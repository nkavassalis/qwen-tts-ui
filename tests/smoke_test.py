# coding=utf-8
"""
Smoke test for the local Qwen3-TTS setup.

Exercises each capability end to end (model load -> GPU swap -> generate -> wav on
disk) and asserts the audio is non-silent, non-clipping and of plausible length.
This is a functional test of the real inference path, not a unit test of mocks.

Usage:
    .venv/bin/python tests/smoke_test.py                 # all three capabilities
    .venv/bin/python tests/smoke_test.py --only custom_voice
    .venv/bin/python tests/smoke_test.py --keep-gpu      # do not evict between cases
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui"))
import app as webui  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "tests", "output")

CASES = {
    "custom_voice": dict(
        text="Hello! This is a smoke test for Qwen three T T S running on my local GPU.",
        language="English",
        speaker="ryan",
        instruct="Very happy and energetic.",
    ),
    "voice_design": dict(
        text="The results were published last Tuesday, and nobody expected them to be this large.",
        language="English",
        instruct="An elderly British male narrator, calm and measured, with a low resonant voice.",
    ),
    "voice_clone": dict(
        # Reference clip from the official Qwen3-TTS README example.
        ref_audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
        ref_text="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
        text="Cloning a voice should still pronounce every word clearly, even on a second pass.",
        language="English",
    ),
}


def check_audio(path: str, wav: np.ndarray, sr: int, min_seconds: float = 0.5) -> None:
    assert wav.ndim == 1, f"expected 1-D audio, got shape {wav.shape}"
    assert len(wav) / sr >= min_seconds, f"audio too short: {len(wav) / sr:.2f}s"
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(wav)))
    assert rms > 1e-3, f"audio is (near) silent, rms={rms:.6f}"
    assert peak <= 1.05, f"audio clips, peak={peak:.3f}"
    assert os.path.getsize(path) > 10_000, f"written wav suspiciously small: {os.path.getsize(path)} bytes"
    print(f"    rms={rms:.4f} peak={peak:.3f} dur={len(wav) / sr:.2f}s -> {path}")


def run_case(hub: webui.ModelHub, capability: str, keep_gpu: bool) -> None:
    cfg = dict(CASES[capability])
    started = time.time()
    with hub.lock:
        tts = hub.acquire(capability)
        load_s = time.time() - started
        if capability == "custom_voice":
            wavs, sr = tts.generate_custom_voice(non_streaming_mode=True, max_new_tokens=256, **cfg)
        elif capability == "voice_design":
            wavs, sr = tts.generate_voice_design(non_streaming_mode=True, max_new_tokens=256, **cfg)
        else:
            prompt = tts.create_voice_clone_prompt(
                ref_audio=cfg["ref_audio"], ref_text=cfg["ref_text"], x_vector_only_mode=False)
            wavs, sr = tts.generate_voice_clone(
                text=cfg["text"], language=cfg["language"], voice_clone_prompt=prompt, max_new_tokens=256)
    gen_s = time.time() - started - load_s

    path = os.path.join(OUT_DIR, f"{capability}.wav")
    sf.write(path, wavs[0], sr)
    print(f"  [{capability}] load={load_s:.1f}s generate={gen_s:.1f}s sr={sr}")
    check_audio(path, wavs[0], sr)

    if not keep_gpu:
        with hub.lock:
            name, resident = hub.on_gpu.popitem(last=False)
            hub._move(resident, torch.device("cpu"))
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    parser.add_argument("--only", choices=list(CASES), action="append")
    parser.add_argument("--keep-gpu", action="store_true")
    parser.add_argument("--max-resident", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA is required for this test"
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{torch.cuda.get_device_capability(0)[0]}"
          f"{torch.cuda.get_device_capability(0)[1]}) torch {torch.__version__}")

    hub = webui.ModelHub(args.models_dir, torch.bfloat16, "sdpa", args.max_resident)
    webui.HUB = hub

    for capability in (args.only or list(CASES)):
        print(f"* {capability}")
        run_case(hub, capability, args.keep_gpu)

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
