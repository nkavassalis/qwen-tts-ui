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
    "dialect": dict(
        # Dialect comes from the *speaker* (checkpoint spk_is_dialect: dylan ->
        # beijing_dialect), not from the language field, which rejects dialect values.
        text="今天天气不错，咱们出去走走吧。",
        language="chinese",
        speaker="dylan",
        instruct="",
    ),
    # The workflow the hosted demo cannot do: design a voice, persist it, then reuse it
    # as a clone speaker in a later request.
    "voice_library": dict(
        text="Morning. Coffee first, opinions later.",
        instruct="A dry, unhurried documentary narrator in his sixties, slightly amused.",
        language="Auto",
        clone_text="Same voice, second line, still unmistakably the same person.",
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


def run_voice_library(hub: webui.ModelHub, keep_gpu: bool) -> None:
    """VoiceDesign -> save_voice -> load_saved_voice -> Base clone, as one user flow."""
    import tempfile

    cfg = CASES["voice_library"]
    tmp = os.path.join(OUT_DIR, "voices")
    os.makedirs(tmp, exist_ok=True)
    original, webui.VOICES_DIR = webui.VOICES_DIR, tmp
    try:
        for stale in os.listdir(tmp):
            os.remove(os.path.join(tmp, stale))

        with hub.lock:
            designed = hub.acquire("voice_design")
            wavs, sr = designed.generate_voice_design(
                text=cfg["text"], language=cfg["language"], instruct=cfg["instruct"],
                non_streaming_mode=True, max_new_tokens=192)
        design_path = os.path.join(OUT_DIR, "voice_library_design.wav")
        sf.write(design_path, wavs[0], sr)
        check_audio(design_path, wavs[0], sr)
        webui.save_voice("Smoke Test Narrator", wavs[0], sr, cfg["text"], cfg["instruct"], cfg["language"])

        choices = webui.voice_choices()
        assert len(choices) == 1, f"designed voice was not registered: {choices}"

        # load_saved_voice returns the gradio Audio shape (sr, waveform) plus the transcript.
        (ref_sr, ref_wav), ref_text, msg = webui.load_saved_voice(choices[0][1])
        assert "Loaded" in msg, f"saved voice failed to load: {msg}"
        assert ref_sr == sr or ref_wav.ndim == 1, "reference should come back as (sr, 1-D waveform)"
        assert ref_text == cfg["text"], "transcript must round-trip; ICL cloning depends on it"

        with hub.lock:
            clone = hub.acquire("voice_clone")
            outs, out_sr = clone.generate_voice_clone(
                text=cfg["clone_text"], language="English",
                voice_clone_prompt=clone.create_voice_clone_prompt(
                    ref_audio=(ref_wav, ref_sr), ref_text=ref_text, x_vector_only_mode=False),
                max_new_tokens=192)
        clone_path = os.path.join(OUT_DIR, "voice_library_clone.wav")
        sf.write(clone_path, outs[0], out_sr)
        print(f"  [voice_library] designed {len(wavs[0]) / sr:.1f}s -> cloned {len(outs[0]) / out_sr:.1f}s")
        check_audio(clone_path, outs[0], out_sr)
    finally:
        webui.VOICES_DIR = original
        if not keep_gpu and "voice_design" in hub.on_gpu:
            pass  # both models get evicted by the caller below


def run_case(hub: webui.ModelHub, capability: str, keep_gpu: bool) -> None:
    if capability == "voice_library":
        run_voice_library(hub, keep_gpu)
        if not keep_gpu:
            with hub.lock:
                while hub.on_gpu:
                    _, resident = hub.on_gpu.popitem(last=False)
                    hub._move(resident, torch.device("cpu"))
                    torch.cuda.empty_cache()
        return

    cfg = dict(CASES[capability])
    if capability == "dialect":
        cfg = {k: v for k, v in cfg.items() if v != ""}
    started = time.time()
    with hub.lock:
        # `dialect` reuses the CustomVoice checkpoint; only the language value differs.
        tts = hub.acquire("custom_voice" if capability == "dialect" else capability)
        load_s = time.time() - started
        if capability in ("custom_voice", "dialect"):
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
    parser.add_argument("--force", action="store_true",
                        help="Run even if the webui is up (unsafe: two copies of the weights).")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA is required for this test"
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{torch.cuda.get_device_capability(0)[0]}"
          f"{torch.cuda.get_device_capability(0)[1]}) torch {torch.__version__}")

    def run() -> int:
        # One checkpoint in host RAM at a time: this test evicts between cases anyway, and
        # pinning all three here alongside a running server has hard-locked the WSL2 VM.
        hub = webui.ModelHub(args.models_dir, torch.bfloat16, "sdpa", args.max_resident,
                             max_cpu_resident=1)
        webui.HUB = hub
        for capability in (args.only or list(CASES)):
            print(f"* {capability}")
            run_case(hub, capability, args.keep_gpu)
        print("\nAll smoke tests passed.")
        return 0

    if args.force:
        print("[smoke] --force: running without the single-owner lock.", flush=True)
        return run()
    try:
        with webui.exclusive_owner("smoke_test"):
            return run()
    except RuntimeError as e:
        print(f"[smoke] refusing to run: {e}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
