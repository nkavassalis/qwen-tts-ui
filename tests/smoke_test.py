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
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import List, Tuple

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
    # Best-of-N: the real Clone callback, scoring each take with Whisper WER then speaker
    # similarity, plus the seed field's reproducibility promise.
    "best_of_n": dict(
        ref_audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
        ref_text="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
        text="Picking the best of several takes should never pick a worse one by accident.",
        seed=7,
    ),
    # Video upload path: spoken by Ryan, wrapped into an mp4, then imported / trimmed /
    # transcribed / cloned — the whole new pipeline in one case.
    "media_pipeline": dict(
        spoken="The quick brown fox jumps over the lazy dog near the river bank.",
        keywords=["quick", "brown", "fox", "jumps", "lazy", "dog", "river"],
        clone_text="Cloning from a video file should sound exactly the same as from a wav.",
    ),
}


# Cases that did not actually run. A skip is not a pass: main() exits non-zero unless
# --allow-skip, because `--only media_pipeline` on a box without ffmpeg used to print
# SKIPPED and still report "All smoke tests passed" with exit 0.
SKIPPED: List[Tuple[str, str]] = []


def skip(case: str, reason: str, note: str = "") -> None:
    """Record a case that did not run, so the exit code can say so."""
    SKIPPED.append((case, reason))
    print(f"  [{case}] SKIPPED: {reason}" + (f" ({note})" if note else ""))


def check_audio(path: str, wav: np.ndarray, sr: int, min_seconds: float = 0.5) -> None:
    assert wav.ndim == 1, f"expected 1-D audio, got shape {wav.shape}"
    assert len(wav) / sr >= min_seconds, f"audio too short: {len(wav) / sr:.2f}s"
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(wav)))
    assert rms > 1e-3, f"audio is (near) silent, rms={rms:.6f}"
    assert peak <= 1.05, f"audio clips, peak={peak:.3f}"
    assert os.path.getsize(path) > 10_000, f"written wav suspiciously small: {os.path.getsize(path)} bytes"
    print(f"    rms={rms:.4f} peak={peak:.3f} dur={len(wav) / sr:.2f}s -> {path}")


def run_voice_library(hub: webui.ModelHub) -> None:
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
        # Both models are evicted by the caller below; nothing to do here.


def run_media_pipeline(hub: webui.ModelHub, use_asr: bool = True) -> None:
    """video upload -> ffmpeg -> waveform/trim helpers -> Whisper -> clone."""
    cfg = CASES["media_pipeline"]
    if not shutil.which(webui.FFMPEG_BIN):
        # A skip is not a pass: main() exits non-zero unless --allow-skip.
        skip("media_pipeline", f"ffmpeg ({webui.FFMPEG_BIN}) not on PATH")

    # The app only reads uploads from inside its media dir (resolve_input_path), so the
    # fixture mp4 has to be built there, not in OUT_DIR next to it.
    webui.MEDIA_DIR = os.path.join(OUT_DIR, "media")
    os.makedirs(webui.MEDIA_DIR, exist_ok=True)

    # 1. synthesize the source speech with a known speaker and known words
    with hub.lock:
        cv = hub.acquire("custom_voice")
        wavs, sr = cv.generate_custom_voice(text=cfg["spoken"], language="English", speaker="ryan",
                                            non_streaming_mode=True, max_new_tokens=192)
    source_wav = os.path.join(OUT_DIR, "media_pipeline_source.wav")
    sf.write(source_wav, wavs[0], sr)
    seconds = len(wavs[0]) / sr

    # 2. wrap it in a real mp4 (video stream + aac audio), like a phone export
    mp4 = os.path.join(webui.MEDIA_DIR, "media_pipeline.mp4")
    proc = subprocess.run(
        [webui.FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c=navy:s=320x240:d={seconds + 0.5:.2f}",
         "-i", source_wav, "-shortest",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", mp4],
        capture_output=True, text=True)
    assert proc.returncode == 0, f"ffmpeg could not build the test mp4: {proc.stderr[-300:]}"
    assert os.path.getsize(mp4) > 10_000, "test mp4 suspiciously small"

    # 3. the app's own import path must see only the audio, through the same callbacks
    #    the browser hits (gr.State is JSON-decoded on the way back, so we round-trip it)
    imported = webui.import_media(mp4)
    state, status = imported[0], imported[1]
    assert "Loaded" in status, f"import failed: {status}"
    assert "video" in status, f"mp4 should be reported as video: {status}"
    import json as _json
    state = _json.loads(_json.dumps(state))          # exactly what the browser sends back
    assert state["duration"] > 0 and state["sample_rate"] == webui.REF_SAMPLE_RATE

    # 4. the visual selection helpers, as the sliders would drive them
    _, clip_audio, info, start_upd, end_upd = webui.update_clip(state, 0.3, 999.0)
    start, end = start_upd["value"], end_upd["value"]
    assert end - start <= webui.REF_MAX_SECONDS + 1e-6, f"selection not clamped: {start}-{end}"
    assert "selected" in info, f"clip guidance missing: {info}"
    _, clip_audio, info, start_upd, end_upd = webui.auto_pick_clip(state)
    start, end = start_upd["value"], end_upd["value"]
    assert end > start, f"auto-pick produced an empty window: {start}-{end}"

    # 5. transcription of exactly what will be cloned
    transcript = cfg["spoken"]
    if use_asr:
        webui.ASR = webui.Transcriber()
        t0 = time.time()
        text_upd, asr_status = webui.transcribe_reference(state, start, end, "English")
        took = time.time() - t0
        assert "Error" not in asr_status, f"transcription failed: {asr_status}"
        transcript = text_upd["value"]
        found = [w for w in cfg["keywords"] if w in transcript.lower()]
        print(f"  [media_pipeline] whisper ({took:.1f}s): {transcript!r}")
        assert len(found) >= len(cfg["keywords"]) * 0.6, \
            f"transcript missed too many words ({found} of {cfg['keywords']}): {transcript!r}"
        webui.ASR.unload()
        webui.ASR = None
    else:
        skip("media_pipeline/whisper", "--no-asr", note=f"using the known transcript for {cfg['spoken']!r}")

    # 6. clone from the trimmed video-derived clip, again via the UI callback
    with hub.lock:
        base = hub.acquire("voice_clone")
        prompt = base.create_voice_clone_prompt(ref_audio=(clip_audio[1], clip_audio[0]),
                                                ref_text=transcript, x_vector_only_mode=False)
        outs, out_sr = base.generate_voice_clone(text=cfg["clone_text"], language="English",
                                                 voice_clone_prompt=prompt, max_new_tokens=192)
    clone_path = os.path.join(OUT_DIR, "media_pipeline_clone.wav")
    sf.write(clone_path, outs[0], out_sr)
    print(f"  [media_pipeline] mp4 {seconds:.1f}s -> clip {end - start:.1f}s -> clone {len(outs[0]) / out_sr:.1f}s")
    check_audio(clone_path, outs[0], out_sr)


def run_best_of_n(hub: webui.ModelHub) -> None:
    """Exercise the Clone callback's candidate scoring and the seed field.

    Goes through webui.generate_voice_clone rather than the model API, so the wiring
    (inputs, scoring, the report the user reads) is what is under test.
    """
    cfg = CASES["best_of_n"]
    os.makedirs(webui.MEDIA_DIR, exist_ok=True)
    ref_path = os.path.join(webui.MEDIA_DIR, "best_of_n_ref.wav")
    if not os.path.exists(ref_path):
        urllib.request.urlretrieve(cfg["ref_audio"], ref_path)
    ref_wav, ref_sr = sf.read(ref_path, dtype="float32")

    webui.ASR = webui.Transcriber()

    def clone(*args, **kwargs):
        """The callback returns (audio, status, take1..4, takes_note); most checks want two."""
        result = webui.generate_voice_clone(*args, **kwargs)
        return result[0], result[1], list(result[2:6]), result[6]

    try:
        t0 = time.time()
        audio, status, takes, takes_note = clone(
            (ref_sr, ref_wav), cfg["ref_text"], cfg["text"], "English", False,
            0.9, 1.0, 50, 1.05, candidates=3, seed=cfg["seed"])
        took = time.time() - t0
        assert audio is not None, f"best-of-N generation failed: {status}"
        sr, wav = audio
        assert "Best of 3" in status, f"no best-of-N report in the status:\n{status}"
        assert status.count("WER") >= 3, f"candidates were not scored by WER:\n{status}"
        assert "kept #" in status, f"no winner named in the status:\n{status}"
        assert "WARNING" not in status, f"a correct transcript must not be flagged:\n{status}"
        assert sr == webui.POSTPROCESS_SR, \
            f"the listening copy should come back at {webui.POSTPROCESS_SR}Hz, got {sr}Hz"
        assert "Post-processing" in status, f"the status should say what was applied:\n{status}"
        # Every candidate must stay auditionable — the auto-pick is only defensible if the
        # human can hear the alternatives and keep a different one.
        visible = [t for t in takes if isinstance(t, dict) and t.get("visible")]
        assert len(visible) == 3, f"expected 3 visible takes, got {len(visible)}"
        assert all("Take #" in (t.get("label") or "") for t in visible), \
            f"takes must be labelled with their scores: {[t.get('label') for t in visible]}"
        assert any("(kept)" in (t.get("label") or "") for t in visible), \
            f"the auto-picked take must be marked: {[t.get('label') for t in visible]}"
        assert all(t.get("value") is not None for t in visible), "each take carries audio"
        assert takes_note.get("visible") is True, "the explanation should show when there are takes"
        _, _, single_takes, single_note = clone((ref_sr, ref_wav), cfg["ref_text"], cfg["text"],
                                                "English", False, 0.9, 1.0, 50, 1.05,
                                                candidates=1, seed=cfg["seed"])
        assert not any(isinstance(t, dict) and t.get("visible") for t in single_takes), \
            "a single candidate must not leave four empty players on screen"
        assert single_note.get("visible") is False, "and no explanation row either"
        raw_pair, _, _, _ = clone(
            (ref_sr, ref_wav), cfg["ref_text"], cfg["text"], "English", False,
            0.9, 1.0, 50, 1.05, postproc="off", candidates=1, seed=cfg["seed"])
        raw_sr, raw = raw_pair
        assert raw_sr == 24000, f"'off' must hand back the vocoder's own 24kHz, got {raw_sr}"
        assert np.all(np.isfinite(raw)) and float(np.max(np.abs(raw))) <= 1.0, \
            "post-processed audio must stay finite and inside full scale"
        print(f"  [best_of_n] {took:.1f}s | {status.splitlines()[-1]}")
        path = os.path.join(OUT_DIR, "best_of_n.wav")
        sf.write(path, wav, sr)
        check_audio(path, wav, sr)

        # The seed field promises reproducibility; two identical calls must agree bitwise.
        first, _, _, _ = clone(
            (ref_sr, ref_wav), cfg["ref_text"], cfg["text"], "English", False,
            0.9, 1.0, 50, 1.05, candidates=1, seed=99)
        second, _, _, _ = clone(
            (ref_sr, ref_wav), cfg["ref_text"], cfg["text"], "English", False,
            0.9, 1.0, 50, 1.05, candidates=1, seed=99)
        assert first[1].shape == second[1].shape, "same seed changed the clip length"
        assert np.array_equal(first[1], second[1]), "same seed did not reproduce the same audio"
        print("  [best_of_n] seed reproduced identical audio across two calls")

        # The fit guard is a plausibility check on words-per-second, so it catches the
        # failure that was actually measured (transcript of the whole upload, or a stub left
        # in) — not a same-length text with the wrong words. Assert exactly that.
        _, too_short, _, _ = clone(
            (ref_sr, ref_wav), "No.", cfg["text"], "English", False,
            0.9, 1.0, 50, 1.05, candidates=1, seed=5)
        assert "WARNING" in too_short and "too short" in too_short, \
            f"a one-word transcript for a multi-second clip was not flagged:\n{too_short}"
        _, too_long, _, _ = clone(
            (ref_sr, ref_wav), " ".join(["padding"] * 160), cfg["text"], "English", False,
            0.9, 1.0, 50, 1.05, candidates=1, seed=5)
        assert "too long" in too_long, \
            f"a 160-word transcript for a short clip was not flagged:\n{too_long}"
        print("  [best_of_n] fit guard flagged both implausible transcripts")
    finally:
        webui.ASR.unload()
        webui.ASR = None


def run_case(hub: webui.ModelHub, capability: str, keep_gpu: bool, use_asr: bool = True) -> None:
    if capability == "media_pipeline":
        run_media_pipeline(hub, use_asr)
        if not keep_gpu:
            with hub.lock:
                while hub.on_gpu:
                    _, resident = hub.on_gpu.popitem(last=False)
                    hub._move(resident, torch.device("cpu"))
                    torch.cuda.empty_cache()
        return

    if capability == "voice_library":
        run_voice_library(hub)
        if not keep_gpu:
            with hub.lock:
                while hub.on_gpu:
                    _, resident = hub.on_gpu.popitem(last=False)
                    hub._move(resident, torch.device("cpu"))
                    torch.cuda.empty_cache()
        return

    if capability == "best_of_n":
        run_best_of_n(hub)
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
    parser.add_argument("--no-asr", action="store_true",
                        help="Skip the Whisper step of the media pipeline case (records a "
                             "skip; combine with --allow-skip to still exit 0).")
    parser.add_argument("--allow-skip", action="store_true",
                        help="Exit 0 even when cases were skipped (a skip is a failure by "
                             "default, so CI cannot quietly go green on a missing ffmpeg).")
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
            run_case(hub, capability, args.keep_gpu, use_asr=not args.no_asr)
        if SKIPPED:
            print(f"\n{len(SKIPPED)} case(s) did not actually run:")
            for case, reason in SKIPPED:
                print(f"  SKIPPED {case}: {reason}")
            if not args.allow_skip:
                print("\nA skipped case is not a pass. Fix the cause, or re-run with "
                      "--allow-skip if the skip is intended.")
                return 1
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
