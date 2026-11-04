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
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
    # The workflow the hosted demo cannot do: design a voice, persist it, then reuse it
    # as a clone speaker in a later request.
    "voice_library": dict(
        text="Morning. Coffee first, opinions later.",
        instruct="A dry, unhurried documentary narrator in his sixties, slightly amused.",
        language="Auto",
        clone_text="Same voice, second line, still unmistakably the same person.",
    ),
    # The one-call endpoint for external tools (scripts/say.py). Reads best_of_n's reference.
    "speak_api": dict(),
    # The chat tab: progressive playback, measured as time-to-first-audio.
    "chat_stream": dict(),
    # Long-text chunking: the same 300 words that measured 4.04 words/s and WER 0.68 in one
    # pass, now through the splitter and the stitcher.
    "long_text": dict(
        sentence=("the quick brown fox jumps over the lazy dog while the patient owl watches "
                  "quietly from the old wooden fence beside the river"),
        repeat=10,
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

        # Through the tab's own callbacks, not save_voice(). Save used to read the audio box,
        # which holds the post-processed listening copy — so every designed voice was stored
        # with the Warm EQ baked in at 48kHz, and a saved voice is a future clone reference,
        # which then gets cloned through the codec a second time.
        audio, status, state = webui.generate_voice_design(
            cfg["text"], cfg["language"], cfg["instruct"], 0.9, 1.0, 50, 1.05, postproc="Warm")
        assert audio is not None, f"design failed: {status}"
        box_sr, _box_wav = audio
        assert box_sr == 48000, f"the player should still get the 48kHz copy, got {box_sr}Hz"
        assert isinstance(state, dict) and state.get("path"), \
            f"the raw take must reach Save through gr.State: {state}"
        raw_wav, sr = sf.read(state["path"], dtype="float32")
        assert sr == 24000, f"the kept take must be the raw decode, got {sr}Hz"

        design_path = os.path.join(OUT_DIR, "voice_library_design.wav")
        sf.write(design_path, raw_wav, sr)
        check_audio(design_path, raw_wav, sr)

        _, save_msg, _ = webui.save_designed_voice("Smoke Test Narrator", state)
        assert "Saved" in save_msg, f"saving the designed take failed: {save_msg}"
        saved_wav, saved_sr = sf.read(os.path.join(tmp, "smoke_test_narrator.wav"), dtype="float32")
        assert saved_sr == 24000, (
            f"a saved voice is a future clone reference and must be the model's own decode; "
            f"got {saved_sr}Hz while the player copy is {box_sr}Hz")
        assert saved_wav.shape == raw_wav.shape and \
            float(np.max(np.abs(saved_wav - raw_wav))) < 1e-3, \
            "the saved samples are not the raw decode — something is post-processing the copy " \
            "(tolerance is PCM_16 quantisation: soundfile writes 16-bit WAV)"

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
        print(f"  [voice_library] designed {len(raw_wav) / sr:.1f}s raw @ {sr}Hz "
              f"(player copy @ {box_sr}Hz) -> saved @ {saved_sr}Hz -> cloned "
              f"{len(outs[0]) / out_sr:.1f}s")
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

    # 1. synthesize the source speech with known words (the voice is whatever the model
    #    invents from this description — only the words and the audio path matter here)
    with hub.lock:
        vd = hub.acquire("voice_design")
        wavs, sr = vd.generate_voice_design(
            text=cfg["spoken"], language="English",
            instruct="A clear, unhurried adult female voice, neutral American English.",
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
        text_upd, asr_status, _lang_upd = webui.transcribe_reference(state, start, end, "English")
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
        """The callback returns (audio, status, take1..4, note, take_state, take_picker).

        Most checks want the first two; the take record is handed back whole so the
        'keep this take' path is exercised on real audio rather than a fixture.
        """
        result = webui.generate_voice_clone(*args, **kwargs)
        return result[0], result[1], list(result[2:6]), result[6], result

    try:
        t0 = time.time()
        audio, status, takes, takes_note, full = clone(
            (ref_sr, ref_wav), cfg["ref_text"], cfg["text"], "English", False,
            0.9, 1.0, 50, 1.05, candidates=3, seed=cfg["seed"])
        took = time.time() - t0
        assert audio is not None, f"best-of-N generation failed: {status}"
        sr, wav = audio
        assert "Auto-pick: take #" in status, f"no best-of-N report in the status:\n{status}"
        assert status.count("WER") >= 3, f"candidates were not scored by WER:\n{status}"
        assert "kept take #" in status, (
            f"the first status line must say which take the player is holding:\n{status}")
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

        # 'Keep this take' runs on the real thing: the state must be JSON (it round-trips
        # through the browser), the files must exist, and saving one must produce a voice
        # whose transcript is the line it was asked to speak.
        state, pick = full[7], full[8]
        json.dumps(state)
        assert len(state["takes"]) == 3, f"one record per candidate: {state['takes']}"
        assert all(t["path"] and os.path.exists(t["path"]) for t in state["takes"]), \
            f"every take must be on disk to be kept later: {state['takes']}"
        assert sum(1 for t in state["takes"] if t["kept"]) == 1, "exactly one auto-pick"
        assert state["text"] == cfg["text"], "the transcript is the requested line"
        # One save control: the reference first, then every take; it defaults to the
        # reference and its label names the auto-pick.
        values = [value for _, value in pick["choices"]]
        assert values == [webui.SAVE_REFERENCE] + [str(t["index"]) for t in state["takes"]], \
            f"the save picker offers the reference and every take: {pick}"
        assert pick["value"] == webui.SAVE_REFERENCE, \
            f"the save picker defaults to the reference: {pick['value']}"
        kept_index = next(t["index"] for t in state["takes"] if t["kept"])
        assert f"auto-pick was take #{kept_index}" in (pick.get("label") or ""), \
            f"the picker must say which one it chose for you: {pick.get('label')}"
        # CONF_GOOD is provisional after the scorer fix; these lines are the calibration data.
        for line in status.splitlines():
            if line.strip().startswith("#"):
                print(f"  [best_of_n] {line.strip()}")

        loser = str(next(t["index"] for t in state["takes"] if not t["kept"]))
        with tempfile.TemporaryDirectory() as vdir:
            original_voices, webui.VOICES_DIR = webui.VOICES_DIR, vdir
            try:
                _, saved_msg, saved_dropdown = webui.save_generated_voice(state, loser, "smoke take")
                assert saved_msg.startswith("Saved take #"), f"saving a take failed: {saved_msg}"
                meta_path = os.path.join(vdir, "smoke_take.json")
                assert os.path.exists(os.path.join(vdir, "smoke_take.wav")), "no wav written"
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
                assert meta["text"] == cfg["text"], f"wrong transcript: {meta}"
                assert 1.0 < meta["seconds"] < 20.0, f"implausible length saved: {meta}"
                assert saved_dropdown.get("value") == "smoke_take", \
                    f"the library dropdown should land on the new voice: {saved_dropdown}"
                # The saved copy is the model's own decode, so the vocoder's rate survives.
                saved_wav, saved_sr = sf.read(os.path.join(vdir, "smoke_take.wav"), dtype="float32")
                assert saved_sr == 24000, \
                    f"a saved voice should be the raw 24kHz decode, not the 48kHz copy: {saved_sr}"
                print(f"  [best_of_n] kept take #{loser} as a voice "
                      f"({meta['seconds']}s @ {saved_sr}Hz) — overruling the auto-pick")
            finally:
                webui.VOICES_DIR = original_voices
        _, _, single_takes, single_note, _ = clone((ref_sr, ref_wav), cfg["ref_text"], cfg["text"],
                                                "English", False, 0.9, 1.0, 50, 1.05,
                                                candidates=1, seed=cfg["seed"])
        assert not any(isinstance(t, dict) and t.get("visible") for t in single_takes), \
            "a single candidate must not leave four empty players on screen"
        assert single_note.get("visible") is False, "and no explanation row either"
        raw_pair, _, _, _, _ = clone(
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
        first, _, _, _, _ = clone(
            (ref_sr, ref_wav), cfg["ref_text"], cfg["text"], "English", False,
            0.9, 1.0, 50, 1.05, candidates=1, seed=99)
        second, _, _, _, _ = clone(
            (ref_sr, ref_wav), cfg["ref_text"], cfg["text"], "English", False,
            0.9, 1.0, 50, 1.05, candidates=1, seed=99)
        assert first[1].shape == second[1].shape, "same seed changed the clip length"
        assert np.array_equal(first[1], second[1]), "same seed did not reproduce the same audio"
        print("  [best_of_n] seed reproduced identical audio across two calls")

        # The fit guard is a plausibility check on words-per-second, so it catches the
        # failure that was actually measured (transcript of the whole upload, or a stub left
        # in) — not a same-length text with the wrong words. Assert exactly that.
        _, too_short, _, _, _ = clone(
            (ref_sr, ref_wav), "No.", cfg["text"], "English", False,
            0.9, 1.0, 50, 1.05, candidates=1, seed=5)
        assert "WARNING" in too_short and "too short" in too_short, \
            f"a one-word transcript for a multi-second clip was not flagged:\n{too_short}"
        _, too_long, _, _, _ = clone(
            (ref_sr, ref_wav), " ".join(["padding"] * 160), cfg["text"], "English", False,
            0.9, 1.0, 50, 1.05, candidates=1, seed=5)
        assert "too long" in too_long, \
            f"a 160-word transcript for a short clip was not flagged:\n{too_long}"
        print("  [best_of_n] fit guard flagged both implausible transcripts")
    finally:
        webui.ASR.unload()
        webui.ASR = None


def run_long_text(hub: webui.ModelHub) -> None:
    """Long text must be chunked, and the result must be a plausible reading.

    What this case used to assert — 'chunked is slower than one pass' — was measured against
    the CustomVoice 'ryan' preset (4.04 words/s, WER 0.68 single pass) which no longer ships.
    Re-measured on the voice that does ship, one pass over the same 300 words came back at
    3.39 words/s with WER 0.62 on one run and 1.40 words/s on another: without chunking this
    checkpoint either rushes or rambles, so an in-run comparison against one pass is not a
    stable instrument (it failed by being *faster* than a degenerate single pass). The chunked
    route measured 3.02 words/s, WER 0.47, and the assertions below are absolute ones on it.

    Measured baselines, kept here because they are what justifies the bands:
      one pass, preset ryan      74.3s audio, 4.04 words/s, WER 0.68
      one pass, designed voice   88.5s / 3.39 and 214s / 1.40 words/s, WER 0.62
      chunked, designed voice    76.1s audio, 3.02 words/s, WER 0.42-0.47 (0.77 once)
    """
    cfg = CASES["long_text"]
    text = " ".join([cfg["sentence"]] * cfg["repeat"])
    words = len(text.split())
    description = "A clear, steady adult voice, neutral American English, unhurried."
    if webui.ASR is None:                      # WER is an assertion here, not an optional extra
        webui.ASR = webui.Transcriber()

    started = time.time()
    audio, status, _state = webui.generate_voice_design(
        text, "English", description, 0.9, 1.0, 50, 1.05, postproc="off")
    took = time.time() - started
    sr, wav = audio
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    seconds = len(wav) / sr
    per_second = words / seconds

    assert "Chunked" in status, f"300 words must go through the chunked path:\n{status}"
    assert webui.split_for_tts(text) != [text], "the splitter should slice a 300-word script"
    # 3.6 is above the 3.39-4.04 one-pass rush: if chunking stopped working this is the
    # assertion that says so, and the 0.6 floor catches the rambling mode instead.
    assert 0.6 < per_second < 3.6, (
        f"{per_second:.2f} words/s — the chunked route measures ~3.0; one pass rushes past 3.4 "
        f"or rambles under 1.5, and both of those are what chunking exists to prevent")
    assert 0.4 * words / 2.5 < seconds < 1.6 * words / 2.5, \
        f"{seconds:.1f}s for {words} words is not a plausible reading"
    if webui.ASR is None or not webui.ASR.enabled:
        raise AssertionError("WER is asserted in this case; Whisper must be available")
    step = int(20 * sr)
    hypotheses = []
    for start in range(0, len(wav), step):
        clip = wav[start:start + step]
        if len(clip) < sr // 2:
            continue
        try:
            hypotheses.append(webui.ASR.transcribe(clip, sr, "English"))
        except Exception:  # noqa: BLE001 - scoring must not fail the case
            pass
    assert hypotheses, "Whisper returned nothing to score"
    wer = webui.word_error_rate(" ".join(hypotheses), text)
    # Loose on purpose. This corpus is one sentence repeated ten times, Whisper hallucinates
    # repeats on that, and the hard 20s windows cut words: the same route measured 0.42, 0.47
    # and 0.77. A tighter bound fails on instrument noise, so this one catches a generation
    # that has stopped being intelligible, and the pacing band above is the strong assertion.
    assert wer < 0.85, f"chunked WER {wer:.2f} — worse than anything this text has measured"

    # Stitching artefacts: no long hole anywhere, and never past full scale.
    #
    # This bound used to be 1.0s and its message blamed the joins. That conflated two
    # different failures. Measured on a clean 16-piece stitch of this exact text, the longest
    # quiet run was 0.64s and it sat 1.76s away from the nearest join — the model pausing
    # inside a sentence, which is prosody, not a stitching bug. A run that paused a little
    # harder (1.04s) failed here while the joins were fine. Join behaviour — trimming and the
    # crossfade — is pinned deterministically in the CPU tier
    # (test_stitch_trims_and_crossfades), where the seam locations are known; what this can
    # meaningfully assert on real speech is that nothing goes quiet long enough to read as a
    # stutter. 1.6s is that guard with the measured 1.04s outlier inside it.
    windows = webui._rms_windows(wav, sr, int(0.02 * sr))
    longest = gap = 0
    previous = None
    for index in np.flatnonzero(windows < webui.SILENCE_RMS):
        gap = gap + 1 if previous == index - 1 else 1
        longest = max(longest, gap)
        previous = index
    assert longest * 0.02 < 1.6, (
        f"a {longest * 0.02:.2f}s hole reads as a stutter — either a piece came back empty, "
        f"or trim_edges stopped trimming (measured clean run: 0.64s longest pause)")
    assert float(np.max(np.abs(wav))) <= 1.0, "stitched audio exceeded full scale"
    # No click assertion here: a global max-diff threshold on real speech is meaningless. The
    # seam behaviour is pinned synthetically in test_stitch_trims_and_crossfades, where the
    # join locations and the expected waveform are both known.

    # (scored above)
    path = os.path.join(OUT_DIR, "long_text.wav")
    sf.write(path, wav, sr)
    check_audio(path, wav, sr)
    print(f"  [long_text] {took:.1f}s | {seconds:.1f}s audio | {per_second:.2f} words/s "
          f"(chunked measures ~3.0; one pass 1.4-4.0) | WER {wer:.2f} (0.42-0.77 measured) | "
          f"{status.splitlines()[1]}")


def run_speak_api(hub: webui.ModelHub) -> None:
    """The one-call endpoint an external tool uses: name a voice, get speech back."""
    cfg = CASES["best_of_n"]
    ref_path = os.path.join(webui.media_dir(), "best_of_n_ref.wav")
    if not os.path.exists(ref_path):
        urllib.request.urlretrieve(cfg["ref_audio"], ref_path)
    ref_wav, ref_sr = sf.read(ref_path, dtype="float32")
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)

    with tempfile.TemporaryDirectory() as vdir:
        original_voices, webui.VOICES_DIR = webui.VOICES_DIR, vdir
        try:
            webui.save_voice(name="Smoke Narrator", wav=ref_wav, sr=ref_sr,
                             text=cfg["ref_text"], instruct="", language="English")
            started = time.time()
            audio, status = webui.speak("Smoke Narrator", "Short sentence, clearly spoken.",
                                        "English")
            took = time.time() - started
            assert audio is not None, f"speak returned nothing:\n{status}"
            sr, wav = audio
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            assert sr in (24000, 48000), f"unexpected sample rate {sr}"
            assert 0.5 < len(wav) / sr < 12.0, f"implausible length: {len(wav)/sr:.1f}s"
            assert float(np.max(np.abs(wav))) <= 1.0 and np.all(np.isfinite(wav)), "bad audio"
            assert "Speak" in status, f"the status should name the path taken:\n{status}"
            assert "ICL" in status, f"a voiced sidecar must select ICL cloning:\n{status}"
            first_status, first_seconds = status, len(wav) / sr

            # A caller that guesses a name gets the list, not a stack trace.
            audio, status = webui.speak("no-such-voice", "hello")
            assert audio is None and status.startswith("Error"), f"unknown voice: {status}"
            assert "smoke_narrator" in status, f"the error should list what exists: {status}"

            # A voice with no transcript must fall back to x-vector rather than fail.
            webui.save_voice(name="Silent Ghost", wav=ref_wav, sr=ref_sr, text="",
                             instruct="", language="English")
            audio, status = webui.speak("Silent Ghost", "Hello there.", "English")
            assert audio is not None and "x-vector" in status, \
                f"a transcript-less voice should use x-vector:\n{status}"

            # A described voice is spoken by Voice Design rather than cloned — this is the
            # route the shipped 'Fox' voice takes, and the only one that keeps its top end
            # (2.67m of energy above 9kHz against 0.57m for cloning a saved take).
            webui.save_voice(name="Described One", wav=None, sr=0, text="", instruct="",
                             language="English", description="A low, unhurried male narrator.",
                             kind="design")
            audio, status = webui.speak("Described One", "Hello from a description.", "English")
            assert audio is not None and "described" in status, \
                f"a described voice must be spoken from its description:\n{status}"
            d_sr, d_wav = audio
            d_wav = np.asarray(d_wav, dtype=np.float32).reshape(-1)
            assert 0.3 < len(d_wav) / d_sr < 12.0, f"implausible length {len(d_wav)/d_sr:.1f}s"
            assert float(np.max(np.abs(d_wav))) > 0.01, "a described voice produced silence"
            assert "described_one" in webui.voices_available(), "and it shows up in --list"

            listing = webui.voices_available()
            assert "smoke_narrator" in listing and "silent_ghost" in listing, listing
            print(f"  [speak_api] {took:.1f}s | one call, name only | "
                  f"{first_status.splitlines()[0]} ({first_seconds:.1f}s)")
        finally:
            webui.VOICES_DIR = original_voices


def run_chat_stream(hub: webui.ModelHub) -> None:
    """The chat tab's whole claim is that you hear the first sentence early."""
    cfg = CASES["best_of_n"]
    ref_path = os.path.join(webui.media_dir(), "best_of_n_ref.wav")
    if not os.path.exists(ref_path):
        urllib.request.urlretrieve(cfg["ref_audio"], ref_path)
    ref_wav, ref_sr = sf.read(ref_path, dtype="float32")
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)

    text = ("The build finished cleanly and the test suite is green. "
            "I rewrote the chunker so long text is spoken at a natural pace instead of rushing. "
            "The seam between pieces uses a short equal-power crossfade, so there is no click. "
            "Tell me if it still sounds hurried anywhere and I will widen the band.")

    with tempfile.TemporaryDirectory() as vdir:
        original_voices, webui.VOICES_DIR = webui.VOICES_DIR, vdir
        try:
            webui.save_voice(name="Smoke Narrator", wav=ref_wav, sr=ref_sr,
                             text=cfg["ref_text"], instruct="", language="English")
            started = time.time()
            first_audio = None
            turns = []
            for turn in webui.chat_say(text, [], "Smoke Narrator", "English"):
                if first_audio is None and str(turn[1]).startswith("<audio"):
                    first_audio = time.time() - started
                turns.append(turn)
            total = time.time() - started

            import base64 as _b64
            import io as _io
            import re as _re

            streamed = []
            for turn in turns:
                match = _re.search(r'src="data:audio/wav;base64,([^"]+)"', str(turn[1]))
                if match:
                    streamed.append(sf.read(_io.BytesIO(_b64.b64decode(match.group(1))),
                                             dtype="float32")[0])
            assert len(streamed) >= 2, f"a 60-word reply should arrive in pieces: {len(streamed)}"
            assert all(len(w) > 0 for w in streamed), "a hand-off carried no audio"
            assert all(float(np.max(np.abs(w))) <= 1.0 for w in streamed), "streamed audio clipped"
            assert all("autoplay" in str(t[1]) and "controls" in str(t[1])
                       for t in turns if str(t[1]).startswith("<audio")), \
                "each piece must autoplay and still offer a manual play button"
            assert first_audio is not None and first_audio < total - 1.0, \
                f"first audio must land well before the end (ttfb {first_audio:.1f}s of {total:.1f}s)"
            said = turns[-1][0][-1]["content"]
            assert said.replace(" ", "") == text.replace(" ", ""), "the log lost or reordered words"
            assert "Said" in turns[-1][2], f"final status: {turns[-1][2]}"
            print(f"  [chat_stream] {total:.1f}s | first audio {first_audio:.1f}s | "
                  f"{len(streamed)} pieces | {turns[-1][2]}")
        finally:
            webui.VOICES_DIR = original_voices


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

    if capability == "long_text":
        run_long_text(hub)
        if not keep_gpu:
            with hub.lock:
                while hub.on_gpu:
                    _, resident = hub.on_gpu.popitem(last=False)
                    hub._move(resident, torch.device("cpu"))
                    torch.cuda.empty_cache()
        return

    if capability == "speak_api":
        run_speak_api(hub)
        if not keep_gpu:
            with hub.lock:
                while hub.on_gpu:
                    _, resident = hub.on_gpu.popitem(last=False)
                    hub._move(resident, torch.device("cpu"))
                    torch.cuda.empty_cache()
        return

    if capability == "chat_stream":
        run_chat_stream(hub)
        if not keep_gpu:
            with hub.lock:
                while hub.on_gpu:
                    _, resident = hub.on_gpu.popitem(last=False)
                    hub._move(resident, torch.device("cpu"))
                    torch.cuda.empty_cache()
        return

    cfg = dict(CASES[capability])
    started = time.time()
    with hub.lock:
        tts = hub.acquire(capability)
        load_s = time.time() - started
        if capability == "voice_design":
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
