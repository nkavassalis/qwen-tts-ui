# coding=utf-8
"""
Fast (CPU-only, no model weights) tests for the WebUI helper logic.

Covers the boundaries that matter for usability:
  * _budget()  — token budget never drops below the floor nor exceeds the cap,
                 so short prompts still work and long prompts cannot run for minutes.
  * _report()  — a generation that hits the cap is flagged as a non-stopping run.
  * _audio_to_tuple / _normalize_audio — gradio audio payloads become clean mono
                 float32 in [-1, 1] at the declared sample rate, whatever the input form.
  * _clone_cache_key — same reference audio hits the cache, different audio/text does not.

Run: .venv/bin/python tests/test_app_logic.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui"))
import app as webui  # noqa: E402


def check(cond: bool, msg: str) -> None:
    assert cond, f"FAILED: {msg}"


def test_budget() -> None:
    check(webui._budget("hi") == webui.MIN_NEW_TOKENS, "very short text must clamp to the floor")
    check(webui._budget("x" * 10_000) == webui.MAX_NEW_TOKENS_CAP, "very long text must clamp to the cap")
    check(webui._budget("a" * 50) == 200, "mid-length text should be 4 tokens/char")
    check(webui._budget("") == webui.MIN_NEW_TOKENS, "empty text must not produce a zero budget")
    check(webui._budget("  hello  ") == webui._budget("hello"), "budget should ignore surrounding whitespace")


def test_report() -> None:
    sr = 24000
    ok = webui._report(np.zeros(sr * 3, dtype=np.float32), sr, 512, "CustomVoice")
    check(ok.startswith("CustomVoice OK"), "normal generation reports OK without warnings")
    check("token cap" not in ok, "normal generation must not warn about the cap")

    runaway = webui._report(np.zeros(sr * 164, dtype=np.float32), sr, webui.MAX_NEW_TOKENS_CAP, "VoiceClone")
    check("token cap" in runaway, "a capped generation must be flagged as non-stopping")

    # Long audio that finished before the cap (large budget) is not flagged.
    long_ok = webui._report(np.zeros(sr * 100, dtype=np.float32), sr, 8192, "VoiceDesign")
    check("token cap" not in long_ok, "long-but-complete audio should not be flagged")


def test_audio_normalisation() -> None:
    sr = 48000
    mono = (np.sin(np.linspace(0, 100, 1000)) * 0.5).astype(np.float32)

    # gradio hands over either (sr, wav) or {"sampling_rate", "data"}; _audio_to_tuple
    # normalises both to (wav float32 mono, sample_rate).
    a_wav, a_sr = webui._audio_to_tuple((sr, mono))
    check((a_sr, a_wav.dtype, a_wav.shape) == (sr, np.float32, (1000,)), "tuple form normalises to float32 mono")

    b_wav, b_sr = webui._audio_to_tuple({"sampling_rate": sr, "data": mono})
    check((b_sr, b_wav.shape) == (sr, (1000,)), "dict form matches tuple form")

    check(webui._audio_to_tuple(None) is None, "missing audio stays None so the caller can error out")
    # Stereo input is mixed down; integer PCM is rescaled; over-range float is normalised.
    stereo = np.stack([mono, mono], axis=-1)
    mixed, _ = webui._audio_to_tuple((sr, stereo))
    check(mixed.ndim == 1 and np.allclose(mixed, mono, atol=1e-6), "stereo is averaged to mono")

    pcm = (mono * 32767).astype(np.int16)
    from_int, _ = webui._audio_to_tuple((sr, pcm))
    check(np.max(np.abs(from_int)) <= 1.0 and np.allclose(from_int, mono, atol=1e-2), "int16 is rescaled")

    loud = (mono * 4.0).astype(np.float32)
    from_loud, _ = webui._audio_to_tuple((sr, loud))
    check(np.max(np.abs(from_loud)) <= 1.0, "over-range float audio is normalised, never clipped raw")


def test_clone_cache_key() -> None:
    wav = np.ones(24000, dtype=np.float32) * 0.1
    key = webui._clone_cache_key((wav, 24000), "hello", False)
    check(key == webui._clone_cache_key((wav.copy(), 24000), "hello", False), "identical reference hits the cache")
    check(key != webui._clone_cache_key((wav, 24000), "hello", True), "x-vector mode changes the key")
    check(key != webui._clone_cache_key((wav, 24000), "hello!", False), "transcript changes the key")
    check(key != webui._clone_cache_key((wav * 2, 24000), "hello", False), "different audio changes the key")


def test_move_reaches_the_speech_tokenizer() -> None:
    """Regression: model.speech_tokenizer is a plain wrapper, not an nn.Module.

    If ModelHub._move forgets it, the vocoder silently runs on CPU and generation
    gets ~25x slower (seen as aten::slow_conv_dilated2d dominating a profile).
    """
    class FakeTokenizer:
        def __init__(self):
            self.model = None
            self.device = "cpu"

    import torch

    class FakeInner(torch.nn.Module):
        """Must be a real nn.Module — _move only descends into Module-typed members."""
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.seen = None

        def to(self, device):
            self.seen = device
            return self

    class FakeModel:
        def __init__(self):
            self.speech_tokenizer = FakeTokenizer()
            self.speech_tokenizer.model = FakeInner()
            self.seen = None

        def to(self, device):
            self.seen = device
            return self

    class FakeTTS:
        def __init__(self):
            self.model = FakeModel()
            self.device = "cpu"

    tts = FakeTTS()
    webui.ModelHub._move(tts, "cuda")
    check(tts.model.seen == "cuda", "talker moved")
    check(tts.model.speech_tokenizer.model.seen == "cuda", "speech tokenizer vocoder moved too")
    check(tts.model.speech_tokenizer.device == "cuda", "speech tokenizer .device cache updated")
    check(tts.device == "cuda", "wrapper .device updated")


def test_language_choices_match_checkpoint() -> None:
    """Dropdown values must match what the model actually accepts.

    Upstream builds `supported_languages` from codec_language_id but *skips* the
    dialect entries (modeling_qwen3_tts.py:1832) — dialect is a speaker attribute
    (`spk_is_dialect`), so offering 'beijing_dialect' would raise at generate time.
    Skipped when the weights are not downloaded.
    """
    import json

    cfg_path = os.path.join(webui.ROOT, "models", "Qwen3-TTS-12Hz-1.7B-CustomVoice", "config.json")
    if not os.path.exists(cfg_path):
        print(f"     (skipped: {cfg_path} not present)")
        return
    talker = json.load(open(cfg_path))["talker_config"]
    accepted = {"auto"} | {k for k in talker["codec_language_id"] if "dialect" not in k}
    # The model lowercases whatever it is given, so compare case-insensitively.
    unknown = {v.lower() for v in webui.LANGUAGE_VALUES} - {"auto"} - accepted
    check(not unknown, f"dropdown offers languages the model rejects: {sorted(unknown)}")
    for dialect in ("beijing_dialect", "sichuan_dialect"):
        check(dialect not in {v.lower() for v in webui.LANGUAGE_VALUES},
              f"{dialect} is a speaker trait, not a language option")
    check("italian" in {v.lower() for v in webui.LANGUAGE_VALUES}, "italian is supported and must stay selectable")


def test_speaker_choices_come_from_the_checkpoint() -> None:
    import json

    choices = webui.speaker_choices(os.path.join(webui.ROOT, "models"))
    values = [value for _, value in choices]
    check("" not in [label for label, _ in choices], "every speaker has a description label")

    by_name = dict((label.split(" — ")[0].lower().replace(" ", "_"), label) for label, _ in choices)
    cfg_path = os.path.join(webui.ROOT, "models", "Qwen3-TTS-12Hz-1.7B-CustomVoice", "config.json")
    if os.path.exists(cfg_path):
        talker = json.load(open(cfg_path))["talker_config"]
        check(set(values) == set(talker["spk_id"]), "dropdown speakers == checkpoint spk_id table")
        for speaker, dialect in talker["spk_is_dialect"].items():
            if dialect:
                label = by_name[speaker]
                check(dialect.replace("_", " ").lower() in label.lower(),
                      f"{speaker} should advertise its dialect in the label: {label!r}")
    else:
        check(len(choices) == 9, "fallback list covers the 9 released speakers")

    # A missing checkpoint must still yield a usable dropdown.
    empty = webui.speaker_choices(os.path.join(webui.ROOT, "does-not-exist"))
    check(len(empty) == 9 and empty[0][1] in webui.SPEAKER_INFO, "falls back to the static speaker list")


def test_voice_library(tmp_dir=None) -> None:
    import tempfile

    wav = (np.sin(np.linspace(0, 200, 24000)) * 0.3).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        original, webui.VOICES_DIR = webui.VOICES_DIR, tmp
        try:
            check(webui.list_voices() == [], "empty library lists nothing")
            check(webui.voice_choices() == [], "empty library offers no choices")

            rec = webui.save_voice("Sleepy Sea Captain!", wav, 24000, "Ahoy me hearties", "gruff old man", "english")
            check(rec["slug"] == "sleepy_sea_captain", "name is slugified for the filename")
            check(os.path.exists(os.path.join(tmp, "sleepy_sea_captain.wav")), "wav written")
            check(rec["seconds"] == 1.0, "duration recorded from the waveform")

            voices = webui.list_voices()
            check([v["slug"] for v in voices] == ["sleepy_sea_captain"], "listed after save")
            check(voices[0]["text"] == "Ahoy me hearties", "transcript persisted (needed for ICL cloning)")
            check(webui.voice_choices()[0][1] == "sleepy_sea_captain", "dropdown exposes the slug as its value")

            audio, text, msg = webui.load_saved_voice("sleepy_sea_captain")
            check(audio[0] == 24000 and audio[1].shape == (24000,), "load returns (sr, wav) for the Audio component")
            check(text == "Ahoy me hearties", "load returns the stored transcript")
            check("Loaded" in msg, "load reports success")

            # Overwriting the same name replaces rather than accumulating.
            webui.save_voice("sleepy sea captain", wav[:12000], 24000, "shorter", "x", "english")
            check(len(webui.list_voices()) == 1, "same name overwrites instead of duplicating")
            check(webui.list_voices()[0]["seconds"] == 0.5, "overwrite updates the clip")

            _, _, msg = webui.load_saved_voice("does_not_exist")
            check("missing" in msg, "loading an unknown voice errors instead of returning silence")

            _, msg, _ = webui.delete_saved_voice("sleepy_sea_captain")
            check("Deleted" in msg and webui.list_voices() == [], "delete removes both files")

            # A half-written entry must not break the list.
            open(os.path.join(tmp, "broken.json"), "w").write("{not json")
            webui.save_voice("good", wav, 24000, "t", "i", "english")
            check([v["slug"] for v in webui.list_voices()] == ["good"], "corrupt metadata is skipped, not fatal")
        finally:
            webui.VOICES_DIR = original


def test_save_designed_voice_requires_generation() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        original, webui.VOICES_DIR = webui.VOICES_DIR, tmp
        try:
            webui._LAST_DESIGN.clear()
            _, msg, _ = webui.save_designed_voice("nobody")
            check("Nothing to save yet" in msg, "cannot save before generating")

            webui._LAST_DESIGN.update(wav=np.zeros(24000, dtype=np.float32), sr=24000,
                                      text="hello", instruct="warm", language="english")
            _, msg, update = webui.save_designed_voice("   ")
            check("name" in msg.lower(), "blank name rejected")
            check(webui.list_voices() == [], "rejected save writes nothing")

            _, msg, update = webui.save_designed_voice("Warm Narrator")
            check("Saved" in msg, "saving the last design works")
            check(update["choices"][0][1] == "warm_narrator", "clone tab dropdown is refreshed with the new voice")
        finally:
            webui.VOICES_DIR = original
            webui._LAST_DESIGN.clear()


def test_slug_rejects_junk() -> None:
    check(webui._slug("  Two  Words!! ") == "two_words", "punctuation and spacing collapse to underscores")
    for bad in ("", "   ", "!!!", "///"):
        try:
            webui._slug(bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass


def test_memory_guards() -> None:
    """The hub must refuse to over-commit rather than swap/ballooning the VM.

    Regression guard for the WSL2 hard-lock: two processes each pinning all three
    checkpoints exhausted the host. Nothing here loads real weights.
    """
    import tempfile

    check(webui.mem_available_mb() > 0, "MemAvailable is readable")

    with tempfile.TemporaryDirectory() as tmp:
        # A fake checkpoint: only the file sizes matter for the budget math.
        cap_dir = os.path.join(tmp, "Qwen3-TTS-12Hz-1.7B-CustomVoice")
        os.makedirs(cap_dir)
        with open(os.path.join(cap_dir, "model.safetensors"), "wb") as fh:
            fh.truncate(2 * 1024 ** 3)  # 2GB sparse
        for name in ("config.json", "generation_config.json"):
            open(os.path.join(cap_dir, name), "w").write("{}")

        hub = webui.ModelHub(tmp, None, "sdpa", 1, max_cpu_resident=2)
        check(1900 < hub.weight_mb("custom_voice") < 2100, f"size estimated from safetensors: {hub.weight_mb('custom_voice')}")

        # Floor above what is available -> MemoryError before any weights are touched.
        guarded = webui.ModelHub(tmp, None, "sdpa", 1, max_cpu_resident=2,
                                 ram_floor_mb=int(webui.mem_available_mb() + 100_000))
        try:
            guarded._get_cpu("custom_voice")
            raise AssertionError("expected MemoryError, got a load")
        except MemoryError as e:
            check("floor" in str(e), f"refusal explains the floor: {e}")
        check(guarded.cpu_cache == {}, "refused load leaves nothing pinned")

    # _free_cpu must never drop the model we are about to need, nor one on the GPU.
    hub = webui.ModelHub(tmp, None, "sdpa", 1, max_cpu_resident=5, ram_floor_mb=0)
    hub.cpu_cache["a"] = object()
    hub.cpu_cache["b"] = object()
    hub.on_gpu["b"] = hub.cpu_cache["b"]
    hub._free_cpu(10 ** 9, protect="a")
    check("b" in hub.cpu_cache, "a model resident on the GPU is not dropped from the CPU cache")


def test_exclusive_owner_lock() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        lock_path = os.path.join(tmp, "test.lock")
        original, webui.LOCK_PATH = webui.LOCK_PATH, lock_path
        try:
            with webui.exclusive_owner("first"):
                try:
                    with webui.exclusive_owner("second"):
                        raise AssertionError("second holder should have been refused")
                except RuntimeError as e:
                    check("first pid=" in str(e), f"refusal names the holder: {e}")
            # released -> reacquirable
            with webui.exclusive_owner("third"):
                pass
        finally:
            webui.LOCK_PATH = original


def test_media_helpers() -> None:
    import tempfile
    import types

    # --- load_media input validation -------------------------------------------------
    for bad in ("", None, "/nope/missing.mp4"):
        try:
            webui.load_media(bad)
            raise AssertionError(f"expected MediaError for {bad!r}")
        except webui.MediaError:
            pass
    with tempfile.NamedTemporaryFile(suffix=".txt") as fh:
        fh.write(b"not media")
        fh.flush()
        try:
            webui.load_media(fh.name)
            raise AssertionError("expected MediaError for a .txt upload")
        except webui.MediaError as e:
            check("Unsupported file type" in str(e), "unsupported type names the extension")

    with tempfile.TemporaryDirectory() as tmp:
        # --- ffmpeg command construction + failure reporting -------------------------
        calls = {}
        real_run, real_which = webui.subprocess.run, webui.shutil.which

        def fake_run(cmd, **kw):
            calls["cmd"] = list(cmd)
            return types.SimpleNamespace(returncode=0, stderr="")

        try:
            webui.subprocess.run = fake_run
            webui.ffmpeg_extract_audio("/in/movie.mp4", "/out/a.wav", sr=24000)
            cmd = calls["cmd"]
            check(cmd[0] == webui.FFMPEG_BIN, "uses the configured ffmpeg")
            check("-vn" in cmd and "-map" in cmd, "drops the video stream and maps the first audio stream")
            check(cmd[cmd.index("-ac") + 1] == "1" and cmd[cmd.index("-ar") + 1] == "24000",
                  "downmixes to mono at the requested rate")
            check(cmd[-1] == "/out/a.wav" and "pcm_s16le" in cmd, "writes PCM wav")

            webui.subprocess.run = lambda cmd, **kw: types.SimpleNamespace(
                returncode=1, stderr="x\nStream map '0:a:0' matches no streams.")
            try:
                webui.ffmpeg_extract_audio("/in/silent.mp4", "/out/a.wav")
                raise AssertionError("expected MediaError")
            except webui.MediaError as e:
                check("matches no streams" in str(e), f"ffmpeg stderr reaches the user: {e}")

            webui.subprocess.run = real_run
            webui.shutil.which = lambda name: None
            try:
                webui.ffmpeg_extract_audio("/in/movie.mp4", "/out/a.wav")
                raise AssertionError("expected MediaError when ffmpeg is missing")
            except webui.MediaError as e:
                check("ffmpeg" in str(e), "missing ffmpeg says so")
        finally:
            webui.subprocess.run, webui.shutil.which = real_run, real_which

        # --- a video upload goes through ffmpeg, an audio upload does not ------------
        sr = 24000
        wav = (np.sin(np.linspace(0, 300, sr)) * 0.4).astype(np.float32)
        real_extract = webui.ffmpeg_extract_audio
        used_ffmpeg = {"called": False}

        def fake_extract(src, dst, sr=24000):
            used_ffmpeg["called"] = True
            sf.write(dst, wav, sr)
            return dst

        try:
            webui.ffmpeg_extract_audio = fake_extract
            video = os.path.join(tmp, "clip.mp4")
            open(video, "wb").write(b"fake mp4 bytes")
            got, got_sr = webui.load_media(video)
            check(used_ffmpeg["called"] and got_sr == 24000 and len(got) == sr,
                  "video is decoded via ffmpeg to mono float32")

            used_ffmpeg["called"] = False
            audio = os.path.join(tmp, "clip.wav")
            sf.write(audio, wav, sr)
            got, got_sr = webui.load_media(audio)
            check(not used_ffmpeg["called"] and len(got) == sr, "wav is read natively, no ffmpeg")

            # stereo + wrong sample rate -> mono at the target rate
            stereo_path = os.path.join(tmp, "stereo44.wav")
            sf.write(stereo_path, np.stack([wav, wav], axis=-1), 44100)
            got, got_sr = webui.load_media(stereo_path)
            expected_seconds = len(wav) / 44100.0
            check(got_sr == 24000 and got.ndim == 1 and abs(len(got) / 24000 - expected_seconds) < 0.02,
                  f"stereo 44.1k is downmixed and resampled (got {got_sr}Hz, {len(got) / 24000:.2f}s "
                  f"vs {expected_seconds:.2f}s)")
        finally:
            webui.ffmpeg_extract_audio = real_extract


def test_trim_window() -> None:
    # empty / inverted selections become a sensible default window from the top
    check(webui.trim_window(30.0, None, None) == (0.0, 6.0), "empty selection takes the target window")
    check(webui.trim_window(30.0, 5.0, 5.0) == (5.0, 11.0), "zero-length selection grows to target")
    check(webui.trim_window(30.0, 10.0, 4.0) == (10.0, 16.0), "inverted selection grows forward")

    # out-of-range is clamped inside the file
    start, end = webui.trim_window(10.0, -5.0, 99.0)
    check(start == 0.0 and end == 10.0, f"clamped to the file: {start}-{end}")

    # too long shrinks around the centre, too short grows around the centre
    start, end = webui.trim_window(60.0, 20.0, 50.0)
    check(abs((end - start) - webui.REF_MAX_SECONDS) < 1e-6 and abs((start + end) / 2 - 35) < 1e-6,
          f"long selection shrunk to max around its centre: {start}-{end}")
    start, end = webui.trim_window(60.0, 20.0, 20.5)
    check(abs((end - start) - webui.REF_MIN_SECONDS) < 1e-6, f"short selection grown to min: {start}-{end}")

    # near the end of the file the window slides back instead of overflowing
    start, end = webui.trim_window(8.0, 7.9, 8.0)
    check(end == 8.0 and abs((end - start) - webui.REF_MIN_SECONDS) < 1e-6, f"stays inside the file: {start}-{end}")

    # a file shorter than the minimum is used whole
    check(webui.trim_window(1.2, 0.0, 1.2) == (0.0, 1.2), "short clips are not trimmed away")
    try:
        webui.trim_window(0.0, 0, 0)
        raise AssertionError("expected MediaError for empty media")
    except webui.MediaError:
        pass


def test_auto_pick_window() -> None:
    sr = 1000
    silence = np.zeros(sr * 20, dtype=np.float32)
    loud = (np.sin(np.linspace(0, 4000, sr * 5)) * 0.9).astype(np.float32)
    wav = np.concatenate([silence, loud, silence])           # loud from 20s..25s
    start, end = webui.auto_pick_window(wav, sr, target=5.0)
    check(abs(start - 20.0) <= 0.3 and abs(end - start - 5.0) < 1e-6,
          f"should land on the loud stretch, got {start}-{end}")

    short = np.ones(sr, dtype=np.float32) * 0.5
    check(webui.auto_pick_window(short, sr, target=5.0) == (0.0, 1.0), "shorter than target -> whole clip")


def test_render_waveform() -> None:
    sr = 8000
    wav = (np.sin(np.linspace(0, 800, sr * 4)) * 0.6).astype(np.float32)
    img = webui.render_waveform(wav, sr, 1.0, 2.5)
    check(img.size == (980, 180), f"expected canvas, got {img.size}")
    check(img.mode == "RGB", "RGB for gradio")

    plain = np.asarray(webui.render_waveform(wav, sr), dtype=np.uint8)
    shaded = np.asarray(img, dtype=np.uint8)
    check(plain.shape == shaded.shape, "same canvas with or without a selection")
    check(np.abs(plain.astype(int) - shaded.astype(int)).sum() > 0,
          "the shaded selection actually changes the drawing")

    empty = webui.render_waveform(np.zeros(0, dtype=np.float32), sr)
    check(empty.size == (980, 180), "empty waveform must not crash the UI")

    # long files still get tick labels rather than one per second
    long_img = webui.render_waveform(np.zeros(sr * 300, dtype=np.float32), sr)
    check(long_img.size == (980, 180), "10-minute clip renders")


def test_clip_guidance() -> None:
    check("too short" in webui.clip_guidance(1.0), "flags a 1s clip")
    check("good length" in webui.clip_guidance(6.0), "endorses the target length")
    check("long" in webui.clip_guidance(30.0), "flags a 30s clip")


def test_transcriber() -> None:
    check(set(webui.ASR_LANGUAGE_CODES) == {v.lower() for v in webui.LANGUAGE_VALUES} - {"auto"},
          "every UI language has a whisper code")

    disabled = webui.Transcriber(model_id="none")
    check(not disabled.enabled, "'none' disables ASR")
    try:
        disabled.transcribe(np.ones(1000, dtype=np.float32), 24000)
        raise AssertionError("expected MediaError when disabled")
    except webui.MediaError as e:
        check("disabled" in str(e), "disabled ASR explains itself")

    empty = webui.Transcriber(model_id="openai/whisper-large-v3", enabled=False)
    try:
        empty.transcribe(np.zeros(0, dtype=np.float32), 24000)
        raise AssertionError("expected MediaError for an empty clip")
    except webui.MediaError as e:
        check("empty clip" in str(e), "empty clip is rejected before loading the model")


def test_media_state_is_json_serialisable() -> None:
    """gr.State round-trips through the browser as JSON — no numpy, no torch objects."""
    import json
    import tempfile

    sr = 24000
    with tempfile.TemporaryDirectory() as tmp:
        original, webui.MEDIA_DIR = webui.MEDIA_DIR, tmp
        try:
            wav = (np.sin(np.linspace(0, 400, sr * 3)) * 0.5).astype(np.float32)
            upload = os.path.join(tmp, "take.wav")
            sf.write(upload, wav, sr)
            state, status, wave, start, end, audio, info = webui.import_media(upload)
            check("Loaded" in status, f"import reported: {status}")
            json.dumps(state)  # must not raise
            check(state["duration"] == 3.0 and state["sample_rate"] == 24000, "duration/rate recorded")
            check(os.path.exists(state["path"]), "clip persisted for later requests")
            check(isinstance(audio, tuple) and audio[1].shape[0] > 0, "trimmed clip returned for the Audio box")

            # later requests arrive with the JSON-decoded state, exactly as the browser sends it
            from_browser = json.loads(json.dumps(state))
            wave2, audio2, info2, start_update, end_update = webui.update_clip(from_browser, 1.0, 3.0)
            s2, e2 = start_update["value"], end_update["value"]   # slider updates, clamped by the server
            check(abs(s2 - 1.0) < 1e-6 and abs(e2 - 3.0) < 1e-6, f"re-slice works from decoded state: {s2}-{e2}")
            check(audio2[1].shape[0] == 48000, f"two seconds of audio returned, got {audio2[1].shape}")

            # a selection shorter than the clone minimum is grown, not rejected
            _, _, _, grown_start, grown_end = webui.update_clip(from_browser, 1.0, 1.5)
            check(abs(grown_end["value"] - grown_start["value"] - webui.REF_MIN_SECONDS) < 1e-6,
                  f"sub-minimum selection grown to the floor: {grown_start['value']}-{grown_end['value']}")
        finally:
            webui.MEDIA_DIR = original


def test_import_media_reports_errors() -> None:
    # A broken upload must come back as a status message, not an exception that kills the request
    state, status, wave, start, end, audio, info = webui.import_media("/definitely/not/here.mp4")
    check(state is None and "Error" in status, "missing file reported in the status box")
    check(wave is None and audio is None, "nothing half-loaded")

    _, _, info_out, *_ = webui.update_clip(None, 0, 5)
    check("Upload a clip first" in info_out, "trimming before uploading is explained")

    result = webui.auto_pick_clip(None)
    check("Error" in result[2], "auto-pick without media explains itself")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("\n", "all passed" if not failures else f"{failures} failed")
    raise SystemExit(1 if failures else 0)
