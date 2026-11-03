# coding=utf-8
"""
Fast (CPU-only, no model weights) tests for the WebUI helper logic.

Covers the boundaries that matter for usability:
  * _budget()  — token budget never drops below the floor nor exceeds the cap,
                 so short prompts still work and long prompts cannot run for minutes.
  * _report()  — a generation that hits the cap is flagged as a non-stopping run.
  * _audio_to_tuple / _normalize_audio — gradio audio payloads become clean mono
                 float32 in [-1, 1] at the declared sample rate, whatever the input form.
  * _clone_cache_key — same reference audio hits the cache, different audio does not
                 (including same-length, same-sum audio, which the old key collided on).
  * resolve_input_path / load_media / _audio_to_tuple — a browser-named path is only
                 ever read from inside the upload dirs (belt to Gradio's own check).
  * prune_media — extracted clips do not accumulate on disk forever.
  * build_ui — the Blocks tree constructs against the installed Gradio.
  * ModelHub.acquire — the VRAM budget only applies to a CUDA target (--device cpu).
  * setup.sh — its download list still matches CAPABILITIES.
  * smoke_test.skip() — a skipped GPU case cannot report success.

Run: .venv/bin/python tests/test_app_logic.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

import gradio as gr
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
    # Same length, same sample multiset (so the same sum), different order: the old
    # (length, float-sum) key collided here, so one user could be served another user's
    # cached clone prompt.
    ramp = np.linspace(-0.1, 0.1, 24000, dtype=np.float32)
    check(webui._clone_cache_key((ramp, 24000), "hello", False)
          != webui._clone_cache_key((np.roll(ramp, 1), 24000), "hello", False),
          "same-length, same-sum, reordered audio must not reuse the cached prompt")


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


def test_save_designed_voice_uses_the_audio_box() -> None:
    """'Save as voice' reads the clip from its inputs, never from a process global.

    It used to persist a module-level "last design" cache, so with two people on the LAN
    one person's Save button could persist the other person's clip.
    """
    wav = (np.sin(np.linspace(0, 200, 24000)) * 0.3).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        original, webui.VOICES_DIR = webui.VOICES_DIR, tmp
        original_media, webui.MEDIA_DIR = webui.MEDIA_DIR, tmp
        try:
            clip_path = os.path.join(tmp, "design.wav")
            sf.write(clip_path, wav, 24000)

            _, msg, _ = webui.save_designed_voice("nobody", None, "text", "warm", "english")
            check("Nothing to save yet" in msg, "cannot save without a clip in the box")

            _, msg, _ = webui.save_designed_voice("   ", clip_path, "text", "warm", "english")
            check("name" in msg.lower(), "blank name rejected")
            check(webui.list_voices() == [], "rejected save writes nothing")

            _, msg, _ = webui.save_designed_voice("x", clip_path, "   ", "warm", "english")
            check("empty" in msg, "a clip with no transcript is rejected (ICL needs it)")
            check(webui.list_voices() == [], "rejected save writes nothing")

            _, msg, update = webui.save_designed_voice("Warm Narrator", clip_path,
                                                       "hello", "warm", "english")
            check("Saved" in msg, "saving the clip in the box works")
            check(update["choices"][0][1] == "warm_narrator", "clone tab dropdown is refreshed with the new voice")
            saved = webui.list_voices()[0]
            check(saved["text"] == "hello" and saved["instruct"] == "warm",
                  "transcript and description come from the Voice Design inputs")

            # A path outside the allowed roots is refused, not read.
            _, msg, _ = webui.save_designed_voice("spy", "/opt/recordings/take.wav",
                                                  "hello", "warm", "english")
            check("outside the upload directory" in msg, f"foreign path refused: {msg}")
            check(webui.list_voices() == [saved], "the refused save added nothing")
        finally:
            webui.VOICES_DIR = original
            webui.MEDIA_DIR = original_media


def test_save_reference_voice_from_the_clone_tab() -> None:
    """A voice built from an upload/video must be saveable, not just loadable.

    The Clone tab had Load and Delete but no Save, so a reference you trimmed and
    transcribed could not be kept — only designed voices could be saved.
    """
    wav = (np.sin(np.linspace(0, 200, 24000 * 5)) * 0.3).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        original, webui.VOICES_DIR = webui.VOICES_DIR, tmp
        original_media, webui.MEDIA_DIR = webui.MEDIA_DIR, tmp
        try:
            clip_path = os.path.join(tmp, "reference.wav")
            sf.write(clip_path, wav, 24000)
            short_path = os.path.join(tmp, "short.wav")
            sf.write(short_path, wav[:int(24000 * (webui.REF_MIN_SECONDS - 0.5))], 24000)

            _, msg, _ = webui.save_reference_voice("nobody", None, "text", "English")
            check("Nothing to save" in msg, "cannot save without a reference clip")

            _, msg, _ = webui.save_reference_voice("  ", clip_path, "text", "English")
            check("name" in msg.lower(), "blank name rejected")

            _, msg, _ = webui.save_reference_voice("tiny", short_path, "text", "English")
            check("too short" in msg, f"a sub-minimum clip is refused: {msg}")
            check(webui.list_voices() == [], "rejected saves write nothing")

            _, msg, update = webui.save_reference_voice("Phone Memo", clip_path,
                                                        "the quick brown fox", "English")
            check("Saved" in msg and "5" in msg, f"saving the reference clip works: {msg}")
            check(update["choices"][0][1] == "phone_memo", "dropdown refreshed with the new voice")
            saved = webui.list_voices()[0]
            check(saved["text"] == "the quick brown fox" and saved["language"] == "English",
                  "the transcript and language are persisted with the clip")
            check(saved["instruct"] == "", "a cloned reference has no voice description")

            # A reference with no transcript is still worth keeping, but say what it costs.
            _, msg, _ = webui.save_reference_voice("No Words", clip_path, "", "Auto")
            check("x-vector only" in msg, f"a transcript-less save explains the consequence: {msg}")

            _, msg, _ = webui.save_reference_voice("spy", "/opt/recordings/take.wav",
                                                   "hello", "English")
            check("outside the upload directory" in msg, f"foreign path refused: {msg}")
            check(len(webui.list_voices()) == 2, "the refused save added nothing")

            # Round trip: what the Clone tab saved, the Clone tab can load back.
            clip, text, status = webui.load_saved_voice("phone_memo")
            check(text == "the quick brown fox", "loading it back restores the transcript")
            check(isinstance(clip, tuple) and clip[0] == 24000, "and the clip at its sample rate")
        finally:
            webui.VOICES_DIR = original
            webui.MEDIA_DIR = original_media


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
        # (Still inside the temp dir: ModelHub is built from `tmp`.)
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
    import types

    with tempfile.TemporaryDirectory() as tmp:
        # Uploads are only read from inside the allowed roots (see resolve_input_path),
        # so every fixture file in this test lives in the media dir.
        original_media, webui.MEDIA_DIR = webui.MEDIA_DIR, tmp
        try:
            _media_helpers_body(tmp)
        finally:
            webui.MEDIA_DIR = original_media


def _media_helpers_body(tmp: str) -> None:
    # --- load_media input validation -------------------------------------------------
    for bad in ("", None):
        try:
            webui.load_media(bad)
            raise AssertionError(f"expected MediaError for {bad!r}")
        except webui.MediaError:
            pass
    missing = os.path.join(tmp, "nope.mp4")
    try:
        webui.load_media(missing)
        raise AssertionError("expected MediaError for a missing upload")
    except webui.MediaError as e:
        check("not found" in str(e).lower(), f"missing file is explained: {e}")
    with tempfile.NamedTemporaryFile(suffix=".txt", dir=tmp, delete=False) as fh:
        fh.write(b"not media")
    try:
        webui.load_media(fh.name)
        raise AssertionError("expected MediaError for a .txt upload")
    except webui.MediaError as e:
        check("Unsupported file type" in str(e), "unsupported type names the extension")

    # --- ffmpeg command construction + failure reporting -----------------------------
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

    # --- a video upload goes through ffmpeg, an audio upload does not ----------------
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

        # the ffmpeg scratch file is cleaned up out of the media dir
        check(not [f for f in os.listdir(tmp) if f.startswith("qwen-tts-media-")],
              "the extracted scratch wav is deleted after import")
    finally:
        webui.ffmpeg_extract_audio = real_extract


def test_client_paths_are_confined_to_the_upload_dirs() -> None:
    """Our own guard on browser-named paths (belt to Gradio's braces).

    Gradio 6's preprocess_data already refuses a path no user uploaded, but the File and
    Audio components themselves do not check `payload.path` at all — so if a future
    Gradio loosens preprocess_data, this is the only thing standing between a client and
    an arbitrary .wav/.mp4 read off this machine. Regression guard for that day.
    """
    with tempfile.TemporaryDirectory() as tmp:
        original_media, webui.MEDIA_DIR = webui.MEDIA_DIR, tmp
        try:
            # An absolute path with a legal extension, outside every allowed root.
            for outside in ("/etc/hostname.wav", "/opt/recordings/secret.mp3",
                            os.path.join(tempfile.gettempdir(), "elsewhere.wav")):
                try:
                    webui.load_media(outside)
                    raise AssertionError(f"expected MediaError for {outside!r}")
                except webui.MediaError as e:
                    check("outside the upload directory" in str(e),
                          f"refusal explains containment, not existence: {e}")

            # A symlink planted inside the media dir must not escape it (realpath check).
            wav = (np.sin(np.linspace(0, 200, 24000)) * 0.3).astype(np.float32)
            real = os.path.join(tmp, "real.wav")
            sf.write(real, wav, 24000)
            link = os.path.join(tmp, "link.wav")
            os.symlink(real, link)
            got, got_sr = webui.load_media(link)
            check(len(got) == 24000, "a symlink inside the media dir still loads")

            escape = os.path.join(tmp, "escape.wav")
            os.symlink("/etc/hostname", escape)
            try:
                webui.load_media(escape)
                raise AssertionError("expected MediaError for a symlink escape")
            except webui.MediaError as e:
                check("outside the upload directory" in str(e), f"symlink escape refused: {e}")

            # URLs are not paths; ffmpeg would happily fetch them.
            try:
                webui.load_media("http://evil.example/x.wav")
                raise AssertionError("expected MediaError for a URL")
            except webui.MediaError as e:
                check("URLs are not accepted" in str(e), f"URL rejected: {e}")

            # The audio path used by the reference box goes through the same guard.
            try:
                webui._audio_to_tuple("/opt/recordings/secret.wav")
                raise AssertionError("expected MediaError from _audio_to_tuple")
            except webui.MediaError as e:
                check("outside the upload directory" in str(e), f"audio box guarded: {e}")

            # Inside the media dir, both entry points work.
            check(webui._audio_to_tuple(real)[1] == 24000, "a clip in the media dir loads")
        finally:
            webui.MEDIA_DIR = original_media


def test_gradio_roundtrip_paths_are_accepted() -> None:
    """The containment rule must not lock us out of the files Gradio writes for us.

    Every (sr, wav) handed to a clip box is written into Gradio's component cache by
    postprocess, and the browser sends that exact path back as an input on the next
    click. If allowed_input_roots() ever stops covering it, 'Clone & generate' starts
    refusing our own clips — a self-inflicted outage, not a security win.
    """
    sr = 24000
    wav = (np.sin(np.linspace(0, 300, sr)) * 0.4).astype(np.float32)
    box = gr.Audio(type="filepath")
    filedata = box.postprocess((sr, wav))
    path = getattr(filedata, "path", None) or (filedata or {}).get("path")
    check(bool(path), "gradio wrote the clip out")
    got, got_sr = webui._audio_to_tuple(path)
    check(got_sr == sr and len(got) > 0, f"our own output round-trips ({got_sr}Hz, {len(got)} samples)")
    check(os.path.realpath(str(path)) in webui.allowed_input_roots()
          or any(os.path.realpath(str(path)).startswith(r + os.sep)
                 for r in webui.allowed_input_roots()),
          "gradio's component cache is an allowed input root")


def test_prune_media() -> None:
    import os as _os
    import time as _time

    with tempfile.TemporaryDirectory() as tmp:
        original_media, webui.MEDIA_DIR = webui.MEDIA_DIR, tmp
        try:
            fresh = os.path.join(tmp, "fresh.wav")
            old = os.path.join(tmp, "old.wav")
            for path in (fresh, old):
                sf.write(path, np.zeros(100, dtype=np.float32), 24000)
            past = _time.time() - 7200
            _os.utime(old, (past, past))
            check(webui.prune_media(max_age_seconds=3600) == 1, "only the stale clip goes")
            check(_os.path.exists(fresh) and not _os.path.exists(old), "fresh survives, stale does not")
            check(webui.prune_media(max_age_seconds=0) == 0, "TTL 0 disables pruning")
            webui.MEDIA_DIR = os.path.join(tmp, "does-not-exist")
            check(webui.prune_media() == 0, "missing media dir is not an error")
        finally:
            webui.MEDIA_DIR = original_media



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


def test_build_ui_constructs() -> None:
    """The whole Blocks tree must build against the installed Gradio.

    Catches the drift class that bit us once: Gradio 6 moved `css` from Blocks() to
    launch() and dropped `show_api`, and a wrong component kwarg or a mis-wired
    .click() input list only shows up when a human clicks. This runs in ~1s, CPU-only.
    """
    class StubHub:
        def status(self) -> str:
            return "stub"

    original, webui.HUB = webui.HUB, StubHub()
    try:
        demo = webui.build_ui()
    finally:
        webui.HUB = original

    check(demo is not None, "build_ui returned a Blocks instance")
    check(len(demo.fns) >= 13, f"every button/slider is wired (found {len(demo.fns)} listeners)")

    audios = [b for b in demo.blocks.values() if isinstance(b, gr.Audio)]
    by_label = {b.label: b for b in audios}
    # CustomVoice + VoiceDesign + Clone output + Clone reference box + the four candidate
    # players that make best-of-N auditionable.
    check(len(audios) == 8, f"three outputs, one reference input, four take players — found {len(audios)}")
    for label in ("Reference clip (the trimmed selection lands here)", "Generated voice"):
        check(label in by_label, f"{label} exists")
    # Anything we read back from the browser is filepath-typed so that we open it
    # ourselves, after resolve_input_path; with type="numpy" Gradio opens the client's
    # path before our code runs, so we never get a chance to check it.
    check(by_label["Reference clip (the trimmed selection lands here)"].type == "filepath",
          "the reference box is filepath-typed so resolve_input_path can guard it")
    check(by_label["Generated voice"].type == "filepath",
          "the Voice Design box is filepath-typed: 'Save as voice' reads it back")


def test_acquire_works_without_cuda() -> None:
    """`--device cpu` used to refuse every load: the VRAM gate ran against 0MB free."""
    import torch

    class FakeTTS:
        pass

    hub = webui.ModelHub("/nonexistent", None, "sdpa", 1)
    hub._get_cpu = lambda capability: FakeTTS()
    hub.weight_mb = lambda capability: 4000.0
    hub._move = lambda tts, device: None

    original = webui.DEVICE
    try:
        webui.DEVICE = torch.device("cpu")
        got = hub.acquire("custom_voice")
        check(isinstance(got, FakeTTS), "cpu target returns the model")
        check("custom_voice" in hub.on_gpu, "cpu target is tracked as resident")

        if not torch.cuda.is_available():
            # With the gate back on and no VRAM at all, it must still refuse loudly.
            hub.on_gpu.clear()
            webui.DEVICE = torch.device("cuda")
            try:
                hub.acquire("voice_design")
                raise AssertionError("expected MemoryError with no VRAM")
            except MemoryError as e:
                check("VRAM" in str(e), f"cuda target still enforces the margin: {e}")
    finally:
        webui.DEVICE = original


def test_default_speaker_is_offered() -> None:
    """The preselected speaker must be one the checkpoint actually has."""
    values = {value for _, value in webui.speaker_choices(os.path.join(webui.ROOT, "does-not-exist"))}
    check(webui.default_speaker(os.path.join(webui.ROOT, "does-not-exist")) in values,
          "fallback default is in the fallback list")
    real = webui.default_speaker(webui.ROOT)
    check(real in {value for _, value in webui.speaker_choices(webui.ROOT)},
          "default is offered by the checkpoint (or the fallback list)")


def test_smoke_test_skips_are_not_passes() -> None:
    """A skipped GPU case must not report success (house rule: a skip is not a pass).

    `--only media_pipeline` on a box without ffmpeg used to print SKIPPED and still exit
    0 saying "All smoke tests passed". The GPU work is stubbed out here; only the
    bookkeeping around it is under test.
    """
    import contextlib
    import io
    import smoke_test
    import types

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(_index=0):
            return "fake-gpu"

        @staticmethod
        def get_device_capability(_index=0):
            return (12, 0)

    real_torch, real_run_case, real_hub = smoke_test.torch, smoke_test.run_case, webui.ModelHub
    real_argv = sys.argv
    real_owner = smoke_test.webui.exclusive_owner
    smoke_test.torch = types.SimpleNamespace(cuda=FakeCuda, bfloat16=object(), __version__="0.0-test")
    smoke_test.webui.ModelHub = lambda *a, **kw: types.SimpleNamespace(lock=None, on_gpu={})

    # The single-owner lock is production behaviour, not what this test is about: without
    # stubbing it the test fails whenever the webui server happens to be running.
    @contextlib.contextmanager
    def fake_owner(_who):
        yield

    smoke_test.webui.exclusive_owner = fake_owner
    try:
        last_output = [""]

        def run(skip_it: bool):
            smoke_test.SKIPPED.clear()
            smoke_test.run_case = (lambda hub, cap, keep_gpu, use_asr=True:
                                   smoke_test.skip(cap, "stubbed") if skip_it else None)
            sys.argv = ["smoke_test.py", "--only", "custom_voice"]
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rc = smoke_test.main()
            last_output[0] = out.getvalue()
            return rc

        check(run(skip_it=False) == 0, "a full run exits 0")
        check(run(skip_it=True) == 1, "a skipped case exits non-zero, not 0")
        check("SKIPPED custom_voice" in last_output[0], "and it says which case was skipped")

        smoke_test.SKIPPED.clear()
        smoke_test.run_case = lambda hub, cap, keep_gpu, use_asr=True: smoke_test.skip(cap, "stubbed")
        sys.argv = ["smoke_test.py", "--only", "custom_voice", "--allow-skip"]
        with contextlib.redirect_stdout(io.StringIO()):
            check(smoke_test.main() == 0, "--allow-skip is the explicit opt-out")
        check(smoke_test.SKIPPED and smoke_test.SKIPPED[0][0] == "custom_voice",
              "the skip is recorded with its case name")
    finally:
        smoke_test.torch, smoke_test.run_case = real_torch, real_run_case
        smoke_test.webui.ModelHub = real_hub
        smoke_test.webui.exclusive_owner = real_owner
        sys.argv = real_argv
        smoke_test.SKIPPED.clear()


def test_setup_sh_knows_every_capability() -> None:
    """setup.sh downloads what app.py can load; the two lists must not drift."""
    setup = os.path.join(webui.ROOT, "setup.sh")
    with open(setup, encoding="utf-8") as fh:
        body = fh.read()
    for capability, (dirname, _) in webui.CAPABILITIES.items():
        check(dirname in body, f"setup.sh does not download the {capability} checkpoint ({dirname})")
    check("Qwen3-TTS-Tokenizer-12Hz" in body, "setup.sh downloads the 12Hz tokenizer")
    for needle in ("whisper-large-v3", '--exclude "*fp32*"'):
        check(needle in body, f"setup.sh is missing {needle!r} (README documents it)")


def test_resolve_credentials() -> None:
    """The UI password must never have to appear in argv (visible via ps/proc)."""
    import contextlib
    import io
    import stat
    import tempfile
    from types import SimpleNamespace

    def args(username=None, password=None, password_file=None):
        return SimpleNamespace(username=username, password=password, password_file=password_file)

    def write_secret(tmp, text="s3cr3t-pass\n", mode=0o600):
        path = os.path.join(tmp, "pw")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(path, mode)
        return path

    def run(a, environ, prompted="from-prompt"):
        out = io.StringIO()
        asked = {}

        def fake_prompt(label):
            asked["label"] = label
            return prompted

        with contextlib.redirect_stdout(out):
            result = webui.resolve_credentials(a, environ=environ, prompt=fake_prompt)
        return result, out.getvalue(), asked

    with tempfile.TemporaryDirectory() as tmp:
        secret_file = write_secret(tmp, "from-file\n")

        # no username -> no auth, whatever else was given
        result, log, _ = run(args(password="x"), {})
        check(result is None, "a password without a username does not enable auth")
        check("UNAUTHENTICATED" in log, "and says so")

        # priority: file > env > flag > prompt
        result, log, _ = run(args(username="u", password="from-flag", password_file=secret_file),
                             {"QWEN_TTS_PASSWORD": "from-env"})
        check(result == ("u", "from-file"), f"password file must win: {result}")
        check("from-flag" not in log and "from-env" not in log, "the secret is never logged")

        result, log, _ = run(args(username="u", password="from-flag"), {"QWEN_TTS_PASSWORD": "from-env"})
        check(result == ("u", "from-env"), f"env must beat the flag: {result}")

        result, log, _ = run(args(username="u", password="from-flag"), {})
        check(result == ("u", "from-flag"), f"flag still works as a last resort: {result}")
        check("ps" in log and "DEPRECATED" not in log, "the argv warning fires only for the flag path")

        result, log, asked = run(args(username="u"), {})
        check(result == ("u", "from-prompt"), f"prompt fills in when nothing else is given: {result}")
        check("u" in asked.get("label", "") and "hidden" in asked.get("label", ""),
              f"prompt names the user and says input is hidden: {asked}")
        check("from-prompt" not in log, "the prompted secret is not echoed")

        # failure modes must fail closed (no auth) rather than half-working
        empty = write_secret(tmp, "\n")
        result, log, _ = run(args(username="u", password_file=empty), {})
        check(result is None and "UNAUTHENTICATED" in log, "an empty secret file does not enable auth")

        result, log, _ = run(args(username="u", password_file=os.path.join(tmp, "nope")), {})
        check(result is None and "cannot read" in log, "a missing secret file is reported, not ignored")

        loose = write_secret(tmp, "loose-secret\n", mode=0o644)
        result, log, _ = run(args(username="u", password_file=loose), {})
        check(result == ("u", "loose-secret"), "a loose file still works (usable default)")
        check("chmod 600" in log, "but it is called out")

        tight = write_secret(tmp, "tight\n", mode=0o600)
        _, log, _ = run(args(username="u", password_file=tight), {})
        check("chmod 600" not in log, "no warning when permissions are already tight")


def test_credentials_come_from_argv_free_sources() -> None:
    """Guard the point of the change: the docs/help steer to file/env, not argv."""
    import inspect

    source = inspect.getsource(webui.main)
    check("--password-file" in source and "QWEN_TTS_PASSWORD" in source,
          "main wires the argv-free sources")
    check("resolve_credentials(args)" in source, "main resolves credentials through the helper")
    check("[(args.username, args.password)]" not in source,
          "the raw argv credential path is gone")


def test_transcript_fit() -> None:
    """The guard against the one failure mode that ruins a clone: transcript != clip.

    Measured on the GPU box: ICL with a mismatched transcript gave 17.4s of audio for a
    5s sentence and WER 1.000 (vs 0.000 matched). Everything else — sliders, temperature,
    non-streaming mode — moved the metrics far less.
    """
    good = " ".join(f"word{i}" for i in range(15))          # 15 words ~ 6s at 2.5 w/s
    check(webui.transcript_fit(6.0, good) == "", "a matching transcript is not flagged")
    check(webui.transcript_fit(6.0, good, x_vector_only=True) == "",
          "x-vector mode ignores the transcript, so nothing to check")
    check(webui.transcript_fit(0.0, good) == "", "no selection yet, no complaint")

    msg = webui.transcript_fit(6.0, "")
    check("No transcript" in msg and "x-vector" in msg, f"empty transcript is flagged: {msg}")

    msg = webui.transcript_fit(6.0, "hello there")
    check("too short" in msg, f"a two-word transcript for 6s is flagged: {msg}")

    msg = webui.transcript_fit(2.0, good)
    check("too long" in msg, f"15 words cannot fit in 2s: {msg}")

    # Boundaries are a factor of 2 either way, so a brisk or slow speaker is not nagged.
    check(webui.transcript_fit(6.0, " ".join(f"w{i}" for i in range(8))) == "",
          "8 words in 6s (1.3 w/s) is inside the window")
    check(webui.transcript_fit(6.0, " ".join(f"w{i}" for i in range(29))) == "",
          "29 words in 6s (4.8 w/s) is inside the window")
    check(webui.transcript_fit(6.0, " ".join(f"w{i}" for i in range(31))) != "",
          "31 words in 6s is past the window")

    # CJK counts characters, not whitespace-separated words, or every Chinese clip would warn.
    chinese = "我" * 20
    check(webui.transcript_fit(5.0, chinese) == "", "20 Chinese characters in 5s reads as normal")
    check("too short" in webui.transcript_fit(12.0, chinese), "the same 20 characters in 12s is thin")
    mixed = good + " " + "我" * 12
    check(webui.transcript_fit(6.0, mixed) == "", "mixed scripts take the better-fitting rate")

    check(webui.speech_units("two words 你好") == (2, 2), "word/character counting splits scripts")
    check(webui.speech_units("") == (0, 0), "empty text has no units")


def test_word_error_rate() -> None:
    """The scoring metric behind best-of-N; it has to behave like the usual WER."""
    wer = webui.word_error_rate
    check(wer("Hello there friend", "Hello there friend") == 0.0, "identical text scores 0")
    check(abs(wer("Hello there mate", "Hello there friend") - 1 / 3) < 1e-9,
          "one substitution in three words is 0.33")
    check(abs(wer("hello THERE friend", "Hello there friend") - 0.0) < 1e-9,
          "case is not an error")
    check(abs(wer("Hello, there! friend.", "Hello there friend") - 0.0) < 1e-9,
          "punctuation is not an error")
    check(wer("", "Hello there friend") == 1.0, "nothing said is 100% wrong")
    check(wer("extra words here that were never asked for", "Hello there friend") > 1.0,
          "rambling exceeds 1.0, which is how the mismatch case shows up")
    check(wer("", "") == 0.0, "two empties agree")
    check(abs(wer("你好世界", "你好世界") - 0.0) < 1e-9, "CJK compares per character")


def test_choose_best_candidate() -> None:
    """Content fidelity wins over timbre; similarity breaks ties."""
    check(webui.choose_best_candidate([{"wer": 0.2, "sim": 0.99},
                                       {"wer": 0.0, "sim": 0.90}]) == 1,
          "a clean take beats a closer-sounding one that mispronounces")
    check(webui.choose_best_candidate([{"wer": 0.1, "sim": 0.95},
                                       {"wer": 0.1, "sim": 0.98}]) == 1,
          "equal WER breaks the tie on speaker similarity")
    check(webui.choose_best_candidate([{"wer": None, "sim": 0.91},
                                       {"wer": None, "sim": 0.97}]) == 1,
          "with ASR off, similarity alone decides")
    check(webui.choose_best_candidate([{"wer": None, "sim": None}]) == 0,
          "a single unscored candidate is still chosen")
    check(webui.choose_best_candidate([{"wer": 0.5, "sim": None},
                                       {"wer": 0.1, "sim": None}]) == 1,
          "WER still ranks when similarity is unavailable")
    try:
        webui.choose_best_candidate([])
        check(False, "empty candidate list should raise")
    except ValueError:
        check(True, "empty candidate list raises instead of returning a bogus index")


def test_clone_tab_wires_the_quality_controls() -> None:
    """Drift guard: the fit check and best-of-N must stay wired to real components."""
    import inspect

    source = inspect.getsource(webui.build_ui)
    check("cl_fit" in source and "clip_fit_status" in source,
          "the Fit check box exists and is driven by the live checker")
    check("cl_candidates" in source and "cl_seed" in source,
          "best-of-N and seed controls are in the Clone tab")
    check("+ [cl_candidates, cl_seed]" in source,
          "they are actually passed into generate_voice_clone")
    check(source.count("clip_fit_status") >= 5,
          "the fit check is wired to transcript, sliders, x-vector and the post-action chains")


def test_postprocess_output() -> None:
    """The 'digital' fix: less near-Nyquist imaging, same length, same likeness.

    The likeness half of that claim is measured in the GPU tier (speaker-encoder cosine);
    here we pin the signal-shape guarantees that make the claim meaningful.
    """
    rng = np.random.default_rng(1)
    sr = 24000
    t = np.arange(int(1.5 * sr)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    hiss = (0.05 * rng.standard_normal(len(t))).astype(np.float32)   # energy up to Nyquist
    wav = np.clip(tone + hiss, -1, 1).astype(np.float32)

    def near_nyquist(x, rate):
        s = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        f = np.fft.rfftfreq(len(x), 1.0 / rate)
        top = f > 0.95 * (rate / 2.0)
        return float(s[top].sum() / (s[f > 0].sum() + 1e-12))

    raw, raw_sr = webui.postprocess_output(wav, sr, "off")
    check(raw_sr == sr and np.array_equal(raw, wav), "off returns the vocoder signal untouched")

    clean, clean_sr = webui.postprocess_output(wav, sr, "clean")
    check(clean_sr == webui.POSTPROCESS_SR, "clean resamples to 48kHz")
    check(abs(len(clean) / clean_sr - len(wav) / sr) < 0.01, "clean preserves duration")
    check(np.all(np.isfinite(clean)) and float(np.max(np.abs(clean))) <= 1.0,
          "clean stays finite and inside full scale")

    warm, warm_sr = webui.postprocess_output(wav, sr, " Warm ")
    check(warm_sr == webui.POSTPROCESS_SR, "warm outputs 48kHz")
    check(abs(len(warm) / warm_sr - len(wav) / sr) < 0.02, "warm preserves duration")
    check(np.all(np.isfinite(warm)) and float(np.max(np.abs(warm))) <= 1.0,
          "warm stays finite and inside full scale")
    check(near_nyquist(warm, warm_sr) < near_nyquist(raw, raw_sr) * 0.5,
          f"warm must actually cut the near-Nyquist imaging "
          f"({near_nyquist(warm, warm_sr):.4f} vs {near_nyquist(raw, raw_sr):.4f})")

    # Robustness: this is polish, so nothing here may be able to break a generation.
    # Blank means "not specified" -> the default; unrecognised -> the raw signal, because
    # applying filters nobody asked for is the worse mistake.
    for mode in ("", None):
        out, out_sr = webui.postprocess_output(wav, sr, mode)
        check(out_sr in (sr, webui.POSTPROCESS_SR) and np.all(np.isfinite(out)),
              f"blank mode '{mode}' falls back to the default without raising")
    for mode in ("nonsense", "loud", "0"):
        out, out_sr = webui.postprocess_output(wav, sr, mode)
        check(out.shape == wav.shape and out_sr == sr,
              f"unrecognised mode '{mode}' returns the raw signal rather than guessing")
    off_upper, off_sr = webui.postprocess_output(wav, sr, " OFF ")
    check(off_sr == sr and np.array_equal(off_upper, wav), "mode matching ignores case and spaces")
    empty, empty_sr = webui.postprocess_output(np.zeros(0, dtype=np.float32), sr, "warm")
    check(empty.size == 0 and empty_sr == sr, "an empty clip passes through")
    tiny, tiny_sr = webui.postprocess_output(np.array([0.1], dtype=np.float32), sr, "warm")
    check(np.all(np.isfinite(tiny)) and tiny_sr in (sr, webui.POSTPROCESS_SR),
          "a one-sample clip does not blow up the filters")

    hot = np.full(sr, 0.99, dtype=np.float32)
    hot_out, _ = webui.postprocess_output(hot, sr, "warm")
    check(float(np.max(np.abs(hot_out))) <= 1.0, "a hot input is never left clipping")

    check(set(webui.POSTPROCESS_MODES) == set(webui.POSTPROCESS_LABELS),
          "every mode has a label for the dropdown")
    check(webui.POSTPROCESS_DEFAULT in webui.POSTPROCESS_MODES, "the default is a real mode")
    check("off" in webui.POSTPROCESS_MODES, "the raw signal stays reachable for A/B and for "
                                            "saving library voices off the model's own decode")


def test_output_options_are_wired() -> None:
    """All three tabs must offer the post-processing choice, not just one."""
    import inspect

    source = inspect.getsource(webui.build_ui)
    check("s_postproc" in source and "output_options" in source, "the dropdown exists")
    check(source.count("+ output_options") == 3,
          "all three generation tabs (CustomVoice, VoiceDesign, Clone) receive the choice")


def test_clone_tab_reads_top_to_bottom() -> None:
    """The Clone tab must read as the three steps it actually is.

    It used to put 'Save this reference as a voice' between the reference section and the
    synthesis section, which made people hunt for the save button in the wrong place and
    wonder what 'Target text' meant. Order is the documentation here, so pin it.
    """
    import inspect

    source = inspect.getsource(webui.build_ui)

    def at(needle: str) -> int:
        index = source.find(needle)
        assert index >= 0, f"missing {needle!r} in the Clone tab"
        return index

    steps = [at("#### 1 · The reference"), at("#### 2 · What it should say"),
             at("#### 3 · Keep it")]
    check(steps == sorted(steps), "the three step headings appear in order")
    check(at('cl_text = gr.Textbox') > at("#### 2 · What it should say"),
          "the text to synthesise sits under step 2, not the reference section")
    check(at('cl_voice_name = gr.Textbox') > at('cl_text = gr.Textbox'),
          "the save controls come after the synthesis controls, not between them")
    check(at('cl_voice_name = gr.Textbox') > at("#### 3 · Keep it"),
          "saving lives in the library step, next to Load/Delete")
    check(at('cl_save_btn = gr.Button') < at('cl_voice = gr.Dropdown'),
          "Save sits directly above the saved-voices list it feeds")
    check(at('cl_audio = gr.Audio') < at('cl_voice_name = gr.Textbox'),
          "the generated audio stays next to the button that produced it")

    # The two text boxes must not be confusable.
    check("Say this in the cloned voice" in source, "the synthesis box says what it is for")
    check("The reference clip says" in source, "the transcript box says it is what the clip says")
    check('label="Target text"' not in source, "'Target text' is gone — it named nothing")


def test_take_labels_explain_the_pick() -> None:
    """The scorer's decision has to be arguable, so the numbers go on the player."""
    full = webui.take_label(2, 0.133, 0.993, True)
    check(full.startswith("Take #2"), f"names the take: {full}")
    check("WER 0.13" in full and "SIM 0.993" in full, f"shows both scores: {full}")
    check("(kept)" in full, "marks the one that was auto-picked")
    check("(kept)" not in webui.take_label(1, 0.0, 0.99, False), "only the winner is marked")
    noscore = webui.take_label(3, None, None, False)
    check(noscore == "Take #3", f"unscored candidates still get a sane label: {noscore}")
    check("WER" in webui.take_label(1, 0.0, None, False) and "SIM" not in webui.take_label(1, 0.0, None, False),
          "a missing similarity is omitted rather than printed as None")
    check("WER" not in webui.take_label(1, None, 0.9, False),
          "a missing WER (transcription off) is omitted too")


def test_candidate_players_are_wired() -> None:
    """Every take must reach the page — otherwise 'best of N' is a black box."""
    import inspect

    source = inspect.getsource(webui.build_ui)
    for n in (1, 2, 3, 4):
        check(f"cl_take_{n} = gr.Audio" in source, f"take player {n} exists")
    check("cl_take_1, cl_take_2, cl_take_3, cl_take_4" in source,
          "the four players are outputs of Clone & generate")
    check("cl_takes_note" in source, "the row that explains the auto-pick exists")


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
            except Exception as e:  # noqa: BLE001 - one crashing case must not hide the rest
                failures += 1
                import traceback
                print(f"ERROR {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print("\n", "all passed" if not failures else f"{failures} failed")
    raise SystemExit(1 if failures else 0)
