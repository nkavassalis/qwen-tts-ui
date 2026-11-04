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
  * docs vs code — the product name, the tab labels and the environment variables in the
                 README are the ones the code actually has, and the ethical-use statement
                 exists in both the README and next to the clone button.

Run: .venv/bin/python tests/test_app_logic.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types

import gradio as gr
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui"))
import app as webui  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    ok = webui._report(np.zeros(sr * 3, dtype=np.float32), sr, 512, "VoiceDesign")
    check(ok.startswith("VoiceDesign OK"), "normal generation reports OK without warnings")
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

    Upstream builds `supported_languages` from codec_language_id but *skips* any dialect
    entry, so a dialect value would raise at generate time. (Dialects were a property of the
    CustomVoice preset speakers, which this app no longer ships — the check stays so the
    dropdown cannot quietly grow one back.) Skipped when the weights are not downloaded.
    """
    import json

    cfg_path = os.path.join(webui.ROOT, "models", "Qwen3-TTS-12Hz-1.7B-VoiceDesign", "config.json")
    if not os.path.exists(cfg_path):
        print(f"     (skipped: {cfg_path} not present)")
        return
    talker = json.load(open(cfg_path))["talker_config"]
    accepted = {"auto"} | {k for k in talker["codec_language_id"] if "dialect" not in k}
    # The model lowercases whatever it is given, so compare case-insensitively.
    offered = {v.lower() for v in webui.LANGUAGE_VALUES}
    unknown = offered - {"auto"} - accepted
    check(not unknown, f"dropdown offers languages the model rejects: {sorted(unknown)}")
    check(not [value for value in offered if "dialect" in value],
          "no dialect value is selectable — upstream would reject it at generate time")
    check("italian" in offered, "italian is supported and must stay selectable")


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
            check("no saved voice" in msg, "loading an unknown voice errors instead of returning silence")

            _, msg = webui.delete_saved_voice("sleepy_sea_captain")
            check("Deleted" in msg and webui.list_voices() == [], "delete removes both files")

            # A half-written entry must not break the list.
            open(os.path.join(tmp, "broken.json"), "w").write("{not json")
            webui.save_voice("good", wav, 24000, "t", "i", "english")
            check([v["slug"] for v in webui.list_voices()] == ["good"], "corrupt metadata is skipped, not fatal")
        finally:
            webui.VOICES_DIR = original


def test_saving_a_designed_voice_keeps_the_raw_decode() -> None:
    """Save must persist the model's own decode, never the post-processed listening copy.

    Two things used to go wrong here, both found in the real app. The Save handler read the
    audio box, which holds the *listening* copy — 48kHz, de-imaged, high-shelf, soft-limited —
    so a designed voice was saved already through an EQ curve, and then used as a clone
    reference that had been through the codec twice. A voice saved from the browser measured
    `sample_rate: 48000`, which is how this surfaced. And before that, Save read a module-level
    "last design" global, so with two people on the LAN one person's Save could persist the
    other's clip.

    So: the path of the raw decode travels through `gr.State` (JSON-safe, per-request), and
    this pins that the bytes on disk are the model's, not the EQ's.
    """
    import threading

    sr = 24000
    raw = (0.3 * np.sin(2 * np.pi * 150 * np.arange(sr * 3) / sr)).astype(np.float32)

    class FakeTTS:
        def generate_voice_design(self, **kwargs):
            return [raw], sr

    class FakeHub:
        def __init__(self):
            self.lock = threading.RLock()

        def acquire(self, capability):
            return FakeTTS()

    original_voices, original_media = webui.VOICES_DIR, webui.MEDIA_DIR
    original_hub, webui.HUB = webui.HUB, FakeHub()
    with tempfile.TemporaryDirectory() as tmp:
        webui.VOICES_DIR, webui.MEDIA_DIR = tmp, tmp
        try:
            audio, status, state = webui.generate_voice_design(
                "hello there", "English", "warm and low", 0.9, 1.0, 50, 1.05, postproc="Warm")
            box_sr, _box_wav = audio
            check(box_sr == 48000,
                  f"the player still gets the post-processed copy, got {box_sr}Hz")
            check(isinstance(state, dict) and state.get("path"),
                  f"Save gets a path through state, not an array: {state}")
            check(state.get("text") == "hello there" and state.get("language") == "English"
                  and state.get("description") == "warm and low",
                  f"the state carries what the clip says and how it was made: {state}")
            kept_sr = sf.read(state["path"], dtype="float32")[1]
            check(kept_sr == 24000, f"the kept clip is the raw decode, got {kept_sr}Hz")

            _, msg, _ = webui.save_designed_voice("nobody", None)
            check("Nothing to save yet" in msg, "cannot save without a generated take")
            _, msg, _ = webui.save_designed_voice("nobody", state["path"])
            check("Nothing to save yet" in msg, "a bare path carries no transcript: refused")
            _, msg, _ = webui.save_designed_voice("   ", state)
            check("name" in msg.lower(), "blank name rejected")
            check(webui.list_voices() == [], "rejected save writes nothing")
            _, msg, _ = webui.save_designed_voice("x", dict(state, text="   "))
            check("empty" in msg, "a take with no transcript is rejected (ICL needs it)")
            check(webui.list_voices() == [], "rejected save writes nothing")

            # The user edits the boxes after Generate (types the next line, tweaks the
            # description) and then saves the take they liked. Save takes no box inputs at
            # all, so what it stores is what the clip was generated from.
            _, msg, update = webui.save_designed_voice("Warm Narrator", state)
            check("hello there" in msg, f"the status says which transcript was saved: {msg}")
            check("Saved" in msg, f"saving the take works: {msg}")
            saved_wav, saved_sr = sf.read(os.path.join(tmp, "warm_narrator.wav"), dtype="float32")
            check(saved_sr == 24000,
                  f"the saved voice is the model's decode, not the {box_sr}Hz player copy: "
                  f"{saved_sr}Hz")
            check(saved_wav.shape == raw.shape and
                  float(np.max(np.abs(saved_wav - raw))) < 1e-3,
                  "the saved samples are the model's decode, EQ-free (the tolerance is "
                  "PCM_16 quantisation: soundfile writes 16-bit WAV, so 'raw' means "
                  "unprocessed, not bit-exact)")
            record = webui.list_voices()[0]
            check(record["sample_rate"] == 24000 and record["seconds"] == 3.0,
                  f"the sidecar describes the raw clip: {record}")
            check(record["text"] == "hello there" and record["description"] == "warm and low"
                  and record["instruct"] == "warm and low" and record["language"] == "English",
                  f"transcript, description and language are the ones the clip was generated "
                  f"from: {record}")
            check(update["choices"][0][1] == "warm_narrator",
                  "the voice dropdowns are refreshed with the new voice")

            # A path outside the allowed roots is refused, not read: the state is client-
            # round-tripped JSON, so whatever arrives here is untrusted.
            _, msg, _ = webui.save_designed_voice(
                "spy", {"path": "/opt/recordings/take.wav", "text": "hello"})
            check("outside the upload directory" in msg, f"foreign path refused: {msg}")
            check(webui.list_voices() == [record], "the refused save added nothing")

            # A failed take-write must not sink the generation, only disable Save.
            def explode(*a, **k):
                raise OSError("disk full")

            original_write, webui._write_take = webui._write_take, explode
            try:
                audio, status, state = webui.generate_voice_design(
                    "hello there", "English", "warm and low", 0.9, 1.0, 50, 1.05)
                check(audio is not None, "audio still comes back when the take cannot be kept")
                check("could not keep the raw clip" in status, f"the status says why: {status}")
                check(state is None, "and Save is disabled rather than saving something stale")
                _, msg, _ = webui.save_designed_voice("Late", state)
                check("Nothing to save yet" in msg, f"stale save refused: {msg}")
            finally:
                webui._write_take = original_write
        finally:
            webui.VOICES_DIR, webui.MEDIA_DIR = original_voices, original_media
            webui.HUB = original_hub


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
        cap_dir = os.path.join(tmp, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        os.makedirs(cap_dir)
        with open(os.path.join(cap_dir, "model.safetensors"), "wb") as fh:
            fh.truncate(2 * 1024 ** 3)  # 2GB sparse
        for name in ("config.json", "generation_config.json"):
            open(os.path.join(cap_dir, name), "w").write("{}")

        hub = webui.ModelHub(tmp, None, "sdpa", 1, max_cpu_resident=2)
        check(1900 < hub.weight_mb("voice_design") < 2100, f"size estimated from safetensors: {hub.weight_mb('voice_design')}")

        # Floor above what is available -> MemoryError before any weights are touched.
        guarded = webui.ModelHub(tmp, None, "sdpa", 1, max_cpu_resident=2,
                                 ram_floor_mb=int(webui.mem_available_mb() + 100_000))
        try:
            guarded._get_cpu("voice_design")
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
        webui.shutil.which = lambda name: f"/usr/bin/{name}"   # hermetic: no real ffmpeg needed
        webui.ffmpeg_extract_audio("/in/movie.mp4", "/out/a.wav", sr=24000)
        cmd = calls["cmd"]
        check(cmd[0] == webui.FFMPEG_BIN, "uses the configured ffmpeg")
        check("-vn" in cmd and "-map" in cmd, "drops the video stream and maps the first audio stream")
        check(cmd[cmd.index("-ac") + 1] == "1" and cmd[cmd.index("-ar") + 1] == "24000",
              "downmixes to mono at the requested rate")
        check(cmd[-1] == "/out/a.wav" and "pcm_s16le" in cmd, "writes PCM wav")
        check(cmd[cmd.index("-t") + 1] == str(webui.MAX_MEDIA_SECONDS),
              "decoding is capped at MAX_MEDIA_SECONDS, so a two-hour upload cannot fill RAM")
        timeouts = {}
        webui.subprocess.run = lambda cmd, **kw: (timeouts.update(kw),
                                                  types.SimpleNamespace(returncode=0, stderr=""))[1]
        webui.ffmpeg_extract_audio("/in/movie.mp4", "/out/a.wav")
        check(timeouts.get("timeout") == webui.FFMPEG_TIMEOUT_SECONDS, "ffmpeg runs under a timeout")

        def hang(cmd, **kw):
            raise webui.subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        webui.subprocess.run = hang
        try:
            webui.ffmpeg_extract_audio("/in/huge.mp4", "/out/a.wav")
            raise AssertionError("expected MediaError on an ffmpeg timeout")
        except webui.MediaError as e:
            check("took longer" in str(e) or "timed out" in str(e).lower(),
                  f"a hung ffmpeg is reported, not left running: {e}")

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
    # The window snaps outwards into the surrounding pauses, so it covers the whole burst
    # and may grow a little past `target` — never shrink below the clone minimum.
    check(19.5 <= start <= 20.05 and 24.95 <= end <= 25.6,
          f"should land on the loud stretch, edges in the silence around it, got {start}-{end}")

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
            return "GPU: stub | RAM: stub"

    original, webui.HUB = webui.HUB, StubHub()
    try:
        demo = webui.build_ui()
    finally:
        webui.HUB = original

    check(demo is not None, "build_ui returned a Blocks instance")
    check(len(demo.fns) >= 25, f"every button/slider is wired (found {len(demo.fns)} listeners)")

    audios = [b for b in demo.blocks.values() if isinstance(b, gr.Audio)]
    by_label = {b.label: b for b in audios}
    # Clone: microphone, reference, result, four takes · Design result · Speak result ·
    # Voices preview. (The chat player is HTML.)
    check(len(audios) == 10, f"expected 10 audio players, found {len(audios)}")
    reference = by_label.get("Reference clip — exactly what will be cloned")
    check(reference is not None, "the reference player exists")
    # Anything we read back from the browser is filepath-typed so that we open it
    # ourselves, after resolve_input_path; with type="numpy" Gradio opens the client's
    # path before our code runs, so we never get a chance to check it.
    check(reference.type == "filepath", "the reference box is filepath-typed so resolve_input_path can guard it")
    check(not reference.interactive, "the reference is a result of step 1, not a second upload box")
    check("Designed voice" in by_label and by_label["Designed voice"].type == "filepath",
          "the Voice Design box is filepath-typed: Save reads it back")

    # The public API surface: scripts/say.py depends on these names.
    names = {d.get("api_name") for d in demo.config["dependencies"]
             if d.get("api_visibility", "public") == "public"}
    for name in ("speak", "voices_available", "generate_voice_clone", "generate_voice_design"):
        check(name in names, f"/{name} stays a public endpoint")
    speak = next(d for d in demo.config["dependencies"] if d.get("api_name") == "speak")
    check(len(speak["inputs"]) == 8, "/speak keeps its 8 positional inputs (scripts/say.py)")
    design_save = [b for b in demo.fns.values() if b.fn is webui.save_designed_voice]
    check(len(design_save) == 1 and len(design_save[0].inputs) == 2
          and isinstance(design_save[0].inputs[1], gr.State),
          "Design's Save reads only the name and the take-state, never the text boxes "
          "(they may have been edited since Generate)")

    statuses = [b for b in demo.blocks.values()
                if isinstance(b, gr.Textbox) and b.label == "Status"]
    check(len(statuses) == 5, f"one Status box per tab, found {len(statuses)}")


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
        got = hub.acquire("voice_design")
        check(isinstance(got, FakeTTS), "cpu target returns the model")
        check("voice_design" in hub.on_gpu, "cpu target is tracked as resident")

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
            sys.argv = ["smoke_test.py", "--only", "voice_design"]
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rc = smoke_test.main()
            last_output[0] = out.getvalue()
            return rc

        check(run(skip_it=False) == 0, "a full run exits 0")
        check(run(skip_it=True) == 1, "a skipped case exits non-zero, not 0")
        check("SKIPPED voice_design" in last_output[0], "and it says which case was skipped")

        smoke_test.SKIPPED.clear()
        smoke_test.run_case = lambda hub, cap, keep_gpu, use_asr=True: smoke_test.skip(cap, "stubbed")
        sys.argv = ["smoke_test.py", "--only", "voice_design", "--allow-skip"]
        with contextlib.redirect_stdout(io.StringIO()):
            check(smoke_test.main() == 0, "--allow-skip is the explicit opt-out")
        check(smoke_test.SKIPPED and smoke_test.SKIPPED[0][0] == "voice_design",
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
    """Drift guard: the reference check and best-of-N must stay wired to real components."""
    import inspect

    source = inspect.getsource(webui.build_ui)
    check("cl_check = gr.Markdown(reference_check" in source,
          "the reference checklist exists and starts filled in")
    check("then_check(" in source and source.count("then_check(") >= 7,
          "every action that changes the reference re-runs the checklist")
    check("cl_ref_text.input" in source, "and so does typing in the transcript")
    check("cl_candidates" in source and "cl_seed" in source and "cl_clean" in source,
          "best-of-N, seed and reference cleanup are in the Clone tab")
    check("+ [cl_candidates, cl_seed, cl_clean]" in source,
          "they are actually passed into generate_voice_clone, in its argument order")
    check("value=3, step=1" in source, "best of 3 is the default — the measured quality lever")


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
    check(source.count("+ output_options") == 4,
          "every generation surface (Design a voice, Clone a voice, Speak, Chat) receives "
          "the choice")


def test_clone_tab_reads_top_to_bottom() -> None:
    """The Clone tab must read as the three steps it actually is, with one Save.

    It used to have two name boxes, two Save buttons, a take radio, a separate library row
    and five status boxes, and nothing said which saved what. Order is the documentation
    here, so pin it.
    """
    import inspect

    source = inspect.getsource(webui.build_ui)

    def at(needle: str) -> int:
        index = source.find(needle)
        assert index >= 0, f"missing {needle!r} in the Clone tab"
        return index

    steps = [at("#### 1 · Reference"), at("#### 2 · What it should say"),
             at("#### 3 · Save it to the voice library")]
    check(steps == sorted(steps), "the three step headings appear in order")
    check(at("cl_text = gr.Textbox") > steps[1], "the text to synthesise sits under step 2")
    check(at("cl_ref_text = gr.Textbox") < steps[1], "the transcript sits under step 1")
    check(at("cl_audio = gr.Audio") < steps[2], "the result sits next to the button that made it")
    check(at("cl_save_what = gr.Radio") > steps[2] and at("cl_save_btn = gr.Button") > steps[2],
          "saving lives in step 3")
    check(source.count("gr.Button(\"Save voice\"") == 2,
          "exactly one Save button on Clone and one on Design")

    # The two text boxes must not be confusable.
    check("Text for the cloned voice" in source, "the synthesis box says what it is for")
    check("What the reference says" in source, "the transcript box says it is what the clip says")
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


def test_defect_report() -> None:
    """The waveform-level half of the scorer: things no transcript can tell you."""
    sr = 24000
    speech = " ".join(f"word{i}" for i in range(15))          # 15 words ~ 6s at 2.5 w/s

    def tone(seconds, amp=0.25, freq=180):
        t = np.arange(int(seconds * sr)) / sr
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    good = webui.defect_report(tone(6.0), sr, speech)
    check(good["penalty"] == 0.0, f"a normal take is not penalised: {good}")
    check(webui.defect_flags(good) == [], "and raises no flags")
    check(abs(good["rate_ratio"] - 1.0) < 0.25, f"rate reads ~1x for a natural clip: {good}")

    padded = np.concatenate([np.zeros(int(0.8 * sr), dtype=np.float32), tone(4.0),
                             np.zeros(int(0.8 * sr), dtype=np.float32)]).astype(np.float32)
    silent = webui.defect_report(padded, sr, speech)
    check(silent["lead_silence"] > 0.6 and silent["tail_silence"] > 0.6,
          f"edge silence is measured: {silent}")
    check(any("silence" in f for f in webui.defect_flags(silent)), f"and flagged: {silent}")
    check(silent["penalty"] > 0, "silence costs penalty")

    hot = (np.sign(tone(6.0)) * 0.9999).astype(np.float32)   # hard clipped
    clipped = webui.defect_report(hot, sr, speech)
    check(clipped["clipped_fraction"] > webui.CLIP_FRACTION, f"clipping measured: {clipped}")
    check("clipping" in webui.defect_flags(clipped), "and flagged")
    check(clipped["penalty"] >= 0.2, "and penalised")

    slow = webui.defect_report(tone(12.0), sr, "two slow words")
    check("rate" in " ".join(webui.defect_flags(slow)), f"a crawl is flagged: {slow}")
    fast = webui.defect_report(tone(2.0), sr, " ".join(f"w{i}" for i in range(30)))
    check("rate" in " ".join(webui.defect_flags(fast)), f"machine-gun is flagged: {fast}")
    check(slow["penalty"] > 0 and fast["penalty"] > 0, "both cost penalty")

    empty = webui.defect_report(np.zeros(0, dtype=np.float32), sr, speech)
    check(empty["penalty"] == 1.0, "an empty take is the worst possible score")

    # CJK is counted per character, or every Chinese take would read as 'too slow'.
    chinese = "我" * 20
    zh = webui.defect_report(tone(5.0), sr, chinese)
    check(zh["penalty"] == 0.0, f"20 characters in 5s reads as a normal rate: {zh}")


def test_candidate_penalty() -> None:
    """Content still beats everything; confidence and defects only break near-ties."""
    pen = webui.candidate_penalty
    check(pen(0.0, -0.10, {"penalty": 0.0}) == 0.0, "a clean take costs nothing")
    good = webui.CONF_GOOD
    check(pen(0.4, good - 1.0, {"penalty": 0.0}) > pen(0.0, good - 1.5, {"penalty": 0.3}),
          "saying the words wrong outweighs a struggling decoder plus defects")
    check(pen(0.0, good - 1.0, {"penalty": 0.0}) < pen(0.0, good - 2.5, {"penalty": 0.0}),
          "a struggling decoder separates two takes with identical WER")
    check(pen(None, None, None) == 0.0, "no signals at all costs nothing (ASR off)")
    check(pen(0.0, good - 1.0, {"penalty": 5.0}) == round(webui.CONF_WEIGHT + webui.DEFECT_CAP, 4),
          "defects are capped so they can never outvote content")
    check(pen(0.0, good, None) == 0.0, "confidence exactly at the good line is free")
    check(pen(0.0, good - 0.4, None) > 0.1, "below the line it is penalised")
    check(pen(0.0, good + 0.6, None) == 0.0, "a clean take, above the line, costs nothing")
    check(pen(0.0, good - 4.0, None) > 1.0,
          "a take Whisper cannot match to the text at all must dominate every other signal")
    check(pen(0.2, None, None) > pen(0.0, None, {"penalty": 0.4}),
          "10% wrong words outweighs a fistful of defect flags")
    check(pen(0.02, None, None) < pen(0.0, good - 3.0, None),
          "a take that is barely decodable can lose over one trivial word error")


def test_choose_best_candidate_uses_the_penalty() -> None:
    metrics = [{"wer": 0.0, "sim": 0.95, "penalty": 0.6},
               {"wer": 0.1, "sim": 0.95, "penalty": 0.1}]
    check(webui.choose_best_candidate(metrics) == 1,
          "a zero-WER take riddled with defects loses to a slightly wrong clean one")
    check(webui.choose_best_candidate([{"wer": 0.05, "sim": 0.99},
                                       {"wer": 0.05, "sim": 0.99}]) == 0,
          "perfectly equal candidates resolve to the first, deterministically")
    check(webui.choose_best_candidate([{"wer": 0.0, "sim": 0.9, "penalty": 0.0},
                                       {"wer": 0.0, "sim": 0.99, "penalty": 0.004}]) == 1,
          "within rounding, similarity still breaks the tie")


def test_take_label_shows_confidence_and_defects() -> None:
    conf = webui.take_label(1, 0.0, 0.99, False, confidence=-0.42)
    check("conf -0.42" in conf, f"confidence is shown so the pick can be questioned: {conf}")
    flagged = webui.take_label(1, 0.0, 0.99, False,
                               defects={"penalty": 0.3, "rate_ratio": 1.0, "lead_silence": 0.9,
                                        "tail_silence": 0.4, "clipped_fraction": 0.0})
    check("silence" in flagged and "⚠" in flagged, f"defects are visible on the player: {flagged}")
    clean = webui.take_label(1, 0.0, 0.99, False, confidence=-0.1,
                             defects={"penalty": 0.0, "rate_ratio": 1.0, "lead_silence": 0.1,
                                      "tail_silence": 0.1, "clipped_fraction": 0.0})
    check("⚠" not in clean, f"a clean take is not covered in warnings: {clean}")


def test_mean_token_logprob_alignment() -> None:
    """The off-by-prefix bug in this arithmetic is silent and total, so pin it.

    Measured on a real clean clip: correct alignment reads -1.9, one token off reads
    -13.5 (worse than a uniform distribution). Nothing errors; the signal just becomes
    garbage that penalises every candidate equally.
    """
    import torch

    vocab, steps = 8, 6
    logits = torch.full((1, steps, vocab), -10.0)
    for position in range(steps):
        logits[0, position, (position + 1) % vocab] = 20.0   # logits[t] predicts token t+1
    tokens = [1, 2, 3, 4]

    aligned = webui._mean_token_logprob(logits, tokens, offset=0)
    check(aligned is not None and aligned > -0.01,
          f"a decoder that is certain reads ~0, got {aligned}")
    misaligned = webui._mean_token_logprob(logits, tokens, offset=2)
    check(misaligned is not None and misaligned < -3,
          f"the same tokens one step off must read as nonsense, got {misaligned}")

    uniform = torch.zeros(1, 4, vocab)
    flat = webui._mean_token_logprob(uniform, [0, 1, 2, 3])
    check(abs(flat - float(np.log(1.0 / vocab))) < 1e-3,
          f"a uniform decoder reads log(1/vocab), got {flat}")

    check(webui._mean_token_logprob(logits, []) is None, "no tokens, no score")
    check(webui._mean_token_logprob(None, [1]) is None, "no logits, no score")
    check(webui._mean_token_logprob(logits[:, :3], tokens, offset=0) is None,
          "not enough steps to score every token -> None rather than a partial average")
    check(webui._mean_token_logprob(logits, tokens, offset=3) is None,
          "offset plus tokens running past the end is refused, not silently truncated")


def test_build_take_state_is_json_safe_and_names_the_winner() -> None:
    """gr.State round-trips through the browser, so the take record must be plain JSON."""
    import json

    results = [
        {"index": 1, "wer": 0.0, "sim": 0.988, "confidence": -2.01,
         "defects": {"penalty": 0.12, "rate_ratio": 1.3, "lead_silence": 0.9, "tail_silence": 0.0,
                     "clipped_fraction": 0.0}},
        {"index": 2, "wer": 0.0, "sim": 0.985, "confidence": -2.09,
         "defects": {"penalty": 0.0, "rate_ratio": 1.2, "lead_silence": 0.1, "tail_silence": 0.0,
                     "clipped_fraction": 0.0}},
    ]
    state = webui.build_take_state(results, 2, ["/tmp/a.wav", None], "  Say this.  ", "English")
    json.dumps(state)                       # raises on anything non-serialisable
    check([t["index"] for t in state["takes"]] == [1, 2], "one entry per candidate, in order")
    check([t["path"] for t in state["takes"]] == ["/tmp/a.wav", None],
          "paths are carried through, including the None from a failed write")
    check([t["kept"] for t in state["takes"]] == [False, True], "the auto-pick is flagged")
    check("silence" in state["takes"][0]["label"], "labels keep the same numbers the players show")
    check(state["text"] == "Say this.", "the line it was asked to say is the transcript")
    check(state["language"] == "English", "language travels with it")
    check(webui.build_take_state([], 0, [], "x", None)["language"] == "Auto",
          "no language means Auto, not None")


def test_save_generated_voice_needs_a_name_and_a_generation() -> None:
    _, status, _ = webui.save_generated_voice({"takes": [{"index": 1, "path": "/x"}],
                                               "text": "hi", "language": "Auto"}, "1", "  ")
    check("name" in status.lower(), f"an empty name is refused: {status}")
    _, status, _ = webui.save_generated_voice(None, "1", "someone")
    check("generate" in status.lower(), f"no generation yet is explained, not crashed: {status}")
    _, status, _ = webui.save_generated_voice({"takes": [{"index": 1, "path": "/x"},
                                                         {"index": 2, "path": "/y"}],
                                               "text": "hi", "language": "Auto"}, None, "someone")
    check("pick" in status.lower(), f"an ambiguous selection asks instead of guessing: {status}")


def test_save_generated_voice_saves_the_chosen_take() -> None:
    """The take you preferred becomes a reusable voice, with the line it spoke as transcript."""
    sr = 24000
    wav = (0.3 * np.sin(2 * np.pi * 160 * np.arange(sr * 3) / sr)).astype(np.float32)
    with tempfile.TemporaryDirectory() as media, tempfile.TemporaryDirectory() as voices:
        original_media, original_voices = webui.MEDIA_DIR, webui.VOICES_DIR
        webui.MEDIA_DIR, webui.VOICES_DIR = media, voices
        try:
            keep = os.path.join(media, "keep.wav")
            other = os.path.join(media, "other.wav")
            sf.write(keep, wav, sr)
            sf.write(other, wav[:sr], sr)
            state = {"takes": [{"index": 1, "path": keep, "label": "Take #1", "kept": False},
                               {"index": 2, "path": other, "label": "Take #2", "kept": True}],
                     "text": "Picking the best of several takes should never pick a worse one.",
                     "language": "English"}
            _, status, dropdown = webui.save_generated_voice(state, "1", "fox take one")
            check("Saved take #1" in status, f"names the take it kept: {status}")
            check("model's own decode" in status,
                  "says it stored the raw decode, not the post-processed copy")
            saved = webui.list_voices()
            check(len(saved) == 1 and saved[0]["name"] == "fox take one", f"in the library: {saved}")
            check(saved[0]["text"].startswith("Picking the best"),
                  f"transcript is the line it was asked to say: {saved[0]['text']}")
            check(saved[0]["language"] == "English" and 2.5 < saved[0]["seconds"] < 3.5,
                  f"metadata is right: {saved[0]}")
            check(os.path.exists(saved[0]["wav"]), "the wav is on disk")
            reloaded, rate = sf.read(saved[0]["wav"], dtype="float32")
            check(rate == sr and len(reloaded) == len(wav), "and it is the chosen take, not the other")
            check(dropdown.get("value") == saved[0]["slug"], "the dropdown lands on the new voice")

            # A single candidate has nothing to choose, so the selection may be blank.
            single = {"takes": [{"index": 1, "path": keep, "label": "Take #1", "kept": True}],
                      "text": "One take.", "language": "Auto"}
            _, status, _ = webui.save_generated_voice(single, None, "solo take")
            check("Saved take #1" in status, f"one candidate needs no selection: {status}")

            # A state forged by a client must not read arbitrary files.
            forged = {"takes": [{"index": 1, "path": "/etc/passwd", "label": "x", "kept": True}],
                      "text": "x", "language": "Auto"}
            _, status, _ = webui.save_generated_voice(forged, "1", "sneaky")
            check(status.startswith("Error"), f"a path outside the upload dirs is refused: {status}")
            check(not os.path.exists(os.path.join(voices, "sneaky.wav")), "and nothing was written")
        finally:
            webui.MEDIA_DIR, webui.VOICES_DIR = original_media, original_voices


def test_take_picker_is_wired() -> None:
    """The one save picker must be fed by generation and read by Save."""
    import inspect

    source = inspect.getsource(webui.build_ui)
    check("cl_take_state = gr.State(value=None)" in source,
          "the take paths travel through gr.State, not a module global")
    check("cl_takes_note, cl_take_state, cl_save_what]" in source,
          "generation populates the players, the note, the state and the save picker")
    check("inputs=[cl_save_what, cl_save_name, cl_ref_audio, cl_ref_text, cl_language,\n"
          "                            cl_take_state]" in source,
          "saving reads the selection and the takes from the page, not from server memory")

    choices = webui.save_choices({"takes": [{"index": 1, "path": "/a", "label": "Take #1"},
                                            {"index": 2, "path": None, "label": "Take #2"}]})
    check(choices[0][1] == webui.SAVE_REFERENCE, "the reference is always the first choice")
    check([value for _, value in choices] == [webui.SAVE_REFERENCE, "1"],
          "a take that failed to reach disk is not offered")
    check(webui.save_choices(None) == choices[:1], "before generating, only the reference")


def test_candidate_players_are_wired() -> None:
    """Every take must reach the page — otherwise 'best of N' is a black box."""
    import inspect

    source = inspect.getsource(webui.build_ui)
    check("cl_takes.append(gr.Audio(" in source and "range(row_start" in source,
          "one take player per candidate is built")
    check("outputs=[cl_audio, cl_status] + cl_takes" in source,
          "the players are outputs of Clone & generate")
    check("cl_takes_note" in source, "the line that explains the auto-pick exists")
    check(webui.CLONE_OUTPUTS == 2 + webui.MAX_CANDIDATES + 3,
          "the output count matches what is wired")
    check(len(webui._clone_failure("x")) == webui.CLONE_OUTPUTS,
          "a validation failure returns a value for every wired output")


# ---------------------------------------------------------------------------
# Long-text chunking
# ---------------------------------------------------------------------------

CHUNK_CORPUS = (
    "Mr. E. Smith drove 3.5 miles to St. John's office (e.g. vs. the old route) at 1. Main "
    "St. and then left. He arrived at 4:30 p.m. sharp, which surprised everyone there. "
    "Professor Li, Ph.D. said it was fine, and Dr. Brown agreed. Visit https://example.com/a.b "
    "or mail someone@example.co.uk for details. J. R. R. Tolkien wrote it. Wait... really? "
    "Yes! And then nothing happened at all, which was the point of the whole exercise. "
    "1. First item 2. Second item 3. Third item, all of which were read aloud by the "
    "presenter who had prepared them carefully the night before the meeting. "
    "La dott. Rossi ha detto che p.es. il treno delle 7.15 parte in ritardo di venti minuti "
    "e che dobbiamo aspettare alla stazione centrale senza lamentarci troppo. "
    "今天天气很好，我们一起去公园散步吧。他看了看表，说时间还早，可以先喝一杯咖啡再出发也不迟。"
)


def test_chunking_is_lossless() -> None:
    """The one failure a TTS splitter must never have: silently losing words."""
    chunks = webui.split_for_tts(CHUNK_CORPUS)
    check(len(chunks) > 3, f"a long corpus should split into pieces, got {len(chunks)}")
    joined = "".join(chunks).replace(" ", "")
    check(joined == CHUNK_CORPUS.replace(" ", ""),
          "concatenating the chunks must reproduce the input, character for character")
    for chunk in chunks:
        check(chunk.strip() == chunk, "chunks are trimmed")
        check(_has_no_placeholder(chunk), f"placeholders must never leak to the model: {chunk}")


def _has_no_placeholder(chunk: str) -> bool:
    return "\x00" not in chunk


def test_chunking_protects_false_boundaries() -> None:
    """A dot that is not a full stop must not become a chunk boundary."""
    tricky = ("Mr. E. Smith drove 3.5 miles to St. John's office at 1. Main St. and then left "
              "the building slowly because his knee was hurting again today. Professor Li "
              "(Ph.D. and very proud of it) waited outside with Dr. Brown and Mrs. Clark, who "
              "had brought the coffee and the biscuits for everyone.")
    chunks = webui.split_for_tts(tricky)
    for chunk in chunks:
        for bad in ("Mr", "St", "Dr", "Mrs", "Prof", "Ph.D"):
            check(not chunk.rstrip().endswith(bad + "."),
                  f"a chunk ends on the abbreviation {bad}. — it would be spoken with final "
                  f"intonation: {chunk!r}")
    check(not any(c.rstrip().endswith("3") for c in chunks), "3.5 was not cut in half")
    ellipsis = ("He said wait... really? yes and then left the room without another word to "
                "anyone there at all that evening after the meeting ended quite abruptly.")
    for chunk in webui.split_for_tts(ellipsis):
        check(not chunk.strip().endswith("..."), "an ellipsis is not a sentence end")
    listing = ("1. First item is quite long indeed and needs to be read carefully 2. Second "
               "item follows it and is also long enough to matter 3. Third item closes the list "
               "that the presenter prepared the night before the meeting.")
    check("".join(webui.split_for_tts(listing)).replace(" ", "") == listing.replace(" ", ""),
          "numbered list markers survive")


def test_chunking_respects_the_measured_band() -> None:
    """Chunks exist to stay inside the region where the model does not rush."""
    chunks = webui.split_for_tts(CHUNK_CORPUS)
    for chunk in chunks:
        units = webui._units(chunk)
        check(units <= int(max(webui.CHUNK_MAX_WORDS, webui.CHUNK_MAX_CJK) * 1.35),
              f"chunk is longer than the band allows ({units}): {chunk[:60]}...")
    short = " ".join(f"word{i}" for i in range(24))
    check(len(webui.split_for_tts(short)) <= 3, "a 24-word run does not shatter")
    tiny = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew " \
           "it! And thanks to you for everything you have done for us this year."
    for chunk in webui.split_for_tts(tiny):
        check(webui._units(chunk) >= 4,
              f"a two-word chunk arrives clipped and toneless: {chunk!r}")


def test_chunking_leaves_short_text_alone() -> None:
    """Below the measured safe zone, chunking must change nothing at all."""
    short = "Picking the best of several takes should never pick a worse one by accident."
    check(webui.split_for_tts(short) == [short.strip()], "short text is one pass")
    check(webui.split_for_tts("") == [], "empty text yields no chunks")
    check(webui.split_for_tts("   ") == [], "whitespace yields no chunks")


def test_chunking_handles_cjk() -> None:
    check("\u3002" in webui._TERMINATORS and "\uff01" in webui._TERMINATORS
          and "\uff1f" in webui._TERMINATORS,
          f"ideographic terminators must be matched by codepoint, not lookalikes: "
          f"{[hex(ord(c)) for c in webui._TERMINATORS]}")
    text = "今天天气很好，我们一起去公园散步吧。他看了看表，说时间还早，可以先喝一杯咖啡再出发也不迟。" \
           "路上的行人很多，每个人都在赶自己的路，没有人注意到那只坐在长椅上的猫。"
    chunks = webui.split_for_tts(text)
    check(len(chunks) >= 2, f"CJK must split by character budget, got {len(chunks)}")
    check("".join(chunks).replace(" ", "") == text.replace(" ", ""), "CJK slicing is lossless")
    for chunk in chunks:
        check(webui._units(chunk) <= int(webui.CHUNK_MAX_CJK * 1.35),
              f"CJK chunk over the band: {webui._units(chunk)}")


def test_chunking_is_deterministic() -> None:
    """The seed field promises reproducibility, which needs identical chunking."""
    first = webui.split_for_tts(CHUNK_CORPUS)
    second = webui.split_for_tts(CHUNK_CORPUS)
    check(first == second, "same input must produce the same chunks")


def test_chunking_refuses_an_unbounded_request() -> None:
    """One GPU, one request: a whole novel must fail loudly, not run for hours."""
    huge = "This is a sentence with a reasonable number of words in it. " * 3000
    try:
        webui.split_for_tts(huge)
    except webui.MediaError as e:
        check("chunk" in str(e).lower(), f"the error should say why: {e}")
    else:
        check(False, "an over-long script must raise MediaError rather than be attempted")


def _tone(seconds, sr=24000, amp=0.3, freq=180, phase=0.0):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def test_stitch_trims_and_crossfades() -> None:
    sr = 24000
    pad = int(0.5 * sr)
    pieces = [np.concatenate([np.zeros(pad, np.float32), _tone(2.0), np.zeros(pad, np.float32)])
              for _ in range(3)]
    joined = webui.stitch_clips(pieces, sr)
    naive = np.concatenate(pieces)
    check(len(joined) < len(naive) - sr,
          f"the padding between pieces must be trimmed ({len(joined)/sr:.2f}s vs "
          f"{len(naive)/sr:.2f}s)")
    check(len(joined) > 3 * 2.0 * sr - sr, "but the speech must not be eaten")
    rms = webui._rms_windows(joined, sr, int(0.02 * sr))
    silent = np.flatnonzero(rms < webui.SILENCE_RMS)
    longest = 0
    run = 0
    previous = None
    for index in silent:
        run = run + 1 if previous == index - 1 else 1
        longest = max(longest, run)
        previous = index
    check(longest * 0.02 < 0.35,
          f"a {longest * 0.02:.2f}s gap inside the stitch reads as a stutter")
    check(float(np.max(np.abs(joined))) <= 1.0, "stitching must stay inside full scale")

    # Clicks are asserted here rather than in the GPU tier: only here do we know where the
    # joins are and what a clean waveform looks like. Unpadded tones at different frequencies
    # and phases put the seam at a loud part of the waveform, which is where a hard splice
    # audibly pops; an equal-power crossfade must not.
    pieces_phased = [_tone(1.0, amp=0.3, freq=f, phase=p)
                     for f, p in ((180.0, 0.0), (311.0, np.pi / 2), (233.0, np.pi))]
    hard = np.concatenate(pieces_phased)
    smooth = webui.stitch_clips(pieces_phased, sr)
    hard_jump = float(np.max(np.abs(np.diff(hard))))
    smooth_jump = float(np.max(np.abs(np.diff(smooth))))
    check(hard_jump > 0.15, f"this fixture should make a hard splice click, got {hard_jump:.3f}")
    check(smooth_jump < hard_jump / 3,
          f"crossfade did not smooth the seam ({smooth_jump:.3f} vs {hard_jump:.3f})")


def test_stitch_edge_cases() -> None:
    sr = 24000
    check(webui.stitch_clips([], sr).size == 0, "no clips, no audio")
    single = webui.stitch_clips([_tone(1.0)], sr)
    check(0.8 * sr < len(single) < 1.1 * sr, f"one clip passes through: {len(single)/sr:.2f}s")
    tiny = webui.stitch_clips([_tone(0.005), _tone(0.005)], sr)
    check(tiny.size > 0 and np.all(np.isfinite(tiny)), "clips shorter than the crossfade survive")
    loud = webui.stitch_clips([_tone(0.5, amp=0.9), _tone(0.5, amp=0.9)], sr)
    check(float(np.max(np.abs(loud))) <= 1.0, "a hot crossfade is scaled, not clipped")


def test_generate_chunked_calls_the_model_the_right_number_of_times() -> None:
    sr = 24000
    calls = []

    def fake(chunk: str):
        calls.append(chunk)
        words, _ = webui.speech_units(chunk)
        return _tone(max(0.4, words / 2.5)), sr

    short = "This is only one sentence and should not be chunked at all."
    wav, rate, notes = webui.generate_chunked(short, "English", fake)
    check(len(calls) == 1, f"short text must be one pass, got {len(calls)}")
    check(notes == [], "and no chunking note")
    check(abs(len(wav) / sr - len(webui.trim_edges(fake(short)[0], sr)) / sr) < 1.0,
          "the single pass returns the model's audio")

    calls.clear()
    long_text = "This is a sentence with a reasonable number of words in it. " * 12
    wav, rate, notes = webui.generate_chunked(long_text, "English", fake)
    check(len(calls) > 3, f"long text must be synthesised in pieces, got {len(calls)}")
    check(any("Chunked" in n for n in notes), f"the status should say so: {notes}")
    check("".join(calls).replace(" ", "") == long_text.strip().replace(" ", ""),
          "every word of the request reaches the model exactly once")
    check(len(wav) / sr > 20.0, f"stitched audio should cover the text: {len(wav)/sr:.1f}s")


def test_generate_chunked_rerolls_a_rushed_chunk() -> None:
    """Tempo acceleration is the failure this path exists to avoid — so retry it once."""
    sr = 24000
    seen = {}

    def fake(chunk: str):
        seen[chunk] = seen.get(chunk, 0) + 1
        words, _ = webui.speech_units(chunk)
        # First take rushes (5 words/s), the retry speaks normally (2.5 words/s).
        rate = 5.0 if seen[chunk] == 1 and words > 10 else 2.5
        return _tone(max(0.4, words / rate)), sr

    long_text = "This is a sentence with a reasonable number of words in it. " * 12
    wav, _, notes = webui.generate_chunked(long_text, "English", fake)
    check(any("re-rolled" in n for n in notes), f"the rushed take should be replaced: {notes}")
    report = webui.defect_report(wav, sr, long_text)
    check(report["rate_ratio"] < 1.8, f"stitched result still rushes: {report}")


def test_callbacks_use_the_chunked_path() -> None:
    """All three capabilities must go through it, or long text still rushes somewhere."""
    import inspect

    source = inspect.getsource(webui)
    for capability in ("generate_voice_design", "generate_voice_clone"):
        start = source.index(f"def {capability}(")
        body = source[start:start + 6000]
        check("generate_chunked(" in body, f"{capability} does not use the chunked path")
        check("_budget(chunk)" in body, f"{capability} does not bound each chunk's tokens")


def test_speak_validates_before_touching_the_gpu() -> None:
    """A network caller gets a sentence it can act on, never a traceback."""
    _, status = webui.speak("", "hello")
    check("voice" in status.lower(), f"a missing voice is named: {status}")
    _, status = webui.speak("someone", "   ")
    check("text is required" in status.lower(), f"empty text is refused: {status}")
    _, status = webui.speak("definitely-not-a-voice", "hello")
    check(status.startswith("Error") and "available" in status.lower(),
          f"an unknown voice says what exists instead of failing obscurely: {status}")


def test_speak_resolves_voices_server_side() -> None:
    """The client names a voice; it never names a path."""
    sr = 24000
    wav = (0.3 * np.sin(2 * np.pi * 160 * np.arange(sr * 3) / sr)).astype(np.float32)
    with tempfile.TemporaryDirectory() as voices:
        original, webui.VOICES_DIR = webui.VOICES_DIR, voices
        try:
            webui.save_voice(name="Ryan Narrator", wav=wav, sr=sr, text="A test line.",
                             instruct="", language="English")
            resolved, error = webui._resolve_voice("Ryan Narrator")
            check(error is None and resolved is not None, f"display name resolves: {error}")
            ref_wav, ref_sr, ref_text = resolved
            check(ref_sr == sr and abs(len(ref_wav) / ref_sr - 3.0) < 0.1,
                  "the clip comes back at its own rate")
            check(ref_text == "A test line.", "the transcript comes from the sidecar")

            resolved, error = webui._resolve_voice("ryan-narrator")
            check(error is None, "the slug resolves too")
            resolved, error = webui._resolve_voice("ryan_narrator")
            check(error is None, "with either separator")

            for hostile in ("../../etc/passwd", "/etc/passwd", "..%2f..%2fetc%2fpasswd",
                            "~/.ssh/id_rsa"):
                resolved, error = webui._resolve_voice(hostile)
                check(resolved is None and error.startswith("Error"),
                      f"a hostile voice name must not resolve: {hostile}")

            listing = webui.voices_available()
            check("ryan_narrator" in listing and "A test line." in listing,
                  f"--list shows the slug and the transcript: {listing}")
            webui.save_voice("Words", None, 0, "", "airy and bright", "English",
                             description="airy and bright", kind="design")
            check("words\tdescribed\tairy and bright" in webui.voices_available(),
                  "a described voice lists as described, not as 'Nones'")
            webui.VOICES_DIR = os.path.join(voices, "empty")
            check("no voices saved" in webui.voices_available(), "an empty library is stated")
        finally:
            webui.VOICES_DIR = original


def test_speak_and_voice_list_are_api_endpoints() -> None:
    """say.py calls these by name; if they stop being endpoints the CLI silently rots."""
    class StubHub:
        def status(self) -> str:
            return "stub"

    original_hub, webui.HUB = webui.HUB, StubHub()
    try:
        demo = webui.build_ui()
        entries = demo.fns.values() if isinstance(demo.fns, dict) else demo.fns
        exposed = {getattr(getattr(entry, "fn", None), "__name__", None) for entry in entries}
        check("speak" in exposed, "speak is wired to a component, so it is a named endpoint")
        check("voices_available" in exposed, "the voice list is callable too (--list)")
    finally:
        webui.HUB = original_hub

    import inspect
    source = inspect.getsource(webui.speak)
    check("_voice_synth(" in source, "speak shares the voice resolution with the chat tab")
    check("_resolve_voice(" in inspect.getsource(webui._voice_synth),
          "resolution happens server-side, in one place")
    check("resolve_input_path" not in source and "resolve_input_path" not in inspect.getsource(webui._voice_synth),
          "speak takes no client path at all — there is nothing to confine")
    check("generate_chunked(" in source, "speak gets the chunked long-text path too")


def test_say_py_is_a_thin_client() -> None:
    """The CLI must stay a client of the two endpoints, not a second implementation."""
    path = os.path.join(os.path.dirname(webui.__file__), "..", "scripts", "say.py")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    check('api_name="/speak"' in source, "it calls the speak endpoint")
    check('api_name="/voices_available"' in source, "and the voice list for --list")
    check("--no-play" in source and "paplay" in source, "it can render without playing")
    check("--password-file" in source and "QWEN_TTS_PASSWORD" in source,
          "credentials follow the server's precedence")
    check("--ref-audio" not in source and "--voice-path" not in source,
          "the CLI must never be able to name a server path")


def test_chat_say_streams_one_piece_at_a_time() -> None:
    """The point of the chat tab is that you hear the first sentence early."""
    sr = 24000
    calls = []

    def fake(chunk: str):
        calls.append(chunk)
        words, _ = webui.speech_units(chunk)
        t = np.arange(int(max(0.4, words / 2.5) * sr)) / sr
        return (0.25 * np.sin(2 * np.pi * 170 * t)).astype(np.float32), sr

    text = ("The build finished cleanly. Three tests failed in the database layer. "
            "I fixed two of them already and the third needs a migration. "
            "Want me to write the migration now or leave it for the morning?")
    turns = list(webui.chat_say(text, [], "someone", "English", synth=fake))

    check(len(turns) >= 4, f"a reply should arrive in several updates, got {len(turns)}")
    check(len(calls) >= 2, f"the text should have been chunked, got {len(calls)} calls")
    streamed = [t for t in turns if isinstance(t[1], str) and t[1].startswith("<audio")]
    check(len(streamed) == len(calls),
          f"one audio hand-off per piece ({len(streamed)} vs {len(calls)})")
    # Decode what the browser would actually receive, rather than trusting the wrapper.
    import base64 as _b64
    import io as _io
    import re as _re

    decoded = []
    for turn in streamed:
        match = _re.search(r'src="data:audio/wav;base64,([^"]+)"', turn[1])
        check(match is not None, "the player carries an embedded wav, not a server path")
        decoded.append(sf.read(_io.BytesIO(_b64.b64decode(match.group(1))), dtype="float32"))
    check(all(len(w) for w, _ in decoded), "every hand-off carries audio")
    check(all(float(np.max(np.abs(w))) <= 1.0 for w, _ in decoded),
          "streamed audio stays inside full scale")
    check(all(sr == 48000 for _, sr in decoded),
          f"the listening copy is the post-processed one, got {set(sr for _, sr in decoded)}")
    check(all("autoplay" in t[1] and "controls" in t[1] for t in streamed),
          "each element autoplays and still offers a manual play button")

    final_messages = turns[-1][0]
    check(final_messages[0]["role"] == "user" and final_messages[1]["role"] == "assistant",
          "the log is a user turn then an assistant turn")
    said = final_messages[1]["content"]
    check(said.replace(" ", "") == text.replace(" ", ""),
          "the log must contain exactly what was asked to be said, in order")
    grew = [len(t[0][-1]["content"]) for t in turns if t[0] and t[0][-1]["role"] == "assistant"]
    check(grew == sorted(grew) and grew[-1] > grew[0], "the reply fills in as it goes")
    check("Said" in turns[-1][2] and "piece" in turns[-1][2], f"final status: {turns[-1][2]}")


def test_chat_say_reports_failures_without_killing_the_chat() -> None:
    turns = list(webui.chat_say("   ", [], "someone", "English", synth=lambda c: (None, 0)))
    check("nothing to say" in turns[-1][2].lower(), f"empty input is answered: {turns[-1][2]}")

    def explode(chunk: str):
        raise webui.MediaError("no saved voice ‘someone’")

    turns = list(webui.chat_say("Hello there friend, this is a reasonably long sentence.",
                                [], "someone", "English", synth=explode))
    check(turns[-1][2].startswith("Error"), f"the status line carries the error: {turns[-1][2]}")
    check("⚠" in turns[-1][0][-1]["content"], "the bubble says what went wrong instead of vanishing")


def test_chat_path_shares_the_chunker_and_locks_per_chunk() -> None:
    """Two invariants that only bite in production: one chunker, and no lock across yields."""
    import inspect

    chat_source = inspect.getsource(webui.chat_say)
    check("iter_chunked(" in chat_source, "chat uses the same splitter as everything else")
    check("_voice_synth(" in chat_source, "and the same voice resolution as /speak")

    synth_source = inspect.getsource(webui._voice_synth)
    check(synth_source.index("def one(chunk") < synth_source.index("with HUB.lock"),
          "the hub lock must be taken inside one(), per chunk — holding it across a yielded "
          "reply can strand the app if the client walks away")

    demo_source = inspect.getsource(webui.build_ui)
    check("ch_player = gr.HTML" in demo_source,
          "the chat player is our own element — Gradio's Audio reuses one <audio> tag and "
          "only autoplays on the first source load, which is what silenced a streamed reply")
    player_source = inspect.getsource(webui._audio_html)
    check("autoplay" in player_source and "controls" in player_source,
          "it autoplays, and a blocked browser is one click away from playing it")
    check("ch_text.submit(chat_say" in demo_source,
          "Enter submits, not just the button")


def test_product_name_is_consistent() -> None:
    """One name, everywhere. It used to be 'Qwen3-TTS — local (1.7B, 16GB Blackwell)'."""
    import inspect

    source = inspect.getsource(webui.build_ui)
    check('title="qwen3-tts-ui"' in source, "the browser tab is titled qwen3-tts-ui")
    check("# qwen3-tts-ui" in source, "the page header says qwen3-tts-ui")
    check("16GB Blackwell" not in source,
          "the header is for people, not a hardware spec sheet; the Runtime line carries that")

    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    check(readme.startswith("# qwen3-tts-ui"), "the README opens with the same name")

    with open(os.path.join(ROOT, "AGENTS.md"), encoding="utf-8") as fh:
        agents = fh.read()
    check("nkavassalis/qwen-tts-ui" in agents and "`nkavassalis/qwen-tts`," not in agents,
          "AGENTS.md points at the repo that exists")


def test_tabs_and_readme_agree() -> None:
    """Docs drift from the UI silently; this is the cheap check that stops it."""
    import inspect
    import re

    labels = re.findall(r'gr\.Tab\("([^"]+)", id=', inspect.getsource(webui.build_ui))
    check(labels == ["Clone a voice", "Design a voice", "Speak", "Chat", "Voices"],
          f"five top-level tabs, in order, found {labels}")
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    missing = [label for label in labels if label not in readme]
    check(not missing, f"every tab label appears in the README: missing {missing}")

    # The footer used to tell people to chunk their own text, which the app has done
    # automatically since the chunker landed.
    source = inspect.getsource(webui)
    check("split very long scripts into chunks" not in source,
          "the footer must not tell users to do what the app now does for them")


def test_environment_variables_are_documented() -> None:
    """Same idea as the setup.sh capability check: a knob nobody can find is a knob nobody uses."""
    import re

    found: set = set()
    for relative in ("webui/app.py", "scripts/say.py", "run.sh", "setup.sh"):
        with open(os.path.join(ROOT, relative), encoding="utf-8") as fh:
            found |= set(re.findall(r"QWEN_TTS_[A-Z_]+", fh.read()))
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    undocumented = sorted(name for name in found if name not in readme)
    check(not undocumented, f"undocumented environment variables: {undocumented}")


def test_ethical_use_is_stated_where_it_is_needed() -> None:
    """Cloning is the harmful-use surface; the warning belongs next to the button."""
    import inspect

    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    check("## Ethical use" in readme, "the README has an ethical-use section")
    for word in ("consent", "impersonat", "synthetic"):
        check(word in readme.lower(), f"ethical use covers {word}")

    clone_source = inspect.getsource(webui.build_ui)
    check("permission to use" in clone_source, "the Clone tab says it before you upload a clip")
    check("Ethical use" in clone_source, "and points at the section rather than replacing it")


def test_the_auto_pick_names_the_take_you_are_hearing() -> None:
    """With N players hidden and one clip in the main player, 'which one is this' must be answered."""
    import inspect

    source = inspect.getsource(webui.generate_voice_clone)
    check("kept take #" in source,
          "the first status line names the take the main player is holding")
    check("Auto-pick: take #" in source, "the takes note leads with the auto-pick, in bold")
    check("the auto-pick was take #" in source,
          "and the picker repeats it, so the override is made with the information")


def test_a_saved_voice_reaches_every_tab() -> None:
    """Saving worked; the Chat tab just never heard about it, so it looked broken.

    The library dropdowns were per-tab, so a take kept on the Clone tab was invisible to
    Speak and Chat until a reload. Every action that mutates the library now chains one
    refresh over every list of voices.
    """
    import inspect

    updates = webui.voice_library_updates()
    check(len(updates) == 6, "four dropdowns, the machine-readable list and the table")
    check(all(isinstance(update, dict) and "choices" in update for update in updates[:4]),
          "each dropdown gets its choices refreshed")
    check(isinstance(updates[4], str), "the --list box gets the text list")
    check(isinstance(updates[5], dict) and "value" in updates[5], "the Voices table gets its rows")

    source = inspect.getsource(webui.build_ui)
    for event in ("cl_save_evt", "vd_save_evt", "vo_delete_evt", "vo_prefer_evt", "vo_refresh_evt"):
        check(event in source, f"{event} must exist so its change can chain the refresh")
    check("event.then(voice_library_updates, outputs=library_outputs" in source,
          "the refresh is chained onto every library-mutating action")
    check("library_outputs = [cl_library, sp_voice, ch_voice, vo_select, sp_list, vo_table]" in source,
          "and it covers Clone, Speak, Chat, Voices and the API list box")
    check("demo.load(voice_library_updates" in source, "a fresh page load is current too")


def test_a_described_voice_is_spoken_from_its_description() -> None:
    """The reason fox ships as words: cloning keeps ~60% of a reference's top end.

    A voice that carries a description must be spoken by Voice Design, not by cloning its
    frozen clip — 2.67m of energy above 9kHz against 0.57m measured. The clip still exists
    for the Clone tab, so the two forms have to coexist without being confused.
    """
    import threading

    sr = 24000
    wav = (0.3 * np.sin(2 * np.pi * 150 * np.arange(sr * 3) / sr)).astype(np.float32)
    calls: list = []

    class FakeTTS:
        def create_voice_clone_prompt(self, **kwargs):
            calls.append(("prompt", kwargs.get("ref_text")))
            return {"fake": "prompt"}

        def generate_voice_clone(self, **kwargs):
            calls.append(("clone", kwargs["text"]))
            return [np.zeros(sr, dtype=np.float32)], sr

        def generate_voice_design(self, **kwargs):
            calls.append(("design", kwargs["instruct"], kwargs["text"]))
            return [np.zeros(sr, dtype=np.float32)], sr

    class FakeHub:
        def __init__(self):
            self.lock = threading.RLock()
            self.tts = FakeTTS()

        def acquire(self, capability):
            calls.append(("acquire", capability))
            return self.tts

    original_voices, original_hub = webui.VOICES_DIR, webui.HUB
    with tempfile.TemporaryDirectory() as vdir:
        webui.VOICES_DIR, webui.HUB = vdir, FakeHub()
        try:
            webui.save_voice(name="Fox", wav=wav, sr=sr, text="a line", instruct="bright and wry",
                             language="English", description="bright and wry", kind="design")
            webui.save_voice(name="Booth", wav=wav, sr=sr, text="another line", instruct="",
                             language="English")

            labels = dict(webui.voice_choices())
            check("Fox (described)" in [label for label, _ in webui.voice_choices()],
                  f"a described voice says so in the dropdown: {list(labels)}")
            check(webui.voice_kind(webui.list_voices()[0]) in ("design", "clip")
                  and webui.voice_kind({"description": "x"}) == "design"
                  and webui.voice_kind({"instruct": "y"}) == "design"
                  and webui.voice_kind({"description": ""}) == "clip",
                  "kind falls back to what the record actually carries")

            calls.clear()
            one, mode = webui._voice_synth("Fox", "English", 0.9, 1.0, 50, 1.05)
            one("hello there")
            check(mode == "described", f"a described voice reports itself: {mode}")
            check(("acquire", "voice_design") in calls,
                  f"described voices use the design checkpoint: {calls}")
            check(any(c[0] == "design" and c[1] == "bright and wry" for c in calls),
                  f"the description is what gets spoken: {calls}")
            check(not any(c[0] == "clone" for c in calls),
                  f"a described voice must not be cloned: {calls}")

            calls.clear()
            one, mode = webui._voice_synth("Booth", "English", 0.9, 1.0, 50, 1.05)
            one("hello there")
            check(mode == "ICL" and ("acquire", "voice_clone") in calls,
                  f"a clip voice still goes the clone route: {mode} {calls}")

            # A described voice with no clip must refuse to be loaded as a reference,
            # with a message that says what to do instead.
            webui.save_voice(name="Ghost", wav=None, sr=0, text="", instruct="low and slow",
                             language="English", description="low and slow", kind="design")
            _, _, msg = webui.load_saved_voice("ghost")
            check(msg.startswith("Error") and "described" in msg and "Speak" in msg,
                  f"loading a recipe as a reference explains the alternative: {msg}")
            audio, status = webui.speak("ghost", "", "English")
            check("text is required" in status.lower(), f"empty text is still refused: {status}")
        finally:
            webui.VOICES_DIR, webui.HUB = original_voices, original_hub


# ---------------------------------------------------------------------------
# Review fixes: scoring, reference handling, library, Clone-tab plumbing
# ---------------------------------------------------------------------------

class _FakeWhisperTokenizer:
    """Just enough of WhisperTokenizer to pin the scorer's token layout."""
    SPECIAL = {"<|startoftranscript|>": 1, "<|en|>": 2, "<|transcribe|>": 3,
               "<|notimestamps|>": 4, "<|endoftext|>": 0}
    eos_token_id = 0

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, text, add_special_tokens=True):
        self.calls.append(add_special_tokens)
        body = [10 + (ord(c) % 50) for c in text]
        if add_special_tokens:                       # what the real tokenizer does by default
            body = [1, 4] + body + [0]
        return types.SimpleNamespace(input_ids=body)

    def convert_tokens_to_ids(self, tokens):
        return [self.SPECIAL[t] for t in tokens]

    def convert_ids_to_tokens(self, token):
        return {v: k for k, v in self.SPECIAL.items()}.get(int(token), "x")


def test_whisper_confidence_scores_only_the_words() -> None:
    """The double-prefix bug: the scorer charged every take for two impossible tokens."""
    import torch

    asr = webui.Transcriber(model_id="openai/whisper-large-v3", enabled=False)
    asr.tokenizer = _FakeWhisperTokenizer()
    ids = asr._text_ids("Hi")
    check(asr.tokenizer.calls == [False], "the text is tokenised without special tokens")
    check(ids[-1] == 0 and 1 not in ids and 4 not in ids,
          f"words plus end-of-text, no start/no-timestamps prefix inside: {ids}")
    check(asr._prefix_ids("en") == [1, 2, 3, 4], "start, language, transcribe, no-timestamps")
    check(asr._prefix_ids(None) == [1, 3, 4], "no language token when it is not known")
    check(asr._detected_code(torch.tensor([[1, 2, 3, 4, 20]])) == "en",
          "the language Whisper picked is read back from its own prefix")
    check(asr._detected_code(torch.tensor([[1, 3, 4, 20]])) is None, "and None when absent")

    # Alignment: a decoder that puts all its mass on the right next token scores ~0 only
    # when the offset is len(prefix) - 1.
    prefix, target, vocab = [1, 2, 3, 4], ids, 64
    fed = prefix + target[:-1]
    logits = torch.full((1, len(fed), vocab), -30.0)
    for t in range(len(fed) - 1):
        logits[0, t, fed[t + 1]] = 30.0
    logits[0, len(fed) - 1, target[-1]] = 30.0
    good = webui._mean_token_logprob(logits, target, offset=len(prefix) - 1)
    bad = webui._mean_token_logprob(logits, target, offset=len(prefix))
    check(good is not None and good > -0.01, f"aligned scoring is ~0 for a perfect decoder: {good}")
    check(bad is None or bad < -10, f"an off-by-one is loud, not subtle: {bad}")


def test_pauses_and_snapping() -> None:
    sr = 8000
    rng = np.random.default_rng(3)
    speech = lambda s: (0.4 * rng.standard_normal(int(s * sr))).astype(np.float32)  # noqa: E731
    quiet = lambda s: (0.0005 * rng.standard_normal(int(s * sr))).astype(np.float32)  # noqa: E731
    wav = np.concatenate([speech(2.0), quiet(0.3), speech(3.0), quiet(0.3), speech(2.0)])
    gap1 = 2.15                                   # centre of the first pause
    found = webui.find_pause(wav, sr, 2.0)
    check(found is not None and abs(found - gap1) <= 0.2, f"nearest pause found: {found}")
    check(webui.find_pause(wav, sr, 3.5, radius=0.2) is None,
          "mid-speech with no pause in reach returns None")
    start, end = webui.snap_to_pauses(wav, sr, 2.05, 5.6)
    check(abs(start - gap1) <= 0.2 and 5.3 <= end <= 5.65, f"both edges moved into pauses: {start}-{end}")
    tone = (0.5 * np.sin(np.linspace(0, 2000, sr * 8))).astype(np.float32)
    check(webui.snap_to_pauses(tone, sr, 1.0, 6.0) == (1.0, 6.0),
          "no pauses at all leaves the selection alone")
    s2, e2 = webui.snap_to_pauses(wav, sr, 2.1, 2.1 + webui.REF_MIN_SECONDS)
    check(e2 - s2 >= webui.REF_MIN_SECONDS - 1e-6, "snapping never shrinks below the clone minimum")


def test_reference_report() -> None:
    sr = 16000
    rng = np.random.default_rng(5)
    body = (0.3 * rng.standard_normal(sr * 6)).astype(np.float32)
    gap = (0.0003 * rng.standard_normal(int(sr * 0.3))).astype(np.float32)
    clean = np.concatenate([gap, body[: sr * 3], gap, body[sr * 3:], gap])
    report = webui.reference_report(clean, sr, "one two three four five six seven eight nine "
                                               "ten eleven twelve thirteen fourteen")
    check("selected" in report and "Clean background" in report, f"clean clip passes: {report}")
    check("Starts and ends in a pause" in report, "edges in silence are recognised")

    noisy = clean + (0.05 * rng.standard_normal(clean.size)).astype(np.float32)
    check("Noisy background" in webui.reference_report(noisy, sr, "x"), "noise is flagged")
    cut = np.concatenate([body[: sr * 2], gap, body[sr * 2: sr * 4], gap, body[sr * 4:]])
    check("mid-word" in webui.reference_report(cut, sr, "x", start=4.0),
          "a selection that cuts speech at both edges is flagged")
    check("4.00s" in webui.reference_report(cut, sr, "x", start=4.0),
          "the flagged edge is given in the source's own timeline")
    hot = np.clip(body * 10, -1, 1)
    check("Clipped" in webui.reference_report(hot, sr, "x"), "clipping is flagged")
    check("x-vector only" in webui.reference_report(clean, sr, "", x_vector_only=True),
          "x-vector mode says the transcript is unused")
    check("No reference" in webui.reference_report(np.zeros(0, np.float32), sr),
          "an empty clip explains itself")
    check("No reference yet" in webui.reference_check(None, ""), "the UI wrapper handles no clip")


def test_prepare_reference() -> None:
    sr = 24000
    t = np.arange(sr * 4) / sr
    voice = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    padded = np.concatenate([np.zeros(sr, np.float32), voice, np.zeros(sr, np.float32)]) + 0.1
    out = webui.prepare_reference(padded, sr)
    check(abs(float(np.mean(out))) < 0.01, "DC offset removed")
    check(4.0 <= len(out) / sr <= 4.5, f"long edge silence trimmed, voice kept: {len(out) / sr:.2f}s")
    check(float(np.max(np.abs(out))) <= 1.0 and np.all(np.isfinite(out)), "finite, in range")
    quiet = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    check(abs(float(np.max(np.abs(webui.prepare_reference(quiet, sr)))) - 0.05) < 0.01,
          "level is left alone — a quiet reference clones as well as a loud one")
    short = np.zeros(int(sr * 0.5), np.float32)
    check(webui.prepare_reference(short, sr).size == short.size,
          "cleanup that would leave too little gives the input back")
    check(webui.prepare_reference(np.zeros(0, np.float32), sr).size == 0, "empty passes through")


def test_saving_never_overwrites() -> None:
    sr = 24000
    wav = (np.sin(np.linspace(0, 300, sr * 4)) * 0.3).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        original_voices, original_media = webui.VOICES_DIR, webui.MEDIA_DIR
        webui.VOICES_DIR = os.path.join(tmp, "voices")
        webui.MEDIA_DIR = os.path.join(tmp, "media")
        try:
            os.makedirs(webui.MEDIA_DIR)
            clip = os.path.join(webui.MEDIA_DIR, "ref.wav")
            sf.write(clip, wav, sr)
            _, msg = webui.save_clone_voice(webui.SAVE_REFERENCE, "Memo", clip, "hello there", "English", None)
            check(msg.startswith("Saved"), f"first save works: {msg}")
            check(webui.voice_exists("memo") and webui.voice_exists(" MEMO "), "the name is now taken")
            _, msg = webui.save_clone_voice(webui.SAVE_REFERENCE, "memo", clip, "other", "English", None)
            check("already in the library" in msg, f"a second save under the same name is refused: {msg}")
            check(webui.list_voices()[0]["text"] == "hello there", "and the original is untouched")
            _, msg, _ = webui.save_designed_voice("Memo", {"path": clip, "text": "t"})
            check("already in the library" in msg, "Design refuses it too")

            take = os.path.join(webui.MEDIA_DIR, "take.wav")
            sf.write(take, wav, sr)
            state = {"takes": [{"index": 1, "path": take, "label": "Take #1"}],
                     "text": "the new line", "language": "English"}
            _, msg = webui.save_clone_voice("1", "Take one", clip, "hello there", "English", state)
            check(msg.startswith("Saved take #1"), f"the dispatcher saves a take when asked: {msg}")
            saved = {v["slug"]: v for v in webui.list_voices()}
            check(saved["take_one"]["text"] == "the new line",
                  "a take is saved with the line it was asked to say, not the reference transcript")
            check(saved["memo"].get("prefer") == "auto", "new voices start on the automatic route")
            check(not any(f.endswith(".tmp") or ".tmp." in f for f in os.listdir(webui.VOICES_DIR)),
                  "atomic writes leave no temp files behind")
        finally:
            webui.VOICES_DIR, webui.MEDIA_DIR = original_voices, original_media


def test_voice_route_and_preference() -> None:
    sr = 24000
    wav = (np.sin(np.linspace(0, 300, sr * 2)) * 0.3).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        original, webui.VOICES_DIR = webui.VOICES_DIR, tmp
        try:
            webui.save_voice("Both", wav, sr, "line", "warm and low", "English",
                             description="warm and low", kind="design")
            webui.save_voice("Clip only", wav, sr, "line", "", "English")
            webui.save_voice("Words only", None, 0, "", "airy", "English", description="airy",
                             kind="design")
            routes = {r["slug"]: webui.voice_route(r) for r in webui.list_voices()}
            check(routes == {"both": "design", "clip_only": "clip", "words_only": "design"},
                  f"auto follows the kind: {routes}")

            webui.set_voice_preference("both", "clip")
            check(webui.voice_route(webui._find_voice("both")[0]) == "clip",
                  "a voice with both can be pinned to its clip")
            check("(" in dict((v, k) for k, v in webui.voice_choices())["both"] and
                  "described" not in dict((v, k) for k, v in webui.voice_choices())["both"],
                  "the dropdown label follows the preference")
            webui.set_voice_preference("words_only", "clip")
            check(webui.voice_route(webui._find_voice("words_only")[0]) == "design",
                  "a preference the voice cannot honour falls back instead of failing")
            msg = webui.save_voice_preference("words_only", "clip")
            check("still uses the other" in msg, f"and the UI says so: {msg}")
            try:
                webui.set_voice_preference("both", "loudly")
                raise AssertionError("expected ValueError")
            except ValueError:
                pass

            rows = webui.voice_table_rows()
            check(len(rows) == 3 and all(len(r) == len(webui.VOICE_TABLE_HEADERS) for r in rows),
                  "one table row per voice, one cell per header")
            words_row = next(r for r in rows if r[1] == "words_only")
            check(words_row[3] == "—" and words_row[2] == "description", f"no clip shown as such: {words_row}")
            details, preview, prefer = webui.voice_details("clip_only")
            check("Clip only" in details and preview is not None and preview[0] == sr,
                  "details come with a playable preview")
            check(webui.voice_details("words_only")[1] is None, "a described voice has no preview")
            check("Pick a voice" in webui.voice_details(None)[0], "nothing selected is explained")

            confirm, msg = webui.ask_delete("both")
            check(confirm["visible"] and "no undo" in msg, "delete asks first")
            _, msg, hidden = webui.confirm_delete("both")
            check("Deleted" in msg and not hidden["visible"] and not webui.voice_exists("both"),
                  "the second click deletes and hides the confirm button")
        finally:
            webui.VOICES_DIR = original


def test_load_voice_for_cloning() -> None:
    sr = 24000
    wav = (np.sin(np.linspace(0, 300, sr * 4)) * 0.3).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        original_voices, original_media = webui.VOICES_DIR, webui.MEDIA_DIR
        webui.VOICES_DIR, webui.MEDIA_DIR = os.path.join(tmp, "v"), os.path.join(tmp, "m")
        try:
            webui.save_voice("Memo", wav, sr, "hello there", "", "French")
            out = webui.load_voice_for_cloning("memo")
            check(len(out) == 9, "import outputs plus transcript and language")
            state, status, _, start, end, audio, _, text, language = out
            check(state and os.path.dirname(state["path"]) == os.path.realpath(webui.MEDIA_DIR)
                  or os.path.dirname(state["path"]) == webui.MEDIA_DIR,
                  "the clip is copied into the media dir, where the request path guard allows it")
            check(start["value"] == 0.0 and abs(end["value"] - 4.0) < 0.01,
                  "the whole clip is selected — its transcript describes all of it")
            check(text["value"] == "hello there" and language["value"] == "French",
                  "transcript and language come back with it")
            check(audio[1].shape[0] == sr * 4, "the reference player gets the clip")
            webui.save_voice("Ghost", None, 0, "", "x", "English", description="x", kind="design")
            out = webui.load_voice_for_cloning("ghost")
            check(out[0] is None and "described voice" in out[1] and len(out) == 9,
                  "a described voice is refused with the full output tuple")
            check("no saved voice" in webui.load_voice_for_cloning("nope")[1], "unknown voice")
        finally:
            webui.VOICES_DIR, webui.MEDIA_DIR = original_voices, original_media


def test_clone_validation_fills_every_output() -> None:
    """Early exits used to return 2 values to 9 outputs — a red toast, not a message."""
    for args, needle in (((None, "", "", "Auto", False), "say"),
                         ((None, "", "hello", "Auto", False), "no reference"),
                         (((24000, np.ones(48000, np.float32) * 0.1), "", "hello", "Auto", False),
                          "transcript")):
        out = webui.generate_voice_clone(*args, 0.9, 1.0, 50, 1.05)
        check(len(out) == webui.CLONE_OUTPUTS, f"{len(out)} outputs for {needle}")
        check(needle in out[1].lower(), f"the status explains it: {out[1]}")
    out = webui.generate_voice_clone((24000, np.ones(48000, np.float32) * 0.1), "hi", "hello",
                                     "Auto", False, 0.9, 1.0, 50, 1.05, seed="abc")
    check("Seed must be an integer" in out[1], "a bad seed is explained")


def test_clone_prompt_cache_is_bounded_lru() -> None:
    class FakeTTS:
        def __init__(self) -> None:
            self.calls = 0

        def create_voice_clone_prompt(self, **kw):
            self.calls += 1
            return [object()]

    tts = FakeTTS()
    webui._CLONE_PROMPTS.clear()
    clips = [np.full(100, i / 100.0, np.float32) for i in range(webui.CLONE_CACHE_SIZE + 1)]
    first = webui._clone_prompt(tts, clips[0], 24000, "a", False)
    check(webui._clone_prompt(tts, clips[0], 24000, "a", False) is first and tts.calls == 1,
          "the same reference is encoded once")
    for clip in clips[1:]:
        webui._clone_prompt(tts, clip, 24000, "a", False)
        webui._clone_prompt(tts, clips[0], 24000, "a", False)    # keep the first one hot
    check(len(webui._CLONE_PROMPTS) == webui.CLONE_CACHE_SIZE, "the cache is bounded")
    check(webui._clone_prompt(tts, clips[0], 24000, "a", False) is first,
          "recently used references survive eviction")
    webui._clone_prompt(tts, clips[0], 24000, "b", False)
    check(tts.calls == webui.CLONE_CACHE_SIZE + 2, "a different transcript is a different prompt")
    webui._CLONE_PROMPTS.clear()


def test_transcribe_reference_sets_the_language() -> None:
    class FakeASR:
        enabled = True

        def transcribe_detailed(self, wav, sr, language=None):
            return {"text": "bonjour à tous", "confidence": -0.2, "code": "fr", "language": "French"}

    sr = 24000
    wav = (np.sin(np.linspace(0, 300, sr * 4)) * 0.3).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        original_media, original_asr = webui.MEDIA_DIR, webui.ASR
        webui.MEDIA_DIR, webui.ASR = tmp, FakeASR()
        try:
            path = os.path.join(tmp, "r.wav")
            sf.write(path, wav, sr)
            state = {"path": path}
            text, status, language = webui.transcribe_reference(state, 0, 4, "Auto", "Auto")
            check(text["value"] == "bonjour à tous", "the transcript fills the box")
            check(language.get("value") == "French" and "French" in status,
                  "an Auto target language is set to what Whisper heard")
            _, _, language = webui.transcribe_reference(state, 0, 4, "Auto", "English")
            check("value" not in language, "a language the user chose is left alone")
            webui.ASR = webui.Transcriber(model_id="none")
            _, status, _ = webui.transcribe_reference(state, 0, 4, "Auto", "Auto")
            check("disabled" in status, "disabled ASR is explained")
        finally:
            webui.MEDIA_DIR, webui.ASR = original_media, original_asr


def test_runtime_status_lives_in_the_sidebar() -> None:
    import inspect

    check("HUB.status()" not in inspect.getsource(webui._ok),
          "generation statuses no longer carry the runtime line")
    original = webui.HUB
    try:
        webui.HUB = types.SimpleNamespace(status=lambda: "GPU: a | RAM: b | on GPU: -")
        check(webui.refresh_status() == "- GPU: a\n- RAM: b\n- on GPU: -", "one bullet per item")
        webui.HUB = None
        check("not loaded" in webui.refresh_status(), "before the hub exists")
    finally:
        webui.HUB = original
    source = inspect.getsource(webui.build_ui)
    check("gr.Sidebar(" in source and "gr.Timer(" in source, "a sidebar panel that refreshes itself")
    check("theme=" not in source.split("gr.Blocks(")[1].split(")")[0],
          "the theme goes to launch() in Gradio 6, not Blocks()")


def test_quick_start_cuts_only_the_first_piece_short() -> None:
    """Chat's first piece is short so the first audio arrives sooner; nothing else moves."""
    def words(t: str) -> int:
        return len(t.split())

    text = ("The build finished cleanly after the second attempt, which took a while. Three "
            "tests failed in the database layer. I fixed two of them already and the third "
            "needs a migration before it can pass. Want me to write it now?")
    normal = webui.split_for_tts(text)
    quick = webui.split_for_tts(text, quick_start=True)
    low, _target, high = webui.FIRST_CHUNK_WORDS
    check(low <= words(quick[0]) <= high, f"the first piece is short: {quick[0]!r}")
    check("".join(quick).replace(" ", "") == text.replace(" ", ""), "quick start is lossless")
    check(quick[2:] == normal[1:] and webui._join(quick[0], quick[1]) == normal[0],
          f"only the first piece is split: {normal} vs {quick}")
    check(quick[0].endswith(","), f"the cut is at punctuation: {quick[0]!r}")

    one_sentence = "Done, the deploy went out and every health check came back green on the first try."
    split = webui.split_for_tts(one_sentence, quick_start=True)
    check(len(split) == 1, f"a 2-word first half is not worth it, the text stays whole: {split}")
    two = "The deploy went out cleanly tonight. Every health check came back green on the first try."
    split = webui.split_for_tts(two, quick_start=True)
    check(split == ["The deploy went out cleanly tonight.",
                    "Every health check came back green on the first try."],
          f"a sentence boundary beats waiting for the whole reply: {split}")
    no_marks = " ".join(["word"] * 22)
    check(webui.split_for_tts(no_marks, quick_start=True) == [no_marks],
          "no punctuation, no cut: never mid-phrase")
    check(webui.split_for_tts("Short and sweet.", quick_start=True) == ["Short and sweet."],
          "a short reply is untouched")
    check(webui.split_for_tts(two) == [two], "without quick_start nothing changes")
    cjk = "今天的构建在第二次尝试后顺利完成，花了一些时间。数据库层有三个测试失败了。"
    split = webui.split_for_tts(cjk, quick_start=True)
    check("".join(split) == cjk and len(split) >= 2, f"CJK is cut too, losslessly: {split}")


def _cache_fixture(tmp: str):
    """A fake hub that counts generations, with voices, media and the cache under tmp."""
    import threading

    sr = 24000
    calls: list = []

    def tone(text: str) -> np.ndarray:
        words, _ = webui.speech_units(text)
        t = np.arange(int(max(0.5, words / 2.6) * sr)) / sr
        return (0.25 * np.sin(2 * np.pi * (150 + len(calls)) * t)).astype(np.float32)

    class FakeTTS:
        def create_voice_clone_prompt(self, **kwargs):
            return {"prompt": kwargs.get("ref_text")}

        def generate_voice_clone(self, **kwargs):
            calls.append(("clone", kwargs["text"]))
            return [tone(kwargs["text"])], sr

        def generate_voice_design(self, **kwargs):
            calls.append(("design", kwargs["text"]))
            return [tone(kwargs["text"])], sr

    class FakeHub:
        def __init__(self):
            self.lock = threading.RLock()

        def acquire(self, capability):
            return FakeTTS()

        def _checkpoint(self, capability):
            raise FileNotFoundError(capability)

    saved = (webui.VOICES_DIR, webui.MEDIA_DIR, webui.HUB, webui.RENDER_CACHE_DIR,
             webui.RENDER_CACHE_MB)
    webui.VOICES_DIR = os.path.join(tmp, "voices")
    webui.MEDIA_DIR = os.path.join(tmp, "media")
    webui.RENDER_CACHE_DIR = os.path.join(tmp, "cache")
    webui.RENDER_CACHE_MB = 64
    webui.HUB = FakeHub()
    os.makedirs(webui.VOICES_DIR)
    ref = (0.3 * np.sin(2 * np.pi * 140 * np.arange(sr * 4) / sr)).astype(np.float32)
    webui.save_voice(name="Booth", wav=ref, sr=sr, text="a line of reference text",
                     instruct="", language="English")
    webui.save_voice(name="Fox", wav=None, sr=0, text="", instruct="bright and wry",
                     language="English", description="bright and wry", kind="design")

    def restore():
        (webui.VOICES_DIR, webui.MEDIA_DIR, webui.HUB, webui.RENDER_CACHE_DIR,
         webui.RENDER_CACHE_MB) = saved
    return calls, restore


def test_render_cache_replays_a_line_already_said() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        calls, restore = _cache_fixture(tmp)
        try:
            line = "Tests passed."
            audio, status = webui.speak("booth", line, "English", postproc="off")
            check(audio is not None and len(calls) == 1, f"first time renders: {status} {calls}")
            check("cache" not in status, f"a fresh render does not claim the cache: {status}")
            first = audio[1]
            calls.clear()
            audio, status = webui.speak("booth", "  Tests   passed. ", "English", postproc="off")
            check(calls == [], f"the second time is a replay, no generation: {calls}")
            check("render cache" in status, f"the status says so: {status}")
            check(audio[1].shape == first.shape and float(np.max(np.abs(audio[1] - first))) < 1e-3,
                  "the replay is the same take (PCM_16 tolerance)")
            audio, _ = webui.speak("booth", line, "English", postproc="Warm")
            check(calls == [] and audio[0] == 48000,
                  "post-processing runs after the cache: Warm replays the raw take too")

            for variant, why in ((("booth", line, "French"), "another language"),
                                 (("booth", line, "English", 0.7), "other sampling"),
                                 (("fox", line, "English"), "another voice"),
                                 (("booth", "Tests failed.", "English"), "other words")):
                calls.clear()
                webui.speak(*variant, postproc="off")
                check(len(calls) == 1, f"{why} is a different render: {calls}")

            # Chat shares the cache: a reply whose sentences were said before replays them.
            calls.clear()
            reply = ("The deploy went out cleanly tonight. Nobody noticed a thing, which is "
                     "how it should be. Every health check came back green on the first try, "
                     "and the error rate stayed flat for the whole hour afterwards.")
            list(webui.chat_say(reply, [], "booth", "English", postproc="off"))
            rendered = len(calls)
            check(rendered >= 2, f"chat rendered its pieces: {calls}")
            calls.clear()
            turns = list(webui.chat_say(reply, [], "booth", "English", postproc="off"))
            check(calls == [], f"the same reply again is all replay: {calls}")
            check("from the render cache" in turns[-1][2] and "from cache" in turns[1][2],
                  f"chat says which pieces were replayed: {turns[1][2]} / {turns[-1][2]}")

            # Deleting the voice deletes its renders; other voices keep theirs.
            fox_files = [f for f in os.listdir(webui.RENDER_CACHE_DIR) if f.startswith("fox__")]
            _, msg = webui.delete_saved_voice("booth")
            left = os.listdir(webui.RENDER_CACHE_DIR)
            check(not any(f.startswith("booth__") for f in left) and "cached render" in msg,
                  f"a deleted voice's renders go with it: {msg} {left}")
            check(sorted(f for f in left if f.startswith("fox__")) == sorted(fox_files),
                  "other voices' renders are untouched")

            webui.RENDER_CACHE_MB = 0
            calls.clear()
            webui.speak("fox", line, "English", postproc="off")
            webui.speak("fox", line, "English", postproc="off")
            check(len(calls) == 2, "QWEN_TTS_RENDER_CACHE_MB=0 turns the cache off")
        finally:
            restore()


def test_render_cache_never_stores_a_bad_take_and_stays_bounded() -> None:
    sr = 24000
    with tempfile.TemporaryDirectory() as tmp:
        saved = webui.RENDER_CACHE_DIR, webui.RENDER_CACHE_MB, webui.RENDER_CACHE_TTL_SECONDS
        webui.RENDER_CACHE_DIR, webui.RENDER_CACHE_MB = os.path.join(tmp, "cache"), 64
        webui.RENDER_CACHE_TTL_SECONDS = 0      # put() prunes; keep the backdated files alive
        try:
            cache = webui.RenderCache("v", "identity")
            calls: list = []

            def rushed(chunk: str):
                calls.append(chunk)
                return np.full(int(0.2 * sr), 0.2, dtype=np.float32), sr   # 0.2s for 9 words

            text = "This sentence has far too many words for its audio."
            list(webui.iter_chunked(text, "English", rushed, cache=cache))
            check(cache.get(text) is None,
                  "a take that failed the rate check is not cached (it would replay forever)")

            good = (0.25 * np.sin(2 * np.pi * 150 * np.arange(sr * 4) / sr)).astype(np.float32)
            cache.put("hello there", good, sr)
            check(cache.get("hello there") is not None, "a good take is cached")
            check(webui.RenderCache("v", "other").get("hello there") is None,
                  "another identity never sees it")

            names = []
            for i in range(6):
                webui.RenderCache("v", f"id{i}").put("line", good, sr)
                names.append(webui.RenderCache("v", f"id{i}")._path("line"))
                os.utime(names[-1], (1000 + i, 1000 + i))
            size_mb = os.path.getsize(names[0]) / (1024 * 1024)
            # 7 files of one size: "hello there" (just used) plus id0..id5 (backdated).
            webui.prune_render_cache(max_age_seconds=0, max_mb=size_mb * 4.5)
            check([os.path.exists(n) for n in names] == [False] * 3 + [True] * 3,
                  "over the cap, the least recently used go first")
            os.utime(names[3], (time.time(), time.time()))
            removed = webui.prune_render_cache(max_age_seconds=3600, max_mb=1000)
            check(removed >= 2 and os.path.exists(names[3]) and not os.path.exists(names[4]),
                  "entries unused for the TTL are dropped; recently used ones stay")
            os.makedirs(webui.RENDER_CACHE_DIR, exist_ok=True)
            with open(os.path.join(webui.RENDER_CACHE_DIR, "v__broken.wav"), "w") as fh:
                fh.write("not a wav")
            broken = webui.RenderCache("v", "x")
            broken._path = lambda chunk: os.path.join(webui.RENDER_CACHE_DIR, "v__broken.wav")
            check(broken.get("anything") is None, "a damaged entry is a miss, not a crash")

            # The slug names a file this class writes, and it arrives from a voice record —
            # which `list_voices()` reads out of the sidecar JSON. So it is disk-supplied text
            # on a write path: same reason `resolve_input_path` exists for reads.
            hostile = webui.RenderCache("../../escape", "identity")
            hostile.put("a line", good, sr)
            written = hostile._path("a line")
            check(os.path.dirname(written) == webui.RENDER_CACHE_DIR and os.path.isfile(written),
                  f"a hostile slug still writes inside the cache dir: {written}")
            check(not os.path.exists(os.path.join(tmp, "..", "escape__a"))
                  and os.listdir(tmp) == ["cache"],
                  f"nothing escaped the cache directory: {os.listdir(tmp)}")
            webui.RenderCache("!!!", "identity").put("a line", good, sr)
            check(any(name.startswith("voice__") for name in os.listdir(webui.RENDER_CACHE_DIR)),
                  "a slug with nothing sluggable in it still gets a safe name")
        finally:
            (webui.RENDER_CACHE_DIR, webui.RENDER_CACHE_MB,
             webui.RENDER_CACHE_TTL_SECONDS) = saved


def test_reference_transcript_check() -> None:
    """Whisper grades the reference's transcript: the right length with the wrong words."""
    import threading

    sr = 24000
    ref = (0.3 * np.sin(2 * np.pi * 140 * np.arange(sr * 4) / sr)).astype(np.float32)
    asked: list = []

    class FakeASR:
        enabled = True
        loaded = True

        def assess(self, wav, sr, intended, language=None):
            asked.append(intended)
            if intended == "the right words here":
                return "the right words here", -0.04
            return "the right words here", -4.2

    class FakeTTS:
        model = None

        def create_voice_clone_prompt(self, **kwargs):
            return {}

        def generate_voice_clone(self, **kwargs):
            t = np.arange(int(1.2 * sr)) / sr
            return [(0.25 * np.sin(2 * np.pi * 170 * t)).astype(np.float32)], sr

    class FakeHub:
        lock = threading.RLock()

        def acquire(self, capability):
            return FakeTTS()

    saved = webui.ASR, webui.HUB, webui.MEDIA_DIR
    with tempfile.TemporaryDirectory() as tmp:
        webui.ASR, webui.HUB, webui.MEDIA_DIR = FakeASR(), FakeHub(), tmp
        webui._REF_TEXT_CHECKS.clear()
        try:
            path = os.path.join(tmp, "ref.wav")
            sf.write(path, ref, sr)
            out = webui.generate_voice_clone(path, "wrong words over here", "Hello.", "English",
                                             False, 0.9, 1.0, 50, 1.05, postproc="off")
            status = out[1]
            check(status.splitlines()[1].startswith("WARNING: Whisper does not hear"),
                  f"a mismatched transcript is the first thing the status says: {status}")
            check("the right words here" in status, "and it shows what Whisper heard")
            check(out[0] is not None, "it warns, it does not block (the threshold is provisional)")

            checklist = webui.reference_check(path, "wrong words over here")
            check("⚠️ Whisper does not hear" in checklist,
                  f"the checklist shows the verdict for this clip + transcript: {checklist}")
            check("Whisper" not in webui.reference_check(path, "something else entirely"),
                  "an edited transcript has no verdict yet: nothing stale is shown")

            asked.clear()
            webui.generate_voice_clone(path, "wrong words over here", "Hello again.", "English",
                                       False, 0.9, 1.0, 50, 1.05, postproc="off")
            check(asked == [], "a retake with the same reference does not re-run the check")

            out = webui.generate_voice_clone(path, "the right words here", "Hello.", "English",
                                             False, 0.9, 1.0, 50, 1.05, postproc="off")
            check("WARNING: Whisper" not in out[1] and "Whisper hears this transcript" in out[1],
                  f"a matching transcript passes, and says so: {out[1]}")
            check("✅ Whisper hears" in webui.reference_check(path, "the right words here"),
                  "and the checklist shows the pass")

            asked.clear()
            webui.generate_voice_clone(path, "", "Hello.", "English", True,
                                       0.9, 1.0, 50, 1.05, postproc="off")
            check(asked == [], "x-vector only ignores the transcript, so it is not checked")

            webui._REF_TEXT_CHECKS.clear()
            webui.ASR.loaded = False
            webui.generate_voice_clone(path, "fresh words", "Hello.", "English", False,
                                       0.9, 1.0, 50, 1.05, postproc="off", candidates=1)
            check(asked == [], "one take with Whisper unloaded: no surprise 3GB load")
        finally:
            webui.ASR, webui.HUB, webui.MEDIA_DIR = saved
            webui._REF_TEXT_CHECKS.clear()


def test_cjk_is_scored_per_character() -> None:
    """WER and the transcript-length check must work on scripts without spaces."""
    exact, one_off = "今日はいい天気ですね", "今日は悪い天気ですね"
    check(webui.word_error_rate(exact, exact) == 0.0, "exact Japanese is WER 0")
    check(0.05 < webui.word_error_rate(one_off, exact) < 0.3,
          f"one word changed is a small non-zero WER: {webui.word_error_rate(one_off, exact)}")
    check(webui.word_error_rate("오늘은 날씨가 좋네요", "오늘은 날씨가 나쁘네요") > 0, "Hangul too")
    check(webui.transcript_fit(3.0, exact) == "", "10 characters fit a 3s clip")
    check("too short" in webui.transcript_fit(3.0, "今日は"), "3 characters do not")
    check("too long" in webui.transcript_fit(3.0, exact * 3), "30 characters do not")



def _lab():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "scripts"))
    import multi_clip_lab
    return multi_clip_lab


def test_multi_clip_helpers() -> None:
    """Folds, medoid, averaged embedding, screening, concat, loudness, the summary."""
    import random
    lab = _lab()
    try:
        lab.make_folds(2, 4)
        check(False, "two clips cannot make a held-out fold")
    except ValueError:
        pass
    folds = lab.make_folds(6, 4)
    check(len(folds) == 4 and len({h for h, _ in folds}) == 4, f"4 distinct held-out: {folds}")
    check(all(h not in b and len(b) == 5 for h, b in folds), "held-out never in its build set")
    check(len(lab.make_folds(3, 10)) == 3, "folds capped at the clip count")

    e = [np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.1, 0.0]), np.array([0.8, 0.2, 0.1]),
         np.array([0.0, 0.0, 1.0])]
    check(lab.medoid_index(e[:3]) == 1, "the middle vector is the medoid")
    check(lab.other_index(3, 0) == 1 and lab.other_index(3, 2) == 0, "other is never the medoid")
    big = [np.array([10.0, 0.0]), np.array([0.0, 1.0])]
    m = lab.mean_embedding(big)
    check(abs(np.linalg.norm(m) - 5.5) < 1e-4, f"rescaled to the mean norm: {np.linalg.norm(m)}")
    check(abs(m[0] - m[1]) < 1e-4, "averages directions, so the big-norm clip does not dominate")
    check(m.dtype == np.float32, "float32 out")
    flags = lab.screen_clips(e)
    check(flags[3]["outlier"] and not any(f["outlier"] for f in flags[:3]),
          f"the orthogonal clip is the outlier: {flags}")

    sr = 1000
    wavs = [np.ones(8000, np.float32), np.ones(8000, np.float32), np.ones(8000, np.float32)]
    joined, text, used = lab.concat_clips(wavs, ["a", "b", "c"], sr, max_seconds=20.0, gap=0.5)
    check(used == 2 and text == "a b" and len(joined) == 16500, f"capped at 20s: {used} {len(joined)}")
    joined, _, used = lab.concat_clips([np.ones(30000, np.float32)], ["long"], sr, max_seconds=20.0)
    check(used == 1 and len(joined) == 30000, "always keeps the first clip")

    rng = np.random.default_rng(0)
    quiet, loud = 0.01 * rng.standard_normal(24000), 0.5 * rng.standard_normal(24000)
    a, b = lab.loudness_match(quiet), lab.loudness_match(loud)
    check(abs(np.sqrt(np.mean(a ** 2)) - np.sqrt(np.mean(b ** 2))) < 1e-3, "same RMS")
    check(np.max(np.abs(b)) <= 0.95 + 1e-6, "peak-limited without clipping")
    check(not np.any(lab.loudness_match(np.zeros(10))), "silence stays silence")
    t = np.arange(48000) / 48000.0
    check(lab.high_band_milli(np.sin(2 * np.pi * 200 * t), 48000) < 1, "a 200Hz tone has no top end")
    check(lab.high_band_milli(np.sin(2 * np.pi * 12000 * t), 48000) > 990, "a 12kHz tone is all top")
    check(lab.high_band_milli(np.ones(100), 16000) is None, "no band above Nyquist")

    trials = lab.build_trials(2, 2, [1, 2], lab.ARM_KEYS, random.Random(0))
    check(len(trials) == 8 and len({t["id"] for t in trials}) == 8, "one trial per fold×sent×seed")
    check(all(sorted(t["order"]) == sorted(lab.ARM_KEYS) for t in trials), "every arm in each")
    check(len({tuple(t["order"]) for t in trials}) > 1, "orders are shuffled")

    takes = [{"trial": "t1", "arm": a, "sim_heldout": s, "wer": 0.0}
             for a, s in (("medoid", 0.80), ("medoid_mean", 0.85), ("other", 0.70))]
    ratings = {"t1": {"scores": {"medoid": 3, "medoid_mean": 4, "other": 3}, "best": "medoid_mean"}}
    rows = {r["arm"]: r for r in lab.summarise(takes, ratings)}
    check(rows["medoid_mean"]["listen_wtl"] == (1, 0, 0), "a listener win against the baseline")
    check(rows["other"]["listen_wtl"] == (0, 1, 0) and rows["other"]["sim_wtl"] == (0, 0, 1),
          "a tie on ears, a loss on SIM")
    check("listen_wtl" not in rows["medoid"], "the baseline is not compared with itself")
    check(rows["medoid_mean"]["best_share"] == 1.0, "picked best every time")
    md = lab.summary_markdown(list(rows.values()), 1)
    check("biased" in md and "medoid_mean" in md, "the table says which column is biased")


def test_multi_clip_lab_end_to_end_with_a_fake_model() -> None:
    """Screen, render every arm, rate blind through the UI's own handlers, read the results."""
    import threading
    import torch
    lab = _lab()
    sr = 24000

    def embed(wav):
        spec = np.abs(np.fft.rfft(np.asarray(wav, np.float64)[:24000], n=24000))[:2000]
        return torch.tensor(spec.reshape(16, -1).sum(axis=1), dtype=torch.bfloat16)

    seen_prompts: list = []

    class FakeModel:
        def extract_speaker_embedding(self, audio, sr):
            assert sr == 24000
            return embed(audio)

    class FakeTTS:
        model = FakeModel()

        def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
            from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
            wav, _ = ref_audio
            return [VoiceClonePromptItem(ref_code=torch.zeros(len(wav) // 1920, 16),
                                         ref_spk_embedding=embed(wav), x_vector_only_mode=False,
                                         icl_mode=True, ref_text=ref_text)]

        def generate_voice_clone(self, text, language, voice_clone_prompt, max_new_tokens, **kw):
            item = voice_clone_prompt[0]
            seen_prompts.append(item)
            assert item.ref_spk_embedding.dtype == torch.bfloat16, "mean cast back to model dtype"
            hz = 120 + float(item.ref_spk_embedding.float().argmax()) * 7 + torch.rand(1).item()
            n = int(sr * 1.5)
            return [(0.3 * np.sin(2 * np.pi * hz * np.arange(n) / sr)).astype(np.float32)], sr

    class FakeHub:
        lock = threading.RLock()

        def acquire(self, capability):
            assert capability == "voice_clone"
            return FakeTTS()

    def speech(hz, seconds=4.0):
        t = np.arange(int(sr * seconds)) / sr
        env = (np.sin(2 * np.pi * 1.5 * t) > -0.3).astype(np.float32)
        return (0.3 * env * np.sin(2 * np.pi * hz * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        clips = [{"name": f"c{i}.wav", "wav": speech(hz), "sr": sr, "text": f"clip number {i}"}
                 for i, hz in enumerate((150, 155, 160, 900))]
        lab_obj = lab.Lab(FakeHub(), asr=None, out_root=tmp)
        clips = lab_obj.screen(clips, "English")
        check(all("embedding" in c and "report" in c for c in clips), "screened")
        run_dir = lab_obj.render(clips, ["One sentence.", "", "Two sentences."], [7], 4,
                                 "English", True)
        run = lab.load_run(os.path.basename(run_dir), tmp)
        check(len(run["trials"]) == 4 * 2, f"4 folds × 2 sentences: {len(run['trials'])}")
        check(len(run["takes"]) == 8 * len(lab.ARM_KEYS), "one take per arm per trial")
        check(all(os.path.isfile(os.path.join(run_dir, d, t["file"]))
                  for t in run["takes"] for d in ("takes", "listen")), "raw and listening copies")
        check(all(t["sim_heldout"] is not None and t["sim_build"] is not None for t in run["takes"]),
              "every take scored against held-out and centroid")
        xv = [p for p in seen_prompts if p.x_vector_only_mode]
        check(xv and all(p.ref_code is None and p.ref_text is None for p in xv),
              "x-vector arm carries no codes and no transcript")
        for f in run["folds"]:
            check(f["heldout"] not in f["build"] and f["medoid"] != f["other"], f"fold sane: {f}")

        demo = lab.build_lab_ui(lab_obj, [])
        fns = {getattr(bf.fn, "__name__", ""): bf.fn for bf in demo.fns.values()}
        run_id = os.path.basename(run_dir)
        out = fns["show_trial"](run_id)
        check(out[0] and "Trial 1 of 8" in out[1] and os.path.isfile(out[2]), "first trial shown")
        n = len(lab.ARM_KEYS)
        out = fns["save_and_next"](run_id, out[0], None, "", *(["3"] * n))
        check("Rate every take" in out[1] and not lab.load_ratings(run_id, tmp),
              "an incomplete rating is refused, nothing saved")
        for i in range(8):
            trial_id = out[0]
            check(bool(trial_id), f"trial {i} available")
            trial = next(t for t in run["trials"] if t["id"] == trial_id)
            scores = ["5" if arm == "medoid_mean" else "2" for arm in trial["order"]]
            best = "ABCDE"[trial["order"].index("medoid_mean")]
            out = fns["save_and_next"](run_id, trial_id, best, "", *scores)
        check(out[0] is None and "All 8 trials rated" in out[1], "done after the last trial")
        ratings = lab.load_ratings(run_id, tmp)
        check(all(r["best"] == "medoid_mean" and r["scores"]["medoid_mean"] == 5
                  for r in ratings.values()), "letters were mapped back to the right arms")
        md = lab.results_markdown(run_id, tmp)
        check("8/0/0" in md and "100%" in md, f"results reflect the ratings:\n{md}")
        check(any(c["outlier"] for c in run["clips"]) and "outlier" in md,
              "the 900Hz clip is flagged and reported")
        try:
            lab.load_run("../" + run_id, tmp)
            check(False, "a run id cannot walk out of the output dir")
        except ValueError:
            pass


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
