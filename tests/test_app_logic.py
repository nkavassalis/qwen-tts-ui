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
