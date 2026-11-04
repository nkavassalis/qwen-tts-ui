# coding=utf-8
"""
qwen3-tts-ui — a local Qwen3-TTS Gradio web UI.

Four tabs: design-a-voice, clone-a-voice, a one-call **Speak** endpoint for scripts
(`scripts/say.py`), and a **Chat** tab that streams a reply sentence by sentence. A saved
voice is clip + transcript, or a description; `voice_kind` picks the route, and the
description is the higher-fidelity one (see the measurement in that docstring).

Adapted from the official Hugging Face Space (Qwen/Qwen3-TTS app.py) for a single
local GPU. Differences from the Space version:

  * Loads the 1.7B checkpoints (the best quality tier) from a local directory.
  * The Space keeps all 5 models resident on the GPU at once (~20GB VRAM). A 16GB
    Blackwell card (RTX 5070 Ti / 5080) cannot do that, so models live in CPU RAM and an
    LRU "GPU slot" moves at most `--max-resident` models onto the GPU on demand (~1-3s swap).
  * No `spaces` (Zero-GPU) dependency, no model-size dropdown, attn_implementation
    is `sdpa` (flash-attn 2 has no sm_120/Blackwell wheels; the Space uses sdpa too).
  * Long text is split on sentence boundaries and stitched, because this checkpoint
    accelerates past ~30-45s of audio per pass and intelligibility goes with the pace.

Voice cloning is the part of this that can do harm: use it on voices you have permission to
use. The README's *Ethical use* section spells out what that means.

Run:  ./run.sh            (or)  .venv/bin/python webui/app.py --ip 127.0.0.1 --port 7860
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gc
import getpass
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from PIL import Image, ImageDraw

from qwen_tts import Qwen3TTSModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.environ.get("QWEN_TTS_MODELS_DIR", os.path.join(ROOT, "models"))

# (tab name, checkpoint dir name, model type)
# Two checkpoints, two ways to make a voice: describe one, or clone one. The third
# upstream capability (CustomVoice, nine preset speakers) is not shipped — see the
# README: the product is your voices, not Alibaba's, and every preset was a path around
# the thing this app is for.
CAPABILITIES = OrderedDict(
    [
        ("voice_design", ("Qwen3-TTS-12Hz-1.7B-VoiceDesign", "voice_design")),
        ("voice_clone", ("Qwen3-TTS-12Hz-1.7B-Base", "base")),
    ]
)

# (UI label, value passed to the model). Values must match `talker_config.codec_language_id`
# keys *and* upstream's supported-language filter, which excludes dialect values: dialects
# were a property of the CustomVoice preset speakers (dylan -> beijing_dialect, eric ->
# sichuan_dialect), and that capability is not shipped. Italian is included — the hosted
# Space omits it by oversight, the checkpoint accepts it.
LANGUAGE_CHOICES = [
    ("Auto (detect)", "Auto"),
    ("Chinese", "Chinese"),
    ("English", "English"),
    ("Japanese", "Japanese"),
    ("Korean", "Korean"),
    ("French", "French"),
    ("German", "German"),
    ("Spanish", "Spanish"),
    ("Portuguese", "Portuguese"),
    ("Russian", "Russian"),
    ("Italian", "Italian"),
]
LANGUAGE_VALUES = {value for _, value in LANGUAGE_CHOICES}

# Designed voices are persisted here so a VoiceDesign result can be reused as a
# clone speaker across requests and restarts (the hosted demo cannot, it is stateless).
VOICES_DIR = os.path.join(ROOT, "voices")

# Only one process may own the checkpoints at a time. Two of them (e.g. the server plus
# tests/smoke_test.py) means two full copies of the weights in RAM, and on WSL2 — where
# the VM balloons towards the host's whole RAM — that has hard-locked the machine.
LOCK_PATH = os.environ.get("QWEN_TTS_LOCK", os.path.join(ROOT, ".qwen-tts.lock"))

# Memory budget. This box is WSL2 with dynamic memory capped at the host's total RAM,
# so over-committing does not just OOM the distro, it can take down Windows.
RAM_FLOOR_MB = int(os.environ.get("QWEN_TTS_RAM_FLOOR_MB", "4096"))
VRAM_MARGIN_MB = int(os.environ.get("QWEN_TTS_VRAM_MARGIN_MB", "1024"))


def mem_available_mb() -> float:
    """MemAvailable from /proc/meminfo, or +inf when the platform does not report it.

    Unknown is deliberately NOT 0.0: `_load()` compares this against the RAM floor, so
    returning 0 on a machine without /proc/meminfo (macOS) would make every single load
    raise MemoryError. Callers that *display* it must handle +inf (see `status()`).
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return float("inf")


def vram_available_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free_b, _ = torch.cuda.mem_get_info()
    return free_b / 2**20


@contextmanager
def exclusive_owner(who: str) -> Iterator[None]:
    """Advisory single-owner lock; raises RuntimeError if someone else holds it."""
    handle = open(LOCK_PATH, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.seek(0)
        holder = handle.read().strip() or "unknown process"
        handle.close()
        raise RuntimeError(
            f"Another Qwen3-TTS process is already holding the weights: {holder}. "
            f"A second one can add ~9GB or more of RAM (up to two ~4.3GB checkpoints plus "
            f"Whisper), which on WSL2 "
            f"can hard-lock the machine. Stop it first (pgrep -af 'webui/app[.]py|smoke_test')."
        )
    handle.seek(0)
    handle.truncate()
    handle.write(f"{who} pid={os.getpid()} started={datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    handle.flush()
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

MAX_NEW_TOKENS_CAP = 2048
# The 12Hz codec emits ~12.5 frames (~1 token) per 0.08s of audio. Budget generously
# ~4 tokens per character so long words / CJK never get truncated, but never let a
# runaway decode burn 3+ minutes (the checkpoint default is max_new_tokens=8192 and
# the model occasionally fails to emit EOS, especially for ICL voice cloning).
TOKENS_PER_CHAR = 4.0
MIN_NEW_TOKENS = 64
DEVICE = torch.device("cuda")

# Checkpoint defaults from generation_config.json, surfaced in the UI so they can be tuned.
SAMPLING_DEFAULTS = dict(temperature=0.9, top_p=1.0, top_k=50, repetition_penalty=1.05)


# Reference-clip guidance for the Base (clone) model. The model card advertises a
# "3-second rapid voice clone"; in practice a clean 3-10s window clones best, and very
# long references waste time and make ICL runs more likely to ramble.
REF_TARGET_SECONDS = 6.0
REF_MIN_SECONDS = 2.0
REF_MAX_SECONDS = 15.0
REF_SAMPLE_RATE = 24000

# Media ingest: ffmpeg is used to strip the audio track out of video uploads.
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv", ".mpeg", ".mpg", ".3gp"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".aiff", ".aif", ".wma", ".amr"}
MEDIA_SUFFIXES = sorted(VIDEO_SUFFIXES | AUDIO_SUFFIXES)
MAX_MEDIA_SECONDS = 600         # only the first 10 minutes of an upload are read
FFMPEG_TIMEOUT_SECONDS = 120

# Transcription of the reference transcript (the ICL clone needs it to match the clip).
ASR_MODEL_ID = "openai/whisper-large-v3"
ASR_LANGUAGE_CODES = {
    "chinese": "zh", "english": "en", "japanese": "ja", "korean": "ko", "french": "fr",
    "german": "de", "spanish": "es", "portuguese": "pt", "russian": "ru", "italian": "it",
}


def _budget(text: str, cap: int = MAX_NEW_TOKENS_CAP) -> int:
    """Token budget scaled to the prompt length (guards against non-stopping runs)."""
    return int(min(cap, max(MIN_NEW_TOKENS, len(text.strip()) * TOKENS_PER_CHAR)))


def _sampling(temperature, top_p, top_k, repetition_penalty) -> Dict[str, Any]:
    return dict(
        temperature=float(temperature), top_p=float(top_p),
        top_k=int(top_k), repetition_penalty=float(repetition_penalty),
    )


# ============================================================================
# Quality: transcript/clip fit, and scoring best-of-N candidates
# ============================================================================
#
# Both exist because of what was measured on the 5070 Ti box (README "Getting the best
# quality"): decode knobs barely move SIM/WER, but a transcript that does not match the
# clip is catastrophic (WER 1.0 and 3x the expected length), and sampling variance alone
# can turn a good take into a bad one (WER 0.000 vs 0.133 across seeds).

WORDS_PER_SECOND = 2.5      # ~150 wpm narration; the fit check's English rate
CHARS_PER_SECOND = 4.0      # CJK is counted per character, not per word
FIT_RATIO_LOW = 0.5
FIT_RATIO_HIGH = 2.0
MAX_CANDIDATES = 4          # best-of-N ceiling; each candidate is a full generation

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_WORD_RE = re.compile(r"[0-9A-Za-z\u00c0-\u024f']+")


def speech_units(text: str) -> Tuple[int, int]:
    """(words, cjk_chars) actually present — the two scripts need different rates."""
    if not text:
        return 0, 0
    cjk = len(_CJK_RE.findall(text))
    words = len(_WORD_RE.findall(_CJK_RE.sub(" ", text)))
    return words, cjk


def transcript_fit(seconds: float, text: str, x_vector_only: bool = False) -> str:
    """Non-blocking warning when the transcript cannot possibly be what the clip says.

    ICL cloning conditions on the reference transcript, so a mismatch does not just sound
    worse — the model keeps decoding (measured: 17.4s of audio for a 5s sentence, WER 1.0
    against 0.0). The check is deliberately coarse: a factor-of-2 window on words-per-
    second, because its job is to catch "transcript of the whole upload" and "left the
    placeholder in", not to nitpick a fast speaker.
    """
    if x_vector_only or not seconds or seconds <= 0:
        return ""
    words, cjk = speech_units(text)
    if not words and not cjk:
        return ("No transcript — ICL cloning needs one. Transcribe the clip, or tick "
                "'x-vector only' to clone from the speaker embedding alone.")
    ratios = []
    if words:
        ratios.append(words / (seconds * WORDS_PER_SECOND))
    if cjk:
        ratios.append(cjk / (seconds * CHARS_PER_SECOND))
    ratio = max(ratios)  # whichever script fits better, so mixed text is not punished
    spoken = (words / WORDS_PER_SECOND) if words else (cjk / CHARS_PER_SECOND)
    if ratio < FIT_RATIO_LOW:
        return (f"Transcript looks too short for a {seconds:.1f}s clip: {words + cjk} word(s) "
                f"is about {spoken:.1f}s of speech. A mismatched transcript makes the model "
                f"ramble — transcribe the selection, or tick 'x-vector only'.")
    if ratio > FIT_RATIO_HIGH:
        return (f"Transcript looks too long for a {seconds:.1f}s clip: {words + cjk} word(s) "
                f"is about {spoken:.1f}s of speech. It probably covers more than the shaded "
                f"selection — cut it down to what that stretch actually says, or widen it.")
    return ""


def word_error_rate(hypothesis: str, reference: str) -> float:
    """Word-level edit distance / reference length; case and punctuation insensitive."""
    def tokens(text: str) -> List[str]:
        return [t.lower() for t in _WORD_RE.findall(text or "")] + list(_CJK_RE.findall(text or ""))

    hyp, ref = tokens(hypothesis), tokens(reference)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, start=1):
        current = [i]
        for j, r in enumerate(ref, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (h != r)))
        previous = current
    return previous[-1] / float(len(ref))


def take_label(index: int, wer: Optional[float], sim: Optional[float], kept: bool,
               confidence: Optional[float] = None, defects: Optional[Dict[str, Any]] = None) -> str:
    """Label for a candidate's player: which take it was and how it scored.

    The scores are shown so the automatic pick is arguable rather than mysterious — the
    player itself is the override, since you can listen to every take and download any one.
    """
    bits = []
    if wer is not None:
        bits.append(f"WER {wer:.2f}")
    if confidence is not None:
        bits.append(f"conf {confidence:+.2f}")
    if sim is not None:
        bits.append(f"SIM {sim:.3f}")
    if defects:
        flags = defect_flags(defects)
        if flags:
            bits.append("⚠ " + ", ".join(flags))
    return f"Take #{index}" + (f" — {' · '.join(bits)}" if bits else "") + \
           (" (kept)" if kept else "")


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------
# WER alone was a weak judge: it is a hard 0/1 per word, so a take that mumbles and a take
# that is merely slightly wrong score the same, and it says nothing about a clip that is
# half silence or full of clipped peaks. Two additions, both cheap and offline:
#
#   * Whisper's own sequence log-probability (`confidence`). Where WER is a guess about
#     discrete words, confidence is a graded answer about the whole audio: garble, doubled
#     words and hallucinated tails all push it down well before WER notices.
#   * A defect report off the waveform: speaking rate against the text length, leading and
#     trailing silence, and clipping. These are the ways a take is "wrong" in a way no
#     transcript-based metric can see.
#
# The weights below are heuristics on a common scale, not a trained model: WER and
# confidence are the signals with evidence behind them (see README), the defect penalties
# are deliberately small so a slightly long tail of silence cannot outvote a take that says
# the words wrong.

# Mean per-token decoder log-prob of the *intended* text (plus end-of-text), teacher-forced
# behind Whisper's own prefix. Clean speech scores close to zero; a take Whisper cannot
# match to the text falls well below -1. The previous value (-2.60) was calibrated against
# a scorer that fed the start-of-transcript prefix twice (see `Transcriber._text_ids`), which
# dragged every take down by the cost of two impossible tokens. -1.0 is provisional until
# it is re-measured on the GPU box (`tests/smoke_test.py --only best_of_n` prints each
# take's confidence).
CONF_GOOD = -1.00          # below this a take is penalised
WER_WEIGHT = 2.0           # content dominates: no defect or confidence gap outranks it
CONF_WEIGHT = 0.4          # a full log-prob below the line costs less than 20% wrong words
DEFECT_CAP = 0.3           # defects alone cannot outrank a take that says the words wrong
RATE_RATIO_LOW, RATE_RATIO_HIGH = 0.6, 1.8   # plausible band around WORDS_PER_SECOND
SILENCE_RMS = 0.005        # 20ms window RMS below this counts as silence
SILENCE_BUDGET = 0.5       # seconds of lead+tail silence tolerated before a penalty
CLIP_FRACTION = 0.001      # share of samples at full scale before it counts as clipping


def _rms_windows(wav: np.ndarray, sr: int, window: int) -> np.ndarray:
    if wav.size == 0:
        return np.zeros(0, dtype=np.float32)
    # ceil, not floor: the last partial window still counts as a window. With floor the pad
    # width went negative for any length that was not an exact multiple of `window`, which
    # numpy rejects — it hid for as long as every caller passed round-number clip lengths.
    frames = int(np.ceil(wav.size / window))
    padded = np.pad(wav, (0, frames * window - wav.size))
    return np.sqrt(np.mean(padded.reshape(frames, window).astype(np.float64) ** 2, axis=1))


def defect_report(wav: np.ndarray, sr: int, text: str) -> Dict[str, Any]:
    """Waveform-level defects, plus the penalty they contribute to ranking.

    Pure and CPU-only on purpose: these are the checks that must keep working when Whisper
    is disabled, and they cost microseconds rather than a GPU pass.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    sr = int(sr) or 1
    seconds = wav.size / float(sr)
    report: Dict[str, Any] = {"seconds": round(seconds, 3), "rate": 0.0, "rate_ratio": 0.0,
                              "lead_silence": 0.0, "tail_silence": 0.0,
                              "clipped_fraction": 0.0, "penalty": 0.0}
    if wav.size == 0:
        report["penalty"] = 1.0
        return report

    words, cjk = speech_units(text)
    if words or cjk:
        expected = (WORDS_PER_SECOND if not cjk or words / max(1, words + cjk) > 0.5
                    else CHARS_PER_SECOND)
        rate = (words + cjk) / max(seconds, 1e-6)
        report["rate"] = round(rate, 2)
        report["rate_ratio"] = round(rate / expected, 2)

    window = max(1, int(0.02 * sr))
    rms = _rms_windows(wav, sr, window)
    if rms.size:
        loud = rms >= SILENCE_RMS
        lead = int(np.argmax(loud)) if loud.any() else int(rms.size)
        tail = int(np.argmax(loud[::-1])) if loud.any() else int(rms.size)
        report["lead_silence"] = round(lead * window / float(sr), 3)
        report["tail_silence"] = round(tail * window / float(sr), 3)

    report["clipped_fraction"] = round(float(np.mean(np.abs(wav) >= 0.999)), 5)

    penalty = 0.0
    if report["rate_ratio"]:
        ratio = report["rate_ratio"]
        if ratio < RATE_RATIO_LOW:
            penalty += min(0.5, (RATE_RATIO_LOW - ratio) * 0.6)
        elif ratio > RATE_RATIO_HIGH:
            penalty += min(0.5, (ratio - RATE_RATIO_HIGH) * 0.4)
    excess = (report["lead_silence"] + report["tail_silence"]) - SILENCE_BUDGET
    if excess > 0:
        penalty += min(0.3, excess * 0.3)
    if report["clipped_fraction"] > CLIP_FRACTION:
        penalty += 0.4 if report["clipped_fraction"] > 0.01 else 0.2
    report["penalty"] = round(penalty, 3)
    return report


def defect_flags(report: Dict[str, Any]) -> List[str]:
    """Human-readable version of the defects, for the player labels and the status line."""
    flags = []
    if report.get("rate_ratio") and (report["rate_ratio"] < RATE_RATIO_LOW
                                     or report["rate_ratio"] > RATE_RATIO_HIGH):
        flags.append(f"rate {report['rate_ratio']:.2f}x")
    if report.get("lead_silence", 0) + report.get("tail_silence", 0) > SILENCE_BUDGET:
        flags.append(f"{report['lead_silence'] + report['tail_silence']:.1f}s silence")
    if report.get("clipped_fraction", 0) > CLIP_FRACTION:
        flags.append("clipping")
    return flags


def candidate_penalty(wer: Optional[float], confidence: Optional[float],
                      defects: Optional[Dict[str, Any]]) -> float:
    """One number per candidate, lower is better. Missing signals cost nothing.

    The weights encode one rule: **saying the words wrong is the worst thing a take can
    do**. WER carries weight 2, so a 20%-wrong take still loses to a take that is quiet,
    slightly silent at the edges and only mildly unintelligible. Confidence and defects
    exist to separate takes whose WER is a tie — which, measured, is most of them.
    """
    penalty = 0.0
    if wer is not None:
        penalty += WER_WEIGHT * float(wer)
    if confidence is not None and confidence < CONF_GOOD:
        penalty += CONF_WEIGHT * (CONF_GOOD - float(confidence))
    if defects:
        penalty += min(DEFECT_CAP, float(defects.get("penalty", 0.0)))
    return round(penalty, 4)


def choose_best_candidate(metrics: List[Dict[str, Optional[float]]]) -> int:
    """Index of the best candidate: lowest penalty, then closest voice, then order.

    Similarity is only a tie-breaker, and only a narrow one — measured across seeds it
    varies by ~0.001, so it decides near-identical takes and nothing else. A candidate
    whose content is wrong must never win on timbre.
    """
    if not metrics:
        raise ValueError("no candidates to choose from")

    def key(item: Tuple[int, Dict[str, Any]]):
        index, m = item
        penalty = m.get("penalty")
        if penalty is None:
            penalty = candidate_penalty(m.get("wer"), m.get("confidence"), m.get("defects"))
        sim = m.get("sim")
        return (round(float(penalty), 2),
                -(float("inf") if sim is None else float(sim)), index)

    return min(enumerate(metrics), key=key)[0]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _to_24k(wav: np.ndarray, sr: int) -> np.ndarray:
    """The speaker encoder only accepts 24kHz mono (it asserts)."""
    wav = np.asarray(wav, dtype=np.float32)
    if int(sr) == 24000:
        return wav
    import librosa
    return np.asarray(librosa.resample(y=wav, orig_sr=int(sr), target_sr=24000), dtype=np.float32)


# ---------------------------------------------------------------------------
# Output post-processing
# ---------------------------------------------------------------------------
# The vocoder is fixed at 24kHz (tokenizer config: output_sample_rate 24000,
# decode_upsample_rate 1920), so nothing can add real bandwidth above 12kHz. But the raw
# decode measured 0.155% of its energy in the top 5% of the band (11.4-12kHz) against
# 0.036% for the same signal resampled properly — a 4x excess of near-Nyquist imaging,
# which is what people hear as "digital". Doing the 24->48kHz step here with soxr HQ also
# stops the browser guessing at it; a cheap linear resampler cost 20% of the high-band
# energy in the same measurement.
#
# Every mode below was scored with the model's own speaker encoder: SIM moved from 0.9900
# (raw) to 0.9893 (clean) and 0.9879 (warm) — inside the 0.001 seed-to-seed spread, i.e.
# likeness is not being traded for the nicer sound.

POSTPROCESS_MODES = ("off", "clean", "warm")
POSTPROCESS_DEFAULT = "warm"
POSTPROCESS_LABELS = {
    "off": "Off (raw 24kHz from the vocoder)",
    "clean": "Clean — HQ resample to 48kHz only",
    "warm": "Warm — 48kHz, de-image, gentle air (recommended)",
}
POSTPROCESS_SR = 48000


def postprocess_output(wav: np.ndarray, sr: int, mode: str = POSTPROCESS_DEFAULT
                       ) -> Tuple[np.ndarray, int]:
    """Shape the decoded audio for listening. Never touches the model or the clone path.

    Returns the raw signal unchanged if the dependencies are missing or the mode is
    unknown — this is polish, and it must not be able to break a generation.
    """
    mode = (mode or POSTPROCESS_DEFAULT).strip().lower()
    wav = np.asarray(wav, dtype=np.float32)
    if mode not in POSTPROCESS_MODES or mode == "off" or wav.size == 0:
        return wav, int(sr)
    try:
        import soxr
        from scipy.signal import butter, sosfiltfilt
    except ImportError:
        return wav, int(sr)

    try:
        out_sr = POSTPROCESS_SR
        out = np.asarray(soxr.resample(wav, int(sr), out_sr, quality="HQ"), dtype=np.float32)
        if mode == "warm":
            # sosfiltfilt needs room for its edge padding; a clip that short is a rounding
            # error anyway, so hand back the resampled signal rather than filtering it.
            if out.size <= 48:
                return out, out_sr
            # Cut the vocoder's imaging near Nyquist, then put back the openness it removes.
            sos = butter(10, 11500.0 / (out_sr / 2.0), btype="low", output="sos")
            out = np.asarray(sosfiltfilt(sos, out.astype(np.float64)), dtype=np.float32)
            sos = butter(2, 8000.0 / (out_sr / 2.0), btype="high", output="sos")
            air = np.asarray(sosfiltfilt(sos, out.astype(np.float64)), dtype=np.float32)
            out = (out + 0.45 * air).astype(np.float32)
            out = (np.tanh(1.25 * out) / np.tanh(1.25)).astype(np.float32)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.0:  # the shelf/saturation can nudge past full scale; keep it legal
            out = (out / peak).astype(np.float32)
        return out, out_sr
    except Exception:  # noqa: BLE001 - polish must never sink a generation
        return wav, int(sr)


# ============================================================================
# Long-text chunking
# ============================================================================
# Measured on this checkpoint: a single pass keeps its timbre (SIM 0.981 and first-vs-last
# 0.986 at 74s of audio) but its **tempo accelerates** — 3.05 words/s at 18 words, 3.39 at
# 60, 3.28 at 150, 4.04 at 300 — and Whisper WER follows it from 0.00 to 0.68. So the limit
# is roughly 30-45 seconds of audio, not the token cap (300 words needs 929 of 2048 tokens;
# raising the cap to 4096 changed nothing, because EOS arrives on its own) and not VRAM
# (7.52GB at 221 tokens vs 7.58GB at 929). Long text is therefore synthesised in
# sentence-sized pieces and stitched: same total tokens, same ~1.35x realtime.

CHUNK_MIN_WORDS, CHUNK_TARGET_WORDS, CHUNK_MAX_WORDS = 12, 18, 26
CHUNK_MIN_CJK, CHUNK_TARGET_CJK, CHUNK_MAX_CJK = 20, 30, 42
CHUNK_THRESHOLD = 25      # at or below this many units: one pass, the measured safe zone
# Chat only: the first piece is cut short so the first audio arrives sooner (generation time
# tracks audio length). Only at punctuation, never mid-phrase, and only when both halves stay
# long enough to carry prosody. Unmeasured for quality — see AGENTS.md.
FIRST_CHUNK_WORDS = (4, 7, 10)      # min, target, max
FIRST_CHUNK_CJK = (6, 10, 16)
MAX_CHUNKS = 40           # one GPU, one request — ask for a shorter script instead
CROSSFADE_MS = 15         # equal-power join; a hard splice clicks
SILENCE_KEEP_MS = 60      # trim the decode's edge padding, keep consonant air

# Dots that do not end a sentence. Replacing them with placeholders before scanning is what
# stops "Mr. Smith drove 3.5 miles to St. John's at 1. Main St." from becoming four chunks
# that the model then reads with sentence-final intonation at every false boundary.
_PROTECT_PATTERNS = (
    r"https?://\S+|\bwww\.\S+",                 # urls
    r"\b[\w.+-]+@[\w.-]+\.\w+",                 # emails
    r"\b(?:[A-Z]\.)+",                           # initials: J. R. R.
    r"\b\d+(?:\.\d+)+\b",                        # decimals, versions, ip-ish
    r"\b(?:Mr|Mrs|Ms|Messrs|Dr|Prof|St|vs|etc|approx|Inc|Ltd|Co|Fig|Jr|Sr|al|No|Dept|Ave|Rd|"
    r"p\.ex|e\.g|i\.e)\.(?=\s|$)",               # abbreviations (incl. common Italian)
    r"\.{3,}|…",                                 # an ellipsis is not a full stop
    r"(?m)^[ \t]*\d+\.(?=\s)",                   # a list marker: "1. first item"
)
# Full-width ！？ and 。 are terminators too — omitting them makes a page of Chinese arrive
# as one enormous chunk, since the ASCII set never matches.
# Full-width ideographic terminators are included by escape, not by literal: typing them
# risks a lookalike (an ASCII '.' silently stood in for U+3002 once), and omitting them makes
# a page of Chinese arrive as one enormous chunk because nothing ever matched.
_TERMINATORS = ".!?…\u3002\uff01\uff1f"
_CLOSERS = "\"'”»’)}]"
_OPENERS = "\"'“«([{（《"
_CLAUSE_RE = re.compile(r"[,;:—–]| and | but | which | that | where | when | perché | ma ")


def _units(text: str) -> int:
    """How long this is *to speak*: words for spaced scripts, characters for CJK."""
    words, cjk = speech_units(text)
    return cjk if cjk > words else words


def _protect(text: str) -> Tuple[str, List[str]]:
    """Blank out dots that must not split; the saved list puts them back verbatim."""
    saved: List[str] = []

    def keep(match: "re.Match[str]") -> str:
        saved.append(match.group(0))
        return f"\x00{len(saved) - 1}\x00"

    protected = text
    for pattern in _PROTECT_PATTERNS:
        protected = re.sub(pattern, keep, protected)
    return protected, saved


def _restore(text: str, saved: List[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m.group(1))], text)


def _is_cjk(ch: str) -> bool:
    return bool(ch) and bool(_CJK_RE.match(ch))


def _join(left: str, right: str) -> str:
    """Glue two pieces back together — no space across a CJK boundary."""
    if not left.strip():
        return right
    if not right.strip():
        return left
    if _is_cjk(left[-1]) or _is_cjk(right[0]):
        return left + right
    return f"{left} {right}"


def _boundary_ahead(prot: str, index: int) -> bool:
    """Does a real sentence follow this terminator? Uppercase, a digit, or CJK."""
    k = index
    while k < len(prot) and prot[k] == " ":
        k += 1
    if k >= len(prot):
        return True
    nxt = prot[k]
    return nxt.isupper() or nxt.isdigit() or _is_cjk(nxt) or nxt in _OPENERS


def _split_sentences(text: str) -> List[str]:
    """Split on true sentence terminators only, keeping every character."""
    prot, saved = _protect(text)
    sentences: List[str] = []
    buffer: List[str] = []
    i, n = 0, len(prot)
    while i < n:
        char = prot[i]
        buffer.append(char)
        i += 1
        if char not in _TERMINATORS:
            continue
        while i < n and prot[i] in _CLOSERS:      # close the quote before the break
            buffer.append(prot[i])
            i += 1
        if _boundary_ahead(prot, i):
            sentence = _restore("".join(buffer), saved).strip()
            if sentence:
                sentences.append(sentence)
            buffer = []
            while i < n and prot[i] == " ":
                i += 1
    tail = _restore("".join(buffer), saved).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_clauses(sentence: str, ceiling: int, depth: int = 0) -> List[str]:
    """Cut an over-long sentence at the clause boundary nearest its middle."""
    if _units(sentence) <= ceiling or depth > 6:
        return [sentence]
    best: Optional[Tuple[str, str]] = None
    best_cost: Optional[int] = None
    for match in _CLAUSE_RE.finditer(sentence):
        left, right = sentence[:match.end()], sentence[match.end():]
        if not left.strip() or not right.strip():
            continue
        cost = abs(_units(left) - _units(right))
        if best_cost is None or cost < best_cost:
            best, best_cost = (left, right), cost
    if best is None:                              # no punctuation to lean on: cut mid-phrase
        words = sentence.split()
        if len(words) < 2:
            return [sentence]
        middle = len(words) // 2
        return (_split_clauses(" ".join(words[:middle]), ceiling, depth + 1)
                + _split_clauses(" ".join(words[middle:]), ceiling, depth + 1))
    # The separator stays with the left piece: a trailing comma keeps the model from using
    # sentence-final intonation in the middle of a thought.
    return (_split_clauses(best[0].strip(), ceiling, depth + 1)
            + _split_clauses(best[1].strip(), ceiling, depth + 1))


def _quick_first(chunks: List[str], cjk: bool) -> List[str]:
    """Cut the first chunk short, at a sentence or punctuation boundary, if one fits.

    Sentence boundaries beat clause punctuation; among those, the cut whose first half is
    closest to the target wins. Word-boundary cuts (" and ", " which ") are not used here:
    ending the very first thing you hear on "and" sounds broken. Lossless — a slice.
    """
    if not chunks:
        return chunks
    low, target, high = FIRST_CHUNK_CJK if cjk else FIRST_CHUNK_WORDS
    first = chunks[0]
    if _units(first) <= high:
        return chunks
    cuts: List[Tuple[int, int]] = []           # (rank, index): rank 0 sentence, 1 clause
    position = 0
    for sentence in _split_sentences(first)[:-1]:
        found = first.find(sentence, position)
        if found < 0:
            break
        position = found + len(sentence)
        cuts.append((0, position))
    cuts += [(1, m.end()) for m in re.finditer(r"[,;:—–，；：、]", first)]
    best: Optional[Tuple[Tuple[int, int], int]] = None
    for rank, index in cuts:
        left, right = first[:index].strip(), first[index:].strip()
        if not left or not right:
            continue
        size = _units(left)
        if not (low <= size <= high) or _units(right) < low:
            continue
        score = (rank, abs(size - target))
        if best is None or score < best[0]:
            best = (score, index)
    if best is None:
        return chunks
    index = best[1]
    return [first[:index].strip(), first[index:].strip()] + chunks[1:]


def split_for_tts(text: str, language: Optional[str] = None,
                  quick_start: bool = False) -> List[str]:
    """Slice text into pieces the model speaks without rushing. Lossless.

    Sentences first, then the measured length band in both directions: fragments are merged
    (a two-word chunk arrives clipped and toneless — prosody needs context) and over-long
    sentences are cut at their middlest clause boundary. Concatenating the result reproduces
    the input with whitespace ignored; nothing is reworded, reordered or dropped.

    `quick_start` (Chat) additionally cuts the first piece short (`_quick_first`), because
    in a streamed reply the wait for the first piece is the wait you notice.
    """
    text = (text or "").strip()
    if not text:
        return []
    cjk = _is_cjk(text.lstrip()[0])
    if _units(text) <= CHUNK_THRESHOLD:
        return _quick_first([text], cjk) if quick_start else [text]
    if cjk:
        low, target, ceiling = CHUNK_MIN_CJK, CHUNK_TARGET_CJK, CHUNK_MAX_CJK
    else:
        low, target, ceiling = CHUNK_MIN_WORDS, CHUNK_TARGET_WORDS, CHUNK_MAX_WORDS

    pieces: List[str] = []
    for sentence in _split_sentences(text):
        pieces.extend(_split_clauses(sentence, ceiling))

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
            continue
        combined = _units(current) + _units(piece)
        if combined <= target or (combined <= ceiling and _units(current) < low):
            current = _join(current, piece)
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)

    # A stub tail is worse than a slightly long chunk: it sounds truncated.
    if len(chunks) > 1 and _units(chunks[-1]) < low:
        if _units(chunks[-2]) + _units(chunks[-1]) <= int(ceiling * 1.35):
            chunks[-2] = _join(chunks[-2], chunks[-1])
            chunks.pop()
    chunks = [c.strip() for c in chunks if c.strip()]
    if quick_start:
        chunks = _quick_first(chunks, cjk)
    if len(chunks) > MAX_CHUNKS:
        raise MediaError(
            f"That is {len(chunks)} chunks — over the {MAX_CHUNKS} allowed in one request on "
            f"one GPU. Split the script into shorter passages and generate them one at a time.")
    return chunks


def trim_edges(wav: np.ndarray, sr: int, keep_ms: int = SILENCE_KEEP_MS) -> np.ndarray:
    """Drop a decode's leading/trailing padding, keeping a little air either side."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    window = max(1, int(0.02 * sr))
    rms = _rms_windows(wav, sr, window)
    if not rms.size:
        return wav
    loud = np.flatnonzero(rms >= SILENCE_RMS)
    if not loud.size:
        return wav
    keep = int(keep_ms / 1000.0 * sr)
    start = max(0, int(loud[0]) * window - keep)
    end = min(wav.size, (int(loud[-1]) + 1) * window + keep)
    return wav[start:end]


def stitch_clips(clips: List[np.ndarray], sr: int,
                 crossfade_ms: int = CROSSFADE_MS) -> np.ndarray:
    """Trim each piece and join them with an equal-power crossfade.

    Naive concatenation of per-chunk decodes gives you the padding of each take as a stutter
    of gaps, and a hard splice clicks because two waveforms rarely agree at the seam.
    """
    pieces = [np.asarray(c, dtype=np.float32).reshape(-1) for c in clips]
    pieces = [trim_edges(c, sr) for c in pieces if c.size]
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    fade = int(crossfade_ms / 1000.0 * sr)
    if fade <= 0 or len(pieces) == 1:
        return np.concatenate(pieces)
    out = pieces[0]
    for nxt in pieces[1:]:
        overlap = min(fade, out.size // 2, nxt.size // 2)
        if overlap <= 0:
            out = np.concatenate([out, nxt])
            continue
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        joined = out[-overlap:] * np.sqrt(1.0 - ramp) + nxt[:overlap] * np.sqrt(ramp)
        out = np.concatenate([out[:-overlap], joined, nxt[overlap:]])
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:                      # the crossfade can sum two peaks together
        out = out / peak
    return out.astype(np.float32)


def iter_chunked(text: str, language: Optional[str], generate_one: Any,
                 progress: Any = None, cache: Any = None, quick_start: bool = False):
    """Yield one dict per spoken piece: index, total, text, wav, sr, rerolled, cached.

    A generator rather than a loop that returns a finished clip, because the chat tab wants
    to hand you each piece the moment it exists — the model cannot stream samples, but it can
    stream sentences, and the first one lands in ~6-9s instead of after the whole reply.

    `cache` (a `RenderCache`, Speak/Chat only) is consulted per piece. A take is stored only
    if it passed the rate and clipping checks — a cached take is replayed every time, so a
    bad one must never get in.
    """
    chunks = split_for_tts((text or "").strip(), language, quick_start=quick_start)
    total = len(chunks)
    for index, chunk in enumerate(chunks):
        if progress is not None:
            try:
                progress((index + 1) / (total + 1.0), desc=f"Chunk {index + 1} of {total}")
            except Exception:  # noqa: BLE001 - a progress bar is never a reason to fail
                pass
        hit = cache.get(chunk) if cache is not None else None
        if hit is not None:
            yield {"index": index, "total": total, "text": chunk, "wav": hit[0], "sr": hit[1],
                   "rerolled": False, "cached": True}
            continue
        wav, sr = generate_one(chunk)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        sr = int(sr)
        rerolled = False
        report = defect_report(wav, sr, chunk)
        if any(flag.startswith("rate") for flag in defect_flags(report)):
            retry, retry_sr = generate_one(chunk)
            retry = np.asarray(retry, dtype=np.float32).reshape(-1)
            second = defect_report(retry, retry_sr, chunk)
            if abs(second["rate_ratio"] - 1.0) < abs(report["rate_ratio"] - 1.0):
                wav, sr, rerolled, report = retry, int(retry_sr), True, second
        if cache is not None:
            flags = defect_flags(report)
            if not any(f.startswith("rate") or f == "clipping" for f in flags):
                cache.put(chunk, wav, sr)
        yield {"index": index, "total": total, "text": chunk, "wav": wav, "sr": sr,
               "rerolled": rerolled, "cached": False}


def generate_chunked(text: str, language: Optional[str], generate_one: Any,
                     progress: Any = None, cache: Any = None) -> Tuple[np.ndarray, int, List[str]]:
    """Synthesise `text`, in pieces when it is long enough to rush.

    `generate_one(chunk_text) -> (wav, sr)` is the model call, injected so this works for all
    three capabilities and can be tested without weights. A piece whose speaking rate lands
    out of band is re-rolled once — that is the failure this whole path exists to avoid, and
    one retry costs ~10s against re-running the whole document.
    """
    pieces: List[np.ndarray] = []
    seconds: List[str] = []
    notes: List[str] = []
    sr = REF_SAMPLE_RATE
    cached = 0
    for piece in iter_chunked(text, language, generate_one, progress, cache=cache):
        sr = piece["sr"]
        cached += bool(piece.get("cached"))
        if piece["rerolled"]:
            notes.append(f"chunk {piece['index'] + 1} re-rolled to hold a natural pace")
        if piece["total"] == 1:
            if cached:
                notes.append("From the render cache (said before in this voice)")
            return piece["wav"], sr, notes      # one pass: the model's own decode, untouched
        pieces.append(piece["wav"])
        seconds.append(f"{len(piece['wav']) / sr:.1f}")
    if not pieces:
        return np.zeros(0, dtype=np.float32), sr, notes
    out = stitch_clips(pieces, sr)
    if cached:
        notes.append(f"{cached} of {len(pieces)} pieces from the render cache")
    notes.append(f"Chunked: {len(pieces)} pieces ({'+'.join(seconds)}s) stitched with a "
                 f"{CROSSFADE_MS}ms crossfade")
    return out, int(sr), notes


# ============================================================================
# Model loading + VRAM management
# ============================================================================
class ModelHub:
    """Keeps loaded models in CPU RAM; only a couple sit on the GPU at once.

    Weights are lazy-loaded from disk on first use (~4.3GB each for the 1.7B
    checkpoints). Before each generation the requested model is moved to the GPU and
    least-recently-used models are pushed back to CPU RAM, so the VRAM working set stays
    bounded. Both tiers are capped and guarded:

      * at most `max_cpu_resident` checkpoints are pinned in host RAM (LRU beyond that,
        re-read from disk on demand). This is the safety valve: on WSL2 the VM will
        happily balloon until the Windows host locks up, so unbounded pinning is unsafe.
      * a model is only loaded/promoted if there is room to spare (`ram_floor_mb`,
        `vram_margin_mb`); otherwise the request fails loudly instead of swapping.
    """

    def __init__(self, models_dir: str, dtype: torch.dtype, attn_impl: str, max_resident: int,
                 max_cpu_resident: int = 2, ram_floor_mb: int = RAM_FLOOR_MB,
                 vram_margin_mb: int = VRAM_MARGIN_MB):
        self.models_dir = models_dir
        self.dtype = dtype
        self.attn_impl = attn_impl
        self.max_resident = max(1, max_resident)
        self.max_cpu_resident = max(1, max_cpu_resident)
        self.ram_floor_mb = ram_floor_mb
        self.vram_margin_mb = vram_margin_mb
        self.cpu_cache: "OrderedDict[str, Qwen3TTSModel]" = OrderedDict()
        self.on_gpu: "OrderedDict[str, Qwen3TTSModel]" = OrderedDict()
        self.lock = threading.RLock()
        self._sizes: Dict[str, int] = {}

    def _checkpoint(self, capability: str) -> str:
        dirname, _ = CAPABILITIES[capability]
        path = os.path.join(self.models_dir, dirname)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Model not found at {path}. Download it with:\n"
                f"  .venv/bin/hf download Qwen/{dirname} --local-dir {path}"
            )
        return path

    def weight_mb(self, capability: str) -> float:
        """Approximate resident size: the checkpoint's safetensors bytes (bf16 on disk)."""
        if capability not in self._sizes:
            total = 0
            for root, _, files in os.walk(self._checkpoint(capability)):
                for fname in files:
                    if fname.endswith(".safetensors"):
                        total += os.path.getsize(os.path.join(root, fname))
            self._sizes[capability] = total
        return self._sizes[capability] / 2**20

    def _free_cpu(self, need_mb: float, protect: Optional[str] = None) -> None:
        """Drop LRU CPU-only models until `need_mb` fits above the RAM floor."""
        candidates = [name for name in self.cpu_cache
                      if name not in self.on_gpu and name != protect]
        while mem_available_mb() - need_mb < self.ram_floor_mb and candidates:
            name = candidates.pop(0)
            print(f"[hub] dropping {name} from CPU RAM (RAM floor {self.ram_floor_mb}MB)", flush=True)
            self.cpu_cache.pop(name, None)
            gc.collect()

    def _load(self, capability: str) -> Qwen3TTSModel:
        path = self._checkpoint(capability)
        need = self.weight_mb(capability)
        self._free_cpu(need, protect=capability)
        if mem_available_mb() - need < self.ram_floor_mb:
            raise MemoryError(
                f"Only {mem_available_mb():.0f}MB available; loading {capability} "
                f"(~{need:.0f}MB) would leave less than the {self.ram_floor_mb}MB floor. "
                f"Free some RAM or lower --max-cpu-resident. On WSL2 also check the host: "
                f"the VM balloons into Windows' own memory."
            )
        print(f"[hub] loading {path} -> CPU RAM (~{need:.0f}MB, "
              f"{mem_available_mb():.0f}MB free) ...", flush=True)
        tts = Qwen3TTSModel.from_pretrained(
            path,
            device_map="cpu",
            dtype=self.dtype,
            attn_implementation=self.attn_impl,
        )
        tts.device = torch.device("cpu")
        # Surfaced because the sampling sliders are only meaningful if the checkpoint's
        # own generate_config.json leaves do_sample on (upstream lets that file override
        # every default in `_merge_generate_kwargs`).
        print(f"[hub] {capability} ready in CPU RAM ({mem_available_mb():.0f}MB free); "
              f"generate defaults: {getattr(tts, 'generate_defaults', 'n/a')}", flush=True)
        return tts

    def _get_cpu(self, capability: str) -> Qwen3TTSModel:
        if capability not in self.cpu_cache:
            if len(self.cpu_cache) >= self.max_cpu_resident:
                for name in list(self.cpu_cache):
                    if name not in self.on_gpu:
                        print(f"[hub] dropping {name} from CPU RAM "
                              f"(max_cpu_resident={self.max_cpu_resident})", flush=True)
                        self.cpu_cache.pop(name)
                        gc.collect()
                        break
            self.cpu_cache[capability] = self._load(capability)
        self.cpu_cache.move_to_end(capability)
        return self.cpu_cache[capability]

    @staticmethod
    def _move(tts: Qwen3TTSModel, device: torch.device) -> None:
        """Move the talker *and* the speech tokenizer to `device`.

        `model.speech_tokenizer` is a Qwen3TTSTokenizer wrapper, not an nn.Module, so
        `model.to(device)` silently leaves the 12Hz vocoder on CPU — which shows up as
        ~50x slower generation (aten::slow_conv_dilated2d dominates the profile). Both
        the wrapper's inner module and its cached `.device` must be updated.
        """
        tts.model.to(device)
        for attr in vars(tts.model).values():
            inner = getattr(attr, "model", None)
            if hasattr(attr, "device") and isinstance(inner, torch.nn.Module):
                inner.to(device)
                attr.device = device
        tts.device = device

    def acquire(self, capability: str) -> Qwen3TTSModel:
        """Return `capability` resident on the GPU, evicting LRU models if needed."""
        with self.lock:
            tts = self._get_cpu(capability)
            if capability in self.on_gpu:
                self.on_gpu.move_to_end(capability)
                return tts

            need = self.weight_mb(capability) + self.vram_margin_mb
            # Only a CUDA target has VRAM to budget. `vram_available_mb()` reports 0 with
            # no CUDA device, which used to make `--device cpu` refuse every load.
            if DEVICE.type == "cuda":
                while vram_available_mb() < need and self.on_gpu:
                    evicted_name, evicted = self.on_gpu.popitem(last=False)
                    print(f"[hub] evicting {evicted_name} -> CPU RAM "
                          f"({vram_available_mb():.0f}MB free, need {need:.0f}MB)", flush=True)
                    self._move(evicted, torch.device("cpu"))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                if vram_available_mb() < need:
                    raise MemoryError(
                        f"Need ~{need:.0f}MB of VRAM for {capability} but only "
                        f"{vram_available_mb():.0f}MB is free. Close other GPU users "
                        f"(nvidia-smi) or lower --max-resident."
                    )

            print(f"[hub] promoting {capability} -> {DEVICE} "
                  f"({vram_available_mb():.0f}MB free)", flush=True)
            self._move(tts, DEVICE)
            self.on_gpu[capability] = tts
            return tts

    def status(self) -> str:
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            vram = f"{free_b / 2**30:.1f}GB free / {total_b / 2**30:.1f}GB total"
        else:
            vram = "no CUDA device"
        ram = mem_available_mb()
        ram_txt = "unknown" if ram == float("inf") else f"{ram / 1024:.1f}GB free"
        return (
            f"GPU: {vram} | RAM: {ram_txt} | "
            f"in CPU RAM: {', '.join(self.cpu_cache) or '-'} | "
            f"on GPU: {', '.join(self.on_gpu) or '-'}"
        )


HUB: Optional[ModelHub] = None


def _ok(audio: Tuple[int, np.ndarray], message: str) -> Tuple[Any, str]:
    """Gradio output pair: (audio, status line).

    The runtime line (VRAM, RAM, which models are where) used to be appended to every
    status, so each tab's status box was half diagnostics. It lives in the sidebar now and
    refreshes on its own.
    """
    return audio, message


def _post_note(mode: str) -> str:
    mode = (mode or POSTPROCESS_DEFAULT).strip().lower()
    if mode not in POSTPROCESS_MODES:
        mode = POSTPROCESS_DEFAULT
    return "" if mode == "off" else f"\nPost-processing: {POSTPROCESS_LABELS[mode]}"


def _present(wav: np.ndarray, sr: int, budget: int, label: str, mode: str,
             extra: Optional[List[str]] = None) -> Tuple[Tuple[int, np.ndarray], str]:
    """Report on the model's own decode; hand back the post-processed listening copy."""
    report = _report(wav, sr, budget, label)
    if extra:
        report += "\n" + "\n".join(extra)
    out_wav, out_sr = postprocess_output(wav, sr, mode)
    return (out_sr, out_wav), report + _post_note(mode)


def _report(wav: np.ndarray, sr: int, budget: int, label: str) -> str:
    """Human-readable result line; flags generations that look like a non-stopping run."""
    seconds = len(wav) / sr
    msg = f"{label} OK — {seconds:.1f}s of audio"
    if budget >= MAX_NEW_TOKENS_CAP and seconds > 120:
        msg += " (hit the token cap — the model did not stop; raise the reference quality or repetition penalty)"
    return msg


# ============================================================================
# Audio helpers (same normalisation as the official Space app)
# ============================================================================
def _normalize_audio(wav, eps: float = 1e-12, clip: bool = True) -> np.ndarray:
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def _audio_to_tuple(audio) -> Optional[Tuple[np.ndarray, int]]:
    """Gradio Audio value -> (wav float32, sample_rate).

    The reference/clip boxes are `type="filepath"`, so the normal case is a path string
    handed over by Gradio — which is client-supplied, hence `resolve_input_path` before
    anything opens it. The tuple/dict forms are kept for tests and for callers that
    already hold a waveform.
    """
    if audio is None:
        return None
    if isinstance(audio, dict) and audio.get("path"):
        return _audio_to_tuple(audio["path"])
    if isinstance(audio, (str, os.PathLike)):
        wav, sr = sf.read(resolve_input_path(audio, "reference audio"), dtype="float32", always_2d=False)
        return _normalize_audio(wav), int(sr)
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        return _normalize_audio(wav), int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        return _normalize_audio(audio["data"]), sr
    return None


# ============================================================================
# Media ingest: video / audio upload -> mono reference waveform
# ============================================================================
class MediaError(RuntimeError):
    """Anything the user can fix: bad file, no audio track, missing ffmpeg."""


# ---------------------------------------------------------------------------
# Path containment for anything the browser names (defense in depth)
# ---------------------------------------------------------------------------
# Gradio 6 does gate inbound file paths: `Blocks.process_api` -> `preprocess_data` ->
# `processing_utils._check_allowed(payload.path, check_in_upload_folder)` refuses to move
# a file into its cache unless a user actually uploaded it (verified live, 2026-11-02:
# posting {"path": "/etc/hostname.wav"} to /gradio_api/call/import_media dies with
# "...because it was not uploaded by a user"). So this guard is the *belt*, not the
# braces. It stays because (a) the components themselves do no such check —
# `components/file.py::_process_single_file` uses `f.path` straight off the payload — so
# a Gradio upgrade that loosens preprocess_data silently reopens an arbitrary media read;
# (b) it also covers paths that reach us by other routes (gr.State contents, and
# QWEN_TTS_ALLOWED_MEDIA_DIRS if someone widens it); and (c) it turns an opaque
# framework-level InvalidPathError into a message the user can act on.
MEDIA_DIR = os.path.join(ROOT, "tmp", "media")

# Uploaded clips are working data, not an archive: they are re-read on every slider
# drag, then nothing needs them. Prune anything older than this (0 disables pruning).
MEDIA_TTL_SECONDS = int(os.environ.get("QWEN_TTS_MEDIA_TTL_SECONDS", str(6 * 3600)))


def media_dir() -> str:
    os.makedirs(MEDIA_DIR, exist_ok=True)
    return MEDIA_DIR


def allowed_input_roots() -> List[str]:
    """Real, absolute directories a client-supplied path may live in."""
    roots = [MEDIA_DIR]
    try:
        from gradio import utils as gradio_utils
        for getter in ("get_upload_folder", "get_cache_folder"):
            fn = getattr(gradio_utils, getter, None)
            if callable(fn):
                roots.append(str(fn()))
    except Exception:  # noqa: BLE001 - gradio moves these helpers between versions
        pass
    # Where Gradio writes the files it hands back to the browser: every (sr, wav) we
    # return to a clip box is saved there by postprocess, and the browser then sends
    # that exact path back to us as an input. Ask the components themselves rather than
    # assuming it is the same as the upload dir — it has not always been.
    try:
        from gradio.components import Audio as _GradioAudio, File as _GradioFile
        for component in (_GradioAudio, _GradioFile):
            cache = getattr(component(type="filepath"), "GRADIO_CACHE", None)
            if cache:
                roots.append(str(cache))
    except Exception:  # noqa: BLE001
        pass
    roots += [p for p in os.environ.get("QWEN_TTS_ALLOWED_MEDIA_DIRS", "").split(os.pathsep) if p]
    return [os.path.realpath(r) for r in roots if r]


def resolve_input_path(path: Any, what: str = "media file") -> str:
    """Validate a client-supplied path and return its real path, or raise MediaError.

    Containment is checked *before* existence, so a probe cannot use this to find out
    whether /opt/recordings/something.wav exists.
    """
    if not path or not isinstance(path, (str, os.PathLike)):
        raise MediaError(f"No {what} was provided.")
    text = os.fspath(path)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        raise MediaError(f"{what.capitalize()} URLs are not accepted — upload the file instead.")
    real = os.path.realpath(os.path.abspath(text))
    if not any(real == root or real.startswith(root + os.sep) for root in allowed_input_roots()):
        raise MediaError(f"Refusing to read a {what} outside the upload directory.")
    if not os.path.isfile(real):
        raise MediaError(f"{what.capitalize()} not found.")
    return real


def prune_media(max_age_seconds: Optional[float] = None) -> int:
    """Delete extracted clips older than `max_age_seconds`; returns how many went.

    The TTL is read from the module global at call time (not bound as a default) so
    tests and `--media-ttl` style overrides can change it after import.
    """
    if max_age_seconds is None:
        max_age_seconds = MEDIA_TTL_SECONDS
    if max_age_seconds <= 0:
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    try:
        entries = os.listdir(MEDIA_DIR)
    except OSError:
        return 0
    for fname in entries:
        path = os.path.join(MEDIA_DIR, fname)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            continue  # someone else is mid-request with this file; leave it
    return removed



def ffmpeg_extract_audio(src: str, dst: str, sr: int = REF_SAMPLE_RATE) -> str:
    """Strip the audio track out of `src` (video or audio) into a mono PCM wav."""
    if not shutil.which(FFMPEG_BIN):
        raise MediaError(
            f"`{FFMPEG_BIN}` not found on PATH — video uploads need ffmpeg "
            f"(apt install ffmpeg, or pass --ffmpeg-bin)."
        )
    # `-t` caps what is decoded: a reference needs seconds, and an uncapped two-hour upload
    # would become ~700MB of float32 in RAM plus a waveform drawn from all of it.
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-t", str(MAX_MEDIA_SECONDS),
           "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise MediaError(f"ffmpeg took longer than {FFMPEG_TIMEOUT_SECONDS}s to read "
                         f"{os.path.basename(str(src))} — try a shorter or simpler file.")
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {proc.returncode}"
        raise MediaError(f"Could not read audio from {os.path.basename(str(src))}: {tail}")
    return dst


def load_media(path: str, sr: int = REF_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load any supported upload as a float32 mono waveform at `sr`.

    The path comes from the browser, so it is confined to `allowed_input_roots()` here,
    once, at the only public entry point.
    """
    return _load_media(resolve_input_path(path), sr)


def _load_media(path: str, sr: int) -> Tuple[np.ndarray, int]:
    """`load_media` for paths we already validated (or wrote ourselves)."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in MEDIA_SUFFIXES:
        raise MediaError(f"Unsupported file type {suffix or '(none)'}. Supported: {', '.join(MEDIA_SUFFIXES)}")

    if suffix not in AUDIO_SUFFIXES:
        # ffmpeg output lands in MEDIA_DIR so the recursive call re-validates inside an
        # allowed root instead of escaping to a system temp dir.
        fd, extracted = tempfile.mkstemp(prefix="qwen-tts-media-", suffix=".wav", dir=media_dir())
        os.close(fd)
        try:
            return _load_media(ffmpeg_extract_audio(path, extracted, sr), sr)
        finally:
            with suppress(OSError):
                os.remove(extracted)

    try:
        frames = int(MAX_MEDIA_SECONDS * sf.info(path).samplerate)
        wav, in_sr = sf.read(path, frames=frames, dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - unsupported codec: let ffmpeg decode it
        fd, extracted = tempfile.mkstemp(prefix="qwen-tts-media-", suffix=".wav", dir=media_dir())
        os.close(fd)
        try:
            return _load_media(ffmpeg_extract_audio(path, extracted, sr), sr)
        finally:
            with suppress(OSError):
                os.remove(extracted)

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1).astype(np.float32)
    if int(in_sr) != int(sr):
        import librosa
        wav = librosa.resample(y=wav, orig_sr=int(in_sr), target_sr=int(sr))
    return wav, int(sr)


def trim_window(duration: float, start: Optional[float], end: Optional[float],
                target: float = REF_TARGET_SECONDS, minimum: float = REF_MIN_SECONDS,
                maximum: float = REF_MAX_SECONDS) -> Tuple[float, float]:
    """Clamp a user selection into a usable clone window.

    Empty selections become a `target`-second window; too-short selections grow around
    their centre, too-long ones shrink, and everything stays inside [0, duration].
    """
    duration = max(0.0, float(duration))
    if duration == 0.0:
        raise MediaError("The selected media has no audio to trim.")
    maximum = min(maximum, duration)
    minimum = min(minimum, maximum)

    start = min(max(0.0, float(start or 0.0)), duration)
    end = min(max(float(end if end is not None else 0.0), 0.0), duration)
    if end <= start:
        start = min(start, max(0.0, duration - target))
        end = min(duration, start + target)

    if end - start < minimum:
        centre = (start + end) / 2.0
        start, end = centre - minimum / 2.0, centre + minimum / 2.0
        start = max(0.0, min(start, duration - minimum))
        end = start + minimum

    if end - start > maximum:
        centre = (start + end) / 2.0
        start, end = centre - maximum / 2.0, centre + maximum / 2.0
        start = max(0.0, min(start, duration - maximum))
        end = start + maximum

    return round(start, 3), round(end, 3)


def slice_audio(wav: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    a = max(0, int(round(start * sr)))
    b = min(len(wav), int(round(end * sr)))
    return wav[a:max(b, a + 1)]


# ---------------------------------------------------------------------------
# Reference analysis: where the pauses are, and how clean the stretch is
# ---------------------------------------------------------------------------
# ICL cloning hands the talker the reference's own codec codes and asks it to *continue*
# them. A reference cut off in the middle of a word asks it to continue half a phoneme,
# and the transcript then has to describe half a word, which Whisper rarely gets right. So
# the selection should start and end in a pause. A pause is found relative to the clip's
# own levels (its noise floor and its speech level), not a fixed dBFS, so the same rule
# works for a quiet phone memo and a loud podcast.

FRAME_SECONDS = 0.02
PAUSE_SEARCH_SECONDS = 0.35     # how far auto-pick moves an edge to land it in a pause
MIN_DYNAMIC_DB = 10.0           # below this spread there are no pauses to find (music, tone)
NOISY_SNR_DB = 20.0             # under this the background is audible in a clone
REF_HIGHPASS_HZ = 60.0          # below a deep voice's fundamental; above mains hum and rumble
REF_EDGE_KEEP_MS = 200          # natural pause kept at each end of a cleaned reference


def _frame_levels(wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """Per-20ms RMS level in dBFS, and the frame length in samples."""
    window = max(1, int(FRAME_SECONDS * sr))
    rms = _rms_windows(np.asarray(wav, dtype=np.float32).reshape(-1), sr, window)
    return 20.0 * np.log10(np.maximum(rms, 1e-6)), window


def _activity(db: np.ndarray) -> Tuple[Optional[np.ndarray], float, float]:
    """(active-frame mask, noise floor dB, speech level dB). Mask is None without pauses."""
    if db.size == 0:
        return None, -120.0, -120.0
    # 5th percentile, not 10th: fluent read speech can have under a tenth of its frames in
    # pauses, and then the "floor" lands on speech and no pause is ever found.
    floor = float(np.percentile(db, 5))
    level = float(np.percentile(db, 95))
    if level - floor < MIN_DYNAMIC_DB:
        return None, floor, level
    threshold = max(floor + 0.35 * (level - floor), 20.0 * np.log10(SILENCE_RMS))
    return db >= threshold, floor, level


def find_pause(wav: np.ndarray, sr: int, at: float,
               radius: float = PAUSE_SEARCH_SECONDS) -> Optional[float]:
    """The pause nearest to `at` seconds, within `radius` — or None if there is none."""
    db, window = _frame_levels(wav, sr)
    active, _, _ = _activity(db)
    if active is None:
        return None
    frame_s = window / float(sr)
    centre = int(at / frame_s)
    span = int(np.ceil(radius / frame_s))
    lo, hi = max(0, centre - span), min(active.size, centre + span + 1)
    quiet = [i for i in range(lo, hi) if not active[i]]
    if not quiet:
        return None
    best = min(quiet, key=lambda i: abs(i - centre))
    return round(min(len(wav) / float(sr), (best + 0.5) * frame_s), 3)


def snap_to_pauses(wav: np.ndarray, sr: int, start: float, end: float,
                   radius: float = PAUSE_SEARCH_SECONDS) -> Tuple[float, float]:
    """Move each edge of a selection into the nearest pause, if one is close enough."""
    snapped_start = find_pause(wav, sr, start, radius)
    snapped_end = find_pause(wav, sr, end, radius)
    new_start = start if snapped_start is None else snapped_start
    new_end = end if snapped_end is None else snapped_end
    if new_end - new_start < REF_MIN_SECONDS:      # snapping must never break the floor
        return round(start, 3), round(end, 3)
    return round(new_start, 3), round(new_end, 3)


def auto_pick_window(wav: np.ndarray, sr: int, target: float = REF_TARGET_SECONDS,
                     step: float = 0.25) -> Tuple[float, float]:
    """Pick the `target`-second stretch with the most speech, then land its edges in pauses.

    It used to pick the *loudest* window, which favours shouting, music beds and clipped
    peaks. Speech coverage is the better proxy for "a clean stretch of one person talking";
    level only breaks ties, and windows with clipped samples are pushed down.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    duration = len(wav) / float(sr)
    if duration <= target:
        return 0.0, round(duration, 3)
    db, window = _frame_levels(wav, sr)
    active, _, _ = _activity(db)
    if active is None:                 # no pauses at all: fall back to level alone
        active = db >= float(np.percentile(db, 50))
    frame_s = window / float(sr)
    per = max(1, int(round(target / frame_s)))
    clipped = np.abs(wav) >= 0.999
    clip_cum = np.concatenate([[0], np.cumsum(clipped)])
    act_cum = np.concatenate([[0], np.cumsum(active.astype(np.float64))])
    db_cum = np.concatenate([[0.0], np.cumsum(db.astype(np.float64))])
    best_score, best_start = -np.inf, 0.0
    position = 0.0
    while position + target <= duration + 1e-6:
        f0 = int(position / frame_s)
        f1 = min(active.size, f0 + per)
        frames = max(1, f1 - f0)
        coverage = (act_cum[f1] - act_cum[f0]) / frames
        level = (db_cum[f1] - db_cum[f0]) / frames
        i0, i1 = int(position * sr), min(len(wav), int((position + target) * sr))
        clip_share = (clip_cum[i1] - clip_cum[i0]) / max(1, i1 - i0)
        score = coverage - 50.0 * clip_share + 0.001 * level
        if score > best_score:
            best_score, best_start = score, position
        position += step
    start, end = round(float(best_start), 3), round(float(min(duration, best_start + target)), 3)
    return snap_to_pauses(wav, sr, start, end)


def reference_report(clip: np.ndarray, sr: int, text: str = "",
                     x_vector_only: bool = False, start: Optional[float] = None) -> str:
    """Markdown checklist for the selected reference: length, noise, speech, edges, transcript.

    Every line is something measured to matter, in the order it costs you: a transcript that
    does not match (WER 1.0, 3x the length), background noise (the largest similarity loss),
    then length and clean edges. Advisory only — nothing here blocks a generation.
    """
    clip = np.asarray(clip, dtype=np.float32).reshape(-1)
    seconds = len(clip) / float(sr) if sr else 0.0
    if not clip.size:
        return "No reference yet — upload a file, record one, or load a saved voice."
    lines: List[str] = []
    guidance = clip_guidance(seconds)
    good_length = REF_MIN_SECONDS <= seconds <= REF_MAX_SECONDS
    lines.append(("✅ " if good_length else "⚠️ ") + guidance)

    db, window = _frame_levels(clip, sr)
    active, floor, level = _activity(db)
    if active is None:
        lines.append("⚠️ No pauses found — this may be music, a tone, or wall-to-wall noise. "
                     "One person talking in a quiet room clones best.")
    else:
        snr = level - floor
        if snr < NOISY_SNR_DB:
            lines.append(f"⚠️ Noisy background (speech is ~{snr:.0f} dB above it). Noise was the "
                         f"biggest similarity loss measured — find a quieter stretch if you can.")
        else:
            lines.append(f"✅ Clean background (speech ~{snr:.0f} dB above the noise floor).")
        coverage = float(np.mean(active))
        if coverage < 0.45:
            lines.append(f"⚠️ Speech fills only {coverage:.0%} of the selection — tighten it "
                         f"around the talking.")
        edge = max(1, int(0.06 / FRAME_SECONDS))
        offset = float(start or 0.0)
        cut = []
        if active[:edge].all():
            cut.append(f"starts mid-word (at {offset:.2f}s)")
        if active[-edge:].all():
            cut.append(f"ends mid-word (at {offset + seconds:.2f}s)")
        if cut:
            lines.append("⚠️ The selection " + " and ".join(cut) + ". Move the edge into a pause "
                         "— **Auto-pick** lands edges in pauses for you.")
        else:
            lines.append("✅ Starts and ends in a pause.")
    clipped = float(np.mean(np.abs(clip) >= 0.999))
    if clipped > CLIP_FRACTION:
        lines.append(f"⚠️ Clipped peaks ({clipped:.2%} of samples at full scale) — the clone "
                     f"inherits the distortion.")

    if x_vector_only:
        lines.append("ℹ️ x-vector only: the transcript is not used (lower similarity).")
    else:
        fit = transcript_fit(seconds, text or "")
        lines.append(("⚠️ " + fit) if fit else "✅ Transcript length fits the clip — still read "
                     "it against what you hear.")
        check_line, check_bad = reference_transcript_line(cached_reference_check(clip, sr, text))
        if check_line:
            lines.append(("⚠️ " if check_bad else "✅ ") + check_line)
    return "\n\n".join(lines)


def prepare_reference(wav: np.ndarray, sr: int) -> np.ndarray:
    """Tidy a reference before it is encoded: DC, rumble, and dead air at the edges.

    Deliberately conservative — nothing here changes the voice. Level is left alone
    (measured: a reference peaking at 0.1 cloned as well as a full-scale one), and there is
    no denoiser, because a denoiser's artefacts are exactly the kind of thing the codec
    faithfully reproduces. What it does remove:
      * DC offset and sub-60Hz rumble (handling noise, HVAC, mains hum's fundamental):
        energy the speaker encoder and codec spend bits on that is not the voice;
      * more than ~0.2s of silence at either end, so the clip the model continues from ends
        in a short natural pause rather than in a long stretch of room tone.
    Falls back to the input on any failure: cleanup must not be able to break a clone.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size < int(0.25 * sr):
        return wav
    try:
        out = wav - float(np.mean(wav))
        try:
            from scipy.signal import butter, sosfiltfilt
            sos = butter(2, REF_HIGHPASS_HZ / (sr / 2.0), btype="high", output="sos")
            out = np.asarray(sosfiltfilt(sos, out.astype(np.float64)), dtype=np.float32)
        except ImportError:
            pass
        out = trim_edges(out, sr, keep_ms=REF_EDGE_KEEP_MS)
        if out.size < int(REF_MIN_SECONDS * sr * 0.75):    # trimming ate the clip: undo it
            return wav
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.0:
            out = out / peak
        return out.astype(np.float32)
    except Exception:  # noqa: BLE001 - polish must never sink a clone
        return wav


def render_waveform(wav: np.ndarray, sr: int, start: Optional[float] = None,
                    end: Optional[float] = None, width: int = 980, height: int = 180) -> Image.Image:
    """Draw a waveform with a second grid and the selected window shaded.

    Hand-drawn with PIL rather than a plotting library so the UI needs no extra
    dependency and the pixel geometry stays testable.
    """
    bg, fg, sel, grid, accent = "#101418", "#5eead4", "#1f6f5c", "#2b323c", "#f0abfc"
    pad_l, pad_r, pad_t, pad_b = 10, 10, 12, 26
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    mid = pad_t + plot_h / 2.0

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    duration = max(len(wav) / float(sr), 1e-6)
    x_of = lambda t: pad_l + (t / duration) * plot_w

    if start is not None and end is not None and end > start:
        draw.rectangle([x_of(start), pad_t, x_of(end), pad_t + plot_h], fill=sel)

    # per-pixel peak envelope
    pixels = max(1, min(plot_w, len(wav)))
    if len(wav):
        per = max(1, len(wav) // pixels)
        usable = (len(wav) // per) * per
        envelope = np.abs(wav[:usable].reshape(-1, per)).max(axis=1)[:pixels]
    else:
        envelope = np.zeros(1)
    scale = (plot_h / 2.0) / max(float(envelope.max()), 1e-6)
    for index, peak in enumerate(envelope):
        x = pad_l + index
        half = max(1.0, float(peak) * scale)
        draw.line([(x, mid - half), (x, mid + half)], fill=fg)

    draw.line([(pad_l, mid), (pad_l + plot_w, mid)], fill=grid)
    tick = 1.0 if duration <= 30 else (5.0 if duration <= 150 else 30.0)
    position = 0.0
    while position <= duration + 1e-6:
        x = x_of(position)
        draw.line([(x, pad_t), (x, pad_t + plot_h)], fill=grid)
        draw.text((x + 2, height - pad_b + 6), f"{position:.0f}s", fill="#9ca3af")
        position += tick

    for value, colour in ((start, accent), (end, accent)):
        if value is not None:
            x = x_of(min(max(float(value), 0.0), duration))
            draw.line([(x, pad_t - 4), (x, pad_t + plot_h + 4)], fill=colour, width=2)
            draw.text((min(x + 3, width - 60), pad_t - 10), f"{value:.1f}s", fill=colour)
    return image


# ============================================================================
# Transcription of the reference clip (the ICL clone needs a matching transcript)
# ============================================================================
def _mean_token_logprob(logits: torch.Tensor, token_ids: List[int],
                        offset: int = 0) -> Optional[float]:
    """Average log-probability the decoder assigned to the tokens it was asked to produce.

    `offset` is the length of the decoder prefix that was fed in but is not being scored
    (Whisper's start/language/task/timestamp tokens): logits at position t predict token
    t+1 of the *fed* sequence, so the first scoreable token sits at `offset`, not 0.
    Getting that wrong is silent and catastrophic — measured, the same clean clip reads
    -13.5 (worse than uniform) with the offset missing and near -0.2 with it.

    Kept separate from the forward pass so the arithmetic is testable without weights:
    a confident decoder gives ~0, a uniform one gives log(1/vocab) ≈ -10.8 for Whisper's
    51,865-token vocabulary.
    """
    if logits is None or not token_ids:
        return None
    try:
        step_logits = logits[0]
        if step_logits.shape[0] < offset + len(token_ids):
            return None
        index = torch.tensor(list(token_ids), device=step_logits.device).unsqueeze(1)
        window = step_logits[offset:offset + len(token_ids)].float()
        log_probs = torch.log_softmax(window, dim=-1)
        return round(float(log_probs.gather(1, index).mean()), 4)
    except Exception:  # noqa: BLE001 - a scoring signal must never break a transcription
        return None


class Transcriber:
    """Lazy Whisper wrapper, loaded only when the user actually asks for a transcript.

    Kept on the GPU once loaded (~3.1GB bf16 for large-v3) — that fits next to one
    resident 1.7B TTS model, and re-loading costs far more than keeping it.
    """

    def __init__(self, model_id: str = ASR_MODEL_ID, dtype: torch.dtype = torch.bfloat16,
                 enabled: bool = True, device: Optional[torch.device] = None):
        self.model_id = model_id
        self.dtype = dtype
        self.enabled = enabled and bool(model_id) and model_id.lower() not in ("none", "off", "")
        self.device = device or DEVICE
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        self.lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.enabled:
            raise MediaError("Transcription is disabled (start with --asr-model to enable it).")
        from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration

        print(f"[asr] loading {self.model_id} -> CPU RAM ...", flush=True)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, dtype=self.dtype, attn_implementation="sdpa")
        self.model.to(self.device)
        self.model.eval()
        print(f"[asr] {self.model_id} ready on {self.device}", flush=True)

    def unload(self) -> None:
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _text_ids(self, text: str) -> List[int]:
        """Token ids for `text` as Whisper would emit them after its prefix: no specials.

        `WhisperTokenizer.__call__` adds `<|startoftranscript|><|notimestamps|>` in front
        and `<|endoftext|>` behind by default. The scorer used to take that output and put
        the generation prefix in front of it *again*, so every take was charged for two
        impossible tokens (start-of-transcript after no-timestamps) and its real words were
        scored in a corrupted context — which is why a clean take read about -1.9 instead
        of near zero, and why short lines read worse than long ones. Leading space because
        Whisper's transcripts begin with one (" Hello", not "Hello").
        """
        body = self.tokenizer(" " + (text or "").strip(), add_special_tokens=False).input_ids
        return [int(t) for t in body] + [int(self.tokenizer.eos_token_id)]

    def _prefix_ids(self, language_code: Optional[str]) -> List[int]:
        """The forced decoder prefix: start, language (if known), transcribe, no-timestamps."""
        tokens = ["<|startoftranscript|>"]
        if language_code:
            tokens.append(f"<|{language_code}|>")
        tokens += ["<|transcribe|>", "<|notimestamps|>"]
        return [int(t) for t in self.tokenizer.convert_tokens_to_ids(tokens)]

    def _detected_code(self, ids: Any) -> Optional[str]:
        """The language token Whisper put in its own prefix, e.g. 'en' — or None."""
        try:
            for token in ids[0][:4].tolist():
                name = self.tokenizer.convert_ids_to_tokens(int(token)) or ""
                match = re.fullmatch(r"<\|([a-z]{2,3})\|>", name)
                if match:
                    return match.group(1)
        except Exception:  # noqa: BLE001 - detection is a convenience
            pass
        return None

    def _confidence(self, features: Any, padding_mask: np.ndarray, text: str,
                    language_code: Optional[str] = None) -> Optional[float]:
        """Mean log-probability Whisper's decoder gives `text`, given the audio.

        A teacher-forced forward pass rather than a read of the beam-search bookkeeping:
        `output_scores` is dropped somewhere inside Whisper's own generation-config juggling
        (it is only forced on for token timestamps and `logprob_threshold`). One forward
        pass is cheap, exact, and does not depend on which decoding strategy ran.

        The prefix is built explicitly (`_prefix_ids`) and the text is tokenised without
        specials (`_text_ids`), so the scored tokens are exactly the words plus end-of-text.
        A per-token mean, so it is comparable across line lengths: near zero for clean
        audio, well below -1 when the decoder cannot find the text in the audio.
        """
        try:
            if not (text or "").strip():
                return None
            target = self._text_ids(text)
            prefix = self._prefix_ids(language_code)
            fed = torch.tensor([prefix + target[:-1]], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                out = self.model(
                    input_features=features,
                    attention_mask=torch.from_numpy(padding_mask).unsqueeze(0).to(self.device),
                    decoder_input_ids=fed)
            # logits[t] predicts fed[t+1], and fed = prefix + text[:-1]: the first scoreable
            # token therefore lines up with logits[prefix_len - 1], not logits[prefix_len].
            return _mean_token_logprob(getattr(out, "logits", None), target,
                                       offset=max(0, len(prefix) - 1))
        except Exception:  # noqa: BLE001 - confidence is a bonus, never a failure
            return None

    def _generate_ids(self, features: Any, padding_mask: np.ndarray,
                      language_code: Optional[str]) -> Any:
        with torch.inference_mode():
            out = self.model.generate(
                features,
                attention_mask=torch.from_numpy(padding_mask).unsqueeze(0).to(self.device),
                language=language_code, task="transcribe",
                num_beams=4, no_repeat_ngram_size=4, return_dict_in_generate=True)
        return getattr(out, "sequences", out)

    def transcribe(self, wav: np.ndarray, sr: int, language: Optional[str] = None) -> str:
        return self.transcribe_with_confidence(wav, sr, language)[0]

    def assess(self, wav: np.ndarray, sr: int, intended: str,
               language: Optional[str] = None) -> Tuple[str, Optional[float]]:
        """(what Whisper heard, how well the audio matches the *intended* text).

        One beam search plus one teacher-forced pass per take. It used to be two beam
        searches — `transcribe_with_confidence` and then `score_transcript`, which ran
        `generate` again only to recover the prefix.

        The confidence scores the text we *asked for*, never Whisper's own transcript:
        measured on a noise-degraded take, scoring its own answer missed the damage while
        scoring the intended text caught it. Whisper is confidently consistent about
        whatever it decided to hear.
        """
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if not wav.size:
            return "", None
        code = ASR_LANGUAGE_CODES.get((language or "").strip().lower())
        with self.lock:
            self.load()
            features, padding_mask = self._features(wav, sr)
            ids = self._generate_ids(features, padding_mask, code)
            heard = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            confidence = self._confidence(features, padding_mask, intended,
                                          code or self._detected_code(ids))
        return heard, confidence

    def score_transcript(self, wav: np.ndarray, sr: int, text: str,
                         language: Optional[str] = None) -> Optional[float]:
        """How well Whisper thinks this audio matches `text` (see `assess`)."""
        if not np.asarray(wav).size or not (text or "").strip():
            return None
        return self.assess(wav, sr, text, language)[1]

    def _features(self, wav: np.ndarray, sr: int) -> Tuple[Any, np.ndarray]:
        """Mel features plus the padding mask, resampled to Whisper's 16kHz."""
        import librosa
        if int(sr) != 16000:
            wav = librosa.resample(y=np.asarray(wav, dtype=np.float32), orig_sr=int(sr), target_sr=16000)
            sr = 16000
        features = self.feature_extractor(
            np.asarray(wav, dtype=np.float32), sampling_rate=int(sr), return_tensors="pt"
        ).input_features.to(self.device, self.dtype)
        # Whisper pads every clip to 30s; without an explicit padding mask it cannot
        # tell silence from audio (and warns, since pad == eos for this checkpoint).
        max_samples = self.feature_extractor.nb_max_frames * 160  # 30s at 16kHz
        padding_mask = np.zeros(max_samples, dtype=np.int64)
        padding_mask[:min(len(wav), max_samples)] = 1
        return features, padding_mask

    def transcribe_detailed(self, wav: np.ndarray, sr: int,
                            language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe; returns text, confidence and the language Whisper used (UI name).

        The language matters for cloning: setting it explicitly instead of Auto is upstream's
        own advice, and Whisper has already worked it out by the time the transcript exists.
        """
        if not np.asarray(wav).size:
            raise MediaError("Nothing to transcribe — pick a non-empty clip.")
        code = ASR_LANGUAGE_CODES.get((language or "").strip().lower())
        with self.lock:
            self.load()
            features, padding_mask = self._features(wav, sr)
            ids = self._generate_ids(features, padding_mask, code)
            text = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            used = code or self._detected_code(ids)
            confidence = self._confidence(features, padding_mask, text, used)
        if not text:
            raise MediaError("Transcription came back empty — the clip may be music or silence.")
        by_code = {c: name.title() for name, c in ASR_LANGUAGE_CODES.items()}
        return {"text": text, "confidence": confidence, "code": used,
                "language": by_code.get(used or "")}

    def transcribe_with_confidence(self, wav: np.ndarray, sr: int,
                                   language: Optional[str] = None) -> Tuple[str, Optional[float]]:
        """Transcribe, and also return how confident Whisper is in its own transcript."""
        result = self.transcribe_detailed(wav, sr, language)
        return result["text"], result["confidence"]


ASR: Optional[Transcriber] = None

# Whisper's verdict on "does the reference say this transcript?". The length check
# (`transcript_fit`) cannot catch a transcript of the right length with the wrong words, and
# a mismatched transcript is the worst clone failure measured (17.4s of audio for a 5s line,
# WER 1.0). The same teacher-forced scorer that grades takes grades the reference instead.
# PROVISIONAL: clean speech reads -0.02..-0.06 and a real Japanese take -0.51 on this scale;
# what a wrong transcript reads has not been measured. It warns, it never blocks, until it is.
REF_TEXT_CONF_WARN = -2.00
_REF_TEXT_CHECKS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
REF_TEXT_CHECK_CACHE = 32


def check_reference_transcript(wav: np.ndarray, sr: int, text: str,
                               may_load: bool = False) -> Optional[Dict[str, Any]]:
    """{confidence, heard, wer} for a reference and its transcript, or None if not run.

    Runs only when Whisper is already loaded, or when `may_load` says this request was going
    to load it anyway (best-of-N scoring) — never a surprise 3GB load. Results are kept per
    (clip content, transcript), so retakes do not re-run it and the checklist can show it.
    """
    text = str(text or "").strip()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if not text or not wav.size or ASR is None or not ASR.enabled:
        return None
    key = _clone_cache_key((wav, int(sr)), text, False)
    if key in _REF_TEXT_CHECKS:
        _REF_TEXT_CHECKS.move_to_end(key)
        return _REF_TEXT_CHECKS[key]
    if not (ASR.loaded or may_load):
        return None
    try:
        heard, confidence = ASR.assess(wav, int(sr), text)
    except Exception:  # noqa: BLE001 - a check that fails is a check not run
        return None
    result = {"confidence": confidence, "heard": heard,
              "wer": word_error_rate(heard, text) if heard else None}
    _REF_TEXT_CHECKS[key] = result
    while len(_REF_TEXT_CHECKS) > REF_TEXT_CHECK_CACHE:
        _REF_TEXT_CHECKS.popitem(last=False)
    return result


def cached_reference_check(wav: np.ndarray, sr: int, text: str) -> Optional[Dict[str, Any]]:
    """The stored verdict for this exact clip + transcript, without running anything."""
    text = str(text or "").strip()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if not text or not wav.size:
        return None
    return _REF_TEXT_CHECKS.get(_clone_cache_key((wav, int(sr)), text, False))


def reference_transcript_line(result: Optional[Dict[str, Any]]) -> Tuple[str, bool]:
    """(one line for a status or checklist, whether it is a warning)."""
    if not result or result.get("confidence") is None:
        return "", False
    confidence = float(result["confidence"])
    if confidence >= REF_TEXT_CONF_WARN:
        return (f"Whisper hears this transcript in the reference (confidence {confidence:+.2f}).",
                False)
    heard = _short(result.get("heard") or "", 90)
    return (f"Whisper does not hear this transcript in the reference (confidence "
            f"{confidence:+.2f}; a matching one reads near 0). It heard: “{heard}”. A transcript "
            f"that does not match its clip is the worst clone failure measured — fix it, or "
            f"press Transcribe.", True)

# (Uploaded clips are kept on disk under MEDIA_DIR — see the path-containment block
# above. gr.State travels to the browser as JSON, so only the clip's path/duration live
# in state; a numpy array there would not survive the round-trip.)


# ============================================================================
# Gradio callbacks
# ============================================================================
def generate_voice_design(text, language, voice_description, temperature, top_p, top_k, rep_penalty,
                          postproc=POSTPROCESS_DEFAULT, progress=gr.Progress(track_tqdm=True)):
    """Speak a description. Returns (audio, status, take-state).

    The third output is the **raw model decode** as a path under `tmp/media/`, for the Save
    button. The audio box carries the post-processed listening copy (48kHz, de-imaged, EQ'd),
    and a saved voice is usually a future clone reference — which should come straight from
    the model, not through an EQ curve. Paths through `gr.State`, never a module global: the
    "last generation" global there is what once let one LAN user's Save persist another's clip.
    """
    if not text or not text.strip():
        return None, "Error: Text is required.", None
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required.", None
    try:
        budget = _budget(text)
        with HUB.lock:
            tts = HUB.acquire("voice_design")

            def one(chunk: str) -> Tuple[np.ndarray, int]:
                wavs, rate = tts.generate_voice_design(
                    text=chunk,
                    language=language,
                    instruct=voice_description.strip(),
                    non_streaming_mode=True,
                    max_new_tokens=_budget(chunk),
                    **_sampling(temperature, top_p, top_k, rep_penalty),
                )
                return wavs[0], rate

            wav, sr, chunk_notes = generate_chunked(text, language, one, progress)
        try:
            raw_path = _write_take(wav, sr)
        except (OSError, ValueError) as e:  # the audio still plays; only Save is lost
            raw_path = None
            chunk_notes.append(f"⚠️ could not keep the raw clip ({type(e).__name__}), so Save "
                               f"is unavailable for this take")
        audio, report = _present(wav, sr, budget, "VoiceDesign", postproc, chunk_notes)
        # Everything Save stores is captured here, at Generate. Reading the text boxes at
        # Save time stored whatever had been typed since: a transcript that does not match
        # the clip is the worst clone failure measured (docs/quality.md).
        state = None if raw_path is None else {
            "path": raw_path, "text": text.strip(), "language": str(language or "Auto"),
            "description": voice_description.strip()}
        return audio, report, state
    except Exception as e:  # noqa: BLE001
        return None, f"Error: {type(e).__name__}: {e}", None


# Reusable clone prompts, keyed by a hash of the uploaded reference audio, so the
# reference encoder is not recomputed for every line (big speedup for repeated takes).
_CLONE_PROMPTS: "OrderedDict[str, Any]" = OrderedDict()
CLONE_CACHE_SIZE = 8


# ============================================================================
# Voice library: designed voices persisted as reusable clone references
# ============================================================================
def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    if not s:
        raise ValueError("Voice name must contain at least one letter or digit.")
    if s in (".", "..") or s.startswith("_"):
        raise ValueError("Invalid voice name.")
    return s


def _voice_paths(slug: str) -> Tuple[str, str]:
    base = os.path.join(VOICES_DIR, slug)
    return f"{base}.wav", f"{base}.json"


def save_voice(name: str, wav: Any = None, sr: int = 0, text: str = "", instruct: str = "",
               language: str = "Auto", description: str = "", kind: str = "") -> Dict[str, Any]:
    """Persist a named voice: a sidecar, plus the wav when there is a clip to keep.

    `wav=None` is a *described* voice — a description and nothing else, which Speak and Chat
    can still talk from. See `voice_kind` for why the description, not the clip, is the
    higher-fidelity form.
    """
    os.makedirs(VOICES_DIR, exist_ok=True)
    slug = _slug(name)
    wav_path, meta_path = _voice_paths(slug)
    record = {
        "name": name.strip(),
        "slug": slug,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": language,
        "instruct": instruct,
        "description": description,
        "kind": kind or ("design" if (description or instruct).strip() else "clip"),
        "text": text,
        "sample_rate": int(sr),
        "seconds": round(len(wav) / float(sr), 2) if wav is not None and sr else None,
        "prefer": "auto",
    }
    # Write-then-rename, so a crash mid-save cannot leave a half-written sidecar that
    # `list_voices` then silently skips.
    if wav is not None:
        tmp_wav = f"{wav_path}.tmp.wav"
        sf.write(tmp_wav, wav, sr)
        os.replace(tmp_wav, wav_path)
    elif os.path.exists(wav_path):
        # Re-saving a name as description-only must not leave the old clip behind, paired
        # with a description it never came from.
        os.remove(wav_path)
    tmp_meta = f"{meta_path}.tmp"
    with open(tmp_meta, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_meta, meta_path)
    return record


def voice_exists(name: str) -> bool:
    """True when saving under `name` would replace a voice already in the library."""
    try:
        return os.path.exists(_voice_paths(_slug(name))[1])
    except ValueError:
        return False


def _taken_message(name: str) -> str:
    return (f"Error: a voice called ‘{name.strip()}’ is already in the library. Pick another "
            f"name, or delete the old one on the Voices tab first — saving never overwrites.")


VOICE_ROUTES = ("auto", "clip", "description")


def voice_route(record: Dict[str, Any]) -> str:
    """How Speak and Chat should say this voice: 'design' or 'clip'.

    A record may carry both a clip and a description. `voice_kind` prefers the description
    (the brighter sound); the record's `prefer` field, set on the Voices tab, can override
    that — a frozen clip is the steadier identity across the sentences of a long reply,
    because a description is re-rolled for every chunk.
    """
    prefer = str(record.get("prefer") or "auto").strip().lower()
    has_clip = os.path.exists(record.get("wav") or _voice_paths(record["slug"])[0])
    has_description = bool(str(record.get("description") or record.get("instruct") or "").strip())
    if prefer == "clip" and has_clip:
        return "clip"
    if prefer == "description" and has_description:
        return "design"
    return voice_kind(record)


def set_voice_preference(slug: str, prefer: str) -> Dict[str, Any]:
    """Persist which route Speak/Chat use for one voice (see `voice_route`)."""
    prefer = (prefer or "auto").strip().lower()
    if prefer not in VOICE_ROUTES:
        raise ValueError(f"unknown route {prefer!r}")
    meta_path = _voice_paths(_slug(slug))[1]
    with open(meta_path, encoding="utf-8") as fh:
        record = json.load(fh)
    record["prefer"] = prefer
    tmp_meta = f"{meta_path}.tmp"
    with open(tmp_meta, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_meta, meta_path)
    return record


def voice_kind(record: Dict[str, Any]) -> str:
    """'design' when the voice carries a description it can be spoken from, else 'clip'.

    Two kinds of voice, one format. A **clip** voice is a reference recording + transcript and
    is spoken by cloning it; a **design** voice carries the description that made it and is
    spoken straight by Voice Design. Measured on the same three lines: speaking a description
    keeps 2.67m of energy above 9kHz, cloning a saved take keeps 0.57m — cloning attenuates
    whatever it is given to roughly 60% of the reference's top end, so when a description
    exists it is the better route, and the clip stays there for the Clone tab and for
    freezing into a fixed identity if you prefer one.
    """
    kind = str(record.get("kind") or "").strip().lower()
    if kind in ("design", "clip"):
        return kind
    return "design" if (record.get("description") or record.get("instruct") or "").strip() \
        else "clip"


def list_voices() -> List[Dict[str, Any]]:
    """All saved voices, newest first."""
    if not os.path.isdir(VOICES_DIR):
        return []
    records = []
    for fname in os.listdir(VOICES_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(VOICES_DIR, fname), encoding="utf-8") as fh:
                record = json.load(fh)
            record["wav"], record["json"] = _voice_paths(record.get("slug") or fname[:-5])
            records.append(record)
        except (OSError, ValueError, KeyError):
            continue  # a half-written voice should not break the whole list
    return sorted(records, key=lambda r: r.get("created", ""), reverse=True)


def voice_choices() -> List[Tuple[str, str]]:
    entries = []
    for record in list_voices():
        if voice_route(record) == "design":
            entries.append((f"{record['name']} (described)", record["slug"]))
        else:
            entries.append((f"{record['name']} ({record.get('seconds', '?')}s)", record["slug"]))
    return entries


def _find_voice(voice: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Locate a saved voice *record* by display name or slug — never by a client path."""
    if not (voice or "").strip():
        return None, "Error: choose a saved voice first."
    wanted = voice.strip()
    slug = _slug(wanted)
    records = list_voices()
    for record in records:
        if record.get("slug") == slug:
            return record, None
    needle = wanted.lower()
    for record in records:
        if record.get("name", "").lower() == needle:
            return record, None
    known = ", ".join(r["slug"] for r in records[:12]) or "none saved yet"
    return None, f"Error: no saved voice ‘{wanted}’. Available: {known}"


def save_designed_voice(name: str, take_state: Any) -> Tuple[Any, str, Any]:
    """Save the clip the last Generate produced as a reusable voice.

    Clip, transcript, description and language all come from `take_state`, which
    `generate_voice_design` filled in at Generate time. Not from the text boxes: they may
    have been edited since, and a transcript that does not match its clip is the worst clone
    failure measured (17s of audio for a 5s line at WER 1.0). Not from a module-level "last
    generation" cache either: with two people on the LAN, that let one person's Save persist
    the other person's clip. Same rule as the Clone tab's `save_generated_voice`.

    The clip comes from the **raw decode** written at generation time, not from the audio
    box. The box holds the post-processed listening copy — 48kHz, de-imaged, high-shelf and
    soft-limited — and a saved voice is usually a future clone reference, so feeding that
    copy back in means cloning an EQ curve. It also means the reference has been through the
    codec twice by the time you clone it, and cloning keeps only ~60% of whatever it is
    handed (`docs/quality.md`).
    """
    if not (name or "").strip():
        return gr.update(), "Error: give the voice a name first.", gr.update()
    if voice_exists(name):
        return gr.update(), _taken_message(name), gr.update()
    if not isinstance(take_state, dict) or not take_state.get("path"):
        return gr.update(), "Nothing to save yet — press Generate first.", gr.update()
    path = take_state["path"]
    text = str(take_state.get("text") or "").strip()
    description = str(take_state.get("description") or "").strip()
    language = str(take_state.get("language") or "Auto")
    if not text:
        return gr.update(), "Error: the text the clip speaks is empty — it is what ICL cloning needs.", gr.update()
    try:
        path = resolve_input_path(path, "generated clip")
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        record = save_voice(
            name=name,
            wav=wav,
            sr=sr,
            text=text,
            instruct=description,
            language=language,
            description=description,
            kind="design",
        )
    except (MediaError, OSError, ValueError) as e:
        return gr.update(), f"Error: {type(e).__name__}: {e}", gr.update()
    return (
        gr.update(),
        f"Saved ‘{record['name']}’ ({record['seconds']}s @ {record['sample_rate']}Hz) with its "
        f"description — the model's own decode, not the post-processed copy. It says "
        f"“{_short(text)}”, as generated (later edits to the boxes are not saved). Speak "
        f"and Chat will speak it from the description (brighter); Voice Clone can load the clip.",
        gr.update(choices=voice_choices(), value=record["slug"]),
    )


def save_reference_voice(name: str, audio: Any, ref_text: str, language: str) -> Tuple[Any, str, Any]:
    """Save the clip in the Voice Clone reference box as a reusable voice.

    Voice Design could save a voice; the Clone tab could only load and delete. So a voice
    you assembled from a video or an upload — trimmed waveform, Whisper transcript and all
    — had nowhere to go, and the next session meant uploading it again. Same on-disk
    format as a designed voice, so both show up in one list.
    """
    if not (name or "").strip():
        return gr.update(), "Error: give the voice a name first.", gr.update()
    if voice_exists(name):
        return gr.update(), _taken_message(name), gr.update()
    try:
        parsed = _audio_to_tuple(audio)
        if parsed is None:
            return (gr.update(),
                    "Nothing to save — upload, record or load a reference in step 1 first.",
                    gr.update())
        wav, sr = parsed
        seconds = len(wav) / float(sr)
        if seconds < REF_MIN_SECONDS:
            return (gr.update(),
                    f"Error: the reference clip is {seconds:.1f}s — too short to clone from "
                    f"(need {REF_MIN_SECONDS}s). Widen the selection on the waveform.",
                    gr.update())
        record = save_voice(name=name, wav=wav, sr=sr, text=(ref_text or "").strip(),
                            instruct="", language=language)
    except (MediaError, OSError, ValueError) as e:
        return gr.update(), f"Error: {type(e).__name__}: {e}", gr.update()
    note = "" if record["text"] else " (no transcript — you will need ‘x-vector only’ to clone from it)"
    return (
        gr.update(),
        f"Saved ‘{record['name']}’ ({record['seconds']}s) from the reference clip{note}. "
        f"It is on the Voices tab, and in Speak and Chat, now.",
        gr.update(choices=voice_choices(), value=record["slug"]),
    )


def build_take_state(results: List[Dict[str, Any]], best_index: int,
                     paths: List[Optional[str]], text: str, language: Any) -> Dict[str, Any]:
    """JSON-safe record of what was just generated, for the 'keep this take' button.

    Paths, never arrays: `gr.State` round-trips through the browser, so anything here has
    to survive JSON. The labels are kept alongside so the picker can show the same numbers
    the players do.
    """
    takes = [{"index": item["index"], "path": path,
              "label": take_label(item["index"], item.get("wer"), item.get("sim"),
                                  item["index"] == best_index, item.get("confidence"),
                                  item.get("defects")),
              "kept": item["index"] == best_index}
             for item, path in zip(results, paths)]
    return {"takes": takes, "text": (text or "").strip(), "language": str(language or "Auto")}


def _write_take(wav: np.ndarray, sr: int) -> str:
    """Persist one candidate so a later Save click can name it.

    The path travels through `gr.State`, never a module-level 'last generation' global:
    that global is what once let one LAN user's Save button persist another's clip. Raw
    model decode, not the post-processed listening copy — a saved voice is usually a future
    clone reference, and a reference is better coming straight from the model than through
    an EQ curve. Pruned by MEDIA_TTL_SECONDS like every other clip under tmp/media/.
    """
    prune_media()
    path = os.path.join(media_dir(), f"{uuid.uuid4().hex}.wav")
    sf.write(path, wav, sr)
    return path


def save_generated_voice(state: Any, selection: Any, name: str) -> Tuple[Any, str, Any]:
    """Save one of the candidate takes from the last generation into the voice library.

    The library used to hold only *references* — the clip a voice was taken from — so the
    take you actually preferred could only leave the browser as a download, and turning it
    into a reusable voice meant uploading it back and transcribing it. The selection comes
    in as an input (which take), and the audio comes from the state this request produced.
    """
    if not (name or "").strip():
        return gr.update(), "Error: give the voice a name first.", gr.update()
    if not isinstance(state, dict) or not state.get("takes"):
        return gr.update(), "Nothing to save yet — press ‘Clone & generate’ first.", gr.update()
    if voice_exists(name):
        return gr.update(), _taken_message(name), gr.update()
    takes = state["takes"]
    wanted = str(selection).strip() if selection is not None else ""
    chosen = next((t for t in takes if str(t.get("index")) == wanted), None)
    if chosen is None and len(takes) == 1:
        chosen = takes[0]          # one candidate: nothing to choose, so take it
    if chosen is None:
        return (gr.update(),
                "Error: pick which take to keep under ‘What to save’.",
                gr.update())
    try:
        path = resolve_input_path(chosen.get("path"))
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        record = save_voice(name=name, wav=wav, sr=sr, text=str(state.get("text") or "").strip(),
                            instruct="", language=str(state.get("language") or "Auto"))
    except (MediaError, OSError, ValueError) as e:
        return gr.update(), f"Error: {type(e).__name__}: {e}", gr.update()
    return (
        gr.update(),
        f"Saved take #{chosen['index']} as ‘{record['name']}’ ({record['seconds']}s, "
        f"voices/{record['slug']}.wav) — the model's own decode, not the post-processed copy. "
        f"It is on the Voices tab, and in Speak and Chat, now.",
        gr.update(choices=voice_choices(), value=record["slug"]),
    )


def load_saved_voice(slug: str) -> Tuple[Any, Any, str]:
    """Fill the Voice Clone reference fields from a saved voice."""
    if not slug:
        return None, gr.update(), "Error: choose a saved voice first."
    record, error = _find_voice(slug)
    if error:
        return None, gr.update(), error
    wav_path = record.get("wav") or _voice_paths(record["slug"])[0]
    meta_path = record.get("json") or _voice_paths(record["slug"])[1]
    if not os.path.exists(wav_path):
        return None, gr.update(), (
            f"Error: ‘{record['name']}’ is a described voice — it has no clip to clone. Use it "
            f"in Speak or Chat, or generate it on the Design a voice tab and save the clip.")
    wav, sr = sf.read(wav_path, dtype="float32")
    text = ""
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            text = json.load(fh).get("text", "")
    return (int(sr), wav), text, f"Loaded ‘{record['name']}’ — reference transcript filled in. Hit ‘Clone & generate’."


def delete_saved_voice(slug: str) -> Tuple[Any, str, Any]:
    if not slug:
        return gr.update(), "Error: choose a saved voice to delete."
    wav_path, meta_path = _voice_paths(_slug(slug))
    removed = []
    for path in (wav_path, meta_path):
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    renders = forget_voice_renders(_slug(slug))
    msg = f"Deleted {', '.join(removed)}" if removed else f"Nothing saved under ‘{slug}’."
    if renders:
        msg += f" and {renders} cached render{'s' if renders != 1 else ''}"
    # Deleting a library entry used to clear the Clone tab's reference box too, throwing
    # away whatever was loaded there even when it was not the voice being deleted.
    return gr.update(choices=voice_choices(), value=None), msg


def refresh_voices() -> Any:
    return gr.update(choices=voice_choices())


def voice_library_updates() -> Tuple[Any, Any, Any, Any, str, Any]:
    """Refresh **every** dropdown that can speak a saved voice, plus the machine-readable list.

    The library dropdowns used to live and die inside the tab that owned them: keep a take on
    the Clone tab and the Chat and Speak tabs never heard about it, so a save that worked
    looked like a save that did nothing. One helper, wired to every action that changes the
    library, so the tabs cannot drift apart again.
    """
    choices = voice_choices()
    return (gr.update(choices=choices), gr.update(choices=choices),
            gr.update(choices=choices), gr.update(choices=choices), voices_available(),
            gr.update(value=voice_table_rows()))


def _resolve_voice(voice: str) -> Tuple[Optional[Tuple[Any, int, str]], Optional[str]]:
    """Turn a client-supplied voice *name* into a server-side (wav, sr, transcript).

    The client never supplies a path: the name is slugged to [a-z0-9_] before it is joined
    onto VOICES_DIR, so there is nothing left to traverse. If the exact slug is missing we
    fall back to matching the display name, because the thing calling this is usually a
    script that read the list once and typed it back.
    """
    record, error = _find_voice(voice)
    if error:
        return None, error
    wav_path = record.get("wav") or _voice_paths(record["slug"])[0]
    meta_path = record.get("json") or _voice_paths(record["slug"])[1]
    if not os.path.exists(wav_path):
        return None, (f"Error: ‘{record['name']}’ is a described voice with no clip. Speak and "
                      f"Chat can talk from its description; Voice Clone needs a clip to clone.")
    try:
        ref_wav, ref_sr = sf.read(wav_path, dtype="float32")
    except (OSError, ValueError) as e:
        return None, f"Error: saved voice ‘{record['slug']}’ could not be read: {type(e).__name__}: {e}"
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)
    ref_text = str(record.get("text") or "").strip()
    return (ref_wav, int(ref_sr), ref_text), None


# ---------------------------------------------------------------------------
# Render cache: Speak and Chat replay a sentence they have already said in this voice.
# ---------------------------------------------------------------------------
RENDER_CACHE_DIR = os.path.join(ROOT, "tmp", "render_cache")
RENDER_CACHE_MB = int(os.environ.get("QWEN_TTS_RENDER_CACHE_MB", "256"))      # 0 disables it
RENDER_CACHE_TTL_SECONDS = int(os.environ.get("QWEN_TTS_RENDER_CACHE_TTL_SECONDS",
                                              str(7 * 24 * 3600)))           # since last use
RENDER_CACHE_VERSION = 1      # bump when anything that shapes a raw decode changes


def _model_tag(capability: str) -> str:
    """Which weights produced a take: the checkpoint folder plus its config's mtime, so
    swapping the weights under the same name does not replay the old model's audio."""
    try:
        path = HUB._checkpoint(capability) if HUB is not None else ""
        stamp = os.stat(os.path.join(path, "config.json")).st_mtime_ns if path else 0
        return f"{os.path.basename(path) or capability}@{stamp}"
    except Exception:  # noqa: BLE001 - a missing checkpoint fails later, at generation
        return capability


class RenderCache:
    """Raw per-sentence decodes on disk, for one voice in one configuration.

    Bound by `_voice_synth` to an *identity*: the voice's content (the cleaned clip's PCM
    hash + transcript + mode, or its description), the language, the sampling settings, the
    model's weights and `RENDER_CACHE_VERSION`. Anything that would change the audio changes
    the identity, so a hit can never be stale; it is only ever a take this exact
    configuration produced. Post-processing runs after the cache, so Warm/Clean/Off share it.

    One take per sentence, replayed verbatim: repeats sound identical. That is the point for
    notifications ("Tests passed.") and it holds a designed voice steady, but it is a change
    from re-rolling every time. Files are named `<slug>__<hash>.wav` so deleting a voice can
    delete its renders — a cache of someone's voice saying things must not outlive the voice.
    """

    def __init__(self, slug: str, identity: str):
        self.slug = slug
        self.identity = identity

    def _path(self, chunk: str) -> str:
        said = " ".join(str(chunk or "").split())
        digest = hashlib.sha256(f"{self.identity}\n{said}".encode("utf-8")).hexdigest()[:40]
        return os.path.join(RENDER_CACHE_DIR, f"{self.slug}__{digest}.wav")

    def get(self, chunk: str) -> Optional[Tuple[np.ndarray, int]]:
        if RENDER_CACHE_MB <= 0:
            return None
        path = self._path(chunk)
        if not os.path.isfile(path):
            return None
        try:
            wav, sr = sf.read(path, dtype="float32")
            os.utime(path)                  # the TTL and the size cap both count from last use
        except Exception:  # noqa: BLE001 - a damaged entry is a miss, never a failure
            return None
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return (wav, int(sr)) if wav.size else None

    def put(self, chunk: str, wav: np.ndarray, sr: int) -> None:
        if RENDER_CACHE_MB <= 0:
            return
        try:
            os.makedirs(RENDER_CACHE_DIR, exist_ok=True)
            path = self._path(chunk)
            temp = os.path.join(RENDER_CACHE_DIR, f".{uuid.uuid4().hex}.tmp.wav")
            sf.write(temp, np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0),
                     int(sr), subtype="PCM_16")
            os.replace(temp, path)
            prune_render_cache()
        except Exception:  # noqa: BLE001 - failing to cache must not fail the reply
            return


def prune_render_cache(max_age_seconds: Optional[float] = None,
                       max_mb: Optional[float] = None) -> int:
    """Drop renders unused for the TTL, then the least recently used until under the cap."""
    max_age_seconds = RENDER_CACHE_TTL_SECONDS if max_age_seconds is None else max_age_seconds
    max_mb = RENDER_CACHE_MB if max_mb is None else max_mb
    try:
        names = os.listdir(RENDER_CACHE_DIR)
    except OSError:
        return 0
    entries = []
    for name in names:
        path = os.path.join(RENDER_CACHE_DIR, name)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
    entries.sort()
    removed = 0
    now = time.time()
    total = sum(size for _, size, _ in entries)
    limit = max(0.0, float(max_mb)) * 1024 * 1024
    for mtime, size, path in entries:
        stale = max_age_seconds > 0 and mtime < now - max_age_seconds
        if not stale and total <= limit:
            continue
        try:
            os.remove(path)
            removed += 1
            total -= size
        except OSError:
            continue
    return removed


def forget_voice_renders(slug: str) -> int:
    """Delete every cached render of one voice (called when the voice is deleted)."""
    removed = 0
    prefix = f"{slug}__"
    try:
        names = os.listdir(RENDER_CACHE_DIR)
    except OSError:
        return 0
    for name in names:
        if name.startswith(prefix):
            try:
                os.remove(os.path.join(RENDER_CACHE_DIR, name))
                removed += 1
            except OSError:
                continue
    return removed


def _sampling_tag(temperature, top_p, top_k, rep_penalty) -> str:
    s = _sampling(temperature, top_p, top_k, rep_penalty)
    return (f"t{s['temperature']:.4f}|p{s['top_p']:.4f}|k{s['top_k']}|"
            f"r{s['repetition_penalty']:.4f}")


def _voice_synth(voice: str, language: str, temperature: float, top_p: float,
                 top_k: int, rep_penalty: float) -> Tuple[Any, str]:
    """Resolve a saved voice into a `generate_one(chunk) -> (wav, sr)` callable.

    The hub lock is taken *per chunk*, not around a whole reply. The chat handler is a
    generator, and a client that walks away mid-reply leaves the generator closed — a lock
    held across yields could then be released late or not at all, and a stuck lock stops the
    whole app. Per-chunk acquisition costs an LRU check and cannot deadlock the box.

    The callable carries a `render_cache` attribute (a `RenderCache` bound to this voice,
    language and sampling) that `iter_chunked` consults before calling it.
    """
    resolved, error = _find_voice(voice)
    if error:
        raise MediaError(error)
    record = resolved
    description = str(record.get("description") or record.get("instruct") or "").strip()
    settings = (f"v{RENDER_CACHE_VERSION}|{language or 'Auto'}|"
                f"{_sampling_tag(temperature, top_p, top_k, rep_penalty)}")
    if voice_route(record) == "design" and description:
        # Speak the description straight: measured 2.67m of energy above 9kHz against 0.57m
        # for cloning a saved take, because cloning keeps only ~60% of whatever reference it
        # is handed. The clip in the sidecar is still what Voice Clone loads.
        def one(chunk: str) -> Tuple[np.ndarray, int]:
            with HUB.lock:
                tts = HUB.acquire("voice_design")
                wavs, rate = tts.generate_voice_design(
                    text=chunk,
                    language=language,
                    instruct=description,
                    max_new_tokens=_budget(chunk),
                    **_sampling(temperature, top_p, top_k, rep_penalty),
                )
            return wavs[0], int(rate)
        one.render_cache = RenderCache(  # type: ignore[attr-defined]
            record["slug"], f"design|{description}|{settings}|{_model_tag('voice_design')}")
        return one, "described"

    resolved, error = _resolve_voice(voice)
    if error:
        raise MediaError(error)
    ref_wav, ref_sr, ref_text = resolved
    ref_wav = prepare_reference(ref_wav, ref_sr)     # same cleanup the Clone tab applies
    use_xvector = not ref_text
    cache: Dict[str, Any] = {}

    def one(chunk: str) -> Tuple[np.ndarray, int]:
        with HUB.lock:
            tts = HUB.acquire("voice_clone")
            if "prompt" not in cache:
                cache["prompt"] = _clone_prompt(tts, ref_wav, ref_sr, ref_text, use_xvector)
            wavs, rate = tts.generate_voice_clone(
                text=chunk,
                language=language,
                voice_clone_prompt=cache["prompt"],
                max_new_tokens=_budget(chunk),
                **_sampling(temperature, top_p, top_k, rep_penalty),
            )
        return wavs[0], int(rate)

    one.render_cache = RenderCache(  # type: ignore[attr-defined]
        record["slug"], f"clone|{_clone_cache_key((ref_wav, ref_sr), ref_text, use_xvector)}|"
                        f"{settings}|{_model_tag('voice_clone')}")
    return one, ("x-vector" if use_xvector else "ICL")


def speak(voice: str, text: str, language: str = "Auto",
          temperature: float = SAMPLING_DEFAULTS["temperature"],
          top_p: float = SAMPLING_DEFAULTS["top_p"],
          top_k: int = SAMPLING_DEFAULTS["top_k"],
          rep_penalty: float = SAMPLING_DEFAULTS["repetition_penalty"],
          postproc: str = POSTPROCESS_DEFAULT) -> Tuple[Any, str]:
    """Say `text` in a saved voice, in one call — the endpoint an external tool hits.

    Everything the Clone tab needs (reference clip, transcript, x-vector fallback) is read
    from the voice's own sidecar on the server, so a caller only has to know the voice's
    name. That is what makes `scripts/say.py` a one-liner from a shell, which is the shape a
    coding assistant actually wants. Same chunking, post-processing and memory floors as the
    tabs; it shares the one GPU, so a long line here queues a UI generation behind it.
    """
    if not (text or "").strip():
        return None, "Error: Text is required."
    try:
        one, mode = _voice_synth(voice, language, temperature, top_p, top_k, rep_penalty)
        budget = _budget(text)
        wav, sr, chunk_notes = generate_chunked(text, language, one,
                                                cache=getattr(one, "render_cache", None))
        return _ok(*_present(wav, sr, budget, f"Speak ({mode})", postproc, chunk_notes))
    except MediaError as e:
        return None, f"Error: {e}"
    except Exception as e:  # noqa: BLE001 - a caller over the network gets a message, not a traceback
        return None, f"Error: {type(e).__name__}: {e}"


def _audio_html(wav: np.ndarray, sr: int) -> str:
    """A self-contained <audio> element for one streamed piece.

    Gradio's Audio component keeps a single <audio> element and, on a value update, swaps
    the src — its autoplay fires on the first source load and not on the ones after, which
    is exactly the "first few pieces play, then it goes quiet" failure on a streamed reply.
    A brand-new element per yield is what the browser needs to honour autoplay again, and a
    data: URI keeps playback independent of Gradio's temp-file serving and its cleanup.
    `controls` stays on so a browser that blocks autoplay is one click away from playing it.
    """
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(wav, dtype=np.float32), int(sr), format="WAV", subtype="PCM_16")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (f'<audio controls autoplay playsinline style="width:100%" '
            f'src="data:audio/wav;base64,{payload}"></audio>')


def chat_say(text: str, history: Any, voice: str, language: str = "Auto",
             temperature: float = SAMPLING_DEFAULTS["temperature"],
             top_p: float = SAMPLING_DEFAULTS["top_p"],
             top_k: int = SAMPLING_DEFAULTS["top_k"],
             rep_penalty: float = SAMPLING_DEFAULTS["repetition_penalty"],
             postproc: str = POSTPROCESS_DEFAULT,
             synth: Any = None, progress: Any = None):
    """Chat-style reply: speak `text` piece by piece, yielding each one as it is ready.

    The model cannot stream samples, so this streams at the only granularity it has —
    sentences. You hear the first one after ~6-9s instead of waiting for the whole reply,
    and the chat log fills in as it goes. `synth` is injected so the flow is testable without
    weights; the real one comes from `_voice_synth`.
    """
    messages = list(history or [])
    wanted = (text or "").strip()

    def snapshot() -> List[dict]:
        """A copy per yield. Mutating one list and handing it out repeatedly means every
        update shows the *final* state once Gradio comes to serialise it — and makes the
        streaming impossible to test."""
        return [dict(message) for message in messages]

    if not wanted:
        yield snapshot(), None, "Nothing to say."
        return
    messages.append({"role": "user", "content": wanted})
    messages.append({"role": "assistant", "content": "…"})
    yield snapshot(), None, "Thinking…"
    try:
        if synth is None:
            synth, mode = _voice_synth(voice, language, temperature, top_p, top_k, rep_penalty)
        else:
            mode = "test"
        spoken = 0.0
        pieces = 0
        replayed = 0
        for piece in iter_chunked(wanted, language, synth, progress,
                                  cache=getattr(synth, "render_cache", None), quick_start=True):
            wav, sr = piece["wav"], piece["sr"]
            listen_wav, listen_sr = postprocess_output(wav, sr, postproc)
            spoken += len(wav) / sr
            pieces += 1
            replayed += bool(piece.get("cached"))
            messages[-1]["content"] = (piece["text"] if pieces == 1
                                       else f"{messages[-1]['content']} {piece['text']}")
            yield (snapshot(), _audio_html(listen_wav, listen_sr),
                   f"Speaking piece {pieces}/{piece['total']} — {spoken:.1f}s so far "
                   f"({mode}, {listen_sr}Hz{', from cache' if piece.get('cached') else ''})")
        from_cache = f", {replayed} from the render cache" if replayed else ""
        yield snapshot(), gr.update(), (f"Said {pieces} piece{'s' if pieces != 1 else ''}, "
                                        f"{spoken:.1f}s of audio in {voice}{from_cache}")
    except MediaError as e:
        messages[-1]["content"] = f"⚠ {e}"
        yield snapshot(), None, f"Error: {e}"
    except Exception as e:  # noqa: BLE001 - the chat must stay usable after a bad reply
        messages[-1]["content"] = f"⚠ {type(e).__name__}: {e}"
        yield snapshot(), None, f"Error: {type(e).__name__}: {e}"


def voices_available() -> str:
    """Machine-readable voice list for `say.py --list`."""
    records = list_voices()
    if not records:
        return "(no voices saved yet — save one on the Clone or Design tab)"

    def line(r: Dict[str, Any]) -> str:
        # A described voice has no clip, so no length: it used to print as "Nones".
        if voice_route(r) == "design":
            return f"{r['slug']}\tdescribed\t{_short(r.get('description') or r.get('instruct'), 60)}"
        return f"{r['slug']}\t{r.get('seconds') or '?'}s\t{_short(r.get('text'), 60)}"
    return "\n".join(line(r) for r in records)



def _clone_cache_key(audio_tuple, ref_text, x_vector_only) -> str:
    """Content hash of the reference clip + transcript + mode.

    Keyed on the PCM bytes, not on (length, float sum): two different clips with equal
    length and a matching sum used to share one cached clone prompt, which on a shared
    LAN meant answering in someone else's voice.
    """
    wav, sr = audio_tuple
    pcm = np.ascontiguousarray(wav, dtype=np.float32).tobytes()
    digest = hashlib.sha256(pcm).hexdigest()[:32]
    return f"{sr}|{wav.shape[0]}|{digest}|{ref_text or ''}|{bool(x_vector_only)}"


CLONE_OUTPUTS = 2 + MAX_CANDIDATES + 3     # audio, status, takes…, note, take state, save picker
SAVE_REFERENCE = "reference"


def _clone_failure(message: str) -> Tuple[Any, ...]:
    """Every exit from `generate_voice_clone` returns one value per wired output.

    The early-validation exits used to return just (audio, status) to a callback wired to
    nine outputs, which Gradio rejects as a server error — so "Target text is required"
    reached the user as a red error toast instead of a status line.
    """
    return (None, message) + tuple(gr.skip() for _ in range(CLONE_OUTPUTS - 2))


def _clone_prompt(tts: Any, wav: np.ndarray, sr: int, ref_text: str, x_vector_only: bool) -> Any:
    """The encoded reference (codec codes + speaker embedding), cached by content.

    One helper for the Clone tab and Speak/Chat, which used to carry two copies of the
    cache logic. A prompt is a few hundred codec frames and one embedding, so keeping the
    last `CLONE_CACHE_SIZE` costs nothing and alternating between references stops
    re-encoding both every time.
    """
    key = _clone_cache_key((wav, sr), ref_text, x_vector_only)
    prompt = _CLONE_PROMPTS.get(key)
    if prompt is None:
        prompt = tts.create_voice_clone_prompt(
            ref_audio=(wav, sr),
            ref_text=(ref_text or "").strip() or None,
            x_vector_only_mode=bool(x_vector_only),
        )
        _CLONE_PROMPTS[key] = prompt
        while len(_CLONE_PROMPTS) > CLONE_CACHE_SIZE:
            _CLONE_PROMPTS.popitem(last=False)
    else:
        _CLONE_PROMPTS.move_to_end(key)
    return prompt


def save_choices(take_state: Any = None) -> List[Tuple[str, str]]:
    """What the Clone tab's one Save button can keep: the reference, or any take."""
    choices = [("The reference clip + its transcript (best for cloning again)", SAVE_REFERENCE)]
    if isinstance(take_state, dict):
        for take in take_state.get("takes") or []:
            if take.get("path"):
                choices.append((f"Generated {take['label']}", str(take["index"])))
    return choices


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only,
                         temperature, top_p, top_k, rep_penalty,
                         postproc=POSTPROCESS_DEFAULT,
                         candidates=1, seed="", clean_reference=True,
                         progress=gr.Progress(track_tqdm=True)):
    if not target_text or not target_text.strip():
        return _clone_failure("Error: type what the cloned voice should say (step 2).")
    try:
        audio_tuple = _audio_to_tuple(ref_audio)
    except (MediaError, OSError, ValueError) as e:
        return _clone_failure(f"Error: {e}")
    if audio_tuple is None:
        return _clone_failure("Error: no reference yet — upload, record or load one in step 1.")
    if not use_xvector_only and not (ref_text or "").strip():
        return _clone_failure("Error: the reference needs a transcript — press Transcribe, or "
                              "tick 'x-vector only' under Reference options.")
    try:
        n = max(1, min(int(candidates or 1), MAX_CANDIDATES))
    except (TypeError, ValueError):
        return _clone_failure("Error: Candidates must be a whole number.")
    seed_text = str(seed or "").strip()
    if seed_text:
        if not re.fullmatch(r"-?\d+", seed_text):
            return _clone_failure("Error: Seed must be an integer (or blank for random).")
        base_seed = int(seed_text) % (2 ** 31)
    else:
        base_seed = int(time.time() * 1000) % (2 ** 31)

    ref_wav, ref_sr = audio_tuple
    ref_seconds = len(ref_wav) / float(ref_sr)
    fit = transcript_fit(ref_seconds, ref_text or "", bool(use_xvector_only))
    text_check = None
    if not use_xvector_only:
        # On the selection as it stands (before cleanup), so the checklist can find it again.
        text_check = check_reference_transcript(ref_wav, ref_sr, ref_text or "", may_load=n > 1)
    if clean_reference:
        ref_wav = prepare_reference(ref_wav, ref_sr)
    asr_language = None if (not language or language == "Auto") else str(language)
    try:
        budget = _budget(target_text)
        with HUB.lock:
            tts = HUB.acquire("voice_clone")
            prompt = _clone_prompt(tts, ref_wav, ref_sr, ref_text, bool(use_xvector_only))

            ref_embedding = None
            try:
                ref_embedding = tts.model.extract_speaker_embedding(
                    audio=_to_24k(ref_wav, ref_sr), sr=24000)
                ref_embedding = ref_embedding.float().cpu().numpy()
            except Exception:  # noqa: BLE001 - similarity is a bonus, never a blocker
                ref_embedding = None

            def one(chunk: str) -> Tuple[np.ndarray, int]:
                wavs, rate = tts.generate_voice_clone(
                    text=chunk,
                    language=language,
                    voice_clone_prompt=prompt,
                    max_new_tokens=_budget(chunk),
                    **_sampling(temperature, top_p, top_k, rep_penalty),
                )
                return wavs[0], rate

            results: List[Dict[str, Any]] = []
            for i in range(n):
                if n > 1:
                    progress((i + 1) / (n + 2.0), desc=f"Candidate {i + 1} of {n}")
                torch.manual_seed(base_seed + i)
                wav, sr, chunk_notes = generate_chunked(target_text, language, one, progress)
                sim = wer = confidence = None
                # Defects are measured whether or not Whisper is available: they are the
                # signals that survive a disabled ASR, and they cost microseconds.
                defects = defect_report(wav, sr, target_text)
                if ref_embedding is not None:
                    try:
                        cand = tts.model.extract_speaker_embedding(
                            audio=_to_24k(wav, sr), sr=24000).float().cpu().numpy()
                        sim = _cosine(ref_embedding, cand)
                    except Exception:  # noqa: BLE001
                        sim = None
                if n > 1 and ASR is not None and ASR.enabled:
                    try:
                        # One beam search for what Whisper heard (WER) and one teacher-forced
                        # pass for how well the audio carries the text we asked for.
                        hypothesis, confidence = ASR.assess(wav, sr, target_text, asr_language)
                        wer = word_error_rate(hypothesis, target_text)
                    except Exception:  # noqa: BLE001 - scoring must not sink the generation
                        wer = confidence = None
                metrics: Dict[str, Any] = {"index": i + 1, "wav": wav, "sr": int(sr),
                                            "sim": sim, "wer": wer, "confidence": confidence,
                                            "defects": defects, "chunks": chunk_notes}
                metrics["penalty"] = candidate_penalty(wer, confidence, defects)
                results.append(metrics)

        best = choose_best_candidate(results)
        chosen = results[best]
        kept_index = chosen["index"]
        # The auto-pick has to be impossible to miss: the player above shows one clip out of N
        # and nothing else on the page said which one you were listening to.
        report = _report(chosen["wav"], chosen["sr"], budget,
                         f"VoiceClone · kept take #{kept_index} of {n}" if n > 1 else "VoiceClone")
        notes: List[str] = []
        if n > 1:
            def fmt(m: Dict[str, Any]) -> str:
                bits = [f"pen {m['penalty']:.2f}"]
                if m["wer"] is not None:
                    bits.append(f"WER {m['wer']:.2f}")
                if m["confidence"] is not None:
                    bits.append(f"conf {m['confidence']:+.2f}")
                if m["sim"] is not None:
                    bits.append(f"SIM {m['sim']:.3f}")
                flags = defect_flags(m.get("defects") or {})
                if flags:
                    bits.append("⚠ " + ", ".join(flags))
                return f"#{m['index']} " + " ".join(bits)
            scored_by = ("WER + Whisper confidence + waveform defects"
                         if any(m["wer"] is not None for m in results)
                         else "waveform defects (transcription off)")
            notes.append(f"Auto-pick: take #{kept_index} of {n}, scored by {scored_by}.")
            notes.extend("  " + fmt(m) for m in results)
        else:
            flags = defect_flags(chosen.get("defects") or {})
            if flags:
                notes.append("Take flags: " + ", ".join(flags) +
                             " — the clip is worth a listen before you keep it")
        if fit:
            notes.append("WARNING: " + fit)
        check_line, check_bad = reference_transcript_line(text_check)
        if check_line:
            if check_bad:
                notes.insert(0, "WARNING: " + check_line)
            else:
                notes.append(check_line)
        if clean_reference:
            notes.append(f"Reference cleaned: {ref_seconds:.1f}s → {len(ref_wav) / ref_sr:.1f}s "
                         f"(DC, rumble, edge silence)")
        if chosen.get("chunks"):
            notes.extend(chosen["chunks"])
        status = report + (("\n" + "\n".join(notes)) if notes else "")
        out_wav, out_sr = postprocess_output(chosen["wav"], chosen["sr"], postproc)
        # Hand back every take, not just the winner: the auto-pick is a first opinion, and
        # the only honest way to let someone overrule it is to let them hear the alternatives.
        # Each one is also written to disk so the Save button can name it later.
        takes: List[Any] = []
        paths: List[Optional[str]] = []
        for item in results:
            try:
                paths.append(_write_take(item["wav"], item["sr"]))
            except (OSError, ValueError):  # noqa: BLE001 - disk trouble must not lose the audio
                paths.append(None)
            if n > 1:
                take_wav, take_sr = postprocess_output(item["wav"], item["sr"], postproc)
                takes.append(gr.update(value=(take_sr, take_wav), visible=True,
                                       label=take_label(item["index"], item["wer"], item["sim"],
                                                        item["index"] == kept_index,
                                                        item.get("confidence"), item.get("defects"))))
            else:
                takes.append(gr.update(value=None, visible=False))
        while len(takes) < MAX_CANDIDATES:
            takes.append(gr.update(value=None, visible=False))
        take_state = build_take_state(results, kept_index, paths, target_text, language)
        # The one Save control offers the reference and every take. It stays on the
        # reference by default — that is what "save this voice" usually means — and the
        # label says which take the auto-pick chose, so overruling it is one click.
        pick_update = gr.update(
            choices=save_choices(take_state), value=SAVE_REFERENCE,
            label=(f"What to save — the auto-pick was take #{kept_index}" if n > 1
                   else "What to save"))
        return (_ok((out_sr, out_wav), status + _post_note(postproc)) + tuple(takes)
                + (gr.update(visible=n > 1), take_state, pick_update))
    except Exception as e:  # noqa: BLE001
        return _clone_failure(f"Error: {type(e).__name__}: {e}")


def clip_guidance(seconds: float, minimum: float = REF_MIN_SECONDS,
                  maximum: float = REF_MAX_SECONDS, target: float = REF_TARGET_SECONDS) -> str:
    """Tell the user whether the selected stretch is a good clone reference."""
    if seconds < minimum:
        verdict = f"too short — use at least {minimum:g}s"
    elif seconds > maximum:
        verdict = f"long — clones best under {maximum:g}s and runs faster"
    elif abs(seconds - target) <= 2.0:
        verdict = "good length"
    else:
        verdict = "usable"
    return f"{seconds:.2f}s selected — {verdict}. Sweet spot {minimum:g}-{maximum:g}s, target {target:g}s."


def _state_audio(state: Any) -> Tuple[np.ndarray, int]:
    if not isinstance(state, dict) or not state.get("path"):
        raise MediaError("Upload a clip first — nothing is loaded.")
    return load_media(state["path"])


def _clip_outputs(state: Any, start: Any, end: Any) -> Tuple[Any, Any, str, float, float]:
    """Clamp the selection and produce (waveform, audio, info, start, end)."""
    wav, sr = _state_audio(state)
    start, end = trim_window(len(wav) / sr, start, end)
    clip = slice_audio(wav, sr, start, end)
    return render_waveform(wav, sr, start, end), (sr, clip), clip_guidance(end - start), start, end


def import_media(file_value: Any) -> Tuple[Any, str, Any, Any, Any, Any, str]:
    """Take a video/audio upload, extract its audio, and pre-select the best window."""
    path = file_value[0] if isinstance(file_value, (list, tuple)) and file_value else file_value
    try:
        prune_media()
        wav, sr = load_media(path)
        suffix = os.path.splitext(str(path))[1].lower()
        kind = "video — audio track extracted" if suffix in VIDEO_SUFFIXES else "audio"
        stored = os.path.join(media_dir(), f"{uuid.uuid4().hex}.wav")
        sf.write(stored, wav, sr)
        state = {"path": stored, "name": os.path.basename(str(path)), "kind": kind,
                 "duration": round(len(wav) / sr, 3), "sample_rate": int(sr)}
        # Start on the best stretch, not on the first few seconds: an intro, a count-in or
        # room tone at the head of a recording used to be the default reference.
        try:
            first = auto_pick_window(wav, sr)
        except (MediaError, ValueError):
            first = (None, None)
        wave, audio, info, start, end = _clip_outputs(state, *first)
        status = (f"Loaded {state['name']} ({kind}): {len(wav) / sr:.1f}s @ {sr}Hz — "
                  f"selected {start:.1f}–{end:.1f}s. Drag the sliders to change it, then "
                  f"Transcribe.")
        duration = round(len(wav) / sr, 2)
        return (state, status, wave,
                gr.update(maximum=duration, value=start),
                gr.update(maximum=duration, value=end),
                audio, info)
    except (MediaError, OSError, ValueError) as e:
        return None, f"Error: {type(e).__name__}: {e}", None, gr.update(), gr.update(), None, ""


def update_clip(state: Any, start: Any, end: Any) -> Tuple[Any, Any, str, Any, Any]:
    """Re-slice after the user drags the selection sliders."""
    try:
        wave, audio, info, start, end = _clip_outputs(state, start, end)
        return wave, audio, info, gr.update(value=start), gr.update(value=end)
    except (MediaError, OSError, ValueError) as e:
        return None, None, f"Error: {type(e).__name__}: {e}", gr.update(), gr.update()


def auto_pick_clip(state: Any) -> Tuple[Any, Any, Any, Any, str]:
    """Jump the selection to the loudest window (drops silence / room tone)."""
    try:
        wav, sr = _state_audio(state)
        start, end = auto_pick_window(wav, sr)
        wave, audio, info, start, end = _clip_outputs(state, start, end)
        return wave, audio, info, gr.update(value=start), gr.update(value=end)
    except (MediaError, OSError, ValueError) as e:
        return None, None, f"Error: {type(e).__name__}: {e}", gr.update(), gr.update()


def transcribe_reference(state: Any, start: Any, end: Any, language: Any,
                         target_language: Any = None) -> Tuple[Any, str, Any]:
    """Whisper the *selected* window into the editable transcript box.

    Returns (transcript, status, target-language update). When the target language is
    still Auto it is set to the language Whisper heard: upstream's own advice is to name
    the language rather than leave it on Auto, and by now we know it.
    """
    if ASR is None or not ASR.enabled:
        return (gr.update(), "Error: transcription is disabled on this server (--asr-model). "
                             "Type the transcript, or tick x-vector only.", gr.update())
    try:
        wav, sr = _state_audio(state)
        start, end = trim_window(len(wav) / sr, start, end)
        clip = slice_audio(wav, sr, start, end)
        t0 = time.time()
        result = ASR.transcribe_detailed(
            clip, sr, None if not language or language == "Auto" else str(language))
    except (MediaError, OSError, ValueError, RuntimeError) as e:
        return gr.update(), f"Error: {type(e).__name__}: {e}", gr.update()
    heard = result.get("language")
    lang_update = gr.update()
    lang_note = ""
    if heard and heard in LANGUAGE_VALUES and (not target_language or target_language == "Auto"):
        lang_update = gr.update(value=heard)
        lang_note = f" Language set to {heard}."
    return (gr.update(value=result["text"]),
            f"Transcribed {end - start:.1f}s in {time.time() - t0:.1f}s.{lang_note} Check the "
            f"text against what you hear and fix any mistakes before cloning.",
            lang_update)


def clip_fit_status(ref_text: Any, start: Any, end: Any, x_vector_only: Any = False) -> Any:
    """Live transcript/clip sanity line for the Clone tab; it never blocks a generation."""
    try:
        seconds = max(0.0, float(end or 0.0) - float(start or 0.0))
    except (TypeError, ValueError):
        return gr.update()
    return gr.update(value=transcript_fit(seconds, ref_text or "", bool(x_vector_only)))


def reference_check(ref_audio: Any, ref_text: Any, x_vector_only: Any = False,
                    start: Any = None) -> str:
    """The Clone tab's reference checklist, recomputed from the clip that will be cloned.

    Reads the reference box itself, not the upload state, so it is right whichever way the
    reference arrived — upload, microphone or library.
    """
    try:
        parsed = _audio_to_tuple(ref_audio)
    except (MediaError, OSError, ValueError) as e:
        return f"⚠️ The reference could not be read: {e}"
    if parsed is None:
        return "*No reference yet — upload, record or load one above.*"
    wav, sr = parsed
    try:
        offset = float(start) if start not in (None, "") else None
    except (TypeError, ValueError):
        offset = None
    return reference_report(wav, sr, str(ref_text or ""), bool(x_vector_only), offset)


def load_voice_for_cloning(slug: Any) -> Tuple[Any, ...]:
    """Put a saved voice back into step 1 of the Clone tab, as if it had been uploaded.

    Returns the `import_media` outputs plus (transcript, language). The whole clip is
    selected: its transcript describes all of it, so trimming would break the match.
    """
    blank = (None, "", None, gr.update(), gr.update(), None, "", gr.update(), gr.update())
    record, error = _find_voice(str(slug or ""))
    if error:
        return (None, error) + blank[2:]
    wav_path = record.get("wav") or _voice_paths(record["slug"])[0]
    if not os.path.exists(wav_path):
        return (None, f"Error: ‘{record['name']}’ is a described voice — it has no clip to "
                      f"clone. Speak and Chat can use it as it is.") + blank[2:]
    try:
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        wav = _normalize_audio(wav)
        prune_media()
        stored = os.path.join(media_dir(), f"{uuid.uuid4().hex}.wav")
        sf.write(stored, wav, sr)
        duration = round(len(wav) / float(sr), 3)
        state = {"path": stored, "name": f"{record['name']} (library)", "kind": "saved voice",
                 "duration": duration, "sample_rate": int(sr)}
        wave, audio, info, start, end = _clip_outputs(state, 0.0, duration)
    except (MediaError, OSError, ValueError) as e:
        return (None, f"Error: {type(e).__name__}: {e}") + blank[2:]
    text = str(record.get("text") or "").strip()
    language = str(record.get("language") or "Auto")
    note = "" if text else " It has no transcript — Transcribe it, or tick x-vector only."
    return (state, f"Loaded ‘{record['name']}’ from the library ({duration:.1f}s).{note}",
            wave, gr.update(maximum=round(duration, 2), value=start),
            gr.update(maximum=round(duration, 2), value=end), audio, info,
            gr.update(value=text),
            gr.update(value=language if language in LANGUAGE_VALUES else "Auto"))


def save_clone_voice(what: Any, name: Any, ref_audio: Any, ref_text: Any, language: Any,
                     take_state: Any) -> Tuple[Any, str]:
    """The Clone tab's one Save button: keep the reference, or one of the generated takes.

    There used to be two name boxes, two Save buttons, a take radio and three status boxes
    for this, and nothing said which one did what.
    """
    choice = str(what or SAVE_REFERENCE)
    if choice == SAVE_REFERENCE:
        name_update, message, _ = save_reference_voice(str(name or ""), ref_audio,
                                                       str(ref_text or ""), str(language or "Auto"))
    else:
        name_update, message, _ = save_generated_voice(take_state, choice, str(name or ""))
    return name_update, message


def refresh_status() -> str:
    """Sidebar runtime panel: what is loaded where, and how much memory is left."""
    if HUB is None:
        return "Models are not loaded yet."
    return "\n".join(f"- {part.strip()}" for part in HUB.status().split("|"))


# ----------------------------------------------------------------------------
# Voices tab
# ----------------------------------------------------------------------------
VOICE_TABLE_HEADERS = ["Name", "Slug", "Speaks from", "Clip", "Language", "Transcript / description"]


def _short(text: Any, limit: int = 70) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def voice_table_rows() -> List[List[Any]]:
    """One row per saved voice for the Voices tab table."""
    rows = []
    for record in list_voices():
        route = voice_route(record)
        has_clip = os.path.exists(record.get("wav") or _voice_paths(record["slug"])[0])
        words = record.get("description") or record.get("instruct") if route == "design" \
            else record.get("text")
        rows.append([record.get("name", record["slug"]), record["slug"],
                     "description" if route == "design" else "clip",
                     f"{record.get('seconds')}s" if has_clip and record.get("seconds") else "—",
                     record.get("language") or "Auto", _short(words)])
    return rows


def voice_details(slug: Any) -> Tuple[str, Any, Any]:
    """(details markdown, preview clip, preference radio) for the selected voice."""
    if not slug:
        return "*Pick a voice to see its details.*", None, gr.update(value="auto")
    record, error = _find_voice(str(slug))
    if error:
        return error, None, gr.update(value="auto")
    wav_path = record.get("wav") or _voice_paths(record["slug"])[0]
    preview = None
    if os.path.exists(wav_path):
        try:
            wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            preview = (int(sr), _normalize_audio(wav))
        except (OSError, ValueError):
            preview = None
    route = voice_route(record)
    description = str(record.get("description") or record.get("instruct") or "").strip()
    lines = [f"### {record.get('name', record['slug'])}",
             f"`{record['slug']}` · saved {str(record.get('created', '?'))[:10]} · "
             f"language {record.get('language') or 'Auto'}",
             f"**Speak and Chat use:** {'its description (Voice Design)' if route == 'design' else 'its clip (cloning)'}"]
    if preview is not None:
        lines.append(f"**Clip:** {record.get('seconds')}s")
    if record.get("text"):
        lines.append(f"**Transcript:** {record['text']}")
    if description:
        lines.append(f"**Description:** {description}")
    if preview is None:
        lines.append("*No clip — this voice can only be spoken from its description.*")
    prefer = str(record.get("prefer") or "auto").lower()
    return "\n\n".join(lines), preview, gr.update(value=prefer if prefer in VOICE_ROUTES else "auto")


def save_voice_preference(slug: Any, prefer: Any) -> str:
    if not slug:
        return "Error: pick a voice first."
    record, error = _find_voice(str(slug))
    if error:
        return error
    try:
        updated = set_voice_preference(record["slug"], str(prefer or "auto"))
    except (OSError, ValueError) as e:
        return f"Error: {type(e).__name__}: {e}"
    updated["wav"], updated["json"] = _voice_paths(updated["slug"])
    route = voice_route(updated)
    note = ""
    if updated["prefer"] != "auto" and \
            (route == "design") != (updated["prefer"] == "description"):
        note = " (this voice has nothing to speak that way from, so it still uses the other)"
    return (f"‘{updated['name']}’ is now spoken from its "
            f"{'description' if route == 'design' else 'clip'} in Speak and Chat{note}.")


def pick_voice_row(evt: gr.SelectData) -> Any:
    """Voices-tab table click -> the slug of that row (column 1)."""
    row = getattr(evt, "row_value", None)
    if isinstance(row, (list, tuple)) and len(row) > 1:
        return gr.update(value=row[1])
    return gr.update()


def ask_delete(slug: Any) -> Tuple[Any, str]:
    """First click of the two-step delete: reveal the confirm button, naming the voice."""
    if not slug:
        return gr.update(visible=False), "Error: pick a voice to delete."
    record, error = _find_voice(str(slug))
    if error:
        return gr.update(visible=False), error
    return (gr.update(visible=True, value=f"Yes, delete ‘{record['name']}’"),
            f"Press the red button to delete ‘{record['name']}’ for good — there is no undo.")


def confirm_delete(slug: Any) -> Tuple[Any, str, Any]:
    dropdown, message = delete_saved_voice(str(slug or ""))
    return dropdown, message, gr.update(visible=False)


UI_THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")
UI_CSS = (".gradio-container {max-width: 1500px !important;} "
          ".gradio-container h1 {margin-bottom: .1em;} "
          ".gradio-container h4 {margin: .4em 0 .2em; opacity: .85;} "
          "#cl-check p, #cl-check li {margin: .1em 0;}")

ETHICS_NOTE = ("> **Use it kindly.** Clone only voices you have permission to use — your own, or "
               "someone else's with their consent — and never pass synthetic audio off as the "
               "real person. Running it locally does not change that. See *Ethical use* in the "
               "README.")


# ============================================================================
# UI
# ============================================================================
def build_ui() -> gr.Blocks:
    """The web UI. Layout rules, so it does not drift back into a wall of boxes:

    * one **Status** box per tab — every action on that tab reports there;
    * settings and the runtime line live in the **sidebar**, not in every status;
    * one **Save** control per tab, saying what it saves;
    * the library is managed in one place, the **Voices** tab, and every list of voices
      refreshes whenever it changes.
    """
    with gr.Blocks(title="qwen3-tts-ui") as demo:
        gr.Markdown(
            "# qwen3-tts-ui\n"
            "**Clone** a voice from a short clip · **Design** one from a description · "
            "**Speak** and **Chat** as any saved voice · manage them under **Voices**. "
            "Settings and the GPU/RAM status are in the sidebar — the › at the top left.")

        with gr.Sidebar(open=False, label="Settings"):
            gr.Markdown("### Runtime")
            rt_status = gr.Markdown(refresh_status())
            rt_timer = gr.Timer(5.0)
            rt_timer.tick(refresh_status, outputs=[rt_status], concurrency_limit=None,
                          show_progress="hidden", api_visibility="private")
            gr.Markdown(
                "*Weights wait in CPU RAM and move to the GPU on demand, so the first request "
                "after switching between Clone and Design pays a few seconds.*")
            gr.Markdown("### Output")
            s_postproc = gr.Dropdown(
                label="Post-processing",
                choices=[(POSTPROCESS_LABELS[m], m) for m in POSTPROCESS_MODES],
                value=POSTPROCESS_DEFAULT,
                info="The vocoder decodes at 24kHz. 'Warm' resamples properly and trims the "
                     "imaging near 12kHz that reads as 'digital'. Saved takes are always the "
                     "unprocessed decode.")
            gr.Markdown("### Sampling")
            gr.Markdown(
                "Checkpoint defaults. Quality comes from the reference and its transcript far "
                "more than from these; lower the temperature or raise the repetition penalty "
                "only if a take rambles or never stops.")
            s_temperature = gr.Slider(0.1, 2.0, value=SAMPLING_DEFAULTS["temperature"], step=0.05,
                                      label="Temperature")
            s_top_p = gr.Slider(0.1, 1.0, value=SAMPLING_DEFAULTS["top_p"], step=0.05, label="Top-p")
            s_top_k = gr.Slider(1, 200, value=SAMPLING_DEFAULTS["top_k"], step=1, label="Top-k")
            s_rep = gr.Slider(1.0, 1.5, value=SAMPLING_DEFAULTS["repetition_penalty"], step=0.01,
                              label="Repetition penalty")
            s_reset = gr.Button("Reset to defaults", size="sm")
            s_reset.click(lambda: (SAMPLING_DEFAULTS["temperature"], SAMPLING_DEFAULTS["top_p"],
                                   SAMPLING_DEFAULTS["top_k"], SAMPLING_DEFAULTS["repetition_penalty"]),
                          outputs=[s_temperature, s_top_p, s_top_k, s_rep],
                          api_visibility="private")
        sampling = [s_temperature, s_top_p, s_top_k, s_rep]
        output_options = [s_postproc]

        with gr.Tabs() as tabs:
            # ---------------------------------------------------------------- Clone
            with gr.Tab("Clone a voice", id="clone"):
                gr.Markdown(ETHICS_NOTE)
                cl_media_state = gr.State(value=None)
                cl_take_state = gr.State(value=None)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1 · Reference — the voice to copy")
                        with gr.Tabs():
                            with gr.Tab("Upload"):
                                cl_upload = gr.File(
                                    label="Video or audio file — only the audio track is used",
                                    file_types=MEDIA_SUFFIXES, type="filepath", height=110)
                            with gr.Tab("Record"):
                                cl_mic = gr.Audio(label="Record 5–10s of natural speech",
                                                  sources=["microphone"], type="filepath")
                            with gr.Tab("From library"):
                                with gr.Row():
                                    cl_library = gr.Dropdown(label="Saved voice", choices=voice_choices(),
                                                             value=None, scale=4)
                                    cl_load_btn = gr.Button("Load", scale=1)
                        cl_wave = gr.Image(label="Waveform — the shaded part is the reference",
                                           type="pil", height=150, interactive=False)
                        with gr.Row():
                            cl_start = gr.Slider(0, 60, value=0, step=0.05, label="Start (s)")
                            cl_end = gr.Slider(0, 60, value=0, step=0.05, label="End (s)")
                        with gr.Row():
                            cl_auto_btn = gr.Button("Auto-pick the best stretch", size="sm")
                        cl_ref_audio = gr.Audio(label="Reference clip — exactly what will be cloned",
                                                type="filepath", interactive=False)
                        with gr.Row(equal_height=True):
                            cl_asr_language = gr.Dropdown(
                                label="Transcript language", scale=2,
                                choices=[("Auto-detect", "Auto")] +
                                        [(k.title(), k) for k in ASR_LANGUAGE_CODES],
                                value="Auto")
                            cl_transcribe_btn = gr.Button("Transcribe the clip", variant="primary",
                                                          scale=1)
                        cl_ref_text = gr.Textbox(
                            label="What the reference says (must match the clip word for word)",
                            lines=3,
                            placeholder="Press Transcribe, then fix anything Whisper got wrong.")
                        with gr.Accordion("Reference options", open=False):
                            cl_clean = gr.Checkbox(
                                label="Clean up the reference (remove DC, rumble below 60Hz and "
                                      "long edge silence)", value=True)
                            cl_xvector = gr.Checkbox(
                                label="x-vector only — no transcript needed, but a weaker likeness",
                                value=False)
                        cl_check = gr.Markdown(reference_check(None, ""), elem_id="cl-check")

                    with gr.Column(scale=1):
                        gr.Markdown("#### 2 · What it should say")
                        cl_text = gr.Textbox(label="Text for the cloned voice", lines=4,
                                             placeholder="New words for the cloned voice to speak…")
                        with gr.Row():
                            cl_language = gr.Dropdown(label="Language", choices=LANGUAGE_CHOICES,
                                                      value="Auto", scale=1,
                                                      info="Transcribe fills this in.")
                            cl_candidates = gr.Slider(1, MAX_CANDIDATES, value=3, step=1, scale=1,
                                                      label="Takes (best of N)",
                                                      info="One generation each; the best is kept.")
                        with gr.Accordion("Seed", open=False):
                            cl_seed = gr.Textbox(show_label=False, value="",
                                                 placeholder="blank = random; a number reproduces a take")
                        cl_btn = gr.Button("Clone & generate", variant="primary")
                        cl_audio = gr.Audio(label="Result (the auto-picked take)", type="numpy")
                        cl_status = gr.Textbox(label="Status", lines=5, max_lines=14,
                                               interactive=False)
                        cl_takes_note = gr.Markdown(
                            visible=False,
                            value="*Every take, scored. The pick ranks Whisper's word error rate, "
                                  "its confidence and waveform defects; speaker similarity only "
                                  "breaks ties. Trust your ears over the numbers.*")
                        cl_takes = []
                        for row_start in range(0, MAX_CANDIDATES, 2):
                            with gr.Row():
                                for i in range(row_start, min(row_start + 2, MAX_CANDIDATES)):
                                    cl_takes.append(gr.Audio(label=f"Take #{i + 1}", type="numpy",
                                                             visible=False))

                        gr.Markdown("#### 3 · Save it to the voice library")
                        cl_save_what = gr.Radio(label="What to save", choices=save_choices(None),
                                                value=SAVE_REFERENCE)
                        with gr.Row(equal_height=True):
                            cl_save_name = gr.Textbox(label="Name", scale=3,
                                                      placeholder="e.g. phone voice memo")
                            cl_save_btn = gr.Button("Save voice", scale=1)

                import_outputs = [cl_media_state, cl_status, cl_wave, cl_start, cl_end,
                                  cl_ref_audio, cl_check]
                slice_outputs = [cl_wave, cl_ref_audio, cl_check, cl_start, cl_end]
                check_inputs = [cl_ref_audio, cl_ref_text, cl_xvector, cl_start]

                def then_check(event: Any) -> Any:
                    return event.then(reference_check, inputs=check_inputs, outputs=[cl_check],
                                      show_progress="hidden", api_visibility="private")

                then_check(cl_upload.upload(import_media, inputs=[cl_upload], outputs=import_outputs,
                                            api_visibility="private"))
                then_check(cl_mic.stop_recording(import_media, inputs=[cl_mic], outputs=import_outputs,
                                                 api_visibility="private"))
                then_check(cl_load_btn.click(load_voice_for_cloning, inputs=[cl_library],
                                             outputs=import_outputs + [cl_ref_text, cl_language],
                                             api_visibility="private"))
                for slider in (cl_start, cl_end):
                    then_check(slider.release(update_clip, inputs=[cl_media_state, cl_start, cl_end],
                                              outputs=slice_outputs, api_visibility="private"))
                then_check(cl_auto_btn.click(auto_pick_clip, inputs=[cl_media_state],
                                             outputs=slice_outputs, api_visibility="private"))
                then_check(cl_transcribe_btn.click(
                    transcribe_reference,
                    inputs=[cl_media_state, cl_start, cl_end, cl_asr_language, cl_language],
                    outputs=[cl_ref_text, cl_status, cl_language], api_visibility="private"))
                # The checklist is cheap (no model), so it runs outside the one-GPU queue and
                # follows the transcript as it is typed.
                for trigger in (cl_ref_text.input, cl_xvector.change):
                    trigger(reference_check, inputs=check_inputs, outputs=[cl_check],
                            concurrency_limit=None, trigger_mode="always_last",
                            show_progress="hidden", api_visibility="private")

                cl_btn.click(generate_voice_clone,
                             inputs=[cl_ref_audio, cl_ref_text, cl_text, cl_language, cl_xvector]
                                    + sampling + output_options + [cl_candidates, cl_seed, cl_clean],
                             outputs=[cl_audio, cl_status] + cl_takes
                                     + [cl_takes_note, cl_take_state, cl_save_what],
                             api_name="generate_voice_clone")
                cl_save_evt = cl_save_btn.click(
                    save_clone_voice,
                    inputs=[cl_save_what, cl_save_name, cl_ref_audio, cl_ref_text, cl_language,
                            cl_take_state],
                    outputs=[cl_save_name, cl_status], api_visibility="private")

            # ---------------------------------------------------------------- Design
            with gr.Tab("Design a voice", id="design"):
                gr.Markdown(
                    "Describe a voice and hear it say a line. **Save** keeps the description and "
                    "the clip: Speak and Chat talk from the description, and the clip can be "
                    "cloned like any other reference.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        vd_instruct = gr.Textbox(
                            label="Voice description", lines=3,
                            value="Speak in an incredulous tone, but with a hint of panic beginning "
                                  "to creep into your voice.",
                            placeholder="Timbre, age, gender, emotion, accent, speaking rate…")
                        vd_text = gr.Textbox(
                            label="Text to say", lines=4,
                            value="It's in the top drawer... wait, it's empty? No way, that's "
                                  "impossible! I'm sure I put it there!")
                        vd_language = gr.Dropdown(label="Language", choices=LANGUAGE_CHOICES,
                                                  value="Auto")
                        vd_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(label="Designed voice", type="filepath")
                        vd_status = gr.Textbox(label="Status", lines=3, interactive=False)
                        vd_take_state = gr.State()
                        gr.Markdown("#### Save this voice")
                        with gr.Row(equal_height=True):
                            vd_voice_name = gr.Textbox(label="Name", scale=3,
                                                       placeholder="e.g. sleepy sea captain")
                            vd_save_btn = gr.Button("Save voice", scale=1)
                vd_btn.click(generate_voice_design,
                             inputs=[vd_text, vd_language, vd_instruct] + sampling + output_options,
                             outputs=[vd_audio, vd_status, vd_take_state],
                             api_name="generate_voice_design")
                vd_save_evt = vd_save_btn.click(
                    save_designed_voice,
                    inputs=[vd_voice_name, vd_take_state],
                    outputs=[vd_voice_name, vd_status, cl_library], api_visibility="private")

            # ---------------------------------------------------------------- Speak
            with gr.Tab("Speak", id="speak"):
                gr.Markdown(
                    "Pick a saved voice and type what to say; long text is split into sentences "
                    "and stitched. This is also the endpoint scripts use — "
                    "`scripts/say.py --voice <slug> --text \"…\"`.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        sp_voice = gr.Dropdown(label="Voice", choices=voice_choices(), value=None,
                                               allow_custom_value=True,
                                               info="‘(described)’ voices speak from their "
                                                    "description; the rest clone their clip. "
                                                    "Change which on the Voices tab.")
                        sp_text = gr.Textbox(
                            label="What to say", lines=4,
                            value="Build finished. Three tests failed, but I already fixed two of them.")
                        sp_language = gr.Dropdown(label="Language", choices=LANGUAGE_CHOICES,
                                                  value="Auto")
                        sp_btn = gr.Button("Speak", variant="primary")
                    with gr.Column(scale=1):
                        sp_audio = gr.Audio(label="Result", type="numpy")
                        sp_status = gr.Textbox(label="Status", lines=3, interactive=False)
                with gr.Accordion("For scripts: the voice list as the API returns it", open=False):
                    sp_list = gr.Textbox(
                        show_label=False, lines=6, interactive=False, value=voices_available(),
                        info="`scripts/say.py --list` prints this; pass a slug back as --voice.")
                sp_btn.click(speak,
                             inputs=[sp_voice, sp_text, sp_language] + sampling + output_options,
                             outputs=[sp_audio, sp_status], api_name="speak")

            # ---------------------------------------------------------------- Chat
            with gr.Tab("Chat", id="chat"):
                gr.Markdown(
                    "Type a line and hear it. Replies are spoken sentence by sentence, so the "
                    "first one plays after a few seconds instead of after the whole reply.")
                ch_log = gr.Chatbot(label="Conversation", height=320)
                with gr.Row(equal_height=True):
                    ch_text = gr.Textbox(label="Say this", scale=5,
                                         placeholder="Type a sentence and press Enter…")
                    ch_voice = gr.Dropdown(label="Voice", choices=voice_choices(), value=None,
                                           scale=2, allow_custom_value=True)
                    ch_language = gr.Dropdown(label="Language", choices=LANGUAGE_CHOICES,
                                              value="Auto", scale=1)
                with gr.Row():
                    ch_send = gr.Button("Speak", variant="primary", scale=3)
                    ch_clear = gr.Button("Clear", scale=1)
                ch_player = gr.HTML(label="Now playing",
                                    value="<em style='opacity:.6'>Each sentence appears here and "
                                          "starts playing on its own.</em>")
                ch_status = gr.Textbox(label="Status", lines=1, interactive=False)
                chat_inputs = [ch_text, ch_log, ch_voice, ch_language] + sampling + output_options
                ch_send.click(chat_say, inputs=chat_inputs, outputs=[ch_log, ch_player, ch_status],
                              api_name="chat_say")
                ch_text.submit(chat_say, inputs=chat_inputs, outputs=[ch_log, ch_player, ch_status],
                               api_visibility="private")
                ch_clear.click(lambda: ([], "", "Cleared."), outputs=[ch_log, ch_player, ch_status],
                               api_visibility="private")

            # ---------------------------------------------------------------- Voices
            with gr.Tab("Voices", id="voices"):
                gr.Markdown(
                    "Every saved voice, in one place. Click a row to select it. Voices are "
                    "files under `voices/` on the server (`<slug>.wav` + `<slug>.json`).")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        vo_table = gr.Dataframe(value=voice_table_rows(), headers=VOICE_TABLE_HEADERS,
                                                interactive=False, wrap=True, max_height=520)
                    with gr.Column(scale=2):
                        with gr.Row(equal_height=True):
                            vo_select = gr.Dropdown(label="Selected voice", choices=voice_choices(),
                                                    value=None, scale=4)
                            vo_refresh_btn = gr.Button("Refresh", scale=1)
                        vo_details = gr.Markdown(voice_details(None)[0])
                        vo_preview = gr.Audio(label="Saved clip", type="numpy", interactive=False)
                        vo_prefer = gr.Radio(
                            label="Speak and Chat say this voice from…",
                            choices=[("Auto — its description if it has one", "auto"),
                                     ("Its clip (cloning: the steadiest identity)", "clip"),
                                     ("Its description (Voice Design: the brightest sound)",
                                      "description")],
                            value="auto")
                        with gr.Row():
                            vo_use_btn = gr.Button("Use as clone reference", variant="primary")
                            vo_delete_btn = gr.Button("Delete…")
                        vo_confirm_btn = gr.Button("Yes, delete", variant="stop", visible=False)
                        vo_status = gr.Textbox(label="Status", lines=2, interactive=False)

                vo_table.select(pick_voice_row, outputs=[vo_select], api_visibility="private")
                vo_select.change(voice_details, inputs=[vo_select],
                                 outputs=[vo_details, vo_preview, vo_prefer],
                                 api_visibility="private").then(
                    lambda: gr.update(visible=False), outputs=[vo_confirm_btn],
                    api_visibility="private")
                vo_prefer_evt = vo_prefer.input(save_voice_preference, inputs=[vo_select, vo_prefer],
                                                outputs=[vo_status], api_visibility="private")
                then_check(vo_use_btn.click(
                    load_voice_for_cloning, inputs=[vo_select],
                    outputs=import_outputs + [cl_ref_text, cl_language],
                    api_visibility="private").then(
                    lambda: gr.Tabs(selected="clone"), outputs=[tabs], api_visibility="private"))
                vo_delete_btn.click(ask_delete, inputs=[vo_select],
                                    outputs=[vo_confirm_btn, vo_status], api_visibility="private")
                vo_delete_evt = vo_confirm_btn.click(
                    confirm_delete, inputs=[vo_select],
                    outputs=[vo_select, vo_status, vo_confirm_btn], api_visibility="private")
                # Kept as a named endpoint: `scripts/say.py --list` calls /voices_available.
                vo_refresh_evt = vo_refresh_btn.click(voices_available, outputs=[sp_list],
                                                      api_name="voices_available")

        # Anything that changes the library refreshes every list of voices on every tab.
        library_outputs = [cl_library, sp_voice, ch_voice, vo_select, sp_list, vo_table]
        for event in (cl_save_evt, vd_save_evt, vo_delete_evt, vo_prefer_evt, vo_refresh_evt):
            event.then(voice_library_updates, outputs=library_outputs, api_visibility="private")
        # A page opened before someone else saved a voice should still see it on reload.
        demo.load(voice_library_updates, outputs=library_outputs, api_visibility="private")

        gr.Markdown(
            "---\nPowered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba Qwen), "
            "1.7B checkpoints, run offline. **Ethical use:** clone only voices you have "
            "permission to use, never impersonate someone to deceive or defraud them, and say "
            "so when audio is synthetic.")
    return demo


def resolve_credentials(args: Any, environ: Optional[Dict[str, str]] = None,
                        prompt: Any = getpass.getpass) -> Optional[Tuple[str, str]]:
    """Resolve the single shared UI credential without ever putting the secret in argv.

    `--password` on the command line is readable by any local user through `ps`/`/proc`,
    so it is the *last* resort and warns. Preferred sources, highest priority first:
    `--password-file`, `QWEN_TTS_PASSWORD`, then the interactive prompt. The secret is
    never printed. Returns None when no username was requested (or nothing usable was
    supplied), which leaves the app unauthenticated — the caller still applies its
    bind-scope warning in that case.
    """
    environ = os.environ if environ is None else environ
    warnings: List[str] = []

    username = args.username or environ.get("QWEN_TTS_USERNAME")
    if not username:
        if args.password or args.password_file or environ.get("QWEN_TTS_PASSWORD"):
            warnings.append("a password was supplied without a username; starting UNAUTHENTICATED")
        if warnings:
            print(f"[webui] warning: {warnings[0]}.", flush=True)
        return None

    secret: Optional[str] = None
    if args.password_file:
        try:
            with open(args.password_file, "r", encoding="utf-8") as fh:
                secret = fh.read().rstrip("\r\n")
        except OSError as e:
            print(f"[webui] error: cannot read --password-file {args.password_file}: {e}", flush=True)
            return None
        try:
            if stat.S_IMODE(os.stat(args.password_file).st_mode) & 0o077:
                print(f"[webui] warning: {args.password_file} is readable by others — "
                      f"chmod 600 it.", flush=True)
        except OSError:
            pass
    elif environ.get("QWEN_TTS_PASSWORD"):
        secret = environ["QWEN_TTS_PASSWORD"]
    elif args.password:
        print("[webui] warning: --password is visible to every local user via `ps`/proc. "
              "Use --password-file or QWEN_TTS_PASSWORD instead.", flush=True)
        secret = args.password
    else:
        secret = prompt(f"qwen3-tts-ui password for user '{username}' (input hidden): ")

    if not secret:
        print("[webui] warning: no password could be resolved; starting UNAUTHENTICATED.", flush=True)
        return None
    return (username, secret)


def main() -> int:
    global HUB, DEVICE

    parser = argparse.ArgumentParser(description="qwen3-tts-ui — local Qwen3-TTS Gradio demo (1.7B models).")
    parser.add_argument("--models-dir", default=MODELS_DIR, help="Directory holding the Qwen3-TTS-* checkpoints.")
    parser.add_argument("--ip", default="127.0.0.1", help="Bind address (0.0.0.0 to expose on the LAN).")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                        help="sdpa is the default; flash-attn 2 has no sm_120 (Blackwell) wheels.")
    parser.add_argument("--max-resident", type=int, default=1,
                        help="How many 1.7B models may sit on the GPU at once (16GB GPU -> 1-2).")
    parser.add_argument("--max-cpu-resident", type=int, default=2,
                        help="How many 1.7B checkpoints may be pinned in host RAM at once "
                             "(~4.3GB each). Lower this on WSL2 if the Windows host gets tight.")
    parser.add_argument("--ram-floor-mb", type=int, default=RAM_FLOOR_MB,
                        help="Refuse to load another checkpoint if it would leave less than "
                             "this much MemAvailable.")
    parser.add_argument("--device", default="cuda", help="cuda / cuda:0 / cpu")
    parser.add_argument("--preload", action="store_true",
                        help="Load checkpoints into CPU RAM at startup instead of lazily "
                             "(bounded by --max-cpu-resident).")
    parser.add_argument("--force", action="store_true",
                        help="Start even if another Qwen3-TTS process owns the weights (unsafe: "
                             "two copies of the weights in RAM can lock the machine).")
    parser.add_argument("--ssl-certfile", default=None)
    parser.add_argument("--ssl-keyfile", default=None)
    parser.add_argument("--ssl-self-signed", action="store_true",
                        help="Tell Gradio not to verify the certificate it is serving "
                             "(set by run.sh --ssl-self-signed). Without this, launch() "
                             "kills itself: its startup self-check does an https request "
                             "to its own server and fails on the self-signed cert.")
    parser.add_argument("--username", default=None,
                        help="Require this username to use the UI. Also reads "
                             "QWEN_TTS_USERNAME. Required in spirit for anything bound "
                             "beyond localhost.")
    parser.add_argument("--password", default=None,
                        help="DEPRECATED: visible in `ps`. Prefer --password-file or "
                             "QWEN_TTS_PASSWORD; with neither you are prompted.")
    parser.add_argument("--password-file", default=None,
                        help="File containing the UI password (chmod 600). Its contents "
                             "never reach the command line or the logs.")
    parser.add_argument("--share", action="store_true", help="Create a public gradio.app link.")
    parser.add_argument("--asr-model", default=ASR_MODEL_ID,
                        help=f"Whisper model for reference transcription (default {ASR_MODEL_ID}); "
                             f"'none' disables the button.")
    parser.add_argument("--no-asr", action="store_true", help="Same as --asr-model none.")
    parser.add_argument("--ffmpeg-bin", default=FFMPEG_BIN, help="ffmpeg executable for video ingest.")
    args = parser.parse_args()

    globals()["FFMPEG_BIN"] = args.ffmpeg_bin

    DEVICE = torch.device(args.device)
    globals()["ASR"] = Transcriber(model_id="none" if args.no_asr else args.asr_model, device=DEVICE)
    print(f"[webui] transcription: {'disabled' if not ASR.enabled else ASR.model_id} | "
          f"ffmpeg: {args.ffmpeg_bin if shutil.which(args.ffmpeg_bin) else 'NOT FOUND (video uploads disabled)'}",
          flush=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    if args.max_cpu_resident * 4.3 * 1024 > mem_available_mb() - args.ram_floor_mb:
        print(f"[webui] warning: --max-cpu-resident {args.max_cpu_resident} needs "
              f"~{args.max_cpu_resident * 4.3:.1f}GB but only {mem_available_mb() / 1024:.1f}GB "
              f"is free; the hub will drop models instead of over-committing.", flush=True)

    HUB = ModelHub(args.models_dir, dtype, args.attn_implementation, args.max_resident,
                   max_cpu_resident=args.max_cpu_resident, ram_floor_mb=args.ram_floor_mb)

    def serve() -> int:
        removed = prune_media()
        if removed:
            print(f"[webui] pruned {removed} stale clip(s) from {MEDIA_DIR}", flush=True)
        removed = prune_render_cache()
        if removed:
            print(f"[webui] pruned {removed} cached render(s) from {RENDER_CACHE_DIR}", flush=True)
        if args.preload:
            for capability in list(CAPABILITIES)[: HUB.max_cpu_resident]:
                HUB._get_cpu(capability)

        demo = build_ui()

        auth = None
        credentials = resolve_credentials(args)
        if credentials:
            auth = [credentials]
        reachable_beyond_localhost = bool(args.share) or args.ip not in ("127.0.0.1", "localhost", "::1")
        if reachable_beyond_localhost and auth is None:
            print("[webui] WARNING: serving unauthenticated on "
                  f"{'a public gradio.app link' if args.share else args.ip}:{args.port}. Anyone who can "
                  "reach it can run the model and upload files to this machine. "
                  "Pass --username/--password.", flush=True)

        launch_kwargs: Dict[str, Any] = dict(
            server_name=args.ip, server_port=args.port, share=args.share, auth=auth,
            inbrowser=False, theme=UI_THEME, css=UI_CSS,
        )
        if args.ssl_certfile and args.ssl_keyfile:
            launch_kwargs.update(ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)
            if args.ssl_self_signed:
                # Gradio's own startup check (blocks.launch -> httpx.get(local_api_url +
                # "startup-events", verify=ssl_verify)) dials its own https port. With a
                # self-signed cert that raises CERTIFICATE_VERIFY_FAILED and launch() dies
                # before the UI is reachable — verified 2026-11-02 on gradio 6.17. This
                # only affects Gradio talking to itself; browsers still see the cert.
                launch_kwargs["ssl_verify"] = False

        print(f"[webui] {HUB.status()}", flush=True)
        demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)
        return 0

    if args.force:
        print("[webui] --force: starting without the single-owner lock.", flush=True)
        return serve()
    try:
        with exclusive_owner("webui"):
            return serve()
    except RuntimeError as e:
        print(f"[webui] refusing to start: {e}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
