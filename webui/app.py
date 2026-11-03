# coding=utf-8
"""
Local Qwen3-TTS Gradio WebUI.

Adapted from the official Hugging Face Space (Qwen/Qwen3-TTS app.py) for a single
local GPU. Differences from the Space version:

  * Loads the 1.7B checkpoints (the best quality tier) from a local directory.
  * The Space keeps all 5 models resident on the GPU at once (~20GB VRAM). A 16GB
    Blackwell card (RTX 5070 Ti / 5080) cannot do that, so models live in CPU RAM and an
    LRU "GPU slot" moves at most `--max-resident` models onto the GPU on demand (~1-3s swap).
  * No `spaces` (Zero-GPU) dependency, no model-size dropdown, attn_implementation
    is `sdpa` (flash-attn 2 has no sm_120/Blackwell wheels; the Space uses sdpa too).

Run:  ./run.sh            (or)  .venv/bin/python webui/app.py --ip 127.0.0.1 --port 7860
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import getpass
import hashlib
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
CAPABILITIES = OrderedDict(
    [
        ("custom_voice", ("Qwen3-TTS-12Hz-1.7B-CustomVoice", "custom_voice")),
        ("voice_design", ("Qwen3-TTS-12Hz-1.7B-VoiceDesign", "voice_design")),
        ("voice_clone", ("Qwen3-TTS-12Hz-1.7B-Base", "base")),
    ]
)

# Speaker metadata from the model card, surfaced as dropdown labels. Dialect is a
# property of the speaker, not a selectable language: the checkpoint carries
# `talker_config.spk_is_dialect` (dylan -> beijing_dialect, eric -> sichuan_dialect)
# and upstream deliberately strips the dialects from `get_supported_languages()`
# (modeling_qwen3_tts.py:1831-1834). So you get the Beijing accent by picking Dylan, and
# passing language="beijing_dialect" is rejected — do not re-add them to the list below.
SPEAKER_INFO = {
    "vivian":     ("Bright, slightly edgy young female voice", "Chinese"),
    "serena":     ("Warm, gentle young female voice", "Chinese"),
    "uncle_fu":   ("Seasoned male voice, low and mellow", "Chinese"),
    "dylan":      ("Youthful Beijing male voice, clear and natural", "Chinese (Beijing dialect)"),
    "eric":       ("Lively Chengdu male voice, slightly husky", "Chinese (Sichuan dialect)"),
    "ryan":       ("Dynamic male voice, strong rhythmic drive", "English"),
    "aiden":      ("Sunny American male voice, clear midrange", "English"),
    "ono_anna":   ("Playful Japanese female voice, light and nimble", "Japanese"),
    "sohee":      ("Warm Korean female voice, rich emotion", "Korean"),
}

SPEAKERS = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan",
            "Serena", "Sohee", "Uncle_fu", "Vivian"]


def speaker_choices(models_dir: str = MODELS_DIR) -> List[Tuple[str, str]]:
    """Speaker dropdown entries, driven by the checkpoint's own speaker table.

    Reads `talker_config.spk_id` from the CustomVoice config (cheap JSON, no weights)
    so the UI can never offer a speaker the checkpoint does not know.
    """
    config_path = os.path.join(models_dir, "Qwen3-TTS-12Hz-1.7B-CustomVoice", "config.json")
    names = [s.lower() for s in SPEAKERS]
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as fh:
                names = list(json.load(fh)["talker_config"]["spk_id"].keys())
        except (OSError, ValueError, KeyError):
            pass  # fall back to the hard-coded list if the checkpoint is absent/corrupt
    choices = []
    for name in names:
        description, native = SPEAKER_INFO.get(name, ("", ""))
        label = " — ".join(part for part in (name.replace("_", " ").title(), description, native) if part)
        choices.append((label, name))
    return choices


def default_speaker(models_dir: str = MODELS_DIR) -> str:
    """The speaker the dropdown preselects — always one the checkpoint actually has.

    Hard-coding "ryan" broke silently whenever the checkpoint's own `spk_id` table
    differed: Gradio accepted a value that was not in `choices`, the box came up empty,
    and Generate died with "Speaker is required".
    """
    values = [value for _, value in speaker_choices(models_dir)]
    if "ryan" in values:
        return "ryan"
    return values[0] if values else "ryan"


# (UI label, value passed to the model). Values must match `talker_config.codec_language_id`
# keys *and* upstream's supported-language filter, which excludes the dialect entries —
# see the note above. The hosted Space also omits Italian, which the model does accept.
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
            f"Two at once means two copies of ~13GB of checkpoints in RAM, which on WSL2 "
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


def take_label(index: int, wer: Optional[float], sim: Optional[float], kept: bool) -> str:
    """Label for a candidate's player: which take it was and how it scored.

    The scores are shown so the automatic pick is arguable rather than mysterious — the
    player itself is the override, since you can listen to every take and download any one.
    """
    bits = []
    if wer is not None:
        bits.append(f"WER {wer:.2f}")
    if sim is not None:
        bits.append(f"SIM {sim:.3f}")
    return f"Take #{index}" + (f" — {' · '.join(bits)}" if bits else "") + \
           (" (kept)" if kept else "")


def choose_best_candidate(metrics: List[Dict[str, Optional[float]]]) -> int:
    """Index of the best candidate: content fidelity first, then speaker similarity.

    WER is the tie-breaker that matters — a candidate that says the wrong words is worse
    no matter how close its timbre is. When ASR is disabled every WER is None and ranking
    falls back to similarity alone.
    """
    if not metrics:
        raise ValueError("no candidates to choose from")

    def key(item: Tuple[int, Dict[str, Optional[float]]]):
        index, m = item
        wer = m.get("wer")
        sim = m.get("sim")
        return (float("inf") if wer is None else round(float(wer), 3),
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
    """Gradio output pair: (audio, status line including live VRAM info)."""
    return audio, f"{message}\n{HUB.status()}"


def _post_note(mode: str) -> str:
    mode = (mode or POSTPROCESS_DEFAULT).strip().lower()
    if mode not in POSTPROCESS_MODES:
        mode = POSTPROCESS_DEFAULT
    return "" if mode == "off" else f"\nPost-processing: {POSTPROCESS_LABELS[mode]}"


def _present(wav: np.ndarray, sr: int, budget: int, label: str,
             mode: str) -> Tuple[Tuple[int, np.ndarray], str]:
    """Report on the model's own decode; hand back the post-processed listening copy."""
    report = _report(wav, sr, budget, label)
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
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
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
        wav, in_sr = sf.read(path, dtype="float32", always_2d=False)
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


def auto_pick_window(wav: np.ndarray, sr: int, target: float = REF_TARGET_SECONDS,
                     step: float = 0.25) -> Tuple[float, float]:
    """Pick the loudest contiguous `target`-second window (skips silence / room tone)."""
    duration = len(wav) / float(sr)
    if duration <= target:
        return 0.0, round(duration, 3)
    squared = wav.astype(np.float64) ** 2
    cumulative = np.concatenate([[0.0], np.cumsum(squared)])
    best_energy, best_start = -1.0, 0.0
    window = int(target * sr)
    position = 0.0
    while position + target <= duration + 1e-6:
        i0 = int(position * sr)
        i1 = min(len(wav), i0 + window)
        energy = (cumulative[i1] - cumulative[i0]) / max(1, i1 - i0)
        if energy > best_energy:
            best_energy, best_start = energy, position
        position += step
    return round(float(best_start), 3), round(float(best_start + target), 3)


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

    def transcribe(self, wav: np.ndarray, sr: int, language: Optional[str] = None) -> str:
        if not wav.size:
            raise MediaError("Nothing to transcribe — pick a non-empty clip.")
        code = ASR_LANGUAGE_CODES.get((language or "").strip().lower())
        with self.lock:
            self.load()
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
            raw_length = min(len(wav), max_samples)
            padding_mask = np.zeros(max_samples, dtype=np.int64)
            padding_mask[:raw_length] = 1
            with torch.inference_mode():
                ids = self.model.generate(
                    features,
                    attention_mask=torch.from_numpy(padding_mask).unsqueeze(0).to(self.device),
                    language=code, task="transcribe",
                    num_beams=4, no_repeat_ngram_size=4)
            text = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        if not text:
            raise MediaError("Transcription came back empty — the clip may be music or silence.")
        return text


ASR: Optional[Transcriber] = None

# (Uploaded clips are kept on disk under MEDIA_DIR — see the path-containment block
# above. gr.State travels to the browser as JSON, so only the clip's path/duration live
# in state; a numpy array there would not survive the round-trip.)


# ============================================================================
# Gradio callbacks
# ============================================================================
def generate_custom_voice(text, language, speaker, instruct, temperature, top_p, top_k, rep_penalty,
                          postproc=POSTPROCESS_DEFAULT, progress=gr.Progress(track_tqdm=True)):
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."
    try:
        budget = _budget(text)
        with HUB.lock:
            tts = HUB.acquire("custom_voice")
            wavs, sr = tts.generate_custom_voice(
                text=text.strip(),
                language=language,
                speaker=speaker.lower().replace(" ", "_"),
                instruct=(instruct or "").strip() or None,
                non_streaming_mode=True,
                max_new_tokens=budget,
                **_sampling(temperature, top_p, top_k, rep_penalty),
            )
        return _ok(*_present(wavs[0], sr, budget, "CustomVoice", postproc))
    except Exception as e:  # noqa: BLE001 - surface errors in the UI instead of killing the server
        return None, f"Error: {type(e).__name__}: {e}"


def generate_voice_design(text, language, voice_description, temperature, top_p, top_k, rep_penalty,
                          postproc=POSTPROCESS_DEFAULT, progress=gr.Progress(track_tqdm=True)):
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."
    try:
        budget = _budget(text)
        with HUB.lock:
            tts = HUB.acquire("voice_design")
            wavs, sr = tts.generate_voice_design(
                text=text.strip(),
                language=language,
                instruct=voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=budget,
                **_sampling(temperature, top_p, top_k, rep_penalty),
            )
        return _ok(*_present(wavs[0], sr, budget, "VoiceDesign", postproc))
    except Exception as e:  # noqa: BLE001
        return None, f"Error: {type(e).__name__}: {e}"


# Reusable clone prompts, keyed by a hash of the uploaded reference audio, so the
# reference encoder is not recomputed for every line (big speedup for repeated takes).
_CLONE_PROMPTS: Dict[str, Any] = {}


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


def save_voice(name: str, wav: np.ndarray, sr: int, text: str, instruct: str, language: str) -> Dict[str, Any]:
    """Persist a generated clip as a named voice (wav + sidecar metadata)."""
    os.makedirs(VOICES_DIR, exist_ok=True)
    slug = _slug(name)
    wav_path, meta_path = _voice_paths(slug)
    record = {
        "name": name.strip(),
        "slug": slug,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": language,
        "instruct": instruct,
        "text": text,
        "sample_rate": int(sr),
        "seconds": round(len(wav) / float(sr), 2),
    }
    sf.write(wav_path, wav, sr)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    return record


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
    return [(f"{r['name']} ({r.get('seconds', '?')}s)", r["slug"]) for r in list_voices()]


def save_designed_voice(name: str, audio: Any, text: str, voice_description: str,
                        language: str) -> Tuple[Any, str, Any]:
    """Save the clip currently in the Voice Design audio box as a reusable clone voice.

    Clip, transcript and description all arrive as inputs rather than from a module-level
    "last generation" cache: a process-global there meant that with two people on the LAN,
    one person's Save button persisted the other person's clip.
    """
    if not (name or "").strip():
        return gr.update(), "Error: give the voice a name first.", gr.update()
    if not (text or "").strip():
        return gr.update(), "Error: the text the clip speaks is empty — it is what ICL cloning needs.", gr.update()
    try:
        parsed = _audio_to_tuple(audio)
        if parsed is None:
            return gr.update(), "Nothing to save yet — generate something in the Voice Design tab first.", gr.update()
        wav, sr = parsed
        record = save_voice(
            name=name,
            wav=wav,
            sr=sr,
            text=text.strip(),
            instruct=(voice_description or "").strip(),
            language=language,
        )
    except (MediaError, OSError, ValueError) as e:
        return gr.update(), f"Error: {type(e).__name__}: {e}", gr.update()
    return (
        gr.update(),
        f"Saved ‘{record['name']}’ ({record['seconds']}s). Pick it under Voice Clone → Saved voices.",
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
    try:
        parsed = _audio_to_tuple(audio)
        if parsed is None:
            return (gr.update(),
                    "Nothing to save — hit ‘Extract audio & show waveform’ first, or upload a "
                    "reference into the clip box.",
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
        f"It is in the Saved voices list below.",
        gr.update(choices=voice_choices(), value=record["slug"]),
    )


def load_saved_voice(slug: str) -> Tuple[Any, Any, str]:
    """Fill the Voice Clone reference fields from a saved voice."""
    if not slug:
        return None, gr.update(), "Error: choose a saved voice first."
    wav_path, meta_path = _voice_paths(_slug(slug))
    if not os.path.exists(wav_path):
        return None, gr.update(), f"Error: saved voice ‘{slug}’ is missing its audio."
    wav, sr = sf.read(wav_path, dtype="float32")
    text = ""
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            text = json.load(fh).get("text", "")
    return (int(sr), wav), text, f"Loaded ‘{slug}’ — reference transcript filled in. Hit ‘Clone & generate’."


def delete_saved_voice(slug: str) -> Tuple[Any, str, Any]:
    if not slug:
        return gr.update(), "Error: choose a saved voice to delete.", gr.update()
    wav_path, meta_path = _voice_paths(_slug(slug))
    removed = []
    for path in (wav_path, meta_path):
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    msg = f"Deleted {', '.join(removed)}" if removed else f"Nothing saved under ‘{slug}’."
    return gr.update(choices=voice_choices(), value=None), msg, gr.update(value=None)


def refresh_voices() -> Any:
    return gr.update(choices=voice_choices())



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


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only,
                         temperature, top_p, top_k, rep_penalty,
                         postproc=POSTPROCESS_DEFAULT,
                         candidates=1, seed="",
                         progress=gr.Progress(track_tqdm=True)):
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."
    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."
    if not use_xvector_only and not (ref_text or "").strip():
        return None, "Error: Reference text is required when 'x-vector only' is off."
    try:
        n = max(1, min(int(candidates or 1), MAX_CANDIDATES))
    except (TypeError, ValueError):
        return None, "Error: Candidates must be a whole number."
    seed_text = str(seed or "").strip()
    if seed_text:
        if not re.fullmatch(r"-?\d+", seed_text):
            return None, "Error: Seed must be an integer (or blank for random)."
        base_seed = int(seed_text) % (2 ** 31)
    else:
        base_seed = int(time.time() * 1000) % (2 ** 31)

    ref_seconds = len(audio_tuple[0]) / float(audio_tuple[1])
    fit = transcript_fit(ref_seconds, ref_text or "", bool(use_xvector_only))
    asr_language = None if (not language or language == "Auto") else str(language)
    try:
        budget = _budget(target_text)
        with HUB.lock:
            tts = HUB.acquire("voice_clone")
            key = _clone_cache_key(audio_tuple, ref_text, use_xvector_only)
            prompt = _CLONE_PROMPTS.get(key)
            if prompt is None:
                prompt = tts.create_voice_clone_prompt(
                    ref_audio=audio_tuple,
                    ref_text=(ref_text or "").strip() or None,
                    x_vector_only_mode=bool(use_xvector_only),
                )
                _CLONE_PROMPTS.clear()
                _CLONE_PROMPTS[key] = prompt

            ref_embedding = None
            try:
                ref_embedding = tts.model.extract_speaker_embedding(
                    audio=_to_24k(audio_tuple[0], audio_tuple[1]), sr=24000)
                ref_embedding = ref_embedding.float().cpu().numpy()
            except Exception:  # noqa: BLE001 - similarity is a bonus, never a blocker
                ref_embedding = None

            results: List[Dict[str, Any]] = []
            for i in range(n):
                if n > 1:
                    progress((i + 1) / (n + 2.0), desc=f"Candidate {i + 1} of {n}")
                torch.manual_seed(base_seed + i)
                wavs, sr = tts.generate_voice_clone(
                    text=target_text.strip(),
                    language=language,
                    voice_clone_prompt=prompt,
                    max_new_tokens=budget,
                    **_sampling(temperature, top_p, top_k, rep_penalty),
                )
                wav = wavs[0]
                sim = wer = None
                if ref_embedding is not None:
                    try:
                        cand = tts.model.extract_speaker_embedding(
                            audio=_to_24k(wav, sr), sr=24000).float().cpu().numpy()
                        sim = _cosine(ref_embedding, cand)
                    except Exception:  # noqa: BLE001
                        sim = None
                if n > 1 and ASR is not None and ASR.enabled:
                    try:
                        wer = word_error_rate(ASR.transcribe(wav, sr, asr_language), target_text)
                    except Exception:  # noqa: BLE001 - scoring must not sink the generation
                        wer = None
                results.append({"index": i + 1, "wav": wav, "sr": int(sr), "sim": sim, "wer": wer})

        best = choose_best_candidate(results)
        chosen = results[best]
        report = _report(chosen["wav"], chosen["sr"], budget, "VoiceClone")
        notes: List[str] = []
        if n > 1:
            def fmt(m: Dict[str, Any]) -> str:
                bits = []
                if m["wer"] is not None:
                    bits.append(f"WER {m['wer']:.2f}")
                if m["sim"] is not None:
                    bits.append(f"SIM {m['sim']:.3f}")
                return f"#{m['index']} " + " ".join(bits)
            scored_by = ("Whisper WER + speaker similarity"
                         if any(m["wer"] is not None for m in results)
                         else "speaker similarity (transcription off)")
            notes.append(f"Best of {n}: kept #{results[best]['index']} ({scored_by}) — "
                         + ", ".join(fmt(m) for m in results))
        if fit:
            notes.append("WARNING: " + fit)
        status = report + (("\n" + "\n".join(notes)) if notes else "")
        out_wav, out_sr = postprocess_output(chosen["wav"], chosen["sr"], postproc)
        # Hand back every take, not just the winner: the auto-pick is a first opinion, and
        # the only honest way to let someone overrule it is to let them hear the alternatives.
        takes: List[Any] = []
        for item in results:
            if n > 1:
                take_wav, take_sr = postprocess_output(item["wav"], item["sr"], postproc)
                takes.append(gr.update(value=(take_sr, take_wav), visible=True,
                                       label=take_label(item["index"], item["wer"], item["sim"],
                                                        item["index"] == results[best]["index"])))
            else:
                takes.append(gr.update(value=None, visible=False))
        while len(takes) < MAX_CANDIDATES:
            takes.append(gr.update(value=None, visible=False))
        return _ok((out_sr, out_wav), status + _post_note(postproc)) + tuple(takes) + (
            gr.update(visible=n > 1),)
    except Exception as e:  # noqa: BLE001
        return None, f"Error: {type(e).__name__}: {e}"


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
        wave, audio, info, start, end = _clip_outputs(state, None, None)
        status = f"Loaded {state['name']} ({kind}): {len(wav) / sr:.1f}s @ {sr}Hz"
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


def transcribe_reference(state: Any, start: Any, end: Any, language: Any) -> Tuple[Any, str]:
    """Whisper the *selected* window into the editable transcript box."""
    if ASR is None:
        return gr.update(), "Error: transcription is disabled on this server (--asr-model)."
    try:
        wav, sr = _state_audio(state)
        start, end = trim_window(len(wav) / sr, start, end)
        clip = slice_audio(wav, sr, start, end)
        t0 = time.time()
        text = ASR.transcribe(clip, sr, None if not language or language == "Auto" else str(language))
        return gr.update(value=text), (
            f"Transcribed {end - start:.1f}s in {time.time() - t0:.1f}s — "
            f"check and edit the text before cloning.")
    except (MediaError, OSError, ValueError, RuntimeError) as e:
        return gr.update(), f"Error: {type(e).__name__}: {e}"


def clip_fit_status(ref_text: Any, start: Any, end: Any, x_vector_only: Any = False) -> Any:
    """Live transcript/clip sanity line for the Clone tab; it never blocks a generation."""
    try:
        seconds = max(0.0, float(end or 0.0) - float(start or 0.0))
    except (TypeError, ValueError):
        return gr.update()
    return gr.update(value=transcript_fit(seconds, ref_text or "", bool(x_vector_only)))


def refresh_status():
    return HUB.status()


# ============================================================================
# UI
# ============================================================================
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS") as demo:
        gr.Markdown(
            """# Qwen3-TTS — local (1.7B, 16GB Blackwell)
**TTS (CustomVoice)**: 9 preset speakers + natural-language style instructions ·
**Voice Design**: invent a voice from a text description ·
**Voice Clone (Base)**: clone any voice from a ~3s reference clip.
Models stay in CPU RAM and are swapped onto the GPU on demand; the first request
after a model switch pays a few seconds of load time.
"""
        )
        status_bar = gr.Textbox(value=HUB.status(), label="Runtime", interactive=False, lines=1)
        refresh = gr.Button("Refresh runtime status", size="sm")
        refresh.click(refresh_status, outputs=[status_bar])

        with gr.Accordion("Sampling settings (advanced)", open=False):
            gr.Markdown(
                "Defaults come from the checkpoint's `generation_config.json`. Lower "
                "temperature / raise repetition penalty if a generation rambles or never stops. "
                "Output length is capped at `min(2048, 4 × text length)` tokens."
            )
            with gr.Row():
                s_temperature = gr.Slider(0.1, 2.0, value=SAMPLING_DEFAULTS["temperature"], step=0.05, label="Temperature")
                s_top_p = gr.Slider(0.1, 1.0, value=SAMPLING_DEFAULTS["top_p"], step=0.05, label="Top-p")
                s_top_k = gr.Slider(1, 200, value=SAMPLING_DEFAULTS["top_k"], step=1, label="Top-k")
                s_rep = gr.Slider(1.0, 1.5, value=SAMPLING_DEFAULTS["repetition_penalty"], step=0.01, label="Repetition penalty")
                s_postproc = gr.Dropdown(
                    label="Output post-processing",
                    choices=[(POSTPROCESS_LABELS[m], m) for m in POSTPROCESS_MODES],
                    value=POSTPROCESS_DEFAULT,
                    info="The vocoder decodes at 24kHz. 'Warm' resamples properly, cuts the "
                         "imaging near 12kHz that reads as 'digital', and adds a little air. "
                         "Measured likeness-neutral.")
        sampling = [s_temperature, s_top_p, s_top_k, s_rep]
        output_options = [s_postproc]

        with gr.Tabs():
            with gr.Tab("TTS (CustomVoice)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        cv_text = gr.Textbox(
                            label="Text to synthesize", lines=4,
                            value="Hello! Welcome to Qwen3-TTS. This runs entirely on my own GPU.",
                            placeholder="Enter the text you want to convert to speech...",
                        )
                        with gr.Row():
                            cv_language = gr.Dropdown(label="Language", choices=LANGUAGE_CHOICES, value="English")
                            cv_speaker = gr.Dropdown(label="Speaker", choices=speaker_choices(), value=default_speaker())
                        cv_instruct = gr.Textbox(
                            label="Style instruction (optional)", lines=2,
                            placeholder="e.g. Speak in a cheerful, energetic tone / 用愤怒的语气说",
                        )
                        cv_btn = gr.Button("Generate speech", variant="primary")
                    with gr.Column(scale=2):
                        cv_audio = gr.Audio(label="Generated audio", type="numpy")
                        cv_status = gr.Textbox(label="Status", lines=2, interactive=False)
                cv_btn.click(generate_custom_voice,
                             inputs=[cv_text, cv_language, cv_speaker, cv_instruct] + sampling + output_options,
                             outputs=[cv_audio, cv_status])

            with gr.Tab("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=2):
                        vd_text = gr.Textbox(
                            label="Text to synthesize", lines=4,
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                        )
                        vd_language = gr.Dropdown(label="Language", choices=LANGUAGE_CHOICES, value="Auto")
                        vd_instruct = gr.Textbox(
                            label="Voice description", lines=3,
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
                            placeholder="Describe timbre, age, gender, emotion, accent, speaking rate...",
                        )
                        vd_btn = gr.Button("Generate voice", variant="primary")
                        with gr.Row():
                            vd_voice_name = gr.Textbox(
                                label="Save as voice (name)", scale=3,
                                placeholder="e.g. sleepy sea captain",
                                info="Keeps the clip as a reusable clone speaker (voices/ on disk).",
                            )
                            vd_save_btn = gr.Button("Save as clone voice", size="sm", scale=1)
                        vd_save_status = gr.Textbox(label="Voice library", lines=1, interactive=False)
                    with gr.Column(scale=2):
                        vd_audio = gr.Audio(label="Generated voice", type="filepath")
                        vd_status = gr.Textbox(label="Status", lines=2, interactive=False)
                vd_btn.click(generate_voice_design,
                             inputs=[vd_text, vd_language, vd_instruct] + sampling + output_options,
                             outputs=[vd_audio, vd_status])

            with gr.Tab("Voice Clone (Base)"):
                gr.Markdown(
                    "Clone a voice from a short clip, then make it say anything — three steps, "
                    "top to bottom. **Quality comes from the reference and its transcript**, not "
                    "from the sliders; see the README's quality section. A clean 3–10s stretch of "
                    "one speaker, with no music or reverb, clones best.")

                gr.Markdown("#### 1 · The reference — the clip the voice is taken from")
                cl_media_state = gr.State(value=None)
                cl_upload = gr.File(label="Upload video or audio — video works, only its audio track is used",
                                    file_types=MEDIA_SUFFIXES, type="filepath")
                with gr.Row():
                    cl_import_btn = gr.Button("Extract audio & show waveform", variant="primary")
                    cl_auto_btn = gr.Button("Auto-pick best clip")
                cl_media_status = gr.Textbox(label="Media", lines=1, interactive=False)
                cl_wave = gr.Image(label="Reference waveform — shaded region is what gets cloned",
                                   type="pil", height=180, interactive=False)
                with gr.Row():
                    cl_start = gr.Slider(0, 60, value=0, step=0.05, label="Start (s)")
                    cl_end = gr.Slider(0, 60, value=0, step=0.05, label="End (s)")
                cl_clip_info = gr.Textbox(label="Selected clip", lines=1, interactive=False)

                with gr.Row():
                    with gr.Column(scale=2):
                        cl_ref_audio = gr.Audio(label="Reference clip (the trimmed selection lands here)",
                                                type="filepath", sources=["upload", "microphone"])
                    with gr.Column(scale=2):
                        cl_asr_language = gr.Dropdown(
                            label="Transcript language",
                            choices=[("Auto-detect", "Auto")] + [(k.title(), k) for k in ASR_LANGUAGE_CODES],
                            value="Auto")
                        cl_transcribe_btn = gr.Button("Transcribe selected clip", variant="primary")
                cl_ref_text = gr.Textbox(
                    label="The reference clip says… (transcript — must match what you hear)", lines=3,
                    placeholder="Transcribe the selection above, then fix anything Whisper got wrong. "
                                "This is what the clip says, not what you want it to say.")
                cl_xvector = gr.Checkbox(
                    label="x-vector only (no transcript needed, lower similarity)", value=False)
                cl_fit = gr.Textbox(label="Fit check", lines=1, interactive=False)

                gr.Markdown("#### 2 · What it should say")
                with gr.Row():
                    with gr.Column(scale=2):
                        cl_text = gr.Textbox(
                            label="Say this in the cloned voice", lines=4,
                            placeholder="New words for the cloned voice to speak…",
                            info="The new line to synthesise — the box above is what the reference "
                                 "says; this is what you want to hear.")
                        cl_language = gr.Dropdown(label="Language", choices=LANGUAGE_CHOICES, value="Auto")
                        with gr.Row():
                            cl_candidates = gr.Slider(1, MAX_CANDIDATES, value=1, step=1,
                                                      label="Candidates (best of N)",
                                                      info="Each extra take costs ~7s. Helps when the "
                                                           "model flubs a take; scored by Whisper WER then "
                                                           "speaker similarity.")
                            cl_seed = gr.Textbox(label="Seed (blank = random)", value="", scale=1,
                                                 info="Same seed + same inputs reproduces the same audio.")
                        cl_btn = gr.Button("Clone & generate", variant="primary")
                    with gr.Column(scale=2):
                        cl_audio = gr.Audio(label="Generated audio (the auto-pick)", type="numpy")
                        cl_status = gr.Textbox(label="Status", lines=2, interactive=False)
                with gr.Row():
                    cl_take_1 = gr.Audio(label="Take #1", type="numpy", visible=False)
                    cl_take_2 = gr.Audio(label="Take #2", type="numpy", visible=False)
                    cl_take_3 = gr.Audio(label="Take #3", type="numpy", visible=False)
                    cl_take_4 = gr.Audio(label="Take #4", type="numpy", visible=False)
                cl_takes_note = gr.Markdown(
                    visible=False,
                    value="*Candidates (best of N)*: **Generated audio** above is the auto-pick "
                          "(best Whisper WER, then speaker similarity). Every take is left below so "
                          "you can listen and overrule it — each has its own download icon, so the "
                          "one you keep does not have to be the one the scorer preferred.")

                gr.Markdown("#### 3 · Keep it — the voice library")
                gr.Markdown(
                    "Saving keeps the **reference clip + transcript** so you do not have to "
                    "re-upload it next time; **Load** puts one back into step 1. The audio you "
                    "generated is kept with the download icon on its player, not here.")
                with gr.Row():
                    cl_voice_name = gr.Textbox(
                        label="Save this reference as a voice (name)", scale=3,
                        placeholder="e.g. phone voice memo",
                        info="Writes voices/<name>.wav + .json on this machine.")
                    cl_save_btn = gr.Button("Save to voice library", size="sm", scale=1)
                cl_save_status = gr.Textbox(label="Save status", lines=1, interactive=False)
                with gr.Row():
                    cl_voice = gr.Dropdown(label="Saved voices (designed or saved from a clip)",
                                           choices=voice_choices(),
                                           value=None, scale=4, allow_custom_value=False,
                                           info="Load fills in the reference clip and its transcript.")
                    cl_load_btn = gr.Button("Load", size="sm", scale=1)
                    cl_delete_btn = gr.Button("Delete", size="sm", scale=1, variant="stop")
                    cl_refresh_btn = gr.Button("Refresh", size="sm", scale=1)
                cl_voice_status = gr.Textbox(label="Voice library", lines=1, interactive=False)

                cl_btn.click(generate_voice_clone,
                             inputs=[cl_ref_audio, cl_ref_text, cl_text, cl_language, cl_xvector] + sampling
                                    + output_options + [cl_candidates, cl_seed],
                             outputs=[cl_audio, cl_status, cl_take_1, cl_take_2, cl_take_3, cl_take_4,
                                      cl_takes_note])

                # Transcript/clip fit is a live, non-blocking check: a transcript that does not
                # match the selection is the worst clone failure mode we measured, and it is
                # invisible until the model has rambled for 17 seconds.
                fit_inputs = [cl_ref_text, cl_start, cl_end, cl_xvector]
                cl_ref_text.change(clip_fit_status, inputs=fit_inputs, outputs=[cl_fit])
                cl_xvector.change(clip_fit_status, inputs=fit_inputs, outputs=[cl_fit])
                cl_start.release(clip_fit_status, inputs=fit_inputs, outputs=[cl_fit])
                cl_end.release(clip_fit_status, inputs=fit_inputs, outputs=[cl_fit])

                cl_import_btn.click(import_media, inputs=[cl_upload],
                                    outputs=[cl_media_state, cl_media_status, cl_wave,
                                             cl_start, cl_end, cl_ref_audio, cl_clip_info]).then(
                    clip_fit_status, inputs=fit_inputs, outputs=[cl_fit])
                cl_start.release(update_clip, inputs=[cl_media_state, cl_start, cl_end],
                                 outputs=[cl_wave, cl_ref_audio, cl_clip_info, cl_start, cl_end])
                cl_end.release(update_clip, inputs=[cl_media_state, cl_start, cl_end],
                               outputs=[cl_wave, cl_ref_audio, cl_clip_info, cl_start, cl_end])
                cl_auto_btn.click(auto_pick_clip, inputs=[cl_media_state],
                                  outputs=[cl_wave, cl_ref_audio, cl_clip_info, cl_start, cl_end]).then(
                    clip_fit_status, inputs=fit_inputs, outputs=[cl_fit])
                cl_transcribe_btn.click(transcribe_reference,
                                        inputs=[cl_media_state, cl_start, cl_end, cl_asr_language],
                                        outputs=[cl_ref_text, cl_media_status]).then(
                    clip_fit_status, inputs=fit_inputs, outputs=[cl_fit])

                vd_save_btn.click(save_designed_voice,
                                  inputs=[vd_voice_name, vd_audio, vd_text, vd_instruct, vd_language],
                                  outputs=[vd_voice_name, vd_save_status, cl_voice])
                cl_save_btn.click(save_reference_voice,
                                  inputs=[cl_voice_name, cl_ref_audio, cl_ref_text, cl_language],
                                  outputs=[cl_voice_name, cl_save_status, cl_voice])
                cl_load_btn.click(load_saved_voice, inputs=[cl_voice],
                                  outputs=[cl_ref_audio, cl_ref_text, cl_voice_status])
                cl_delete_btn.click(delete_saved_voice, inputs=[cl_voice],
                                    outputs=[cl_voice, cl_voice_status, cl_ref_audio])
                cl_refresh_btn.click(refresh_voices, outputs=[cl_voice])

        gr.Markdown(
            "---\nPowered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba Qwen). "
            "Long texts are capped at `max_new_tokens=2048`; split very long scripts into chunks."
        )
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
        secret = prompt(f"Qwen3-TTS password for user '{username}' (input hidden): ")

    if not secret:
        print("[webui] warning: no password could be resolved; starting UNAUTHENTICATED.", flush=True)
        return None
    return (username, secret)


def main() -> int:
    global HUB, DEVICE

    parser = argparse.ArgumentParser(description="Local Qwen3-TTS Gradio demo (1.7B models).")
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
            inbrowser=False, css=".gradio-container {max-width: none !important;}",
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
