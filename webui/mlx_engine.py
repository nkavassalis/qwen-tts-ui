"""The MLX backend: the same app, on Apple Silicon, without CUDA.

Why this file exists as a shim rather than a fork. Every model in this app is loaded through
`ModelHub._load()` and promoted through `ModelHub._move()` — that is invariant 2, and it is the
only place that touches a model object. So the Apple Silicon path is one more implementation of
the same small surface, selected by platform, and the CUDA path is left exactly as it was. The
alternative — branching inside 3,800 lines of UI callbacks — would have put a second way to load
a model into the file, which is precisely what the invariant forbids.

What the surface has to be, taken from the call sites rather than invented:

    generate_voice_clone(text, language, ref_audio, ref_text, x_vector_only_mode,
                         max_new_tokens, **sampling) -> (list_of_wav, sample_rate)
    generate_voice_design(text, instruct, language, max_new_tokens, **sampling)
    create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode) -> list
    speaker_embedding(wav, sr) -> 1-D float array

`mlx_audio`'s own API differs from `Qwen3TTSModel` in four ways this file absorbs, and each is a
place where a naive port would have silently produced something wrong rather than failing:

  * `generate()` is a **generator** of `GenerationResult`, not a function returning a tuple.
    Dropping all but the last yielded segment would quietly truncate multi-segment text.
  * It wants `lang_code` ("en", "ja"), not the display name the UI carries ("English").
  * It wants `max_tokens`, not `max_new_tokens` — a name mismatch here means the runaway-EOS
    guard (`_budget()`, cap 2048) silently never applies, which is the 163-seconds-for-one-
    sentence bug all over again.
  * Reference audio is a path or an `mx.array`, not the `(numpy, sr)` tuple the app passes
    around, so it is written to a short-lived WAV and handed over by name.

Confidence is *not* a gap. The take-scoring confidence comes from Whisper's teacher-forced
forward pass in `Transcriber`, not from the TTS model's logits, so best-of-N keeps WER,
confidence, defect penalties and the speaker-similarity tie-break on the Mac too.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# mlx-audio's language codes; the UI carries display names ("English"), the model wants "en".
_LANG_CODES: Dict[str, str] = {
    "auto": "auto", "chinese": "zh", "english": "en", "german": "de", "italian": "it",
    "portuguese": "pt", "spanish": "es", "japanese": "ja", "korean": "ko", "french": "fr",
    "russian": "ru", "thai": "th", "vietnamese": "vi", "indonesian": "id", "arabic": "ar",
}


REF_SAMPLE_RATE = 24000
# One temp WAV per distinct reference, evicted oldest-first. Bounded because these are people's
# voices sitting in the OS temp dir, and macOS reaps that directory on its own schedule — which
# is also why the prompt item keeps the array: a reaped file can be rewritten rather than
# silently turning into whatever the model does with a missing reference.
REF_CACHE_SIZE = 8


class MLXError(RuntimeError):
    """Raised when the MLX backend cannot do something; surfaced as a status line."""


def _to_numpy(audio: Any) -> np.ndarray:
    """mlx array / numpy / list -> 1-D float32, without importing mlx at module scope."""
    for attr in ("numpy",):
        if hasattr(audio, attr):
            audio = getattr(audio, attr)
            break
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _ref_key(wav: np.ndarray, sr: int) -> str:
    """Content hash for a reference clip.

    The cache used to be one slot keyed on the literal string "path", so the second voice you
    cloned from reused the *first* voice's temp file — measured: prompt #2's file was byte-
    identical to fox's and scored SIM 1.000 against fox, 0.924 against the voice actually being
    cloned. Every metric stayed green because the words were still perfect; only the speaker was
    wrong. Same lesson the app's own clone-prompt cache already learned, including why `(length,
    sum)` is not enough — it collided.
    """
    pcm = np.ascontiguousarray(wav, dtype=np.float32).tobytes()
    return hashlib.sha256(f"{int(sr)}|".encode() + pcm).hexdigest()[:32]


def _write_reference(wav: np.ndarray, sr: int, keep: Dict[str, str]) -> str:
    """The model takes reference audio by path; the app has it as (numpy, sr)."""
    import soundfile as sf

    key = _ref_key(wav, sr)
    path = keep.get(key)
    if path and os.path.exists(path):
        return path
    fd, path = tempfile.mkstemp(prefix="qwen-tts-mlx-ref-", suffix=".wav")
    os.close(fd)
    sf.write(path, np.asarray(wav, dtype=np.float32).reshape(-1), int(sr))
    while len(keep) >= REF_CACHE_SIZE:
        for oldest in list(keep):
            stale = keep.pop(oldest)
            if os.path.exists(stale):
                try:
                    os.unlink(stale)
                except OSError:
                    pass
            break
    keep[key] = path
    return path


class _ModelShim:
    """`tts.model.extract_speaker_embedding(...)` is how the app reaches the encoder today;
    MLX's array type has no `.float().cpu().numpy()`, so the shim returns plain numpy and the
    call sites go through `speaker_embedding()` instead of the torch idiom."""

    def __init__(self, engine: "MLXModel") -> None:
        self._engine = engine

    def extract_speaker_embedding(self, audio: Any = None, sr: int = 24000) -> np.ndarray:
        return self._engine.speaker_embedding(audio, sr)


class MLXModel:
    """One loaded Qwen3-TTS checkpoint under MLX, shaped like `Qwen3TTSModel`."""

    # Read by ModelHub._move() and speaker_embedding() to pick a path. An explicit marker
    # rather than isinstance or duck-typing, so a torch model can never be mistaken for one
    # and quietly skip the move that keeps the vocoder on the GPU.
    _mlx_backend = True

    _load_lock = threading.Lock()

    def __init__(self, model: Any, capability: str, repo: str) -> None:
        self._model = model
        self.capability = capability
        self.repo = repo
        self.model = _ModelShim(self)
        # The torch path sets tts.device; nothing here moves tensors across devices, but the
        # attribute exists so shared code that reads it does not have to special-case us.
        self.device = "mlx"
        self._ref_cache: Dict[str, str] = {}

    # -- loading -------------------------------------------------------------

    @classmethod
    def load(cls, repo: str, capability: str, models_dir: Optional[str] = None) -> "MLXModel":
        """Import mlx_audio lazily: the CPU test tier must import app.py on a box with no MLX.

        A local `models-mlx/<name>` directory wins over the Hub repo id, so ./setup.sh's
        download is what actually runs — otherwise a fresh Mac silently re-fetches the weights
        into the Hugging Face cache on first use and the disk footprint doubles.
        """
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as e:  # pragma: no cover - depends on the platform
            raise MLXError(
                f"mlx-audio is not installed ({e}). Run setup.sh on this Mac, or start the "
                "server with --device cpu to use the PyTorch path."
            ) from e
        source = repo
        if models_dir:
            local = os.path.join(models_dir + "-mlx", repo.rsplit("/", 1)[-1])
            if os.path.isdir(local) and any(f.endswith(".safetensors")
                                            for _, _, fs in os.walk(local) for f in fs):
                source = local
        with cls._load_lock:
            print(f"[hub] loading {source} -> MLX ...", flush=True)
            model = load_model(source, lazy=False)
        print(f"[hub] {capability} ready on MLX ({getattr(model, 'sample_rate', '?')}Hz)",
              flush=True)
        return cls(model, capability, repo)

    # -- shared plumbing -----------------------------------------------------

    def _sampling(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Map our sampling names onto theirs. Names that differ are the dangerous ones: an
        unmapped `max_new_tokens` would leave the runaway guard off."""
        out: Dict[str, Any] = {}
        for key in ("temperature", "top_p", "top_k", "repetition_penalty", "speed"):
            if kwargs.get(key) is not None:
                out[key] = kwargs[key]
        if kwargs.get("max_new_tokens") is not None:
            out["max_tokens"] = int(kwargs["max_new_tokens"])
        return out

    def _run(self, **kwargs: Any) -> Tuple[List[np.ndarray], int]:
        results = list(self._model.generate(**kwargs))
        if not results:
            raise MLXError("the model returned no audio")
        wavs = [_to_numpy(r.audio) for r in results]
        rate = int(getattr(results[0], "sample_rate", 0) or self._model.sample_rate)
        return wavs, rate

    def _reference(self, ref_audio: Any) -> Tuple[Optional[str], Optional[str]]:
        """Kept for the measurement scripts; `_reference_from` is what the app path uses."""
        wav, sr = self._as_wave(ref_audio)
        return (_write_reference(wav, sr, self._ref_cache) if wav is not None else None), None

    def create_voice_clone_prompt(self, ref_audio: Any = None, ref_text: Optional[str] = None,
                                  x_vector_only_mode: bool = False) -> List[Any]:
        """The app caches whatever this returns and hands it straight back as
        `voice_clone_prompt` — which is the *only* reference the clone path ever sends. So this
        is not a hint, it is the payload: it has to carry the audio, the transcript and the mode.
        """
        wav, sr = self._as_wave(ref_audio)
        path = _write_reference(wav, sr, self._ref_cache) if wav is not None else None
        # The array rides along deliberately: macOS reaps /var/folders on its own schedule, and a
        # prompt whose file has vanished must be able to rewrite it instead of sending mlx-audio a
        # path that no longer exists.
        return [{"mlx": True, "ref_path": path, "ref_text": ref_text,
                 "x_vector_only": bool(x_vector_only_mode),
                 "wav": None if wav is None else np.ascontiguousarray(wav, dtype=np.float32),
                 "sr": sr}]

    @staticmethod
    def _as_wave(ref_audio: Any) -> Tuple[Optional[np.ndarray], int]:
        """Accept (wav, sr), a path, or None; return (1-D float32, sample_rate)."""
        if ref_audio is None:
            return None, REF_SAMPLE_RATE
        if isinstance(ref_audio, (tuple, list)) and len(ref_audio) == 2:
            wav, sr = ref_audio
            return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)
        import soundfile as sf

        data, sr = sf.read(str(ref_audio), dtype="float32")
        return np.asarray(data, dtype=np.float32).reshape(-1), int(sr)

    @staticmethod
    def _prompt_item(prompt: Any) -> Optional[Dict[str, Any]]:
        """Unwrap what `create_voice_clone_prompt` produced, in whatever shape it came back."""
        if prompt is None:
            return None
        item = prompt[0] if isinstance(prompt, (list, tuple)) and len(prompt) else prompt
        if isinstance(item, dict) and item.get("mlx"):
            return item
        return None

    def _reference_from(self, ref_audio: Any, ref_text: Optional[str],
                        kwargs: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Resolve the reference from either argument style.

        The app calls `generate_voice_clone(text=, language=, voice_clone_prompt=)` and nothing
        else — no `ref_audio`, no `ref_text`. A shim that only understands the explicit form
        accepts the call, drops the prompt, and mlx-audio then takes its *non-ICL* branch:
        a fluent, correctly-worded rendering in the checkpoint's default speaker. That is how a
        clone of a woman's voice came out as "a generic male voice" while every quality metric
        stayed green — WER and confidence measure the words, and the words were perfect.

        So a clone with no reachable reference is refused here rather than generated. A wrong
        voice that passes every check is worse than an error nobody can miss.
        """
        item = self._prompt_item(kwargs.pop("voice_clone_prompt", None))
        if item is not None:
            if item.get("x_vector_only"):
                raise MLXError(
                    "x-vector-only cloning is not available on the MLX backend. Untick it, or "
                    "run the PyTorch path (--device cpu).")
            if ref_audio is None:
                wav, sr = item.get("wav"), int(item.get("sr") or REF_SAMPLE_RATE)
                path = item.get("ref_path")
                if wav is None:
                    raise MLXError(
                        "the cached clone prompt reached the MLX backend without its reference "
                        "audio. Clear the voice's cached take and generate again.")
                wav = np.asarray(wav, dtype=np.float32).reshape(-1)
                # Re-keyed by content, never by slot: this is what makes a second voice's take
                # actually use the second voice.
                path = _write_reference(wav, sr, self._ref_cache)
                return path, (ref_text if ref_text is not None else item.get("ref_text"))
        if ref_audio is None:
            raise MLXError(
                "the clone request reached the MLX backend with no reference audio, which would "
                "render the checkpoint's default speaker instead of the cloned voice. Send "
                "voice_clone_prompt from create_voice_clone_prompt, or ref_audio + ref_text.")
        wav, sr = self._as_wave(ref_audio)
        return _write_reference(wav, sr, self._ref_cache), ref_text

    # -- the surface the app calls ------------------------------------------

    def generate_voice_clone(self, text: str, language: Optional[str] = None,
                             ref_audio: Any = None, ref_text: Optional[str] = None,
                             x_vector_only_mode: bool = False, **kwargs: Any
                             ) -> Tuple[List[np.ndarray], int]:
        if x_vector_only_mode:
            # Not exposed by mlx-audio's Qwen3-TTS generate(); refusing is honest, running it
            # as a normal clone would silently mean something different from what was asked.
            raise MLXError(
                "x-vector-only cloning is not available on the MLX backend. Untick it, or run "
                "the PyTorch path (--device cpu).")
        path, text_ref = self._reference_from(ref_audio, ref_text, kwargs)
        return self._run(
            text=text,
            lang_code=_LANG_CODES.get((language or "").strip().lower(), "auto"),
            ref_audio=path, ref_text=text_ref, stream=False,
            **self._sampling(kwargs))

    def generate_voice_design(self, text: str, instruct: Optional[str] = None,
                              language: Optional[str] = None, **kwargs: Any
                              ) -> Tuple[List[np.ndarray], int]:
        return self._run(
            text=text, instruct=instruct or "",
            lang_code=_LANG_CODES.get((language or "").strip().lower(), "auto"),
            stream=False, **self._sampling(kwargs))

    def speaker_embedding(self, wav: Any, sr: int = 24000) -> np.ndarray:
        """Cosine-similarity embedding for best-of-N. mlx-audio exposes it; the array type is
        not torch, so it is normalised to numpy here rather than at the call site.

        mlx's `extract_speaker_embedding` is annotated `audio: mx.array` and its first act is
        `mx.concatenate([left_pad, sample, right_pad])` — hand it a numpy array and that raises
        a TypeError about incompatible function arguments. The app calls this inside best-of-N
        under `except Exception`, so the failure was silent: candidates were ranked on penalties
        alone and nothing said the speaker tie-break had stopped existing.
        """
        import mlx.core as mx

        fn = getattr(self._model, "extract_speaker_embedding", None)
        if fn is None:
            raise MLXError("this mlx-audio build has no speaker encoder; SIM tie-break off")
        if isinstance(wav, (tuple, list)) and len(wav) == 2:
            wav, sr = wav
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        if int(sr) != 24000:
            # mlx refuses anything but 24kHz for this encoder, and the app hands us whatever the
            # candidate came back at. Resampling here keeps the tie-break alive instead of
            # letting a 16kHz candidate silently opt out of it.
            import librosa

            arr = librosa.resample(arr, orig_sr=int(sr), target_sr=24000)
        emb = fn(audio=mx.array(arr), sr=24000)
        return _to_numpy(emb)

    # -- housekeeping --------------------------------------------------------

    def close(self) -> None:
        for path in list(self._ref_cache.values()):
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
        self._ref_cache.clear()
