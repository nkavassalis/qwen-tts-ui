# coding=utf-8
"""
Local Qwen3-TTS Gradio WebUI.

Adapted from the official Hugging Face Space (Qwen/Qwen3-TTS app.py) for a single
local GPU. Differences from the Space version:

  * Loads the 1.7B checkpoints (the best quality tier) from a local directory.
  * The Space keeps all 5 models resident on the GPU at once (~20GB VRAM). A 16GB
    RTX 5070 Ti cannot do that, so models live in CPU RAM and an LRU "GPU slot"
    moves at most `--max-resident` models onto the GPU on demand (~1-3s swap).
  * No `spaces` (Zero-GPU) dependency, no model-size dropdown, attn_implementation
    is `sdpa` (flash-attn 2 has no sm_120/Blackwell wheels; the Space uses sdpa too).

Run:  ./run.sh            (or)  .venv/bin/python webui/app.py --ip 127.0.0.1 --port 7860
"""

from __future__ import annotations

import argparse
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

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

SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan",
    "Serena", "Sohee", "Uncle_fu", "Vivian",
]
LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", "French",
    "German", "Spanish", "Portuguese", "Russian", "Italian",
]

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


def _budget(text: str, cap: int = MAX_NEW_TOKENS_CAP) -> int:
    """Token budget scaled to the prompt length (guards against non-stopping runs)."""
    return int(min(cap, max(MIN_NEW_TOKENS, len(text.strip()) * TOKENS_PER_CHAR)))


def _sampling(temperature, top_p, top_k, repetition_penalty) -> Dict[str, Any]:
    return dict(
        temperature=float(temperature), top_p=float(top_p),
        top_k=int(top_k), repetition_penalty=float(repetition_penalty),
    )


# ============================================================================
# Model loading + VRAM management
# ============================================================================
class ModelHub:
    """Keeps every loaded model in CPU RAM; only a few sit on the GPU at once.

    Weights are lazy-loaded from disk the first time a capability is used and then
    pinned in host RAM (~4GB each). Before each generation the requested model is
    moved to the GPU; least-recently-used models are pushed back to CPU RAM so the
    working set stays inside VRAM.
    """

    def __init__(self, models_dir: str, dtype: torch.dtype, attn_impl: str, max_resident: int):
        self.models_dir = models_dir
        self.dtype = dtype
        self.attn_impl = attn_impl
        self.max_resident = max(1, max_resident)
        self.cpu_cache: "OrderedDict[str, Qwen3TTSModel]" = OrderedDict()
        self.on_gpu: "OrderedDict[str, Qwen3TTSModel]" = OrderedDict()
        self.lock = threading.RLock()

    def _checkpoint(self, capability: str) -> str:
        dirname, _ = CAPABILITIES[capability]
        path = os.path.join(self.models_dir, dirname)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Model not found at {path}. Download it with:\n"
                f"  .venv/bin/hf download Qwen/{dirname} --local-dir {path}"
            )
        return path

    def _load(self, capability: str) -> Qwen3TTSModel:
        path = self._checkpoint(capability)
        print(f"[hub] loading {path} -> CPU RAM ...", flush=True)
        tts = Qwen3TTSModel.from_pretrained(
            path,
            device_map="cpu",
            dtype=self.dtype,
            attn_implementation=self.attn_impl,
        )
        tts.device = torch.device("cpu")
        print(f"[hub] {capability} ready in CPU RAM", flush=True)
        return tts

    def _get_cpu(self, capability: str) -> Qwen3TTSModel:
        if capability not in self.cpu_cache:
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

            while len(self.on_gpu) >= self.max_resident:
                evicted_name, evicted = self.on_gpu.popitem(last=False)
                print(f"[hub] evicting {evicted_name} -> CPU RAM", flush=True)
                self._move(evicted, torch.device("cpu"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"[hub] promoting {capability} -> {DEVICE}", flush=True)
            self._move(tts, DEVICE)
            self.on_gpu[capability] = tts
            return tts

    def status(self) -> str:
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            vram = f"{free_b / 2**30:.1f}GB free / {total_b / 2**30:.1f}GB total"
        else:
            vram = "no CUDA device"
        return (
            f"GPU: {vram} | in CPU RAM: {', '.join(self.cpu_cache) or '-'} | "
            f"on GPU: {', '.join(self.on_gpu) or '-'}"
        )


HUB: Optional[ModelHub] = None


def _ok(audio: Tuple[int, np.ndarray], message: str) -> Tuple[Any, str]:
    """Gradio output pair: (audio, status line including live VRAM info)."""
    return audio, f"{message}\n{HUB.status()}"


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
    """Gradio Audio value -> (wav float32, sample_rate)."""
    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        return _normalize_audio(wav), int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        return _normalize_audio(audio["data"]), sr
    return None


# ============================================================================
# Gradio callbacks
# ============================================================================
def generate_custom_voice(text, language, speaker, instruct, temperature, top_p, top_k, rep_penalty,
                          progress=gr.Progress(track_tqdm=True)):
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
        return _ok((sr, wavs[0]), _report(wavs[0], sr, budget, "CustomVoice"))
    except Exception as e:  # noqa: BLE001 - surface errors in the UI instead of killing the server
        return None, f"Error: {type(e).__name__}: {e}"


def generate_voice_design(text, language, voice_description, temperature, top_p, top_k, rep_penalty,
                          progress=gr.Progress(track_tqdm=True)):
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
        return _ok((sr, wavs[0]), _report(wavs[0], sr, budget, "VoiceDesign"))
    except Exception as e:  # noqa: BLE001
        return None, f"Error: {type(e).__name__}: {e}"


# Reusable clone prompts, keyed by a hash of the uploaded reference audio, so the
# reference encoder is not recomputed for every line (big speedup for repeated takes).
_CLONE_PROMPTS: Dict[str, Any] = {}


def _clone_cache_key(audio_tuple, ref_text, x_vector_only) -> str:
    wav, sr = audio_tuple
    return f"{sr}|{wav.shape[0]}|{float(wav.sum()):.4f}|{ref_text or ''}|{bool(x_vector_only)}"


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only,
                         temperature, top_p, top_k, rep_penalty,
                         progress=gr.Progress(track_tqdm=True)):
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."
    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."
    if not use_xvector_only and not (ref_text or "").strip():
        return None, "Error: Reference text is required when 'x-vector only' is off."
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
            wavs, sr = tts.generate_voice_clone(
                text=target_text.strip(),
                language=language,
                voice_clone_prompt=prompt,
                max_new_tokens=budget,
                **_sampling(temperature, top_p, top_k, rep_penalty),
            )
        return _ok((sr, wavs[0]), _report(wavs[0], sr, budget, "VoiceClone"))
    except Exception as e:  # noqa: BLE001
        return None, f"Error: {type(e).__name__}: {e}"


def refresh_status():
    return HUB.status()


# ============================================================================
# UI
# ============================================================================
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS") as demo:
        gr.Markdown(
            """# Qwen3-TTS — local (1.7B, RTX 5070 Ti)
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
        sampling = [s_temperature, s_top_p, s_top_k, s_rep]

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
                            cv_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="English")
                            cv_speaker = gr.Dropdown(label="Speaker", choices=SPEAKERS, value="Ryan")
                        cv_instruct = gr.Textbox(
                            label="Style instruction (optional)", lines=2,
                            placeholder="e.g. Speak in a cheerful, energetic tone / 用愤怒的语气说",
                        )
                        cv_btn = gr.Button("Generate speech", variant="primary")
                    with gr.Column(scale=2):
                        cv_audio = gr.Audio(label="Generated audio", type="numpy")
                        cv_status = gr.Textbox(label="Status", lines=2, interactive=False)
                cv_btn.click(generate_custom_voice,
                             inputs=[cv_text, cv_language, cv_speaker, cv_instruct] + sampling,
                             outputs=[cv_audio, cv_status])

            with gr.Tab("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=2):
                        vd_text = gr.Textbox(
                            label="Text to synthesize", lines=4,
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                        )
                        vd_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto")
                        vd_instruct = gr.Textbox(
                            label="Voice description", lines=3,
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
                            placeholder="Describe timbre, age, gender, emotion, accent, speaking rate...",
                        )
                        vd_btn = gr.Button("Generate voice", variant="primary")
                    with gr.Column(scale=2):
                        vd_audio = gr.Audio(label="Generated audio", type="numpy")
                        vd_status = gr.Textbox(label="Status", lines=2, interactive=False)
                vd_btn.click(generate_voice_design,
                             inputs=[vd_text, vd_language, vd_instruct] + sampling,
                             outputs=[vd_audio, vd_status])

            with gr.Tab("Voice Clone (Base)"):
                gr.Markdown("Upload (or record) a short clean reference clip, give its transcript, then synthesize new text.")
                with gr.Row():
                    with gr.Column(scale=2):
                        cl_ref_audio = gr.Audio(label="Reference audio", type="numpy",
                                                sources=["upload", "microphone"])
                        cl_ref_text = gr.Textbox(label="Reference transcript", lines=2,
                                                 placeholder="Exact text spoken in the reference audio...")
                        cl_xvector = gr.Checkbox(
                            label="x-vector only (no transcript needed, lower similarity)", value=False)
                    with gr.Column(scale=2):
                        cl_text = gr.Textbox(label="Target text", lines=4,
                                             placeholder="What the cloned voice should say...")
                        cl_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto")
                        cl_btn = gr.Button("Clone & generate", variant="primary")
                with gr.Row():
                    cl_audio = gr.Audio(label="Generated audio", type="numpy")
                    cl_status = gr.Textbox(label="Status", lines=2, interactive=False)
                cl_btn.click(generate_voice_clone,
                             inputs=[cl_ref_audio, cl_ref_text, cl_text, cl_language, cl_xvector] + sampling,
                             outputs=[cl_audio, cl_status])

        gr.Markdown(
            "---\nPowered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba Qwen). "
            "Long texts are capped at `max_new_tokens=2048`; split very long scripts into chunks."
        )
    return demo


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
    parser.add_argument("--device", default="cuda", help="cuda / cuda:0 / cpu")
    parser.add_argument("--preload", action="store_true",
                        help="Load all three checkpoints into CPU RAM at startup instead of lazily.")
    parser.add_argument("--ssl-certfile", default=None)
    parser.add_argument("--ssl-keyfile", default=None)
    parser.add_argument("--share", action="store_true", help="Create a public gradio.app link.")
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    HUB = ModelHub(args.models_dir, dtype, args.attn_implementation, args.max_resident)

    if args.preload:
        for capability in CAPABILITIES:
            HUB._get_cpu(capability)

    demo = build_ui()
    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip, server_port=args.port, share=args.share,
        inbrowser=False, css=".gradio-container {max-width: none !important;}",
    )
    if args.ssl_certfile and args.ssl_keyfile:
        launch_kwargs.update(ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)

    print(f"[webui] {HUB.status()}", flush=True)
    demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
