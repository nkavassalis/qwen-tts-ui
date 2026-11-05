#!/usr/bin/env bash
# Reproducible environment setup for the local Qwen3-TTS deployment.
#
# Two first-party platforms, one script:
#
#   Linux + CUDA     .venv (Python 3.12), torch 2.8.0+cu128, pinned upstream qwen-tts,
#                    the 1.7B checkpoints, Whisper.
#   macOS (M-series) .venv, torch CPU (for Whisper), mlx-audio, the mlx-community
#                    ports of the same two checkpoints in both quantisations, Whisper.
#
# The engine is chosen by `uname`; nothing else about the app differs. Set
# QWEN_TTS_ENGINE=torch on an M-series Mac to force the PyTorch path instead.
#
#   ./setup.sh              # everything (skips work already done)
#   ./setup.sh --models     # only (re-)download the TTS checkpoints
#   ./setup.sh --asr        # only (re-)fetch Whisper
#   ./setup.sh --voices     # only (re-)create the shipped voices
set -euo pipefail
cd "$(dirname "$0")"

ONLY=""
case "${1:-}" in
  --models)  ONLY=models ;;
  --asr)     ONLY=asr ;;
  --voices)  ONLY=voices ;;
  "")        ;;
  *) echo "usage: $0 [--models|--asr|--voices]" >&2; exit 2 ;;
esac

# Engine selection mirrors webui/app.py::default_engine(). Keep the two in step: the app
# picks MLX on Apple Silicon and torch everywhere else, and a setup.sh that disagrees would
# install a runtime the server then refuses to use.
ENGINE="${QWEN_TTS_ENGINE:-auto}"
if [[ $ENGINE == auto ]]; then
  if [[ $(uname -s) == Darwin && $(uname -m) == arm64 ]]; then ENGINE=mlx; else ENGINE=torch; fi
fi
echo "platform: $(uname -s) $(uname -m) -> engine: $ENGINE"

# QwenLM/Qwen3-TTS ref to install. A branch or tag name is fetched shallowly; a full
# 40-char commit SHA is fetched with full history (GitHub refuses shallow SHA fetches).
# Freeze this to the SHA your ./repo is actually on before trusting a fresh install:
#   git -C repo rev-parse HEAD
UPSTREAM_REF="${QWEN_TTS_UPSTREAM_REF:-main}"

REPO_DIR="repo"
MODELS_DIR="models"
MODELS=(
  Qwen3-TTS-12Hz-1.7B-Base
  Qwen3-TTS-12Hz-1.7B-VoiceDesign
  Qwen3-TTS-Tokenizer-12Hz
)
# Must stay in sync with CAPABILITIES in webui/app.py — tests/test_app_logic.py::
# test_setup_sh_knows_every_capability enforces it.

# The MLX ports of the same two checkpoints, in both quantisations, so --quant is a lever
# that works out of the box instead of a flag that triggers a surprise download the first
# time you try it. bf16 is what every measurement in docs/quality.md was taken on; 8bit is
# smaller and faster. There is no tokenizer to fetch on this path: the codec ships inside
# each repo. Must stay in sync with MLX_REPOS in webui/app.py.
MLX_MODELS=(
  mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
  mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
  mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
  mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit
)

# Transcription of reference clips. Fetched into the Hugging Face cache (~/.cache),
# NOT into models/. The excludes matter: without them the repo drags ~24GB of flax/tf/
# fp32 duplicates of the one 2.9GB safetensors file we use.
ASR_MODEL="${QWEN_TTS_ASR_MODEL:-openai/whisper-large-v3}"

have_checkpoint() {  # config.json plus at least one weight file == usable
  local dir="$1"
  [[ -s "$dir/config.json" ]] || return 1
  [[ -s "$dir/model.safetensors" ]] && return 0
  compgen -G "$dir/*.safetensors" >/dev/null
}

if [[ -z $ONLY || $ONLY == asr ]]; then
  command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

  [[ -d .venv ]] || uv venv --python 3.12 .venv

  if [[ $ENGINE == mlx ]]; then
    # torch (CPU build) is still installed: Whisper runs on it, and the reference-cleanup
    # and waveform code uses it. It is NOT the TTS runtime on this path — mlx-audio is.
    uv pip install --python .venv/bin/python "torch" "torchaudio"
    uv pip install --python .venv/bin/python -r requirements.txt
    # mlx-audio carries the Qwen3-TTS talker, the 12Hz speech tokenizer and the speaker
    # encoder. Pinned deliberately: an unpinned sampling or float8 change upstream would
    # alter the audio behind our measurements without anything here noticing.
    uv pip install --python .venv/bin/python "mlx-audio"

    .venv/bin/python - <<'PY'
import mlx.core as mx
import torch
print(f"mlx {mx.__version__} on {mx.default_device()} | torch {torch.__version__} "
      "(Whisper and the audio helpers only)")
PY
  else

  # torch 2.8.0+cu128 is the oldest stable wheel line that supports sm_120 (RTX 50xx).
  # Pinned to the PyTorch cu128 index explicitly: the plain PyPI default is whatever CUDA
  # build that release happened to ship, which is not a promise we want to rely on.
  uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.8.0" "torchaudio==2.8.0"

  uv pip install --python .venv/bin/python -r requirements.txt

  [[ -d $REPO_DIR ]] || git clone https://github.com/QwenLM/Qwen3-TTS.git "$REPO_DIR"
  if [[ $UPSTREAM_REF =~ ^[0-9a-f]{40}$ ]]; then
    ( cd "$REPO_DIR" && git fetch origin && git checkout "$UPSTREAM_REF" )
  else
    ( cd "$REPO_DIR" && git fetch --depth 1 origin "$UPSTREAM_REF" && git checkout FETCH_HEAD )
  fi
  # -e (editable) because upstream's own package data + the finetuning/ tree live there.
  uv pip install --python .venv/bin/python -e "$REPO_DIR"

  .venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch has no CUDA — check the driver / wheel variant"
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"torch {torch.__version__} cuda {torch.version.cuda} on {name} sm_{cap[0]}{cap[1]}")
if cap < (12, 0):
    # Not fatal: everything below sm_120 works on the cu128 wheels too. This box is
    # Blackwell, so the message is a reminder to re-check kernels, not a gate.
    print(f"NOTE: {name} is sm_{cap[0]}{cap[1]} (pre-Blackwell). Fine on these wheels; "
          "if you hit 'no kernel image available', the torch build does not cover your "
          "architecture — pick the matching wheel from https://download.pytorch.org/whl.")
PY
  fi
fi

if [[ -z $ONLY || $ONLY == models ]]; then
  if [[ $ENGINE == mlx ]]; then
    mkdir -p "$MODELS_DIR-mlx"
    for m in "${MLX_MODELS[@]}"; do
      target="$MODELS_DIR-mlx/${m##*/}"
      if have_checkpoint "$target"; then
        echo "have ${m##*/}"
        continue
      fi
      echo "downloading $m"
      .venv/bin/hf download "$m" --local-dir "$target"
    done
  else
  mkdir -p "$MODELS_DIR"
  for m in "${MODELS[@]}"; do
    if have_checkpoint "$MODELS_DIR/$m"; then
      echo "have $m"
      continue
    fi
    echo "downloading $m"
    .venv/bin/hf download "Qwen/$m" --local-dir "$MODELS_DIR/$m"
  done
  fi
fi

if [[ -z $ONLY || $ONLY == asr ]]; then
  [[ -x .venv/bin/hf ]] || { echo "run ./setup.sh without --asr first (needs .venv)" >&2; exit 1; }
  echo "fetching $ASR_MODEL (2.9GB, into the Hugging Face cache)"
  .venv/bin/hf download "$ASR_MODEL" --exclude "*fp32*" "*.bin" "*.msgpack" "*.h5" "*.ot"
fi

# The shipped voice. Fox is a *description*, not a recording: this project does not
# distribute anyone's voice (see README → Ethical use), so the repo carries the words and
# your GPU renders the sound. The description is written first because it needs no GPU at
# all — Speak and Chat work from it — and the frozen clip is a bonus for Voice Clone.
FOX_DESCRIPTION="A bright, expressive adult female voice in her late twenties, slightly nasal, quick and emotional delivery that lifts at the end of phrases, casual American English, warm but sharp at the edges, a little breathless when she is excited."

if [[ -z $ONLY || $ONLY == voices ]]; then
  mkdir -p voices
  if [[ ! -s voices/fox.json ]]; then
    echo "creating the shipped voice 'Fox' (description)"
    .venv/bin/python scripts/make_voice.py --name Fox --no-clip \
      --description "$FOX_DESCRIPTION" || echo "WARNING: could not create Fox"
  else
    echo "have fox"
  fi
  if [[ -s voices/fox.json && ! -s voices/fox.wav ]]; then
    echo "rendering Fox's frozen clip (needs the GPU and the weights)"
    .venv/bin/python scripts/make_voice.py --name Fox --force \
      --description "$FOX_DESCRIPTION" \
      || echo "NOTE: no clip rendered — Fox stays description-only, which Speak and Chat still speak."
  fi
fi

echo
echo "Done. Start the UI with ./run.sh"
