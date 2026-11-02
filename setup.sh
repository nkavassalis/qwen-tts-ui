#!/usr/bin/env bash
# Reproducible environment setup for the local Qwen3-TTS deployment.
#
# Creates .venv (Python 3.12), installs CUDA-enabled torch, installs the qwen-tts
# package from the pinned upstream checkout, and downloads the 1.7B checkpoints.
#
#   ./setup.sh              # everything (skips work already done)
#   ./setup.sh --models     # only (re-)download weights
set -euo pipefail
cd "$(dirname "$0")"

ONLY=""
[[ "${1:-}" == "--models" ]] && ONLY=models

UPSTREAM_TAG="main"          # QwenLM/Qwen3-TTS; pin a commit here for full reproducibility
REPO_DIR="repo"
MODELS_DIR="models"
MODELS=(
  Qwen3-TTS-12Hz-1.7B-CustomVoice
  Qwen3-TTS-12Hz-1.7B-VoiceDesign
  Qwen3-TTS-12Hz-1.7B-Base
  Qwen3-TTS-Tokenizer-12Hz
)

if [[ -z $ONLY ]]; then
  command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

  [[ -d .venv ]] || uv venv --python 3.12 .venv

  # torch 2.8.0+cu128 is the oldest stable wheel line that supports sm_120 (RTX 50xx).
  uv pip install --python .venv/bin/python "torch==2.8.0" "torchaudio==2.8.0"

  [[ -d $REPO_DIR ]] || git clone https://github.com/QwenLM/Qwen3-TTS.git "$REPO_DIR"
  ( cd "$REPO_DIR" && git fetch --depth 1 origin "$UPSTREAM_TAG" && git checkout FETCH_HEAD )
  uv pip install --python .venv/bin/python -e "$REPO_DIR"

  .venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch has no CUDA — check the driver / wheel variant"
cap = torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__} cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
assert cap >= (12, 0), "expected a Blackwell GPU; older GPUs work too but re-check the wheel"
PY
fi

mkdir -p "$MODELS_DIR"
for m in "${MODELS[@]}"; do
  if [[ -f "$MODELS_DIR/$m/model.safetensors" ]]; then
    echo "have $m"
    continue
  fi
  echo "downloading $m"
  .venv/bin/hf download "Qwen/$m" --local-dir "$MODELS_DIR/$m"
done

echo
echo "Done. Start the UI with ./run.sh"
