#!/usr/bin/env bash
# Launch qwen3-tts-ui — the local Qwen3-TTS Gradio web UI.
#
#   ./run.sh                     -> http://127.0.0.1:7860  (localhost only)
#   ./run.sh --ip 0.0.0.0        -> expose on the LAN (pass --username/--password!)
#   ./run.sh --preload           -> load up to --max-cpu-resident checkpoints into RAM at
#                                   startup (default 2 = both shipped checkpoints, ~8.6GB;
#                                   the memory floor refuses rather than swapping)
#   ./run.sh --ssl-self-signed --ip 0.0.0.0 --port 7860
#                                -> HTTPS, needed for browser mic access on non-localhost
#   ./run.sh --no-asr            -> skip Whisper entirely (frees ~3GB of VRAM when loaded)
#   ./run.sh --asr-model openai/whisper-large-v3-turbo   -> swap the transcription model
#
# Any other flags are forwarded to webui/app.py (see --help).
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv/bin/python"
if [[ ! -x $VENV ]]; then
  echo "No .venv — run ./setup.sh first." >&2
  exit 1
fi

ARGS=()
for a in "$@"; do
  if [[ $a == "--ssl-self-signed" ]]; then
    command -v openssl >/dev/null || { echo "--ssl-self-signed needs openssl on PATH" >&2; exit 1; }
    if [[ ! -f certs/key.pem || ! -f certs/cert.pem ]]; then
      mkdir -p certs   # openssl writes both files, so a half-written pair is rewritten
      openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem \
        -days 365 -nodes -subj "/CN=localhost" 2>/dev/null
      echo "generated self-signed certs in ./certs"
    fi
    ARGS+=(--ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem --ssl-self-signed)
  else
    ARGS+=("$a")
  fi
done

exec "$VENV" webui/app.py ${ARGS[@]+"${ARGS[@]}"}
