#!/usr/bin/env bash
# Launch the local Qwen3-TTS Gradio web UI.
#
#   ./run.sh                 -> http://127.0.0.1:7860  (localhost only)
#   ./run.sh --ip 0.0.0.0    -> expose on the LAN
#   ./run.sh --preload       -> load all three 1.7B checkpoints into RAM up front
#   ./run.sh --ssl-self-signed --ip 0.0.0.0 --port 7860
#                            -> HTTPS, needed for browser mic access on non-localhost
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
    if [[ ! -f certs/key.pem ]]; then
      mkdir -p certs
      openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem \
        -days 365 -nodes -subj "/CN=localhost" 2>/dev/null
      echo "generated self-signed certs in ./certs"
    fi
    ARGS+=(--ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem)
  else
    ARGS+=("$a")
  fi
done

exec "$VENV" webui/app.py "${ARGS[@]}"
