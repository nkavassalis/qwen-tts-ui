#!/usr/bin/env python3
# coding=utf-8
"""Talk to me: render text in a saved Qwen3-TTS voice and play it out loud.

Built to be called by *something else* — a coding assistant, a build hook, a CI job —
which is why this is a shell command rather than an SDK call:

    scripts/say.py --voice fox-take-4 --text "Build finished. Three tests failed."
    echo "The migration is done." | scripts/say.py --voice fox-take-4
    scripts/say.py --list
    scripts/say.py --voice ryan --text "Coffee is ready." --out kitchen.wav

Only the voice *name* crosses the network. The reference clip and its transcript are
read from voices/ on the server, so a caller cannot name a path, and a script does not
need to know anything about the voice library beyond the slug it read from --list.

Long text is chunked and stitched server-side, the same as in the web UI. Expect
~1.4x realtime, with the first sentence audible after ~6-9s if you keep --out off and
the player starts as soon as the file lands.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

DEFAULT_URL = os.environ.get("QWEN_TTS_URL", "http://127.0.0.1:7860")
PLAYERS = (
    ("paplay", []),
    ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ("aplay", []),
    ("afplay", []),          # macOS
)


def resolve_credentials(args: argparse.Namespace) -> tuple:
    """Same precedence the server uses: file, then env, then flag (warned), then none."""
    if args.password_file:
        with open(os.path.expanduser(args.password_file), encoding="utf-8") as fh:
            return args.username, fh.read().strip()
    if os.environ.get("QWEN_TTS_PASSWORD"):
        return args.username, os.environ["QWEN_TTS_PASSWORD"]
    if args.password:
        print("warning: --password is visible in `ps`/proc; prefer --password-file or "
              "QWEN_TTS_PASSWORD", file=sys.stderr)
        return args.username, args.password
    return (args.username, None) if args.username else (None, None)


def pick_player():
    for name, extra in PLAYERS:
        if shutil.which(name):
            return [name] + extra
    return None


def audio_path(result, client) -> tuple:
    """Pull (local_path, status_text) out of the /speak response."""
    if isinstance(result, (list, tuple)):
        audio, status = (list(result) + [None, None])[:2]
    else:
        audio, status = result, None
    if audio is None:
        return None, str(status or "the server returned no audio")
    if isinstance(audio, dict):
        audio = audio.get("path") or audio.get("url") or audio.get("value")
    if isinstance(audio, (list, tuple)) and audio:
        audio = audio[0]
    if not isinstance(audio, str):
        return None, f"unexpected audio payload from the server: {type(audio).__name__}"
    if audio.startswith(("http://", "https://")):
        # gradio_client normally downloads file outputs for us; if it handed back a URL,
        # fetch it rather than handing a URL to paplay.
        suffix = os.path.splitext(audio.split("?")[0])[1] or ".wav"
        fd, local = tempfile.mkstemp(suffix=suffix, prefix="qwen-tts-say-")
        os.close(fd)
        import urllib.request
        with urllib.request.urlopen(audio, timeout=60) as response, open(local, "wb") as fh:
            fh.write(response.read())
        return local, str(status or "")
    return audio, str(status or "")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Speak text through a running qwen-tts server in a saved voice.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"default {DEFAULT_URL}")
    parser.add_argument("--voice", help="saved voice name or slug (see --list)")
    parser.add_argument("--text", help="what to say; '-' or omitted reads stdin")
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--postproc", default="warm", choices=("off", "clean", "warm"),
                        help="listening polish; 'off' keeps the raw 24kHz decode")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--top-k", dest="top_k", type=int, default=50)
    parser.add_argument("--repetition-penalty", dest="rep_penalty", type=float, default=1.05)
    parser.add_argument("--out", help="keep the wav here instead of a temp file")
    parser.add_argument("--no-play", action="store_true", help="render only, do not play")
    parser.add_argument("--list", action="store_true", help="print the voices on the server")
    parser.add_argument("--username", default=os.environ.get("QWEN_TTS_USERNAME"))
    parser.add_argument("--password-file", dest="password_file")
    parser.add_argument("--password")
    parser.add_argument("--insecure", action="store_true",
                        help="skip TLS verification (the server's --ssl-self-signed)")
    args = parser.parse_args(argv)

    try:
        from gradio_client import Client
        import httpx
    except ImportError:
        print("gradio_client is required: .venv/bin/pip install gradio_client", file=sys.stderr)
        return 2

    username, password = resolve_credentials(args)
    auth = (username, password) if username and password else None
    kwargs = {"verbose": False, "ssl_verify": not args.insecure}
    if auth:
        kwargs["auth"] = auth
    try:
        client = Client(args.url, **kwargs)
    except (httpx.ConnectError, OSError) as e:
        print(f"cannot reach {args.url}: {e}\nis the server running?  ./run.sh", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001 - a friendly line beats a traceback in a terminal
        print(f"cannot talk to {args.url}: {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    if args.list:
        try:
            print(client.predict(api_name="/voices_available"))
        except Exception as e:  # noqa: BLE001
            print(f"--list failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 3
        return 0

    if not args.voice:
        print("error: --voice is required (scripts/say.py --list to see them)", file=sys.stderr)
        return 2
    text = args.text
    if text in (None, "-"):
        text = sys.stdin.read()
    if not text.strip():
        print("error: nothing to say (pass --text or pipe it on stdin)", file=sys.stderr)
        return 2

    try:
        result = client.predict(
            args.voice, text.strip(), args.language, args.temperature, args.top_p,
            args.top_k, args.rep_penalty, args.postproc, api_name="/speak")
    except Exception as e:  # noqa: BLE001
        print(f"speak failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    path, status = audio_path(result, client)
    if path is None:
        print(f"server refused: {status}", file=sys.stderr)
        return 3
    if args.out:
        shutil.copyfile(path, args.out)
        print(f"wrote {args.out}")
    for line in str(status).splitlines():
        if line.strip() and not line.startswith("GPU:"):
            print(line)
    if not args.no_play:
        player = pick_player()
        if not player:
            print("no player found (paplay/ffplay/aplay/afplay) — audio is at "
                  f"{path}", file=sys.stderr)
            return 0
        subprocess.run(player + [path], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
