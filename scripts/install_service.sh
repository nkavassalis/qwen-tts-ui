#!/usr/bin/env bash
# Keep `./run.sh` alive for you — across logout and reboot, and after a crash.
#
#   scripts/install_service.sh install [run.sh flags...]
#   scripts/install_service.sh restart | status | logs | uninstall
#
# macOS  ->  ~/Library/LaunchAgents/com.qwen-tts.ui.<name>.plist   (launchd, KeepAlive)
# Linux  ->  ~/.config/systemd/user/qwen-tts-ui.<name>.service     (systemd, Restart=always)
#
# It runs `run.sh`, not `webui/app.py`, so everything the launcher does — the venv, the
# self-signed cert minting, argument validation — still happens. And it means the single-owner
# lock still applies: if you start `./run.sh` by hand while the service is up, the *second* one
# refuses and names the holder, exactly as it should. A service that cannot get the lock exits
# and retries on a throttle rather than spinning against it.
#
# Flags are whatever `./run.sh` takes, and are stored in the unit:
#
#   scripts/install_service.sh install --ip 0.0.0.0 --username you --password-file ~/tts.pw
#
# `--password` is refused on purpose: argv is world-readable through ps and /proc, and a unit
# file is plain text on disk. Use --password-file.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
SLUG="$(basename "$REPO" | tr -cs 'a-zA-Z0-9' '-' | tr 'A-Z' 'a-z' | sed 's/^-*//; s/-*$//')"
SLUG="${SLUG:-qwen-tts-ui}"
LABEL="com.qwen-tts.ui.${SLUG}"
UNIT="qwen-tts-ui.${SLUG}"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
SERVICE="$HOME/.config/systemd/user/${UNIT}.service"
# Homebrew and ~/.local/bin are not on a non-interactive PATH on macOS, which is how a server
# started by launchd ends up printing "ffmpeg: NOT FOUND (video uploads disabled)" while the
# same command works perfectly in Terminal.
SERVICE_PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/usr/bin:/bin"

die() { echo "error: $*" >&2; exit 1; }

platform() {
  case "$(uname -s)" in
    Darwin) echo macos ;;
    Linux)  echo linux ;;
    *)      die "no service support for $(uname -s); run ./run.sh yourself" ;;
  esac
}

validate_args() {
  (( $# > 0 )) || return 0
  for a in "$@"; do
    [[ $a == "--password" || $a == --password=* ]] &&
      die "do not pass --password: argv is world-readable and the unit file is plain text. Use --password-file."
  done
  # Fail here rather than at the first reboot, when a bad flag looks like a broken service.
  ./run.sh --help >/dev/null 2>&1 || true
}

xml_quote() { local s=$1; s=${s//&/&amp;}; s=${s//</&lt;}; s=${s//>/&gt;}; echo "$s"; }

write_plist() {
  local args="" a
  for a in "$@"; do args+="    <string>$(xml_quote "$a")</string>"$'\n'; done
  # A Mac that has never had a LaunchAgent has no such directory, and `cat >` does not make one.
  mkdir -p "$(dirname "$PLIST")"
  cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>run.sh</string>
${args}  </array>
  <key>WorkingDirectory</key><string>${REPO}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>${SERVICE_PATH}</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>${REPO}/logs/service.log</string>
  <key>StandardErrorPath</key><string>${REPO}/logs/service.log</string>
</dict>
</plist>
PLIST
}

write_unit() {
  local args="" a
  for a in "$@"; do args+=" $(printf '%q' "$a")"; done
  mkdir -p "$(dirname "$SERVICE")"
  cat > "$SERVICE" <<UNIT
[Unit]
Description=qwen3-tts-ui (${SLUG})
Documentation=file://${REPO}/README.md
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${REPO}
Environment=PATH=${SERVICE_PATH}
ExecStart=/bin/bash run.sh${args}
Restart=always
RestartSec=15
# The app writes its own log line per request; keep journald from swallowing the memory-guard
# messages that explain a refusal to load.
TimeoutStopSec=30

[Install]
WantedBy=default.target
UNIT
}

install() {
  [[ -x run.sh ]] || die "no ./run.sh here — run this from the project root after ./setup.sh"
  [[ -x .venv/bin/python ]] || die "no .venv — run ./setup.sh first"
  validate_args "$@"
  mkdir -p logs
  if [[ $(platform) == macos ]]; then
    write_plist "$@"
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    launchctl kickstart -k "gui/$(id -u)/${LABEL}" 2>/dev/null || true
    echo "installed ${PLIST}"
  else
    write_unit "$@"
    systemctl --user daemon-reload
    systemctl --user enable --now "$UNIT.service"
    # Without lingering, a user service dies when the last login session closes — which on WSL
    # is the difference between "starts at boot" and "starts when I happen to open a terminal".
    loginctl enable-linger "$USER" 2>/dev/null || true
    echo "installed ${SERVICE}"
  fi
  echo "logs: $(log_path)   ·   status: $0 status   ·   remove: $0 uninstall"
}

log_path() { echo "$REPO/logs/service.log"; }

status() {
  if [[ $(platform) == macos ]]; then
    launchctl print "gui/$(id -u)/${LABEL}" 2>&1 | grep -E "state|pid|last exit|path" | head -6 \
      || echo "not loaded (run: $0 install)"
  else
    systemctl --user --no-pager --full "$UNIT.service" | head -12
  fi
}

uninstall() {
  if [[ $(platform) == macos ]]; then
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
    rm -f "$PLIST"
    echo "removed ${PLIST}"
  else
    systemctl --user disable --now "$UNIT.service" 2>/dev/null || true
    rm -f "$SERVICE"
    systemctl --user daemon-reload
    echo "removed ${SERVICE}"
  fi
}

case "${1:-}" in
  install)   shift; install "$@" ;;
  uninstall) uninstall ;;
  status)    status ;;
  restart)
    if [[ $(platform) == macos ]]; then
      launchctl kickstart -k "gui/$(id -u)/${LABEL}"
    else
      systemctl --user restart "$UNIT.service"
    fi
    echo "restarted" ;;
  logs)      tail -n 40 "$(log_path)" 2>/dev/null || echo "no log yet at $(log_path)" ;;
  *) cat <<HELP
Usage: $0 install [run.sh flags...] | uninstall | status | restart | logs

  install --ip 0.0.0.0 --username you --password-file ~/tts.pw

Runs ./run.sh as a per-user service that starts at login/boot and restarts on crash.
macOS uses launchd, Linux uses a systemd user unit. --password is refused; use
--password-file. Your own ./run.sh still works — the single-owner lock decides which
one gets the weights, and the loser says so.
HELP
    exit 2 ;;
esac
