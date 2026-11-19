#!/usr/bin/env bash
# Install aws_capsolver from this checkout without root.
#
#   deploy/install.sh              venv + dependencies + keys.json + rendered system unit,
#                                  then prints the sudo commands to install the unit
#   deploy/install.sh --user-unit  same, but installs and starts a systemd --user unit
#                                  (no sudo; boot start needs `loginctl enable-linger`)
#   deploy/install.sh --no-deps    skip pip (re-render the unit only)
#
# Environment: PYTHON (default python3), PORT (8790), HOST (127.0.0.1),
# TORCH_INDEX_URL (CPU wheels by default), WITH_AUDIO=0 to skip the audio solver deps.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
PORT="${PORT:-8790}"
HOST="${HOST:-127.0.0.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
WITH_AUDIO="${WITH_AUDIO:-1}"
USER_UNIT=0
DEPS=1
for arg in "$@"; do
  case "$arg" in
    --user-unit) USER_UNIT=1 ;;
    --no-deps) DEPS=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\033[1m==> %s\033[0m\n' "$*"; }
cd "$APP_DIR"

# 1. virtualenv + dependencies ------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  log "creating .venv with $PYTHON"
  "$PYTHON" -m venv .venv
fi
VPY="$APP_DIR/.venv/bin/python"
if [ "$DEPS" = 1 ]; then
  log "installing CPU-only torch from $TORCH_INDEX_URL"
  "$VPY" -m pip install --upgrade pip >/dev/null
  if ! "$VPY" -c "import torch" 2>/dev/null; then
    "$VPY" -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision
  fi
  log "installing service dependencies"
  "$VPY" -m pip install -r requirements.txt numpy open_clip_torch
  if [ "$WITH_AUDIO" = 1 ]; then
    "$VPY" -m pip install faster-whisper av scipy
  fi
fi

# 2. state and keys -------------------------------------------------------------
mkdir -p data weights
if [ ! -f keys.json ]; then
  KEY="$("$VPY" -c 'import secrets; print(secrets.token_hex(16))')"
  printf '{"%s": {"name": "default", "balance": 1000}}\n' "$KEY" > keys.json
  chmod 600 keys.json
  log "created keys.json with client key $KEY"
fi

# 3. systemd unit -------------------------------------------------------------------
RENDERED="$APP_DIR/deploy/aws-capsolver.rendered.service"
RUN_USER="$(id -un)"
RUN_GROUP="$(id -gn)"
sed -e "s#/opt/aws_capsolver#${APP_DIR}#g" \
    -e "s#^User=.*#User=${RUN_USER}#" \
    -e "s#^Group=.*#Group=${RUN_GROUP}#" \
    -e "s#^Environment=AWSCS_HOST=.*#Environment=AWSCS_HOST=${HOST}#" \
    -e "s#^Environment=AWSCS_PORT=.*#Environment=AWSCS_PORT=${PORT}#" \
    deploy/aws-capsolver.service > "$RENDERED"

if [ "$USER_UNIT" = 1 ]; then
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  ENV_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/aws-capsolver/env"
  mkdir -p "$UNIT_DIR" "$(dirname "$ENV_FILE")"
  [ -f "$ENV_FILE" ] || printf '# overrides, e.g.\n# AWSCS_WORKERS=8\n# HF_HUB_OFFLINE=1\n' > "$ENV_FILE"
  # user units cannot switch user and some sandboxing needs privileges
  sed -E -e '/^(User|Group|PrivateTmp|ProtectSystem|ProtectKernelTunables|ProtectControlGroups|RestrictSUIDSGID)=/d' \
      -e "s#^EnvironmentFile=.*#EnvironmentFile=-${ENV_FILE}#" \
      -e 's#^WantedBy=.*#WantedBy=default.target#' \
      -e 's#^After=network-online.target#After=network.target#' \
      -e '/^Wants=network-online.target/d' \
      "$RENDERED" > "$UNIT_DIR/aws-capsolver.service"
  systemctl --user daemon-reload
  systemctl --user enable --now aws-capsolver.service
  log "user unit installed: $UNIT_DIR/aws-capsolver.service (env: $ENV_FILE)"
  echo "    status: systemctl --user status aws-capsolver; logs: journalctl --user -u aws-capsolver -f"
  echo "    to start it at boot without logging in, run once:  sudo loginctl enable-linger $RUN_USER"
else
  log "rendered system unit: $RENDERED"
  cat <<STEPS

Run these as root to install the system service (not executed by this script):

  sudo install -m 0644 "$RENDERED" /etc/systemd/system/aws-capsolver.service
  sudo touch /etc/default/aws-capsolver        # optional AWSCS_* overrides, one per line
  sudo systemctl daemon-reload
  sudo systemctl enable --now aws-capsolver
  systemctl status aws-capsolver; journalctl -u aws-capsolver -f

STEPS
fi

log "health check once running:  curl -s http://${HOST}:${PORT}/health"
