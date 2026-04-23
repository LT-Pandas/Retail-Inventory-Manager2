#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="retail-inventory-manager.service"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN_DEFAULT="$REPO_DIR/mp-venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
RUN_ARGS="${RUN_ARGS:---headless --button-controlled --button-gpio-pin 17}"
TARGET_USER="${TARGET_USER:-${SUDO_USER:-$USER}}"
TARGET_GROUP="${TARGET_GROUP:-$TARGET_USER}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not found or not executable: $PYTHON_BIN" >&2
  echo "Tip: set PYTHON_BIN=/path/to/python before running this installer." >&2
  exit 1
fi

TMP_FILE="$(mktemp)"
cat > "$TMP_FILE" <<UNIT
[Unit]
Description=Retail Inventory Manager CV service
After=network-online.target bluetooth.target
Wants=network-online.target

[Service]
Type=simple
User=$TARGET_USER
Group=$TARGET_GROUP
WorkingDirectory=$REPO_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON_BIN $REPO_DIR/run_cv.py $RUN_ARGS
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo cp "$TMP_FILE" "/etc/systemd/system/$SERVICE_NAME"
rm -f "$TMP_FILE"

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "Installed and started $SERVICE_NAME"
echo "Check status: sudo systemctl status $SERVICE_NAME"
echo "View logs:    journalctl -u $SERVICE_NAME -f"
