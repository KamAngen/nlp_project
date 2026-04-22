#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"
RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-cpu}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m legal_agent.cli web-ui \
  --config configs/web_ui.yaml \
  --study-config configs/study_agent.yaml \
  --host "$HOST" \
  --port "$PORT" \
  --retrieval-device "$RETRIEVAL_DEVICE"
