#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"
export HF_ENDPOINT="https://hf-mirror.com"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m legal_agent.cli download-disc-law --config configs/defaults.yaml
