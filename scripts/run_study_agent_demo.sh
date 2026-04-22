#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

cd "${PROJECT_ROOT}"
"$PYTHON_BIN" -m legal_agent.cli chat \
	--config configs/defaults.yaml \
	--study-config configs/study_agent.yaml \
	"$@"