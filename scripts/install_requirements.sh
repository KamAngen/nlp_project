#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
# On Mac, we don't need the CUDA-specific index. Standard pip install is better.
"$PYTHON_BIN" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
"$PYTHON_BIN" -m pip install -e . "pytest>=8.3.0"
