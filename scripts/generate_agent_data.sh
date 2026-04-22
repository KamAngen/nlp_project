#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"
TRAIN_COUNT="${TRAIN_COUNT:-1600}"
EVAL_COUNT="${EVAL_COUNT:-200}"
RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-cuda:0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m legal_agent.cli build-data \
  --config configs/defaults.yaml \
  --study-config configs/study_agent.yaml \
  --train-count "$TRAIN_COUNT" \
  --eval-count "$EVAL_COUNT" \
  --retrieval-device "$RETRIEVAL_DEVICE"
