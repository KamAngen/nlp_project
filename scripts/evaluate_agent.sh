#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"
RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-cuda:0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m legal_agent.cli evaluate \
  --config configs/defaults.yaml \
  --output-dir outputs/eval_base \
  --retrieval-device "$RETRIEVAL_DEVICE"

"$PYTHON_BIN" -m legal_agent.cli evaluate \
  --config configs/defaults.yaml \
  --adapter-path ckpt/unified_agent_qlora \
  --output-dir outputs/eval_adapter \
  --retrieval-device "$RETRIEVAL_DEVICE"

"$PYTHON_BIN" -m legal_agent.cli export-case-studies \
  --base-results outputs/eval_base/eval_details.jsonl \
  --adapted-results outputs/eval_adapter/eval_details.jsonl \
  --output outputs/eval_case_studies.md
