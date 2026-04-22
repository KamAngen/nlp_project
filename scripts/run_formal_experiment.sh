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

if [ ! -f "$PROJECT_ROOT/data/disc_law/disc_law_normalized.jsonl" ]; then
  echo "[1/7] Download and normalize DISC-Law-SFT"
  "$PYTHON_BIN" -m legal_agent.cli download-disc-law --config configs/defaults.yaml
else
  echo "[1/7] DISC-Law-SFT already prepared; skip download"
fi

echo "[2/7] Build legal knowledge base"
bash "$SCRIPT_DIR/build_knowledge_base.sh"

echo "[3/7] Build formal datasets"
"$PYTHON_BIN" -m legal_agent.cli build-data \
  --config configs/defaults.yaml \
  --study-config configs/study_agent.yaml \
  --retrieval-device "$RETRIEVAL_DEVICE"

echo "[4/7] Train formal adapter"
"$PYTHON_BIN" -m legal_agent.cli train --config configs/defaults.yaml

echo "[5/7] Evaluate base model"
"$PYTHON_BIN" -m legal_agent.cli evaluate \
  --config configs/defaults.yaml \
  --output-dir outputs/eval_base \
  --retrieval-device "$RETRIEVAL_DEVICE"

echo "[6/7] Evaluate post-trained adapter"
"$PYTHON_BIN" -m legal_agent.cli evaluate \
  --config configs/defaults.yaml \
  --adapter-path ckpt/unified_agent_qlora \
  --output-dir outputs/eval_adapter \
  --retrieval-device "$RETRIEVAL_DEVICE"

echo "[7/7] Export side-by-side case studies and reports"
"$PYTHON_BIN" -m legal_agent.cli export-case-studies \
  --base-results outputs/eval_base/eval_details.jsonl \
  --adapted-results outputs/eval_adapter/eval_details.jsonl \
  --output outputs/eval_case_studies.md \
  --max-examples 3

mkdir -p outputs/representative_examples
cp outputs/eval_case_studies.md outputs/representative_examples/README.md
"$PYTHON_BIN" scripts/generate_report_docs.py

echo "Formal experiment completed. Representative examples: outputs/representative_examples/README.md"