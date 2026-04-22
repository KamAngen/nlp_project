#!/usr/bin/env bash
set -euo pipefail

# 自动下载基础模型到 ./models 下的标准目录。
#
# 默认通过 huggingface_hub.snapshot_download 下载，支持配合 HF_ENDPOINT=https://hf-mirror.com
# 使用国内镜像。
#
# 示例：
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_models.sh
#   BGE_MODEL_ID=BAAI/bge-small-zh-v1.5 QWEN_MODEL_ID=Qwen/Qwen3-4B bash scripts/download_models.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"
MODELS_DIR="$PROJECT_ROOT/models"

EMB_DIR="$MODELS_DIR/embeddings/bge-small-zh"
QWEN_DIR="$MODELS_DIR/qwen/Qwen3_4B"

BGE_MODEL_ID="${BGE_MODEL_ID:-BAAI/bge-small-zh-v1.5}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-4B}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-}"

exists_nonempty_dir() {
  [ -d "$1" ] && [ "$(ls -A "$1")" ]
}

echo "Project root: $PROJECT_ROOT"
mkdir -p "$MODELS_DIR"

if exists_nonempty_dir "$EMB_DIR"; then
  echo "[skip] Embedding model already exists at $EMB_DIR"
else
  echo "Downloading BGE model '$BGE_MODEL_ID' -> $EMB_DIR"
  "$PYTHON_BIN" - <<PY || (echo "Python 下载失败，请确认 huggingface_hub 已安装并可用" >&2; exit 3)
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(r"$EMB_DIR")
target.mkdir(parents=True, exist_ok=True)
print('snapshot_download:', '$BGE_MODEL_ID', '->', target)
snapshot_download(
    repo_id="$BGE_MODEL_ID",
    local_dir=str(target),
    local_dir_use_symlinks=False,
    endpoint="$HF_ENDPOINT_VALUE" or None,
    resume_download=True,
)
print('done')
PY
fi

if exists_nonempty_dir "$QWEN_DIR"; then
  echo "[skip] Qwen model already exists at $QWEN_DIR"
else
  echo "Downloading QWEN model '$QWEN_MODEL_ID' -> $QWEN_DIR"
  "$PYTHON_BIN" - <<PY || (echo "Python 下载失败，请确认 huggingface_hub 已安装并可用" >&2; exit 5)
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(r"$QWEN_DIR")
target.mkdir(parents=True, exist_ok=True)
print('snapshot_download:', '$QWEN_MODEL_ID', '->', target)
snapshot_download(
    repo_id="$QWEN_MODEL_ID",
    local_dir=str(target),
    local_dir_use_symlinks=False,
    endpoint="$HF_ENDPOINT_VALUE" or None,
    resume_download=True,
)
print('done')
PY
fi

echo "完成：请检查 $MODELS_DIR 下是否包含 embeddings/bge-small-zh 与 qwen/Qwen3_4B。"
