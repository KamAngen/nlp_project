#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"
KAGGLE_ARTIFACT_DATASET="${KAGGLE_ARTIFACT_DATASET:-angennn/nlp-chinese-law-agent}"
KAGGLE_TMP_DIR="${KAGGLE_TMP_DIR:-$PROJECT_ROOT/data/.downloads/kaggle_artifacts}"
TARGET_ARTIFACT_ROOT="${TARGET_ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}"
KAGGLE_BIN="${KAGGLE_BIN:-$(resolve_env_tool kaggle)}"

if ! "$KAGGLE_BIN" --help >/dev/null 2>&1; then
  echo "[error] 未找到 kaggle 命令。请先在当前环境安装 kaggle，并配置 ~/.kaggle/kaggle.json。" >&2
  exit 2
fi

mkdir -p "$KAGGLE_TMP_DIR" "$TARGET_ARTIFACT_ROOT"
echo "[artifacts] downloading Kaggle dataset: $KAGGLE_ARTIFACT_DATASET"
"$KAGGLE_BIN" datasets download -d "$KAGGLE_ARTIFACT_DATASET" -p "$KAGGLE_TMP_DIR" --unzip

artifact_path="$(find "$KAGGLE_TMP_DIR" -path '*/artifacts/rag/metadata.json' | head -n 1)"
if [ -z "$artifact_path" ]; then
  artifact_path="$(find "$KAGGLE_TMP_DIR" -path '*/rag/metadata.json' | head -n 1)"
fi
if [ -z "$artifact_path" ]; then
  echo "[error] Kaggle 数据集中未找到可识别的 artifacts/rag/metadata.json。" >&2
  exit 3
fi

extracted_root="$(cd "$(dirname "$artifact_path")/.." && pwd)"
if [ -d "$extracted_root/artifacts" ]; then
  cp -a "$extracted_root/artifacts"/. "$TARGET_ARTIFACT_ROOT"/
else
  cp -a "$extracted_root"/. "$TARGET_ARTIFACT_ROOT"/
fi

echo "[done] artifacts downloaded into $TARGET_ARTIFACT_ROOT"