#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-configs/defaults.yaml}"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
PREPARED_LAW_DIR="${PREPARED_LAW_DIR:-$PROJECT_ROOT/data/law_files}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}"
KAGGLE_ARTIFACT_DATASET="${KAGGLE_ARTIFACT_DATASET:-angennn/nlp-chinese-law-agent}"
KAGGLE_TMP_DIR="${KAGGLE_TMP_DIR:-$PROJECT_ROOT/data/.downloads/kaggle_artifacts}"

has_prepared_law_data() {
	[ -f "$PREPARED_LAW_DIR/catalogs/law_catalog_master.csv" ] && [ -d "$PREPARED_LAW_DIR/files" ]
}

has_built_artifacts() {
	[ -f "$ARTIFACT_ROOT/law_chunks.jsonl" ] && [ -f "$ARTIFACT_ROOT/rag/metadata.json" ]
}

download_artifacts_from_kaggle() {
	local kaggle_bin
	kaggle_bin="${KAGGLE_BIN:-$(resolve_env_tool kaggle)}"

	if ! "$kaggle_bin" --help >/dev/null 2>&1; then
		echo "[error] 未找到 kaggle 命令。请先在当前环境安装 kaggle，并配置 ~/.kaggle/kaggle.json。" >&2
		exit 2
	fi

	mkdir -p "$KAGGLE_TMP_DIR"
	echo "[artifacts] downloading built RAG artifacts from Kaggle: $KAGGLE_ARTIFACT_DATASET"
	"$kaggle_bin" datasets download -d "$KAGGLE_ARTIFACT_DATASET" -p "$KAGGLE_TMP_DIR" --unzip

	local artifact_path
	artifact_path="$(find "$KAGGLE_TMP_DIR" -path '*/artifacts/rag/metadata.json' | head -n 1)"
	if [ -z "$artifact_path" ]; then
		artifact_path="$(find "$KAGGLE_TMP_DIR" -path '*/rag/metadata.json' | head -n 1)"
	fi
	if [ -z "$artifact_path" ]; then
		echo "[error] Kaggle 数据集中未找到可识别的 artifacts/rag/metadata.json。" >&2
		exit 4
	fi

	local extracted_root
	extracted_root="$(cd "$(dirname "$artifact_path")/.." && pwd)"
	mkdir -p "$ARTIFACT_ROOT"
	if [ -d "$extracted_root/artifacts" ]; then
		cp -a "$extracted_root/artifacts"/. "$ARTIFACT_ROOT"/
	else
		cp -a "$extracted_root"/. "$ARTIFACT_ROOT"/
	fi
}

cd "$PROJECT_ROOT"
if has_prepared_law_data; then
	"$PYTHON_BIN" -m legal_agent.cli repair-law-docs --config "$CONFIG_PATH" --law-dir "$PREPARED_LAW_DIR"
	"$PYTHON_BIN" -m legal_agent.cli build-corpus --config "$CONFIG_PATH"
	"$PYTHON_BIN" -m legal_agent.cli build-index --config "$CONFIG_PATH" --embedding-device "$EMBEDDING_DEVICE"
elif ! has_built_artifacts; then
	download_artifacts_from_kaggle
fi
