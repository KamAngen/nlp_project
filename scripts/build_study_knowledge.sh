#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"
APP_CONFIG_PATH="${APP_CONFIG_PATH:-configs/defaults.yaml}"
STUDY_CONFIG_PATH="${STUDY_CONFIG_PATH:-configs/study_agent.yaml}"
RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-cpu}"
QUESTION_COUNT="${QUESTION_COUNT:-180}"
CASE_COUNT="${CASE_COUNT:-96}"
COMMON_COUNT="${COMMON_COUNT:-24}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
AUTO_DOWNLOAD_DISC_LAW="${AUTO_DOWNLOAD_DISC_LAW:-0}"

cd "${PROJECT_ROOT}"
cmd=(
	"$PYTHON_BIN" -m legal_agent.cli build-study-kb
	--config "$STUDY_CONFIG_PATH"
	--app-config "$APP_CONFIG_PATH"
	--retrieval-device "$RETRIEVAL_DEVICE"
	--question-count "$QUESTION_COUNT"
	--case-count "$CASE_COUNT"
	--common-count "$COMMON_COUNT"
)

if [ "$FORCE_REBUILD" = "1" ]; then
	cmd+=(--force-rebuild)
fi

if [ "$AUTO_DOWNLOAD_DISC_LAW" = "1" ]; then
	cmd+=(--auto-download-disc-law)
fi

"${cmd[@]}" "$@"