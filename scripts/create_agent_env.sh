#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENT_ENV_NAME="${AGENT_ENV_NAME:-agent_env}"

cd "${PROJECT_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] 未找到 conda，请先安装 Conda 或 Mamba。" >&2
  exit 127
fi

if conda env list | awk '{print $1}' | grep -qx "$AGENT_ENV_NAME"; then
  conda env update -n "$AGENT_ENV_NAME" -f environment.agent_env.yml --prune
else
  conda env create -n "$AGENT_ENV_NAME" -f environment.agent_env.yml
fi

AGENT_ENV_NAME="$AGENT_ENV_NAME" bash "$SCRIPT_DIR/install_requirements.sh"