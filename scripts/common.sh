#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENT_ENV_NAME="${AGENT_ENV_NAME:-agent_env}"

resolve_python_bin() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi

  if [ "${CONDA_DEFAULT_ENV:-}" = "$AGENT_ENV_NAME" ] && command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    local resolved
    resolved="$(conda run -n "$AGENT_ENV_NAME" python -c 'import sys; print(sys.executable)' 2>/dev/null | tail -n 1 || true)"
    if [ -n "$resolved" ]; then
      printf '%s\n' "$resolved"
      return 0
    fi
  fi

  cat >&2 <<EOF
[error] 无法解析 Python 解释器。
请先执行以下任一操作：
1. export AGENT_ENV_NAME=agent_env && conda activate "$AGENT_ENV_NAME"
2. export PYTHON_BIN=/path/to/python
EOF
  exit 127
}

resolve_env_tool() {
  local tool_name="$1"

  if [ -n "${PYTHON_BIN:-}" ]; then
    local sibling_tool
    sibling_tool="$(cd "$(dirname "$PYTHON_BIN")" && pwd)/$tool_name"
    if [ -x "$sibling_tool" ]; then
      printf '%s\n' "$sibling_tool"
      return 0
    fi
  fi

  if [ "${CONDA_DEFAULT_ENV:-}" = "$AGENT_ENV_NAME" ] && command -v "$tool_name" >/dev/null 2>&1; then
    command -v "$tool_name"
    return 0
  fi

  cat >&2 <<EOF
[error] 无法解析 ${tool_name} 可执行文件。
请确认它已安装在环境 "$AGENT_ENV_NAME" 中，或手动把它加入 PATH。
EOF
  exit 127
}

PYTHON_BIN="$(resolve_python_bin)"
export SCRIPT_DIR PROJECT_ROOT AGENT_ENV_NAME PYTHON_BIN
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"