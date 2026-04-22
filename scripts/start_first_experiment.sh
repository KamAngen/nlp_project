#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

bash "$PROJECT_ROOT/scripts/download_disc_law.sh"
bash "$PROJECT_ROOT/scripts/build_knowledge_base.sh"
bash "$PROJECT_ROOT/scripts/generate_agent_data.sh"
