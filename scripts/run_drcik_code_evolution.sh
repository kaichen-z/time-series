#!/usr/bin/env bash
# Run one numbers-only Dr-CiK Coding Agent evolution (drcik_agent.run_code_evolution).
# Usage: ./run_drcik_code_evolution.sh --sample-dir <path> --output <path> [more args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

cd "$REPO_ROOT"
"$PYTHON" -m drcik_agent.run_code_evolution "$@"
