#!/usr/bin/env bash
# Compatibility entry point. The canonical evolving-agent runners live beside
# the package under evolving_loop/scripts/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec "$REPO_ROOT/evolving_loop/scripts/run_llm_only_evolutions.sh" "$@"
