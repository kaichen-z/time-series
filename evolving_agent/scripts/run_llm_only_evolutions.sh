#!/usr/bin/env bash
# Compatibility entrypoint requested for evolving_agent/scripts/. The maintained
# implementation follows the repository's current script layout at scripts/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../../scripts/run_llm_only_evolutions.sh" "$@"
