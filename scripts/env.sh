#!/usr/bin/env bash
# Shared configuration for evolving_loop/coding_agent scripts. Sourced, not executed directly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python3}"

# Where every run's library/results/log files live, one subdirectory per mode.
EA_RUNS_DIR="${EA_RUNS_DIR:-$REPO_ROOT/runs}"

# Overridable knobs -- leave unset to use baseline.py's own defaults.
EA_SEED="${EA_SEED:-0}"
EA_LIMIT="${EA_LIMIT:-}"
EA_MODEL_ID="${EA_MODEL_ID:-}"
EA_DEVICE="${EA_DEVICE:-}"
EA_TASKS_FILE="${EA_TASKS_FILE:-}"

ea_run_dir() {
    # One directory per mode, e.g. runs/library/, runs/fresh/ -- keeps the two conditions' artifacts apart.
    local mode="$1"
    echo "$EA_RUNS_DIR/$mode"
}
