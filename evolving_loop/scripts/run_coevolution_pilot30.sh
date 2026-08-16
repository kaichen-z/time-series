#!/usr/bin/env bash
# Frozen first-stage protocol for complete LLM-only three-agent co-evolution.
#
# Usage:
#   evolving_loop/scripts/run_coevolution_pilot30.sh /path/to/Dr-CiK_public/tasks
#
# Override any EA_* value when scaling beyond the first pilot. The default run
# is intentionally one generation with four screened children and one promotion.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASKS_FILE="${1:-${EA_TASKS_FILE:-}}"
if [[ -z "$TASKS_FILE" ]]; then
    echo "Usage: $0 TASKS_FILE" >&2
    exit 2
fi

export EA_RUNS_DIR="${EA_RUNS_DIR:-$SCRIPT_DIR/../../runs/coevolution_pilot30}"
export EA_GENERATIONS="${EA_GENERATIONS:-1}"
export EA_CHILDREN="${EA_CHILDREN:-4}"
export EA_LIMIT="${EA_LIMIT:-30}"
export EA_EVOLVE_TARGET="${EA_EVOLVE_TARGET:-auto}"
export EA_DEV_FRACTION="${EA_DEV_FRACTION:-0.20}"
export EA_HOLDOUT_FRACTION="${EA_HOLDOUT_FRACTION:-0.20}"
export EA_SEED="${EA_SEED:-7}"
export EA_SUCCESSIVE_HALVING="${EA_SUCCESSIVE_HALVING:-1}"
export EA_SCREEN_TRAIN_TASKS="${EA_SCREEN_TRAIN_TASKS:-6}"
export EA_SCREEN_DEV_TASKS="${EA_SCREEN_DEV_TASKS:-2}"
export EA_SCREEN_PROMOTE="${EA_SCREEN_PROMOTE:-1}"
export EA_SCREEN_TOLERANCE="${EA_SCREEN_TOLERANCE:-0.01}"

exec "$SCRIPT_DIR/run_llm_only_evolutions.sh" "$TASKS_FILE" genome
