#!/usr/bin/env bash
# Run the coding-skill baseline for one mode.
# Usage: ./run_baseline.sh <library|fresh> [extra baseline.py args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

MODE="${1:?usage: run_baseline.sh <library|fresh> [--limit N] [extra baseline.py args...]}"
shift || true

if [[ "$MODE" != "library" && "$MODE" != "fresh" ]]; then
    echo "error: mode must be 'library' or 'fresh', got '$MODE'" >&2
    exit 1
fi

RUN_DIR="$(ea_run_dir "$MODE")"
mkdir -p "$RUN_DIR"

ARGS=(
    --mode "$MODE"
    --seed "$EA_SEED"
    --library-path "$RUN_DIR/skill_library.json"
    --results-path "$RUN_DIR/results.jsonl"
    --log-file "$RUN_DIR/run.log"
)
[[ -n "$EA_LIMIT" ]] && ARGS+=(--limit "$EA_LIMIT")
[[ -n "$EA_MODEL_ID" ]] && ARGS+=(--model-id "$EA_MODEL_ID")
[[ -n "$EA_DEVICE" ]] && ARGS+=(--device "$EA_DEVICE")
[[ -n "$EA_TASKS_FILE" ]] && ARGS+=(--tasks-file "$EA_TASKS_FILE")

cd "$REPO_ROOT"
echo "Running baseline (mode=$MODE) -> $RUN_DIR"
"$PYTHON" -m evolving_agent.coding_agent.baseline "${ARGS[@]}" "$@"
