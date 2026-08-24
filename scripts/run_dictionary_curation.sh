#!/usr/bin/env bash
# Run Phase 1 dictionary curation: an LLM implements each statistical method, the
# sandbox runs it, and measured Train/Dev sMAE decides what is kept.
#
# Usage:
#   scripts/run_dictionary_curation.sh smoke
#   scripts/run_dictionary_curation.sh full
#
# `smoke` curates against a handful of tasks to check the wiring end to end.
# `full` uses the frozen entity-disjoint 80 Train / 20 Dev public split.
# This spends real LLM calls: one per unimplemented method per generation.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

MODE="${1:-smoke}"

NA_TASKS_FILE="${NA_TASKS_FILE:-/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK/data/tasks/train.jsonl}"
NA_SPLIT_FILE="${NA_SPLIT_FILE:-$REPO_ROOT/splits/drcik_public_80_20_99_v1.json}"
NA_BASE_METHODS="${NA_BASE_METHODS:-$REPO_ROOT/numerical_agent/datasets/forecast_method_dataset_v001.json}"
NA_RUNS_DIR="${NA_RUNS_DIR:-$EA_RUNS_DIR/dictionary_curation}"
NA_LLM_BACKEND="${NA_LLM_BACKEND:-qwen}"
NA_CODEX_MODEL="${NA_CODEX_MODEL:-gpt-5.6-sol}"
NA_REASONING_EFFORT="${NA_REASONING_EFFORT:-high}"
NA_CODEX_TIMEOUT="${NA_CODEX_TIMEOUT:-900}"
NA_MODEL_ID="${NA_MODEL_ID:-Qwen/Qwen3.5-27B}"
NA_DEVICE="${NA_DEVICE:-}"
NA_GENERATIONS="${NA_GENERATIONS:-3}"
# Child prompts now receive distinct implementation objectives. Keep one child as the
# conservative default; raise NA_CHILDREN when the compute budget permits real search.
NA_CHILDREN="${NA_CHILDREN:-1}"
NA_MAX_REVISIONS="${NA_MAX_REVISIONS:-1}"
NA_MAX_IMPLEMENTATION_ATTEMPTS="${NA_MAX_IMPLEMENTATION_ATTEMPTS:-3}"
NA_ACCEPTED_MAX_ERROR="${NA_ACCEPTED_MAX_ERROR:-50.0}"
NA_SPECIALIZED_MAX_ERROR="${NA_SPECIALIZED_MAX_ERROR:-100.0}"
NA_MIN_SUCCESS_RATE="${NA_MIN_SUCCESS_RATE:-0.8}"
NA_SELECTION_FOLDS="${NA_SELECTION_FOLDS:-3}"
NA_SELECTION_HORIZON="${NA_SELECTION_HORIZON:-8}"
NA_DRY_RUN="${NA_DRY_RUN:-0}"

die() {
    echo "error: $*" >&2
    exit 2
}

case "$MODE" in
    smoke) NA_TRAIN_LIMIT="${NA_TRAIN_LIMIT:-4}"; NA_DEV_LIMIT="${NA_DEV_LIMIT:-2}" ;;
    full)  NA_TRAIN_LIMIT="${NA_TRAIN_LIMIT:-0}"; NA_DEV_LIMIT="${NA_DEV_LIMIT:-0}" ;;
    *)     die "mode must be smoke or full; got '$MODE'" ;;
esac

[[ -f "$NA_SPLIT_FILE" ]] || die "split file does not exist: $NA_SPLIT_FILE"
[[ -f "$NA_BASE_METHODS" ]] || die "base method dictionary does not exist: $NA_BASE_METHODS"
[[ -f "$NA_TASKS_FILE" ]] || die "Dr-CiK tasks file does not exist: $NA_TASKS_FILE"

OUTPUT_ROOT="$NA_RUNS_DIR/$MODE"
EXPERIMENT_CONFIG="$OUTPUT_ROOT/experiment.json"
mkdir -p "$OUTPUT_ROOT"

BUILD_COMMAND=(
    "$PYTHON" -m numerical_agent build-experiment
    --tasks-file "$NA_TASKS_FILE"
    --split-file "$NA_SPLIT_FILE"
    --output "$EXPERIMENT_CONFIG"
    --generations "$NA_GENERATIONS"
    --children-per-generation "$NA_CHILDREN"
    --max-revisions-per-method "$NA_MAX_REVISIONS"
    --max-implementation-attempts "$NA_MAX_IMPLEMENTATION_ATTEMPTS"
    --accepted-max-error "$NA_ACCEPTED_MAX_ERROR"
    --specialized-max-error "$NA_SPECIALIZED_MAX_ERROR"
    --min-success-rate "$NA_MIN_SUCCESS_RATE"
    --selection-folds "$NA_SELECTION_FOLDS"
    --selection-horizon "$NA_SELECTION_HORIZON"
)
[[ "$NA_TRAIN_LIMIT" != "0" ]] && BUILD_COMMAND+=(--train-limit "$NA_TRAIN_LIMIT")
[[ "$NA_DEV_LIMIT" != "0" ]] && BUILD_COMMAND+=(--dev-limit "$NA_DEV_LIMIT")

CURATE_COMMAND=(
    "$PYTHON" -m numerical_agent curate
    --experiment-config "$EXPERIMENT_CONFIG"
    --base-methods "$NA_BASE_METHODS"
    --provider llm
    --llm-backend "$NA_LLM_BACKEND"
    --output-dir "$OUTPUT_ROOT"
)
if [[ "$NA_LLM_BACKEND" == "codex" ]]; then
    CURATE_COMMAND+=(
        --codex-model "$NA_CODEX_MODEL"
        --codex-reasoning-effort "$NA_REASONING_EFFORT"
        --codex-timeout "$NA_CODEX_TIMEOUT"
        --codex-cache-dir "$OUTPUT_ROOT/codex-cache"
    )
elif [[ "$NA_LLM_BACKEND" == "qwen" ]]; then
    CURATE_COMMAND+=(--model-id "$NA_MODEL_ID")
    [[ -n "$NA_DEVICE" ]] && CURATE_COMMAND+=(--device "$NA_DEVICE")
fi

METHOD_COUNT="$("$PYTHON" -c "import json,sys; methods=json.load(open(sys.argv[1]))['methods']; print(sum((m.get('family') or m.get('definition', {}).get('family')) == 'statistical' for m in methods))" "$NA_BASE_METHODS")"

cat <<EOF
Dictionary curation
  mode:          $MODE
  methods:       $METHOD_COUNT
  split:         $NA_SPLIT_FILE
  train limit:   ${NA_TRAIN_LIMIT/0/all (80)}
  dev limit:     ${NA_DEV_LIMIT/0/all (20)}
  backend:       $NA_LLM_BACKEND
  generations:   $NA_GENERATIONS (x $NA_CHILDREN child)
  llm calls:     up to $((METHOD_COUNT * NA_CHILDREN)) per generation
  output root:   $OUTPUT_ROOT
  dry run:       $NA_DRY_RUN
EOF

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

if [[ "$NA_DRY_RUN" == "1" ]]; then
    print_command "${BUILD_COMMAND[@]}"
    print_command "${CURATE_COMMAND[@]}"
    exit 0
fi

cd "$REPO_ROOT"
"${BUILD_COMMAND[@]}" || die "could not build the experiment config"
"${CURATE_COMMAND[@]}" | tee "$OUTPUT_ROOT/summary.json"
status="${PIPESTATUS[0]}"
[[ "$status" -eq 0 ]] || die "curation failed; see $OUTPUT_ROOT"

echo "Vetted dictionary: $OUTPUT_ROOT/working_dictionary.json"
echo "Per-method scores: $OUTPUT_ROOT/method_evaluations.jsonl"
echo "Generated code:    $OUTPUT_ROOT/best_artifact/methods/"
