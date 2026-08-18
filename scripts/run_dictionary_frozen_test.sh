#!/usr/bin/env bash
# Evaluate one frozen numerical dictionary on the sealed 99-task Public Test split.
# This command never invokes an LLM and never writes evolution artifacts.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

NA_TASKS_FILE="${NA_TASKS_FILE:-/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK/data/tasks/train.jsonl}"
NA_SPLIT_FILE="${NA_SPLIT_FILE:-$REPO_ROOT/splits/drcik_public_80_20_99_v1.json}"
NA_RUNS_DIR="${NA_RUNS_DIR:-$EA_RUNS_DIR/dictionary_curation}"
NA_EXPERIMENT_CONFIG="${NA_EXPERIMENT_CONFIG:-$NA_RUNS_DIR/full/experiment.json}"
NA_DICTIONARY="${NA_DICTIONARY:-$NA_RUNS_DIR/full/working_dictionary.json}"
NA_FROZEN_OUTPUT_DIR="${NA_FROZEN_OUTPUT_DIR:-$NA_RUNS_DIR/frozen_public_test}"
NA_DRY_RUN="${NA_DRY_RUN:-0}"

die() {
    echo "error: $*" >&2
    exit 2
}

[[ -f "$NA_TASKS_FILE" ]] || die "tasks file does not exist: $NA_TASKS_FILE"
[[ -f "$NA_SPLIT_FILE" ]] || die "split file does not exist: $NA_SPLIT_FILE"
[[ -f "$NA_EXPERIMENT_CONFIG" ]] || die "experiment config does not exist: $NA_EXPERIMENT_CONFIG"
[[ -f "$NA_DICTIONARY" ]] || die "working dictionary does not exist: $NA_DICTIONARY"

COMMAND=(
    "$PYTHON" -m numerical_agent evaluate-frozen
    --tasks-file "$NA_TASKS_FILE"
    --split-file "$NA_SPLIT_FILE"
    --experiment-config "$NA_EXPERIMENT_CONFIG"
    --dictionary "$NA_DICTIONARY"
    --output-dir "$NA_FROZEN_OUTPUT_DIR"
)

cat <<EOF
Frozen dictionary evaluation
  split:       $NA_SPLIT_FILE
  dictionary:  $NA_DICTIONARY
  output:      $NA_FROZEN_OUTPUT_DIR
  LLM calls:   none
  write-back:  disabled
  dry run:     $NA_DRY_RUN
EOF

if [[ "$NA_DRY_RUN" == "1" ]]; then
    printf '  '
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

"${COMMAND[@]}"
