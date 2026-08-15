#!/usr/bin/env bash
# Evaluate two frozen Harness Genomes and two Codex baselines on the same
# entity-disjoint fresh 30-task Dr-CiK manifest.
#
# Usage:
#   evolving_agent/scripts/run_fresh30_four_method_eval.sh smoke
#   evolving_agent/scripts/run_fresh30_four_method_eval.sh full
#
# `smoke` runs only the first manifest task. `full` runs all 30. Actual runs
# execute two methods at a time and retry process failures without converting
# transient Codex/network errors into forecasting scores.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-python3}"
MODE="${1:-smoke}"
EA_DRY_RUN="${EA_DRY_RUN:-0}"
EA_EXPERIMENT_ROOT="${EA_EXPERIMENT_ROOT:-$REPO_ROOT/runs/fresh30_four_method_20260815}"
EA_EVAL_ROOT="${EA_EVAL_ROOT:-$EA_EXPERIMENT_ROOT}"
EA_MANIFEST="${EA_MANIFEST:-$EA_EXPERIMENT_ROOT/manifests/fresh30.json}"
EA_TASKS_FILE="${EA_TASKS_FILE:-$REPO_ROOT/external/Dr-CiK/full-download/Dr-CiK_public/tasks}"
EA_SAMPLE_ROOT="${EA_SAMPLE_ROOT:-$(dirname "$EA_TASKS_FILE")}"
EA_V000_POLICY="${EA_V000_POLICY:-$EA_EXPERIMENT_ROOT/policies/retry2_v000.json}"
EA_V003_POLICY="${EA_V003_POLICY:-$EA_EXPERIMENT_ROOT/policies/retry2_v003.json}"
EA_CODEX_MODEL="${EA_CODEX_MODEL:-gpt-5.6-sol}"
EA_REASONING_EFFORT="${EA_REASONING_EFFORT:-high}"
EA_CODEX_TIMEOUT="${EA_CODEX_TIMEOUT:-900}"
EA_SAMPLES="${EA_SAMPLES:-25}"
EA_MAX_ATTEMPTS="${EA_MAX_ATTEMPTS:-3}"
EA_CHRONOS_MODEL="${EA_CHRONOS_MODEL:-amazon/chronos-bolt-small}"

die() {
    echo "error: $*" >&2
    exit 2
}

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
    die "mode must be smoke or full; got '$MODE'"
fi
if [[ "$EA_DRY_RUN" != "0" && "$EA_DRY_RUN" != "1" ]]; then
    die "EA_DRY_RUN must be 0 or 1"
fi

for required in "$EA_MANIFEST" "$EA_V000_POLICY" "$EA_V003_POLICY"; do
    [[ -f "$required" ]] || die "required artifact does not exist: $required"
done
if [[ "$EA_DRY_RUN" != "1" && ! -e "$EA_TASKS_FILE" ]]; then
    die "Dr-CiK task path does not exist: $EA_TASKS_FILE"
fi

TASK_IDS=()
while IFS= read -r task_id; do
    [[ -n "$task_id" ]] && TASK_IDS+=("$task_id")
done < <(
    "$PYTHON" - "$EA_MANIFEST" "$MODE" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
task_ids = [task["benchmark_id"] for task in manifest["tasks"]]
if sys.argv[2] == "smoke":
    task_ids = task_ids[:1]
print("\n".join(task_ids))
PY
)
[[ "${#TASK_IDS[@]}" -gt 0 ]] || die "manifest selected no tasks"

TASK_ARGS=()
for task_id in "${TASK_IDS[@]}"; do
    TASK_ARGS+=(--task-id "$task_id")
done
EXPECTED_TASKS="${#TASK_IDS[@]}"

OUTPUT_ROOT="$EA_EVAL_ROOT/outputs/$MODE"
LOG_ROOT="$EA_EVAL_ROOT/logs/$MODE"
CACHE_ROOT="$EA_EVAL_ROOT/cache"
STATE_ROOT="$EA_EVAL_ROOT/state"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CACHE_ROOT" "$STATE_ROOT"

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

method_command() {
    local method="$1"
    local output_dir="$OUTPUT_ROOT/$method"
    local cache_dir="$CACHE_ROOT/$method"
    local state_dir="$STATE_ROOT/$method"
    mkdir -p "$output_dir" "$cache_dir" "$state_dir"

    case "$method" in
        retry2_v000|retry2_v003)
            local policy="$EA_V000_POLICY"
            [[ "$method" == "retry2_v003" ]] && policy="$EA_V003_POLICY"
            METHOD_COMMAND=(
                "$PYTHON" -m evolving_agent
                --inference genome
                --tasks-file "$EA_TASKS_FILE"
                "${TASK_ARGS[@]}"
                --policy-path "$policy"
                --output-dir "$output_dir"
                --score-public
                --samples "$EA_SAMPLES"
                --setting llm_only
                --llm-backend codex
                --codex-model "$EA_CODEX_MODEL"
                --codex-reasoning-effort "$EA_REASONING_EFFORT"
                --codex-timeout "$EA_CODEX_TIMEOUT"
                --codex-cache-dir "$cache_dir"
                --library-path "$state_dir/coding_skills.json"
                --retrieval-library-path "$state_dir/retrieval_skills.json"
                --decision-library-path "$state_dir/decision_skills.json"
            )
            ;;
        codex_direct|codex_contract)
            local baseline="codex-direct"
            [[ "$method" == "codex_contract" ]] && baseline="codex-contract"
            METHOD_COMMAND=(
                "$PYTHON" -m evolving_agent
                --baseline "$baseline"
                --sample-dir "$EA_SAMPLE_ROOT"
                "${TASK_ARGS[@]}"
                --output-dir "$output_dir"
                --samples "$EA_SAMPLES"
                --seed 7
                --backbone chronos
                --chronos-model-id "$EA_CHRONOS_MODEL"
                --chronos-local-files-only
                --codex-model "$EA_CODEX_MODEL"
                --codex-reasoning-effort "$EA_REASONING_EFFORT"
                --codex-timeout "$EA_CODEX_TIMEOUT"
                --codex-cache-dir "$cache_dir"
            )
            ;;
        *)
            die "unknown method: $method"
            ;;
    esac
}

is_complete() {
    local summary="$OUTPUT_ROOT/$1/summary.json"
    [[ -f "$summary" ]] || return 1
    "$PYTHON" - "$summary" "$EXPECTED_TASKS" <<'PY' >/dev/null 2>&1
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if summary.get("num_tasks") == int(sys.argv[2]) else 1)
PY
}

run_method() {
    local method="$1"
    local log="$LOG_ROOT/$method.log"
    local attempt=1
    if is_complete "$method"; then
        echo "[$method] already complete ($EXPECTED_TASKS tasks); skipping"
        return 0
    fi
    method_command "$method"
    if [[ "$EA_DRY_RUN" == "1" ]]; then
        echo "[$method] dry run"
        print_command "${METHOD_COMMAND[@]}"
        return 0
    fi
    while [[ "$attempt" -le "$EA_MAX_ATTEMPTS" ]]; do
        {
            echo "[$method] attempt $attempt/$EA_MAX_ATTEMPTS started $(date -u +%FT%TZ)"
            print_command "${METHOD_COMMAND[@]}"
        } >>"$log"
        if "${METHOD_COMMAND[@]}" >>"$log" 2>&1 && is_complete "$method"; then
            echo "[$method] complete"
            echo "[$method] completed $(date -u +%FT%TZ)" >>"$log"
            return 0
        fi
        echo "[$method] attempt $attempt failed; retrying without scoring the failure" | tee -a "$log" >&2
        attempt=$((attempt + 1))
    done
    echo "[$method] failed after $EA_MAX_ATTEMPTS attempts; see $log" >&2
    return 1
}

run_wave() {
    local first="$1"
    local second="$2"
    local first_pid second_pid first_status second_status
    run_method "$first" &
    first_pid=$!
    run_method "$second" &
    second_pid=$!
    wait "$first_pid"; first_status=$?
    wait "$second_pid"; second_status=$?
    [[ "$first_status" -eq 0 && "$second_status" -eq 0 ]]
}

cat <<EOF
Fresh-30 four-method evaluation
  mode:              $MODE
  manifest:          $EA_MANIFEST
  tasks:             $EXPECTED_TASKS
  model:             $EA_CODEX_MODEL ($EA_REASONING_EFFORT)
  trajectories:      $EA_SAMPLES
  method concurrency: 2
  output root:       $OUTPUT_ROOT
  dry run:           $EA_DRY_RUN
EOF

cd "$REPO_ROOT"
if [[ "$EA_DRY_RUN" == "1" ]]; then
    run_method retry2_v000
    run_method retry2_v003
    run_method codex_direct
    run_method codex_contract
elif run_wave retry2_v000 retry2_v003 && run_wave codex_direct codex_contract; then
    echo "All four methods completed for mode=$MODE."
else
    echo "At least one method did not complete for mode=$MODE." >&2
    exit 1
fi
