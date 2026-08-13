#!/usr/bin/env bash
# Run the prompt, genome, and source evolution experiments under one controlled
# LLM-only protocol.
#
# Usage:
#   evolving_agent/scripts/run_llm_only_evolutions.sh /path/to/Dr-CiK_public/tasks [all|prompt|genome|source]
#
# Important environment overrides:
#   EA_RUNS_DIR, EA_CODEX_MODEL, EA_GENERATIONS, EA_CHILDREN, EA_SEED,
#   EA_DEV_FRACTION, EA_HOLDOUT_FRACTION, EA_LIMIT, EA_DRY_RUN
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-python3}"
TASKS_FILE="${1:-${EA_TASKS_FILE:-}}"
MODE="${2:-all}"

EA_RUNS_DIR="${EA_RUNS_DIR:-$REPO_ROOT/runs/llm_only_evolution}"
EA_CODEX_MODEL="${EA_CODEX_MODEL:-gpt-5.6-sol}"
EA_REASONING_EFFORT="${EA_REASONING_EFFORT:-high}"
EA_GENERATIONS="${EA_GENERATIONS:-2}"
EA_CHILDREN="${EA_CHILDREN:-3}"
EA_SEED="${EA_SEED:-7}"
EA_DEV_FRACTION="${EA_DEV_FRACTION:-0.25}"
EA_HOLDOUT_FRACTION="${EA_HOLDOUT_FRACTION:-0.20}"
EA_LIMIT="${EA_LIMIT:-}"
EA_DRY_RUN="${EA_DRY_RUN:-0}"
EA_CODEX_TIMEOUT="${EA_CODEX_TIMEOUT:-900}"
EA_SOURCE_ENGINEER_TIMEOUT="${EA_SOURCE_ENGINEER_TIMEOUT:-1800}"
EA_SOURCE_TEST_TIMEOUT="${EA_SOURCE_TEST_TIMEOUT:-300}"
EA_SOURCE_EVAL_TIMEOUT="${EA_SOURCE_EVAL_TIMEOUT:-7200}"

usage() {
    cat <<'EOF'
Usage:
  evolving_agent/scripts/run_llm_only_evolutions.sh TASKS_FILE [all|prompt|genome|source]

The default mode is "all". All selected modes use the same:
  - Coding setting: llm_only
  - LLM backend/model/reasoning effort
  - entity split seed and Train/Dev/Public-Holdout fractions
  - generations and children per generation

Examples:
  evolving_agent/scripts/run_llm_only_evolutions.sh external/Dr-CiK/Dr-CiK_public/tasks
  EA_GENERATIONS=1 EA_CHILDREN=1 EA_LIMIT=5 \
    evolving_agent/scripts/run_llm_only_evolutions.sh external/Dr-CiK/Dr-CiK_public/tasks prompt
  EA_DRY_RUN=1 evolving_agent/scripts/run_llm_only_evolutions.sh /data/tasks all
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

if [[ -z "$TASKS_FILE" ]]; then
    usage >&2
    die "TASKS_FILE is required (or set EA_TASKS_FILE)"
fi
if [[ "$MODE" != "all" && "$MODE" != "prompt" && "$MODE" != "genome" && "$MODE" != "source" ]]; then
    usage >&2
    die "mode must be all, prompt, genome, or source; got '$MODE'"
fi
if [[ "$EA_DRY_RUN" != "1" && ! -e "$TASKS_FILE" ]]; then
    die "task path does not exist: $TASKS_FILE"
fi

TASKS_FILE="$(
    if [[ -e "$TASKS_FILE" ]]; then
        cd "$(dirname "$TASKS_FILE")"
        printf '%s/%s\n' "$PWD" "$(basename "$TASKS_FILE")"
    else
        printf '%s\n' "$TASKS_FILE"
    fi
)"

if [[ "$MODE" == "all" ]]; then
    MODES=(prompt genome source)
else
    MODES=("$MODE")
fi

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

execute() {
    print_command "$@"
    if [[ "$EA_DRY_RUN" != "1" ]]; then
        "$@"
    fi
}

if [[ "$EA_DRY_RUN" != "1" && ("$MODE" == "all" || "$MODE" == "source") ]]; then
    if ! git -C "$REPO_ROOT" diff --quiet HEAD --; then
        die "source evolution requires a clean tracked worktree; commit or stash tracked changes first"
    fi
fi

mkdir -p "$EA_RUNS_DIR"
cd "$REPO_ROOT"

cat <<EOF
LLM-only evolution protocol
  tasks:              $TASKS_FILE
  modes:              ${MODES[*]}
  model:              $EA_CODEX_MODEL
  reasoning effort:   $EA_REASONING_EFFORT
  generations:        $EA_GENERATIONS
  children:           $EA_CHILDREN
  seed:               $EA_SEED
  dev fraction:       $EA_DEV_FRACTION
  holdout fraction:   $EA_HOLDOUT_FRACTION
  task limit:          ${EA_LIMIT:-none (full public set)}
  output root:         $EA_RUNS_DIR
  dry run:             $EA_DRY_RUN
EOF

for evolution_mode in "${MODES[@]}"; do
    run_dir="$EA_RUNS_DIR/$evolution_mode"
    mkdir -p "$run_dir"
    args=(
        "$PYTHON" -m evolving_agent
        --evolution "$evolution_mode"
        --tasks-file "$TASKS_FILE"
        --setting llm_only
        --llm-backend codex
        --codex-model "$EA_CODEX_MODEL"
        --codex-reasoning-effort "$EA_REASONING_EFFORT"
        --codex-timeout "$EA_CODEX_TIMEOUT"
        --codex-cache-dir "$run_dir/codex-cache"
        --generations "$EA_GENERATIONS"
        --children "$EA_CHILDREN"
        --seed "$EA_SEED"
        --dev-fraction "$EA_DEV_FRACTION"
        --holdout-fraction "$EA_HOLDOUT_FRACTION"
        --split-manifest-path "$run_dir/split_manifest.json"
        --trace-path "$run_dir/evolution_trace.json"
        --checkpoint-path "$run_dir/checkpoint.json"
        --progress-path "$run_dir/progress.jsonl"
        --library-path "$run_dir/coding_skills.json"
        --retrieval-library-path "$run_dir/retrieval_skills.json"
        --decision-library-path "$run_dir/decision_skills.json"
        --source-engineer-timeout "$EA_SOURCE_ENGINEER_TIMEOUT"
        --source-test-timeout "$EA_SOURCE_TEST_TIMEOUT"
        --source-eval-timeout "$EA_SOURCE_EVAL_TIMEOUT"
    )
    if [[ "$evolution_mode" == "source" ]]; then
        args+=(--source-patch-path "$run_dir/best_source.patch")
    else
        args+=(--policy-path "$run_dir/best_policy.json")
    fi
    if [[ -n "$EA_LIMIT" ]]; then
        args+=(--limit "$EA_LIMIT")
    fi

    echo
    echo "Running $evolution_mode evolution -> $run_dir"
    execute "${args[@]}"
done

echo
if [[ "$EA_DRY_RUN" == "1" ]]; then
    echo "Dry run complete; no evolution command was executed."
else
    echo "Selected LLM-only evolution runs completed."
    echo "Freeze each artifact before evaluating its manifest's Public Holdout or Hidden Test."
fi
