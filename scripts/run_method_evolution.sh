#!/usr/bin/env bash
# Evolve the forecasting-method module: measure every method on the frozen Train split,
# hand the whole file plus its measurements to an LLM, apply the operations it returns,
# and commit each generation to the module's own git repository.
#
# Usage:
#   scripts/run_method_evolution.sh          # one generation, qwen
#   ME_GENERATIONS=3 scripts/run_method_evolution.sh
#   ME_LLM_BACKEND=codex scripts/run_method_evolution.sh
#
# Re-running continues from the current commit; it never re-seeds. A one-stage generation uses
# one LLM call; configuring ME_SELECTOR_CODEX_MODEL enables a selector plus mutator call.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

ME_REPO="${ME_REPO:-$EA_RUNS_DIR/method_evolution/v001}"
ME_TASKS_FILE_WAS_SET="${ME_TASKS_FILE+x}"
ME_TASKS_FILE="${ME_TASKS_FILE:-/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK/data/tasks/train.jsonl}"
ME_SPLIT_FILE="${ME_SPLIT_FILE:-$REPO_ROOT/splits/drcik_public_80_20_99_v1.json}"
ME_GENERATIONS="${ME_GENERATIONS:-5}"
ME_EVOLUTION_STRATEGY="${ME_EVOLUTION_STRATEGY:-batch}"
ME_OUTCOME_CACHE_DIR="${ME_OUTCOME_CACHE_DIR:-$ME_REPO/outcome-cache}"
ME_MAX_TARGETS="${ME_MAX_TARGETS:-3}"
ME_SCREEN_TASKS="${ME_SCREEN_TASKS:-4}"
ME_FULL_EVALUATION_CANDIDATES="${ME_FULL_EVALUATION_CANDIDATES:-3}"
ME_FAILURE_JUDGE="${ME_FAILURE_JUDGE:-0}"
ME_LLM_BACKEND="${ME_LLM_BACKEND:-qwen}"
ME_MODEL_ID="${ME_MODEL_ID:-}"
ME_DEVICE="${ME_DEVICE:-}"
ME_CODEX_MODEL="${ME_CODEX_MODEL:-gpt-5.6-sol}"
ME_REASONING_EFFORT="${ME_REASONING_EFFORT:-high}"
ME_SELECTOR_CODEX_MODEL="${ME_SELECTOR_CODEX_MODEL:-}"
ME_SELECTOR_REASONING_EFFORT="${ME_SELECTOR_REASONING_EFFORT:-medium}"
ME_CODEX_TIMEOUT="${ME_CODEX_TIMEOUT:-900}"
ME_TRAIN_LIMIT="${ME_TRAIN_LIMIT:-0}"
ME_VALIDATION_TAIL="${ME_VALIDATION_TAIL:-0}"
ME_FOUNDATION_PORTFOLIO="${ME_FOUNDATION_PORTFOLIO:-none}"
ME_POLICY_OUTCOME_CACHE_DIR="${ME_POLICY_OUTCOME_CACHE_DIR:-$ME_REPO/policy-outcome-cache}"
ME_POLICY_MAX_TARGETS="${ME_POLICY_MAX_TARGETS:-3}"
ME_TSFM_RUNTIMES="${ME_TSFM_RUNTIMES:-}"
ME_TSFM_WORKERS_CONFIG="${ME_TSFM_WORKERS_CONFIG:-}"
ME_ACKNOWLEDGED_MODEL_LICENSES="${ME_ACKNOWLEDGED_MODEL_LICENSES:-}"
ME_MODEL_CACHE_DIR="${ME_MODEL_CACHE_DIR:-}"
ME_CHRONOS_DEVICE_MAP="${ME_CHRONOS_DEVICE_MAP:-cpu}"
ME_DRY_RUN="${ME_DRY_RUN:-0}"

die() {
    echo "error: $*" >&2
    exit 2
}

[[ -f "$ME_REPO/methods.py" ]] || die "no seeded module at $ME_REPO/methods.py"
[[ -d "$ME_REPO/.git" ]] || die "$ME_REPO is not a git repository; the seed commit is missing"
if [[ "$ME_FOUNDATION_PORTFOLIO" != "none" ]]; then
    [[ -f "$ME_REPO/policies.py" ]] || die "foundation portfolio requires $ME_REPO/policies.py"
fi
[[ -f "$ME_SPLIT_FILE" ]] || die "split file does not exist: $ME_SPLIT_FILE"
if [[ ! -f "$ME_TASKS_FILE" && ! -d "$ME_TASKS_FILE" && !( "$ME_DRY_RUN" == "1" && -z "$ME_TASKS_FILE_WAS_SET" ) ]]; then
    die "Dr-CiK tasks file does not exist: $ME_TASKS_FILE"
fi

COMMAND=(
    "$PYTHON" -m numerical_agent.run_evolution
    --repo "$ME_REPO"
    --split-file "$ME_SPLIT_FILE"
    --tasks-file "$ME_TASKS_FILE"
    --generations "$ME_GENERATIONS"
    --llm-backend "$ME_LLM_BACKEND"
    --evolution-strategy "$ME_EVOLUTION_STRATEGY"
    --outcome-cache-dir "$ME_OUTCOME_CACHE_DIR"
    --max-targets "$ME_MAX_TARGETS"
    --screen-tasks "$ME_SCREEN_TASKS"
    --full-evaluation-candidates "$ME_FULL_EVALUATION_CANDIDATES"
    --foundation-portfolio "$ME_FOUNDATION_PORTFOLIO"
    --policy-outcome-cache-dir "$ME_POLICY_OUTCOME_CACHE_DIR"
    --policy-max-targets "$ME_POLICY_MAX_TARGETS"
)
[[ -n "$ME_TSFM_RUNTIMES" ]] && COMMAND+=(--tsfm-runtimes "$ME_TSFM_RUNTIMES")
[[ -n "$ME_TSFM_WORKERS_CONFIG" ]] && COMMAND+=(--tsfm-workers-config "$ME_TSFM_WORKERS_CONFIG")
[[ -n "$ME_ACKNOWLEDGED_MODEL_LICENSES" ]] && COMMAND+=(--acknowledged-model-licenses "$ME_ACKNOWLEDGED_MODEL_LICENSES")
[[ -n "$ME_MODEL_CACHE_DIR" ]] && COMMAND+=(--model-cache-dir "$ME_MODEL_CACHE_DIR")
[[ -n "$ME_CHRONOS_DEVICE_MAP" ]] && COMMAND+=(--chronos-device-map "$ME_CHRONOS_DEVICE_MAP")
[[ "$ME_FAILURE_JUDGE" == "1" ]] && COMMAND+=(--failure-judge)
case "$ME_LLM_BACKEND" in
    codex)
        COMMAND+=(--codex-model "$ME_CODEX_MODEL"
                  --codex-reasoning-effort "$ME_REASONING_EFFORT"
                  --codex-timeout "$ME_CODEX_TIMEOUT"
                  --codex-cache-dir "$ME_REPO/codex-cache")
        if [[ -n "$ME_SELECTOR_CODEX_MODEL" ]]; then
            COMMAND+=(--selector-codex-model "$ME_SELECTOR_CODEX_MODEL"
                      --selector-codex-reasoning-effort "$ME_SELECTOR_REASONING_EFFORT")
        fi
        ;;
    qwen)
        [[ -n "$ME_MODEL_ID" ]] && COMMAND+=(--model-id "$ME_MODEL_ID")
        [[ -n "$ME_DEVICE" ]] && COMMAND+=(--device "$ME_DEVICE")
        ;;
esac
[[ "$ME_TRAIN_LIMIT" != "0" ]] && COMMAND+=(--train-limit "$ME_TRAIN_LIMIT")
[[ "$ME_VALIDATION_TAIL" != "0" ]] && COMMAND+=(--validation-tail "$ME_VALIDATION_TAIL")

METHOD_COUNT="$(grep -c '^def ' "$ME_REPO/methods.py")"
HEAD_COMMIT="$(git -C "$ME_REPO" rev-parse --short HEAD 2>/dev/null || echo none)"

cat <<EOF
Method-module evolution
  repo:        $ME_REPO
  at commit:   $HEAD_COMMIT
  methods:     $METHOD_COUNT
  split:       $ME_SPLIT_FILE
  backend:     $ME_LLM_BACKEND
  strategy:    $ME_EVOLUTION_STRATEGY
  portfolio:   $ME_FOUNDATION_PORTFOLIO
  generations: $ME_GENERATIONS
  dry run:     $ME_DRY_RUN
EOF

if [[ "$ME_DRY_RUN" == "1" ]]; then
    printf '  '
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

cd "$REPO_ROOT"
"${COMMAND[@]}" || die "evolution failed; see $ME_REPO/run_evolution_trace.jsonl"

echo
echo "History:"
git -C "$ME_REPO" log --format='  %h %s' | head -5
echo
echo "Module:      $ME_REPO/methods.py"
echo "Metrics:     $ME_REPO/generation_*_metrics.json"
echo "Transcripts: $ME_REPO/transcripts/"
echo "Trace:       $ME_REPO/run_evolution_trace.jsonl"
echo "Diff:        git -C $ME_REPO log -p -1"
