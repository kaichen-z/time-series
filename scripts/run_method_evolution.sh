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
# Re-running continues from the current commit; it never re-seeds. This spends one real
# LLM call per generation.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

ME_REPO="${ME_REPO:-$EA_RUNS_DIR/method_evolution/v001}"
ME_TASKS_FILE="${ME_TASKS_FILE:-/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK/data/tasks/train.jsonl}"
ME_SPLIT_FILE="${ME_SPLIT_FILE:-$REPO_ROOT/splits/drcik_public_80_20_99_v1.json}"
ME_GENERATIONS="${ME_GENERATIONS:-5}"
ME_LLM_BACKEND="${ME_LLM_BACKEND:-qwen}"
ME_MODEL_ID="${ME_MODEL_ID:-}"
ME_DEVICE="${ME_DEVICE:-}"
ME_CODEX_MODEL="${ME_CODEX_MODEL:-gpt-5.6-sol}"
ME_REASONING_EFFORT="${ME_REASONING_EFFORT:-high}"
ME_CODEX_TIMEOUT="${ME_CODEX_TIMEOUT:-900}"
ME_DRY_RUN="${ME_DRY_RUN:-0}"

die() {
    echo "error: $*" >&2
    exit 2
}

[[ -f "$ME_SPLIT_FILE" ]] || die "split file does not exist: $ME_SPLIT_FILE"
[[ -f "$ME_TASKS_FILE" ]] || die "Dr-CiK tasks file does not exist: $ME_TASKS_FILE"
[[ -f "$ME_REPO/methods.py" ]] || die "no seeded module at $ME_REPO/methods.py"
[[ -d "$ME_REPO/.git" ]] || die "$ME_REPO is not a git repository; the seed commit is missing"

COMMAND=(
    "$PYTHON" -m numerical_agent.run_evolution
    --repo "$ME_REPO"
    --split-file "$ME_SPLIT_FILE"
    --tasks-file "$ME_TASKS_FILE"
    --generations "$ME_GENERATIONS"
    --llm-backend "$ME_LLM_BACKEND"
)
case "$ME_LLM_BACKEND" in
    codex)
        COMMAND+=(--codex-model "$ME_CODEX_MODEL"
                  --codex-reasoning-effort "$ME_REASONING_EFFORT"
                  --codex-timeout "$ME_CODEX_TIMEOUT"
                  --codex-cache-dir "$ME_REPO/codex-cache")
        ;;
    qwen)
        [[ -n "$ME_MODEL_ID" ]] && COMMAND+=(--model-id "$ME_MODEL_ID")
        [[ -n "$ME_DEVICE" ]] && COMMAND+=(--device "$ME_DEVICE")
        ;;
esac

METHOD_COUNT="$(grep -c '^def ' "$ME_REPO/methods.py")"
HEAD_COMMIT="$(git -C "$ME_REPO" rev-parse --short HEAD 2>/dev/null || echo none)"

cat <<EOF
Method-module evolution
  repo:        $ME_REPO
  at commit:   $HEAD_COMMIT
  methods:     $METHOD_COUNT
  split:       $ME_SPLIT_FILE
  backend:     $ME_LLM_BACKEND
  generations: $ME_GENERATIONS  (one real LLM call each)
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
