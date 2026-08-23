#!/usr/bin/env bash
# Build the standalone method repository from every eligible catalog definition.
# Re-running after interruption resumes from per-method checkpoints under .bootstrap/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

ME_REPO="${ME_REPO:-$EA_RUNS_DIR/method_evolution/v001}"
ME_CATALOG="${ME_CATALOG:-$REPO_ROOT/numerical_agent/datasets/forecast_method_dataset_v002.json}"
ME_FAMILY="${ME_FAMILY:-statistical}"
ME_LLM_BACKEND="${ME_LLM_BACKEND:-codex}"
ME_CODEX_MODEL="${ME_CODEX_MODEL:-gpt-5.6-sol}"
ME_REASONING_EFFORT="${ME_REASONING_EFFORT:-high}"
ME_CODEX_TIMEOUT="${ME_CODEX_TIMEOUT:-900}"
ME_ATTEMPTS_PER_METHOD="${ME_ATTEMPTS_PER_METHOD:-2}"
ME_DRY_RUN="${ME_DRY_RUN:-0}"

die() {
    echo "error: $*" >&2
    exit 2
}

[[ -f "$ME_CATALOG" ]] || die "catalog does not exist: $ME_CATALOG"
[[ ! -f "$ME_REPO/methods.py" ]] || die "$ME_REPO already contains methods.py"

METHOD_COUNT="$($PYTHON - "$ME_CATALOG" "$ME_FAMILY" <<'PY'
import sys
from numerical_agent.evolution.seed import seed_definitions
print(len(seed_definitions(sys.argv[1], family=sys.argv[2])[0]))
PY
)"

COMMAND=(
    "$PYTHON" -m numerical_agent.bootstrap_evolution
    --repo "$ME_REPO"
    --catalog "$ME_CATALOG"
    --family "$ME_FAMILY"
    --attempts-per-method "$ME_ATTEMPTS_PER_METHOD"
    --llm-backend "$ME_LLM_BACKEND"
)
case "$ME_LLM_BACKEND" in
    codex)
        COMMAND+=(--codex-model "$ME_CODEX_MODEL"
                  --codex-reasoning-effort "$ME_REASONING_EFFORT"
                  --codex-timeout "$ME_CODEX_TIMEOUT"
                  --codex-cache-dir "$ME_REPO/codex-cache")
        ;;
esac

cat <<EOF
Method-repository bootstrap
  repo:             $ME_REPO
  catalog:          $ME_CATALOG
  family:           $ME_FAMILY
  methods selected: $METHOD_COUNT
  backend:          $ME_LLM_BACKEND
  attempts/method:  $ME_ATTEMPTS_PER_METHOD
  dry run:          $ME_DRY_RUN
EOF

if [[ "$ME_DRY_RUN" == "1" ]]; then
    printf '  '
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

cd "$REPO_ROOT"
"${COMMAND[@]}"

echo
echo "Repository: $ME_REPO"
echo "Module:     $ME_REPO/methods.py"
echo "Summary:    $ME_REPO/bootstrap_summary.json"
echo "History:"
git -C "$ME_REPO" log --format='  %h %s' -3
