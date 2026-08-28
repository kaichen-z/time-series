#!/usr/bin/env bash
# Write the seed methods module: one LLM call per catalog definition, foundation models verbatim.
#
# Usage:
#   scripts/run_method_bootstrap.sh                       # full 98-method seed with Haiku
#   MB_DRY_RUN=1 scripts/run_method_bootstrap.sh          # report what it would write
#   MB_LIMIT=3 scripts/run_method_bootstrap.sh            # a 3-method smoke test
#
# This spends one real LLM call per model-written method. The client caches by prompt, so
# re-running after an interruption resumes instead of paying for every method again.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

MB_REPO="${MB_REPO:-$EA_RUNS_DIR/method_evolution/full98}"
MB_DEFINITIONS="${MB_DEFINITIONS:-$MB_REPO/definitions.json}"
MB_LLM_BACKEND="${MB_LLM_BACKEND:-claude}"
MB_CLAUDE_MODEL="${MB_CLAUDE_MODEL:-haiku}"
MB_CACHE_DIR="${MB_CACHE_DIR:-$MB_REPO/claude-cache}"
MB_TIMEOUT="${MB_TIMEOUT:-900}"
MB_LIMIT="${MB_LIMIT:-0}"
MB_DRY_RUN="${MB_DRY_RUN:-0}"

die() {
    echo "error: $*" >&2
    exit 2
}

[[ -f "$MB_DEFINITIONS" ]] || die "no definitions file at $MB_DEFINITIONS"

COMMAND=(
    "$PYTHON" -m numerical_agent.run_bootstrap
    --repo "$MB_REPO"
    --definitions "$MB_DEFINITIONS"
    --llm-backend "$MB_LLM_BACKEND"
)
case "$MB_LLM_BACKEND" in
    claude)
        COMMAND+=(--claude-model "$MB_CLAUDE_MODEL"
                  --claude-cache-dir "$MB_CACHE_DIR"
                  --claude-timeout "$MB_TIMEOUT")
        ;;
esac
[[ "$MB_LIMIT" != "0" ]] && COMMAND+=(--limit "$MB_LIMIT")
[[ "$MB_DRY_RUN" == "1" ]] && COMMAND+=(--dry-run)

cat <<EOF
Method-module bootstrap
  repo:        $MB_REPO
  definitions: $MB_DEFINITIONS
  backend:     $MB_LLM_BACKEND ($MB_CLAUDE_MODEL)
  limit:       $MB_LIMIT  (0 means every definition)
  dry run:     $MB_DRY_RUN
EOF

cd "$REPO_ROOT" || die "cannot enter $REPO_ROOT"
PYTHONPATH="$REPO_ROOT" exec "${COMMAND[@]}"
