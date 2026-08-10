#!/usr/bin/env bash
# The one and only scoring pass on the held-out test split. Run this ONCE, at the end.
#
#   EA_CODING_BUNDLE=... EA_RETRIEVAL_BUNDLE=... EA_DECISION_BUNDLE=... ./scripts/05_final_eval.sh
#
# The test split exists to give an honest number. Every time you look at it and then change
# something, that number gets a little less honest. final_eval.py refuses to overwrite an
# existing summary for exactly this reason; --force is an override, not a normal step.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

SEEDS="$EA_REPO_ROOT/evolving_agents/bundles"
CODING="${EA_CODING_BUNDLE:-$SEEDS/coding/v000.json}"
RETRIEVAL="${EA_RETRIEVAL_BUNDLE:-$SEEDS/retrieval/v000.json}"
DECISION="${EA_DECISION_BUNDLE:-$SEEDS/decision/v000.json}"
OUT="${EA_FINAL_OUT:-$EA_RESULTS_DIR/final_eval}"

for path in "$CODING" "$RETRIEVAL" "$DECISION"; do
  [[ -f "$path" ]] || ea_die "bundle not found: $path"
done

if [[ "$CODING" == *"/v000.json" && "$RETRIEVAL" == *"/v000.json" && "$DECISION" == *"/v000.json" ]]; then
  ea_info "WARNING: every bundle is still a SEED. This scores the un-evolved system."
  ea_info "         That is a legitimate baseline number, but it is not your result."
fi

ea_resolve_devices
ea_check_disk "$EA_OUT_ROOT" 10
ea_show_config
ea_info "frozen triple -> $OUT"

python3 "$EA_REPO_ROOT/scripts/final_eval.py" \
  $(ea_task_source) \
  --output-dir "$OUT" \
  --coding-bundle "$CODING" \
  --retrieval-bundle "$RETRIEVAL" \
  --decision-bundle "$DECISION" \
  --worker-model-id "$EA_WORKER_MODEL" \
  ${EA_WORKER_DEVICE:+--worker-device "$EA_WORKER_DEVICE"} \
  --cache-dir "$EA_CACHE_DIR" \
  --runs-dir "$EA_RUNS_DIR" \
  --seed "$EA_SEED" \
  --trace-level "$EA_TRACE_LEVEL" \
  "$@"

ea_info "final summary: $OUT/summary.json"
