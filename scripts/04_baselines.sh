#!/usr/bin/env bash
# Run the reference systems the evolved pipeline has to beat. These are also the ablations.
#
#   ./scripts/04_baselines.sh                    # all five, on the dev split
#   ./scripts/04_baselines.sh chronos-only       # just one
#   EA_SPLIT=evolve ./scripts/04_baselines.sh
#
#   chronos-only      text-blind foundation model alone. EVERYTHING must beat this.
#   naive-rag         BM25 top-k -> one prompt -> numbers parsed from free text (the strawman)
#   coding-only       the Coding Agent with no text, isolating what Loop A alone buys
#   frozen-system     all three agents on SEED bundles, isolating what evolution itself buys
#   oracle-retrieval  feed exactly the labeled supporting docs: the ceiling retrieval could reach
#
# LAFP is deliberately absent: compare against its published numbers, do not reimplement it here.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ALL_BASELINES=(chronos-only naive-rag coding-only frozen-system oracle-retrieval)
SPLIT="${EA_SPLIT:-dev}"
SEEDS="$EA_REPO_ROOT/evolving_agents/bundles"

if (( $# > 0 )); then
  BASELINES=("$@")
else
  BASELINES=("${ALL_BASELINES[@]}")
fi

ea_resolve_devices
ea_check_disk "$EA_OUT_ROOT" 10
ea_show_config
ea_info "split: $SPLIT | baselines: ${BASELINES[*]}"

for baseline in "${BASELINES[@]}"; do
  out="$EA_RESULTS_DIR/baseline_${baseline}_${SPLIT}"
  LOG_FILE="$(ea_log_path "baseline_${baseline}")"
  ea_info "--- $baseline (log: $LOG_FILE) ---"
  python3 -m evolving_agents.cli run-baselines \
    $(ea_task_source) \
    --baseline "$baseline" \
    --split "$SPLIT" \
    --output-dir "$out" \
    --runs-dir "$EA_RUNS_DIR" \
    --cache-dir "$EA_CACHE_DIR" \
    --coding-bundle "${EA_CODING_BUNDLE:-$SEEDS/coding/v000.json}" \
    --retrieval-bundle "${EA_RETRIEVAL_BUNDLE:-$SEEDS/retrieval/v000.json}" \
    --decision-bundle "${EA_DECISION_BUNDLE:-$SEEDS/decision/v000.json}" \
    --worker-model-id "$EA_WORKER_MODEL" \
    ${EA_WORKER_DEVICE:+--worker-device "$EA_WORKER_DEVICE"} \
    --seed "$EA_SEED" \
    --trace-level "$EA_TRACE_LEVEL" \
    $(ea_log_flags "$LOG_FILE") > "$out.summary.json"
done

ea_info "summaries:"
for baseline in "${BASELINES[@]}"; do
  summary="$EA_RESULTS_DIR/baseline_${baseline}_${SPLIT}/summary.json"
  [[ -f "$summary" ]] && printf '  %-18s %s\n' "$baseline" "$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['mean_metrics'])" "$summary")"
done
