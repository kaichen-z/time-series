#!/usr/bin/env bash
# Loop C: evolve all three bundles jointly against end-to-end forecast error.
#
# An individual is a (coding, retrieval, decision) triple. Mutations mostly target the
# decision bundle; seed it with Loop A's and Loop B's winners.
#
#   EA_CODING_BUNDLE=.../coding/v007.json \
#   EA_RETRIEVAL_BUNDLE=.../retrieval/v004.json \
#   ./scripts/03_evolve_system.sh
#
# Scores use dr_cik's LOCAL PROXY metrics (sMAE/sRMSE/sCRPS), not Dr-CiK's private
# official scorer -- every summary this writes carries that note. Treat the numbers as
# relative comparisons between bundles, not as leaderboard-comparable absolutes.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

RUN_NAME="${EA_RUN_NAME:-loop_c}"
CKPT="$EA_CKPT_ROOT/$RUN_NAME"
SEEDS="$EA_REPO_ROOT/evolving_agents/bundles"
CODING="${EA_CODING_BUNDLE:-$SEEDS/coding/v000.json}"
RETRIEVAL="${EA_RETRIEVAL_BUNDLE:-$SEEDS/retrieval/v000.json}"
DECISION="${EA_DECISION_BUNDLE:-$SEEDS/decision/v000.json}"

for path in "$CODING" "$RETRIEVAL" "$DECISION"; do
  [[ -f "$path" ]] || ea_die "bundle not found: $path"
done

LOG_FILE="$(ea_log_path "$RUN_NAME")"

ea_resolve_devices
ea_check_disk "$EA_OUT_ROOT" 20
ea_show_config
ea_info "seed triple: $(basename "$(dirname "$CODING")")/$(basename "$CODING") + $(basename "$RETRIEVAL") + $(basename "$DECISION")"

python3 -m evolving_agents.cli \
  evolve-system \
  $(ea_task_source) \
  --checkpoint-dir "$CKPT" \
  --bundles-dir "$EA_BUNDLES_DIR" \
  --runs-dir "$EA_RUNS_DIR" \
  --cache-dir "$EA_CACHE_DIR" \
  --coding-bundle "$CODING" \
  --retrieval-bundle "$RETRIEVAL" \
  --decision-bundle "$DECISION" \
  --worker-model-id "$EA_WORKER_MODEL" \
  --evolver-model-id "$EA_EVOLVER_MODEL" \
  ${EA_WORKER_DEVICE:+--worker-device "$EA_WORKER_DEVICE"} \
  ${EA_EVOLVER_DEVICE:+--evolver-device "$EA_EVOLVER_DEVICE"} \
  --generations "$EA_GENERATIONS" \
  --population-size "$EA_POPULATION" \
  --keep-elite "$EA_KEEP_ELITE" \
  --minibatch-size "$EA_MINIBATCH" \
  --stall-patience "$EA_STALL_PATIENCE" \
  --seed "$EA_SEED" \
  --trace-level "$EA_TRACE_LEVEL" \
  $(ea_log_flags "$LOG_FILE") \
  "$@" > "$EA_RESULTS_DIR/$RUN_NAME.summary.json"

ea_info "summary:"
cat "$EA_RESULTS_DIR/$RUN_NAME.summary.json"
ea_report_log "$LOG_FILE"

ea_info "done. when you are finished tuning, score the test split ONCE:"
ea_info "  ./scripts/05_final_eval.sh"
