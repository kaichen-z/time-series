#!/usr/bin/env bash
# Loop B: evolve the Retrieval Agent against the document role labels.
#
# Scores a bundle by F1 of the documents it kept against the supporting/distractor
# labels, plus a bonus when its evidence actually improves the forecast produced by a
# FROZEN coding+decision stack. Run Loop A first: its winner is what gets frozen here.
#
#   EA_FROZEN_CODING=/path/to/coding/v007.json ./scripts/02_evolve_retrieval.sh
#   EA_BONUS_WEIGHT=0 ./scripts/02_evolve_retrieval.sh   # label F1 only, no forecast arm
#
# The bonus arm runs the whole stack twice per task (with and without evidence), so it
# is roughly 3x the cost of --bonus-weight 0. Start at 0 if you are just checking wiring.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

RUN_NAME="${EA_RUN_NAME:-loop_b}"
CKPT="$EA_CKPT_ROOT/$RUN_NAME"
FROZEN_CODING="${EA_FROZEN_CODING:-$EA_REPO_ROOT/evolving_agents/bundles/coding/v000.json}"
FROZEN_DECISION="${EA_FROZEN_DECISION:-$EA_REPO_ROOT/evolving_agents/bundles/decision/v000.json}"
BONUS_WEIGHT="${EA_BONUS_WEIGHT:-0.2}"

[[ -f "$FROZEN_CODING" ]] || ea_die "frozen coding bundle not found: $FROZEN_CODING (run scripts/01_evolve_coding.sh first, or set EA_FROZEN_CODING)"
[[ -f "$FROZEN_DECISION" ]] || ea_die "frozen decision bundle not found: $FROZEN_DECISION"

if [[ "$FROZEN_CODING" == *"/v000.json" ]]; then
  ea_info "NOTE: freezing the SEED coding bundle. That is fine for a wiring check, but for real"
  ea_info "      results run Loop A first and point EA_FROZEN_CODING at its winner."
fi

ea_resolve_devices
ea_check_disk "$EA_OUT_ROOT" 20
ea_show_config
ea_info "frozen coding   $FROZEN_CODING"
ea_info "frozen decision $FROZEN_DECISION"
ea_info "bonus weight    $BONUS_WEIGHT"

python3 -m evolving_agents.cli \
  evolve-retrieval \
  $(ea_task_source) \
  --checkpoint-dir "$CKPT" \
  --bundles-dir "$EA_BUNDLES_DIR" \
  --runs-dir "$EA_RUNS_DIR" \
  --cache-dir "$EA_CACHE_DIR" \
  --frozen-coding-bundle "$FROZEN_CODING" \
  --frozen-decision-bundle "$FROZEN_DECISION" \
  --bonus-weight "$BONUS_WEIGHT" \
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
  "$@" | tee "$EA_RESULTS_DIR/$RUN_NAME.summary.json"

ea_info "done. feed this winner plus Loop A's into Loop C:"
ea_info "  EA_CODING_BUNDLE=... EA_RETRIEVAL_BUNDLE=... ./scripts/03_evolve_system.sh"
