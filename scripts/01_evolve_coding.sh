#!/usr/bin/env bash
# Loop A: evolve the Coding Agent against hindcast error. Run this FIRST and LONGEST.
#
# Needs no labels at all -- it scores a bundle by replaying its best hypothesis on past
# windows carved out of each series' own history. Its winning bundle is the input Loops
# B and C both freeze, so nothing downstream is meaningful until this has run.
#
#   ./scripts/01_evolve_coding.sh                 # full dataset, G=10 P=6
#   EA_USE_SAMPLE=1 ./scripts/01_evolve_coding.sh # 3-task sample, for wiring checks
#   EA_GENERATIONS=3 ./scripts/01_evolve_coding.sh
#
# Resumable: re-running with the same checkpoint dir skips completed generations.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

RUN_NAME="${EA_RUN_NAME:-loop_a}"
CKPT="$EA_CKPT_ROOT/$RUN_NAME"

ea_resolve_devices
ea_check_disk "$EA_OUT_ROOT" 20
ea_show_config
ea_info "checkpoints -> $CKPT (delete it to start over; keep it to resume)"

python3 -m evolving_agents.cli \
  evolve-coding \
  $(ea_task_source) \
  --checkpoint-dir "$CKPT" \
  --bundles-dir "$EA_BUNDLES_DIR" \
  --runs-dir "$EA_RUNS_DIR" \
  --cache-dir "$EA_CACHE_DIR" \
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

ea_info "done. the winning bundle id is 'best_individual' in the summary above."
ea_info "pass its file to Loop B, e.g.:"
ea_info "  EA_FROZEN_CODING=$EA_BUNDLES_DIR/coding/vNNN.json ./scripts/02_evolve_retrieval.sh"
