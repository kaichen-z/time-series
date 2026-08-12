#!/usr/bin/env bash
# Fastest possible end-to-end check: 1 generation on the 3-task sample dir.
#
# Use this to confirm the models load, the sandbox runs, and the trace looks right
# BEFORE committing to a multi-hour Loop A. Expect a few minutes, mostly model loading.
#
#   ./scripts/00_smoke.sh

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

export EA_USE_SAMPLE=1
export EA_GENERATIONS=1
export EA_POPULATION=2
export EA_KEEP_ELITE=1
export EA_MINIBATCH=1
export EA_STALL_PATIENCE=0
export EA_TRACE_LEVEL="${EA_TRACE_LEVEL:-full}"

LOG_FILE="$(ea_log_path smoke)"

ea_resolve_devices
ea_check_disk "$EA_OUT_ROOT" 5
ea_show_config

SMOKE_OUT="$EA_OUT_ROOT/smoke"
rm -rf "$SMOKE_OUT"
mkdir -p "$SMOKE_OUT"

ea_info "running a 1-generation Loop A smoke test on the sample split"
python3 -m evolving_agents.cli \
  evolve-coding \
  $(ea_task_source) \
  --split-file "$SMOKE_OUT/splits.json" \
  --checkpoint-dir "$SMOKE_OUT/ckpt" \
  --bundles-dir "$SMOKE_OUT/bundles" \
  --runs-dir "$SMOKE_OUT/runs" \
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
  --dev-limit 1 \
  --n-windows 1 \
  --seed "$EA_SEED" \
  --trace-level "$EA_TRACE_LEVEL" \
  $(ea_log_flags "$LOG_FILE")

ea_report_log "$LOG_FILE"
ea_info "smoke output under $SMOKE_OUT"
ea_info "run records:   $(wc -l < "$SMOKE_OUT/runs/loop_a.jsonl" 2>/dev/null || echo 0) line(s)"
ea_info "reasoning:     $(find "$SMOKE_OUT/runs/reasoning" -name '*.txt' 2>/dev/null | wc -l) file(s)"
ea_info "checkpoints:   $(ls "$SMOKE_OUT/ckpt" 2>/dev/null | tr '\n' ' ')"
