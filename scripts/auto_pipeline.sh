#!/usr/bin/env bash
# Auto-pipeline: wait for 80-train effect cards, then automatically run
#   ② generate train semantic refs  ->  ③ evolve the controller on train (CVaR + LOO)
# Runs unattended (launch with nohup). Everything logs to .scratch/auto_pipeline.log.
set -u
cd "$(dirname "$0")/.." || exit 1
export TMPDIR="$PWD/.scratch"
export PYTHONPATH="$PWD"
PY=.venv/bin/python
FP=f8d9d5862942
CARDS=".scratch/effect_cards_train_${FP}.json"
GEN_PID=777877          # the running 80-train generator
LOG=.scratch/auto_pipeline.log

count() { $PY -c "import json;print(len(json.load(open('$CARDS'))))" 2>/dev/null || echo 0; }
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "watcher started; waiting for 80-train cards ($(count)/80)"
# wait until >=78 cards OR the generator process has exited
while :; do
  c=$(count)
  if [ "$c" -ge 78 ]; then say "cards ready ($c/80)"; break; fi
  if ! ps -p "$GEN_PID" >/dev/null 2>&1; then say "generator ended at $c/80; proceeding"; break; fi
  sleep 120
done

say "step ② generating train semantic refs"
CARDS_FILE="effect_cards_train_${FP}.json" $PY scripts/gen_semantic_refs.py >> "$LOG" 2>&1
say "step ② done"

say "step ③ evolving controller on train (CVaR + LOO)"
CARDS_FILE="effect_cards_train_${FP}.json" $PY scripts/evolve_controller.py > .scratch/evolve_train.log 2>&1
say "step ③ done -> .scratch/evolve_train.log"
say "AUTO-PIPELINE COMPLETE"
