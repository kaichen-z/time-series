#!/usr/bin/env bash
# Auto-pipeline (retry mode): the 80-train cards are ready. Repeatedly try to generate
# train semantic refs until the LLM quota is back (retry every 20 min), then evolve the
# controller on train (CVaR + LOO). Runs unattended (launch with nohup).
set -u
cd "$(dirname "$0")/.." || exit 1
export TMPDIR="$PWD/.scratch"
export PYTHONPATH="$PWD"
PY=.venv/bin/python
FP=f8d9d5862942
export CARDS_FILE="effect_cards_train_${FP}.json"
LOG=.scratch/auto_pipeline.log

# how many of the 80 train tasks currently have a saved semantic ref
train_refs() {
  $PY - <<'PYEOF' 2>/dev/null || echo 0
import json
tr=json.load(open("splits/drcik_public_80_20_99_v3.json"))["partitions"]["train"]["task_ids"]
r=json.load(open(".scratch/semantic_refs.json")) if __import__("os").path.exists(".scratch/semantic_refs.json") else {}
print(sum(1 for t in tr if t in r))
PYEOF
}
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

say "retry-watcher started; train refs $(train_refs)/80"
for attempt in $(seq 1 24); do          # up to 24 tries * 20min = 8h
  $PY scripts/gen_semantic_refs.py >> "$LOG" 2>&1
  n=$(train_refs)
  say "attempt $attempt: train refs $n/80"
  [ "$n" -ge 78 ] && break
  say "quota likely still limited; sleeping 20m"
  sleep 1200
done

say "step ③ evolving controller on train (CVaR + LOO)"
$PY scripts/evolve_controller.py > .scratch/evolve_train.log 2>&1
say "step ③ done -> .scratch/evolve_train.log"
say "AUTO-PIPELINE COMPLETE"
