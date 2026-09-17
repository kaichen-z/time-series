#!/usr/bin/env bash
# Supervisor: run the big Haiku evolution, auto-resume on token/quota exhaustion.
# Resumes from checkpoint each attempt; waits 30 min between failed attempts.
set -u
cd /home/yiqi/yiyun/draft/time-series
RUN=runs/evolution_v2/real-30m-haiku-big-20260916-r1
MANIFEST=configs/evolution_v2/real/real-30m-toto-claude-server.json
LOG=/tmp/evolve-haiku-big.log

SAVED_CACHE=/tmp/haiku-p2p3-cache

for attempt in $(seq 1 40); do
  # On the very first launch (fresh dir), replay the saved P2/P3 Haiku cache:
  # wait for run_manifest.json, then copy cache into llm-cache before P2 calls.
  if [ -d "$SAVED_CACHE" ] && [ ! -f "$RUN/run_manifest.json" ]; then
    ( until [ -f "$RUN/run_manifest.json" ]; do sleep 1; done
      mkdir -p "$RUN/llm-cache"
      cp "$SAVED_CACHE"/*.json "$RUN/llm-cache/" 2>/dev/null
      echo "INJECTOR: seeded $(ls "$RUN/llm-cache" | wc -l) cache files $(date)" >> "$LOG" ) &
  fi
  echo "SUPERVISOR: attempt $attempt starting $(date)" >> "$LOG"
  EVOLVE_CLAUDE_MODEL=haiku .venv/bin/python -m evolving_loop.v2 real-evolve \
    --manifest "$MANIFEST" --output-dir "$RUN" --no-time-limit \
    --p2-generations 40 --p3-steps 8 --p4-candidates 3 \
    --eval-train-size 8 --eval-dev-size 3 --eval-folds 2 >> "$LOG" 2>&1
  rc=$?
  phase=$(python3 -c "import json;print(json.load(open('$RUN/checkpoint.json')).get('phase',''))" 2>/dev/null)
  echo "SUPERVISOR: attempt $attempt exited rc=$rc phase=$phase $(date)" >> "$LOG"
  if [ "$phase" = "COMPLETE" ]; then
    echo "SUPERVISOR: run COMPLETE after $attempt attempt(s) $(date)" >> "$LOG"
    break
  fi
  if [ "$rc" -eq 0 ]; then
    # Clean exit that is not COMPLETE is not a token/quota problem; resuming will
    # not help. Stop and surface for inspection instead of looping.
    echo "SUPERVISOR: clean exit rc=0 but phase=$phase (not token exhaustion); stopping for inspection $(date)" >> "$LOG"
    break
  fi
  echo "SUPERVISOR: crash rc=$rc (likely token/quota exhaustion), sleeping 1800s then resume $(date)" >> "$LOG"
  sleep 1800
done
echo "SUPERVISOR: supervisor exiting $(date)" >> "$LOG"
