#!/bin/bash
# Alternating co-evolution: numerical -> decision -> numerical -> decision; 3 agents per phase, 2 episodes x 5 submissions.
R=$1; DSN=$2; A=$(dirname $0)/../../scripts/coevolution/run_agent.py; T=$(dirname $0)/TASK_$DSN.md
phase () {  # $1 role, $2 round tag
  echo "{\"phase\": \"$1\"}" > $R/shared/phase.json; echo "== phase $1 round $2 $(date -u +%H:%M)" >> $R/phases.log
  for i in 1 2; do
    case $1 in
      numerical) F=("per-dataset/per-group blending using backtests and method errors" "robust combination of more foundation models (moirai, chronos) and shrinkage" "cold start and level/trend handling that generalise across datasets");;
      decision) F=("when to trust document corrections on tmmd/timesx without hurting drcik" "magnitude calibration relative to the new base forecast" "window/edge handling and physical constraints that generalise");;
    esac
    ROLE="YOUR ROLE: $1 (submit with --role $1). Round $2 of the alternating co-evolution. Focus: ${F[$((i-1))]}"
    TASK_FILE=$T AGENT_MODEL=${AGENT_MODEL:-gpt-6-sol} python3 $A $R ${1:0:1}$i$2 2 5 "$ROLE" > $R/agent_${1:0:1}$i$2.log 2>&1 &
  done
  wait
}
phase numerical a; phase decision a; phase numerical b; phase decision b
echo "== done $(date -u +%H:%M)" >> $R/phases.log
