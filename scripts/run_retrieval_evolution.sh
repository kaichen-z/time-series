#!/usr/bin/env bash
set -euo pipefail

TASKS_FILE=${TASKS_FILE:-external/Dr-CiK/sample/tasks.jsonl}
SPLIT_FILE=${SPLIT_FILE:-splits/drcik_public_80_20_99_v1.json}
MODEL=${MODEL:-gpt-5.4}
EFFORT=${EFFORT:-high}
RUN_DIR=${RUN_DIR:-runs/retrieval_evolution/formal_80_20}

command=(
  python -m evolving_loop.cli
  --evolution retrieval
  --tasks-file "$TASKS_FILE"
  --split-manifest "$SPLIT_FILE"
  --retrieval-mode two-stage
  --llm-backend codex
  --codex-model "$MODEL"
  --codex-reasoning-effort "$EFFORT"
  --generations 3
  --screen-train-tasks 8
  --screen-promote 2
  --checkpoint-path "$RUN_DIR/checkpoint.json"
  --progress-path "$RUN_DIR/progress.jsonl"
  --policy-path "$RUN_DIR/best_policy.json"
  --trace-path "$RUN_DIR/evolution_trace.json"
)

printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
