#!/usr/bin/env bash
set -euo pipefail

TASKS_FILE=${TASKS_FILE:-external/Dr-CiK/sample/tasks.jsonl}
SPLIT_FILE=${SPLIT_FILE:-splits/drcik_public_80_20_99_v1.json}
MODEL=${MODEL:-gpt-5.4}
EFFORT=${EFFORT:-high}
RUN_DIR=${RUN_DIR:-runs/retrieval_evolution/formal_80_20}
AUTHORITY_PATH=${AUTHORITY_PATH:-runs/retrieval_evolution_authority/formal_80_20.json}
AUTHORITY_HEAD_PATH=${AUTHORITY_HEAD_PATH:-${AUTHORITY_PATH}.head}
AUTHORITY_DIR=${AUTHORITY_PATH%/*}
OPERATOR_AUTHORITY_KEY=${RETRIEVAL_CHECKPOINT_AUTHORITY_KEY:-}

if [[ ${#OPERATOR_AUTHORITY_KEY} -lt 32 ]]; then
  printf 'RETRIEVAL_CHECKPOINT_AUTHORITY_KEY must provide the operator authority key\n' >&2
  exit 2
fi

if [[ "$AUTHORITY_DIR" == "$AUTHORITY_PATH" ]]; then
  printf 'AUTHORITY_PATH must include a protected directory component\n' >&2
  exit 2
fi
if [[ ! -d "$AUTHORITY_DIR" ]]; then
  if [[ -e "$AUTHORITY_DIR" ]]; then
    printf 'AUTHORITY_PATH parent exists but is not a directory\n' >&2
    exit 2
  fi
  (umask 077 && mkdir -p -- "$AUTHORITY_DIR")
fi

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
  --run-root "$RUN_DIR"
  --checkpoint-authority-path "$AUTHORITY_PATH"
  --checkpoint-authority-head-path "$AUTHORITY_HEAD_PATH"
  --checkpoint-path "$RUN_DIR/checkpoint.json"
  --progress-path "$RUN_DIR/progress.jsonl"
  --policy-path "$RUN_DIR/best_policy.json"
  --trace-path "$RUN_DIR/evolution_trace.json"
)

printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
