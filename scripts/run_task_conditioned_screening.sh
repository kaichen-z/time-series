#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_REPO="${RUN_REPO:-$ROOT_DIR/runs/method_filtering/combined103_full_80_20_99_20260823}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/task_conditioned_screening/formal_80_20_20260823}"
TASKS_FILE="${TASKS_FILE:-$ROOT_DIR/external/Dr-CiK/full-download/Dr-CiK_public/tasks}"
SPLIT_FILE="${SPLIT_FILE:-$ROOT_DIR/splits/drcik_public_80_20_99_v1.json}"
TARGET_BATCHES_FILE="${TARGET_BATCHES_FILE:-$RUN_REPO/filter_target_batches.json}"
TRAIN_LIMIT="${SCREEN_TRAIN:-80}"
DEV_LIMIT="${SCREEN_DEV:-20}"
CODEX_MODEL="${SCREEN_CODEX_MODEL:-gpt-5.6-luna}"
CODEX_REASONING="${SCREEN_CODEX_REASONING:-low}"
BASELINE_METHOD="${SCREEN_BASELINE_METHOD:-toto_2_0}"
MIN_CANDIDATES="${SCREEN_MIN_CANDIDATES:-12}"
MAX_CANDIDATES="${SCREEN_MAX_CANDIDATES:-}"
MIN_UNIQUE="${SCREEN_MIN_UNIQUE_DICTIONARIES:-3}"
MAX_JACCARD="${SCREEN_MAX_MEAN_JACCARD:-0.995}"
MIN_GROUP_SUPPORT="${SCREEN_MIN_GROUP_SUPPORT:-4}"
MIN_DEV_ORACLE_RETENTION="${SCREEN_MIN_DEV_ORACLE_RETENTION:-0.9}"
BATCH_SIZE="${SCREEN_BATCH_SIZE:-8}"
REFINEMENT_GENERATIONS="${SCREEN_REFINEMENT_GENERATIONS:-3}"
REFINEMENT_BATCH_SIZE="${SCREEN_REFINEMENT_BATCH_SIZE:-24}"
WORKERS_CONFIG="${NA_TSFM_WORKERS_CONFIG:-$ROOT_DIR/runs/method_evolution/local_tsfm_workers.json}"
MODEL_CACHE="${NA_MODEL_CACHE_DIR:-$ROOT_DIR/outputs/model-cache}"

MAX_CANDIDATE_ARGS=()
if [[ -n "$MAX_CANDIDATES" ]]; then
  MAX_CANDIDATE_ARGS=(--screen-max-candidates "$MAX_CANDIDATES")
fi

cd "$ROOT_DIR"
python -m numerical_agent.run_task_conditioned_screening \
  --repo "$RUN_REPO" \
  --split-file "$SPLIT_FILE" \
  --tasks-file "$TASKS_FILE" \
  --outcome-cache-dir "$RUN_REPO/outcome-cache" \
  --policy-outcome-cache-dir "$RUN_REPO/policy-outcome-cache" \
  --target-batches-file "$TARGET_BATCHES_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "$TRAIN_LIMIT" \
  --dev-limit "$DEV_LIMIT" \
  --seed-manifest "$RUN_REPO/seed_manifest.json" \
  --codex-model "$CODEX_MODEL" \
  --codex-reasoning-effort "$CODEX_REASONING" \
  --baseline-method "$BASELINE_METHOD" \
  --screen-min-candidates "$MIN_CANDIDATES" \
  "${MAX_CANDIDATE_ARGS[@]}" \
  --screen-min-unique-dictionaries "$MIN_UNIQUE" \
  --screen-max-mean-jaccard "$MAX_JACCARD" \
  --screen-min-group-support "$MIN_GROUP_SUPPORT" \
  --screen-min-dev-oracle-retention "$MIN_DEV_ORACLE_RETENTION" \
  --screen-batch-size "$BATCH_SIZE" \
  --screen-refinement-generations "$REFINEMENT_GENERATIONS" \
  --screen-refinement-batch-size "$REFINEMENT_BATCH_SIZE" \
  --tsfm-runtimes chronos,timesfm \
  --chronos-device-map cpu \
  --model-cache-dir "$MODEL_CACHE" \
  --tsfm-workers-config "$WORKERS_CONFIG" \
  --acknowledged-model-licenses CC-BY-NC-4.0
