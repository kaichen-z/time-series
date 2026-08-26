#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_REPO="${RUN_REPO:-$ROOT_DIR/runs/method_filtering/combined103_full_80_20_99_20260823}"
SCREENING_DIR="${SCREENING_DIR:-$ROOT_DIR/runs/task_conditioned_screening/formal_80_20_20260823}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/numerical_selector/formal_80_20_20260823}"
HINDCAST_CACHE="${HINDCAST_CACHE:-$ROOT_DIR/runs/numerical_selector/hindcast-cache}"
TASKS_FILE="${TASKS_FILE:-$ROOT_DIR/external/Dr-CiK/full-download/Dr-CiK_public/tasks}"
SPLIT_FILE="${SPLIT_FILE:-$ROOT_DIR/splits/drcik_public_80_20_99_v1.json}"
TRAIN_LIMIT="${SELECTOR_TRAIN:-80}"
DEV_LIMIT="${SELECTOR_DEV:-20}"
GENERATIONS="${SELECTOR_GENERATIONS:-3}"
TRAIN_VALIDATION_FOLDS="${SELECTOR_TRAIN_VALIDATION_FOLDS:-4}"
CODEX_MODEL="${SELECTOR_CODEX_MODEL:-gpt-5.6-luna}"
CODEX_REASONING="${SELECTOR_CODEX_REASONING:-low}"
WORKERS_CONFIG="${NA_TSFM_WORKERS_CONFIG:-$ROOT_DIR/runs/method_evolution/local_tsfm_workers.json}"
MODEL_CACHE="${NA_MODEL_CACHE_DIR:-$ROOT_DIR/outputs/model-cache}"

cd "$ROOT_DIR"
python -m numerical_agent.run_selector_evolution \
  --repo "$RUN_REPO" \
  --screening-dir "$SCREENING_DIR" \
  --split-file "$SPLIT_FILE" \
  --tasks-file "$TASKS_FILE" \
  --outcome-cache-dir "$RUN_REPO/outcome-cache" \
  --policy-outcome-cache-dir "$RUN_REPO/policy-outcome-cache" \
  --hindcast-cache-dir "$HINDCAST_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "$TRAIN_LIMIT" \
  --dev-limit "$DEV_LIMIT" \
  --generations "$GENERATIONS" \
  --train-validation-folds "$TRAIN_VALIDATION_FOLDS" \
  --codex-model "$CODEX_MODEL" \
  --codex-reasoning-effort "$CODEX_REASONING" \
  --tsfm-runtimes chronos,timesfm \
  --chronos-device-map cpu \
  --model-cache-dir "$MODEL_CACHE" \
  --tsfm-workers-config "$WORKERS_CONFIG" \
  --acknowledged-model-licenses CC-BY-NC-4.0
