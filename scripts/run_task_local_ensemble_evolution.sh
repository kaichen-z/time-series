#!/usr/bin/env bash
set -euo pipefail

command=(
  python -m numerical_agent.run_task_local_ensemble_evolution
  --repo "${TASK_LOCAL_REPO:?}"
  --split-file "${TASK_LOCAL_SPLIT_FILE:?}"
  --tasks-file "${TASK_LOCAL_TASKS_FILE:?}"
  --anchor-release-dir "${TASK_LOCAL_ANCHOR_RELEASE_DIR:?}"
  --forecast-store "${TASK_LOCAL_FORECAST_STORE:?}"
  --output-dir "${TASK_LOCAL_OUTPUT_DIR:?}"
)

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}" "$@"
