#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  printf '%s\n' "python -m numerical_agent.run_champion_evolution --repo ${CHAMPION_REPO:?} --split-file ${CHAMPION_SPLIT_FILE:?} --tasks-file ${CHAMPION_TASKS_FILE:?} --generations ${CHAMPION_GENERATIONS:?} --proposer-model ${CHAMPION_MODEL:?} --proposer-reasoning-effort ${CHAMPION_REASONING:?} --candidate-minimum-gain ${CHAMPION_MIN_GAIN:?} --research-target-gain ${CHAMPION_TARGET_GAIN:?} --output-dir ${CHAMPION_OUTPUT_DIR:?} --authority-root ${CHAMPION_AUTHORITY_ROOT:?} --authority-identity ${CHAMPION_AUTHORITY_IDENTITY:?}"
  exit 0
fi

exec python -m numerical_agent.run_champion_evolution \
  --repo "${CHAMPION_REPO:?}" \
  --split-file "${CHAMPION_SPLIT_FILE:?}" \
  --tasks-file "${CHAMPION_TASKS_FILE:?}" \
  --generations "${CHAMPION_GENERATIONS:?}" \
  --proposer-model "${CHAMPION_MODEL:?}" \
  --proposer-reasoning-effort "${CHAMPION_REASONING:?}" \
  --candidate-minimum-gain "${CHAMPION_MIN_GAIN:?}" \
  --research-target-gain "${CHAMPION_TARGET_GAIN:?}" \
  --output-dir "${CHAMPION_OUTPUT_DIR:?}" \
  --authority-root "${CHAMPION_AUTHORITY_ROOT:?}" \
  --authority-identity "${CHAMPION_AUTHORITY_IDENTITY:?}" \
  "$@"
