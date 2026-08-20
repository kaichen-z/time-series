#!/usr/bin/env bash
set -euo pipefail

python_command=${PYTHON:-python3}
exec "$python_command" -m numerical_agent.tsfm.smoke "$@"
