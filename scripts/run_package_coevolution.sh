#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${TIME_SERIES_PYTHON:-python3}"
exec "$PYTHON_BIN" -u -m evolving_loop.run_package_coevolution "$@"
