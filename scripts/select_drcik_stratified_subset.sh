#!/usr/bin/env bash
# Build a stratified frozen Dr-CiK task subset (drcik_agent.select_stratified_subset).
# Usage: ./select_drcik_stratified_subset.sh --output <path> [--size N] [--seed N] [--exclude ID]...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

cd "$REPO_ROOT"
"$PYTHON" -m drcik_agent.select_stratified_subset "$@"
