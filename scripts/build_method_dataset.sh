#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python - <<'PY'
from pathlib import Path

from numerical_agent.collection.catalog_v001 import write_catalog_manifests

root = Path(".")
sources, methods = write_catalog_manifests(
    root / "numerical_agent/dictionaries/statistical_base_methods_v000.json",
    root / "numerical_agent/datasets/source_registry_v001.jsonl",
    root / "numerical_agent/datasets/method_candidates_v001.jsonl",
)
print(f"Generated {len(methods)} method candidates from {len(sources)} sources.")
PY

python -m numerical_agent build-dataset \
  --sources numerical_agent/datasets/source_registry_v001.jsonl \
  --methods numerical_agent/datasets/method_candidates_v001.jsonl \
  --queries numerical_agent/datasets/collection_queries_v001.json \
  --collection-journal numerical_agent/datasets/collection_journal_v001.json \
  --output numerical_agent/datasets/forecast_method_dataset_v001.json \
  --audit-output numerical_agent/datasets/collection_audit_v001.json \
  --sha256-output numerical_agent/datasets/forecast_method_dataset_v001.sha256

python -m pytest -q \
  tests/test_collection_catalog.py \
  tests/test_collection.py \
  tests/test_numerical_cli.py \
  tests/test_scripts.py

echo "Release: numerical_agent/datasets/forecast_method_dataset_v001.json"
echo "Checksum: numerical_agent/datasets/forecast_method_dataset_v001.sha256"
