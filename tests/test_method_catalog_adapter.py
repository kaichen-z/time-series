from __future__ import annotations

import json
from pathlib import Path

import pytest

from numerical_agent.collection.catalog_adapter import tool_dictionary_from_payload


ROOT = Path(__file__).resolve().parents[1]


def test_release_catalog_imports_as_an_executable_statistical_dictionary() -> None:
    payload = json.loads(
        (
            ROOT / "numerical_agent/datasets/forecast_method_dataset_v001.json"
        ).read_text()
    )

    dictionary = tool_dictionary_from_payload(
        payload, allowed_families=("statistical",)
    )

    assert dictionary.dictionary_id == "forecast_method_dataset_v001.statistical.v000"
    assert dictionary.methods
    assert all(
        record.definition.family == "statistical" for record in dictionary.methods
    )
    assert dictionary.methods[0].definition.method_id.startswith("method_")
    assert all(record.candidate is None for record in dictionary.methods)


def test_unbound_tool_dictionary_payload_is_not_an_active_catalog_seed() -> None:
    payload = {
        "dictionary_id": "existing",
        "parent_dictionary_id": None,
        "generation": 0,
        "methods": [
            {
                "method_id": "m1",
                "family": "statistical",
                "description": "existing method",
            }
        ],
    }

    with pytest.raises(ValueError, match="metric policy"):
        tool_dictionary_from_payload(payload, allowed_families=("statistical",))


def test_release_import_rejects_combined_methods_without_their_parent_families() -> (
    None
):
    payload = json.loads(
        (
            ROOT / "numerical_agent/datasets/forecast_method_dataset_v001.json"
        ).read_text()
    )

    with pytest.raises(ValueError, match="requires excluded parent methods"):
        tool_dictionary_from_payload(payload, allowed_families=("combined",))
