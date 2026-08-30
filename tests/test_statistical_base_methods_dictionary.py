from __future__ import annotations

import json
from pathlib import Path

from numerical_agent.dictionary import ToolDictionary


DICTIONARY_PATH = (
    Path(__file__).resolve().parent.parent
    / "numerical_agent"
    / "dictionaries"
    / "statistical_base_methods_v000.json"
)


def _load() -> ToolDictionary:
    payload = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
    return ToolDictionary.from_legacy_report_payload(payload)


def test_statistical_base_methods_dictionary_parses_and_round_trips() -> None:
    dictionary = _load()

    assert dictionary.dictionary_id == "statistical_base_methods_v000"
    assert dictionary.parent_dictionary_id is None
    assert dictionary.generation == 0
    assert ToolDictionary.from_payload(dictionary.to_payload()) == dictionary


def test_statistical_base_methods_dictionary_method_count_is_sane() -> None:
    dictionary = _load()

    assert 30 <= len(dictionary.methods) <= 50


def test_statistical_base_methods_dictionary_entries_are_well_formed() -> None:
    dictionary = _load()

    for record in dictionary.methods:
        definition = record.definition
        assert definition.family == "statistical"
        assert record.status == "unimplemented"
        assert definition.assumptions, f"{definition.method_id} has no assumptions"
        assert definition.failure_conditions, f"{definition.method_id} has no failure_conditions"
        assert definition.description.strip()
