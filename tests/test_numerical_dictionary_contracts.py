"""Tests for numerical_agent/dictionary: ToolDictionary/MethodRecord contracts and the statistical-base-methods seed dictionary."""
from __future__ import annotations

import pytest
import json
from typing import Sequence
from numerical_agent.config import DictionaryCurationConfig
from numerical_agent.dictionary import (
    MethodCandidate,
    MethodDefinition,
    MethodRecord,
    ToolDictionary,
)
from numerical_agent.providers import RuntimeRegistry
from pathlib import Path


def test_dictionary_rejects_duplicate_method_ids() -> None:
    method = MethodDefinition("m1", "statistical", "external method")

    with pytest.raises(ValueError, match="duplicate"):
        ToolDictionary("d0", None, 0, (method, method))


def test_dictionary_accepts_external_method_without_implementation() -> None:
    method = MethodDefinition("m1", "foundation", "provided by collaborator")

    dictionary = ToolDictionary("d0", None, 0, (method,))

    assert dictionary.methods[0].definition.method_id == "m1"
    assert dictionary.methods[0].status == "unimplemented"
    assert dictionary.methods[0].candidate is None


def test_dictionary_rejects_unknown_family_and_dependency() -> None:
    with pytest.raises(ValueError, match="family"):
        MethodDefinition("m1", "unknown", "bad")  # type: ignore[arg-type]

    dependent = MethodDefinition(
        "combined", "combined", "depends on missing method", dependencies=("missing",)
    )
    with pytest.raises(ValueError, match="dependency"):
        ToolDictionary("d0", None, 0, (dependent,))


def test_dictionary_json_round_trip_preserves_external_payload() -> None:
    definition = MethodDefinition(
        "m1",
        "statistical",
        "external method",
        implementation_spec={"provider_hint": "collaborator"},
    )
    candidate = MethodCandidate(
        method_id="m1",
        provider="fake",
        implementation_kind="opaque",
        implementation={"quality": 2.5},
        version=2,
        parent_version=1,
    )
    original = ToolDictionary(
        "d1",
        "d0",
        1,
        (MethodRecord(definition, candidate, "accepted", 1, {"smae": 8.0}),),
    )

    restored = ToolDictionary.from_payload(original.to_payload())

    assert restored == original


class FakeRuntime:
    def supports(self, candidate: MethodCandidate) -> bool:
        return candidate.provider == "fake"

    def forecast(
        self,
        candidate: MethodCandidate,
        history: Sequence[float],
        horizon: int,
        frequency: str,
    ) -> Sequence[float]:
        return [float(candidate.implementation["quality"])] * horizon


def test_runtime_registry_returns_structured_unavailable_result() -> None:
    registry = RuntimeRegistry({"fake": FakeRuntime()})
    supported = MethodCandidate("m1", "fake", "opaque", {"quality": 1.0})
    missing = MethodCandidate("m2", "missing", "opaque", {})

    assert registry.resolve(supported).available
    unavailable = registry.resolve(missing)
    assert not unavailable.available
    assert unavailable.runtime is None
    assert "missing" in unavailable.reason


def test_curation_config_validates_actions_and_revision_budget() -> None:
    config = DictionaryCurationConfig(
        max_revisions_per_method=1,
        max_implementation_attempts=3,
        min_success_rate=0.8,
    )
    assert config.allowed_actions == ("keep", "revise", "quarantine", "discard")
    assert config.allowed_families == ("statistical",)

    with pytest.raises(ValueError, match="max_revisions_per_method"):
        DictionaryCurationConfig(max_revisions_per_method=-1)
    with pytest.raises(ValueError, match="max_implementation_attempts"):
        DictionaryCurationConfig(max_implementation_attempts=0)
    with pytest.raises(ValueError, match="min_success_rate"):
        DictionaryCurationConfig(min_success_rate=1.1)


def test_method_record_round_trip_preserves_implementation_attempts() -> None:
    definition = MethodDefinition("m1", "statistical", "external method")
    original = ToolDictionary(
        "d1",
        "d0",
        1,
        (MethodRecord(definition, implementation_attempts=2),),
    )

    restored = ToolDictionary.from_payload(original.to_payload())

    assert restored.methods[0].implementation_attempts == 2

DICTIONARY_PATH = (
    Path(__file__).resolve().parent.parent
    / "numerical_agent"
    / "dictionaries"
    / "statistical_base_methods_v000.json"
)


def _load() -> ToolDictionary:
    payload = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
    return ToolDictionary.from_payload(payload)


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
