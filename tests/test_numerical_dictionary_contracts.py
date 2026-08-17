from __future__ import annotations

from typing import Sequence

import pytest

from numerical_agent.config import DictionaryCurationConfig
from numerical_agent.dictionary import (
    MethodCandidate,
    MethodDefinition,
    MethodRecord,
    ToolDictionary,
)
from numerical_agent.providers import RuntimeRegistry


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
        (MethodRecord(definition, candidate, "accepted", 1, {"smape": 8.0}),),
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
