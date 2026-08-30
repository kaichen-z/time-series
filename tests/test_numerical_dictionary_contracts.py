from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import pytest

from common.evolution_core.contracts import metric_policy_metadata

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


@pytest.mark.parametrize(
    "mutation",
    (
        "flat_row",
        "missing_record_field",
        "missing_definition_field",
        "string_method_id",
        "string_numeric",
        "bool_int",
        "string_summary",
        "candidate_string_numeric",
        "candidate_bool_int",
        "unknown_outer",
        "unknown_record",
        "unknown_definition",
        "unknown_candidate",
        "mismatched_ids",
    ),
)
def test_active_dictionary_requires_exact_canonical_nested_records(mutation) -> None:
    original = ToolDictionary(
        "active",
        None,
        1,
        (MethodRecord(
            MethodDefinition("m1", "statistical", "active method"),
            MethodCandidate("m1", "fake", "opaque", {"quality": 1.0}),
            "accepted",
            1,
            {"smae": 1.0},
            2,
        ),),
    ).to_payload()
    payload = deepcopy(original)
    record = payload["methods"][0]
    if mutation == "flat_row":
        payload["methods"] = [deepcopy(record["definition"])]
    elif mutation == "missing_record_field":
        record.pop("candidate")
    elif mutation == "missing_definition_field":
        record["definition"].pop("assumptions")
    elif mutation == "string_method_id":
        record["definition"]["method_id"] = 1
        record["candidate"]["method_id"] = 1
    elif mutation == "string_numeric":
        record["revision_count"] = "1"
    elif mutation == "bool_int":
        record["implementation_attempts"] = True
    elif mutation == "string_summary":
        record["train_summary"]["smae"] = "1.0"
    elif mutation == "candidate_string_numeric":
        record["candidate"]["version"] = "1"
    elif mutation == "candidate_bool_int":
        record["candidate"]["version"] = True
    elif mutation == "unknown_outer":
        payload["unexpected"] = "forged"
    elif mutation == "unknown_record":
        record["unexpected"] = "forged"
    elif mutation == "unknown_definition":
        record["definition"]["unexpected"] = "forged"
    elif mutation == "unknown_candidate":
        record["candidate"]["unexpected"] = "forged"
    elif mutation == "mismatched_ids":
        record["candidate"]["method_id"] = "other"

    with pytest.raises(ValueError):
        ToolDictionary.from_payload(payload)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: {**payload, "schema_version": 1},
        lambda payload: {
            key: value for key, value in payload.items()
            if key not in {"metric_policy", "metric_policy_fingerprint"}
        },
        lambda payload: {**payload, "schema_version": 2.0},
        lambda payload: {
            **payload,
            "generation": True,
        },
        lambda payload: {
            **payload,
            "methods": [
                {
                    key: value for key, value in payload["methods"][0].items()
                    if key != "status"
                }
            ],
        },
    ),
)
def test_active_dictionary_rejects_legacy_unbound_or_defaulted_fields(mutate) -> None:
    dictionary = ToolDictionary(
        "active",
        None,
        0,
        (MethodDefinition("m1", "statistical", "active method"),),
    )

    with pytest.raises(ValueError, match="schema|metric policy|generation|status"):
        ToolDictionary.from_payload(mutate(dictionary.to_payload()))


def test_legacy_dictionary_reader_is_explicit_and_report_only() -> None:
    payload = {
        "schema_version": 1,
        "dictionary_id": "legacy",
        "parent_dictionary_id": None,
        "methods": [
            {
                "method_id": "m1",
                "family": "statistical",
                "description": "historical report method",
            }
        ],
    }

    with pytest.raises(ValueError):
        ToolDictionary.from_payload(payload)
    restored = ToolDictionary.from_legacy_payload(payload)

    assert restored.generation == 0
    assert restored.methods[0].status == "unimplemented"
    assert restored.to_payload() == {
        "schema_version": 2,
        **metric_policy_metadata(),
        "dictionary_id": "legacy",
        "parent_dictionary_id": None,
        "generation": 0,
        "methods": [restored.methods[0].to_payload()],
    }
    assert ToolDictionary.from_legacy_report_payload(payload) == restored


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
