from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from evolving_loop.v2 import (
    EvolutionV2Config,
    KernelProtocolCommitment,
    SanitizedEvolutionFeedback,
    canonical_v2_bytes,
    fingerprint_payload,
    load_v2_config,
    require_sha256,
)


def sha256_for(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def valid_protocol_payload() -> dict[str, object]:
    return {
        field: sha256_for(field)
        for field in (
            "task_materializer",
            "split_manifest",
            "metric_policy",
            "label_firewall",
            "artifact_validator",
            "sandbox_policy",
            "promotion_policy",
        )
    }


def valid_config_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile": "formal",
        "seed": 7,
        "scheduler": "ucb",
        "enabled_mutation_scopes": ["numerical", "retrieval", "decision", "joint"],
        "archive_capacities": {"numerical": 8, "joint": 4},
        "hyperband": {"resource_levels": [8, 32, 80]},
        "runtime_fingerprints": {
            "python": sha256_for("python-runtime"),
            "forecast": sha256_for("forecast-runtime"),
        },
        "kernel_protocol": valid_protocol_payload(),
        "hard_limit_seconds": 14_400,
        "finalization_reserve_fraction": 0.2,
        "runner": "production",
    }


def valid_feedback_payload() -> dict[str, object]:
    return {
        "parent_sha256": sha256_for("parent"),
        "train_evaluation_sha256": sha256_for("train-evaluation"),
        "train_objectives": {"smae": 0.5, "failures": 0},
        "train_behavior_descriptors": {"trend": "high"},
        "failure_categories": ["timeout"],
        "remaining_proposal_budget": {"children": 3, "tokens": 1_000},
    }


def test_v2_canonical_bytes_are_order_independent_and_strict():
    left = canonical_v2_bytes({"b": [2, 1], "a": "x"})
    right = canonical_v2_bytes({"a": "x", "b": [2, 1]})
    assert left == right == b'{"a":"x","b":[2,1]}\n'
    assert fingerprint_payload({"b": [2, 1], "a": "x"}) == hashlib.sha256(
        left
    ).hexdigest()
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            canonical_v2_bytes({"value": bad})


@pytest.mark.parametrize(
    "payload, error, message",
    [
        ({1: "value"}, ValueError, "keys must be strings"),
        ({"path": Path("artifact.json")}, TypeError, "Path"),
        ({"callback": lambda: None}, TypeError, "function"),
    ],
)
def test_v2_canonical_bytes_reject_non_json_values(payload, error, message):
    with pytest.raises(error, match=message):
        canonical_v2_bytes(payload)


def test_require_sha256_accepts_only_canonical_lowercase_hex():
    valid = "0123456789abcdef" * 4
    assert require_sha256(valid, "artifact") == valid
    for invalid in (True, 123, "abc", valid.upper(), "g" * 64):
        with pytest.raises(ValueError, match="artifact"):
            require_sha256(invalid, "artifact")


def test_protocol_fingerprint_commits_to_every_l0_binding():
    payload = valid_protocol_payload()
    baseline = KernelProtocolCommitment.from_payload(payload).fingerprint()

    for field in payload:
        changed = dict(payload)
        changed[field] = sha256_for(f"changed-{field}")
        commitment = KernelProtocolCommitment.from_payload(changed)
        assert commitment.fingerprint() != baseline


def test_protocol_commitment_has_an_exact_sha256_schema():
    payload = valid_protocol_payload()
    with pytest.raises(ValueError, match="exact schema"):
        KernelProtocolCommitment.from_payload(payload | {"caller_protocol": "v2"})
    payload["metric_policy"] = "not-a-sha"
    with pytest.raises(ValueError, match="metric_policy"):
        KernelProtocolCommitment.from_payload(payload)


def test_sanitized_feedback_is_closed_and_rejects_evaluator_only_keys():
    payload = valid_feedback_payload()
    feedback = SanitizedEvolutionFeedback.from_payload(payload)
    assert feedback.to_payload() == payload

    for reserved in (
        "dev_comparison",
        "dev_metrics",
        "public_ids",
        "future_values",
        "evaluator_labels",
        "holdout",
    ):
        hostile = dict(payload)
        hostile["remaining_proposal_budget"] = {"nested": [{reserved: []}]}
        with pytest.raises(ValueError, match=reserved):
            SanitizedEvolutionFeedback.from_payload(hostile)

    harmless = dict(payload)
    harmless["train_behavior_descriptors"] = {"label_firewall_runtime": "sealed"}
    assert SanitizedEvolutionFeedback.from_payload(harmless).to_payload() == harmless

    with pytest.raises(ValueError, match="exact schema"):
        SanitizedEvolutionFeedback.from_payload(payload | {"notes": "not closed"})


def test_sanitized_feedback_cannot_be_mutated_after_validation():
    payload = valid_feedback_payload()
    payload["remaining_proposal_budget"] = {"nested": [{"children": 3}]}
    feedback = SanitizedEvolutionFeedback.from_payload(payload)

    with pytest.raises(TypeError):
        feedback.train_objectives["dev_metrics"] = {}
    nested = feedback.remaining_proposal_budget["nested"]
    with pytest.raises(TypeError):
        nested[0]["future_values"] = []

    assert feedback.to_payload() == payload


def test_v2_config_rejects_unknown_keys_and_wrong_boolean_integer_aliases():
    payload = valid_config_payload()
    payload["seed"] = True
    with pytest.raises(ValueError, match="seed"):
        EvolutionV2Config.from_payload(payload)
    payload = valid_config_payload() | {"unowned_control": True}
    with pytest.raises(ValueError, match="exact schema"):
        EvolutionV2Config.from_payload(payload)


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema_version", 2),
        ("profile", "benchmark"),
        ("scheduler", "round_robin"),
        ("enabled_mutation_scopes", []),
        ("enabled_mutation_scopes", ["numerical", "numerical"]),
        ("enabled_mutation_scopes", ["numerical", "source"]),
        ("archive_capacities", {"numerical": 0}),
        ("runtime_fingerprints", {}),
        ("hard_limit_seconds", True),
        ("finalization_reserve_fraction", 1),
        ("runner", "dry_run"),
    ],
)
def test_v2_config_rejects_invalid_field_values(field, value):
    payload = valid_config_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        EvolutionV2Config.from_payload(payload)


def test_v2_config_requires_valid_runtime_and_protocol_fingerprints():
    payload = valid_config_payload()
    payload["runtime_fingerprints"] = {"python": "not-a-sha"}
    with pytest.raises(ValueError, match="runtime_fingerprints.python"):
        EvolutionV2Config.from_payload(payload)

    payload = valid_config_payload()
    payload["kernel_protocol"] = valid_protocol_payload() | {"extra": sha256_for("x")}
    with pytest.raises(ValueError, match="exact schema"):
        EvolutionV2Config.from_payload(payload)


def test_v2_config_mappings_cannot_be_mutated_after_validation():
    payload = valid_config_payload()
    config = EvolutionV2Config.from_payload(payload)

    with pytest.raises(TypeError):
        config.archive_capacities["numerical"] = 0
    with pytest.raises(AttributeError):
        config.hyperband["resource_levels"].append(160)
    with pytest.raises(TypeError):
        config.runtime_fingerprints["python"] = "not-a-sha"

    assert config.to_payload() == payload


@pytest.mark.parametrize(
    "field, value",
    [
        ("hard_limit_seconds", 14_399),
        ("finalization_reserve_fraction", 0.19),
    ],
)
def test_formal_config_rejects_hard_limit_or_reserve_drift(field, value):
    payload = valid_config_payload()
    payload[field] = value
    with pytest.raises(ValueError, match="formal"):
        EvolutionV2Config.from_payload(payload)


def test_load_v2_config_uses_duplicate_key_rejecting_parser(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key.*schema_version"):
        load_v2_config(path)


def test_load_v2_config_returns_a_validated_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(canonical_v2_bytes(valid_config_payload()))
    config = load_v2_config(path)
    assert config.profile == "formal"
    assert config.kernel_protocol.fingerprint() == KernelProtocolCommitment.from_payload(
        valid_protocol_payload()
    ).fingerprint()
