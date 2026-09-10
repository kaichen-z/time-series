from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math

import pytest

from evolving_loop.v2 import canonical_v2_bytes
from evolving_loop.v2.budget import BudgetPlan
from evolving_loop.v2.contracts import KernelProtocolCommitment
from evolving_loop.v2.numerical_qd.config import (
    NumericalQDConfigV2,
    load_numerical_qd_config,
)


TOP_LEVEL_FIELDS = {
    "schema_version", "profile", "seed", "kernel_protocol",
    "runtime_fingerprints", "budget", "fixed_bundle_components",
    "descriptor_policy", "mutation", "map_elites", "hyperband", "proposer",
    "adapter",
}


def sha256_for(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def protocol_payload() -> dict[str, object]:
    return {
        field: sha256_for(field)
        for field in (
            "task_materializer", "split_manifest", "metric_policy", "label_firewall",
            "artifact_validator", "sandbox_policy", "promotion_policy",
        )
    }


def descriptor_policy_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "trend_low_threshold": 0.15,
        "trend_high_threshold": 0.75,
        "intermittency_high_threshold": 0.30,
        "regime_shift_threshold": 0.50,
        "horizon_short_threshold": 0.10,
        "horizon_medium_threshold": 0.30,
        "seasonality_threshold": 0.30,
        "short_seasonal_lag_max": 24,
        "variance_floor": 1e-12,
        "seasonal_lags": {"H": [24, 168], "D": [7, 30], "W": [52], "M": [12], "Q": [4]},
    }


def resource_payload() -> dict[str, int | float]:
    return {
        "wall_seconds": 14_400.0,
        "task_executions": 10_000,
        "llm_calls": 100,
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "gpu_seconds": 20_000.0,
        "subprocesses": 10_000,
        "artifact_bytes": 1_000_000_000,
    }


def valid_config_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile": "formal",
        "seed": 7,
        "kernel_protocol": protocol_payload(),
        "runtime_fingerprints": {"python": sha256_for("python"), "worker": sha256_for("worker")},
        "budget": {
            "hard_limit_seconds": 14_400,
            "finalization_reserve_fraction": 0.2,
            "ceilings": resource_payload(),
        },
        "fixed_bundle_components": {
            "retrieval_release_sha256": sha256_for("retrieval-release"),
            "decision_policy_sha256": sha256_for("decision-policy"),
            "harness_policy_sha256": sha256_for("harness-policy"),
            "archive_snapshot_sha256": sha256_for("initial-archive"),
            "scheduler_state_sha256": sha256_for("scheduler-state"),
        },
        "descriptor_policy": descriptor_policy_payload(),
        "mutation": {
            "operators": ["add", "repair", "fork", "combine", "route", "specialize", "crossover", "remove", "quarantine", "policy_tune"],
            "max_parents_per_child": 2,
            "max_inventory_size": 64,
        },
        "map_elites": {
            "cell_capacity": 4,
            "sampling_weights": {
                "underexplored": 40,
                "elites": 30,
                "failure_matched": 20,
                "stepping_stones": 10,
            },
        },
        "hyperband": {
            "brackets": {
                "explore": [8, 32, 80],
                "confirm": [32, 80],
                "replay": [80],
            },
            "reduction_factor": 2,
        },
        "proposer": {
            "provider": "hybrid",
            "max_proposals_per_generation": 16,
            "max_response_bytes": 65_536,
        },
        "adapter": {
            "task_timeout_seconds": 300.0,
            "max_forecast_values": 10_000,
        },
    }


def test_config_has_exact_top_level_schema_and_typed_project_one_bindings():
    payload = valid_config_payload()
    config = NumericalQDConfigV2.from_payload(payload)
    assert set(config.to_payload()) == TOP_LEVEL_FIELDS
    assert config.to_payload() == payload
    assert isinstance(config.kernel_protocol, KernelProtocolCommitment)
    assert isinstance(config.budget, BudgetPlan)
    assert config.budget.search_deadline_seconds == 11_520
    assert config.descriptor_policy.fingerprint() == hashlib.sha256(
        config.descriptor_policy.canonical_bytes()
    ).hexdigest()
    assert config.canonical_bytes() == canonical_v2_bytes(payload)
    assert config.fingerprint() == hashlib.sha256(config.canonical_bytes()).hexdigest()


@pytest.mark.parametrize("field", sorted(TOP_LEVEL_FIELDS))
def test_config_rejects_every_missing_top_level_field(field):
    payload = valid_config_payload()
    del payload[field]
    with pytest.raises(ValueError, match="exact schema"):
        NumericalQDConfigV2.from_payload(payload)


def test_config_rejects_additional_or_nonstring_fields():
    for extra in ("future_values", "dev_metrics", 1):
        with pytest.raises(ValueError, match="exact schema|keys must be strings"):
            NumericalQDConfigV2.from_payload(valid_config_payload() | {extra: None})


def test_fixed_bundle_components_bind_generation_zero_host_identities():
    config = NumericalQDConfigV2.from_payload(valid_config_payload())
    assert set(config.fixed_bundle_components) == {
        "retrieval_release_sha256", "decision_policy_sha256", "harness_policy_sha256",
        "archive_snapshot_sha256", "scheduler_state_sha256",
    }
    for field, value in config.fixed_bundle_components.items():
        assert len(value) == 64, field
    hostile = copy.deepcopy(valid_config_payload())
    hostile["fixed_bundle_components"]["numerical_release_sha256"] = sha256_for("caller-selected")
    with pytest.raises(ValueError, match="exact schema"):
        NumericalQDConfigV2.from_payload(hostile)


def test_every_nested_mapping_is_detached_and_frozen():
    payload = valid_config_payload()
    config = NumericalQDConfigV2.from_payload(payload)
    before = config.canonical_bytes()
    with pytest.raises(TypeError):
        config.runtime_fingerprints["python"] = sha256_for("changed")
    with pytest.raises(TypeError):
        config.map_elites["sampling_weights"]["elites"] = 100
    with pytest.raises(AttributeError):
        config.hyperband["brackets"]["explore"].append(160)
    payload["map_elites"]["sampling_weights"]["elites"] = 100
    assert config.canonical_bytes() == before


def test_direct_dataclass_replacement_cannot_bypass_validation_or_freezing():
    config = NumericalQDConfigV2.from_payload(valid_config_payload())
    with pytest.raises(ValueError, match="profile"):
        replace(config, profile="public")
    changed = replace(
        config,
        map_elites={
            "cell_capacity": 4,
            "sampling_weights": {
                "underexplored": 40,
                "elites": 30,
                "failure_matched": 20,
                "stepping_stones": 10,
            },
        },
    )
    with pytest.raises(TypeError):
        changed.map_elites["sampling_weights"]["elites"] = 100


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), 2),
        (("profile",), "public"),
        (("seed",), True),
        (("runtime_fingerprints",), {}),
        (("fixed_bundle_components", "retrieval_release_sha256"), "bad"),
        (("mutation", "operators"), ["add", "add"]),
        (("mutation", "operators"), ["rewrite_kernel"]),
        (("mutation", "max_parents_per_child"), 0),
        (("mutation", "max_inventory_size"), -1),
        (("map_elites", "cell_capacity"), 0),
        (("map_elites", "sampling_weights", "underexplored"), 39),
        (("hyperband", "brackets", "explore"), [8, 16, 80]),
        (("hyperband", "brackets", "confirm"), [32]),
        (("hyperband", "brackets", "replay"), [80, 160]),
        (("hyperband", "reduction_factor"), 0),
        (("proposer", "provider"), "llm"),
        (("proposer", "max_proposals_per_generation"), 0),
        (("proposer", "max_response_bytes"), 0),
        (("adapter", "task_timeout_seconds"), math.inf),
        (("adapter", "max_forecast_values"), 0),
        (("budget", "ceilings", "wall_seconds"), math.nan),
        (("budget", "ceilings", "task_executions"), True),
    ],
)
def test_config_rejects_invalid_limits_weights_rungs_provider_or_resources(path, value):
    payload = copy.deepcopy(valid_config_payload())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        NumericalQDConfigV2.from_payload(payload)


@pytest.mark.parametrize(
    "field,value",
    [("hard_limit_seconds", 14_399), ("finalization_reserve_fraction", 0.19)],
)
def test_formal_profile_rejects_budget_drift(field, value):
    payload = valid_config_payload()
    payload["budget"][field] = value
    with pytest.raises(ValueError, match="formal"):
        NumericalQDConfigV2.from_payload(payload)


def test_deterministic_provider_and_nonformal_budget_are_allowed():
    payload = valid_config_payload()
    payload["profile"] = "smoke"
    payload["budget"]["hard_limit_seconds"] = 60
    payload["budget"]["finalization_reserve_fraction"] = 0.1
    payload["proposer"]["provider"] = "deterministic"
    config = NumericalQDConfigV2.from_payload(payload)
    assert config.profile == "smoke"
    assert config.proposer["provider"] == "deterministic"


def test_load_config_rejects_duplicate_keys_and_noncanonical_file_bytes(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key.*schema_version"):
        load_numerical_qd_config(duplicate)

    noncanonical = tmp_path / "pretty.json"
    noncanonical.write_text(json.dumps(valid_config_payload(), indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        load_numerical_qd_config(noncanonical)


def test_load_config_accepts_canonical_bytes(tmp_path):
    path = tmp_path / "config.json"
    payload = valid_config_payload()
    path.write_bytes(canonical_v2_bytes(payload))
    assert load_numerical_qd_config(path).to_payload() == payload
