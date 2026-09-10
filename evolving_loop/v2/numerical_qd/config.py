"""Closed configuration contract for Project 2 Numerical QD runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from common.payload import strict_json_loads

from ..budget import BudgetPlan, ResourceUse
from ..contracts import (
    KernelProtocolCommitment,
    _freeze_json_value,
    _require_exact_schema,
    _strict_json_value,
    canonical_v2_bytes,
    fingerprint_payload,
    require_sha256,
)
from .contracts import MUTATION_OPERATORS
from .descriptors import DescriptorPolicyV2


_CONFIG_FIELDS = (
    "schema_version",
    "profile",
    "seed",
    "kernel_protocol",
    "runtime_fingerprints",
    "budget",
    "fixed_bundle_components",
    "descriptor_policy",
    "mutation",
    "map_elites",
    "hyperband",
    "proposer",
    "adapter",
)
_BUDGET_FIELDS = ("hard_limit_seconds", "finalization_reserve_fraction", "ceilings")
_FIXED_BUNDLE_FIELDS = (
    "retrieval_release_sha256",
    "decision_policy_sha256",
    "harness_policy_sha256",
    "archive_snapshot_sha256",
    "scheduler_state_sha256",
)
_MUTATION_FIELDS = ("operators", "max_parents_per_child", "max_inventory_size")
_MAP_ELITES_FIELDS = ("cell_capacity", "sampling_weights")
_SAMPLING_WEIGHTS = {
    "underexplored": 40,
    "elites": 30,
    "failure_matched": 20,
    "stepping_stones": 10,
}
_HYPERBAND_FIELDS = ("brackets", "reduction_factor")
_BRACKETS = {"explore": (8, 32, 80), "confirm": (32, 80), "replay": (80,)}
_PROPOSER_FIELDS = (
    "provider",
    "max_proposals_per_generation",
    "max_response_bytes",
)
_ADAPTER_FIELDS = ("task_timeout_seconds", "max_forecast_values")


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _exact_mapping(value: object, fields: tuple[str, ...], name: str) -> dict[str, object]:
    strict = _strict_json_value(value, field=name)
    return _require_exact_schema(strict, fields, field=name)


def _validate_runtime_fingerprints(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("runtime_fingerprints must be a non-empty object")
    strict = _strict_json_value(value, field="runtime_fingerprints")
    assert isinstance(strict, dict)
    result: dict[str, str] = {}
    for name, digest in strict.items():
        if not name:
            raise ValueError("runtime_fingerprints keys must be non-empty")
        result[name] = require_sha256(digest, f"runtime_fingerprints.{name}")
    return _freeze_json_value(result)  # type: ignore[return-value]


def _validate_budget(value: object, profile: str) -> BudgetPlan:
    payload = _exact_mapping(value, _BUDGET_FIELDS, "budget")
    hard_limit = _positive_int(payload["hard_limit_seconds"], "budget.hard_limit_seconds")
    reserve = payload["finalization_reserve_fraction"]
    if type(reserve) is not float or not math.isfinite(reserve) or not 0.0 <= reserve < 1.0:
        raise ValueError(
            "budget.finalization_reserve_fraction must be a finite float in [0, 1)"
        )
    if profile == "formal" and (hard_limit, reserve) != (14_400, 0.2):
        raise ValueError(
            "formal profile requires hard_limit_seconds=14400 and "
            "finalization_reserve_fraction=0.2"
        )
    ceilings_payload = payload["ceilings"]
    if not isinstance(ceilings_payload, Mapping):
        raise ValueError("budget.ceilings must be an object")
    try:
        ceilings = ResourceUse.from_payload(ceilings_payload)
    except ValueError as error:
        raise ValueError(f"invalid budget.ceilings: {error}") from error
    return BudgetPlan(hard_limit, reserve, ceilings)


def _validate_fixed_bundle(value: object) -> Mapping[str, str]:
    payload = _exact_mapping(value, _FIXED_BUNDLE_FIELDS, "fixed_bundle_components")
    result = {
        field: require_sha256(payload[field], f"fixed_bundle_components.{field}")
        for field in _FIXED_BUNDLE_FIELDS
    }
    return _freeze_json_value(result)  # type: ignore[return-value]


def _validate_mutation(value: object) -> Mapping[str, object]:
    payload = _exact_mapping(value, _MUTATION_FIELDS, "mutation")
    operators_value = payload["operators"]
    if not isinstance(operators_value, (list, tuple)) or not operators_value:
        raise ValueError("mutation.operators must be a non-empty list")
    operators = tuple(operators_value)
    if any(type(operator) is not str or operator not in MUTATION_OPERATORS for operator in operators):
        raise ValueError("mutation.operators contains an unknown operator")
    if len(operators) != len(set(operators)):
        raise ValueError("mutation.operators must be unique")
    result = {
        "operators": operators,
        "max_parents_per_child": _positive_int(
            payload["max_parents_per_child"], "mutation.max_parents_per_child"
        ),
        "max_inventory_size": _positive_int(
            payload["max_inventory_size"], "mutation.max_inventory_size"
        ),
    }
    return _freeze_json_value(result)  # type: ignore[return-value]


def _validate_map_elites(value: object) -> Mapping[str, object]:
    payload = _exact_mapping(value, _MAP_ELITES_FIELDS, "map_elites")
    capacity = _positive_int(payload["cell_capacity"], "map_elites.cell_capacity")
    if capacity > 4:
        raise ValueError("map_elites.cell_capacity must be in [1, 4]")
    weights = _exact_mapping(
        payload["sampling_weights"], tuple(_SAMPLING_WEIGHTS), "map_elites.sampling_weights"
    )
    if any(type(value) is not int for value in weights.values()) or weights != _SAMPLING_WEIGHTS:
        raise ValueError("map_elites.sampling_weights must be exactly 40/30/20/10")
    result = {
        "cell_capacity": capacity,
        "sampling_weights": dict(weights),
    }
    return _freeze_json_value(result)  # type: ignore[return-value]


def _validate_hyperband(value: object) -> Mapping[str, object]:
    payload = _exact_mapping(value, _HYPERBAND_FIELDS, "hyperband")
    brackets = _exact_mapping(payload["brackets"], tuple(_BRACKETS), "hyperband.brackets")
    for name, expected in _BRACKETS.items():
        configured = brackets[name]
        if (
            not isinstance(configured, (list, tuple))
            or any(type(resource) is not int for resource in configured)
            or tuple(configured) != expected
        ):
            raise ValueError(f"hyperband.brackets.{name} must be exactly {list(expected)}")
    result = {
        "brackets": {name: resources for name, resources in _BRACKETS.items()},
        "reduction_factor": _positive_int(
            payload["reduction_factor"], "hyperband.reduction_factor"
        ),
    }
    return _freeze_json_value(result)  # type: ignore[return-value]


def _validate_proposer(value: object) -> Mapping[str, object]:
    payload = _exact_mapping(value, _PROPOSER_FIELDS, "proposer")
    provider = payload["provider"]
    if type(provider) is not str or provider not in {"hybrid", "deterministic"}:
        raise ValueError("proposer.provider must be hybrid or deterministic")
    result = {
        "provider": provider,
        "max_proposals_per_generation": _positive_int(
            payload["max_proposals_per_generation"],
            "proposer.max_proposals_per_generation",
        ),
        "max_response_bytes": _positive_int(
            payload["max_response_bytes"], "proposer.max_response_bytes"
        ),
    }
    return _freeze_json_value(result)  # type: ignore[return-value]


def _validate_adapter(value: object) -> Mapping[str, object]:
    payload = _exact_mapping(value, _ADAPTER_FIELDS, "adapter")
    timeout = payload["task_timeout_seconds"]
    if type(timeout) is not float or not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("adapter.task_timeout_seconds must be a positive finite float")
    result = {
        "task_timeout_seconds": timeout,
        "max_forecast_values": _positive_int(
            payload["max_forecast_values"], "adapter.max_forecast_values"
        ),
    }
    return _freeze_json_value(result)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class NumericalQDConfigV2:
    """Immutable, exact configuration for one Numerical-only QD run."""

    schema_version: int
    profile: Literal["smoke", "pilot", "formal"]
    seed: int
    kernel_protocol: KernelProtocolCommitment
    runtime_fingerprints: Mapping[str, str]
    budget: BudgetPlan
    fixed_bundle_components: Mapping[str, str]
    descriptor_policy: DescriptorPolicyV2
    mutation: Mapping[str, object]
    map_elites: Mapping[str, object]
    hyperband: Mapping[str, object]
    proposer: Mapping[str, object]
    adapter: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be exactly 1")
        if type(self.profile) is not str or self.profile not in {"smoke", "pilot", "formal"}:
            raise ValueError("profile must be smoke, pilot, or formal")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.kernel_protocol) is not KernelProtocolCommitment:
            raise ValueError("kernel_protocol must be a KernelProtocolCommitment")
        if type(self.budget) is not BudgetPlan:
            raise ValueError("budget must be a BudgetPlan")
        if self.profile == "formal" and (
            self.budget.hard_limit_seconds,
            self.budget.finalization_reserve_fraction,
        ) != (14_400, 0.2):
            raise ValueError(
                "formal profile requires hard_limit_seconds=14400 and "
                "finalization_reserve_fraction=0.2"
            )
        if type(self.descriptor_policy) is not DescriptorPolicyV2:
            raise ValueError("descriptor_policy must be a DescriptorPolicyV2")
        object.__setattr__(
            self,
            "runtime_fingerprints",
            _validate_runtime_fingerprints(self.runtime_fingerprints),
        )
        object.__setattr__(
            self,
            "fixed_bundle_components",
            _validate_fixed_bundle(self.fixed_bundle_components),
        )
        object.__setattr__(self, "mutation", _validate_mutation(self.mutation))
        object.__setattr__(self, "map_elites", _validate_map_elites(self.map_elites))
        object.__setattr__(self, "hyperband", _validate_hyperband(self.hyperband))
        object.__setattr__(self, "proposer", _validate_proposer(self.proposer))
        object.__setattr__(self, "adapter", _validate_adapter(self.adapter))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "NumericalQDConfigV2":
        values = _require_exact_schema(payload, _CONFIG_FIELDS, field="numerical_qd_config")
        if type(values["schema_version"]) is not int or values["schema_version"] != 1:
            raise ValueError("schema_version must be exactly 1")
        profile = values["profile"]
        if type(profile) is not str or profile not in {"smoke", "pilot", "formal"}:
            raise ValueError("profile must be smoke, pilot, or formal")
        seed = values["seed"]
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        protocol_payload = values["kernel_protocol"]
        if not isinstance(protocol_payload, Mapping):
            raise ValueError("kernel_protocol must be an object with the exact schema")
        descriptor_payload = values["descriptor_policy"]
        if not isinstance(descriptor_payload, Mapping):
            raise ValueError("descriptor_policy must be an object with the exact schema")
        return cls(
            schema_version=1,
            profile=profile,  # type: ignore[arg-type]
            seed=seed,
            kernel_protocol=KernelProtocolCommitment.from_payload(protocol_payload),
            runtime_fingerprints=_validate_runtime_fingerprints(values["runtime_fingerprints"]),
            budget=_validate_budget(values["budget"], profile),
            fixed_bundle_components=_validate_fixed_bundle(values["fixed_bundle_components"]),
            descriptor_policy=DescriptorPolicyV2.from_payload(descriptor_payload),
            mutation=_validate_mutation(values["mutation"]),
            map_elites=_validate_map_elites(values["map_elites"]),
            hyperband=_validate_hyperband(values["hyperband"]),
            proposer=_validate_proposer(values["proposer"]),
            adapter=_validate_adapter(values["adapter"]),
        )

    def to_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "seed": self.seed,
            "kernel_protocol": self.kernel_protocol.to_payload(),
            "runtime_fingerprints": self.runtime_fingerprints,
            "budget": {
                "hard_limit_seconds": self.budget.hard_limit_seconds,
                "finalization_reserve_fraction": self.budget.finalization_reserve_fraction,
                "ceilings": self.budget.ceilings.to_payload(),
            },
            "fixed_bundle_components": self.fixed_bundle_components,
            "descriptor_policy": self.descriptor_policy.to_payload(),
            "mutation": self.mutation,
            "map_elites": self.map_elites,
            "hyperband": self.hyperband,
            "proposer": self.proposer,
            "adapter": self.adapter,
        }
        strict = _strict_json_value(payload)
        assert isinstance(strict, dict)
        return strict

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


def load_numerical_qd_config(path: str | Path) -> NumericalQDConfigV2:
    """Read a duplicate-free Numerical QD config in exact canonical form."""
    source = Path(path)
    raw = source.read_bytes()
    payload = strict_json_loads(
        raw.decode("utf-8"), context=f"Evolution V2 Numerical QD config {source}"
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"Evolution V2 Numerical QD config {source} must be a JSON object")
    if raw != canonical_v2_bytes(payload):
        raise ValueError(f"Evolution V2 Numerical QD config {source} must use canonical bytes")
    return NumericalQDConfigV2.from_payload(payload)
