"""Strict, content-addressed contracts for the isolated Evolution V2 kernel."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from common.payload import strict_json_loads


JsonScalar = str | int | float | bool | None

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PROTOCOL_FIELDS = (
    "task_materializer",
    "split_manifest",
    "metric_policy",
    "label_firewall",
    "artifact_validator",
    "sandbox_policy",
    "promotion_policy",
)
_FEEDBACK_FIELDS = (
    "parent_sha256",
    "train_evaluation_sha256",
    "train_objectives",
    "train_behavior_descriptors",
    "failure_categories",
    "remaining_proposal_budget",
)
_CONFIG_FIELDS = (
    "schema_version",
    "profile",
    "seed",
    "scheduler",
    "enabled_mutation_scopes",
    "archive_capacities",
    "hyperband",
    "runtime_fingerprints",
    "kernel_protocol",
    "hard_limit_seconds",
    "finalization_reserve_fraction",
    "runner",
)
_RESERVED_FEEDBACK_KEYS = frozenset(
    {
        "dev_comparison",
        "dev_metrics",
        "public_ids",
        "future_values",
        "evaluator_labels",
        "holdout",
    }
)


@runtime_checkable
class EvolutionArtifact(Protocol):
    """Semantic interface implemented by every evolvable V2 artifact."""

    @property
    def artifact_kind(self) -> str: ...

    @property
    def schema_version(self) -> int: ...

    def to_payload(self) -> Mapping[str, object]: ...

    def canonical_bytes(self) -> bytes: ...

    def fingerprint(self) -> str: ...


def _strict_json_value(value: object, *, field: str = "payload") -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError(f"{field} keys must be strings")
        return {
            key: _strict_json_value(item, field=f"{field}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item, field=f"{field}[]") for item in value]
    raise TypeError(f"{field} contains non-JSON value {type(value).__name__}")


def _freeze_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def canonical_v2_bytes(payload: Mapping[str, object]) -> bytes:
    """Return the one legal compact JSON representation for a V2 payload."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    plain = _strict_json_value(payload)
    return (
        json.dumps(
            plain,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def fingerprint_payload(payload: Mapping[str, object]) -> str:
    """Hash the exact canonical V2 bytes for ``payload``."""
    return hashlib.sha256(canonical_v2_bytes(payload)).hexdigest()


def require_sha256(value: object, field: str = "sha256") -> str:
    """Require one canonical, lowercase SHA-256 hexadecimal string."""
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical lowercase SHA-256 string")
    return value


def _require_exact_schema(
    payload: object, expected: Sequence[str], *, field: str
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} must be an object with the exact schema")
    if any(type(key) is not str for key in payload):
        raise ValueError(f"{field} keys must be strings")
    actual = set(payload)
    expected_set = set(expected)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        unexpected = sorted(actual - expected_set)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        suffix = f" ({', '.join(details)})" if details else ""
        raise ValueError(f"{field} must use the exact schema{suffix}")
    return dict(payload)


def _require_int(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _require_choice(value: object, field: str, choices: frozenset[str]) -> str:
    if type(value) is not str or value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{field} must be one of: {expected}")
    return value


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    plain = _strict_json_value(value, field=field)
    assert isinstance(plain, dict)
    return plain


@dataclass(frozen=True, slots=True)
class KernelProtocolCommitment:
    """Content identities for every immutable L0 protocol binding."""

    task_materializer: str
    split_manifest: str
    metric_policy: str
    label_firewall: str
    artifact_validator: str
    sandbox_policy: str
    promotion_policy: str

    def __post_init__(self) -> None:
        for field in _PROTOCOL_FIELDS:
            require_sha256(getattr(self, field), f"kernel_protocol.{field}")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "KernelProtocolCommitment":
        values = _require_exact_schema(payload, _PROTOCOL_FIELDS, field="kernel_protocol")
        return cls(**{field: require_sha256(values[field], field) for field in _PROTOCOL_FIELDS})

    def to_payload(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in _PROTOCOL_FIELDS}

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


def _reject_reserved_feedback_keys(value: object, *, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _RESERVED_FEEDBACK_KEYS:
                raise ValueError(f"{field} contains reserved evaluator-only key {key}")
            _reject_reserved_feedback_keys(item, field=f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_reserved_feedback_keys(item, field=f"{field}[]")


@dataclass(frozen=True, slots=True)
class SanitizedEvolutionFeedback:
    """Closed, primitive Train-only information made visible to proposers."""

    parent_sha256: str
    train_evaluation_sha256: str
    train_objectives: Mapping[str, object]
    train_behavior_descriptors: Mapping[str, object]
    failure_categories: tuple[str, ...]
    remaining_proposal_budget: Mapping[str, object]

    def __post_init__(self) -> None:
        require_sha256(self.parent_sha256, "parent_sha256")
        require_sha256(self.train_evaluation_sha256, "train_evaluation_sha256")
        objectives = _require_mapping(self.train_objectives, "train_objectives")
        descriptors = _require_mapping(
            self.train_behavior_descriptors, "train_behavior_descriptors"
        )
        budget = _require_mapping(
            self.remaining_proposal_budget, "remaining_proposal_budget"
        )
        if not isinstance(self.failure_categories, (list, tuple)) or any(
            type(category) is not str or not category
            for category in self.failure_categories
        ):
            raise ValueError("failure_categories must be a list of non-empty strings")
        categories = tuple(self.failure_categories)
        for field, value in (
            ("train_objectives", objectives),
            ("train_behavior_descriptors", descriptors),
            ("failure_categories", categories),
            ("remaining_proposal_budget", budget),
        ):
            _reject_reserved_feedback_keys(value, field=field)
        object.__setattr__(self, "train_objectives", _freeze_json_value(objectives))
        object.__setattr__(
            self,
            "train_behavior_descriptors",
            _freeze_json_value(descriptors),
        )
        object.__setattr__(self, "failure_categories", categories)
        object.__setattr__(
            self,
            "remaining_proposal_budget",
            _freeze_json_value(budget),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SanitizedEvolutionFeedback":
        values = _require_exact_schema(payload, _FEEDBACK_FIELDS, field="feedback")
        return cls(
            parent_sha256=values["parent_sha256"],  # type: ignore[arg-type]
            train_evaluation_sha256=values["train_evaluation_sha256"],  # type: ignore[arg-type]
            train_objectives=values["train_objectives"],  # type: ignore[arg-type]
            train_behavior_descriptors=values["train_behavior_descriptors"],  # type: ignore[arg-type]
            failure_categories=values["failure_categories"],  # type: ignore[arg-type]
            remaining_proposal_budget=values["remaining_proposal_budget"],  # type: ignore[arg-type]
        )

    def to_payload(self) -> dict[str, object]:
        payload = {
            "parent_sha256": self.parent_sha256,
            "train_evaluation_sha256": self.train_evaluation_sha256,
            "train_objectives": self.train_objectives,
            "train_behavior_descriptors": self.train_behavior_descriptors,
            "failure_categories": self.failure_categories,
            "remaining_proposal_budget": self.remaining_proposal_budget,
        }
        plain = _strict_json_value(payload)
        assert isinstance(plain, dict)
        return plain


@dataclass(frozen=True, slots=True)
class EvolutionV2Config:
    """Closed Project 1 configuration; later algorithms are data only here."""

    schema_version: int
    profile: Literal["smoke", "pilot", "formal", "public"]
    seed: int
    scheduler: Literal["ucb", "thompson"]
    enabled_mutation_scopes: tuple[
        Literal["numerical", "retrieval", "decision", "joint"], ...
    ]
    archive_capacities: Mapping[str, int]
    hyperband: Mapping[str, object]
    runtime_fingerprints: Mapping[str, str]
    kernel_protocol: KernelProtocolCommitment
    hard_limit_seconds: int
    finalization_reserve_fraction: float
    runner: Literal["deterministic_fake", "production"]

    def __post_init__(self) -> None:
        schema_version = _require_int(self.schema_version, "schema_version")
        if schema_version != 1:
            raise ValueError("schema_version must be exactly 1")
        profile = _require_choice(
            self.profile, "profile", frozenset({"smoke", "pilot", "formal", "public"})
        )
        _require_int(self.seed, "seed")
        _require_choice(self.scheduler, "scheduler", frozenset({"ucb", "thompson"}))

        scopes_value = self.enabled_mutation_scopes
        if not isinstance(scopes_value, (list, tuple)):
            raise ValueError("enabled_mutation_scopes must be a list")
        scopes = tuple(scopes_value)
        valid_scopes = frozenset({"numerical", "retrieval", "decision", "joint"})
        if not scopes:
            raise ValueError("enabled_mutation_scopes must not be empty")
        if any(type(scope) is not str or scope not in valid_scopes for scope in scopes):
            raise ValueError("enabled_mutation_scopes contains an unknown scope")
        if len(scopes) != len(set(scopes)):
            raise ValueError("enabled_mutation_scopes must be unique")

        capacities_plain = _require_mapping(self.archive_capacities, "archive_capacities")
        if not capacities_plain:
            raise ValueError("archive_capacities must not be empty")
        capacities: dict[str, int] = {}
        for name, value in capacities_plain.items():
            if not name:
                raise ValueError("archive_capacities keys must not be empty")
            capacities[name] = _require_int(
                value, f"archive_capacities.{name}", positive=True
            )

        hyperband = _require_mapping(self.hyperband, "hyperband")
        runtime_plain = _require_mapping(
            self.runtime_fingerprints, "runtime_fingerprints"
        )
        if not runtime_plain:
            raise ValueError("runtime_fingerprints must not be empty")
        runtimes: dict[str, str] = {}
        for name, value in runtime_plain.items():
            if not name:
                raise ValueError("runtime_fingerprints keys must not be empty")
            runtimes[name] = require_sha256(value, f"runtime_fingerprints.{name}")

        if not isinstance(self.kernel_protocol, KernelProtocolCommitment):
            raise ValueError("kernel_protocol must be a KernelProtocolCommitment")
        hard_limit = _require_int(
            self.hard_limit_seconds, "hard_limit_seconds", positive=True
        )
        reserve = self.finalization_reserve_fraction
        if type(reserve) is not float or not math.isfinite(reserve):
            raise ValueError("finalization_reserve_fraction must be a finite float")
        if not 0.0 <= reserve < 1.0:
            raise ValueError("finalization_reserve_fraction must be in [0, 1)")
        _require_choice(
            self.runner,
            "runner",
            frozenset({"deterministic_fake", "production"}),
        )
        if profile == "formal" and (hard_limit, reserve) != (14_400, 0.2):
            raise ValueError(
                "formal profile requires hard_limit_seconds=14400 and "
                "finalization_reserve_fraction=0.2"
            )

        object.__setattr__(self, "enabled_mutation_scopes", scopes)
        object.__setattr__(
            self, "archive_capacities", _freeze_json_value(capacities)
        )
        object.__setattr__(self, "hyperband", _freeze_json_value(hyperband))
        object.__setattr__(
            self, "runtime_fingerprints", _freeze_json_value(runtimes)
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "EvolutionV2Config":
        values = _require_exact_schema(payload, _CONFIG_FIELDS, field="config")
        protocol_payload = values["kernel_protocol"]
        if not isinstance(protocol_payload, Mapping):
            raise ValueError("kernel_protocol must be an object with the exact schema")
        return cls(
            schema_version=values["schema_version"],  # type: ignore[arg-type]
            profile=values["profile"],  # type: ignore[arg-type]
            seed=values["seed"],  # type: ignore[arg-type]
            scheduler=values["scheduler"],  # type: ignore[arg-type]
            enabled_mutation_scopes=values["enabled_mutation_scopes"],  # type: ignore[arg-type]
            archive_capacities=values["archive_capacities"],  # type: ignore[arg-type]
            hyperband=values["hyperband"],  # type: ignore[arg-type]
            runtime_fingerprints=values["runtime_fingerprints"],  # type: ignore[arg-type]
            kernel_protocol=KernelProtocolCommitment.from_payload(protocol_payload),
            hard_limit_seconds=values["hard_limit_seconds"],  # type: ignore[arg-type]
            finalization_reserve_fraction=values["finalization_reserve_fraction"],  # type: ignore[arg-type]
            runner=values["runner"],  # type: ignore[arg-type]
        )

    def to_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "seed": self.seed,
            "scheduler": self.scheduler,
            "enabled_mutation_scopes": self.enabled_mutation_scopes,
            "archive_capacities": self.archive_capacities,
            "hyperband": self.hyperband,
            "runtime_fingerprints": self.runtime_fingerprints,
            "kernel_protocol": self.kernel_protocol.to_payload(),
            "hard_limit_seconds": self.hard_limit_seconds,
            "finalization_reserve_fraction": self.finalization_reserve_fraction,
            "runner": self.runner,
        }
        plain = _strict_json_value(payload)
        assert isinstance(plain, dict)
        return plain


def load_v2_config(path: str | Path) -> EvolutionV2Config:
    """Load one V2 config using strict JSON and its closed schema."""
    source = Path(path)
    payload = strict_json_loads(
        source.read_text(encoding="utf-8"), context=f"Evolution V2 config {source}"
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"Evolution V2 config {source} must be a JSON object")
    return EvolutionV2Config.from_payload(payload)
