"""Canonical contracts for one editable DGM-lite policy source file."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..contracts import canonical_v2_bytes, fingerprint_payload, require_sha256


ARM_ORDER = ("numerical", "retrieval", "decision", "joint")
_ARM_SET = frozenset(ARM_ORDER)
_VARIANT_FIELDS = (
    "schema_version",
    "parent_source_sha256",
    "operator",
    "files",
    "protocol_fingerprint",
    "runtime_fingerprint",
)
_REQUEST_FIELDS = (
    "schema_version",
    "enabled_arms",
    "step",
    "seed",
    "train_reward_by_arm",
)
_MAX_SOURCE_BYTES = 8192


def _exact_object(value: object, fields: tuple[str, ...], name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object with the exact schema")
    actual = set(value)
    expected = set(fields)
    if actual != expected:
        raise ValueError(f"{name} must use the exact schema")
    return dict(value)


def _schema_version(value: object, name: str) -> None:
    if type(value) is not int or value != 1:
        raise ValueError(f"{name}.schema_version must be exactly 1")


def _source_files(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"policy.py"}:
        raise ValueError("files must be exactly {'policy.py': source}")
    source = value.get("policy.py")
    if type(source) is not str:
        raise ValueError("files.policy.py must be a string")
    try:
        source_bytes = source.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("files.policy.py must be UTF-8 encodable") from error
    if not source_bytes or len(source_bytes) > _MAX_SOURCE_BYTES:
        raise ValueError("files.policy.py must be 1..8192 UTF-8 bytes")
    return MappingProxyType({"policy.py": source})


def _canonical_arms(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or isinstance(value, (str, bytes)):
        raise ValueError("enabled_arms must be a non-empty canonical arm list")
    arms = tuple(value)
    if not arms or any(type(arm) is not str or arm not in _ARM_SET for arm in arms):
        raise ValueError("enabled_arms must contain known arms")
    expected = tuple(arm for arm in ARM_ORDER if arm in arms)
    if arms != expected:
        raise ValueError("enabled_arms must be a non-empty canonical subset")
    return arms


def _rewards(value: object, arms: tuple[str, ...]) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(arms):
        raise ValueError("train_reward_by_arm must contain precisely enabled arms")
    result: dict[str, float] = {}
    for arm in arms:
        number = value[arm]
        if type(number) not in (int, float) or not math.isfinite(float(number)):
            raise ValueError(f"train_reward_by_arm.{arm} must be a finite scalar")
        result[arm] = float(number)
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class SourceVariantV2:
    """The full content-addressed candidate manifest, with one editable file."""

    schema_version: int
    parent_source_sha256: str | None
    operator: str
    files: Mapping[str, str]
    protocol_fingerprint: str
    runtime_fingerprint: str

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "SourceVariantV2")
        if self.parent_source_sha256 is not None:
            require_sha256(self.parent_source_sha256, "parent_source_sha256")
        if type(self.operator) is not str or not self.operator.strip():
            raise ValueError("operator must be a non-empty string")
        if self.operator == "seed" and self.parent_source_sha256 is not None:
            raise ValueError("seed source must not have a parent_source_sha256")
        if self.operator != "seed" and self.parent_source_sha256 is None:
            raise ValueError("non-seed source must have a parent_source_sha256")
        object.__setattr__(self, "files", _source_files(self.files))
        require_sha256(self.protocol_fingerprint, "protocol_fingerprint")
        require_sha256(self.runtime_fingerprint, "runtime_fingerprint")

    @classmethod
    def seed(
        cls, source: str, protocol_fingerprint: str, runtime_fingerprint: str
    ) -> "SourceVariantV2":
        return cls(1, None, "seed", {"policy.py": source}, protocol_fingerprint, runtime_fingerprint)

    @classmethod
    def child(
        cls, parent: "SourceVariantV2", source: str, operator: str
    ) -> "SourceVariantV2":
        if not isinstance(parent, cls):
            raise TypeError("parent must be a SourceVariantV2")
        return cls(
            1,
            parent.fingerprint(),
            operator,
            {"policy.py": source},
            parent.protocol_fingerprint,
            parent.runtime_fingerprint,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SourceVariantV2":
        values = _exact_object(payload, _VARIANT_FIELDS, "SourceVariantV2")
        return cls(
            values["schema_version"],
            values["parent_source_sha256"],
            values["operator"],
            values["files"],
            values["protocol_fingerprint"],
            values["runtime_fingerprint"],
        )

    @property
    def source(self) -> str:
        return self.files["policy.py"]

    def source_text_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parent_source_sha256": self.parent_source_sha256,
            "operator": self.operator,
            "files": dict(self.files),
            "protocol_fingerprint": self.protocol_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


@dataclass(frozen=True, slots=True)
class SourceRequestV2:
    """The only aggregate Train data a source policy may observe."""

    schema_version: int
    enabled_arms: tuple[str, ...]
    step: int
    seed: int
    train_reward_by_arm: Mapping[str, float]

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "SourceRequestV2")
        arms = _canonical_arms(self.enabled_arms)
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        object.__setattr__(self, "enabled_arms", arms)
        object.__setattr__(self, "train_reward_by_arm", _rewards(self.train_reward_by_arm, arms))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SourceRequestV2":
        values = _exact_object(payload, _REQUEST_FIELDS, "SourceRequestV2")
        return cls(
            values["schema_version"],
            values["enabled_arms"],
            values["step"],
            values["seed"],
            values["train_reward_by_arm"],
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "enabled_arms": list(self.enabled_arms),
            "step": self.step,
            "seed": self.seed,
            "train_reward_by_arm": dict(self.train_reward_by_arm),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())
