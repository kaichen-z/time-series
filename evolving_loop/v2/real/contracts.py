"""Strict contracts for bounded real Evolution V2 runs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal

from common.payload import strict_json_loads
from ..contracts import (
    _require_exact_schema, _strict_json_value,
    canonical_v2_bytes, fingerprint_payload, require_sha256,
)

_PROFILE_ALLOCATIONS = {
    "real-30m": {"p2": 840, "p3": 360, "p4": 120, "p5": 120, "finalization": 360},
    "real-1h": {"p2": 1680, "p3": 720, "p4": 240, "p5": 240, "finalization": 720},
}
_ROLE_ORDER = (
    "split",
    "tasks",
    "numerical_seed",
    "numerical_source_seed",
    "forecast_cache",
    "retrieval_seed",
    "source_seed",
)
_RUNTIME_ROLE_ORDER = ("python", "runtime", "task_loader", "forecast_store", "model_cache", "codex_cli")
P3_NUMERICAL_MODE = "p3_dictionary"


def require_p3_numerical_mode(value: object) -> Literal["p3_dictionary"]:
    """Admit the sole Numerical proposal mode for new real P3 runs."""
    if type(value) is not str or value != P3_NUMERICAL_MODE:
        raise ValueError("numerical_mode must be exactly p3_dictionary")
    return value


def _contract_payload(value: object, cls):
    if isinstance(value, cls):
        return value
    return cls.from_payload(value)


class _Canonical:
    @classmethod
    def from_payload(cls, payload):
        values = _require_exact_schema(payload, tuple(f.name for f in fields(cls)), field=cls.__name__)
        # Admit only JSON-compatible primitives before constructing contracts;
        # this also prevents callers from smuggling mutable/non-JSON objects in.
        normalized = _strict_json_value(values, field=cls.__name__)
        assert isinstance(normalized, dict)
        return cls(**normalized)

    def to_payload(self):
        return _strict_json_value({f.name: _plain(getattr(self, f.name)) for f in fields(self)})

    def canonical_bytes(self):
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self):
        return fingerprint_payload(self.to_payload())


def _plain(value):
    if isinstance(value, _Canonical):
        return value.to_payload()
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _version(value):
    if type(value) is not int or value != 1:
        raise ValueError("schema_version must be exactly 1")


def _path(value, field):
    if type(value) is not str or not value or value.startswith("/"):
        raise ValueError(f"{field} must be a relative path")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{field} must be a confined relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{field} must be a confined relative path")
    return value


def _rows(value, cls, field):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    result = tuple(_contract_payload(v, cls) for v in value)
    roles = tuple(getattr(v, "role") for v in result)
    if cls is RealInputFileV2 and any(role not in _ROLE_ORDER for role in roles):
        raise ValueError(f"{field} roles must be known")
    if len(roles) != len(set(roles)):
        raise ValueError(f"{field} roles must be unique")
    role_order = _ROLE_ORDER if cls is RealInputFileV2 else _RUNTIME_ROLE_ORDER
    if cls is RealRuntimeLocationV2 and any(role not in role_order for role in roles):
        raise ValueError(f"{field} roles must be known")
    order = {name: i for i, name in enumerate(role_order)}
    if roles != tuple(sorted(roles, key=lambda r: order.get(r, len(order)))):
        raise ValueError(f"{field} must use canonical role order")
    return result


@dataclass(frozen=True, slots=True)
class RealScheduleV2(_Canonical):
    schema_version: int
    profile: str
    allocations: Mapping[str, int]
    total_seconds: int
    finalization_reserve_seconds: int

    def __post_init__(self):
        _version(self.schema_version)
        if self.profile not in _PROFILE_ALLOCATIONS:
            raise ValueError("profile must be an approved real profile")
        expected = _PROFILE_ALLOCATIONS[self.profile]
        if dict(self.allocations) != expected:
            raise ValueError("allocations must match the approved profile exactly")
        if self.total_seconds != sum(expected.values()) or self.finalization_reserve_seconds != expected["finalization"]:
            raise ValueError("total_seconds and finalization reserve must match profile")
        object.__setattr__(self, "allocations", MappingProxyType(dict(expected)))


@dataclass(frozen=True, slots=True)
class RealModelBindingV2(_Canonical):
    schema_version: int
    name: str
    reasoning_effort: Literal["medium"]

    def __post_init__(self):
        _version(self.schema_version)
        if self.name != "gpt-5.6-luna" or self.reasoning_effort != "medium":
            raise ValueError("model binding must be gpt-5.6-luna/medium")


@dataclass(frozen=True, slots=True)
class RealInputFileV2(_Canonical):
    role: str
    relative_path: str
    sha256: str

    def __post_init__(self):
        if type(self.role) is not str or not self.role:
            raise ValueError("role must be non-empty")
        _path(self.relative_path, "relative_path")
        require_sha256(self.sha256)


@dataclass(frozen=True, slots=True)
class RealRuntimeLocationV2(_Canonical):
    role: str
    relative_path: str
    identity_sha256: str

    def __post_init__(self):
        if type(self.role) is not str or not self.role:
            raise ValueError("role must be non-empty")
        _path(self.relative_path, "relative_path")
        require_sha256(self.identity_sha256, "identity_sha256")


@dataclass(frozen=True, slots=True)
class RealEvolutionManifestV2(_Canonical):
    schema_version: int
    profile: str
    model: RealModelBindingV2 | Mapping[str, object]
    files: tuple[RealInputFileV2, ...] | Sequence[Mapping[str, object]]
    runtime_locations: tuple[RealRuntimeLocationV2, ...] | Sequence[Mapping[str, object]]
    l0_fingerprints: Mapping[str, str]

    def __post_init__(self):
        _version(self.schema_version)
        schedule = PROFILE_SCHEDULES.get(self.profile)
        if schedule is None:
            raise ValueError("profile must be an approved real profile")
        model = _contract_payload(self.model, RealModelBindingV2)
        files = _rows(self.files, RealInputFileV2, "files")
        locations = _rows(self.runtime_locations, RealRuntimeLocationV2, "runtime_locations")
        l0 = dict(self.l0_fingerprints)
        for name, digest in l0.items():
            require_sha256(digest, f"l0_fingerprints.{name}")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "runtime_locations", locations)
        object.__setattr__(self, "l0_fingerprints", MappingProxyType(l0))


@dataclass(frozen=True, slots=True)
class RealStageRecordV2(_Canonical):
    stage: str
    grant_seconds: int
    charged_seconds: int
    status: Literal["complete", "incomplete", "failed"]
    completion_sha256: str | None
    preceding_progress_sha256: str | None

    def __post_init__(self):
        if type(self.stage) is not str or self.stage not in {"p2", "p3", "p4", "p5"}:
            raise ValueError("stage must be p2, p3, p4, or p5")
        if type(self.grant_seconds) is not int or self.grant_seconds <= 0:
            raise ValueError("grant_seconds must be positive")
        if type(self.charged_seconds) is not int or self.charged_seconds < 0:
            raise ValueError("charged_seconds must be non-negative")
        if self.status not in {"complete", "incomplete", "failed"}:
            raise ValueError("invalid stage status")
        if self.completion_sha256 is not None:
            require_sha256(self.completion_sha256, "completion_sha256")
        if self.preceding_progress_sha256 is not None:
            require_sha256(self.preceding_progress_sha256, "preceding_progress_sha256")


@dataclass(frozen=True, slots=True)
class RealEvolutionCheckpointV2(_Canonical):
    phase: str
    stage_records: tuple[RealStageRecordV2, ...]
    active_stage: str | None
    carry_seconds: int
    budget_checkpoint: Mapping[str, object]
    handoff_sha256s: Mapping[str, str]
    completion_sha256: str | None

    def __post_init__(self):
        object.__setattr__(self, "stage_records", tuple(_contract_payload(v, RealStageRecordV2) for v in self.stage_records))
        if self.active_stage is not None and self.active_stage not in {"p2", "p3", "p4", "p5"}:
            raise ValueError("invalid active_stage")
        if type(self.carry_seconds) is not int or self.carry_seconds < 0:
            raise ValueError("carry_seconds must be non-negative")
        for key, value in self.handoff_sha256s.items():
            require_sha256(value, f"handoff_sha256s.{key}")
        if self.completion_sha256 is not None:
            require_sha256(self.completion_sha256, "completion_sha256")


@dataclass(frozen=True, slots=True)
class RealRunResultV2(_Canonical):
    status: Literal["complete", "incomplete", "failed"]
    manifest_sha256: str
    model_binding_sha256: str
    stage_records: tuple[RealStageRecordV2, ...]
    completion_sha256: str | None
    public_test_accessed: bool

    def __post_init__(self):
        if self.status not in {"complete", "incomplete", "failed"}:
            raise ValueError("status must be complete, incomplete, or failed")
        require_sha256(self.manifest_sha256, "manifest_sha256")
        require_sha256(self.model_binding_sha256, "model_binding_sha256")
        object.__setattr__(self, "stage_records", tuple(_contract_payload(v, RealStageRecordV2) for v in self.stage_records))
        if self.completion_sha256 is not None:
            require_sha256(self.completion_sha256, "completion_sha256")
        if type(self.public_test_accessed) is not bool:
            raise ValueError("public_test_accessed must be a boolean")


PROFILE_SCHEDULES = MappingProxyType({
    name: RealScheduleV2(1, name, allocations, sum(allocations.values()), allocations["finalization"])
    for name, allocations in _PROFILE_ALLOCATIONS.items()
})


def load_real_schedule(path):
    payload = strict_json_loads(path.read_text(encoding="utf-8"), context=f"real profile {path}")
    return RealScheduleV2.from_payload(payload)
