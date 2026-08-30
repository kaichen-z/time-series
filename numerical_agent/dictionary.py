"""JSON-compatible schemas for externally supplied numerical methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence, cast

from common.evolution_core.contracts import load_active_release, metric_policy_metadata
from common.payload import require_strings as _tuple_of_strings

from .config import ALLOWED_FAMILIES, METHOD_STATUSES


MethodFamily = Literal["statistical", "foundation", "combined"]
MethodStatus = Literal[
    "unimplemented",
    "accepted",
    "specialized",
    "quarantined",
    "unavailable",
    "discarded",
]


@dataclass(frozen=True)
class MethodDefinition:
    method_id: str
    family: MethodFamily
    description: str
    assumptions: tuple[str, ...] = ()
    failure_conditions: tuple[str, ...] = ()
    implementation_spec: Mapping[str, object] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    status: MethodStatus = "unimplemented"

    def __post_init__(self) -> None:
        if not self.method_id.strip():
            raise ValueError("method_id must not be empty")
        if self.family not in ALLOWED_FAMILIES:
            raise ValueError(f"unsupported method family: {self.family!r}")
        if not self.description.strip():
            raise ValueError("description must not be empty")
        if self.status not in METHOD_STATUSES:
            raise ValueError(f"unsupported method status: {self.status!r}")

    def to_payload(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "family": self.family,
            "description": self.description,
            "assumptions": list(self.assumptions),
            "failure_conditions": list(self.failure_conditions),
            "implementation_spec": dict(self.implementation_spec),
            "dependencies": list(self.dependencies),
            "status": self.status,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MethodDefinition":
        spec = payload.get("implementation_spec", {})
        if not isinstance(spec, Mapping):
            raise ValueError("implementation_spec must be an object")
        return cls(
            method_id=str(payload["method_id"]),
            family=cast(MethodFamily, payload["family"]),
            description=str(payload["description"]),
            assumptions=_tuple_of_strings(payload.get("assumptions"), "assumptions"),
            failure_conditions=_tuple_of_strings(
                payload.get("failure_conditions"), "failure_conditions"
            ),
            implementation_spec=dict(spec),
            dependencies=_tuple_of_strings(payload.get("dependencies"), "dependencies"),
            status=cast(MethodStatus, payload.get("status", "unimplemented")),
        )


@dataclass(frozen=True)
class MethodCandidate:
    method_id: str
    provider: str
    implementation_kind: str
    implementation: Mapping[str, object]
    version: int = 1
    parent_version: int | None = None

    def __post_init__(self) -> None:
        if not self.method_id or not self.provider or not self.implementation_kind:
            raise ValueError("candidate identifiers must not be empty")
        if self.version <= 0:
            raise ValueError("candidate version must be positive")
        if self.parent_version is not None and self.parent_version <= 0:
            raise ValueError("candidate parent_version must be positive")

    def to_payload(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "provider": self.provider,
            "implementation_kind": self.implementation_kind,
            "implementation": dict(self.implementation),
            "version": self.version,
            "parent_version": self.parent_version,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MethodCandidate":
        implementation = payload.get("implementation", {})
        if not isinstance(implementation, Mapping):
            raise ValueError("candidate implementation must be an object")
        parent_version = payload.get("parent_version")
        return cls(
            method_id=str(payload["method_id"]),
            provider=str(payload["provider"]),
            implementation_kind=str(payload["implementation_kind"]),
            implementation=dict(implementation),
            version=int(payload.get("version", 1)),
            parent_version=int(parent_version) if parent_version is not None else None,
        )


@dataclass(frozen=True)
class MethodRecord:
    definition: MethodDefinition
    candidate: MethodCandidate | None = None
    status: MethodStatus = "unimplemented"
    revision_count: int = 0
    train_summary: Mapping[str, float] = field(default_factory=dict)
    implementation_attempts: int = 0

    def __post_init__(self) -> None:
        if self.status not in METHOD_STATUSES:
            raise ValueError(f"unsupported method status: {self.status!r}")
        if self.revision_count < 0:
            raise ValueError("revision_count must be non-negative")
        if self.implementation_attempts < 0:
            raise ValueError("implementation_attempts must be non-negative")
        if self.candidate is not None and self.candidate.method_id != self.definition.method_id:
            raise ValueError("candidate method_id does not match its definition")

    def to_payload(self) -> dict[str, object]:
        return {
            "definition": self.definition.to_payload(),
            "candidate": self.candidate.to_payload() if self.candidate else None,
            "status": self.status,
            "revision_count": self.revision_count,
            "train_summary": dict(self.train_summary),
            "implementation_attempts": self.implementation_attempts,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MethodRecord":
        definition = payload.get("definition")
        if not isinstance(definition, Mapping):
            raise ValueError("method record definition must be an object")
        candidate_payload = payload.get("candidate")
        if candidate_payload is not None and not isinstance(candidate_payload, Mapping):
            raise ValueError("method record candidate must be an object or null")
        summary = payload.get("train_summary", {})
        if not isinstance(summary, Mapping):
            raise ValueError("train_summary must be an object")
        return cls(
            definition=MethodDefinition.from_payload(definition),
            candidate=(
                MethodCandidate.from_payload(candidate_payload)
                if isinstance(candidate_payload, Mapping)
                else None
            ),
            status=cast(MethodStatus, payload.get("status", "unimplemented")),
            revision_count=int(payload.get("revision_count", 0)),
            train_summary={str(key): float(value) for key, value in summary.items()},
            implementation_attempts=int(payload.get("implementation_attempts", 0)),
        )


@dataclass(frozen=True)
class ToolDictionary:
    dictionary_id: str
    parent_dictionary_id: str | None
    generation: int
    methods: tuple[MethodRecord | MethodDefinition, ...]

    def __post_init__(self) -> None:
        if not self.dictionary_id:
            raise ValueError("dictionary_id must not be empty")
        if self.generation < 0:
            raise ValueError("dictionary generation must be non-negative")
        normalized = tuple(
            method
            if isinstance(method, MethodRecord)
            else MethodRecord(method, status=method.status)
            for method in self.methods
        )
        if not normalized:
            raise ValueError("dictionary must contain at least one method")
        method_ids = [record.definition.method_id for record in normalized]
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("dictionary contains duplicate method IDs")
        known_ids = set(method_ids)
        for record in normalized:
            missing = set(record.definition.dependencies) - known_ids
            if missing:
                raise ValueError(
                    f"method {record.definition.method_id!r} has unknown dependency {sorted(missing)!r}"
                )
        object.__setattr__(self, "methods", normalized)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            **metric_policy_metadata(),
            "dictionary_id": self.dictionary_id,
            "parent_dictionary_id": self.parent_dictionary_id,
            "generation": self.generation,
            "methods": [cast(MethodRecord, method).to_payload() for method in self.methods],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ToolDictionary":
        load_active_release(payload)
        if "generation" not in payload or type(payload["generation"]) is not int:
            raise ValueError("active dictionary generation must be an explicit integer")
        return cls._from_payload(payload, require_explicit_status=True)

    @classmethod
    def from_legacy_report_payload(
        cls, payload: Mapping[str, object]
    ) -> "ToolDictionary":
        """Parse a historical dictionary for reporting, never active evolution."""
        return cls._from_payload(payload, require_explicit_status=False)

    @classmethod
    def _from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        require_explicit_status: bool,
    ) -> "ToolDictionary":
        methods = payload.get("methods")
        if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes)):
            raise ValueError("dictionary methods must be a list")
        records = []
        for method in methods:
            if not isinstance(method, Mapping):
                raise ValueError("dictionary method must be an object")
            if require_explicit_status and "status" not in method:
                raise ValueError("active dictionary method status must be explicit")
            if "definition" in method:
                records.append(MethodRecord.from_payload(method))
            else:
                records.append(MethodRecord(MethodDefinition.from_payload(method)))
        parent = payload.get("parent_dictionary_id")
        return cls(
            dictionary_id=str(payload["dictionary_id"]),
            parent_dictionary_id=str(parent) if parent is not None else None,
            generation=int(payload.get("generation", 0)),
            methods=tuple(records),
        )
