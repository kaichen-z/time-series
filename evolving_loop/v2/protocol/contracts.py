"""Canonical, immutable L1 infrastructure-protocol contracts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evolving_loop.v2.contracts import (
    canonical_v2_bytes,
    fingerprint_payload,
    require_sha256,
)


KINDS = (
    "backbone",
    "loader",
    "verifier_strategy",
    "diagnostic_metric",
    "schema_migration",
)

_COMPONENT_FIELDS = (
    "kind",
    "implementation_id",
    "implementation_version",
    "artifact_schema_version",
)
_PROTOCOL_FIELDS = (
    "schema_version",
    "protocol_version",
    "parent_protocol_sha256",
    "l0_commitment_sha256",
    "components",
)
_PROPOSAL_FIELDS = (
    "schema_version",
    "parent_protocol_sha256",
    "kind",
    "replacement",
    "compatibility_corpus_sha256",
)
_RELEASE_FIELDS = (
    "schema_version",
    "protocol_sha256",
    "l0_commitment_sha256",
    "evidence_sha256",
    "frozen_bundle_sha256",
    "runtime_fingerprint",
)


def _exact_payload(payload: object, fields: Sequence[str], *, name: str) -> dict[str, object]:
    if not isinstance(payload, Mapping) or any(type(key) is not str for key in payload):
        raise ValueError(f"{name} must be an object with the exact schema")
    actual = set(payload)
    expected = set(fields)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        suffix = f" ({', '.join(details)})" if details else ""
        raise ValueError(f"{name} must use the exact schema{suffix}")
    return dict(payload)


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _schema_v1(value: object, *, field: str = "schema_version") -> int:
    if type(value) is not int or value != 1:
        raise ValueError(f"{field} must be supported version 1")
    return value


def _kind(value: object, *, field: str = "kind") -> str:
    if type(value) is not str or value not in KINDS:
        raise ValueError(f"{field} must be one of: {', '.join(KINDS)}")
    return value


def _implementation_id(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("implementation_id must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ProtocolComponentV2:
    """One Host-resolved, versioned L1 component selection."""

    kind: str
    implementation_id: str
    implementation_version: int
    artifact_schema_version: int

    def __post_init__(self) -> None:
        _kind(self.kind)
        _implementation_id(self.implementation_id)
        _positive_int(self.implementation_version, field="implementation_version")
        _positive_int(self.artifact_schema_version, field="artifact_schema_version")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ProtocolComponentV2":
        values = _exact_payload(payload, _COMPONENT_FIELDS, name="protocol component")
        return cls(
            _kind(values["kind"]),
            _implementation_id(values["implementation_id"]),
            _positive_int(values["implementation_version"], field="implementation_version"),
            _positive_int(
                values["artifact_schema_version"], field="artifact_schema_version"
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "artifact_schema_version": self.artifact_schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


@dataclass(frozen=True, slots=True)
class InfrastructureProtocolV2:
    """A complete, content-addressed L1 manifest bound to immutable L0."""

    schema_version: int
    protocol_version: int
    parent_protocol_sha256: str | None
    l0_commitment_sha256: str
    components: tuple[ProtocolComponentV2, ...]

    def __post_init__(self) -> None:
        _schema_v1(self.schema_version)
        protocol_version = _positive_int(self.protocol_version, field="protocol_version")
        if protocol_version == 1:
            if self.parent_protocol_sha256 is not None:
                raise ValueError("seed protocol must not have a parent_protocol_sha256")
        elif self.parent_protocol_sha256 is None:
            raise ValueError("child protocol must have a parent_protocol_sha256")
        else:
            require_sha256(self.parent_protocol_sha256, "parent_protocol_sha256")
        require_sha256(self.l0_commitment_sha256, "l0_commitment_sha256")
        if not isinstance(self.components, (list, tuple)):
            raise ValueError("components must be an ordered component list")
        components = tuple(self.components)
        if len(components) != len(KINDS) or any(
            not isinstance(component, ProtocolComponentV2) for component in components
        ):
            raise ValueError("components must contain exactly one component per kind")
        if tuple(component.kind for component in components) != KINDS:
            raise ValueError("components must use the exact canonical kind order")
        object.__setattr__(self, "components", components)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "InfrastructureProtocolV2":
        values = _exact_payload(payload, _PROTOCOL_FIELDS, name="infrastructure protocol")
        components_payload = values["components"]
        if not isinstance(components_payload, (list, tuple)):
            raise ValueError("components must be an ordered component list")
        parent = values["parent_protocol_sha256"]
        if parent is not None:
            require_sha256(parent, "parent_protocol_sha256")
        return cls(
            _schema_v1(values["schema_version"]),
            _positive_int(values["protocol_version"], field="protocol_version"),
            parent,
            require_sha256(values["l0_commitment_sha256"], "l0_commitment_sha256"),
            tuple(ProtocolComponentV2.from_payload(item) for item in components_payload),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "parent_protocol_sha256": self.parent_protocol_sha256,
            "l0_commitment_sha256": self.l0_commitment_sha256,
            "components": [component.to_payload() for component in self.components],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


@dataclass(frozen=True, slots=True)
class ProtocolProposalV2:
    """A one-component L1 replacement proposed against one exact parent."""

    schema_version: int
    parent_protocol_sha256: str
    kind: str
    replacement: ProtocolComponentV2
    compatibility_corpus_sha256: str

    def __post_init__(self) -> None:
        _schema_v1(self.schema_version)
        require_sha256(self.parent_protocol_sha256, "parent_protocol_sha256")
        _kind(self.kind)
        if not isinstance(self.replacement, ProtocolComponentV2):
            raise ValueError("replacement must be a protocol component")
        if self.replacement.kind != self.kind:
            raise ValueError("replacement.kind must match proposal kind")
        require_sha256(self.compatibility_corpus_sha256, "compatibility_corpus_sha256")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ProtocolProposalV2":
        values = _exact_payload(payload, _PROPOSAL_FIELDS, name="protocol proposal")
        return cls(
            _schema_v1(values["schema_version"]),
            require_sha256(values["parent_protocol_sha256"], "parent_protocol_sha256"),
            _kind(values["kind"]),
            ProtocolComponentV2.from_payload(values["replacement"]),
            require_sha256(
                values["compatibility_corpus_sha256"], "compatibility_corpus_sha256"
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parent_protocol_sha256": self.parent_protocol_sha256,
            "kind": self.kind,
            "replacement": self.replacement.to_payload(),
            "compatibility_corpus_sha256": self.compatibility_corpus_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())

    def to_child(self, parent: InfrastructureProtocolV2) -> InfrastructureProtocolV2:
        """Apply this proposal to exactly one matching component of ``parent``."""
        if not isinstance(parent, InfrastructureProtocolV2):
            raise ValueError("parent must be an infrastructure protocol")
        if self.parent_protocol_sha256 != parent.fingerprint():
            raise ValueError("proposal parent_protocol_sha256 does not match parent")
        if self.replacement.kind != self.kind:
            raise ValueError("replacement.kind must match proposal kind")
        old = next(item for item in parent.components if item.kind == self.kind)
        if self.replacement == old:
            raise ValueError("replacement must change the selected component")
        components = tuple(
            self.replacement if item.kind == self.kind else item
            for item in parent.components
        )
        return InfrastructureProtocolV2(
            1,
            parent.protocol_version + 1,
            parent.fingerprint(),
            parent.l0_commitment_sha256,
            components,
        )


@dataclass(frozen=True, slots=True)
class ProtocolReleaseV2:
    """Host-authorized publication binding a protocol, evidence, and runtime."""

    schema_version: int
    protocol_sha256: str
    l0_commitment_sha256: str
    evidence_sha256: str
    frozen_bundle_sha256: str
    runtime_fingerprint: str

    def __post_init__(self) -> None:
        _schema_v1(self.schema_version)
        for field in _RELEASE_FIELDS[1:]:
            require_sha256(getattr(self, field), field)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ProtocolReleaseV2":
        values = _exact_payload(payload, _RELEASE_FIELDS, name="protocol release")
        return cls(
            _schema_v1(values["schema_version"]),
            require_sha256(values["protocol_sha256"], "protocol_sha256"),
            require_sha256(values["l0_commitment_sha256"], "l0_commitment_sha256"),
            require_sha256(values["evidence_sha256"], "evidence_sha256"),
            require_sha256(values["frozen_bundle_sha256"], "frozen_bundle_sha256"),
            require_sha256(values["runtime_fingerprint"], "runtime_fingerprint"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "l0_commitment_sha256": self.l0_commitment_sha256,
            "evidence_sha256": self.evidence_sha256,
            "frozen_bundle_sha256": self.frozen_bundle_sha256,
            "runtime_fingerprint": self.runtime_fingerprint,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())
