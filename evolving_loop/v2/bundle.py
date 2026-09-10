"""Immutable identity and mutation-ownership contract for Evolution V2."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Protocol

from .contracts import canonical_v2_bytes, require_sha256


MutationTarget = Literal["numerical", "retrieval", "decision", "joint"]

_BUNDLE_FIELDS = (
    "schema_version",
    "generation",
    "parent_bundle_sha256",
    "numerical_release_sha256",
    "numerical_registry_sha256",
    "retrieval_release_sha256",
    "decision_policy_sha256",
    "harness_policy_sha256",
    "archive_snapshot_sha256",
    "scheduler_state_sha256",
    "protocol_fingerprint",
    "runtime_fingerprints",
    "acceptance_evidence_sha256",
)
_PRINCIPAL_FIELDS = {
    "numerical": ("numerical_release_sha256", "numerical_registry_sha256"),
    "retrieval": ("retrieval_release_sha256",),
    "decision": ("decision_policy_sha256",),
}
_NON_PRINCIPAL_AUTHORITY_FIELDS = (
    "harness_policy_sha256",
    "archive_snapshot_sha256",
    "scheduler_state_sha256",
    "protocol_fingerprint",
    "runtime_fingerprints",
)


class BundleContractError(ValueError):
    """Raised when a Bundle violates lineage or mutation ownership."""


def _bundle_sha256(value: object, field: str) -> str:
    try:
        return require_sha256(value, field)
    except ValueError as error:
        raise BundleContractError(str(error)) from error


class _CanonicalEvidenceBinding(Protocol):
    """Kernel-side evidence interface needed to seal one accepted Child."""

    candidate_bundle_sha256: str

    def to_payload(self) -> Mapping[str, object]: ...

    def canonical_bytes(self) -> bytes: ...


@dataclass(frozen=True, slots=True)
class EvolutionBundleV2:
    """Complete content-addressed state needed to replay one evolution state."""

    schema_version: int
    generation: int
    parent_bundle_sha256: str | None
    numerical_release_sha256: str
    numerical_registry_sha256: str
    retrieval_release_sha256: str
    decision_policy_sha256: str
    harness_policy_sha256: str
    archive_snapshot_sha256: str
    scheduler_state_sha256: str
    protocol_fingerprint: str
    runtime_fingerprints: Mapping[str, str]
    acceptance_evidence_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise BundleContractError("schema_version must be integer 2")
        if type(self.generation) is not int or self.generation < 0:
            raise BundleContractError("generation must be a non-negative integer")

        if self.generation == 0:
            if self.parent_bundle_sha256 is not None:
                raise BundleContractError("seed Bundle must not claim a Parent")
            if self.acceptance_evidence_sha256 is not None:
                raise BundleContractError("seed Bundle must not carry acceptance evidence")
        elif self.parent_bundle_sha256 is None:
            raise BundleContractError(
                "positive-generation Bundle requires a Parent fingerprint"
            )

        if self.parent_bundle_sha256 is not None:
            _bundle_sha256(self.parent_bundle_sha256, "parent_bundle_sha256")

        for field in (
            "numerical_release_sha256",
            "numerical_registry_sha256",
            "retrieval_release_sha256",
            "decision_policy_sha256",
            "harness_policy_sha256",
            "archive_snapshot_sha256",
            "scheduler_state_sha256",
            "protocol_fingerprint",
        ):
            _bundle_sha256(getattr(self, field), field)

        if self.acceptance_evidence_sha256 is not None:
            _bundle_sha256(
                self.acceptance_evidence_sha256, "acceptance_evidence_sha256"
            )

        if not isinstance(self.runtime_fingerprints, Mapping):
            raise BundleContractError("runtime_fingerprints must be a non-empty object")
        runtime = dict(self.runtime_fingerprints)
        if not runtime:
            raise BundleContractError("runtime_fingerprints must be non-empty")
        if any(type(key) is not str or not key for key in runtime):
            raise BundleContractError("runtime_fingerprints keys must be non-empty strings")
        for key, value in runtime.items():
            _bundle_sha256(value, f"runtime_fingerprints.{key}")
        object.__setattr__(
            self,
            "runtime_fingerprints",
            MappingProxyType(dict(sorted(runtime.items()))),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "EvolutionBundleV2":
        if not isinstance(payload, Mapping) or any(
            type(key) is not str for key in payload
        ):
            raise BundleContractError("Bundle payload must use the exact schema")
        actual = set(payload)
        expected = set(_BUNDLE_FIELDS)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            suffix = f" ({', '.join(details)})" if details else ""
            raise BundleContractError(
                f"Bundle payload must use the exact schema{suffix}"
            )
        return cls(**{field: payload[field] for field in _BUNDLE_FIELDS})  # type: ignore[arg-type]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "parent_bundle_sha256": self.parent_bundle_sha256,
            "numerical_release_sha256": self.numerical_release_sha256,
            "numerical_registry_sha256": self.numerical_registry_sha256,
            "retrieval_release_sha256": self.retrieval_release_sha256,
            "decision_policy_sha256": self.decision_policy_sha256,
            "harness_policy_sha256": self.harness_policy_sha256,
            "archive_snapshot_sha256": self.archive_snapshot_sha256,
            "scheduler_state_sha256": self.scheduler_state_sha256,
            "protocol_fingerprint": self.protocol_fingerprint,
            "runtime_fingerprints": dict(self.runtime_fingerprints),
            "acceptance_evidence_sha256": self.acceptance_evidence_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def provisional_child(
        self, target: MutationTarget, changes: Mapping[str, object]
    ) -> "EvolutionBundleV2":
        """Create a direct provisional Child from candidate-owned identities."""
        if not isinstance(changes, Mapping):
            raise BundleContractError("Child changes must be a principal mapping")
        unknown = sorted(
            str(key) for key in changes if key not in _PRINCIPAL_FIELDS
        )
        if unknown:
            raise BundleContractError(
                f"Child changes may contain only principal scopes; invalid={unknown}"
            )
        updates: dict[str, object] = {
            "generation": self.generation + 1,
            "parent_bundle_sha256": self.fingerprint(),
            "acceptance_evidence_sha256": None,
        }
        for scope, value in changes.items():
            if scope == "numerical":
                if type(value) is not tuple or len(value) != 2:
                    raise BundleContractError(
                        "changes.numerical must be an atomic pair of release and "
                        "registry SHA-256 identities"
                    )
                release_sha256 = _bundle_sha256(
                    value[0], "changes.numerical.release_sha256"
                )
                registry_sha256 = _bundle_sha256(
                    value[1], "changes.numerical.registry_sha256"
                )
                updates["numerical_release_sha256"] = release_sha256
                updates["numerical_registry_sha256"] = registry_sha256
            else:
                updates[_PRINCIPAL_FIELDS[scope][0]] = _bundle_sha256(
                    value, f"changes.{scope}"
                )
        child = replace(self, **updates)
        validate_child_scope(self, child, target)
        return child


def principal_fingerprints(bundle: EvolutionBundleV2) -> Mapping[str, str]:
    """Return disjoint principal identities without Host-owned metadata."""
    if not isinstance(bundle, EvolutionBundleV2):
        raise BundleContractError("principal fingerprints require an EvolutionBundleV2")
    fingerprints = {}
    for scope, fields in _PRINCIPAL_FIELDS.items():
        payload = {field: getattr(bundle, field) for field in fields}
        fingerprints[scope] = hashlib.sha256(canonical_v2_bytes(payload)).hexdigest()
    return MappingProxyType(fingerprints)


def changed_scopes(
    parent: EvolutionBundleV2, child: EvolutionBundleV2
) -> tuple[str, ...]:
    """Recompute principal scope changes from Bundle contents."""
    parent_fingerprints = principal_fingerprints(parent)
    child_fingerprints = principal_fingerprints(child)
    return tuple(
        scope
        for scope in _PRINCIPAL_FIELDS
        if parent_fingerprints[scope] != child_fingerprints[scope]
    )


def validate_child_scope(
    parent: EvolutionBundleV2,
    child: EvolutionBundleV2,
    target: MutationTarget,
) -> tuple[str, ...]:
    """Validate direct lineage, Host ownership, and the actual principal diff."""
    if not isinstance(parent, EvolutionBundleV2) or not isinstance(
        child, EvolutionBundleV2
    ):
        raise BundleContractError("scope validation requires Parent and Child Bundles")
    if target not in {*_PRINCIPAL_FIELDS, "joint"}:
        raise BundleContractError("Child mutation target is invalid")
    if child.generation != parent.generation + 1:
        raise BundleContractError("Child generation must increment Parent generation")
    if child.parent_bundle_sha256 != parent.fingerprint():
        raise BundleContractError("Child must bind the exact Parent fingerprint")
    if child.acceptance_evidence_sha256 is not None:
        raise BundleContractError("provisional Child must clear acceptance evidence")
    for field in _NON_PRINCIPAL_AUTHORITY_FIELDS:
        if getattr(child, field) != getattr(parent, field):
            raise BundleContractError(f"provisional Child changed Host-owned {field}")

    release_changed = (
        child.numerical_release_sha256 != parent.numerical_release_sha256
    )
    registry_changed = (
        child.numerical_registry_sha256 != parent.numerical_registry_sha256
    )
    if release_changed != registry_changed:
        raise BundleContractError(
            "Numerical scope must change release and registry identities together"
        )

    scopes = changed_scopes(parent, child)
    if target == "joint":
        if len(scopes) < 2:
            raise BundleContractError("joint Child must change at least two scopes")
    else:
        if len(scopes) != 1:
            raise BundleContractError(
                "single-target Child must change exactly one principal scope"
            )
        if scopes[0] != target:
            raise BundleContractError(
                "Child changed scope does not match the declared target"
            )
    return scopes


def _seal_acceptance(
    parent: EvolutionBundleV2,
    provisional_child: EvolutionBundleV2,
    target: MutationTarget,
    evidence: _CanonicalEvidenceBinding,
    archive_snapshot_sha256: str,
    scheduler_state_sha256: str,
) -> EvolutionBundleV2:
    """Kernel-only acceptance seal over a freshly revalidated provisional Child."""
    if provisional_child.acceptance_evidence_sha256 is not None:
        raise BundleContractError("acceptance evidence may be sealed only once")

    validate_child_scope(parent, provisional_child, target)
    child_sha256 = provisional_child.fingerprint()
    try:
        bound_child_sha256 = evidence.candidate_bundle_sha256
        evidence_payload = evidence.to_payload()
        evidence_bytes = evidence.canonical_bytes()
    except (AttributeError, TypeError) as error:
        raise BundleContractError(
            "acceptance evidence must be a canonical evidence-binding object"
        ) from error

    _bundle_sha256(bound_child_sha256, "evidence.candidate_bundle_sha256")
    if bound_child_sha256 != child_sha256:
        raise BundleContractError(
            "acceptance evidence must bind the exact provisional Child fingerprint"
        )
    if not isinstance(evidence_payload, Mapping):
        raise BundleContractError("acceptance evidence payload must be an object")
    if evidence_payload.get("candidate_bundle_sha256") != bound_child_sha256:
        raise BundleContractError(
            "acceptance evidence payload must bind the exact provisional Child fingerprint"
        )
    canonical_evidence_bytes = canonical_v2_bytes(evidence_payload)
    if type(evidence_bytes) is not bytes or evidence_bytes != canonical_evidence_bytes:
        raise BundleContractError("acceptance evidence bytes must be canonical")

    evidence_sha256 = hashlib.sha256(canonical_evidence_bytes).hexdigest()
    archive = _bundle_sha256(archive_snapshot_sha256, "archive_snapshot_sha256")
    scheduler = _bundle_sha256(
        scheduler_state_sha256, "scheduler_state_sha256"
    )
    return replace(
        provisional_child,
        acceptance_evidence_sha256=evidence_sha256,
        archive_snapshot_sha256=archive,
        scheduler_state_sha256=scheduler,
    )


__all__ = [
    "BundleContractError",
    "EvolutionBundleV2",
    "MutationTarget",
    "changed_scopes",
    "principal_fingerprints",
    "validate_child_scope",
]
