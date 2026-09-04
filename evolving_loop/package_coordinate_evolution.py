"""Single-coordinate orchestration for a frozen Numerical/Retrieval/Decision triad."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.package_numerical_supply import (
    NumericalSupplyError,
    NumericalSupplyRelease,
    parse_numerical_supply_release,
)
from evolving_loop.package_registry import FrozenNumericalPackageRegistry


PackageCoordinateTarget = Literal["numerical", "retrieval", "decision"]
PackageCoordinateName = Literal["seed", "numerical", "retrieval", "decision"]
_MODULES = ("numerical", "retrieval", "decision")
_RUNTIME_KEYS = frozenset(
    {
        "bridge_runtime",
        "numerical_runtime",
        "retrieval_runtime",
        "decision_runtime",
        "retrieval_verifier",
        "metric_policy",
        "model_runtime",
        "llm_runtime",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: object, message: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(message)
    return value


@dataclass(frozen=True)
class PackageCoordinateBundle:
    """Canonical schema-2 authority for one package-coordinate state."""

    generation: int
    coordinate: PackageCoordinateName
    parent_sha256: str | None
    numerical_release_payload: Mapping[str, object]
    numerical_release_sha256: str
    numerical_manifest_sha256: str
    policy: HarnessPolicy
    runtime_fingerprints: Mapping[str, str]
    acceptance_evidence_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("package bundle generation must be non-negative")
        if self.coordinate not in {"seed", *_MODULES}:
            raise ValueError("package bundle coordinate is invalid")
        if (self.generation == 0) != (self.coordinate == "seed"):
            raise ValueError("package bundle generation zero requires coordinate='seed'")
        if self.generation == 0:
            if self.parent_sha256 is not None:
                raise ValueError("package seed must not claim a Parent SHA")
        elif self.parent_sha256 is None:
            raise ValueError("package bundle Child requires a direct Parent SHA")
        else:
            _require_sha256(
                self.parent_sha256, "package bundle parent fingerprint must be canonical"
            )
        if not isinstance(self.numerical_release_payload, Mapping):
            raise ValueError("package bundle Numerical release payload must be a mapping")
        try:
            release = parse_numerical_supply_release(
                _plain(self.numerical_release_payload)
            )
        except NumericalSupplyError as error:
            raise ValueError("package bundle Numerical release payload is invalid") from error
        if release.fingerprint != _require_sha256(
            self.numerical_release_sha256,
            "package bundle Numerical release must be canonical",
        ):
            raise ValueError("package bundle Numerical release digest mismatch")
        _require_sha256(
            self.numerical_manifest_sha256,
            "package bundle Numerical manifest must be canonical",
        )
        if not isinstance(self.policy, HarnessPolicy):
            raise ValueError("package bundle requires a HarnessPolicy")
        runtime = dict(self.runtime_fingerprints)
        if set(runtime) != _RUNTIME_KEYS or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in runtime.values()
        ):
            raise ValueError("package bundle runtime fingerprints are incomplete")
        if self.acceptance_evidence_sha256 is not None:
            _require_sha256(
                self.acceptance_evidence_sha256,
                "package bundle acceptance evidence must be canonical",
            )
            if self.coordinate == "seed":
                raise ValueError("package seed cannot carry acceptance evidence")
        object.__setattr__(
            self,
            "numerical_release_payload",
            _freeze(release.to_payload()),
        )
        object.__setattr__(
            self, "runtime_fingerprints", MappingProxyType(dict(sorted(runtime.items())))
        )

    @property
    def public_test_accessed(self) -> bool:
        return False

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "generation": self.generation,
            "coordinate": self.coordinate,
            "parent_sha256": self.parent_sha256,
            "numerical_release_payload": _plain(self.numerical_release_payload),
            "numerical_release_sha256": self.numerical_release_sha256,
            "numerical_manifest_sha256": self.numerical_manifest_sha256,
            "policy": self.policy.to_payload(),
            "runtime_fingerprints": dict(self.runtime_fingerprints),
            "acceptance_evidence_sha256": self.acceptance_evidence_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_payload())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def with_numerical(
        self,
        release: NumericalSupplyRelease,
        registry: FrozenNumericalPackageRegistry,
    ) -> "PackageCoordinateBundle":
        if not isinstance(release, NumericalSupplyRelease):
            raise ValueError("package coordinate Numerical Child requires a release")
        if not isinstance(registry, FrozenNumericalPackageRegistry):
            raise ValueError("package coordinate Numerical Child requires a registry")
        if registry.release_sha256 != release.fingerprint:
            raise ValueError("package coordinate Numerical Child release mismatch")
        return PackageCoordinateBundle(
            generation=self.generation + 1,
            coordinate="numerical",
            parent_sha256=self.fingerprint(),
            numerical_release_payload=release.to_payload(),
            numerical_release_sha256=release.fingerprint,
            numerical_manifest_sha256=registry.fingerprint,
            policy=self.policy,
            runtime_fingerprints=self.runtime_fingerprints,
            acceptance_evidence_sha256=None,
        )

    def with_policy(
        self, policy: HarnessPolicy, target: Literal["retrieval", "decision"]
    ) -> "PackageCoordinateBundle":
        if not isinstance(policy, HarnessPolicy):
            raise ValueError("package coordinate Child requires a HarnessPolicy")
        if target not in {"retrieval", "decision"}:
            raise ValueError("package coordinate policy Child target is invalid")
        child = self._provisional_policy_child(policy, target)
        if _changed_modules(self, child) != (target,):
            raise ValueError("package coordinate Child crossed module ownership")
        return child

    def _provisional_policy_child(
        self, policy: HarnessPolicy, target: Literal["retrieval", "decision"]
    ) -> "PackageCoordinateBundle":
        """Build a traceable policy candidate before ownership acceptance."""
        return PackageCoordinateBundle(
            generation=self.generation + 1,
            coordinate=target,
            parent_sha256=self.fingerprint(),
            numerical_release_payload=self.numerical_release_payload,
            numerical_release_sha256=self.numerical_release_sha256,
            numerical_manifest_sha256=self.numerical_manifest_sha256,
            policy=policy,
            runtime_fingerprints=self.runtime_fingerprints,
            acceptance_evidence_sha256=None,
        )

    def seal_acceptance(self, evidence_sha256: str) -> "PackageCoordinateBundle":
        if self.coordinate == "seed" or self.parent_sha256 is None:
            raise ValueError("only a provisional direct Child can be sealed")
        if self.acceptance_evidence_sha256 is not None:
            raise ValueError("package acceptance evidence may be sealed only once")
        _require_sha256(
            evidence_sha256, "package acceptance evidence must be canonical"
        )
        return PackageCoordinateBundle(
            generation=self.generation,
            coordinate=self.coordinate,
            parent_sha256=self.parent_sha256,
            numerical_release_payload=self.numerical_release_payload,
            numerical_release_sha256=self.numerical_release_sha256,
            numerical_manifest_sha256=self.numerical_manifest_sha256,
            policy=self.policy,
            runtime_fingerprints=self.runtime_fingerprints,
            acceptance_evidence_sha256=evidence_sha256,
        )


@dataclass(frozen=True)
class PackageCoordinateState:
    bundle: PackageCoordinateBundle
    registry: FrozenNumericalPackageRegistry

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, PackageCoordinateBundle):
            raise ValueError("package state requires a package bundle")
        if not isinstance(self.registry, FrozenNumericalPackageRegistry):
            raise ValueError("package state requires a frozen Numerical registry")
        if self.registry.fingerprint != self.bundle.numerical_manifest_sha256:
            raise ValueError("package state registry manifest mismatch")
        if self.registry.release_sha256 != self.bundle.numerical_release_sha256:
            raise ValueError("package state Numerical release mismatch")

    def with_numerical(
        self, release: NumericalSupplyRelease, registry: FrozenNumericalPackageRegistry
    ) -> "PackageCoordinateState":
        return PackageCoordinateState(self.bundle.with_numerical(release, registry), registry)

    def with_policy(
        self, policy: HarnessPolicy, target: Literal["retrieval", "decision"]
    ) -> "PackageCoordinateState":
        return PackageCoordinateState(
            self.bundle.with_policy(policy, target), self.registry
        )

    def seal_acceptance(self, evidence_sha256: str) -> "PackageCoordinateState":
        return PackageCoordinateState(self.bundle.seal_acceptance(evidence_sha256), self.registry)


def package_principal_fingerprints(bundle: PackageCoordinateBundle) -> Mapping[str, str]:
    """Return disjoint identities without acceptance or coordinate metadata."""
    if not isinstance(bundle, PackageCoordinateBundle):
        raise ValueError("principal fingerprints require a package bundle")
    policy = bundle.policy
    retrieval_payload = {
        "release_payload": policy.to_payload().get("retrieval_release_payload"),
        "release_sha256": policy.retrieval_release_sha256,
        "retrieval_runtime": bundle.runtime_fingerprints["retrieval_runtime"],
        "retrieval_verifier": bundle.runtime_fingerprints["retrieval_verifier"],
    }
    decision_payload = {
        "decision_prompt": policy.decision_prompt,
        "decision_skills": _plain(policy.decision_skills),
        "enable_evidence_adjustments": policy.enable_evidence_adjustments,
        "max_evidence_adjustments": policy.max_evidence_adjustments,
        "decision_aggregation": policy.decision_aggregation,
        "decision_runtime": bundle.runtime_fingerprints["decision_runtime"],
    }
    numerical_payload = {
        "release_payload": _plain(bundle.numerical_release_payload),
        "manifest": bundle.numerical_manifest_sha256,
        "bridge_runtime": bundle.runtime_fingerprints["bridge_runtime"],
        "numerical_runtime": bundle.runtime_fingerprints["numerical_runtime"],
        "model_runtime": bundle.runtime_fingerprints["model_runtime"],
        "metric_policy": bundle.runtime_fingerprints["metric_policy"],
    }
    return MappingProxyType(
        {
            "numerical": _digest(numerical_payload),
            "retrieval": _digest(retrieval_payload),
            "decision": _digest(decision_payload),
        }
    )


def _changed_modules(
    parent: PackageCoordinateBundle, child: PackageCoordinateBundle
) -> tuple[str, ...]:
    parent_fingerprints = package_principal_fingerprints(parent)
    child_fingerprints = package_principal_fingerprints(child)
    return tuple(
        module
        for module in _MODULES
        if parent_fingerprints[module] != child_fingerprints[module]
    )


@dataclass(frozen=True)
class PackageCoordinateStep:
    generation: int
    target: PackageCoordinateTarget
    accepted: bool
    reason: str
    parent_fingerprints: Mapping[str, str]
    child_fingerprints: Mapping[str, str]
    accepted_fingerprints: Mapping[str, str]
    changed_modules: tuple[str, ...]
    parent_bytes_sha256: str
    child_bytes_sha256: str
    accepted_bytes_sha256: str
    parent_registry_sha256: str
    child_registry_sha256: str
    accepted_registry_sha256: str
    public_test_accessed: bool = False

    def __post_init__(self) -> None:
        if self.target not in _MODULES:
            raise ValueError("invalid package coordinate target")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("package coordinate step requires a reason")
        for field_name in (
            "parent_fingerprints",
            "child_fingerprints",
            "accepted_fingerprints",
        ):
            fingerprints = dict(getattr(self, field_name))
            if set(fingerprints) != set(_MODULES):
                raise ValueError("package coordinate trace must bind every module")
            object.__setattr__(self, field_name, MappingProxyType(dict(fingerprints)))
        for value in (
            self.parent_bytes_sha256,
            self.child_bytes_sha256,
            self.accepted_bytes_sha256,
            self.parent_registry_sha256,
            self.child_registry_sha256,
            self.accepted_registry_sha256,
        ):
            _require_sha256(value, "package coordinate trace hash must be canonical")
        if self.accepted and self.public_test_accessed:
            raise ValueError("a Public-accessed package coordinate cannot be accepted")
        if not self.accepted and (
            self.accepted_bytes_sha256 != self.parent_bytes_sha256
            or self.accepted_registry_sha256 != self.parent_registry_sha256
        ):
            raise ValueError("rejected package coordinate must preserve the exact Parent")


def _step(
    *,
    generation: int,
    target: PackageCoordinateTarget,
    parent: PackageCoordinateState,
    child: PackageCoordinateState,
    accepted_state: PackageCoordinateState,
    accepted: bool,
    reason: str,
    public_test_accessed: bool,
) -> PackageCoordinateStep:
    changed = _changed_modules(parent.bundle, child.bundle)
    if accepted:
        if child.bundle.parent_sha256 != parent.bundle.fingerprint():
            raise ValueError("accepted package coordinate Child has detached lineage")
        if changed != (target,):
            raise ValueError("accepted package coordinate Child crossed module ownership")
        if accepted_state.bundle.acceptance_evidence_sha256 is None:
            raise ValueError("accepted package coordinate Child must be sealed")
    return PackageCoordinateStep(
        generation=generation,
        target=target,
        accepted=accepted,
        reason=reason,
        parent_fingerprints=package_principal_fingerprints(parent.bundle),
        child_fingerprints=package_principal_fingerprints(child.bundle),
        accepted_fingerprints=package_principal_fingerprints(accepted_state.bundle),
        changed_modules=changed,
        parent_bytes_sha256=parent.bundle.fingerprint(),
        child_bytes_sha256=child.bundle.fingerprint(),
        accepted_bytes_sha256=accepted_state.bundle.fingerprint(),
        parent_registry_sha256=parent.registry.fingerprint,
        child_registry_sha256=child.registry.fingerprint,
        accepted_registry_sha256=accepted_state.registry.fingerprint,
        public_test_accessed=public_test_accessed,
    )


_PHASE_TARGETS: tuple[PackageCoordinateTarget, ...] = ("numerical", "retrieval", "decision")


class PackageCoordinatePhase(Protocol):
    """A per-coordinate successive-halving phase runner."""

    target: PackageCoordinateTarget

    def run(
        self,
        parent: "PackageCoordinateState",
        initial: "PackageCoordinateState",
        *,
        generation: int,
    ) -> object: ...


def _skip_step(
    generation: int, state: "PackageCoordinateState", reason: str
) -> PackageCoordinateStep:
    return _step(
        generation=generation,
        target="decision",
        parent=state,
        child=state,
        accepted_state=state,
        accepted=False,
        reason=reason,
        public_test_accessed=False,
    )


class PackageCoordinateController:
    """Run two ordered Numerical -> Retrieval -> Decision cycles with early stopping."""

    def __init__(
        self,
        numerical_phase: PackageCoordinatePhase | None,
        retrieval_phase: PackageCoordinatePhase | None,
        decision_phase: PackageCoordinatePhase | None,
        *,
        cycles: int = 2,
    ) -> None:
        if type(cycles) is not int or cycles < 1:
            raise ValueError("package coordinate controller requires at least one cycle")
        self._phases: tuple[
            tuple[PackageCoordinateTarget, PackageCoordinatePhase | None], ...
        ] = (
            ("numerical", numerical_phase),
            ("retrieval", retrieval_phase),
            ("decision", decision_phase),
        )
        self.cycles = cycles

    def run(
        self,
        parent: "PackageCoordinateState",
        initial: "PackageCoordinateState",
    ) -> tuple["PackageCoordinateState", tuple[PackageCoordinateStep, ...]]:
        if not isinstance(parent, PackageCoordinateState) or not isinstance(
            initial, PackageCoordinateState
        ):
            raise ValueError("package coordinate controller requires package states")
        current = parent
        trace: list[PackageCoordinateStep] = []
        generation = 0
        for _cycle in range(self.cycles):
            cycle_accepted = False
            for target, phase in self._phases:
                if phase is None:
                    generation += 1
                    continue
                if (
                    target == "decision"
                    and not current.bundle.policy.has_accepted_retrieval_release
                ):
                    trace.append(
                        _skip_step(
                            generation,
                            current,
                            "Decision phase requires a non-v000 accepted Retrieval release",
                        )
                    )
                    generation += 1
                    continue
                outcome = phase.run(current, initial, generation=generation)
                step, current = self._apply(generation, target, current, outcome)
                trace.append(step)
                cycle_accepted = cycle_accepted or step.accepted
                generation += 1
            if not cycle_accepted:
                break
        return current, tuple(trace)

    @staticmethod
    def _apply(
        generation: int,
        target: PackageCoordinateTarget,
        current: "PackageCoordinateState",
        outcome: object,
    ) -> tuple[PackageCoordinateStep, "PackageCoordinateState"]:
        reason = getattr(outcome, "reason", "")
        if not isinstance(reason, str) or not reason:
            reason = "package coordinate phase produced no reason"
        public = bool(getattr(outcome, "public_test_accessed", False))
        selected = getattr(outcome, "selected", None)
        accepted = bool(getattr(outcome, "accepted", False))

        def rejected(step_reason: str) -> tuple[PackageCoordinateStep, "PackageCoordinateState"]:
            return (
                _step(
                    generation=generation,
                    target=target,
                    parent=current,
                    child=current,
                    accepted_state=current,
                    accepted=False,
                    reason=step_reason,
                    public_test_accessed=public,
                ),
                current,
            )

        if getattr(outcome, "target", None) != target:
            return rejected("package phase returned the wrong coordinate target")
        if public:
            return rejected("Public Regression access is forbidden")
        if not accepted:
            return rejected(reason)
        if (
            not getattr(outcome, "improved", False)
            or not isinstance(selected, PackageCoordinateState)
            or selected.bundle.acceptance_evidence_sha256 is None
            or selected.bundle.parent_sha256 != current.bundle.fingerprint()
            or _changed_modules(current.bundle, selected.bundle) != (target,)
        ):
            return rejected("package coordinate Child failed a lineage or ownership gate")
        if target == "numerical" and (
            selected.bundle.numerical_release_sha256
            == current.bundle.numerical_release_sha256
            or selected.bundle.numerical_manifest_sha256
            == current.bundle.numerical_manifest_sha256
        ):
            return rejected("accepted Numerical Child did not replace release and registry")
        if target in ("retrieval", "decision") and (
            selected.registry.fingerprint != current.registry.fingerprint
        ):
            return rejected("accepted contextual Child changed the Numerical registry")
        step = _step(
            generation=generation,
            target=target,
            parent=current,
            child=selected,
            accepted_state=selected,
            accepted=True,
            reason=reason,
            public_test_accessed=False,
        )
        return step, selected
