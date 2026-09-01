"""Single-coordinate orchestration for a frozen Numerical/Retrieval/Decision triad."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.coordinate_evolution import (
    CoordinatePhaseOutcome,
    CoordinatePhaseRunner,
)
from evolving_loop.data import ContextTask


PackageCoordinateTarget = Literal["retrieval", "decision"]
_MODULES = ("numerical", "retrieval", "decision")
_RUNTIME_KEYS = frozenset(
    {
        "bridge_runtime",
        "retrieval_runtime",
        "decision_runtime",
        "retrieval_verifier",
        "metric_policy",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class PackageCoordinateBundle:
    """Accepted triad authority with canonical transitive provenance."""

    generation: int
    parent_sha256: str | None
    numerical_manifest_sha256: str
    policy: HarnessPolicy
    runtime_fingerprints: Mapping[str, str]

    def __post_init__(self) -> None:
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("package bundle generation must be non-negative")
        if self.parent_sha256 is not None and _SHA256.fullmatch(
            self.parent_sha256
        ) is None:
            raise ValueError("package bundle parent fingerprint must be canonical")
        if _SHA256.fullmatch(self.numerical_manifest_sha256) is None:
            raise ValueError("package bundle Numerical manifest must be canonical")
        if not isinstance(self.policy, HarnessPolicy):
            raise ValueError("package bundle requires a HarnessPolicy")
        runtime = dict(self.runtime_fingerprints)
        if set(runtime) != _RUNTIME_KEYS or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in runtime.values()
        ):
            raise ValueError("package bundle runtime fingerprints are incomplete")
        object.__setattr__(
            self,
            "runtime_fingerprints",
            MappingProxyType(dict(sorted(runtime.items()))),
        )

    @property
    def public_test_accessed(self) -> bool:
        return False

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "generation": self.generation,
            "parent_sha256": self.parent_sha256,
            "numerical_manifest_sha256": self.numerical_manifest_sha256,
            "policy": self.policy.to_payload(),
            "runtime_fingerprints": dict(self.runtime_fingerprints),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_payload())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def with_policy(self, policy: HarnessPolicy) -> "PackageCoordinateBundle":
        if not isinstance(policy, HarnessPolicy):
            raise ValueError("package coordinate Child requires a HarnessPolicy")
        return PackageCoordinateBundle(
            generation=self.generation + 1,
            parent_sha256=self.fingerprint(),
            numerical_manifest_sha256=self.numerical_manifest_sha256,
            policy=policy,
            runtime_fingerprints=self.runtime_fingerprints,
        )


def package_principal_fingerprints(
    bundle: PackageCoordinateBundle,
) -> Mapping[str, str]:
    """Return disjoint transitive identities for the three accepted modules."""
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
        "manifest": bundle.numerical_manifest_sha256,
        "bridge_runtime": bundle.runtime_fingerprints["bridge_runtime"],
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
    parent: PackageCoordinateBundle,
    child: PackageCoordinateBundle,
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
    public_test_accessed: bool = False

    def __post_init__(self) -> None:
        if self.target not in {"retrieval", "decision"}:
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
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(fingerprints)),
            )
        for value in (
            self.parent_bytes_sha256,
            self.child_bytes_sha256,
            self.accepted_bytes_sha256,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("package coordinate bytes hash must be canonical")
        if self.accepted and self.public_test_accessed:
            raise ValueError("a Public-accessed package coordinate cannot be accepted")


def _step(
    *,
    generation: int,
    target: PackageCoordinateTarget,
    parent: PackageCoordinateBundle,
    child: PackageCoordinateBundle,
    accepted_bundle: PackageCoordinateBundle,
    accepted: bool,
    reason: str,
    public_test_accessed: bool,
) -> PackageCoordinateStep:
    return PackageCoordinateStep(
        generation=generation,
        target=target,
        accepted=accepted,
        reason=reason,
        parent_fingerprints=package_principal_fingerprints(parent),
        child_fingerprints=package_principal_fingerprints(child),
        accepted_fingerprints=package_principal_fingerprints(accepted_bundle),
        changed_modules=_changed_modules(parent, child),
        parent_bytes_sha256=parent.fingerprint(),
        child_bytes_sha256=child.fingerprint(),
        accepted_bytes_sha256=accepted_bundle.fingerprint(),
        public_test_accessed=public_test_accessed,
    )


class PackageCoordinateController:
    """Accept already gated Retrieval and Decision phases one at a time."""

    def __init__(
        self,
        retrieval_phase: CoordinatePhaseRunner | None,
        decision_phase: CoordinatePhaseRunner | None,
    ) -> None:
        self.retrieval_phase = retrieval_phase
        self.decision_phase = decision_phase

    def run(
        self,
        parent: PackageCoordinateBundle,
        train_tasks: Sequence[ContextTask],
        dev_tasks: Sequence[ContextTask],
    ) -> tuple[PackageCoordinateBundle, tuple[PackageCoordinateStep, ...]]:
        if not isinstance(parent, PackageCoordinateBundle):
            raise ValueError("package coordinate controller requires a bundle")
        current = parent
        trace: list[PackageCoordinateStep] = []
        phases: tuple[
            tuple[PackageCoordinateTarget, CoordinatePhaseRunner | None], ...
        ] = (
            ("retrieval", self.retrieval_phase),
            ("decision", self.decision_phase),
        )
        for generation, (target, runner) in enumerate(phases):
            if runner is None:
                continue
            if target == "decision" and not current.policy.has_accepted_retrieval_release:
                trace.append(
                    _step(
                        generation=generation,
                        target=target,
                        parent=current,
                        child=current,
                        accepted_bundle=current,
                        accepted=False,
                        reason=(
                            "Decision phase requires a non-v000 accepted Retrieval release"
                        ),
                        public_test_accessed=False,
                    )
                )
                continue
            outcome = runner.run(current.policy, train_tasks, dev_tasks)
            if not isinstance(outcome, CoordinatePhaseOutcome):
                raise ValueError("package coordinate runner returned an invalid outcome")
            if outcome.target != target:
                reason = "package phase returned the wrong coordinate target"
                child = current
                accepted = False
            else:
                child = (
                    current.with_policy(outcome.bundle)
                    if outcome.accepted
                    else current
                )
                changed = _changed_modules(current, child)
                lineage_valid = (
                    outcome.bundle.version != current.policy.version
                    and outcome.bundle.parent == current.policy.version
                )
                accepted = bool(
                    outcome.accepted
                    and outcome.improved
                    and not outcome.public_test_accessed
                    and lineage_valid
                    and changed == (target,)
                )
                if outcome.public_test_accessed:
                    reason = "Public Regression access is forbidden"
                elif not lineage_valid and outcome.accepted:
                    reason = "package coordinate Child has detached lineage"
                elif outcome.accepted and changed != (target,):
                    reason = "package coordinate Child crossed module ownership"
                elif outcome.accepted and not outcome.improved:
                    reason = "package coordinate Child reported no improvement"
                else:
                    reason = outcome.reason
            accepted_bundle = child if accepted else current
            trace.append(
                _step(
                    generation=generation,
                    target=target,
                    parent=current,
                    child=child,
                    accepted_bundle=accepted_bundle,
                    accepted=accepted,
                    reason=reason,
                    public_test_accessed=outcome.public_test_accessed,
                )
            )
            current = accepted_bundle
        return current, tuple(trace)
