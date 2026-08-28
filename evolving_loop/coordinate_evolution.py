"""Coordinate-isolated Retrieval and Decision evolution.

The controller is deliberately policy-only: phase engines own Train/Dev
evaluation, trusted release publication, and acceptance.  This module accepts
only their already gated outcomes, verifies exact coordinate ownership, and
never loads Public Regression tasks or grants release authority from JSON.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Protocol, Sequence

from evolving_loop.co_evolution import (
    CoEvolutionEngine,
    HarnessPolicy,
    embed_retrieval_release,
)
from evolving_loop.data import ContextTask
from evolving_loop.retrieval_agent.evolution import (
    RetrievalEvolutionEngine,
    RetrievalEvolutionResult,
)
from evolving_loop.retrieval_agent.policy import (
    _load_retrieval_release_for_operator,
)


CoordinateTarget = Literal["retrieval", "decision"]
CoordinatePhase = Literal["retrieval", "decision", "alternate"]
_MODULES = ("numerical_morphology", "retrieval", "decision")


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


@dataclass(frozen=True)
class CoordinateDiagnostics:
    """Comparable, separately attributed weakness signals.

    Retrieval gain is higher-is-better; Decision regret is lower-is-better.
    Consequently ``-retrieval_gain`` and ``decision_regret`` are the two
    higher-is-weaker values used by the alternating selector.
    """

    retrieval_gain: float
    decision_regret: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "retrieval_gain", _finite(self.retrieval_gain, "retrieval_gain")
        )
        object.__setattr__(
            self, "decision_regret", _finite(self.decision_regret, "decision_regret")
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "CoordinateDiagnostics":
        if "retrieval_gain" in raw:
            retrieval_gain = raw["retrieval_gain"]
        else:
            gains = tuple(
                raw[key]
                for key in (
                    "mean_retrieval_contextual_oracle_smae_gain",
                    "mean_retrieval_contextual_oracle_srmse_gain",
                )
                if key in raw
            )
            if not gains:
                raise ValueError("coordinate diagnostics omit Retrieval gain")
            retrieval_gain = sum(
                _finite(value, "Retrieval gain") for value in gains
            ) / len(gains)
        if "decision_regret" in raw:
            decision_regret = raw["decision_regret"]
        else:
            regrets = tuple(
                raw[key]
                for key in (
                    "mean_selection_smae_regret",
                    "mean_selection_srmse_regret",
                )
                if key in raw
            )
            if not regrets:
                raise ValueError("coordinate diagnostics omit Decision regret")
            decision_regret = sum(
                _finite(value, "Decision regret") for value in regrets
            ) / len(regrets)
        return cls(
            retrieval_gain=_finite(retrieval_gain, "retrieval_gain"),
            decision_regret=_finite(decision_regret, "decision_regret"),
        )

    @property
    def weakest(self) -> CoordinateTarget:
        return (
            "retrieval"
            if -self.retrieval_gain >= self.decision_regret
            else "decision"
        )


@dataclass(frozen=True)
class CoordinateEvolutionConfig:
    phase: CoordinatePhase = "retrieval"
    generations: int = 1

    def __post_init__(self) -> None:
        if self.phase not in {"retrieval", "decision", "alternate"}:
            raise ValueError("invalid coordinate phase")
        if (
            isinstance(self.generations, bool)
            or not isinstance(self.generations, int)
            or self.generations < 1
        ):
            raise ValueError("coordinate generations must be a positive integer")


@dataclass(frozen=True)
class CoordinatePhaseOutcome:
    target: CoordinateTarget
    bundle: HarnessPolicy
    accepted: bool
    improved: bool
    reason: str
    public_test_accessed: bool = False

    def __post_init__(self) -> None:
        if self.target not in {"retrieval", "decision"}:
            raise ValueError("invalid coordinate target")
        if not isinstance(self.bundle, HarnessPolicy):
            raise ValueError("coordinate phase must return a HarnessPolicy")
        if type(self.accepted) is not bool or type(self.improved) is not bool:
            raise ValueError("coordinate phase gates must be booleans")
        if type(self.public_test_accessed) is not bool:
            raise ValueError("Public access flag must be a boolean")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("coordinate phase must explain its result")


class CoordinatePhaseRunner(Protocol):
    def run(
        self,
        parent: HarnessPolicy,
        train_tasks: Sequence[ContextTask],
        dev_tasks: Sequence[ContextTask],
    ) -> CoordinatePhaseOutcome: ...


@dataclass(frozen=True)
class CoordinateEvolutionStep:
    generation: int
    target: CoordinateTarget
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
        if self.public_test_accessed:
            raise ValueError("Public Regression access is forbidden")
        for field_name in (
            "parent_fingerprints",
            "child_fingerprints",
            "accepted_fingerprints",
        ):
            value = getattr(self, field_name)
            if set(value) != set(_MODULES):
                raise ValueError("coordinate trace must bind every principal module")
            object.__setattr__(self, field_name, MappingProxyType(dict(value)))


def _fingerprint(value: object) -> str:
    return HarnessPolicy.retrieval_payload_fingerprint({"value": value})


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _retrieval_coordinate_payload(genome) -> dict[str, object]:
    payload = genome.to_payload()
    payload.pop("version")
    payload.pop("parent")
    return payload


def _next_bundle_version(parent: HarnessPolicy, retrieval_version: str) -> str:
    if not (
        parent.version.startswith("v")
        and parent.version[1:].isdigit()
        and retrieval_version.startswith("v")
        and retrieval_version[1:].isdigit()
    ):
        raise ValueError("coordinate bundle versions must use vNNN identities")
    number = max(int(parent.version[1:]) + 1, int(retrieval_version[1:]))
    return f"v{number:03d}"


def principal_module_fingerprints(policy: HarnessPolicy) -> Mapping[str, str]:
    """Return disjoint fingerprints for every principal accepted coordinate."""
    payload = policy.to_payload()
    numerical_morphology = {
        key: payload[key]
        for key in (
            "coding_generation_prompt",
            "coding_revision_prompt",
            "coding_initial_programs",
            "coding_mutations",
            "coding_mutation_children",
            "coding_validation_folds",
            "coding_validation_horizon",
            "coding_skills",
            "workflow",
        )
    }
    retrieval = {
        key: payload.get(key)
        for key in (
            "retrieval_prompt",
            "retrieval_skills",
            "retrieval_release_payload",
            "retrieval_release_sha256",
        )
    }
    decision = {
        key: payload[key]
        for key in (
            "decision_prompt",
            "decision_skills",
            "enable_evidence_adjustments",
            "max_evidence_adjustments",
            "decision_aggregation",
        )
    }
    return MappingProxyType(
        {
            "numerical_morphology": _fingerprint(numerical_morphology),
            "retrieval": _fingerprint(retrieval),
            "decision": _fingerprint(decision),
        }
    )


class RetrievalEvolutionPhaseAdapter:
    """Wire the typed Retrieval engine to an operator-loaded accepted release."""

    def __init__(
        self,
        engine: RetrievalEvolutionEngine,
        *,
        parent_release_path: str
        | Path
        | Callable[[HarnessPolicy], str | Path],
        accepted_release_path: Callable[[RetrievalEvolutionResult], str | Path],
    ) -> None:
        if not isinstance(engine, RetrievalEvolutionEngine):
            raise ValueError("Retrieval coordinate requires RetrievalEvolutionEngine")
        if not callable(accepted_release_path):
            raise ValueError("Retrieval coordinate requires an accepted-release resolver")
        if not isinstance(parent_release_path, (str, Path)) and not callable(
            parent_release_path
        ):
            raise ValueError("Retrieval coordinate requires a Parent release resolver")
        self.engine = engine
        self.parent_release_path = parent_release_path
        self.accepted_release_path = accepted_release_path

    def run(
        self,
        parent: HarnessPolicy,
        train_tasks: Sequence[ContextTask],
        dev_tasks: Sequence[ContextTask],
    ) -> CoordinatePhaseOutcome:
        parent_genome = parent.retrieval_genome
        if parent_genome is None:
            return CoordinatePhaseOutcome(
                "retrieval", parent, False, False, "missing Retrieval seed release"
            )
        parent_path = (
            self.parent_release_path(parent)
            if callable(self.parent_release_path)
            else self.parent_release_path
        )
        if not isinstance(parent_path, (str, Path)):
            raise ValueError("Parent release resolver must return a filesystem path")
        trusted_parent = _load_retrieval_release_for_operator(parent_path)
        expected_parent = embed_retrieval_release(
            parent, trusted_parent, changelog=parent.changelog
        )
        if (
            trusted_parent.genome.fingerprint() != parent_genome.fingerprint()
            or trusted_parent.manifest["state"] not in {"seed", "accepted"}
            or expected_parent.retrieval_release_sha256
            != parent.retrieval_release_sha256
            or expected_parent.retrieval_release_payload
            != parent.retrieval_release_payload
        ):
            return CoordinatePhaseOutcome(
                "retrieval",
                parent,
                False,
                False,
                "operator-loaded Parent release does not match the bundle",
            )
        result = self.engine.evolve(parent_genome, train_tasks, dev_tasks)
        if not isinstance(result, RetrievalEvolutionResult):
            raise ValueError("Retrieval engine returned an invalid result")
        if not result.accepted:
            return CoordinatePhaseOutcome(
                "retrieval",
                parent,
                False,
                False,
                ";".join(result.rejection_reasons) or "Retrieval Dev rejected",
            )
        winner = result.release_genome
        if winner is None:
            return CoordinatePhaseOutcome(
                "retrieval", parent, False, False, "accepted result omitted winner"
            )
        if (
            result.original_parent.fingerprint() != parent_genome.fingerprint()
            or result.train_winner.fingerprint() != winner.fingerprint()
            or result.selected_genome.fingerprint() != winner.fingerprint()
        ):
            return CoordinatePhaseOutcome(
                "retrieval",
                parent,
                False,
                False,
                "accepted Retrieval result has inconsistent lineage",
            )
        if _retrieval_coordinate_payload(winner) == _retrieval_coordinate_payload(
            parent_genome
        ):
            return CoordinatePhaseOutcome(
                "retrieval",
                parent,
                False,
                False,
                "accepted Retrieval result made no principal-module improvement",
            )
        release_path = self.accepted_release_path(result)
        if not isinstance(release_path, (str, Path)):
            raise ValueError("accepted-release resolver must return a filesystem path")
        # Only this private operator loader may turn the published artifacts into
        # a release object used by the accepted bundle.
        release = _load_retrieval_release_for_operator(release_path)
        if (
            release.genome.version == "v000"
            or release.manifest["state"] != "accepted"
            or _retrieval_coordinate_payload(release.genome)
            != _retrieval_coordinate_payload(winner)
            or release.genome.parent != parent_genome.version
            or int(release.genome.version[1:])
            != int(parent_genome.version[1:]) + 1
        ):
            return CoordinatePhaseOutcome(
                "retrieval",
                parent,
                False,
                False,
                "operator-loaded release does not match the accepted Retrieval winner",
            )
        embedded = embed_retrieval_release(
            parent,
            release,
            changelog=";".join(result.acceptance_reasons)
            or f"Accepted Retrieval {release.genome.version}.",
        )
        candidate = replace(
            embedded,
            version=_next_bundle_version(parent, release.genome.version),
            parent=parent.version,
        )
        return CoordinatePhaseOutcome(
            "retrieval",
            candidate,
            True,
            candidate.retrieval_release_sha256
            != parent.retrieval_release_sha256,
            "Retrieval Train/Dev gates accepted",
        )


class DecisionEvolutionPhaseAdapter:
    """Wire the existing targeted Genome engine as the Decision coordinate."""

    def __init__(
        self,
        engine: CoEvolutionEngine,
        *,
        accepted_release_path: str
        | Path
        | Callable[[HarnessPolicy], str | Path],
    ) -> None:
        if not isinstance(engine, CoEvolutionEngine):
            raise ValueError("Decision coordinate requires CoEvolutionEngine")
        if engine.config.mode != "genome" or engine.config.target != "decision":
            raise ValueError(
                "Decision coordinate engine must use genome mode and target=decision"
            )
        if not isinstance(accepted_release_path, (str, Path)) and not callable(
            accepted_release_path
        ):
            raise ValueError("Decision coordinate requires an accepted release resolver")
        self.engine = engine
        self.accepted_release_path = accepted_release_path

    def run(
        self,
        parent: HarnessPolicy,
        train_tasks: Sequence[ContextTask],
        dev_tasks: Sequence[ContextTask],
    ) -> CoordinatePhaseOutcome:
        if not parent.has_accepted_retrieval_release:
            return CoordinatePhaseOutcome(
                "decision",
                parent,
                False,
                False,
                "Decision phase requires a non-v000 accepted Retrieval release",
            )
        release_path = (
            self.accepted_release_path(parent)
            if callable(self.accepted_release_path)
            else self.accepted_release_path
        )
        if not isinstance(release_path, (str, Path)):
            raise ValueError("accepted release resolver must return a filesystem path")
        trusted_release = _load_retrieval_release_for_operator(release_path)
        expected_parent = embed_retrieval_release(
            parent, trusted_release, changelog=parent.changelog
        )
        if (
            trusted_release.genome.version == "v000"
            or trusted_release.manifest["state"] != "accepted"
            or expected_parent.retrieval_release_sha256
            != parent.retrieval_release_sha256
            or expected_parent.retrieval_release_payload
            != parent.retrieval_release_payload
        ):
            return CoordinatePhaseOutcome(
                "decision",
                parent,
                False,
                False,
                "operator-loaded accepted Retrieval release does not match the bundle",
            )
        if (
            parent.version.startswith("v")
            and parent.version[1:].isdigit()
        ):
            self.engine._version = max(
                self.engine._version, int(parent.version[1:]) + 1
            )
        candidate, trace = self.engine.evolve(parent, train_tasks, dev_tasks)
        if not isinstance(candidate, HarnessPolicy) or not isinstance(trace, tuple):
            raise ValueError("Decision engine returned an invalid result")
        accepted_version = (
            getattr(trace[-1], "accepted_version", None) if trace else None
        )
        if (
            candidate.version != parent.version
            and accepted_version == candidate.version
            and candidate.parent != parent.version
        ):
            return CoordinatePhaseOutcome(
                "decision",
                parent,
                False,
                False,
                "Decision engine returned detached bundle lineage",
            )
        accepted = (
            candidate.version != parent.version
            and accepted_version == candidate.version
            and candidate.parent == parent.version
        )
        if not accepted:
            return CoordinatePhaseOutcome(
                "decision", parent, False, False, "Decision phase had no Dev gain"
            )
        parent_fingerprints = principal_module_fingerprints(parent)
        candidate_fingerprints = principal_module_fingerprints(candidate)
        changed = tuple(
            module
            for module in _MODULES
            if candidate_fingerprints[module] != parent_fingerprints[module]
        )
        if changed != ("decision",):
            return CoordinatePhaseOutcome(
                "decision",
                parent,
                False,
                False,
                (
                    "Decision engine crossed coordinate ownership"
                    if changed
                    else "Decision engine accepted no principal-module improvement"
                ),
            )
        return CoordinatePhaseOutcome(
            "decision",
            candidate,
            True,
            True,
            "Decision Train/Dev gates accepted",
        )


class CoordinateEvolutionController:
    """Execute one accepted coordinate at a time and reject ownership drift."""

    def __init__(
        self,
        retrieval_phase: CoordinatePhaseRunner | None,
        decision_phase: CoordinatePhaseRunner | None,
        config: CoordinateEvolutionConfig | None = None,
        *,
        diagnostics: Callable[
            [HarnessPolicy], CoordinateDiagnostics | Mapping[str, object]
        ]
        | None = None,
    ) -> None:
        self.retrieval_phase = retrieval_phase
        self.decision_phase = decision_phase
        self.config = config or CoordinateEvolutionConfig()
        self.diagnostics = diagnostics

    def _target(self, generation: int, bundle: HarnessPolicy) -> CoordinateTarget:
        if self.config.phase != "alternate":
            return self.config.phase
        if generation == 0:
            return "retrieval"
        if generation == 1:
            return "decision"
        if self.diagnostics is None:
            raise ValueError("alternate coordinate evolution requires module diagnostics")
        raw = self.diagnostics(bundle)
        diagnostics = (
            raw
            if isinstance(raw, CoordinateDiagnostics)
            else CoordinateDiagnostics.from_mapping(raw)
            if isinstance(raw, Mapping)
            else None
        )
        if diagnostics is None:
            raise ValueError("alternate coordinate diagnostics are invalid")
        return diagnostics.weakest

    def run(
        self,
        seed: HarnessPolicy,
        train_tasks: Sequence[ContextTask],
        dev_tasks: Sequence[ContextTask],
    ) -> tuple[HarnessPolicy, tuple[CoordinateEvolutionStep, ...]]:
        if not isinstance(seed, HarnessPolicy):
            raise ValueError("coordinate seed must be a HarnessPolicy")
        if self.config.phase in {"decision", "alternate"} and not (
            seed.has_accepted_retrieval_release
        ):
            raise ValueError(
                "Decision and alternate phases require a non-v000 accepted Retrieval release"
            )
        if self.config.phase == "retrieval" and seed.retrieval_genome is None:
            raise ValueError("Retrieval phase requires an embedded v000 or accepted release")
        incumbent = seed
        trace: list[CoordinateEvolutionStep] = []
        for generation in range(self.config.generations):
            target = self._target(generation, incumbent)
            phase = (
                self.retrieval_phase if target == "retrieval" else self.decision_phase
            )
            if phase is None:
                raise ValueError(f"coordinate {target} phase is not configured")
            parent = incumbent
            parent_fingerprints = principal_module_fingerprints(parent)
            parent_bytes_sha256 = _bytes_sha256(parent.canonical_bytes())
            outcome = phase.run(parent, train_tasks, dev_tasks)
            if not isinstance(outcome, CoordinatePhaseOutcome):
                raise ValueError("coordinate phase returned an invalid outcome")
            if outcome.public_test_accessed:
                raise ValueError("Public Regression access is forbidden")
            candidate = outcome.bundle
            child_fingerprints = principal_module_fingerprints(candidate)
            changed = tuple(
                module
                for module in _MODULES
                if child_fingerprints[module] != parent_fingerprints[module]
            )
            accepted = (
                outcome.target == target
                and outcome.accepted
                and outcome.improved
                and changed == (target,)
            )
            reason = outcome.reason
            if outcome.target != target:
                reason = "phase returned the wrong coordinate target"
            elif outcome.accepted and outcome.improved and changed != (target,):
                reason = (
                    "coordinate ownership violation"
                    if changed
                    else "accepted phase made no principal-module improvement"
                )
            if accepted:
                incumbent = candidate
            else:
                # The exact object and bytes survive every rejected/no-op/drift path.
                incumbent = parent
            accepted_fingerprints = principal_module_fingerprints(incumbent)
            trace.append(
                CoordinateEvolutionStep(
                    generation=generation,
                    target=target,
                    accepted=accepted,
                    reason=reason,
                    parent_fingerprints=parent_fingerprints,
                    child_fingerprints=child_fingerprints,
                    accepted_fingerprints=accepted_fingerprints,
                    changed_modules=changed,
                    parent_bytes_sha256=parent_bytes_sha256,
                    child_bytes_sha256=_bytes_sha256(candidate.canonical_bytes()),
                    accepted_bytes_sha256=_bytes_sha256(
                        incumbent.canonical_bytes()
                    ),
                    public_test_accessed=False,
                )
            )
        return incumbent, tuple(trace)


__all__ = [
    "CoordinateDiagnostics",
    "CoordinateEvolutionConfig",
    "CoordinateEvolutionController",
    "CoordinateEvolutionStep",
    "CoordinatePhaseOutcome",
    "DecisionEvolutionPhaseAdapter",
    "RetrievalEvolutionPhaseAdapter",
    "principal_module_fingerprints",
]
