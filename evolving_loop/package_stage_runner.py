"""Target-agnostic 8/32/80/20 successive-halving package phase runner."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from common.data import Task as DataTask
from common.evolution_core.task_feedback import TaskEvidenceProjection
from evolving_loop.data import ContextTask
from evolving_loop.package_candidate_proposal import (
    PackageCandidate,
    PackageProposalFeedback,
)
from evolving_loop.package_coordinate_evolution import (
    PackageCoordinateState,
    PackageCoordinateTarget,
)
from evolving_loop.package_metrics import (
    PackageEvaluation,
    PackageGateConfig,
    package_full_gate_failures,
    package_rank_key,
    package_screen_failures,
)
from numerical_agent.evolution.task_local_evolution import (
    GroupFoldManifest,
    build_group_fold_manifest,
)


_STAGE_ORDER: tuple[str, ...] = (
    "screen8",
    "screen32",
    "train80",
    "dev20",
)
_STAGE_SIZES: Mapping[str, int] = {
    "screen8": 8,
    "screen32": 32,
    "train80": 80,
    "dev20": 20,
}
_PROMOTE_LIMITS: Mapping[str, int] = {
    "screen8": 2,
    "screen32": 1,
    "train80": 1,
    "dev20": 1,
}
_SCREEN_STAGES: frozenset[str] = frozenset({"screen8", "screen32"})
_FULL_STAGE_KIND: Mapping[str, str] = {
    "train80": "train",
    "dev20": "dev",
}
_DEFAULT_SEED = 20260903
_GATE_NAMES: tuple[str, ...] = (
    "minimum_relative_joint_gain",
    "maximum_task_joint_regret",
    "p95_srmse",
)


class PackageStageError(ValueError):
    """Raised when the registered stage schedule or halving contract is violated."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _entity_group_order(tasks: Sequence[DataTask], seed: int) -> tuple[tuple[str, ...], ...]:
    """Deterministic SHA-seeded order over indivisible entity task groups."""
    groups: dict[str, list[str]] = {}
    for task in tasks:
        groups.setdefault(task.entity_name, []).append(task.task_id)
    ordered = sorted(
        (
            (
                hashlib.sha256(f"{seed}{entity}".encode("utf-8")).hexdigest(),
                entity,
                tuple(sorted(task_ids)),
            )
            for entity, task_ids in groups.items()
        ),
        key=lambda item: (item[0], item[1]),
    )
    return tuple(task_ids for _sha, _entity, task_ids in ordered)


def _take_whole_groups(
    group_order: Sequence[tuple[str, ...]], size: int, *, stage: str
) -> tuple[str, ...]:
    chosen: list[str] = []
    for group in group_order:
        if len(chosen) + len(group) > size:
            continue
        chosen.extend(group)
        if len(chosen) == size:
            return tuple(sorted(chosen))
    raise PackageStageError(
        f"{stage} cannot be formed from whole task groups at size {size}"
    )


@dataclass(frozen=True)
class PackageStageSchedule:
    """Registered nested 8/32/80/20 universe with five-fold Train cross-fit."""

    seed: int
    screen8_ids: tuple[str, ...]
    screen32_ids: tuple[str, ...]
    train80_ids: tuple[str, ...]
    dev20_ids: tuple[str, ...]
    fold_manifest: GroupFoldManifest

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise PackageStageError("stage schedule seed must be an exact integer")
        stages = {
            "screen8": self.screen8_ids,
            "screen32": self.screen32_ids,
            "train80": self.train80_ids,
            "dev20": self.dev20_ids,
        }
        for stage, ids in stages.items():
            if (
                type(ids) is not tuple
                or len(ids) != _STAGE_SIZES[stage]
                or tuple(sorted(ids)) != ids
                or len(set(ids)) != len(ids)
                or any(type(task_id) is not str or not task_id for task_id in ids)
            ):
                raise PackageStageError(f"{stage} membership is not a registered set")
        if not set(self.screen8_ids) <= set(self.screen32_ids):
            raise PackageStageError("screen8 must nest inside screen32")
        if not set(self.screen32_ids) <= set(self.train80_ids):
            raise PackageStageError("screen32 must nest inside train80")
        if not set(self.dev20_ids).isdisjoint(self.train80_ids):
            raise PackageStageError("Dev must be disjoint from the Train partition")
        if not isinstance(self.fold_manifest, GroupFoldManifest):
            raise PackageStageError("stage schedule requires a GroupFoldManifest")
        if self.fold_manifest.fold_count != 5:
            raise PackageStageError("Train cross-fit requires exactly five folds")
        if set(self.fold_manifest.task_fold_map) != set(self.train80_ids):
            raise PackageStageError("fold manifest must cover the exact Train universe")

    @classmethod
    def build(
        cls,
        train_tasks: Sequence[DataTask],
        dev_tasks: Sequence[DataTask],
        *,
        seed: int = _DEFAULT_SEED,
    ) -> "PackageStageSchedule":
        if type(seed) is not int:
            raise PackageStageError("stage schedule seed must be an exact integer")
        train = tuple(train_tasks)
        dev = tuple(dev_tasks)
        if len(train) != 80 or any(type(task) is not DataTask for task in train):
            raise PackageStageError("registered Train partition requires exactly 80 tasks")
        if len(dev) != 20 or any(type(task) is not DataTask for task in dev):
            raise PackageStageError("registered Dev partition requires exactly 20 tasks")
        fold_manifest = build_group_fold_manifest(train, seed=seed, fold_count=5)
        group_order = _entity_group_order(train, seed)
        screen32 = _take_whole_groups(group_order, 32, stage="screen32")
        screen32_groups = tuple(
            group for group in group_order if set(group) <= set(screen32)
        )
        screen8 = _take_whole_groups(screen32_groups, 8, stage="screen8")
        return cls(
            seed=seed,
            screen8_ids=screen8,
            screen32_ids=screen32,
            train80_ids=tuple(sorted(task.task_id for task in train)),
            dev20_ids=tuple(sorted(task.task_id for task in dev)),
            fold_manifest=fold_manifest,
        )

    @property
    def counts(self) -> tuple[int, int, int, int]:
        return (8, 32, 80, 20)

    def stage_ids(self, stage: str) -> tuple[str, ...]:
        if stage not in _STAGE_SIZES:
            raise PackageStageError(f"unknown stage {stage!r}")
        return {
            "screen8": self.screen8_ids,
            "screen32": self.screen32_ids,
            "train80": self.train80_ids,
            "dev20": self.dev20_ids,
        }[stage]

    def tasks_for(
        self, stage: str, task_map: Mapping[str, ContextTask]
    ) -> tuple[ContextTask, ...]:
        """The only accessor that returns task objects; verifies exact membership."""
        ids = self.stage_ids(stage)
        missing = tuple(task_id for task_id in ids if task_id not in task_map)
        if missing:
            raise PackageStageError(
                f"host task map is missing {stage} tasks: {missing[:3]}"
            )
        resolved = tuple(task_map[task_id] for task_id in ids)
        if any(
            not isinstance(task, ContextTask) or task.numeric.task_id != task_id
            for task, task_id in zip(resolved, ids, strict=True)
        ):
            raise PackageStageError(f"host {stage} tasks do not match their IDs")
        return resolved

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "seed": self.seed,
            "screen8_ids": list(self.screen8_ids),
            "screen32_ids": list(self.screen32_ids),
            "train80_ids": list(self.train80_ids),
            "dev20_ids": list(self.dev20_ids),
            "fold_manifest": self.fold_manifest.to_payload(),
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_payload())


@dataclass(frozen=True)
class PackageStageEvidence:
    """One immutable stage/candidate scoring record."""

    stage: str
    parent_sha256: str
    candidate_sha256: str
    parent_evaluation: PackageEvaluation
    child_evaluation: PackageEvaluation
    gate_failures: tuple[str, ...]
    opened: bool

    def __post_init__(self) -> None:
        if self.stage not in _STAGE_SIZES:
            raise PackageStageError(f"stage evidence has unknown stage {self.stage!r}")
        if not isinstance(self.parent_evaluation, PackageEvaluation) or not isinstance(
            self.child_evaluation, PackageEvaluation
        ):
            raise PackageStageError("stage evidence requires PackageEvaluation values")
        object.__setattr__(self, "gate_failures", tuple(self.gate_failures))

    def to_payload(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "parent_sha256": self.parent_sha256,
            "candidate_sha256": self.candidate_sha256,
            "parent_evaluation": self.parent_evaluation.to_payload(),
            "child_evaluation": self.child_evaluation.to_payload(),
            "gate_failures": list(self.gate_failures),
            "opened": self.opened,
        }


@dataclass(frozen=True)
class PackageCoordinatePhaseOutcome:
    """The gated result of one coordinate generation."""

    target: PackageCoordinateTarget
    parent: PackageCoordinateState
    finalist: PackageCandidate | None
    selected: PackageCoordinateState
    accepted: bool
    improved: bool
    reason: str
    evidence: tuple[PackageStageEvidence, ...]
    public_test_accessed: bool = False

    def __post_init__(self) -> None:
        if self.target not in ("numerical", "retrieval", "decision"):
            raise PackageStageError("phase outcome target is invalid")
        if not isinstance(self.parent, PackageCoordinateState) or not isinstance(
            self.selected, PackageCoordinateState
        ):
            raise PackageStageError("phase outcome requires package states")
        if not isinstance(self.reason, str) or not self.reason:
            raise PackageStageError("phase outcome requires a reason")
        if self.accepted != self.improved:
            raise PackageStageError("package acceptance and improvement must agree")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if self.accepted:
            if self.finalist is None:
                raise PackageStageError("an accepted phase requires a finalist")
            if self.selected.bundle.acceptance_evidence_sha256 is None:
                raise PackageStageError("an accepted phase must seal its finalist")
            if self.public_test_accessed:
                raise PackageStageError("a Public-accessed phase cannot be accepted")
        elif self.selected.bundle.fingerprint() != self.parent.bundle.fingerprint():
            raise PackageStageError("a rejected phase must preserve the exact Parent")


@runtime_checkable
class PackageStageArtifactSink(Protocol):
    """Minimal append-only boundary; Task 8 supplies the durable implementation."""

    def record_candidate_evidence(
        self, target: str, generation: int, proposal_sha256: str, payload: Mapping[str, object]
    ) -> None: ...

    def record_stage_evidence(self, evidence: PackageStageEvidence) -> None: ...

    def record_acceptance_evidence(
        self, evidence_sha256: str, payload: Mapping[str, object]
    ) -> None: ...

    def contains_evidence(self, evidence_sha256: str) -> bool: ...


class InMemoryPackageArtifactSink:
    """A non-durable sink for deterministic tests and fake-LLM end-to-end runs."""

    def __init__(self) -> None:
        self.candidate_evidence: list[dict[str, object]] = []
        self.stage_evidence: list[PackageStageEvidence] = []
        self.acceptance_evidence: dict[str, Mapping[str, object]] = {}

    def record_candidate_evidence(
        self, target: str, generation: int, proposal_sha256: str, payload: Mapping[str, object]
    ) -> None:
        self.candidate_evidence.append(
            {
                "target": target,
                "generation": generation,
                "proposal_sha256": proposal_sha256,
                "payload": dict(payload),
            }
        )

    def record_stage_evidence(self, evidence: PackageStageEvidence) -> None:
        self.stage_evidence.append(evidence)

    def record_acceptance_evidence(
        self, evidence_sha256: str, payload: Mapping[str, object]
    ) -> None:
        if evidence_sha256 in self.acceptance_evidence:
            raise PackageStageError("acceptance evidence digest already recorded")
        if _digest(dict(payload)) != evidence_sha256:
            raise PackageStageError("acceptance evidence digest does not bind its payload")
        self.acceptance_evidence[evidence_sha256] = dict(payload)

    def contains_evidence(self, evidence_sha256: str) -> bool:
        return evidence_sha256 in self.acceptance_evidence


@runtime_checkable
class PackagePhaseEvaluator(Protocol):
    """Injected evaluator boundary (real `PackagePipelineEvaluator` or a fake)."""

    def evaluate(
        self,
        bundle: object,
        registry: object,
        tasks: Sequence[ContextTask],
        *,
        stage: str,
        cache_only: bool = False,
    ) -> PackageEvaluation: ...


class RetrievalAcceptedPublisher(Protocol):
    """Rebase behavior onto the next accepted Retrieval release and re-embed it."""

    def publish(
        self, finalist: PackageCandidate, phase_parent: PackageCoordinateState
    ) -> PackageCoordinateState: ...


class PackageTaskFeedbackProvider(Protocol):
    """Supply one Parent-bound sanitized projection to a Numerical phase."""

    def for_numerical(
        self,
        state: PackageCoordinateState,
        *,
        generation: int,
    ) -> TaskEvidenceProjection | None: ...


class PackageCoordinatePhaseRunner:
    """Run one coordinate generation through the registered halving schedule."""

    def __init__(
        self,
        target: PackageCoordinateTarget,
        proposer: object,
        evaluator: PackagePhaseEvaluator,
        schedule: PackageStageSchedule,
        task_map: Mapping[str, ContextTask],
        gate_config: PackageGateConfig,
        artifact_store: PackageStageArtifactSink,
        *,
        retrieval_publisher: RetrievalAcceptedPublisher | None = None,
        feedback_manager: PackageTaskFeedbackProvider | None = None,
        child_count: int = 3,
    ) -> None:
        if target not in ("numerical", "retrieval", "decision"):
            raise PackageStageError("phase runner target is invalid")
        if not callable(getattr(proposer, "propose", None)):
            raise PackageStageError("phase runner requires a candidate proposer")
        if not isinstance(evaluator, PackagePhaseEvaluator):
            raise PackageStageError("phase runner requires an evaluate(...) boundary")
        if not isinstance(schedule, PackageStageSchedule):
            raise PackageStageError("phase runner requires a PackageStageSchedule")
        if not isinstance(gate_config, PackageGateConfig):
            raise PackageStageError("phase runner requires a PackageGateConfig")
        if not isinstance(artifact_store, PackageStageArtifactSink):
            raise PackageStageError("phase runner requires an artifact sink")
        if type(child_count) is not int or child_count != 3:
            raise PackageStageError("a formal coordinate generation proposes three Children")
        if target == "retrieval" and retrieval_publisher is None:
            raise PackageStageError("the Retrieval coordinate requires an accepted publisher")
        if feedback_manager is not None and not callable(
            getattr(feedback_manager, "for_numerical", None)
        ):
            raise PackageStageError("task feedback provider requires for_numerical(...)")
        self.target = target
        self.proposer = proposer
        self.evaluator = evaluator
        self.schedule = schedule
        self.task_map = dict(task_map)
        self.gate_config = gate_config
        self.artifact_store = artifact_store
        self.retrieval_publisher = retrieval_publisher
        self.feedback_manager = feedback_manager
        self.child_count = child_count

    # -- evaluation helpers -------------------------------------------------

    def _evaluate(
        self, state: PackageCoordinateState, stage: str, *, cache_only: bool = False
    ) -> PackageEvaluation:
        tasks = self.schedule.tasks_for(stage, self.task_map)
        evaluation = self.evaluator.evaluate(
            state.bundle, state.registry, tasks, stage=stage, cache_only=cache_only
        )
        if not isinstance(evaluation, PackageEvaluation):
            raise PackageStageError("phase evaluator returned an invalid evaluation")
        return evaluation

    def _gate_failures(
        self,
        stage: str,
        child_eval: PackageEvaluation,
        parent_eval: PackageEvaluation,
        initial_eval: PackageEvaluation | None,
    ) -> tuple[str, ...]:
        if stage in _SCREEN_STAGES:
            return package_screen_failures(child_eval, parent_eval, self.gate_config)
        return package_full_gate_failures(
            child_eval,
            parent_eval,
            self.gate_config,
            stage=_FULL_STAGE_KIND[stage],
            fold_manifest=self.schedule.fold_manifest if stage == "train80" else None,
            initial=initial_eval,
        )

    # -- the phase --------------------------------------------------------

    def run(
        self,
        parent: PackageCoordinateState,
        initial: PackageCoordinateState,
        *,
        generation: int,
    ) -> PackageCoordinatePhaseOutcome:
        if not isinstance(parent, PackageCoordinateState) or not isinstance(
            initial, PackageCoordinateState
        ):
            raise PackageStageError("phase runner requires package states")
        if type(generation) is not int or generation < 0:
            raise PackageStageError("phase generation must be non-negative")

        parent_screen = self._evaluate(parent, "screen8")
        task_evidence = (
            self.feedback_manager.for_numerical(parent, generation=generation)
            if self.target == "numerical" and self.feedback_manager is not None
            else None
        )
        feedback = PackageProposalFeedback.from_evaluations(
            parent=parent_screen,
            rejected_children=(),
            gate_names=_GATE_NAMES,
            structures=(),
            task_evidence=task_evidence,
        )
        candidates = tuple(
            self.proposer.propose(
                parent, feedback, generation=generation, child_count=self.child_count
            )
        )
        if len(candidates) != self.child_count or any(
            not isinstance(candidate, PackageCandidate) for candidate in candidates
        ):
            raise PackageStageError("proposer must return exactly three PackageCandidates")
        if tuple(candidate.slot for candidate in candidates) != (0, 1, 2) or any(
            candidate.target != self.target for candidate in candidates
        ):
            raise PackageStageError("proposer returned an off-target or unordered slot set")
        for candidate in candidates:
            self.artifact_store.record_candidate_evidence(
                self.target,
                generation,
                candidate.proposal_sha256,
                {
                    "slot": candidate.slot,
                    "invalid_reason": candidate.invalid_reason,
                    "bundle_sha256": candidate.state.bundle.fingerprint(),
                },
            )

        evidence: list[PackageStageEvidence] = []
        active: list[PackageCandidate] = [
            candidate for candidate in candidates if candidate.invalid_reason is None
        ]
        if not active:
            return self._reject(
                parent, candidates, evidence, "every proposed Child was structurally invalid"
            )

        finalist: PackageCandidate | None = None
        finalist_prepub: PackageEvaluation | None = None
        for stage in _STAGE_ORDER:
            parent_eval = self._evaluate(parent, stage)
            initial_eval = (
                self._evaluate(initial, stage) if stage not in _SCREEN_STAGES else None
            )
            ranked: list[tuple[tuple[float, ...], PackageCandidate, PackageEvaluation]] = []
            for candidate in active:
                child_eval = self._evaluate(candidate.state, stage)
                failures = self._gate_failures(stage, child_eval, parent_eval, initial_eval)
                record = PackageStageEvidence(
                    stage=stage,
                    parent_sha256=parent.bundle.fingerprint(),
                    candidate_sha256=candidate.state.bundle.fingerprint(),
                    parent_evaluation=parent_eval,
                    child_evaluation=child_eval,
                    gate_failures=failures,
                    opened=True,
                )
                evidence.append(record)
                self.artifact_store.record_stage_evidence(record)
                if not failures:
                    ranked.append(
                        (
                            (*package_rank_key(child_eval), candidate.state.bundle.fingerprint()),
                            candidate,
                            child_eval,
                        )
                    )
            ranked.sort(key=lambda item: item[0])
            promoted = ranked[: _PROMOTE_LIMITS[stage]]
            if not promoted:
                return self._reject(
                    parent, candidates, evidence, f"no Child passed the {stage} gate"
                )
            active = [candidate for _key, candidate, _eval in promoted]
            if stage == "dev20":
                finalist = active[0]
                finalist_prepub = promoted[0][2]

        assert finalist is not None and finalist_prepub is not None
        return self._accept(parent, initial, candidates, finalist, finalist_prepub, evidence)

    # -- terminal transitions -------------------------------------------

    def _reject(
        self,
        parent: PackageCoordinateState,
        candidates: Sequence[PackageCandidate],
        evidence: Sequence[PackageStageEvidence],
        reason: str,
    ) -> PackageCoordinatePhaseOutcome:
        return PackageCoordinatePhaseOutcome(
            target=self.target,
            parent=parent,
            finalist=None,
            selected=parent,
            accepted=False,
            improved=False,
            reason=reason,
            evidence=tuple(evidence),
            public_test_accessed=False,
        )

    def _accept(
        self,
        parent: PackageCoordinateState,
        initial: PackageCoordinateState,
        candidates: Sequence[PackageCandidate],
        finalist: PackageCandidate,
        prepublication: PackageEvaluation,
        evidence: Sequence[PackageStageEvidence],
    ) -> PackageCoordinatePhaseOutcome:
        selected_state = finalist.state
        if self.target == "retrieval":
            assert self.retrieval_publisher is not None
            selected_state = self.retrieval_publisher.publish(finalist, parent)
            if not isinstance(selected_state, PackageCoordinateState):
                raise PackageStageError("Retrieval publisher returned an invalid state")
            if (
                selected_state.bundle.parent_sha256 != parent.bundle.fingerprint()
                or selected_state.bundle.coordinate != "retrieval"
            ):
                raise PackageStageError("Retrieval publisher broke direct lineage")
        # Immediate cache-backed replay; a miss never reruns the model here.
        try:
            replayed = self.evaluator.evaluate(
                selected_state.bundle,
                selected_state.registry,
                self.schedule.tasks_for("dev20", self.task_map),
                stage="dev20",
                cache_only=True,
            )
        except (RuntimeError, LookupError, KeyError) as error:  # cache unavailable
            return PackageCoordinatePhaseOutcome(
                target=self.target,
                parent=parent,
                finalist=None,
                selected=parent,
                accepted=False,
                improved=False,
                reason=f"replay_cache_miss:{type(error).__name__}",
                evidence=tuple(evidence),
                public_test_accessed=False,
            )
        if replayed.result_bytes() != prepublication.result_bytes():
            return self._reject(
                parent, candidates, evidence, "cache-backed replay did not reproduce the finalist"
            )

        acceptance_payload = {
            "schema_version": 1,
            "target": self.target,
            "parent_bundle_sha256": parent.bundle.fingerprint(),
            "finalist_bundle_sha256": finalist.state.bundle.fingerprint(),
            "selected_bundle_sha256": selected_state.bundle.fingerprint(),
            "proposal_sha256": finalist.proposal_sha256,
            "schedule_sha256": self.schedule.fingerprint,
            "stage_evidence": [record.to_payload() for record in evidence],
            "replay_result_sha256": hashlib.sha256(replayed.result_bytes()).hexdigest(),
        }
        evidence_sha256 = _digest(acceptance_payload)
        self.artifact_store.record_acceptance_evidence(evidence_sha256, acceptance_payload)
        if not self.artifact_store.contains_evidence(evidence_sha256):
            raise PackageStageError("acceptance evidence was not durably recorded")
        sealed = selected_state.seal_acceptance(evidence_sha256)
        return PackageCoordinatePhaseOutcome(
            target=self.target,
            parent=parent,
            finalist=finalist,
            selected=sealed,
            accepted=True,
            improved=True,
            reason="accepted after cache-backed replay",
            evidence=tuple(evidence),
            public_test_accessed=False,
        )


__all__ = [
    "InMemoryPackageArtifactSink",
    "PackageCoordinatePhaseOutcome",
    "PackageCoordinatePhaseRunner",
    "PackagePhaseEvaluator",
    "PackageStageArtifactSink",
    "PackageStageError",
    "PackageStageEvidence",
    "PackageStageSchedule",
    "PackageTaskFeedbackProvider",
    "RetrievalAcceptedPublisher",
]
