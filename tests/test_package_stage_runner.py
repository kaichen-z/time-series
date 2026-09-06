"""Registered 8/32/80/20 successive-halving package phase runner."""
from __future__ import annotations

from dataclasses import replace

import pytest

from common.data import Task as DataTask
from common.evolution_core.task_feedback import TaskEvidenceProjection
from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.data import ContextTask, Document
from evolving_loop.package_candidate_proposal import PackageCandidate
from evolving_loop.package_coordinate_evolution import PackageCoordinateState
from evolving_loop.package_metrics import (
    PackageEvaluation,
    PackageGateConfig,
    PackageTaskScore,
)
from evolving_loop.package_stage_runner import (
    InMemoryPackageArtifactSink,
    PackageCoordinatePhaseOutcome,
    PackageCoordinatePhaseRunner,
    PackageStageError,
    PackageStageSchedule,
)
from common.data import Task as ContextNumericTask
from tests.test_package_coordinate_evolution import _bundle


# --------------------------------------------------------------------------
# task universes
# --------------------------------------------------------------------------


def _data_tasks(prefix: str, count: int, *, per_entity: int) -> tuple[DataTask, ...]:
    return tuple(
        DataTask(
            task_id=f"{prefix}_{index:03d}",
            history_values=tuple(float(index + offset) for offset in range(6 + index)),
            future_values=(float(index + 3), float(index + 4)),
            prediction_length=2,
            frequency="D",
            seasonal_period="7",
            entity_name=f"{prefix}_entity_{index // per_entity:03d}",
        )
        for index in range(count)
    )


TRAIN_80 = _data_tasks("train", 80, per_entity=4)
DEV_20 = _data_tasks("dev", 20, per_entity=1)


def _context_task(task_id: str) -> ContextTask:
    return ContextTask(
        numeric=ContextNumericTask(
            task_id=task_id,
            history_values=(1.0, 2.0, 3.0) * 4,
            future_values=(1.0, 2.0),
            prediction_length=2,
            frequency="D",
            seasonal_period=None,
            entity_name="Entity Z",
        ),
        target_name="sales",
        target_description="Daily sales",
        history_timestamps=tuple(f"2026-01-{index + 1:02d}" for index in range(12)),
        future_timestamps=("2026-01-13", "2026-01-14"),
        documents=(Document("doc", "text", role="supporting"),),
        gt_evidence=("text",),
        labels_public=True,
    )


def _task_map() -> dict[str, ContextTask]:
    ids = [task.task_id for task in (*TRAIN_80, *DEV_20)]
    return {task_id: _context_task(task_id) for task_id in ids}


def _schedule() -> PackageStageSchedule:
    return PackageStageSchedule.build(TRAIN_80, DEV_20, seed=20260903)


# --------------------------------------------------------------------------
# fake evaluator + proposer
# --------------------------------------------------------------------------


def _rows(task_ids, error: float) -> tuple[PackageTaskScore, ...]:
    return tuple(
        PackageTaskScore(
            task_id=task_id,
            entity_name="Entity Z",
            final_smae=error,
            final_srmse=error,
            final_smae_raw=error,
            final_srmse_raw=error,
            final_forecast=(1.0, 2.0),
            numerical_oracle_smae=error,
            numerical_oracle_srmse=error,
            numerical_candidate_count=2,
            smae_clipped=False,
            srmse_clipped=False,
            invalid_count=0,
            catastrophic_count=0,
            fallback_count=0,
            selected_candidate_id="safe_anchor",
            numerical_package_sha256="1" * 64,
            final_retrieval_sha256="2" * 64,
            final_decision_sha256="3" * 64,
        )
        for task_id in task_ids
    )


def _evaluation(candidate_sha: str, task_ids, error: float) -> PackageEvaluation:
    return PackageEvaluation.from_rows(candidate_sha, _rows(task_ids, error), tuple(task_ids))


class _ScriptedEvaluator:
    """Return a uniform-error evaluation whose error depends on the bundle."""

    def __init__(
        self,
        errors: dict[str, float],
        *,
        cache_ok: bool = True,
        stage_errors: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.errors = errors
        self.cache_ok = cache_ok
        self.stage_errors = stage_errors or {}
        self.stage_counts: dict[str, list[str]] = {}

    def error_for(self, bundle, stage: str) -> float:
        return self.stage_errors.get(
            (bundle.fingerprint(), stage), self.errors.get(bundle.fingerprint(), 1.0)
        )

    def evaluate(self, bundle, registry, tasks, *, stage, cache_only=False):
        task_ids = tuple(task.numeric.task_id for task in tasks)
        if cache_only:
            if not self.cache_ok:
                raise RuntimeError("cache-only package evaluation is unavailable")
            return _evaluation(bundle.fingerprint(), task_ids, self.error_for(bundle, stage))
        self.stage_counts.setdefault(stage, []).append(bundle.fingerprint())
        return _evaluation(bundle.fingerprint(), task_ids, self.error_for(bundle, stage))

    def child_stage_counts(self, stage: str, parent_sha: str) -> int:
        return sum(1 for sha in self.stage_counts.get(stage, ()) if sha != parent_sha)

    def child_fingerprints_seen_on(self, stage: str, parent_sha: str) -> set[str]:
        return {sha for sha in self.stage_counts.get(stage, ()) if sha != parent_sha}


class _DecisionProposer:
    """Emit three Decision Children by rewriting the owned Decision prompt."""

    def __init__(self, prompts: tuple[str, str, str]) -> None:
        self.prompts = prompts

    def propose(self, parent, feedback, *, generation, child_count):
        assert child_count == 3
        children = []
        for slot, suffix in enumerate(self.prompts):
            child_policy = replace(
                parent.bundle.policy,
                decision_prompt=f"{parent.bundle.policy.decision_prompt} :: {suffix}",
            )
            state = parent.with_policy(child_policy, target="decision")
            children.append(
                PackageCandidate(
                    slot=slot,
                    target="decision",
                    state=state,
                    proposal_sha256=state.bundle.fingerprint(),
                )
            )
        return tuple(children)


def _runner(tmp_path, evaluator, proposer, *, sink=None):
    _task, parent = _bundle(tmp_path)
    runner = PackageCoordinatePhaseRunner(
        target="decision",
        proposer=proposer,
        evaluator=evaluator,
        schedule=_schedule(),
        task_map=_task_map(),
        gate_config=PackageGateConfig(),
        artifact_store=sink or InMemoryPackageArtifactSink(),
        child_count=3,
    )
    return parent, runner


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------


def test_schedule_is_nested_registered_and_group_aware() -> None:
    schedule = _schedule()
    assert len(schedule.screen8_ids) == 8
    assert len(schedule.screen32_ids) == 32
    assert len(schedule.train80_ids) == 80
    assert len(schedule.dev20_ids) == 20
    assert set(schedule.screen8_ids) <= set(schedule.screen32_ids)
    assert set(schedule.screen32_ids) <= set(schedule.train80_ids)
    assert set(schedule.train80_ids).isdisjoint(schedule.dev20_ids)
    assert schedule.fold_manifest.fold_count == 5
    assert set(schedule.fold_manifest.task_fold_map) == set(schedule.train80_ids)
    assert schedule.counts == (8, 32, 80, 20)
    assert schedule.stage_ids("train80") == schedule.train80_ids
    with pytest.raises(PackageStageError, match="unknown stage"):
        schedule.stage_ids("calibration16")
    assert len(schedule.fingerprint) == 64


def test_schedule_rejects_a_non_eighty_train_partition() -> None:
    with pytest.raises(PackageStageError, match="80"):
        PackageStageSchedule.build(TRAIN_80[:79], DEV_20, seed=20260903)


def test_tasks_for_verifies_exact_membership() -> None:
    schedule = _schedule()
    task_map = _task_map()
    resolved = schedule.tasks_for("screen8", task_map)
    assert tuple(task.numeric.task_id for task in resolved) == schedule.screen8_ids
    del task_map[schedule.screen8_ids[0]]
    with pytest.raises(PackageStageError, match="missing"):
        schedule.tasks_for("screen8", task_map)


# --------------------------------------------------------------------------
# halving
# --------------------------------------------------------------------------


def test_only_one_child_can_open_full_train_and_dev(tmp_path) -> None:
    proposer = _DecisionProposer(("a", "b", "c"))
    _task, parent = _bundle(tmp_path)
    # child errors: two survive the 8-task screen, one wins from 32 onward.
    child_policies = [
        replace(parent.bundle.policy, decision_prompt=f"{parent.bundle.policy.decision_prompt} :: {s}")
        for s in ("a", "b", "c")
    ]
    child_states = [parent.with_policy(policy, target="decision") for policy in child_policies]
    errors = {parent.bundle.fingerprint(): 1.0}
    errors[child_states[0].bundle.fingerprint()] = 0.90
    errors[child_states[1].bundle.fingerprint()] = 0.95
    errors[child_states[2].bundle.fingerprint()] = 1.20
    evaluator = _ScriptedEvaluator(errors)
    runner = PackageCoordinatePhaseRunner(
        target="decision",
        proposer=proposer,
        evaluator=evaluator,
        schedule=_schedule(),
        task_map=_task_map(),
        gate_config=PackageGateConfig(),
        artifact_store=InMemoryPackageArtifactSink(),
        child_count=3,
    )
    outcome = runner.run(parent, parent, generation=0)
    parent_sha = parent.bundle.fingerprint()
    assert evaluator.child_stage_counts("screen8", parent_sha) == 3
    assert evaluator.child_stage_counts("screen32", parent_sha) <= 2
    assert evaluator.child_stage_counts("train80", parent_sha) <= 1
    assert evaluator.child_stage_counts("calibration16", parent_sha) == 0
    assert evaluator.child_stage_counts("dev20", parent_sha) <= 1
    assert len(evaluator.child_fingerprints_seen_on("dev20", parent_sha)) <= 1
    assert outcome.accepted is True
    assert outcome.selected.bundle.acceptance_evidence_sha256 is not None


@pytest.mark.parametrize(
    "failure,expected_last_stage",
    (
        ("screen_regression", "screen8"),
        ("dev_regression", "dev20"),
        ("replay_cache_miss", "dev20"),
    ),
)
def test_phase_stops_and_preserves_parent_on_a_failed_gate(
    tmp_path, failure, expected_last_stage
) -> None:
    proposer = _DecisionProposer(("a", "b", "c"))
    _task, parent = _bundle(tmp_path)
    child_states = [
        parent.with_policy(
            replace(parent.bundle.policy, decision_prompt=f"{parent.bundle.policy.decision_prompt} :: {s}"),
            target="decision",
        )
        for s in ("a", "b", "c")
    ]
    parent_sha = parent.bundle.fingerprint()
    errors = {parent_sha: 1.0}
    stage_errors: dict[tuple[str, str], float] = {}
    cache_ok = True
    if failure == "screen_regression":
        for state in child_states:
            errors[state.bundle.fingerprint()] = 1.5
    elif failure == "dev_regression":
        for state in child_states:
            errors[state.bundle.fingerprint()] = 0.90
            stage_errors[(state.bundle.fingerprint(), "dev20")] = 1.5
    elif failure == "replay_cache_miss":
        for state in child_states:
            errors[state.bundle.fingerprint()] = 0.90
        cache_ok = False
    evaluator = _ScriptedEvaluator(errors, cache_ok=cache_ok, stage_errors=stage_errors)
    runner = PackageCoordinatePhaseRunner(
        target="decision",
        proposer=proposer,
        evaluator=evaluator,
        schedule=_schedule(),
        task_map=_task_map(),
        gate_config=PackageGateConfig(),
        artifact_store=InMemoryPackageArtifactSink(),
        child_count=3,
    )
    outcome = runner.run(parent, parent, generation=0)
    assert outcome.accepted is False
    assert outcome.selected.bundle.fingerprint() == parent_sha
    assert outcome.evidence[-1].stage == expected_last_stage


def test_structurally_invalid_slots_reject_the_phase(tmp_path) -> None:
    _task, parent = _bundle(tmp_path)

    class _InvalidProposer:
        def propose(self, parent, feedback, *, generation, child_count):
            return tuple(
                PackageCandidate(
                    slot=slot,
                    target="decision",
                    state=parent,
                    proposal_sha256=("%064x" % slot),
                    invalid_reason="materialization_failed",
                )
                for slot in range(3)
            )

    runner = PackageCoordinatePhaseRunner(
        target="decision",
        proposer=_InvalidProposer(),
        evaluator=_ScriptedEvaluator({parent.bundle.fingerprint(): 1.0}),
        schedule=_schedule(),
        task_map=_task_map(),
        gate_config=PackageGateConfig(),
        artifact_store=InMemoryPackageArtifactSink(),
        child_count=3,
    )
    outcome = runner.run(parent, parent, generation=0)
    assert outcome.accepted is False
    assert outcome.selected.bundle.fingerprint() == parent.bundle.fingerprint()
    assert "invalid" in outcome.reason


def test_formal_numerical_phase_injects_generation_bound_task_feedback(
    tmp_path,
) -> None:
    _task, parent = _bundle(tmp_path)
    projection = TaskEvidenceProjection(
        source_bundle_sha256=parent.bundle.fingerprint(),
        request_namespace_sha256="b" * 64,
        cases=(),
    )

    class FeedbackManager:
        def __init__(self) -> None:
            self.calls = []

        def for_numerical(self, state, *, generation):
            self.calls.append((state.bundle.fingerprint(), generation))
            return projection

    class CapturingProposer:
        def __init__(self) -> None:
            self.task_evidence = None

        def propose(self, state, feedback, *, generation, child_count):
            self.task_evidence = feedback.task_evidence
            return tuple(
                PackageCandidate(
                    slot=slot,
                    target="numerical",
                    state=state,
                    proposal_sha256=f"{slot:064x}",
                    invalid_reason="materialization_failed",
                )
                for slot in range(child_count)
            )

    manager = FeedbackManager()
    proposer = CapturingProposer()
    runner = PackageCoordinatePhaseRunner(
        target="numerical",
        proposer=proposer,
        evaluator=_ScriptedEvaluator({parent.bundle.fingerprint(): 1.0}),
        schedule=_schedule(),
        task_map=_task_map(),
        gate_config=PackageGateConfig(),
        artifact_store=InMemoryPackageArtifactSink(),
        feedback_manager=manager,
        child_count=3,
    )

    outcome = runner.run(parent, parent, generation=3)

    assert outcome.accepted is False
    assert manager.calls == [(parent.bundle.fingerprint(), 3)]
    assert proposer.task_evidence is projection


def test_accepted_phase_records_sealed_acceptance_evidence(tmp_path) -> None:
    proposer = _DecisionProposer(("a", "b", "c"))
    _task, parent = _bundle(tmp_path)
    child_states = [
        parent.with_policy(
            replace(parent.bundle.policy, decision_prompt=f"{parent.bundle.policy.decision_prompt} :: {s}"),
            target="decision",
        )
        for s in ("a", "b", "c")
    ]
    errors = {parent.bundle.fingerprint(): 1.0}
    for offset, state in enumerate(child_states):
        errors[state.bundle.fingerprint()] = 0.90 + offset * 0.01
    sink = InMemoryPackageArtifactSink()
    evaluator = _ScriptedEvaluator(errors)
    runner = PackageCoordinatePhaseRunner(
        target="decision",
        proposer=proposer,
        evaluator=evaluator,
        schedule=_schedule(),
        task_map=_task_map(),
        gate_config=PackageGateConfig(),
        artifact_store=sink,
        child_count=3,
    )
    outcome = runner.run(parent, parent, generation=0)
    assert isinstance(outcome, PackageCoordinatePhaseOutcome)
    assert outcome.accepted is True
    assert outcome.improved is True
    sealed = outcome.selected.bundle.acceptance_evidence_sha256
    assert sealed is not None
    assert sink.contains_evidence(sealed)
    assert len(sink.candidate_evidence) == 3
