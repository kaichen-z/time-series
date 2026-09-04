"""Package-native evolution CLI, cache, and Public-firewall contracts."""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.data import Task as DataTask
from common.evolution_core.task_feedback import TaskEvidenceProjection
from evolving_loop.co_evolution import HarnessPolicy, embed_retrieval_release
from evolving_loop.data import ContextTask, load_context_tasks_by_ids
from evolving_loop.package_artifacts import (
    PackageArtifactError,
    PackageArtifactStore,
    PackageCacheMissError,
)
from evolving_loop.run_package_coevolution import (
    PackageCacheBackedEvaluator,
    _CheckpointRecorder,
    _InteractionFeedbackManager,
    _SmokeNumericalProposer,
    _build_registry,
    _configuration_identity,
    _early_resume_guard,
    _interaction_smoke_gate_failures,
    _interaction_smoke_is_complete,
    _run_controller,
    _resume_controller,
    _state_from_payload,
    _step_payload,
    _validate_mode,
    build_parser,
    main,
)
from evolving_loop.package_coordinate_evolution import PackageCoordinateStep
from evolving_loop.package_candidate_proposal import PackageProposalFeedback
from evolving_loop.package_task_feedback import PackageTaskFeedbackLedger
from evolving_loop.package_stage_runner import PackageCoordinatePhaseOutcome
from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.policy import RetrievalRelease
from tests.test_package_coordinate_evolution import _bundle
from tests.test_package_stage_runner import _evaluation
from tests.test_package_task_feedback import _verified_state_and_result
from tests.test_task_evidence_feedback import _case


def _record(task_id: str, *, labeled: bool = True) -> dict[str, object]:
    return {
        "benchmark_id": task_id,
        "entity_name": f"entity_{task_id}",
        "target_name": "target",
        "frequency": "1 day",
        "prediction_length": 1,
        "history_values": [1.0, 2.0],
        "future_values": [3.0] if labeled else [None],
        "documents": [],
        "labels_public": labeled,
    }


def _context_task(task_id: str) -> ContextTask:
    return ContextTask(
        numeric=DataTask(
            task_id=task_id,
            history_values=(1.0, 2.0),
            future_values=(3.0,),
            prediction_length=1,
            frequency="D",
            seasonal_period="7",
            entity_name=f"entity_{task_id}",
        ),
        target_name="target",
        target_description="target",
        history_timestamps=("2026-01-01", "2026-01-02"),
        future_timestamps=("2026-01-03",),
        documents=(),
    )


def test_build_registry_thaws_frozen_numerical_anchor(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, state = _bundle(tmp_path)
    release = parse_numerical_supply_release(
        state.bundle.to_payload()["numerical_release_payload"]
    )
    package = state.registry.package_for(task)
    monkeypatch.setattr(
        "evolving_loop.run_package_coevolution.run_numerical_loop",
        lambda *_args, **_kwargs: package,
    )
    materializer = SimpleNamespace(
        screening_policy=None,
        forecast_store=SimpleNamespace(forecast=lambda *_args: ()),
        combined_policies=(),
        decision_policy=None,
        hindcast_config=None,
        source_fingerprints={},
        runtime_fingerprints={},
    )

    rebuilt = _build_registry((task,), release, materializer)

    assert rebuilt.release_sha256 == release.fingerprint
    assert (
        rebuilt.package_for(task).final_forecast == package.protected_baseline.forecast
    )


def test_smoke_numerical_proposer_thaws_frozen_anchor_before_fallback(tmp_path) -> None:
    task, state = _bundle(tmp_path)
    proposer = _SmokeNumericalProposer(
        SimpleNamespace(propose=lambda *_args, **_kwargs: ()),
        object(),
        (),
        object(),
        (task,),
    )

    candidates = proposer.propose(
        state,
        PackageProposalFeedback({}, (), (), ()),
        generation=0,
        child_count=1,
    )

    assert candidates[0].invalid_reason == "materialization_failed"


def test_step_payload_serializes_frozen_coordinate_trace(tmp_path) -> None:
    _task, state = _bundle(tmp_path)
    fingerprints = {
        "numerical": "1" * 64,
        "retrieval": "2" * 64,
        "decision": "3" * 64,
    }
    step = PackageCoordinateStep(
        generation=0,
        target="numerical",
        accepted=False,
        reason="kept parent",
        parent_fingerprints=fingerprints,
        child_fingerprints=fingerprints,
        accepted_fingerprints=fingerprints,
        changed_modules=("numerical",),
        parent_bytes_sha256="4" * 64,
        child_bytes_sha256="5" * 64,
        accepted_bytes_sha256="4" * 64,
        parent_registry_sha256="6" * 64,
        child_registry_sha256="7" * 64,
        accepted_registry_sha256="6" * 64,
    )

    payload = _step_payload(step, state)
    restored = json.loads(json.dumps(payload, sort_keys=True))

    assert restored["parent_fingerprints"] == fingerprints
    assert restored["child_fingerprints"] == fingerprints
    assert restored["accepted_fingerprints"] == fingerprints
    assert restored["changed_modules"] == ["numerical"]
    assert restored["accepted_bundle_payload"]["schema_version"] == 2


def test_state_from_payload_thaws_frozen_numerical_release(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, state = _bundle(tmp_path)
    monkeypatch.setattr(
        "evolving_loop.run_package_coevolution._build_registry",
        lambda *_args, **_kwargs: state.registry,
    )

    restored = _state_from_payload(
        state.bundle.to_payload(),
        tasks=(task,),
        materializer=object(),
        library=RetrievalSkillLibrary(tmp_path / "skills.json", persist=False),
    )

    assert restored.bundle.fingerprint() == state.bundle.fingerprint()
    assert restored.registry is state.registry


def _split_manifest() -> dict[str, object]:
    train_ids = [f"train_{index:03d}" for index in range(80)]
    dev_ids = [f"dev_{index:03d}" for index in range(20)]
    public_ids = [f"public_{index:03d}" for index in range(99)]
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset": "ServiceNow/Dr-CiK",
        "source_split": "public_dev",
        "seed": 20260816,
        "grouping": "entity_disjoint",
        "stratification_features": [
            "frequency",
            "horizon_bin",
            "reasoning_hops",
            "origin",
        ],
        "selection_uses_future_values": False,
        "selection_uses_gt_evidence": False,
        "selection_uses_document_labels": False,
        "target_sizes": {"train": 80, "dev": 20, "public_test": 99},
        "actual_sizes": {"train": 80, "dev": 20, "public_test": 99},
        "partitions": {
            "train": {"task_ids": train_ids, "entities": train_ids, "distribution": {}},
            "dev": {"task_ids": dev_ids, "entities": dev_ids, "distribution": {}},
            "public_test": {
                "task_ids": public_ids,
                "entities": public_ids,
                "distribution": {},
            },
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return payload


def test_parser_has_no_public_input_or_evaluation_flag() -> None:
    parser = build_parser()

    destinations = {action.dest for action in parser._actions}

    assert "public_tasks" not in destinations
    assert "evaluate_public" not in destinations
    assert "public_output" not in destinations


def test_interaction_smoke_requires_two_cycles_one_child_and_explicit_feedback_mode(
    tmp_path,
) -> None:
    required = [
        "--repo",
        str(tmp_path / "repo"),
        "--split-file",
        str(tmp_path / "split.json"),
        "--tasks-file",
        str(tmp_path / "tasks"),
        "--numerical-champion-release",
        str(tmp_path / "champion.json"),
        "--forecast-store",
        str(tmp_path / "forecasts"),
        "--retrieval-seed-release",
        str(tmp_path / "retrieval"),
        "--retrieval-skills",
        str(tmp_path / "skills.json"),
        "--output-dir",
        str(tmp_path / "output"),
        "--authority-dir",
        str(tmp_path / "authority"),
    ]
    treatment = build_parser().parse_args(
        [
            *required,
            "--interaction-smoke",
            "--children-per-coordinate",
            "1",
            "--feedback-mode",
            "task",
        ]
    )
    control = build_parser().parse_args(
        [
            *required,
            "--interaction-smoke",
            "--children-per-coordinate",
            "1",
            "--feedback-mode",
            "none",
        ]
    )

    _validate_mode(treatment)
    _validate_mode(control)
    assert _configuration_identity(treatment) != _configuration_identity(control)

    forbidden = build_parser().parse_args([*required, "--feedback-mode", "task"])
    with pytest.raises(ValueError, match="interaction"):
        _validate_mode(forbidden)


def test_interaction_feedback_manager_persists_cycle1_projection_for_cycle2(
    tmp_path,
) -> None:
    task, state, result = _verified_state_and_result(tmp_path)
    ledger = PackageTaskFeedbackLedger({task.numeric.task_id: "train"})
    ledger.record(state.bundle, task, result)
    store = PackageArtifactStore(tmp_path / "artifacts")
    manager = _InteractionFeedbackManager(
        ledger,
        (task.numeric.task_id,),
        store,
        feedback_mode="task",
    )

    manager.finish_cycle(state, generation=2)
    projection = manager.for_numerical(state, generation=3)

    assert projection is not None
    assert len(projection.cases) == 1
    assert store.load_task_feedback(3) == projection


def test_skipped_cycle1_decision_persists_explicit_empty_treatment_feedback(
    tmp_path,
) -> None:
    seed = RetrievalRelease.load(
        Path("evolving_loop/retrieval_agent/releases/v000")
    )
    policy = embed_retrieval_release(
        HarnessPolicy(), seed, changelog="Bind the Retrieval seed."
    )
    task, state = _bundle(tmp_path, policy=policy)
    ledger = PackageTaskFeedbackLedger({task.numeric.task_id: "train"})
    store = PackageArtifactStore(tmp_path / "artifacts")
    manager = _InteractionFeedbackManager(
        ledger,
        (task.numeric.task_id,),
        store,
        feedback_mode="task",
    )
    recorder = _CheckpointRecorder(
        store,
        run_sha256="a" * 64,
        schedule_sha256="b" * 64,
        initial=state,
        checkpoint=None,
    )

    class RejectingPhase:
        def __init__(self, target, feedback_manager=None) -> None:
            self.target = target
            self.feedback_manager = feedback_manager

        def run(self, parent, _initial, *, generation):
            return PackageCoordinatePhaseOutcome(
                target=self.target,
                parent=parent,
                finalist=None,
                selected=parent,
                accepted=False,
                improved=False,
                reason=f"rejected generation {generation}",
                evidence=(),
            )

    current = _run_controller(
        (
            RejectingPhase("numerical"),
            RejectingPhase("retrieval", manager),
            RejectingPhase("decision", manager),
        ),
        cycles=1,
        offset=0,
        parent=state,
        initial=state,
        recorder=recorder,
    )

    projection = manager.for_numerical(current, generation=3)
    assert projection is not None
    assert projection.cases == ()
    assert store.load_task_feedback(3) == projection
    assert tuple(step["target"] for step in recorder.steps) == (
        "numerical",
        "retrieval",
        "decision",
    )


def test_interaction_retrieval_gate_ignores_gain_but_keeps_integrity_failures() -> None:
    gain_only = (
        "minimum_relative_joint_gain",
        "mean_smae",
        "p95_srmse",
        "improving_folds",
    )
    assert _interaction_smoke_gate_failures("retrieval", gain_only) == ()
    assert _interaction_smoke_gate_failures(
        "retrieval", (*gain_only, "task_coverage", "fallback_count")
    ) == ("task_coverage", "fallback_count")
    assert _interaction_smoke_gate_failures("decision", gain_only) == gain_only


def test_interaction_completion_requires_six_invoked_phases_and_matching_feedback() -> (
    None
):
    steps = tuple(
        {
            "generation": generation,
            "target": target,
            "reason": "phase invoked",
            "public_test_accessed": False,
        }
        for generation, target in enumerate(("numerical", "retrieval", "decision") * 2)
    )
    treatment = TaskEvidenceProjection(
        source_bundle_sha256="a" * 64,
        request_namespace_sha256="b" * 64,
        cases=(_case(),),
    )
    control = TaskEvidenceProjection(
        source_bundle_sha256="a" * 64,
        request_namespace_sha256="b" * 64,
        cases=(),
    )

    assert _interaction_smoke_is_complete(steps, "task", treatment) is True
    assert _interaction_smoke_is_complete(steps, "none", control) is True
    assert _interaction_smoke_is_complete(steps[:-1], "task", treatment) is False
    skipped = (
        *steps[:2],
        {**steps[2], "reason": "Decision phase requires a release"},
        *steps[3:],
    )
    assert _interaction_smoke_is_complete(skipped, "task", treatment) is False


@pytest.mark.parametrize("resume_from", ("fresh", "partial_cycle", "complete_cycle"))
def test_interaction_runs_two_cycles_when_every_coordinate_is_rejected(
    tmp_path, resume_from
) -> None:
    _task, state = _bundle(tmp_path)
    recorder = _CheckpointRecorder(
        PackageArtifactStore(tmp_path / "artifacts"),
        run_sha256="a" * 64,
        schedule_sha256="b" * 64,
        initial=state,
        checkpoint=None,
    )

    class RejectingPhase:
        def __init__(self, target: str) -> None:
            self.target = target

        def run(self, parent, _initial, *, generation):
            return PackageCoordinatePhaseOutcome(
                target=self.target,
                parent=parent,
                finalist=None,
                selected=parent,
                accepted=False,
                improved=False,
                reason=f"rejected generation {generation}",
                evidence=(),
            )

    phases = tuple(
        RejectingPhase(target) for target in ("numerical", "retrieval", "decision")
    )
    current = state
    if resume_from == "partial_cycle":
        current = _run_controller(
            (phases[0], None, None),
            cycles=1,
            offset=0,
            parent=current,
            initial=state,
            recorder=recorder,
        )
        assert len(recorder.steps) == 1
    elif resume_from == "complete_cycle":
        current = _resume_controller(
            SimpleNamespace(cycles=1, interaction_smoke=True),
            phases=phases,
            initial=state,
            current=current,
            recorder=recorder,
        )
        assert len(recorder.steps) == 3

    _resume_controller(
        SimpleNamespace(cycles=2, interaction_smoke=True),
        phases=phases,
        initial=state,
        current=current,
        recorder=recorder,
    )

    assert tuple(step["generation"] for step in recorder.steps) == tuple(range(6))
    assert tuple(step["target"] for step in recorder.steps) == (
        "numerical",
        "retrieval",
        "decision",
    ) * 2


def test_smoke_numerical_proposer_forwards_cycle_feedback(tmp_path) -> None:
    _task, state = _bundle(tmp_path)
    projection = TaskEvidenceProjection(
        source_bundle_sha256=state.bundle.fingerprint(),
        request_namespace_sha256="b" * 64,
        cases=(_case(),),
    )
    feedback = PackageProposalFeedback(
        parent_summary={"task_count": 1},
        rejected_summaries=(),
        gate_names=(),
        structures=(),
        task_evidence=projection,
    )

    class CapturingChampion:
        def __init__(self) -> None:
            self.task_evidence = None

        def propose(self, _anchor, _aggregate, *, generation, task_evidence=None):
            self.task_evidence = task_evidence
            return ()

    champion = CapturingChampion()
    proposer = _SmokeNumericalProposer(
        champion,
        SimpleNamespace(),
        (),
        SimpleNamespace(task_fold_map={}),
        (),
    )

    children = proposer.propose(state, feedback, generation=3, child_count=1)

    assert children[0].invalid_reason == "materialization_failed"
    assert champion.task_evidence is projection


def test_runner_loads_only_train_and_dev_membership(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _split_manifest()
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    seen_ids: list[str] = []
    executed: list[tuple[str, ...]] = []

    def load_only_requested(_path, ids):
        seen_ids.extend(ids)
        return tuple(_context_task(task_id) for task_id in ids)

    monkeypatch.setattr(
        "evolving_loop.run_package_coevolution.load_context_tasks_by_ids",
        load_only_requested,
    )
    monkeypatch.setattr(
        "evolving_loop.run_package_coevolution._execute_run",
        lambda _args, _split, tasks: executed.append(
            tuple(task.numeric.task_id for task in tasks)
        )
        or 0,
    )

    required = {
        name: tmp_path / name
        for name in (
            "tasks",
            "champion",
            "forecasts",
            "retrieval",
            "skills",
            "output",
            "authority",
            "repo",
        )
    }
    argv = [
        "--repo",
        str(required["repo"]),
        "--split-file",
        str(split_path),
        "--tasks-file",
        str(required["tasks"]),
        "--numerical-champion-release",
        str(required["champion"]),
        "--forecast-store",
        str(required["forecasts"]),
        "--retrieval-seed-release",
        str(required["retrieval"]),
        "--retrieval-skills",
        str(required["skills"]),
        "--output-dir",
        str(required["output"]),
        "--authority-dir",
        str(required["authority"]),
    ]

    assert main(argv) == 0
    train_ids = set(split["partitions"]["train"]["task_ids"])
    dev_ids = set(split["partitions"]["dev"]["task_ids"])
    public_ids = set(split["partitions"]["public_test"]["task_ids"])
    assert set(seen_ids) == train_ids | dev_ids
    assert public_ids.isdisjoint(seen_ids)
    assert set(executed[0]) == train_ids | dev_ids


def test_id_filtered_loader_returns_requested_order_without_constructing_public(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(_record(task_id))
            for task_id in ("train_b", "public_secret", "train_a")
        )
        + "\n",
        encoding="utf-8",
    )
    converted: list[str] = []
    from evolving_loop import data

    original = data._to_context_task

    def record_conversion(payload):
        converted.append(payload["benchmark_id"])
        return original(payload)

    monkeypatch.setattr(data, "_to_context_task", record_conversion)

    tasks = load_context_tasks_by_ids(path, ("train_a", "train_b"))

    assert tuple(task.numeric.task_id for task in tasks) == ("train_a", "train_b")
    assert converted == ["train_a", "train_b"]


@pytest.mark.parametrize("ids", [(), ("task_a", "task_a"), ("",)])
def test_id_filtered_loader_rejects_invalid_requested_membership(tmp_path, ids) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(_record("task_a")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requested task IDs"):
        load_context_tasks_by_ids(path, ids)


def test_id_filtered_loader_rejects_missing_duplicate_unlabeled_or_unexpected_records(
    tmp_path,
) -> None:
    missing = tmp_path / "missing.jsonl"
    missing.write_text(json.dumps(_record("task_a")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_context_tasks_by_ids(missing, ("task_b",))

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        json.dumps(_record("task_a")) + "\n" + json.dumps(_record("task_a")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_context_tasks_by_ids(duplicate, ("task_a",))

    unlabeled = tmp_path / "unlabeled.jsonl"
    unlabeled.write_text(
        json.dumps(_record("task_a", labeled=False)) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unlabeled"):
        load_context_tasks_by_ids(unlabeled, ("task_a",))

    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "task_a.json").write_text(
        json.dumps(_record("different_id")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unexpected"):
        load_context_tasks_by_ids(directory, ("task_a",))


class _CountingEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, bundle, _registry, tasks, *, stage, cache_only=False):
        assert cache_only is False
        self.calls += 1
        return _evaluation(
            bundle.fingerprint(),
            tuple(task.numeric.task_id for task in tasks),
            error=0.5,
        )


def test_cache_backed_evaluator_replays_exact_bundle_stage_task_set_and_runtime(
    tmp_path,
) -> None:
    task, state = _bundle(tmp_path)
    delegate = _CountingEvaluator()
    evaluator = PackageCacheBackedEvaluator(
        delegate,
        tmp_path / "cache",
        runtime_fingerprints=state.bundle.runtime_fingerprints,
    )

    live = evaluator.evaluate(state.bundle, state.registry, (task,), stage="dev20")
    replay = evaluator.evaluate(
        state.bundle,
        state.registry,
        (task,),
        stage="dev20",
        cache_only=True,
    )

    assert delegate.calls == 1
    assert replay == live
    with pytest.raises(PackageCacheMissError, match="cache-only"):
        evaluator.evaluate(
            state.bundle,
            state.registry,
            (task,),
            stage="calibration16",
            cache_only=True,
        )

    changed_runtime = {
        **dict(state.bundle.runtime_fingerprints),
        "llm_runtime": "9" * 64,
    }
    other = PackageCacheBackedEvaluator(
        delegate,
        tmp_path / "cache",
        runtime_fingerprints=changed_runtime,
    )
    with pytest.raises(PackageCacheMissError, match="cache-only"):
        other.evaluate(
            state.bundle,
            state.registry,
            (task,),
            stage="dev20",
            cache_only=True,
        )


def test_cache_backed_evaluator_promotes_published_bundle_without_live_call(
    tmp_path,
) -> None:
    task, state = _bundle(tmp_path)
    delegate = _CountingEvaluator()
    evaluator = PackageCacheBackedEvaluator(
        delegate,
        tmp_path / "cache",
        runtime_fingerprints=state.bundle.runtime_fingerprints,
    )
    live = evaluator.evaluate(state.bundle, state.registry, (task,), stage="dev20")
    published = state.with_policy(
        replace(
            state.bundle.policy,
            decision_prompt=state.bundle.policy.decision_prompt + " :: published",
        ),
        target="decision",
    )

    evaluator.promote_cached_bundle(
        state.bundle,
        published.bundle,
        state.registry,
        (task,),
        stage="dev20",
    )
    replay = evaluator.evaluate(
        published.bundle,
        published.registry,
        (task,),
        stage="dev20",
        cache_only=True,
    )

    assert delegate.calls == 1
    assert replay.candidate_sha256 == published.bundle.fingerprint()
    assert replay.result_bytes() == live.result_bytes()


def test_completion_records_formality_and_coordinate_counts(tmp_path) -> None:
    store = PackageArtifactStore(tmp_path)

    store.complete(
        {"schema_version": 2, "final": True},
        accepted_steps=2,
        rejected_steps=1,
        formal_run=False,
    )

    completion = json.loads(
        (tmp_path / "evaluation_complete.json").read_text(encoding="utf-8")
    )
    assert completion["accepted_steps"] == 2
    assert completion["rejected_steps"] == 1
    assert completion["formal_run"] is False
    assert completion["public_test_accessed"] is False


def test_resume_rejects_runtime_fingerprint_change(tmp_path) -> None:
    parser = build_parser()
    common = [
        "--repo",
        str(tmp_path / "repo"),
        "--split-file",
        str(tmp_path / "split.json"),
        "--tasks-file",
        str(tmp_path / "tasks"),
        "--numerical-champion-release",
        str(tmp_path / "champion.json"),
        "--forecast-store",
        str(tmp_path / "forecasts"),
        "--retrieval-seed-release",
        str(tmp_path / "retrieval"),
        "--retrieval-skills",
        str(tmp_path / "skills.json"),
        "--output-dir",
        str(tmp_path / "output"),
        "--authority-dir",
        str(tmp_path / "authority"),
    ]
    original = parser.parse_args(common)
    output = tmp_path / "output"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps({"configuration_sha256": _configuration_identity(original)}),
        encoding="utf-8",
    )
    resumed = parser.parse_args([*common, "--resume", "--model", "different-model"])

    with pytest.raises(PackageArtifactError, match="runtime"):
        _early_resume_guard(resumed)


def test_resume_of_complete_run_stops_before_model_construction(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--repo",
            str(tmp_path / "repo"),
            "--split-file",
            str(tmp_path / "split.json"),
            "--tasks-file",
            str(tmp_path / "tasks"),
            "--numerical-champion-release",
            str(tmp_path / "champion.json"),
            "--forecast-store",
            str(tmp_path / "forecasts"),
            "--retrieval-seed-release",
            str(tmp_path / "retrieval"),
            "--retrieval-skills",
            str(tmp_path / "skills.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--authority-dir",
            str(tmp_path / "authority"),
            "--resume",
        ]
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps({"configuration_sha256": _configuration_identity(args)}),
        encoding="utf-8",
    )
    (output / "evaluation_complete.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (output / "final_bundle.json").write_text("{}", encoding="utf-8")

    assert _early_resume_guard(args) is True
