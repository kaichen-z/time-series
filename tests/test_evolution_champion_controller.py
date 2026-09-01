"""Build-only Champion evolution controller regressions."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace

import pytest

import numerical_agent.evolution.champion_controller as controller_module
from common.data import Task
from common.payload import canonical_json_bytes
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
    champion_fingerprint,
)
from numerical_agent.evolution.champion_controller import (
    ChampionArtifactStore,
    ChampionCheckpoint,
    ChampionControllerError,
    ChampionEvolutionController,
    ChampionEvolutionConfig,
    ChampionLifecycleError,
    ChampionRunManifest,
    canonical_release_bytes,
    partition_train_tasks,
    run_build_evolution,
)
from numerical_agent.evolution.champion_evidence import (
    ChampionGateConfig,
    ChampionHistoryDiagnostic,
    ChampionTaskRow,
    ProposerEvidence,
)
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.screening import TaskProfile


def _profile(task_id: str, index: int) -> TaskProfile:
    return TaskProfile(
        task_id=task_id,
        frequency="D",
        history_length=100 + index,
        horizon=2,
        zero_fraction=0.0,
        signed=False,
        integer_valued=False,
        trend_direction="up",
        trend_strength=0.5,
        periodicity_periods=(7,),
        periodicity_strength=float(index % 8) / 8.0,
        periodicity_confidence=0.75,
        outlier_fraction=0.0,
        noise_relative_scale=0.2,
        likely_stationary=False,
        stationarity_score=0.25,
        recent_regime_start=None,
        recent_regime_confidence=0.1,
        intermittency_adi=1.0,
        intermittency_cv2=0.1,
    )


def _parent() -> ChampionRelease:
    recipe = ChampionRecipe(
        name="active_parent",
        kind="select",
        parents=("baseline_leaf",),
        fallback_parent="baseline_leaf",
        assumptions=(
            EvolutionAssumption(
                assumption_id="parent_history",
                candidate_name="baseline_leaf",
                feature="history_length",
                direction="above",
                horizon_region="full",
                operator="select",
                rationale="History length supports the baseline forecast.",
                failure_condition="History length no longer supports the baseline forecast.",
            ),
        ),
    )
    return ChampionRelease(
        policy=FittedChampionPolicy(
            recipe=recipe,
            thresholds=(("parent_history", 0.0),),
        ),
        source_hashes=(("baseline_leaf", "0" * 64),),
        metric_policy_fingerprint="1" * 64,
        lineage=("active_parent",),
    )


def _history_diagnostic(
    name: str, *, eligible: bool = True
) -> ChampionHistoryDiagnostic:
    return ChampionHistoryDiagnostic.from_candidate(
        CandidateDiagnostics.synthetic(
            name=name,
            family="statistical",
            median_mase=0.5,
            eligible=eligible,
        )
    )


def _rows(
    count: int = 64, *, enriched: bool = False
) -> tuple[ChampionTaskRow, ...]:
    errors = {
        "baseline_leaf": 2.0,
        "better_a": 0.8,
        "better_b": 0.9,
        "better_c": 1.0,
        "better_d": 1.1,
        "better_e": 1.2,
        "worse_a": 2.8,
        "worse_b": 2.9,
        "worse_c": 3.0,
        "worse_d": 3.1,
        "worse_e": 3.2,
    }
    rows: list[ChampionTaskRow] = []
    for index in range(count):
        task_id = f"build_case_{index:03d}"
        profile = _profile(task_id, index)
        truth = (10.0 + index, 12.0 + index)
        history = (10.0,) * profile.history_length if enriched else None
        for candidate_name, error in errors.items():
            rows.append(
                ChampionTaskRow(
                    task_id=task_id,
                    candidate_name=candidate_name,
                    profile=profile,
                    truth=truth,
                    forecast=tuple(value + error for value in truth),
                    fold=index % 5,
                    split="build",
                    history=history,
                    diagnostic=(
                        _history_diagnostic(candidate_name) if enriched else None
                    ),
                )
            )
    return tuple(rows)


def _config(*, generations: int = 1) -> ChampionEvolutionConfig:
    task_ids = tuple(f"build_case_{index:03d}" for index in range(64))
    return ChampionEvolutionConfig(
        generations=generations,
        screen_task_ids=(task_ids[:8], task_ids[:32], task_ids),
        gate_config=ChampionGateConfig(minimum_improved_folds=0),
    )


def _proposal_batch(prefix: str, generation: int = 0) -> tuple[ChampionRecipe, ...]:
    return tuple(
        ChampionRecipe(
            name=f"proposal_{generation}_{index}",
            kind="select",
            parents=(f"{prefix}_{letter}",),
            fallback_parent=f"{prefix}_{letter}",
            assumptions=(
                EvolutionAssumption(
                    assumption_id=f"history_{generation}_{index}",
                    candidate_name=f"{prefix}_{letter}",
                    feature="history_length",
                    direction="above",
                    horizon_region="full",
                    operator="select",
                    rationale="History length supports the candidate forecast.",
                    failure_condition="History length stops supporting the candidate forecast.",
                ),
            ),
        )
        for index, letter in enumerate("abcde")
    )


@dataclass
class RecordingProposer:
    prefix: str
    calls: int = 0

    def __call__(
        self,
        parent: ChampionRelease,
        evidence: ProposerEvidence,
    ) -> tuple[ChampionRecipe, ...]:
        assert parent is PARENT
        assert evidence.label == "adaptive_train_build_diagnostic"
        generation = self.calls
        self.calls += 1
        return _proposal_batch(self.prefix, generation)


@dataclass
class DiverseProposer:
    def __call__(
        self,
        parent: ChampionRelease,
        evidence: ProposerEvidence,
    ) -> tuple[ChampionRecipe, ...]:
        assert parent is PARENT

        def assumption(index: int, kind: str, candidate: str) -> EvolutionAssumption:
            return EvolutionAssumption(
                assumption_id=f"diverse_history_{index}",
                candidate_name=candidate,
                feature="history_length",
                direction="above",
                horizon_region="full",
                operator=kind,  # type: ignore[arg-type]
                rationale="History length supports the candidate forecast.",
                failure_condition="History length stops supporting the candidate forecast.",
            )

        recipes: list[ChampionRecipe] = []
        for index, letter in enumerate("abc"):
            candidate = f"better_{letter}"
            recipes.append(
                ChampionRecipe(
                    name=f"select_proposal_{index}",
                    kind="select",
                    parents=(candidate,),
                    fallback_parent=candidate,
                    assumptions=(assumption(index, "select", candidate),),
                )
            )
        for index, kind in enumerate(("weighted", "median"), start=3):
            recipes.append(
                ChampionRecipe(
                    name=f"{kind}_proposal",
                    kind=kind,  # type: ignore[arg-type]
                    parents=("better_d", "better_e"),
                    fallback_parent="better_d",
                    assumptions=(assumption(index, kind, "better_d"),),
                )
            )
        return tuple(recipes)


@dataclass
class AllOperatorProposer:
    def __call__(
        self,
        parent: ChampionRelease,
        evidence: ProposerEvidence,
    ) -> tuple[ChampionRecipe, ...]:
        kinds = (
            "select",
            "route",
            "horizon_route",
            "weighted",
            "median",
            "bounded_overlay",
        )
        recipes: list[ChampionRecipe] = []
        for index, kind in enumerate(kinds):
            parents = ("better_a",) if kind == "select" else (
                "baseline_leaf",
                "better_a",
            )
            recipes.append(
                ChampionRecipe(
                    name=f"all_operator_{kind}",
                    kind=kind,  # type: ignore[arg-type]
                    parents=parents,
                    fallback_parent=parents[0],
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"operator_history_{index}",
                            candidate_name=parents[-1],
                            feature="periodicity_confidence",
                            direction="above",
                            horizon_region="full",
                            operator=kind,  # type: ignore[arg-type]
                            rationale="History diagnostics support this operator.",
                            failure_condition="History diagnostics stop supporting this operator.",
                        ),
                    ),
                )
            )
        return tuple(recipes)


PARENT = _parent()
ROWS_64 = _rows()


def _task(task_id: str, entity_name: str) -> Task:
    return Task(
        task_id=task_id,
        history_values=(1.0, 2.0, 3.0),
        future_values=(4.0,),
        prediction_length=1,
        frequency="D",
        seasonal_period=None,
        entity_name=entity_name,
    )


TRAIN_80 = tuple(
    _task(f"train_{index:03d}", f"entity_{index // 4:03d}")
    for index in range(80)
)
DEV_20 = tuple(
    _task(f"dev_{index:03d}", f"dev_entity_{index:03d}")
    for index in range(20)
)


def _lifecycle_config(parts) -> ChampionEvolutionConfig:
    build_ids = tuple(task.task_id for task in parts.build)
    return ChampionEvolutionConfig(
        build_task_ids=build_ids,
        screen_task_ids=(build_ids[:8], build_ids[:32], build_ids),
        gate_config=ChampionGateConfig(minimum_improved_folds=0),
    )


def _manifest(parts, config: ChampionEvolutionConfig) -> ChampionRunManifest:
    return ChampionRunManifest(
        schema_version=1,
        partition_seed=20260901,
        source_hashes=(("dev_tasks", "b" * 64), ("train_tasks", "a" * 64)),
        train_tasks=tuple(
            (task.task_id, task.entity_name) for task in TRAIN_80
        ),
        dev_tasks=tuple((task.task_id, task.entity_name) for task in DEV_20),
        split_manifest_fingerprint="c" * 64,
        build_tasks=tuple(
            (task.task_id, task.entity_name) for task in parts.build
        ),
        calibration_tasks=tuple(
            (task.task_id, task.entity_name) for task in parts.calibration
        ),
        dictionary_hashes=PARENT.source_hashes,
        forecast_store_fingerprint="d" * 64,
        metric_policy_fingerprint=PARENT.metric_policy_fingerprint,
        proposal_model="deterministic-test-proposer",
        proposal_config_fingerprint="e" * 64,
        schedule_fingerprint=config.fingerprint,
        numeric_grid_fingerprint="f" * 64,
        candidate_minimum_gain=config.candidate_minimum_gain,
        research_target_gain=config.research_target_gain,
        runtime_fingerprint="9" * 64,
    )


@dataclass
class LifecycleRows:
    calibration_error: float = 0.5
    dev_error: float = 0.5

    def __post_init__(self) -> None:
        self.opens: list[str] = []

    def __call__(
        self, tasks: tuple[Task, ...], split: str
    ) -> tuple[ChampionTaskRow, ...]:
        self.opens.append(split)
        error = {
            "build": 0.5,
            "calibration": self.calibration_error,
            "dev": self.dev_error,
        }[split]
        rows: list[ChampionTaskRow] = []
        for index, task in enumerate(tasks):
            profile = replace(
                _profile(task.task_id, index),
                history_length=len(task.history_values),
                horizon=task.prediction_length,
            )
            for candidate_name, candidate_error in {
                "baseline_leaf": 2.0,
                "better_a": error,
                "better_b": error + 0.01,
                "better_c": error + 0.02,
                "better_d": error + 0.03,
                "better_e": error + 0.04,
            }.items():
                rows.append(
                    ChampionTaskRow(
                        task_id=task.task_id,
                        candidate_name=candidate_name,
                        profile=profile,
                        truth=task.future_values,
                        forecast=tuple(
                            value + candidate_error for value in task.future_values
                        ),
                        fold=index % 5,
                        split=split,  # type: ignore[arg-type]
                        history=task.history_values,
                        diagnostic=_history_diagnostic(candidate_name),
                    )
                )
        return tuple(rows)


@dataclass
class SerializedRecordingProposer:
    serialized_inputs: str = ""

    def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
        self.serialized_inputs += repr((parent, evidence))
        return _proposal_batch("better")


def _lifecycle_controller(tmp_path, proposer, rows) -> ChampionEvolutionController:
    parts = partition_train_tasks(
        TRAIN_80,
        build_size=64,
        calibration_size=16,
        seed=20260901,
    )
    config = _lifecycle_config(parts)
    return ChampionEvolutionController(
        manifest=_manifest(parts, config),
        config=config,
        proposer=proposer,
        row_provider=rows,
        artifact_store=ChampionArtifactStore(tmp_path),
    )


def test_internal_partition_is_exact_deterministic_and_entity_disjoint() -> None:
    first = partition_train_tasks(
        TRAIN_80,
        build_size=64,
        calibration_size=16,
        seed=20260901,
    )
    second = partition_train_tasks(
        tuple(reversed(TRAIN_80)),
        build_size=64,
        calibration_size=16,
        seed=20260901,
    )

    assert len(first.build) == 64
    assert len(first.calibration) == 16
    assert {task.entity_name for task in first.build}.isdisjoint(
        {task.entity_name for task in first.calibration}
    )
    assert tuple(task.task_id for task in first.build) == tuple(
        task.task_id for task in second.build
    )
    assert tuple(task.task_id for task in first.calibration) == tuple(
        task.task_id for task in second.calibration
    )


def test_partition_fails_when_entity_groups_cannot_make_exact_sizes() -> None:
    tasks = (
        _task("large_1", "large"),
        _task("large_2", "large"),
        _task("small_1", "small"),
        _task("small_2", "small"),
    )

    with pytest.raises(ChampionLifecycleError, match="entity-disjoint"):
        partition_train_tasks(tasks, build_size=3, calibration_size=1, seed=1)


@pytest.mark.parametrize(
    "tasks, message",
    (
        (
            (
                _task("duplicate", "first"),
                _task("duplicate", "second"),
            ),
            "duplicate task IDs",
        ),
        (
            (
                _task("first", "known"),
                _task("second", "unknown"),
            ),
            "unknown entity",
        ),
        (
            (
                _task("first", "Model_K"),
                _task("second", "Model_K"),
            ),
            "duplicate entity",
        ),
    ),
)
def test_partition_rejects_duplicate_and_unknown_identities(tasks, message) -> None:
    with pytest.raises(ChampionLifecycleError, match=message):
        partition_train_tasks(tasks, build_size=1, calibration_size=1, seed=1)


def test_manifest_fingerprint_binds_every_registered_input(tmp_path) -> None:
    parts = partition_train_tasks(
        TRAIN_80,
        build_size=64,
        calibration_size=16,
        seed=20260901,
    )
    config = _lifecycle_config(parts)
    manifest = _manifest(parts, config)
    variants = (
        replace(
            manifest,
            source_hashes=(
                ("dev_tasks", "b" * 64),
                ("train_tasks", "0" * 64),
            ),
        ),
        replace(manifest, train_tasks=tuple(reversed(manifest.train_tasks))),
        replace(manifest, dev_tasks=tuple(reversed(manifest.dev_tasks))),
        replace(manifest, split_manifest_fingerprint="0" * 64),
        replace(manifest, build_tasks=tuple(reversed(manifest.build_tasks))),
        replace(
            manifest,
            calibration_tasks=tuple(reversed(manifest.calibration_tasks)),
        ),
        replace(manifest, dictionary_hashes=(("baseline_leaf", "2" * 64),)),
        replace(manifest, forecast_store_fingerprint="0" * 64),
        replace(manifest, metric_policy_fingerprint="0" * 64),
        replace(manifest, proposal_model="different-model"),
        replace(manifest, proposal_config_fingerprint="0" * 64),
        replace(manifest, schedule_fingerprint="0" * 64),
        replace(manifest, numeric_grid_fingerprint="0" * 64),
        replace(manifest, candidate_minimum_gain=0.006),
        replace(manifest, research_target_gain=0.051),
        replace(manifest, runtime_fingerprint="0" * 64),
        replace(manifest, partition_seed=20260902),
    )

    assert all(
        variant.input_fingerprint != manifest.input_fingerprint
        for variant in variants
    )
    for index, variant in enumerate(variants):
        store = ChampionArtifactStore(tmp_path / f"variant_{index:02d}")
        store.bind_manifest(manifest)
        with pytest.raises(ChampionLifecycleError, match="manifest|drift"):
            store.bind_manifest(variant)


def test_checkpoint_keeps_exact_frozen_sanitized_feedback() -> None:
    feedback = ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=(),
        comparisons=(),
    )
    checkpoint = ChampionCheckpoint(
        schema_version=1,
        input_fingerprint="1" * 64,
        completed_stage="build_generation_1",
        active_parent=PARENT,
        proposal_archive=(),
        sanitized_feedback=feedback,
    )

    assert checkpoint.sanitized_feedback is feedback


def test_calibration_is_never_returned_to_the_proposer(tmp_path) -> None:
    proposer = SerializedRecordingProposer()

    outcome = _lifecycle_controller(
        tmp_path, proposer, LifecycleRows()
    ).evolve(PARENT, TRAIN_80, DEV_20)
    parts = partition_train_tasks(
        TRAIN_80,
        build_size=64,
        calibration_size=16,
        seed=20260901,
    )

    assert outcome.release is not PARENT
    assert not any(
        task.task_id in proposer.serialized_inputs for task in parts.calibration
    )
    assert "calibration" not in proposer.serialized_inputs.lower()


def test_calibration_rejection_keeps_parent_and_never_reads_dev(tmp_path) -> None:
    class ExplodingSequence:
        def __iter__(self):
            raise AssertionError("Dev must remain unopened")

        def __len__(self):
            raise AssertionError("Dev must remain unopened")

    rows = LifecycleRows(calibration_error=3.0)
    outcome = _lifecycle_controller(
        tmp_path, SerializedRecordingProposer(), rows
    ).evolve(PARENT, TRAIN_80, ExplodingSequence())

    assert outcome.release is PARENT
    assert outcome.dev_report is None
    assert rows.opens == ["build", "calibration"]


def test_dev_rejection_preserves_prior_release_bytes(tmp_path) -> None:
    rows = LifecycleRows(dev_error=3.0)
    outcome = _lifecycle_controller(
        tmp_path, SerializedRecordingProposer(), rows
    ).evolve(PARENT, TRAIN_80, DEV_20)

    assert outcome.release is PARENT
    assert canonical_release_bytes(outcome.release) == canonical_release_bytes(PARENT)
    assert (tmp_path / "champion_release.json").read_bytes() == canonical_release_bytes(
        PARENT
    )
    assert outcome.dev_report is not None


def test_manifest_drift_fails_before_task_or_model_execution(tmp_path) -> None:
    rows = LifecycleRows()
    proposer = SerializedRecordingProposer()
    controller = _lifecycle_controller(tmp_path, proposer, rows)
    controller.artifact_store.bind_manifest(controller.manifest)
    payload = json.loads((tmp_path / "run_manifest.json").read_text())
    payload["runtime_fingerprint"] = "8" * 64
    (tmp_path / "run_manifest.json").write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ChampionLifecycleError, match="manifest|drift"):
        controller.evolve(PARENT, TRAIN_80, DEV_20)

    assert rows.opens == []
    assert proposer.serialized_inputs == ""


def test_valid_checkpoint_resume_skips_build_and_model_execution(tmp_path) -> None:
    class InterruptAtCalibration:
        def __init__(self) -> None:
            self.delegate = LifecycleRows()

        def __call__(self, tasks, split):
            if split == "calibration":
                raise RuntimeError("simulated interruption")
            return self.delegate(tasks, split)

    first_proposer = SerializedRecordingProposer()
    with pytest.raises(ChampionLifecycleError, match="calibration row provider failed"):
        _lifecycle_controller(
            tmp_path, first_proposer, InterruptAtCalibration()
        ).evolve(PARENT, TRAIN_80, DEV_20)

    resumed_proposer = SerializedRecordingProposer()
    resumed_rows = LifecycleRows()
    outcome = _lifecycle_controller(
        tmp_path, resumed_proposer, resumed_rows
    ).evolve(PARENT, TRAIN_80, DEV_20)

    assert outcome.build_result is None
    assert resumed_proposer.serialized_inputs == ""
    assert resumed_rows.opens == ["calibration", "dev"]


def test_foreign_checkpoint_fails_before_task_or_model_execution(tmp_path) -> None:
    class InterruptAtCalibration:
        def __init__(self) -> None:
            self.delegate = LifecycleRows()

        def __call__(self, tasks, split):
            if split == "calibration":
                raise RuntimeError("simulated interruption")
            return self.delegate(tasks, split)

    with pytest.raises(ChampionLifecycleError, match="calibration row provider failed"):
        _lifecycle_controller(
            tmp_path, SerializedRecordingProposer(), InterruptAtCalibration()
        ).evolve(PARENT, TRAIN_80, DEV_20)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["input_fingerprint"] = "7" * 64
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint))
    rows = LifecycleRows()
    proposer = SerializedRecordingProposer()

    with pytest.raises(ChampionLifecycleError, match="stale|foreign"):
        _lifecycle_controller(tmp_path, proposer, rows).evolve(
            PARENT, TRAIN_80, DEV_20
        )

    assert rows.opens == []
    assert proposer.serialized_inputs == ""


def test_duplicate_key_checkpoint_json_fails_before_callbacks(tmp_path) -> None:
    controller = _lifecycle_controller(
        tmp_path, SerializedRecordingProposer(), LifecycleRows()
    )
    controller.artifact_store.bind_manifest(controller.manifest)
    controller.artifact_store.ensure_release(PARENT)
    (tmp_path / "checkpoint.json").write_text(
        '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
    )
    rows = LifecycleRows()
    proposer = SerializedRecordingProposer()

    with pytest.raises(ChampionLifecycleError, match="malformed JSON"):
        _lifecycle_controller(tmp_path, proposer, rows).evolve(
            PARENT, TRAIN_80, DEV_20
        )

    assert rows.opens == []
    assert proposer.serialized_inputs == ""


def test_checkpoint_rejects_nested_cached_pass_authority_before_callbacks(
    tmp_path,
) -> None:
    class InterruptAtCalibration:
        def __init__(self) -> None:
            self.delegate = LifecycleRows()

        def __call__(self, tasks, split):
            if split == "calibration":
                raise RuntimeError("simulated interruption")
            return self.delegate(tasks, split)

    with pytest.raises(ChampionLifecycleError, match="calibration row provider failed"):
        _lifecycle_controller(
            tmp_path, SerializedRecordingProposer(), InterruptAtCalibration()
        ).evolve(PARENT, TRAIN_80, DEV_20)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    comparisons = checkpoint["sanitized_feedback"]["comparisons"]
    assert comparisons
    comparisons[0]["passed"] = True
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint))
    rows = LifecycleRows()
    proposer = SerializedRecordingProposer()

    with pytest.raises(ChampionLifecycleError, match="feedback|malformed|forbidden"):
        _lifecycle_controller(tmp_path, proposer, rows).evolve(
            PARENT, TRAIN_80, DEV_20
        )

    assert rows.opens == []
    assert proposer.serialized_inputs == ""


def test_stored_calibration_pass_cannot_open_dev(monkeypatch, tmp_path) -> None:
    class InterruptAtCalibration:
        def __init__(self) -> None:
            self.delegate = LifecycleRows()

        def __call__(self, tasks, split):
            if split == "calibration":
                raise RuntimeError("simulated interruption")
            return self.delegate(tasks, split)

    with pytest.raises(ChampionLifecycleError, match="calibration row provider failed"):
        _lifecycle_controller(
            tmp_path, SerializedRecordingProposer(), InterruptAtCalibration()
        ).evolve(PARENT, TRAIN_80, DEV_20)
    compare = controller_module.compare_champion
    calibration_calls = 0

    def changing_comparison(parent, child, gate):
        nonlocal calibration_calls
        result = compare(parent, child, gate)
        if child.policy_name.startswith("calibration_child_"):
            calibration_calls += 1
            if calibration_calls > 3:
                return replace(
                    result,
                    accepted=False,
                    failures=("primary_smae_regression",),
                )
        return result

    monkeypatch.setattr(controller_module, "compare_champion", changing_comparison)
    rows = LifecycleRows()
    outcome = _lifecycle_controller(
        tmp_path, SerializedRecordingProposer(), rows
    ).evolve(PARENT, TRAIN_80, DEV_20)

    assert outcome.calibration_report is not None
    assert all(
        candidate.comparison is not None and candidate.comparison.accepted
        for candidate in outcome.calibration_report.candidates
    )
    assert outcome.release is PARENT
    assert outcome.dev_report is None
    assert rows.opens == ["calibration"]


def test_stored_dev_pass_cannot_publish_release(monkeypatch, tmp_path) -> None:
    compare = controller_module.compare_champion
    dev_calls = 0

    def changing_comparison(parent, child, gate):
        nonlocal dev_calls
        result = compare(parent, child, gate)
        if child.policy_name.startswith("dev_child_"):
            dev_calls += 1
            if dev_calls > 1:
                return replace(
                    result,
                    accepted=False,
                    failures=("primary_smae_regression",),
                )
        return result

    monkeypatch.setattr(controller_module, "compare_champion", changing_comparison)
    outcome = _lifecycle_controller(
        tmp_path, SerializedRecordingProposer(), LifecycleRows()
    ).evolve(PARENT, TRAIN_80, DEV_20)

    assert outcome.dev_report is not None
    comparison = outcome.dev_report.candidates[0].comparison
    assert comparison is not None and comparison.accepted
    assert outcome.release is PARENT
    assert (tmp_path / "champion_release.json").read_bytes() == canonical_release_bytes(
        PARENT
    )


def test_release_rejects_normalized_lineage_replay(tmp_path) -> None:
    parent = replace(PARENT, lineage=("active_parent", "Model_K"))

    class LineageCollisionProposer:
        def __call__(self, received, evidence) -> tuple[ChampionRecipe, ...]:
            proposals = list(_proposal_batch("better"))
            proposals[0] = replace(proposals[0], name="Model_K")
            return tuple(proposals)

    controller = _lifecycle_controller(
        tmp_path, LineageCollisionProposer(), LifecycleRows()
    )

    with pytest.raises(ChampionLifecycleError, match="lineage|replayed"):
        controller.evolve(parent, TRAIN_80, DEV_20)

    assert (tmp_path / "champion_release.json").read_bytes() == canonical_release_bytes(
        parent
    )


def test_completed_lifecycle_reports_cannot_be_replayed(tmp_path) -> None:
    _lifecycle_controller(
        tmp_path, SerializedRecordingProposer(), LifecycleRows()
    ).evolve(PARENT, TRAIN_80, DEV_20)
    rows = LifecycleRows()
    proposer = SerializedRecordingProposer()

    with pytest.raises(ChampionLifecycleError, match="overwritten|replayed"):
        _lifecycle_controller(tmp_path, proposer, rows).evolve(
            PARENT, TRAIN_80, DEV_20
        )

    assert rows.opens == []
    assert proposer.serialized_inputs == ""


def test_hostile_calibration_callback_cannot_replace_durable_parent(tmp_path) -> None:
    class ReleaseTamperingRows(LifecycleRows):
        def __call__(self, tasks, split):
            result = super().__call__(tasks, split)
            if split == "calibration":
                tampered = replace(PARENT, lineage=("active_parent", "tampered"))
                (tmp_path / "champion_release.json").write_bytes(
                    canonical_release_bytes(tampered)
                )
            return result

    rows = ReleaseTamperingRows()

    with pytest.raises(ChampionLifecycleError, match="release|Parent|drift"):
        _lifecycle_controller(
            tmp_path, SerializedRecordingProposer(), rows
        ).evolve(PARENT, TRAIN_80, DEV_20)

    assert rows.opens == ["build", "calibration"]


def test_artifact_store_rejects_symlinked_ancestor(tmp_path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ChampionLifecycleError, match="symlink|alias"):
        ChampionArtifactStore(alias / "run")


def test_artifact_store_rejects_broken_symlinked_ancestor(tmp_path) -> None:
    alias = tmp_path / "broken_alias"
    alias.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(ChampionLifecycleError, match="symlink|alias"):
        ChampionArtifactStore(alias / "run")


def test_artifact_store_rejects_hardlinked_manifest(tmp_path) -> None:
    root = tmp_path / "run"
    controller = _lifecycle_controller(
        root, SerializedRecordingProposer(), LifecycleRows()
    )
    path = controller.artifact_store.bind_manifest(controller.manifest)
    os.link(path, tmp_path / "foreign_manifest.json")

    with pytest.raises(ChampionLifecycleError, match="alias|immutable"):
        controller.artifact_store.bind_manifest(controller.manifest)


def test_artifact_store_rejects_hardlinked_release_archive(tmp_path) -> None:
    store = ChampionArtifactStore(tmp_path / "run")
    store.ensure_release(PARENT)
    archive = (
        store.root / "releases" / f"{champion_fingerprint(PARENT)}.json"
    )
    os.link(archive, tmp_path / "foreign_release.json")

    with pytest.raises(ChampionLifecycleError, match="alias|immutable"):
        store.publish_release(PARENT)


def test_artifact_store_rejects_nonfinite_json(tmp_path) -> None:
    store = ChampionArtifactStore(tmp_path)
    (tmp_path / "checkpoint.json").write_text(
        '{"schema_version":1,"score":1e999}\n', encoding="utf-8"
    )

    with pytest.raises(ChampionLifecycleError, match="nonfinite"):
        store.load_checkpoint()


def test_lifecycle_artifacts_are_exact_canonical_json(tmp_path) -> None:
    _lifecycle_controller(
        tmp_path, SerializedRecordingProposer(), LifecycleRows()
    ).evolve(PARENT, TRAIN_80, DEV_20)

    artifacts = tuple(sorted(tmp_path.rglob("*.json")))
    assert artifacts
    for artifact in artifacts:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert artifact.read_bytes() == canonical_json_bytes(payload)


def test_interrupted_publication_preserves_last_release(monkeypatch, tmp_path) -> None:
    store = ChampionArtifactStore(tmp_path)
    store.ensure_release(PARENT)
    prior_bytes = (tmp_path / "champion_release.json").read_bytes()
    next_release = replace(
        PARENT,
        lineage=("active_parent", "next_release"),
    )
    replace_file = controller_module.os.replace

    def interrupted(source, destination):
        if str(destination).endswith("champion_release.json"):
            raise OSError("simulated publication interruption")
        return replace_file(source, destination)

    monkeypatch.setattr(controller_module.os, "replace", interrupted)

    with pytest.raises(OSError, match="publication interruption"):
        store.publish_release(next_release)

    assert (tmp_path / "champion_release.json").read_bytes() == prior_bytes


def test_successive_halving_runs_fixed_8_32_64_schedule() -> None:
    result = run_build_evolution(PARENT, ROWS_64, RecordingProposer("better"), _config())

    assert result.generations[0].stage_counts == (8, 32, 64)
    assert result.generations[0].full_build_children <= 3


def test_rejected_child_feedback_does_not_change_the_mutation_parent() -> None:
    parent = _parent()
    proposer = RecordingProposer("worse")
    global PARENT
    previous = PARENT
    PARENT = parent
    try:
        result = run_build_evolution(parent, ROWS_64, proposer, _config(generations=2))
    finally:
        PARENT = previous

    assert all(
        generation.mutation_parent_sha256 == champion_fingerprint(parent)
        for generation in result.generations
    )
    assert proposer.calls == 2
    assert result.active_parent is parent


def test_build_feedback_is_labeled_non_independent() -> None:
    result = run_build_evolution(PARENT, ROWS_64, RecordingProposer("better"), _config())

    assert result.score_label == "adaptive_train_build_diagnostic"
    assert result.independent_generalization_claim is False
    assert all(
        attempt.score_label == "adaptive_train_build_diagnostic"
        and attempt.independent_generalization_claim is False
        for attempt in result.generations[0].attempts
    )
    assert len(result.generations[0].feedback.comparisons) == len(
        result.generations[0].attempts
    )


def test_threshold_defaults_and_membership_are_fingerprinted() -> None:
    default = ChampionEvolutionConfig()
    candidate_changed = replace(default, candidate_minimum_gain=0.006)
    target_changed = replace(default, research_target_gain=0.051)

    assert default.candidate_minimum_gain == 0.005
    assert default.research_target_gain == 0.05
    assert default.fingerprint != candidate_changed.fingerprint
    assert default.fingerprint != target_changed.fingerprint
    assert _config().fingerprint != _config().screen_membership_sha256


def test_candidate_threshold_gates_but_research_target_does_not() -> None:
    high_target = replace(_config(), research_target_gain=0.99)
    high_candidate = replace(
        _config(), candidate_minimum_gain=0.99, research_target_gain=1.0
    )

    target_result = run_build_evolution(
        PARENT, ROWS_64, RecordingProposer("better"), high_target
    )
    candidate_result = run_build_evolution(
        PARENT, ROWS_64, RecordingProposer("better"), high_candidate
    )

    assert target_result.shortlist
    assert candidate_result.shortlist == ()


def test_halving_preserves_distinct_recipe_kinds_before_a_second_kind() -> None:
    result = run_build_evolution(PARENT, ROWS_64, DiverseProposer(), _config())

    kinds = tuple(policy.recipe.kind for policy in result.shortlist)
    assert len(kinds) == 3
    assert set(kinds) == {"select", "weighted", "median"}


def test_controller_rejects_normalized_duplicate_policy_ids() -> None:
    class DuplicatePolicyProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            names = ("Model_K", "Model_K", "unique_two", "unique_three", "unique_four")
            return tuple(
                ChampionRecipe(
                    name=name,
                    kind="select",
                    parents=("better_a",),
                    fallback_parent="better_a",
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"unique_assumption_{index}",
                            candidate_name="better_a",
                            feature="history_length",
                            direction="above",
                            horizon_region="full",
                            operator="select",
                            rationale="History length supports the candidate forecast.",
                            failure_condition=(
                                "History length stops supporting the candidate forecast."
                            ),
                        ),
                    ),
                )
                for index, name in enumerate(names)
            )

    with pytest.raises(ChampionControllerError, match="duplicate policy"):
        run_build_evolution(PARENT, ROWS_64, DuplicatePolicyProposer(), _config())


def test_hostile_callback_cannot_leave_the_exact_parent_mutated() -> None:
    parent = _parent()
    original_sha256 = champion_fingerprint(parent)

    class MutatingProposer:
        def __call__(self, received, evidence) -> tuple[ChampionRecipe, ...]:
            assert received is parent
            object.__setattr__(received.policy.recipe, "name", "tampered_parent")
            return _proposal_batch("better")

    with pytest.raises(ChampionControllerError, match="mutate the active Parent"):
        run_build_evolution(parent, ROWS_64, MutatingProposer(), _config())

    assert parent.policy.recipe.name == "active_parent"
    assert champion_fingerprint(parent) == original_sha256


def test_win_tie_loss_is_diagnostic_and_never_a_standalone_gate() -> None:
    mixed_rows = list(ROWS_64)
    for row in ROWS_64:
        if row.candidate_name != "baseline_leaf":
            continue
        assert row.truth is not None
        index = int(row.task_id.rsplit("_", 1)[1])
        error = 0.0 if index < 4 else 2.05
        mixed_rows.append(
            replace(
                row,
                candidate_name="mixed_leaf",
                forecast=tuple(value + error for value in row.truth),
            )
        )

    class MixedProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            candidates = ("mixed_leaf", "worse_b", "worse_c", "worse_d", "worse_e")
            return tuple(
                ChampionRecipe(
                    name=f"mixed_batch_{index}",
                    kind="select",
                    parents=(candidate,),
                    fallback_parent=candidate,
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"mixed_history_{index}",
                            candidate_name=candidate,
                            feature="history_length",
                            direction="above",
                            horizon_region="full",
                            operator="select",
                            rationale="History length supports the candidate forecast.",
                            failure_condition=(
                                "History length stops supporting the candidate forecast."
                            ),
                        ),
                    ),
                )
                for index, candidate in enumerate(candidates)
            )

    gate = ChampionGateConfig(
        tail_regression_tolerance=1.0,
        maximum_task_regret_smae=1.0,
        maximum_task_regret_srmse=1.0,
        minimum_improved_folds=0,
    )
    result = run_build_evolution(
        PARENT,
        tuple(mixed_rows),
        MixedProposer(),
        replace(_config(), gate_config=gate),
    )

    full = next(
        attempt
        for attempt in result.generations[0].attempts
        if attempt.policy.recipe.parents == ("mixed_leaf",)
        and attempt.stage_task_counts[-1] == 64
    )
    assert full.comparison is not None
    assert full.comparison.wtl.losses > full.comparison.wtl.wins
    assert result.shortlist


def test_malformed_proposer_output_fails_closed() -> None:
    class MalformedProposer:
        def __call__(self, parent, evidence) -> object:
            return {"recipes": []}

    with pytest.raises(ChampionControllerError, match="exact tuple or list"):
        run_build_evolution(PARENT, ROWS_64, MalformedProposer(), _config())


def test_row_universe_and_split_drift_fail_before_proposer_execution() -> None:
    rows = list(ROWS_64)
    rows[0] = replace(rows[0], split="calibration")
    proposer = RecordingProposer("better")

    with pytest.raises(ChampionControllerError, match="Build rows only"):
        run_build_evolution(PARENT, tuple(rows), proposer, _config())

    assert proposer.calls == 0


def test_hostile_callback_cannot_mutate_sanitized_feedback() -> None:
    class FeedbackMutator:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            object.__setattr__(evidence, "label", "tampered")
            return _proposal_batch("better")

    with pytest.raises(ChampionControllerError, match="sanitized Build feedback"):
        run_build_evolution(PARENT, ROWS_64, FeedbackMutator(), _config())


def test_unscorable_child_is_typed_pruned_without_aborting_valid_siblings() -> None:
    class MixedValidityProposer:
        calls = 0
        feedback: list[ProposerEvidence] = []

        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            self.feedback.append(evidence)
            generation = self.calls
            self.calls += 1
            valid = _proposal_batch("better", generation)[:1]
            invalid = tuple(
                ChampionRecipe(
                    name=f"overlay_{generation}_{index}",
                    kind="bounded_overlay",
                    parents=("better_a", "better_b"),
                    fallback_parent="better_a",
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"overlay_history_{generation}_{index}",
                            candidate_name="better_b",
                            feature="periodicity_confidence",
                            direction="above",
                            horizon_region="full",
                            operator="bounded_overlay",
                            rationale="History length supports the candidate forecast.",
                            failure_condition=(
                                "History length stops supporting the candidate forecast."
                            ),
                        ),
                    ),
                )
                for index in range(4)
            )
            return valid + invalid

    proposer = MixedValidityProposer()
    result = run_build_evolution(
        PARENT, ROWS_64, proposer, _config(generations=2)
    )

    invalid = tuple(
        attempt
        for attempt in result.generations[0].attempts
        if attempt.status == "invalid"
    )
    assert invalid
    assert all(attempt.comparison is None for attempt in invalid)
    assert all(attempt.invalid_reason == "unscorable_child" for attempt in invalid)
    generation_feedback = result.generations[0].feedback
    valid_count = len(result.generations[0].attempts) - len(invalid)
    assert len(generation_feedback.comparisons) == valid_count
    assert len(generation_feedback.invalid_attempts) == len(invalid)
    assert (
        len(generation_feedback.comparisons)
        + len(generation_feedback.invalid_attempts)
        == len(result.generations[0].attempts)
    )
    invalid_fingerprints = {
        champion_fingerprint(attempt.policy) for attempt in invalid
    }
    assert {
        item.structure_sha256 for item in generation_feedback.invalid_attempts
    } == invalid_fingerprints
    assert all(
        item.kind == "bounded_overlay"
        and item.stage_support == 8
        and item.reason_code == "unscorable_child"
        for item in generation_feedback.invalid_attempts
    )
    invalid_payloads = generation_feedback.to_payload()["invalid_attempts"]
    assert all(
        set(item) == {"structure_sha256", "kind", "stage_support", "reason_code"}
        for item in invalid_payloads
    )
    assert not any(
        forbidden in repr(invalid_payloads).casefold()
        for forbidden in (
            "overlay_0_",
            "better_a",
            "accepted",
            "rejected",
            "passed",
            "smae",
            "srmse",
            "truth",
            "forecast",
            "task_id",
            "split",
            "gate",
        )
    )
    assert proposer.feedback[1].invalid_attempts == generation_feedback.invalid_attempts
    assert not invalid_fingerprints & {
        champion_fingerprint(policy) for policy in result.shortlist
    }
    assert result.shortlist
    assert result.active_parent is PARENT


def test_deterministic_smoke_uses_only_the_4_8_schedule() -> None:
    task_ids = tuple(f"build_case_{index:03d}" for index in range(8))
    config = ChampionEvolutionConfig(
        build_size=8,
        calibration_size=2,
        screen_sizes=(4, 8),
        screen_task_ids=(task_ids[:4], task_ids),
        gate_config=ChampionGateConfig(minimum_improved_folds=0),
    )

    result = run_build_evolution(PARENT, _rows(8), RecordingProposer("better"), config)

    assert result.generations[0].stage_counts == (4, 8)
    assert result.config.build_task_ids == task_ids
    assert result.config_fingerprint == result.config.fingerprint


def test_history_only_materializer_scores_all_six_task3_operators() -> None:
    result = run_build_evolution(
        PARENT,
        _rows(enriched=True),
        AllOperatorProposer(),
        _config(),
    )

    valid_kinds = {
        attempt.policy.recipe.kind
        for attempt in result.generations[0].attempts
        if attempt.comparison is not None
    }
    assert valid_kinds == {
        "select",
        "route",
        "horizon_route",
        "weighted",
        "median",
        "bounded_overlay",
    }


def test_overlay_forecast_materialization_never_reads_future_truth(monkeypatch) -> None:
    rows = [row for row in _rows(1, enriched=True) if row.candidate_name in {
        "baseline_leaf",
        "better_a",
    }]

    class PoisonTruth:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("future truth reached history-only materialization")

    for row in rows:
        object.__setattr__(row, "truth", PoisonTruth())
    assumption = EvolutionAssumption(
        assumption_id="poison_free_history",
        candidate_name="better_a",
        feature="periodicity_confidence",
        direction="above",
        horizon_region="full",
        operator="bounded_overlay",
        rationale="History diagnostics support the overlay.",
        failure_condition="History diagnostics stop supporting the overlay.",
    )
    policy = FittedChampionPolicy(
        recipe=ChampionRecipe(
            name="poison_free_overlay",
            kind="bounded_overlay",
            parents=("baseline_leaf", "better_a"),
            fallback_parent="baseline_leaf",
            assumptions=(assumption,),
        ),
        thresholds=(("poison_free_history", 0.75),),
        overlay_alpha=0.5,
        correction_cap=0.1,
    )
    runtime_calls = []
    execute = controller_module.execute_champion

    def history_only_execute(policy, forecasts, diagnostics, profile, history, horizon):
        runtime_calls.append((forecasts, diagnostics, profile, history, horizon))
        assert all(diagnostic.folds == () for diagnostic in diagnostics.values())
        assert all(diagnostic.fold_forecasts == () for diagnostic in diagnostics.values())
        assert all(diagnostic.fold_truths == () for diagnostic in diagnostics.values())
        assert all(diagnostic.long_horizon_fold is None for diagnostic in diagnostics.values())
        return execute(policy, forecasts, diagnostics, profile, history, horizon)

    monkeypatch.setattr(controller_module, "execute_champion", history_only_execute)

    forecast, failure = controller_module._policy_forecast(
        policy, {row.candidate_name: row for row in rows}
    )

    assert len(runtime_calls) == 1
    assert failure is None
    assert forecast == pytest.approx((11.4, 13.4))


def test_valid_unsatisfied_history_condition_materializes_the_declared_fallback() -> None:
    rows = [
        row
        for row in _rows(1, enriched=True)
        if row.candidate_name in {"baseline_leaf", "better_a"}
    ]
    specialist_index = next(
        index for index, row in enumerate(rows) if row.candidate_name == "better_a"
    )
    specialist = rows[specialist_index]
    rows[specialist_index] = replace(
        specialist,
        diagnostic=_history_diagnostic("better_a", eligible=False),
    )
    assumption = EvolutionAssumption(
        assumption_id="unsatisfied_history",
        candidate_name="better_a",
        feature="periodicity_confidence",
        direction="above",
        horizon_region="full",
        operator="weighted",
        rationale="History diagnostics support the specialist.",
        failure_condition="History diagnostics do not support the specialist.",
    )
    policy = FittedChampionPolicy(
        recipe=ChampionRecipe(
            name="fallback_on_unsatisfied_history",
            kind="weighted",
            parents=("baseline_leaf", "better_a"),
            fallback_parent="baseline_leaf",
            assumptions=(assumption,),
        ),
        thresholds=(("unsatisfied_history", 0.0),),
        weights=(0.5, 0.5),
    )

    forecast, failure = controller_module._policy_forecast(
        policy, {row.candidate_name: row for row in rows}
    )

    assert failure is None
    assert forecast == rows[0].forecast


def test_proposal_policy_id_cannot_collide_with_the_active_parent() -> None:
    class ParentCollisionProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            return (parent.policy.recipe, *_proposal_batch("better")[:4])

    with pytest.raises(ChampionControllerError, match="duplicate policy ID"):
        run_build_evolution(PARENT, ROWS_64, ParentCollisionProposer(), _config())


def test_invalid_nested_parent_is_rejected_before_the_proposer() -> None:
    parent = _parent()
    object.__setattr__(parent.policy.recipe, "name", "")

    class CountingProposer:
        calls = 0

        def __call__(self, received, evidence) -> tuple[ChampionRecipe, ...]:
            type(self).calls += 1
            return _proposal_batch("better")

    with pytest.raises(ChampionControllerError, match="active Parent"):
        run_build_evolution(parent, ROWS_64, CountingProposer(), _config())

    assert CountingProposer.calls == 0


def test_callback_recipe_alias_cannot_corrupt_prior_generation_fitted_ids() -> None:
    class AliasingProposer:
        calls = 0
        first_batch: tuple[ChampionRecipe, ...] | None = None

        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            if self.first_batch is not None:
                object.__setattr__(
                    self.first_batch[0], "name", "mutated_external_alias"
                )
            batch = _proposal_batch("better", self.calls)
            if self.first_batch is None:
                self.first_batch = batch
            self.calls += 1
            return batch

    result = run_build_evolution(
        PARENT, ROWS_64, AliasingProposer(), _config(generations=2)
    )

    assert all(
        attempt.fitted_id == champion_fingerprint(attempt.policy)
        for generation in result.generations
        for attempt in generation.attempts
    )
    assert all(
        attempt.policy.recipe.name != "mutated_external_alias"
        for generation in result.generations
        for attempt in generation.attempts
    )


def test_callback_cannot_loosen_or_leave_the_registered_gate_mutated() -> None:
    gate = ChampionGateConfig(minimum_improved_folds=0)
    config = replace(_config(), gate_config=gate)

    class GateMutator:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            object.__setattr__(gate, "minimum_coverage", 0.0)
            return _proposal_batch("better")

    original_fingerprint = config.fingerprint
    with pytest.raises(ChampionControllerError, match="config"):
        run_build_evolution(PARENT, ROWS_64, GateMutator(), config)

    assert gate.minimum_coverage == 1.0
    assert config.fingerprint == original_fingerprint


def test_registered_fold_gate_remains_authoritative_on_the_first_screen() -> None:
    all_ids = tuple(f"build_case_{index:03d}" for index in range(64))
    first_stage = tuple(f"build_case_{index:03d}" for index in range(0, 40, 5))
    second_stage = first_stage + tuple(
        task_id for task_id in all_ids if task_id not in first_stage
    )[:24]
    config = replace(
        _config(),
        screen_task_ids=(first_stage, second_stage, all_ids),
        gate_config=ChampionGateConfig(minimum_improved_folds=2),
    )

    result = run_build_evolution(
        PARENT, ROWS_64, RecordingProposer("better"), config
    )

    assert all(attempt.stage_task_counts == (8,) for attempt in result.generations[0].attempts)
    assert result.generations[0].full_build_children == 0
    assert result.shortlist == ()


def test_normalized_row_inventory_collision_fails_before_proposer() -> None:
    rows = list(ROWS_64)
    for index in range(64):
        task_id = f"build_case_{index:03d}"
        source = next(
            row
            for row in ROWS_64
            if row.task_id == task_id and row.candidate_name == "baseline_leaf"
        )
        rows.extend(
            (
                replace(source, candidate_name="Model_K"),
                replace(source, candidate_name="Model_K"),
            )
        )
    proposer = RecordingProposer("better")

    with pytest.raises(ChampionControllerError, match="normalized row inventory"):
        run_build_evolution(PARENT, tuple(rows), proposer, _config())

    assert proposer.calls == 0


def test_proposal_name_cannot_collide_with_materialized_row_inventory() -> None:
    class InventoryCollisionProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            first, *remaining = _proposal_batch("better")
            return (replace(first, name="better_a"), *remaining)

    with pytest.raises(ChampionControllerError, match="materialized row inventory"):
        run_build_evolution(PARENT, ROWS_64, InventoryCollisionProposer(), _config())
