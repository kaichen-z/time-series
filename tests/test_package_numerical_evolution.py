"""Numerical package proposal, fitting, and materialization boundaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import pytest

from common.evolution_core.task_feedback import TaskEvidenceProjection
from common.data import Task as DataTask
import evolving_loop.package_numerical_evolution as numerical_evolution
from evolving_loop.data import ContextTask, Document
from evolving_loop.package_numerical_evolution import (
    FrozenNumericalDiagnosticsRegistry,
    NumericalCoordinateCandidate,
    NumericalPackageEvolutionError,
    NumericalPackageMaterializer,
    NumericalPackageProposer,
    fit_numerical_recipe,
)
from evolving_loop.package_numerical_supply import (
    NumericalAlternativeSpec,
    NumericalSupplyRelease,
)
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
)
from numerical_agent.evolution.champion_evidence import ChampionTaskRow
from numerical_agent.evolution.champion_controller import ChampionProposerAdapter
from numerical_agent.evolution.champion_evidence import (
    MorphologyAggregate,
    ProposerEvidence,
)
from numerical_agent.evolution.champion_proposal import expand_recipe
from numerical_agent.evolution.execution import Task as RuntimeTask
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    HindcastConfig,
)
from numerical_agent.evolution.screening import (
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
    profile_task,
)
from numerical_agent.evolution.specialist_atlas import AtlasPolicy, fit_atlas_release
from numerical_agent.evolution.task_local_evolution import (
    TaskLocalTaskRow,
    build_group_fold_manifest,
)
from tests.test_task_evidence_feedback import _case


def _recipe(candidate_name: str = "seasonal_naive") -> ChampionRecipe:
    assumption = EvolutionAssumption(
        assumption_id=f"{candidate_name}_history",
        candidate_name=candidate_name,
        feature="history_length",
        direction="above",
        horizon_region="full",
        operator="select",
        rationale="History length supports the reviewed candidate.",
        failure_condition="History length no longer supports the candidate.",
    )
    return ChampionRecipe(
        name=f"select_{candidate_name}",
        kind="select",
        parents=(candidate_name,),
        fallback_parent=candidate_name,
        assumptions=(assumption,),
    )


def _parent() -> ChampionRelease:
    recipe = _recipe("toto_2_0")
    return ChampionRelease(
        policy=FittedChampionPolicy(
            recipe=recipe,
            thresholds=((recipe.assumptions[0].assumption_id, 0.0),),
        ),
        source_hashes=(("dictionary", hashlib.sha256(b"reviewed").hexdigest()),),
        metric_policy_fingerprint="1" * 64,
        lineage=("toto_2_0",),
    )


def _tasks(count: int = 80) -> tuple[DataTask, ...]:
    return tuple(
        DataTask(
            task_id=f"build_case_{index:03d}",
            history_values=tuple(float(index + offset) for offset in range(8 + index)),
            future_values=(float(index + 10), float(index + 11)),
            prediction_length=2,
            frequency="D",
            seasonal_period="7",
            entity_name=f"Entity {index:03d}",
        )
        for index in range(count)
    )


def _build_rows(tasks: tuple[DataTask, ...]) -> tuple[ChampionTaskRow, ...]:
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    return tuple(
        ChampionTaskRow(
            task_id=task.task_id,
            candidate_name=candidate_name,
            profile=profile_task(
                RuntimeTask(
                    task_id=task.task_id,
                    history=task.history_values,
                    horizon=task.prediction_length,
                    frequency=task.frequency,
                    future=(),
                )
            ),
            truth=task.future_values,
            forecast=tuple(
                value + (0.5 if candidate_name == "seasonal_naive" else 1.0)
                for value in task.future_values
            ),
            fold=manifest.task_fold_map[task.task_id],
            split="build",
            history=task.history_values,
        )
        for task in tasks
        for candidate_name in ("toto_2_0", "seasonal_naive")
    )


def test_fit_numerical_recipe_uses_host_grid_and_deterministic_rank() -> None:
    tasks = _tasks()
    rows = _build_rows(tasks)
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    recipe = _recipe()

    fitted = fit_numerical_recipe(recipe, rows, manifest, _parent())

    assert fitted.full_build_policy in expand_recipe(recipe, rows)
    assert fitted.recipe == recipe
    assert tuple(fold for fold, _policy in fitted.build_fold_policies) == (
        0,
        1,
        2,
        3,
        4,
    )
    assert fitted.full_build_task_ids == tuple(task.task_id for task in tasks)
    assert len(fitted.numerical_score_sha256) == 64
    assert fitted == fit_numerical_recipe(recipe, rows, manifest, _parent())


def test_fit_numerical_recipe_never_accepts_dev_or_public_rows() -> None:
    tasks = _tasks()
    rows = _build_rows(tasks)
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    contaminated = (
        *rows[:-1],
        ChampionTaskRow(**{**rows[-1].__dict__, "split": "dev"}),
    )

    with pytest.raises(NumericalPackageEvolutionError, match="Train"):
        fit_numerical_recipe(_recipe(), contaminated, manifest, _parent())


def test_each_build_fold_policy_excludes_its_held_out_groups() -> None:
    tasks = _tasks()
    rows = _build_rows(tasks)
    manifest = build_group_fold_manifest(tasks, seed=20260903)

    fitted = fit_numerical_recipe(_recipe(), rows, manifest, _parent())

    recorded = dict(fitted.fold_training_task_ids)
    for fold, _policy in fitted.build_fold_policies:
        held_out = {
            task_id
            for _group_sha, task_ids, assigned_fold in manifest.groups
            if assigned_fold == fold
            for task_id in task_ids
        }
        assert set(recorded[fold]).isdisjoint(held_out)
        assert set(recorded[fold]) == set(fitted.full_build_task_ids) - held_out


def test_fit_numerical_recipe_is_independent_of_build_row_order() -> None:
    tasks = _tasks()
    rows = _build_rows(tasks)
    manifest = build_group_fold_manifest(tasks, seed=20260903)

    forward = fit_numerical_recipe(_recipe(), rows, manifest, _parent())
    reversed_rows = fit_numerical_recipe(
        _recipe(), tuple(reversed(rows)), manifest, _parent()
    )

    assert reversed_rows == forward


def _supply_parent() -> NumericalSupplyRelease:
    return NumericalSupplyRelease(
        schema_version=1,
        version="n000",
        parent_sha256=None,
        anchor_release_payload=_parent().to_payload(),
        alternatives=(),
        atlas_release_sha256=None,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
    )


def _registry_for_release(
    release: NumericalSupplyRelease,
    task_ids: tuple[str, ...] | None = None,
) -> FrozenNumericalPackageRegistry:
    registry = object.__new__(FrozenNumericalPackageRegistry)
    registry._release_sha256 = release.fingerprint
    if task_ids is None:
        task_ids = tuple(task.numeric.task_id for task in _evolution_tasks())
    registry._packages = {task_id: None for task_id in task_ids}
    return registry


def _proposal_batch() -> tuple[ChampionRecipe, ...]:
    structures = (
        ("seasonal_naive", "history_length", "above"),
        ("seasonal_naive", "horizon", "above"),
        ("seasonal_naive", "trend_strength", "below"),
        ("toto_2_0", "history_length", "below"),
        ("toto_2_0", "horizon_ratio", "above"),
    )
    return tuple(
        ChampionRecipe(
            name=f"proposal_{index}",
            kind="select",
            parents=(candidate,),
            fallback_parent=candidate,
            assumptions=(
                EvolutionAssumption(
                    assumption_id=f"proposal_assumption_{index}",
                    candidate_name=candidate,
                    feature=feature,
                    direction=direction,
                    horizon_region="full",
                    operator="select",
                    rationale="The reviewed structural feature supports this candidate.",
                    failure_condition="The reviewed structural feature stops supporting it.",
                ),
            ),
        )
        for index, (candidate, feature, direction) in enumerate(structures)
    )


def _evolution_tasks() -> tuple[ContextTask, ...]:
    return tuple(
        ContextTask(
            numeric=DataTask(
                task_id=(
                    f"build_case_{index:03d}"
                    if index < 80
                    else f"evolution_{index:03d}"
                ),
                history_values=tuple(
                    float(index + offset) for offset in range(8 + index)
                ),
                future_values=(float(index + 10), float(index + 11)),
                prediction_length=2,
                frequency="D",
                seasonal_period="7",
                entity_name=f"Entity {index:03d}",
            ),
            target_name="demand",
            target_description="daily demand",
            history_timestamps=tuple(f"h{offset}" for offset in range(8 + index)),
            future_timestamps=("f0", "f1"),
            documents=(Document("doc", "history context", "supporting", "fact"),),
            gt_evidence=("doc",),
            labels_public=True,
        )
        for index in range(100)
    )


@dataclass
class _RecordingMaterializer:
    seen_tasks: tuple[ContextTask, ...] = ()

    def materialize(
        self,
        parent_release: NumericalSupplyRelease,
        fit,
        tasks,
        *,
        version: str,
        generation: int,
    ) -> NumericalCoordinateCandidate:
        self.seen_tasks = tuple(tasks)
        release = NumericalSupplyRelease(
            schema_version=1,
            version=version,
            parent_sha256=parent_release.fingerprint,
            anchor_release_payload=_parent().to_payload(),
            alternatives=(),
            atlas_release_sha256=None,
            source_fingerprints=parent_release.source_fingerprints,
            runtime_fingerprints=parent_release.runtime_fingerprints,
        )
        return NumericalCoordinateCandidate(
            release=release,
            registry=_registry_for_release(release),
            proposal_sha256=hashlib.sha256(
                f"{generation}:{fit.numerical_score_sha256}".encode()
            ).hexdigest(),
        )


def _package_proposer(materializer: _RecordingMaterializer) -> NumericalPackageProposer:
    tasks = _tasks()
    return NumericalPackageProposer(
        proposer=ChampionProposerAdapter.scripted(
            identity="package_numerical_fixture",
            proposal_batches=(_proposal_batch(),),
            config={"fixture": True},
        ),
        materializer=materializer,
        build_rows=_build_rows(tasks),
        fold_manifest=build_group_fold_manifest(tasks, seed=20260903),
        tasks=_evolution_tasks(),
    )


def test_numerical_proposer_returns_three_direct_lineage_states() -> None:
    parent_release = _supply_parent()
    parent_registry = _registry_for_release(parent_release)
    materializer = _RecordingMaterializer()
    proposer = _package_proposer(materializer)

    children = proposer.propose(
        parent_release,
        parent_registry,
        ProposerEvidence("adaptive_train_build_diagnostic", False, (), ()),
        generation=0,
        child_count=3,
    )

    assert len(children) == 3
    assert len({child.proposal_sha256 for child in children}) == 3


def test_numerical_package_proposer_accepts_sanitized_task_evidence() -> None:
    parent_release = _supply_parent()
    projection = TaskEvidenceProjection(
        source_bundle_sha256="a" * 64,
        request_namespace_sha256="b" * 64,
        cases=(_case(),),
    )

    children = _package_proposer(_RecordingMaterializer()).propose(
        parent_release,
        _registry_for_release(parent_release),
        ProposerEvidence("adaptive_train_build_diagnostic", False, (), ()),
        generation=0,
        child_count=3,
        task_evidence=projection,
    )

    assert len(children) == 3
    assert all(child.invalid_reason is None for child in children)
    assert all(
        child.release.parent_sha256 == parent_release.fingerprint for child in children
    )
    assert all(
        child.registry.release_sha256 == child.release.fingerprint for child in children
    )


def test_numerical_materialization_receives_history_only_tasks() -> None:
    parent_release = _supply_parent()
    materializer = _RecordingMaterializer()
    proposer = _package_proposer(materializer)

    proposer.propose(
        parent_release,
        _registry_for_release(parent_release),
        ProposerEvidence("adaptive_train_build_diagnostic", False, (), ()),
        generation=0,
        child_count=3,
    )

    assert materializer.seen_tasks
    assert all(task.numeric.future_values == () for task in materializer.seen_tasks)
    assert all(task.gt_evidence == () for task in materializer.seen_tasks)
    assert all(task.labels_public is False for task in materializer.seen_tasks)
    assert all(
        document.role is None and document.subtype is None
        for task in materializer.seen_tasks
        for document in task.documents
    )


class _NeverForecastStore:
    def forecast(self, *_args, **_kwargs):
        raise AssertionError("label firewall must reject before forecasting")


def test_numerical_materializer_rejects_label_bearing_task_inputs() -> None:
    tasks = _tasks()
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    fit = fit_numerical_recipe(_recipe(), _build_rows(tasks), manifest, _parent())
    labeled = _evolution_tasks()
    materializer = NumericalPackageMaterializer(
        forecast_store=_NeverForecastStore(),
        screening_policy=ScreeningPolicy(
            (
                ScreeningEntry(
                    "toto_2_0",
                    "tsfm",
                    "keep",
                    ApplicabilityPolicy(),
                    "reviewed anchor",
                ),
                ScreeningEntry(
                    "seasonal_naive",
                    "statistical",
                    "keep",
                    ApplicabilityPolicy(),
                    "reviewed alternative",
                ),
            ),
            ("toto_2_0",),
        ),
        fold_manifest=manifest,
        original_tasks=labeled,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
    )

    with pytest.raises(NumericalPackageEvolutionError, match="label-free"):
        materializer.materialize(
            _supply_parent(),
            fit,
            labeled,
            version="n001",
            generation=0,
        )


class _FixtureForecastStore:
    def forecast(self, name, history, horizon, _frequency):
        offset = 0.0 if name == "toto_2_0" else 1.0
        return (float(history[-1]) + offset,) * horizon


def test_numerical_materializer_builds_complete_registry_with_toto_anchor() -> None:
    build_tasks = _tasks()
    manifest = build_group_fold_manifest(build_tasks, seed=20260903)
    fit = fit_numerical_recipe(_recipe(), _build_rows(build_tasks), manifest, _parent())
    original_tasks = _evolution_tasks()
    diagnostics = {
        task.numeric.task_id: {
            name: CandidateDiagnostics.synthetic(
                name=name,
                family=family,
                median_mase=score,
                median_smae=score,
                recent_smae=score,
                worst_smae=score,
                median_srmse=score,
                recent_srmse=score,
                worst_srmse=score,
                worst_smae_raw=score,
                worst_srmse_raw=score,
            )
            for name, family, score in (
                ("toto_2_0", "tsfm", 0.1),
                ("seasonal_naive", "statistical", 0.2),
            )
        }
        for task in original_tasks
    }
    diagnostics_registry = FrozenNumericalDiagnosticsRegistry.build(
        original_tasks, diagnostics, HindcastConfig()
    )
    screening = ScreeningPolicy(
        (
            ScreeningEntry(
                "toto_2_0",
                "tsfm",
                "keep",
                ApplicabilityPolicy(),
                "reviewed anchor",
            ),
            ScreeningEntry(
                "seasonal_naive",
                "statistical",
                "keep",
                ApplicabilityPolicy(),
                "reviewed alternative",
            ),
        ),
        ("toto_2_0",),
    )
    materializer = NumericalPackageMaterializer(
        forecast_store=_FixtureForecastStore(),
        screening_policy=screening,
        fold_manifest=manifest,
        original_tasks=original_tasks,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
        diagnostics_registry=diagnostics_registry,
    )
    sanitized = tuple(
        replace(
            task,
            numeric=task.numeric_view(),
            documents=tuple(
                replace(document, role=None, subtype=None)
                for document in task.documents
            ),
            gt_evidence=(),
            labels_public=False,
        )
        for task in original_tasks
    )

    candidate = materializer.materialize(
        _supply_parent(),
        fit,
        sanitized,
        version="n001",
        generation=0,
    )

    assert candidate.registry.task_ids == tuple(
        sorted(task.numeric.task_id for task in original_tasks)
    )
    assert candidate.release.parent_sha256 == _supply_parent().fingerprint
    assert (
        candidate.release.source_fingerprints["diagnostics_registry"]
        == diagnostics_registry.fingerprint
    )
    assert candidate.release.alternatives[0].candidate_id == _recipe().name
    for task in original_tasks:
        package = candidate.registry.package_for(task)
        assert package.protected_baseline.name == "toto_2_0"
        assert {item.name for item in package.ranked_alternatives} == {
            "toto_2_0",
            _recipe().name,
        }


def _atlas_fit_rows(tasks: tuple[DataTask, ...]) -> tuple[TaskLocalTaskRow, ...]:
    rows: list[TaskLocalTaskRow] = []
    for task in tasks:
        profile = profile_task(
            RuntimeTask(
                task.task_id,
                task.history_values,
                task.prediction_length,
                task.frequency,
                (),
            )
        )
        for name, family, offset, score in (
            ("toto_2_0", "tsfm", 2.0, 0.7),
            ("seasonal_naive", "statistical", 0.2, 0.2),
        ):
            forecast = tuple(value + offset for value in task.future_values)
            rows.append(
                TaskLocalTaskRow(
                    task_id=task.task_id,
                    candidate_name=name,
                    family=family,
                    profile=profile,
                    history=task.history_values,
                    truth=task.future_values,
                    forecast=forecast,
                    diagnostic=CandidateDiagnostics.synthetic(
                        name=name,
                        family=family,
                        median_mase=score,
                        fold_forecasts=(forecast,) * 3,
                        fold_truths=(task.future_values,) * 3,
                        median_smae=score,
                        recent_smae=score,
                        worst_smae=score,
                        median_srmse=score,
                        recent_srmse=score,
                        worst_srmse=score,
                        worst_smae_raw=score,
                        worst_srmse_raw=score,
                    ),
                    split="train",
                )
            )
    return tuple(rows)


def test_numerical_materializer_includes_only_policy_accepted_atlas_routes() -> None:
    build_tasks = _tasks()
    manifest = build_group_fold_manifest(build_tasks, seed=20260903)
    fit = fit_numerical_recipe(_recipe(), _build_rows(build_tasks), manifest, _parent())
    atlas_release = fit_atlas_release(
        _atlas_fit_rows(build_tasks), manifest, AtlasPolicy()
    )
    original_tasks = _evolution_tasks()
    diagnostics = {
        task.numeric.task_id: {
            name: CandidateDiagnostics.synthetic(
                name=name,
                family=family,
                median_mase=score,
                fold_forecasts=((1.0, 2.0),) * 3,
                fold_truths=((1.0, 2.0),) * 3,
                median_smae=score,
                recent_smae=score,
                worst_smae=score,
                median_srmse=score,
                recent_srmse=score,
                worst_srmse=score,
                worst_smae_raw=score,
                worst_srmse_raw=score,
            )
            for name, family, score in (
                ("toto_2_0", "tsfm", 0.7),
                ("seasonal_naive", "statistical", 0.2),
            )
        }
        for task in original_tasks
    }
    screening = ScreeningPolicy(
        (
            ScreeningEntry(
                "toto_2_0",
                "tsfm",
                "keep",
                ApplicabilityPolicy(),
                "reviewed anchor",
            ),
            ScreeningEntry(
                "seasonal_naive",
                "statistical",
                "keep",
                ApplicabilityPolicy(),
                "reviewed alternative",
            ),
        ),
        ("toto_2_0",),
    )
    materializer = NumericalPackageMaterializer(
        forecast_store=_FixtureForecastStore(),
        screening_policy=screening,
        fold_manifest=manifest,
        original_tasks=original_tasks,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
        atlas_release=atlas_release,
        diagnostics_registry=FrozenNumericalDiagnosticsRegistry.build(
            original_tasks, diagnostics, HindcastConfig()
        ),
    )
    sanitized = tuple(
        replace(
            task,
            numeric=task.numeric_view(),
            documents=tuple(
                replace(document, role=None, subtype=None)
                for document in task.documents
            ),
            gt_evidence=(),
            labels_public=False,
        )
        for task in original_tasks
    )

    candidate = materializer.materialize(
        _supply_parent(), fit, sanitized, version="n001", generation=0
    )

    package = candidate.registry.package_for(original_tasks[-1])
    assert package.protected_baseline.name == "toto_2_0"
    assert "atlas_70_30" in {item.name for item in package.ranked_alternatives}


def _screening() -> ScreeningPolicy:
    return ScreeningPolicy(
        (
            ScreeningEntry(
                "toto_2_0",
                "tsfm",
                "keep",
                ApplicabilityPolicy(),
                "reviewed anchor",
            ),
            ScreeningEntry(
                "seasonal_naive",
                "statistical",
                "keep",
                ApplicabilityPolicy(),
                "reviewed alternative",
            ),
        ),
        ("toto_2_0",),
    )


def _history_diagnostics(
    tasks: tuple[ContextTask, ...],
) -> dict[str, dict[str, CandidateDiagnostics]]:
    return {
        task.numeric.task_id: {
            name: CandidateDiagnostics.synthetic(
                name=name,
                family=family,
                median_mase=score,
                fold_forecasts=((1.0, 2.0),) * 3,
                fold_truths=((1.0, 2.0),) * 3,
                median_smae=score,
                recent_smae=score,
                worst_smae=score,
                median_srmse=score,
                recent_srmse=score,
                worst_srmse=score,
                worst_smae_raw=score,
                worst_srmse_raw=score,
            )
            for name, family, score in (
                ("toto_2_0", "tsfm", 0.7),
                ("seasonal_naive", "statistical", 0.2),
            )
        }
        for task in tasks
    }


def _label_free(tasks: tuple[ContextTask, ...]) -> tuple[ContextTask, ...]:
    return tuple(
        replace(
            task,
            numeric=task.numeric_view(),
            documents=tuple(
                replace(document, role=None, subtype=None)
                for document in task.documents
            ),
            gt_evidence=(),
            labels_public=False,
        )
        for task in tasks
    )


def test_build_atlas_materialization_uses_only_the_matching_fold_model_pool() -> None:
    build_tasks = _tasks()
    manifest = build_group_fold_manifest(build_tasks, seed=20260903)
    fit = fit_numerical_recipe(_recipe(), _build_rows(build_tasks), manifest, _parent())
    atlas = fit_atlas_release(_atlas_fit_rows(build_tasks), manifest, AtlasPolicy())
    assert len(atlas.full_build_model.selected_pool) > 1
    atlas = replace(
        atlas,
        full_build_model=replace(
            atlas.full_build_model,
            selected_pool=("toto_2_0",),
            records=(),
        ),
    )
    original_tasks = _evolution_tasks()
    materializer = NumericalPackageMaterializer(
        forecast_store=_FixtureForecastStore(),
        screening_policy=_screening(),
        fold_manifest=manifest,
        original_tasks=original_tasks,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
        atlas_release=atlas,
        diagnostics_registry=FrozenNumericalDiagnosticsRegistry.build(
            original_tasks, _history_diagnostics(original_tasks), HindcastConfig()
        ),
    )

    candidate = materializer.materialize(
        _supply_parent(),
        fit,
        _label_free(original_tasks),
        version="n001",
        generation=0,
    )

    build_package = candidate.registry.package_for(original_tasks[0])
    assert "atlas_70_30" in {item.name for item in build_package.ranked_alternatives}


def test_materializer_rejects_recipe_fit_from_a_different_fold_manifest() -> None:
    build_tasks = _tasks()
    fitted_manifest = build_group_fold_manifest(build_tasks, seed=20260903)
    execution_manifest = build_group_fold_manifest(build_tasks, seed=20260904)
    fit = fit_numerical_recipe(
        _recipe(), _build_rows(build_tasks), fitted_manifest, _parent()
    )
    original_tasks = _evolution_tasks()
    materializer = NumericalPackageMaterializer(
        forecast_store=_NeverForecastStore(),
        screening_policy=_screening(),
        fold_manifest=execution_manifest,
        original_tasks=original_tasks,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
    )

    with pytest.raises(NumericalPackageEvolutionError, match="manifest"):
        materializer.materialize(
            _supply_parent(),
            fit,
            _label_free(original_tasks),
            version="n001",
            generation=0,
        )


def test_materializer_rejects_recipe_fit_from_a_different_parent() -> None:
    build_tasks = _tasks()
    manifest = build_group_fold_manifest(build_tasks, seed=20260903)
    fit = fit_numerical_recipe(_recipe(), _build_rows(build_tasks), manifest, _parent())
    other_anchor = replace(_parent(), source_hashes=(("dictionary", "9" * 64),))
    other_parent = NumericalSupplyRelease(
        schema_version=1,
        version="n000",
        parent_sha256=None,
        anchor_release_payload=other_anchor.to_payload(),
        alternatives=(),
        atlas_release_sha256=None,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
    )
    original_tasks = _evolution_tasks()
    materializer = NumericalPackageMaterializer(
        forecast_store=_NeverForecastStore(),
        screening_policy=_screening(),
        fold_manifest=manifest,
        original_tasks=original_tasks,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
    )

    with pytest.raises(NumericalPackageEvolutionError, match="Parent"):
        materializer.materialize(
            other_parent,
            fit,
            _label_free(original_tasks),
            version="n001",
            generation=0,
        )


def test_materializer_rejects_tampered_per_fold_training_complements() -> None:
    build_tasks = _tasks()
    manifest = build_group_fold_manifest(build_tasks, seed=20260903)
    fit = fit_numerical_recipe(_recipe(), _build_rows(build_tasks), manifest, _parent())
    memberships = dict(fit.fold_training_task_ids)
    tampered = replace(
        fit,
        fold_training_task_ids=tuple(
            (fold, memberships[(fold + 1) % 5]) for fold in range(5)
        ),
    )
    original_tasks = _evolution_tasks()
    materializer = NumericalPackageMaterializer(
        forecast_store=_NeverForecastStore(),
        screening_policy=_screening(),
        fold_manifest=manifest,
        original_tasks=original_tasks,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
    )

    with pytest.raises(NumericalPackageEvolutionError, match="membership"):
        materializer.materialize(
            _supply_parent(),
            tampered,
            _label_free(original_tasks),
            version="n001",
            generation=0,
        )


def test_materializer_rejects_an_atlas_release_from_another_manifest() -> None:
    build_tasks = _tasks()
    fitted_manifest = build_group_fold_manifest(build_tasks, seed=20260903)
    execution_manifest = build_group_fold_manifest(build_tasks, seed=20260904)
    atlas = fit_atlas_release(
        _atlas_fit_rows(build_tasks), fitted_manifest, AtlasPolicy()
    )

    with pytest.raises(NumericalPackageEvolutionError, match="Atlas.*manifest"):
        NumericalPackageMaterializer(
            forecast_store=_FixtureForecastStore(),
            screening_policy=_screening(),
            fold_manifest=execution_manifest,
            original_tasks=_evolution_tasks(),
            source_fingerprints={"dictionary": "2" * 64},
            runtime_fingerprints={"materializer": "3" * 64},
            atlas_release=atlas,
        )


def test_formal_proposer_requires_exactly_three_slots() -> None:
    parent = _supply_parent()
    proposer = _package_proposer(_RecordingMaterializer())

    with pytest.raises(NumericalPackageEvolutionError, match="three"):
        proposer.propose(
            parent,
            _registry_for_release(parent),
            ProposerEvidence("adaptive_train_build_diagnostic", False, (), ()),
            generation=0,
            child_count=2,
        )


@dataclass
class _MalformedResultMaterializer:
    failure: str
    cached: NumericalCoordinateCandidate | None = None

    def materialize(
        self,
        parent_release: NumericalSupplyRelease,
        fit,
        tasks,
        *,
        version: str,
        generation: int,
    ) -> NumericalCoordinateCandidate:
        del tasks, generation
        if self.failure == "duplicate" and self.cached is not None:
            return self.cached
        release = NumericalSupplyRelease(
            schema_version=1,
            version="n999" if self.failure == "version" else version,
            parent_sha256=(
                "f" * 64 if self.failure == "parent" else parent_release.fingerprint
            ),
            anchor_release_payload=_parent().to_payload(),
            alternatives=(),
            atlas_release_sha256=None,
            source_fingerprints=parent_release.source_fingerprints,
            runtime_fingerprints=parent_release.runtime_fingerprints,
        )
        candidate = NumericalCoordinateCandidate(
            release=release,
            registry=_registry_for_release(release),
            proposal_sha256=fit.numerical_score_sha256,
        )
        if self.failure == "duplicate":
            self.cached = candidate
        return candidate


@pytest.mark.parametrize(
    ("failure", "valid_count"),
    (("version", 0), ("parent", 0), ("duplicate", 1)),
)
def test_formal_proposer_rejects_malformed_or_duplicate_valid_results(
    failure: str,
    valid_count: int,
) -> None:
    parent = _supply_parent()
    proposer = _package_proposer(_MalformedResultMaterializer(failure))

    children = proposer.propose(
        parent,
        _registry_for_release(parent),
        ProposerEvidence("adaptive_train_build_diagnostic", False, (), ()),
        generation=0,
        child_count=3,
    )

    assert sum(child.invalid_reason is None for child in children) == valid_count
    assert len({child.proposal_sha256 for child in children}) == 3


def test_proposer_revalidates_feedback_against_every_build_task_identity() -> None:
    parent = _supply_parent()
    feedback = ProposerEvidence(
        "adaptive_train_build_diagnostic",
        False,
        (
            MorphologyAggregate(
                group_id="history:build_case_000",
                support=1,
                candidate_name="seasonal_naive",
                mean_delta_smae=0.0,
                mean_delta_srmse=0.0,
                coverage=1.0,
                p95_regret_smae=0.0,
                p95_regret_srmse=0.0,
            ),
        ),
        (),
    )

    with pytest.raises(NumericalPackageEvolutionError, match="identity"):
        _package_proposer(_RecordingMaterializer()).propose(
            parent,
            _registry_for_release(parent),
            feedback,
            generation=0,
            child_count=3,
        )


def test_proposer_rejects_incomplete_build_rows_before_the_callback() -> None:
    tasks = _tasks()
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    omitted_task_id = tasks[0].task_id
    incomplete_rows = tuple(
        row for row in _build_rows(tasks) if row.task_id != omitted_task_id
    )

    with pytest.raises(NumericalPackageEvolutionError, match="Train row universe"):
        NumericalPackageProposer(
            proposer=ChampionProposerAdapter.scripted(
                identity="incomplete_rows_fixture",
                proposal_batches=(_proposal_batch(),),
                config={"fixture": True},
            ),
            materializer=_RecordingMaterializer(),
            build_rows=incomplete_rows,
            fold_manifest=manifest,
            tasks=_evolution_tasks(),
        )


def test_proposer_binds_tasks_to_the_parent_registry_exactly() -> None:
    parent = _supply_parent()
    registered = tuple(task.numeric.task_id for task in _evolution_tasks())
    wrong_registry_ids = (*registered[:-1], "public_case_100")

    with pytest.raises(NumericalPackageEvolutionError, match="Parent registry"):
        _package_proposer(_RecordingMaterializer()).propose(
            parent,
            _registry_for_release(parent, wrong_registry_ids),
            ProposerEvidence("adaptive_train_build_diagnostic", False, (), ()),
            generation=0,
            child_count=3,
        )


def test_proposer_and_materializer_reject_non_100_task_universes() -> None:
    tasks = _tasks()
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    extra = replace(
        _evolution_tasks()[-1],
        numeric=replace(
            _evolution_tasks()[-1].numeric,
            task_id="public_case_100",
        ),
    )
    oversized = (*_evolution_tasks(), extra)

    with pytest.raises(NumericalPackageEvolutionError, match="100"):
        NumericalPackageProposer(
            proposer=ChampionProposerAdapter.scripted(
                identity="oversized_fixture",
                proposal_batches=(_proposal_batch(),),
                config={"fixture": True},
            ),
            materializer=_RecordingMaterializer(),
            build_rows=_build_rows(tasks),
            fold_manifest=manifest,
            tasks=oversized,
        )
    with pytest.raises(NumericalPackageEvolutionError, match="100"):
        NumericalPackageMaterializer(
            forecast_store=_FixtureForecastStore(),
            screening_policy=_screening(),
            fold_manifest=manifest,
            original_tasks=oversized,
            source_fingerprints={"dictionary": "2" * 64},
            runtime_fingerprints={"materializer": "3" * 64},
        )


def _retained_combined_spec() -> NumericalAlternativeSpec:
    assumptions = tuple(
        EvolutionAssumption(
            assumption_id=f"retained_{name}",
            candidate_name=name,
            feature="history_length",
            direction="above",
            horizon_region="full",
            operator="weighted",
            rationale="Reviewed histories support this retained supplier.",
            failure_condition="Reviewed histories stop supporting this supplier.",
        )
        for name in ("toto_2_0", "seasonal_naive")
    )
    recipe = ChampionRecipe(
        name="retained_weighted",
        kind="weighted",
        parents=("toto_2_0", "seasonal_naive"),
        fallback_parent="toto_2_0",
        assumptions=assumptions,
    )

    def policy(weight: float) -> FittedChampionPolicy:
        return FittedChampionPolicy(
            recipe=recipe,
            thresholds=tuple(
                (assumption.assumption_id, 0.0) for assumption in assumptions
            ),
            weights=(weight, 1.0 - weight),
        )

    return NumericalAlternativeSpec(
        candidate_id=recipe.name,
        family="combined",
        materializer_kind="champion",
        recipe_payload=recipe.to_payload(),
        full_build_policy_payload=policy(0.5).to_payload(),
        build_fold_policy_payloads=tuple(
            (fold, policy(0.40 + fold * 0.01).to_payload()) for fold in range(5)
        ),
        assumption_ids=tuple(item.assumption_id for item in assumptions),
        failure_conditions=tuple(item.failure_condition for item in assumptions),
    )


def _seed_only_retained_combined_spec() -> NumericalAlternativeSpec:
    retained = _retained_combined_spec()
    payload = retained.to_payload()
    return NumericalAlternativeSpec(
        candidate_id=retained.candidate_id,
        family=retained.family,
        materializer_kind=retained.materializer_kind,
        recipe_payload=payload["recipe_payload"],
        full_build_policy_payload=payload["full_build_policy_payload"],
        build_fold_policy_payloads=tuple(
            (fold, payload["full_build_policy_payload"]) for fold in range(5)
        ),
        assumption_ids=retained.assumption_ids,
        failure_conditions=retained.failure_conditions,
    )


def test_materializer_rematerializes_every_retained_parent_alternative() -> None:
    build_tasks = _tasks()
    manifest = build_group_fold_manifest(build_tasks, seed=20260903)
    fit = fit_numerical_recipe(_recipe(), _build_rows(build_tasks), manifest, _parent())
    parent = NumericalSupplyRelease(
        schema_version=1,
        version="n000",
        parent_sha256=None,
        anchor_release_payload=_parent().to_payload(),
        alternatives=(_retained_combined_spec(),),
        atlas_release_sha256=None,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
    )
    original_tasks = _evolution_tasks()
    materializer = NumericalPackageMaterializer(
        forecast_store=_FixtureForecastStore(),
        screening_policy=_screening(),
        fold_manifest=manifest,
        original_tasks=original_tasks,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
        diagnostics_registry=FrozenNumericalDiagnosticsRegistry.build(
            original_tasks, _history_diagnostics(original_tasks), HindcastConfig()
        ),
    )

    candidate = materializer.materialize(
        parent,
        fit,
        _label_free(original_tasks),
        version="n001",
        generation=0,
    )

    assert {
        item.name
        for item in candidate.registry.package_for(
            original_tasks[0]
        ).ranked_alternatives
    } == {
        "toto_2_0",
        "select_seasonal_naive",
        "retained_weighted",
    }


def test_materializer_emits_v2_without_collapsing_repeated_family_parent_catalog() -> None:
    build_tasks = _tasks()
    manifest = build_group_fold_manifest(build_tasks, seed=20260903)
    fit = fit_numerical_recipe(_recipe(), _build_rows(build_tasks), manifest, _parent())
    retained = _retained_combined_spec()
    retained_payload = retained.to_payload()
    repeated_family = NumericalAlternativeSpec(
        candidate_id="retained_weighted_2",
        family="combined",
        materializer_kind="champion",
        recipe_payload=retained_payload["recipe_payload"],
        full_build_policy_payload=retained_payload["full_build_policy_payload"],
        build_fold_policy_payloads=tuple(
            (fold, policy)
            for fold, policy in retained_payload["build_fold_policy_payloads"]
        ),
        assumption_ids=retained.assumption_ids,
        failure_conditions=retained.failure_conditions,
    )
    parent = NumericalSupplyRelease(
        schema_version=2,
        version="n000",
        parent_sha256=None,
        anchor_release_payload=_parent().to_payload(),
        alternatives=(retained, repeated_family),
        atlas_release_sha256=None,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
    )
    original_tasks = _evolution_tasks()
    materializer = NumericalPackageMaterializer(
        forecast_store=_FixtureForecastStore(),
        screening_policy=_screening(),
        fold_manifest=manifest,
        original_tasks=original_tasks,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
        diagnostics_registry=FrozenNumericalDiagnosticsRegistry.build(
            original_tasks, _history_diagnostics(original_tasks), HindcastConfig()
        ),
    )

    candidate = materializer.materialize(
        parent,
        fit,
        _label_free(original_tasks),
        version="n001",
        generation=0,
    )

    assert candidate.release.schema_version == 2
    assert tuple(item.candidate_id for item in candidate.release.alternatives) == (
        "retained_weighted",
        "retained_weighted_2",
        _recipe().name,
    )


def test_materializer_refits_and_preserves_v2_seed_catalog_in_first_child() -> None:
    build_tasks = _tasks()
    manifest = build_group_fold_manifest(build_tasks, seed=20260903)
    fit = fit_numerical_recipe(_recipe(), _build_rows(build_tasks), manifest, _parent())
    parent = NumericalSupplyRelease(
        schema_version=2,
        version="n000",
        parent_sha256=None,
        anchor_release_payload=_parent().to_payload(),
        alternatives=(_seed_only_retained_combined_spec(),),
        atlas_release_sha256=None,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
    )
    original_tasks = _evolution_tasks()
    materializer = NumericalPackageMaterializer(
        forecast_store=_FixtureForecastStore(),
        screening_policy=_screening(),
        fold_manifest=manifest,
        original_tasks=original_tasks,
        source_fingerprints={"dictionary": "2" * 64},
        runtime_fingerprints={"materializer": "3" * 64},
        build_rows=_build_rows(build_tasks),
        diagnostics_registry=FrozenNumericalDiagnosticsRegistry.build(
            original_tasks, _history_diagnostics(original_tasks), HindcastConfig()
        ),
    )

    candidate = materializer.materialize(
        parent,
        fit,
        _label_free(original_tasks),
        version="n001",
        generation=0,
    )

    assert tuple(item.candidate_id for item in candidate.release.alternatives) == (
        "retained_weighted",
        _recipe().name,
    )


def test_diagnostics_registry_is_frozen_and_bound_to_history_and_policy() -> None:
    registry_type = getattr(
        numerical_evolution, "FrozenNumericalDiagnosticsRegistry", None
    )
    assert registry_type is not None
    tasks = _evolution_tasks()
    safe_tasks = _label_free(tasks)
    diagnostics = _history_diagnostics(tasks)
    policy = HindcastConfig()
    registry = registry_type.build(tasks, diagnostics, policy)
    fingerprint = registry.fingerprint
    first_task = tasks[0].numeric.task_id
    diagnostics[first_task]["seasonal_naive"] = replace(
        diagnostics[first_task]["seasonal_naive"],
        median_smae=9.0,
    )

    frozen = registry.diagnostics_for(safe_tasks[0], policy)

    assert registry.fingerprint == fingerprint
    assert frozen["seasonal_naive"].median_smae == 0.2
    changed_history = replace(
        safe_tasks[0],
        numeric=replace(
            safe_tasks[0].numeric,
            history_values=tuple(
                value + 1.0 for value in safe_tasks[0].numeric.history_values
            ),
        ),
    )
    with pytest.raises(NumericalPackageEvolutionError, match="history"):
        registry.diagnostics_for(changed_history, policy)
    with pytest.raises(NumericalPackageEvolutionError, match="hindcast"):
        registry.diagnostics_for(
            safe_tasks[0],
            HindcastConfig(folds=4, min_successful_folds=2),
        )


def test_materializer_rejects_unregistered_diagnostics_mappings() -> None:
    build_tasks = _tasks()
    manifest = build_group_fold_manifest(build_tasks, seed=20260903)

    with pytest.raises(NumericalPackageEvolutionError, match="frozen.*diagnostics"):
        NumericalPackageMaterializer(
            forecast_store=_FixtureForecastStore(),
            screening_policy=_screening(),
            fold_manifest=manifest,
            original_tasks=_evolution_tasks(),
            source_fingerprints={"dictionary": "2" * 64},
            runtime_fingerprints={"materializer": "3" * 64},
            diagnostics_registry=_history_diagnostics(_evolution_tasks()),
        )
