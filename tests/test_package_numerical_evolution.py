"""Numerical package proposal, fitting, and materialization boundaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import pytest

from common.data import Task as DataTask
from evolving_loop.data import ContextTask, Document
from evolving_loop.package_numerical_evolution import (
    NumericalCoordinateCandidate,
    NumericalPackageEvolutionError,
    NumericalPackageMaterializer,
    NumericalPackageProposer,
    fit_numerical_recipe,
)
from evolving_loop.package_numerical_supply import NumericalSupplyRelease
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
)
from numerical_agent.evolution.champion_evidence import ChampionTaskRow
from numerical_agent.evolution.champion_controller import ChampionProposerAdapter
from numerical_agent.evolution.champion_evidence import ProposerEvidence
from numerical_agent.evolution.champion_proposal import expand_recipe
from numerical_agent.evolution.execution import Task as RuntimeTask
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
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


def _tasks(count: int = 64) -> tuple[DataTask, ...]:
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

    with pytest.raises(NumericalPackageEvolutionError, match="Build"):
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
) -> FrozenNumericalPackageRegistry:
    registry = object.__new__(FrozenNumericalPackageRegistry)
    registry._release_sha256 = release.fingerprint
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
                    if index < 64
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
    labeled = tuple(
        ContextTask(
            numeric=task,
            target_name="demand",
            target_description="daily demand",
            history_timestamps=tuple(
                f"h{index}" for index in range(len(task.history_values))
            ),
            future_timestamps=("f0", "f1"),
            documents=(Document("doc", "context", "supporting", "fact"),),
            gt_evidence=("doc",),
            labels_public=True,
        )
        for task in tasks
    )
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
        diagnostics_by_task=diagnostics,
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
    fit = fit_numerical_recipe(
        _recipe(), _build_rows(build_tasks), manifest, _parent()
    )
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
        diagnostics_by_task=diagnostics,
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
    assert "atlas_70_30" in {
        item.name for item in package.ranked_alternatives
    }
