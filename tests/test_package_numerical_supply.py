from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace

import pytest

from common.data import Task as ContextNumericTask
from evolving_loop.data import ContextTask
from evolving_loop.package_numerical_supply import (
    NumericalAlternativeSpec,
    NumericalSupplyError,
    NumericalSupplyRelease,
    bound_numerical_package,
    build_package_registry,
    parse_numerical_supply_release,
)
from evolving_loop.package_registry import (
    FrozenNumericalPackageRegistry,
    PackageRegistryError,
)
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
    champion_fingerprint,
)
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_handoff import task_input_fingerprint
from numerical_agent.evolution.numerical_package import (
    NumericalForecastPackage,
    RankedNumericalForecast,
)
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    SelectionDecision,
)
from numerical_agent.evolution.screening import profile_task


def _policy(candidate_id: str, threshold: float = 0.0) -> FittedChampionPolicy:
    assumption = EvolutionAssumption(
        assumption_id=f"{candidate_id}_ready",
        candidate_name=candidate_id,
        feature="history_length",
        direction="above",
        horizon_region="full",
        operator="select",
        rationale="The candidate is valid for the reviewed history.",
        failure_condition="The candidate no longer matches the history.",
    )
    return FittedChampionPolicy(
        recipe=ChampionRecipe(
            name=f"select_{candidate_id}",
            kind="select",
            parents=(candidate_id,),
            fallback_parent=candidate_id,
            assumptions=(assumption,),
        ),
        thresholds=((assumption.assumption_id, threshold),),
    )


def _anchor_release() -> ChampionRelease:
    return ChampionRelease(
        policy=_policy("safe_anchor"),
        source_hashes=(("dictionary", hashlib.sha256(b"reviewed").hexdigest()),),
        metric_policy_fingerprint="1" * 64,
        lineage=("safe_anchor",),
    )


def _alternative(candidate_id: str, family: str) -> NumericalAlternativeSpec:
    policy = _policy(candidate_id)
    return NumericalAlternativeSpec(
        candidate_id=candidate_id,
        family=family,
        materializer_kind={
            "statistical": "dictionary",
            "tsfm": "champion",
            "combined": "bounded_overlay",
            "atlas_overlay": "atlas",
        }[family],
        recipe_payload=policy.recipe.to_payload(),
        full_build_policy_payload=policy.to_payload(),
        build_fold_policy_payloads=tuple(
            (fold, _policy(candidate_id, float(fold + 1)).to_payload())
            for fold in range(5)
        ),
        assumption_ids=(f"{candidate_id}_ready",),
        failure_conditions=("The candidate no longer matches the history.",),
    )


def _supply_release(
    *, alternatives: tuple[NumericalAlternativeSpec, ...] | None = None
) -> NumericalSupplyRelease:
    if alternatives is None:
        alternatives = (
            _alternative("seasonal_naive", "statistical"),
            _alternative("toto_2_0", "tsfm"),
            _alternative("weighted_pair", "combined"),
            _alternative("atlas_70_30", "atlas_overlay"),
        )
    return NumericalSupplyRelease(
        schema_version=1,
        version="n001",
        parent_sha256="2" * 64,
        anchor_release_payload=_anchor_release().to_payload(),
        alternatives=alternatives,
        atlas_release_sha256="3" * 64,
        source_fingerprints={"dictionary": "4" * 64},
        runtime_fingerprints={"materializer": "5" * 64},
    )


def _ranked(name: str, family: str, forecast: Sequence[float]) -> RankedNumericalForecast:
    values = tuple(float(value) for value in forecast)
    return RankedNumericalForecast(
        rank=1,
        name=name,
        family=family,
        forecast=values,
        diagnostics=CandidateDiagnostics.synthetic(
            name=name,
            family=family,
            median_mase=0.1,
            fold_forecasts=(values,) * 3,
            fold_truths=((1.0, 2.0),) * 3,
            median_smae=0.1,
            recent_smae=0.1,
            worst_smae=0.1,
            median_srmse=0.1,
            recent_srmse=0.1,
            worst_srmse=0.1,
            worst_smae_raw=0.1,
            worst_srmse_raw=0.1,
        ),
    )


def _wide_package() -> NumericalForecastPackage:
    alternatives = (
        _ranked("safe_anchor", "tsfm", (9.0, 9.0)),
        _ranked("seasonal_naive", "statistical", (2.0, 2.0)),
        _ranked("toto_2_0", "tsfm", (3.0, 3.0)),
        _ranked("weighted_pair", "combined", (4.0, 4.0)),
        _ranked("atlas_70_30", "atlas_overlay", (5.0, 5.0)),
    )
    anchor = alternatives[0]
    return NumericalForecastPackage(
        task_profile=profile_task(Task("supply-runtime", (1.0, 2.0, 3.0) * 12, 2, "D", ())),
        active_candidate_names=tuple(item.name for item in alternatives),
        candidate_diagnostics={item.name: item.diagnostics for item in alternatives},
        morphology_card=None,
        accepted_assumptions=(),
        rejected_assumptions={},
        selection_decision=SelectionDecision(
            mode="single",
            selected=(anchor.name,),
            weights=(1.0,),
            forecast=anchor.forecast,
            confidence=0.0,
            reason_codes=("frozen_champion",),
            rejected={},
            baseline_name=anchor.name,
            considered_candidates=tuple(item.name for item in alternatives),
        ),
        final_forecast=anchor.forecast,
        protected_baseline=anchor,
        ranked_alternatives=tuple(
            RankedNumericalForecast(
                rank=index,
                name=item.name,
                family=item.family,
                forecast=item.forecast,
                diagnostics=item.diagnostics,
            )
            for index, item in enumerate(alternatives, start=1)
        ),
        retrieval_handoff=(),
        component_fingerprints={
            "task_input": "6" * 64,
            "champion_release": champion_fingerprint(_anchor_release()),
            "champion_recipe": champion_fingerprint(_anchor_release().policy.recipe),
            "champion_assumptions": champion_fingerprint(
                _anchor_release().policy.recipe.assumptions
            ),
        },
    )


def _materialized_forecasts() -> dict[str, RankedNumericalForecast]:
    return {
        "safe_anchor": _ranked("safe_anchor", "tsfm", (9.0, 9.0)),
        "seasonal_naive": _ranked("seasonal_naive", "statistical", (2.0, 2.0)),
        "toto_2_0": _ranked("toto_2_0", "tsfm", (3.0, 3.0)),
        "weighted_pair": _ranked("weighted_pair", "combined", (4.0, 4.0)),
        "atlas_70_30": _ranked("atlas_70_30", "atlas_overlay", (5.0, 5.0)),
    }


def _registry_tasks() -> tuple[ContextTask, ContextTask]:
    return tuple(
        ContextTask(
            numeric=ContextNumericTask(
                task_id=f"registry-{index}",
                history_values=(1.0, 2.0, 3.0) * 12,
                future_values=(1.0, 2.0),
                prediction_length=2,
                frequency="D",
                seasonal_period=None,
                entity_name=f"Registry Entity {index}",
            ),
            target_name="sales",
            target_description="Daily sales",
            history_timestamps=tuple(
                f"2026-01-{position + 1:02d}" for position in range(36)
            ),
            future_timestamps=("2026-02-06", "2026-02-07"),
            documents=(),
        )
        for index in range(2)
    )  # type: ignore[return-value]


def _package_for_task(
    task: ContextTask, release: NumericalSupplyRelease
) -> NumericalForecastPackage:
    profile = profile_task(
        Task(
            task.numeric.task_id,
            task.numeric.history_values,
            task.numeric.prediction_length,
            task.numeric.frequency,
            (),
        )
    )
    anchor = _anchor_release()
    source = replace(
        _wide_package(),
        task_profile=profile,
        component_fingerprints={
            "task_input": task_input_fingerprint(
                task_id=task.numeric.task_id,
                history=task.numeric.history_values,
                frequency=task.numeric.frequency,
                horizon=task.numeric.prediction_length,
            ),
            "champion_release": champion_fingerprint(anchor),
            "champion_recipe": champion_fingerprint(anchor.policy.recipe),
            "champion_assumptions": champion_fingerprint(
                anchor.policy.recipe.assumptions
            ),
        },
    )
    return bound_numerical_package(source, release, _materialized_forecasts())


def test_supply_release_binds_anchor_parent_and_four_diverse_alternatives():
    release = _supply_release(
        alternatives=(
            _alternative("seasonal_naive", "statistical"),
            _alternative("toto_2_0", "tsfm"),
            _alternative("weighted_pair", "combined"),
            _alternative("atlas_70_30", "atlas_overlay"),
        )
    )

    assert release.version == "n001"
    assert tuple(item.family for item in release.alternatives) == (
        "statistical",
        "tsfm",
        "combined",
        "atlas_overlay",
    )
    assert len(release.alternatives) == 4
    assert len(release.fingerprint) == 64


def test_supply_release_rejects_five_additional_alternatives():
    with pytest.raises(NumericalSupplyError, match="four additional"):
        _supply_release(
            alternatives=tuple(
                _alternative(f"candidate_{index}", "statistical")
                for index in range(5)
            )
        )


def test_schema_v1_supply_rejects_repeated_family_before_catalog_migration() -> None:
    with pytest.raises(NumericalSupplyError, match="one alternative per family"):
        _supply_release(
            alternatives=(
                _alternative("seasonal_naive", "statistical"),
                _alternative("drift", "statistical"),
            )
        )


def test_schema_v1_supply_replay_preserves_legacy_payload_byte_for_byte() -> None:
    release = _supply_release()
    payload = release.to_payload()

    parsed = parse_numerical_supply_release(payload)

    assert payload["schema_version"] == 1
    assert parsed.to_payload() == payload


def test_schema_v2_supply_round_trips_complete_ordered_repeated_family_catalog() -> None:
    families = (
        "statistical",
        "tsfm",
        "combined",
        "atlas_overlay",
    )
    alternatives = tuple(
        _alternative(f"candidate_{index:02d}", families[index % len(families)])
        for index in range(12)
    )
    release = NumericalSupplyRelease(
        schema_version=2,
        version="n001",
        parent_sha256="2" * 64,
        anchor_release_payload=_anchor_release().to_payload(),
        alternatives=alternatives,
        atlas_release_sha256="3" * 64,
        source_fingerprints={"dictionary": "4" * 64},
        runtime_fingerprints={"materializer": "5" * 64},
    )

    parsed = parse_numerical_supply_release(release.to_payload())

    assert parsed.to_payload() == release.to_payload()
    assert tuple(item.candidate_id for item in parsed.alternatives) == tuple(
        f"candidate_{index:02d}" for index in range(12)
    )


def test_non_seed_supply_rejects_full_build_policy_reused_for_every_fold():
    policy = _policy("seasonal_naive")
    repeated_policy = NumericalAlternativeSpec(
        candidate_id="seasonal_naive",
        family="statistical",
        materializer_kind="dictionary",
        recipe_payload=policy.recipe.to_payload(),
        full_build_policy_payload=policy.to_payload(),
        build_fold_policy_payloads=tuple(
            (fold, policy.to_payload()) for fold in range(5)
        ),
        assumption_ids=("seasonal_naive_ready",),
        failure_conditions=("The candidate no longer matches the history.",),
    )

    with pytest.raises(NumericalSupplyError, match="cross-fitted"):
        _supply_release(alternatives=(repeated_policy,))


def test_non_seed_supply_allows_one_cross_fitted_fold_to_equal_full_by_coincidence():
    policy = _policy("seasonal_naive")
    cross_fitted = NumericalAlternativeSpec(
        candidate_id="seasonal_naive",
        family="statistical",
        materializer_kind="dictionary",
        recipe_payload=policy.recipe.to_payload(),
        full_build_policy_payload=policy.to_payload(),
        build_fold_policy_payloads=tuple(
            (
                fold,
                (
                    policy.to_payload()
                    if fold == 0
                    else _policy("seasonal_naive", float(fold)).to_payload()
                ),
            )
            for fold in range(5)
        ),
        assumption_ids=("seasonal_naive_ready",),
        failure_conditions=("The candidate no longer matches the history.",),
    )

    release = _supply_release(alternatives=(cross_fitted,))

    assert release.alternatives == (cross_fitted,)


def test_bounded_package_keeps_anchor_plus_one_per_family_and_deduplicates_vectors():
    source = _wide_package()
    materialized = {
        "safe_anchor": _ranked("safe_anchor", "tsfm", (9.0, 9.0)),
        "seasonal_naive": _ranked("seasonal_naive", "statistical", (2.0, 2.0)),
        "toto_2_0": _ranked("toto_2_0", "tsfm", (9.0, 9.0)),
        "weighted_pair": _ranked("weighted_pair", "combined", (3.0, 3.0)),
        "atlas_70_30": _ranked("atlas_70_30", "atlas_overlay", (4.0, 4.0)),
    }

    package = bound_numerical_package(source, _supply_release(), materialized)

    assert tuple(item.name for item in package.ranked_alternatives) == (
        "safe_anchor",
        "seasonal_naive",
        "weighted_pair",
        "atlas_70_30",
    )
    assert package.selection_decision.selected == ("safe_anchor",)
    assert package.protected_baseline.name == "safe_anchor"
    assert len(package.ranked_alternatives) <= 5


def test_schema_v2_binds_repeated_family_duplicate_vectors_in_release_order() -> None:
    source = _wide_package()
    release = NumericalSupplyRelease(
        schema_version=2,
        version="n001",
        parent_sha256="2" * 64,
        anchor_release_payload=_anchor_release().to_payload(),
        alternatives=(
            _alternative("seasonal_naive", "statistical"),
            _alternative("drift", "statistical"),
        ),
        atlas_release_sha256=None,
        source_fingerprints={"dictionary": "4" * 64},
        runtime_fingerprints={"materializer": "5" * 64},
    )
    materialized = {
        "safe_anchor": _ranked("safe_anchor", "tsfm", (9.0, 9.0)),
        "seasonal_naive": _ranked("seasonal_naive", "statistical", (2.0, 2.0)),
        "drift": _ranked("drift", "statistical", (2.0, 2.0)),
    }

    package = bound_numerical_package(source, release, materialized)

    assert tuple(item.name for item in package.ranked_alternatives) == (
        "safe_anchor",
        "seasonal_naive",
        "drift",
    )


def test_schema_v2_package_uses_verified_task_shortlist_not_full_supply() -> None:
    """The frozen supply remains complete while a task package is local-only."""
    from numerical_agent.evolution.task_shortlist import (
        TaskCandidateShortlistV1,
        TaskShortlistPolicyV1,
    )

    source = _wide_package()
    release = NumericalSupplyRelease(
        schema_version=2,
        version="n001",
        parent_sha256="2" * 64,
        anchor_release_payload=_anchor_release().to_payload(),
        alternatives=(
            _alternative("seasonal_naive", "statistical"),
            _alternative("drift", "statistical"),
            _alternative("weighted_pair", "combined"),
        ),
        atlas_release_sha256=None,
        source_fingerprints={"dictionary": "4" * 64},
        runtime_fingerprints={"materializer": "5" * 64},
    )
    policy = TaskShortlistPolicyV1()
    shortlist = TaskCandidateShortlistV1(
        1, "1" * 64, "4" * 64, policy.fingerprint(),
        ("safe_anchor", "drift"), (), True, False,
    )
    materialized = {
        "safe_anchor": _ranked("safe_anchor", "tsfm", (9.0, 9.0)),
        "seasonal_naive": _ranked("seasonal_naive", "statistical", (2.0, 2.0)),
        "drift": _ranked("drift", "statistical", (3.0, 3.0)),
        "weighted_pair": _ranked("weighted_pair", "combined", (4.0, 4.0)),
    }

    package = bound_numerical_package(source, release, materialized, shortlist=shortlist,
                                      hindcast_diagnostics_sha256="5" * 64)

    assert tuple(item.candidate_id for item in release.alternatives) == (
        "seasonal_naive", "drift", "weighted_pair",
    )
    assert package.active_candidate_names == ("safe_anchor", "drift")
    assert set(package.candidate_diagnostics) == {"safe_anchor", "drift"}
    assert package.component_fingerprints["task_shortlist"] == shortlist.fingerprint()
    assert package.component_fingerprints["shortlist_policy"] == policy.fingerprint()
    assert package.component_fingerprints["dictionary"] == "4" * 64
    assert package.component_fingerprints["hindcast_diagnostics"] == "5" * 64


def test_bounded_package_projects_verified_alternative_assumption_to_safe_handoff():
    source = _wide_package()
    release = _supply_release(
        alternatives=(_alternative("seasonal_naive", "statistical"),)
    )

    package = bound_numerical_package(
        source,
        release,
        _materialized_forecasts(),
        history=(1.0, 2.0, 3.0) * 12,
        task_fold=None,
    )

    assert package.morphology_card is not None
    assert tuple(
        grounding.candidate_names for grounding in package.accepted_assumptions
    ) == (("seasonal_naive",),)
    assert len(package.retrieval_handoff) == 1
    assert set(package.retrieval_handoff[0]) == {
        "assumption_id",
        "kind",
        "claim",
        "failure_condition",
    }
    assert "seasonal_naive" not in json.dumps(
        [dict(item) for item in package.retrieval_handoff]
    )
    assert (
        package.component_fingerprints["morphology_card"]
        == package.morphology_card.fingerprint
    )


def test_bounded_package_uses_release_policy_and_clears_stale_morphology_on_rejection():
    source = _wide_package()
    permissive = _supply_release(
        alternatives=(_alternative("seasonal_naive", "statistical"),)
    )
    populated = bound_numerical_package(
        source,
        permissive,
        _materialized_forecasts(),
        history=(1.0, 2.0, 3.0) * 12,
        task_fold=None,
    )
    strict_policy = _policy("seasonal_naive", 1_000.0)
    base_alternative = _alternative("seasonal_naive", "statistical")
    strict_alternative = NumericalAlternativeSpec(
        candidate_id=base_alternative.candidate_id,
        family=base_alternative.family,
        materializer_kind=base_alternative.materializer_kind,
        recipe_payload=base_alternative.to_payload()["recipe_payload"],
        full_build_policy_payload=strict_policy.to_payload(),
        build_fold_policy_payloads=tuple(
            (fold, payload)
            for fold, payload in base_alternative.to_payload()[
                "build_fold_policy_payloads"
            ]
        ),
        assumption_ids=base_alternative.assumption_ids,
        failure_conditions=base_alternative.failure_conditions,
    )

    rejected = bound_numerical_package(
        populated,
        _supply_release(alternatives=(strict_alternative,)),
        _materialized_forecasts(),
        history=(1.0, 2.0, 3.0) * 12,
        task_fold=None,
    )

    disabled = hashlib.sha256(b'{"enabled":false}').hexdigest()
    assert rejected.morphology_card is None
    assert rejected.accepted_assumptions == ()
    assert rejected.retrieval_handoff == ()
    assert rejected.component_fingerprints["morphology_card"] == disabled


def test_bounded_package_rejects_a_materialized_anchor_with_a_different_forecast():
    materialized = {
        **_materialized_forecasts(),
        "safe_anchor": _ranked("safe_anchor", "tsfm", (1.0, 1.0)),
    }
    with pytest.raises(NumericalSupplyError, match="exact protected anchor"):
        bound_numerical_package(
            _wide_package(),
            _supply_release(),
            materialized,
        )


def test_fixed_atlas_blend_is_never_promoted_to_anchor():
    package = bound_numerical_package(
        _wide_package(),
        _supply_release(),
        _materialized_forecasts(),
    )

    assert package.protected_baseline.name != "atlas_70_30"


def test_fixed_atlas_blend_is_rejected_as_the_source_anchor():
    source = _wide_package()
    atlas = next(
        item for item in source.ranked_alternatives if item.name == "atlas_70_30"
    )

    with pytest.raises(NumericalSupplyError, match="fixed Atlas"):
        bound_numerical_package(
            replace(source, protected_baseline=atlas),
            _supply_release(),
            _materialized_forecasts(),
        )


def test_registry_binds_exact_supply_release_and_task_universe():
    tasks = _registry_tasks()
    release = _supply_release()
    registry = build_package_registry(tasks, release, _package_for_task)

    assert registry.release_sha256 == release.fingerprint
    assert registry.task_ids == tuple(sorted(task.numeric.task_id for task in tasks))
    assert registry.manifest["release_sha256"] == release.fingerprint


def test_registry_builder_rejects_package_champion_provenance_that_drifts_from_anchor():
    tasks = _registry_tasks()
    release = _supply_release()

    def package_builder(task: ContextTask, supply: NumericalSupplyRelease):
        package = _package_for_task(task, supply)
        return replace(
            package,
            component_fingerprints={
                **dict(package.component_fingerprints),
                "champion_release": "0" * 64,
            },
        )

    with pytest.raises(NumericalSupplyError, match="Champion provenance"):
        build_package_registry(tasks, release, package_builder)


def test_registry_release_sha256_is_read_only():
    registry = build_package_registry(
        _registry_tasks(), _supply_release(), _package_for_task
    )

    with pytest.raises(AttributeError):
        registry.release_sha256 = "0" * 64


def test_registry_rejects_missing_or_extra_task_packages():
    tasks = _registry_tasks()
    release = _supply_release()
    packages = tuple(_package_for_task(task, release) for task in tasks)

    with pytest.raises(PackageRegistryError, match="complete task coverage"):
        FrozenNumericalPackageRegistry(
            entries=((tasks[0], packages[0]),),
            release_sha256=release.fingerprint,
            expected_task_ids=(tasks[0].numeric.task_id, tasks[1].numeric.task_id),
        )
