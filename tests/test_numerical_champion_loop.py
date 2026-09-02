"""Runtime integration contract for a frozen Numerical Champion."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace

import pytest
import numerical_agent.evolution.numerical_loop as loop_module
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
    champion_fingerprint,
)
from numerical_agent.evolution.champion_runtime import (
    ChampionExecution,
    execute_champion as real_execute,
)
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_loop import run_numerical_loop
from numerical_agent.evolution.numerical_package import NumericalForecastPackage
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    SelectionArithmetic,
    SelectionDecision,
    replay_selection_forecast,
)
from numerical_agent.evolution.screening import (
    ApplicabilityClause,
    ApplicabilityPolicy,
    FeatureTest,
    ScreeningEntry,
    ScreeningPolicy,
)
from numerical_agent.evolution.task_local_ensemble import (
    TaskLocalEnsembleRelease,
    TaskLocalTournamentPolicy,
)
from numerical_agent.evolution.task_local_confidence import (
    ConfidenceEvidenceRecord,
    ConfidencePolicy,
    HierarchicalEvidenceBank,
    WeightRecipe,
    beta_win_probability,
)


def _task() -> Task:
    return Task("champion-runtime", (1.0, 2.0, 3.0) * 12, 2, "D", ())


def _screening() -> ScreeningPolicy:
    entries = (
        ScreeningEntry("safe_anchor", "tsfm", "keep", ApplicabilityPolicy(), "fixture"),
        ScreeningEntry(
            "specialist", "statistical", "keep", ApplicabilityPolicy(), "fixture"
        ),
    )
    return ScreeningPolicy(entries, ("safe_anchor",))


def _diagnostics() -> dict[str, CandidateDiagnostics]:
    truth = (1.0, 2.0)
    return {
        name: CandidateDiagnostics.synthetic(
            name=name,
            family=family,
            median_mase=error,
            fold_forecasts=(forecast,) * 3,
            fold_truths=(truth,) * 3,
            median_smae=error,
            recent_smae=error,
            worst_smae=error,
            median_srmse=error,
            recent_srmse=error,
            worst_srmse=error,
            worst_smae_raw=error,
            worst_srmse_raw=error,
        )
        for name, family, error, forecast in (
            ("safe_anchor", "tsfm", 0.1, (7.0, 7.0)),
            ("specialist", "statistical", 0.2, (1.0, 2.0)),
        )
    }


def _release(*, lineage: tuple[str, ...] = ("champion_a",)) -> ChampionRelease:
    assumption = EvolutionAssumption(
        assumption_id="history_ready",
        candidate_name="specialist",
        feature="history_length",
        direction="above",
        horizon_region="full",
        operator="select",
        rationale="The reviewed specialist applies to this history length.",
        failure_condition="The history length leaves the reviewed range.",
    )
    return ChampionRelease(
        policy=FittedChampionPolicy(
            recipe=ChampionRecipe(
                name="select_specialist",
                kind="select",
                parents=("specialist",),
                fallback_parent="specialist",
                assumptions=(assumption,),
            ),
            thresholds=(("history_ready", 0.0),),
        ),
        source_hashes=(("dictionary", hashlib.sha256(b"reviewed").hexdigest()),),
        metric_policy_fingerprint="1" * 64,
        lineage=lineage,
    )


def _run(
    *,
    release: ChampionRelease | None = None,
    calls: Counter[str] | None = None,
    screening: ScreeningPolicy | None = None,
    task_local_release: TaskLocalEnsembleRelease | None = None,
    diagnostics: dict[str, CandidateDiagnostics] | None = None,
):
    runner_calls = calls if calls is not None else Counter()

    def runner(name: str, _history: tuple[float, ...], horizon: int, _frequency: str):
        runner_calls[name] += 1
        return (
            (7.0,) * horizon
            if name == "safe_anchor"
            else tuple(float(i + 1) for i in range(horizon))
        )

    return run_numerical_loop(
        _task(),
        screening_policy=_screening() if screening is None else screening,
        candidate_runner=runner,
        diagnostics=_diagnostics() if diagnostics is None else diagnostics,
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        champion_release=release,
        task_local_release=task_local_release,
    )


def _task_local_release(anchor: ChampionRelease) -> TaskLocalEnsembleRelease:
    return TaskLocalEnsembleRelease(
        schema_version=1,
        anchor_release_sha256=champion_fingerprint(anchor),
        anchor_name="specialist",
        policy=TaskLocalTournamentPolicy(anchor_name="specialist"),
        default_candidate_names=("specialist", "safe_anchor"),
        group_supplies=(),
        grouping_fingerprint="0" * 64,
        oof_report_sha256="1" * 64,
        source_hashes=anchor.source_hashes,
        metric_policy_fingerprint=anchor.metric_policy_fingerprint,
        lineage=("task_local_v1",),
    )


def test_horizon_weighted_arithmetic_replays_early_and_late_weights() -> None:
    arithmetic = SelectionArithmetic(
        "horizon_weighted",
        inputs=(
            SelectionArithmetic("leaf", candidate_name="toto_2_0"),
            SelectionArithmetic("leaf", candidate_name="seasonal_naive"),
        ),
        horizon_stops=(2, 4),
        region_weights=((0.6, 0.4), (1.0, 0.0)),
    )
    decision = SelectionDecision(
        mode="combined",
        selected=("toto_2_0", "seasonal_naive"),
        weights=(0.8, 0.2),
        forecast=(8.8, 8.8, 20.0, 20.0),
        confidence=0.0,
        reason_codes=("task_local_ensemble", "activated"),
        rejected={},
        combination_type="task_local_horizon_weighted",
        arithmetic=arithmetic,
    )

    replayed = replay_selection_forecast(
        decision,
        {
            "toto_2_0": (8.0, 8.0, 20.0, 20.0),
            "seasonal_naive": (10.0, 10.0, 0.0, 0.0),
        },
    )

    assert replayed == (8.8, 8.8, 20.0, 20.0)


def test_horizon_weighted_arithmetic_rejects_noncovering_or_unnormalized_rows() -> None:
    leaves = (
        SelectionArithmetic("leaf", candidate_name="toto_2_0"),
        SelectionArithmetic("leaf", candidate_name="seasonal_naive"),
    )
    with pytest.raises(ValueError, match="region weights"):
        SelectionArithmetic(
            "horizon_weighted",
            inputs=leaves,
            horizon_stops=(2, 4),
            region_weights=((0.6, 0.4), (0.8, 0.3)),
        )


def test_v2_regional_package_replays_and_materializes_each_leaf_once() -> None:
    champion = _release()
    calls: Counter[str] = Counter()
    task = Task("regional", (1.0, 2.0, 3.0) * 12, 4, "D", ())
    truth = (7.0, 7.0, 3.0, 4.0)
    diagnostics = {
        "specialist": CandidateDiagnostics.synthetic(
            name="specialist",
            family="statistical",
            median_mase=1.0,
            fold_forecasts=((1.0, 2.0, 3.0, 4.0),) * 5,
            fold_truths=(truth,) * 5,
            median_smae=1.0,
            median_srmse=1.0,
        ),
        "safe_anchor": CandidateDiagnostics.synthetic(
            name="safe_anchor",
            family="tsfm",
            median_mase=1.0,
            fold_forecasts=((7.0, 7.0, 0.0, 0.0),) * 5,
            fold_truths=(truth,) * 5,
            median_smae=1.0,
            median_srmse=1.0,
        ),
    }
    confidence_policy = ConfidencePolicy(
        exact_minimum_support=8,
        coarse_minimum_support=8,
        global_minimum_support=8,
    )
    recipe = WeightRecipe("early", ("specialist", "safe_anchor"), (5, 5))
    evidence = HierarchicalEvidenceBank.build(
        confidence_policy,
        records=(
            ConfidenceEvidenceRecord(
                level="global",
                group_key=HierarchicalEvidenceBank.global_group_key(),
                recipe=recipe,
                independent_groups=8,
                task_support=8,
                wins=7,
                ties=0,
                losses=1,
                posterior_win_probability=beta_win_probability(7, 1),
                robust_margin_smae=0.1,
                robust_margin_srmse=0.1,
                p90_regret_smae_raw=0.0,
                p90_regret_srmse_raw=0.0,
                failure_count=0,
                clipped_smae_count=0,
                clipped_srmse_count=0,
            ),
        ),
        fit_group_ids=tuple(f"{index:064x}" for index in range(1, 9)),
    )
    release = TaskLocalEnsembleRelease(
        schema_version=2,
        anchor_release_sha256=champion_fingerprint(champion),
        anchor_name="specialist",
        policy=TaskLocalTournamentPolicy(anchor_name="specialist"),
        default_candidate_names=("specialist", "safe_anchor"),
        group_supplies=(),
        grouping_fingerprint="0" * 64,
        oof_report_sha256="1" * 64,
        source_hashes=champion.source_hashes,
        metric_policy_fingerprint=champion.metric_policy_fingerprint,
        lineage=("task_local_confidence_v2",),
        confidence_evidence=evidence,
    )

    def runner(
        name: str, _history: tuple[float, ...], horizon: int, _frequency: str
    ) -> tuple[float, ...]:
        calls[name] += 1
        return (
            tuple(float(index + 1) for index in range(horizon))
            if name == "specialist"
            else (7.0,) * horizon
        )

    package = run_numerical_loop(
        task,
        screening_policy=_screening(),
        candidate_runner=runner,
        diagnostics=diagnostics,
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        champion_release=champion,
        task_local_release=release,
    )

    assert package.final_forecast == (4.0, 4.5, 3.0, 4.0)
    assert package.selection_decision.arithmetic is not None
    assert package.selection_decision.arithmetic.operation == "horizon_weighted"
    assert replay_selection_forecast(
        package.selection_decision,
        {item.name: item.forecast for item in package.ranked_alternatives},
    ) == package.final_forecast
    assert calls == Counter({"safe_anchor": 1, "specialist": 1})
    assert set(package.component_fingerprints) >= {
        "task_local_confidence",
        "task_local_result",
    }


def test_task_local_release_replays_package_without_rerunning_leaves() -> None:
    champion = _release()
    calls: Counter[str] = Counter()
    truth = (1.0, 2.0)
    diagnostics = {
        "specialist": CandidateDiagnostics.synthetic(
            name="specialist",
            family="statistical",
            median_mase=1.0,
            fold_forecasts=((0.0, 0.0),) * 3,
            fold_truths=(truth,) * 3,
            median_smae=1.0,
            median_srmse=1.0,
        ),
        "safe_anchor": CandidateDiagnostics.synthetic(
            name="safe_anchor",
            family="tsfm",
            median_mase=0.0,
            fold_forecasts=(truth,) * 3,
            fold_truths=(truth,) * 3,
            median_smae=0.0,
            median_srmse=0.0,
        ),
    }

    package = _run(
        release=champion,
        task_local_release=_task_local_release(champion),
        diagnostics=diagnostics,
        calls=calls,
    )

    assert package.selection_decision.selected == ("specialist", "safe_anchor")
    assert package.selection_decision.weights == (0.5, 0.5)
    assert package.final_forecast == (4.0, 4.5)
    assert calls == Counter({"safe_anchor": 1, "specialist": 1})
    assert set(package.component_fingerprints) >= {
        "task_local_release",
        "task_local_policy",
        "task_local_group_supply",
    }


def test_legacy_champion_path_is_unchanged_without_task_local_release() -> None:
    first = _run(release=_release())
    second = _run(release=_release(), task_local_release=None)

    assert first == second
    assert "task_local_release" not in first.component_fingerprints


def test_champion_selects_only_an_already_materialized_ranked_forecast(
    monkeypatch,
) -> None:
    # This fails if Champion execution receives hindcast truth or if it reruns a leaf.
    observed: list[object] = []

    def checked_execute(policy, forecasts, diagnostics, profile, history, horizon):
        observed.append((forecasts, diagnostics, profile, history, horizon))
        assert all(item.folds == () for item in diagnostics.values())
        assert all(item.fold_forecasts == () for item in diagnostics.values())
        assert all(item.fold_truths == () for item in diagnostics.values())
        assert all(item.long_horizon_fold is None for item in diagnostics.values())
        return real_execute(policy, forecasts, diagnostics, profile, history, horizon)

    monkeypatch.setattr(loop_module, "execute_champion", checked_execute)
    calls: Counter[str] = Counter()

    package = _run(release=_release(), calls=calls)

    assert package.selection_decision.selected == ("specialist",)
    assert package.selection_decision.selected[0] in {
        item.name for item in package.ranked_alternatives
    }
    assert package.final_forecast == (1.0, 2.0)
    assert calls == Counter({"safe_anchor": 1, "specialist": 1})
    assert len(observed) == 1
    assert package.selection_decision.assumption_ids == ("history_ready",)
    assert set(package.component_fingerprints) >= {
        "champion_release",
        "champion_recipe",
        "champion_assumptions",
    }
    assert not hasattr(package, "champion_release")
    assert not hasattr(package, "champion_thresholds")


def test_champion_release_fingerprint_changes_package_identity() -> None:
    first = _run(release=_release(lineage=("champion_a",)))
    second = _run(release=_release(lineage=("champion_b",)))

    assert (
        first.component_fingerprints["champion_release"]
        != second.component_fingerprints["champion_release"]
    )
    assert (
        first.component_fingerprints["champion_recipe"]
        == second.component_fingerprints["champion_recipe"]
    )
    assert (
        first.component_fingerprints["champion_assumptions"]
        == second.component_fingerprints["champion_assumptions"]
    )


def test_package_rejects_partial_or_non_hash_champion_fingerprints() -> None:
    package = _run(release=_release())

    with pytest.raises(ValueError, match="Champion component fingerprints"):
        replace(
            package,
            component_fingerprints={
                **dict(package.component_fingerprints),
                "champion_release": "not-a-sha256",
            },
        )
    with pytest.raises(ValueError, match="Champion component fingerprints"):
        replace(
            package,
            component_fingerprints={
                key: value
                for key, value in package.component_fingerprints.items()
                if key != "champion_assumptions"
            },
        )


def test_legacy_package_keeps_legacy_named_component_fingerprints() -> None:
    package = _run()

    legacy = replace(
        package,
        component_fingerprints={
            **dict(package.component_fingerprints),
            "champion_release": "legacy-tag",
        },
    )

    assert legacy.component_fingerprints["champion_release"] == "legacy-tag"
    assert legacy == replace(
        package,
        component_fingerprints={
            **dict(package.component_fingerprints),
            "champion_release": "legacy-tag",
        },
    )


def test_legacy_package_never_infers_champion_provenance_from_contents() -> None:
    legacy = _run()
    champion = _run(release=_release())

    collision = replace(
        legacy,
        selection_decision=replace(
            legacy.selection_decision,
            reason_codes=("frozen_champion", "legacy_reason"),
        ),
        component_fingerprints={
            **dict(legacy.component_fingerprints),
            **{
                key: champion.component_fingerprints[key]
                for key in (
                    "champion_release",
                    "champion_recipe",
                    "champion_assumptions",
                )
            },
        },
    )

    assert type(collision) is NumericalForecastPackage
    assert collision.selection_decision.reason_codes == (
        "frozen_champion",
        "legacy_reason",
    )


def test_champion_package_subtype_preserves_its_provenance_invariant() -> None:
    champion = _run(release=_release())

    assert type(champion) is not NumericalForecastPackage
    assert type(replace(champion)) is type(champion)
    with pytest.raises(ValueError, match="fixed provenance marker"):
        replace(
            champion,
            selection_decision=replace(
                champion.selection_decision,
                reason_codes=("legacy_reason",),
            ),
        )
    with pytest.raises(ValueError, match="Champion component fingerprints"):
        replace(
            champion,
            component_fingerprints={
                key: value
                for key, value in champion.component_fingerprints.items()
                if key != "champion_assumptions"
            },
        )


class _FloatSubclass(float):
    pass


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "malformed",
    (
        pytest.param(
            ChampionExecution((10**10000, 2.0), ("specialist",), (), None),
            id="huge_int",
        ),
        ChampionExecution((_FloatSubclass(1.0), 2.0), ("specialist",), (), None),
        ChampionExecution(("1.0", 2.0), ("specialist",), (), None),
        ChampionExecution((object(), 2.0), ("specialist",), (), None),
        ChampionExecution((1.0, 2.0), (_StringSubclass("specialist"),), (), None),
    ),
)
def test_hostile_champion_execution_values_never_escape_fallback(
    monkeypatch, malformed: ChampionExecution
) -> None:
    monkeypatch.setattr(loop_module, "execute_champion", lambda *_args: malformed)

    package = _run(release=_release())

    assert package.selection_decision.selected == ("specialist",)
    assert package.fallback_reason == "champion_execution_failed:invalid_result"


@pytest.mark.parametrize(
    "malformed",
    (
        ChampionExecution((1, 2), None, (), None),  # type: ignore[arg-type]
        ChampionExecution((1.0,), ("specialist",), (), None),
        ChampionExecution((1.0, 2.0), ("unknown",), (), None),
        ChampionExecution((1.0, 2.0), ("specialist",), ("",), None),
        ChampionExecution((1.0, 2.0), ("specialist",), ("not an id",), None),
        ChampionExecution((1.0, 2.0), ("specialist",), (), ""),
    ),
)
def test_malformed_champion_execution_fields_use_fixed_materialized_fallback(
    monkeypatch, malformed: ChampionExecution
) -> None:
    monkeypatch.setattr(loop_module, "execute_champion", lambda *_args: malformed)

    package = _run(release=_release())

    assert package.selection_decision.selected == ("specialist",)
    assert package.final_forecast == (1.0, 2.0)
    assert package.fallback_reason == "champion_execution_failed:invalid_result"


def test_public_champion_package_hides_recipe_operator_and_threshold_text() -> None:
    release = _release()

    package = _run(release=release)

    assert package.selection_decision.reason_codes == (
        "frozen_champion",
        "champion_materialized_selection",
    )
    assert package.selection_decision.assumption_ids == ("history_ready",)
    assert all(
        value != release.policy.recipe.kind
        for value in package.selection_decision.reason_codes
    )
    assert all(
        "threshold" not in value for value in package.selection_decision.reason_codes
    )
    assert all(
        len(package.component_fingerprints[key]) == 64
        for key in (
            "champion_release",
            "champion_recipe",
            "champion_assumptions",
        )
    )
    assert not hasattr(package, "champion_thresholds")


def test_champion_arithmetic_is_host_materialized_as_a_ranked_candidate() -> None:
    release = _release()
    weighted = replace(
        release,
        policy=FittedChampionPolicy(
            recipe=ChampionRecipe(
                name="weighted_runtime_vector",
                kind="weighted",
                parents=("safe_anchor", "specialist"),
                fallback_parent="safe_anchor",
                assumptions=(
                    replace(
                        release.policy.recipe.assumptions[0],
                        candidate_name="specialist",
                        operator="weighted",
                    ),
                ),
            ),
            thresholds=(("history_ready", 0.0),),
            weights=(0.5, 0.5),
        ),
    )

    package = _run(release=weighted)

    assert package.selection_decision.selected == ("weighted_runtime_vector",)
    assert package.final_forecast == (4.0, 4.5)
    assert package.fallback_reason is None
    materialized = {
        item.name: item.forecast for item in package.ranked_alternatives
    }
    assert materialized["weighted_runtime_vector"] == (4.0, 4.5)
    assert package.candidate_diagnostics[
        "weighted_runtime_vector"
    ].reason_code == "frozen_champion_release"


def test_champion_exact_fallback_is_materialized_even_when_screening_excludes_it() -> None:
    screening = ScreeningPolicy(
        (
            ScreeningEntry(
                "safe_anchor",
                "tsfm",
                "specialized",
                ApplicabilityPolicy(
                    (
                        ApplicabilityClause(
                            feature_tests=(FeatureTest("history_length", ">", 1000),)
                        ),
                    )
                ),
                "not active for this short history",
            ),
            ScreeningEntry(
                "specialist",
                "statistical",
                "keep",
                ApplicabilityPolicy(),
                "fixture",
            ),
        ),
        ("specialist",),
    )
    release = _release()
    exact_fallback = replace(
        release,
        policy=FittedChampionPolicy(
            recipe=ChampionRecipe(
                name="safe_anchor_release",
                kind="select",
                parents=("safe_anchor",),
                fallback_parent="safe_anchor",
                assumptions=(
                    replace(
                        release.policy.recipe.assumptions[0],
                        candidate_name="safe_anchor",
                    ),
                ),
            ),
            thresholds=(("history_ready", 1000.0),),
        ),
    )
    calls: Counter[str] = Counter()

    package = _run(release=exact_fallback, calls=calls, screening=screening)

    assert package.selection_decision.selected == ("safe_anchor",)
    assert package.final_forecast == (7.0, 7.0)
    assert package.fallback_reason == "champion_execution_fallback"
    assert calls == Counter({"safe_anchor": 1, "specialist": 1})
