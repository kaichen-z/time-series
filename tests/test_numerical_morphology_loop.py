"""End-to-end contract for the morphology-guided Numerical loop."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import FrozenInstanceError, replace

import pytest

from numerical_agent.evolution import NumericalForecastPackage, run_numerical_loop
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.morphology import (
    AssumptionGrounding,
    MorphologyCard,
    MorphologyObservation,
    MorphologyToolCall,
)
from numerical_agent.evolution.numerical_handoff import safe_retrieval_projection
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    HindcastConfig,
    HindcastFold,
    SelectionDecision,
    replay_selection_forecast,
    select_assumption_guided_forecast,
    select_numerical_forecast,
)
from numerical_agent.evolution.portfolio import CombinedPolicy
from numerical_agent.evolution.screening import (
    ApplicabilityClause,
    ApplicabilityPolicy,
    FeatureTest,
    ScreeningEntry,
    ScreeningPolicy,
    profile_task,
)


def test_numerical_loop_public_api_is_importable() -> None:
    assert NumericalForecastPackage is not None
    assert callable(run_numerical_loop)


def _task(*, horizon: int = 3) -> Task:
    history = tuple(float(value) for value in [1, 2, 3] * 28)
    return Task("inference-1", history, horizon, "D", ())


def _entry(
    name: str,
    family: str,
    *,
    applicability: ApplicabilityPolicy = ApplicabilityPolicy(),
) -> ScreeningEntry:
    return ScreeningEntry(name, family, "keep", applicability, "reviewed test candidate")


def _screening(*entries: ScreeningEntry) -> ScreeningPolicy:
    fallback = tuple(
        name
        for name in ("toto_2_0", "seasonal_specialist")
        if any(entry.name == name for entry in entries)
    )
    return ScreeningPolicy(tuple(entries), fallback)


def _diagnostic(
    name: str,
    family: str,
    *,
    forecast: tuple[float, ...] = (1.0, 2.0, 3.0),
    truth: tuple[float, ...] = (1.0, 2.0, 3.0),
    median_mase: float = 0.2,
    **changes: object,
) -> CandidateDiagnostics:
    return replace(
        CandidateDiagnostics.synthetic(
            name=name,
            family=family,
            median_mase=median_mase,
            fold_forecasts=(forecast,) * 3,
            fold_truths=(truth,) * 3,
        ),
        **changes,
    )


def _audited_diagnostic(
    diagnostic: CandidateDiagnostics,
    *,
    forecast: tuple[float, ...],
    truth: tuple[float, ...],
    scale: float = 1.0,
) -> CandidateDiagnostics:
    folds = tuple(
        HindcastFold(
            train_end=10 * (index + 1),
            validation_end=10 * (index + 1) + len(fold_truth),
            status="success",
            forecast=tuple(float(value) for value in fold_forecast),
            truth=tuple(float(value) for value in fold_truth),
            mase_scale=float(scale),
        )
        for index, (fold_forecast, fold_truth) in enumerate(
            zip(diagnostic.fold_forecasts, diagnostic.fold_truths, strict=True)
        )
    )
    return replace(
        diagnostic,
        folds=folds,
        long_horizon_fold=HindcastFold(
            train_end=24,
            validation_end=48,
            status="success",
            forecast=forecast,
            truth=truth,
            mase_scale=float(scale),
        ),
        long_horizon_coverage=1.0,
    )


def _run_legacy_selection_package(
    task: Task,
    policy: DecisionPolicy,
    diagnostics: dict[str, CandidateDiagnostics],
    forecasts: dict[str, tuple[float, ...]],
    *,
    conditioned_names: tuple[str, ...] = (),
) -> tuple[SelectionDecision, NumericalForecastPackage]:
    active_names = tuple(diagnostics)
    expected = select_numerical_forecast(
        policy,
        profile=profile_task(task),
        active_names=active_names,
        diagnostics=diagnostics,
        forecasts=forecasts,
        history=task.history,
        conditioned_names=conditioned_names,
    )
    fallback = next(
        (
            name
            for name in active_names
            if diagnostics[name].family == "tsfm"
        ),
        active_names[0],
    )
    package = run_numerical_loop(
        task,
        screening_policy=ScreeningPolicy(
            tuple(
                _entry(
                    name,
                    diagnostics[name].family,
                    applicability=(
                        ApplicabilityPolicy(
                            (
                                ApplicabilityClause(
                                    feature_tests=(
                                        FeatureTest("history_length", ">=", 1),
                                    )
                                ),
                            )
                        )
                        if name in conditioned_names
                        else ApplicabilityPolicy()
                    ),
                )
                for name in active_names
            ),
            (fallback,),
        ),
        candidate_runner=lambda name, history, horizon, frequency: forecasts[name],
        diagnostics=diagnostics,
        decision_policy=policy,
    )
    return expected, package


def _legacy_replay_scenario(
    kind: str,
) -> tuple[SelectionDecision, NumericalForecastPackage]:
    if kind == "residual_correction":
        task = Task("residual", (0.0,) * 20, 2, "D", ())
        truths = ((0.0, 0.0),) * 3
        diagnostics = {
            "toto_2_0": _diagnostic(
                "toto_2_0",
                "tsfm",
                forecast=(2.0, 2.0),
                truth=(0.0, 0.0),
                median_mase=2.0,
            ),
            "wild_stat": _diagnostic(
                "wild_stat",
                "statistical",
                forecast=(-6.0, -6.0),
                truth=(0.0, 0.0),
                median_mase=6.0,
            ),
        }
        assert all(item.fold_truths == truths for item in diagnostics.values())
        policy = DecisionPolicy(
            ensemble_enabled=True,
            ensemble_weight_grid=(),
            ensemble_residual_strengths=(0.5,),
            ensemble_correction_clip=0.5,
            ensemble_min_diversity=0.1,
            ensemble_min_improvement=0.01,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.05,
        )
        return _run_legacy_selection_package(
            task,
            policy,
            diagnostics,
            {"toto_2_0": (2.0, 2.0), "wild_stat": (-6.0, -6.0)},
        )
    if kind == "protected_statistical_residual":
        task = Task("protected-residual", (8.0, 9.0, 10.0, 11.0), 2, "D", ())
        truths = ((10.0, 10.0),) * 3
        diagnostics = {
            "toto_2_0": _audited_diagnostic(
                _diagnostic(
                    "toto_2_0",
                    "tsfm",
                    forecast=(12.0, 12.0),
                    truth=(10.0, 10.0),
                    median_mase=2.0,
                ),
                forecast=(12.0, 12.0),
                truth=(10.0, 10.0),
            ),
            "downward_specialist": _audited_diagnostic(
                _diagnostic(
                    "downward_specialist",
                    "statistical",
                    forecast=(0.0, 0.0),
                    truth=(10.0, 10.0),
                    median_mase=10.0,
                ),
                forecast=(0.0, 0.0),
                truth=(10.0, 10.0),
            ),
        }
        assert all(item.fold_truths == truths for item in diagnostics.values())
        policy = DecisionPolicy(
            baseline_strategy="protected_joint_residual",
            ensemble_enabled=False,
            recent_regime_first=False,
            tsfm_router_min_improvement=0.02,
            ensemble_residual_strengths=(0.2,),
            ensemble_correction_clip=1.0,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        )
        return _run_legacy_selection_package(
            task,
            policy,
            diagnostics,
            {
                "toto_2_0": (12.0, 12.0),
                "downward_specialist": (0.0, 0.0),
            },
            conditioned_names=("downward_specialist",),
        )
    if kind == "tsfm_median_portfolio":
        task = Task("median", (10.0,) * 40, 2, "D", ())
        truths = ((10.0, 10.0),) * 3
        final_forecasts = {
            "toto_2_0": (4.0, 8.0),
            "timesfm_2_5": (14.0, 15.0),
            "chronos_bolt": (10.0, 10.0),
        }
        diagnostics = {
            name: _audited_diagnostic(
                _diagnostic(
                    name,
                    "tsfm",
                    forecast=forecast,
                    truth=(10.0, 10.0),
                    median_mase=1.0 + index * 0.1,
                ),
                forecast=forecast,
                truth=(10.0, 10.0),
            )
            for index, (name, forecast) in enumerate(final_forecasts.items())
        }
        assert all(item.fold_truths == truths for item in diagnostics.values())
        policy = DecisionPolicy(
            baseline_strategy="conservative_tsfm_portfolio",
            tsfm_router_min_improvement=0.02,
            tsfm_router_blend_weight=0.5,
            ensemble_enabled=False,
            recent_regime_first=False,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        )
        return _run_legacy_selection_package(
            task, policy, diagnostics, final_forecasts
        )
    if kind == "equal_fmean":
        task = Task("fmean", (0.0,) * 40, 1, "D", ())
        final_forecasts = {
            "low": (0.1,),
            "middle": (0.2,),
            "high": (0.4,),
        }
        truth = (statistics.fmean((0.1, 0.2, 0.4)),)
        diagnostics = {
            name: _diagnostic(
                name,
                "statistical",
                forecast=forecast,
                truth=truth,
                median_mase=1.0,
            )
            for name, forecast in final_forecasts.items()
        }
        policy = DecisionPolicy(
            ensemble_enabled=True,
            ensemble_max_members=3,
            ensemble_min_diversity=0.0,
            ensemble_min_improvement=0.01,
            recent_regime_first=False,
        )
        return _run_legacy_selection_package(
            task, policy, diagnostics, final_forecasts
        )
    if kind == "tsfm_shrinkage_overlay":
        task = Task("tsfm-blend", (0.0, 1.0) * 20, 2, "D", ())
        truth = (0.0, 1.0)
        truths = (truth,) * 3
        inputs = (
            ("toto_2_0", 1.0, (3.0, 4.0)),
            ("timesfm_2_5", 0.8, (1.0, 2.0)),
            ("chronos_bolt", 0.5, (2.0, 3.0)),
        )
        diagnostics = {
            name: _audited_diagnostic(
                _diagnostic(
                    name,
                    "tsfm",
                    forecast=forecast,
                    truth=truth,
                    median_mase=median_mase,
                ),
                forecast=forecast,
                truth=truth,
            )
            for name, median_mase, forecast in inputs
        }
        assert all(item.fold_truths == truths for item in diagnostics.values())
        policy = DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            tsfm_router_blend_weight=0.1,
            ensemble_enabled=False,
            recent_regime_first=False,
        )
        return _run_legacy_selection_package(
            task,
            policy,
            diagnostics,
            {name: forecast for name, _, forecast in inputs},
        )
    if kind == "statistical_shrinkage_overlay":
        task = Task("statistical-blend", (0.0, 1.0) * 20, 2, "D", ())
        truth = (0.0, 1.0)
        truths = (truth,) * 3
        inputs = (
            ("toto_2_0", "tsfm", 0.5, (2.0, 3.0)),
            ("seasonal_specialist", "statistical", 5.0, (1.0, 2.0)),
        )
        diagnostics = {
            name: _audited_diagnostic(
                _diagnostic(
                    name,
                    family,
                    forecast=forecast,
                    truth=truth,
                    median_mase=median_mase,
                ),
                forecast=forecast,
                truth=truth,
            )
            for name, family, median_mase, forecast in inputs
        }
        assert all(item.fold_truths == truths for item in diagnostics.values())
        policy = DecisionPolicy(
            baseline_strategy="conservative_combined",
            tsfm_router_blend_weight=0.1,
            ensemble_enabled=False,
            recent_regime_first=False,
        )
        return _run_legacy_selection_package(
            task,
            policy,
            diagnostics,
            {name: forecast for name, _, _, forecast in inputs},
            conditioned_names=("seasonal_specialist",),
        )
    if kind == "joint_tsfm_statistical_portfolio":
        task = Task("joint-blend", (1.75,) * 40, 2, "D", ())
        truth = (1.75, 1.75)
        truths = (truth,) * 3
        inputs = (
            ("toto_2_0", "tsfm", 1.0, (2.5, 1.5)),
            ("timesfm_2_5", "tsfm", 1.0, (1.5, 2.5)),
            ("seasonal_specialist", "statistical", 4.0, (1.0, 1.0)),
        )
        diagnostics = {
            name: _audited_diagnostic(
                _diagnostic(
                    name,
                    family,
                    forecast=forecast,
                    truth=truth,
                    median_mase=median_mase,
                ),
                forecast=forecast,
                truth=truth,
            )
            for name, family, median_mase, forecast in inputs
        }
        assert all(item.fold_truths == truths for item in diagnostics.values())
        policy = DecisionPolicy(
            baseline_strategy="conservative_joint_portfolio",
            tsfm_router_min_improvement=0.02,
            tsfm_router_blend_weight=0.1,
            ensemble_enabled=False,
            recent_regime_first=False,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        )
        return _run_legacy_selection_package(
            task,
            policy,
            diagnostics,
            {name: forecast for name, _, _, forecast in inputs},
            conditioned_names=("seasonal_specialist",),
        )
    raise AssertionError(f"unknown replay scenario {kind!r}")


def _combined() -> tuple[CombinedPolicy, ...]:
    return (
        CombinedPolicy(
            "combined_mean",
            ("toto_2_0", "seasonal_specialist"),
            "weighted_mean",
            (0.5, 0.5),
            fallback_parent="toto_2_0",
        ),
        CombinedPolicy(
            "combined_route",
            ("toto_2_0", "seasonal_specialist"),
            "route",
            signal="periodicity_strength",
            threshold=0.5,
            above_parent="seasonal_specialist",
            below_parent="toto_2_0",
            fallback_parent="toto_2_0",
        ),
    )


def _diagnostics_for_active(
    *, horizon: int = 3
) -> dict[str, CandidateDiagnostics]:
    truth = tuple(float(index + 1) for index in range(horizon))
    return {
        "toto_2_0": _diagnostic(
            "toto_2_0",
            "tsfm",
            forecast=tuple(10.0 for _ in range(horizon)),
            truth=truth,
            median_mase=1.0,
        ),
        "seasonal_specialist": _diagnostic(
            "seasonal_specialist",
            "statistical",
            forecast=truth,
            truth=truth,
            median_mase=0.1,
        ),
        "combined_mean": _diagnostic(
            "combined_mean", "combined", forecast=truth, truth=truth, median_mase=0.2
        ),
        "combined_route": _diagnostic(
            "combined_route", "combined", forecast=truth, truth=truth, median_mase=0.3
        ),
    }


def _runner(calls: Counter[tuple[str, tuple[float, ...], int, str]]):
    def run(
        name: str, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        calls[(name, tuple(history), horizon, frequency)] += 1
        if name == "toto_2_0":
            return tuple(10.0 for _ in range(horizon))
        if name == "seasonal_specialist":
            return tuple(float(index + 1) for index in range(horizon))
        if name == "inactive_flat_only":
            raise AssertionError("screened candidate executed")
        raise AssertionError(f"Combined policy was sent to the leaf runner: {name}")

    return run


def _policy_entries(*, include_inactive: bool = False) -> tuple[ScreeningEntry, ...]:
    entries = [
        _entry("toto_2_0", "tsfm"),
        _entry("seasonal_specialist", "statistical"),
        _entry("combined_mean", "combined"),
        _entry("combined_route", "combined"),
    ]
    if include_inactive:
        entries.insert(
            0,
            _entry(
                "inactive_flat_only",
                "statistical",
                applicability=ApplicabilityPolicy(
                    (
                        ApplicabilityClause(
                            feature_tests=(FeatureTest("trend_direction", "==", "flat"),)
                        ),
                    )
                ),
            ),
        )
    return tuple(entries)


def _card(*assumptions: AssumptionGrounding, history_length: int = 84) -> MorphologyCard:
    broad = MorphologyToolCall("broad", "detect_periodicity", 0, history_length)
    recent = MorphologyToolCall(
        "recent", "detect_periodicity", history_length // 2, history_length
    )
    return MorphologyCard(
        "The recent periodic shape is stable.",
        "The full history contains a repeated cycle.",
        (broad, recent),
        (
            MorphologyObservation(broad, {"strength": 1.0}),
            MorphologyObservation(recent, {"strength": 1.0}),
        ),
        assumptions,
    )


def _assumption(
    assumption_id: str,
    candidate: str,
    *,
    kind: str = "seasonality",
    claim: str = "The observed cycle persists through the forecast horizon.",
) -> AssumptionGrounding:
    return AssumptionGrounding(
        assumption_id,
        kind,
        claim,
        "The cycle disappears or changes phase.",
        ("broad", "recent"),
        (candidate,),
        0.9,
    )


class _FixedReasoner:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def reason(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.result


def test_task_profile_screens_before_leaf_materialization_and_combined_reuses_leaves() -> None:
    task = Task("trend", tuple(float(value) for value in range(1, 85)), 3, "D", ())
    calls: Counter[tuple[str, tuple[float, ...], int, str]] = Counter()
    diagnostics = _diagnostics_for_active()

    package = run_numerical_loop(
        task,
        screening_policy=_screening(*_policy_entries(include_inactive=True)),
        candidate_runner=_runner(calls),
        combined_policies=_combined(),
        diagnostics=diagnostics,
        decision_policy=DecisionPolicy(ensemble_enabled=False),
    )

    assert package.task_profile.trend_direction == "up"
    assert "inactive_flat_only" not in package.active_candidate_names
    assert {key[0] for key in calls} == {"toto_2_0", "seasonal_specialist"}
    assert all(count == 1 for count in calls.values())
    assert {item.name for item in package.ranked_alternatives} >= {
        "combined_mean",
        "combined_route",
    }
    assert len(package.final_forecast) == task.horizon
    assert all(math.isfinite(value) for value in package.final_forecast)


def test_history_only_hindcasts_memoize_shared_leaf_invocations_by_prefix() -> None:
    task = Task("history-only", tuple(float(value) for value in [1, 2, 3] * 10), 2, "D", ())
    calls: Counter[tuple[str, tuple[float, ...], int, str]] = Counter()

    package = run_numerical_loop(
        task,
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(calls),
        combined_policies=_combined(),
        hindcast_config=HindcastConfig(folds=2, min_successful_folds=2),
        decision_policy=DecisionPolicy(min_successful_folds=2, ensemble_enabled=False),
    )

    assert all(count == 1 for count in calls.values())
    assert all(key[1] != task.future for key in calls)
    assert package.candidate_diagnostics["combined_mean"].successful_folds == 2
    assert package.candidate_diagnostics["combined_route"].successful_folds == 2


def test_valid_grounded_top_k_guides_only_executed_forecasts_and_projects_four_fields() -> None:
    task = _task()
    calls: Counter[tuple[str, tuple[float, ...], int, str]] = Counter()
    card = _card(
        _assumption("cycle_primary", "seasonal_specialist"),
        _assumption("cycle_duplicate", "combined_route"),
    )
    reasoner = _FixedReasoner(card)

    package = run_numerical_loop(
        task,
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(calls),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False, assumption_top_k=2),
        morphology_reasoner=reasoner,
        component_fingerprints={"portfolio": "reviewed-portfolio-v1"},
    )

    assert package.selection_decision.selected == ("seasonal_specialist",)
    assert package.final_forecast == tuple(float(index + 1) for index in range(task.horizon))
    assert tuple(item.assumption_id for item in package.accepted_assumptions) == (
        "cycle_primary",
    )
    assert package.rejected_assumptions == {"cycle_duplicate": "diversity_rejected"}
    assert tuple(map(set, package.retrieval_handoff)) == (
        {"assumption_id", "kind", "claim", "failure_condition"},
    )
    assert package.retrieval_handoff[0]["kind"] == "seasonality"
    assert reasoner.calls[0]["active_names"] == package.active_candidate_names
    assert package.component_fingerprints["portfolio"] == "reviewed-portfolio-v1"
    with pytest.raises(TypeError):
        package.retrieval_handoff[0]["claim"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        package.component_fingerprints["portfolio"] = "mutated"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        package.final_forecast = (999.0,) * task.horizon  # type: ignore[misc]


def test_grounded_guidance_retains_a_nonflagship_protected_tsfm_anchor() -> None:
    task = _task()
    entries = (
        _entry("safe_anchor", "tsfm"),
        _entry("guided_specialist", "statistical"),
    )
    truth = (1.0, 2.0, 3.0)
    diagnostics = {
        "safe_anchor": _diagnostic(
            "safe_anchor", "tsfm", forecast=truth, truth=truth, median_mase=1.0
        ),
        "guided_specialist": _diagnostic(
            "guided_specialist",
            "statistical",
            forecast=(100.0, 100.0, 100.0),
            truth=truth,
            median_mase=0.1,
        ),
    }

    def runner(name: str, history: tuple[float, ...], horizon: int, frequency: str):
        del history, frequency
        value = 10.0 if name == "safe_anchor" else 1.0
        return tuple(value for _ in range(horizon))

    package = run_numerical_loop(
        task,
        screening_policy=ScreeningPolicy(entries, ("safe_anchor",)),
        candidate_runner=runner,
        diagnostics=diagnostics,
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        morphology_reasoner=_FixedReasoner(
            _card(_assumption("guided", "guided_specialist"))
        ),
    )

    assert package.protected_baseline.name == "safe_anchor"
    assert package.selection_decision.selected == ("safe_anchor",)
    assert package.final_forecast == (10.0, 10.0, 10.0)


def test_grounded_guidance_compares_against_exact_all_statistical_safe_anchor() -> None:
    task = _task()
    entries = (
        _entry("safe_stat", "statistical"),
        _entry("guided_stat", "statistical"),
    )
    truth = (1.0, 2.0, 3.0)
    diagnostics = {
        "safe_stat": _diagnostic(
            "safe_stat", "statistical", forecast=truth, truth=truth, median_mase=0.1
        ),
        "guided_stat": _diagnostic(
            "guided_stat",
            "statistical",
            forecast=(100.0, 100.0, 100.0),
            truth=truth,
            median_mase=0.2,
        ),
    }

    def runner(name: str, history: tuple[float, ...], horizon: int, frequency: str):
        del history, frequency
        value = 10.0 if name == "safe_stat" else 1.0
        return tuple(value for _ in range(horizon))

    package = run_numerical_loop(
        task,
        screening_policy=ScreeningPolicy(entries, ("safe_stat",)),
        candidate_runner=runner,
        diagnostics=diagnostics,
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        morphology_reasoner=_FixedReasoner(
            _card(_assumption("guided", "guided_stat"))
        ),
    )

    assert package.protected_baseline.name == "safe_stat"
    assert package.selection_decision.selected == ("safe_stat",)
    assert package.selection_decision.baseline_name == "safe_stat"
    assert package.final_forecast == (10.0, 10.0, 10.0)


def test_safe_handoff_does_not_confuse_a_morphology_word_with_identity_leakage() -> None:
    task = _task()
    entries = (_entry("toto_2_0", "tsfm"), _entry("cycle", "statistical"))
    truth = (1.0, 2.0, 3.0)
    diagnostics = {
        "toto_2_0": _diagnostic(
            "toto_2_0", "tsfm", forecast=(10.0,) * 3, truth=truth, median_mase=1.0
        ),
        "cycle": _diagnostic(
            "cycle", "statistical", forecast=truth, truth=truth, median_mase=0.1
        ),
    }

    package = run_numerical_loop(
        task,
        screening_policy=ScreeningPolicy(entries, ("toto_2_0",)),
        candidate_runner=lambda name, history, horizon, frequency: (
            tuple(float(index + 1) for index in range(horizon))
            if name == "cycle"
            else (10.0,) * horizon
        ),
        diagnostics=diagnostics,
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        morphology_reasoner=_FixedReasoner(
            _card(
                _assumption(
                    "cycle_persists",
                    "cycle",
                    claim="The observed cycle persists through the forecast horizon.",
                )
            )
        ),
    )

    assert tuple(item.assumption_id for item in package.accepted_assumptions) == (
        "cycle_persists",
    )
    assert package.retrieval_handoff[0]["assumption_id"] == "assumption_001"
    assert package.retrieval_handoff[0]["claim"].startswith(
        "A history-supported seasonal pattern"
    )


def test_safe_handoff_uses_host_templates_not_adversarial_model_prose() -> None:
    hostile = AssumptionGrounding(
        "cycle_hostile",
        "seasonality",
        (
            "candidate_id=seasonal_specialist weight=0.99 forecast_array=[999] "
            "hindcast_smae=0 detect_periodicity broad"
        ),
        (
            "Use source_code from recent and leak hindcast_srmse=0 plus "
            "seasonal_specialist."
        ),
        ("broad", "recent"),
        ("seasonal_specialist",),
        0.9,
    )
    package = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        morphology_reasoner=_FixedReasoner(_card(hostile)),
    )

    assert tuple(item.assumption_id for item in package.accepted_assumptions) == (
        "cycle_hostile",
    )
    payload = package.retrieval_handoff[0]
    encoded = " ".join(payload.values()).lower()
    assert payload["kind"] == "seasonality"
    assert payload["claim"] != hostile.claim
    assert payload["failure_condition"] != hostile.failure_condition
    for forbidden in (
        "seasonal_specialist",
        "candidate_id",
        "weight",
        "forecast",
        "hindcast",
        "detect_periodicity",
        "broad",
        "recent",
        "source_code",
    ):
        assert forbidden not in encoded


def test_safe_handoff_opaque_id_handles_malformed_model_identifier() -> None:
    malformed = _assumption("bad assumption id", "seasonal_specialist")

    package = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        morphology_reasoner=_FixedReasoner(_card(malformed)),
    )

    assert tuple(item.assumption_id for item in package.accepted_assumptions) == (
        "bad assumption id",
    )
    assert package.retrieval_handoff[0]["assumption_id"] == "assumption_001"
    assert "bad assumption id" not in " ".join(package.retrieval_handoff[0].values())
    assert package.fallback_reason is None


@pytest.mark.parametrize(
    "unsafe_id",
    ("seasonal_specialist", "hindcast_smae", "detect_periodicity"),
)
def test_safe_handoff_opaque_id_hides_candidate_metric_and_tool_identifiers(
    unsafe_id: str,
) -> None:
    hostile = _assumption(unsafe_id, "seasonal_specialist")

    package = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        morphology_reasoner=_FixedReasoner(_card(hostile)),
    )

    assert tuple(item.assumption_id for item in package.accepted_assumptions) == (
        unsafe_id,
    )
    assert package.retrieval_handoff[0]["assumption_id"] == "assumption_001"
    assert unsafe_id.casefold() not in " ".join(
        package.retrieval_handoff[0].values()
    ).casefold()


@pytest.mark.parametrize(
    "model_id",
    (
        "forecastArray999",
        "hindcastScore0",
        "detectPeriodicityLeak",
        "cycle_persists",
    ),
)
def test_safe_handoff_replaces_every_model_id_with_an_opaque_host_id(
    model_id: str,
) -> None:
    task = _task()
    entries = (_entry("toto_2_0", "tsfm"), _entry("cycle", "statistical"))
    truth = (1.0, 2.0, 3.0)
    diagnostics = {
        "toto_2_0": _diagnostic(
            "toto_2_0", "tsfm", forecast=(10.0,) * 3, truth=truth, median_mase=1.0
        ),
        "cycle": _diagnostic(
            "cycle", "statistical", forecast=truth, truth=truth, median_mase=0.1
        ),
    }

    package = run_numerical_loop(
        task,
        screening_policy=ScreeningPolicy(entries, ("toto_2_0",)),
        candidate_runner=lambda name, history, horizon, frequency: (
            tuple(float(index + 1) for index in range(horizon))
            if name == "cycle"
            else (10.0,) * horizon
        ),
        diagnostics=diagnostics,
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        morphology_reasoner=_FixedReasoner(
            _card(_assumption(model_id, "cycle"))
        ),
    )

    assert tuple(item.assumption_id for item in package.accepted_assumptions) == (
        model_id,
    )
    assert package.retrieval_handoff[0]["assumption_id"] == "assumption_001"
    assert model_id.casefold() not in " ".join(
        package.retrieval_handoff[0].values()
    ).casefold()


def test_safe_handoff_host_ids_are_stable_for_identical_accepted_order() -> None:
    accepted = (
        _assumption("model_generated_first", "seasonal_specialist"),
        _assumption(
            "model_generated_second",
            "seasonal_specialist",
            kind="trend",
        ),
    )

    first = safe_retrieval_projection(accepted, {})[2]
    second = safe_retrieval_projection(accepted, {})[2]

    assert tuple(item["assumption_id"] for item in first) == (
        "assumption_001",
        "assumption_002",
    )
    assert first == second


@pytest.mark.parametrize(
    ("candidate", "diagnostic_changes", "expected_reason"),
    [
        ("too_few", {"successful_folds": 2}, "insufficient_successful_folds"),
        ("bad_tail", {"worst_mase": 10.1}, "catastrophic_hindcast_tail"),
        ("exploded", {"explosion": True}, "catastrophic_hindcast_tail"),
        (
            "low_coverage",
            {"long_horizon_coverage": 0.5},
            "insufficient_long_horizon_coverage",
        ),
    ],
)
def test_assumption_guidance_cannot_bypass_protected_reliability_gates(
    candidate: str,
    diagnostic_changes: dict[str, object],
    expected_reason: str,
) -> None:
    task = _task()
    entries = (*_policy_entries(), _entry(candidate, "statistical"))
    diagnostics = _diagnostics_for_active()
    diagnostics[candidate] = _diagnostic(
        candidate,
        "statistical",
        **diagnostic_changes,
    )
    if candidate == "low_coverage":
        diagnostics[candidate] = replace(
            diagnostics[candidate],
            long_horizon_fold=HindcastFold(
                train_end=80,
                validation_end=84,
                status="success",
                forecast=(1.0, 2.0, 3.0),
                truth=(1.0, 2.0, 3.0),
                mase=0.0,
                mase_scale=1.0,
            ),
        )

    def runner(name: str, history: tuple[float, ...], horizon: int, frequency: str):
        del history, frequency
        return tuple(99.0 if name == candidate else 10.0 for _ in range(horizon))

    package = run_numerical_loop(
        task,
        screening_policy=_screening(*entries),
        candidate_runner=runner,
        combined_policies=_combined(),
        diagnostics=diagnostics,
        decision_policy=DecisionPolicy(
            ensemble_enabled=False,
            fallback_to_best_available=False,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
        ),
        morphology_reasoner=_FixedReasoner(_card(_assumption("hostile", candidate))),
    )

    assert package.final_forecast == (10.0, 10.0, 10.0)
    assert package.protected_baseline is not None
    assert package.protected_baseline.name == "toto_2_0"
    assert package.accepted_assumptions == ()
    assert package.rejected_assumptions == {"hostile": expected_reason}
    assert package.fallback_reason == "morphology_consistency_rejected"


@pytest.mark.parametrize("result", [{"not": "a card"}, RuntimeError("bad morphology")])
def test_malformed_or_failed_morphology_returns_protected_safe_anchor(result: object) -> None:
    class MalformedReasoner(_FixedReasoner):
        def reason(self, **kwargs: object) -> object:
            if isinstance(self.result, Exception):
                raise self.result
            return super().reason(**kwargs)

    package = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        morphology_reasoner=MalformedReasoner(result),
    )

    assert package.selection_decision.selected == ("toto_2_0",)
    assert package.final_forecast == (10.0, 10.0, 10.0)
    assert package.morphology_card is None
    assert package.accepted_assumptions == ()
    assert package.retrieval_handoff == ()
    assert package.fallback_reason.startswith("morphology_reasoner_failed:")


def test_absent_morphology_reasoner_is_exactly_legacy_selector_behavior() -> None:
    task = _task()
    diagnostics = _diagnostics_for_active()
    policy = DecisionPolicy(ensemble_enabled=False)
    expected = select_numerical_forecast(
        policy,
        profile=None,
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={
            "toto_2_0": (10.0, 10.0, 10.0),
            "seasonal_specialist": (1.0, 2.0, 3.0),
            "combined_mean": (5.5, 6.0, 6.5),
            "combined_route": (1.0, 2.0, 3.0),
        },
        history=task.history,
    )

    package = run_numerical_loop(
        task,
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=diagnostics,
        decision_policy=policy,
    )

    assert package.selection_decision == expected
    assert package.fallback_reason is None
    assert package.morphology_card is None


def test_absent_reasoner_preserves_enabled_legacy_assumption_guidance() -> None:
    task = _task()
    diagnostics = _diagnostics_for_active()
    forecasts = {
        "toto_2_0": (10.0, 10.0, 10.0),
        "seasonal_specialist": (1.0, 2.0, 3.0),
        "combined_mean": (5.5, 6.0, 6.5),
        "combined_route": (1.0, 2.0, 3.0),
    }
    families = {
        "toto_2_0": "tsfm",
        "seasonal_specialist": "statistical",
        "combined_mean": "combined",
        "combined_route": "combined",
    }
    policy = DecisionPolicy(
        ensemble_enabled=False,
        assumption_guidance_enabled=True,
        assumption_top_k=3,
    )
    expected = select_assumption_guided_forecast(
        policy,
        profile=profile_task(task),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts=forecasts,
        families=families,
        history=task.history,
    )

    package = run_numerical_loop(
        task,
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=diagnostics,
        decision_policy=policy,
    )

    assert expected.assumption_ids
    assert package.selection_decision == expected


def test_absent_morphology_preserves_legacy_best_available_fallback() -> None:
    task = _task()
    diagnostic = _diagnostic(
        "stat_only",
        "statistical",
        median_mase=0.5,
        eligible=False,
        successful_folds=0,
    )
    policy = DecisionPolicy(ensemble_enabled=False, fallback_to_best_available=True)
    expected = select_numerical_forecast(
        policy,
        active_names=("stat_only",),
        diagnostics={"stat_only": diagnostic},
        forecasts={"stat_only": (4.0, 4.0, 4.0)},
    )

    package = run_numerical_loop(
        task,
        screening_policy=ScreeningPolicy(
            (_entry("stat_only", "statistical"),), ("stat_only",)
        ),
        candidate_runner=lambda name, history, horizon, frequency: (4.0,) * horizon,
        diagnostics={"stat_only": diagnostic},
        decision_policy=policy,
    )

    assert package.selection_decision == expected
    assert package.protected_baseline.name == "stat_only"


def test_package_detaches_mutable_precomputed_diagnostic_containers() -> None:
    mutable_forecasts = [[1.0, 2.0, 3.0] for _ in range(3)]
    mutable_truths = [[1.0, 2.0, 3.0] for _ in range(3)]
    diagnostic = replace(
        _diagnostic("stat_only", "statistical"),
        fold_forecasts=mutable_forecasts,
        fold_truths=mutable_truths,
    )

    package = run_numerical_loop(
        _task(),
        screening_policy=ScreeningPolicy(
            (_entry("stat_only", "statistical"),), ("stat_only",)
        ),
        candidate_runner=lambda name, history, horizon, frequency: (4.0,) * horizon,
        diagnostics={"stat_only": diagnostic},
        decision_policy=DecisionPolicy(ensemble_enabled=False),
    )
    mutable_forecasts[0][0] = 999.0
    mutable_truths.append([999.0])

    frozen = package.candidate_diagnostics["stat_only"]
    assert frozen.fold_forecasts == ((1.0, 2.0, 3.0),) * 3
    assert frozen.fold_truths == ((1.0, 2.0, 3.0),) * 3


def test_package_detaches_every_mutable_selection_container() -> None:
    base = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
    )
    selected = list(base.selection_decision.selected)
    weights = list(base.selection_decision.weights)
    forecast = list(base.selection_decision.forecast)
    reason_codes = list(base.selection_decision.reason_codes)
    rejected = {"outside": "unchanged"}
    assumption_ids = list(base.selection_decision.assumption_ids)
    assumption_kinds = list(base.selection_decision.assumption_kinds)
    considered = list(base.selection_decision.considered_candidates)
    forged = replace(
        base.selection_decision,
        selected=selected,
        weights=weights,
        forecast=forecast,
        reason_codes=reason_codes,
        rejected=rejected,
        assumption_ids=assumption_ids,
        assumption_kinds=assumption_kinds,
        considered_candidates=considered,
    )

    package = replace(base, selection_decision=forged)
    selected.clear()
    weights[0] = 0.0
    forecast[0] = 999.0
    reason_codes.clear()
    rejected["mutated"] = "yes"
    assumption_ids.append("mutated")
    assumption_kinds.append("mutated")
    considered.append("mutated")

    decision = package.selection_decision
    assert decision.selected == base.selection_decision.selected
    assert decision.weights == base.selection_decision.weights
    assert decision.forecast == base.selection_decision.forecast
    assert decision.reason_codes == base.selection_decision.reason_codes
    assert decision.rejected == {"outside": "unchanged"}
    assert decision.assumption_ids == base.selection_decision.assumption_ids
    assert decision.assumption_kinds == base.selection_decision.assumption_kinds
    assert decision.considered_candidates == base.selection_decision.considered_candidates
    with pytest.raises(TypeError):
        decision.rejected["mutated"] = "yes"  # type: ignore[index]


def test_package_rejects_selected_active_name_without_materialized_alternative() -> None:
    base = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
    )
    forged = replace(
        base.selection_decision,
        selected=("unmaterialized",),
        weights=(1.0,),
        forecast=(9.0, 9.0, 9.0),
    )

    with pytest.raises(ValueError, match="materialized"):
        replace(
            base,
            active_candidate_names=(*base.active_candidate_names, "unmaterialized"),
            selection_decision=forged,
            final_forecast=forged.forecast,
        )


def test_package_rejects_forged_single_forecast_for_materialized_name() -> None:
    base = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
    )
    forged = replace(base.selection_decision, forecast=(999.0, 999.0, 999.0))

    with pytest.raises(ValueError, match="materialized forecast"):
        replace(base, selection_decision=forged, final_forecast=forged.forecast)


@pytest.mark.parametrize(
    "weights",
    ((-0.25, 1.25), (0.25, 0.5), (0.5, 0.5000000000005)),
)
def test_package_rejects_negative_or_non_normalized_weights(
    weights: tuple[float, float],
) -> None:
    base = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
    )
    forged = replace(
        base.selection_decision,
        mode="ensemble",
        selected=("toto_2_0", "seasonal_specialist"),
        weights=weights,
        forecast=(3.25, 4.0, 4.75),
        combination_type=None,
    )

    with pytest.raises(ValueError, match="weights"):
        replace(base, selection_decision=forged, final_forecast=forged.forecast)


@pytest.mark.parametrize(
    "forecast",
    ((999.0, 999.0, 999.0), (3.2500000000005, 4.0, 4.75)),
)
def test_package_rejects_arbitrary_legacy_ensemble_forecast(
    forecast: tuple[float, float, float],
) -> None:
    base = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
    )
    forged = replace(
        base.selection_decision,
        mode="ensemble",
        selected=("toto_2_0", "seasonal_specialist"),
        weights=(0.25, 0.75),
        forecast=forecast,
        combination_type=None,
        arithmetic=None,
    )

    with pytest.raises(ValueError, match="weighted combination"):
        replace(base, selection_decision=forged, final_forecast=forged.forecast)


def test_package_accepts_valid_legacy_weighted_ensemble() -> None:
    base = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
    )
    weighted = (3.25, 4.0, 4.75)
    legacy = replace(
        base.selection_decision,
        mode="ensemble",
        selected=("toto_2_0", "seasonal_specialist"),
        weights=(0.25, 0.75),
        forecast=weighted,
        combination_type=None,
        arithmetic=None,
    )

    package = replace(base, selection_decision=legacy, final_forecast=weighted)

    assert package.selection_decision.forecast == weighted
    assert package.final_forecast == weighted


def test_package_rejects_non_weighted_multi_member_selection_mode() -> None:
    base = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
    )
    weighted = (3.25, 4.0, 4.75)
    forged = replace(
        base.selection_decision,
        mode="combined",
        selected=("toto_2_0", "seasonal_specialist"),
        weights=(0.25, 0.75),
        forecast=weighted,
        combination_type="residual_correction",
        arithmetic=None,
    )

    with pytest.raises(ValueError, match="trusted arithmetic replay"):
        replace(base, selection_decision=forged, final_forecast=weighted)


def test_package_rejects_morphology_guided_multi_member_selection() -> None:
    base = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        morphology_reasoner=_FixedReasoner(
            _card(_assumption("cycle_primary", "seasonal_specialist"))
        ),
    )
    weighted = (3.25, 4.0, 4.75)
    forged = replace(
        base.selection_decision,
        mode="ensemble",
        selected=("toto_2_0", "seasonal_specialist"),
        weights=(0.25, 0.75),
        forecast=weighted,
        combination_type=None,
    )

    with pytest.raises(ValueError, match="Morphology-guided selection must be single"):
        replace(base, selection_decision=forged, final_forecast=weighted)


def test_package_rejects_active_card_ensemble_even_without_accepted_assumptions() -> None:
    base = run_numerical_loop(
        _task(),
        screening_policy=_screening(*_policy_entries()),
        candidate_runner=_runner(Counter()),
        combined_policies=_combined(),
        diagnostics=_diagnostics_for_active(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
    )
    weighted = (3.25, 4.0, 4.75)
    forged = replace(
        base.selection_decision,
        mode="ensemble",
        selected=("toto_2_0", "seasonal_specialist"),
        weights=(0.25, 0.75),
        forecast=weighted,
        combination_type=None,
    )

    with pytest.raises(ValueError, match="Morphology-guided selection must be single"):
        replace(
            base,
            morphology_card=_card(
                _assumption("rejected_by_consistency", "seasonal_specialist")
            ),
            accepted_assumptions=(),
            retrieval_handoff=(),
            selection_decision=forged,
            final_forecast=weighted,
        )


@pytest.mark.parametrize(
    "kind",
    (
        "residual_correction",
        "protected_statistical_residual",
        "tsfm_median_portfolio",
    ),
)
def test_no_reasoner_package_preserves_real_legacy_non_linear_decision(
    kind: str,
) -> None:
    expected, package = _legacy_replay_scenario(kind)

    assert expected.combination_type == kind
    assert package.selection_decision == expected
    assert package.final_forecast == expected.forecast


def test_no_reasoner_package_replays_equal_weight_ensemble_with_fmean() -> None:
    expected, package = _legacy_replay_scenario("equal_fmean")

    assert expected.mode == "ensemble"
    assert expected.forecast == (0.23333333333333336,)
    assert package.selection_decision == expected
    assert package.final_forecast == expected.forecast


@pytest.mark.parametrize(
    ("kind", "selected", "legacy_forecast", "generic_weighted_forecast"),
    (
        (
            "tsfm_shrinkage_overlay",
            ("chronos_bolt", "timesfm_2_5"),
            (1.9, 2.9000000000000004),
            (1.9000000000000001, 2.9000000000000004),
        ),
        (
            "statistical_shrinkage_overlay",
            ("toto_2_0", "seasonal_specialist"),
            (1.9, 2.9000000000000004),
            (1.9000000000000001, 2.9000000000000004),
        ),
        (
            "joint_tsfm_statistical_portfolio",
            ("timesfm_2_5", "toto_2_0", "seasonal_specialist"),
            (1.9, 1.9),
            (1.9000000000000001, 1.9000000000000001),
        ),
    ),
)
def test_no_reasoner_package_replays_shrinkage_with_exact_blend_primitive(
    kind: str,
    selected: tuple[str, ...],
    legacy_forecast: tuple[float, ...],
    generic_weighted_forecast: tuple[float, ...],
) -> None:
    expected, package = _legacy_replay_scenario(kind)
    materialized = {item.name: item.forecast for item in package.ranked_alternatives}

    assert expected.combination_type == kind
    assert expected.selected == selected
    assert expected.forecast == legacy_forecast
    assert math.nextafter(legacy_forecast[0], math.inf) == generic_weighted_forecast[0]
    assert replay_selection_forecast(expected, materialized) == legacy_forecast
    assert package.selection_decision == expected
    assert package.final_forecast == legacy_forecast

    forged = replace(expected, forecast=generic_weighted_forecast)
    with pytest.raises(ValueError, match="replay"):
        replace(
            package,
            selection_decision=forged,
            final_forecast=generic_weighted_forecast,
        )


@pytest.mark.parametrize(
    "kind",
    (
        "residual_correction",
        "protected_statistical_residual",
        "tsfm_median_portfolio",
        "equal_fmean",
    ),
)
def test_package_rejects_forged_forecast_for_every_replay_mode(kind: str) -> None:
    _, package = _legacy_replay_scenario(kind)
    forged_forecast = (
        package.selection_decision.forecast[0] + 1.0,
        *package.selection_decision.forecast[1:],
    )
    forged = replace(package.selection_decision, forecast=forged_forecast)

    with pytest.raises(ValueError, match="replay"):
        replace(package, selection_decision=forged, final_forecast=forged_forecast)


def test_component_fingerprint_is_invariant_to_combined_policy_input_order() -> None:
    task = _task()
    policies = _combined()
    kwargs = {
        "screening_policy": _screening(*_policy_entries()),
        "diagnostics": _diagnostics_for_active(),
        "decision_policy": DecisionPolicy(ensemble_enabled=False),
    }

    forward = run_numerical_loop(
        task,
        candidate_runner=_runner(Counter()),
        combined_policies=policies,
        **kwargs,
    )
    reversed_order = run_numerical_loop(
        task,
        candidate_runner=_runner(Counter()),
        combined_policies=tuple(reversed(policies)),
        **kwargs,
    )

    assert forward.component_fingerprints == reversed_order.component_fingerprints


def test_leaf_memo_does_not_swallow_process_control_exceptions() -> None:
    def interrupted(
        name: str, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        del name, history, horizon, frequency
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_numerical_loop(
            _task(),
            screening_policy=ScreeningPolicy(
                (_entry("stat_only", "statistical"),), ("stat_only",)
            ),
            candidate_runner=interrupted,
            diagnostics={
                "stat_only": _diagnostic("stat_only", "statistical")
            },
            decision_policy=DecisionPolicy(ensemble_enabled=False),
        )
