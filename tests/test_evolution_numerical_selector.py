from __future__ import annotations

from dataclasses import asdict, replace
import math
import random

import pytest

from common.metrics import drcik_point_metrics
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    HindcastConfig,
    HindcastFold,
    diagnose_candidate,
    hindcast_cache_key,
    pairwise_diversity,
    passes_independent_scaled_regret,
    select_assumption_guided_forecast,
    select_numerical_forecast,
    select_protected_safe_anchor,
)
from numerical_agent.evolution import numerical_selector as selector_module
from numerical_agent.evolution.screening import TaskProfile, profile_task


def test_exported_scaled_regret_guard_uses_raw_error_when_capped_metrics_tie() -> None:
    assert not passes_independent_scaled_regret(
        candidate_forecasts=((8.0, 8.0),),
        reference_forecasts=((7.0, 7.0),),
        truths=((1.0, 1.0),),
        max_smae_regret=0.02,
        max_srmse_regret=0.02,
    )


def _task(history=tuple(float(i) for i in range(1, 41)), horizon=5):
    return Task("hidden-id", history, horizon, "D", (999.0,) * horizon)


def test_hindcasts_use_only_historical_prefixes_and_not_task_future():
    seen = []

    def runner(name, history, horizon, frequency):
        seen.append((name, history, horizon, frequency))
        return tuple(history[-1] + i + 1 for i in range(horizon))

    diagnostic = diagnose_candidate(
        _task(), "linear", "statistical", runner, HindcastConfig(folds=3)
    )

    assert [len(item[1]) for item in seen] == [25, 30, 35]
    assert all(999.0 not in item[1] for item in seen)
    assert diagnostic.successful_folds == 3
    assert diagnostic.eligible
    assert diagnostic.median_mase == pytest.approx(0.0)
    assert diagnostic.recent_mase == pytest.approx(0.0)
    assert diagnostic.worst_mase == pytest.approx(0.0)
    assert diagnostic.mase_mad == pytest.approx(0.0)
    assert diagnostic.median_rmsse == pytest.approx(0.0)


def test_hindcast_folds_record_capped_and_raw_scaled_metrics():
    diagnostic = diagnose_candidate(
        _task(tuple(float(index + 1) for index in range(40)), horizon=5),
        "bad", "statistical", lambda *_: (10_000.0,) * 5,
        HindcastConfig(folds=3),
    )

    fold = diagnostic.folds[0]
    assert fold.smae == pytest.approx(5.0)
    assert fold.srmse == pytest.approx(5.0)
    assert fold.smae_raw is not None and fold.smae_raw > fold.smae
    assert fold.srmse_raw is not None and fold.srmse_raw > fold.srmse
    assert fold.smae_clipped and fold.srmse_clipped


def test_joint_summaries_follow_per_fold_distribution_on_cross_trading_folds() -> None:
    """Pairing marginal summaries invents a joint score no observed fold produced."""
    forecasts = iter(
        (
            (5.0, 5.0, 5.0, 5.0),  # sMAE=4, sRMSE=4, joint=4
            (1.0, 1.0, 1.0, 9.0),  # sMAE=2, sRMSE=4, joint=3
            (1.0, 1.0, 1.0, 11.0),  # sMAE=2.5, sRMSE=5, joint=3.75
        )
    )
    diagnostic = diagnose_candidate(
        _task((1.0,) * 40, horizon=4),
        "cross_trading",
        "statistical",
        lambda *_: next(forecasts),
        HindcastConfig(folds=3),
    )

    assert diagnostic.median_joint_scaled_error == pytest.approx(3.75)
    assert diagnostic.worst_joint_scaled_error == pytest.approx(4.0)
    assert diagnostic.median_joint_scaled_error != pytest.approx(
        (diagnostic.median_smae + diagnostic.median_srmse) / 2.0
    )
    assert diagnostic.worst_joint_scaled_error != pytest.approx(
        (diagnostic.worst_smae + diagnostic.worst_srmse) / 2.0
    )


def test_hindcast_identity_is_independent_of_the_screening_policy():
    """Changing only the active-dictionary policy must not invalidate a forecast."""
    task = _task()
    config = HindcastConfig(folds=3)

    first = hindcast_cache_key(
        task,
        "same_method",
        "statistical",
        config,
        screening_policy_hash="part-1",
        runtime_settings={"checkpoint": "same"},
    )
    second = hindcast_cache_key(
        task,
        "same_method",
        "statistical",
        config,
        screening_policy_hash="part-2",
        runtime_settings={"checkpoint": "same"},
    )

    assert first == second


def test_hindcast_handles_failures_constants_and_short_histories():
    calls = 0

    def flaky(name, history, horizon, frequency):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return (history[-1],) * horizon

    constant = _task((4.0,) * 20, 3)
    diagnostic = diagnose_candidate(
        constant, "constant", "statistical", flaky, HindcastConfig(folds=3)
    )
    assert diagnostic.successful_folds == 2
    assert diagnostic.eligible
    assert math.isfinite(diagnostic.median_mase)
    assert diagnostic.folds[0].status == "failed"

    short = diagnose_candidate(
        _task((1.0, 2.0, 3.0), 2),
        "short",
        "statistical",
        lambda *args: (3.0, 3.0),
        HindcastConfig(folds=3),
    )
    assert not short.eligible
    assert short.reason_code == "insufficient_history"


def test_pairwise_diversity_is_scale_normalized():
    left = ((1.0, 2.0), (2.0, 3.0))
    same = ((1.0, 2.0), (2.0, 3.0))
    other = ((3.0, 0.0), (4.0, 1.0))
    assert pairwise_diversity(left, same) == pytest.approx(0.0)
    assert pairwise_diversity(left, other) > 0.0


def _diagnostic(name, *, median, recent=None, worst=None, mad=0.1, family="statistical", forecasts=None, truths=None):
    return CandidateDiagnostics.synthetic(
        name=name,
        family=family,
        median_mase=median,
        recent_mase=median if recent is None else recent,
        worst_mase=median if worst is None else worst,
        mase_mad=mad,
        fold_forecasts=forecasts or ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0)),
        fold_truths=truths or ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0)),
    )


def test_selector_ignores_mase_when_scaled_metrics_disagree():
    """Active ranking is Dr-CiK scaled-only, even when legacy MASE disagrees."""
    diagnostics = {
        "anchor": CandidateDiagnostics.synthetic(
            name="anchor", family="statistical", median_mase=0.01,
            median_smae=1.0, median_srmse=1.0,
        ),
        "challenger": CandidateDiagnostics.synthetic(
            name="challenger", family="statistical", median_mase=100.0,
            median_smae=0.8, median_srmse=0.8,
        ),
    }

    result = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False),
        active_names=("anchor", "challenger"),
        diagnostics=diagnostics,
        forecasts={"anchor": (1.0, 2.0), "challenger": (1.0, 2.0)},
        history=(0.0, 1.0, 2.0),
    )

    assert result.selected == ("challenger",)


def test_safe_anchor_blocks_one_metric_tail_regression():
    """A raw sRMSE tail regression cannot displace the protected Toto anchor."""
    diagnostics = {
        "toto_2_0": CandidateDiagnostics.synthetic(
            name="toto_2_0", family="tsfm", median_mase=10.0,
            median_smae=1.0, median_srmse=1.0,
        ),
        "challenger": CandidateDiagnostics.synthetic(
            name="challenger", family="statistical", median_mase=0.01,
            median_smae=0.6, median_srmse=1.2, worst_srmse_raw=2.0,
        ),
    }

    result = select_protected_safe_anchor(
        DecisionPolicy(),
        active_names=("toto_2_0", "challenger"),
        diagnostics=diagnostics,
        forecasts={"toto_2_0": (1.0, 2.0), "challenger": (1.0, 2.0)},
        horizon=2,
        fallback_reason="test",
    )

    assert result.selected == ("toto_2_0",)


def test_active_policy_rejects_legacy_error_ranking_fields():
    with pytest.raises(ValueError, match="unsupported"):
        DecisionPolicy(ranking_order=("median_mase",))


@pytest.mark.parametrize(
    "ranking_order",
    (
        ("median_smae",),
        (
            "median_joint_scaled_error",
            "recent_joint_scaled_error",
            "worst_joint_scaled_error",
            "median_smae",
        ),
        (
            "recent_smae",
            "median_joint_scaled_error",
            "recent_joint_scaled_error",
            "worst_joint_scaled_error",
            "median_smae",
            "median_srmse",
        ),
    ),
)
def test_active_policy_requires_the_complete_dual_metric_ranking_contract(
    ranking_order,
):
    with pytest.raises(ValueError, match="complete.*sMAE.*sRMSE"):
        DecisionPolicy(ranking_order=ranking_order)


def test_active_policy_parser_requires_explicit_legacy_migration_flag():
    with pytest.raises(ValueError, match="allow_legacy"):
        DecisionPolicy.from_payload({"catastrophic_mase": 2.0})

    migrated = DecisionPolicy.from_payload(
        {"catastrophic_mase": 2.0}, allow_legacy=True
    )

    assert not hasattr(migrated, "catastrophic_mase")
    assert migrated.catastrophic_smae_raw == pytest.approx(10.0)
    assert migrated.catastrophic_srmse_raw == pytest.approx(10.0)


def test_active_policy_payload_requires_every_canonical_field():
    payload = asdict(DecisionPolicy())
    payload.pop("median_mase", None)
    payload.pop("long_horizon_max_regret")

    with pytest.raises(ValueError, match="missing"):
        DecisionPolicy.from_payload(payload)

    assert DecisionPolicy.from_payload(payload, allow_legacy=True) == DecisionPolicy()


def test_hindcast_config_has_no_active_catastrophic_mase_and_legacy_is_explicit():
    active = asdict(HindcastConfig())
    assert "catastrophic_mase" not in active
    with pytest.raises(ValueError, match="legacy"):
        HindcastConfig.from_payload({**active, "catastrophic_mase": 2.0})

    migrated = HindcastConfig.from_payload(
        {"folds": 3, "catastrophic_mase": 2.0}, allow_legacy=True
    )
    assert migrated == HindcastConfig()
    assert not hasattr(migrated, "catastrophic_mase")


def test_unreliable_toto_cannot_displace_a_reliable_scaled_challenger():
    diagnostics = {
        "toto_2_0": CandidateDiagnostics.synthetic(
            name="toto_2_0", family="tsfm", median_mase=0.01,
            median_smae=0.1, median_srmse=0.1,
            worst_smae_raw=float("inf"),
        ),
        "challenger": CandidateDiagnostics.synthetic(
            name="challenger", family="statistical", median_mase=100.0,
            median_smae=0.8, median_srmse=0.8,
        ),
    }

    result = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False),
        active_names=tuple(diagnostics), diagnostics=diagnostics,
        forecasts={name: (1.0, 2.0) for name in diagnostics}, history=(1.0, 2.0),
    )

    assert result.selected == ("challenger",)


@pytest.mark.parametrize("field", (
    "catastrophic_smae_raw",
    "catastrophic_srmse_raw",
    "max_smae_fold_regret",
    "max_srmse_fold_regret",
))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_scaled_safety_thresholds_require_finite_values(field, value):
    with pytest.raises(ValueError, match="scaled safety thresholds"):
        DecisionPolicy(**{field: value})


def test_policy_parser_rejects_legacy_ranking_without_legacy_flag():
    with pytest.raises(ValueError, match="allow_legacy"):
        DecisionPolicy.from_payload({"ranking_order": ["median_mase"]})


def test_explicit_legacy_reader_rejects_median_smape_ranking_without_a_surrogate():
    payload = asdict(DecisionPolicy())
    payload["ranking_order"] = ["median_smape"]

    with pytest.raises(ValueError, match="median_smape cannot be migrated"):
        DecisionPolicy.from_payload(payload, allow_legacy=True)


def test_explicit_legacy_reader_normalizes_unpaired_ranking_to_canonical_pair() -> None:
    payload = asdict(DecisionPolicy())
    payload["ranking_order"] = [
        "median_mase",
        "recent_mase",
        "worst_mase",
        "mase_mad",
        "median_rmsse",
    ]

    assert DecisionPolicy.from_payload(payload, allow_legacy=True) == DecisionPolicy()


def _with_long_horizon_audit(diagnostic, *, forecast, truth, coverage, scale=1.0):
    def scaled_fields(fold_truth, fold_forecast):
        metrics = drcik_point_metrics(fold_truth, fold_forecast)
        return {
            key: metrics[key]
            for key in ("smae", "srmse", "smae_raw", "srmse_raw", "smae_clipped", "srmse_clipped")
        }

    folds = tuple(
        HindcastFold(
            train_end=10 * (index + 1),
            validation_end=10 * (index + 1) + len(fold_truth),
            status="success",
            forecast=tuple(float(value) for value in fold_forecast),
            truth=tuple(float(value) for value in fold_truth),
            mase_scale=float(scale),
            **scaled_fields(fold_truth, fold_forecast),
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
            forecast=tuple(float(value) for value in forecast),
            truth=tuple(float(value) for value in truth),
            mase_scale=float(scale),
            **scaled_fields(truth, forecast),
        ),
        long_horizon_coverage=float(coverage),
    )


def test_residual_correction_scale_is_independent_of_legacy_mase_diagnostics():
    base = _diagnostic(
        "anchor",
        median=1.0,
        forecasts=((2.0, 3.0),) * 3,
        truths=((1.0, 5.0),) * 3,
    )
    small_legacy_scale = _with_long_horizon_audit(
        base,
        forecast=(2.0, 3.0),
        truth=(1.0, 5.0),
        coverage=1.0,
        scale=0.01,
    )
    large_legacy_scale = _with_long_horizon_audit(
        base,
        forecast=(2.0, 3.0),
        truth=(1.0, 5.0),
        coverage=1.0,
        scale=10_000.0,
    )

    assert selector_module._fold_correction_scales(
        small_legacy_scale
    ) == selector_module._fold_correction_scales(large_legacy_scale)
    assert selector_module._fold_correction_scales(small_legacy_scale) == (4.0,) * 3


def test_context_preserving_long_horizon_audit_keeps_original_rank_folds():
    task = _task(tuple(float(index) for index in range(156)), horizon=100)
    calls = []

    def runner(name, history, horizon, frequency):
        del name, frequency
        calls.append((len(history), horizon))
        return tuple(history[-1] for _ in range(horizon))

    diagnostic = diagnose_candidate(
        task,
        "long_horizon_sensitive",
        "statistical",
        runner,
        HindcastConfig(folds=3, long_horizon_audit=True),
    )

    assert calls == [(39, 39), (78, 39), (117, 39), (104, 52)]
    assert len(diagnostic.folds) == 3
    assert diagnostic.long_horizon_fold is not None
    assert len(diagnostic.long_horizon_fold.truth) == 52
    assert diagnostic.long_horizon_coverage == pytest.approx(0.52)


def test_long_horizon_penalty_is_applied_only_when_task_route_matches():
    truths = ((-10.0, -10.0),) * 3
    baseline = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((1.0, 1.0),) * 3, truths=truths,
        ),
        forecast=(1.0,) * 4,
        truth=(-10.0,) * 4,
        coverage=1.0,
    )
    challenger = _with_long_horizon_audit(
        _diagnostic(
            "challenger", median=0.5,
            forecasts=((-1.0, -1.0),) * 3, truths=truths,
        ),
        forecast=(3.0,) * 4,
        truth=(-10.0,) * 4,
        coverage=1.0,
    )
    policy = DecisionPolicy(
        ensemble_enabled=False,
        recent_regime_first=False,
        ensemble_min_improvement=0.05,
        ensemble_min_fold_wins=2,
        ensemble_max_worst_fold_regret=0.05,
        long_horizon_audit_enabled=True,
        long_horizon_penalty_weight=1.0,
        long_horizon_route_feature="horizon_ratio",
        long_horizon_route_operator="at_least",
        long_horizon_route_threshold=0.5,
    )
    common = {
        "active_names": ("toto_2_0", "challenger"),
        "diagnostics": {"toto_2_0": baseline, "challenger": challenger},
        "forecasts": {"toto_2_0": (1.0,) * 4, "challenger": (0.5,) * 4},
    }

    matched = select_numerical_forecast(
        policy,
        profile=_profile(history_length=100, horizon=60),
        **common,
    )
    unmatched = select_numerical_forecast(
        policy,
        profile=_profile(history_length=100, horizon=10),
        **common,
    )

    assert matched.selected == ("challenger",)
    assert unmatched.selected == ("challenger",)


def test_task_conditioned_audit_scores_the_exact_combined_forecast():
    truths = ((0.0, 0.0),) * 3
    baseline = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=2.0,
            forecasts=((2.0, 2.0),) * 3, truths=truths,
        ),
        forecast=(1.0,) * 4,
        truth=(0.0,) * 4,
        coverage=1.0,
    )
    specialist = _with_long_horizon_audit(
        _diagnostic(
            "seasonal", family="statistical", median=6.0,
            forecasts=((-6.0, -6.0),) * 3, truths=truths,
        ),
        forecast=(9.0,) * 4,
        truth=(0.0,) * 4,
        coverage=1.0,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=True,
            ensemble_weight_grid=(0.75,),
            ensemble_residual_strengths=(),
            ensemble_min_diversity=0.1,
            ensemble_min_improvement=0.01,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.05,
            long_horizon_audit_enabled=True,
            long_horizon_penalty_weight=1.0,
            long_horizon_route_feature="horizon_ratio",
            long_horizon_route_operator="at_least",
            long_horizon_route_threshold=0.5,
        ),
        profile=_profile(history_length=100, horizon=60),
        active_names=("toto_2_0", "seasonal"),
        diagnostics={"toto_2_0": baseline, "seasonal": specialist},
        forecasts={"toto_2_0": (1.0,) * 4, "seasonal": (9.0,) * 4},
        history=(0.0,) * 48,
    )

    assert decision.mode == "single"
    assert decision.selected == ("toto_2_0",)


def test_change_aware_guard_rejects_override_with_long_horizon_regret():
    truths = ((0.0, 0.0),) * 3
    baseline = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((1.0, 1.0),) * 3, truths=truths,
        ),
        forecast=(1.0,) * 4,
        truth=(0.0,) * 4,
        coverage=1.0,
    )
    challenger = _with_long_horizon_audit(
        _diagnostic(
            "challenger", median=0.5,
            forecasts=((-1.0, -1.0),) * 3, truths=truths,
        ),
        forecast=(1.5,) * 4,
        truth=(0.0,) * 4,
        coverage=1.0,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=False,
            recent_regime_first=False,
            ensemble_min_improvement=0.05,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.05,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.02,
        ),
        active_names=("toto_2_0", "challenger"),
        diagnostics={"toto_2_0": baseline, "challenger": challenger},
        forecasts={"toto_2_0": (1.0,) * 4, "challenger": (0.5,) * 4},
    )

    assert decision.selected == ("toto_2_0",)
    assert "stable_baseline_protection" in decision.reason_codes


def test_change_aware_guard_allows_stable_override_with_sufficient_audit_coverage():
    truths = ((-10.0, -10.0),) * 3
    baseline = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((1.0, 1.0),) * 3, truths=truths,
        ),
        forecast=(1.0,) * 4,
        truth=(-10.0,) * 4,
        coverage=0.75,
    )
    challenger = _with_long_horizon_audit(
        _diagnostic(
            "challenger", median=0.5,
            forecasts=((-1.0, -1.0),) * 3, truths=truths,
        ),
        forecast=(1.02,) * 4,
        truth=(-10.0,) * 4,
        coverage=0.75,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=False,
            recent_regime_first=False,
            ensemble_min_improvement=0.05,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.05,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.02,
        ),
        active_names=("toto_2_0", "challenger"),
        diagnostics={"toto_2_0": baseline, "challenger": challenger},
        forecasts={"toto_2_0": (1.0,) * 4, "challenger": (0.5,) * 4},
    )

    assert decision.selected == ("challenger",)


def test_change_aware_guard_fails_closed_when_audit_coverage_is_too_short():
    truths = ((0.0, 0.0),) * 3
    baseline = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((1.0, 1.0),) * 3, truths=truths,
        ),
        forecast=(1.0,) * 4,
        truth=(0.0,) * 4,
        coverage=0.5,
    )
    challenger = _with_long_horizon_audit(
        _diagnostic(
            "challenger", median=0.5,
            forecasts=((0.5, 0.5),) * 3, truths=truths,
        ),
        forecast=(0.5,) * 4,
        truth=(0.0,) * 4,
        coverage=0.5,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=False,
            recent_regime_first=False,
            ensemble_min_improvement=0.05,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.05,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.02,
        ),
        active_names=("toto_2_0", "challenger"),
        diagnostics={"toto_2_0": baseline, "challenger": challenger},
        forecasts={"toto_2_0": (1.0,) * 4, "challenger": (0.5,) * 4},
    )

    assert decision.selected == ("toto_2_0",)


def test_conservative_tsfm_router_keeps_toto_when_timesfm_audit_regresses():
    truths = ((0.0, 0.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((1.0, 1.0),) * 3, truths=truths,
        ),
        forecast=(1.0,) * 4,
        truth=(0.0,) * 4,
        coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=0.5,
            forecasts=((0.5, 0.5),) * 3, truths=truths,
        ),
        forecast=(1.01,) * 4,
        truth=(0.0,) * 4,
        coverage=1.0,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            ensemble_enabled=False,
            recent_regime_first=False,
            ensemble_min_improvement=0.02,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.0,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=("toto_2_0", "timesfm_2_5"),
        diagnostics={"toto_2_0": toto, "timesfm_2_5": timesfm},
        forecasts={"toto_2_0": (1.0,) * 4, "timesfm_2_5": (0.5,) * 4},
    )

    assert decision.selected == ("toto_2_0",)
    assert decision.baseline_name == "toto_2_0"


def test_conservative_tsfm_router_requires_three_of_four_strict_wins():
    truths = ((-10.0, -10.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((1.0, 1.0),) * 3, truths=truths,
        ),
        forecast=(1.0,) * 4,
        truth=(-10.0,) * 4,
        coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=0.95,
            forecasts=((0.5, 0.5), (0.5, 0.5), (1.0, 1.0)), truths=truths,
        ),
        forecast=(0.5,) * 4,
        truth=(-10.0,) * 4,
        coverage=1.0,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            ensemble_enabled=False,
            recent_regime_first=False,
            ensemble_min_improvement=0.02,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.0,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=("toto_2_0", "timesfm_2_5"),
        diagnostics={"toto_2_0": toto, "timesfm_2_5": timesfm},
        forecasts={"toto_2_0": (1.0,) * 4, "timesfm_2_5": (0.5,) * 4},
    )

    assert decision.selected == ("timesfm_2_5",)
    assert decision.baseline_name == "timesfm_2_5"
    assert "conservative_tsfm_router" in decision.reason_codes


def test_conservative_tsfm_router_rejects_submargin_anchor_change():
    truths = ((0.0, 0.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((1.0, 1.0),) * 3, truths=truths,
        ),
        forecast=(1.0,) * 4,
        truth=(0.0,) * 4,
        coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=0.99,
            forecasts=((0.99, 0.99),) * 3, truths=truths,
        ),
        forecast=(0.99,) * 4,
        truth=(0.0,) * 4,
        coverage=1.0,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            ensemble_enabled=False,
            recent_regime_first=False,
            ensemble_min_improvement=0.02,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.0,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=("toto_2_0", "timesfm_2_5"),
        diagnostics={"toto_2_0": toto, "timesfm_2_5": timesfm},
        forecasts={"toto_2_0": (1.0,) * 4, "timesfm_2_5": (0.99,) * 4},
    )

    assert decision.selected == ("toto_2_0",)
    assert decision.baseline_name == "toto_2_0"


def test_conservative_router_checks_statistical_challenger_against_routed_anchor():
    truths = ((-10.0, -10.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((1.0, 1.0),) * 3, truths=truths,
        ),
        forecast=(1.0,) * 4,
        truth=(-10.0,) * 4,
        coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=0.9,
            forecasts=((0.5, 0.5),) * 3, truths=truths,
        ),
        forecast=(0.5,) * 4,
        truth=(-10.0,) * 4,
        coverage=1.0,
    )
    statistical = _with_long_horizon_audit(
        _diagnostic(
            "robust_trend", family="statistical", median=0.95,
            forecasts=((0.7, 0.7),) * 3, truths=truths,
        ),
        forecast=(0.7,) * 4,
        truth=(-10.0,) * 4,
        coverage=1.0,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            ensemble_enabled=False,
            recent_regime_first=False,
            ensemble_min_improvement=0.02,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.0,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=("toto_2_0", "timesfm_2_5", "robust_trend"),
        diagnostics={
            "toto_2_0": toto,
            "timesfm_2_5": timesfm,
            "robust_trend": statistical,
        },
        forecasts={
            "toto_2_0": (1.0,) * 4,
            "timesfm_2_5": (0.5,) * 4,
            "robust_trend": (0.7,) * 4,
        },
    )

    assert decision.selected == ("timesfm_2_5",)
    assert decision.baseline_name == "timesfm_2_5"


def test_conservative_tsfm_router_rejects_lower_smae_with_higher_srmse():
    truths = ((10.0, 10.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((9.0, 9.0),) * 3, truths=truths,
        ),
        forecast=(9.0, 9.0),
        truth=(10.0, 10.0),
        coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=0.9,
            forecasts=((10.0, 8.2),) * 3, truths=truths,
        ),
        forecast=(10.0, 8.2),
        truth=(10.0, 10.0),
        coverage=1.0,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            ensemble_enabled=False,
            recent_regime_first=False,
            ensemble_min_improvement=0.02,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.0,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=("toto_2_0", "timesfm_2_5"),
        diagnostics={"toto_2_0": toto, "timesfm_2_5": timesfm},
        forecasts={"toto_2_0": (9.0, 9.0), "timesfm_2_5": (10.0, 8.2)},
    )

    assert decision.selected == ("toto_2_0",)
    assert decision.baseline_name == "toto_2_0"


def test_conservative_combination_rejects_lower_smae_with_higher_srmse():
    truths = ((10.0, 10.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=2.0,
            forecasts=((8.0, 8.0),) * 3, truths=truths,
        ),
        forecast=(8.0, 8.0),
        truth=(10.0, 10.0),
        coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=1.0,
            forecasts=((9.0, 9.0),) * 3, truths=truths,
        ),
        forecast=(9.0, 9.0),
        truth=(10.0, 10.0),
        coverage=1.0,
    )
    specialist = _with_long_horizon_audit(
        _diagnostic(
            "spiky_specialist", family="statistical", median=4.5,
            forecasts=((14.0, 5.0),) * 3, truths=truths,
        ),
        forecast=(14.0, 5.0),
        truth=(10.0, 10.0),
        coverage=1.0,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            ensemble_enabled=True,
            recent_regime_first=False,
            ensemble_weight_grid=(0.8,),
            ensemble_residual_strengths=(),
            ensemble_min_diversity=0.1,
            ensemble_min_improvement=0.02,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.0,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=("toto_2_0", "timesfm_2_5", "spiky_specialist"),
        diagnostics={
            "toto_2_0": toto,
            "timesfm_2_5": timesfm,
            "spiky_specialist": specialist,
        },
        forecasts={
            "toto_2_0": (8.0, 8.0),
            "timesfm_2_5": (9.0, 9.0),
            "spiky_specialist": (14.0, 5.0),
        },
        history=(10.0,) * 40,
    )

    assert decision.mode == "single"
    assert decision.selected == ("timesfm_2_5",)


def test_conservative_tsfm_router_preserves_safe_anchor_when_no_reviewed_route():
    truths = ((-10.0, -10.0),) * 3
    toto = _diagnostic(
        "toto_2_0", family="tsfm", median=1.0,
        forecasts=((1.0, 1.0),) * 3, truths=truths,
    )
    chronos = _diagnostic(
        "chronos_bolt", family="tsfm", median=0.5,
        forecasts=((-1.0, -1.0), (-1.0, -1.0), (1.01, 1.01)), truths=truths,
    )
    common = {
        "active_names": ("toto_2_0", "chronos_bolt"),
        "diagnostics": {"toto_2_0": toto, "chronos_bolt": chronos},
        "forecasts": {"toto_2_0": (1.0, 1.0), "chronos_bolt": (0.5, 0.5)},
    }

    safe_anchor = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False),
        **common,
    )
    routed = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            tsfm_router_min_improvement=0.02,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert safe_anchor.selected == ("chronos_bolt",)
    assert routed.selected == safe_anchor.selected
    assert routed.forecast == safe_anchor.forecast


def test_conservative_tsfm_soft_overlay_shrinks_toward_reviewed_timesfm():
    truths = ((0.0, 1.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((3.0, 4.0),) * 3, truths=truths,
        ),
        forecast=(3.0, 4.0), truth=(0.0, 1.0), coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=0.8,
            forecasts=((1.0, 2.0),) * 3, truths=truths,
        ),
        forecast=(1.0, 2.0), truth=(0.0, 1.0), coverage=1.0,
    )
    chronos = _with_long_horizon_audit(
        _diagnostic(
            "chronos_bolt", family="tsfm", median=0.5,
            forecasts=((2.0, 3.0),) * 3, truths=truths,
        ),
        forecast=(2.0, 3.0), truth=(0.0, 1.0), coverage=1.0,
    )
    common = {
        "active_names": ("toto_2_0", "timesfm_2_5", "chronos_bolt"),
        "diagnostics": {
            "toto_2_0": toto,
            "timesfm_2_5": timesfm,
            "chronos_bolt": chronos,
        },
        "forecasts": {
            "toto_2_0": (3.0, 4.0),
            "timesfm_2_5": (1.0, 2.0),
            "chronos_bolt": (2.0, 3.0),
        },
    }

    parent = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False),
        **common,
    )
    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            tsfm_router_blend_weight=0.1,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert parent.selected == ("chronos_bolt",)
    assert child.mode == "combined"
    assert child.selected == ("chronos_bolt", "timesfm_2_5")
    assert child.weights == pytest.approx((0.9, 0.1))
    assert child.forecast == pytest.approx((1.9, 2.9))
    assert child.combination_type == "tsfm_shrinkage_overlay"
    assert "conservative_tsfm_soft_overlay" in child.reason_codes


def test_conservative_tsfm_soft_overlay_abstains_when_blend_hurts_parent():
    truths = ((0.0, 1.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((3.0, 4.0),) * 3, truths=truths,
        ),
        forecast=(3.0, 4.0), truth=(0.0, 1.0), coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=0.8,
            forecasts=((1.0, 2.0),) * 3, truths=truths,
        ),
        forecast=(1.0, 2.0), truth=(0.0, 1.0), coverage=1.0,
    )
    chronos = _with_long_horizon_audit(
        _diagnostic(
            "chronos_bolt", family="tsfm", median=0.5,
            forecasts=((0.5, 1.5),) * 3, truths=truths,
        ),
        forecast=(0.5, 1.5), truth=(0.0, 1.0), coverage=1.0,
    )
    common = {
        "active_names": ("toto_2_0", "timesfm_2_5", "chronos_bolt"),
        "diagnostics": {
            "toto_2_0": toto,
            "timesfm_2_5": timesfm,
            "chronos_bolt": chronos,
        },
        "forecasts": {
            "toto_2_0": (3.0, 4.0),
            "timesfm_2_5": (1.0, 2.0),
            "chronos_bolt": (0.5, 1.5),
        },
    }

    parent = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False),
        **common,
    )
    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            tsfm_router_blend_weight=0.1,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert child == parent


def test_conservative_tsfm_soft_overlay_rejects_a_missing_ordinary_fold():
    truths = ((0.0, 1.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((3.0, 4.0),) * 3, truths=truths,
        ),
        forecast=(3.0, 4.0), truth=(0.0, 1.0), coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=0.8,
            forecasts=((1.0, 2.0),) * 3, truths=truths,
        ),
        forecast=(1.0, 2.0), truth=(0.0, 1.0), coverage=1.0,
    )
    timesfm = replace(
        timesfm,
        folds=(
            timesfm.folds[0],
            replace(timesfm.folds[1], status="failed", forecast=()),
            timesfm.folds[2],
        ),
    )
    anchor = _with_long_horizon_audit(
        _diagnostic(
            "chronos_bolt", family="tsfm", median=0.5,
            forecasts=((2.0, 3.0),) * 3, truths=truths,
        ),
        forecast=(2.0, 3.0), truth=(0.0, 1.0), coverage=1.0,
    )
    common = {
        "active_names": tuple(("toto_2_0", "timesfm_2_5", "chronos_bolt")),
        "diagnostics": {
            "toto_2_0": toto, "timesfm_2_5": timesfm, "chronos_bolt": anchor,
        },
        "forecasts": {
            "toto_2_0": (3.0, 4.0), "timesfm_2_5": (1.0, 2.0),
            "chronos_bolt": (2.0, 3.0),
        },
    }

    parent = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False), **common
    )
    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            tsfm_router_blend_weight=0.1,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert child == parent


def test_conservative_tsfm_soft_overlay_rejects_misaligned_fold_boundaries():
    truths = ((0.0, 1.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((3.0, 4.0),) * 3, truths=truths,
        ),
        forecast=(3.0, 4.0), truth=(0.0, 1.0), coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=0.8,
            forecasts=((1.0, 2.0),) * 3, truths=truths,
        ),
        forecast=(1.0, 2.0), truth=(0.0, 1.0), coverage=1.0,
    )
    timesfm = replace(
        timesfm,
        folds=(
            timesfm.folds[0],
            replace(timesfm.folds[1], train_end=timesfm.folds[1].train_end + 1),
            timesfm.folds[2],
        ),
    )
    anchor = _with_long_horizon_audit(
        _diagnostic(
            "chronos_bolt", family="tsfm", median=0.5,
            forecasts=((2.0, 3.0),) * 3, truths=truths,
        ),
        forecast=(2.0, 3.0), truth=(0.0, 1.0), coverage=1.0,
    )
    common = {
        "active_names": tuple(("toto_2_0", "timesfm_2_5", "chronos_bolt")),
        "diagnostics": {
            "toto_2_0": toto, "timesfm_2_5": timesfm, "chronos_bolt": anchor,
        },
        "forecasts": {
            "toto_2_0": (3.0, 4.0), "timesfm_2_5": (1.0, 2.0),
            "chronos_bolt": (2.0, 3.0),
        },
    }

    parent = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False), **common
    )
    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            tsfm_router_blend_weight=0.1,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert child == parent


def test_conservative_tsfm_soft_overlay_uses_largest_strictly_safe_weight():
    truths = ((10.0, 10.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((30.0, 30.0),) * 3, truths=truths,
        ),
        forecast=(30.0, 30.0), truth=(10.0, 10.0), coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=0.8,
            forecasts=((1.0, 7.0),) * 3, truths=truths,
        ),
        forecast=(7.0, 7.0), truth=(10.0, 10.0), coverage=1.0,
    )
    anchor = _with_long_horizon_audit(
        _diagnostic(
            "chronos_bolt", family="tsfm", median=0.5,
            forecasts=((11.0, 11.0),) * 3, truths=truths,
        ),
        forecast=(11.0, 11.0), truth=(10.0, 10.0), coverage=1.0,
    )
    common = {
        "active_names": ("toto_2_0", "timesfm_2_5", "chronos_bolt"),
        "diagnostics": {
            "toto_2_0": toto,
            "timesfm_2_5": timesfm,
            "chronos_bolt": anchor,
        },
        "forecasts": {
            "toto_2_0": (30.0, 30.0),
            "timesfm_2_5": (1.0, 7.0),
            "chronos_bolt": (11.0, 11.0),
        },
    }

    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            tsfm_router_blend_weight=0.25,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert child.mode == "combined"
    assert child.selected == ("chronos_bolt", "timesfm_2_5")
    assert child.weights == pytest.approx((0.9, 0.1))
    assert child.forecast == pytest.approx((10.0, 10.6))


def test_conservative_tsfm_soft_overlay_can_fall_back_to_five_percent():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=1.0,
                forecasts=((50.0, 50.0),) * 3, truths=truths,
            ),
            forecast=(50.0, 50.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "timesfm_2_5": _with_long_horizon_audit(
            _diagnostic(
                "timesfm_2_5", family="tsfm", median=0.8,
                forecasts=((-19.0, 11.0),) * 3, truths=truths,
            ),
            forecast=(7.0, 7.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "chronos_bolt": _with_long_horizon_audit(
            _diagnostic(
                "chronos_bolt", family="tsfm", median=0.5,
                forecasts=((11.0, 11.0),) * 3, truths=truths,
            ),
            forecast=(11.0, 11.0), truth=(10.0, 10.0), coverage=1.0,
        ),
    }

    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            tsfm_router_blend_weight=0.25,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={
            "toto_2_0": (50.0, 50.0),
            "timesfm_2_5": (-19.0, 11.0),
            "chronos_bolt": (11.0, 11.0),
        },
    )

    assert child.weights == pytest.approx((0.95, 0.05))
    assert child.forecast == pytest.approx((9.5, 11.0))


def test_conservative_tsfm_blend_weight_is_bounded():
    with pytest.raises(ValueError, match="blend weight"):
        DecisionPolicy(tsfm_router_blend_weight=0.6)


def test_conservative_combined_adds_a_strictly_safe_statistical_specialist():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=0.5,
                forecasts=((11.0, 11.0),) * 3, truths=truths,
            ),
            forecast=(11.0, 11.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "seasonal_specialist": _with_long_horizon_audit(
            _diagnostic(
                "seasonal_specialist", family="statistical", median=5.0,
                forecasts=((7.0, 7.0),) * 3, truths=truths,
            ),
            forecast=(7.0, 7.0), truth=(10.0, 10.0), coverage=1.0,
        ),
    }
    common = {
        "active_names": tuple(diagnostics),
        "diagnostics": diagnostics,
        "forecasts": {
            "toto_2_0": (11.0, 11.0),
            "seasonal_specialist": (7.0, 7.0),
        },
        "conditioned_names": ("seasonal_specialist",),
    }

    parent = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False),
        **common,
    )
    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_combined",
            tsfm_router_blend_weight=0.25,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert parent.selected == ("toto_2_0",)
    assert child.mode == "combined"
    assert child.selected == ("toto_2_0", "seasonal_specialist")
    assert child.weights == pytest.approx((0.75, 0.25))
    assert child.forecast == pytest.approx((10.0, 10.0))
    assert child.combination_type == "statistical_shrinkage_overlay"
    assert "conservative_statistical_soft_overlay" in child.reason_codes


def test_conservative_combined_abstains_when_any_fold_regresses():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=0.5,
                forecasts=((11.0, 11.0),) * 3, truths=truths,
            ),
            forecast=(11.0, 11.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "unstable_specialist": _with_long_horizon_audit(
            _diagnostic(
                "unstable_specialist", family="statistical", median=5.0,
                forecasts=((7.0, 7.0), (15.0, 15.0), (7.0, 7.0)), truths=truths,
            ),
            forecast=(7.0, 7.0), truth=(10.0, 10.0), coverage=1.0,
        ),
    }
    common = {
        "active_names": tuple(diagnostics),
        "diagnostics": diagnostics,
        "forecasts": {
            "toto_2_0": (11.0, 11.0),
            "unstable_specialist": (7.0, 7.0),
        },
        "conditioned_names": ("unstable_specialist",),
    }

    parent = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False), **common
    )
    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_combined",
            tsfm_router_blend_weight=0.25,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert child == parent


def test_conservative_combined_abstains_when_srmse_regresses_alone():
    """Lower sMAE cannot admit an overlay whose sRMSE becomes worse."""
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=0.5,
                forecasts=((12.0, 12.0),) * 3, truths=truths,
            ),
            forecast=(12.0, 12.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "srmse_regressing_specialist": _with_long_horizon_audit(
            _diagnostic(
                "srmse_regressing_specialist", family="statistical", median=5.0,
                forecasts=((-28.0, 30.0),) * 3, truths=truths,
            ),
            forecast=(-28.0, 30.0), truth=(10.0, 10.0), coverage=1.0,
        ),
    }
    common = {
        "active_names": tuple(diagnostics),
        "diagnostics": diagnostics,
        "forecasts": {
            "toto_2_0": (12.0, 12.0),
            "srmse_regressing_specialist": (-28.0, 30.0),
        },
        "conditioned_names": ("srmse_regressing_specialist",),
    }
    parent = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False), **common
    )
    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_combined",
            tsfm_router_blend_weight=0.25,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert child == parent


def test_conservative_combined_rejects_submargin_fold_improvements():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=0.5,
                forecasts=((11.0, 11.0),) * 3, truths=truths,
            ),
            forecast=(11.0, 11.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "tiny_correction": _with_long_horizon_audit(
            _diagnostic(
                "tiny_correction", family="statistical", median=5.0,
                forecasts=((10.99, 10.99),) * 3, truths=truths,
            ),
            forecast=(10.99, 10.99), truth=(10.0, 10.0), coverage=1.0,
        ),
    }
    common = {
        "active_names": tuple(diagnostics),
        "diagnostics": diagnostics,
        "forecasts": {
            "toto_2_0": (11.0, 11.0),
            "tiny_correction": (10.99, 10.99),
        },
        "conditioned_names": ("tiny_correction",),
    }

    parent = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False), **common
    )
    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_combined",
            tsfm_router_min_improvement=0.02,
            tsfm_router_blend_weight=0.25,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert child == parent


def test_conservative_combined_does_not_use_an_unconditioned_statistical_method():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=0.5,
                forecasts=((11.0, 11.0),) * 3, truths=truths,
            ),
            forecast=(11.0, 11.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "broad_statistical": _with_long_horizon_audit(
            _diagnostic(
                "broad_statistical", family="statistical", median=5.0,
                forecasts=((7.0, 7.0),) * 3, truths=truths,
            ),
            forecast=(7.0, 7.0), truth=(10.0, 10.0), coverage=1.0,
        ),
    }
    common = {
        "active_names": tuple(diagnostics),
        "diagnostics": diagnostics,
        "forecasts": {
            "toto_2_0": (11.0, 11.0),
            "broad_statistical": (7.0, 7.0),
        },
        "conditioned_names": (),
    }

    parent = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False), **common
    )
    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_combined",
            tsfm_router_blend_weight=0.25,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert child == parent


def test_conservative_combined_rejects_misaligned_long_audit_boundaries():
    truths = ((10.0, 10.0),) * 3
    anchor = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=0.5,
            forecasts=((11.0, 11.0),) * 3, truths=truths,
        ),
        forecast=(11.0, 11.0), truth=(10.0, 10.0), coverage=1.0,
    )
    specialist = _with_long_horizon_audit(
        _diagnostic(
            "seasonal_specialist", family="statistical", median=5.0,
            forecasts=((7.0, 7.0),) * 3, truths=truths,
        ),
        forecast=(7.0, 7.0), truth=(10.0, 10.0), coverage=1.0,
    )
    specialist = replace(
        specialist,
        long_horizon_fold=replace(
            specialist.long_horizon_fold,
            train_end=specialist.long_horizon_fold.train_end + 1,
        ),
    )
    common = {
        "active_names": ("toto_2_0", "seasonal_specialist"),
        "diagnostics": {
            "toto_2_0": anchor,
            "seasonal_specialist": specialist,
        },
        "forecasts": {
            "toto_2_0": (11.0, 11.0),
            "seasonal_specialist": (7.0, 7.0),
        },
        "conditioned_names": ("seasonal_specialist",),
    }

    parent = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, recent_regime_first=False), **common
    )
    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_combined",
            tsfm_router_blend_weight=0.25,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        **common,
    )

    assert child == parent


def test_conservative_combined_search_includes_a_task_conditioned_specialist():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=0.5,
                forecasts=((11.0, 11.0),) * 3, truths=truths,
            ),
            forecast=(11.0, 11.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        **{
            f"broad_{index}": _with_long_horizon_audit(
                _diagnostic(
                    f"broad_{index}", family="statistical", median=1.0 + index,
                    forecasts=((12.0, 12.0),) * 3, truths=truths,
                ),
                forecast=(12.0, 12.0), truth=(10.0, 10.0), coverage=1.0,
            )
            for index in range(3)
        },
        "matched_specialist": _with_long_horizon_audit(
            _diagnostic(
                "matched_specialist", family="statistical", median=9.0,
                forecasts=((7.0, 7.0),) * 3, truths=truths,
            ),
            forecast=(7.0, 7.0), truth=(10.0, 10.0), coverage=1.0,
        ),
    }

    child = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_combined",
            tsfm_router_blend_weight=0.25,
            ensemble_enabled=False,
            recent_regime_first=False,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={name: diagnostic.long_horizon_fold.forecast
                   for name, diagnostic in diagnostics.items()},
        conditioned_names=("matched_specialist",),
    )

    assert child.selected == ("toto_2_0", "matched_specialist")
    assert "task_conditioned_statistical_specialist" in child.reason_codes


def test_joint_portfolio_combines_two_complementary_tsfms_before_statistics():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=1.0,
                forecasts=((13.0, 9.0),) * 3, truths=truths,
            ),
            forecast=(13.0, 9.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "timesfm_2_5": _with_long_horizon_audit(
            _diagnostic(
                "timesfm_2_5", family="tsfm", median=1.0,
                forecasts=((9.0, 13.0),) * 3, truths=truths,
            ),
            forecast=(9.0, 13.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "chronos_bolt": _with_long_horizon_audit(
            _diagnostic(
                "chronos_bolt", family="tsfm", median=3.0,
                forecasts=((16.0, 16.0),) * 3, truths=truths,
            ),
            forecast=(16.0, 16.0), truth=(10.0, 10.0), coverage=1.0,
        ),
    }

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm_portfolio",
            tsfm_router_min_improvement=0.02,
            tsfm_router_blend_weight=0.5,
            ensemble_enabled=False,
            recent_regime_first=False,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={
            name: diagnostic.long_horizon_fold.forecast
            for name, diagnostic in diagnostics.items()
        },
    )

    assert decision.mode == "combined"
    assert decision.selected == ("timesfm_2_5", "toto_2_0")
    assert decision.weights == pytest.approx((0.5, 0.5))
    assert decision.forecast == pytest.approx((11.0, 11.0))
    assert decision.combination_type == "tsfm_weighted_portfolio"
    assert "conservative_multi_tsfm_portfolio" in decision.reason_codes


def test_conservative_single_tsfm_never_selects_a_statistical_challenger():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=1.0,
                forecasts=((11.0, 11.0),) * 3, truths=truths,
            ),
            forecast=(11.0, 11.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "perfect_statistical": _with_long_horizon_audit(
            _diagnostic(
                "perfect_statistical", family="statistical", median=0.0,
                forecasts=truths, truths=truths,
            ),
            forecast=(10.0, 10.0), truth=(10.0, 10.0), coverage=1.0,
        ),
    }

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_single_tsfm",
            ensemble_enabled=False,
            recent_regime_first=False,
            long_horizon_guard_enabled=True,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={
            "toto_2_0": (11.0, 11.0),
            "perfect_statistical": (10.0, 10.0),
        },
    )

    assert decision.mode == "single"
    assert decision.selected == ("toto_2_0",)


def test_conservative_single_tsfm_falls_back_to_toto_when_audits_are_incomplete():
    diagnostic = CandidateDiagnostics.synthetic(
        name="toto_2_0",
        family="tsfm",
        median_mase=math.inf,
        eligible=False,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_single_tsfm",
            ensemble_enabled=False,
        ),
        active_names=("toto_2_0",),
        diagnostics={"toto_2_0": diagnostic},
        forecasts={"toto_2_0": (4.0, 4.0)},
    )

    assert decision.selected == ("toto_2_0",)
    assert decision.forecast == (4.0, 4.0)
    assert "unverified_tsfm_fallback" in decision.reason_codes


def test_joint_portfolio_adds_one_conditioned_statistical_specialist():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=1.0,
                forecasts=((13.0, 9.0),) * 3, truths=truths,
            ),
            forecast=(13.0, 9.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "timesfm_2_5": _with_long_horizon_audit(
            _diagnostic(
                "timesfm_2_5", family="tsfm", median=1.0,
                forecasts=((9.0, 13.0),) * 3, truths=truths,
            ),
            forecast=(9.0, 13.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "seasonal_specialist": _with_long_horizon_audit(
            _diagnostic(
                "seasonal_specialist", family="statistical", median=4.0,
                forecasts=((7.0, 7.0),) * 3, truths=truths,
            ),
            forecast=(7.0, 7.0), truth=(10.0, 10.0), coverage=1.0,
        ),
    }

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_joint_portfolio",
            tsfm_router_min_improvement=0.02,
            tsfm_router_blend_weight=0.25,
            ensemble_enabled=False,
            recent_regime_first=False,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={
            name: diagnostic.long_horizon_fold.forecast
            for name, diagnostic in diagnostics.items()
        },
        conditioned_names=("seasonal_specialist",),
    )

    assert decision.mode == "combined"
    assert decision.selected == ("timesfm_2_5", "toto_2_0", "seasonal_specialist")
    assert decision.weights == pytest.approx((0.375, 0.375, 0.25))
    assert decision.forecast == pytest.approx((10.0, 10.0))
    assert decision.combination_type == "joint_tsfm_statistical_portfolio"
    assert "task_conditioned_statistical_specialist" in decision.reason_codes


def test_joint_portfolio_abstains_when_the_long_audit_regresses():
    truths = ((10.0, 10.0),) * 3
    toto = _with_long_horizon_audit(
        _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((13.0, 9.0),) * 3, truths=truths,
        ),
        forecast=(10.0, 10.0), truth=(10.0, 10.0), coverage=1.0,
    )
    timesfm = _with_long_horizon_audit(
        _diagnostic(
            "timesfm_2_5", family="tsfm", median=1.0,
            forecasts=((9.0, 13.0),) * 3, truths=truths,
        ),
        forecast=(14.0, 14.0), truth=(10.0, 10.0), coverage=1.0,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm_portfolio",
            tsfm_router_min_improvement=0.02,
            tsfm_router_blend_weight=0.5,
            ensemble_enabled=False,
            recent_regime_first=False,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=("toto_2_0", "timesfm_2_5"),
        diagnostics={"toto_2_0": toto, "timesfm_2_5": timesfm},
        forecasts={
            "toto_2_0": (10.0, 10.0),
            "timesfm_2_5": (14.0, 14.0),
        },
    )

    assert decision.mode == "single"
    assert decision.selected == ("toto_2_0",)


def test_protected_single_tsfm_keeps_parent_when_one_history_fold_regresses():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=2.0,
                forecasts=((12.0, 12.0),) * 3, truths=truths,
            ),
            forecast=(12.0, 12.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "timesfm_2_5": _with_long_horizon_audit(
            _diagnostic(
                "timesfm_2_5", family="tsfm", median=0.0,
                forecasts=((10.0, 10.0), (10.0, 10.0), (13.0, 13.0)),
                truths=truths,
            ),
            forecast=(10.0, 10.0), truth=(10.0, 10.0), coverage=1.0,
        ),
    }

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="protected_single_tsfm",
            ensemble_enabled=False,
            recent_regime_first=False,
            tsfm_router_min_improvement=0.02,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={"toto_2_0": (12.0, 12.0), "timesfm_2_5": (10.0, 10.0)},
    )

    assert decision.selected == ("toto_2_0",)
    assert "protected_parent_reference" in decision.reason_codes


def test_protected_topk_single_tsfm_preserves_the_minimax_parent():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0",
                family="tsfm",
                median=1.0,
                worst=20.0,
                forecasts=((11.0, 11.0), (11.0, 11.0), (30.0, 30.0)),
                truths=truths,
            ),
            forecast=(11.0, 11.0),
            truth=(10.0, 10.0),
            coverage=1.0,
        ),
        "timesfm_2_5": _with_long_horizon_audit(
            _diagnostic(
                "timesfm_2_5",
                family="tsfm",
                median=1.1,
                worst=1.1,
                forecasts=((11.1, 11.1),) * 3,
                truths=truths,
            ),
            forecast=(11.1, 11.1),
            truth=(10.0, 10.0),
            coverage=1.0,
        ),
    }

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="protected_topk_single_tsfm",
            assumption_guidance_enabled=True,
            ensemble_enabled=False,
            recent_regime_first=False,
            tsfm_router_min_improvement=0.02,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={"toto_2_0": (11.0, 11.0), "timesfm_2_5": (11.1, 11.1)},
    )

    assert decision.selected == ("timesfm_2_5",)
    assert "protected_parent_reference" in decision.reason_codes


def test_protected_multi_tsfm_portfolio_can_beat_the_parent_safely():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=2.0,
                forecasts=((13.0, 9.0),) * 3, truths=truths,
            ),
            forecast=(13.0, 9.0), truth=(10.0, 10.0), coverage=1.0,
        ),
        "timesfm_2_5": _with_long_horizon_audit(
            _diagnostic(
                "timesfm_2_5", family="tsfm", median=2.0,
                forecasts=((9.0, 13.0),) * 3, truths=truths,
            ),
            forecast=(9.0, 13.0), truth=(10.0, 10.0), coverage=1.0,
        ),
    }

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="protected_tsfm_portfolio",
            ensemble_enabled=False,
            recent_regime_first=False,
            tsfm_router_min_improvement=0.02,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={"toto_2_0": (13.0, 9.0), "timesfm_2_5": (9.0, 13.0)},
    )

    assert decision.selected == ("timesfm_2_5", "toto_2_0")
    assert decision.weights == pytest.approx((0.5, 0.5))
    assert decision.forecast == pytest.approx((11.0, 11.0))
    assert decision.combination_type == "protected_tsfm_weighted_portfolio"


def test_protected_joint_uses_a_clipped_statistical_residual_correction():
    truths = ((10.0, 10.0),) * 3
    diagnostics = {
        "toto_2_0": _with_long_horizon_audit(
            _diagnostic(
                "toto_2_0", family="tsfm", median=2.0,
                forecasts=((12.0, 12.0),) * 3, truths=truths,
            ),
            forecast=(12.0, 12.0), truth=(10.0, 10.0), coverage=1.0, scale=1.0,
        ),
        "downward_specialist": _with_long_horizon_audit(
            _diagnostic(
                "downward_specialist", family="statistical", median=10.0,
                forecasts=((0.0, 0.0),) * 3, truths=truths,
            ),
            forecast=(0.0, 0.0), truth=(10.0, 10.0), coverage=1.0, scale=1.0,
        ),
    }

    decision = select_numerical_forecast(
        DecisionPolicy(
            baseline_strategy="protected_joint_residual",
            ensemble_enabled=False,
            recent_regime_first=False,
            tsfm_router_min_improvement=0.02,
            ensemble_residual_strengths=(0.2,),
            ensemble_correction_clip=1.0,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={"toto_2_0": (12.0, 12.0), "downward_specialist": (0.0, 0.0)},
        history=(8.0, 9.0, 10.0, 11.0),
        conditioned_names=("downward_specialist",),
    )

    assert decision.selected == ("toto_2_0", "downward_specialist")
    assert decision.forecast == pytest.approx((11.8, 11.8))
    assert decision.combination_type == "protected_statistical_residual"
    assert "bounded_statistical_residual" in decision.reason_codes


def _profile(**changes):
    payload = {
        "task_id": "history-only-id",
        "frequency": "D",
        "history_length": 40,
        "horizon": 2,
        "zero_fraction": 0.0,
        "signed": False,
        "integer_valued": False,
        "trend_direction": "flat",
        "trend_strength": 0.1,
        "periodicity_periods": (),
        "periodicity_strength": 0.0,
        "periodicity_confidence": 0.0,
        "outlier_fraction": 0.0,
        "noise_relative_scale": 0.2,
        "likely_stationary": False,
        "stationarity_score": 0.2,
        "recent_regime_start": None,
        "recent_regime_confidence": 0.0,
        "intermittency_adi": 1.0,
        "intermittency_cv2": 0.1,
    }
    payload.update(changes)
    return TaskProfile(**payload)


def test_assumption_guidance_excludes_methods_unrelated_to_supported_history():
    truths = ((-10.0, -10.0),) * 3
    diagnostics = {
        "toto_2_0": _diagnostic(
            "toto_2_0", family="tsfm", median=2.0, worst=2.0,
            forecasts=((2.0, 2.0),) * 3, truths=truths,
        ),
        "seasonal_naive": _diagnostic(
            "seasonal_naive", median=0.0, worst=0.0,
            forecasts=truths, truths=truths,
        ),
        "linear_trend_regression": _diagnostic(
            "linear_trend_regression", median=0.0, worst=0.0,
            forecasts=truths, truths=truths,
        ),
    }
    decision = select_assumption_guided_forecast(
        DecisionPolicy(
            assumption_guidance_enabled=True,
            assumption_top_k=2,
            assumption_candidates_per_hypothesis=1,
            assumption_min_confidence=0.25,
            ensemble_enabled=False,
            ensemble_min_improvement=0.0,
            ensemble_min_fold_wins=3,
            ensemble_max_worst_fold_regret=0.0,
        ),
        profile=_profile(
            periodicity_periods=(7,),
            periodicity_strength=0.9,
            periodicity_confidence=0.9,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={name: (0.0, 0.0) for name in diagnostics},
        families={
            "toto_2_0": "tsfm",
            "seasonal_naive": "statistical",
            "linear_trend_regression": "statistical",
        },
        history=tuple(float(index % 7) for index in range(42)),
    )

    assert decision.selected == ("seasonal_naive",)
    assert "periodic_persistence_p7" in decision.assumption_ids
    assert "linear_trend_regression" not in decision.considered_candidates


def test_assumption_guidance_always_retains_reviewed_tsfm_anchor():
    diagnostics = {
        "toto_2_0": _diagnostic("toto_2_0", family="tsfm", median=1.0),
        "unsupported_specialist": _diagnostic("unsupported_specialist", median=0.01),
    }
    decision = select_assumption_guided_forecast(
        DecisionPolicy(
            assumption_guidance_enabled=True,
            assumption_top_k=1,
            assumption_candidates_per_hypothesis=1,
            ensemble_enabled=False,
        ),
        profile=_profile(),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={name: (1.0, 2.0) for name in diagnostics},
        families={"toto_2_0": "tsfm", "unsupported_specialist": "statistical"},
        history=(1.0,) * 40,
    )

    assert decision.selected == ("toto_2_0",)
    assert decision.baseline_name == "toto_2_0"
    assert decision.considered_candidates == ("toto_2_0",)


def test_assumption_guidance_rejects_period_seen_only_at_recent_cutoffs():
    """Removing cross-cutoff validation must re-admit a temporary cycle."""
    rng = random.Random(42)
    history = tuple(rng.gauss(0.0, 1.0) for _ in range(42)) + tuple(
        float((index % 7) * 3) for index in range(21)
    )
    task = Task("temporary-cycle", history, 7, "D", (0.0,) * 7)
    profile = profile_task(task)
    assert profile.periodicity_periods[0] == 7

    truths = ((0.0, 0.0),) * 3
    diagnostics = {
        "toto_2_0": _diagnostic(
            "toto_2_0", family="tsfm", median=1.0,
            forecasts=((1.0, 1.0),) * 3, truths=truths,
        ),
        "combined_timesfm_seasonal": _diagnostic(
            "combined_timesfm_seasonal", family="combined", median=0.1,
            forecasts=((0.1, 0.1),) * 3, truths=truths,
        ),
    }
    decision = select_assumption_guided_forecast(
        DecisionPolicy(
            assumption_guidance_enabled=True,
            assumption_top_k=3,
            assumption_candidates_per_hypothesis=1,
            assumption_min_confidence=0.0,
            ensemble_enabled=False,
            recent_regime_first=False,
            ensemble_min_improvement=0.0,
            ensemble_min_fold_wins=1,
            ensemble_max_worst_fold_regret=1.0,
        ),
        profile=profile,
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={name: (0.0,) * 7 for name in diagnostics},
        families={
            "toto_2_0": "tsfm",
            "combined_timesfm_seasonal": "combined",
        },
        history=history,
    )

    assert "periodic_persistence_p7" not in decision.assumption_ids
    assert "combined_timesfm_seasonal" not in decision.considered_candidates


def test_disabling_assumption_guidance_preserves_the_flat_selector_contract():
    diagnostics = {
        "alpha": _diagnostic("alpha", median=0.1),
        "beta": _diagnostic("beta", median=0.2),
    }
    forecasts = {name: (1.0, 2.0) for name in diagnostics}
    policy = DecisionPolicy(assumption_guidance_enabled=False, ensemble_enabled=False)

    flat = select_numerical_forecast(
        policy,
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts=forecasts,
    )
    guided_api = select_assumption_guided_forecast(
        policy,
        profile=_profile(),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts=forecasts,
        families={name: "statistical" for name in diagnostics},
    )

    assert guided_api == flat


def test_selector_rejects_inactive_failed_and_catastrophic_candidates():
    diagnostics = {
        "good": _diagnostic("good", median=0.8),
        "inactive": _diagnostic("inactive", median=0.1),
        "failed": CandidateDiagnostics.synthetic(
            name="failed", family="statistical", median_mase=0.01, eligible=False
        ),
        "tail": _diagnostic("tail", median=0.2, worst=20.0),
    }
    forecasts = {name: (1.0, 2.0) for name in diagnostics}
    decision = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False),
        active_names=("good", "failed", "tail"),
        diagnostics=diagnostics,
        forecasts=forecasts,
    )
    assert decision.selected == ("good",)
    assert decision.mode == "single"


def test_minimax_baseline_strategy_anchors_to_the_tsfm_with_the_best_worst_fold():
    diagnostics = {
        "toto_2_0": _diagnostic(
            "toto_2_0", family="tsfm", median=1.0, recent=0.9, worst=5.0
        ),
        "timesfm_2_5": _diagnostic(
            "timesfm_2_5", family="tsfm", median=1.1, recent=1.0, worst=1.2
        ),
    }
    forecasts = {name: (1.0, 2.0) for name in diagnostics}

    fixed = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, baseline_strategy="toto_first"),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts=forecasts,
    )
    minimax = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, baseline_strategy="minimax_tsfm"),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts=forecasts,
    )

    assert fixed.baseline_name == "toto_2_0"
    assert minimax.baseline_name == "timesfm_2_5"


def test_selector_uses_pareto_then_recent_and_deterministic_name_tie_break():
    diagnostics = {
        "zeta": _diagnostic("zeta", median=1.0, recent=0.8, worst=1.2),
        "alpha": _diagnostic("alpha", median=1.0, recent=0.8, worst=1.2),
        "dominated": _diagnostic("dominated", median=1.2, recent=1.4, worst=2.0),
    }
    decision = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={name: (1.0, 2.0) for name in diagnostics},
    )
    assert decision.selected == ("alpha",)
    assert "pareto_front" in decision.reason_codes


def test_guarded_ensemble_requires_diversity_and_historical_improvement():
    truths = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    diagnostics = {
        "positive": _diagnostic(
            "positive", median=1.0,
            forecasts=((2.0, 2.0), (2.0, 2.0), (2.0, 2.0)), truths=truths,
        ),
        "negative": _diagnostic(
            "negative", median=1.0,
            forecasts=((-2.0, -2.0), (-2.0, -2.0), (-2.0, -2.0)), truths=truths,
        ),
    }
    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=True,
            ensemble_min_diversity=0.1,
            ensemble_min_improvement=0.01,
        ),
        active_names=("positive", "negative"),
        diagnostics=diagnostics,
        forecasts={"positive": (2.0, 2.0), "negative": (-2.0, -2.0)},
    )
    assert decision.mode == "single"
    assert decision.selected == ("negative",)

    duplicate = dict(diagnostics)
    duplicate["negative"] = _diagnostic(
        "negative", median=1.0,
        forecasts=diagnostics["positive"].fold_forecasts, truths=truths,
    )
    single = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=True, ensemble_min_diversity=0.1),
        active_names=("positive", "negative"),
        diagnostics=duplicate,
        forecasts={"positive": (2.0, 2.0), "negative": (2.0, 2.0)},
    )
    assert single.mode == "single"


def test_same_family_ensemble_rejects_audit_srmse_regret_despite_joint_gain():
    """A scalar joint gain cannot hide the ensemble's long-audit RMSE regression."""
    truths = ((10.0, 10.0),) * 3
    anchor = _with_long_horizon_audit(
        _diagnostic(
            "anchor", median=0.2,
            forecasts=((13.0, 9.0),) * 3, truths=truths,
        ),
        forecast=(12.0, 12.0), truth=(10.0, 10.0), coverage=1.0,
    )
    peer = _with_long_horizon_audit(
        _diagnostic(
            "peer", median=0.3,
            forecasts=((9.0, 13.0),) * 3, truths=truths,
        ),
        forecast=(8.0, 14.0), truth=(10.0, 10.0), coverage=1.0,
    )

    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=True,
            ensemble_min_diversity=0.1,
            ensemble_min_fold_wins=2,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=1.0,
            long_horizon_max_regret=0.0,
        ),
        active_names=("anchor", "peer"),
        diagnostics={"anchor": anchor, "peer": peer},
        forecasts={"anchor": (13.0, 9.0), "peer": (9.0, 13.0)},
    )

    assert decision.mode == "single"
    assert decision.selected == ("anchor",)


def test_dynamic_combined_searches_asymmetric_tsfm_statistical_weights():
    truths = ((0.5, 0.5),) * 3
    diagnostics = {
        "toto_2_0": _diagnostic(
            "toto_2_0",
            family="tsfm",
            median=2.0,
            forecasts=((2.0, 2.0),) * 3,
            truths=truths,
        ),
        "seasonal": _diagnostic(
            "seasonal",
            family="statistical",
            median=6.0,
            forecasts=((-6.0, -6.0),) * 3,
            truths=truths,
        ),
    }
    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=True,
            ensemble_weight_grid=(0.75,),
            ensemble_residual_strengths=(),
            ensemble_min_diversity=0.1,
            ensemble_min_improvement=0.01,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.05,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={"toto_2_0": (2.0, 2.0), "seasonal": (-6.0, -6.0)},
        history=(0.0,) * 20,
    )

    assert decision.mode == "combined"
    assert decision.combination_type == "weighted_blend"
    assert decision.selected == ("toto_2_0", "seasonal")
    assert decision.weights == pytest.approx((0.75, 0.25))
    assert decision.forecast == pytest.approx((0.0, 0.0))


def test_conditioned_specialist_expands_guarded_combination_search():
    truths = ((0.5, 0.5),) * 3
    diagnostics = {
        "toto_2_0": _diagnostic(
            "toto_2_0",
            family="tsfm",
            median=2.0,
            forecasts=((2.0, 2.0),) * 3,
            truths=truths,
        ),
        **{
            f"broad_{index}": _diagnostic(
                f"broad_{index}",
                family="statistical",
                median=0.1 + index * 0.01,
                forecasts=((2.0, 2.0),) * 3,
                truths=truths,
            )
            for index in range(3)
        },
        "matched_specialist": _diagnostic(
            "matched_specialist",
            family="statistical",
            median=6.0,
            forecasts=((-6.0, -6.0),) * 3,
            truths=truths,
        ),
    }

    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=True,
            ensemble_weight_grid=(0.75,),
            ensemble_residual_strengths=(),
            ensemble_min_diversity=0.1,
            ensemble_min_improvement=0.01,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.05,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={
            "toto_2_0": (2.0, 2.0),
            "broad_0": (2.0, 2.0),
            "broad_1": (2.0, 2.0),
            "broad_2": (2.0, 2.0),
            "matched_specialist": (-6.0, -6.0),
        },
        history=(0.0,) * 20,
        conditioned_names=("matched_specialist",),
    )

    assert decision.mode == "combined"
    assert decision.selected == ("toto_2_0", "matched_specialist")
    assert "task_conditioned_specialist" in decision.reason_codes


def test_dynamic_combined_rejects_median_gain_with_bad_worst_fold():
    truths = ((0.0, 0.0),) * 3
    diagnostics = {
        "toto_2_0": _diagnostic(
            "toto_2_0",
            family="tsfm",
            median=1.0,
            forecasts=((1.0, 1.0),) * 3,
            truths=truths,
        ),
        "unstable_stat": _diagnostic(
            "unstable_stat",
            family="statistical",
            median=1.0,
            forecasts=((-1.0, -1.0), (-1.0, -1.0), (10.0, 10.0)),
            truths=truths,
        ),
    }
    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=True,
            ensemble_weight_grid=(0.75,),
            ensemble_residual_strengths=(),
            ensemble_min_diversity=0.1,
            ensemble_min_improvement=0.01,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.05,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={"toto_2_0": (1.0, 1.0), "unstable_stat": (-1.0, -1.0)},
        history=(0.0,) * 20,
    )

    assert decision.mode == "single"
    assert decision.selected == ("toto_2_0",)


def test_dynamic_combined_can_use_clipped_residual_correction():
    truths = ((0.5, 0.5),) * 3
    diagnostics = {
        "toto_2_0": _diagnostic(
            "toto_2_0",
            family="tsfm",
            median=2.0,
            forecasts=((2.0, 2.0),) * 3,
            truths=truths,
        ),
        "wild_stat": _diagnostic(
            "wild_stat",
            family="statistical",
            median=6.0,
            forecasts=((-6.0, -6.0),) * 3,
            truths=truths,
        ),
    }
    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=True,
            ensemble_weight_grid=(),
            ensemble_residual_strengths=(0.5,),
            ensemble_correction_clip=0.5,
            ensemble_min_diversity=0.1,
            ensemble_min_improvement=0.01,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.05,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={"toto_2_0": (2.0, 2.0), "wild_stat": (-6.0, -6.0)},
        history=(0.0,) * 20,
    )

    assert decision.mode == "combined"
    assert decision.combination_type == "residual_correction"
    assert decision.weights == pytest.approx((0.5, 0.5))
    assert decision.forecast == pytest.approx((1.75, 1.75))


def test_stable_tsfm_baseline_blocks_tail_risky_single_challenger():
    truths = ((0.0, 0.0),) * 3
    diagnostics = {
        "toto_2_0": _diagnostic(
            "toto_2_0",
            family="tsfm",
            median=1.0,
            recent=1.0,
            worst=1.0,
            forecasts=((1.0, 1.0),) * 3,
            truths=truths,
        ),
        "tail_risky": _diagnostic(
            "tail_risky",
            family="statistical",
            median=0.5,
            recent=2.0,
            worst=2.0,
            forecasts=((0.5, 0.5), (0.5, 0.5), (2.0, 2.0)),
            truths=truths,
        ),
    }
    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=False,
            recent_regime_first=False,
            ensemble_min_improvement=0.01,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.05,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={"toto_2_0": (1.0, 1.0), "tail_risky": (0.5, 0.5)},
        history=(0.0,) * 20,
    )

    assert decision.selected == ("toto_2_0",)
    assert "stable_baseline_protection" in decision.reason_codes


def test_fixed_combined_challenger_uses_stricter_baseline_protection():
    """Treating a fixed Combined method like a generic single must fail this test."""
    truths = ((0.0, 0.0),) * 3
    diagnostics = {
        "toto_2_0": _diagnostic(
            "toto_2_0", family="tsfm", median=1.0, recent=1.0, worst=1.0,
            forecasts=((1.0, 1.0),) * 3, truths=truths,
        ),
        "combined_timesfm_seasonal": _diagnostic(
            "combined_timesfm_seasonal", family="combined",
            median=0.96, recent=1.01, worst=1.01,
            forecasts=((0.96, 0.96), (0.96, 0.96), (1.01, 1.01)),
            truths=truths,
        ),
    }
    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=False,
            recent_regime_first=False,
            ensemble_min_improvement=0.02,
            ensemble_min_fold_wins=2,
            ensemble_max_worst_fold_regret=0.05,
        ),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={
            "toto_2_0": (1.0, 1.0),
            "combined_timesfm_seasonal": (0.96, 0.96),
        },
        history=(0.0,) * 20,
    )

    assert decision.selected == ("toto_2_0",)
    assert "stable_baseline_protection" in decision.reason_codes


def test_unreliable_toto_cannot_displace_the_reliable_final_forecast():
    truths = ((0.0, 0.0),) * 3
    diagnostics = {
        "toto_2_0": CandidateDiagnostics.synthetic(
            name="toto_2_0",
            family="tsfm",
            median_mase=20.0,
            eligible=False,
            fold_forecasts=((1.0, 1.0),) * 3,
            fold_truths=truths,
        ),
        "tempting_stat": _diagnostic(
            "tempting_stat",
            family="statistical",
            median=0.1,
            forecasts=((0.0, 0.0),) * 3,
            truths=truths,
        ),
    }

    decision = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=True),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={"toto_2_0": (1.0, 1.0), "tempting_stat": (0.0, 0.0)},
        history=(0.0,) * 20,
    )

    assert decision.mode == "single"
    assert decision.selected == ("tempting_stat",)
    assert decision.baseline_name == "toto_2_0"
    assert "unverified_baseline_fallback" not in decision.reason_codes


def test_selector_never_returns_more_than_three_members():
    diagnostics = {
        f"m{i}": _diagnostic(f"m{i}", median=1.0 + i * 0.01)
        for i in range(5)
    }
    decision = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=True, ensemble_max_members=3),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={name: (1.0, 2.0) for name in diagnostics},
    )
    assert len(decision.selected) <= 3


def test_selector_uses_history_only_best_available_fallback_for_coverage():
    diagnostics = {
        "weak": CandidateDiagnostics.synthetic(
            name="weak", family="statistical", median_mase=2.0, eligible=False
        ),
        "less_weak": CandidateDiagnostics.synthetic(
            name="less_weak", family="statistical", median_mase=1.0, eligible=False
        ),
    }
    decision = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False, fallback_to_best_available=True),
        active_names=("weak", "less_weak"),
        diagnostics=diagnostics,
        forecasts={"weak": (4.0,), "less_weak": (3.0,)},
    )
    assert decision.selected == ("less_weak",)
    assert decision.forecast == (3.0,)
    assert "conservative_best_available_fallback" in decision.reason_codes
