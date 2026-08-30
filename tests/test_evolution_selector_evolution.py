from __future__ import annotations

import json
from dataclasses import replace

import pytest

from common.llm import LLMResponse
import numerical_agent.evolution.selector_evolution as selector_evolution
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics, DecisionPolicy
from numerical_agent.evolution.selector_evolution import (
    SELECTOR_SYSTEM,
    bounded_combined_candidates,
    bounded_joint_portfolio_candidates,
    DecisionCase,
    SelectorEvolutionError,
    apply_decision_response,
    compare_train_crossfolds,
    compare_decisions,
    evaluate_decision,
    evolve_selector_train_then_dev,
    evolve_selector_generations,
    evolve_selector_once,
    parse_decision_source,
    render_decision_source,
)


def test_selector_prompt_discloses_all_strict_numeric_ranges() -> None:
    assert "strictly between 0.5 and 1.0" in SELECTOR_SYSTEM
    assert "in (0, 0.5]" in SELECTOR_SYSTEM
    assert "between one and three" in SELECTOR_SYSTEM
    assert "tsfm_router_min_improvement" in SELECTOR_SYSTEM
    assert "tsfm_router_blend_weight" in SELECTOR_SYSTEM
    assert "within [0, 1]" in SELECTOR_SYSTEM


class FakeAgent:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def complete(self, **kwargs):
        self.requests.append(kwargs)
        return LLMResponse(text=json.dumps(self.payload))


class SequenceAgent(FakeAgent):
    def __init__(self, payloads):
        super().__init__(None)
        self.payloads = iter(payloads)

    def complete(self, **kwargs):
        self.requests.append(kwargs)
        return LLMResponse(text=json.dumps(next(self.payloads)))


def _diag(name, score):
    truth = ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0))
    forecast = tuple(tuple(value + score for value in fold) for fold in truth)
    return CandidateDiagnostics.synthetic(
        name=name,
        family="statistical",
        median_mase=score,
        recent_mase=score,
        worst_mase=score,
        fold_forecasts=forecast,
        fold_truths=truth,
    )


def _case(task_id, good="a"):
    task = Task(task_id, tuple(float(i) for i in range(1, 21)), 2, "D", (1.0, 2.0))
    diagnostics = {"a": _diag("a", 0.1), "b": _diag("b", 0.2)}
    forecasts = {"a": (1.0, 2.0), "b": (4.0, 5.0)}
    if good == "b":
        diagnostics = {"a": _diag("a", 0.2), "b": _diag("b", 0.1)}
        forecasts = {"a": (4.0, 5.0), "b": (1.0, 2.0)}
    return DecisionCase(task, ("a", "b"), diagnostics, forecasts, {"a": "statistical", "b": "statistical"})


def _ranking_sensitive_case(task_id, *, child_is_better):
    task = Task(task_id, tuple(float(i) for i in range(1, 21)), 2, "D", (1.0, 2.0))
    diagnostics = {
        "a": CandidateDiagnostics.synthetic(
            name="a",
            family="statistical",
            median_mase=0.1,
            recent_mase=0.3,
            worst_mase=0.3,
        ),
        "b": CandidateDiagnostics.synthetic(
            name="b",
            family="statistical",
            median_mase=0.2,
            recent_mase=0.1,
            worst_mase=0.2,
        ),
    }
    forecasts = {
        "a": (4.0, 5.0) if child_is_better else (1.0, 2.0),
        "b": (1.0, 2.0) if child_is_better else (4.0, 5.0),
    }
    return DecisionCase(
        task,
        ("a", "b"),
        diagnostics,
        forecasts,
        {"a": "statistical", "b": "statistical"},
    )


def test_decision_policy_round_trip_and_strict_mutation_schema():
    parent = DecisionPolicy(ensemble_enabled=False)
    assert parse_decision_source(render_decision_source(parent)) == parent
    child = apply_decision_response(parent, json.dumps({
        "summary": "prefer recent evidence",
        "policy": {
            "ranking_order": ["recent_mase", "median_mase", "worst_mase", "mase_mad"],
            "recent_regime_first": True,
            "min_successful_folds": 2,
            "catastrophic_mase": 8.0,
            "ensemble_enabled": False,
            "ensemble_max_members": 2,
            "ensemble_min_diversity": 0.1,
            "ensemble_min_improvement": 0.02,
            "ensemble_weight_grid": [0.7, 0.8, 0.9],
            "ensemble_residual_strengths": [0.1, 0.25, 0.5],
            "ensemble_correction_clip": 1.5,
            "ensemble_min_fold_wins": 2,
            "ensemble_max_worst_fold_regret": 0.05,
            "fallback_to_best_available": True,
        },
    }))
    assert child.recent_regime_first
    assert child.catastrophic_mase == parent.catastrophic_mase

    for forbidden in ("future", "split", "candidates", "scorer", "screening_policy"):
        with pytest.raises(SelectorEvolutionError):
            apply_decision_response(parent, json.dumps({"summary": "bad", "policy": {forbidden: 1}}))


def test_guarded_combination_parameters_round_trip_through_evolution_schema():
    policy = DecisionPolicy(
        ensemble_weight_grid=(0.65, 0.85),
        ensemble_residual_strengths=(0.2, 0.4),
        ensemble_correction_clip=0.75,
        ensemble_min_fold_wins=3,
        ensemble_max_worst_fold_regret=0.025,
    )
    assert parse_decision_source(render_decision_source(policy)) == policy

    payload = {
        "ranking_order": list(policy.ranking_order),
        "recent_regime_first": policy.recent_regime_first,
        "min_successful_folds": policy.min_successful_folds,
        "catastrophic_mase": policy.catastrophic_mase,
        "ensemble_enabled": policy.ensemble_enabled,
        "ensemble_max_members": policy.ensemble_max_members,
        "ensemble_min_diversity": policy.ensemble_min_diversity,
        "ensemble_min_improvement": policy.ensemble_min_improvement,
        "ensemble_weight_grid": [0.7, 0.9],
        "ensemble_residual_strengths": [0.1, 0.25],
        "ensemble_correction_clip": 1.0,
        "ensemble_min_fold_wins": 2,
        "ensemble_max_worst_fold_regret": 0.04,
        "fallback_to_best_available": policy.fallback_to_best_available,
    }
    child = apply_decision_response(
        policy,
        json.dumps({"summary": "tighten tail protection", "policy": payload}),
    )
    assert child.ensemble_weight_grid == (0.7, 0.9)
    assert child.ensemble_residual_strengths == (0.1, 0.25)
    assert child.ensemble_max_worst_fold_regret == pytest.approx(0.04)


def test_task_conditioned_long_horizon_route_round_trips_and_legacy_defaults_are_safe():
    policy = DecisionPolicy(
        long_horizon_audit_enabled=True,
        long_horizon_penalty_weight=0.5,
        long_horizon_route_feature="periodicity_strength",
        long_horizon_route_operator="at_least",
        long_horizon_route_threshold=0.5,
    )
    assert parse_decision_source(render_decision_source(policy)) == policy

    legacy_source = render_decision_source(DecisionPolicy())
    for field in (
        "long_horizon_audit_enabled",
        "long_horizon_penalty_weight",
        "long_horizon_route_feature",
        "long_horizon_route_operator",
        "long_horizon_route_threshold",
    ):
        legacy_source = "\n".join(
            line
            for line in legacy_source.splitlines()
            if not line.lstrip().startswith(repr(field))
        )

    parsed = parse_decision_source(legacy_source)
    assert parsed.long_horizon_audit_enabled is False
    assert parsed.long_horizon_penalty_weight == 0.0


def test_task_conditioned_long_horizon_grid_is_typed_and_bounded():
    candidates = selector_evolution.bounded_long_horizon_route_candidates(DecisionPolicy())

    assert candidates[0] == DecisionPolicy()
    assert len(candidates) == len(set(candidates))
    assert len(candidates) <= 200
    assert any(
        candidate.long_horizon_route_feature == "horizon_ratio"
        and candidate.long_horizon_route_operator == "at_least"
        and candidate.long_horizon_route_threshold == pytest.approx(0.25)
        for candidate in candidates
    )
    assert any(
        candidate.long_horizon_route_feature == "periodicity_strength"
        and candidate.long_horizon_route_operator == "at_most"
        and candidate.long_horizon_penalty_weight == 1.0
        for candidate in candidates
    )


def test_change_aware_baseline_guard_grid_contains_exactly_four_children():
    parent = DecisionPolicy()

    candidates = selector_evolution.bounded_baseline_guard_candidates(parent)

    assert candidates[0] == parent
    assert len(candidates) == 5
    assert len(candidates) == len(set(candidates))
    children = candidates[1:]
    assert {
        (candidate.baseline_strategy, candidate.long_horizon_max_regret)
        for candidate in children
    } == {
        ("toto_first", 0.0),
        ("toto_first", 0.02),
        ("minimax_tsfm", 0.0),
        ("minimax_tsfm", 0.02),
    }
    assert all(candidate.long_horizon_guard_enabled for candidate in children)
    assert all(candidate.long_horizon_min_coverage == pytest.approx(0.75) for candidate in children)


def test_conservative_tsfm_search_contains_only_two_fixed_soft_overlay_children():
    parent = DecisionPolicy()

    candidates = selector_evolution.bounded_conservative_tsfm_candidates(parent)

    assert candidates[0] == parent
    assert len(candidates) == 3
    assert len(candidates) == len(set(candidates))
    children = candidates[1:]
    assert {candidate.tsfm_router_min_improvement for candidate in children} == {0.02}
    assert {candidate.tsfm_router_blend_weight for candidate in children} == {0.1, 0.25}
    assert all(candidate.baseline_strategy == "conservative_tsfm" for candidate in children)
    assert all(
        candidate.ensemble_min_improvement == parent.ensemble_min_improvement
        for candidate in children
    )
    assert all(
        candidate.ensemble_max_worst_fold_regret
        == pytest.approx(parent.ensemble_max_worst_fold_regret)
        for candidate in children
    )
    assert all(not candidate.long_horizon_guard_enabled for candidate in children)
    assert all(candidate.long_horizon_min_coverage == pytest.approx(0.75) for candidate in children)
    assert all(candidate.long_horizon_max_regret == pytest.approx(0.0) for candidate in children)
    assert parse_decision_source(render_decision_source(children[0])) == children[0]


def test_conservative_combined_search_fixes_the_two_percent_fold_margin():
    parent = DecisionPolicy(tsfm_router_min_improvement=0.0)

    candidates = selector_evolution.bounded_conservative_combined_candidates(parent)

    assert candidates[0] == parent
    assert {candidate.tsfm_router_min_improvement for candidate in candidates[1:]} == {
        0.02
    }


def test_conservative_combined_search_does_not_rebase_a_non_toto_parent():
    parent = DecisionPolicy(baseline_strategy="minimax_tsfm")

    candidates = selector_evolution.bounded_conservative_combined_candidates(parent)

    assert candidates == (parent,)


def test_conservative_combined_search_does_not_rebase_a_guarded_parent():
    parent = DecisionPolicy(
        long_horizon_guard_enabled=True,
        long_horizon_min_coverage=0.5,
        long_horizon_max_regret=0.02,
    )

    candidates = selector_evolution.bounded_conservative_combined_candidates(parent)

    assert candidates == (parent,)


def test_joint_portfolio_search_exposes_four_ablation_children():
    parent = DecisionPolicy()

    candidates = bounded_joint_portfolio_candidates(parent)

    assert candidates[0] == parent
    assert len(candidates) == 5
    assert [candidate.baseline_strategy for candidate in candidates[1:]] == [
        "conservative_single_tsfm",
        "conservative_tsfm_portfolio",
        "conservative_tsfm_statistical",
        "conservative_joint_portfolio",
    ]
    assert all(not candidate.ensemble_enabled for candidate in candidates[1:])
    assert candidates[1].tsfm_router_blend_weight == pytest.approx(0.0)
    assert candidates[2].tsfm_router_blend_weight == pytest.approx(0.5)
    assert candidates[3].tsfm_router_blend_weight == pytest.approx(0.25)
    assert candidates[4].tsfm_router_blend_weight == pytest.approx(0.25)


def test_protected_portfolio_search_exposes_r1_r2_r3_without_changing_parent():
    parent = DecisionPolicy()

    candidates = selector_evolution.bounded_protected_portfolio_candidates(parent)

    assert candidates[0] == parent
    assert [candidate.baseline_strategy for candidate in candidates[1:]] == [
        "protected_single_tsfm",
        "protected_tsfm_portfolio",
        "protected_joint_residual",
    ]
    assert all(candidate.ensemble_enabled == parent.ensemble_enabled for candidate in candidates)
    assert candidates[-1].ensemble_residual_strengths == pytest.approx((0.05, 0.1, 0.2))


def test_protected_topk_search_preserves_learned_assumptions_in_every_child():
    parent = DecisionPolicy(
        baseline_strategy="minimax_tsfm",
        assumption_guidance_enabled=True,
        assumption_top_k=3,
        assumption_candidates_per_hypothesis=1,
        assumption_min_confidence=0.6,
    )

    candidates = selector_evolution.bounded_protected_topk_candidates(parent)

    assert candidates[0] == parent
    assert [candidate.baseline_strategy for candidate in candidates[1:]] == [
        "protected_topk_single_tsfm",
        "protected_topk_tsfm_portfolio",
        "protected_topk_joint_residual",
    ]
    assert all(candidate.assumption_guidance_enabled for candidate in candidates)
    assert all(candidate.assumption_top_k == 3 for candidate in candidates)
    assert all(
        candidate.assumption_candidates_per_hypothesis == 1
        for candidate in candidates
    )
    assert all(
        candidate.assumption_min_confidence == pytest.approx(0.6)
        for candidate in candidates
    )
    assert all(
        candidate.ensemble_residual_strengths == parent.ensemble_residual_strengths
        for candidate in candidates
    )
    assert all(
        candidate.ensemble_correction_clip == parent.ensemble_correction_clip
        for candidate in candidates
    )


def test_policy_parser_accepts_legacy_source_without_soft_overlay_weight():
    source = render_decision_source(DecisionPolicy()).replace(
        " 'tsfm_router_blend_weight': 0.0,\n",
        "",
    )

    assert parse_decision_source(source) == DecisionPolicy()


def test_minimax_baseline_strategy_round_trips_and_is_in_bounded_search():
    minimax = DecisionPolicy(baseline_strategy="minimax_tsfm")

    assert parse_decision_source(render_decision_source(minimax)) == minimax
    candidates = bounded_combined_candidates(
        DecisionPolicy(),
        DecisionPolicy(),
        available_hindcast_folds=3,
    )

    assert any(candidate.baseline_strategy == "minimax_tsfm" for candidate in candidates)


def test_assumption_policy_round_trips_and_bounded_search_enables_top_k():
    policy = DecisionPolicy(
        assumption_guidance_enabled=True,
        assumption_top_k=4,
        assumption_candidates_per_hypothesis=2,
        assumption_min_confidence=0.35,
    )

    assert parse_decision_source(render_decision_source(policy)) == policy
    candidates = bounded_combined_candidates(
        DecisionPolicy(),
        DecisionPolicy(),
        available_hindcast_folds=3,
    )

    assert any(candidate.assumption_guidance_enabled for candidate in candidates)
    assert {candidate.assumption_top_k for candidate in candidates if candidate.assumption_guidance_enabled} >= {3, 5}
    assert any(
        candidate.assumption_guidance_enabled
        and candidate.baseline_strategy == "minimax_tsfm"
        for candidate in candidates
    )


def test_legacy_policy_source_defaults_to_flat_selection_without_assumptions():
    policy = DecisionPolicy()
    source = render_decision_source(policy)
    for field in (
        "assumption_guidance_enabled",
        "assumption_top_k",
        "assumption_candidates_per_hypothesis",
        "assumption_min_confidence",
    ):
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(repr(field)))

    parsed = parse_decision_source(source)

    assert parsed.assumption_guidance_enabled is False
    assert parsed.assumption_top_k == 5


def test_pre_combined_policy_source_remains_backward_compatible():
    source = render_decision_source(DecisionPolicy())
    omitted = {
        "baseline_strategy",
        "assumption_guidance_enabled",
        "assumption_top_k",
        "assumption_candidates_per_hypothesis",
        "assumption_min_confidence",
        "ensemble_weight_grid",
        "ensemble_residual_strengths",
        "ensemble_correction_clip",
        "ensemble_min_fold_wins",
        "ensemble_max_worst_fold_regret",
    }
    source = "\n".join(
        line
        for line in source.splitlines()
        if not any(line.lstrip().startswith(repr(field)) for field in omitted)
    )

    parsed = parse_decision_source(source)

    assert parsed == DecisionPolicy()


def test_evaluator_scores_final_forecasts_and_gate_rejects_mean_regression():
    cases = (_case("t1"), _case("t2"))
    parent = evaluate_decision(DecisionPolicy(ensemble_enabled=False), cases)
    assert parent.coverage == 1.0
    assert parent.mean_mase == pytest.approx(0.0)
    assert parent.catastrophic_rate == 0.0

    regressed = parent.__class__(
        task_count=2,
        coverage=1.0,
        mean_mase=1.0,
        median_mase=1.0,
        mean_mae=1.0,
        median_mae=1.0,
        mean_smape=1.0,
        catastrophic_rate=0.0,
        mean_active_oracle_regret=1.0,
        method_diversity=1,
        family_diversity=1,
        ensemble_rate=0.0,
        fallback_rate=0.0,
    )
    gate = compare_decisions(parent, parent, parent, regressed)
    assert not gate.accepted


def test_gate_uses_clipped_smae_as_primary_and_guards_srmse_and_tail():
    observed = evaluate_decision(DecisionPolicy(ensemble_enabled=False), (_case("t1"),))
    parent = replace(
        observed,
        mean_smae=1.0,
        mean_srmse=1.0,
        smae_clipped_count=0,
        srmse_clipped_count=0,
        p90_smae=1.0,
        p95_smae=1.0,
    )
    improved = replace(
        parent,
        mean_smae=0.9,
        mean_mase=parent.mean_mase + 10.0,
    )

    assert compare_decisions(parent, improved, parent, improved).accepted

    worse_srmse = replace(improved, mean_srmse=1.01)
    assert not compare_decisions(parent, improved, parent, worse_srmse).accepted

    worse_tail = replace(improved, p95_smae=1.02)
    assert not compare_decisions(parent, improved, parent, worse_tail).accepted

    more_clipped = replace(improved, smae_clipped_count=1)
    assert not compare_decisions(parent, improved, parent, more_clipped).accepted

    missing = replace(improved, coverage=0.99)
    assert not compare_decisions(parent, improved, parent, missing).accepted


def test_combined_child_is_rejected_when_train_srmse_regresses() -> None:
    """A Train sMAE gain must not hide regression in the paired scaled metric."""
    observed = evaluate_decision(DecisionPolicy(ensemble_enabled=False), (_case("t1"),))
    parent = replace(
        observed,
        mean_smae=1.0,
        mean_srmse=1.0,
        p90_smae=1.0,
        p95_smae=1.0,
    )
    train_child = replace(parent, mean_smae=0.8, mean_srmse=1.001)
    dev_child = replace(parent, mean_smae=0.9, mean_srmse=0.9)

    assert not compare_decisions(parent, train_child, parent, dev_child).accepted


def test_combined_child_accepts_pareto_gain_in_srmse_with_smae_unchanged() -> None:
    """Requiring an sMAE win would incorrectly reject a valid two-metric Pareto gain."""
    observed = evaluate_decision(DecisionPolicy(ensemble_enabled=False), (_case("t1"),))
    parent = replace(
        observed,
        mean_smae=1.0,
        mean_srmse=1.0,
        p90_smae=1.0,
        p95_smae=1.0,
    )
    child = replace(parent, mean_srmse=0.9)

    assert compare_decisions(parent, child, parent, child).accepted


def test_active_oracle_regret_uses_the_same_scaled_smae_contract():
    task = Task("t", tuple(float(i) for i in range(1, 21)), 2, "D", (1.0, 2.0))
    case = DecisionCase(
        task,
        ("selected", "oracle"),
        {"selected": _diag("selected", 0.1), "oracle": _diag("oracle", 0.2)},
        {"selected": (4.0, 5.0), "oracle": (1.0, 2.0)},
        {"selected": "statistical", "oracle": "statistical"},
    )

    score = evaluate_decision(DecisionPolicy(ensemble_enabled=False), (case,))

    # Selected MAE is 3 and the mean absolute target scale is 1.5, so sMAE=2.
    # The active oracle is perfect, giving regret (2 - 0) / (1 + 0) = 2.
    assert score.mean_active_oracle_regret == pytest.approx(2.0)


def test_decision_score_observes_assumption_breadth_before_final_selection():
    truth = ((0.0, 0.0),) * 3
    task = Task(
        "periodic",
        tuple(float(index % 7) for index in range(70)),
        2,
        "D",
        (0.0, 0.0),
    )
    diagnostics = {
        "toto_2_0": CandidateDiagnostics.synthetic(
            name="toto_2_0", family="tsfm", median_mase=1.0,
            fold_forecasts=((1.0, 1.0),) * 3, fold_truths=truth,
        ),
        "seasonal_naive": CandidateDiagnostics.synthetic(
            name="seasonal_naive", family="statistical", median_mase=0.0,
            fold_forecasts=truth, fold_truths=truth,
        ),
        "combined_timesfm_seasonal": CandidateDiagnostics.synthetic(
            name="combined_timesfm_seasonal", family="combined", median_mase=0.2,
            fold_forecasts=((0.2, 0.2),) * 3, fold_truths=truth,
        ),
    }
    case = DecisionCase(
        task,
        tuple(diagnostics),
        diagnostics,
        {name: (0.0, 0.0) for name in diagnostics},
        {
            "toto_2_0": "tsfm",
            "seasonal_naive": "statistical",
            "combined_timesfm_seasonal": "combined",
        },
    )

    score = evaluate_decision(
        DecisionPolicy(
            assumption_guidance_enabled=True,
            assumption_top_k=3,
            assumption_candidates_per_hypothesis=2,
            assumption_min_confidence=0.2,
            ensemble_enabled=False,
        ),
        (case,),
    )

    assert score.mean_assumption_count >= 2.0
    assert score.mean_considered_candidates == pytest.approx(3.0)
    assert score.mean_considered_families == pytest.approx(3.0)
    assert score.assumption_kind_diversity >= 2


def test_evolution_prompt_contains_train_aggregates_not_ids_or_futures(tmp_path):
    parent = DecisionPolicy(ensemble_enabled=False)
    agent = FakeAgent({
        "summary": "no unsafe change",
        "policy": {
            "ranking_order": list(parent.ranking_order),
            "recent_regime_first": parent.recent_regime_first,
            "min_successful_folds": parent.min_successful_folds,
            "catastrophic_mase": parent.catastrophic_mase,
            "ensemble_enabled": parent.ensemble_enabled,
            "ensemble_max_members": parent.ensemble_max_members,
            "ensemble_min_diversity": parent.ensemble_min_diversity,
            "ensemble_min_improvement": parent.ensemble_min_improvement,
            "ensemble_weight_grid": list(parent.ensemble_weight_grid),
            "ensemble_residual_strengths": list(parent.ensemble_residual_strengths),
            "ensemble_correction_clip": parent.ensemble_correction_clip,
            "ensemble_min_fold_wins": parent.ensemble_min_fold_wins,
            "ensemble_max_worst_fold_regret": parent.ensemble_max_worst_fold_regret,
            "fallback_to_best_available": parent.fallback_to_best_available,
        },
    })
    result = evolve_selector_once(
        parent,
        (_case("train-secret", good="a"),),
        (_case("dev-secret", good="b"),),
        agent,
        generation=1,
        screening_policy_hash="screen-sha",
        transcript_dir=tmp_path,
    )
    prompt = agent.requests[0]["messages"][0]["content"]
    assert "train-secret" not in prompt
    assert "dev-secret" not in prompt
    assert "future" not in prompt.lower()
    assert "mase" not in prompt.lower()
    assert "smape" not in prompt.lower()
    assert "screen-sha" in prompt
    assert not result.accepted


def test_evolution_rejects_child_requiring_more_folds_than_exist(tmp_path):
    parent = DecisionPolicy(ensemble_enabled=True, min_successful_folds=3)
    payload = {
        "ranking_order": list(parent.ranking_order),
        "recent_regime_first": parent.recent_regime_first,
        "min_successful_folds": 4,
        "catastrophic_mase": parent.catastrophic_mase,
        "ensemble_enabled": parent.ensemble_enabled,
        "ensemble_max_members": parent.ensemble_max_members,
        "ensemble_min_diversity": parent.ensemble_min_diversity,
        "ensemble_min_improvement": parent.ensemble_min_improvement,
        "ensemble_weight_grid": list(parent.ensemble_weight_grid),
        "ensemble_residual_strengths": list(parent.ensemble_residual_strengths),
        "ensemble_correction_clip": parent.ensemble_correction_clip,
        "ensemble_min_fold_wins": parent.ensemble_min_fold_wins,
        "ensemble_max_worst_fold_regret": parent.ensemble_max_worst_fold_regret,
        "fallback_to_best_available": parent.fallback_to_best_available,
    }
    agent = FakeAgent({"summary": "require unavailable evidence", "policy": payload})

    result = evolve_selector_once(
        parent,
        (_case("train-secret"),),
        (_case("dev-secret"),),
        agent,
        generation=1,
        screening_policy_hash="screen-sha",
        transcript_dir=tmp_path,
        available_hindcast_folds=3,
    )

    assert result.child == parent
    assert not result.accepted
    assert "available hindcast folds" in result.gate.reason
    request = json.loads(agent.requests[0]["messages"][0]["content"])
    assert request["available_hindcast_folds"] == 3


def test_evolution_passes_prior_rejections_without_task_labels(tmp_path):
    parent = DecisionPolicy(ensemble_enabled=False)
    payload = {
        "ranking_order": list(parent.ranking_order),
        "recent_regime_first": parent.recent_regime_first,
        "min_successful_folds": parent.min_successful_folds,
        "catastrophic_mase": parent.catastrophic_mase,
        "ensemble_enabled": parent.ensemble_enabled,
        "ensemble_max_members": parent.ensemble_max_members,
        "ensemble_min_diversity": parent.ensemble_min_diversity,
        "ensemble_min_improvement": parent.ensemble_min_improvement,
        "ensemble_weight_grid": list(parent.ensemble_weight_grid),
        "ensemble_residual_strengths": list(parent.ensemble_residual_strengths),
        "ensemble_correction_clip": parent.ensemble_correction_clip,
        "ensemble_min_fold_wins": parent.ensemble_min_fold_wins,
        "ensemble_max_worst_fold_regret": parent.ensemble_max_worst_fold_regret,
        "fallback_to_best_available": parent.fallback_to_best_available,
    }
    agent = FakeAgent({"summary": "try a different safe direction", "policy": payload})

    evolve_selector_once(
        parent,
        (_case("train-secret"),),
        (_case("dev-secret"),),
        agent,
        generation=2,
        screening_policy_hash="screen-sha",
        transcript_dir=tmp_path,
        available_hindcast_folds=3,
        prior_rejections=("Generation 1: Train sMAE increased",),
    )

    request = json.loads(agent.requests[0]["messages"][0]["content"])
    assert request["prior_rejections"] == ["Generation 1: Train sMAE increased"]
    assert "train-secret" not in json.dumps(request)
    assert "dev-secret" not in json.dumps(request)


def test_generation_sequence_feeds_rejection_reason_to_next_child(tmp_path):
    parent = DecisionPolicy(ensemble_enabled=False, min_successful_folds=3)

    def response(minimum_folds):
        return {
            "summary": "bounded proposal",
            "policy": {
                "ranking_order": list(parent.ranking_order),
                "recent_regime_first": parent.recent_regime_first,
                "min_successful_folds": minimum_folds,
                "catastrophic_mase": parent.catastrophic_mase,
                "ensemble_enabled": parent.ensemble_enabled,
                "ensemble_max_members": parent.ensemble_max_members,
                "ensemble_min_diversity": parent.ensemble_min_diversity,
                "ensemble_min_improvement": parent.ensemble_min_improvement,
                "ensemble_weight_grid": list(parent.ensemble_weight_grid),
                "ensemble_residual_strengths": list(parent.ensemble_residual_strengths),
                "ensemble_correction_clip": parent.ensemble_correction_clip,
                "ensemble_min_fold_wins": parent.ensemble_min_fold_wins,
                "ensemble_max_worst_fold_regret": parent.ensemble_max_worst_fold_regret,
                "fallback_to_best_available": parent.fallback_to_best_available,
            },
        }

    agent = SequenceAgent((response(4), response(3)))
    frozen, generations = evolve_selector_generations(
        parent,
        (_case("train-secret"),),
        (_case("dev-secret"),),
        agent,
        generations=2,
        available_hindcast_folds=3,
        screening_policy_hash="screen-sha",
        transcript_dir=tmp_path,
    )

    assert frozen == parent
    assert len(generations) == 2
    second_request = json.loads(agent.requests[1]["messages"][0]["content"])
    assert second_request["prior_rejections"] == [
        "Generation 1: Invalid child proposal: min_successful_folds exceeds "
        "available hindcast folds (4 > 3)"
    ]


def test_bounded_combined_candidates_cover_operators_without_exceeding_fold_budget():
    parent = DecisionPolicy(min_successful_folds=3)
    proposal = replace(
        parent,
        ranking_order=("recent_mase", "median_mase", "worst_mase", "mase_mad"),
    )

    candidates = bounded_combined_candidates(
        parent,
        proposal,
        available_hindcast_folds=3,
    )

    assert proposal in candidates
    assert all(candidate.min_successful_folds <= 3 for candidate in candidates)
    operator_modes = {
        (
            bool(candidate.ensemble_weight_grid),
            bool(candidate.ensemble_residual_strengths),
        )
        for candidate in candidates
    }
    assert operator_modes >= {(True, False), (False, True), (True, True)}
    assert len(candidates) == len(set(candidates))
    assert len(candidates) <= 64


def test_train_evolution_uses_dev_only_for_one_final_read_only_gate(tmp_path):
    parent = DecisionPolicy(
        ranking_order=("median_mase", "recent_mase", "worst_mase", "mase_mad"),
        recent_regime_first=False,
        ensemble_enabled=False,
    )
    child = replace(
        parent,
        ranking_order=("recent_mase", "median_mase", "worst_mase", "mase_mad"),
    )

    def response(policy):
        return {
            "summary": "prefer the recent-fold winner",
            "policy": {
                "ranking_order": list(policy.ranking_order),
                "recent_regime_first": policy.recent_regime_first,
                "min_successful_folds": policy.min_successful_folds,
                "catastrophic_mase": policy.catastrophic_mase,
                "ensemble_enabled": policy.ensemble_enabled,
                "ensemble_max_members": policy.ensemble_max_members,
                "ensemble_min_diversity": policy.ensemble_min_diversity,
                "ensemble_min_improvement": policy.ensemble_min_improvement,
                "ensemble_weight_grid": list(policy.ensemble_weight_grid),
                "ensemble_residual_strengths": list(policy.ensemble_residual_strengths),
                "ensemble_correction_clip": policy.ensemble_correction_clip,
                "ensemble_min_fold_wins": policy.ensemble_min_fold_wins,
                "ensemble_max_worst_fold_regret": policy.ensemble_max_worst_fold_regret,
                "fallback_to_best_available": policy.fallback_to_best_available,
            },
        }

    agent = SequenceAgent((response(child), response(child)))
    result = evolve_selector_train_then_dev(
        parent,
        (_ranking_sensitive_case("train-secret", child_is_better=True),),
        (_ranking_sensitive_case("dev-secret", child_is_better=False),),
        agent,
        generations=2,
        available_hindcast_folds=3,
        screening_policy_hash="screen-sha",
        transcript_dir=tmp_path,
    )

    assert result.generations[0].accepted
    second_request = json.loads(agent.requests[1]["messages"][0]["content"])
    assert second_request["current_policy"]["ranking_order"][0] == "recent_joint_scaled_error"
    assert "dev-secret" not in json.dumps(agent.requests)
    assert result.train_winner.ranking_order[0] == "recent_joint_scaled_error"
    assert not result.final_gate.accepted
    assert result.frozen == parent


def test_train_crossfold_gate_rejects_average_gain_with_one_unstable_entity_group():
    """Removing the cross-fold gate would accept a policy with one severe group regression."""
    parent = DecisionPolicy(
        ranking_order=("median_mase", "recent_mase", "worst_mase", "mase_mad"),
        recent_regime_first=False,
        ensemble_enabled=False,
    )
    child = replace(
        parent,
        ranking_order=("recent_mase", "median_mase", "worst_mase", "mase_mad"),
    )
    cases = tuple(
        replace(
            _ranking_sensitive_case(
                f"train-{index}",
                child_is_better=index < 3,
            ),
            group_id=f"entity-{index}",
        )
        for index in range(4)
    )

    assert evaluate_decision(child, cases).mean_smae < evaluate_decision(parent, cases).mean_smae
    gate = compare_train_crossfolds(parent, child, cases, folds=4)

    assert not gate.accepted
    assert "fold" in gate.reason.lower()


def test_train_crossfold_gate_accepts_policy_improving_every_entity_group():
    parent = DecisionPolicy(
        ranking_order=("median_mase", "recent_mase", "worst_mase", "mase_mad"),
        recent_regime_first=False,
        ensemble_enabled=False,
    )
    child = replace(
        parent,
        ranking_order=("recent_mase", "median_mase", "worst_mase", "mase_mad"),
    )
    cases = tuple(
        replace(
            _ranking_sensitive_case(f"train-{index}", child_is_better=True),
            group_id=f"entity-{index}",
        )
        for index in range(4)
    )

    gate = compare_train_crossfolds(parent, child, cases, folds=4)

    assert gate.accepted


def test_activation_aware_gate_accepts_two_improved_and_two_unchanged_folds():
    base = replace(
        evaluate_decision(DecisionPolicy(ensemble_enabled=False), (_case("base"),)),
        mean_smae=1.0,
        mean_srmse=1.0,
        p90_smae=1.0,
        p95_smae=1.0,
        mean_active_oracle_regret=0.1,
    )
    improved = replace(
        base,
        mean_smae=0.9,
        mean_srmse=0.95,
        p90_smae=0.95,
        p95_smae=0.95,
        mean_active_oracle_regret=0.09,
    )

    gate = selector_evolution.compare_activation_aware_fold_scores(
        ((base, improved), (base, improved), (base, base), (base, base)),
        matched_counts=(3, 3, 0, 0),
        total_matched=6,
        total_tasks=20,
    )

    assert gate.accepted
    assert "2/4" in gate.reason


def test_activation_aware_gate_rejects_any_material_fold_regression():
    base = replace(
        evaluate_decision(DecisionPolicy(ensemble_enabled=False), (_case("base"),)),
        mean_smae=1.0,
        mean_srmse=1.0,
        p90_smae=1.0,
        p95_smae=1.0,
        mean_active_oracle_regret=0.1,
    )
    improved = replace(base, mean_smae=0.9, mean_srmse=0.95)
    regressed = replace(base, mean_smae=1.02)

    gate = selector_evolution.compare_activation_aware_fold_scores(
        ((base, improved), (base, improved), (base, regressed), (base, base)),
        matched_counts=(3, 3, 2, 0),
        total_matched=8,
        total_tasks=20,
    )

    assert not gate.accepted
    assert "sMAE" in gate.reason


def test_change_aware_crossfold_gate_counts_only_changed_final_forecasts():
    parent = DecisionPolicy(
        ranking_order=("median_mase", "recent_mase", "worst_mase", "mase_mad"),
        recent_regime_first=False,
        ensemble_enabled=False,
    )
    child = replace(
        parent,
        ranking_order=("recent_mase", "median_mase", "worst_mase", "mase_mad"),
    )
    cases = (
        replace(_ranking_sensitive_case("changed-1", child_is_better=True), group_id="e1"),
        replace(_ranking_sensitive_case("changed-2", child_is_better=True), group_id="e2"),
        replace(_case("unchanged-1"), group_id="e3"),
        replace(_case("unchanged-2"), group_id="e4"),
    )

    gate = selector_evolution.compare_change_aware_crossfolds(
        parent,
        child,
        cases,
        folds=4,
    )

    assert gate.accepted
    assert "2/4" in gate.reason


@pytest.mark.parametrize("matched", (1, 17))
def test_activation_aware_gate_rejects_trivial_route_coverage(matched):
    base = replace(
        evaluate_decision(DecisionPolicy(ensemble_enabled=False), (_case("base"),)),
        mean_smae=1.0,
        mean_srmse=1.0,
        p90_smae=1.0,
        p95_smae=1.0,
    )
    improved = replace(base, mean_smae=0.9, mean_srmse=0.95)

    gate = selector_evolution.compare_activation_aware_fold_scores(
        ((base, improved), (base, improved), (base, base), (base, base)),
        matched_counts=(1, 1, 0, 0),
        total_matched=matched,
        total_tasks=20,
    )

    assert not gate.accepted
    assert "coverage" in gate.reason.lower()


def test_activation_aware_gate_allows_two_fold_abstaining_overlay_support():
    base = replace(
        evaluate_decision(DecisionPolicy(ensemble_enabled=False), (_case("base"),)),
        mean_smae=1.0,
        mean_srmse=1.0,
        p90_smae=1.0,
        p95_smae=1.0,
        mean_active_oracle_regret=0.1,
    )
    improved = replace(
        base,
        mean_smae=0.99,
        mean_srmse=0.999,
        mean_active_oracle_regret=0.09,
    )

    gate = selector_evolution.compare_activation_aware_fold_scores(
        ((base, base), (base, improved), (base, improved), (base, base)),
        matched_counts=(0, 1, 1, 0),
        total_matched=2,
        total_tasks=80,
        minimum_matches=2,
    )

    assert gate.accepted
    assert "2/4" in gate.reason


def test_train_only_evolution_rejects_child_that_fails_crossfold_stability(tmp_path):
    parent = DecisionPolicy(
        ranking_order=("median_mase", "recent_mase", "worst_mase", "mase_mad"),
        recent_regime_first=False,
        ensemble_enabled=False,
    )
    child = replace(
        parent,
        ranking_order=("recent_mase", "median_mase", "worst_mase", "mase_mad"),
    )
    payload = {
        "summary": "prefer recent evidence",
        "policy": {
            "ranking_order": list(child.ranking_order),
            "recent_regime_first": child.recent_regime_first,
            "min_successful_folds": child.min_successful_folds,
            "catastrophic_mase": child.catastrophic_mase,
            "ensemble_enabled": child.ensemble_enabled,
            "ensemble_max_members": child.ensemble_max_members,
            "ensemble_min_diversity": child.ensemble_min_diversity,
            "ensemble_min_improvement": child.ensemble_min_improvement,
            "ensemble_weight_grid": list(child.ensemble_weight_grid),
            "ensemble_residual_strengths": list(child.ensemble_residual_strengths),
            "ensemble_correction_clip": child.ensemble_correction_clip,
            "ensemble_min_fold_wins": child.ensemble_min_fold_wins,
            "ensemble_max_worst_fold_regret": child.ensemble_max_worst_fold_regret,
            "fallback_to_best_available": child.fallback_to_best_available,
        },
    }
    train = tuple(
        replace(
            _ranking_sensitive_case(f"train-{index}", child_is_better=index < 3),
            group_id=f"entity-{index}",
        )
        for index in range(4)
    )

    result = evolve_selector_train_then_dev(
        parent,
        train,
        (_ranking_sensitive_case("dev", child_is_better=True),),
        FakeAgent(payload),
        generations=1,
        available_hindcast_folds=3,
        train_validation_folds=4,
        screening_policy_hash="screen-sha",
        transcript_dir=tmp_path,
    )

    assert not result.generations[0].accepted
    assert "fold" in result.generations[0].gate.reason.lower()
    assert result.train_winner == parent
