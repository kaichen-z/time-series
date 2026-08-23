from __future__ import annotations

import json

import pytest

from common.llm import LLMResponse
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics, DecisionPolicy
from numerical_agent.evolution.selector_evolution import (
    DecisionCase,
    SelectorEvolutionError,
    apply_decision_response,
    compare_decisions,
    evaluate_decision,
    evolve_selector_once,
    parse_decision_source,
    render_decision_source,
)


class FakeAgent:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def complete(self, **kwargs):
        self.requests.append(kwargs)
        return LLMResponse(text=json.dumps(self.payload))


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
        },
    }))
    assert child.recent_regime_first
    assert child.catastrophic_mase == 8.0

    for forbidden in ("future", "split", "candidates", "scorer", "screening_policy"):
        with pytest.raises(SelectorEvolutionError):
            apply_decision_response(parent, json.dumps({"summary": "bad", "policy": {forbidden: 1}}))


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
    assert "screen-sha" in prompt
    assert not result.accepted
