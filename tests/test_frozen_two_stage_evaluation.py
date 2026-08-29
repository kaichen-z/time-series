from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import numerical_agent.evaluate_frozen_two_stage as frozen_evaluation
from numerical_agent.evaluate_frozen_two_stage import (
    ForecastResult,
    _hindcast_config_for_policy,
    _paired_counts,
    _report,
    _selector_forecast,
    build_parser,
    score_forecast_results,
    verify_frozen_policies,
)
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.module import MODULE_HEADER, parse_module
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics, DecisionPolicy
from numerical_agent.evolution.portfolio import CombinedPolicy, PolicyPortfolio
from numerical_agent.evolution.selector_evolution import DecisionCase


ROOT = Path(__file__).resolve().parents[1]


class _StopAfterRankingValidation(Exception):
    pass


def _ranking_module():
    names = (
        "seasonal_naive",
        "holt_damped_trend",
        "croston_sba",
        "robust_loess_trend",
        "median_seasonal_profile_forecast",
        *(f"statistical_{index}" for index in range(88)),
    )
    source = "\n\n".join(
        f'''def {name}(history, horizon, frequency):
    """Frozen evaluator namespace fixture."""
    return [0.0] * horizon
'''
        for name in names
    )
    return parse_module(MODULE_HEADER + "\n\n" + source)


def _ranking_portfolio(combined_count: int) -> PolicyPortfolio:
    portfolio = PolicyPortfolio.flagship5()
    while len(portfolio.combined) < combined_count:
        index = len(portfolio.combined)
        portfolio = portfolio.add_combined(CombinedPolicy(
            f"combined_extra_{index}",
            ("toto_2_0", "seasonal_naive"),
            "median",
            fallback_parent="toto_2_0",
        ))
    return portfolio


def _run_until_ranking_validation(
    tmp_path, monkeypatch, *, combined_count: int, ranking_mutation: str | None = None
):
    repo = tmp_path / "repo"
    screen = tmp_path / "screen"
    selector = tmp_path / "selector"
    output = tmp_path / "output"
    repo.mkdir(); screen.mkdir(); selector.mkdir()
    (screen / "frozen_screening_policy.py").write_text("screen", encoding="utf-8")
    (selector / "frozen_decision_policy.py").write_text("decision", encoding="utf-8")
    module = _ranking_module()
    portfolio = _ranking_portfolio(combined_count)
    ranking = [*module.names(), *portfolio.names]
    if ranking_mutation == "missing":
        ranking.pop()
    elif ranking_mutation == "duplicate":
        ranking[-1] = ranking[0]
    elif ranking_mutation == "extra":
        ranking.append("unexpected_candidate")
    (selector / "selector_manifest.json").write_text(
        json.dumps({"frozen_global_ranking": ranking}), encoding="utf-8"
    )
    task = Task("public", (1.0, 2.0), 1, "D", (3.0,))
    monkeypatch.setattr(
        frozen_evaluation, "verify_frozen_policies", lambda *args: ("screen", "decision")
    )
    monkeypatch.setattr(
        frozen_evaluation, "_public_test_tasks", lambda *args: (task,) * 99
    )
    monkeypatch.setattr(frozen_evaluation, "read_module", lambda *args: module)
    monkeypatch.setattr(frozen_evaluation, "read_policy_file", lambda *args: portfolio)
    monkeypatch.setattr(frozen_evaluation, "parse_screening_source", lambda *args: object())
    monkeypatch.setattr(frozen_evaluation, "build_filter_dictionary", lambda *args: object())
    monkeypatch.setattr(
        frozen_evaluation, "migrate_filter_dictionary", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        frozen_evaluation, "parse_decision_source", lambda *args: DecisionPolicy()
    )
    monkeypatch.setattr(
        frozen_evaluation,
        "_training_outcomes",
        lambda *args: (_ for _ in ()).throw(_StopAfterRankingValidation()),
    )
    argv = [
        "--repo", str(repo),
        "--screening-dir", str(screen),
        "--selector-dir", str(selector),
        "--tasks-file", "unused-tasks",
        "--outcome-cache-dir", str(tmp_path / "outcome-cache"),
        "--policy-outcome-cache-dir", str(tmp_path / "policy-cache"),
        "--hindcast-cache-dir", str(tmp_path / "hindcast-cache"),
        "--output-dir", str(output),
    ]
    return argv, len(module.names()) + len(portfolio.names)


def test_frozen_evaluation_runner_prefers_project_virtualenv():
    source = (ROOT / "scripts" / "evaluate_frozen_two_stage.sh").read_text(
        encoding="utf-8"
    )
    assert 'PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"' in source
    assert '"$PYTHON_BIN" -m numerical_agent.evaluate_frozen_two_stage' in source


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def test_freeze_verifier_rejects_hash_mismatch_and_existing_completion(tmp_path):
    screen = tmp_path / "screen"
    selector = tmp_path / "selector"
    output = tmp_path / "out"
    screen.mkdir(); selector.mkdir(); output.mkdir()
    screen_hash = _write(screen / "frozen_screening_policy.py", "screen")
    decision_hash = _write(selector / "frozen_decision_policy.py", "decision")
    (screen / "screening_manifest.json").write_text(json.dumps({
        "frozen_screening_policy_sha256": screen_hash,
        "public_test_accessed": False,
    }))
    (selector / "selector_manifest.json").write_text(json.dumps({
        "screening_policy_sha256": screen_hash,
        "frozen_decision_policy_sha256": decision_hash,
        "public_test_accessed": False,
    }))
    assert verify_frozen_policies(screen, selector, output) == (screen_hash, decision_hash)
    (selector / "frozen_decision_policy.py").write_text("changed")
    with pytest.raises(ValueError, match="decision"):
        verify_frozen_policies(screen, selector, output)
    (selector / "frozen_decision_policy.py").write_text("decision")
    (output / "evaluation_complete.json").write_text("{}")
    with pytest.raises(ValueError, match="already"):
        verify_frozen_policies(screen, selector, output)


def test_evaluation_cli_has_no_llm_or_mutation_options():
    parser = build_parser()
    options = {action.dest for action in parser._actions}
    assert "codex_model" not in options
    assert "generations" not in options
    assert "llm_backend" not in options


@pytest.mark.parametrize("combined_count, expected_count", ((5, 103), (6, 104)))
def test_frozen_evaluator_accepts_exact_runtime_ranking_namespace(
    tmp_path, monkeypatch, combined_count, expected_count
):
    argv, candidate_count = _run_until_ranking_validation(
        tmp_path, monkeypatch, combined_count=combined_count
    )

    with pytest.raises(_StopAfterRankingValidation):
        frozen_evaluation.main(argv)

    assert candidate_count == expected_count


@pytest.mark.parametrize("ranking_mutation", ("missing", "duplicate", "extra"))
def test_frozen_evaluator_rejects_ranking_namespace_before_forecasting(
    tmp_path, monkeypatch, ranking_mutation
):
    argv, _ = _run_until_ranking_validation(
        tmp_path,
        monkeypatch,
        combined_count=5,
        ranking_mutation=ranking_mutation,
    )

    with pytest.raises(ValueError, match="selector ranking namespace mismatch"):
        frozen_evaluation.main(argv)


def test_frozen_evaluator_builds_long_audit_for_change_aware_guard():
    parent = _hindcast_config_for_policy(DecisionPolicy())
    guarded = _hindcast_config_for_policy(
        DecisionPolicy(long_horizon_guard_enabled=True)
    )
    routed = _hindcast_config_for_policy(
        DecisionPolicy(
            baseline_strategy="conservative_tsfm",
            tsfm_router_min_improvement=0.02,
        )
    )
    portfolio = _hindcast_config_for_policy(
        DecisionPolicy(baseline_strategy="conservative_joint_portfolio")
    )

    assert parent.long_horizon_audit is False
    assert guarded.long_horizon_audit is True
    assert routed.long_horizon_audit is True
    assert portfolio.long_horizon_audit is True
    assert guarded.folds == parent.folds == 3


def test_score_reports_mean_median_rmsse_and_diversity():
    task = Task("t", (1.0, 2.0, 3.0), 2, "D", (4.0, 5.0))
    perfect = ForecastResult("t", (4.0, 5.0), ("a",), ("statistical",), "single")
    score = score_forecast_results((task,), (perfect,))
    assert score["coverage"] == 1.0
    assert score["mean_mase"] == 0.0
    assert score["median_mase"] == 0.0
    assert score["mean_rmsse"] == 0.0
    assert score["method_diversity"] == 1
    assert score["family_diversity"] == 1


def test_score_reports_drcik_aligned_point_metrics_standard_errors_and_tails():
    tasks = (
        Task("easy", (1.0, 2.0, 3.0), 2, "D", (2.0, 2.0)),
        Task("clipped", (1.0, 2.0, 3.0), 2, "D", (1.0, 1.0)),
    )
    results = (
        ForecastResult("easy", (3.0, 3.0), ("a",), ("statistical",), "single"),
        ForecastResult("clipped", (11.0, 11.0), ("b",), ("tsfm",), "single"),
    )

    score = score_forecast_results(tasks, results)

    assert score["mean_smae"] == pytest.approx(2.75)
    assert score["se_smae"] == pytest.approx(2.25)
    assert score["mean_srmse"] == pytest.approx(2.75)
    assert score["se_srmse"] == pytest.approx(2.25)
    assert score["p90_smae"] == pytest.approx(4.55)
    assert score["p95_smae"] == pytest.approx(4.775)
    assert score["smae_clipped_count"] == 1
    assert score["smae_clipped_rate"] == pytest.approx(0.5)
    assert score["srmse_clipped_count"] == 1
    assert score["srmse_clipped_rate"] == pytest.approx(0.5)
    clipped = next(row for row in score["per_task"] if row["task_id"] == "clipped")
    assert clipped["smae_raw"] == pytest.approx(10.0)
    assert clipped["smae"] == pytest.approx(5.0)
    assert clipped["smae_clipped"] is True


def test_frozen_report_leads_with_drcik_point_metrics():
    task = Task("t", (1.0, 2.0, 3.0), 1, "D", (2.0,))
    result = ForecastResult("t", (3.0,), ("a",), ("statistical",), "single")
    score = score_forecast_results((task,), (result,))
    report = _report({
        "task_count": 1,
        "screening_policy_sha256": "screen",
        "decision_policy_sha256": "decision",
        "llm_calls": 0,
        "mutation_calls": 0,
        "rows": {"candidate": score},
        "paired_vs_A": {"candidate": {"wins": 0, "ties": 1, "losses": 0}},
    })

    assert "Mean sMAE" in report
    assert "sMAE SE" in report
    assert "Mean sRMSE" in report
    assert "P90/P95 sMAE" in report
    assert "Clipped sMAE/sRMSE" in report


def test_paired_counts_compare_the_reported_smae_metric():
    comparison = _paired_counts(
        ({"task_id": "t", "mase": 0.1, "smae": 2.0},),
        {"t": 1.0},
    )

    assert comparison == {"wins": 0, "ties": 0, "losses": 1, "missing": 0}


def test_frozen_selector_records_history_only_top_k_assumptions():
    truth_folds = ((0.0, 0.0),) * 3
    diagnostics = {
        "toto_2_0": CandidateDiagnostics.synthetic(
            name="toto_2_0", family="tsfm", median_mase=2.0,
            worst_mase=2.0, recent_mase=2.0,
            fold_forecasts=((2.0, 2.0),) * 3, fold_truths=truth_folds,
        ),
        "seasonal_naive": CandidateDiagnostics.synthetic(
            name="seasonal_naive", family="statistical", median_mase=0.0,
            worst_mase=0.0, recent_mase=0.0,
            fold_forecasts=truth_folds, fold_truths=truth_folds,
        ),
    }
    case = DecisionCase(
        Task(
            "periodic",
            tuple(float(index % 7) for index in range(70)),
            2,
            "D",
            (0.0, 0.0),
        ),
        ("toto_2_0", "seasonal_naive"),
        diagnostics,
        {"toto_2_0": (2.0, 2.0), "seasonal_naive": (0.0, 0.0)},
        {"toto_2_0": "tsfm", "seasonal_naive": "statistical"},
    )
    policy = DecisionPolicy(
        assumption_guidance_enabled=True,
        assumption_top_k=3,
        assumption_candidates_per_hypothesis=1,
        assumption_min_confidence=0.2,
        ensemble_enabled=False,
        ensemble_min_improvement=0.0,
        ensemble_min_fold_wins=3,
        ensemble_max_worst_fold_regret=0.0,
    )

    result = _selector_forecast(case, policy)

    assert result.assumption_ids
    assert any(value.startswith("periodic_persistence") for value in result.assumption_ids)
