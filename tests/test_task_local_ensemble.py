"""History-only runtime contracts for the task-local Numerical tournament."""

from __future__ import annotations

import math

import pytest

from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.task_local_ensemble import (
    TaskLocalTournamentPolicy,
    execute_task_local_ensemble,
)


TRUTHS = ((10.0, 10.0), (10.0, 10.0), (10.0, 10.0))


def _diagnostic(
    name: str,
    forecasts: tuple[tuple[float, ...], ...],
    *,
    family: str,
) -> CandidateDiagnostics:
    return CandidateDiagnostics.synthetic(
        name=name,
        family=family,
        median_mase=1.0,
        fold_forecasts=forecasts,
        fold_truths=TRUTHS,
        median_smae=1.0,
        median_srmse=1.0,
    )


def _policy(**overrides: object) -> TaskLocalTournamentPolicy:
    values: dict[str, object] = {
        "anchor_name": "toto_2_0",
        "minimum_successful_folds": 3,
        "minimum_joint_improvement": 0.02,
        "maximum_worst_joint_regret": 0.25,
        "maximum_raw_smae": 10.0,
        "maximum_raw_srmse": 10.0,
    }
    values.update(overrides)
    return TaskLocalTournamentPolicy(**values)


def _valid_inputs() -> dict[str, object]:
    anchor_folds = ((8.0, 8.0),) * 3
    specialist_folds = ((10.0, 10.0),) * 3
    return {
        "candidate_names": ("toto_2_0", "seasonal_naive"),
        "forecasts": {
            "toto_2_0": (8.0, 8.0),
            "seasonal_naive": (10.0, 10.0),
        },
        "diagnostics": {
            "toto_2_0": _diagnostic("toto_2_0", anchor_folds, family="tsfm"),
            "seasonal_naive": _diagnostic(
                "seasonal_naive", specialist_folds, family="statistical"
            ),
        },
        "horizon": 2,
    }


def test_local_tournament_returns_exact_anchor_when_specialist_is_invalid() -> None:
    inputs = _valid_inputs()
    inputs["forecasts"] = {
        "toto_2_0": (8.0, 8.0),
        "seasonal_naive": (math.nan, 10.0),
    }

    result = execute_task_local_ensemble(_policy(), **inputs)

    assert result.forecast == (8.0, 8.0)
    assert result.selected_names == ("toto_2_0",)
    assert result.weights == (1.0,)
    assert result.activated is False
    assert result.fallback_reason == "no_eligible_specialist"


def test_local_tournament_weights_are_deterministic_normalized_and_anchor_heavy() -> None:
    inputs = _valid_inputs()
    first = execute_task_local_ensemble(_policy(), **inputs)
    second = execute_task_local_ensemble(
        _policy(),
        candidate_names=("seasonal_naive", "toto_2_0"),
        forecasts=dict(reversed(tuple(inputs["forecasts"].items()))),  # type: ignore[union-attr]
        diagnostics=dict(reversed(tuple(inputs["diagnostics"].items()))),  # type: ignore[union-attr]
        horizon=2,
    )

    assert first == second
    assert first.activated is True
    assert first.selected_names == ("toto_2_0", "seasonal_naive")
    assert first.weights == (0.5, 0.5)
    assert sum(first.weights) == pytest.approx(1.0, abs=1e-12)
    assert first.weights[first.selected_names.index("toto_2_0")] >= 0.5
    assert first.maximum_fold_regret == 0.0


def test_smae_only_gain_cannot_hide_srmse_regression() -> None:
    anchor_folds = ((9.0, 9.0),) * 3
    specialist_folds = ((10.0, 8.2),) * 3
    result = execute_task_local_ensemble(
        _policy(),
        candidate_names=("toto_2_0", "uneven_specialist"),
        forecasts={"toto_2_0": (9.0, 9.0), "uneven_specialist": (10.0, 8.2)},
        diagnostics={
            "toto_2_0": _diagnostic("toto_2_0", anchor_folds, family="tsfm"),
            "uneven_specialist": _diagnostic(
                "uneven_specialist", specialist_folds, family="statistical"
            ),
        },
        horizon=2,
    )

    assert result.activated is False
    assert result.forecast == (9.0, 9.0)
    assert result.fallback_reason == "no_pareto_safe_weight"


def test_worst_fold_regression_rejects_better_median_weight() -> None:
    anchor_folds = ((8.0, 8.0),) * 3
    specialist_folds = ((10.0, 10.0), (10.0, 10.0), (0.0, 0.0))
    result = execute_task_local_ensemble(
        _policy(maximum_worst_joint_regret=0.1),
        candidate_names=("toto_2_0", "unstable_specialist"),
        forecasts={"toto_2_0": (8.0, 8.0), "unstable_specialist": (10.0, 10.0)},
        diagnostics={
            "toto_2_0": _diagnostic("toto_2_0", anchor_folds, family="tsfm"),
            "unstable_specialist": _diagnostic(
                "unstable_specialist", specialist_folds, family="statistical"
            ),
        },
        horizon=2,
    )

    assert result.activated is False
    assert result.forecast == (8.0, 8.0)
    assert result.fallback_reason == "no_eligible_specialist"


def test_tournament_rejects_more_than_eight_candidates() -> None:
    inputs = _valid_inputs()
    with pytest.raises(ValueError, match="at most eight"):
        execute_task_local_ensemble(
            _policy(),
            candidate_names=("toto_2_0", *(f"method_{index}" for index in range(8))),
            forecasts=inputs["forecasts"],
            diagnostics=inputs["diagnostics"],
            horizon=2,
        )
