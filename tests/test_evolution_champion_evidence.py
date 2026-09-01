"""Trusted scoring and sanitized Build-evidence contracts."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from numerical_agent.evolution.champion_evidence import (
    ChampionEvidenceError,
    ChampionGateConfig,
    ChampionTaskRow,
    WinTieLoss,
    compare_champion,
    sanitize_build_evidence,
    score_policy,
)
from numerical_agent.evolution.screening import TaskProfile


def _profile(task_id: str, **changes: object) -> TaskProfile:
    values: dict[str, object] = {
        "task_id": task_id,
        "frequency": "D",
        "history_length": 200,
        "horizon": 4,
        "zero_fraction": 0.0,
        "signed": False,
        "integer_valued": False,
        "trend_direction": "up",
        "trend_strength": 0.8,
        "periodicity_periods": (7,),
        "periodicity_strength": 0.8,
        "periodicity_confidence": 0.9,
        "outlier_fraction": 0.0,
        "noise_relative_scale": 0.2,
        "likely_stationary": False,
        "stationarity_score": 0.2,
        "recent_regime_start": None,
        "recent_regime_confidence": 0.1,
        "intermittency_adi": 1.0,
        "intermittency_cv2": 0.1,
    }
    values.update(changes)
    return TaskProfile(**values)  # type: ignore[arg-type]


def _row(
    task_id: str,
    candidate_name: str,
    forecast: tuple[float, ...] | None,
    *,
    truth: tuple[float, ...] | None = (1.0, 1.0, 1.0, 1.0),
    profile: TaskProfile | None = None,
    failure_reason: str | None = None,
    fold: int = 0,
    split: str = "build",
) -> ChampionTaskRow:
    return ChampionTaskRow(
        task_id=task_id,
        candidate_name=candidate_name,
        profile=profile or _profile(task_id),
        truth=truth,
        forecast=forecast,
        failure_reason=failure_reason,
        fold=fold,
        split=split,
    )


def _paired_rows(
    parent_errors: tuple[float, ...],
    child_errors: tuple[float, ...],
    *,
    folds: tuple[int, ...] | None = None,
) -> tuple[ChampionTaskRow, ...]:
    if folds is None:
        folds = tuple(index % 5 for index in range(len(parent_errors)))
    rows: list[ChampionTaskRow] = []
    for index, (parent_error, child_error, fold) in enumerate(
        zip(parent_errors, child_errors, folds)
    ):
        task_id = f"secret_task_{index}"
        rows.extend(
            (
                _row(task_id, "parent", (1.0 + parent_error,) * 4, fold=fold),
                _row(task_id, "child", (1.0 + child_error,) * 4, fold=fold),
            )
        )
    return tuple(rows)


def _scores(
    parent_errors: tuple[float, ...],
    child_errors: tuple[float, ...],
    *,
    folds: tuple[int, ...] | None = None,
):
    rows = _paired_rows(parent_errors, child_errors, folds=folds)
    return rows, score_policy(rows, "parent"), score_policy(rows, "child")


def _gate(**changes: object) -> ChampionGateConfig:
    values: dict[str, object] = {
        "minimum_joint_improvement": 0.0,
        "primary_regression_tolerance": 0.0,
        "tail_regression_tolerance": 0.0,
        "minimum_coverage": 1.0,
        "maximum_failure_rate": 0.0,
        "maximum_coverage_regression": 0.0,
        "maximum_failure_rate_increase": 0.0,
        "maximum_clipped_count_increase": 0,
        "maximum_task_regret_smae": 1.0,
        "maximum_task_regret_srmse": 1.0,
        "minimum_improved_folds": 0,
        "tie_tolerance": 1e-12,
    }
    values.update(changes)
    return ChampionGateConfig(**values)  # type: ignore[arg-type]


def test_score_policy_uses_canonical_capped_and_raw_metric_authority() -> None:
    rows = (
        _row("a", "candidate", (1.0, 1.0, 1.0, 25.0)),
        _row("b", "candidate", (1.5, 1.5, 1.5, 1.5)),
    )

    result = score_policy(rows, "candidate")

    assert result.mean_smae == pytest.approx(2.75)
    assert result.mean_srmse == pytest.approx(2.75)
    assert result.max_smae_raw == pytest.approx(6.0)
    assert result.max_srmse_raw == pytest.approx(12.0)
    assert result.p90_smae_raw == pytest.approx(5.45)
    assert result.p95_srmse_raw == pytest.approx(11.425)
    assert result.smae_clipped_count == 1
    assert result.srmse_clipped_count == 1
    assert result.coverage == 1.0
    assert result.failure_count == 0


def test_score_policy_counts_explicit_failures_without_inspecting_raw_exception() -> None:
    rows = (
        _row("a", "candidate", (1.0,) * 4),
        _row(
            "b",
            "candidate",
            None,
            truth=None,
            failure_reason="RuntimeError: entity-secret failed on Public row",
        ),
    )

    result = score_policy(rows, "candidate")

    assert result.coverage == 0.5
    assert result.failure_rate == 0.5
    assert result.failure_count == 1


@pytest.mark.parametrize(
    "rows, message",
    (
        (
            lambda: (
                _row("a", "candidate", (1.0,) * 4),
                _row("a", "candidate", (1.0,) * 4),
            ),
            "duplicate candidate/task",
        ),
        (
            lambda: (
                _row(
                    "a",
                    "candidate",
                    (1.0,) * 4,
                    profile=_profile("different"),
                ),
            ),
            "mislabeled",
        ),
        (
            lambda: (_row("a", "candidate", (1.0,) * 3),),
            "complete",
        ),
        (
            lambda: (_row("a", "candidate", (1.0, 1.0, float("nan"), 1.0)),),
            "finite",
        ),
    ),
)
def test_score_policy_fails_closed_on_invalid_rows(rows, message: str) -> None:
    with pytest.raises(ChampionEvidenceError, match=message):
        score_policy(rows(), "candidate")


def test_score_policy_rejects_missing_candidate_task_pair() -> None:
    rows = (
        _row("a", "parent", (1.0,) * 4),
        _row("b", "parent", (1.0,) * 4),
        _row("a", "child", (1.0,) * 4),
    )

    with pytest.raises(ChampionEvidenceError, match="complete task coverage"):
        score_policy(rows, "child")


@pytest.mark.parametrize(
    "row",
    (
        lambda: _row("a", "candidate", (True, 1.0, 1.0, 1.0)),
        lambda: _row(
            "a",
            "candidate",
            (1.0,) * 4,
            profile=_profile("a", trend_strength=1),
        ),
    ),
)
def test_rows_reject_wrong_exact_numeric_types(row) -> None:
    with pytest.raises(ChampionEvidenceError, match="exact|float"):
        score_policy((row(),), "candidate")


def test_pair_gate_rejects_srmse_regression_hidden_by_joint_mean() -> None:
    # Uneven horizons make RMSE react more strongly than MAE to one outlier.
    rows = (
        _row("a", "parent", (2.0, 2.0, 2.0, 2.0)),
        _row("a", "child", (1.0, 1.0, 1.0, 3.1)),
    )
    parent = score_policy(rows, "parent")
    child = score_policy(rows, "child")

    result = compare_champion(parent, child, _gate())

    assert child.mean_smae < parent.mean_smae
    assert child.mean_srmse > parent.mean_srmse
    assert result.accepted is False
    assert "mean_srmse" in result.failures


@pytest.mark.parametrize(
    "metric",
    ("p90_smae_raw", "p95_smae_raw", "p90_srmse_raw", "p95_srmse_raw"),
)
def test_each_raw_tail_is_an_independent_gate(metric: str) -> None:
    _, parent, child = _scores((1.0,) * 10, (0.5,) * 10)
    object.__setattr__(child, metric, 99.0)

    result = compare_champion(parent, child, _gate())

    assert result.accepted is False
    assert metric in result.failures


def test_win_count_is_reported_but_not_a_standalone_gate() -> None:
    # Four large wins outweigh six small losses while both paired means improve.
    _, parent, child = _scores(
        (2.0,) * 4 + (1.0,) * 6,
        (0.5,) * 4 + (1.1,) * 6,
    )

    result = compare_champion(
        parent,
        child,
        _gate(maximum_task_regret_smae=0.2, maximum_task_regret_srmse=0.2),
    )

    assert result.accepted is True
    assert result.wtl == WinTieLoss(wins=4, ties=0, losses=6)


def test_coverage_clipping_regret_and_fold_stability_remain_independent_gates() -> None:
    rows, parent, child = _scores((1.0,) * 10, (0.5,) * 10)

    clipped = replace(child, smae_clipped_count=1)
    assert "smae_clipped_count" in compare_champion(parent, clipped, _gate()).failures

    low_coverage = replace(
        child,
        successful_tasks=9,
        coverage=0.9,
        failure_rate=0.1,
        failure_count=1,
    )
    coverage_result = compare_champion(parent, low_coverage, _gate())
    assert {"coverage", "failure_rate"} <= set(coverage_result.failures)

    _, regret_parent, regret_child = _scores((1.0, 1.0), (0.0, 1.2))
    regret_result = compare_champion(
        regret_parent,
        regret_child,
        _gate(maximum_task_regret_smae=0.1, maximum_task_regret_srmse=0.1),
    )
    assert "max_regret_smae" in regret_result.failures
    assert "max_regret_srmse" in regret_result.failures

    unstable = compare_champion(parent, child, _gate(minimum_improved_folds=6))
    assert "fold_stability" in unstable.failures
    assert unstable.total_folds == 5
    assert unstable.improved_folds == 5
    assert len(rows) == 20


def _walk(value: object):
    if hasattr(value, "__dataclass_fields__"):
        yield from _walk(vars(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def test_sanitized_build_evidence_contains_only_anonymous_reviewed_aggregates() -> None:
    rows, parent, child = _scores(
        (1.0, 1.0, 1.0, 1.0),
        (0.5, 0.5, 1.5, 0.5),
    )
    comparison = compare_champion(
        parent,
        child,
        _gate(maximum_task_regret_smae=1.0, maximum_task_regret_srmse=1.0),
    )
    evidence = sanitize_build_evidence(rows, (comparison,))

    periodic = next(
        aggregate
        for aggregate in evidence.morphology
        if aggregate.group_id == "periodicity_strength:high"
    )
    assert periodic.support == 4
    assert periodic.candidate_name == "child"
    assert periodic.mean_delta_smae == pytest.approx(-0.25)
    assert periodic.mean_delta_srmse == pytest.approx(-0.25)
    assert periodic.coverage == 1.0
    assert periodic.p95_regret_smae == pytest.approx(0.425)
    assert periodic.p95_regret_srmse == pytest.approx(0.425)

    payload = evidence.to_payload()
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False).lower()
    forbidden = (
        "secret_task_",
        "task_id",
        "truth",
        "forecast",
        "entity",
        "runtimeerror",
        "public row",
        '"dev"',
    )
    assert not any(token in serialized for token in forbidden)
    assert all(not isinstance(value, TaskProfile) for value in _walk(evidence))
    assert payload["label"] == "adaptive_train_build_diagnostic"
    assert payload["independent_generalization_claim"] is False


def test_sanitizer_rejects_mislabeled_rows_and_forbidden_candidate_markers() -> None:
    rows, parent, child = _scores((1.0,), (0.5,))
    comparison = compare_champion(parent, child, _gate())

    mislabeled = tuple(replace(row, split="dev") for row in rows)
    with pytest.raises(ChampionEvidenceError, match="Build"):
        sanitize_build_evidence(mislabeled, (comparison,))

    hostile_rows = tuple(
        replace(row, candidate_name="public_child") if row.candidate_name == "child" else row
        for row in rows
    )
    hostile_child = score_policy(hostile_rows, "public_child")
    hostile_comparison = compare_champion(parent, hostile_child, _gate())
    with pytest.raises(ChampionEvidenceError, match="forbidden"):
        sanitize_build_evidence(hostile_rows, (hostile_comparison,))


def test_sanitizer_revalidates_comparison_fields_and_frequency_markers() -> None:
    rows, parent, child = _scores((1.0,), (0.5,))
    comparison = compare_champion(parent, child, _gate())
    object.__setattr__(comparison, "child_p90_smae_raw", "not-a-number")
    with pytest.raises(ChampionEvidenceError, match="child_p90_smae_raw"):
        sanitize_build_evidence(rows, (comparison,))

    marked_rows = tuple(
        replace(row, profile=replace(row.profile, frequency="Public daily"))
        for row in rows
    )
    marked_parent = score_policy(marked_rows, "parent")
    marked_child = score_policy(marked_rows, "child")
    marked_comparison = compare_champion(marked_parent, marked_child, _gate())
    with pytest.raises(ChampionEvidenceError, match="Dev/Public"):
        sanitize_build_evidence(marked_rows, (marked_comparison,))


def test_evidence_contracts_are_frozen() -> None:
    rows, _, child = _scores((1.0,), (0.5,))
    with pytest.raises(FrozenInstanceError):
        child.coverage = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        rows[0].forecast = (9.0,) * 4  # type: ignore[misc]
