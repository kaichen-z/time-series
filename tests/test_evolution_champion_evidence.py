"""Trusted scoring and sanitized Build-evidence contracts."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

import numerical_agent.evolution.champion_evidence as champion_evidence
from numerical_agent.evolution.champion_evidence import (
    ChampionEvidenceError,
    ChampionGateConfig,
    ChampionHistoryDiagnostic,
    ChampionTaskRow,
    WinTieLoss,
    compare_champion,
    sanitize_build_evidence,
    score_policy,
)
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
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


def test_build_row_binds_exact_history_only_diagnostics_without_fold_arrays() -> None:
    source = CandidateDiagnostics.synthetic(
        name="candidate",
        family="statistical",
        median_mase=0.5,
        fold_forecasts=((1.0,),),
        fold_truths=((9.0,),),
    )
    diagnostic = ChampionHistoryDiagnostic.from_candidate(source)
    history = (1.0,) * 200

    row = ChampionTaskRow(
        task_id="history_bound",
        candidate_name="candidate",
        profile=_profile("history_bound"),
        truth=(1.0, 1.0, 1.0, 1.0),
        forecast=(1.0, 1.0, 1.0, 1.0),
        history=history,
        diagnostic=diagnostic,
    )

    assert row.history is history
    assert row.diagnostic is diagnostic
    assert not hasattr(diagnostic, "fold_truths")
    assert not hasattr(diagnostic, "fold_forecasts")

    with pytest.raises(ChampionEvidenceError):
        replace(row, history=list(history))  # type: ignore[arg-type]
    with pytest.raises(ChampionEvidenceError):
        replace(row, history=history[:-1])
    with pytest.raises(ChampionEvidenceError):
        replace(row, diagnostic=replace(diagnostic, name="other"))


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


def test_scoring_rejects_conflicting_truths_before_building_policy_scores() -> None:
    rows = (
        _row("same", "parent", (1.0,) * 4, truth=(1.0,) * 4),
        _row("same", "child", (2.0,) * 4, truth=(2.0,) * 4),
    )

    with pytest.raises(ChampionEvidenceError, match="truth|universe"):
        score_policy(rows, "parent")


def test_comparison_binds_the_exact_truth_universe_across_separate_scores() -> None:
    parent = score_policy(
        (_row("same", "parent", (1.0,) * 4, truth=(1.0,) * 4),),
        "parent",
    )
    child = score_policy(
        (_row("same", "child", (2.0,) * 4, truth=(2.0,) * 4),),
        "child",
    )

    with pytest.raises(ChampionEvidenceError, match="truth|universe"):
        compare_champion(parent, child, _gate())


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
    _, parent, child = _scores((4.0,) * 10, (0.0,) * 8 + (10.0,) * 2)

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
    rows, parent, child = _scores((4.0,) * 10, (0.0,) * 9 + (10.0,))

    clipped_result = compare_champion(
        parent,
        child,
        _gate(maximum_task_regret_smae=5.0, maximum_task_regret_srmse=5.0),
    )
    assert "smae_clipped_count" in clipped_result.failures
    assert "srmse_clipped_count" in clipped_result.failures

    failed_rows = (
        _row("a", "parent", (2.0,) * 4),
        _row("b", "parent", (2.0,) * 4),
        _row("a", "child", (1.5,) * 4),
        _row("b", "child", None, failure_reason="runtime failed"),
    )
    failed_parent = score_policy(failed_rows, "parent")
    failed_child = score_policy(failed_rows, "child")
    coverage_result = compare_champion(failed_parent, failed_child, _gate())
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


def test_common_successful_pairs_own_primary_and_tail_authority() -> None:
    rows = (
        _row("easy", "parent", (1.1,) * 4),
        _row("hard", "parent", (6.0,) * 4),
        _row("easy", "child", (1.2,) * 4),
        _row("hard", "child", None, failure_reason="bounded failure"),
    )
    parent = score_policy(rows, "parent")
    child = score_policy(rows, "child")

    result = compare_champion(
        parent,
        child,
        _gate(
            minimum_coverage=0.5,
            maximum_failure_rate=0.5,
            maximum_coverage_regression=0.5,
            maximum_failure_rate_increase=0.5,
        ),
    )

    assert child.mean_smae < parent.mean_smae  # Unpaired aggregates are misleading.
    assert result.mean_delta_smae == pytest.approx(0.1)
    assert result.mean_delta_srmse == pytest.approx(0.1)
    assert {"mean_smae", "mean_srmse"} <= set(result.failures)
    assert {
        "p90_smae", "p95_smae", "p90_srmse", "p95_srmse",
        "p90_smae_raw", "p95_smae_raw", "p90_srmse_raw", "p95_srmse_raw",
    } <= set(result.failures)
    assert result.wtl == WinTieLoss(wins=0, ties=0, losses=1)


def test_public_comparison_rejects_forged_or_incoherent_scores() -> None:
    _, parent, child = _scores((1.0, 1.0), (0.5, 0.5))

    with pytest.raises(ChampionEvidenceError, match="coherent|coverage"):
        replace(child, coverage=0.5)

    object.__setattr__(child, "mean_smae", 0.0)
    with pytest.raises(ChampionEvidenceError, match="coherent|mean_smae"):
        compare_champion(parent, child, _gate())


@pytest.mark.parametrize(
    "field",
    (
        "total_tasks",
        "successful_tasks",
        "failure_count",
        "coverage",
        "failure_rate",
        "smae_clipped_count",
        "srmse_clipped_count",
        "smae_clipped_rate",
        "srmse_clipped_rate",
        "fold_count",
        "mean_smae",
        "mean_srmse",
        "median_smae",
        "median_srmse",
        "p90_smae",
        "p95_smae",
        "max_smae",
        "mean_smae_raw",
        "median_srmse_raw",
        "p90_smae_raw",
        "p95_srmse_raw",
        "max_srmse_raw",
    ),
)
def test_every_derived_score_field_is_exactly_bound_to_task_metrics(field: str) -> None:
    rows = (
        _row("a", "candidate", (1.0, 1.0, 1.0, 25.0), fold=0),
        _row("b", "candidate", (1.5,) * 4, fold=1),
        _row("c", "candidate", None, failure_reason="bounded failure", fold=2),
    )
    score = score_policy(rows, "candidate")
    current = getattr(score, field)
    mutation = current + (1 if type(current) is int else 1e-6)

    with pytest.raises(ChampionEvidenceError, match=field):
        replace(score, **{field: mutation})


def test_score_coherence_rejects_even_sub_tolerance_metric_forgery() -> None:
    _, _, child = _scores((1.0,), (0.5,))

    with pytest.raises(ChampionEvidenceError, match="mean_smae"):
        replace(child, mean_smae=child.mean_smae + 5e-13)


def test_joint_gain_is_diagnostic_only_and_not_an_acceptance_threshold() -> None:
    _, parent, child = _scores((1.0,) * 5, (0.999,) * 5)

    result = compare_champion(parent, child, _gate())

    assert result.accepted is True
    assert 0.0 < result.joint_improvement < 0.005
    assert "joint_improvement" not in result.failures


def test_fold_stability_requires_independent_smae_srmse_pareto_behavior() -> None:
    rows = (
        _row("a", "parent", (2.0, 2.0, 2.0, 2.0), fold=0),
        _row("a", "child", (1.0, 1.0, 1.0, 3.1), fold=0),
    )

    result = compare_champion(
        score_policy(rows, "parent"),
        score_policy(rows, "child"),
        _gate(minimum_improved_folds=1),
    )

    assert result.improved_folds == 0
    assert "fold_stability" in result.failures


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
    assert not {
        "accepted", "rejected", "passed", "failures", "gate", "gates"
    } & set(payload["comparisons"][0])


def test_sanitizer_recomputes_diagnostics_instead_of_copying_decision_authority() -> None:
    rows, parent, child = _scores((1.0,), (0.5,))
    comparison = compare_champion(parent, child, _gate())
    object.__setattr__(comparison, "mean_delta_smae", -999.0)
    object.__setattr__(comparison, "accepted", False)
    object.__setattr__(comparison, "failures", ("forged_gate",))

    payload = sanitize_build_evidence(rows, (comparison,)).to_payload()

    assert payload["comparisons"][0]["mean_delta_smae"] == pytest.approx(-0.5)
    assert "accepted" not in payload["comparisons"][0]
    assert "failures" not in payload["comparisons"][0]


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


def test_sanitizer_ignores_forged_comparison_fields_and_blocks_frequency_markers() -> None:
    rows, parent, child = _scores((1.0,), (0.5,))
    comparison = compare_champion(parent, child, _gate())
    object.__setattr__(comparison, "child_p90_smae_raw", "not-a-number")
    payload = sanitize_build_evidence(rows, (comparison,)).to_payload()
    assert payload["comparisons"][0]["p90_smae_raw"] == pytest.approx(0.5)

    marked_rows = tuple(
        replace(row, profile=replace(row.profile, frequency="Public daily"))
        for row in rows
    )
    marked_parent = score_policy(marked_rows, "parent")
    marked_child = score_policy(marked_rows, "child")
    marked_comparison = compare_champion(marked_parent, marked_child, _gate())
    with pytest.raises(ChampionEvidenceError, match="Dev/Public"):
        sanitize_build_evidence(marked_rows, (marked_comparison,))

    camel_rows = tuple(
        replace(row, profile=replace(row.profile, frequency="PublicDaily"))
        for row in rows
    )
    camel_comparison = compare_champion(
        score_policy(camel_rows, "parent"),
        score_policy(camel_rows, "child"),
        _gate(),
    )
    with pytest.raises(ChampionEvidenceError, match="Dev/Public"):
        sanitize_build_evidence(camel_rows, (camel_comparison,))


def test_frequency_groups_are_preregistered_buckets_and_never_task_identities() -> None:
    rows, parent, child = _scores((1.0,), (0.5,))
    custom_rows = tuple(
        replace(row, profile=replace(row.profile, frequency="tenant-secret-frequency"))
        for row in rows
    )
    custom_comparison = compare_champion(
        score_policy(custom_rows, "parent"),
        score_policy(custom_rows, "child"),
        _gate(),
    )
    evidence = sanitize_build_evidence(custom_rows, (custom_comparison,))
    group_ids = {item.group_id for item in evidence.morphology}
    assert "frequency:other" in group_ids
    assert not any("tenant" in group_id for group_id in group_ids)

    identity_rows = (
        _row("daily", "parent", (2.0,) * 4),
        _row("daily", "child", (1.5,) * 4),
    )
    identity_comparison = compare_champion(
        score_policy(identity_rows, "parent"),
        score_policy(identity_rows, "child"),
        _gate(),
    )
    with pytest.raises(ChampionEvidenceError, match="identity"):
        sanitize_build_evidence(identity_rows, (identity_comparison,))


def _named_child_evidence(child_name: str, *, task_id: str = "secret_task_0"):
    rows = (
        _row(task_id, "parent", (2.0,) * 4),
        _row(task_id, child_name, (1.5,) * 4),
    )
    comparison = compare_champion(
        score_policy(rows, "parent"),
        score_policy(rows, child_name),
        _gate(),
    )
    return rows, comparison


def test_sanitizer_rejects_embedded_normalized_task_identity_sequences() -> None:
    rows, comparison = _named_child_evidence("model_secret_task_0")

    with pytest.raises(ChampionEvidenceError, match="identity"):
        sanitize_build_evidence(rows, (comparison,))


@pytest.mark.parametrize(
    ("task_id", "child_name"),
    (
        ("租户", "model_租户"),
        ("Ｔｅｎａｎｔ", "model_tenant"),
    ),
)
def test_sanitizer_rejects_unicode_normalized_embedded_task_identities(
    task_id: str,
    child_name: str,
) -> None:
    rows, comparison = _named_child_evidence(child_name, task_id=task_id)

    with pytest.raises(ChampionEvidenceError, match="identity"):
        sanitize_build_evidence(rows, (comparison,))


@pytest.mark.parametrize(
    "child_name",
    (
        "DEVops_model",
        "devops_model",
        "myDEVops_model",
        "myＤＥＶops_model",
        "modelPUBLICdaily",
        "DevModel",
        "DEVModel",
        "dev_model",
        "PublicModel",
    ),
)
def test_sanitizer_blocks_reserved_markers_across_identifier_forms(
    child_name: str,
) -> None:
    rows, comparison = _named_child_evidence(child_name)

    with pytest.raises(ChampionEvidenceError, match="forbidden|Dev/Public"):
        sanitize_build_evidence(rows, (comparison,))


@pytest.mark.parametrize(
    "frequency",
    (
        "dev-daily",
        "devops-daily",
        "public.daily",
        "DEVops-daily",
        "Public-Daily",
    ),
)
def test_sanitizer_blocks_reserved_markers_across_separator_forms(
    frequency: str,
) -> None:
    rows, comparison = _named_child_evidence("child")
    marked_rows = tuple(
        replace(row, profile=replace(row.profile, frequency=frequency))
        for row in rows
    )
    marked_comparison = compare_champion(
        score_policy(marked_rows, "parent"),
        score_policy(marked_rows, "child"),
        _gate(),
    )

    with pytest.raises(ChampionEvidenceError, match="Dev/Public"):
        sanitize_build_evidence(marked_rows, (marked_comparison,))


@pytest.mark.parametrize(
    "child_name",
    (
        "entity_secret",
        "truth_model",
        "exception_payload",
        "hidden_task",
        "EntitySecret",
        "TruthModel",
        "ＴｒｕｔｈModel",
        "EXCEPTIONPayload",
        "myHIDDENtask",
    ),
)
def test_sanitizer_blocks_sensitive_markers_across_identifier_forms(
    child_name: str,
) -> None:
    rows, comparison = _named_child_evidence(child_name)

    with pytest.raises(ChampionEvidenceError, match="forbidden|sensitive"):
        sanitize_build_evidence(rows, (comparison,))


@pytest.mark.parametrize(
    "value",
    (
        "hidden-task",
        "model.entity",
        "truth/payload",
        "EXCEPTION-payload",
    ),
)
def test_recursive_payload_validator_blocks_sensitive_separator_forms(
    value: str,
) -> None:
    with pytest.raises(ChampionEvidenceError, match="forbidden|sensitive"):
        champion_evidence._assert_sanitized({"label": value})


@pytest.mark.parametrize(
    "child_name",
    (
        "device_model",
        "developer_model",
        "devonian_model",
        "publicity_model",
        "republic_model",
        "myDeviceModel",
        "modelPublicityDaily",
        "myRepublicModel",
        "identity_model",
        "truthful_model",
        "ｔｒｕｔｈｆｕｌ_model",
        "exceptional_model",
        "unhidden_model",
        "model_secret_task_01",
        "model_secret_tasks_0",
    ),
)
def test_reserved_and_identity_checks_allow_benign_near_misses(
    child_name: str,
) -> None:
    rows, comparison = _named_child_evidence(child_name)

    payload = sanitize_build_evidence(rows, (comparison,)).to_payload()

    assert payload["comparisons"][0]["candidate_name"] == child_name


def test_sanitizer_builds_the_full_identity_index_once(monkeypatch) -> None:
    builder = getattr(champion_evidence, "_build_identity_index", None)
    assert callable(builder), "sanitizer requires one explicit identity-index boundary"
    calls: list[tuple[str, ...]] = []

    def counted(task_ids: tuple[str, ...]):
        calls.append(task_ids)
        return builder(task_ids)

    monkeypatch.setattr(champion_evidence, "_build_identity_index", counted)
    rows, parent, child = _scores((1.0,) * 25, (0.5,) * 25)
    comparison = compare_champion(parent, child, _gate())

    sanitize_build_evidence(rows, (comparison,))

    assert calls == [tuple(f"secret_task_{index}" for index in range(25))]


def test_evidence_contracts_are_frozen() -> None:
    rows, _, child = _scores((1.0,), (0.5,))
    with pytest.raises(FrozenInstanceError):
        child.coverage = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        rows[0].forecast = (9.0,) * 4  # type: ignore[misc]
