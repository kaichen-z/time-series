from __future__ import annotations

from dataclasses import replace

import pytest

from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.execution import CRASHED, NOT_APPLICABLE, SUCCESS, Outcome
from numerical_agent.evolution.screening import (
    ActiveDictionary,
    ApplicabilityClause,
    ApplicabilityPolicy,
    FeatureTest,
    ScreeningEntry,
    ScreeningGateResult,
    ScreeningPolicy,
    ScreeningScore,
    ScreeningConstraints,
    TaskProfile,
    compare_screening,
    evaluate_screening,
    materialize_active_dictionary,
    profile_task,
    task_group_tags,
)


def _task(history: tuple[float, ...], *, horizon: int = 3) -> Task:
    return Task("private-task-id", history, horizon, "1 day", (999.0,) * horizon)


def test_profile_is_history_only_and_hides_task_identity_and_future() -> None:
    profile = profile_task(_task((0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 5.0)))

    assert isinstance(profile, TaskProfile)
    assert profile.history_length == 9
    assert profile.horizon == 3
    assert profile.zero_fraction == pytest.approx(2.0 / 3.0)
    assert profile.intermittency_adi == pytest.approx(3.0)
    assert profile.intermittency_cv2 == pytest.approx(1.0 / 128.0)
    assert profile.signed is False
    assert profile.integer_valued is True
    assert profile.to_public_payload().keys() == {
        "frequency",
        "history_length",
        "horizon",
        "zero_fraction",
        "signed",
        "integer_valued",
        "trend_direction",
        "trend_strength",
        "periodicity_periods",
        "periodicity_strength",
        "periodicity_confidence",
        "outlier_fraction",
        "noise_relative_scale",
        "likely_stationary",
        "stationarity_score",
        "recent_regime_start",
        "recent_regime_confidence",
        "intermittency_adi",
        "intermittency_cv2",
    }
    assert "private-task-id" not in repr(profile.to_public_payload())
    assert "999.0" not in repr(profile.to_public_payload())


def test_profile_detects_periodic_trending_signed_and_recent_regime_histories() -> None:
    periodic = profile_task(_task(tuple([1.0, 3.0, 2.0] * 12)))
    trending = profile_task(_task(tuple(float(index) for index in range(30))))
    signed = profile_task(_task((-3.0, -1.0, 0.0, 2.0, 4.0, 7.0, 9.0, 12.0)))
    regime = profile_task(_task((1.0,) * 20 + (20.0,) * 20))

    assert 3 in periodic.periodicity_periods
    assert periodic.periodicity_strength >= 0.8
    assert trending.trend_direction == "up"
    assert trending.trend_strength >= 0.9
    assert signed.signed is True
    assert regime.recent_regime_start is not None
    assert regime.recent_regime_confidence > 0.0


def test_task_groups_cover_every_approved_history_only_dimension() -> None:
    task = Task(
        "hidden-id",
        tuple([0.0, 0.0, 5.0] * 20),
        12,
        "1 day",
        (999.0,) * 12,
    )

    groups = task_group_tags(profile_task(task))

    assert {group.split(":", 1)[0] for group in groups} == {
        "periodicity",
        "trend",
        "intermittency",
        "regime",
        "frequency",
        "history",
        "horizon",
    }
    assert "frequency:1 day" in groups
    assert "history:le_168" in groups
    assert "horizon:le_24" in groups
    assert "hidden-id" not in repr(groups)
    assert "999.0" not in repr(groups)


def test_task_group_tags_are_executable_applicability_conditions() -> None:
    profile = profile_task(_task(tuple([1.0, 3.0, 2.0] * 12)))

    assert ApplicabilityClause(("periodicity:strong",)).matches(profile)


def test_profile_is_deterministic_and_rejects_empty_history() -> None:
    task = _task((1.0, 2.0, 5.0, 3.0, 8.0, 4.0, 9.0, 5.0))

    assert profile_task(task) == profile_task(task)
    with pytest.raises(ValueError, match="history must not be empty"):
        profile_task(_task(()))


def test_applicability_is_or_across_clauses_and_and_inside_each_clause() -> None:
    policy = ApplicabilityPolicy(
        (
            ApplicabilityClause(
                all_tags=("intermittent", "nonnegative"),
            ),
            ApplicabilityClause(
                feature_tests=(FeatureTest("periodicity_strength", ">=", 0.8),),
            ),
        )
    )
    intermittent = profile_task(_task((0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0)))
    periodic = profile_task(_task(tuple([1.0, 3.0, 2.0] * 12)))
    unrelated = profile_task(_task((1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98)))

    assert policy.match(intermittent) == 0
    assert policy.match(periodic) == 1
    assert policy.match(unrelated) is None
    assert ApplicabilityPolicy(()).match(unrelated) == -1


@pytest.mark.parametrize(
    ("field", "operator", "value", "message"),
    (
        ("future", ">", 0.0, "unsupported profile field"),
        ("zero_fraction", "contains", 0.0, "unsupported feature operator"),
        ("zero_fraction", ">", float("nan"), "finite"),
    ),
)
def test_feature_tests_reject_unreviewed_expressions(
    field: str, operator: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FeatureTest(field, operator, value)


def test_materialization_records_active_and_excluded_reasons() -> None:
    profile = profile_task(_task((0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 5.0)))
    policy = ScreeningPolicy(
        entries=(
            ScreeningEntry(
                "croston", "statistical", "specialized",
                ApplicabilityPolicy((ApplicabilityClause(("intermittent", "nonnegative")),)),
                "intermittent demand specialist",
            ),
            ScreeningEntry(
                "seasonal", "statistical", "specialized",
                ApplicabilityPolicy((ApplicabilityClause(feature_tests=(
                    FeatureTest("periodicity_strength", ">", 1.0),
                )),)),
                "strong seasonality specialist",
            ),
            ScreeningEntry(
                "timesfm", "tsfm", "keep", ApplicabilityPolicy(()), "broad TSFM",
            ),
        ),
        fallback_names=("croston", "timesfm"),
    )

    active = materialize_active_dictionary(policy, profile)

    assert isinstance(active, ActiveDictionary)
    assert [candidate.name for candidate in active.active] == ["croston", "timesfm"]
    assert active.active[0].matched_clause == 0
    assert active.active[0].reason_codes == ("intermittent", "nonnegative")
    assert active.excluded[0].name == "seasonal"
    assert active.excluded[0].reason_code == "applicability_not_matched"
    assert not active.fallback_applied


def test_materialization_applies_reviewed_fallback_and_family_invariants() -> None:
    profile = profile_task(_task((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)))
    impossible = ApplicabilityPolicy((ApplicabilityClause(("intermittent",)),))
    policy = ScreeningPolicy(
        entries=(
            ScreeningEntry("croston", "statistical", "specialized", impossible, "specialist"),
            ScreeningEntry("naive", "statistical", "keep", ApplicabilityPolicy(()), "fallback"),
            ScreeningEntry("timesfm", "tsfm", "specialized", impossible, "fallback"),
            ScreeningEntry("toto", "tsfm", "specialized", impossible, "fallback"),
        ),
        fallback_names=("naive", "timesfm", "toto"),
    )

    active = materialize_active_dictionary(policy, profile)

    assert [candidate.name for candidate in active.active] == ["naive", "timesfm", "toto"]
    assert active.fallback_applied
    assert {candidate.family for candidate in active.active} == {"statistical", "tsfm"}


def _screen_policy(*, broken_status: str = "keep", oracle_rule: ApplicabilityPolicy | None = None):
    return ScreeningPolicy(
        entries=(
            ScreeningEntry("stable", "statistical", "keep", ApplicabilityPolicy(()), "stable"),
            ScreeningEntry("oracle", "statistical", "specialized" if oracle_rule else "keep",
                           oracle_rule or ApplicabilityPolicy(()), "best specialist"),
            ScreeningEntry("broken", "statistical", broken_status, ApplicabilityPolicy(()), "bad"),
            ScreeningEntry("skip", "combined", "keep", ApplicabilityPolicy(()), "not applicable"),
            ScreeningEntry("timesfm", "tsfm", "keep", ApplicabilityPolicy(()), "broad"),
        ),
        fallback_names=("stable", "timesfm", "oracle"),
    )


def _screen_tasks(prefix: str = "task") -> tuple[Task, ...]:
    return (
        Task(f"{prefix}-dense", (1.0,) * 12, 2, "1 day", (1.0, 1.0)),
        Task(f"{prefix}-intermittent", (0.0, 0.0, 4.0) * 4, 2, "1 day", (4.0, 0.0)),
    )


def _screen_outcomes(tasks: tuple[Task, ...]) -> tuple[Outcome, ...]:
    rows = []
    for task in tasks:
        rows.extend(
            (
                Outcome("stable", task.task_id, SUCCESS, mase=1.0, mae=1.0, smape=1.0),
                Outcome("oracle", task.task_id, SUCCESS, mase=0.5, mae=0.5, smape=0.5),
                Outcome("broken", task.task_id, CRASHED, detail="crash"),
                Outcome("skip", task.task_id, NOT_APPLICABLE),
                Outcome("timesfm", task.task_id, SUCCESS, mase=1.2, mae=1.2, smape=1.2),
            )
        )
    return tuple(rows)


def test_screening_score_is_independent_of_a_final_forecast_selector() -> None:
    tasks = _screen_tasks()
    outcomes = _screen_outcomes(tasks)
    parent = evaluate_screening(_screen_policy(), tasks, outcomes)
    child = evaluate_screening(_screen_policy(broken_status="repair"), tasks, outcomes)

    assert isinstance(parent, ScreeningScore)
    assert child.coverage == 1.0
    assert child.active_success_rate > parent.active_success_rate
    assert child.failure_exposure < parent.failure_exposure
    assert child.not_applicable_exposure > parent.not_applicable_exposure
    assert child.compression < parent.compression
    assert child.global_oracle_retention == 1.0
    assert child.mean_active_oracle_regret == 0.0
    assert set(child.active_counts) == {task.task_id for task in tasks}
    assert child.unique_active_dictionaries == 1
    assert child.mean_pairwise_jaccard == 1.0
    assert child.min_active_candidates == child.max_active_candidates == 4


def test_screening_gate_rejects_compression_that_loses_the_dev_oracle() -> None:
    train = _screen_tasks("train")
    dev = _screen_tasks("dev")
    outcomes = _screen_outcomes(train + dev)
    parent = _screen_policy()
    impossible_oracle = ApplicabilityPolicy((ApplicabilityClause(("signed",)),))
    child = _screen_policy(broken_status="repair", oracle_rule=impossible_oracle)

    result = compare_screening(
        evaluate_screening(parent, train, outcomes),
        evaluate_screening(child, train, outcomes),
        evaluate_screening(parent, dev, outcomes),
        evaluate_screening(child, dev, outcomes),
    )

    assert isinstance(result, ScreeningGateResult)
    assert not result.accepted
    assert "oracle retention" in result.reason


def test_final_screening_gate_enforces_pool_bounds_and_task_diversity() -> None:
    tasks = _screen_tasks()
    outcomes = _screen_outcomes(tasks)
    parent = _screen_policy()
    child = _screen_policy(broken_status="repair")
    parent_score = evaluate_screening(parent, tasks, outcomes)
    child_score = evaluate_screening(child, tasks, outcomes)
    constraints = ScreeningConstraints(
        baseline_method="timesfm",
        min_active_candidates=2,
        max_active_candidates=4,
        min_unique_active_dictionaries=2,
        max_mean_pairwise_jaccard=0.99,
        min_group_support=1,
        required_conditioned_families=(),
    )

    result = compare_screening(
        parent_score,
        child_score,
        parent_score,
        child_score,
        constraints=constraints,
        enforce_final_constraints=True,
    )

    assert not result.accepted
    assert "task-conditioned diversity" in result.reason


def test_final_screening_gate_requires_conditions_for_all_method_families() -> None:
    tasks = _screen_tasks()
    outcomes = _screen_outcomes(tasks)
    parent = _screen_policy()
    child = _screen_policy(broken_status="repair")
    constraints = ScreeningConstraints(
        baseline_method="timesfm",
        min_active_candidates=1,
        max_active_candidates=5,
        min_unique_active_dictionaries=1,
        max_mean_pairwise_jaccard=1.0,
        min_group_support=1,
        required_conditioned_families=("statistical", "tsfm", "combined"),
    )

    result = compare_screening(
        evaluate_screening(parent, tasks, outcomes),
        evaluate_screening(child, tasks, outcomes),
        evaluate_screening(parent, tasks, outcomes),
        evaluate_screening(child, tasks, outcomes),
        constraints=constraints,
        enforce_final_constraints=True,
    )

    assert not result.accepted
    assert "conditioned entries" in result.reason


def test_conditioned_family_count_requires_a_rule_to_match_a_task() -> None:
    tasks = _screen_tasks()
    outcomes = _screen_outcomes(tasks)
    impossible = ApplicabilityPolicy((ApplicabilityClause(("signed",)),))
    policy = ScreeningPolicy(
        (
            ScreeningEntry("stable", "statistical", "keep", ApplicabilityPolicy(()), "stable"),
            ScreeningEntry("oracle", "statistical", "keep", ApplicabilityPolicy(()), "oracle"),
            ScreeningEntry("broken", "statistical", "repair", ApplicabilityPolicy(()), "bad"),
            ScreeningEntry("skip", "combined", "specialized", impossible, "unmatched"),
            ScreeningEntry("timesfm", "tsfm", "specialized", impossible, "unmatched"),
        ),
        ("stable", "timesfm", "oracle"),
    )

    score = evaluate_screening(policy, tasks, outcomes)

    assert score.conditioned_entries_by_family["combined"] == 0
    assert score.conditioned_entries_by_family["tsfm"] == 0


def test_screening_gate_rejects_train_failure_exposure_increase() -> None:
    train = _screen_tasks("train")
    dev = _screen_tasks("dev")
    outcomes = _screen_outcomes(train + dev)
    parent = _screen_policy(broken_status="repair")
    child = _screen_policy(broken_status="keep")

    result = compare_screening(
        evaluate_screening(parent, train, outcomes),
        evaluate_screening(child, train, outcomes),
        evaluate_screening(parent, dev, outcomes),
        evaluate_screening(parent, dev, outcomes),
    )

    assert not result.accepted
    assert "Train failure exposure" in result.reason


def test_screening_gate_uses_absolute_failure_burden_when_compressing_successes() -> None:
    """Removing a reliable broad candidate must not manufacture a failure regression."""
    train = _screen_tasks("train")
    dev = _screen_tasks("dev")
    outcomes = _screen_outcomes(train + dev)
    parent = _screen_policy()
    dense_only = ApplicabilityPolicy((ApplicabilityClause(("dense",)),))
    child = ScreeningPolicy(
        tuple(
            replace(entry, status="specialized", applicability=dense_only)
            if entry.name == "stable"
            else entry
            for entry in parent.entries
        ),
        parent.fallback_names,
    )
    train_parent = evaluate_screening(parent, train, outcomes)
    train_child = evaluate_screening(child, train, outcomes)
    dev_parent = evaluate_screening(parent, dev, outcomes)
    dev_child = evaluate_screening(child, dev, outcomes)

    assert train_child.failure_exposure > train_parent.failure_exposure
    assert train_child.mean_active_failures == train_parent.mean_active_failures
    result = compare_screening(
        train_parent,
        train_child,
        dev_parent,
        dev_child,
    )

    assert result.accepted
    assert "compression" in result.improved_dimensions


def test_screening_gate_requires_full_train_and_dev_oracle_retention() -> None:
    train = _screen_tasks("train")
    dev = _screen_tasks("dev")
    outcomes = _screen_outcomes(train + dev)
    parent = _screen_policy()
    only_dense = ApplicabilityPolicy((ApplicabilityClause(("dense",)),))
    child = _screen_policy(broken_status="repair", oracle_rule=only_dense)

    result = compare_screening(
        evaluate_screening(parent, train, outcomes),
        evaluate_screening(child, train, outcomes),
        evaluate_screening(parent, dev, outcomes),
        evaluate_screening(child, dev, outcomes),
    )

    assert not result.accepted
    assert "100%" in result.reason


def test_final_gate_allows_bounded_dev_oracle_loss_when_regret_remains_small() -> None:
    tasks = _screen_tasks()
    outcomes = _screen_outcomes(tasks)
    parent = evaluate_screening(_screen_policy(), tasks, outcomes)
    child = evaluate_screening(
        _screen_policy(broken_status="repair"), tasks, outcomes
    )
    dev_child = replace(
        child,
        global_oracle_retention=0.9,
        mean_active_oracle_regret=0.005,
    )
    constraints = ScreeningConstraints(
        baseline_method="timesfm",
        min_active_candidates=1,
        max_active_candidates=103,
        min_unique_active_dictionaries=1,
        max_mean_pairwise_jaccard=1.0,
        min_group_support=1,
        min_dev_oracle_retention=0.9,
        required_conditioned_families=(),
    )

    result = compare_screening(
        parent,
        child,
        parent,
        dev_child,
        constraints=constraints,
        enforce_final_constraints=True,
    )

    assert result.accepted
