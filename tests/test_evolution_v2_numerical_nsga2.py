from __future__ import annotations

import itertools
import json
import math

import pytest

from evolving_loop.v2.numerical_qd.contracts import (
    ConstraintReportV2,
    MorphologyCellV2,
    NumericalObjectiveVectorV2,
    NumericalQDEntryV2,
)
from evolving_loop.v2.numerical_qd.nsga2 import (
    constraint_compare,
    crowding_distances,
    non_dominated_fronts,
    select_survivors,
)


CELL = MorphologyCellV2("low", "none", "low", "stable", "short", "statistical")
BASE_OBJECTIVES = (0.5, 0.6, 0.9, 0.7, 0.1)


def entry(
    marker: str,
    objectives: tuple[float, float, float, float, float] = BASE_OBJECTIVES,
    violations: tuple[str, ...] = (),
) -> NumericalQDEntryV2:
    return NumericalQDEntryV2(
        schema_version=1,
        genome_sha256=marker * 64,
        evaluation_sha256=marker * 64,
        cell=CELL,
        task_ids=("task-a",),
        objectives=NumericalObjectiveVectorV2(*objectives),
        constraints=ConstraintReportV2(not violations, violations),
        train_diagnostic_categories=(),
    )


def artifact_order(entries):
    return tuple(sorted(entries, key=lambda item: item.fingerprint()))


def test_feasible_always_dominates_infeasible():
    feasible = entry("a", (9.0, 9.0, 9.0, 9.0, 9.0))
    infeasible = entry("b", (0.0, 0.0, 0.0, 0.0, 0.0), ("coverage",))

    assert constraint_compare(feasible, infeasible) == -1
    assert constraint_compare(infeasible, feasible) == 1


def test_infeasible_order_uses_only_violation_count_names_and_artifact_sha():
    fewer_but_worse = entry("c", (9.0, 9.0, 9.0, 9.0, 9.0), ("scope",))
    more_but_better = entry(
        "d", (0.0, 0.0, 0.0, 0.0, 0.0), ("coverage", "scope")
    )
    lexical_first = entry("e", violations=("coverage",))
    lexical_second = entry("f", violations=("scope",))
    same_violations = (
        entry("1", violations=("coverage",)),
        entry("2", violations=("coverage",)),
    )
    sha_first, sha_second = artifact_order(same_violations)

    assert constraint_compare(fewer_but_worse, more_but_better) == -1
    assert constraint_compare(lexical_first, lexical_second) == -1
    assert constraint_compare(sha_first, sha_second) == -1


def test_feasible_pareto_dominance_requires_a_strict_objective_improvement():
    better = entry("a", (0.4, 0.6, 0.9, 0.7, 0.1))
    worse = entry("b", BASE_OBJECTIVES)
    identical = entry("c", BASE_OBJECTIVES)

    assert constraint_compare(better, worse) == -1
    assert constraint_compare(worse, better) == 1
    assert constraint_compare(worse, identical) == 0


def test_pareto_front_keeps_tradeoffs_without_scalarization_and_orders_by_sha():
    tradeoffs = (
        entry("a", (0.1, 0.9, 0.5, 0.5, 0.5)),
        entry("b", (0.5, 0.5, 0.5, 0.5, 0.5)),
        entry("c", (0.9, 0.1, 0.5, 0.5, 0.5)),
    )
    expected = artifact_order(tradeoffs)

    assert non_dominated_fronts(tradeoffs) == (expected,)
    for permutation in itertools.permutations(tradeoffs):
        assert non_dominated_fronts(permutation) == (expected,)


def test_fronts_put_identical_objectives_together_and_dominated_entries_later():
    tied = (entry("a"), entry("b"))
    dominated = entry("c", (0.6, 0.7, 1.0, 0.8, 0.2))

    assert non_dominated_fronts((*reversed(tied), dominated)) == (
        artifact_order(tied),
        (dominated,),
    )


def test_fronts_reject_duplicate_artifact_sha_and_nonfinite_objectives():
    candidate = entry("a")
    with pytest.raises(ValueError, match="duplicate artifact SHA"):
        non_dominated_fronts((candidate, candidate))

    invalid_objectives = object.__new__(NumericalObjectiveVectorV2)
    for name, value in zip(
        ("mean_capped_smae", "mean_capped_srmse", "p95_capped_srmse",
         "mean_raw_joint_error", "normalized_execution_cost"),
        (math.inf, 0.6, 0.9, 0.7, 0.1),
    ):
        object.__setattr__(invalid_objectives, name, value)
    invalid = NumericalQDEntryV2(
        1, "b" * 64, "b" * 64, CELL, ("task-a",), invalid_objectives,
        ConstraintReportV2(True, ()), (),
    )
    with pytest.raises(ValueError, match="finite float"):
        non_dominated_fronts((invalid,))


def test_crowding_marks_boundaries_and_sums_normalized_objective_distances():
    front = (
        entry("a", (0.0, 0.0, 0.0, 0.0, 0.0)),
        entry("b", (0.25, 0.25, 0.25, 0.25, 0.25)),
        entry("c", (1.0, 1.0, 1.0, 1.0, 1.0)),
    )
    low, middle, high = sorted(front, key=lambda item: item.objectives.mean_capped_smae)
    distances = crowding_distances(tuple(reversed(front)))

    assert distances[low.fingerprint()] == math.inf
    assert distances[middle.fingerprint()] == pytest.approx(5.0)
    assert distances[high.fingerprint()] == math.inf


def test_zero_range_objectives_contribute_zero_to_crowding():
    front = (
        entry("a", (0.0, 2.0, 2.0, 2.0, 2.0)),
        entry("b", (0.5, 2.0, 2.0, 2.0, 2.0)),
        entry("c", (1.0, 2.0, 2.0, 2.0, 2.0)),
    )
    low, middle, high = sorted(front, key=lambda item: item.objectives.mean_capped_smae)
    distances = crowding_distances(front)

    assert distances[low.fingerprint()] == math.inf
    assert distances[middle.fingerprint()] == pytest.approx(1.0)
    assert distances[high.fingerprint()] == math.inf

    all_tied = (entry("d"), entry("e"), entry("f"))
    assert crowding_distances(all_tied) == {
        candidate.fingerprint(): 0.0 for candidate in all_tied
    }


def test_crowding_normalization_stays_finite_for_extreme_finite_objectives():
    front = (
        entry("a", (-1e308, -1e308, -1e308, -1e308, -1e308)),
        entry("b", (0.0, 0.0, 0.0, 0.0, 0.0)),
        entry("c", (1e308, 1e308, 1e308, 1e308, 1e308)),
    )

    distances = crowding_distances(front)

    assert distances[front[1].fingerprint()] == pytest.approx(5.0)


def test_select_survivors_applies_capacity_crowding_and_sha_ties_deterministically():
    diverse = (
        entry("a", (0.0, 2.0, 1.0, 1.0, 1.0)),
        entry("b", (1.0, 1.0, 1.0, 1.0, 1.0)),
        entry("c", (2.0, 0.0, 1.0, 1.0, 1.0)),
    )
    boundary = {diverse[0].fingerprint(), diverse[2].fingerprint()}
    assert {item.fingerprint() for item in select_survivors(diverse, 2)} == boundary

    tied = (entry("d"), entry("e"), entry("f"))
    expected = artifact_order(tied)[:2]
    for permutation in itertools.permutations(tied):
        assert select_survivors(permutation, 2) == expected


def test_select_survivors_orders_complete_fronts_and_never_exposes_infinity():
    first = entry("a", (0.4, 0.5, 0.8, 0.6, 0.0))
    tied_second_front = (entry("b"), entry("c"))

    survivors = select_survivors((*reversed(tied_second_front), first), 3)

    assert survivors == (first, *artifact_order(tied_second_front))
    json.dumps([item.to_payload() for item in survivors], allow_nan=False)
    assert select_survivors((first,), 0) == ()
    for capacity in (-1, True, 1.5):
        with pytest.raises(ValueError, match="capacity"):
            select_survivors((first,), capacity)
