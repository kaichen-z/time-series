"""Deterministic constrained NSGA-II ranking for Numerical QD entries."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import (
    ConstraintReportV2,
    NumericalObjectiveVectorV2,
    NumericalQDEntryV2,
)


_OBJECTIVE_FIELDS = (
    "mean_capped_smae",
    "mean_capped_srmse",
    "p95_capped_srmse",
    "mean_raw_joint_error",
    "normalized_execution_cost",
)


@dataclass(frozen=True, slots=True)
class _PreparedEntry:
    artifact_sha: str
    entry: NumericalQDEntryV2
    objectives: tuple[float, ...]
    constraints: ConstraintReportV2


def _prepare_entry(entry: NumericalQDEntryV2) -> _PreparedEntry:
    if type(entry) is not NumericalQDEntryV2:
        raise TypeError("entries must be NumericalQDEntryV2 artifacts")
    if type(entry.objectives) is not NumericalObjectiveVectorV2:
        raise TypeError("entry objectives must be NumericalObjectiveVectorV2")
    if type(entry.constraints) is not ConstraintReportV2:
        raise TypeError("entry constraints must be ConstraintReportV2")

    objectives = tuple(getattr(entry.objectives, name) for name in _OBJECTIVE_FIELDS)
    if any(type(value) is not float or not math.isfinite(value) for value in objectives):
        raise ValueError("every objective must be a finite float")
    return _PreparedEntry(entry.fingerprint(), entry, objectives, entry.constraints)


def _prepare_entries(entries) -> tuple[_PreparedEntry, ...]:
    prepared = tuple(_prepare_entry(entry) for entry in entries)
    identities = tuple(item.artifact_sha for item in prepared)
    if len(identities) != len(set(identities)):
        raise ValueError("entries contain a duplicate artifact SHA")
    return tuple(sorted(prepared, key=lambda item: item.artifact_sha))


def _compare(left: _PreparedEntry, right: _PreparedEntry) -> int:
    left_feasible = left.constraints.feasible
    right_feasible = right.constraints.feasible
    if left_feasible != right_feasible:
        return -1 if left_feasible else 1

    if not left_feasible:
        left_key = (
            len(left.constraints.violations),
            tuple(sorted(left.constraints.violations)),
            left.artifact_sha,
        )
        right_key = (
            len(right.constraints.violations),
            tuple(sorted(right.constraints.violations)),
            right.artifact_sha,
        )
        return (left_key > right_key) - (left_key < right_key)

    left_dominates = (
        all(left_value <= right_value
            for left_value, right_value in zip(left.objectives, right.objectives))
        and any(left_value < right_value
                for left_value, right_value in zip(left.objectives, right.objectives))
    )
    if left_dominates:
        return -1
    right_dominates = (
        all(right_value <= left_value
            for left_value, right_value in zip(left.objectives, right.objectives))
        and any(right_value < left_value
                for left_value, right_value in zip(left.objectives, right.objectives))
    )
    return 1 if right_dominates else 0


def constraint_compare(left: NumericalQDEntryV2, right: NumericalQDEntryV2) -> int:
    """Return -1/1 when left/right constraint-dominates, otherwise zero."""

    return _compare(_prepare_entry(left), _prepare_entry(right))


def non_dominated_fronts(entries) -> tuple[tuple[NumericalQDEntryV2, ...], ...]:
    """Partition entries into deterministic ascending-rank Pareto fronts."""

    prepared = _prepare_entries(entries)
    if not prepared:
        return ()

    dominated: dict[str, list[_PreparedEntry]] = {
        item.artifact_sha: [] for item in prepared
    }
    domination_count = {item.artifact_sha: 0 for item in prepared}
    for index, left in enumerate(prepared):
        for right in prepared[index + 1:]:
            comparison = _compare(left, right)
            if comparison < 0:
                dominated[left.artifact_sha].append(right)
                domination_count[right.artifact_sha] += 1
            elif comparison > 0:
                dominated[right.artifact_sha].append(left)
                domination_count[left.artifact_sha] += 1

    current = tuple(item for item in prepared if domination_count[item.artifact_sha] == 0)
    fronts: list[tuple[NumericalQDEntryV2, ...]] = []
    while current:
        fronts.append(tuple(item.entry for item in current))
        following: list[_PreparedEntry] = []
        for item in current:
            for candidate in dominated[item.artifact_sha]:
                domination_count[candidate.artifact_sha] -= 1
                if domination_count[candidate.artifact_sha] == 0:
                    following.append(candidate)
        current = tuple(sorted(following, key=lambda item: item.artifact_sha))
    return tuple(fronts)


def crowding_distances(front) -> dict[str, float]:
    """Return internal crowding distances keyed by entry artifact SHA."""

    prepared = _prepare_entries(front)
    distances = {item.artifact_sha: 0.0 for item in prepared}
    if not prepared:
        return distances

    for objective_index in range(len(_OBJECTIVE_FIELDS)):
        ordered = sorted(
            prepared,
            key=lambda item: (item.objectives[objective_index], item.artifact_sha),
        )
        minimum = ordered[0].objectives[objective_index]
        maximum = ordered[-1].objectives[objective_index]
        if maximum == minimum:
            continue
        distances[ordered[0].artifact_sha] = math.inf
        distances[ordered[-1].artifact_sha] = math.inf
        scale = max(abs(minimum), abs(maximum))
        crosses_zero = minimum < 0.0 < maximum
        objective_range = (
            maximum / scale - minimum / scale
            if crosses_zero else maximum - minimum
        )
        for index in range(1, len(ordered) - 1):
            artifact_sha = ordered[index].artifact_sha
            if math.isinf(distances[artifact_sha]):
                continue
            previous_value = ordered[index - 1].objectives[objective_index]
            next_value = ordered[index + 1].objectives[objective_index]
            gap = (
                next_value / scale - previous_value / scale
                if crosses_zero else next_value - previous_value
            )
            distances[artifact_sha] += gap / objective_range
    return distances


def _crowding_order(front) -> tuple[NumericalQDEntryV2, ...]:
    distances = crowding_distances(front)
    return tuple(
        sorted(
            front,
            key=lambda entry: (-distances[entry.fingerprint()], entry.fingerprint()),
        )
    )


def select_survivors(entries, capacity: int) -> tuple[NumericalQDEntryV2, ...]:
    """Select capacity-bounded survivors by rank, crowding, then SHA."""

    if type(capacity) is not int or capacity < 0:
        raise ValueError("capacity must be a non-negative integer")
    fronts = non_dominated_fronts(entries)
    survivors: list[NumericalQDEntryV2] = []
    for front in fronts:
        remaining = capacity - len(survivors)
        if remaining <= 0:
            break
        survivors.extend(_crowding_order(front)[:remaining])
    return tuple(survivors)
