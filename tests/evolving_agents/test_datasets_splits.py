"""Series-level splitting: one entity never straddles two splits, and membership is deterministic."""

from __future__ import annotations

import json

import pytest
from dr_cik.models import ForecastTask

from evolving_agents.harness.datasets import (
    DrCikSplits,
    SPLIT_NAMES,
    assign_entities,
    load_drcik_splits,
    write_split_file,
)

from .conftest import SAMPLE_DIR, requires_sample


def _task(index: int, entity: str) -> ForecastTask:
    return ForecastTask(
        benchmark_id=f"task_{index}",
        entity_name=entity,
        target_name="t",
        target_description="d",
        frequency="H",
        prediction_length=2,
        seasonal_period=None,
        history_timestamps=("a", "b"),
        history_values=(1.0, 2.0),
        future_timestamps=("c", "d"),
        future_values=(3.0, 4.0),
        documents=(),
        gt_evidence=(),
        labels_public=True,
    )


# 30 tasks across 10 entities, deliberately uneven: one entity owns 8 of them.
TASKS = (
    [_task(index, "Big") for index in range(8)]
    + [_task(100 + index, f"Entity{index % 9}") for index in range(22)]
)


def test_no_entity_straddles_two_splits() -> None:
    assignment = assign_entities(TASKS, seed=7)
    by_id = {task.benchmark_id: task for task in TASKS}
    entity_to_splits: dict[str, set[str]] = {}
    for split_name, ids in assignment.items():
        for benchmark_id in ids:
            entity_to_splits.setdefault(by_id[benchmark_id].entity_name, set()).add(split_name)
    straddling = {entity: splits for entity, splits in entity_to_splits.items() if len(splits) > 1}
    assert not straddling, f"entities appear in multiple splits: {straddling}"


def test_every_task_lands_in_exactly_one_split() -> None:
    assignment = assign_entities(TASKS, seed=7)
    placed = [benchmark_id for ids in assignment.values() for benchmark_id in ids]
    assert sorted(placed) == sorted(task.benchmark_id for task in TASKS)
    assert len(placed) == len(set(placed))


def test_assignment_is_deterministic_for_a_seed() -> None:
    assert assign_entities(TASKS, seed=7) == assign_entities(TASKS, seed=7)


def test_a_different_seed_gives_a_different_assignment() -> None:
    assert assign_entities(TASKS, seed=7) != assign_entities(TASKS, seed=99)


def test_task_counts_land_near_the_target_fractions() -> None:
    assignment = assign_entities(TASKS, seed=7, fractions=(0.6, 0.2, 0.2))
    total = len(TASKS)
    assert len(assignment["evolve"]) / total > 0.4  # the 8-task entity makes exactness impossible
    assert all(assignment[name] for name in SPLIT_NAMES)


def test_split_file_round_trips(tmp_path) -> None:
    assignment = assign_entities(TASKS, seed=7)
    path = write_split_file(assignment, tmp_path / "splits.json")
    assert json.loads(path.read_text(encoding="utf-8")) == assignment


def test_named_rejects_an_unknown_split() -> None:
    splits = DrCikSplits(evolve=[], dev=[], test=[])
    with pytest.raises(ValueError):
        splits.named("train")


@requires_sample
def test_sample_dir_splits_are_disjoint_and_stable(tmp_path) -> None:
    split_file = tmp_path / "splits.json"
    first = load_drcik_splits(sample_dir=SAMPLE_DIR, seed=7, split_file=split_file)
    second = load_drcik_splits(sample_dir=SAMPLE_DIR, seed=7, split_file=split_file)

    ids = [{task.benchmark_id for task in split} for split in (first.evolve, first.dev, first.test)]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert {task.benchmark_id for task in second.evolve} == ids[0]
    assert sum(len(group) for group in ids) > 0


@requires_sample
def test_loaded_tasks_all_have_public_labels() -> None:
    splits = load_drcik_splits(sample_dir=SAMPLE_DIR, seed=7, split_file=None)
    for split in (splits.evolve, splits.dev, splits.test):
        assert all(task.labels_public for task in split)
