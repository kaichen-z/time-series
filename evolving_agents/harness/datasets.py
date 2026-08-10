"""Loads Dr-CiK tasks and splits them by entity, so one series never straddles two splits."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

from dr_cik.data import load_sample_tasks, load_tasks
from dr_cik.models import ForecastTask

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("evolve", "dev", "test")
DEFAULT_SPLIT_FILE = Path(__file__).parent / "drcik_splits_seed7.json"
DEFAULT_FRACTIONS = (0.6, 0.2, 0.2)


@dataclass(frozen=True)
class DrCikSplits:
    """The three disjoint task sets evolution, model selection, and final eval may each touch."""

    evolve: list[ForecastTask]
    dev: list[ForecastTask]
    test: list[ForecastTask]

    def named(self, name: str) -> list[ForecastTask]:
        """Return one split by name."""
        if name not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {name!r}, expected one of {SPLIT_NAMES}")
        return getattr(self, name)


def load_labeled_tasks(data_dir: str | Path | None = None, sample_dir: str | Path | None = None) -> list[ForecastTask]:
    """Load every task that carries public labels, from either the full dataset or the sample dir."""
    if sample_dir:
        return [task for task in load_sample_tasks(sample_dir) if task.labels_public]
    if not data_dir:
        raise ValueError("one of data_dir or sample_dir is required")
    return load_tasks(data_dir=data_dir, labels_public=True)


def assign_entities(
    tasks: list[ForecastTask], seed: int = 7, fractions: tuple[float, float, float] = DEFAULT_FRACTIONS
) -> dict[str, list[str]]:
    """Group tasks by entity, then bin-pack whole entities to hit the target task-count fractions.

    Dr-CiK reuses one underlying series across several differently-worded tasks, so splitting at
    task level would put the same series in both evolve and test; entities are therefore atomic.
    Packing by task count (not entity count) matters because entity sizes are very uneven.
    """
    by_entity: dict[str, list[str]] = {}
    for task in tasks:
        by_entity.setdefault(task.entity_name, []).append(task.benchmark_id)

    entities = sorted(by_entity)
    random.Random(seed).shuffle(entities)
    total = len(tasks)
    targets = [fraction * total for fraction in fractions]
    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}

    entity_of: dict[str, str] = {}
    for entity in entities:
        members = sorted(by_entity[entity])
        deficits = [targets[index] - len(assigned[name]) for index, name in enumerate(SPLIT_NAMES)]
        chosen = SPLIT_NAMES[deficits.index(max(deficits))]
        assigned[chosen].extend(members)
        for benchmark_id in members:
            entity_of[benchmark_id] = entity

    _fill_empty_splits(assigned, by_entity, entity_of)
    return {name: sorted(ids) for name, ids in assigned.items()}


def _fill_empty_splits(assigned: dict[str, list[str]], by_entity: dict[str, list[str]], entity_of: dict[str, str]) -> None:
    """Move one whole entity into any split greedy packing left empty.

    With few entities the deficit rule can starve a split entirely, and an empty test split would
    silently turn `final_eval` into a no-op. Entities stay atomic, so this cannot leak a series.
    """
    for name in SPLIT_NAMES:
        if assigned[name] or len(by_entity) < len(SPLIT_NAMES):
            continue
        donor = max(SPLIT_NAMES, key=lambda other: len({entity_of[bid] for bid in assigned[other]}))
        donor_entities = {entity_of[benchmark_id] for benchmark_id in assigned[donor]}
        if len(donor_entities) < 2:
            continue
        moved = min(donor_entities, key=lambda entity: (len(by_entity[entity]), entity))
        assigned[donor] = [bid for bid in assigned[donor] if entity_of[bid] != moved]
        assigned[name].extend(by_entity[moved])
        logger.info("split %r was empty; moved entity %r from %r to fill it", name, moved, donor)


def write_split_file(assignment: dict[str, list[str]], path: str | Path) -> Path:
    """Persist a split assignment atomically so every loop reads identical membership."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp = resolved.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(assignment, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, resolved)
    return resolved


def load_drcik_splits(
    data_dir: str | Path | None = None,
    sample_dir: str | Path | None = None,
    seed: int = 7,
    split_file: str | Path | None = DEFAULT_SPLIT_FILE,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> DrCikSplits:
    """Return the evolve/dev/test splits, reusing the committed membership file when it exists."""
    tasks = load_labeled_tasks(data_dir=data_dir, sample_dir=sample_dir)
    by_id = {task.benchmark_id: task for task in tasks}

    path = Path(split_file).expanduser() if split_file else None
    if path is not None and path.is_file():
        assignment = json.loads(path.read_text(encoding="utf-8"))
        known = {benchmark_id for ids in assignment.values() for benchmark_id in ids}
        missing = set(by_id) - known
        if missing:
            # The sample dir is a strict subset of the full dataset, so a committed split can
            # legitimately not mention every loaded task; recompute rather than silently drop them.
            logger.info("split file %s covers %d/%d loaded tasks, recomputing", path, len(known & set(by_id)), len(by_id))
            assignment = assign_entities(tasks, seed=seed, fractions=fractions)
    else:
        assignment = assign_entities(tasks, seed=seed, fractions=fractions)
        # Only the full dataset may author the canonical file: the sample dir holds a handful of
        # tasks, and letting it write here would ship a split that silently covers almost nothing.
        if path is not None and data_dir and not sample_dir:
            write_split_file(assignment, path)
            logger.info("wrote split membership to %s", path)

    splits = DrCikSplits(
        evolve=[by_id[bid] for bid in assignment["evolve"] if bid in by_id],
        dev=[by_id[bid] for bid in assignment["dev"] if bid in by_id],
        test=[by_id[bid] for bid in assignment["test"] if bid in by_id],
    )
    logger.info("splits: evolve=%d dev=%d test=%d", len(splits.evolve), len(splits.dev), len(splits.test))
    return splits
