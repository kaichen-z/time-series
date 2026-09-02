"""Train-only grouping and evolution helpers for task-local Numerical ensembles."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from common.data import Task as DataTask
from common.payload import canonical_json_bytes

from .execution import Task as RuntimeTask
from .screening import TaskProfile, profile_task


_GROUPING_SCHEMA = 1
_GROUPING_IMPLEMENTATION = hashlib.sha256(
    b"task-local-group-folds-v1:entity+history+source:balanced-morphology"
).hexdigest()


def _canonical_identity(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class GroupFoldManifest:
    """Immutable assignment of indivisible history-only task groups to folds."""

    schema_version: int
    seed: int
    fold_count: int
    groups: tuple[tuple[str, tuple[str, ...], int], ...]
    grouping_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != _GROUPING_SCHEMA:
            raise ValueError("group fold manifest schema must be exactly one")
        if type(self.seed) is not int:
            raise ValueError("group fold seed must be an exact integer")
        if type(self.fold_count) is not int or self.fold_count < 2:
            raise ValueError("group fold count must be at least two")
        if type(self.groups) is not tuple or not self.groups:
            raise ValueError("group fold manifest requires groups")
        seen_tasks: set[str] = set()
        seen_groups: set[str] = set()
        previous = ""
        for item in self.groups:
            if type(item) is not tuple or len(item) != 3:
                raise ValueError("group rows must be exact triples")
            group_sha, task_ids, fold = item
            if (
                type(group_sha) is not str
                or len(group_sha) != 64
                or group_sha <= previous
                or group_sha in seen_groups
            ):
                raise ValueError("group identities must be unique sorted SHA-256 values")
            if (
                type(task_ids) is not tuple
                or not task_ids
                or tuple(sorted(task_ids)) != task_ids
                or any(type(task_id) is not str or not task_id for task_id in task_ids)
                or seen_tasks.intersection(task_ids)
            ):
                raise ValueError("group task memberships must be unique sorted identifiers")
            if type(fold) is not int or not 0 <= fold < self.fold_count:
                raise ValueError("group fold is outside the registered fold universe")
            previous = group_sha
            seen_groups.add(group_sha)
            seen_tasks.update(task_ids)
        if set(range(self.fold_count)) != {fold for _sha, _tasks, fold in self.groups}:
            raise ValueError("every registered fold must contain at least one group")
        if self.grouping_fingerprint != _GROUPING_IMPLEMENTATION:
            raise ValueError("grouping implementation fingerprint mismatch")

    @property
    def task_fold_map(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                task_id: fold
                for _group_sha, task_ids, fold in self.groups
                for task_id in task_ids
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "fold_count": self.fold_count,
            "groups": [
                {"group_sha256": group_sha, "task_ids": list(task_ids), "fold": fold}
                for group_sha, task_ids, fold in self.groups
            ],
            "grouping_fingerprint": self.grouping_fingerprint,
        }


class _DisjointSet:
    def __init__(self, task_ids: Sequence[str]) -> None:
        self.parent = {task_id: task_id for task_id in task_ids}

    def find(self, task_id: str) -> str:
        parent = self.parent[task_id]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while task_id != parent:
            next_id = self.parent[task_id]
            self.parent[task_id] = parent
            task_id = next_id
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _history_fingerprint(task: DataTask) -> str:
    return _sha256({"history": list(task.history_values)})


def _task_profile(task: DataTask) -> TaskProfile:
    return profile_task(
        RuntimeTask(
            task.task_id,
            tuple(task.history_values),
            task.prediction_length,
            task.frequency,
            (),
        )
    )


def task_morphology_key(profile: TaskProfile) -> str:
    """Return a stable task-identity-free morphology bucket."""
    if type(profile) is not TaskProfile:
        raise TypeError("morphology grouping requires an exact TaskProfile")
    history_bucket = (
        "short" if profile.history_length < 64 else
        "medium" if profile.history_length < 256 else
        "long"
    )
    ratio = profile.horizon / profile.history_length
    horizon_bucket = "short" if ratio <= 0.1 else "medium" if ratio <= 0.3 else "long"
    trend = (
        profile.trend_direction
        if profile.trend_strength >= 0.35 and profile.trend_direction != "flat"
        else "flat"
    )
    payload = {
        "frequency": _canonical_identity(profile.frequency),
        "history": history_bucket,
        "horizon": horizon_bucket,
        "trend": trend,
        "periodic": bool(
            profile.periodicity_periods and profile.periodicity_confidence >= 0.5
        ),
        "intermittent": bool(
            profile.zero_fraction >= 0.5 or profile.intermittency_adi >= 1.32
        ),
        "recent_regime": bool(
            profile.recent_regime_start is not None
            and profile.recent_regime_confidence >= 0.5
        ),
        "signed": profile.signed,
    }
    return _sha256(payload)


def build_group_fold_manifest(
    tasks: Sequence[DataTask],
    *,
    seed: int,
    fold_count: int = 5,
    source_series_ids: Mapping[str, str] | None = None,
) -> GroupFoldManifest:
    """Assign history-only transitive task groups to deterministic balanced folds."""
    if type(seed) is not int:
        raise ValueError("group fold seed must be an exact integer")
    if type(fold_count) is not int or fold_count < 2:
        raise ValueError("group fold count must be at least two")
    supplied = tuple(tasks)
    if not supplied or any(type(task) is not DataTask for task in supplied):
        raise ValueError("group folds require exact numeric DataTask records")
    task_ids = tuple(task.task_id for task in supplied)
    if (
        any(type(task_id) is not str or not task_id for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
    ):
        raise ValueError("group folds require unique nonempty task IDs")
    for task in supplied:
        if (
            not task.history_values
            or any(not math.isfinite(float(value)) for value in task.history_values)
            or type(task.entity_name) is not str
            or not _canonical_identity(task.entity_name)
        ):
            raise ValueError("group folds require finite history and nonempty entity")
    source_map = dict(source_series_ids or {})
    if set(source_map) - set(task_ids) or any(
        type(value) is not str or not _canonical_identity(value)
        for value in source_map.values()
    ):
        raise ValueError("source-series identities must bind known tasks to nonempty strings")

    ordered = tuple(sorted(supplied, key=lambda task: task.task_id))
    disjoint = _DisjointSet(task_ids)
    indexes: tuple[dict[str, str], ...] = ({}, {}, {})
    for task in ordered:
        keys = (
            _canonical_identity(task.entity_name),
            _history_fingerprint(task),
            _canonical_identity(source_map[task.task_id]) if task.task_id in source_map else "",
        )
        for index, key in zip(indexes, keys, strict=True):
            if not key:
                continue
            prior = index.get(key)
            if prior is None:
                index[key] = task.task_id
            else:
                disjoint.union(prior, task.task_id)

    members: dict[str, list[str]] = defaultdict(list)
    by_id = {task.task_id: task for task in ordered}
    for task_id in task_ids:
        members[disjoint.find(task_id)].append(task_id)
    grouped: list[tuple[str, tuple[str, ...], str]] = []
    for task_group in members.values():
        member_ids = tuple(sorted(task_group))
        group_sha = _sha256({"task_ids": list(member_ids)})
        strata = sorted({task_morphology_key(_task_profile(by_id[item])) for item in member_ids})
        grouped.append((group_sha, member_ids, strata[0]))
    if len(grouped) < fold_count:
        raise ValueError("indivisible task groups cannot populate every requested fold")

    fold_sizes = [0] * fold_count
    stratum_counts: list[dict[str, int]] = [defaultdict(int) for _ in range(fold_count)]
    assignments: list[tuple[str, tuple[str, ...], int]] = []
    ranked_groups = sorted(
        grouped,
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(f"{seed}\0{item[0]}".encode("utf-8")).hexdigest(),
        ),
    )
    for group_sha, member_ids, stratum in ranked_groups:
        fold = min(
            range(fold_count),
            key=lambda value: (
                fold_sizes[value],
                stratum_counts[value][stratum],
                value,
            ),
        )
        fold_sizes[fold] += len(member_ids)
        stratum_counts[fold][stratum] += len(member_ids)
        assignments.append((group_sha, member_ids, fold))
    assignments.sort(key=lambda item: item[0])
    return GroupFoldManifest(
        schema_version=_GROUPING_SCHEMA,
        seed=seed,
        fold_count=fold_count,
        groups=tuple(assignments),
        grouping_fingerprint=_GROUPING_IMPLEMENTATION,
    )
