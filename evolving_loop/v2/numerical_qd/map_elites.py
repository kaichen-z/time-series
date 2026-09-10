"""Immutable bounded niche snapshots and reproducible SHA-256 parent draws."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..contracts import canonical_v2_bytes, fingerprint_payload, require_sha256
from .contracts import (
    MorphologyCellV2, NumericalQDEntryV2, TrainMutationFeedbackV2,
    _CanonicalContract, _nested, _nonnegative_int, _schema_version,
    _sequence, _sorted_strings,
)
from .nsga2 import non_dominated_fronts, select_survivors


_CATEGORY_WEIGHTS = (
    ("underexplored", 40), ("elite", 30),
    ("failure_matched", 20), ("stepping_stone", 10),
)


class CounterRandom:
    """Counter counts hash blocks, including rejected attempts, not draws.

    Every block hashes canonical {seed, stream, counter} JSON. Full 256-bit
    blocks are concatenated for larger bounds; rejecting the incomplete final
    interval prevents modulo bias. No process-global RNG state is consulted.
    """

    __slots__ = ("_seed", "_stream", "_counter")

    def __init__(self, seed: int, stream: str, counter: int = 0):
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if type(stream) is not str or not stream.strip():
            raise ValueError("stream must be a non-empty string")
        _nonnegative_int(counter, "counter")
        self._seed, self._stream, self._counter = seed, stream, counter

    @property
    def seed(self):
        return self._seed

    @property
    def stream(self):
        return self._stream

    @property
    def counter(self):
        return self._counter

    def to_payload(self):
        return {"seed": self.seed, "stream": self.stream, "counter": self.counter}

    def randbelow(self, bound: int) -> int:
        if type(bound) is not int or bound <= 0:
            raise ValueError("bound must be a positive integer")
        blocks = max(1, ((bound - 1).bit_length() + 255) // 256)
        space = 1 << (256 * blocks)
        limit = space - space % bound
        while True:
            value = 0
            for _ in range(blocks):
                digest = hashlib.sha256(canonical_v2_bytes(self.to_payload())).digest()
                self._counter += 1
                value = (value << 256) | int.from_bytes(digest, "big")
            if value < limit:
                return value % bound


@dataclass(frozen=True, slots=True)
class ArchiveInsertionV2(_CanonicalContract):
    """One canonical batch appended to the retained entry history."""

    schema_version: int
    entry_sha256s: tuple[str, ...]

    def __post_init__(self):
        _schema_version(self.schema_version)
        object.__setattr__(self, "entry_sha256s", _sorted_strings(
            self.entry_sha256s, "entry_sha256s", sha=True, nonempty=True,
        ))


@dataclass(frozen=True, slots=True)
class _CellRecord(_CanonicalContract):
    cell: MorphologyCellV2
    visit_count: int
    entry_sha256s: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "cell", _nested(self.cell, MorphologyCellV2))
        _nonnegative_int(self.visit_count, "visit_count")
        object.__setattr__(self, "entry_sha256s", _sorted_strings(
            self.entry_sha256s, "entry_sha256s", sha=True, nonempty=True,
        ))


def _advance_cells(cells, batch, entries, capacity):
    previous = {record.cell.fingerprint(): record for record in cells}
    added = {}
    for sha in batch.entry_sha256s:
        candidate = entries[sha]
        added.setdefault(candidate.cell.fingerprint(), []).append(candidate)
    for cell_sha, candidates in added.items():
        old = previous.get(cell_sha)
        incumbents = tuple(entries[sha] for sha in old.entry_sha256s) if old else ()
        survivors = select_survivors((*incumbents, *candidates), capacity)
        previous[cell_sha] = _CellRecord(
            candidates[0].cell,
            (old.visit_count if old else 0) + len(candidates),
            tuple(sorted(candidate.fingerprint() for candidate in survivors)),
        )
    return tuple(previous[sha] for sha in sorted(previous))


def _snapshot_payload(capacity, cells, entries, log, parent):
    return {
        "schema_version": 1,
        "capacity": capacity,
        "cells": [cell.to_payload() for cell in cells],
        "entries": {sha: entries[sha].to_payload() for sha in sorted(entries)},
        "insertion_log": [item.to_payload() for item in log],
        "snapshot_parent_sha256": parent,
    }


def _sample_category(pools, random_stream):
    available = tuple((name, weight) for name, weight in _CATEGORY_WEIGHTS if pools[name])
    total = sum(weight for _, weight in available)
    if not total:
        raise ValueError("cannot sample an empty archive")
    draw = random_stream.randbelow(total)
    for name, weight in available:
        if draw < weight:
            return name
        draw -= weight
    raise ValueError("random stream returned an out-of-range draw")


@dataclass(frozen=True, slots=True)
class NumericalQDArchive(_CanonicalContract):
    """A content-addressed snapshot; insert returns a new immutable snapshot.

    Visits count unique evaluated entries admitted to a cell's append history.
    Sampling is pure with respect to the archive. All historical entries remain
    available as lineage stepping stones when they cease to be rank-zero elites;
    the controller is responsible for supplying lineage-compatible history.
    """

    capacity: int = 4
    schema_version: int = 1
    cells: tuple[_CellRecord, ...] = ()
    entries: Mapping[str, NumericalQDEntryV2] = field(default_factory=dict)
    insertion_log: tuple[ArchiveInsertionV2, ...] = ()
    snapshot_parent_sha256: str | None = None

    def __post_init__(self):
        _schema_version(self.schema_version)
        if type(self.capacity) is not int or not 1 <= self.capacity <= 4:
            raise ValueError("capacity must be an integer between 1 and 4")
        if not isinstance(self.entries, Mapping):
            raise ValueError("entries must be an immutable entry table")
        entries = {}
        for sha, candidate in self.entries.items():
            require_sha256(sha, "entry SHA")
            candidate = _nested(candidate, NumericalQDEntryV2)
            # Reparse even supplied instances: do not trust bypassed dataclasses.
            candidate = NumericalQDEntryV2.from_payload(candidate.to_payload())
            if candidate.fingerprint() != sha:
                raise ValueError("entry SHA identity conflicts with payload")
            entries[sha] = candidate
        cells = tuple(_nested(value, _CellRecord) for value in _sequence(self.cells, "cells"))
        cell_shas = tuple(record.cell.fingerprint() for record in cells)
        _sorted_strings(cell_shas, "cells", sha=True)
        log = tuple(_nested(value, ArchiveInsertionV2) for value in _sequence(self.insertion_log, "insertion_log"))
        if self.snapshot_parent_sha256 is not None:
            require_sha256(self.snapshot_parent_sha256, "snapshot_parent_sha256")

        # Replay authenticates references, visit counts, survivor membership,
        # complete history and the immediate parent identity before publication.
        replay_entries, replay_cells, replay_log, parent = {}, (), (), None
        for batch in log:
            previous_sha = fingerprint_payload(_snapshot_payload(
                self.capacity, replay_cells, replay_entries, replay_log, parent,
            ))
            for sha in batch.entry_sha256s:
                if sha not in entries or sha in replay_entries:
                    raise ValueError("insertion log has missing or duplicate entry SHA")
                replay_entries[sha] = entries[sha]
            replay_cells = _advance_cells(replay_cells, batch, entries, self.capacity)
            replay_log += (batch,)
            parent = previous_sha
        if set(replay_entries) != set(entries):
            raise ValueError("insertion log must account for every retained entry")
        if replay_cells != cells:
            raise ValueError("cells disagree with insertion history")
        if self.snapshot_parent_sha256 != parent:
            raise ValueError("snapshot parent SHA disagrees with insertion history")
        object.__setattr__(self, "entries", MappingProxyType(dict(sorted(entries.items()))))
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "insertion_log", log)

    def insert(self, entries) -> NumericalQDArchive:
        retained = dict(self.entries)
        added = set()
        for candidate in entries:
            if type(candidate) is not NumericalQDEntryV2:
                raise TypeError("entries must be NumericalQDEntryV2 artifacts")
            sha = candidate.fingerprint()
            if sha in retained:
                if retained[sha].canonical_bytes() != candidate.canonical_bytes():
                    raise ValueError("conflicting payload for the same entry SHA")
                continue
            retained[sha] = candidate
            added.add(sha)
        if not added:
            return self
        batch = ArchiveInsertionV2(1, tuple(sorted(added)))
        return NumericalQDArchive(
            capacity=self.capacity,
            cells=_advance_cells(self.cells, batch, retained, self.capacity),
            entries=retained,
            insertion_log=(*self.insertion_log, batch),
            snapshot_parent_sha256=self.fingerprint(),
        )

    def parent_categories(self, feedback: TrainMutationFeedbackV2):
        if type(feedback) is not TrainMutationFeedbackV2:
            raise TypeError("feedback must be sanitized TrainMutationFeedbackV2")
        feedback = TrainMutationFeedbackV2.from_payload(feedback.to_payload())
        underexplored, elites = set(), set()
        minimum_visits = min((cell.visit_count for cell in self.cells), default=0)
        for cell in self.cells:
            survivors = tuple(self.entries[sha] for sha in cell.entry_sha256s)
            if cell.visit_count == minimum_visits:
                underexplored.update(cell.entry_sha256s)
            elites.update(entry.fingerprint() for entry in non_dominated_fronts(survivors)[0])
        diagnostics = set(feedback.diagnostic_categories)
        matched = {
            sha for sha, entry in self.entries.items()
            if diagnostics.intersection(entry.train_diagnostic_categories)
        }
        pools = {
            "underexplored": underexplored,
            "elite": elites,
            "failure_matched": matched,
            "stepping_stone": set(self.entries) - elites,
        }
        return MappingProxyType({
            name: tuple(self.entries[sha] for sha in sorted(shas))
            for name, shas in pools.items()
        })

    def sample_parent(self, feedback: TrainMutationFeedbackV2, random_stream) -> NumericalQDEntryV2:
        pools = self.parent_categories(feedback)
        category = _sample_category(pools, random_stream)
        candidates = pools[category]
        return candidates[random_stream.randbelow(len(candidates))]
