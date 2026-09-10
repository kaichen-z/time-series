from __future__ import annotations

import itertools
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from evolving_loop.v2.numerical_qd.contracts import (
    ConstraintReportV2, MorphologyCellV2, NumericalObjectiveVectorV2,
    NumericalQDEntryV2, TrainMutationFeedbackV2,
)
from evolving_loop.v2.numerical_qd.map_elites import (
    ArchiveInsertionV2, CounterRandom, NumericalQDArchive, _sample_category,
)


CELL = MorphologyCellV2("low", "none", "low", "stable", "short", "statistical")
OTHER_CELL = replace(CELL, trend="high")


def entry(marker, score=1.0, cell=CELL, violations=(), diagnostics=()):
    return NumericalQDEntryV2(
        1, marker * 64, marker * 64, cell, ("task-a",),
        NumericalObjectiveVectorV2(*(float(score),) * 5),
        ConstraintReportV2(not violations, violations), diagnostics,
    )


def feedback(categories=()):
    return TrainMutationFeedbackV2("train", "repair", True, False, False, categories)


class Draws:
    def __init__(self, *draws):
        self.draws = iter(draws)

    def randbelow(self, bound):
        expected_bound, result = next(self.draws)
        assert bound == expected_bound
        assert 0 <= result < bound
        return result


def test_counter_stream_replays_from_exact_counter_and_has_canonical_golden_values():
    first = CounterRandom(7, "parents")
    # Canonical V2 bytes end in a newline, including counter payloads.
    assert tuple(first.randbelow(10_000) for _ in range(5)) == (3780, 2360, 4241, 3758, 4440)
    assert first.to_payload() == {"seed": 7, "stream": "parents", "counter": 5}
    resumed = CounterRandom(**json.loads(json.dumps(first.to_payload())))
    assert resumed.randbelow(10_000) == first.randbelow(10_000) == 747


@pytest.mark.parametrize("args", [(True, "p"), (1, ""), (1, " "), (1, "p", True), (1, "p", -1), (1.0, "p")])
def test_counter_rejects_invalid_identity_or_counter(args):
    with pytest.raises(ValueError):
        CounterRandom(*args)


@pytest.mark.parametrize("bound", [0, -1, True, 1.0])
def test_counter_rejects_invalid_bound_without_consumption(bound):
    stream = CounterRandom(7, "p")
    with pytest.raises(ValueError):
        stream.randbelow(bound)
    assert stream.counter == 0


def test_counter_rejection_consumes_and_persists_every_hash_attempt(monkeypatch):
    # Force one out-of-range hash, then an accepted one: biased modulo gives 5.
    from evolving_loop.v2.numerical_qd import map_elites
    payloads = []
    digests = iter(((2**256 - 1).to_bytes(32, "big"), (12).to_bytes(32, "big")))

    class Digest:
        def __init__(self, payload):
            payloads.append(json.loads(payload))
        def digest(self):
            return next(digests)

    monkeypatch.setattr(map_elites.hashlib, "sha256", Digest)
    stream = CounterRandom(7, "parents")
    assert stream.randbelow(10) == 2
    assert stream.counter == 2
    assert payloads == [
        {"seed": 7, "stream": "parents", "counter": 0},
        {"seed": 7, "stream": "parents", "counter": 1},
    ]


def test_counter_supports_bounds_larger_than_one_digest():
    stream = CounterRandom(7, "large")
    value = stream.randbelow(2**300)
    assert 0 <= value < 2**300
    assert stream.counter == 2
    assert value == CounterRandom(7, "large").randbelow(2**300)


def test_insert_places_genome_in_multiple_cells_with_immutable_append_history():
    archive = NumericalQDArchive(capacity=1)
    before = archive.to_payload()
    parent_sha = archive.fingerprint()
    candidate = entry("a")
    other = replace(candidate, cell=OTHER_CELL)
    parent = archive
    archive = parent.insert((candidate, other))
    assert parent.to_payload() == before
    assert parent.fingerprint() == parent_sha
    result = archive.insertion_log[0]
    assert isinstance(result, ArchiveInsertionV2)
    assert result.entry_sha256s == tuple(sorted((candidate.fingerprint(), other.fingerprint())))
    assert len(archive.cells) == 2
    assert len(archive.entries) == 2
    assert archive.snapshot_parent_sha256 == parent_sha
    assert before["entries"] == {}
    history = archive.insertion_log
    with pytest.raises(FrozenInstanceError):
        result.entry_sha256s = ()
    with pytest.raises(TypeError):
        archive.entries[candidate.fingerprint()] = other
    with pytest.raises(FrozenInstanceError):
        archive.cells[0].visit_count = 100
    with pytest.raises(FrozenInstanceError):
        archive.snapshot_parent_sha256 = None
    archive = archive.insert((entry("b", score=0.5),))
    assert archive.insertion_log[:1] == history
    assert len(history) == 1
    assert candidate.fingerprint() in archive.entries


def test_insert_is_idempotent_and_batch_order_invariant():
    candidates = (entry("a"), entry("b", 0.5), entry("c", cell=OTHER_CELL))
    snapshots = []
    for order in itertools.permutations(candidates):
        archive = NumericalQDArchive(capacity=2)
        archive = archive.insert(order)
        snapshots.append(archive.canonical_bytes())
        previous = archive.fingerprint()
        assert archive.insert((candidates[0], candidates[0])) is archive
        assert archive.fingerprint() == previous
        assert len(archive.insertion_log) == 1
    assert len(set(snapshots)) == 1


def test_conflicting_same_sha_is_rejected_atomically(monkeypatch):
    archive = NumericalQDArchive()
    a, b = entry("a"), entry("b")
    archive = archive.insert((a,))
    before = archive.canonical_bytes()
    original = NumericalQDEntryV2.fingerprint
    monkeypatch.setattr(NumericalQDEntryV2, "fingerprint", lambda self: original(a))
    with pytest.raises(ValueError, match="conflict"):
        archive.insert((b,))
    assert archive.canonical_bytes() == before


@pytest.mark.parametrize("capacity", [0, 5, True, 1.5, -1])
def test_archive_capacity_must_be_integer_one_to_four(capacity):
    with pytest.raises(ValueError):
        NumericalQDArchive(capacity=capacity)


def test_constrained_survival_caps_cells_but_retains_nonelite_stepping_stones():
    archive = NumericalQDArchive(capacity=4)
    feasible = tuple(entry(marker, score) for marker, score in zip("abcde", (1, 2, 3, 4, 5)))
    failed = entry("f", 0.0, violations=("coverage",), diagnostics=("coverage",))
    archive = archive.insert((*reversed(feasible), failed))
    assert set(archive.cells[0].entry_sha256s) == {e.fingerprint() for e in feasible[:4]}
    assert archive.cells[0].visit_count == 6
    assert len(archive.entries) == 6
    pools = archive.parent_categories(feedback(("coverage",)))
    assert pools["elite"] == (feasible[0],)
    assert pools["failure_matched"] == (failed,)
    assert set(pools["stepping_stone"]) == set((*feasible[1:], failed))


def test_underexplored_uses_lowest_cell_visit_count_and_candidate_sha_order():
    archive = NumericalQDArchive(capacity=2)
    a, b, c = entry("a"), entry("b", 2.0), entry("c", cell=OTHER_CELL)
    archive = archive.insert((a, b, c))
    pools = archive.parent_categories(feedback())
    assert pools["underexplored"] == (c,)
    assert pools["elite"] == tuple(sorted((a, c), key=lambda e: e.fingerprint()))
    assert pools["failure_matched"] == ()
    assert pools["stepping_stone"] == (b,)
    assert archive.sample_parent(feedback(), Draws((80, 0), (1, 0))) == c
    assert archive.sample_parent(feedback(), Draws((80, 79), (1, 0))) == b


@pytest.mark.parametrize("draw", range(100))
def test_parent_category_exact_master_weight_boundaries(draw):
    pools = dict(zip(("underexplored", "elite", "failure_matched", "stepping_stone"), (("a",), ("b",), ("c",), ("d",))))
    expected = "underexplored" if draw < 40 else "elite" if draw < 70 else "failure_matched" if draw < 90 else "stepping_stone"
    assert _sample_category(pools, Draws((100, draw))) == expected


@pytest.mark.parametrize("present", itertools.product((False, True), repeat=4))
def test_every_empty_category_combination_renormalizes_exact_integer_mass(present):
    categories = ("underexplored", "elite", "failure_matched", "stepping_stone")
    pools = {name: (name,) if enabled else () for name, enabled in zip(categories, present)}
    expected = [name for name, weight, enabled in zip(categories, (40, 30, 20, 10), present) if enabled for _ in range(weight)]
    if not expected:
        with pytest.raises(ValueError, match="empty"):
            _sample_category(pools, Draws())
        return
    assert [_sample_category(pools, Draws((len(expected), draw))) for draw in range(len(expected))] == expected


def test_sampling_and_resume_are_invariant_to_entry_order_and_do_not_mutate_snapshot():
    entries = (entry("a"), entry("b", 2.0), entry("c", cell=OTHER_CELL))
    archive = NumericalQDArchive(capacity=2)
    archive = archive.insert(entries)
    restored = NumericalQDArchive.from_payload(json.loads(archive.canonical_bytes()))
    assert restored.canonical_bytes() == archive.canonical_bytes()
    original_sha = archive.fingerprint()
    random = CounterRandom(3, "parents")
    archive.sample_parent(feedback(), random)
    resumed = CounterRandom(**random.to_payload())
    assert [archive.sample_parent(feedback(), random) for _ in range(20)] == [restored.sample_parent(feedback(), resumed) for _ in range(20)]
    assert archive.fingerprint() == original_sha
    restored_payload = restored.to_payload()
    restored_payload["entries"].clear()
    restored_payload["cells"][0]["entry_sha256s"].clear()
    assert restored.fingerprint() == original_sha


def test_snapshot_restore_rejects_unknown_fields_bad_identity_and_corrupt_references():
    archive = NumericalQDArchive()
    archive = archive.insert((entry("a"),))
    payload = archive.to_payload()
    payload["dev_metrics"] = {}
    with pytest.raises(ValueError):
        NumericalQDArchive.from_payload(payload)
    payload = archive.to_payload()
    payload["entries"][entry("a").fingerprint()]["genome_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="SHA|identity"):
        NumericalQDArchive.from_payload(payload)
    payload = archive.to_payload()
    payload["cells"][0]["entry_sha256s"] = ["f" * 64]
    with pytest.raises(ValueError):
        NumericalQDArchive.from_payload(payload)
    payload = archive.to_payload()
    payload["insertion_log"] = []
    with pytest.raises(ValueError):
        NumericalQDArchive.from_payload(payload)
    payload = archive.to_payload()
    payload["snapshot_parent_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="parent SHA"):
        NumericalQDArchive.from_payload(payload)
    payload = archive.to_payload()
    payload["cells"][0]["visit_count"] = 100
    with pytest.raises(ValueError):
        NumericalQDArchive.from_payload(payload)


def test_public_sampling_uses_each_category_and_sorted_member_draw():
    archive = NumericalQDArchive(capacity=2)
    a = entry("a", score=1.0)
    b = entry("b", score=2.0, diagnostics=("coverage",))
    c = entry("c", cell=OTHER_CELL)
    archive = archive.insert((a, b, c))
    categories = feedback(("coverage",))
    # Unique underexplored c, sorted elites a/c, failure specialist b, step b.
    elite_order = sorted((a, c), key=lambda e: e.fingerprint())
    for draw in range(100):
        if draw < 40:
            expected, count = c, 1
        elif draw < 70:
            expected, count = elite_order[1], 2
        else:
            expected, count = b, 1
        assert archive.sample_parent(categories, Draws((100, draw), (count, count - 1))) == expected


def test_incremental_history_replays_identical_survivors_and_parent_chain():
    archive = NumericalQDArchive(capacity=1)
    for candidate in (entry("a", 3), entry("b", 2), entry("c", 1)):
        parent_sha = archive.fingerprint()
        archive = archive.insert((candidate,))
        assert archive.snapshot_parent_sha256 == parent_sha
        restored = NumericalQDArchive.from_payload(archive.to_payload())
        assert restored.fingerprint() == archive.fingerprint()
    assert archive.cells[0].entry_sha256s == (candidate.fingerprint(),)
    assert archive.cells[0].visit_count == 3
    assert len(archive.insertion_log) == 3


def test_empty_archive_and_unsanitized_feedback_are_rejected_before_drawing():
    with pytest.raises(ValueError, match="empty"):
        NumericalQDArchive().sample_parent(feedback(), Draws())
    archive = NumericalQDArchive()
    archive = archive.insert((entry("a"),))
    with pytest.raises((TypeError, ValueError)):
        archive.sample_parent({"split": "dev", "diagnostic_categories": ["coverage"]}, Draws())
