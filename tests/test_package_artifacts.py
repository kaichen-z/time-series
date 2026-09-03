"""Immutable caches, append-only trace artifacts, and exact resume."""
from __future__ import annotations

import json

import pytest

from evolving_loop.package_artifacts import (
    PackageArtifactError,
    PackageArtifactStore,
    PackageCacheKey,
    PackageCacheMissError,
    PackageCheckpoint,
    PackageInferenceCache,
)


_RUN = "a" * 64
_SCHEDULE = "b" * 64


def _cache_key(
    *,
    layer="numerical",
    task_sha256="1" * 64,
    candidate_sha256="2" * 64,
    dependency_fingerprints=None,
) -> PackageCacheKey:
    return PackageCacheKey(
        layer=layer,
        task_sha256=task_sha256,
        candidate_sha256=candidate_sha256,
        dependency_fingerprints=dependency_fingerprints or {"runtime": "3" * 64},
    )


def _bundle_payload(marker: str) -> dict:
    return {"schema_version": 2, "marker": marker}


def _step(generation: int, *, parent: str, accepted: str) -> dict:
    return {
        "generation": generation,
        "target": "numerical",
        "accepted": True,
        "parent_bytes_sha256": parent,
        "accepted_bytes_sha256": accepted,
        "accepted_bundle_payload": _bundle_payload(f"gen{generation}"),
    }


def _checkpoint(
    *,
    completed_steps=(),
    consumed_stages=(),
    current=None,
) -> PackageCheckpoint:
    steps = tuple(completed_steps)
    current_payload = (
        current
        if current is not None
        else (steps[-1]["accepted_bundle_payload"] if steps else _bundle_payload("seed"))
    )
    return PackageCheckpoint(
        schema_version=1,
        run_sha256=_RUN,
        schedule_sha256=_SCHEDULE,
        initial_bundle_payload=_bundle_payload("seed"),
        current_bundle_payload=current_payload,
        completed_steps=steps,
        candidate_fingerprints=(),
        cache_fingerprints=(),
        consumed_stages=tuple(consumed_stages),
    )


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def test_cache_key_changes_for_each_transitive_dependency() -> None:
    base = _cache_key()
    assert base.fingerprint != _cache_key(candidate_sha256="b" * 64).fingerprint
    assert base.fingerprint != _cache_key(task_sha256="c" * 64).fingerprint
    assert (
        base.fingerprint
        != _cache_key(dependency_fingerprints={"runtime": "d" * 64}).fingerprint
    )
    assert base.fingerprint != _cache_key(layer="retrieval").fingerprint


def test_cache_only_mismatch_fails_closed_without_callback(tmp_path) -> None:
    calls: list[str] = []
    cache = PackageInferenceCache(tmp_path, cache_only=True)
    with pytest.raises(PackageCacheMissError, match="cache-only"):
        cache.get_or_compute(_cache_key(), lambda: calls.append("called"))
    assert calls == []


def test_cache_writes_once_and_reuses_only_on_exact_key(tmp_path) -> None:
    cache = PackageInferenceCache(tmp_path, cache_only=False)
    key = _cache_key()
    value = cache.get_or_compute(key, lambda: {"forecast": [1.0, 2.0]})
    assert value == {"forecast": [1.0, 2.0]}
    # a second compute with the same key returns the cached bytes, not the new value
    reused = cache.get_or_compute(key, lambda: {"forecast": [9.9]})
    assert reused == {"forecast": [1.0, 2.0]}
    # cache-only now hits
    hit = PackageInferenceCache(tmp_path, cache_only=True).get_or_compute(
        key, lambda: pytest.fail("should not compute")
    )
    assert hit == {"forecast": [1.0, 2.0]}


# --------------------------------------------------------------------------
# checkpoint + store
# --------------------------------------------------------------------------


def test_resume_rejects_schedule_or_bundle_drift(tmp_path) -> None:
    store = PackageArtifactStore(tmp_path)
    store.write_checkpoint(_checkpoint())
    with pytest.raises(PackageArtifactError, match="schedule"):
        store.load_checkpoint(expected_run_sha256=_RUN, expected_schedule_sha256="e" * 64)
    with pytest.raises(PackageArtifactError, match="run identity"):
        store.load_checkpoint(expected_run_sha256="e" * 64, expected_schedule_sha256=_SCHEDULE)


def test_consumed_dev_stage_cannot_be_evaluated_twice(tmp_path) -> None:
    store = PackageArtifactStore(tmp_path)
    store.claim_stage("coordinate-2-dev20", candidate_sha256="c" * 64)
    store.commit_stage("coordinate-2-dev20", evaluation_sha256="d" * 64)
    with pytest.raises(PackageArtifactError, match="already consumed"):
        store.claim_stage("coordinate-2-dev20", candidate_sha256="c" * 64)
    assert store.committed_stage_evaluation("coordinate-2-dev20") == "d" * 64


def test_resume_continues_from_first_incomplete_stage(tmp_path) -> None:
    store = PackageArtifactStore(tmp_path)
    parent = "0" * 64
    accepted = "1" * 64
    step0 = _step(0, parent=parent, accepted=accepted)
    store.append_coordinate_step(step0)
    checkpoint = _checkpoint(
        completed_steps=(step0,),
        consumed_stages=("screen8",),
        current=step0["accepted_bundle_payload"],
    )
    store.write_checkpoint(checkpoint)
    resumed = store.load_checkpoint(
        expected_run_sha256=_RUN, expected_schedule_sha256=_SCHEDULE
    )
    assert resumed.next_coordinate_generation == 1
    assert resumed.next_stage == "screen32"
    assert resumed.current_bundle_payload == step0["accepted_bundle_payload"]


def test_coordinate_trace_rejects_out_of_order_or_detached_generation(tmp_path) -> None:
    store = PackageArtifactStore(tmp_path)
    store.append_coordinate_step(_step(0, parent="0" * 64, accepted="1" * 64))
    with pytest.raises(PackageArtifactError, match="next expected generation"):
        store.append_coordinate_step(_step(2, parent="1" * 64, accepted="2" * 64))
    with pytest.raises(PackageArtifactError, match="preceding Parent"):
        store.append_coordinate_step(_step(1, parent="9" * 64, accepted="2" * 64))


def test_run_manifest_and_completion_are_write_once(tmp_path) -> None:
    store = PackageArtifactStore(tmp_path)
    store.write_run_manifest({"seed": 20260903})
    store.write_run_manifest({"seed": 20260903})  # identical is idempotent
    with pytest.raises(PackageArtifactError, match="immutable artifact"):
        store.write_run_manifest({"seed": 1})
    final_sha = store.complete({"schema_version": 2, "final": True})
    assert len(final_sha) == 64
    complete = json.loads((tmp_path / "evaluation_complete.json").read_text("utf-8"))
    assert complete["status"] == "complete"
    assert complete["public_test_accessed"] is False
    with pytest.raises(PackageArtifactError, match="already complete"):
        store.complete({"schema_version": 2, "final": True})


def test_store_is_a_durable_stage_artifact_sink(tmp_path) -> None:
    store = PackageArtifactStore(tmp_path)
    payload = {"slot": 0, "bundle_sha256": "a" * 64}
    from evolving_loop.package_artifacts import _digest

    evidence_sha = _digest(payload)
    store.record_acceptance_evidence(evidence_sha, payload)
    assert store.contains_evidence(evidence_sha) is True
    store.record_candidate_evidence("numerical", 0, "f" * 64, {"slot": 0})
    assert (tmp_path / "candidate_evidence" / "numerical-0-ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff.json").exists()
