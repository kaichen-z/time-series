from __future__ import annotations

import math

import pytest

from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.store import StoreContractError, V2RunStore, write_atomic_json


def sha256_for(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_v2_store_creates_only_the_versioned_layout(tmp_path):
    store = V2RunStore.create(tmp_path / "run")
    assert {
        path.relative_to(store.root).as_posix()
        for path in store.root.rglob("*")
        if path.is_dir()
    } == {
        "acceptance",
        "archive",
        "archive/objects",
        "candidates",
        "evaluations",
        "canary",
    }


def test_write_once_is_idempotent_but_never_overwrites(tmp_path):
    store = V2RunStore.create(tmp_path / "run")
    manifest = {"schema_version": 1, "system": "evolution_v2"}
    store.write_run_manifest(manifest)
    store.write_run_manifest(dict(reversed(tuple(manifest.items()))))
    assert (store.root / "run_manifest.json").read_bytes() == canonical_v2_bytes(
        manifest
    )

    with pytest.raises(StoreContractError, match="immutable"):
        store.write_run_manifest({"schema_version": 1, "system": "legacy"})


def test_first_manifest_write_must_declare_evolution_v2(tmp_path):
    store = V2RunStore.create(tmp_path / "run")
    with pytest.raises(StoreContractError, match="system.*evolution_v2"):
        store.write_run_manifest({"schema_version": 1, "system": "legacy"})
    assert not (store.root / "run_manifest.json").exists()


def test_store_mutates_only_checkpoint_and_active_bundle_atomically(tmp_path):
    store = V2RunStore.create(tmp_path / "run")
    store.write_checkpoint({"generation": 1})
    store.write_checkpoint({"generation": 2})
    store.publish_active_bundle({"generation": 1})
    store.publish_active_bundle({"generation": 2})

    assert (store.root / "checkpoint.json").read_bytes() == canonical_v2_bytes(
        {"generation": 2}
    )
    assert (store.root / "accepted_bundle.json").read_bytes() == canonical_v2_bytes(
        {"generation": 2}
    )
    with pytest.raises(StoreContractError, match="only checkpoint.json and accepted_bundle.json"):
        write_atomic_json(store.root / "budget_plan.json", {"budget": 2})


def test_budget_completion_and_content_objects_are_write_once(tmp_path):
    store = V2RunStore.create(tmp_path / "run")
    candidate = sha256_for("candidate")
    evidence = sha256_for("evidence")
    source = sha256_for("source")

    calls = (
        lambda payload: store.write_budget_plan(payload),
        lambda payload: store.write_candidate(candidate, payload),
        lambda payload: store.write_evaluation(candidate, "train", payload),
        lambda payload: store.write_acceptance(evidence, payload),
        lambda payload: store.write_canary(source, payload),
        lambda payload: store.write_completion(payload),
    )
    for write in calls:
        write({"value": 1})
        write({"value": 1})
        with pytest.raises(StoreContractError, match="immutable"):
            write({"value": 2})


def test_store_uses_nested_candidate_evaluation_paths_and_validates_components(tmp_path):
    store = V2RunStore.create(tmp_path / "run")
    candidate = sha256_for("candidate")
    evidence = sha256_for("evidence")
    source = sha256_for("source")

    assert store.write_candidate(candidate, {"kind": "candidate"}) == (
        store.root / "candidates" / f"{candidate}.json"
    )
    assert store.write_evaluation(candidate, "train_full", {"status": "passed"}) == (
        store.root / "evaluations" / candidate / "train_full.json"
    )
    assert store.write_acceptance(evidence, {"status": "accepted"}) == (
        store.root / "acceptance" / f"{evidence}.json"
    )
    assert store.write_canary(source, {"status": "passed"}) == (
        store.root / "canary" / f"{source}.json"
    )

    for invalid in ("short", "A" * 64, "g" * 64, "../" + "a" * 64):
        with pytest.raises(StoreContractError, match="canonical lowercase SHA-256"):
            store.write_candidate(invalid, {})
    for invalid_stage in ("", ".", "..", "../dev", "train/dev"):
        with pytest.raises(StoreContractError, match="stage"):
            store.write_evaluation(candidate, invalid_stage, {})


def test_store_rejects_non_finite_values_before_any_write(tmp_path):
    store = V2RunStore.create(tmp_path / "run")
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(StoreContractError, match="finite"):
            store.write_budget_plan({"value": value})
    assert not (store.root / "budget_plan.json").exists()


def test_jsonl_progress_and_promotion_history_append_canonical_fsynced_records(
    tmp_path, monkeypatch
):
    store = V2RunStore.create(tmp_path / "run")
    fsynced: list[int] = []
    monkeypatch.setattr("evolving_loop.v2.store.os.fsync", fsynced.append)

    store.append_progress({"step": 1})
    store.append_progress({"step": 2})
    store.append_promotion({"action": "activate", "bundle": sha256_for("bundle")})

    assert (store.root / "progress.jsonl").read_bytes() == (
        canonical_v2_bytes({"step": 1}) + canonical_v2_bytes({"step": 2})
    )
    assert (store.root / "promotion_history.jsonl").read_bytes() == canonical_v2_bytes(
        {"action": "activate", "bundle": sha256_for("bundle")}
    )
    assert len(fsynced) == 3


def test_create_accepts_empty_or_v2_run_but_refuses_legacy_non_empty_directory(
    tmp_path,
):
    empty = tmp_path / "empty"
    empty.mkdir()
    store = V2RunStore.create(empty)
    store.write_run_manifest({"schema_version": 1, "system": "evolution_v2"})
    assert V2RunStore.create(empty).root == empty

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "run_manifest.json").write_text(
        '{"schema_version":1,"system":"legacy"}\n', encoding="utf-8"
    )
    with pytest.raises(StoreContractError, match="refuse.*non-empty"):
        V2RunStore.create(legacy)

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "state.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(StoreContractError, match="refuse.*non-empty"):
        V2RunStore.create(unknown)
