from __future__ import annotations

import math
from pathlib import Path

import pytest

from evolving_loop.v2 import store as store_module
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
        store.root / "candidates" / candidate / "proposal.json"
    )
    store.write_candidate(candidate, {"kind": "candidate"})
    with pytest.raises(StoreContractError, match="immutable"):
        store.write_candidate(candidate, {"kind": "changed"})
    assert not (store.root / "candidates" / f"{candidate}.json").exists()
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


def test_completion_uses_exact_write_once_evaluation_complete_path(tmp_path):
    store = V2RunStore.create(tmp_path / "run")
    completion = {"status": "complete"}

    assert store.write_completion(completion) == store.root / "evaluation_complete.json"
    store.write_completion({"status": "complete"})
    with pytest.raises(StoreContractError, match="immutable"):
        store.write_completion({"status": "incomplete"})
    assert not (store.root / "completion.json").exists()


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
    fsynced_directories: list[Path] = []
    monkeypatch.setattr("evolving_loop.v2.store.os.fsync", fsynced.append)
    monkeypatch.setattr(
        store_module, "_fsync_directory", fsynced_directories.append
    )

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
    assert store.root in fsynced_directories
    assert store.root.parent in fsynced_directories


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


@pytest.mark.parametrize(
    "manifest, message",
    [
        (
            b'{"schema_version":1,"system":"evolution_v2","value":1e999}\n',
            "finite",
        ),
        (
            b'{"system": "evolution_v2", "schema_version": 1}\n',
            "canonical",
        ),
    ],
)
def test_resume_rejects_invalid_manifest_before_creating_layout(
    tmp_path, manifest, message
):
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_manifest.json").write_bytes(manifest)

    with pytest.raises(StoreContractError, match=message):
        V2RunStore.create(root)

    assert [path.name for path in root.iterdir()] == ["run_manifest.json"]


def test_atomic_publication_fsyncs_new_directories_and_destination_parent_in_order(
    tmp_path, monkeypatch
):
    store = V2RunStore.create(tmp_path / "run")
    candidate = sha256_for("new-candidate")
    destination = store.root / "evaluations" / candidate / "train.json"
    events: list[tuple[str, Path]] = []
    real_replace = store_module.os.replace

    def observed_replace(source, target):
        real_replace(source, target)
        events.append(("replace", Path(target)))

    monkeypatch.setattr(store_module.os, "replace", observed_replace)
    monkeypatch.setattr(
        store_module,
        "_fsync_directory",
        lambda path: events.append(("fsync_directory", Path(path))),
        raising=False,
    )

    store.write_evaluation(candidate, "train", {"status": "passed"})

    candidate_directory = destination.parent
    assert ("fsync_directory", candidate_directory.parent) in events
    assert events[-2:] == [
        ("replace", destination),
        ("fsync_directory", candidate_directory),
    ]


def test_candidate_proposal_durably_publishes_new_candidate_directory(
    tmp_path, monkeypatch
):
    store = V2RunStore.create(tmp_path / "run")
    candidate = sha256_for("durable-candidate")
    destination = store.root / "candidates" / candidate / "proposal.json"
    events: list[tuple[str, Path]] = []
    real_replace = store_module.os.replace

    def observed_replace(source, target):
        real_replace(source, target)
        events.append(("replace", Path(target)))

    monkeypatch.setattr(store_module.os, "replace", observed_replace)
    monkeypatch.setattr(
        store_module,
        "_fsync_directory",
        lambda path: events.append(("fsync_directory", Path(path))),
    )

    store.write_candidate(candidate, {"kind": "candidate"})

    candidate_directory = destination.parent
    assert ("fsync_directory", candidate_directory.parent) in events
    assert events[-2:] == [
        ("replace", destination),
        ("fsync_directory", candidate_directory),
    ]


def test_new_jsonl_filename_is_directory_durable_after_file_fsync(
    tmp_path, monkeypatch
):
    store = V2RunStore.create(tmp_path / "run")
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        store_module.os, "fsync", lambda descriptor: events.append(("file", descriptor))
    )
    monkeypatch.setattr(
        store_module,
        "_fsync_directory",
        lambda path: events.append(("directory", Path(path))),
        raising=False,
    )

    store.append_progress({"step": 1})

    file_fsync = next(
        index for index, event in enumerate(events) if event[0] == "file"
    )
    assert ("directory", store.root) in events[:file_fsync]
    assert ("directory", store.root.parent) in events[:file_fsync]
    assert events[file_fsync + 1 :] == [("directory", store.root)]


def test_existing_jsonl_append_republishes_existing_directory_entry(
    tmp_path, monkeypatch
):
    store = V2RunStore.create(tmp_path / "run")
    store.append_progress({"step": 1})
    fsynced_directories: list[Path] = []
    monkeypatch.setattr(
        store_module, "_fsync_directory", fsynced_directories.append
    )

    store.append_progress({"step": 2})

    assert store.root in fsynced_directories
    assert store.root.parent in fsynced_directories


def test_directory_fsync_failures_are_not_swallowed(tmp_path, monkeypatch):
    store = V2RunStore.create(tmp_path / "run")

    def fail_fsync(_path):
        raise OSError("directory fsync failed")

    monkeypatch.setattr(
        store_module, "_fsync_directory", fail_fsync, raising=False
    )

    with pytest.raises(OSError, match="directory fsync failed"):
        store.write_checkpoint({"generation": 1})
