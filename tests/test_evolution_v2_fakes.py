import hashlib
import importlib
import json
from dataclasses import replace

import pytest

from evolving_loop.v2.contracts import canonical_v2_bytes


@pytest.fixture
def api():
    return importlib.import_module("evolving_loop.v2.fakes")


def files(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def read(path):
    return json.loads(path.read_bytes())


def test_fake_loop_accepts_then_rejects_and_is_byte_reproducible(tmp_path, api):
    first = api.run_fake_kernel(tmp_path / "first", api.smoke_config(seed=7))
    second = api.run_fake_kernel(tmp_path / "second", api.smoke_config(seed=7))
    assert first.accepted_steps == 1
    assert first.rejected_steps == 1
    assert first.runner == "deterministic_fake"
    assert first.completed_stage_ids == ("fake-numerical", "fake-retrieval")
    assert (
        first.accepted_bundle.canonical_bytes()
        == second.accepted_bundle.canonical_bytes()
    )
    assert {k: v[0] for k, v in files(tmp_path / "first").items()} == {
        k: v[0] for k, v in files(tmp_path / "second").items()
    }
    completion = read(first.completion_path)
    assert completion["status"] == "deterministic_fake_complete"
    assert completion["public_test_accessed"] is False
    assert completion["budget_usage"]["task_executions"] == 40
    assert completion["budget_usage"]["llm_calls"] == 0
    assert (
        first.accepted_bundle.numerical_release_sha256
        == hashlib.sha256(b"v2-fake:7:numerical-child-release").hexdigest()
    )
    assert len(list((tmp_path / "first" / "evaluations").glob("*/closed.json"))) == 2


def test_fake_resume_preserves_rejected_parent_and_matches_uninterrupted(tmp_path, api):
    root = tmp_path / "run"
    partial = api.run_fake_kernel(
        root, api.smoke_config(), stop_after_stage="fake-numerical"
    )
    parent_bytes = partial.accepted_bundle.canonical_bytes()
    assert not partial.completion_path.exists()
    immutable = {
        k: v
        for k, v in files(root).items()
        if k.startswith(("evaluations/", "acceptance/", "archive/objects/"))
    }
    resumed = api.run_fake_kernel(root, api.smoke_config(), resume=True)
    assert resumed.accepted_bundle.canonical_bytes() == parent_bytes
    assert all(files(root)[k] == v for k, v in immutable.items())
    assert resumed.completed_stage_ids == ("fake-numerical", "fake-retrieval")
    full = api.run_fake_kernel(tmp_path / "full", api.smoke_config())
    assert resumed.accepted_bundle == full.accepted_bundle
    assert {k: v[0] for k, v in files(root).items()} == {
        k: v[0] for k, v in files(tmp_path / "full").items()
    }


def test_completed_resume_verifies_without_writes(tmp_path, api):
    result = api.run_fake_kernel(tmp_path, api.smoke_config())
    before = files(tmp_path)
    resumed = api.run_fake_kernel(tmp_path, api.smoke_config(), resume=True)
    assert resumed == result
    assert files(tmp_path) == before


def test_seed_changes_artifacts(tmp_path, api):
    first = api.run_fake_kernel(tmp_path / "a", api.smoke_config(seed=1))
    second = api.run_fake_kernel(tmp_path / "b", api.smoke_config(seed=2))
    assert first.accepted_bundle.fingerprint() != second.accepted_bundle.fingerprint()


def test_checkpoint_has_exact_fields_and_closed_ledger(tmp_path, api):
    api.run_fake_kernel(tmp_path, api.smoke_config())
    checkpoint = read(tmp_path / "checkpoint.json")
    assert set(checkpoint) == {
        "config_sha256",
        "active_bundle_sha256",
        "archive_snapshot_sha256",
        "budget",
        "completed_stage_ids",
        "accepted_steps",
        "rejected_steps",
        "public_test_accessed",
    }
    assert checkpoint["budget"]["open_reservations"] == []
    assert checkpoint["budget"]["finalization_started"] is True
    assert checkpoint["public_test_accessed"] is False


@pytest.mark.parametrize(
    "change",
    [
        "config",
        "checkpoint",
        "kernel_checkpoint",
        "index",
        "object",
        "evidence",
        "rejected_evidence",
        "evaluation",
        "completion",
        "extra",
        "missing_completion",
        "missing_progress",
        "manifest_extra",
    ],
)
def test_resume_rejects_tampering_before_writing(tmp_path, api, change):
    config = api.smoke_config()
    result = api.run_fake_kernel(tmp_path, config)
    if change == "config":
        config = replace(config, scheduler="thompson")
    elif change == "extra":
        (tmp_path / "evaluations" / "partial.json").write_text("{}")
    elif change == "missing_completion":
        result.completion_path.unlink()
    elif change == "missing_progress":
        (tmp_path / "progress.jsonl").unlink()
    elif change == "manifest_extra":
        path = tmp_path / "run_manifest.json"
        payload = read(path)
        payload["uncommitted"] = "tampered"
        path.write_bytes(canonical_v2_bytes(payload))
    else:
        if change == "checkpoint":
            path = tmp_path / "checkpoint.json"
            payload = read(path)
            payload["accepted_steps"] = 2
            path.write_bytes(canonical_v2_bytes(payload))
        else:
            path = {
                "kernel_checkpoint": tmp_path / "kernel" / "checkpoint.json",
                "index": tmp_path / "archive" / "index.jsonl",
                "object": next((tmp_path / "archive" / "objects").glob("*.json")),
                "evidence": next((tmp_path / "acceptance").glob("*.json")),
                "rejected_evidence": next(
                    p
                    for p in (tmp_path / "acceptance").glob("*.json")
                    if read(p)["decision"] == "reject"
                ),
                "evaluation": next((tmp_path / "evaluations").glob("*/train.json")),
                "completion": result.completion_path,
            }[change]
            path.write_bytes(b"{}\n")
    before = files(tmp_path)
    with pytest.raises(ValueError):
        api.run_fake_kernel(tmp_path, config, resume=True)
    assert files(tmp_path) == before


def test_unreferenced_atomic_temporary_file_is_ignored(tmp_path, api):
    api.run_fake_kernel(tmp_path, api.smoke_config(), stop_after_stage="fake-numerical")
    temporary = tmp_path / ".checkpoint.json.interrupted.tmp"
    temporary.write_bytes(b"partial")
    assert (
        api.run_fake_kernel(tmp_path, api.smoke_config(), resume=True).rejected_steps
        == 1
    )
    assert temporary.read_bytes() == b"partial"


def test_exhausted_budget_never_opens_evaluation_or_completes(tmp_path, api):
    config = replace(api.smoke_config(), hard_limit_seconds=1)
    with pytest.raises(ValueError, match="budget|reserve"):
        api.run_fake_kernel(tmp_path, config)
    assert not list(tmp_path.glob("evaluations/*/*.json"))
    assert not (tmp_path / "evaluation_complete.json").exists()


def test_partial_stage_is_not_repaired(tmp_path, api, monkeypatch):
    from evolving_loop.v2.kernel import EvolutionKernel

    original = EvolutionKernel.close_evaluation

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("interruption after reservation")

    monkeypatch.setattr(EvolutionKernel, "close_evaluation", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        api.run_fake_kernel(tmp_path, api.smoke_config())
    before = files(tmp_path)
    with pytest.raises(ValueError):
        api.run_fake_kernel(tmp_path, api.smoke_config(), resume=True)
    assert files(tmp_path) == before


@pytest.mark.parametrize(
    "updates",
    [
        {"runner": "production"},
        {"profile": "pilot"},
        {"enabled_mutation_scopes": ("decision",)},
    ],
)
def test_fake_rejects_unsupported_configuration_before_writes(tmp_path, api, updates):
    with pytest.raises(ValueError):
        api.run_fake_kernel(tmp_path, replace(api.smoke_config(), **updates))
    assert files(tmp_path) == {}
