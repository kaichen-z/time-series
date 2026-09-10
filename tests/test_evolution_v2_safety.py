"""Hostile Project 1 data-boundary gates; OS source isolation is Project 4."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from evolving_loop.v2.archive import (
    ArchiveContractError,
    ArchiveRecord,
    EvolutionArchive,
)
from evolving_loop.v2.budget import BudgetLedger, BudgetPlan, ResourceUse
from evolving_loop.v2.bundle import BundleContractError, EvolutionBundleV2
from evolving_loop.v2.cli import main
from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.fakes import fake_seed_bundle, run_fake_kernel, smoke_config
from evolving_loop.v2.kernel import EvolutionKernel, KernelAuthorityError
from evolving_loop.v2.store import StoreContractError, V2RunStore


def snapshot(root):
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def kernel(tmp_path):
    config = smoke_config()
    return EvolutionKernel(
        V2RunStore.create(tmp_path / "run"),
        config.kernel_protocol,
        BudgetLedger(
            BudgetPlan(600, 0.2, ResourceUse(task_executions=100)),
            monotonic=lambda: 0.0,
        ),
        seed=fake_seed_bundle(config),
    )


def numerical_child(parent):
    return parent.provisional_child("numerical", {"numerical": ("a" * 64, "b" * 64)})


def close(kernel, parent, child, *, dev=None, train=None):
    permit = kernel.reserve_evaluation(child, ResourceUse(task_executions=2))
    assert permit.allowed
    evaluation = kernel.close_evaluation(
        parent,
        child,
        permit=permit,
        status="passed",
        train_objectives={"loss": 0.4} if train is None else train,
        train_behavior_descriptors={"family": "flat"},
        dev_comparison=dev
        or {
            "passed": True,
            "parent_metrics": {"loss": 0.6},
            "candidate_metrics": {"loss": 0.4},
        },
        resource_use=ResourceUse(task_executions=1),
    )
    return evaluation, permit


@pytest.mark.parametrize(
    "field",
    [
        "protocol_fingerprint",
        "runtime_fingerprints",
        "harness_policy_sha256",
        "archive_snapshot_sha256",
        "scheduler_state_sha256",
        "parent_bundle_sha256",
        "acceptance_evidence_sha256",
    ],
)
def test_each_candidate_authority_mutation_fails_closed(kernel, field):
    parent = kernel.active_bundle()
    pointer = (kernel.store.root / "accepted_bundle.json").read_bytes()
    history = kernel.promotion_host.history()
    archive = snapshot(kernel.archive.root)
    value = {"fake": "f" * 64} if field == "runtime_fingerprints" else "f" * 64
    child = replace(numerical_child(parent), **{field: value})
    evaluation, permit = close(kernel, parent, child)
    with pytest.raises(KernelAuthorityError):
        kernel.evaluate_transition(
            parent, child, target="numerical", evaluation=evaluation, permit=permit
        )
    assert kernel.active_bundle().canonical_bytes() == parent.canonical_bytes()
    assert (kernel.store.root / "accepted_bundle.json").read_bytes() == pointer
    assert kernel.promotion_host.history() == history
    assert snapshot(kernel.archive.root) == archive
    assert kernel.budget.charged_use.task_executions == 1


@pytest.mark.parametrize(
    "attack",
    ["release_without_registry", "single_target_multiple_scopes", "joint_one_scope"],
)
def test_scope_attacks_cannot_promote(kernel, attack):
    parent = kernel.active_bundle()
    child = numerical_child(parent)
    target = "numerical"
    if attack == "release_without_registry":
        child = replace(
            child, numerical_registry_sha256=parent.numerical_registry_sha256
        )
    elif attack == "single_target_multiple_scopes":
        child = replace(child, retrieval_release_sha256="c" * 64)
    else:
        target = "joint"
    evaluation, permit = close(kernel, parent, child)
    with pytest.raises(KernelAuthorityError):
        kernel.evaluate_transition(
            parent, child, target=target, evaluation=evaluation, permit=permit
        )
    assert kernel.active_bundle().canonical_bytes() == parent.canonical_bytes()
    assert len(kernel.promotion_host.history()) == 1
    assert not tuple((kernel.store.root / "acceptance").iterdir())


@pytest.mark.parametrize("location", ["train", "dev"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_nested_metrics_never_close_or_promote(kernel, location, value):
    parent = kernel.active_bundle()
    child = numerical_child(parent)
    nested = {"by_task": [{"metrics": {"loss": value}}]}
    kwargs = (
        {"train": nested}
        if location == "train"
        else {
            "dev": {"passed": True, "parent_metrics": {}, "candidate_metrics": nested}
        }
    )
    with pytest.raises(ValueError, match="finite"):
        close(kernel, parent, child, **kwargs)
    assert kernel.active_bundle().canonical_bytes() == parent.canonical_bytes()
    assert not tuple((kernel.store.root / "evaluations").iterdir())
    assert not tuple((kernel.store.root / "acceptance").iterdir())


@pytest.mark.parametrize(
    "bad", ["../escape", "a/../../escape", "/tmp/escape", "..\\escape"]
)
@pytest.mark.parametrize(
    "sink", ["candidate", "evaluation_sha", "evaluation_stage", "acceptance", "canary"]
)
def test_sha_and_stage_traversal_rejected_before_any_write(kernel, bad, sink):
    before = snapshot(kernel.store.root.parent)
    store = kernel.store
    with pytest.raises(StoreContractError):
        if sink == "candidate":
            store.write_candidate(bad, {})
        elif sink == "evaluation_sha":
            store.write_evaluation(bad, "train", {})
        elif sink == "evaluation_stage":
            store.write_evaluation("a" * 64, bad, {})
        elif sink == "acceptance":
            store.write_acceptance(bad, {})
        else:
            store.write_canary(bad, {})
    assert snapshot(kernel.store.root.parent) == before


@pytest.mark.parametrize(
    "key",
    [
        "dev_metrics",
        "dev_comparison",
        "public_ids",
        "future_values",
        "evaluator_labels",
        "holdout",
    ],
)
@pytest.mark.parametrize(
    "field", ["train_objectives", "train_behavior_descriptors", "resource_use"]
)
def test_nested_evaluator_archive_keys_fail_before_index_or_object_write(
    kernel, key, field
):
    before = snapshot(kernel.archive.root)
    record = json.loads(kernel.archive.index.read_text().splitlines()[0])["record"]
    record[field] = {"outer": [{"inner": {key: "HOSTILE_SENTINEL"}}]}
    with pytest.raises(ArchiveContractError, match="evaluator-only"):
        ArchiveRecord.from_payload(record)
    assert snapshot(kernel.archive.root) == before


@pytest.mark.parametrize(
    "artifact", ["pointer", "manifest", "archive_index", "archive_object"]
)
def test_duplicate_authoritative_json_keys_fail_closed(kernel, artifact):
    parent = kernel.active_bundle()
    paths = {
        "pointer": kernel.store.root / "accepted_bundle.json",
        "manifest": kernel.store.root / "run_manifest.json",
        "archive_index": kernel.archive.index,
        "archive_object": kernel.archive.root
        / "objects"
        / f"{parent.fingerprint()}.json",
    }
    path = paths[artifact]
    raw = path.read_text()
    payload = json.loads(raw)
    key = next(iter(payload))
    path.write_text(
        "{" + json.dumps(key) + ":" + json.dumps(payload[key]) + "," + raw[1:]
    )
    before = snapshot(kernel.store.root)
    with pytest.raises((KernelAuthorityError, ArchiveContractError)):
        EvolutionKernel.resume(kernel.store, kernel.budget.plan, monotonic=lambda: 0.0)
    assert snapshot(kernel.store.root) == before


def test_mutable_pointer_without_host_evidence_is_not_authority(kernel):
    parent = kernel.active_bundle()
    forged = replace(numerical_child(parent), acceptance_evidence_sha256="f" * 64)
    kernel.store.publish_active_bundle(forged.to_payload())
    before = snapshot(kernel.store.root)
    with pytest.raises(KernelAuthorityError):
        kernel.active_bundle()
    with pytest.raises(KernelAuthorityError):
        kernel.promotion_host.activate(forged)
    with pytest.raises(KernelAuthorityError):
        EvolutionKernel.resume(kernel.store, kernel.budget.plan, monotonic=lambda: 0.0)
    assert snapshot(kernel.store.root) == before


def assert_primitives(value):
    assert type(value) in (dict, list, str, int, float, bool, type(None))
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str
            assert key not in {
                "store",
                "kernel",
                "archive",
                "promotion_host",
                "callback",
                "source",
                "module",
                "path",
            }
            assert_primitives(item)
    elif type(value) is list:
        for item in value:
            assert_primitives(item)
    elif type(value) is str:
        assert "/" not in value and "\\" not in value


def test_actual_fake_proposal_boundary_contains_only_canonical_primitives(
    tmp_path, monkeypatch
):
    """Observe before serialization, so JSON cannot conceal a leaked live object."""
    proposals = []
    write_candidate = V2RunStore.write_candidate

    def inspect(store, identity, payload):
        assert_primitives(payload)
        encoded = canonical_v2_bytes(payload)
        assert json.loads(encoded) == payload
        assert (
            EvolutionBundleV2.from_payload(payload["candidate"]).fingerprint()
            == identity
        )
        proposals.append(encoded)
        return write_candidate(store, identity, payload)

    monkeypatch.setattr(V2RunStore, "write_candidate", inspect)
    result = run_fake_kernel(tmp_path / "run", smoke_config())
    assert len(proposals) == 2
    assert result.accepted_steps == result.rejected_steps == 1


@pytest.mark.parametrize(
    "field",
    [
        "source",
        "source_code",
        "module",
        "module_name",
        "path",
        "source_path",
        "kernel",
        "store",
        "archive",
        "promotion_host",
        "callback",
    ],
)
def test_bundle_proposal_parser_rejects_code_paths_and_capabilities(kernel, field):
    parent = kernel.active_bundle()
    payload = numerical_child(parent).to_payload()
    payload[field] = "import evolving_loop.v2.kernel; raise RuntimeError('executed')"
    before = snapshot(kernel.store.root)
    with pytest.raises(BundleContractError):
        EvolutionBundleV2.from_payload(payload)
    assert snapshot(kernel.store.root) == before


@pytest.mark.parametrize(
    "capability", ["kernel", "store", "archive", "promotion_host", "callback", "path"]
)
def test_bundle_parser_rejects_live_capability_in_declared_field(kernel, capability):
    capabilities = {
        "kernel": kernel,
        "store": kernel.store,
        "archive": kernel.archive,
        "promotion_host": kernel.promotion_host,
        "callback": lambda: pytest.fail("candidate callback executed"),
        "path": Path("/tmp/candidate-source.py"),
    }
    payload = numerical_child(kernel.active_bundle()).to_payload()
    payload["retrieval_release_sha256"] = capabilities[capability]
    with pytest.raises(BundleContractError):
        EvolutionBundleV2.from_payload(payload)


@pytest.mark.skip(
    reason="Required Project 4 integration: separate candidate process, read-only kernel code root, import/write isolation; Project 1 tests only the data/API boundary"
)
def test_project4_hostile_source_process_cannot_import_or_write_kernel():
    raise AssertionError("Project 4 sandbox integration has not been implemented")


@pytest.mark.parametrize("passed", [True, False])
def test_dev_sentinels_remain_only_in_evaluator_evidence(kernel, passed):
    parent = kernel.active_bundle()
    child = numerical_child(parent)
    sentinels = ("DEV_PARENT_752ae38", "DEV_CHILD_a9814bc")
    evaluation, permit = close(
        kernel,
        parent,
        child,
        dev={
            "passed": passed,
            "parent_metrics": {"sentinel": sentinels[0]},
            "candidate_metrics": {"sentinel": sentinels[1]},
        },
    )
    result = kernel.evaluate_transition(
        parent, child, target="numerical", evaluation=evaluation, permit=permit
    )
    assert (result.fingerprint() != parent.fingerprint()) is passed
    evidence_paths = set()
    for path in kernel.store.root.rglob("*"):
        if not path.is_file():
            continue
        raw = path.read_text()
        if any(sentinel in raw for sentinel in sentinels):
            relative = path.relative_to(kernel.store.root)
            assert (
                relative.parts[0] == "acceptance"
                or relative == Path("evaluations") / child.fingerprint() / "closed.json"
            )
            assert all(sentinel in raw for sentinel in sentinels)
            evidence_paths.add(relative)
    assert len(evidence_paths) == 2
    assert all(
        sentinel not in json.dumps(kernel.budget.checkpoint()) for sentinel in sentinels
    )
    next_child = result.provisional_child("retrieval", {"retrieval": "c" * 64})
    assert all(
        sentinel not in json.dumps(next_child.to_payload()) for sentinel in sentinels
    )
    EvolutionArchive(kernel.archive.root)


def test_fake_next_proposal_and_checkpoints_do_not_receive_dev_sentinel(
    tmp_path, monkeypatch
):
    sentinel = "FAKE_DEV_ONLY_c7805e2"
    close_evaluation = EvolutionKernel.close_evaluation
    write_candidate = V2RunStore.write_candidate
    proposals = []

    def evaluator(kernel, parent, child, **kwargs):
        comparison = kwargs["dev_comparison"]
        kwargs["dev_comparison"] = comparison | {
            "candidate_metrics": comparison["candidate_metrics"] | {"trace": sentinel}
        }
        return close_evaluation(kernel, parent, child, **kwargs)

    def proposer_boundary(store, identity, payload):
        assert_primitives(payload)
        encoded = canonical_v2_bytes(payload)
        assert sentinel.encode() not in encoded
        proposals.append(encoded)
        return write_candidate(store, identity, payload)

    monkeypatch.setattr(EvolutionKernel, "close_evaluation", evaluator)
    monkeypatch.setattr(V2RunStore, "write_candidate", proposer_boundary)
    root = tmp_path / "run"
    result = run_fake_kernel(root, smoke_config())
    assert result.accepted_steps == result.rejected_steps == 1
    assert len(proposals) == 2
    containing = []
    for relative, (raw, _mtime) in snapshot(root).items():
        if sentinel.encode() in raw:
            path = Path(relative)
            assert path.parts[0] == "acceptance" or (
                path.parts[0] == "evaluations" and path.name == "closed.json"
            )
            containing.append(relative)
    assert len(containing) == 4


def test_public_validation_output_sentinel_cannot_feed_back_to_source(tmp_path, capsys):
    source = tmp_path / "run"
    result = run_fake_kernel(source, smoke_config())
    before = snapshot(source)
    output = tmp_path / "PUBLIC_SENTINEL_f319be7"
    assert (
        main(
            [
                "public-evaluate",
                "--bundle",
                str(source / "accepted_bundle.json"),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    sentinel = "validated_only_no_public_evaluator"
    assert summary["status"] == sentinel
    assert summary["public_test_accessed"] is False
    assert summary["bundle_sha256"] == result.accepted_bundle.fingerprint()
    assert sentinel in (output / "evaluation_complete.json").read_text()
    assert snapshot(source) == before
    for raw, _mtime in before.values():
        assert sentinel.encode() not in raw
        assert output.name.encode() not in raw
    run_fake_kernel(source, smoke_config(), resume=True)
    assert snapshot(source) == before
