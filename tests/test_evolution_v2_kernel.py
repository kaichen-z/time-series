from dataclasses import replace
import json

import pytest

from evolving_loop.v2 import kernel as api
from evolving_loop.v2.bundle import EvolutionBundleV2
from evolving_loop.v2.budget import BudgetLedger, BudgetPlan, ResourceUse, StagePermit
from evolving_loop.v2.contracts import (
    KernelProtocolCommitment,
    canonical_v2_bytes,
    fingerprint_payload,
)
from evolving_loop.v2.store import V2RunStore


def protocol():
    return KernelProtocolCommitment(*[str(i) * 64 for i in range(1, 8)])


def seed():
    return EvolutionBundleV2(
        2,
        0,
        None,
        *[str(i) * 64 for i in range(1, 8)],
        protocol().fingerprint(),
        {"python": "a" * 64},
        None,
    )


@pytest.fixture
def kernel(tmp_path):
    plan = BudgetPlan(1000, 0.2, ResourceUse(wall_seconds=800.0, task_executions=100))
    return api.EvolutionKernel(
        V2RunStore.create(tmp_path / "run"),
        protocol(),
        BudgetLedger(plan, monotonic=lambda: 0.0),
        seed=seed(),
    )


def child(parent):
    return parent.provisional_child("retrieval", {"retrieval": "b" * 64})


def evaluation(kernel, parent, candidate, *, passed=True, status="passed", actual=1):
    permit = kernel.budget.reserve_stage(
        kernel.stage_id(candidate), ResourceUse(task_executions=2)
    )
    result = kernel.close_evaluation(
        parent,
        candidate,
        permit=permit,
        status=status,
        train_objectives={"loss": 0.5},
        train_behavior_descriptors={"family": "flat"},
        dev_comparison={
            "passed": passed,
            "parent_metrics": {"loss": 0.6},
            "candidate_metrics": {"loss": 0.4},
        },
        resource_use=ResourceUse(task_executions=actual),
    )
    return result, permit


def transition(kernel, *, passed=True, status="passed", actual=1):
    parent = kernel.active_bundle()
    candidate = child(parent)
    result, permit = evaluation(
        kernel, parent, candidate, passed=passed, status=status, actual=actual
    )
    accepted = kernel.evaluate_transition(
        parent, candidate, target="retrieval", evaluation=result, permit=permit
    )
    return parent, candidate, accepted


def test_acceptance_binds_provisional_child_and_archives_both_states(kernel):
    parent, candidate, accepted = transition(kernel)
    evidence = kernel.load_acceptance(accepted.acceptance_evidence_sha256)
    assert evidence.parent_bundle_sha256 == parent.fingerprint()
    assert evidence.candidate_bundle_sha256 == candidate.fingerprint()
    assert evidence.protocol_fingerprint == protocol().fingerprint()
    assert evidence.runtime_fingerprints == parent.runtime_fingerprints
    assert evidence.decision == "accept"
    assert evidence.dev_comparison["candidate_metrics"]["loss"] == 0.4
    assert "sealed_bundle_sha256" not in evidence.to_payload()
    assert kernel.archive.lineage(accepted.fingerprint()) == (
        parent.fingerprint(),
        candidate.fingerprint(),
        accepted.fingerprint(),
    )
    assert kernel.active_bundle().fingerprint() == accepted.fingerprint()
    assert kernel.budget.charged_use.task_executions == 1
    with pytest.raises(TypeError):
        evidence.dev_comparison["candidate_metrics"]["loss"] = 1


@pytest.mark.parametrize(
    "passed,status", [(False, "passed"), (True, "failed"), (True, "invalid")]
)
def test_rejection_keeps_exact_parent_and_train_only_records(kernel, passed, status):
    parent, candidate, returned = transition(kernel, passed=passed, status=status)
    assert returned is parent
    assert kernel.active_bundle().canonical_bytes() == parent.canonical_bytes()
    assert kernel.archive.lineage(candidate.fingerprint()) == (
        parent.fingerprint(),
        candidate.fingerprint(),
    )
    assert len(kernel.promotion_host.history()) == 1
    assert kernel.budget.charged_use.task_executions == 1
    for path in (kernel.archive.index, kernel.store.root / "progress.jsonl"):
        assert "dev_comparison" not in path.read_text()
        assert "candidate_metrics" not in path.read_text()
    record = json.loads(kernel.archive.index.read_text().splitlines()[-1])["record"]
    assert record["evaluation_status"] == status


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_fingerprint", "9" * 64),
        ("runtime_fingerprints", {"python": "9" * 64}),
        ("acceptance_evidence_sha256", "9" * 64),
        ("harness_policy_sha256", "9" * 64),
        ("parent_bundle_sha256", "9" * 64),
        ("decision_policy_sha256", "9" * 64),
    ],
)
def test_scope_and_authority_smuggling_fails_closed_and_accounts_use(
    kernel, field, value
):
    parent = kernel.active_bundle()
    candidate = replace(child(parent), **{field: value})
    result, permit = evaluation(kernel, parent, candidate)
    with pytest.raises(api.KernelAuthorityError):
        kernel.evaluate_transition(
            parent, candidate, target="retrieval", evaluation=result, permit=permit
        )
    assert kernel.active_bundle() == parent
    assert kernel.budget.charged_use.task_executions == 1


@pytest.mark.parametrize(
    "permit",
    [None, StagePermit(True, None, "f" * 64), StagePermit(False, "denied", None)],
)
def test_missing_or_fabricated_permit_cannot_transition(kernel, permit):
    parent = kernel.active_bundle()
    candidate = child(parent)
    result, _ = evaluation(kernel, parent, candidate)
    with pytest.raises(api.KernelAuthorityError, match="permit"):
        kernel.evaluate_transition(
            parent, candidate, target="retrieval", evaluation=result, permit=permit
        )
    assert kernel.active_bundle() == parent


def test_candidate_fabricated_or_replaced_closed_evaluation_is_rejected(kernel):
    parent = kernel.active_bundle()
    candidate = child(parent)
    result, permit = evaluation(kernel, parent, candidate, passed=False)
    forged = replace(
        result,
        dev_comparison={"passed": True, "parent_metrics": {}, "candidate_metrics": {}},
    )
    with pytest.raises(api.KernelAuthorityError, match="closed evaluation"):
        kernel.evaluate_transition(
            parent, candidate, target="retrieval", evaluation=forged, permit=permit
        )
    assert kernel.active_bundle() == parent
    assert kernel.budget.charged_use.task_executions == 1


def test_budget_overrun_rejects_and_accounts_full_actual(kernel):
    parent, _, returned = transition(kernel, actual=3)
    assert returned is parent
    assert kernel.budget.charged_use.task_executions == 3
    assert kernel.budget.checkpoint()["exhausted_reason"] == "budget_overrun"


def test_automatic_promotion_rollback_and_restart(kernel):
    parent, _, accepted = transition(kernel)
    restored = kernel.promotion_host.rollback(reason="canary_failed")
    assert restored == parent
    history = kernel.promotion_host.history()
    assert [r["action"] for r in history] == ["seed", "activate", "rollback"]
    assert history[1]["bundle_sha256"] == accepted.fingerprint()
    assert (
        history[1]["acceptance_evidence_sha256"] == accepted.acceptance_evidence_sha256
    )
    resumed = api.EvolutionKernel.resume(
        kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
    )
    assert resumed.active_bundle() == parent
    assert resumed.budget.charged_use.task_executions == 1
    assert resumed.promotion_host.history() == history


def test_restart_reconciles_committed_activation_and_rollback_once(kernel, monkeypatch):
    parent = kernel.active_bundle()
    publish = kernel.store.publish_active_bundle

    def fail(payload):
        raise OSError("injected pointer failure")

    monkeypatch.setattr(kernel.store, "publish_active_bundle", fail)
    with pytest.raises(OSError, match="pointer failure"):
        transition(kernel)
    assert (
        json.loads((kernel.store.root / "accepted_bundle.json").read_text())
        == parent.to_payload()
    )
    monkeypatch.setattr(kernel.store, "publish_active_bundle", publish)
    resumed = api.EvolutionKernel.resume(
        kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
    )
    assert resumed.active_bundle().generation == 1
    assert len(resumed.promotion_host.history()) == 2
    monkeypatch.setattr(kernel.store, "publish_active_bundle", fail)
    with pytest.raises(OSError):
        resumed.promotion_host.rollback(reason="integrity_failed")
    monkeypatch.setattr(kernel.store, "publish_active_bundle", publish)
    again = api.EvolutionKernel.resume(
        kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
    )
    assert again.active_bundle() == parent
    assert len(again.promotion_host.history()) == 3


def test_resume_requires_full_manifest_commitment(kernel):
    path = kernel.store.root / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["kernel_protocol"]["metric_policy"] = "f" * 64
    path.write_bytes(canonical_v2_bytes(manifest))
    with pytest.raises(api.KernelAuthorityError, match="protocol"):
        api.EvolutionKernel.resume(
            kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
        )


def test_host_rejects_unsealed_unknown_and_missing_evidence(kernel):
    parent, candidate, accepted = transition(kernel)
    with pytest.raises(api.KernelAuthorityError):
        kernel.promotion_host.activate(candidate)
    with pytest.raises(api.KernelAuthorityError):
        kernel.promotion_host.rollback(
            reason="please", bundle_sha256=parent.fingerprint()
        )
    with pytest.raises(api.KernelAuthorityError):
        kernel.promotion_host.rollback(reason="canary_failed", bundle_sha256="e" * 64)
    (
        kernel.store.root / "acceptance" / f"{accepted.acceptance_evidence_sha256}.json"
    ).unlink()
    with pytest.raises(api.KernelAuthorityError):
        kernel.promotion_host.activate(accepted)


def test_archive_omission_and_evidence_corruption_fail_closed(kernel):
    _, _, accepted = transition(kernel)
    path = (
        kernel.store.root / "acceptance" / f"{accepted.acceptance_evidence_sha256}.json"
    )
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'"accept"', b'"reject"'))
    with pytest.raises(api.KernelAuthorityError):
        kernel.load_acceptance(accepted.acceptance_evidence_sha256)
    path.write_bytes(raw)
    lines = kernel.archive.index.read_bytes().splitlines(keepends=True)
    kernel.archive.index.write_bytes(b"".join(lines[:-1]))
    with pytest.raises(api.KernelAuthorityError):
        kernel.promotion_host.activate(accepted)


def test_train_payload_rejects_dev_and_non_json_host_objects(kernel):
    parent = kernel.active_bundle()
    candidate = child(parent)
    permit = kernel.budget.reserve_stage(kernel.stage_id(candidate), ResourceUse())
    for payload in (
        {"dev_comparison": {}},
        {"nested": {"public_ids": []}},
        {"host": kernel},
    ):
        with pytest.raises((api.KernelAuthorityError, ValueError, TypeError)):
            kernel.close_evaluation(
                parent,
                candidate,
                permit=permit,
                status="passed",
                train_objectives=payload,
                train_behavior_descriptors={},
                dev_comparison={
                    "passed": True,
                    "parent_metrics": {},
                    "candidate_metrics": {},
                },
                resource_use=ResourceUse(),
            )


def test_retrying_default_rollback_does_not_reactivate_failed_bundle(kernel):
    parent, _, _ = transition(kernel)
    kernel.promotion_host.rollback(reason="canary_failed")
    retried = kernel.promotion_host.rollback(reason="canary_failed")
    assert retried == parent
    assert len(kernel.promotion_host.history()) == 3


@pytest.mark.parametrize("field", ["active_bundle_sha256", "archive_snapshot_sha256"])
def test_resume_rejects_checkpoint_with_unrelated_identity_even_if_rehashed(
    kernel, field
):
    path = kernel.store.root / "checkpoint.json"
    checkpoint = json.loads(path.read_text())
    checkpoint[field] = "f" * 64
    checkpoint["checkpoint_sha256"] = fingerprint_payload(
        {k: v for k, v in checkpoint.items() if k != "checkpoint_sha256"}
    )
    path.write_bytes(canonical_v2_bytes(checkpoint))
    with pytest.raises(api.KernelAuthorityError, match="checkpoint"):
        api.EvolutionKernel.resume(
            kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
        )


def test_active_kernel_reconstruction_cannot_replace_charged_ledger_with_empty_one(
    kernel,
):
    transition(kernel)
    empty = BudgetLedger(kernel.budget.plan, monotonic=lambda: 0.0)
    with pytest.raises(api.KernelAuthorityError, match="budget|reservation"):
        api.EvolutionKernel(kernel.store, protocol(), empty, seed=seed())


def test_failure_before_promotion_accounts_use_and_leaves_pointer_unchanged(
    kernel, monkeypatch
):
    parent = kernel.active_bundle()

    def fail(identity, payload):
        raise OSError("injected acceptance write failure")

    monkeypatch.setattr(kernel.store, "write_acceptance", fail)
    with pytest.raises(OSError, match="acceptance write failure"):
        transition(kernel)
    assert kernel.active_bundle() == parent
    assert kernel.budget.charged_use.task_executions == 1
    assert len(kernel.promotion_host.history()) == 1


def test_stale_parent_and_consumed_permit_fail_closed(kernel):
    parent = kernel.active_bundle()
    stale_child = parent.provisional_child("decision", {"decision": "e" * 64})
    result, permit = evaluation(kernel, parent, stale_child)
    transition(kernel)
    with pytest.raises(api.KernelAuthorityError, match="Parent"):
        kernel.evaluate_transition(
            parent, stale_child, target="decision", evaluation=result, permit=permit
        )
    assert kernel.budget.charged_use.task_executions == 2
    with pytest.raises(api.KernelAuthorityError, match="permit"):
        kernel.evaluate_transition(
            parent, stale_child, target="decision", evaluation=result, permit=permit
        )
    assert kernel.budget.charged_use.task_executions == 2


def test_exact_evidence_and_closed_evaluation_schemas_reject_extra_authority(kernel):
    _, _, accepted = transition(kernel)
    evidence = kernel.load_acceptance(accepted.acceptance_evidence_sha256)
    payload = evidence.to_payload()
    payload["promote"] = True
    with pytest.raises(ValueError, match="exact schema"):
        api.AcceptanceEvidence.from_payload(payload)
    closed_path = (
        kernel.store.root
        / "evaluations"
        / evidence.candidate_bundle_sha256
        / "closed.json"
    )
    closed = json.loads(closed_path.read_text())
    closed["host"] = {}
    with pytest.raises(ValueError, match="exact schema"):
        api.ClosedEvaluation.from_payload(closed)


def test_acceptance_reader_rejects_omitted_provisional_archive_record(kernel):
    parent, candidate, _ = transition(kernel, passed=False)
    (evidence_path,) = (kernel.store.root / "acceptance").glob("*.json")
    lines = kernel.archive.index.read_bytes().splitlines(keepends=True)
    kernel.archive.index.write_bytes(b"".join(lines[:-1]))
    with pytest.raises(api.KernelAuthorityError, match="archive|indexed"):
        kernel.load_acceptance(evidence_path.stem)
    assert kernel.active_bundle() == parent


@pytest.mark.parametrize(
    "target,changes",
    [
        ("numerical", {"numerical": ("c" * 64, "d" * 64)}),
        ("decision", {"decision": "c" * 64}),
        ("joint", {"decision": "c" * 64, "retrieval": "d" * 64}),
    ],
)
def test_kernel_accepts_each_declared_mutation_scope(kernel, target, changes):
    parent = kernel.active_bundle()
    candidate = parent.provisional_child(target, changes)
    result, permit = evaluation(kernel, parent, candidate)
    accepted = kernel.evaluate_transition(
        parent, candidate, target=target, evaluation=result, permit=permit
    )
    assert accepted.generation == 1
    assert kernel.load_acceptance(accepted.acceptance_evidence_sha256).target == target
