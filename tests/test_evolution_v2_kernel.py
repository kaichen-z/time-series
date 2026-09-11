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
from evolving_loop.v2.fakes import FakeClock, run_fake_kernel, smoke_config


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
    permit = kernel.reserve_evaluation(candidate, ResourceUse(task_executions=2))
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


@pytest.mark.parametrize("passed", [True, False])
def test_closed_record_binds_dev_only_in_decision_evidence_and_resumes(kernel, passed):
    _, candidate, accepted = transition(kernel, passed=passed)
    (evidence_path,) = (kernel.store.root / "acceptance").glob("*.json")
    evidence = kernel.load_acceptance(evidence_path.stem)
    path = kernel.store.root / "evaluations" / candidate.fingerprint() / "closed.json"
    closed = json.loads(path.read_text())
    assert "dev_comparison" not in closed
    assert closed["dev_comparison_sha256"] == fingerprint_payload(
        evidence.dev_comparison
    )
    assert evidence.evaluation_sha256 == fingerprint_payload(closed)
    closure = json.loads(path.with_name("budget_closure.json").read_text())
    assert closure["evaluation_sha256"] == evidence.evaluation_sha256
    resumed = api.EvolutionKernel.resume(
        kernel.store, kernel.budget.plan, monotonic=FakeClock()
    )
    assert resumed.active_bundle() == accepted
    assert resumed.load_acceptance(evidence_path.stem) == evidence


def test_rehashed_rejected_evidence_cannot_change_dev_bound_to_closed_record(kernel):
    transition(kernel, passed=False)
    (path,) = (kernel.store.root / "acceptance").glob("*.json")
    evidence = json.loads(path.read_text())
    evidence["dev_comparison"]["candidate_metrics"]["loss"] = 0.123
    forged_sha = fingerprint_payload(evidence)
    kernel.store.write_acceptance(forged_sha, evidence)
    checkpoint = json.loads(kernel.checkpoint_path.read_text())
    checkpoint["completed_transitions"][evidence["candidate_bundle_sha256"]][
        "acceptance_evidence_sha256"
    ] = forged_sha
    rehash_checkpoint(checkpoint)
    kernel.checkpoint_path.write_bytes(canonical_v2_bytes(checkpoint))
    with pytest.raises(api.KernelAuthorityError, match="Dev|dev_comparison"):
        api.EvolutionKernel.resume(
            kernel.store, kernel.budget.plan, monotonic=FakeClock()
        )


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
    permit = kernel.reserve_evaluation(candidate, ResourceUse())
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


@pytest.mark.parametrize("operation", ["reserve", "finalize"])
def test_live_checkpoint_write_rejects_changed_body_with_stale_checksum(
    kernel, operation
):
    candidate = child(kernel.active_bundle())
    path = kernel.checkpoint_path
    checkpoint = json.loads(path.read_text())
    checkpoint["active_bundle_sha256"] = "f" * 64
    corrupted = canonical_v2_bytes(checkpoint)
    path.write_bytes(corrupted)
    with pytest.raises(api.KernelAuthorityError, match="checkpoint"):
        if operation == "reserve":
            kernel.reserve_evaluation(candidate, ResourceUse(task_executions=2))
        else:
            kernel.finalize()
    assert path.read_bytes() == corrupted


def rehash_checkpoint(payload):
    payload["checkpoint_sha256"] = fingerprint_payload(
        {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
    )


def test_resume_rejects_elapsed_rewind_even_when_both_checkpoint_layers_are_rehashed(
    tmp_path,
):
    config = smoke_config()
    run_fake_kernel(tmp_path, config, stop_after_stage="fake-numerical")
    for path in (tmp_path / "checkpoint.json", tmp_path / "kernel/checkpoint.json"):
        checkpoint = json.loads(path.read_text())
        assert checkpoint["budget"]["prior_elapsed_wall_seconds"] == 1.0
        checkpoint["budget"]["prior_elapsed_wall_seconds"] = 0.0
        rehash_checkpoint(checkpoint["budget"])
        if "checkpoint_sha256" in checkpoint:
            rehash_checkpoint(checkpoint)
        path.write_bytes(canonical_v2_bytes(checkpoint))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(api.KernelAuthorityError, match="elapsed"):
        run_fake_kernel(tmp_path, config, resume=True)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize(
    "boundary,elapsed", [("budget_before", 20.0), ("budget_after", 0.0)]
)
def test_resume_rejects_closure_elapsed_moving_backwards(tmp_path, boundary, elapsed):
    clock = FakeClock()
    plan = BudgetPlan(1000, 0.2, ResourceUse(task_executions=100))
    kernel = api.EvolutionKernel(
        V2RunStore.create(tmp_path / "run"),
        protocol(),
        BudgetLedger(plan, monotonic=clock),
        seed=seed(),
    )
    parent = kernel.active_bundle()
    candidate = child(parent)
    result, permit = evaluation(kernel, parent, candidate)
    clock.advance(10.0)
    with pytest.raises(api.KernelAuthorityError, match="permit"):
        kernel.evaluate_transition(
            parent, candidate, target="retrieval", evaluation=result
        )
    path = (
        kernel.store.root
        / "evaluations"
        / candidate.fingerprint()
        / "budget_closure.json"
    )
    closure = json.loads(path.read_text())
    closure[boundary]["prior_elapsed_wall_seconds"] = elapsed
    rehash_checkpoint(closure[boundary])
    path.write_bytes(canonical_v2_bytes(closure))
    checkpoint = json.loads(kernel.checkpoint_path.read_text())
    checkpoint["budget_closures"][permit.reservation_sha256]["closure_sha256"] = (
        fingerprint_payload(closure)
    )
    rehash_checkpoint(checkpoint)
    kernel.checkpoint_path.write_bytes(canonical_v2_bytes(checkpoint))
    before = {p: p.read_bytes() for p in kernel.store.root.rglob("*") if p.is_file()}
    with pytest.raises(api.KernelAuthorityError, match="elapsed"):
        api.EvolutionKernel.resume(kernel.store, plan, monotonic=FakeClock())
    assert {
        p: p.read_bytes() for p in kernel.store.root.rglob("*") if p.is_file()
    } == before


def test_resume_preserves_elapsed_closure_and_excludes_downtime(tmp_path, monkeypatch):
    clock = FakeClock()
    plan = BudgetPlan(1000, 0.2, ResourceUse(task_executions=100))
    kernel = api.EvolutionKernel(
        V2RunStore.create(tmp_path / "run"),
        protocol(),
        BudgetLedger(plan, monotonic=clock),
        seed=seed(),
    )
    parent = kernel.active_bundle()
    candidate = child(parent)
    result, permit = evaluation(kernel, parent, candidate, passed=False)
    clock.advance(797.0)
    close_stage = kernel.budget.close_stage

    def advancing_close(*args):
        clock.advance(2.0)
        return close_stage(*args)

    monkeypatch.setattr(kernel.budget, "close_stage", advancing_close)
    kernel.evaluate_transition(
        parent, candidate, target="retrieval", evaluation=result, permit=permit
    )
    fresh_clock = FakeClock(10000.0)
    resumed = api.EvolutionKernel.resume(kernel.store, plan, monotonic=fresh_clock)
    assert resumed.budget.elapsed_wall_seconds == 799.0
    assert resumed.budget.can_open_stage(ResourceUse()).allowed
    fresh_clock.advance(1.0)
    assert resumed.budget.elapsed_wall_seconds == 800.0
    assert resumed.budget.can_open_stage(ResourceUse()).reason == "finalization_reserve"


def test_active_kernel_reconstruction_cannot_replace_charged_ledger_with_empty_one(
    kernel,
):
    transition(kernel)
    empty = BudgetLedger(kernel.budget.plan, monotonic=lambda: 0.0)
    with pytest.raises(api.KernelAuthorityError, match="fresh|budget|reservation"):
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
    with pytest.raises(api.KernelAuthorityError, match="Parent|permit"):
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
    in_memory = api.ClosedEvaluation.from_record_payload(
        closed, dev_comparison=evidence.dev_comparison
    ).to_payload()
    in_memory["host"] = {}
    with pytest.raises(ValueError, match="exact schema"):
        api.ClosedEvaluation.from_payload(in_memory)
    closed["host"] = {}
    with pytest.raises(ValueError, match="exact schema"):
        api.ClosedEvaluation.from_record_payload(
            closed, dev_comparison=evidence.dev_comparison
        )


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


def test_kernel_seals_host_scheduler_state_without_granting_candidate_ownership(kernel):
    parent = kernel.active_bundle()
    candidate = child(parent)
    result, permit = evaluation(kernel, parent, candidate)
    scheduler_sha = "9" * 64

    accepted = kernel.evaluate_transition(
        parent,
        candidate,
        target="retrieval",
        evaluation=result,
        permit=permit,
        host_scheduler_state_sha256=scheduler_sha,
    )

    evidence = kernel.load_acceptance(accepted.acceptance_evidence_sha256)
    assert accepted.scheduler_state_sha256 == scheduler_sha
    assert evidence.scheduler_state_sha256 == scheduler_sha


def test_joint_numerical_transition_validates_and_archives_the_release_pair(
    kernel, monkeypatch
):
    parent = kernel.active_bundle()
    candidate = parent.provisional_child(
        "joint",
        {
            "numerical": ("c" * 64, "d" * 64),
            "retrieval": "e" * 64,
        },
    )
    result, permit = evaluation(kernel, parent, candidate)
    release_pair = tuple(
        sorted(
            (
                candidate.numerical_release_sha256,
                candidate.numerical_registry_sha256,
            )
        )
    )

    validated = []

    def validate_release_pair(bundle, train):
        validated.append(bundle)
        assert bundle.numerical_release_sha256 == candidate.numerical_release_sha256
        assert bundle.numerical_registry_sha256 == candidate.numerical_registry_sha256
        assert train["candidate_bundle_sha256"] == candidate.fingerprint()
        return release_pair

    monkeypatch.setattr(kernel, "_numerical_release_references", validate_release_pair)
    accepted = kernel.evaluate_transition(
        parent,
        candidate,
        target="joint",
        evaluation=result,
        permit=permit,
    )

    records = [
        json.loads(line)["record"]
        for line in kernel.archive.index.read_text().splitlines()
    ]
    accepted_record = next(
        row for row in records if row["artifact_sha256"] == accepted.fingerprint()
    )
    assert validated[0] == candidate
    assert accepted in validated[1:]
    assert accepted_record["accepted_release_sha256s"] == list(release_pair)


def test_budget_overrun_cannot_be_resealed_as_allowed_by_self_hash(kernel):
    transition(kernel, actual=3)
    (path,) = (kernel.store.root / "acceptance").glob("*.json")
    evidence = api.AcceptanceEvidence.from_payload(json.loads(path.read_text()))
    forged = replace(evidence, budget_allowed=True, decision="accept")
    kernel.store.write_acceptance(forged.fingerprint(), forged.to_payload())
    with pytest.raises(api.KernelAuthorityError, match="closure|budget|transition"):
        kernel.load_acceptance(forged.fingerprint())


@pytest.mark.parametrize("failure", ["missing", "fabricated", "copied", "pointer"])
def test_authentic_evaluation_is_accounted_once_across_early_failures(kernel, failure):
    parent = kernel.active_bundle()
    candidate = child(parent)
    result, permit = evaluation(kernel, parent, candidate)
    if failure == "missing":
        supplied = None
    elif failure == "fabricated":
        supplied = StagePermit(True, None, "f" * 64)
    elif failure == "copied":
        supplied = replace(permit)
    else:
        supplied = permit
        drift = replace(parent, numerical_release_sha256="e" * 64)
        kernel.store.publish_active_bundle(drift.to_payload())
    for _ in range(2):
        with pytest.raises(api.KernelAuthorityError):
            kernel.evaluate_transition(
                parent,
                candidate,
                target="retrieval",
                evaluation=result,
                permit=supplied,
            )
        assert kernel.budget.charged_use.task_executions == 1
        assert not kernel.budget.checkpoint()["open_reservations"]


def test_resume_rejects_legacy_raw_dev_record_from_early_failure(kernel):
    parent = kernel.active_bundle()
    candidate = child(parent)
    result, permit = evaluation(kernel, parent, candidate)
    with pytest.raises(api.KernelAuthorityError, match="permit"):
        kernel.evaluate_transition(
            parent,
            candidate,
            target="retrieval",
            evaluation=result,
            permit=None,
        )

    assert not list((kernel.store.root / "acceptance").glob("*.json"))
    closed_path = (
        kernel.store.root / "evaluations" / candidate.fingerprint() / "closed.json"
    )
    legacy_record = result.to_payload()
    assert "dev_comparison" in legacy_record
    closed_path.write_bytes(canonical_v2_bytes(legacy_record))

    closure_path = closed_path.with_name("budget_closure.json")
    closure = json.loads(closure_path.read_text())
    closure["evaluation_sha256"] = fingerprint_payload(legacy_record)
    closure_path.write_bytes(canonical_v2_bytes(closure))

    checkpoint = json.loads(kernel.checkpoint_path.read_text())
    checkpoint["budget_closures"][permit.reservation_sha256]["closure_sha256"] = (
        fingerprint_payload(closure)
    )
    rehash_checkpoint(checkpoint)
    kernel.checkpoint_path.write_bytes(canonical_v2_bytes(checkpoint))
    before = {p: p.read_bytes() for p in kernel.store.root.rglob("*") if p.is_file()}

    with pytest.raises(
        api.KernelAuthorityError, match="closed evaluation record schema"
    ):
        api.EvolutionKernel.resume(
            kernel.store, kernel.budget.plan, monotonic=FakeClock()
        )
    assert {
        p: p.read_bytes() for p in kernel.store.root.rglob("*") if p.is_file()
    } == before


def test_completed_activation_pointer_drift_is_not_recovered_as_pending(kernel):
    parent, _, _ = transition(kernel)
    kernel.store.publish_active_bundle(parent.to_payload())
    for read_active in (
        kernel.active_bundle,
        lambda: api.EvolutionKernel.resume(
            kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
        ),
    ):
        with pytest.raises(api.KernelAuthorityError, match="pointer|committed"):
            read_active()
        assert (
            json.loads((kernel.store.root / "accepted_bundle.json").read_text())
            == parent.to_payload()
        )


@pytest.mark.parametrize(
    "missing", ["run_manifest.json", "budget_plan.json", "checkpoint.json"]
)
def test_established_authority_files_are_never_reconstructed(kernel, missing):
    transition(kernel, passed=False)
    path = kernel.store.root / missing
    path.unlink()
    with pytest.raises(api.KernelAuthorityError):
        api.EvolutionKernel(kernel.store, protocol(), kernel.budget, seed=seed())
    assert not path.exists()
    with pytest.raises(api.KernelAuthorityError):
        api.EvolutionKernel.resume(
            kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
        )
    assert not path.exists()


@pytest.mark.parametrize("artifact", ["train", "closed", "evidence"])
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_resume_verifies_completed_rejected_transition_artifacts(
    kernel, artifact, damage
):
    _, candidate, _ = transition(kernel, passed=False)
    if artifact == "evidence":
        (path,) = (kernel.store.root / "acceptance").glob("*.json")
    else:
        path = (
            kernel.store.root
            / "evaluations"
            / candidate.fingerprint()
            / f"{artifact}.json"
        )
    if damage == "missing":
        path.unlink()
    else:
        payload = json.loads(path.read_text())
        payload["unexpected"] = "tampered"
        path.write_bytes(canonical_v2_bytes(payload))
    with pytest.raises(api.KernelAuthorityError):
        api.EvolutionKernel.resume(
            kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
        )


@pytest.mark.parametrize("damage", ["evidence", "object"])
@pytest.mark.parametrize("reason", ["integrity_failed", "safety_failed"])
def test_integrity_rollback_restores_verified_parent_despite_corrupted_source(
    kernel, damage, reason
):
    parent, _, accepted = transition(kernel)
    if damage == "evidence":
        path = (
            kernel.store.root
            / "acceptance"
            / f"{accepted.acceptance_evidence_sha256}.json"
        )
    else:
        path = kernel.archive.objects / f"{accepted.fingerprint()}.json"
    path.write_bytes(b"corrupt\n")
    restored = kernel.promotion_host.rollback(reason=reason)
    assert restored == parent
    assert kernel.active_bundle() == parent
    checkpoint = json.loads((kernel.store.root / "checkpoint.json").read_text())
    assert checkpoint["terminal_recovery"]["status"] == "recovered_requires_new_epoch"
    assert (
        checkpoint["terminal_recovery"]["invalidated_bundle_sha256"]
        == accepted.fingerprint()
    )
    with pytest.raises(api.KernelAuthorityError, match="new epoch"):
        api.EvolutionKernel.resume(
            kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
        )
    with pytest.raises(api.KernelAuthorityError, match="new epoch"):
        kernel.reserve_evaluation(child(parent), ResourceUse())
    assert path.read_bytes() == b"corrupt\n"


@pytest.mark.parametrize("damage", ["parent", "index"])
def test_integrity_rollback_never_skips_destination_or_index_corruption(kernel, damage):
    parent, _, accepted = transition(kernel)
    (kernel.archive.objects / f"{accepted.fingerprint()}.json").write_bytes(
        b"corrupt\n"
    )
    path = (
        kernel.archive.index
        if damage == "index"
        else kernel.archive.objects / f"{parent.fingerprint()}.json"
    )
    path.write_bytes(b"corrupt\n")
    with pytest.raises(api.KernelAuthorityError):
        kernel.promotion_host.rollback(reason="integrity_failed")
    assert (
        json.loads((kernel.store.root / "accepted_bundle.json").read_text())
        == accepted.to_payload()
    )


def test_copied_live_permit_cannot_mint_host_evaluation(kernel):
    parent = kernel.active_bundle()
    candidate = child(parent)
    result, original = evaluation(kernel, parent, candidate)
    with pytest.raises(api.KernelAuthorityError, match="permit"):
        kernel.evaluate_transition(
            parent,
            candidate,
            target="retrieval",
            evaluation=result,
            permit=StagePermit(True, None, original.reservation_sha256),
        )


def test_resume_with_open_reservation_fails_closed_and_never_reissues_permit(kernel):
    candidate = child(kernel.active_bundle())
    permit = kernel.reserve_evaluation(candidate, ResourceUse(task_executions=2))
    assert permit.allowed
    with pytest.raises(api.KernelAuthorityError, match="open reservation"):
        api.EvolutionKernel.resume(
            kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
        )


def test_wrong_issued_permit_cannot_redirect_actual_accounting(kernel):
    parent = kernel.active_bundle()
    first = parent.provisional_child("decision", {"decision": "c" * 64})
    _, first_permit = evaluation(kernel, parent, first)
    second = child(parent)
    second_eval, _ = evaluation(kernel, parent, second, actual=2)
    with pytest.raises(api.KernelAuthorityError):
        kernel.evaluate_transition(
            parent,
            second,
            target="retrieval",
            evaluation=second_eval,
            permit=first_permit,
        )
    assert kernel.budget.charged_use.task_executions == 2
    open_stages = kernel.budget.checkpoint()["open_reservations"]
    assert [stage["stage_id"] for stage in open_stages] == [kernel.stage_id(first)]


@pytest.mark.parametrize("failed_stage", ["train", "closed", "budget_closure"])
def test_accounting_retries_persist_original_closure_after_artifact_failure(
    kernel, monkeypatch, failed_stage
):
    parent = kernel.active_bundle()
    candidate = child(parent)
    result, permit = evaluation(kernel, parent, candidate)
    write = kernel.store.write_evaluation
    failed = False

    def fail_once(identity, stage, payload):
        nonlocal failed
        if stage == failed_stage and not failed:
            failed = True
            raise OSError("one-shot evaluation artifact failure")
        return write(identity, stage, payload)

    monkeypatch.setattr(kernel.store, "write_evaluation", fail_once)
    with pytest.raises(OSError, match="one-shot"):
        kernel.evaluate_transition(
            parent, candidate, target="retrieval", evaluation=result, permit=permit
        )
    closure_path = (
        kernel.store.root
        / "evaluations"
        / candidate.fingerprint()
        / "budget_closure.json"
    )
    closure_bytes = closure_path.read_bytes()
    closure = json.loads(closure_bytes)
    assert closure["allowed"] is True
    assert closure["reason"] is None
    assert (
        closure["budget_before"]["open_reservations"][0]["reservation_sha256"]
        == permit.reservation_sha256
    )
    assert closure["budget_after"]["charged_use"]["task_executions"] == 1
    with pytest.raises(api.KernelAuthorityError, match="permit"):
        kernel.evaluate_transition(
            parent, candidate, target="retrieval", evaluation=result, permit=permit
        )
    assert closure_path.read_bytes() == closure_bytes
    resumed = api.EvolutionKernel.resume(
        kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
    )
    assert resumed.budget.charged_use.task_executions == 1
    assert resumed.active_bundle() == parent


def test_unpersisted_accounting_closure_leaves_durable_open_reservation(
    kernel, monkeypatch
):
    parent = kernel.active_bundle()
    candidate = child(parent)
    result, permit = evaluation(kernel, parent, candidate)

    def fail(*args):
        raise OSError("persistent artifact failure")

    monkeypatch.setattr(kernel.store, "write_evaluation", fail)
    with pytest.raises(OSError):
        kernel.evaluate_transition(
            parent, candidate, target="retrieval", evaluation=result, permit=permit
        )
    assert kernel.budget.charged_use.task_executions == 1
    checkpoint = json.loads((kernel.store.root / "checkpoint.json").read_text())
    assert (
        checkpoint["budget"]["open_reservations"][0]["reservation_sha256"]
        == permit.reservation_sha256
    )
    with pytest.raises(api.KernelAuthorityError, match="open reservation"):
        api.EvolutionKernel.resume(
            kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
        )


@pytest.mark.parametrize("boundary", ["history", "pointer"])
def test_pending_checkpoint_failure_retry_is_durable_before_publication_and_resume(
    kernel, monkeypatch, boundary
):
    parent = kernel.active_bundle()
    write_checkpoint = kernel.store.write_checkpoint
    failed = False

    def fail_pending_once(payload):
        nonlocal failed
        if payload["pending_publication"] is not None and not failed:
            failed = True
            raise OSError("intent checkpoint failure")
        return write_checkpoint(payload)

    monkeypatch.setattr(kernel.store, "write_checkpoint", fail_pending_once)
    with pytest.raises(OSError, match="intent checkpoint"):
        transition(kernel)
    checkpoint_path = kernel.store.root / "checkpoint.json"
    assert json.loads(checkpoint_path.read_text())["pending_publication"] is None
    assert len(kernel.promotion_host.history()) == 1
    append = kernel.store.append_promotion
    publish = kernel.store.publish_active_bundle

    def append_at_boundary(event):
        checkpoint = json.loads(checkpoint_path.read_text())
        assert checkpoint["pending_publication"] == event
        result = append(event)
        if boundary == "history":
            raise OSError("simulated crash after history")
        return result

    def publish_at_boundary(payload):
        checkpoint = json.loads(checkpoint_path.read_text())
        assert checkpoint["pending_publication"][
            "bundle_sha256"
        ] == fingerprint_payload(payload)
        raise OSError("simulated crash before pointer")

    monkeypatch.setattr(kernel.store, "append_promotion", append_at_boundary)
    monkeypatch.setattr(kernel.store, "publish_active_bundle", publish_at_boundary)
    with pytest.raises(OSError, match="simulated crash"):
        kernel.active_bundle()
    assert (
        json.loads((kernel.store.root / "accepted_bundle.json").read_text())
        == parent.to_payload()
    )
    monkeypatch.setattr(kernel.store, "append_promotion", append)
    monkeypatch.setattr(kernel.store, "publish_active_bundle", publish)
    resumed = api.EvolutionKernel.resume(
        kernel.store, kernel.budget.plan, monotonic=lambda: 0.0
    )
    assert resumed.active_bundle().generation == 1
    assert len(resumed.promotion_host.history()) == 2


def test_repeated_intent_checkpoint_failure_never_advances_publication(
    kernel, monkeypatch
):
    parent = kernel.active_bundle()
    write = kernel.store.write_checkpoint

    def fail_pending(payload):
        if payload["pending_publication"] is not None:
            raise OSError("intent persistence unavailable")
        return write(payload)

    monkeypatch.setattr(kernel.store, "write_checkpoint", fail_pending)
    with pytest.raises(OSError):
        transition(kernel)
    for _ in range(2):
        with pytest.raises(OSError):
            kernel.active_bundle()
        assert len(kernel.promotion_host.history()) == 1
        assert (
            json.loads((kernel.store.root / "accepted_bundle.json").read_text())
            == parent.to_payload()
        )
