"""Durable QD authority rejects changed bytes and incomplete operations."""
from dataclasses import replace
import builtins
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import tempfile

import pytest

from evolving_loop.v2 import store as core_store
from evolving_loop.v2.budget import BudgetLedger, ResourceUse
from evolving_loop.v2.kernel import EvolutionKernel
from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
from evolving_loop.v2.numerical_qd import persistence as api
from evolving_loop.v2.numerical_qd.contracts import (
    HyperbandBracketV2, HyperbandBudgetOutcomeV2, HyperbandStateV2, HyperbandTaskResultV2,
    NumericalGenomeV2, NumericalInventoryV2, NumericalMemberV2,
    NumericalQDCheckpointV2, NumericalQDEntryV2, NumericalMutationPolicyV2,
    NumericalProposerPromptV2,
)
from evolving_loop.v2.numerical_qd.hyperband import advance_hyperband, evaluation_cache_key
from evolving_loop.v2.numerical_qd.map_elites import NumericalQDArchive
from tests.test_evolution_v2_kernel import kernel, child as kernel_child
from tests.test_evolution_v2_numerical_hyperband import (
    CELL, SPLIT, PROTOCOL, METRIC, DESCRIPTOR, ADAPTER, RUNTIME,
    evaluation, manifest, sha,
)
from tests.test_evolution_v2_numerical_mutation import parent_state
from tests.test_evolution_v2_numerical_config import valid_config_payload
from tests.test_package_numerical_evolution import _recipe


SOURCE = b'def seasonal_naive(history, horizon, frequency):\n    """Forecast the final level."""\n    return [float(history[-1])] * horizon\n'
SOURCE_SHA = hashlib.sha256(SOURCE).hexdigest()


def persist(store, value):
    payload = value.to_payload() if hasattr(value, "to_payload") else value
    identity = payload.get("checkpoint_sha256", fingerprint_payload(payload))
    store.write_object(identity, payload)
    return identity


def test_create_uses_the_same_repo_output_allowlist(kernel, monkeypatch):
    # Treat the already valid Kernel root as a repository root. A Numerical
    # subtree must not be created there, even though all Kernel files are valid.
    monkeypatch.setattr(api, "__file__", str(kernel.store.root / "evolving_loop/v2/numerical_qd/persistence.py"))
    before = files(kernel.store.root)
    with pytest.raises(ValueError):
        api.NumericalQDRunStore.create(kernel.store.root)
    assert files(kernel.store.root) == before
    assert not (kernel.store.root / "numerical_qd").exists()


@pytest.mark.parametrize("fault", ["payload", "resource_payload", "invalid_sha", "passed_status", "task_payload"])
def test_live_partial_writer_rejects_unpaid_unpersisted_content(tmp_path, fault):
    from evolving_loop.v2.numerical_qd.runner import _MaterialAccounting, _KernelWork, _persist
    from evolving_loop.v2.numerical_qd.artifacts import ArtifactKindV2
    from tests.test_evolution_v2_kernel import protocol, seed
    from evolving_loop.v2.budget import BudgetPlan
    plan = BudgetPlan(1000, 0.2, ResourceUse(wall_seconds=800.0, task_executions=100, artifact_bytes=1000000))
    kernel = EvolutionKernel(core_store.V2RunStore.create(tmp_path / "run"), protocol(),
        BudgetLedger(plan, monotonic=lambda: 0.0), seed=seed())
    store, args, kernel, _, state = world.__wrapped__(kernel)
    _MaterialAccounting(store, kernel)
    fixed = manifest()
    _persist(store, fixed)
    work = _KernelWork(kernel, kernel.active_bundle(), 1)
    permit = work.reserve_stage("partial-test", ResourceUse(task_executions=1))
    work.close_stage(permit, ResourceUse(task_executions=1), status="failed")
    budget = json.loads(kernel.checkpoint_path.read_bytes())["budget"]
    outcome = HyperbandBudgetOutcomeV2("failed", "artifact_bytes_exhausted", permit.reservation_sha256,
        ResourceUse(task_executions=1).to_payload(), budget["checkpoint_sha256"], budget)
    task = fixed.tasks[0]
    row = {"task_sha256": task.task_sha256, "task_id": task.task_id,
        "candidate_sha256": args["active_genome_sha256"], "cache_key": "a" * 64, "result_sha256": "b" * 64,
        "status": "persistence_denied", "resource_use": ResourceUse(task_executions=1).to_payload()}
    if fault == "payload":
        row["payload"] = "NEVER_STORE_UNPAID_PAYLOAD"
    elif fault == "resource_payload":
        row["resource_use"]["payload"] = "NEVER_STORE_UNPAID_PAYLOAD"
    elif fault == "invalid_sha":
        row["result_sha256"] = "NEVER_STORE_UNPAID_PAYLOAD"
    elif fault == "passed_status":
        row["status"] = "passed"
    else:
        row["task_id"] = {"payload": "NEVER_STORE_UNPAID_PAYLOAD"}
    value = {"closed_partial_rung": {"state": state.to_payload(), "manifest_sha256": fixed.fingerprint(),
        "reason": "artifact_bytes_exhausted", "task_results": [], "unpersisted_task_results": [row],
        "budget_outcome": outcome.to_payload()}}
    before = files(store.root)
    with pytest.raises(ValueError):
        store.write_object(fingerprint_payload(value), value, kind=ArtifactKindV2.PARTIAL_RUNG)
    assert files(store.root) == before
    assert all(b"NEVER_STORE_UNPAID_PAYLOAD" not in data for data in files(store.root).values())


@pytest.mark.parametrize("fault", ["missing", "unpaid", "mismatch", "noncanonical", "directory", "symlink", "paid"])
def test_live_partial_requires_exact_prepaid_task_result(tmp_path, fault):
    from evolving_loop.v2.numerical_qd.runner import _MaterialAccounting, _KernelWork, _persist
    from evolving_loop.v2.numerical_qd.artifacts import ArtifactKindV2
    from tests.test_evolution_v2_kernel import protocol, seed
    from evolving_loop.v2.budget import BudgetPlan
    plan = BudgetPlan(1000, 0.2, ResourceUse(wall_seconds=800.0, task_executions=100, artifact_bytes=1000000))
    kernel = EvolutionKernel(core_store.V2RunStore.create(tmp_path / "run"), protocol(),
        BudgetLedger(plan, monotonic=lambda: 0.0), seed=seed())
    store, args, kernel, _, state = world.__wrapped__(kernel)
    _MaterialAccounting(store, kernel)
    fixed = manifest()
    _persist(store, fixed)
    task = fixed.tasks[0]
    evaluated = evaluation(args["active_genome_sha256"], (task.task_id,), subset=task.fingerprint())
    key = evaluation_cache_key(evaluated.genome_sha256, task.task_sha256, SPLIT,
        METRIC, DESCRIPTOR, RUNTIME, PROTOCOL, ADAPTER)
    result = HyperbandTaskResultV2(evaluated.genome_sha256, task.task_id, key, "passed", evaluated, False, None)
    path = store.directory / f"results/{result.candidate_sha256}/{task.task_sha256}.json"
    if fault == "unpaid":
        core_store.write_once_json(path, result.to_payload())
    elif fault != "missing":
        store.write_task_result(task.task_sha256, result)
    if fault == "mismatch":
        result = replace(result, evaluation=replace(evaluated,
            objectives=replace(evaluated.objectives, mean_raw_joint_error=987654321.125)))
    elif fault == "noncanonical":
        path.write_text(json.dumps(result.to_payload(), indent=2))
    elif fault in {"directory", "symlink"}:
        path.unlink()
        if fault == "directory":
            path.mkdir()
        else:
            path.symlink_to(store.directory / f"objects/{fixed.fingerprint()}.json")
    work = _KernelWork(kernel, kernel.active_bundle(), 1)
    permit = work.reserve_stage("partial-exact-result", ResourceUse(task_executions=1))
    work.close_stage(permit, ResourceUse(task_executions=1), status="failed")
    checkpoint = json.loads(kernel.checkpoint_path.read_bytes())
    budget = checkpoint["budget"]
    outcome = HyperbandBudgetOutcomeV2("failed", "timeout", permit.reservation_sha256,
        ResourceUse(task_executions=1).to_payload(), budget["checkpoint_sha256"], budget)
    value = {"closed_partial_rung": {"state": state.to_payload(), "manifest_sha256": fixed.fingerprint(),
        "reason": "timeout", "task_results": [{"task_sha256": task.task_sha256, "result": result.to_payload()}],
        "unpersisted_task_results": [], "budget_outcome": outcome.to_payload()}}
    before = files(store.root)
    if fault == "paid":
        _persist(store, value, kind=ArtifactKindV2.PARTIAL_RUNG)
        assert json.loads(path.read_bytes()) == result.to_payload()
        _persist(store, checkpoint, kind=ArtifactKindV2.KERNEL_CHECKPOINT)
        _persist(store, budget, kind=ArtifactKindV2.BUDGET_CHECKPOINT)
        updated = args | {"kernel_checkpoint_sha256": checkpoint["checkpoint_sha256"],
                          "budget_checkpoint_sha256": budget["checkpoint_sha256"]}
        store.write_state(**updated)
        closed = files(store.root)
        resume(store, updated)
        assert files(store.root) == closed
    else:
        with pytest.raises(ValueError):
            _persist(store, value, kind=ArtifactKindV2.PARTIAL_RUNG)
        assert files(store.root) == before
        assert all(b"987654321.125" not in data for data in files(store.root).values())


def test_kernel_material_receipt_rejects_payload_before_control_publication(kernel):
    parent = kernel.active_bundle()
    child = parent.provisional_child("numerical", {"numerical": ("a" * 64, "b" * 64)})
    permit = kernel.reserve_evaluation(child, ResourceUse())
    before = files(kernel.store.root)
    with pytest.raises(ValueError):
        kernel.close_evaluation(parent, child, permit=permit, status="passed",
            train_objectives={"material_receipts": [{"kind": "task_result", "relative_path": "results/result.json",
                "content_sha256": "a" * 64, "size_bytes": 1, "payload": "MATERIAL_RECEIPT_SENTINEL"}]},
            train_behavior_descriptors={}, dev_comparison={"passed": False, "parent_metrics": {}, "candidate_metrics": {}},
            resource_use=ResourceUse(), account_only=True)
    assert files(kernel.store.root) == before
    assert all(b"MATERIAL_RECEIPT_SENTINEL" not in raw for raw in files(kernel.store.root).values())


@pytest.fixture
def world(kernel):
    store = api.NumericalQDRunStore.create(kernel.store.root)
    store.write_source(SOURCE_SHA, SOURCE)
    recipe = _recipe().to_payload()
    policy_sha = persist(store, recipe)
    member = NumericalMemberV2("seed", "statistical", SOURCE_SHA, policy_sha, (), (), "active")
    state = parent_state(inventory=NumericalInventoryV2(1, (member,)))
    inventory_sha = persist(store, state.inventory)
    policy_sha = persist(store, state.mutation_policy)
    prompt_sha = persist(store, state.proposer_prompt)
    config = valid_config_payload()
    config.update(profile="smoke", seed=42, kernel_protocol=kernel.protocol.to_payload())
    config["budget"] = {key: value for key, value in kernel.budget.plan.to_payload().items()
                        if key in {"hard_limit_seconds", "finalization_reserve_fraction", "ceilings"}}
    config["hyperband"]["reduction_factor"] = 3
    config_sha = persist(store, config)
    input_sha = persist(store, {"input": "frozen"})
    screen_sha = persist(store, {"screen": []})
    combined_sha = persist(store, {"policies": []})
    genome = NumericalGenomeV2(1, 0, (), "add", inventory_sha, screen_sha,
        combined_sha, policy_sha, prompt_sha, RUNTIME, PROTOCOL)
    genome_sha = persist(store, genome)
    archive = NumericalQDArchive()
    hyperband = HyperbandStateV2(HyperbandBracketV2.registered("explore"),
        (genome_sha,), 3, SPLIT, PROTOCOL, ())
    kernel_payload = json.loads(kernel.checkpoint_path.read_bytes())
    budget_sha = persist(store, kernel_payload["budget"])
    args = dict(config_sha256=config_sha, input_sha256s={"seed": input_sha},
        active_bundle_sha256=persist(store, kernel.active_bundle()), active_genome_sha256=genome_sha,
        qd_snapshot_sha256=persist(store, archive), mutation_policy_sha256=policy_sha,
        proposer_prompt_sha256=prompt_sha, hyperband_state_sha256=persist(store, hyperband),
        counter={"seed": 42, "stream": "numerical", "counter": 0},
        kernel_checkpoint_sha256=kernel_payload["checkpoint_sha256"], budget_checkpoint_sha256=budget_sha)
    return store, args, kernel, archive, hyperband


def resume(store, args):
    return store.resume(**{key: args[key] for key in (
        "config_sha256", "input_sha256s", "kernel_checkpoint_sha256", "budget_checkpoint_sha256")})


def files(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def add_rung(world):
    store, args, kernel, archive, state = world
    fixed = manifest()
    persist(store, fixed)
    for task in fixed.tasks:
        row = evaluation(args["active_genome_sha256"], (task.task_id,), subset=task.fingerprint())
        key = evaluation_cache_key(row.genome_sha256, task.task_sha256, SPLIT,
            METRIC, DESCRIPTOR, RUNTIME, PROTOCOL, ADAPTER)
        store.write_task_result(task.task_sha256, HyperbandTaskResultV2(
            row.genome_sha256, task.task_id, key, "passed", row, False, None))
    aggregate = evaluation(args["active_genome_sha256"], fixed.task_ids, subset=fixed.fingerprint())
    parent = kernel.active_bundle()
    candidate = kernel_child(parent)
    reservation = kernel.reserve_evaluation(candidate, ResourceUse(task_executions=8))
    closed = kernel.close_evaluation(parent, candidate, permit=reservation, status="passed",
        train_objectives={"loss": 1.0}, train_behavior_descriptors={"family": "statistical"},
        dev_comparison={"passed": False, "parent_metrics": {"loss": 1.0}, "candidate_metrics": {"loss": 2.0}},
        resource_use=ResourceUse(task_executions=8))
    # Real issuer-owned accounting and rejected transition keep the active seed.
    kernel.evaluate_transition(parent, candidate, target="retrieval", evaluation=closed, permit=reservation)
    kernel_payload = json.loads(kernel.checkpoint_path.read_bytes())
    captured = kernel_payload["budget"]
    outcome = HyperbandBudgetOutcomeV2("completed", None, reservation.reservation_sha256,
        ResourceUse(task_executions=8).to_payload(), captured["checkpoint_sha256"], captured)
    advanced = advance_hyperband(state, fixed, (aggregate,), outcome).state
    rung = advanced.rungs[-1]
    store.write_rung(rung)
    entry = NumericalQDEntryV2(1, aggregate.genome_sha256, aggregate.fingerprint(), CELL,
        fixed.task_ids, aggregate.objectives, aggregate.constraints, ())
    store.append_qd_entry(entry)
    archive = archive.insert((entry,))
    updated = args | dict(qd_snapshot_sha256=persist(store, archive),
        hyperband_state_sha256=persist(store, advanced),
        kernel_checkpoint_sha256=kernel_payload["checkpoint_sha256"],
        budget_checkpoint_sha256=outcome.ledger_checkpoint_sha256)
    return updated, rung, entry


def test_exact_layout_and_completed_resume_are_read_only(world):
    store, args, *_ = world
    args, rung, entry = add_rung(world)
    checkpoint = store.write_state(**args)
    before = files(store.root)
    assert resume(store, args) == checkpoint
    assert files(store.root) == before
    assert checkpoint.checkpoint_sha256 == fingerprint_payload({
        key: value for key, value in checkpoint.to_payload().items() if key != "checkpoint_sha256"})
    assert (store.root / "numerical_qd/manifest.json").is_file()
    assert (store.root / "numerical_qd/checkpoint.json").is_file()
    assert (store.root / f"numerical_qd/rungs/{rung.fingerprint()}.json").is_file()
    assert not (store.root / "numerical_qd/evaluation_complete.json").exists()
    before_log = (store.root / "numerical_qd/archive/entries.jsonl").read_bytes()
    store.append_qd_entry(entry)
    store.write_rung(rung)
    assert (store.root / "numerical_qd/archive/entries.jsonl").read_bytes() == before_log


@pytest.mark.parametrize("change", ["tasks", "objectives", "cell"])
def test_qd_entry_rejects_evidence_not_equal_to_committed_cell_evaluation(world, change):
    store, *_ = world
    _, _, entry = add_rung(world)
    if change == "tasks":
        bad = replace(entry, task_ids=entry.task_ids[:1])
    elif change == "objectives":
        bad = replace(entry, objectives=replace(entry.objectives, mean_capped_smae=0.0))
    else:
        bad = replace(entry, cell=replace(entry.cell, trend="high"))
    with pytest.raises(ValueError, match="evaluation"):
        store.append_qd_entry(bad)


@pytest.mark.parametrize("value", ["../escape", "/tmp/escape", "a" * 63, "A" * 64])
def test_write_rejects_path_traversal_and_invalid_sha(world, value):
    store, *_ = world
    with pytest.raises(ValueError): store.write_object(value, {"x": 1})
    with pytest.raises(ValueError): store.write_source(value, SOURCE)


def test_wrong_hash_noncanonical_input_and_source_digest_are_rejected(world):
    store, *_ = world
    with pytest.raises(ValueError): store.write_object("f" * 64, {"x": 1})
    with pytest.raises(ValueError): store.write_object(sha('bad'), b'{ "x": 1 }')
    with pytest.raises(ValueError): store.write_source(SOURCE_SHA, SOURCE + b"\n")


def test_immutable_retry_never_repairs_changed_source_or_object(world):
    store, args, *_ = world
    path = store.root / f"numerical_qd/sources/{SOURCE_SHA}.py"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError): store.write_source(SOURCE_SHA, SOURCE)
    assert path.read_bytes() == b"changed"
    object_path = store.root / f'numerical_qd/objects/{args["config_sha256"]}.json'
    object_path.write_bytes(b'{"profile":"changed"}\n')
    with pytest.raises(ValueError): store.write_object(args["config_sha256"], {"profile": "test"})


@pytest.mark.parametrize("target", ["source", "policy", "genome", "task", "rung", "budget", "archive", "hyperband"])
def test_resume_rejects_missing_or_changed_referenced_bytes(world, target):
    store, args, *_ = world
    args, rung, _ = add_rung(world)
    store.write_state(**args)
    subtree = store.root / "numerical_qd"
    paths = {"source": subtree / f"sources/{SOURCE_SHA}.py",
        "policy": subtree / f'objects/{args["mutation_policy_sha256"]}.json',
        "genome": subtree / f'objects/{args["active_genome_sha256"]}.json',
        "task": next((subtree / "results").rglob("*.json")),
        "rung": subtree / f"rungs/{rung.fingerprint()}.json",
        "budget": subtree / f'objects/{args["budget_checkpoint_sha256"]}.json',
        "archive": subtree / "archive/entries.jsonl",
        "hyperband": subtree / f'objects/{args["hyperband_state_sha256"]}.json'}
    path = paths[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError): resume(store, args)
    path.unlink()
    with pytest.raises(ValueError): resume(store, args)


@pytest.mark.parametrize("field", ["config_sha256", "input_sha256s", "kernel_checkpoint_sha256", "budget_checkpoint_sha256"])
def test_resume_cross_binds_supplied_identities(world, field):
    store, args, *_ = world
    store.write_state(**args)
    changed = args | {field: {"seed": "f" * 64} if field == "input_sha256s" else "f" * 64}
    with pytest.raises(ValueError): resume(store, changed)


@pytest.mark.parametrize("rehash", [False, True])
def test_changed_counter_fails_against_immutable_checkpoint_even_if_rehashed(world, rehash):
    store, args, *_ = world
    store.write_state(**args)
    path = store.root / "numerical_qd/checkpoint.json"
    payload = json.loads(path.read_bytes())
    payload["counter"]["counter"] += 1
    if rehash:
        payload["checkpoint_sha256"] = fingerprint_payload({k: v for k, v in payload.items() if k != "checkpoint_sha256"})
    path.write_bytes(canonical_v2_bytes(payload))
    with pytest.raises(ValueError): resume(store, args)


def test_only_unreferenced_hidden_atomic_temp_may_be_ignored(world):
    store, args, *_ = world
    store.write_state(**args)
    path = store.root / "numerical_qd/objects/.checkpoint.json.crash.tmp"
    path.write_bytes(b"partial")
    assert resume(store, args).counter["counter"] == 0
    path.rename(path.with_name("orphan.json"))
    with pytest.raises(ValueError): resume(store, args)


def test_no_missing_policy_ancestry_or_sources_before_evaluation(world):
    store, args, *_ = world
    objects = store.root / "numerical_qd/objects"
    genome = json.loads((objects / f'{args["active_genome_sha256"]}.json').read_bytes())
    inventory = json.loads((objects / f'{genome["inventory_sha256"]}.json').read_bytes())
    member = inventory["members"][0]
    (store.root / f'numerical_qd/objects/{member["policy_sha256"]}.json').unlink()
    with pytest.raises(ValueError): store.verify_candidate(args["active_genome_sha256"])
    with pytest.raises(ValueError): store.write_state(**args)


def test_partial_rung_and_extra_uncommitted_task_cannot_be_checkpointed(world):
    store, args, *_ = world
    store.write_state(**args)
    persist(store, manifest())
    with pytest.raises(ValueError): resume(store, args)
    with pytest.raises(ValueError): store.write_state(**args)


def test_hash_chain_rejects_prefix_reordering_and_torn_append(world):
    store, args, *_ = world
    args, _, entry = add_rung(world)
    from evolving_loop.v2.numerical_qd.contracts import NumericalEvaluationV2
    original = NumericalEvaluationV2.from_payload(store._object(entry.evaluation_sha256))
    changed = replace(original, cells=(replace(CELL, trend="high"),))
    persist(store, changed)
    second = replace(entry, evaluation_sha256=changed.fingerprint(), cell=changed.cells[0])
    store.append_qd_entry(second)
    path = store.root / "numerical_qd/archive/entries.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    assert len(lines) == 2
    path.write_bytes(b"".join(reversed(lines)))
    with pytest.raises(ValueError): store.append_qd_entry(entry)
    path.write_bytes(b"".join(lines) + b'{"partial":')
    with pytest.raises(ValueError): store.append_qd_entry(entry)


def test_exact_budget_snapshot_is_persisted_from_task6_outcome(world):
    store, args, *_ = world
    args, rung, _ = add_rung(world)
    path = store.root / f'numerical_qd/objects/{args["budget_checkpoint_sha256"]}.json'
    assert path.read_bytes() == canonical_v2_bytes(rung.budget_outcome.to_payload()["ledger_checkpoint"])


def test_root_completion_requires_finalized_kernel_and_is_write_once(world):
    store, args, kernel, *_ = world
    store.write_state(**args)
    with pytest.raises(ValueError): store.write_completion({"status": "numerical_qd_complete"})
    kernel.finalize()
    payload = json.loads(kernel.checkpoint_path.read_bytes())
    args = args | dict(kernel_checkpoint_sha256=payload["checkpoint_sha256"],
        budget_checkpoint_sha256=persist(store, payload["budget"]))
    store.write_state(**args)
    path = store.write_completion({"status": "numerical_qd_complete"})
    assert path == store.root / "evaluation_complete.json"
    with pytest.raises(ValueError): store.write_completion({"status": "numerical_qd_complete", "other": True})
    assert not (store.root / "numerical_qd/evaluation_complete.json").exists()


def test_create_refuses_legacy_source_and_symlink_destinations(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "checkpoint.json").write_text('{}')
    with pytest.raises(ValueError): api.NumericalQDRunStore.create(legacy)
    source = Path(__file__).resolve().parents[1] / "numerical_agent" / "task8-forbidden-run"
    with pytest.raises(ValueError): api.NumericalQDRunStore.create(source)
    assert not source.exists()
    linked = tmp_path / "linked"
    linked.symlink_to(legacy, target_is_directory=True)
    with pytest.raises(ValueError): api.NumericalQDRunStore.create(linked)


def test_replaced_directory_symlink_never_writes_outside_run(world, tmp_path):
    store, *_ = world
    destination = store.root / "numerical_qd/proposals"
    destination.rmdir()
    destination.symlink_to(tmp_path, target_is_directory=True)
    payload = {"x": 1}
    with pytest.raises(ValueError): store.write_proposal_attempt(fingerprint_payload(payload), payload)
    assert not (tmp_path / f"{fingerprint_payload(payload)}.json").exists()


def test_failure_after_each_persistence_step_never_resumes_partial_authority(world, monkeypatch):
    store, args, *_ = world
    baseline = files(store.root)
    # Count actual durable writes, then crash after each one in a fresh replica.
    writes = []
    real = core_store._atomic_write
    def record(path, data):
        real(path, data)
        writes.append(path)
    with monkeypatch.context() as patch:
        patch.setattr(core_store, "_atomic_write", record)
        store.write_state(**args)
    assert len(writes) >= 3  # exact Kernel snapshot, immutable checkpoint, pointer
    import shutil
    for step in range(1, len(writes) + 1):
        root = store.root.parent / f"crash-{step}"
        shutil.copytree(store.root, root)
        for path in tuple(root.rglob("*")):
            if path.is_file() and str(path.relative_to(root)) not in baseline:
                path.unlink()
        for name, data in baseline.items():
            (root / name).write_bytes(data)
        replica = api.NumericalQDRunStore(root)
        count = 0
        def crash(path, data):
            nonlocal count
            real(path, data)
            count += 1
            if count == step: raise OSError("injected power loss")
        with monkeypatch.context() as patch:
            patch.setattr(core_store, "_atomic_write", crash)
            with pytest.raises(OSError, match="power loss"): replica.write_state(**args)
        if step == len(writes):
            assert resume(replica, args).counter["counter"] == 0
        else:
            with pytest.raises(ValueError): resume(replica, args)


def test_checkpoint_contract_is_closed_and_deeply_immutable(world):
    store, args, *_ = world
    checkpoint = store.write_state(**args)
    with pytest.raises(TypeError): checkpoint.counter["counter"] = 2
    with pytest.raises(TypeError): checkpoint.completed_operation_sha256s["../../bad"] = "a" * 64
    with pytest.raises(ValueError): NumericalQDCheckpointV2.from_payload(checkpoint.to_payload() | {"open_rung": {}})


def test_identical_state_retry_is_byte_identical_and_creates_no_operation(world):
    store, args, *_ = world
    first = store.write_state(**args)
    before = files(store.root)
    assert store.write_state(**args) == first
    assert files(store.root) == before


@pytest.mark.parametrize("name", ["accepted_bundle.json", "archive/index.jsonl", "promotion_history.jsonl"])
def test_resume_verifies_kernel_referenced_authority_bytes(world, name):
    store, args, *_ = world
    store.write_state(**args)
    path = store.root / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError): resume(store, args)


def test_state_publication_cannot_bless_truncated_archive_prefix(world):
    store, args, _, empty, _ = world
    args, _, _ = add_rung(world)
    store.write_state(**args)
    (store.root / "numerical_qd/archive/entries.jsonl").write_bytes(b"")
    with pytest.raises(ValueError): store.write_state(**(args | {"qd_snapshot_sha256": empty.fingerprint()}))


def test_state_publication_cannot_roll_back_completed_hyperband_rung(world):
    store, args, _, _, initial = world
    args, _, _ = add_rung(world)
    with pytest.raises(ValueError):
        store.write_state(**(args | {"hyperband_state_sha256": initial.fingerprint()}))


def test_budget_closure_ids_cannot_hide_decreased_charge(world):
    store, args, *_ = world
    args, rung, _ = add_rung(world)
    captured = rung.budget_outcome.to_payload()["ledger_checkpoint"]
    captured["charged_use"]["task_executions"] = 0
    captured["checkpoint_sha256"] = BudgetLedger.checkpoint_sha256(captured)
    with pytest.raises(ValueError):
        store.write_state(**(args | {"budget_checkpoint_sha256": persist(store, captured)}))


def test_source_and_structural_policy_full_ancestry_roundtrip(world):
    store, args, *_ = world
    original, original_sources, policies = store.verify_candidate(args["active_genome_sha256"])
    inventory = NumericalInventoryV2.from_payload(json.loads(
        (store.root / f"numerical_qd/objects/{original.inventory_sha256}.json").read_bytes()))
    parent = inventory.members[0]
    structural = dict(schema_version=1, operator="fork", parents=[parent.to_payload()], applicability_cells=[])
    policy_sha = persist(store, structural)
    child = replace(parent, member_id="child", parent_ids=(parent.member_id,), policy_sha256=policy_sha)
    child_genome = replace(original, generation=1, parent_genome_sha256s=(original.fingerprint(),),
        inventory_sha256=persist(store, NumericalInventoryV2(1, (child,))))
    child_sha = persist(store, child_genome)
    _, sources, recovered = store.verify_candidate(child_sha)
    assert sources == {SOURCE_SHA: SOURCE}
    assert recovered[policy_sha] == structural
    assert recovered[parent.policy_sha256] == policies[parent.policy_sha256]
    (store.root / f"numerical_qd/objects/{parent.policy_sha256}.json").unlink()
    with pytest.raises(ValueError): store.verify_candidate(child_sha)


def test_successful_proposal_attempt_persists_exact_normalized_sources(world):
    store, *_ = world
    payload = dict(provider="deterministic", resource_use=ResourceUse().to_payload(), failure_reason=None,
        proposals=[], source_sha256s=[SOURCE_SHA], attempts=[dict(provider="deterministic",
        resource_use=ResourceUse().to_payload(), failure_reason=None)])
    identity = fingerprint_payload(payload)
    path = store.write_proposal_attempt(identity, payload)
    assert path == store.root / f"numerical_qd/proposals/{identity}.json"
    assert path.read_bytes() == canonical_v2_bytes(payload)
    assert store.write_proposal_attempt(identity, payload) == path


def contextual_attempt(world, generation=1, counter=1):
    from evolving_loop.v2.numerical_qd.proposers import primitive_proposer_request
    store, args, *_ = world
    genome = store._object(args["active_genome_sha256"])
    state = parent_state(inventory=NumericalInventoryV2.from_payload(store._object(genome["inventory_sha256"])))
    request = primitive_proposer_request(parent_genome=genome, parent_state=state.to_payload(), selected_cells=[],
        train_feedback=[], remaining_budget=ResourceUse(wall_seconds=1.0).to_payload(),
        allowed_mutation_operators=sorted(state.mutation_policy.operators), counter_draw=0,
        max_proposals=1, max_response_bytes=1024)
    request_sha = persist(store, request)
    args = args | {"counter": dict(args["counter"]) | {"counter": counter}}
    store.write_state(**args)
    batch = {"provider": "deterministic", "resource_use": ResourceUse().to_payload(),
        "failure_reason": "empty", "proposals": [], "source_sha256s": [],
        "attempts": [{"provider": "deterministic", "resource_use": ResourceUse().to_payload(), "failure_reason": "empty"}]}
    return args, {"context": {"generation": generation, "parent_genome_sha256": args["active_genome_sha256"],
        "request_sha256": request_sha, "counter": {"seed": 42, "stream": "numerical", "start": counter - 1, "end": counter}},
        "batch": batch}


def test_contextual_identical_batches_remain_distinct_completion_attempts(world):
    store, _, kernel, *_ = world
    args, first = contextual_attempt(world)
    first_path = store.write_proposal_attempt(fingerprint_payload(first), first)
    args, second = contextual_attempt((store, args, *world[2:]), generation=2, counter=2)
    second_path = store.write_proposal_attempt(fingerprint_payload(second), second)
    assert first["batch"] == second["batch"] and first_path != second_path
    kernel.finalize()
    current = json.loads(kernel.checkpoint_path.read_bytes())
    args.update(kernel_checkpoint_sha256=current["checkpoint_sha256"],
                budget_checkpoint_sha256=persist(store, current["budget"]))
    store.write_state(**args)
    complete = json.loads(store.write_completion({"status": "numerical_qd_complete"}).read_bytes())
    assert complete["summary"]["provider_attempts"] == 2
    assert complete["summary"]["llm_calls"] == 0


@pytest.mark.parametrize("damage", ["generation", "parent", "request", "counter", "seed", "range", "extra"])
def test_contextual_attempt_rejects_unbound_request_or_counter(world, damage):
    store, *_ = world
    _, payload = contextual_attempt(world)
    if damage == "generation": payload["context"]["generation"] = 0
    elif damage == "parent": payload["context"]["parent_genome_sha256"] = "f" * 64
    elif damage == "request": payload["context"]["request_sha256"] = "f" * 64
    elif damage == "counter": payload["context"]["counter"]["end"] = 2
    elif damage == "seed": payload["context"]["counter"]["seed"] = 43
    elif damage == "range": payload["context"]["counter"]["start"] = 1
    else: payload["context"]["extra"] = True
    with pytest.raises(ValueError):
        store.write_proposal_attempt(fingerprint_payload(payload), payload)


def test_open_budget_checkpoint_cannot_publish_state(world):
    store, args, kernel, *_ = world
    ledger = BudgetLedger(kernel.budget.plan, monotonic=lambda: 0.0)
    ledger.reserve_stage("open-rung", ResourceUse(task_executions=1))
    with pytest.raises(ValueError):
        store.write_state(**(args | {"budget_checkpoint_sha256": persist(store, ledger.checkpoint())}))


def test_every_rung_and_archive_write_boundary_leaves_rejectable_partial_state(world, monkeypatch):
    import shutil
    store, args, kernel, archive, state = world
    store.write_state(**args)
    baseline = files(store.root)
    steps = []
    original_atomic, original_append = core_store._atomic_write, core_store.append_jsonl
    def capture_atomic(path, data):
        original_atomic(path, data)
        steps.append(("atomic", str(path)))
    def capture_append(path, payload):
        original_append(path, payload)
        steps.append(("append", str(path)))
    with monkeypatch.context() as patch:
        patch.setattr(core_store, "_atomic_write", capture_atomic)
        patch.setattr(core_store, "append_jsonl", capture_append)
        updated, _, _ = add_rung(world)
        store.write_state(**updated)
    assert len(steps) >= 17 and any(kind == "append" for kind, _ in steps)
    for boundary in range(1, len(steps) + 1):
        root = store.root.parent / f"rung-crash-{boundary}"
        shutil.copytree(store.root, root)
        for path in root.rglob("*"):
            if path.is_file() and str(path.relative_to(root)) not in baseline: path.unlink()
        for name, data in baseline.items(): (root / name).write_bytes(data)
        replica = api.NumericalQDRunStore(root)
        replica_kernel = EvolutionKernel.resume(core_store.V2RunStore(root), kernel.budget.plan, monotonic=lambda: 0.0)
        count = 0
        def after_write():
            nonlocal count
            count += 1
            if count == boundary: raise OSError("injected rung crash")
        def crash_atomic(path, data):
            original_atomic(path, data)
            after_write()
        def crash_append(path, payload):
            original_append(path, payload)
            after_write()
        with monkeypatch.context() as patch:
            patch.setattr(core_store, "_atomic_write", crash_atomic)
            patch.setattr(core_store, "append_jsonl", crash_append)
            # Kernel finally-accounting can replace the injected OSError with
            # its own closed-authority ValueError after the durable write.
            with pytest.raises((OSError, ValueError)):
                changed, _, _ = add_rung((replica, args, replica_kernel, archive, state))
                replica.write_state(**changed)
        assert count >= boundary
        if boundary == len(steps):
            assert resume(replica, updated).hyperband_state_sha256 == updated["hyperband_state_sha256"]
        else:
            with pytest.raises(ValueError): resume(replica, args)


@pytest.mark.parametrize("field,change", [("schema", None), ("seed", 999)])
def test_initial_state_requires_strict_config_and_committed_rng_seed(world, field, change):
    store, args, *_ = world
    if field == "schema":
        args = args | {"config_sha256": persist(store, {"profile": "test"})}
    else:
        args = args | {"counter": args["counter"] | {"seed": change}}
    with pytest.raises(ValueError): store.write_state(**args)


def test_state_cannot_advance_after_root_completion(world):
    store, args, kernel, *_ = world
    kernel.finalize()
    payload = json.loads(kernel.checkpoint_path.read_bytes())
    args.update(kernel_checkpoint_sha256=payload["checkpoint_sha256"],
                budget_checkpoint_sha256=persist(store, payload["budget"]))
    store.write_state(**args)
    store.write_completion({"status": "numerical_qd_complete"})
    before = files(store.root)
    with pytest.raises(ValueError): store.write_state(**(args | {"counter": args["counter"] | {"counter": 1}}))
    assert files(store.root) == before


def test_unpublished_immutable_checkpoint_cannot_be_blessed_on_retry(world, monkeypatch):
    store, args, *_ = world
    def crash(path, payload):
        raise OSError("before authority pointer")
    with monkeypatch.context() as patch:
        patch.setattr(core_store, "write_atomic_json", crash)
        with pytest.raises(OSError): store.write_state(**args)
    with pytest.raises(ValueError): store.write_state(**args)


@pytest.mark.parametrize("operation", ["source", "proposal"])
def test_crash_after_source_or_proposal_write_cannot_resume_previous_state(world, monkeypatch, operation):
    store, args, *_ = world
    store.write_state(**args)
    real = core_store._atomic_write
    def crash(path, data):
        real(path, data)
        raise OSError("artifact persisted before crash")
    payload = dict(provider="deterministic", resource_use=ResourceUse().to_payload(), failure_reason=None,
        proposals=[], source_sha256s=[], attempts=[dict(provider="deterministic",
        resource_use=ResourceUse().to_payload(), failure_reason=None)])
    with monkeypatch.context() as patch:
        patch.setattr(core_store, "_atomic_write", crash)
        with pytest.raises(OSError):
            if operation == "source":
                source = SOURCE + b"\n"
                store.write_source(hashlib.sha256(source).hexdigest(), source)
            else:
                store.write_proposal_attempt(fingerprint_payload(payload), payload)
    with pytest.raises(ValueError): resume(store, args)


def test_every_fsync_boundary_of_source_write_preserves_previous_authority_or_fails_closed(world, monkeypatch):
    import shutil
    store, args, *_ = world
    store.write_state(**args)
    real = core_store.os.fsync
    steps = []
    source = SOURCE + b"\n"
    identity = hashlib.sha256(source).hexdigest()
    def record(descriptor):
        real(descriptor)
        steps.append(descriptor)
    with monkeypatch.context() as patch:
        patch.setattr(core_store.os, "fsync", record)
        store.write_source(identity, source)
    assert len(steps) >= 2
    for boundary in range(1, len(steps) + 1):
        root = store.root.parent / f"fsync-crash-{boundary}"
        shutil.copytree(store.root, root)
        destination = root / f"numerical_qd/sources/{identity}.py"
        destination.unlink()
        replica = api.NumericalQDRunStore(root)
        count = 0
        def crash(descriptor):
            nonlocal count
            real(descriptor)
            count += 1
            if count == boundary: raise OSError("fsync crash")
        with monkeypatch.context() as patch:
            patch.setattr(core_store.os, "fsync", crash)
            with pytest.raises(OSError): replica.write_source(identity, source)
        if destination.exists():
            with pytest.raises(ValueError): resume(replica, args)
        else:
            assert resume(replica, args).counter["counter"] == 0


def test_runner_policy_credit_can_advance_without_replacing_active_genome(world):
    store, args, *_ = world
    original = store.write_state(**args)
    objects = store.root / "numerical_qd/objects"
    policy = NumericalMutationPolicyV2.from_payload(json.loads(
        (objects / f'{args["mutation_policy_sha256"]}.json').read_bytes()))
    operators = dict(policy.operators)
    operators["add"] = replace(operators["add"], attempts=1)
    policy = replace(policy, operators=operators)
    prompt = NumericalProposerPromptV2.from_payload(json.loads(
        (objects / f'{args["proposer_prompt_sha256"]}.json').read_bytes()))
    updated_prompt = replace(prompt, template="Retry using Train diagnostics.", parent_prompt_sha256=prompt.fingerprint())
    args = args | {"mutation_policy_sha256": persist(store, policy),
                   "proposer_prompt_sha256": persist(store, updated_prompt)}
    changed = store.write_state(**args)
    assert changed.active_genome_sha256 == original.active_genome_sha256
    assert resume(store, args).proposer_prompt_sha256 == updated_prompt.fingerprint()


def test_candidate_verification_requires_entire_parent_genome_closure(world):
    store, args, *_ = world
    original, _, _ = store.verify_candidate(args["active_genome_sha256"])
    objects = store.root / "numerical_qd/objects"
    inventory = NumericalInventoryV2.from_payload(json.loads((objects / f"{original.inventory_sha256}.json").read_bytes()))
    child_inventory = replace(inventory, members=(replace(inventory.members[0], member_id="independent-child"),))
    child = replace(original, generation=1, parent_genome_sha256s=(original.fingerprint(),),
        inventory_sha256=persist(store, child_inventory))
    identity = persist(store, child)
    (objects / f"{original.inventory_sha256}.json").unlink()
    with pytest.raises(ValueError): store.verify_candidate(identity)


def test_checkpoint_rejects_missing_unexecuted_hyperband_candidate(world):
    store, args, _, _, state = world
    state = replace(state, candidate_sha256s=tuple(sorted((args["active_genome_sha256"], "f" * 64))))
    with pytest.raises(ValueError):
        store.write_state(**(args | {"hyperband_state_sha256": persist(store, state)}))


def test_mismatched_config_budget_cannot_be_bound_to_kernel(world):
    store, args, *_ = world
    config = json.loads((store.root / f'numerical_qd/objects/{args["config_sha256"]}.json').read_bytes())
    config["budget"]["hard_limit_seconds"] += 1
    with pytest.raises(ValueError): store.write_state(**(args | {"config_sha256": persist(store, config)}))


def test_creation_requires_kernel_initialized_root_without_writing_placeholder_manifest(tmp_path):
    root = tmp_path / "uninitialized"
    with pytest.raises(ValueError): api.NumericalQDRunStore.create(root)
    assert not root.exists()


def test_full_genome_history_is_not_limited_to_structural_policy_depth(world):
    store, args, *_ = world
    genome, _, _ = store.verify_candidate(args["active_genome_sha256"])
    for generation in range(1, 67):
        genome = replace(genome, generation=generation, parent_genome_sha256s=(genome.fingerprint(),))
        persist(store, genome)
    reread, sources, _ = store.verify_candidate(genome.fingerprint())
    assert reread.generation == 66
    assert sources == {SOURCE_SHA: SOURCE}


def forge_runner_checkpoint(store, checkpoint, **changes):
    """Simulate fully rehashed persisted authority; semantic checks must reject it."""
    body = checkpoint.to_payload()
    body.pop("checkpoint_sha256")
    body.update(changes, completed_operation_sha256s=store._catalog())
    changed = NumericalQDCheckpointV2.seal(**body)
    persist(store, changed)
    (store.directory / "checkpoint.json").write_bytes(changed.canonical_bytes())
    return changed


def finalized_state(world, *, rung=False):
    store, args, kernel, *_ = world
    if rung:
        args, _, _ = add_rung(world)
    kernel.finalize()
    payload = json.loads(kernel.checkpoint_path.read_bytes())
    args = args | {"kernel_checkpoint_sha256": payload["checkpoint_sha256"],
                   "budget_checkpoint_sha256": persist(store, payload["budget"])}
    return args, store.write_state(**args)


def expected_completion(world, checkpoint, *, rung=False, providers=0, llm_calls=0, llm_attempts=0):
    store, _, kernel, *_ = world
    bundle = kernel.active_bundle()
    return {"schema_version": 1, "status": "numerical_qd_complete",
        "runner_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "kernel_checkpoint_sha256": checkpoint.kernel_checkpoint_sha256,
        "budget_checkpoint_sha256": checkpoint.budget_checkpoint_sha256,
        "active_bundle_sha256": checkpoint.active_bundle_sha256,
        "active_genome_sha256": checkpoint.active_genome_sha256,
        "qd_snapshot_sha256": checkpoint.qd_snapshot_sha256,
        "summary": {"supply_sha256": bundle.numerical_release_sha256,
            "registry_sha256": bundle.numerical_registry_sha256,
            "bundle_sha256": bundle.fingerprint(), "qd_snapshot_sha256": checkpoint.qd_snapshot_sha256,
            "mutation_policy_sha256": checkpoint.mutation_policy_sha256,
            "proposer_prompt_sha256": checkpoint.proposer_prompt_sha256,
            "occupied_cells": int(rung), "accepted_count": 0, "rejected_count": int(rung),
            "provider_attempts": providers, "llm_attempts": llm_attempts,
            "llm_calls": llm_calls, "llm_provider_used": llm_calls > 0,
            "budget": json.loads(kernel.checkpoint_path.read_bytes())["budget"],
            "dev_accessed": rung, "public_test_accessed": False}}


@pytest.mark.parametrize("operation", ["write_state", "resume", "completion"])
def test_r1_runner_budget_must_exactly_match_kernel_owned_budget(world, operation):
    store, args, kernel, *_ = world
    if operation == "completion":
        args, checkpoint = finalized_state(world)
    else:
        checkpoint = store.write_state(**args)
    ledger = BudgetLedger(kernel.budget.plan, monotonic=lambda: 0.0)
    reservation = ledger.reserve_stage("unrelated-eight-tasks", ResourceUse(task_executions=8))
    ledger.close_stage(reservation, ResourceUse(task_executions=8))
    if operation == "completion": ledger.begin_finalization()
    bad_sha = persist(store, ledger.checkpoint())
    changed = args | {"budget_checkpoint_sha256": bad_sha}
    assert json.loads(kernel.checkpoint_path.read_bytes())["budget"]["charged_use"]["task_executions"] == 0
    if operation == "write_state":
        with pytest.raises(ValueError): store.write_state(**changed)
    else:
        forge_runner_checkpoint(store, checkpoint, budget_checkpoint_sha256=bad_sha)
        with pytest.raises(ValueError):
            if operation == "resume": resume(store, changed)
            else: store.write_completion({"status": "numerical_qd_complete"})


def test_r1_final_runner_uses_current_kernel_budget_and_retains_exact_historical_rung(world):
    store, *_ = world
    args, rung, _ = add_rung(world)
    historical = rung.budget_outcome.to_payload()["ledger_checkpoint"]
    historical_path = store.directory / f'objects/{historical["checkpoint_sha256"]}.json'
    kernel = world[2]
    kernel.finalize()
    current = json.loads(kernel.checkpoint_path.read_bytes())
    args = args | {"kernel_checkpoint_sha256": current["checkpoint_sha256"],
                   "budget_checkpoint_sha256": persist(store, current["budget"])}
    checkpoint = store.write_state(**args)
    assert checkpoint.budget_checkpoint_sha256 == current["budget"]["checkpoint_sha256"]
    assert checkpoint.budget_checkpoint_sha256 != historical["checkpoint_sha256"]
    assert historical_path.read_bytes() == canonical_v2_bytes(historical)
    assert resume(store, args) == checkpoint


@pytest.mark.parametrize("name", ["run_manifest.json", "accepted_bundle.json", "checkpoint.json",
    "budget_plan.json", "promotion_history.jsonl", "archive", "archive/index.jsonl",
    "archive/objects", "archive_object", "acceptance", "acceptance_evidence", "evaluations", "evaluation_record"])
def test_r2_same_byte_external_symlinks_fail_before_kernel_resume(world, tmp_path, monkeypatch, name):
    store, args, *_ = world
    args, _, _ = add_rung(world)
    store.write_state(**args)
    if name == "archive_object": path = next((store.root / "archive/objects").glob("*.json"))
    elif name == "acceptance_evidence": path = next((store.root / "acceptance").glob("*.json"))
    elif name == "evaluation_record": path = next((store.root / "evaluations").rglob("closed.json"))
    else: path = store.root / name
    external = tmp_path / "external-authority"
    path.rename(external)
    path.symlink_to(external, target_is_directory=external.is_dir())
    entered = []
    real_resume = EvolutionKernel.resume
    def observed(cls, *args, **kwargs):
        entered.append(True)
        return real_resume(*args, **kwargs)
    monkeypatch.setattr(EvolutionKernel, "resume", classmethod(observed))
    with pytest.raises(ValueError): resume(store, args)
    assert not entered


@pytest.mark.parametrize("rung", [False, True])
def test_r3_completion_has_closed_truthful_summary_and_exact_resume_is_noop(world, rung):
    store, *_ = world
    args, checkpoint = finalized_state(world, rung=rung)
    path = store.write_completion({"status": "numerical_qd_complete"})
    assert json.loads(path.read_bytes()) == expected_completion(world, checkpoint, rung=rung)
    before = files(store.root)
    assert resume(store, args) == checkpoint
    assert files(store.root) == before


@pytest.mark.parametrize("damage", ["partial", "noncanonical", "unknown", "wrong_status", "runner", "kernel",
    "budget_sha", "bundle", "occupied_cells", "accepted_count", "provider_attempts", "llm_provider_used",
    "budget_payload", "dev_accessed", "public_test_accessed", "boolean_count", "real_model_claim"])
def test_r3_resume_rejects_malformed_or_mismatched_existing_completion(world, damage):
    store, *_ = world
    args, checkpoint = finalized_state(world)
    payload = expected_completion(world, checkpoint)
    if damage == "unknown": payload["unknown"] = True
    elif damage == "wrong_status": payload["status"] = "joint_complete"
    elif damage in {"runner", "kernel", "budget_sha", "bundle"}:
        field = {"runner": "runner_checkpoint_sha256", "kernel": "kernel_checkpoint_sha256",
                 "budget_sha": "budget_checkpoint_sha256", "bundle": "active_bundle_sha256"}[damage]
        payload[field] = "f" * 64
    elif damage in {"occupied_cells", "accepted_count", "provider_attempts"}: payload["summary"][damage] = 100
    elif damage in {"llm_provider_used", "dev_accessed", "public_test_accessed"}: payload["summary"][damage] = True
    elif damage == "budget_payload": payload["summary"]["budget"]["charged_use"]["task_executions"] = 8
    elif damage == "boolean_count": payload["summary"]["accepted_count"] = False
    elif damage == "real_model_claim": payload["summary"]["real_model_used"] = True
    data = b'{"status":' if damage == "partial" else (
        json.dumps(payload, indent=2).encode() if damage == "noncanonical" else canonical_v2_bytes(payload))
    (store.root / "evaluation_complete.json").write_bytes(data)
    with pytest.raises(ValueError): resume(store, args)


def test_r3_premature_root_completion_is_rejected(world):
    store, args, *_ = world
    checkpoint = store.write_state(**args)
    (store.root / "evaluation_complete.json").write_bytes(canonical_v2_bytes(expected_completion(world, checkpoint)))
    with pytest.raises(ValueError): resume(store, args)


@pytest.mark.parametrize("field", ["occupied_cells", "dev_accessed", "public_test_accessed", "provider_attempts"])
def test_r3_completion_writer_rejects_caller_summary_claims(world, field):
    store, *_ = world
    _, checkpoint = finalized_state(world)
    payload = expected_completion(world, checkpoint)
    payload["summary"][field] = True if field.endswith("accessed") else 42
    with pytest.raises(ValueError): store.write_completion(payload)
    assert not (store.root / "evaluation_complete.json").exists()


@pytest.mark.parametrize("field", ["task_materializer", "split_manifest", "metric_policy", "label_firewall",
    "artifact_validator", "sandbox_policy", "promotion_policy"])
@pytest.mark.parametrize("operation", ["write_state", "resume"])
def test_r4_config_protocol_must_match_every_kernel_commitment(world, field, operation):
    store, args, *_ = world
    checkpoint = store.write_state(**args) if operation == "resume" else None
    config = json.loads((store.directory / f'objects/{args["config_sha256"]}.json').read_bytes())
    config["kernel_protocol"][field] = "f" * 64
    changed = args | {"config_sha256": persist(store, config)}
    if checkpoint is not None:
        forge_runner_checkpoint(store, checkpoint, config_sha256=changed["config_sha256"])
    with pytest.raises(ValueError):
        if operation == "write_state": store.write_state(**changed)
        else: resume(store, changed)


@pytest.mark.parametrize("name", ["proposals", "results", "rungs", "archive"])
def test_r5_missing_required_empty_directory_is_corruption(world, name):
    store, args, *_ = world
    store.write_state(**args)
    (store.directory / name).rmdir()
    with pytest.raises(ValueError): resume(store, args)


@pytest.mark.parametrize("temporary", [False, True])
def test_r5_unreferenced_result_directory_is_not_ignored(world, temporary):
    store, args, *_ = world
    store.write_state(**args)
    path = store.directory / "results" / ("f" * 64)
    path.mkdir()
    if temporary: (path / ".task.json.crash.tmp").write_bytes(b"partial")
    with pytest.raises(ValueError): resume(store, args)
    with pytest.raises(ValueError): store.write_state(**args)


def test_r5_crash_after_result_directory_creation_fails_closed(world, monkeypatch):
    store, args, *_ = world
    store.write_state(**args)
    task = manifest().tasks[0]
    value = evaluation(args["active_genome_sha256"], (task.task_id,), subset=task.fingerprint())
    key = evaluation_cache_key(value.genome_sha256, task.task_sha256, SPLIT, METRIC, DESCRIPTOR, RUNTIME, PROTOCOL, ADAPTER)
    result = HyperbandTaskResultV2(value.genome_sha256, task.task_id, key, "passed", value, False, None)
    real = core_store._ensure_directory
    def crash(directory):
        real(directory)
        raise OSError("directory published before file")
    with monkeypatch.context() as patch:
        patch.setattr(core_store, "_ensure_directory", crash)
        with pytest.raises(OSError): store.write_task_result(task.task_sha256, result)
    assert (store.directory / "results" / value.genome_sha256).is_dir()
    with pytest.raises(ValueError): resume(store, args)


def test_r3_failed_llm_attempts_do_not_claim_an_actual_model_call(world):
    store, *_ = world
    attempts = [dict(provider="llm", resource_use=ResourceUse().to_payload(), failure_reason="unavailable"),
                dict(provider="deterministic", resource_use=ResourceUse().to_payload(), failure_reason=None)]
    payload = dict(provider="hybrid", resource_use=ResourceUse().to_payload(), failure_reason=None,
                   proposals=[], source_sha256s=[], attempts=attempts)
    store.write_proposal_attempt(fingerprint_payload(payload), payload)
    args, checkpoint = finalized_state(world)
    path = store.write_completion({"status": "numerical_qd_complete"})
    assert json.loads(path.read_bytes()) == expected_completion(world, checkpoint, providers=2, llm_attempts=1)
    assert resume(store, args) == checkpoint


def test_r3_completion_cannot_claim_unaccounted_llm_calls(world):
    store, *_ = world
    use = ResourceUse(llm_calls=1)
    payload = dict(provider="llm", resource_use=use.to_payload(), failure_reason=None,
        proposals=[], source_sha256s=[], attempts=[dict(provider="llm", resource_use=use.to_payload(), failure_reason=None)])
    store.write_proposal_attempt(fingerprint_payload(payload), payload)
    finalized_state(world)
    with pytest.raises(ValueError): store.write_completion({"status": "numerical_qd_complete"})


def forbid_content_io(monkeypatch):
    """Fail immediately instead of ever opening a FIFO/device in a regression."""
    attempted = []
    def forbidden(path, *args, **kwargs):
        attempted.append(str(path))
        raise AssertionError(f"content I/O before filesystem rejection: {path}")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    return attempted


def replace_with_nonregular(path, kind, tmp_path, monkeypatch):
    if kind in {"character_device", "block_device"}:
        # Creating device nodes requires privileges. Emulate only their lstat
        # type bits; every production check and all other metadata stay real.
        original_lstat = Path.lstat
        mode = stat.S_IFCHR if kind == "character_device" else stat.S_IFBLK
        def device_lstat(candidate, *args, **kwargs):
            value = original_lstat(candidate, *args, **kwargs)
            if candidate == path:
                return os.stat_result((mode | stat.S_IMODE(value.st_mode), *tuple(value)[1:]))
            return value
        monkeypatch.setattr(Path, "lstat", device_lstat)
        return
    original = tmp_path / "external-original"
    path.rename(original)
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "socket":
        # Bind at a short path to respect macOS AF_UNIX pathname limits, then
        # move the actual socket inode into the authority location.
        with tempfile.TemporaryDirectory(prefix="qd-socket-") as directory:
            bound = Path(directory) / "s"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(bound))
                bound.rename(path)
    elif kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        path.symlink_to(original)
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize("operation", ["write_state", "resume", "write_completion"])
@pytest.mark.parametrize("name", ["checkpoint.json", "numerical_qd/checkpoint.json"])
@pytest.mark.parametrize("kind", ["fifo", "socket", "character_device", "block_device", "directory", "symlink"])
def test_r2_preflight_rejects_nonregular_checkpoints_before_any_content_io(
        world, tmp_path, monkeypatch, operation, name, kind):
    store, *_ = world
    args, _ = finalized_state(world)
    replace_with_nonregular(store.root / name, kind, tmp_path, monkeypatch)
    attempted = forbid_content_io(monkeypatch)
    with pytest.raises(api.NumericalQDStoreError):
        if operation == "write_state": store.write_state(**args)
        elif operation == "resume": resume(store, args)
        else: store.write_completion({"status": "numerical_qd_complete"})
    assert attempted == []


@pytest.mark.parametrize("operation", ["create", "write_object", "write_source", "verify_candidate",
    "write_proposal_attempt", "write_task_result", "write_rung", "append_qd_entry"])
@pytest.mark.parametrize("name", ["checkpoint.json", "numerical_qd/checkpoint.json"])
def test_r2_preflight_guards_every_public_artifact_entry_point(
        world, tmp_path, monkeypatch, operation, name):
    store, *_ = world
    args, rung, entry = add_rung(world)
    store.write_state(**args)
    task = rung.manifest.tasks[0]
    result = HyperbandTaskResultV2.from_payload(json.loads((store.directory /
        f"results/{args['active_genome_sha256']}/{task.task_sha256}.json").read_bytes()))
    object_payload = {"retry": "same bytes"}
    object_sha = persist(store, object_payload)
    proposal = dict(provider="deterministic", resource_use=ResourceUse().to_payload(), failure_reason=None,
        proposals=[], source_sha256s=[SOURCE_SHA], attempts=[])
    calls = {
        "create": lambda: api.NumericalQDRunStore.create(store.root),
        "write_object": lambda: store.write_object(object_sha, object_payload),
        "write_source": lambda: store.write_source(SOURCE_SHA, SOURCE),
        "verify_candidate": lambda: store.verify_candidate(args["active_genome_sha256"]),
        "write_proposal_attempt": lambda: store.write_proposal_attempt(fingerprint_payload(proposal), proposal),
        "write_task_result": lambda: store.write_task_result(task.task_sha256, result),
        "write_rung": lambda: store.write_rung(rung),
        "append_qd_entry": lambda: store.append_qd_entry(entry),
    }
    replace_with_nonregular(store.root / name, "fifo", tmp_path, monkeypatch)
    attempted = forbid_content_io(monkeypatch)
    with pytest.raises(api.NumericalQDStoreError): calls[operation]()
    assert attempted == []


@pytest.mark.parametrize("name", ["run_manifest.json", "accepted_bundle.json", "archive/index.jsonl",
    "numerical_qd/manifest.json", "numerical_qd/objects", "numerical_qd/sources"])
@pytest.mark.parametrize("kind", ["fifo", "symlink"])
def test_r2_preflight_rejects_unsafe_authority_and_ancestors_before_reading_checkpoint(
        world, tmp_path, monkeypatch, name, kind):
    store, args, *_ = world
    store.write_state(**args)
    replace_with_nonregular(store.root / name, kind, tmp_path, monkeypatch)
    attempted = forbid_content_io(monkeypatch)
    with pytest.raises(api.NumericalQDStoreError): resume(store, args)
    assert attempted == []


@pytest.mark.parametrize("operation", ["resume", "write_completion"])
def test_r2_preflight_requires_runner_checkpoint_when_reopening(world, monkeypatch, operation):
    store, args, *_ = world
    assert not (store.directory / "checkpoint.json").exists()
    attempted = forbid_content_io(monkeypatch)
    with pytest.raises(api.NumericalQDStoreError):
        if operation == "resume": resume(store, args)
        else: store.write_completion({"status": "numerical_qd_complete"})
    assert attempted == []


def test_r2_preflight_allows_only_initial_phase_absence(kernel):
    assert not (kernel.store.root / "numerical_qd").exists()
    store = api.NumericalQDRunStore.create(kernel.store.root)
    assert not (store.directory / "checkpoint.json").exists()
    payload = {"phase": "before first runner checkpoint"}
    identity = persist(store, payload)
    assert json.loads((store.directory / f"objects/{identity}.json").read_bytes()) == payload
