"""Durable QD authority rejects changed bytes and incomplete operations."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from evolving_loop.v2 import store as core_store
from evolving_loop.v2.budget import BudgetLedger, ResourceUse
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
from tests.test_evolution_v2_kernel import kernel
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
    ledger = BudgetLedger(kernel.budget.plan, monotonic=lambda: 0.0)
    reservation = ledger.reserve_stage("rung-0", ResourceUse(task_executions=8))
    ledger.close_stage(reservation, ResourceUse(task_executions=8))
    captured = ledger.checkpoint()
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
    second = replace(entry, cell=replace(CELL, trend="high"))
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
            with pytest.raises(OSError, match="rung crash"):
                changed, _, _ = add_rung((replica, args, kernel, archive, state))
                replica.write_state(**changed)
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
