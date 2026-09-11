"""Real Kernel material capabilities cannot be forged by ordinary feedback."""
import copy
import hashlib
import json

import pytest

from evolving_loop.v2.budget import BudgetLedger, BudgetPlan, ResourceUse
from evolving_loop.v2.kernel import EvolutionKernel
from evolving_loop.v2.store import V2RunStore
from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
from tests.test_evolution_v2_kernel import protocol, seed


DATA = b"def forecast():\n    return [1.0]\n"
SHA = hashlib.sha256(DATA).hexdigest()
RELATIVE = f"sources/{SHA}.py"


@pytest.fixture
def kernel(tmp_path):
    plan = BudgetPlan(1000, 0.2, ResourceUse(wall_seconds=800.0, artifact_bytes=1000000))
    value = EvolutionKernel(V2RunStore.create(tmp_path / "run"), protocol(),
        BudgetLedger(plan, monotonic=lambda: 0.0), seed=seed())
    NumericalQDRunStore.create(value.store.root)
    return value


def opened(kernel, tag="b"):
    child = kernel.active_bundle().provisional_child("numerical", {"numerical": (tag * 64, "c" * 64)})
    return child, kernel.reserve_evaluation(child, ResourceUse(artifact_bytes=1000))


def close(kernel, child, permit, *, objectives=None, size=0):
    return kernel.close_evaluation(kernel.active_bundle(), child, permit=permit, status="passed",
        train_objectives=objectives or {}, train_behavior_descriptors={},
        dev_comparison={"passed": False, "parent_metrics": {}, "candidate_metrics": {}},
        resource_use=ResourceUse(artifact_bytes=size), account_only=True)


def prepare(kernel, permit, *, relative=RELATIVE, kind="source"):
    return kernel.prepare_material_write(permit, relative_path=relative, kind=kind,
        content_sha256=SHA, size_bytes=len(DATA))


def path(kernel):
    return kernel.store.root / "numerical_qd" / RELATIVE


def files(kernel):
    return {p.relative_to(kernel.store.root): p.read_bytes()
            for p in kernel.store.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("existing,empty", [(False, False), (True, False), (False, True)])
def test_ordinary_material_feedback_cannot_create_authority(kernel, existing, empty):
    child, permit = opened(kernel)
    if existing:
        path(kernel).write_bytes(DATA)
    receipt = {"kind": "source", "relative_path": RELATIVE, "content_sha256": SHA, "size_bytes": len(DATA)}
    before = files(kernel)
    with pytest.raises(ValueError):
        close(kernel, child, permit, objectives={"material_receipts": [] if empty else [receipt]}, size=len(DATA))
    assert files(kernel) == before


def test_kernel_material_capability_seals_exact_bytes_once(kernel):
    child, permit = opened(kernel)
    capability = prepare(kernel, permit)
    path(kernel).write_bytes(DATA)
    kernel.register_material_write(permit, capability)
    with pytest.raises(ValueError):
        kernel.register_material_write(permit, capability)
    close(kernel, child, permit, size=len(DATA))
    train = json.loads((kernel.store.root / f"evaluations/{child.fingerprint()}/train.json").read_bytes())
    assert train["train_objectives"]["material_receipts"] == [{"kind": "source", "relative_path": RELATIVE,
        "content_sha256": SHA, "size_bytes": len(DATA)}]
    resumed = EvolutionKernel.resume(kernel.store, kernel.budget.plan, monotonic=lambda: 0.0)
    resumed.finalize()
    assert resumed.budget.charged_use.artifact_bytes == len(DATA)


@pytest.mark.parametrize("fault", ["copied_capability", "cross_permit", "missing", "empty", "changed", "symlink", "directory"])
def test_register_requires_original_capability_and_exact_regular_bytes(kernel, fault):
    child, permit = opened(kernel)
    capability = prepare(kernel, permit)
    if fault != "missing":
        path(kernel).write_bytes(DATA)
    if fault == "copied_capability":
        capability = copy.copy(capability)
    elif fault == "cross_permit":
        _, permit = opened(kernel, "d")
    elif fault == "empty":
        path(kernel).write_bytes(b"")
    elif fault == "changed":
        path(kernel).write_bytes(DATA.replace(b"1.0", b"2.0"))
    elif fault in {"symlink", "directory"}:
        path(kernel).unlink()
        if fault == "directory":
            path(kernel).mkdir()
        else:
            path(kernel).symlink_to(kernel.store.root / "run_manifest.json")
    before = files(kernel)
    with pytest.raises(ValueError):
        kernel.register_material_write(permit, capability)
    assert files(kernel) == before


@pytest.mark.parametrize("fault", ["existing_unregistered", "traversal", "absolute", "symlink_parent", "control_kind"])
def test_material_prepare_requires_new_safe_billable_path(kernel, fault):
    _, permit = opened(kernel)
    relative, kind = RELATIVE, "source"
    if fault == "existing_unregistered":
        path(kernel).write_bytes(DATA)
    elif fault == "traversal":
        relative = "../run_manifest.json"
    elif fault == "absolute":
        relative = str(path(kernel))
    elif fault == "symlink_parent":
        path(kernel).parent.rmdir()
        path(kernel).parent.symlink_to(kernel.store.root, target_is_directory=True)
    else:
        kind = "partial_rung"
    before = files(kernel)
    with pytest.raises(ValueError):
        prepare(kernel, permit, relative=relative, kind=kind)
    assert files(kernel) == before


def test_open_capability_blocks_close_resume_and_finalize_until_explicit_abort(kernel):
    child, permit = opened(kernel)
    capability = prepare(kernel, permit)
    with pytest.raises(ValueError):
        close(kernel, child, permit)
    with pytest.raises(ValueError):
        kernel.finalize()
    with pytest.raises(ValueError):
        EvolutionKernel.resume(kernel.store, kernel.budget.plan, monotonic=lambda: 0.0)
    kernel.abort_material_write(permit, capability)
    with pytest.raises(ValueError):
        kernel.abort_material_write(permit, capability)
    close(kernel, child, permit)
    assert kernel.budget.charged_use == ResourceUse()
    kernel.finalize()


def test_unregistered_file_cannot_be_silently_aborted_or_resumed(kernel):
    child, permit = opened(kernel)
    capability = prepare(kernel, permit)
    path(kernel).write_bytes(DATA)
    with pytest.raises(ValueError):
        kernel.abort_material_write(permit, capability)
    with pytest.raises(ValueError):
        close(kernel, child, permit, size=len(DATA))
    with pytest.raises(ValueError):
        EvolutionKernel.resume(kernel.store, kernel.budget.plan, monotonic=lambda: 0.0)


@pytest.mark.parametrize("boundary", ["close", "resume", "finalize"])
def test_registered_material_mutation_fails_closed_at_every_authority_boundary(kernel, boundary):
    child, permit = opened(kernel)
    capability = prepare(kernel, permit)
    path(kernel).write_bytes(DATA)
    kernel.register_material_write(permit, capability)
    if boundary != "close":
        close(kernel, child, permit, size=len(DATA))
    path(kernel).write_bytes(DATA.replace(b"1.0", b"2.0"))
    before = files(kernel)
    with pytest.raises(ValueError):
        if boundary == "close":
            close(kernel, child, permit, size=len(DATA))
        elif boundary == "resume":
            EvolutionKernel.resume(kernel.store, kernel.budget.plan, monotonic=lambda: 0.0)
        else:
            kernel.finalize()
    assert files(kernel) == before


def test_material_capability_cannot_replay_across_closed_reservations(kernel):
    child, permit = opened(kernel)
    capability = prepare(kernel, permit)
    path(kernel).write_bytes(DATA)
    kernel.register_material_write(permit, capability)
    close(kernel, child, permit, size=len(DATA))
    _, later = opened(kernel, "d")
    with pytest.raises(ValueError):
        kernel.register_material_write(later, capability)
    with pytest.raises(ValueError):
        prepare(kernel, later)


@pytest.mark.parametrize("after_write", [False, True])
def test_host_material_write_failure_closes_only_actual_registered_bytes(kernel, after_write):
    from evolving_loop.v2.numerical_qd.runner import _MaterialAccounting
    store = NumericalQDRunStore(kernel.store.root)
    accounting = _MaterialAccounting(store, kernel)
    def dispatch():
        if after_write:
            path(kernel).write_bytes(DATA)
        raise OSError("injected Host write failure")
    with pytest.raises(OSError, match="Host write failure"):
        accounting.write(RELATIVE, DATA, dispatch, kind="source")
    assert not kernel.budget.checkpoint()["open_reservations"]
    assert kernel.budget.charged_use.artifact_bytes == (len(DATA) if after_write else 0)
    train = json.loads(next((kernel.store.root / "evaluations").rglob("train.json")).read_bytes())
    assert train["status"] == "failed"
    assert bool(train["train_objectives"].get("material_receipts")) == after_write
    EvolutionKernel.resume(kernel.store, kernel.budget.plan, monotonic=lambda: 0.0).finalize()
