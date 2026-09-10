"""Closed synthetic fixtures exercising the real V2 authority boundary.

The two stages are predetermined: Numerical improves every synthetic objective,
then Retrieval regresses. No forecast runtime, dataset, or external client is
loaded. These values make no claim about real forecasting quality.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path

from .budget import BudgetLedger, BudgetPlan, ResourceUse
from .bundle import EvolutionBundleV2
from .contracts import (
    EvolutionV2Config,
    KernelProtocolCommitment,
    canonical_v2_bytes,
    fingerprint_payload,
)
from .kernel import EvolutionKernel, KernelAuthorityError, _read
from .store import V2RunStore, write_once_json


STAGES = ("fake-numerical", "fake-retrieval")
_STAGE_USE = ResourceUse(wall_seconds=1.0, task_executions=20)


@dataclass
class FakeClock:
    seconds: float = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


def fake_sha256(seed: int, name: str) -> str:
    return hashlib.sha256(f"v2-fake:{seed}:{name}".encode()).hexdigest()


def smoke_config(seed: int = 7) -> EvolutionV2Config:
    protocol = KernelProtocolCommitment(
        **{
            field.name: fake_sha256(seed, field.name)
            for field in fields(KernelProtocolCommitment)
        }
    )
    return EvolutionV2Config(
        1,
        "smoke",
        seed,
        "ucb",
        ("numerical", "retrieval"),
        {"numerical": 4},
        {"resource_levels": [8]},
        {"fake": fake_sha256(seed, "runtime")},
        protocol,
        600,
        0.2,
        "deterministic_fake",
    )


def fake_seed_bundle(config: EvolutionV2Config) -> EvolutionBundleV2:
    return EvolutionBundleV2(
        2,
        0,
        None,
        *(
            fake_sha256(config.seed, name)
            for name in (
                "numerical-release",
                "numerical-registry",
                "retrieval-release",
                "decision-policy",
                "harness-policy",
                "archive-snapshot",
                "scheduler-state",
            )
        ),
        config.kernel_protocol.fingerprint(),
        config.runtime_fingerprints,
        None,
    )


@dataclass(frozen=True)
class FakeRunResult:
    accepted_bundle: EvolutionBundleV2
    completed_stage_ids: tuple[str, ...]
    accepted_steps: int
    rejected_steps: int
    completion_path: Path
    runner: str = "deterministic_fake"


def _child(parent, seed, stage):
    target = stage.removeprefix("fake-")
    changes = {
        target: (
            (
                fake_sha256(seed, "numerical-child-release"),
                fake_sha256(seed, "numerical-child-registry"),
            )
            if target == "numerical"
            else fake_sha256(seed, "retrieval-child-release")
        )
    }
    return parent.provisional_child(target, changes)


def _proposal(parent, child, stage):
    return {
        "runner": "deterministic_fake",
        "stage_id": stage,
        "target": stage.removeprefix("fake-"),
        "parent_bundle_sha256": parent.fingerprint(),
        "candidate": child.to_payload(),
    }


def _checkpoint(kernel, config, stages):
    return {
        "config_sha256": fingerprint_payload(config.to_payload()),
        "active_bundle_sha256": kernel.active_bundle().fingerprint(),
        "archive_snapshot_sha256": kernel.archive.snapshot_sha256(),
        "budget": kernel.budget.checkpoint(),
        "completed_stage_ids": list(stages),
        "accepted_steps": int(len(stages) >= 1),
        "rejected_steps": int(len(stages) == 2),
        "public_test_accessed": False,
    }


def _completion(checkpoint):
    return {
        "status": "deterministic_fake_complete",
        "runner": "deterministic_fake",
        "final_bundle_sha256": checkpoint["active_bundle_sha256"],
        "accepted_steps": checkpoint["accepted_steps"],
        "rejected_steps": checkpoint["rejected_steps"],
        "budget_usage": checkpoint["budget"]["charged_use"],
        "public_test_accessed": False,
    }


def _equal(actual, expected, context):
    if canonical_v2_bytes(actual) != canonical_v2_bytes(expected):
        raise KernelAuthorityError(f"fake {context} mismatch")


def _verify_run(kernel, config, seed, checkpoint, authority):
    """Revalidate stage order and every durable file without repairing state."""
    root = kernel.store.root
    stages = checkpoint.get("completed_stage_ids")
    if not isinstance(stages, list) or tuple(stages) not in ((), STAGES[:1], STAGES):
        raise KernelAuthorityError("fake completed stage order mismatch")
    _equal(checkpoint, _checkpoint(kernel, config, stages), "checkpoint")
    _equal(
        _read(root / "run_manifest.json"),
        {
            "system": "evolution_v2",
            "kernel_protocol": config.kernel_protocol.to_payload(),
            "runtime_fingerprints": dict(config.runtime_fingerprints),
            "seed_bundle_sha256": seed.fingerprint(),
            "budget_plan_sha256": kernel.budget.plan.fingerprint(),
        },
        "run manifest",
    )
    for name in ("active_bundle_sha256", "archive_snapshot_sha256", "budget"):
        if canonical_v2_bytes({name: checkpoint[name]}) != canonical_v2_bytes(
            {name: authority[name]}
        ):
            raise KernelAuthorityError(f"fake/kernel checkpoint {name} mismatch")
    expected_files = {
        "run_manifest.json",
        "budget_plan.json",
        "checkpoint.json",
        "kernel/checkpoint.json",
        "fake_run.json",
        "archive/index.jsonl",
        "accepted_bundle.json",
        "promotion_history.jsonl",
        f"archive/objects/{seed.fingerprint()}.json",
    }
    _equal(
        _read(root / "fake_run.json"),
        {
            "config_sha256": fingerprint_payload(config.to_payload()),
            "seed_bundle_sha256": seed.fingerprint(),
            "runner": "deterministic_fake",
        },
        "configuration/seed commitment",
    )
    transitions, parent, expected_progress = {}, seed, []
    for stage in stages:
        child = _child(parent, config.seed, stage)
        identity = child.fingerprint()
        ref = authority["completed_transitions"].get(identity)
        if ref is None or ref["decision"] != (
            "accept" if stage == STAGES[0] else "reject"
        ):
            raise KernelAuthorityError("fake completed transition mismatch")
        transitions[identity] = ref
        proposal_path = f"candidates/{identity}/proposal.json"
        _equal(_read(root / proposal_path), _proposal(parent, child, stage), "proposal")
        expected_files.update(
            {
                proposal_path,
                f"archive/objects/{identity}.json",
                f"acceptance/{ref['acceptance_evidence_sha256']}.json",
            }
        )
        for name in ("train", "closed", "budget_closure"):
            expected_files.add(f"evaluations/{identity}/{name}.json")
        train = _read(root / f"evaluations/{identity}/train.json")
        expected_progress.append(
            canonical_v2_bytes(
                {
                    "candidate_bundle_sha256": identity,
                    "train_evaluation_sha256": fingerprint_payload(train),
                    "status": train["status"],
                    "resource_use": train["resource_use"],
                }
            )
        )
        if ref["decision"] == "accept":
            sealed_sha = ref["sealed_bundle_sha256"]
            expected_files.add(f"archive/objects/{sealed_sha}.json")
            parent = EvolutionBundleV2.from_payload(
                _read(root / f"archive/objects/{sealed_sha}.json", sealed_sha)
            )
    if transitions != authority["completed_transitions"] or len(
        authority["budget_closures"]
    ) != len(stages):
        raise KernelAuthorityError("fake partial or unexpected stage")
    if parent.canonical_bytes() != kernel.active_bundle().canonical_bytes():
        raise KernelAuthorityError("fake active Parent mismatch")
    if stages:
        expected_files.add("progress.jsonl")
        progress = root / "progress.jsonl"
        if not progress.is_file() or progress.read_bytes() != b"".join(
            expected_progress
        ):
            raise KernelAuthorityError("fake progress mismatch")
    if checkpoint["budget"]["finalization_started"]:
        if tuple(stages) != STAGES:
            raise KernelAuthorityError("fake premature finalization")
        _equal(
            _read(root / "evaluation_complete.json"),
            _completion(checkpoint),
            "completion",
        )
        expected_files.add("evaluation_complete.json")
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise KernelAuthorityError("fake run cannot contain symbolic links")
        if path.is_file():
            relative = str(path.relative_to(root))
            if relative in expected_files:
                actual_files.add(relative)
            elif path.name.startswith(".") and path.name.endswith(".tmp"):
                continue  # Only unreferenced atomic-write remnants are ignorable.
            else:
                raise KernelAuthorityError(f"fake unreferenced artifact: {relative}")
    if actual_files != expected_files:
        raise KernelAuthorityError("fake immutable artifact is missing")
    return tuple(stages)


def run_fake_kernel(
    output_dir: str | Path,
    config: EvolutionV2Config,
    *,
    seed_bundle: EvolutionBundleV2 | None = None,
    clock: FakeClock | None = None,
    resume: bool = False,
    stop_after_stage: str | None = None,
) -> FakeRunResult:
    """Run or verify/resume the two fixed smoke stages through kernel APIs."""
    if (
        config.profile != "smoke"
        or config.runner != "deterministic_fake"
        or not {"numerical", "retrieval"}.issubset(config.enabled_mutation_scopes)
    ):
        raise ValueError(
            "fake runner requires smoke config with Numerical and Retrieval"
        )
    if stop_after_stage is not None and stop_after_stage not in STAGES:
        raise ValueError("unknown fake stop stage")
    clock = clock if clock is not None else FakeClock()
    seed = seed_bundle if seed_bundle is not None else fake_seed_bundle(config)
    if (
        seed.generation != 0
        or seed.runtime_fingerprints != config.runtime_fingerprints
        or seed.protocol_fingerprint != config.kernel_protocol.fingerprint()
    ):
        raise ValueError("fake seed/config binding mismatch")
    root = Path(output_dir)
    kernel_path = root / "kernel" / "checkpoint.json"
    plan = BudgetPlan.from_config(
        config, ceilings=ResourceUse(wall_seconds=2.0, task_executions=40)
    )
    if resume:
        # Refuse partial publication before kernel resume can reconcile it.
        checkpoint = _read(root / "checkpoint.json")
        authority = _read(kernel_path)
        if (
            authority.get("pending_publication") is not None
            or authority.get("terminal_recovery") is not None
        ):
            raise KernelAuthorityError("fake partial publication cannot be resumed")
        _equal(
            _read(root / "fake_run.json"),
            {
                "config_sha256": fingerprint_payload(config.to_payload()),
                "seed_bundle_sha256": seed.fingerprint(),
                "runner": "deterministic_fake",
            },
            "configuration/seed commitment",
        )
        kernel = EvolutionKernel.resume(
            V2RunStore(root), plan, monotonic=clock, checkpoint_path=kernel_path
        )
        stages = _verify_run(kernel, config, seed, checkpoint, authority)
    else:
        budget = BudgetLedger(plan, monotonic=clock)
        if not budget.can_open_stage(_STAGE_USE + _STAGE_USE).allowed:
            raise ValueError(
                "fake budget cannot cover both stages and finalization reserve"
            )
        kernel = EvolutionKernel(
            V2RunStore.create(root),
            config.kernel_protocol,
            budget,
            seed=seed,
            checkpoint_path=kernel_path,
        )
        write_once_json(
            root / "fake_run.json",
            {
                "config_sha256": fingerprint_payload(config.to_payload()),
                "seed_bundle_sha256": seed.fingerprint(),
                "runner": "deterministic_fake",
            },
        )
        stages = ()
        checkpoint = _checkpoint(kernel, config, stages)
        kernel.store.write_checkpoint(checkpoint)
    for stage in STAGES[len(stages) :]:
        parent = kernel.active_bundle()
        child = _child(parent, config.seed, stage)
        permit = kernel.reserve_evaluation(child, _STAGE_USE)
        if not permit.allowed:
            raise ValueError(f"fake budget denied stage: {permit.reason}")
        kernel.store.write_candidate(
            child.fingerprint(), _proposal(parent, child, stage)
        )
        improved = stage == STAGES[0]
        clock.advance(1.0)
        evaluation = kernel.close_evaluation(
            parent,
            child,
            permit=permit,
            status="passed" if improved else "failed",
            train_objectives={
                "synthetic_smae": 0.4 if improved else 0.8,
                "synthetic_srmse": 0.4 if improved else 0.8,
            },
            train_behavior_descriptors={"family": "deterministic_fake"},
            dev_comparison={
                "passed": improved,
                "parent_metrics": {"synthetic_loss": 0.6 if improved else 0.4},
                "candidate_metrics": {"synthetic_loss": 0.4 if improved else 0.8},
            },
            resource_use=_STAGE_USE,
        )
        kernel.evaluate_transition(
            parent,
            child,
            target=stage.removeprefix("fake-"),
            evaluation=evaluation,
            permit=permit,
        )
        stages += (stage,)
        checkpoint = _checkpoint(kernel, config, stages)
        kernel.store.write_checkpoint(checkpoint)
        if stop_after_stage == stage:
            break
    if (
        stages == STAGES
        and stop_after_stage != STAGES[-1]
        and not kernel.budget.finalization_started
    ):
        kernel.finalize()
        checkpoint = _checkpoint(kernel, config, stages)
        kernel.store.write_checkpoint(checkpoint)
        kernel.store.write_completion(_completion(checkpoint))
    return FakeRunResult(
        kernel.active_bundle(),
        stages,
        checkpoint["accepted_steps"],
        checkpoint["rejected_steps"],
        root / "evaluation_complete.json",
    )


__all__ = [
    "FakeClock",
    "FakeRunResult",
    "fake_seed_bundle",
    "fake_sha256",
    "run_fake_kernel",
    "smoke_config",
]
