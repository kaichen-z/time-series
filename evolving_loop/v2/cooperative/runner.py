"""Bounded cooperative Bundle search over the Evolution V2 Kernel."""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from common.payload import canonical_json_bytes, strict_json_loads

from ..budget import BudgetLedger, BudgetPlan, ResourceUse
from ..bundle import EvolutionBundleV2
from ..contracts import SanitizedEvolutionFeedback, canonical_v2_bytes, fingerprint_payload
from ..kernel import EvolutionKernel, KernelAuthorityError
from ..numerical_qd.adapters import frozen_local_evidence_references
from ..numerical_qd.artifacts import validate_artifact
from ..store import V2RunStore, _atomic_write, append_jsonl, write_once_json
from .adapters import (
    CooperativeArtifactCatalog,
    CooperativePipelineAdapter,
    DecisionModuleV2,
    FrozenNumericalArtifactsV2,
    RetrievalModuleV2,
)
from .contracts import (
    ARM_ORDER,
    CooperativeCheckpointV2,
    CooperativeRunResultV2,
    CooperativeSchedulerStateV2,
    SchedulerArmStateV2,
)
from .proposals import propose_bundle_candidate
from .schedulers import record_outcome, select_arm


@dataclass(frozen=True, slots=True)
class _Aggregate:
    bundle_sha256: str
    stage: str
    task_universe_sha256: str
    evaluation_sha256: str
    task_count: int
    coverage: float
    mean_smae: float
    mean_srmse: float
    mean_joint: float
    invalid_count: int
    catastrophic_count: int
    fallback_count: int
    public_test_accessed: bool

    @classmethod
    def from_evaluation(cls, bundle, stage, task_universe_sha256, evaluation):
        return cls(
            bundle.fingerprint(),
            stage,
            task_universe_sha256,
            evaluation.fingerprint,
            evaluation.task_count,
            evaluation.coverage,
            evaluation.mean_smae,
            evaluation.mean_srmse,
            evaluation.mean_joint,
            evaluation.invalid_count,
            evaluation.catastrophic_count,
            evaluation.fallback_count,
            evaluation.public_test_accessed,
        )

    @classmethod
    def from_payload(cls, payload):
        expected = {
            "kind",
            "schema_version",
            "bundle_sha256",
            "stage",
            "task_universe_sha256",
            "evaluation_sha256",
            "task_count",
            "coverage",
            "mean_smae",
            "mean_srmse",
            "mean_joint",
            "invalid_count",
            "catastrophic_count",
            "fallback_count",
            "public_test_accessed",
        }
        if set(payload) != expected or payload["kind"] != "cooperative_aggregate":
            raise ValueError("invalid cooperative aggregate")
        if payload["schema_version"] != 1:
            raise ValueError("invalid cooperative aggregate schema")
        return cls(**{name: payload[name] for name in expected - {"kind", "schema_version"}})

    def to_payload(self):
        return {"kind": "cooperative_aggregate", "schema_version": 1, **asdict(self)}


def train_eligible(parent, child, tolerance):
    return (
        child.coverage == 1.0
        and child.invalid_count <= parent.invalid_count
        and child.catastrophic_count <= parent.catastrophic_count
        and child.mean_smae <= parent.mean_smae + tolerance
        and child.mean_srmse <= parent.mean_srmse + tolerance
    )


def dev_passed(parent, child, tolerance):
    parent_joint = (parent.mean_smae + parent.mean_srmse) / 2.0
    child_joint = (child.mean_smae + child.mean_srmse) / 2.0
    return train_eligible(parent, child, tolerance) and (
        child_joint < parent_joint - tolerance
    )


def _read(path, identity=None):
    raw = Path(path).read_bytes()
    value = strict_json_loads(raw.decode("utf-8"), context=str(path))
    if type(value) is not dict or canonical_v2_bytes(value) != raw:
        raise KernelAuthorityError("cooperative artifact must be canonical JSON")
    if identity is not None and fingerprint_payload(value) != identity:
        raise KernelAuthorityError("cooperative object digest mismatch")
    return value


def _read_frozen_evidence(path, identity, kind):
    """Read only an envelope-bound Task-4 object in its exact serialization."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != identity:
        raise KernelAuthorityError("cooperative evidence object digest mismatch")
    try:
        validate_artifact(kind, raw)
        value = strict_json_loads(raw.decode("utf-8"), context=str(path))
    except (UnicodeError, TypeError, ValueError) as error:
        raise KernelAuthorityError(
            "cooperative evidence object is not its exact typed serialization"
        ) from error
    if type(value) is not dict:
        raise KernelAuthorityError("cooperative evidence object must be an object")
    return value


def _payload(value):
    if hasattr(value, "to_payload"):
        return value.to_payload()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("cooperative config must provide a canonical payload")


def _plain_task(value):
    if hasattr(value, "to_payload"):
        return value.to_payload()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {key: _plain_task(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_task(item) for item in value]
    return value


def _parts(seed_artifacts):
    if isinstance(seed_artifacts, Mapping):
        values = tuple(seed_artifacts.get(name) for name in ("numerical", "retrieval", "decision"))
    elif isinstance(seed_artifacts, (tuple, list)) and len(seed_artifacts) == 3:
        values = tuple(seed_artifacts)
    else:
        values = tuple(getattr(seed_artifacts, name, None) for name in ("numerical", "retrieval", "decision"))
    numerical, retrieval, decision = values
    if type(numerical) is not FrozenNumericalArtifactsV2:
        raise TypeError("seed numerical artifacts must be FrozenNumericalArtifactsV2")
    if type(retrieval) is not RetrievalModuleV2 or type(decision) is not DecisionModuleV2:
        raise TypeError("seed Retrieval and Decision artifacts use V2 module contracts")
    return numerical, retrieval, decision


def _task_splits(tasks):
    if not isinstance(tasks, Mapping):
        raise TypeError("cooperative tasks must be a split mapping")
    for name, values in tasks.items():
        if "public" in str(name).casefold() and values:
            raise ValueError("Public task membership is forbidden in cooperative search")
    try:
        train, dev = tuple(tasks["train"]), tuple(tasks["dev"])
    except (KeyError, TypeError) as error:
        raise ValueError("cooperative tasks require Train and Dev splits") from error
    if not train or not dev:
        raise ValueError("cooperative Train and Dev splits must be non-empty")
    return train, dev


def _input_sha256s(numerical, retrieval, decision, train, dev, adapters):
    alternatives = getattr(adapters.get("numerical"), "_alternatives", ())
    decision_adapter = adapters.get("decision")
    pairs = [
        {
            "release_sha256": item.release.fingerprint,
            "registry_sha256": item.registry.fingerprint,
            "envelope_sha256": item.envelope.fingerprint(),
        }
        for item in alternatives
    ]
    return {
        "seed_numerical_release": numerical.release.fingerprint,
        "seed_numerical_registry": numerical.registry.fingerprint,
        "seed_numerical_envelope": numerical.envelope.fingerprint(),
        "seed_retrieval": retrieval.fingerprint(),
        "seed_decision": decision.fingerprint(),
        "numerical_alternatives": fingerprint_payload({"pairs": pairs}),
        "decision_operator": fingerprint_payload(
            {
                "prompts": list(getattr(decision_adapter, "_prompts", ())),
                "settings_cycle": getattr(
                    decision_adapter, "_settings_cycle", False
                ),
            }
        ),
        "train_tasks": fingerprint_payload({"tasks": [_plain_task(task) for task in train]}),
        "dev_tasks": fingerprint_payload({"tasks": [_plain_task(task) for task in dev]}),
    }


def _initial_scheduler(control, discount):
    enabled = tuple(name for name in ARM_ORDER if name in control.enabled_mutation_scopes)
    return CooperativeSchedulerStateV2(
        1,
        control.scheduler,
        control.seed,
        0,
        0,
        float(discount),
        {name: SchedulerArmStateV2(0, 0, 0.0, 0.0) for name in enabled},
    )


def _write_object(root, identity, payload):
    if fingerprint_payload(payload) != identity:
        raise ValueError("cooperative object identity mismatch")
    return write_once_json(root / "objects" / f"{identity}.json", payload)


def _write_legacy_object(root, identity, payload):
    """Persist Task-4 evidence under its original canonical JSON identity."""
    data = canonical_json_bytes(payload)
    if hashlib.sha256(data).hexdigest() != identity:
        raise ValueError("cooperative legacy object identity mismatch")
    destination = root / "objects" / f"{identity}.json"
    if destination.exists():
        if destination.read_bytes() != data:
            raise ValueError("cooperative legacy object content changed")
        return destination
    _atomic_write(destination, data)
    return destination


def _cache_key(bundle, stage, task_universe_sha256):
    return _cache_key_sha(bundle.fingerprint(), stage, task_universe_sha256)


def _cache_key_sha(bundle_sha256, stage, task_universe_sha256):
    return fingerprint_payload(
        {
            "bundle_sha256": bundle_sha256,
            "stage": stage,
            "task_universe_sha256": task_universe_sha256,
        }
    )


def _load_catalog_and_cache(root, catalog, numerical, retrieval, decision, adapters):
    catalog.add_numerical(numerical)
    alternatives = tuple(getattr(adapters.get("numerical"), "_alternatives", ()))
    for item in alternatives:
        catalog.add_numerical(item)
    catalog.add_retrieval(retrieval)
    catalog.add_decision(decision)
    cache = {}
    objects = root / "objects"
    if not objects.exists():
        return cache
    evidence_refs = {}
    for pair in (numerical, *alternatives):
        for identity, kind in frozen_local_evidence_references(pair.envelope).items():
            previous = evidence_refs.get(identity)
            if previous is not None and previous != kind:
                raise KernelAuthorityError(
                    "cooperative evidence object has conflicting typed bindings"
                )
            evidence_refs[identity] = kind
    for path in objects.glob("*.json"):
        if path.stem in evidence_refs:
            _read_frozen_evidence(path, path.stem, evidence_refs[path.stem])
            continue
        value = _read(path, path.stem)
        if value.get("kind") == "cooperative_aggregate":
            aggregate = _Aggregate.from_payload(value)
            cache[_cache_key_from_aggregate(aggregate)] = aggregate
        elif set(value) == {
            "schema_version", "source_release_sha256", "genome_payload", "skills_payload"
        }:
            catalog.add_retrieval(RetrievalModuleV2.from_payload(value))
        elif set(value) == {
            "schema_version", "prompt", "skills", "enable_evidence_adjustments",
            "max_evidence_adjustments", "aggregation"
        }:
            catalog.add_decision(DecisionModuleV2.from_payload(value))
    return cache


def _cache_key_from_aggregate(aggregate):
    return _cache_key_sha(
        aggregate.bundle_sha256,
        aggregate.stage,
        aggregate.task_universe_sha256,
    )


def _restore_cache_aliases(cache, rows, train_universe, dev_universe):
    for row in rows:
        if row["decision"] != "accept":
            continue
        for stage, universe in (("train", train_universe), ("dev", dev_universe)):
            aggregate = cache.get(
                _cache_key_sha(row["candidate_sha256"], stage, universe)
            )
            if aggregate is not None:
                cache[_cache_key_sha(row["active_after"], stage, universe)] = aggregate


def _evaluate_cached(root, cache, pipeline, bundle, tasks, stage, universe_sha):
    key = _cache_key(bundle, stage, universe_sha)
    if key in cache:
        return cache[key], 0, 0
    evaluation = pipeline.evaluate(bundle, tasks, stage)
    if evaluation.candidate_sha256 != bundle.fingerprint():
        raise ValueError("pipeline aggregate does not bind the evaluated Bundle")
    if evaluation.public_test_accessed:
        raise ValueError("Public evaluation is forbidden in cooperative search")
    aggregate = _Aggregate.from_evaluation(bundle, stage, universe_sha, evaluation)
    payload = aggregate.to_payload()
    identity = fingerprint_payload(payload)
    path = root / "objects" / f"{identity}.json"
    existed = path.exists()
    _write_object(root, identity, payload)
    cache[key] = aggregate
    return aggregate, len(tasks), 0 if existed else len(canonical_v2_bytes(payload))


def _feedback(parent_sha, parent, child, normalized_cost):
    denominator = max(abs(parent.mean_joint), 1e-12)
    improvement = (parent.mean_joint - child.mean_joint) / denominator
    return SanitizedEvolutionFeedback(
        parent_sha256=parent_sha,
        train_evaluation_sha256=child.evaluation_sha256,
        train_objectives={
            "normalized_cost": float(normalized_cost),
            "relative_joint_improvement": float(improvement),
        },
        train_behavior_descriptors={
            "catastrophic_count": child.catastrophic_count,
            "fallback_count": child.fallback_count,
            "invalid_count": child.invalid_count,
        },
        failure_categories=tuple(
            name
            for name, count in (
                ("catastrophic", child.catastrophic_count),
                ("fallback", child.fallback_count),
                ("invalid", child.invalid_count),
            )
            if count
        ),
        remaining_proposal_budget={},
    )


def _neutral_feedback(parent_sha):
    return SanitizedEvolutionFeedback(
        parent_sha,
        fingerprint_payload({"cooperative_feedback": "initial", "parent": parent_sha}),
        {"normalized_cost": 0.0, "relative_joint_improvement": 0.0},
        {"catastrophic_count": 0, "fallback_count": 0, "invalid_count": 0},
        (),
        {},
    )


def _persist_feedback(root, feedback):
    payload = feedback.to_payload()
    identity = fingerprint_payload(payload)
    _write_object(root, identity, payload)
    return identity


def _progress(root):
    path = root / "cooperative_progress.jsonl"
    if not path.exists():
        return []
    rows = []
    for raw in path.read_bytes().splitlines(keepends=True):
        value = strict_json_loads(raw.decode("utf-8"), context=str(path))
        if type(value) is not dict or canonical_v2_bytes(value) != raw:
            raise KernelAuthorityError("cooperative progress must be canonical JSONL")
        rows.append(value)
    return rows


def _kernel_checkpoint_sha(root):
    return _read(root / "checkpoint.json")["checkpoint_sha256"]


def _checkpoint(root, config_sha, inputs, kernel, state, accepted, rejected, completed):
    checkpoint = CooperativeCheckpointV2.seal(
        schema_version=1,
        config_sha256=config_sha,
        input_sha256s=inputs,
        active_bundle_sha256=kernel.active_bundle().fingerprint(),
        scheduler_state=state,
        scheduler_state_sha256=state.fingerprint(),
        next_step=state.completed_step,
        accepted_steps=accepted,
        rejected_steps=rejected,
        completed_candidate_sha256s=completed,
        kernel_checkpoint_sha256=_kernel_checkpoint_sha(root),
    )
    _atomic_write(root / "cooperative_checkpoint.json", checkpoint.canonical_bytes())
    return checkpoint


def _plan(control, ceilings):
    if isinstance(ceilings, Mapping):
        ceilings = ResourceUse.from_payload(ceilings)
    return BudgetPlan.from_config(control, ceilings=ceilings)


def _available_estimate(kernel, task_executions, remaining_steps):
    if type(remaining_steps) is not int or remaining_steps < 1:
        raise ValueError("remaining_steps must be a positive integer")
    charged = kernel.budget.charged_use
    ceilings = kernel.budget.plan.ceilings
    values = {
        name: max(0.0 if type(getattr(ceilings, name)) is float else 0,
                  getattr(ceilings, name) - getattr(charged, name))
        for name in ResourceUse.field_names()
    }
    remaining_wall_seconds = min(
        values["wall_seconds"],
        max(0.0, kernel.budget.plan.search_deadline_seconds - kernel.budget.elapsed_wall_seconds),
    )
    values["wall_seconds"] = remaining_wall_seconds / (remaining_steps + 1)
    values["task_executions"] = task_executions
    return ResourceUse.from_payload(values)


def _host_resource_snapshot(adapters) -> ResourceUse:
    reporter = adapters.get("resource_reporter")
    if reporter is None:
        return ResourceUse()
    if not callable(reporter):
        raise TypeError("cooperative Host resource reporter must be callable")
    value = reporter()
    if type(value) is not ResourceUse:
        raise TypeError("cooperative Host resource reporter must return ResourceUse")
    if value.wall_seconds or value.task_executions or value.artifact_bytes:
        raise ValueError("cooperative Host reporter may only report external resources")
    return value


def _host_resource_delta(adapters, before: ResourceUse) -> ResourceUse:
    after = _host_resource_snapshot(adapters)
    values = {
        name: getattr(after, name) - getattr(before, name)
        for name in ResourceUse.field_names()
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("cooperative Host resource counters moved backwards")
    return ResourceUse.from_payload(values)


def _completed_result(root, checkpoint, rows):
    payload = _read(root / "evaluation_complete.json")
    result = CooperativeRunResultV2.from_payload(payload)
    if (
        result.active_bundle_sha256 != checkpoint.active_bundle_sha256
        or result.scheduler_state_sha256 != checkpoint.scheduler_state_sha256
        or result.accepted_steps != checkpoint.accepted_steps
        or result.rejected_steps != checkpoint.rejected_steps
        or result.attempted_arms != tuple(row["arm"] for row in rows)
    ):
        raise KernelAuthorityError("cooperative completion/checkpoint mismatch")
    return result


def _verify_progress(rows, checkpoint):
    if (
        len(rows) != checkpoint.next_step
        or tuple(row.get("step") for row in rows) != tuple(range(len(rows)))
        or tuple(row.get("candidate_sha256") for row in rows)
        != checkpoint.completed_candidate_sha256s
        or sum(row.get("decision") == "accept" for row in rows)
        != checkpoint.accepted_steps
        or sum(row.get("decision") != "accept" for row in rows)
        != checkpoint.rejected_steps
    ):
        raise KernelAuthorityError("cooperative progress/checkpoint mismatch")
    if rows and (
        rows[-1].get("active_after") != checkpoint.active_bundle_sha256
        or rows[-1].get("scheduler_state_sha256")
        != checkpoint.scheduler_state_sha256
    ):
        raise KernelAuthorityError("cooperative progress/checkpoint mismatch")
    for previous, current in zip(rows, rows[1:]):
        if previous.get("active_after") != current.get("active_before"):
            raise KernelAuthorityError("cooperative progress/checkpoint mismatch")
    for row in rows:
        changed = row.get("active_before") != row.get("active_after")
        if changed != (row.get("decision") == "accept"):
            raise KernelAuthorityError("cooperative progress/checkpoint mismatch")


def run_cooperative_evolution(
    output_dir,
    config,
    seed_artifacts,
    tasks,
    adapters,
    *,
    resume=False,
    stop_after=None,
):
    """Run at most four closed cooperative candidates and resume exactly."""
    if type(resume) is not bool:
        raise TypeError("resume must be boolean")
    if not isinstance(adapters, Mapping):
        raise TypeError("cooperative adapters must be a mapping")
    config_payload = _payload(config)
    config_sha = fingerprint_payload(config_payload)
    control = config.control
    max_steps = config.max_steps
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("cooperative max_steps must be positive")
    if stop_after is not None and (type(stop_after) is not int or not 0 <= stop_after <= max_steps):
        raise ValueError("stop_after must be a closed-step count")
    numerical, retrieval, decision = _parts(seed_artifacts)
    train, dev = _task_splits(tasks)
    inputs = _input_sha256s(numerical, retrieval, decision, train, dev, adapters)
    plan = _plan(control, config.resource_ceilings)
    clock = adapters.get("monotonic", time.monotonic)
    if not callable(clock):
        raise TypeError("cooperative monotonic clock must be callable")
    root = Path(output_dir)

    if resume:
        store = V2RunStore(root)
        checkpoint = CooperativeCheckpointV2.from_payload(
            _read(root / "cooperative_checkpoint.json")
        )
        if checkpoint.config_sha256 != config_sha or dict(checkpoint.input_sha256s) != inputs:
            raise KernelAuthorityError("cooperative resume configuration/input mismatch")
        kernel = EvolutionKernel.resume(store, plan, monotonic=clock)
        if checkpoint.kernel_checkpoint_sha256 != _kernel_checkpoint_sha(root):
            raise KernelAuthorityError("cooperative/Kernel checkpoint mismatch")
        if checkpoint.active_bundle_sha256 != kernel.active_bundle().fingerprint():
            raise KernelAuthorityError("cooperative active Bundle mismatch")
        rows = _progress(root)
        _verify_progress(rows, checkpoint)
        if (root / "evaluation_complete.json").exists():
            return _completed_result(root, checkpoint, rows)
        state = checkpoint.scheduler_state
        accepted, rejected = checkpoint.accepted_steps, checkpoint.rejected_steps
        completed = checkpoint.completed_candidate_sha256s
    else:
        state = _initial_scheduler(control, config.discount)
        seed = EvolutionBundleV2(
            2,
            0,
            None,
            numerical.release.fingerprint,
            numerical.registry.fingerprint,
            retrieval.fingerprint(),
            decision.fingerprint(),
            fingerprint_payload({"cooperative_harness": config_sha}),
            fingerprint_payload({"cooperative_archive_seed": inputs}),
            state.fingerprint(),
            control.kernel_protocol.fingerprint(),
            control.runtime_fingerprints,
            None,
        )
        store = V2RunStore.create(root)
        kernel = EvolutionKernel(
            store,
            control.kernel_protocol,
            BudgetLedger(plan, monotonic=clock),
            seed=seed,
        )
        rows, accepted, rejected, completed = [], 0, 0, ()
        _write_object(root, state.fingerprint(), state.to_payload())
        _checkpoint(root, config_sha, inputs, kernel, state, accepted, rejected, completed)

    catalog = CooperativeArtifactCatalog(
        lambda identity, payload: _write_object(root, identity, payload),
        lambda identity, payload: _write_legacy_object(root, identity, payload),
    )
    cache = _load_catalog_and_cache(
        root, catalog, numerical, retrieval, decision, adapters
    )
    pipeline = adapters.get("pipeline")
    if pipeline is None or not callable(getattr(pipeline, "evaluate", None)):
        raise TypeError("cooperative adapters require a pipeline evaluator")
    if isinstance(pipeline, CooperativePipelineAdapter):
        pipeline.catalog = catalog

    if rows:
        feedback_payload = _read(
            root / "objects" / f"{rows[-1]['feedback_sha256']}.json",
            rows[-1]["feedback_sha256"],
        )
        feedback = SanitizedEvolutionFeedback.from_payload(feedback_payload)
    else:
        feedback = _neutral_feedback(kernel.active_bundle().fingerprint())

    train_universe = inputs["train_tasks"]
    dev_universe = inputs["dev_tasks"]
    _restore_cache_aliases(cache, rows, train_universe, dev_universe)
    for step in range(state.completed_step, max_steps):
        if stop_after is not None and state.completed_step >= stop_after:
            break
        parent = kernel.active_bundle()
        arm = select_arm(state)
        candidate = propose_bundle_candidate(
            parent, arm, catalog, adapters, feedback, step
        )
        active_before = parent.fingerprint()
        if candidate is None:
            candidate_sha = fingerprint_payload(
                {"no_candidate": arm, "parent": active_before, "step": step}
            )
            state = record_outcome(
                state,
                arm,
                train_reward=-1.0,
                normalized_cost=0.0,
                accepted=False,
            )
            _write_object(root, state.fingerprint(), state.to_payload())
            feedback = SanitizedEvolutionFeedback(
                active_before,
                candidate_sha,
                {"normalized_cost": 0.0, "relative_joint_improvement": -1.0},
                {"catastrophic_count": 0, "fallback_count": 0, "invalid_count": 1},
                ("no_candidate",),
                {},
            )
            feedback_sha = _persist_feedback(root, feedback)
            decision_value = "no_candidate"
            active_after = active_before
            rejected += 1
        else:
            child = candidate.to_child(parent)
            candidate_sha = child.fingerprint()
            permit = kernel.reserve_evaluation(
                child,
                _available_estimate(
                    kernel,
                    2 * (len(train) + len(dev)),
                    max_steps - step,
                ),
            )
            if not permit.allowed:
                raise KernelAuthorityError(f"cooperative evaluation denied: {permit.reason}")
            store.write_candidate(candidate_sha, candidate.to_payload())
            started = clock()
            host_before = _host_resource_snapshot(adapters)
            parent_train, parent_tasks, parent_bytes = _evaluate_cached(
                root, cache, pipeline, parent, train, "train", train_universe
            )
            child_train, child_tasks, child_bytes = _evaluate_cached(
                root, cache, pipeline, child, train, "train", train_universe
            )
            train_tasks = parent_tasks + child_tasks
            normalized_cost = float(
                config.task_cost_weight * train_tasks / max(1, 2 * len(train))
            )
            relative = (
                (parent_train.mean_joint - child_train.mean_joint)
                / max(abs(parent_train.mean_joint), 1e-12)
            )
            train_reward = float(
                relative
                - (1.0 if child_train.invalid_count else 0.0)
                - 0.5
                * child_train.catastrophic_count
                / max(1, child_train.task_count)
            )
            eligible = train_eligible(
                parent_train, child_train, config.acceptance_tolerance
            )
            state = record_outcome(
                state,
                arm,
                train_reward=train_reward,
                normalized_cost=normalized_cost,
                accepted=eligible,
            )
            _write_object(root, state.fingerprint(), state.to_payload())
            dev_tasks = dev_bytes = 0
            parent_dev = child_dev = None
            if eligible:
                parent_dev, parent_dev_tasks, parent_dev_bytes = _evaluate_cached(
                    root, cache, pipeline, parent, dev, "dev", dev_universe
                )
                child_dev, child_dev_tasks, child_dev_bytes = _evaluate_cached(
                    root, cache, pipeline, child, dev, "dev", dev_universe
                )
                dev_tasks = parent_dev_tasks + child_dev_tasks
                dev_bytes = parent_dev_bytes + child_dev_bytes
            passed = bool(
                eligible
                and parent_dev is not None
                and child_dev is not None
                and dev_passed(parent_dev, child_dev, config.acceptance_tolerance)
            )
            elapsed = float(clock() - started)
            resource_use = ResourceUse(
                wall_seconds=elapsed,
                task_executions=train_tasks + dev_tasks,
                artifact_bytes=parent_bytes + child_bytes + dev_bytes,
            ) + _host_resource_delta(adapters, host_before)
            closed = kernel.close_evaluation(
                parent,
                child,
                permit=permit,
                status="passed" if eligible else "failed",
                train_objectives={
                    "coverage": child_train.coverage,
                    "mean_smae": child_train.mean_smae,
                    "mean_srmse": child_train.mean_srmse,
                    "mean_joint": child_train.mean_joint,
                    "normalized_cost": normalized_cost,
                    "relative_joint_improvement": relative,
                },
                train_behavior_descriptors={
                    "catastrophic_count": child_train.catastrophic_count,
                    "fallback_count": child_train.fallback_count,
                    "invalid_count": child_train.invalid_count,
                },
                dev_comparison={
                    "passed": passed,
                    "parent_metrics": (
                        {}
                        if parent_dev is None
                        else {
                            "mean_smae": parent_dev.mean_smae,
                            "mean_srmse": parent_dev.mean_srmse,
                            "mean_joint": parent_dev.mean_joint,
                        }
                    ),
                    "candidate_metrics": (
                        {}
                        if child_dev is None
                        else {
                            "mean_smae": child_dev.mean_smae,
                            "mean_srmse": child_dev.mean_srmse,
                            "mean_joint": child_dev.mean_joint,
                        }
                    ),
                },
                resource_use=resource_use,
            )
            active = kernel.evaluate_transition(
                parent,
                child,
                target=arm,
                evaluation=closed,
                permit=permit,
                host_scheduler_state_sha256=state.fingerprint(),
            )
            active_after = active.fingerprint()
            decision_value = "accept" if active_after != active_before else "reject"
            if decision_value == "accept":
                cache[_cache_key_sha(active_after, "train", train_universe)] = child_train
                if child_dev is not None:
                    cache[_cache_key_sha(active_after, "dev", dev_universe)] = child_dev
            accepted += int(decision_value == "accept")
            rejected += int(decision_value == "reject")
            feedback = _feedback(
                active_after, parent_train, child_train, normalized_cost
            )
            feedback_sha = _persist_feedback(root, feedback)

        completed = (*completed, candidate_sha)
        row = {
            "schema_version": 1,
            "step": step,
            "arm": arm,
            "candidate_sha256": candidate_sha,
            "decision": decision_value,
            "active_before": active_before,
            "active_after": active_after,
            "scheduler_state_sha256": state.fingerprint(),
            "feedback_sha256": feedback_sha,
        }
        append_jsonl(root / "cooperative_progress.jsonl", row)
        rows.append(row)
        _checkpoint(root, config_sha, inputs, kernel, state, accepted, rejected, completed)
    if state.completed_step == max_steps and stop_after is None:
        kernel.finalize()
        checkpoint = _checkpoint(
            root, config_sha, inputs, kernel, state, accepted, rejected, completed
        )
        result = CooperativeRunResultV2(
            1,
            "cooperative_complete",
            checkpoint.active_bundle_sha256,
            state.fingerprint(),
            tuple(row["arm"] for row in rows),
            accepted,
            rejected,
            False,
        )
        store.write_completion(result.to_payload())
        return result

    return CooperativeRunResultV2(
        1,
        "cooperative_stopped",
        kernel.active_bundle().fingerprint(),
        state.fingerprint(),
        tuple(row["arm"] for row in rows),
        accepted,
        rejected,
        False,
    )


__all__ = ["dev_passed", "run_cooperative_evolution", "train_eligible"]
