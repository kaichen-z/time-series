"""Small resumable Host runner for the DGM-lite source research epoch."""
from __future__ import annotations

import copy
import dataclasses
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

from common.payload import strict_json_loads

from ..budget import BudgetLedger, BudgetPlan, ResourceUse
from ..contracts import canonical_v2_bytes, fingerprint_payload
from ..store import write_atomic_json, write_once_json
from .archive import SourceArchiveV2, propose_sources
from .authority import SourceAuthorityV2
from .contracts import SourceConfigV2, SourceRunResultV2, SourceVariantV2
from .meta import SourceValidationV2


class SourceRunnerError(ValueError):
    """Raised before evaluation when a source run cannot be safely resumed."""


_STOPS = {"candidate:1", "candidate:2", "validation", "canary"}


def _read(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
        value = strict_json_loads(raw.decode("utf-8"), context=str(path))
    except (OSError, UnicodeError, ValueError) as error:
        raise SourceRunnerError(f"cannot read {path.name}") from error
    if not isinstance(value, Mapping) or canonical_v2_bytes(value) != raw:
        raise SourceRunnerError(f"{path.name} is not canonical JSON")
    return value


def _case_sha(case: object) -> str:
    seed = getattr(case, "seed_source", None)
    if not isinstance(seed, SourceVariantV2):
        raise TypeError("source case requires seed_source")
    tasks = tuple(getattr(case, "train_tasks", ())) + tuple(getattr(case, "dev_tasks", ()))
    identities = []
    for task in tasks:
        try:
            identities.append(dataclasses.asdict(task))
        except AttributeError as error:
            raise TypeError("source case task has no stable identity") from error
    try:
        members = {
            field.name: getattr(case, field.name)
            for field in dataclasses.fields(case)
        }
    except (TypeError, AttributeError):
        members = vars(case)
    sources = {
        name: value.fingerprint()
        for name, value in members.items()
        if isinstance(value, SourceVariantV2)
    }
    bundle = getattr(case, "seed_bundle", None)
    bundle_sha = bundle.fingerprint() if callable(getattr(bundle, "fingerprint", None)) else None
    return fingerprint_payload({"seed": seed.fingerprint(), "sources": sources, "tasks": identities,
                                "seed_bundle": bundle_sha, "input_digest": getattr(case, "input_digest", None),
                                "protocol": seed.protocol_fingerprint, "runtime": seed.runtime_fingerprint})


def _evidence(stage: str, value: SourceValidationV2) -> dict[str, object]:
    return {"schema_version": 1, "stage": stage, "parent_source_sha256": value.parent_source_sha256,
            "finalist_source_sha256": value.finalist_source_sha256, "passed": value.passed,
            "reason": value.reason, "commitment_sha256": value.commitment_sha256,
            "replay_fingerprint": value.replay_fingerprint,
            "evaluation_fingerprints": list(value.evaluation_fingerprints)}


def _cache_checkpoint(evaluator: object) -> Mapping[str, object]:
    method = getattr(evaluator, "cache_checkpoint", None)
    if not callable(method):
        raise SourceRunnerError("source evaluator does not expose durable cache authority")
    checkpoint = method()
    if not isinstance(checkpoint, Mapping):
        raise SourceRunnerError("source evaluator cache checkpoint is invalid")
    return checkpoint


def _restore_cache(evaluator: object, checkpoint: object) -> None:
    method = getattr(evaluator, "restore_cache", None)
    if not callable(method):
        raise SourceRunnerError("source evaluator does not expose durable cache authority")
    try:
        method(checkpoint)
    except (TypeError, ValueError) as error:
        raise SourceRunnerError("source evaluator cache checkpoint is invalid") from error


def _state(root: Path, state: dict[str, object], ledger: BudgetLedger, evaluator: object) -> None:
    body = dict(state)
    body["budget_checkpoint"] = ledger.checkpoint()
    body["evaluator_cache"] = _cache_checkpoint(evaluator)
    state["budget_checkpoint"] = body["budget_checkpoint"]
    state["evaluator_cache"] = body["evaluator_cache"]
    body["checkpoint_sha256"] = fingerprint_payload(body)
    write_atomic_json(root / "checkpoint.json", body)


def _new_state(config: SourceConfigV2, case_sha: str, seed: SourceVariantV2,
               ledger: BudgetLedger, evaluator: object) -> dict[str, object]:
    return {"schema_version": 1, "config_sha256": config.fingerprint(), "case_sha256": case_sha,
            "phase": "search", "draw_counter": 0, "completed_stage_ids": [],
            "parent": None, "candidate_batch": [],
            "candidates": [], "finalist": None, "validation_passed": None, "validation_receipt": None,
            "proposed": 0, "eligible": 0, "activated": 0, "rolled_back": 0,
            "seed_source_sha256": seed.fingerprint(), "budget_checkpoint": ledger.checkpoint(),
            "evaluator_cache": _cache_checkpoint(evaluator)}


def _verify_state(state: Mapping[str, object], config: SourceConfigV2, case_sha: str) -> dict[str, object]:
    values = dict(state)
    claimed = values.pop("checkpoint_sha256", None)
    if not isinstance(claimed, str) or claimed != fingerprint_payload(values):
        raise SourceRunnerError("source checkpoint digest mismatch")
    required = {"schema_version", "config_sha256", "case_sha256", "phase", "draw_counter", "completed_stage_ids",
                "parent", "candidate_batch", "candidates", "finalist", "validation_passed", "validation_receipt",
                "proposed", "eligible", "activated", "rolled_back", "seed_source_sha256", "budget_checkpoint",
                "evaluator_cache"}
    if set(values) != required or values["schema_version"] != 1:
        raise SourceRunnerError("source checkpoint schema mismatch")
    if values["config_sha256"] != config.fingerprint() or values["case_sha256"] != case_sha:
        raise SourceRunnerError("source resume configuration/input mismatch")
    if values["phase"] not in {"search", "validation", "canary_pending", "complete"}:
        raise SourceRunnerError("source checkpoint phase is invalid")
    if (not isinstance(values["completed_stage_ids"], list)
            or not isinstance(values["candidate_batch"], list)
            or not isinstance(values["candidates"], list)):
        raise SourceRunnerError("source checkpoint state is invalid")
    return values


def _budget_plan(config: SourceConfigV2) -> BudgetPlan:
    return BudgetPlan(config.hard_limit_seconds, 0.1, config.resource_ceilings)


def _resume_budget(plan: BudgetPlan, checkpoint: object) -> BudgetLedger:
    if not isinstance(checkpoint, Mapping):
        raise SourceRunnerError("source checkpoint budget is invalid")
    try:
        verified = BudgetLedger.resume(plan, checkpoint, monotonic=time.monotonic)
    except ValueError as error:
        raise SourceRunnerError("source checkpoint budget is invalid") from error
    if verified.checkpoint()["open_reservations"]:
        raise SourceRunnerError("in-flight source budget reservations cannot resume")
    if plan.no_time_limit:
        return verified
    # Each closed-checkpoint invocation gets a new bounded wall-time window;
    # accumulated task/subprocess charges and closed stages remain authoritative.
    fresh = copy.deepcopy(dict(checkpoint))
    fresh["prior_elapsed_wall_seconds"] = 0.0
    fresh["checkpoint_sha256"] = BudgetLedger.checkpoint_sha256(fresh)
    try:
        return BudgetLedger.resume(plan, fresh, monotonic=time.monotonic)
    except ValueError as error:
        raise SourceRunnerError("source checkpoint budget is invalid") from error


def _train_estimate(case: object) -> ResourceUse:
    train_count = len(tuple(getattr(case, "train_tasks", ())))
    arms = len(tuple(getattr(getattr(case, "evaluator", None), "enabled_arms", ())))
    fold_count = int(
        getattr(
            getattr(case, "evaluator", None),
            "_fold_count",
            getattr(case, "fold_count", 2),
        )
    )
    if train_count < 1 or arms <= 0 or fold_count < 1 or train_count % fold_count != 0:
        raise SourceRunnerError(
            "source budget estimator requires a positive Train multiple of fold_count"
        )
    held_count = train_count // fold_count
    task_executions = fold_count * ((arms + 3) * held_count) + fold_count * train_count
    return ResourceUse(task_executions=task_executions, subprocesses=fold_count)


def _stage_estimate(case: object, stage: str) -> ResourceUse:
    train = _train_estimate(case)
    if stage.startswith("candidate:"):
        return train
    dev_count = len(tuple(getattr(case, "dev_tasks", ())))
    train_copies = 1 if stage == "validation" else 2
    return ResourceUse(
        task_executions=train_copies * train.task_executions + 2 * dev_count,
        subprocesses=train_copies * train.subprocesses,
    )


def _actual_use(value: object) -> ResourceUse:
    task_cost = getattr(value, "actual_task_cost", None)
    source_invocations = getattr(value, "actual_source_invocations", None)
    if type(task_cost) is not int or task_cost < 0 or type(source_invocations) is not int or source_invocations < 0:
        raise SourceRunnerError("source evaluator did not report valid resource use")
    return ResourceUse(task_executions=task_cost, subprocesses=source_invocations)


def _reserve(ledger: BudgetLedger, stage: str, estimate: ResourceUse):
    gate = ledger.can_open_stage(estimate)
    if not gate.allowed:
        return None
    permit = ledger.reserve_stage(stage, estimate)
    if not permit.allowed:
        raise SourceRunnerError(f"source budget reservation failed: {permit.reason}")
    return permit


def _result(archive: SourceArchiveV2, authority: SourceAuthorityV2, state: Mapping[str, object], *, complete: bool) -> SourceRunResultV2:
    return SourceRunResultV2(1, "source_evolution_complete" if complete else "source_evolution_checkpointed",
        authority.active_source().fingerprint(), archive.snapshot_sha256(), int(state["proposed"]), int(state["eligible"]),
        int(state["activated"]), int(state["rolled_back"]), tuple(state["completed_stage_ids"]), False)


def _finish(root: Path, archive: SourceArchiveV2, authority: SourceAuthorityV2,
            state: dict[str, object], ledger: BudgetLedger, evaluator: object) -> SourceRunResultV2:
    state["phase"] = "complete"
    result = _result(archive, authority, state, complete=True)
    write_once_json(root / "evaluation_complete.json", result.to_payload())
    _state(root, state, ledger, evaluator)
    return result


def run_source_evolution(output_dir: str | Path, config: SourceConfigV2, case: object, *, resume: bool = False,
                         stop_after: str | None = None) -> SourceRunResultV2:
    """Run exactly one source epoch, stopping only at closed durable stages."""
    if not isinstance(config, SourceConfigV2):
        raise TypeError("config must be SourceConfigV2")
    if stop_after is not None and stop_after not in _STOPS:
        raise ValueError("stop_after must be candidate:1, candidate:2, validation, or canary")
    root = Path(output_dir)
    seed = getattr(case, "seed_source", None)
    evaluator = getattr(case, "evaluator", None)
    if not isinstance(seed, SourceVariantV2) or evaluator is None:
        raise TypeError("source case requires seed_source and evaluator")
    if config.protocol_fingerprint != seed.protocol_fingerprint or config.runtime_fingerprint != seed.runtime_fingerprint:
        raise SourceRunnerError("source config commitments do not match frozen input")
    case_sha = _case_sha(case)
    plan = _budget_plan(config)
    checkpoint = root / "checkpoint.json"
    complete_file = root / "evaluation_complete.json"
    if resume:
        if not checkpoint.exists():
            raise SourceRunnerError("resume requires a source checkpoint")
        state = _verify_state(_read(checkpoint), config, case_sha)
        ledger = _resume_budget(plan, state["budget_checkpoint"])
        _restore_cache(evaluator, state["evaluator_cache"])
        archive = SourceArchiveV2(root / "source_archive")
        authority = SourceAuthorityV2(root / "authority", seed)
        if authority.checkpoint_path.exists():
            raise SourceRunnerError("in-flight canary cannot resume")
        if complete_file.exists():
            stored = SourceRunResultV2.from_payload(_read(complete_file))
            if state["phase"] != "complete" or stored.canonical_bytes() != _result(archive, authority, state, complete=True).canonical_bytes():
                raise SourceRunnerError("completed source result mismatch")
            return stored
    else:
        if root.exists() and any(root.iterdir()):
            raise SourceRunnerError("source output directory is non-empty; use resume")
        root.mkdir(parents=True, exist_ok=True)
        archive = SourceArchiveV2(root / "source_archive")
        archive.add(seed)
        archive.close(seed.fingerprint(), status="eligible")
        authority = SourceAuthorityV2(root / "authority", seed)
        ledger = BudgetLedger(plan, monotonic=time.monotonic)
        state = _new_state(config, case_sha, seed, ledger, evaluator)
        _state(root, state, ledger, evaluator)

    if state["phase"] == "search":
        if not state["candidate_batch"]:
            parent = archive.sample(int(state["draw_counter"]))
            candidates = propose_sources(parent, draw_counter=int(state["draw_counter"]), limit=config.max_candidates)
            state["parent"] = parent.to_payload()
            state["candidate_batch"] = [candidate.to_payload() for candidate in candidates]
            _state(root, state, ledger, evaluator)
        else:
            if not isinstance(state["parent"], Mapping):
                raise SourceRunnerError("search checkpoint has no persisted parent")
            parent = SourceVariantV2.from_payload(state["parent"])
            candidates = tuple(SourceVariantV2.from_payload(value) for value in state["candidate_batch"])
            if any(candidate.parent_source_sha256 != parent.fingerprint() for candidate in candidates):
                raise SourceRunnerError("persisted candidate batch does not bind parent")
        for index, candidate in enumerate(candidates, start=1):
            if index <= len(state["candidates"]):
                continue
            stage = f"candidate:{index}"
            estimate = _stage_estimate(case, stage)
            permit = _reserve(ledger, stage, estimate)
            if permit is None:
                _state(root, state, ledger, evaluator)
                return _result(archive, authority, state, complete=False)
            archive.add(candidate)
            try:
                train = evaluator.train(candidate)
                actual = _actual_use(train)
                eligible = bool(train.feasible)
                archive.close(candidate.fingerprint(), status="eligible" if eligible else "terminal", train_gain=float(train.mean_gain), task_cost=int(train.task_cost))
                entry = {"source": candidate.to_payload(), "gain": float(train.mean_gain), "cost": int(train.task_cost), "eligible": eligible}
            except Exception:
                archive.close(candidate.fingerprint(), status="terminal")
                entry = {"source": candidate.to_payload(), "gain": 0.0, "cost": 0, "eligible": False}
                actual = estimate
            closed = ledger.close_stage(permit, actual)
            state["candidates"].append(entry)
            state["proposed"] = int(state["proposed"]) + 1
            state["eligible"] = int(state["eligible"]) + int(entry["eligible"])
            state["draw_counter"] = int(state["draw_counter"]) + 1
            state["completed_stage_ids"].append(stage)
            _state(root, state, ledger, evaluator)
            if stop_after == stage or not closed.allowed:
                return _result(archive, authority, state, complete=False)
        viable = [entry for entry in state["candidates"] if entry["eligible"] and entry["gain"] > 1e-12]
        if not viable:
            return _finish(root, archive, authority, state, ledger, evaluator)
        finalist = min(viable, key=lambda entry: (-entry["gain"], entry["cost"], SourceVariantV2.from_payload(entry["source"]).fingerprint()))
        state["finalist"] = finalist["source"]
        state["phase"] = "validation"
        _state(root, state, ledger, evaluator)

    finalist_payload = state["finalist"]
    if not isinstance(finalist_payload, Mapping):
        raise SourceRunnerError("validation phase has no finalist")
    finalist = SourceVariantV2.from_payload(finalist_payload)
    if state["phase"] == "validation":
        estimate = _stage_estimate(case, "validation")
        permit = _reserve(ledger, "validation", estimate)
        if permit is None:
            _state(root, state, ledger, evaluator)
            return _result(archive, authority, state, complete=False)
        validation = evaluator.validate(authority.active_source(), finalist)
        payload = _evidence("validation", validation)
        write_once_json(root / "sealed" / f"{fingerprint_payload(payload)}.json", payload)
        state["validation_passed"] = validation.passed
        state["validation_receipt"] = payload
        state["completed_stage_ids"].append("validation")
        state["phase"] = "canary_pending" if validation.passed else "complete"
        closed = ledger.close_stage(permit, _actual_use(validation))
        _state(root, state, ledger, evaluator)
        if stop_after == "validation" or not closed.allowed:
            return _result(archive, authority, state, complete=False)
        if not validation.passed:
            return _finish(root, archive, authority, state, ledger, evaluator)

    if state["phase"] == "canary_pending":
        receipt = state["validation_receipt"]
        if not isinstance(receipt, Mapping):
            raise SourceRunnerError("canary pending state has no sealed validation receipt")
        validation = SimpleNamespace(**dict(receipt))
        if not validation.passed:
            raise SourceRunnerError("sealed validation did not pass")
        estimate = _stage_estimate(case, "canary")
        permit = _reserve(ledger, "canary", estimate)
        if permit is None:
            _state(root, state, ledger, evaluator)
            return _result(archive, authority, state, complete=False)
        authority.begin_canary(finalist, validation)
        canary = evaluator.canary(authority.active_source(), finalist, epoch_seed=config.seed + 1)
        authority.finish_canary(canary)
        closed = ledger.close_stage(permit, _actual_use(canary))
        if canary.passed:
            state["activated"] = int(state["activated"]) + 1
        else:
            state["rolled_back"] = int(state["rolled_back"]) + 1
        state["completed_stage_ids"].append("canary")
        _state(root, state, ledger, evaluator)
        if not closed.allowed:
            return _result(archive, authority, state, complete=False)
        return _finish(root, archive, authority, state, ledger, evaluator)
    return _finish(root, archive, authority, state, ledger, evaluator)


__all__ = ["SourceRunnerError", "run_source_evolution"]
