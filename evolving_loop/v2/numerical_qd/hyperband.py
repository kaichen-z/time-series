"""Pure Hyperband advancement and a separate budgeted Host execution boundary.

The Host owns task materialization, per-task metrics, aggregate evaluations, and
persistence. This module returns immutable artifacts and never opens data files.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
import hashlib
import math

from ..budget import BudgetLedger, ResourceUse
from .config import NumericalQDConfigV2, _validate_hyperband
from .contracts import (
    HyperbandAdvanceV2, HyperbandBracketV2, HyperbandBudgetOutcomeV2,
    HyperbandExecutionV2, HyperbandRungV2, HyperbandStateV2, HyperbandTaskResultV2,
    NumericalEvaluationV2, NumericalQDEntryV2, RungManifestV2, TaskCacheRowV2,
    _cache_identity, _hyperband_survivor_count,
)
from .nsga2 import select_survivors


def fixed_rung_manifest(task_groups, resource, split_sha256, protocol_sha256) -> RungManifestV2:
    """Commit an exact lexicographic prefix; reject a boundary inside an entity."""

    return RungManifestV2(resource, split_sha256, protocol_sha256, task_groups)


def pack_fold_groups(groups):
    """Partition authenticated whole groups into the fixed nested rung increments."""

    rows = tuple(groups)
    seen_tasks = set()
    previous_sha = ""
    for row in rows:
        if type(row) is not tuple or len(row) != 3:
            raise ValueError("authenticated fold groups must be exact triples")
        group_sha, task_ids, fold = row
        if (
            type(group_sha) is not str
            or len(group_sha) != 64
            or group_sha <= previous_sha
            or type(task_ids) is not tuple
            or not task_ids
            or tuple(sorted(task_ids)) != task_ids
            or seen_tasks.intersection(task_ids)
            or type(fold) is not int
        ):
            raise ValueError("invalid authenticated fold group")
        previous_sha = group_sha
        seen_tasks.update(task_ids)
    capacities = (8, 24, 48)
    if len(seen_tasks) != sum(capacities):
        raise ValueError("cannot pack authenticated fold groups into fixed rungs")

    @lru_cache(maxsize=None)
    def assign(index, remaining):
        if index == len(rows):
            return () if remaining == (0, 0, 0) else None
        size = len(rows[index][1])
        for bin_index, available in enumerate(remaining):
            if size > available:
                continue
            following = list(remaining)
            following[bin_index] -= size
            suffix = assign(index + 1, tuple(following))
            if suffix is not None:
                return (bin_index, *suffix)
        return None

    assignment = assign(0, capacities)
    if assignment is None:
        raise ValueError("cannot pack authenticated fold groups into fixed rungs")
    rank = 0
    packed = []
    for bin_index in range(len(capacities)):
        bin_rows = []
        for (group_sha, task_ids, _fold), assigned_bin in zip(
            rows, assignment, strict=True
        ):
            if assigned_bin != bin_index:
                continue
            entity_id = f"fold-group-{rank:03d}-{group_sha}"
            bin_rows.append((entity_id, task_ids))
            rank += 1
        packed.append(tuple(bin_rows))
    return tuple(packed)


def evaluation_cache_key(candidate, task, split, metric, descriptor, runtime, protocol, adapter,
                         local_evidence_sha256=None) -> str:
    """Bind independent SHA identities; task may be exact bytes or their SHA256."""

    task_sha = hashlib.sha256(task).hexdigest() if type(task) is bytes else task
    return _cache_identity(candidate, task_sha, split, metric, descriptor, runtime, protocol, adapter,
                           local_evidence_sha256)


def choose_bracket(candidate_count, cache_coverage, remaining_budget, config) -> HyperbandBracketV2:
    """Choose before reservation using usable search seconds and exact coverage.

    Coverage is the Host-verified fraction of exact Train80 candidate/task rows.
    A single candidate or full cache prefers replay; coverage >= 32/80 prefers
    confirm; otherwise explore. Fall back to a cheaper initial rung if needed.
    A full config supplies its task timeout as the conservative per-task cost;
    a standalone Hyperband config uses one second. The ledger is authoritative.
    """

    if type(candidate_count) is not int or candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    for name, value in (("cache_coverage", cache_coverage), ("remaining_budget", remaining_budget)):
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= cache_coverage <= 1.0 or remaining_budget <= 0.0:
        raise ValueError("invalid cache coverage or exhausted remaining budget")
    policy = _validate_hyperband(config.hyperband if isinstance(config, NumericalQDConfigV2) else config)
    task_seconds = config.adapter["task_timeout_seconds"] if isinstance(config, NumericalQDConfigV2) else 1.0
    preferred = "replay" if candidate_count == 1 or cache_coverage == 1.0 else (
        "confirm" if cache_coverage >= 0.4 else "explore"
    )
    names = ("replay", "confirm", "explore") if preferred == "replay" else (
        ("confirm", "explore") if preferred == "confirm" else ("explore",)
    )
    for name in names:
        first = policy["brackets"][name][0]
        # Coverage is global: only replay can assume every cached task is in its
        # first rung. For a prefix, use the worst-case overlap with Train80.
        guaranteed_hits = max(0.0, 80.0 * cache_coverage - (80 - first))
        seconds = candidate_count * (first - guaranteed_hits) * task_seconds
        if seconds <= remaining_budget:
            return HyperbandBracketV2.registered(name)
    raise ValueError("remaining budget cannot fit an initial rung")


def _validate_next_manifest(state, manifest):
    if type(state) is not HyperbandStateV2 or type(manifest) is not RungManifestV2:
        raise TypeError("state and manifest must be Hyperband artifacts")
    if state.complete:
        raise ValueError("Hyperband bracket is already complete")
    if manifest.resource != state.bracket.resources[len(state.rungs)]:
        raise ValueError("manifest resource does not match the next rung")
    if (manifest.split_sha256, manifest.protocol_sha256) != (state.split_sha256, state.protocol_sha256):
        raise ValueError("manifest split/protocol does not match state")
    if state.rungs and manifest.task_groups != state.rungs[0].manifest.task_groups:
        raise ValueError("rungs must use the same committed Train universe")


def advance_hyperband(state, manifest, evaluations, budget_outcome) -> HyperbandAdvanceV2:
    """Pure transition from full completed aggregate evaluations and closed work.

    Ranking consumes Task 4's constraint dominance and Pareto/crowding selection.
    Failed/blocked executions cannot advance or produce a checkpointable rung.
    """

    _validate_next_manifest(state, manifest)
    budget = HyperbandBudgetOutcomeV2.from_payload(budget_outcome.to_payload())
    if budget.status != "completed":
        raise ValueError("advancement requires completed budgeted work")
    values = tuple(NumericalEvaluationV2.from_payload(value.to_payload()) for value in evaluations)
    if tuple(sorted(value.genome_sha256 for value in values)) != state.active_candidates:
        raise ValueError("partial rung or duplicate/mismatched candidate evaluations")
    index = len(state.rungs)
    count = _hyperband_survivor_count(state.bracket, len(state.candidate_sha256s), index,
                                     len(values), state.reduction_factor)
    entries = tuple(NumericalQDEntryV2(
        1, value.genome_sha256, value.fingerprint(), value.cells[0], value.task_ids,
        value.objectives, value.constraints, value.train_diagnostic_categories,
    ) for value in values)
    survivors = tuple(sorted(value.genome_sha256 for value in select_survivors(entries, count)))
    rung = HyperbandRungV2(index, manifest, values, survivors, budget)
    following = replace(state, rungs=(*state.rungs, rung))
    return HyperbandAdvanceV2(following, rung.evaluations)


def _scale_use(use, count):
    return ResourceUse(**{name: value * count for name, value in use.to_payload().items()})


def _task_evaluation(value, candidate, task, metric, descriptor, runtime, adapter):
    if type(value) is not NumericalEvaluationV2:
        raise ValueError("Host callback must return a single-task NumericalEvaluationV2")
    value = NumericalEvaluationV2.from_payload(value.to_payload())
    if (value.genome_sha256 != candidate or value.task_ids != (task.task_id,)
            or value.task_subset_sha256 != task.fingerprint()
            or value.split_sha256 != task.split_sha256
            or value.protocol_fingerprint != task.protocol_sha256
            or value.metric_policy_sha256 != metric or value.descriptor_policy_sha256 != descriptor
            or value.execution_adapter_sha256 != adapter or value.runtime_fingerprints != runtime):
        raise ValueError("task evaluation identity mismatch")
    return value


def _read_cache(cache, key, candidate, task, metric, descriptor, runtime, adapter,
                local_evidence_sha256=None):
    payload = cache.get(key)
    if payload is None:
        return None
    try:
        row = TaskCacheRowV2.from_payload(payload.to_payload() if type(payload) is TaskCacheRowV2 else payload)
        if (row.cache_key != key or row.task_sha256 != task.task_sha256
                or row.local_evidence_sha256 != local_evidence_sha256):
            return None
        _task_evaluation(row.evaluation, candidate, task, metric, descriptor, runtime, adapter)
        return row
    except (ValueError, TypeError, KeyError):
        return None


def execute_hyperband_rung(
    state, manifest, ledger, evaluate_task, *, metric_sha256, descriptor_sha256,
    runtime_fingerprints, adapter_sha256, estimate_per_task, cache=None,
    local_evidence_sha256_for=None,
) -> HyperbandExecutionV2:
    """Execute missing exact tasks under one real BudgetLedger reservation.

    ``evaluate_task(candidate_sha256, TrainTaskV2, account)`` is the Host boundary:
    it loads/verifies committed task bytes and returns a single-task evaluation
    whose task_subset_sha256 is the TrainTaskV2 fingerprint. The Host reports
    auxiliary actual ResourceUse increments through ``account`` as work occurs,
    including before failures. This helper counts each callback as one task and
    measures callback wall time using the ledger's monotonic clock; account must
    leave those two fields zero. Cache bookkeeping is covered by the global
    elapsed deadline, without recharging cached task execution time. Aggregate
    evaluation/persistence remains caller-owned.

    Cached evaluations retain their original bracket/rung and resource evidence.
    Current-rung cost is exclusively budget_outcome.resource_use. An interrupted
    or failed execution cannot become a completed HyperbandStateV2 rung.
    Persist budget_outcome.ledger_checkpoint exactly as returned; a later ledger
    checkpoint includes a new elapsed time and therefore has a different SHA.
    """

    _validate_next_manifest(state, manifest)
    if type(ledger) is not BudgetLedger or type(estimate_per_task) is not ResourceUse:
        raise TypeError("execution requires BudgetLedger and ResourceUse")
    if estimate_per_task.task_executions != 1 or estimate_per_task.wall_seconds <= 0.0:
        raise ValueError("per-task estimate must include one task and positive wall time")
    if not callable(evaluate_task):
        raise TypeError("evaluate_task must be callable")
    if local_evidence_sha256_for is not None and not callable(local_evidence_sha256_for):
        raise TypeError("local_evidence_sha256_for must be callable")
    cache = {} if cache is None else cache
    if not isinstance(cache, Mapping):
        raise TypeError("cache must be a mapping")
    checkpoint = ledger.checkpoint()
    if checkpoint["open_reservations"]:
        raise ValueError("resume rejects open or partial rung reservations")

    def outcome(status, reason, reservation=None, use=ResourceUse()):
        snapshot = ledger.checkpoint()
        return HyperbandBudgetOutcomeV2(status, reason, reservation, use.to_payload(),
                                       snapshot["checkpoint_sha256"], snapshot)

    # This gate precedes cache lookup and the Host's task-data access.
    gate = ledger.can_open_stage(ResourceUse())
    if not gate.allowed:
        return HyperbandExecutionV2((), (), outcome("blocked", gate.reason))
    work = []
    for candidate in state.active_candidates:
        for task in manifest.tasks:
            local_evidence_sha256 = (None if local_evidence_sha256_for is None
                                     else local_evidence_sha256_for(candidate, task))
            key = evaluation_cache_key(candidate, task.task_sha256, manifest.split_sha256,
                                       metric_sha256, descriptor_sha256, runtime_fingerprints,
                                       manifest.protocol_sha256, adapter_sha256, local_evidence_sha256)
            cached = _read_cache(cache, key, candidate, task, metric_sha256,
                                 descriptor_sha256, runtime_fingerprints, adapter_sha256,
                                 local_evidence_sha256)
            work.append((candidate, task, key, cached, local_evidence_sha256))
    missing = sum(cached is None for _, _, _, cached, _ in work)
    estimate = _scale_use(estimate_per_task, missing)
    stage_id = f"hyperband:{state.fingerprint()}:{manifest.fingerprint()}"
    reservation = ledger.reserve_stage(stage_id, estimate)
    if not reservation.allowed:
        if reservation.reason in ("stage_closed", "stage_already_open"):
            raise ValueError("resume rejects a closed/open rung without its completed state")
        return HyperbandExecutionV2((), (), outcome("blocked", reservation.reason))

    auxiliary = ResourceUse()
    tasks_started = 0
    callback_wall_seconds = 0.0
    results, rows = [], []
    failure = None

    def account(use):
        nonlocal auxiliary
        if type(use) is not ResourceUse or use.task_executions or use.wall_seconds:
            raise ValueError("account accepts auxiliary ResourceUse; task/wall are measured by Host wrapper")
        auxiliary = auxiliary + use

    try:
        for candidate, task, key, cached, local_evidence_sha256 in work:
            if ledger.finalization_started or ledger.elapsed_wall_seconds >= ledger.plan.search_deadline_seconds:
                failure = "finalization_reserve"
                break
            if cached is not None:
                value = cached.evaluation
                rows.append(cached)
            else:
                tasks_started += 1
                callback_started = ledger.elapsed_wall_seconds
                try:
                    value = evaluate_task(candidate, task, account)
                except Exception:
                    failure = "execution_failure"
                    results.append(HyperbandTaskResultV2(candidate, task.task_id, key, "invalid", None, False, failure,
                                                         local_evidence_sha256, 2 if local_evidence_sha256 is not None else 1))
                    break
                finally:
                    callback_wall_seconds += ledger.elapsed_wall_seconds - callback_started
                try:
                    value = _task_evaluation(value, candidate, task, metric_sha256,
                                             descriptor_sha256, runtime_fingerprints, adapter_sha256)
                except (ValueError, TypeError):
                    failure = "invalid_response"
                    results.append(HyperbandTaskResultV2(candidate, task.task_id, key, "invalid", None, False, failure,
                                                         local_evidence_sha256, 2 if local_evidence_sha256 is not None else 1))
                    break
                if value.task_statuses[task.task_id] == "passed" and value.constraints.feasible:
                    rows.append(TaskCacheRowV2(key, task.task_sha256, value, local_evidence_sha256,
                                                2 if local_evidence_sha256 is not None else 1))
            results.append(HyperbandTaskResultV2(candidate, task.task_id, key,
                                                value.task_statuses[task.task_id], value,
                                                cached is not None, None, local_evidence_sha256,
                                                2 if local_evidence_sha256 is not None else 1))
            consumed = auxiliary + ResourceUse(task_executions=tasks_started,
                                               wall_seconds=callback_wall_seconds)
            if any(getattr(consumed, name) > getattr(estimate, name) for name in ResourceUse.field_names()):
                failure = "budget_overrun"
                break
    finally:
        actual = auxiliary + ResourceUse(task_executions=tasks_started,
                                         wall_seconds=callback_wall_seconds)
        closed = ledger.close_stage(reservation, actual)
    if not closed.allowed:
        failure = closed.reason
    # Check the final callback too: a deadline crossing cannot promote a rung.
    if ledger.finalization_started or ledger.elapsed_wall_seconds >= ledger.plan.search_deadline_seconds:
        failure = failure or "finalization_reserve"
    status = "failed" if failure else "completed"
    return HyperbandExecutionV2(tuple(results), tuple(rows),
                                outcome(status, failure, reservation.reservation_sha256, actual))
