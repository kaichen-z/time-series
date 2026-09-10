from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from evolving_loop.v2.budget import BudgetLedger, BudgetPlan, ResourceUse
from evolving_loop.v2.numerical_qd.contracts import (
    ConstraintReportV2, HyperbandBracketV2, HyperbandBudgetOutcomeV2,
    HyperbandStateV2, MorphologyCellV2, NumericalEvaluationV2,
    NumericalObjectiveVectorV2, RungManifestV2, TaskCacheRowV2, TrainTaskV2,
)
from evolving_loop.v2.numerical_qd.hyperband import (
    advance_hyperband, choose_bracket, evaluation_cache_key,
    execute_hyperband_rung, fixed_rung_manifest,
)


def sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


SPLIT, PROTOCOL = sha("train split"), sha("protocol")
METRIC, DESCRIPTOR, ADAPTER = sha("metric"), sha("descriptor"), sha("adapter")
RUNTIME = {"python": sha("python")}
CELL = MorphologyCellV2("low", "none", "low", "stable", "short", "statistical")
CONFIG = {"brackets": {"explore": [8, 32, 80], "confirm": [32, 80], "replay": [80]},
          "reduction_factor": 3}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def ledger(clock):
    return BudgetLedger(BudgetPlan(1000, 0.2, ResourceUse(
        wall_seconds=1000.0, task_executions=1000, llm_calls=1000,
        input_tokens=1000, output_tokens=1000, gpu_seconds=1000.0,
        subprocesses=1000, artifact_bytes=10000)), monotonic=clock)


def groups(count=80, size=1):
    return {f"entity-{start:03}": tuple(TrainTaskV2(
        f"task-{index:03}", f"entity-{start:03}", sha(f"bytes-{index}"),
        "train", SPLIT, PROTOCOL,
    ) for index in range(start, min(start + size, count)))
            for start in range(0, count, size)}


def manifest(resource=8, task_groups=None):
    return fixed_rung_manifest(groups() if task_groups is None else task_groups,
                               resource, SPLIT, PROTOCOL)


def state(count=3, name="explore", reduction=3):
    return HyperbandStateV2(
        HyperbandBracketV2.registered(name), tuple(sorted(sha(f"child-{i}") for i in range(count))),
        reduction, SPLIT, PROTOCOL, (),
    )


def evaluation(candidate, tasks, *, subset, bracket="explore", rung=0,
               objective=1.0, invalid=False):
    return NumericalEvaluationV2(
        1, candidate, sha("supply"), sha("registry"), subset, SPLIT, METRIC,
        DESCRIPTOR, ADAPTER, RUNTIME, PROTOCOL, "train", bracket, rung,
        tuple(sorted(tasks)), {task: "invalid" if invalid else "passed" for task in tasks},
        NumericalObjectiveVectorV2(*(objective,) * 5),
        ConstraintReportV2(not invalid, ("coverage",) if invalid else ()),
        (CELL,), ("execution_failure",) if invalid else (), (), ResourceUse().to_payload(),
    )


def completed_budget():
    return HyperbandBudgetOutcomeV2("completed", None, sha("reservation"),
                                   ResourceUse().to_payload(), sha("checkpoint"))


def rung_evaluations(current, rung_manifest):
    return tuple(evaluation(candidate, rung_manifest.task_ids,
                            subset=rung_manifest.fingerprint(), bracket=current.bracket.name,
                            rung=len(current.rungs), objective=float(index + 1))
                 for index, candidate in enumerate(current.active_candidates))


def run(current, rung_manifest, budget, callback, **kwargs):
    return execute_hyperband_rung(
        current, rung_manifest, budget, callback, metric_sha256=METRIC,
        descriptor_sha256=DESCRIPTOR, runtime_fingerprints=RUNTIME,
        adapter_sha256=ADAPTER,
        estimate_per_task=ResourceUse(wall_seconds=1.0, task_executions=1,
                                      subprocesses=1, gpu_seconds=1.0), **kwargs,
    )


def host(clock, budget, *, fail=False, invalid=False):
    def evaluate_task(candidate, task, account):
        assert len(budget.checkpoint()["open_reservations"]) == 1
        clock.advance(0.25)
        account(ResourceUse(subprocesses=1, gpu_seconds=0.125))
        if fail:
            raise RuntimeError("callback failure")
        return evaluation(candidate, (task.task_id,), subset=task.fingerprint(), invalid=invalid)
    return evaluate_task


@pytest.mark.parametrize("name,rungs", [("explore", (8, 32, 80)),
                                        ("confirm", (32, 80)), ("replay", (80,))])
def test_registered_brackets_have_exact_rungs(name, rungs):
    assert HyperbandBracketV2.registered(name).resources == rungs


@pytest.mark.parametrize("name,rungs", [("other", (8, 32, 80)), ("explore", (8, 32)),
                                        ("confirm", (32, 80, 100)), ("replay", (80, 32)),
                                        ("explore", (8.0, 32, 80))])
def test_bracket_contract_rejects_alternative_or_malformed_resources(name, rungs):
    with pytest.raises(ValueError):
        HyperbandBracketV2(name, rungs)


def test_manifests_are_cumulative_deterministic_and_keep_whole_entities():
    universe = groups(size=4)
    small = manifest(8, universe)
    medium = manifest(32, dict(reversed(tuple(universe.items()))))
    full = manifest(80, universe)
    assert small.task_ids == tuple(f"task-{i:03}" for i in range(8))
    assert medium.task_ids[:8] == small.task_ids
    assert full.task_ids[:32] == medium.task_ids
    assert small.canonical_bytes() == manifest(8, dict(reversed(tuple(universe.items())))).canonical_bytes()
    assert RungManifestV2.from_payload(small.to_payload()) == small
    with pytest.raises(TypeError):
        small.task_groups["other"] = ()


@pytest.mark.parametrize("resource,size,count", [(8, 3, 80), (80, 1, 79), (0, 1, 80),
                                                  (20, 1, 80), (True, 1, 80)])
def test_manifest_rejects_partial_entity_or_unavailable_or_unregistered_resource(resource, size, count):
    with pytest.raises(ValueError):
        manifest(resource, groups(count=count, size=size))


@pytest.mark.parametrize("field,value", [("split_sha256", sha("other split")),
                                         ("protocol_sha256", sha("other protocol")),
                                         ("entity_id", "wrong entity")])
def test_manifest_rejects_wrong_task_commitments(field, value):
    universe = groups()
    first = universe["entity-000"][0]
    universe["entity-000"] = (replace(first, **{field: value}),)
    with pytest.raises(ValueError):
        manifest(task_groups=universe)


def test_manifest_rejects_duplicate_task_ids_and_nontrain_tasks():
    universe = groups()
    universe["entity-001"] = (replace(universe["entity-001"][0], task_id="task-000"),)
    with pytest.raises(ValueError, match="unique"):
        manifest(task_groups=universe)
    with pytest.raises(ValueError):
        replace(groups()["entity-000"][0], split="dev")


@pytest.mark.parametrize("field", ["candidate", "task", "split", "metric", "descriptor",
                                    "runtime", "protocol", "adapter"])
def test_cache_identity_binds_every_dependency_independently(field):
    identity = dict(candidate=sha("candidate"), task=b"task bytes", split=SPLIT,
                    metric=METRIC, descriptor=DESCRIPTOR, runtime=RUNTIME,
                    protocol=PROTOCOL, adapter=ADAPTER)
    original = evaluation_cache_key(**identity)
    identity[field] = (b"changed task bytes" if field == "task" else
                       {"python": sha("changed")} if field == "runtime" else sha("changed"))
    assert evaluation_cache_key(**identity) != original
    assert len(original) == 64


@pytest.mark.parametrize("count,coverage,remaining,want", [
    (3, 0.0, 1000.0, "explore"), (3, 0.5, 1000.0, "confirm"),
    (3, 1.0, 1.0, "replay"), (1, 0.0, 80.0, "replay"),
])
def test_choose_bracket_uses_count_exact_coverage_and_remaining_budget(count, coverage, remaining, want):
    assert choose_bracket(count, coverage, remaining, CONFIG).name == want


@pytest.mark.parametrize("count,coverage,remaining", [(0, 0.0, 100.0), (3, -0.1, 100.0),
                                                     (3, 1.1, 100.0), (3, float("nan"), 100.0),
                                                     (3, 0.0, float("inf")), (3, 0.0, 0.0)])
def test_choose_bracket_rejects_invalid_or_unaffordable_plan(count, coverage, remaining):
    with pytest.raises(ValueError):
        choose_bracket(count, coverage, remaining, CONFIG)


def test_three_children_promote_three_two_one_with_frozen_completed_rungs():
    current = state()
    original = current.canonical_bytes()
    populations = [len(current.active_candidates)]
    for resource in (8, 32, 80):
        committed = manifest(resource)
        outcome = advance_hyperband(current, committed, rung_evaluations(current, committed), completed_budget())
        current = outcome.state
        populations.append(len(current.active_candidates))
        assert current.rungs[-1].manifest == committed
        assert outcome.evaluations == current.rungs[-1].evaluations
        assert HyperbandStateV2.from_payload(current.to_payload()) == current
        json.dumps(outcome.to_payload(), allow_nan=False)
    assert populations == [3, 2, 1, 1]
    assert current.complete
    assert state().canonical_bytes() == original


@pytest.mark.parametrize("count,reduction,want", [(10, 3, 3), (4, 8, 1), (8, 2, 4)])
def test_larger_populations_use_integer_reduction_with_one_survivor_minimum(count, reduction, want):
    current = state(count, reduction=reduction)
    committed = manifest()
    result = advance_hyperband(current, committed, rung_evaluations(current, committed), completed_budget())
    assert len(result.state.active_candidates) == want


def test_promotion_uses_constraints_and_pareto_ranking():
    current = state()
    committed = manifest()
    rows = list(rung_evaluations(current, committed))
    rows[0] = replace(rows[0], objectives=NumericalObjectiveVectorV2(*(0.0,) * 5),
                      constraints=ConstraintReportV2(False, ("coverage",)))
    result = advance_hyperband(current, committed, tuple(reversed(rows)), completed_budget())
    assert set(result.state.active_candidates) == {rows[1].genome_sha256, rows[2].genome_sha256}


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_split", "wrong_protocol", "wrong_subset",
                                   "wrong_task", "wrong_rung", "wrong_candidate"])
def test_advance_rejects_partial_or_mismatched_evaluations(change):
    current, committed = state(), manifest()
    rows = list(rung_evaluations(current, committed))
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    else:
        changes = {"wrong_split": {"split_sha256": sha("wrong")},
                   "wrong_protocol": {"protocol_fingerprint": sha("wrong")},
                   "wrong_subset": {"task_subset_sha256": sha("wrong")},
                   "wrong_task": {"task_ids": ("other",), "task_statuses": {"other": "passed"}},
                   "wrong_rung": {"rung": 1}, "wrong_candidate": {"genome_sha256": sha("wrong")}}
        rows[0] = replace(rows[0], **changes[change])
    with pytest.raises(ValueError):
        advance_hyperband(current, committed, rows, completed_budget())


@pytest.mark.parametrize("change", ["open", "partial", "skipped", "survivor", "resource"])
def test_resume_rejects_open_partial_or_inconsistent_rungs(change):
    current, committed = state(), manifest()
    result = advance_hyperband(current, committed, rung_evaluations(current, committed), completed_budget())
    payload = result.state.to_payload()
    rung = payload["rungs"][0]
    if change == "open":
        rung["status"] = "open"
    elif change == "partial":
        rung["evaluations"].pop()
    elif change == "skipped":
        rung["index"] = 1
    elif change == "survivor":
        rung["survivor_sha256s"] = [sha("outsider")]
    else:
        rung["manifest"]["resource"] = 32
    with pytest.raises(ValueError):
        HyperbandStateV2.from_payload(payload)


def test_execution_reserves_rung_before_callbacks_and_charges_actual_use():
    clock = FakeClock()
    budget = ledger(clock)
    result = run(state(count=1), manifest(), budget, host(clock, budget))
    assert result.budget_outcome.status == "completed"
    assert len(result.task_results) == len(result.cache_rows) == 8
    assert budget.charged_use == ResourceUse(wall_seconds=2.0, task_executions=8,
                                            subprocesses=8, gpu_seconds=1.0)
    assert budget.checkpoint()["open_reservations"] == []
    assert result.budget_outcome.ledger_checkpoint_sha256 == budget.checkpoint()["checkpoint_sha256"]
    json.dumps(result.to_payload(), allow_nan=False)


def test_callback_failure_closes_and_charges_actual_work_as_invalid_uncached_result():
    clock = FakeClock()
    budget = ledger(clock)
    result = run(state(count=1), manifest(), budget, host(clock, budget, fail=True))
    assert result.budget_outcome.status == "failed"
    assert result.task_results[0].status == "invalid"
    assert result.task_results[0].failure_category == "execution_failure"
    assert result.cache_rows == ()
    assert budget.charged_use == ResourceUse(wall_seconds=0.25, task_executions=1,
                                            subprocesses=1, gpu_seconds=0.125)
    assert budget.checkpoint()["open_reservations"] == []


@pytest.mark.parametrize("mode", ["deadline", "estimate", "finalization"])
def test_finalization_blocks_before_task_or_cache_access(mode):
    clock = FakeClock()
    budget = ledger(clock)
    if mode == "deadline":
        clock.advance(800.0)
    elif mode == "estimate":
        clock.advance(795.0)
    else:
        budget.begin_finalization()

    class UnreadCache(dict):
        def get(self, key, default=None):
            raise AssertionError("cache accessed after finalization boundary")

    def forbidden(*args):
        raise AssertionError("task accessed after finalization boundary")

    result = run(state(count=1), manifest(), budget, forbidden,
                 cache={} if mode == "estimate" else UnreadCache())
    assert result.budget_outcome.status == "blocked"
    assert result.task_results == ()
    assert budget.charged_use == ResourceUse()
    assert budget.checkpoint()["open_reservations"] == []


def test_deadline_is_rechecked_between_tasks_and_failure_work_is_charged():
    clock = FakeClock()
    budget = ledger(clock)

    def crosses_deadline(candidate, task, account):
        clock.advance(800.0)
        account(ResourceUse(subprocesses=2))
        return evaluation(candidate, (task.task_id,), subset=task.fingerprint())

    result = run(state(count=1), manifest(), budget, crosses_deadline)
    assert len(result.task_results) == 1
    assert result.budget_outcome.status == "failed"
    assert budget.charged_use.task_executions == 1
    assert budget.charged_use.wall_seconds == 800.0
    assert budget.checkpoint()["open_reservations"] == []


def test_completed_exact_tasks_are_reused_after_resume_without_double_charge():
    clock = FakeClock()
    budget = ledger(clock)
    initial = state(count=1)
    small = manifest()
    executed = run(initial, small, budget, host(clock, budget))
    cache = {row.cache_key: row.to_payload() for row in executed.cache_rows}
    advanced = advance_hyperband(initial, small, rung_evaluations(initial, small), executed.budget_outcome)
    restored_state = HyperbandStateV2.from_payload(advanced.state.to_payload())
    restored_clock = FakeClock()
    restored_budget = BudgetLedger.resume(budget.plan, budget.checkpoint(), monotonic=restored_clock)
    result = run(restored_state, manifest(32), restored_budget, host(restored_clock, restored_budget), cache=cache)
    assert sum(item.cache_hit for item in result.task_results) == 8
    assert restored_budget.charged_use.task_executions == 32
    assert restored_budget.charged_use.wall_seconds == 8.0
    assert len(result.cache_rows) == 32


@pytest.mark.parametrize("mode", ["missing", "malformed", "key", "task_bytes", "split", "metric",
                                  "descriptor", "runtime", "protocol", "adapter", "candidate", "failed"])
def test_missing_malformed_or_mismatched_rows_are_charged_misses(mode):
    clock = FakeClock()
    budget = ledger(clock)
    current = state(count=1)
    committed = manifest()
    task = committed.tasks[0]
    candidate = current.active_candidates[0]
    key = evaluation_cache_key(candidate, task.task_sha256, SPLIT, METRIC, DESCRIPTOR, RUNTIME, PROTOCOL, ADAPTER)
    row = TaskCacheRowV2(key, task.task_sha256,
                         evaluation(candidate, (task.task_id,), subset=task.fingerprint())).to_payload()
    if mode == "malformed":
        row = {"oops": True}
    elif mode == "key":
        row["cache_key"] = sha("wrong")
    elif mode == "task_bytes":
        row["task_sha256"] = sha("wrong")
    elif mode == "failed":
        row["evaluation"]["task_statuses"][task.task_id] = "failed"
    elif mode != "missing":
        field = {"split": "split_sha256", "metric": "metric_policy_sha256",
                 "descriptor": "descriptor_policy_sha256", "runtime": "runtime_fingerprints",
                 "protocol": "protocol_fingerprint", "adapter": "execution_adapter_sha256",
                 "candidate": "genome_sha256"}[mode]
        row["evaluation"][field] = {"python": sha("wrong")} if mode == "runtime" else sha("wrong")
    result = run(current, committed, budget, host(clock, budget), cache={} if mode == "missing" else {key: row})
    assert budget.charged_use.task_executions == 8
    assert not any(item.cache_hit for item in result.task_results)


def test_invalid_host_results_are_completed_but_never_cached_as_success():
    clock = FakeClock()
    budget = ledger(clock)
    result = run(state(count=1), manifest(), budget, host(clock, budget, invalid=True))
    assert result.budget_outcome.status == "completed"
    assert all(item.status == "invalid" for item in result.task_results)
    assert result.cache_rows == ()


def test_execution_rejects_an_open_ledger_rung_on_resume():
    clock = FakeClock()
    budget = ledger(clock)
    budget.reserve_stage("interrupted-rung", ResourceUse(task_executions=8))
    resumed = BudgetLedger.resume(budget.plan, budget.checkpoint(), monotonic=clock)
    with pytest.raises(ValueError, match="open"):
        run(state(count=1), manifest(), resumed, host(clock, resumed))


def test_runtime_digest_and_its_mapping_produce_the_same_cache_identity():
    from evolving_loop.v2.contracts import fingerprint_payload

    args = (sha("candidate"), b"task bytes", SPLIT, METRIC, DESCRIPTOR)
    assert evaluation_cache_key(*args, RUNTIME, PROTOCOL, ADAPTER) == evaluation_cache_key(
        *args, fingerprint_payload(RUNTIME), PROTOCOL, ADAPTER,
    )


def test_all_cached_replay_has_no_repeated_task_or_callback_wall_charge():
    class TickingClock(FakeClock):
        def __call__(self):
            self.now += 0.0001
            return self.now

    clock = TickingClock()
    budget = ledger(clock)
    current, committed = state(count=1, name="replay"), manifest(80)
    candidate = current.active_candidates[0]
    cache = {}
    for task in committed.tasks:
        key = evaluation_cache_key(candidate, task.task_sha256, SPLIT, METRIC, DESCRIPTOR,
                                   RUNTIME, PROTOCOL, ADAPTER)
        cache[key] = TaskCacheRowV2(key, task.task_sha256,
                                    evaluation(candidate, (task.task_id,), subset=task.fingerprint()))

    def forbidden(*args):
        raise AssertionError("exact cache hit executed again")

    result = run(current, committed, budget, forbidden, cache=cache)
    assert result.budget_outcome.status == "completed"
    assert len(result.task_results) == 80
    assert all(value.cache_hit for value in result.task_results)
    assert budget.charged_use == ResourceUse()


@pytest.mark.parametrize("name,resources", [("confirm", (32, 80)), ("replay", (80,))])
def test_every_bracket_round_trips_completed_prefixes(name, resources):
    current = state(count=2, name=name)
    for resource in resources:
        committed = manifest(resource)
        result = advance_hyperband(current, committed, rung_evaluations(current, committed), completed_budget())
        current = HyperbandStateV2.from_payload(result.state.to_payload())
    assert current.complete
    assert len(current.active_candidates) == 1
    with pytest.raises(ValueError, match="complete"):
        advance_hyperband(current, committed, (), completed_budget())


@pytest.mark.parametrize("change", ["protocol", "task_bytes"])
def test_next_rung_cannot_change_the_committed_universe(change):
    current, committed = state(), manifest()
    current = advance_hyperband(current, committed, rung_evaluations(current, committed), completed_budget()).state
    universe = groups()
    task = universe["entity-079"][0]
    universe["entity-079"] = (replace(task, task_sha256=sha("changed bytes")),)
    changed = manifest(32, universe)
    if change == "protocol":
        current = replace(current, rungs=(), protocol_sha256=sha("different protocol"))
        changed = manifest()
    with pytest.raises(ValueError):
        advance_hyperband(current, changed, rung_evaluations(current, changed), completed_budget())


@pytest.mark.parametrize("status", ["blocked", "failed"])
def test_pure_advance_rejects_unclosed_or_failed_budget_outcome(status):
    current, committed = state(), manifest()
    denied = HyperbandBudgetOutcomeV2(status, "finalization_reserve",
                                      None if status == "blocked" else sha("reservation"),
                                      ResourceUse().to_payload(), sha("checkpoint"))
    with pytest.raises(ValueError, match="completed"):
        advance_hyperband(current, committed, rung_evaluations(current, committed), denied)
    assert current.rungs == ()


@pytest.mark.parametrize("bad_response", [None, {"bad": float("inf")}])
def test_malformed_callback_response_is_invalid_and_actual_work_is_charged(bad_response):
    clock = FakeClock()
    budget = ledger(clock)

    def malformed(candidate, task, account):
        clock.advance(0.5)
        account(ResourceUse(subprocesses=1))
        return bad_response

    result = run(state(count=1), manifest(), budget, malformed)
    assert result.budget_outcome.status == "failed"
    assert result.task_results[0].failure_category == "invalid_response"
    assert budget.charged_use == ResourceUse(wall_seconds=0.5, task_executions=1, subprocesses=1)
    assert not budget.checkpoint()["open_reservations"]
    json.dumps(result.to_payload(), allow_nan=False)


def test_process_interruption_still_closes_actual_callback_work():
    clock = FakeClock()
    budget = ledger(clock)

    def interrupted(candidate, task, account):
        clock.advance(0.5)
        account(ResourceUse(gpu_seconds=0.25))
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(state(count=1), manifest(), budget, interrupted)
    assert budget.charged_use == ResourceUse(wall_seconds=0.5, task_executions=1, gpu_seconds=0.25)
    assert not budget.checkpoint()["open_reservations"]


def test_closed_ledger_without_completed_rung_checkpoint_cannot_repeat_work():
    clock = FakeClock()
    budget = ledger(clock)
    current, committed = state(count=1), manifest()
    run(current, committed, budget, host(clock, budget))
    before = budget.charged_use
    with pytest.raises(ValueError, match="completed state"):
        run(current, committed, budget, host(clock, budget))
    assert budget.charged_use == before
