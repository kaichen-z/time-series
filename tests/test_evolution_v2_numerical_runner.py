"""End-to-end Kernel authority with real mutation, QD, ranking and legacy fitting."""
from dataclasses import replace
import hashlib
import json

import pytest

from common.llm import LLMResponse
from evolving_loop.package_numerical_evolution import NumericalPackageMaterializer
from evolving_loop.v2.budget import ResourceUse
from evolving_loop.v2.fakes import FakeClock
from evolving_loop.v2.kernel import EvolutionKernel, KernelAuthorityError
from evolving_loop.v2.numerical_qd.adapters import LegacyNumericalAdapter
from evolving_loop.v2.numerical_qd.config import NumericalQDConfigV2
from evolving_loop.v2.numerical_qd.runner import run_numerical_qd
from evolving_loop.v2.store import V2RunStore
from numerical_agent.evolution.screening import ScreeningPolicy
from numerical_agent.evolution.task_local_evolution import build_group_fold_manifest
from tests.test_evolution_v2_numerical_adapters import CheapStore, SOURCE, screening_policy
from tests.test_evolution_v2_numerical_config import valid_config_payload
from tests.test_package_numerical_evolution import _evolution_tasks, _supply_parent
from tests.test_evolution_v2_kernel import kernel, child


def fixture(seed=7, *, task_budget=1840, provider="deterministic", clock=None, raw_seed=False):
    tasks = []
    for i, task in enumerate(_evolution_tasks()):
        history = (float(i),) * 20 if i < 40 else task.numeric.history_values
        tasks.append(replace(task, numeric=replace(task.numeric,
            history_values=history, future_values=(history[-1] + 1.0,) * 2,
            entity_name=f"entity-{i // 2:03d}")))
    tasks = tuple(tasks)
    folds = build_group_fold_manifest(tuple(t.numeric for t in tasks[:80]), seed=20260903)
    screen = screening_policy()
    screen = ScreeningPolicy(tuple(entry for entry in screen.entries if entry.name != "lagged"), screen.fallback_names)
    source = SOURCE.split("\ndef lagged")[0]
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    materializer = NumericalPackageMaterializer(forecast_store=CheapStore(), screening_policy=screen,
        fold_manifest=folds, original_tasks=tasks, source_fingerprints={"dictionary": source_sha},
        runtime_fingerprints={"materializer": "3" * 64})
    adapter = LegacyNumericalAdapter(materializer=materializer, tasks=tasks,
        fold_manifest=folds, sources={source_sha: source})
    adapter.monotonic = clock or FakeClock()
    payload = valid_config_payload()
    payload.update(profile="smoke", seed=seed, runtime_fingerprints={"materializer": "3" * 64})
    payload["mutation"]["operators"] = ["repair"]
    payload["proposer"].update(provider=provider, max_proposals_per_generation=3)
    payload["budget"]["ceilings"]["task_executions"] = task_budget
    payload["adapter"]["task_timeout_seconds"] = 0.01
    supply = _supply_parent()
    if not raw_seed:
        from evolving_loop.v2.numerical_qd.runner import _seed_registry
        from evolving_loop.v2.numerical_qd.adapters import import_numerical_seed
        supply = import_numerical_seed(supply, _seed_registry(supply, adapter), tasks=adapter.tasks)
    return NumericalQDConfigV2.from_payload(payload), supply, folds, adapter


def files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def run_fixture(root, **kwargs):
    options = {key: kwargs.pop(key) for key in tuple(kwargs) if key in {"seed", "task_budget", "provider", "clock"}}
    return run_numerical_qd(root, *fixture(**options), **kwargs)


def test_qd_runner_evolves_supply_policy_and_prompt(tmp_path):
    result = run_fixture(tmp_path / "run")
    assert result.status == "numerical_qd_complete"
    assert result.occupied_cells >= 2
    assert result.accepted_steps == 1
    assert result.rejected_steps == 1
    assert result.mutation_policy_sha256 != result.seed_mutation_policy_sha256
    assert result.proposer_prompt_sha256 != result.seed_proposer_prompt_sha256
    assert result.public_test_accessed is False
    assert result.budget["charged_use"]["task_executions"] == 1840
    assert result.budget["open_reservations"] == []


def test_acceptance_promotes_pair_and_rejection_keeps_exact_parent(tmp_path):
    config, supply, manifest, adapter = fixture()
    first = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    pointer = (tmp_path / "run/accepted_bundle.json").read_bytes()
    assert first.active_bundle.numerical_release_sha256 != supply.release.fingerprint
    seed_registry = json.loads((tmp_path / "run/run_manifest.json").read_bytes())["seed_bundle_sha256"]
    seed_bundle = json.loads((tmp_path / f"run/archive/objects/{seed_registry}.json").read_bytes())
    assert first.active_bundle.numerical_registry_sha256 != seed_bundle["numerical_registry_sha256"]
    resumed = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, resume=True)
    assert resumed.active_bundle.canonical_bytes() == first.active_bundle.canonical_bytes()
    assert (tmp_path / "run/accepted_bundle.json").read_bytes() == pointer
    records = [json.loads(line)["record"] for line in (tmp_path / "run/archive/index.jsonl").read_bytes().splitlines()]
    accepted = [row for row in records if row["artifact_kind"] == "acceptance_seal"]
    assert len(accepted) == 1
    assert accepted[0]["accepted_release_sha256s"] == sorted([
        first.active_bundle.numerical_release_sha256, first.active_bundle.numerical_registry_sha256])
    assert resumed.rejected_steps == 1


def test_stop_resume_is_byte_identical_and_completed_run_is_mtime_noop(tmp_path):
    run_fixture(tmp_path / "full")
    run_fixture(tmp_path / "resumed", stop_after=1)
    result = run_fixture(tmp_path / "resumed", resume=True)
    assert files(tmp_path / "full") == files(tmp_path / "resumed")
    before = {name: path.stat().st_mtime_ns for name in files(tmp_path / "resumed")
              for path in [tmp_path / "resumed" / name]}
    repeated = run_fixture(tmp_path / "resumed", resume=True)
    assert repeated.active_bundle == result.active_bundle
    assert before == {name: (tmp_path / "resumed" / name).stat().st_mtime_ns for name in before}


class Client:
    def __init__(self, response):
        self.response = response

    def complete(self, **kwargs):
        request = json.loads(kwargs["messages"][0]["content"])["request"]
        assert "future_values" not in json.dumps(request)
        if isinstance(self.response, Exception):
            raise self.response
        if self.response == "legal":
            state = request["parent_state"]
            member = state["inventory"]["members"][0]
            replacement = member | {"member_id": f'fresh_{request["counter_draw"]}',
                "source_sha256": "code", "parent_ids": [member["member_id"]]}
            response = {"source_candidates": [{"local_id": "code", "code": SOURCE.split("\ndef lagged")[0]}],
                "proposals": [{"operator": "repair", "reason": "Train repair", "member_id": member["member_id"], "replacement": replacement}]}
            return LLMResponse(json.dumps(response))
        return LLMResponse(self.response)


@pytest.mark.parametrize("response,reason,calls", [("{", "malformed", 1), (TimeoutError(), "timeout", 1),
                                                  (None, "unavailable", 0), ("legal", None, 1)])
def test_provider_attempts_are_kernel_charged_and_fallback_is_deterministic(tmp_path, response, reason, calls):
    client = None if response is None else Client(response)
    result = run_fixture(tmp_path / "run", task_budget=920, provider="hybrid", llm_client=client)
    attempts = [json.loads(path.read_bytes())["batch"] for path in (tmp_path / "run/numerical_qd/proposals").glob("*.json")]
    assert attempts[0]["attempts"][0]["failure_reason"] == reason
    assert result.budget["charged_use"]["llm_calls"] == calls
    assert result.accepted_steps == 1
    if reason:
        assert attempts[0]["attempts"][-1]["provider"] == "deterministic"
    checkpoint = json.loads((tmp_path / "run/checkpoint.json").read_bytes())
    assert set(checkpoint["budget_closures"]) == set(checkpoint["budget"]["closed_reservation_sha256s"])
    assert not checkpoint["budget"]["open_reservations"]


def test_account_only_closure_is_kernel_owned_and_cannot_be_promoted(kernel):
    parent = kernel.active_bundle()
    candidate = child(parent)
    permit = kernel.reserve_evaluation(candidate, ResourceUse(task_executions=2))
    result = kernel.close_evaluation(parent, candidate, permit=permit, status="passed",
        train_objectives={}, train_behavior_descriptors={},
        dev_comparison={"passed": False, "parent_metrics": {}, "candidate_metrics": {}},
        resource_use=ResourceUse(task_executions=1), account_only=True)
    assert kernel.budget.charged_use.task_executions == 1
    with pytest.raises(KernelAuthorityError):
        kernel.evaluate_transition(parent, candidate, target="retrieval", evaluation=result, permit=permit)
    kernel.finalize()
    restored = EvolutionKernel.resume(V2RunStore(kernel.store.root), kernel.budget.plan, monotonic=lambda: 0.0)
    assert restored.active_bundle() == parent


def test_account_only_cannot_carry_dev(kernel):
    parent = kernel.active_bundle()
    candidate = child(parent)
    permit = kernel.reserve_evaluation(candidate, ResourceUse(task_executions=1))
    with pytest.raises(KernelAuthorityError, match="Dev"):
        kernel.close_evaluation(parent, candidate, permit=permit, status="passed",
            train_objectives={}, train_behavior_descriptors={},
            dev_comparison={"passed": True, "parent_metrics": {"loss": 1.0}, "candidate_metrics": {"loss": 0.0}},
            resource_use=ResourceUse(), account_only=True)


def test_no_feasible_child_closes_work_preserves_parent_and_never_scores_dev(tmp_path):
    config, supply, manifest, adapter = fixture(task_budget=1, provider="hybrid")
    class InvalidSource(Client):
        def complete(self, **kwargs):
            response = super().complete(**kwargs)
            payload = json.loads(response.text)
            payload["source_candidates"][0]["code"] = SOURCE.split("\ndef lagged")[0].replace(
                "float(history[-1]) + 1.0", "float('nan')")
            return LLMResponse(json.dumps(payload))
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, InvalidSource("legal"))
    assert result.active_bundle.numerical_release_sha256 == supply.release.fingerprint
    assert result.accepted_steps == result.rejected_steps == 0
    assert result.occupied_cells == 0
    assert json.loads((tmp_path / "run/evaluation_complete.json").read_bytes())["summary"]["dev_accessed"] is False
    steps = [json.loads(path.read_bytes())["numerical_qd_step"]
        for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
        if "numerical_qd_step" in json.loads(path.read_bytes())]
    assert steps[0]["status"] == "no_feasible_child"
    assert not result.budget["open_reservations"]


def test_completed_but_infeasible_train_rung_records_closed_no_improvement(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    from evolving_loop.v2.numerical_qd.contracts import ConstraintReportV2
    config, supply, manifest, adapter = fixture(task_budget=920)
    original = runner.evaluate_numerical_child
    def constrained(*args, **kwargs):
        value = original(*args, **kwargs)
        return replace(value, constraints=ConstraintReportV2(False, ("joint_regret",)))
    monkeypatch.setattr(runner, "evaluate_numerical_child", constrained)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    steps = [json.loads(path.read_bytes())["numerical_qd_step"]
             for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
             if "numerical_qd_step" in json.loads(path.read_bytes())]
    assert steps[0]["status"] == "no_feasible_child"
    assert result.budget["charged_use"]["task_executions"] == 880
    assert result.occupied_cells == result.accepted_steps == 0


class ThreeChildren(Client):
    def complete(self, **kwargs):
        request = json.loads(kwargs["messages"][0]["content"])["request"]
        member = request["parent_state"]["inventory"]["members"][0]
        return LLMResponse(json.dumps({"source_candidates": [
            {"local_id": f"source_{i}", "code": SOURCE.split("\ndef lagged")[0]} for i in range(3)],
            "proposals": [{"operator": "add", "reason": "Train alternative", "member": member | {
                "member_id": f"new_{i}", "source_sha256": f"source_{i}", "parent_ids": []}} for i in range(3)]}))


@pytest.mark.parametrize("boundary,expected_tasks", [(0, 2400), (1, 2424), (2, 2472), (3, 2520)])
def test_each_rung_and_dev_deadline_blocks_new_dispatch(tmp_path, monkeypatch, boundary, expected_tasks):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=2560, provider="hybrid")
    payload = config.to_payload()
    payload["mutation"]["operators"] = ["add"]
    config = NumericalQDConfigV2.from_payload(payload)
    original_rung, original_freeze = runner._rung, runner.freeze_qd_supply
    calls = []
    def rung(*args, **kwargs):
        if len(calls) == boundary:
            adapter.monotonic.advance(config.budget.search_deadline_seconds)
        calls.append(len(calls))
        return original_rung(*args, **kwargs)
    def freeze(*args, **kwargs):
        value = original_freeze(*args, **kwargs)
        if boundary == 3:
            adapter.monotonic.advance(config.budget.search_deadline_seconds)
        return value
    monkeypatch.setattr(runner, "_rung", rung)
    monkeypatch.setattr(runner, "freeze_qd_supply", freeze)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, ThreeChildren("legal"))
    assert result.budget["charged_use"]["task_executions"] == expected_tasks
    assert result.accepted_steps == result.rejected_steps == 0
    assert not result.budget["open_reservations"]
    assert result.budget["finalization_started"]
    complete = json.loads((tmp_path / "run/evaluation_complete.json").read_bytes())
    assert complete["summary"]["dev_accessed"] is False
    assert len(list((tmp_path / "run/numerical_qd/rungs").glob("*.json"))) == min(boundary, 3)


def test_provider_receives_the_counter_already_persisted(tmp_path):
    class CounterClient(Client):
        def complete(self, **kwargs):
            checkpoint = json.loads((tmp_path / "run/numerical_qd/checkpoint.json").read_bytes())
            assert checkpoint["counter"]["counter"] > 0
            return super().complete(**kwargs)
    run_fixture(tmp_path / "run", task_budget=920, provider="hybrid", llm_client=CounterClient("legal"))


def test_kernel_resume_requires_the_exact_accepted_typed_pair(tmp_path):
    config, supply, manifest, adapter = fixture(task_budget=920)
    run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    checkpoints = json.loads((tmp_path / "run/checkpoint.json").read_bytes())
    accepted_candidate = next(candidate for candidate, value in checkpoints["completed_transitions"].items()
                              if value["decision"] == "accept")
    train = json.loads((tmp_path / f"run/evaluations/{accepted_candidate}/train.json").read_bytes())
    identity = train["train_behavior_descriptors"]["numerical_artifacts_sha256"]
    (tmp_path / f"run/numerical_qd/objects/{identity}.json").unlink()
    with pytest.raises(KernelAuthorityError):
        EvolutionKernel.resume(V2RunStore(tmp_path / "run"), config.budget, monotonic=adapter.monotonic)


def test_complete_explore_has_exact_cumulative_charges_and_a_frozen_winner(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=2560, provider="hybrid")
    payload = config.to_payload()
    payload["mutation"]["operators"] = ["add"]
    config = NumericalQDConfigV2.from_payload(payload)
    selected = []
    original = runner.freeze_qd_supply
    def freeze(*args, **kwargs):
        result = original(*args, **kwargs)
        selected.extend(result.selected_genome_sha256s)
        return result
    monkeypatch.setattr(runner, "freeze_qd_supply", freeze)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, ThreeChildren("legal"))
    rungs = sorted((json.loads(path.read_bytes()) for path in (tmp_path / "run/numerical_qd/rungs").glob("*.json")),
                   key=lambda row: row["index"])
    assert [len(row["evaluations"]) for row in rungs] == [3, 2, 1]
    assert [row["budget_outcome"]["resource_use"]["task_executions"] for row in rungs] == [24, 48, 48]
    assert result.budget["charged_use"]["task_executions"] == 2560
    assert result.accepted_steps == 1
    assert len(list((tmp_path / "run/numerical_qd/results").rglob("*.json"))) == 120
    steps = [json.loads(path.read_bytes())["numerical_qd_step"]
             for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
             if "numerical_qd_step" in json.loads(path.read_bytes())]
    assert steps[0]["winner_genome_sha256"] in selected


def test_task_timeout_closes_actual_partial_work_without_a_rung(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=920)
    original = runner.evaluate_numerical_child
    def evaluate(*args, **kwargs):
        value = original(*args, **kwargs)
        adapter.monotonic.advance(0.02)
        return value
    monkeypatch.setattr(runner, "evaluate_numerical_child", evaluate)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    assert result.budget["charged_use"]["task_executions"] == 801
    assert result.budget["charged_use"]["wall_seconds"] == 0.02
    assert not list((tmp_path / "run/numerical_qd/rungs").glob("*.json"))
    assert not list((tmp_path / "run/numerical_qd/results").rglob("*.json"))
    assert result.occupied_cells == result.accepted_steps == 0
    assert not result.budget["open_reservations"]


def test_finalization_reserve_before_proposal_never_opens_work(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
    config, supply, manifest, adapter = fixture(task_budget=920)
    original = NumericalQDRunStore.write_state
    count = 0
    def checkpoint(*args, **kwargs):
        nonlocal count
        value = original(*args, **kwargs)
        count += 1
        if count == 1:
            adapter.monotonic.advance(config.budget.search_deadline_seconds)
        return value
    monkeypatch.setattr(NumericalQDRunStore, "write_state", checkpoint)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    assert result.budget["charged_use"] == ResourceUse().to_payload()
    assert result.budget["closed_reservation_sha256s"] == []
    assert not list((tmp_path / "run/numerical_qd/proposals").glob("*.json"))


def test_sources_policy_and_proposal_are_persisted_before_atomic_mutation(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    original = runner.apply_mutation
    def mutate(state, proposal):
        payload = proposal.to_payload()
        member = payload["replacement"]
        assert (tmp_path / f'run/numerical_qd/objects/{member["policy_sha256"]}.json').is_file()
        assert (tmp_path / f'run/numerical_qd/sources/{member["source_sha256"]}.py').is_file()
        assert any(payload in json.loads(path.read_bytes())["batch"]["proposals"]
                   for path in (tmp_path / "run/numerical_qd/proposals").glob("*.json"))
        return original(state, proposal)
    monkeypatch.setattr(runner, "apply_mutation", mutate)
    run_fixture(tmp_path / "run", task_budget=920)


def test_unresolved_source_policy_is_rejected_before_atomic_mutation(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    class UnknownPolicy(Client):
        def complete(self, **kwargs):
            response = json.loads(super().complete(**kwargs).text)
            response["proposals"][0]["replacement"]["policy_sha256"] = "f" * 64
            return LLMResponse(json.dumps(response))
    def forbidden_mutation(*args, **kwargs):
        raise AssertionError("unresolved policy crossed atomic mutation boundary")
    monkeypatch.setattr(runner, "apply_mutation", forbidden_mutation)
    result = run_fixture(tmp_path / "run", task_budget=1, provider="hybrid",
                         llm_client=UnknownPolicy("legal"), stop_after=1)
    assert result.budget["charged_use"]["task_executions"] == 0
    assert result.accepted_steps == 0


def test_no_feasible_generation_continues_with_distinct_context_and_train_attempt_credit(tmp_path):
    config, supply, manifest, adapter = fixture(task_budget=921, provider="hybrid")
    class OnceInvalid(Client):
        def __init__(self):
            super().__init__("legal")
            self.count = 0
        def complete(self, **kwargs):
            response = super().complete(**kwargs)
            self.count += 1
            if self.count == 1:
                payload = json.loads(response.text)
                payload["source_candidates"][0]["code"] = SOURCE.split("\ndef lagged")[0].replace(
                    "float(history[-1]) + 1.0", "float('nan')")
                return LLMResponse(json.dumps(payload))
            return response
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, OnceInvalid())
    assert result.accepted_steps == 1
    assert result.budget["charged_use"]["task_executions"] == 921
    records = [json.loads(path.read_bytes()) for path in (tmp_path / "run/numerical_qd/proposals").glob("*.json")]
    assert sorted(record["context"]["generation"] for record in records) == [1, 2]
    policy = json.loads((tmp_path / f"run/numerical_qd/objects/{result.mutation_policy_sha256}.json").read_bytes())
    assert policy["operators"]["repair"]["attempts"] == 2
    assert policy["operators"]["repair"]["feasible"] == 1
    assert json.loads((tmp_path / "run/evaluation_complete.json").read_bytes())["summary"]["provider_attempts"] == 2


def test_ticking_clock_keeps_proposal_estimate_bounded_and_uses_exact_kernel_budget(tmp_path):
    class TickClock:
        def __init__(self):
            self.seconds = 0.0
        def __call__(self):
            self.seconds += 0.0000001
            return self.seconds
    result = run_fixture(tmp_path / "run", task_budget=920, clock=TickClock())
    assert result.accepted_steps == 1
    checkpoint = json.loads((tmp_path / "run/checkpoint.json").read_bytes())
    completed = json.loads((tmp_path / "run/evaluation_complete.json").read_bytes())
    assert completed["summary"]["budget"] == checkpoint["budget"] == result.budget


def test_zero_worker_capacity_stops_without_repeated_proposal_dispatch(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=920)
    payload = config.to_payload()
    payload["budget"]["ceilings"]["subprocesses"] = 0
    config = NumericalQDConfigV2.from_payload(payload)
    original = runner.DeterministicProposalProvider.propose
    count = 0
    def propose(*args, **kwargs):
        nonlocal count
        count += 1
        assert count == 1, "dispatched another proposal after materialization budget denial"
        return original(*args, **kwargs)
    monkeypatch.setattr(runner.DeterministicProposalProvider, "propose", propose)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    assert result.budget["charged_use"]["task_executions"] == 1
    assert result.accepted_steps == 0


def test_mid_generation_checkpoint_cannot_silently_resample_or_replay(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore, NumericalQDStoreError
    config, supply, manifest, adapter = fixture()
    original = NumericalQDRunStore.write_state
    def checkpoint(store, **kwargs):
        result = original(store, **kwargs)
        if store._object(kwargs["hyperband_state_sha256"])["rungs"]:
            raise RuntimeError("stopped after a closed rung")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(NumericalQDRunStore, "write_state", checkpoint)
        with pytest.raises(RuntimeError, match="closed rung"):
            run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    before = files(tmp_path / "run")
    with pytest.raises(NumericalQDStoreError, match="unfinished generation"):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, resume=True, stop_after=1)
    assert files(tmp_path / "run") == before


@pytest.mark.parametrize("failure_at", [1, 3])
def test_materialization_charges_only_dispatches_begun_even_on_failure(tmp_path, failure_at):
    config, supply, manifest, adapter = fixture()
    class FailingAnchor(CheapStore):
        calls = 0
        def forecast(self, *args):
            self.calls += 1
            if self.calls == failure_at:
                raise ValueError("anchor failed")
            return super().forecast(*args)
    adapter.materializer.forecast_store = FailingAnchor()
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    assert adapter.materializer.forecast_store.calls == failure_at
    assert result.budget["charged_use"]["task_executions"] == 2 * failure_at
    assert result.budget["charged_use"]["subprocesses"] == 1
    assert not result.budget["open_reservations"]
    assert not list((tmp_path / "run/numerical_qd/rungs").glob("*.json"))


def test_successful_materialization_exact_dispatch_charge(tmp_path):
    result = run_fixture(tmp_path / "run", task_budget=5000, stop_after=1)
    assert result.accepted_steps == 1
    # 100 distinct histories x 2 methods x (full horizon + 3 hindcasts),
    # followed by Train80 and the Dev20 Parent/Child pair.
    assert result.budget["charged_use"]["task_executions"] == 920
    assert result.budget["charged_use"]["subprocesses"] == 1


def freeze_material_bytes(root, seed_sha):
    total = sum(path.stat().st_size for path in (root / "numerical_qd/proposals").glob("*.json"))
    for path in (root / "numerical_qd/objects").glob("*.json"):
        payload = json.loads(path.read_bytes())
        if set(payload) == {"supply", "registry"} and payload["registry"]["release_sha256"] != seed_sha:
            total += path.stat().st_size
    return total


def test_freeze_material_output_is_exactly_billed_once(tmp_path):
    config, supply, manifest, adapter = fixture(task_budget=920)
    root = tmp_path / "run"
    result = run_numerical_qd(root, config, supply, manifest, adapter)
    assert result.accepted_steps == 1
    assert result.budget["charged_use"]["artifact_bytes"] == freeze_material_bytes(root, supply.release.fingerprint)
    before = files(root)
    resumed = run_numerical_qd(root, config, supply, manifest, adapter, resume=True)
    assert resumed.budget == result.budget
    assert files(root) == before


@pytest.mark.parametrize("after_output", [False, True])
def test_freeze_failure_bills_only_material_already_produced(tmp_path, monkeypatch, after_output):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=920)
    root = tmp_path / "run"
    if after_output:
        original = runner._persist
        def persist(store, value):
            result = original(store, value)
            if type(value) is dict and set(value) == {"supply", "registry"} and (
                    value["registry"]["release_sha256"] != supply.release.fingerprint):
                raise ValueError("freeze failed after material output")
            return result
        monkeypatch.setattr(runner, "_persist", persist)
    else:
        def freeze(*args, **kwargs):
            raise ValueError("freeze failed before material output")
        monkeypatch.setattr(runner, "freeze_qd_supply", freeze)
    result = run_numerical_qd(root, config, supply, manifest, adapter, stop_after=1)
    assert result.budget["charged_use"]["task_executions"] == 880
    assert result.budget["charged_use"]["artifact_bytes"] == freeze_material_bytes(root, supply.release.fingerprint)
    assert not result.budget["open_reservations"]
    assert result.active_bundle.numerical_release_sha256 == supply.release.fingerprint
    before = files(root)
    resumed = run_numerical_qd(root, config, supply, manifest, adapter, resume=True, stop_after=1)
    assert resumed.budget == result.budget
    assert files(root) == before


@pytest.mark.parametrize("failure_at", [1, 7])
def test_partial_dev_failure_charges_only_comparisons_started(tmp_path, monkeypatch, failure_at):
    from evolving_loop.v2.numerical_qd import runner
    original = runner.drcik_point_metrics
    calls = 0
    def metric(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise ValueError("Dev metric failed")
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "drcik_point_metrics", metric)
    result = run_fixture(tmp_path / "run", stop_after=1)
    assert calls == failure_at
    assert result.accepted_steps == 0
    assert result.budget["charged_use"]["task_executions"] == 880 + failure_at
    assert not result.budget["open_reservations"]


@pytest.mark.parametrize("mode", ["success", "task_overrun", "deadline"])
def test_raw_seed_bootstrap_is_precommitted_charged_and_never_replayed(tmp_path, monkeypatch, mode):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=5000 if mode != "task_overrun" else 1, raw_seed=True)
    calls = 0
    class BootstrapStore(CheapStore):
        def forecast(self, *args):
            nonlocal calls
            preflight = json.loads((tmp_path / "run/seed_bootstrap_preflight.json").read_bytes())
            assert preflight["stage"] == "seed_bootstrap"
            assert not (tmp_path / "run/run_manifest.json").exists()
            calls += 1
            if mode == "deadline" and calls == 1:
                adapter.monotonic.advance(config.budget.search_deadline_seconds + 1.0)
            return super().forecast(*args)
    adapter.materializer.forecast_store = BootstrapStore()
    original = runner._seed_registry
    def seed_registry(*args, **kwargs):
        assert (tmp_path / "run/seed_bootstrap_preflight.json").is_file()
        result = original(*args, **kwargs)
        adapter.materializer.forecast_store = CheapStore()
        return result
    monkeypatch.setattr(runner, "_seed_registry", seed_registry)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    assert calls > 0
    assert result.budget["charged_use"]["task_executions"] == calls + (920 if mode == "success" else 0)
    assert result.accepted_steps == int(mode == "success")
    assert result.budget["charged_use"]["wall_seconds"] == (
        config.budget.search_deadline_seconds + 1.0 if mode == "deadline" else 0.0)
    assert not result.budget["open_reservations"]
    if mode != "success":
        assert result.status == "numerical_qd_complete"
        assert not list((tmp_path / "run/numerical_qd/proposals").glob("*.json"))
    before = files(tmp_path / "run")
    resumed = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, resume=True, stop_after=1)
    assert resumed.budget == result.budget
    assert files(tmp_path / "run") == before


def test_bootstrap_bridge_exposes_real_time_to_every_later_stage(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
    config, supply, manifest, adapter = fixture(task_budget=5000, raw_seed=True)
    original = NumericalQDRunStore.write_state
    def checkpoint(*args, **kwargs):
        result = original(*args, **kwargs)
        adapter.monotonic.advance(config.budget.search_deadline_seconds)
        return result
    monkeypatch.setattr(NumericalQDRunStore, "write_state", checkpoint)
    result = runner.run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    assert result.budget["charged_use"]["task_executions"] > 0
    assert result.accepted_steps == 0
    assert not list((tmp_path / "run/numerical_qd/proposals").glob("*.json"))


def test_raw_seed_preflight_zero_budget_never_opens_a_forecast(tmp_path):
    config, supply, manifest, adapter = fixture(task_budget=0, raw_seed=True)
    class ForbiddenStore:
        def forecast(self, *args):
            raise AssertionError("bootstrap forecast opened without admission")
    adapter.materializer.forecast_store = ForbiddenStore()
    with pytest.raises(ValueError, match="preflight budget denied"):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)


def test_bootstrap_authority_cannot_be_removed_from_the_kernel_manifest(tmp_path):
    from evolving_loop.v2.contracts import canonical_v2_bytes
    config, supply, manifest, adapter = fixture(task_budget=1, raw_seed=True)
    run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    path = tmp_path / "run/run_manifest.json"
    payload = json.loads(path.read_bytes())
    payload.pop("seed_bootstrap_preflight_sha256")
    path.write_bytes(canonical_v2_bytes(payload))
    with pytest.raises(KernelAuthorityError, match="bootstrap"):
        EvolutionKernel.resume(V2RunStore(tmp_path / "run"), config.budget, monotonic=adapter.monotonic)


def test_external_resource_declaration_is_rechecked_before_run_task_access(tmp_path):
    config, supply, manifest, adapter = fixture()
    class GPUStore(CheapStore):
        resource_kinds = ("gpu_seconds",)
        def forecast(self, *args):
            raise AssertionError("GPU access before reporter preflight")
    adapter.materializer.forecast_store = GPUStore()
    with pytest.raises(ValueError, match="resource reporter"):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    assert not (tmp_path / "run").exists()


def test_forecast_size_limit_is_enforced_before_any_task_access(tmp_path):
    config, supply, manifest, adapter = fixture(raw_seed=True)
    payload = config.to_payload()
    payload["adapter"]["max_forecast_values"] = 1
    config = NumericalQDConfigV2.from_payload(payload)
    with pytest.raises(ValueError, match="forecast.*limit"):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    assert not (tmp_path / "run").exists()


def test_reported_external_failure_closes_with_exact_gpu_native_token_use(tmp_path):
    config, supply, manifest, adapter = fixture()
    class GPUStore(CheapStore):
        resource_kinds = ("gpu_seconds", "subprocesses", "output_tokens")
        used = ResourceUse()
        def forecast(self, *args):
            self.used += ResourceUse(gpu_seconds=0.5, subprocesses=1, output_tokens=7)
            raise ValueError("GPU operation failed after beginning")
    external = GPUStore()
    adapter.materializer.forecast_store = external
    adapter.resource_reporter = lambda: external.used
    adapter.resource_reporter_sha256 = "a" * 64
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    charged = ResourceUse.from_payload(result.budget["charged_use"])
    assert charged.gpu_seconds == 0.5
    assert charged.output_tokens == 7
    assert charged.subprocesses == 2  # Verified source worker + reported native work.
    assert charged.task_executions == 2
    assert not result.budget["open_reservations"]


def test_rung_host_reporter_work_is_charged_even_when_evaluation_fails(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture()
    used = ResourceUse()
    adapter.resource_kinds = ("gpu_seconds", "output_tokens")
    adapter.resource_reporter = lambda: used
    adapter.resource_reporter_sha256 = "a" * 64
    def evaluate(*args, **kwargs):
        nonlocal used
        used += ResourceUse(gpu_seconds=0.25, output_tokens=3)
        raise ValueError("Host evaluation failed after GPU work")
    monkeypatch.setattr(runner, "evaluate_numerical_child", evaluate)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    assert result.budget["charged_use"]["gpu_seconds"] == 0.25
    assert result.budget["charged_use"]["output_tokens"] == 3
    assert result.budget["charged_use"]["task_executions"] == 801
