"""Unlimited wall time retains accounting, finite work, and terminal outcomes."""
from dataclasses import replace
import json

import pytest

from common.llm import TransientLLMError
from evolving_loop.v2.budget import BudgetLedger, BudgetPlan, ResourceUse
from evolving_loop.v2.real.runner import run_real_evolution, RealRunnerError
from tests.test_evolution_v2_budget import FakeClock
from tests.test_evolution_v2_real_runner import RunnerCase
from tests.test_evolution_v2_protocol_compatibility import compatibility_case
from tests.test_evolution_v2_cooperative_runner import run_case


def test_unlimited_ledger_accounts_wall_time_and_enforces_other_resources():
    clock = FakeClock()
    plan = BudgetPlan(0, 0.1, ResourceUse(llm_calls=2))
    ledger = BudgetLedger(plan, monotonic=clock)
    clock.advance(10_000_000.0)
    permit = ledger.reserve_stage("slow", ResourceUse(llm_calls=1))
    assert permit.allowed
    assert ledger.close_stage(permit, ResourceUse(wall_seconds=5_000_000.0, llm_calls=1)).allowed
    assert ledger.charge(ResourceUse(wall_seconds=5_000_000.0)).allowed
    resumed = BudgetLedger.resume(plan, ledger.checkpoint(), monotonic=clock)
    assert resumed.elapsed_wall_seconds == 10_000_000.0
    assert resumed.charged_use.wall_seconds == 10_000_000.0
    assert resumed.can_open_stage(ResourceUse(llm_calls=1)).allowed
    assert resumed.can_open_stage(ResourceUse(llm_calls=2)).reason == "llm_calls_exhausted"
    resumed.begin_finalization()
    assert resumed.can_open_stage(ResourceUse()).reason == "finalization_started"


def test_unlimited_reservation_still_rejects_non_time_overrun():
    ledger = BudgetLedger(BudgetPlan(0, 0.0, ResourceUse(llm_calls=2)), monotonic=lambda: 0.0)
    permit = ledger.reserve_stage("one", ResourceUse(llm_calls=1))
    assert ledger.close_stage(permit, ResourceUse(llm_calls=2)).reason == "budget_overrun"


def test_root_unlimited_completes_slow_stages_and_rejects_mode_change(tmp_path):
    case = RunnerCase(tmp_path)
    case.stage_durations = {stage: 1_000_000 for stage in case.stage_durations}
    contexts = []
    original = case._run
    def run(stage, context):
        contexts.append(context)
        return original(stage, context)
    case._run = run
    result = run_real_evolution(case.output, case.manifest, case.ports,
        monotonic=case.clock, p2_generations=3, no_time_limit=True)
    assert result.status == "complete"
    assert all(context.grant_seconds == 0 and context.deadline_monotonic is None for context in contexts)
    assert [record.charged_seconds for record in result.stage_records] == [1_000_000] * 4
    checkpoint = json.loads((case.output / "checkpoint.json").read_text())
    assert checkpoint["budget_checkpoint"]["prior_elapsed_wall_seconds"] == 4_000_000.0
    assert json.loads((case.output / "run_manifest.json").read_text())["no_time_limit"] is True
    assert run_real_evolution(case.output, case.manifest, case.ports,
        monotonic=case.clock, p2_generations=3, no_time_limit=True) == result
    with pytest.raises(RealRunnerError, match="manifest"):
        run_real_evolution(case.output, case.manifest, case.ports,
            monotonic=case.clock, p2_generations=3)


def test_root_unlimited_requires_finite_generation_cap_before_writing(tmp_path):
    case = RunnerCase(tmp_path)
    with pytest.raises(ValueError, match="p2_generations"):
        run_real_evolution(case.output, case.manifest, case.ports, no_time_limit=True)
    assert not case.output.exists()


def test_unlimited_seal_failure_keeps_measured_stage_charge(tmp_path):
    from tests.test_evolution_v2_real_runner import SimulatedSealFailure
    case = RunnerCase(tmp_path)
    case.seal_failure_stage = "p2"
    with pytest.raises(SimulatedSealFailure):
        run_real_evolution(case.output, case.manifest, case.ports,
            monotonic=case.clock, p2_generations=1, no_time_limit=True)
    checkpoint = json.loads((case.output / "checkpoint.json").read_text())
    assert checkpoint["phase"] == "FAILED"
    assert checkpoint["stage_records"][0]["charged_seconds"] == 840


def test_stage_deadline_exception_persists_terminal_failure_and_diagnostics(tmp_path):
    case = RunnerCase(tmp_path)
    original = case._run
    def run(stage, context):
        if stage == "p3":
            case.clock.advance(13)
            raise TransientLLMError("Codex CLI stage deadline is exhausted")
        return original(stage, context)
    case._run = run
    with pytest.raises(TransientLLMError, match="deadline"):
        case.run()
    checkpoint = json.loads((case.output / "checkpoint.json").read_text())
    assert checkpoint["phase"] == "FAILED"
    assert checkpoint["active_stage"] is None
    assert checkpoint["stage_records"][-1]["charged_seconds"] == 13
    failure = json.loads((case.output / "p3/root_stage_error.json").read_text())
    assert failure["error_type"] == "TransientLLMError"
    assert "deadline is exhausted" in failure["message"]
    assert case.resume().status == "failed"


def test_unlimited_cli_requires_cap_before_host_creation(tmp_path, monkeypatch, capsys):
    from evolving_loop.v2 import cli
    monkeypatch.setattr(cli, "build_real_host", lambda *a, **k: pytest.fail("must validate first"))
    assert cli.main(["real-evolve", "--manifest", str(tmp_path / "missing"),
        "--output-dir", str(tmp_path / "out"), "--no-time-limit"]) == 2
    assert "p2_generations" in capsys.readouterr().err


def test_unlimited_p2_runs_generations_with_advancing_clock(tmp_path):
    from tests.test_evolution_v2_numerical_runner import fixture
    from evolving_loop.v2.numerical_qd.runner import run_numerical_qd
    class Clock:
        now = 0.0
        def __call__(self):
            self.now += 100_000.0
            return self.now
    config, supply, folds, adapter = fixture(task_budget=2760, clock=Clock())
    config = replace(config, budget=BudgetPlan(0, 0.2,
        replace(config.budget.ceilings, wall_seconds=0.0)))
    result = run_numerical_qd(tmp_path / "p2", config, supply, folds, adapter, finalize_after=2)
    assert result.status == "numerical_qd_complete"
    assert len(list((tmp_path / "p2/numerical_qd/proposals").glob("*.json"))) == 2
    completion = json.loads((tmp_path / "p2/evaluation_complete.json").read_text())
    assert completion["summary"]


@pytest.mark.parametrize("provider", ["deterministic", "llm", "hybrid"])
def test_unlimited_proposers_accept_zero_wall_remaining(provider):
    from tests.test_evolution_v2_numerical_proposers import request_args, ScriptedClient, raw_response
    from evolving_loop.v2.numerical_qd.proposers import (
        DeterministicProposalProvider, LLMProposalProvider, HybridProposalProvider)
    request = request_args(no_time_limit=True)
    request["remaining_budget"]["wall_seconds"] = 0.0
    clock = FakeClock()
    class SlowClient(ScriptedClient):
        def complete(self, **kwargs):
            clock.advance(100_000.0)
            return super().complete(**kwargs)
    llm = LLMProposalProvider(SlowClient(json.dumps(raw_response())), monotonic=clock)
    deterministic = DeterministicProposalProvider(monotonic=clock)
    ledger = BudgetLedger(BudgetPlan(0, 0.1, ResourceUse.from_payload(request["remaining_budget"])), monotonic=clock)
    providers = {"llm": llm, "deterministic": deterministic,
        "hybrid": HybridProposalProvider(llm, deterministic, ledger)}
    batch = providers[provider].propose(request)
    assert batch.failure_reason is None
    assert batch.proposals
    if provider != "deterministic":
        assert batch.resource_use.wall_seconds == 100_000.0


def test_unlimited_hyperband_keeps_preferred_bracket_without_time_estimate():
    from evolving_loop.v2.numerical_qd.hyperband import choose_bracket
    from tests.test_evolution_v2_numerical_config import valid_config_payload
    assert choose_bracket(1, 0.0, None, valid_config_payload()["hyperband"]).name == "replay"


def test_unlimited_p3_finishes_all_four_steps_with_slow_evaluation(run_case, monkeypatch):
    from tests import test_evolution_v2_cooperative_runner as fixtures
    original = fixtures._control
    monkeypatch.setattr(fixtures, "_control", lambda scheduler: replace(original(scheduler), hard_limit_seconds=0))
    class SlowClock(FakeClock):
        def advance(self, seconds):
            super().advance(100_000.0 * seconds)
    monkeypatch.setattr(fixtures, "FakeClock", SlowClock)
    result = run_case.run(scheduler="ucb")
    assert result.status == "cooperative_complete"
    assert len(run_case.progress()) == 4


def test_unlimited_p4_preserves_elapsed_accounting_on_resume(monkeypatch):
    from evolving_loop.v2.source.runner import _resume_budget
    from evolving_loop.v2.source import runner
    clock = FakeClock()
    monkeypatch.setattr(runner.time, "monotonic", clock)
    plan = BudgetPlan(0, 0.1, ResourceUse())
    ledger = BudgetLedger(plan, monotonic=clock)
    clock.advance(500_000.0)
    assert _resume_budget(plan, ledger.checkpoint()).elapsed_wall_seconds == 500_000.0


def test_unlimited_p4_finishes_finite_candidates(tmp_path, monkeypatch):
    from evolving_loop.v2.source import SourceConfigV2, run_source_evolution, runner
    from tests.build_evolution_v2_source_fixture import build_source_case
    clock = FakeClock()
    monkeypatch.setattr(runner.time, "monotonic", clock)
    config = replace(SourceConfigV2.smoke(seed=17), hard_limit_seconds=0)
    case = build_source_case(tmp_path / "inputs")
    original = case.evaluator.train
    def slow_train(*args):
        clock.advance(100_000.0)
        return original(*args)
    monkeypatch.setattr(case.evaluator, "train", slow_train)
    result = run_source_evolution(tmp_path / "p4", config, case)
    assert result.status == "source_evolution_complete"
    assert result.proposed == config.max_candidates


def test_unlimited_p5_finishes_finite_proposals(compatibility_case, tmp_path):
    from tests.test_evolution_v2_protocol_runner import _config, _manifest
    from evolving_loop.v2.protocol import ProtocolRuntimeRegistry, run_protocol_evolution
    class Clock:
        now = 0.0
        def __call__(self):
            self.now += 100_000.0
            return self.now
    case = compatibility_case
    result = run_protocol_evolution(tmp_path, _config() | {"hard_limit_seconds": 0},
        _manifest(case), ProtocolRuntimeRegistry(), host_inputs=case.inputs, monotonic=Clock())
    assert result["status"] == "protocol_evolution_complete"
    assert result["accepted"] + result["rejected"] == 2


def test_real_derived_configs_use_explicit_unlimited_time_and_finite_work():
    from types import SimpleNamespace
    from evolving_loop.v2.real.runner import _derived_numerical_config, _derived_cooperative_config
    p2 = _derived_numerical_config(grant_seconds=0, runtime_fingerprints={"real_host": "1" * 64})
    p3 = _derived_cooperative_config(SimpleNamespace(grant_seconds=0), SimpleNamespace(resource_reporter_sha256="1" * 64))
    assert p2["budget"]["hard_limit_seconds"] == 0
    assert p2["budget"]["ceilings"]["wall_seconds"] == 0.0
    assert p2["adapter"]["task_timeout_seconds"] > 0
    assert p3["control"]["hard_limit_seconds"] == 0
    assert p3["resource_ceilings"]["wall_seconds"] == 0.0
    assert p3["max_steps"] == 4
