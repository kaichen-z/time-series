from __future__ import annotations

import json
import subprocess
from pathlib import Path

from common.llm import FakeLLMClient
from numerical_agent.dictionary import MethodCandidate
from numerical_agent.evolution import commit_module, init_repo
from numerical_agent.evolution.cache import OutcomeCache
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.module import MODULE_HEADER, parse_module, write_module
from numerical_agent.evolution.policy_targetwise import evolve_policies_once
from numerical_agent.evolution.portfolio import (
    FLAGSHIP_METHOD_IDS,
    PolicyOutcomeCache,
    PolicyPortfolio,
    read_policy_file,
    write_policy_file,
)
from numerical_agent.evolution.prompts import POLICY_MUTATE_SYSTEM, POLICY_SELECT_SYSTEM
from numerical_agent.providers import RuntimeRegistry


class OffsetRuntime:
    def supports(self, candidate: MethodCandidate) -> bool:
        return candidate.method_id in FLAGSHIP_METHOD_IDS

    def forecast(self, candidate, history, horizon, frequency):
        del candidate, frequency
        return [float(history[-1]) + 2.0] * horizon


def _registry() -> RuntimeRegistry:
    runtime = OffsetRuntime()
    return RuntimeRegistry({"timesfm": runtime, "chronos": runtime, "tsfm_worker": runtime})


def _repo(path: Path) -> Path:
    repo = init_repo(path)
    functions = []
    for name in (
        "seasonal_naive",
        "holt_damped_trend",
        "croston_sba",
        "robust_loess_trend",
        "median_seasonal_profile_forecast",
    ):
        functions.append(
            f'''def {name}(history, horizon, frequency):
    """Use as a statistical parent for policy evolution tests."""
    return [float(history[-1]) - 2.0] * horizon
'''
        )
    write_module(repo / "methods.py", parse_module(MODULE_HEADER + "\n\n" + "\n\n".join(functions)))
    write_policy_file(repo / "policies.py", PolicyPortfolio.flagship5())
    subprocess.run(["git", "add", "methods.py", "policies.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "seed portfolio"], cwd=repo, check=True)
    return repo


def _tasks(prefix: str) -> tuple[Task, ...]:
    return (
        Task(f"{prefix}1", tuple(float(value) for value in range(1, 29)), 2, "1 day", (30.0, 30.0)),
        Task(f"{prefix}2", tuple(float(value) for value in range(2, 30)), 2, "1 day", (31.0, 31.0)),
    )


def test_combined_policy_child_is_screened_validated_and_committed(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    selector = FakeLLMClient([json.dumps({"targets": [{
        "name": "combined_timesfm_seasonal",
        "action": "repair",
        "reason": "the TSFM parent is consistently more accurate",
    }]})])
    replacement = PolicyPortfolio.flagship5().combined[0].to_payload()
    replacement["weights"] = (0.90, 0.10)
    mutator = FakeLLMClient([json.dumps({
        "replacement": replacement,
        "reason": "increase TSFM weight after lower Train MASE",
    })])

    result = evolve_policies_once(
        repo,
        _tasks("train"),
        mutator,
        selector,
        generation=1,
        outcome_cache=OutcomeCache(tmp_path / "method-cache"),
        policy_cache=PolicyOutcomeCache(tmp_path / "policy-cache"),
        validation_tasks=_tasks("dev"),
        runtimes=_registry(),
        screen_tasks=1,
        max_targets=1,
        full_evaluation_candidates=1,
        isolate_methods=False,
    )

    assert result.candidates[0].accepted and result.candidates[0].promoted
    assert read_policy_file(repo / "policies.py").combined[0].weights == (0.90, 0.10)
    assert result.candidate_count == 15  # five test Python parents + ten policies


def test_policy_mutator_can_change_combined_parents(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    selector = FakeLLMClient([json.dumps({"targets": [{
        "name": "combined_timesfm_seasonal",
        "action": "repair",
        "reason": "test immutable lineage",
    }]})])
    replacement = PolicyPortfolio.flagship5().combined[0].to_payload()
    replacement["parents"] = ("timesfm_2_5", "holt_damped_trend")
    replacement["weights"] = (0.90, 0.10)
    mutator = FakeLLMClient(
        [json.dumps({"replacement": replacement, "reason": "swap parent and improve weight"})]
    )

    result = evolve_policies_once(
        repo, _tasks("train"), mutator, selector, generation=1,
        outcome_cache=OutcomeCache(tmp_path / "method-cache"),
        policy_cache=PolicyOutcomeCache(tmp_path / "policy-cache"),
        validation_tasks=_tasks("dev"), runtimes=_registry(), screen_tasks=1,
        max_targets=1, full_evaluation_candidates=1, isolate_methods=False,
    )

    assert result.candidates[0].accepted and result.candidates[0].promoted
    assert read_policy_file(repo / "policies.py").combined[0].parents == (
        "timesfm_2_5",
        "holt_damped_trend",
    )


def test_targetwise_combined_prompts_use_canonical_policy_schema() -> None:
    prompt = f"{POLICY_SELECT_SYSTEM}\n{POLICY_MUTATE_SYSTEM}"

    for obsolete in (
        "tsfm_parent",
        "statistical_parent",
        "tsfm_when",
        "route_signal",
        "route_threshold",
        "blend-versus-route mode",
    ):
        assert obsolete not in prompt
    for field in (
        "name",
        "parents",
        "operator",
        "weights",
        "signal",
        "threshold",
        "above_parent",
        "below_parent",
        "fallback_parent",
    ):
        assert field in prompt
    assert "parent identities are immutable" not in prompt
