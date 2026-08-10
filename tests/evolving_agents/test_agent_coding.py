"""Coding Agent generation, sandbox retry, and hindcast ranking, all against a scripted LLM."""

from __future__ import annotations

import dataclasses
import json

import pytest
from dr_cik.llm import FakeLLMClient

from evolving_agents.agents.coding import CodingAgent, CodingAgentConfig
from evolving_agents.bundles import load_seed
from evolving_agents.harness.hindcast import carve_hindcast_windows
from evolving_agents.models import NumericTaskView

PERIOD = 12
HISTORY = tuple(float(10 + (index % PERIOD)) for index in range(120))
VIEW = NumericTaskView(benchmark_id="task_x", history_values=HISTORY, prediction_length=6, frequency="H", seasonal_period=PERIOD)
WINDOWS = carve_hindcast_windows(HISTORY, 6, "H", n_windows=3)


def _hypothesis(assumption: str, code: str) -> str:
    return json.dumps({"assumption": assumption, "code": code})


SEASONAL = _hypothesis(
    "a 12-step cycle dominates",
    "def forecast(history, horizon, frequency):\n    tail = history[-12:]\n    return [tail[i % 12] for i in range(horizon)]\n",
)
FLAT = _hypothesis("the level is flat", "def forecast(history, horizon, frequency):\n    return [history[-1]] * horizon\n")
BUGGY = _hypothesis("this one is broken", "def forecast(history, horizon, frequency):\n    return undefined_name\n")
UNSAFE = _hypothesis("this one escapes", "import os\ndef forecast(history, horizon, frequency):\n    return [0.0] * horizon\n")


def _agent(responses: list[str], **overrides) -> CodingAgent:
    bundle = load_seed("coding")
    hyperparameters = {**bundle.hyperparameters, "k_hypotheses": overrides.pop("k_hypotheses", 3), "m_keep": overrides.pop("m_keep", 2)}
    return CodingAgent(
        FakeLLMClient(responses=responses),
        dataclasses.replace(bundle, hyperparameters=hyperparameters),
        CodingAgentConfig(**overrides),
    )


def test_ranks_the_better_hypothesis_first() -> None:
    result = _agent([SEASONAL, FLAT, FLAT]).run(VIEW, WINDOWS)
    assert len(result.candidates) == 2
    assert result.candidates[0].hypothesis.assumption_text == "a 12-step cycle dominates"
    assert result.candidates[0].rank == 1
    assert result.candidates[0].hindcast_score == pytest.approx(0.0, abs=1e-9)
    assert result.candidates[1].hindcast_score > result.candidates[0].hindcast_score


def test_m_keep_caps_the_ranked_candidates() -> None:
    result = _agent([SEASONAL, FLAT, SEASONAL], m_keep=1).run(VIEW, WINDOWS)
    assert len(result.candidates) == 1
    assert len(result.all_candidates) == 3


def test_broken_code_is_retried_once_then_kept_only_in_all_candidates() -> None:
    # 3 generated hypotheses, then one retry response for the buggy one that is still broken.
    agent = _agent([SEASONAL, BUGGY, FLAT, BUGGY])
    result = agent.run(VIEW, WINDOWS)

    assumptions = [candidate.hypothesis.assumption_text for candidate in result.candidates]
    assert "this one is broken" not in assumptions
    assert "this one is broken" in [candidate.hypothesis.assumption_text for candidate in result.all_candidates]
    assert result.llm_call_count == 4  # 3 generations + 1 retry


def test_a_successful_retry_rescues_the_candidate() -> None:
    # The 4th response is the retry for BUGGY, carrying code that actually runs.
    result = _agent([BUGGY, FLAT, FLAT, FLAT]).run(VIEW, WINDOWS)
    rescued = [candidate for candidate in result.all_candidates if candidate.hypothesis.assumption_text == "this one is broken"]
    assert rescued and rescued[0].sandbox_result.ok
    assert "this one is broken" in [candidate.hypothesis.assumption_text for candidate in result.candidates]


def test_unsafe_code_never_ranks() -> None:
    result = _agent([UNSAFE, FLAT, FLAT, UNSAFE], k_hypotheses=3).run(VIEW, WINDOWS)
    assert "this one escapes" not in [candidate.hypothesis.assumption_text for candidate in result.candidates]
    unsafe = [candidate for candidate in result.all_candidates if candidate.hypothesis.assumption_text == "this one escapes"][0]
    assert "not allowed" in unsafe.sandbox_result.error


def test_unparseable_response_is_recorded_as_a_parse_failure() -> None:
    result = _agent(["not json at all", FLAT, FLAT]).run(VIEW, WINDOWS)
    assert any(step.kind == "parse_failure" for step in result.steps)
    assert len(result.all_candidates) == 2


def test_response_missing_code_is_skipped() -> None:
    result = _agent([json.dumps({"assumption": "no code here"}), FLAT, FLAT]).run(VIEW, WINDOWS)
    assert "no code here" not in [candidate.hypothesis.assumption_text for candidate in result.all_candidates]


def test_reasoning_block_is_captured_on_the_hypothesis() -> None:
    wrapped = f"<think>The dips repeat every 12 steps.</think>{SEASONAL}"
    result = _agent([wrapped, FLAT, FLAT]).run(VIEW, WINDOWS)
    assert result.candidates[0].hypothesis.reasoning == "The dips repeat every 12 steps."


def test_all_candidates_failing_yields_no_ranked_candidates() -> None:
    result = _agent([BUGGY, BUGGY, BUGGY, BUGGY, BUGGY, BUGGY]).run(VIEW, WINDOWS)
    assert result.candidates == ()
    assert len(result.all_candidates) == 3


def test_running_without_hindcast_windows_still_returns_candidates() -> None:
    result = _agent([SEASONAL, FLAT, FLAT]).run(VIEW, ())
    assert len(result.candidates) == 2
    assert all(candidate.hindcast_score is None for candidate in result.candidates)


def test_backbone_is_offered_to_generated_code() -> None:
    uses_backbone = _hypothesis(
        "trust the foundation model",
        "def forecast(history, horizon, frequency, backbone):\n    return list(backbone)\n",
    )
    result = _agent([uses_backbone, FLAT, FLAT]).run(VIEW, (), backbone=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    assert result.candidates[0].forecast.mean == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_revision_request_reaches_the_prompt() -> None:
    llm = FakeLLMClient(responses=[SEASONAL, FLAT, FLAT])
    bundle = dataclasses.replace(load_seed("coding"), hyperparameters={"k_hypotheses": 3, "m_keep": 2})
    CodingAgent(llm, bundle).run(VIEW, (), revision_request="add a +20% level shift at t=3")
    assert "add a +20% level shift at t=3" in llm.calls[0]["messages"][0]["content"]
