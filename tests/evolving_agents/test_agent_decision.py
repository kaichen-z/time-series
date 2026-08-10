"""Decision Agent judging, and the rule that weights are computed in code, never taken from the LLM."""

from __future__ import annotations

import json

import pytest
from dr_cik.llm import FakeLLMClient
from dr_cik.models import EvidenceItem, Forecast

from evolving_agents.agents.decision import DecisionAgent, blend, compute_weights
from evolving_agents.bundles import load_seed
from evolving_agents.models import CodingCandidate, Hypothesis, SandboxResult

CONTRADICTS = json.dumps({"contradicts": True, "reason": "the assumption presumes no disruption"})
CLEAN = json.dumps({"contradicts": False, "reason": "unrelated commentary"})
EVIDENCE = (EvidenceItem(claim="Load is shed 02:00-06:00.", source_doc_ids=("doc_7",)),)


def _candidate(name: str, values: tuple[float, ...], score: float | None) -> CodingCandidate:
    return CodingCandidate(
        hypothesis=Hypothesis(hypothesis_id=name, assumption_text=f"assumption {name}", code="def forecast(): pass"),
        sandbox_result=SandboxResult(ok=True, forecast=values, error=None, duration_ms=1.0, code_hash=name),
        forecast=Forecast(mean=values, samples=(values,), method=name),
        hindcast_score=score,
    )


GOOD = _candidate("h0", (10.0, 10.0), 0.1)
BAD = _candidate("h1", (20.0, 20.0), 0.9)


def _agent(responses: list[str]) -> tuple[DecisionAgent, FakeLLMClient]:
    llm = FakeLLMClient(responses=responses)
    return DecisionAgent(llm, load_seed("decision")), llm


def test_weights_sum_to_one_and_favour_the_lower_error() -> None:
    weights = compute_weights((GOOD, BAD))
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["h0"] > weights["h1"]


def test_weights_handle_a_missing_hindcast_score() -> None:
    weights = compute_weights((GOOD, _candidate("h2", (5.0, 5.0), None)))
    assert sum(weights.values()) == pytest.approx(1.0)


def test_weights_are_even_for_a_single_candidate() -> None:
    assert compute_weights((GOOD,)) == {"h0": pytest.approx(1.0)}


def test_blend_is_the_weighted_mean() -> None:
    forecast = blend((GOOD, BAD), {"h0": 0.75, "h1": 0.25})
    assert forecast.mean == (pytest.approx(12.5), pytest.approx(12.5))


def test_weights_are_never_taken_from_the_model_response() -> None:
    # The model returns bogus weights; they must be ignored entirely.
    rogue = json.dumps({"contradicts": False, "reason": "fine", "weights": {"h0": 0.01, "h1": 0.99}})
    agent, _ = _agent([rogue, rogue])
    result = agent.decide((GOOD, BAD), EVIDENCE)
    assert result.weights["h0"] > result.weights["h1"]  # coded softmax, not the model's numbers
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_a_contradicted_candidate_is_discarded() -> None:
    agent, _ = _agent([CONTRADICTS, CLEAN])
    result = agent.decide((GOOD, BAD), EVIDENCE)
    assert result.weights == {"h1": pytest.approx(1.0)}
    discarded = [entry for entry in result.audit if not entry.kept]
    assert discarded[0].candidate_id == "h0"
    assert discarded[0].contradicting_evidence_ids == ("doc_7",)


def test_all_contradicted_still_forecasts_and_asks_for_a_revision() -> None:
    agent, _ = _agent([CONTRADICTS, CONTRADICTS])
    result = agent.decide((GOOD, BAD), EVIDENCE)
    assert result.revision_request is not None
    assert result.final_forecast.mean  # a forecast is still produced
    assert all(not entry.kept for entry in result.audit)


def test_revision_can_be_suppressed() -> None:
    agent, _ = _agent([CONTRADICTS, CONTRADICTS])
    assert agent.decide((GOOD, BAD), EVIDENCE, allow_revision=False).revision_request is None


def test_no_evidence_keeps_every_candidate_without_any_llm_call() -> None:
    agent, llm = _agent([])
    result = agent.decide((GOOD, BAD), ())
    assert all(entry.kept for entry in result.audit)
    assert result.llm_call_count == 0
    assert llm.calls == []


def test_an_unreadable_judgement_keeps_the_candidate() -> None:
    agent, _ = _agent(["I am not JSON", "I am not JSON"])
    result = agent.decide((GOOD, BAD), EVIDENCE)
    assert all(entry.kept for entry in result.audit)
    assert result.revision_request is None


def test_deciding_without_candidates_is_an_error() -> None:
    agent, _ = _agent([])
    with pytest.raises(ValueError):
        agent.decide((), EVIDENCE)


def test_the_judging_prompt_shows_one_candidate_and_one_claim() -> None:
    agent, llm = _agent([CLEAN, CLEAN])
    agent.decide((GOOD, BAD), EVIDENCE)
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "assumption h0" in prompt
    assert "Load is shed" in prompt
    assert "assumption h1" not in prompt  # candidates are judged one at a time
