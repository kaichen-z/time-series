"""End-to-end task wiring, and the rule that only one revision round trip is ever honored."""

from __future__ import annotations

import dataclasses
import json

from dr_cik.llm import FakeLLMClient
from dr_cik.models import Document, ForecastTask

from evolving_agents.agents.coding import CodingAgent
from evolving_agents.agents.decision import DecisionAgent
from evolving_agents.agents.retrieval import RetrievalAgent
from evolving_agents.bundles import load_seed
from evolving_agents.harness.orchestrator import run_task

SUPPORTING = "Building B sheds load 2025-10-16 02:00-06:00 for maintenance."
FLAT = json.dumps({"assumption": "flat level", "code": "def forecast(history, horizon, frequency):\n    return [history[-1]] * horizon\n"})
BUGGY = json.dumps({"assumption": "broken", "code": "def forecast(history, horizon, frequency):\n    return nope\n"})
KEEP = json.dumps({"evidence": [{"claim": SUPPORTING, "source_doc_ids": ["doc_7"]}]})
NO_EVIDENCE = json.dumps({"evidence": []})
CONTRADICTS = json.dumps({"contradicts": True, "reason": "ignores the outage"})
CLEAN = json.dumps({"contradicts": False, "reason": "unrelated"})


def _task() -> ForecastTask:
    return ForecastTask(
        benchmark_id="task_1",
        entity_name="Arbor Gardens",
        target_name="electricity",
        target_description="hourly draw",
        frequency="H",
        prediction_length=4,
        seasonal_period=None,
        history_timestamps=tuple(str(index) for index in range(40)),
        history_values=tuple(float(100 + index % 7) for index in range(40)),
        future_timestamps=("a", "b", "c", "d"),
        future_values=(103.0, 104.0, 105.0, 106.0),
        documents=(
            Document("doc_7", f"Arbor Gardens electricity. {SUPPORTING}", role="supporting", subtype=None),
            Document("doc_9", "General sector commentary about efficiency.", role="distractor", subtype="noisy"),
        ),
        gt_evidence=({"id": "E1", "evidence": SUPPORTING},),
        labels_public=True,
    )


def _agents(coding_responses, retrieval_responses, decision_responses, k: int = 2):
    coding_bundle = dataclasses.replace(load_seed("coding"), hyperparameters={"k_hypotheses": k, "m_keep": 2, "temperature": 0.8})
    return (
        CodingAgent(FakeLLMClient(responses=coding_responses), coding_bundle),
        RetrievalAgent(FakeLLMClient(responses=retrieval_responses), load_seed("retrieval")),
        DecisionAgent(FakeLLMClient(responses=decision_responses), load_seed("decision")),
    )


def test_full_pipeline_produces_a_scored_forecast() -> None:
    coding, retrieval, decision = _agents([FLAT, FLAT], [KEEP], [CLEAN, CLEAN])
    trace = run_task(_task(), coding, retrieval, decision, n_windows=2)

    assert len(trace.forecast.mean) == 4
    assert trace.metrics["smae"] is not None
    assert len(trace.retrieval_result.kept) == 1
    assert not trace.revised


def test_only_one_revision_round_trip_is_honored() -> None:
    # Decision contradicts everything every time, so it would keep asking forever if uncapped.
    coding, retrieval, decision = _agents([FLAT] * 8, [KEEP], [CONTRADICTS] * 20)
    trace = run_task(_task(), coding, retrieval, decision, n_windows=1)

    assert trace.revised
    # A second request is refused, so the final decision carries none.
    assert trace.decision_result.revision_request is None
    assert trace.forecast.mean


def test_no_evidence_means_no_contradiction_calls() -> None:
    coding, retrieval, decision = _agents([FLAT, FLAT], [NO_EVIDENCE], [])
    trace = run_task(_task(), coding, retrieval, decision, n_windows=1)
    assert trace.retrieval_result.kept == ()
    assert trace.decision_result.llm_call_count == 0
    assert all(entry.kept for entry in trace.decision_result.audit)


def test_a_task_where_every_hypothesis_fails_still_returns_a_forecast() -> None:
    coding, retrieval, decision = _agents([BUGGY] * 8, [NO_EVIDENCE], [])
    trace = run_task(_task(), coding, retrieval, decision, n_windows=1)
    assert trace.coding_result.candidates == ()
    assert len(trace.forecast.mean) == 4
    assert "fallback" in trace.forecast.method


def test_running_without_a_retrieval_agent_skips_evidence() -> None:
    coding, _retrieval, decision = _agents([FLAT, FLAT], [], [])
    trace = run_task(_task(), coding, None, decision, n_windows=1)
    assert trace.retrieval_result.kept == ()
    assert trace.metrics["smae"] is not None


def test_fixed_evidence_bypasses_the_retrieval_agent() -> None:
    from evolving_agents.models import RetrievalEvidenceOutput
    from dr_cik.models import EvidenceItem

    coding, retrieval, decision = _agents([FLAT, FLAT], [], [CLEAN, CLEAN])
    injected = RetrievalEvidenceOutput(
        kept=(EvidenceItem(claim=SUPPORTING, source_doc_ids=("doc_7",)),), considered_doc_ids=("doc_7",)
    )
    trace = run_task(_task(), coding, retrieval, decision, n_windows=1, fixed_evidence=injected)
    assert trace.retrieval_result is injected
    assert retrieval.llm.calls == []


def test_metrics_include_the_retrieval_diagnostics() -> None:
    coding, retrieval, decision = _agents([FLAT, FLAT], [KEEP], [CLEAN, CLEAN])
    trace = run_task(_task(), coding, retrieval, decision, n_windows=1)
    assert "supp_doc_recall_cited" in trace.metrics
    assert "distractor_avoidance_cited" in trace.metrics
