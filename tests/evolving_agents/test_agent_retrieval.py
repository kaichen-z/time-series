"""Retrieval Agent filtering, citation validation, and the label-blindness of its prompt."""

from __future__ import annotations

import json

from dr_cik.llm import FakeLLMClient
from dr_cik.models import Document, ForecastTask

from evolving_agents.agents.retrieval import RetrievalAgent
from evolving_agents.bundles import load_seed

SUPPORTING = "Building B sheds all non-essential load 2025-10-16 02:00-06:00 for scheduled electrical maintenance."
DISTRACTOR = "The commercial real estate sector continues to focus on energy efficiency across 2025."


def _task(documents=None) -> ForecastTask:
    return ForecastTask(
        benchmark_id="task_1",
        entity_name="Arbor Gardens",
        target_name="electricity consumption",
        target_description="hourly campus draw in watts",
        frequency="H",
        prediction_length=4,
        seasonal_period=24,
        history_timestamps=tuple(str(index) for index in range(30)),
        history_values=tuple(float(100 + index % 5) for index in range(30)),
        future_timestamps=("a", "b", "c", "d"),
        future_values=(1.0, 2.0, 3.0, 4.0),
        documents=documents
        if documents is not None
        else (
            Document("doc_7", f"Arbor Gardens electricity. {SUPPORTING}", role="supporting", subtype=None),
            Document("doc_9", f"Arbor Gardens commentary. {DISTRACTOR}", role="distractor", subtype="noisy"),
        ),
        gt_evidence=({"id": "E1", "evidence": SUPPORTING},),
        labels_public=True,
    )


def _agent(responses: list[str]) -> tuple[RetrievalAgent, FakeLLMClient]:
    llm = FakeLLMClient(responses=responses)
    return RetrievalAgent(llm, load_seed("retrieval")), llm


KEEP_ONE = json.dumps({"evidence": [{"claim": "Load is shed 02:00-06:00 on 2025-10-16.", "source_doc_ids": ["doc_7"], "direction": "down", "timing": "2025-10-16"}]})


def test_keeps_the_cited_document() -> None:
    agent, _ = _agent([KEEP_ONE])
    result = agent.run(_task().agent_view())
    assert len(result.kept) == 1
    assert result.kept[0].source_doc_ids == ("doc_7",)
    assert result.llm_call_count == 1


def test_empty_evidence_is_a_valid_result() -> None:
    agent, _ = _agent([json.dumps({"evidence": []})])
    result = agent.run(_task().agent_view())
    assert result.kept == ()
    assert result.considered_doc_ids  # it still looked, it just kept nothing


def test_hallucinated_document_ids_are_dropped() -> None:
    response = json.dumps({"evidence": [{"claim": "invented", "source_doc_ids": ["doc_999"]}]})
    agent, _ = _agent([response])
    result = agent.run(_task().agent_view())
    assert result.kept[0].source_doc_ids == ()  # the claim survives, the bogus citation does not


def test_unparseable_response_keeps_nothing_without_raising() -> None:
    agent, _ = _agent(["I think document 7 matters."])
    result = agent.run(_task().agent_view())
    assert result.kept == ()
    assert any(step.kind == "parse_failure" for step in result.steps)


def test_no_individual_document_is_tagged_with_its_role() -> None:
    # The system prompt may discuss distractors in general -- the agent needs to know they exist.
    # What must never appear is a per-document label saying which one this is.
    agent, llm = _agent([KEEP_ONE])
    agent.run(_task().agent_view())
    documents_section = llm.calls[0]["messages"][0]["content"].split("Candidate documents")[1]
    for leaked in ("supporting", "distractor", "role", "subtype"):
        assert leaked not in documents_section.lower(), f"the document block leaked the label {leaked!r}"


def test_two_documents_differing_only_by_role_are_indistinguishable() -> None:
    shared = "Arbor Gardens electricity note with identical wording."
    documents = (
        Document("doc_a", shared, role="supporting", subtype=None),
        Document("doc_b", shared, role="distractor", subtype="confounder"),
    )
    agent, llm = _agent([json.dumps({"evidence": []})])
    agent.run(_task(documents=documents).agent_view())
    section = llm.calls[0]["messages"][0]["content"].split("Candidate documents")[1]
    assert section.count(shared) == 2  # both rendered identically, nothing distinguishes them


def test_prompt_contains_the_document_text_and_ids() -> None:
    agent, llm = _agent([KEEP_ONE])
    agent.run(_task().agent_view())
    prompt = llm.calls[0]["messages"][0]["content"]
    assert SUPPORTING[:40] in prompt
    assert "[doc_7]" in prompt


def test_an_empty_document_pool_short_circuits_the_llm() -> None:
    llm = FakeLLMClient(responses=[])
    agent = RetrievalAgent(llm, load_seed("retrieval"))
    result = agent.run(_task(documents=()).agent_view())
    assert result.kept == ()
    assert result.llm_call_count == 0
    assert llm.calls == []


def test_first_stage_caps_the_candidates() -> None:
    documents = tuple(
        Document(f"doc_{index}", f"Arbor Gardens electricity note {index}. {SUPPORTING}", role="distractor", subtype="noisy")
        for index in range(30)
    )
    agent, _ = _agent([json.dumps({"evidence": []})])
    result = agent.run(_task(documents=documents).agent_view())
    assert 0 < len(result.considered_doc_ids) <= agent.config.first_stage_top_n


def test_reasoning_is_stripped_before_json_parsing() -> None:
    agent, _ = _agent([f"<think>doc_7 is dated and quantitative; doc_9 is vague.</think>{KEEP_ONE}"])
    assert len(agent.run(_task().agent_view()).kept) == 1
