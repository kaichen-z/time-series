"""OpenDR's ReAct loop: normal finish, forced finish, and parse-failure fallback."""

from __future__ import annotations

import json

from dr_cik.agents.opendr import OpenDRAgent, OpenDRConfig
from dr_cik.llm import FakeLLMClient

from .conftest import requires_sample


@requires_sample
def test_opendr_finishes_after_plan_and_one_search(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    doc_id = view.documents[0].document_id
    plan = json.dumps({"sub_questions": ["q1"], "initial_queries": ["irradiance"]})
    search = json.dumps({"thought": "t", "action": {"name": "search", "args": {"query": "irradiance"}}})
    finish = json.dumps(
        {"thought": "t2", "action": {"name": "finish", "args": {"report": "r", "evidence": [{"claim": "c1", "source_doc_ids": [doc_id]}]}}}
    )
    llm = FakeLLMClient(responses=[plan, search, finish])
    result = OpenDRAgent(llm, OpenDRConfig(max_steps=6)).run(view)

    assert result.stop_reason == "finished"
    assert result.llm_call_count == 3
    assert result.report.evidence[0].source_doc_ids == (doc_id,)
    assert [step.kind for step in result.steps] == ["plan", "search", "finish"]


@requires_sample
def test_opendr_hallucinated_doc_id_is_dropped(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    plan = json.dumps({"sub_questions": [], "initial_queries": []})
    finish = json.dumps(
        {"thought": "t", "action": {"name": "finish", "args": {"report": "r", "evidence": [{"claim": "c1", "source_doc_ids": ["not_a_real_doc"]}]}}}
    )
    llm = FakeLLMClient(responses=[plan, finish])
    result = OpenDRAgent(llm, OpenDRConfig(max_steps=6)).run(view)
    assert result.report.evidence[0].source_doc_ids == ()


@requires_sample
def test_opendr_forces_finish_when_step_budget_exhausted(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    plan = json.dumps({"sub_questions": [], "initial_queries": []})
    forced_finish = json.dumps({"thought": "t", "action": {"name": "finish", "args": {"report": "final", "evidence": []}}})

    def responder(system: str, messages: list[dict[str, str]]) -> str:
        content = messages[0]["content"]
        if "Plan your research" in content:
            return plan
        if "Step budget exhausted" in content:
            return forced_finish
        return json.dumps({"thought": "t", "action": {"name": "search", "args": {"query": f"query-{len(content)}"}}})

    llm = FakeLLMClient(responses=responder)
    result = OpenDRAgent(llm, OpenDRConfig(max_steps=2)).run(view)

    assert result.stop_reason == "step_budget_exhausted_forced_finish"
    assert result.report.report_markdown == "final"
    assert result.llm_call_count == 4  # plan + 2 react steps + 1 forced finish


@requires_sample
def test_opendr_degrades_on_two_consecutive_parse_failures(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    plan = json.dumps({"sub_questions": [], "initial_queries": []})

    def responder(system: str, messages: list[dict[str, str]]) -> str:
        return plan if "Plan your research" in messages[0]["content"] else "not valid json"

    llm = FakeLLMClient(responses=responder)
    result = OpenDRAgent(llm, OpenDRConfig(max_steps=6)).run(view)

    assert result.stop_reason == "parse_failure"
    assert result.report.evidence == ()
    parse_failures = [step for step in result.steps if step.kind == "parse_failure"]
    assert len(parse_failures) == 2
