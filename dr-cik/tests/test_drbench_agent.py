"""DRBench's deterministic search -> per-doc brief -> synthesize cascade."""

from __future__ import annotations

import json

from dr_cik.agents.drbench import DRBenchAgent, DRBenchConfig
from dr_cik.llm import FakeLLMClient

from .conftest import requires_sample


def _brief(document_id: str, relevant: bool) -> str:
    return json.dumps({"document_id": document_id, "relevant": relevant, "brief": "b" if relevant else "", "key_claims": []})


@requires_sample
def test_drbench_call_count_matches_retrieved_doc_count_plus_one(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    top_k = 3

    def responder(system: str, messages: list[dict[str, str]]) -> str:
        content = messages[0]["content"]
        if content.startswith("Document"):
            document_id = content.split("Document ")[1].split(":")[0]
            return _brief(document_id, relevant=True)
        return json.dumps({"report": "r", "evidence": [{"claim": "c1", "source_doc_ids": []}]})

    llm = FakeLLMClient(responses=responder)
    result = DRBenchAgent(llm, DRBenchConfig(top_k_search=top_k)).run(view)

    assert result.stop_reason == "synthesized"
    assert result.llm_call_count == top_k + 1


@requires_sample
def test_drbench_excludes_irrelevant_briefs_from_synthesis_prompt(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    seen_document_ids: list[str] = []

    def responder(system: str, messages: list[dict[str, str]]) -> str:
        content = messages[0]["content"]
        if content.startswith("Document"):
            document_id = content.split("Document ")[1].split(":")[0]
            seen_document_ids.append(document_id)
            relevant = len(seen_document_ids) == 1  # only the first doc is relevant
            return _brief(document_id, relevant=relevant)
        # this is the synthesis call: only the relevant doc's brief should be in the brief block
        # (the corpus id list earlier in the prompt legitimately lists every document id)
        brief_block = content.split("Relevant document briefs:")[1]
        assert f"[{seen_document_ids[0]}]" in brief_block
        for irrelevant_id in seen_document_ids[1:]:
            assert f"[{irrelevant_id}]" not in brief_block
        return json.dumps({"report": "r", "evidence": []})

    llm = FakeLLMClient(responses=responder)
    DRBenchAgent(llm, DRBenchConfig(top_k_search=3)).run(view)


@requires_sample
def test_drbench_degrades_on_malformed_synthesis(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()

    def responder(system: str, messages: list[dict[str, str]]) -> str:
        content = messages[0]["content"]
        if content.startswith("Document"):
            document_id = content.split("Document ")[1].split(":")[0]
            return _brief(document_id, relevant=False)
        return "not valid json"

    llm = FakeLLMClient(responses=responder)
    result = DRBenchAgent(llm, DRBenchConfig(top_k_search=2)).run(view)

    assert result.stop_reason == "parse_failure"
    assert result.report.evidence == ()
