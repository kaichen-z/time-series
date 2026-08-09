"""DRBench reproduction: a deterministic search -> per-document brief -> synthesize cascade."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..llm import JsonExtractionError, LLMClient, parse_json_object
from ..models import AgentReport, AgentResult, AgentStep, TaskView
from ..retrieval import build_index
from .common import AGENT_SYSTEM_PREAMBLE, documents_by_id, parse_evidence_list, render_task_brief, trend_word

logger = logging.getLogger(__name__)

DEGRADED_REPORT = "Synthesis failed to produce well-formed output; no verified evidence."


@dataclass(frozen=True)
class DRBenchConfig:
    """Tunables for the DRBench search/brief/synthesize cascade."""

    top_k_search: int = 16
    retriever: str = "bm25"
    temperature: float = 0.0


def _build_query(task_view: TaskView) -> str:
    return f"{task_view.target_name} {task_view.target_description} {task_view.entity_name} {trend_word(task_view.history_values)}"


class DRBenchAgent:
    """Heuristic search, then one LLM brief per document, then one synthesis call."""

    def __init__(self, llm: LLMClient, config: DRBenchConfig | None = None) -> None:
        self.llm = llm
        self.config = config or DRBenchConfig()

    def run(self, task_view: TaskView) -> AgentResult:
        index = build_index(task_view.documents, retriever=self.config.retriever)
        by_id = documents_by_id(task_view.documents)
        valid_ids = set(by_id)
        brief_text = render_task_brief(task_view)
        steps: list[AgentStep] = []
        call_count = 0

        query = _build_query(task_view)
        results = index.search(query, top_k=self.config.top_k_search)
        retrieved_doc_ids = list(dict.fromkeys(chunk.document_id for chunk, _score in results))
        logger.info("drbench[%s]: step 0/search - query=%r (retriever=%s) -> %d document(s): %s", task_view.benchmark_id, query, self.config.retriever, len(retrieved_doc_ids), retrieved_doc_ids)
        steps.append(AgentStep(step_index=0, kind="search", payload={"query": query, "document_ids": retrieved_doc_ids}))

        briefs: list[dict[str, object]] = []
        for step_index, document_id in enumerate(retrieved_doc_ids, start=1):
            logger.info("drbench[%s]: step %d/%d - briefing document %s", task_view.benchmark_id, step_index, len(retrieved_doc_ids), document_id)
            document = by_id[document_id]
            prompt = (
                f"{brief_text}\n\n"
                f'Document {document_id}:\n"""{document.text}"""\n\n'
                'Is this document relevant to the forecasting task above? Respond with exactly one '
                'JSON object: {"document_id": "...", "relevant": true|false, '
                '"brief": "<=400 chars, empty if not relevant>", "key_claims": ["..."]}'
            )
            response = self.llm.complete(system=AGENT_SYSTEM_PREAMBLE, messages=[{"role": "user", "content": prompt}], temperature=self.config.temperature)
            call_count += 1
            try:
                parsed = parse_json_object(response.text)
            except JsonExtractionError:
                logger.warning("drbench[%s]: brief for %s failed to parse JSON, skipping", task_view.benchmark_id, document_id)
                steps.append(AgentStep(step_index=step_index, kind="brief_parse_failure", payload={"document_id": document_id, "raw": response.text}))
                continue
            # Trust the id we briefed, not the one the model echoed: a hallucinated id would be
            # shown to the synthesizer, cited, then dropped by parse_evidence_list as out-of-corpus.
            parsed["document_id"] = document_id
            steps.append(AgentStep(step_index=step_index, kind="brief", payload=parsed))
            if parsed.get("relevant"):
                logger.info("drbench[%s]: document %s marked relevant", task_view.benchmark_id, document_id)
                briefs.append(parsed)
            else:
                logger.info("drbench[%s]: document %s marked not relevant", task_view.benchmark_id, document_id)

        logger.info("drbench[%s]: step %d/synthesize - %d relevant brief(s)", task_view.benchmark_id, len(retrieved_doc_ids) + 1, len(briefs))
        synthesis_prompt = (
            f"{brief_text}\n\nRelevant document briefs:\n"
            + "\n".join(f"[{item.get('document_id')}] {item.get('brief', '')}" for item in briefs)
            + '\n\nRespond with exactly one JSON object: {"report": "<markdown>", '
            '"evidence": [{"claim": "...", "source_doc_ids": ["doc_id", ...]}]}'
        )

        response = self.llm.complete(system=AGENT_SYSTEM_PREAMBLE, messages=[{"role": "user", "content": synthesis_prompt}], temperature=self.config.temperature)
        call_count += 1
        try:
            synthesis = parse_json_object(response.text)
            evidence = parse_evidence_list(synthesis.get("evidence") or [], valid_ids)
            report = AgentReport(report_markdown=str(synthesis.get("report", "")), evidence=evidence)
            steps.append(AgentStep(step_index=len(retrieved_doc_ids) + 1, kind="synthesize", payload=synthesis))
            stop_reason = "synthesized"
        except JsonExtractionError:
            logger.warning("drbench[%s]: synthesis failed to parse JSON, degraded report", task_view.benchmark_id)
            report = AgentReport(report_markdown=DEGRADED_REPORT, evidence=())
            steps.append(AgentStep(step_index=len(retrieved_doc_ids) + 1, kind="parse_failure", payload={"raw": response.text}))
            stop_reason = "parse_failure"

        logger.info("drbench[%s]: done (%s, %d LLM call(s))", task_view.benchmark_id, stop_reason, call_count)
        return AgentResult(report=report, steps=tuple(steps), stop_reason=stop_reason, llm_call_count=call_count)
