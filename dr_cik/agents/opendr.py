"""OpenDR reproduction: plan, then a ReAct search/finish loop, then a report."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..llm import JsonExtractionError, LLMClient, parse_json_object
from ..models import AgentReport, AgentResult, AgentStep, TaskView
from ..retrieval import build_index
from .common import AGENT_SYSTEM_PREAMBLE, parse_evidence_list, render_task_brief

logger = logging.getLogger(__name__)

PLAN_SCHEMA = (
    'Respond with exactly one JSON object: {"sub_questions": ["..."], "initial_queries": ["..."]}'
)
REACT_SCHEMA = (
    "Respond with exactly one JSON object, either:\n"
    '{"thought": "...", "action": {"name": "search", "args": {"query": "..."}}}\n'
    "or, once you have enough evidence:\n"
    '{"thought": "...", "action": {"name": "finish", "args": {"report": "<markdown>", '
    '"evidence": [{"claim": "...", "source_doc_ids": ["doc_id", ...]}]}}}'
)
FORCED_FINISH_INSTRUCTION = (
    "Step budget exhausted. Call finish now using only the evidence you have already found."
)
DEGRADED_REPORT = "Step budget exhausted; no verified evidence was produced."


@dataclass(frozen=True)
class OpenDRConfig:
    """Tunables for the OpenDR ReAct loop."""

    max_steps: int = 6
    max_search_results: int = 5
    retriever: str = "bm25"
    temperature: float = 0.0


class OpenDRAgent:
    """Plan -> ReAct search/finish loop -> report, matching Dr-CiK's OpenDR description."""

    def __init__(self, llm: LLMClient, config: OpenDRConfig | None = None) -> None:
        self.llm = llm
        self.config = config or OpenDRConfig()

    def run(self, task_view: TaskView) -> AgentResult:
        logger.info("opendr[%s]: step 0/plan - building index (retriever=%s)", task_view.benchmark_id, self.config.retriever)
        index = build_index(task_view.documents, retriever=self.config.retriever)
        valid_ids = {document.document_id for document in task_view.documents}
        brief = render_task_brief(task_view)
        steps: list[AgentStep] = []
        call_count = 0

        plan_prompt = f"{brief}\n\nPlan your research before searching.\n{PLAN_SCHEMA}"
        response = self.llm.complete(system=AGENT_SYSTEM_PREAMBLE, messages=[{"role": "user", "content": plan_prompt}], temperature=self.config.temperature)
        call_count += 1
        try:
            plan = parse_json_object(response.text)
        except JsonExtractionError:
            logger.warning("opendr[%s]: plan step failed to parse JSON, falling back to a bare query", task_view.benchmark_id)
            plan = {"sub_questions": [], "initial_queries": [task_view.target_name + " " + task_view.entity_name]}
        logger.info("opendr[%s]: plan - %d sub-question(s), %d initial quer(y/ies)", task_view.benchmark_id, len(plan.get("sub_questions", [])), len(plan.get("initial_queries", [])))
        steps.append(AgentStep(step_index=0, kind="plan", payload=plan))

        transcript_lines: list[str] = []
        seen_queries: set[str] = set()
        consecutive_parse_failures = 0
        finished_report: AgentReport | None = None
        stop_reason = "max_steps_reached"

        for step_index in range(1, self.config.max_steps + 1):
            logger.info("opendr[%s]: step %d/%d - asking the LLM for the next action", task_view.benchmark_id, step_index, self.config.max_steps)
            remaining = self.config.max_steps - step_index + 1
            prompt = (
                f"{brief}\n\nPlan: {plan}\n\nTranscript so far:\n"
                + ("\n".join(transcript_lines) if transcript_lines else "(nothing yet)")
                + f"\n\nYou have {remaining} step(s) left.\n{REACT_SCHEMA}"
            )
            response = self.llm.complete(system=AGENT_SYSTEM_PREAMBLE, messages=[{"role": "user", "content": prompt}], temperature=self.config.temperature)
            call_count += 1
            try:
                turn = parse_json_object(response.text)
            except JsonExtractionError:
                consecutive_parse_failures += 1
                logger.warning("opendr[%s]: step %d - failed to parse JSON action (%d consecutive)", task_view.benchmark_id, step_index, consecutive_parse_failures)
                steps.append(AgentStep(step_index=step_index, kind="parse_failure", payload={"raw": response.text}))
                if consecutive_parse_failures >= 2:
                    stop_reason = "parse_failure"
                    break
                continue
            consecutive_parse_failures = 0

            action = turn.get("action") or {}
            name = action.get("name")
            args = action.get("args") or {}

            if name == "finish":
                evidence = parse_evidence_list(args.get("evidence") or [], valid_ids)
                logger.info("opendr[%s]: step %d - finish (%d evidence item(s))", task_view.benchmark_id, step_index, len(evidence))
                finished_report = AgentReport(report_markdown=str(args.get("report", "")), evidence=evidence)
                steps.append(AgentStep(step_index=step_index, kind="finish", payload=turn))
                stop_reason = "finished"
                break

            if name == "search":
                query = str(args.get("query", "")).strip()
                normalized = query.lower()
                document_ids: list[str] = []
                if not query or normalized in seen_queries:
                    logger.info("opendr[%s]: step %d - search skipped (empty or repeated query %r)", task_view.benchmark_id, step_index, query)
                    observation = "That query is empty or already searched. Try a different query or call finish."
                else:
                    seen_queries.add(normalized)
                    results = index.search(query, top_k=self.config.max_search_results)
                    document_ids = list(dict.fromkeys(chunk.document_id for chunk, _score in results))
                    logger.info("opendr[%s]: step %d - search(%r) -> %d document(s): %s", task_view.benchmark_id, step_index, query, len(document_ids), document_ids)
                    observation = "\n".join(
                        f"[{chunk.document_id} | {chunk.chunk_id} | score={score:.3f}] {chunk.text[:280]}"
                        for chunk, score in results
                    ) or "No matching documents."
                steps.append(AgentStep(step_index=step_index, kind="search", payload={"query": query, "document_ids": document_ids, "observation": observation}))
                transcript_lines.append(f"Thought: {turn.get('thought', '')}")
                transcript_lines.append(f"Action: search({query!r})")
                transcript_lines.append(f"Observation: {observation}")
                continue

            logger.warning("opendr[%s]: step %d - unknown action %r", task_view.benchmark_id, step_index, name)
            steps.append(AgentStep(step_index=step_index, kind="unknown_action", payload=turn))
            transcript_lines.append("Observation: Unknown action. Only 'search' and 'finish' are available.")

        if finished_report is None and stop_reason == "parse_failure":
            logger.warning("opendr[%s]: giving up after repeated parse failures, degraded report", task_view.benchmark_id)
            finished_report = AgentReport(report_markdown=DEGRADED_REPORT, evidence=())

        if finished_report is None:
            logger.info("opendr[%s]: step budget exhausted, forcing a finish call", task_view.benchmark_id)
            response = self.llm.complete(
                system=AGENT_SYSTEM_PREAMBLE,
                messages=[{"role": "user", "content": f"{brief}\n\n{FORCED_FINISH_INSTRUCTION}\n{REACT_SCHEMA}"}],
                temperature=self.config.temperature,
            )
            call_count += 1
            try:
                turn = parse_json_object(response.text)
                args = (turn.get("action") or {}).get("args") or {}
                evidence = parse_evidence_list(args.get("evidence") or [], valid_ids)
                finished_report = AgentReport(report_markdown=str(args.get("report", "")), evidence=evidence)
                steps.append(AgentStep(step_index=self.config.max_steps + 1, kind="forced_finish", payload=turn))
                stop_reason = "step_budget_exhausted_forced_finish"
            except JsonExtractionError:
                logger.warning("opendr[%s]: forced finish also failed to parse, degraded report", task_view.benchmark_id)
                finished_report = AgentReport(report_markdown=DEGRADED_REPORT, evidence=())
                steps.append(AgentStep(step_index=self.config.max_steps + 1, kind="forced_finish_failed", payload={"raw": response.text}))
                stop_reason = "step_budget_exhausted_forced_finish_failed"

        logger.info("opendr[%s]: done (%s, %d LLM call(s))", task_view.benchmark_id, stop_reason, call_count)
        return AgentResult(report=finished_report, steps=tuple(steps), stop_reason=stop_reason, llm_call_count=call_count)
