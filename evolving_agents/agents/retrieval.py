"""The Retrieval Agent: a cheap lexical filter, then an LLM that keeps only concrete dated evidence."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from dr_cik.agents.common import parse_evidence_list
from dr_cik.llm import JsonExtractionError, LLMClient, parse_json_object
from dr_cik.retrieval import build_index

from ..harness.trace import TraceEvent, emit, emit_llm_response
from ..models import AgentStep, Bundle, RetrievalEvidenceOutput
from .common import extract_reasoning, render_fewshot_block, render_numeric_brief, render_system_prompt

logger = logging.getLogger(__name__)

EVIDENCE_SCHEMA = (
    'Respond with exactly one JSON object: {"evidence": [{"claim": "<the specific, dated, quantitative effect>", '
    '"source_doc_ids": ["<id of the document it came from>"], "direction": "up" | "down" | "shape", '
    '"timing": "<when it applies>"}]}\n'
    'Return {"evidence": []} if no document meets the bar. Cite only ids listed below; never invent one.'
)
_DOC_CHAR_LIMIT = 1200


@dataclass(frozen=True)
class RetrievalAgentConfig:
    """First-stage breadth and generation settings for the Retrieval Agent."""

    first_stage_top_n: int = 12
    retriever: str = "bm25"
    temperature: float = 0.0
    max_output_tokens: int = 1200


class RetrievalAgent:
    """Filters a document pool cheaply, then keeps only what an LLM can tie to a concrete effect."""

    def __init__(self, llm: LLMClient, bundle: Bundle, config: RetrievalAgentConfig | None = None) -> None:
        self.llm = llm
        self.bundle = bundle
        base = config or RetrievalAgentConfig()
        settings = bundle.hyperparameters
        self.config = replace(
            base,
            first_stage_top_n=int(settings.get("first_stage_top_n", base.first_stage_top_n)),
            retriever=str(settings.get("retriever", base.retriever)),
            temperature=float(settings.get("temperature", base.temperature)),
        )

    def _first_stage(self, task_view, query: str) -> list[str]:
        """Rank the pool lexically and return the ids worth showing the model."""
        if not task_view.documents:
            return []
        index = build_index(task_view.documents, retriever=self.config.retriever)
        results = index.search(query, top_k=self.config.first_stage_top_n)
        return list(dict.fromkeys(chunk.document_id for chunk, _score in results))

    def _build_prompt(self, task_view, summary: str, doc_ids: list[str]) -> str:
        """Assemble the user message: series summary, then the candidate documents' text."""
        by_id = {document.document_id: document for document in task_view.documents}
        rendered = "\n\n".join(f"[{doc_id}]\n{by_id[doc_id].text[:_DOC_CHAR_LIMIT]}" for doc_id in doc_ids if doc_id in by_id)
        parts = [
            f"Series under forecast: {task_view.target_description or task_view.target_name} for {task_view.entity_name}.",
            summary,
            f"Candidate documents ({len(doc_ids)}):\n{rendered or '(none retrieved)'}",
        ]
        fewshots = render_fewshot_block(self.bundle)
        if fewshots:
            parts.append(fewshots)
        parts.append(EVIDENCE_SCHEMA)
        return "\n\n".join(parts)

    def run(self, task_view, numeric_summary: str | None = None, generation: int | None = None) -> RetrievalEvidenceOutput:
        """Return the evidence worth keeping; an empty result is a valid, expected outcome."""
        task_id = task_view.benchmark_id
        emit(TraceEvent(task_id=task_id, agent="retrieval", event_type="agent_start", generation=generation, detail={}))
        summary = numeric_summary or render_numeric_brief(task_view)
        query = " ".join([task_view.entity_name, task_view.target_name, task_view.target_description or ""]).strip()

        doc_ids = self._first_stage(task_view, query)
        emit(
            TraceEvent(
                task_id=task_id,
                agent="retrieval.filter",
                event_type="tool_result",
                generation=generation,
                detail={"retriever": self.config.retriever, "considered": len(doc_ids), "pool": len(task_view.documents)},
            )
        )
        steps: list[AgentStep] = [
            AgentStep(step_index=0, kind="first_stage", payload={"query": query, "document_ids": doc_ids})
        ]
        if not doc_ids:
            emit(TraceEvent(task_id=task_id, agent="retrieval", event_type="agent_end", generation=generation, detail={"kept": 0}))
            return RetrievalEvidenceOutput(kept=(), considered_doc_ids=(), steps=tuple(steps), llm_call_count=0)

        response = self.llm.complete(
            system=render_system_prompt(self.bundle),
            messages=[{"role": "user", "content": self._build_prompt(task_view, summary, doc_ids)}],
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        reasoning, answer = extract_reasoning(response.text)
        emit_llm_response(task_id, "retrieval", answer, reasoning, model_id=getattr(self.llm, "model_id", "?"), generation=generation)

        try:
            parsed = parse_json_object(answer)
        except JsonExtractionError:
            logger.warning("retrieval[%s]: response was not valid JSON, keeping nothing", task_id)
            steps.append(AgentStep(step_index=1, kind="parse_failure", payload={"raw": answer[:500]}))
            return RetrievalEvidenceOutput(
                kept=(), considered_doc_ids=tuple(doc_ids), steps=tuple(steps), llm_call_count=1
            )

        # Only ids actually shown to the model are citable, so a hallucinated id is dropped here.
        kept = parse_evidence_list(parsed.get("evidence") or [], set(doc_ids))
        steps.append(AgentStep(step_index=1, kind="evidence", payload={"kept": len(kept)}))
        logger.info("retrieval[%s]: considered %d document(s), kept %d claim(s)", task_id, len(doc_ids), len(kept))
        emit(TraceEvent(task_id=task_id, agent="retrieval", event_type="agent_end", generation=generation, detail={"kept": len(kept)}))
        return RetrievalEvidenceOutput(
            kept=kept, considered_doc_ids=tuple(doc_ids), steps=tuple(steps), llm_call_count=1
        )
