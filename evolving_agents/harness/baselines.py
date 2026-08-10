"""Reference systems the evolved pipeline must beat; these double as the ablation table."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

from dr_cik.agents.common import render_task_brief
from dr_cik.evaluation import development_metrics
from dr_cik.llm import JsonExtractionError, LLMClient, parse_json_object
from dr_cik.models import EvidenceItem, Forecast
from dr_cik.retrieval import build_index

from ..agents.coding import CodingAgent
from ..agents.decision import DecisionAgent
from ..agents.retrieval import RetrievalAgent
from ..models import Bundle, RetrievalEvidenceOutput, TaskTrace, to_numeric_view
from .orchestrator import _fallback_forecast, run_task

logger = logging.getLogger(__name__)

BASELINES = ("chronos-only", "naive-rag", "coding-only", "frozen-system", "oracle-retrieval")

NAIVE_RAG_SYSTEM = (
    "You are a forecaster. Given a series summary and some retrieved documents, output the next values."
)
NAIVE_RAG_SCHEMA = 'Respond with exactly one JSON object: {"forecast": [v1, v2, ...]} with exactly N numbers.'

# LAFP (arXiv:2607.24892) is compared against published numbers; reimplementing it is out of scope
# for this pass, so it is deliberately absent rather than present as a stub that looks runnable.
LAFP_STATUS = "not implemented: compare against the paper's reported numbers instead"


@dataclass(frozen=True)
class BaselineResult:
    """One baseline's forecast and proxy metrics for one task."""

    benchmark_id: str
    baseline: str
    forecast: Forecast
    metrics: dict[str, float | None]


def chronos_only(task, forecaster, num_samples: int = 25) -> BaselineResult:
    """Baseline 1: the text-blind foundation model alone; everything else must beat this."""
    forecast = forecaster.forecast(task.agent_view(), num_samples=num_samples)
    return BaselineResult(task.benchmark_id, "chronos-only", forecast, development_metrics(task, forecast, (), set(), None))


def naive_rag(task, llm: LLMClient, top_k: int = 5, judge: LLMClient | None = None) -> BaselineResult:
    """Baseline 2: BM25 top-k straight into one prompt, with numbers parsed from the reply.

    This is the one place a forecast is read out of LLM free text rather than executed code. That is
    the point: it is the strawman the real system must beat, not a candidate we would ever ship.
    """
    view = task.agent_view()
    horizon = view.prediction_length
    doc_ids: list[str] = []
    if view.documents:
        index = build_index(view.documents, retriever="bm25")
        query = f"{view.entity_name} {view.target_name} {view.target_description or ''}"
        doc_ids = list(dict.fromkeys(chunk.document_id for chunk, _score in index.search(query, top_k=top_k)))

    by_id = {document.document_id: document for document in view.documents}
    context = "\n\n".join(f"[{doc_id}] {by_id[doc_id].text[:1000]}" for doc_id in doc_ids if doc_id in by_id)
    prompt = (
        f"{render_task_brief(view)}\n\nRetrieved documents:\n{context or '(none)'}\n\n"
        f"Forecast the next {horizon} values. {NAIVE_RAG_SCHEMA.replace('N', str(horizon))}"
    )
    response = llm.complete(system=NAIVE_RAG_SYSTEM, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_output_tokens=8 * horizon + 200)

    forecast = _parse_forecast(response.text, horizon) or _fallback_forecast(to_numeric_view(view))
    evidence = tuple(EvidenceItem(claim=f"retrieved {doc_id}", source_doc_ids=(doc_id,)) for doc_id in doc_ids)
    return BaselineResult(
        task.benchmark_id, "naive-rag", forecast, development_metrics(task, forecast, evidence, set(doc_ids), judge)
    )


def _parse_forecast(text: str, horizon: int) -> Forecast | None:
    """Read a forecast array out of an LLM reply, or None if it is unusable."""
    try:
        parsed = parse_json_object(text)
    except JsonExtractionError:
        return None
    raw = parsed.get("forecast")
    if not isinstance(raw, list) or len(raw) < horizon:
        return None
    try:
        values = tuple(float(value) for value in raw[:horizon])
    except (TypeError, ValueError):
        return None
    return Forecast(mean=values, samples=(values,), method="naive-rag:llm-text")


def coding_only(task, llm: LLMClient, coding_bundle: Bundle, n_windows: int = 2) -> BaselineResult:
    """Baseline 3: the Coding Agent with no text at all, isolating what Loop A alone buys."""
    trace = run_task(task, CodingAgent(llm, coding_bundle), None, None, n_windows=n_windows)
    return BaselineResult(task.benchmark_id, "coding-only", trace.forecast, trace.metrics)


def frozen_system(task, llm: LLMClient, coding: Bundle, retrieval: Bundle, decision: Bundle, judge: LLMClient | None = None, n_windows: int = 2) -> TaskTrace:
    """Baseline 4: all three agents on their seed bundles, isolating what evolution itself buys."""
    return run_task(
        task, CodingAgent(llm, coding), RetrievalAgent(llm, retrieval), DecisionAgent(llm, decision), judge=judge, n_windows=n_windows
    )


def oracle_retrieval(task, llm: LLMClient, coding: Bundle, decision: Bundle, judge: LLMClient | None = None, n_windows: int = 2) -> TaskTrace:
    """Baseline 6: feed exactly the labeled supporting documents, bounding what retrieval could add.

    This is the only baseline permitted to read Document.role, and it is a diagnostic ceiling only --
    never a configuration whose output may influence bundle selection.
    """
    supporting = [document for document in task.documents if document.role == "supporting"]
    evidence = RetrievalEvidenceOutput(
        kept=tuple(EvidenceItem(claim=document.text[:400], source_doc_ids=(document.document_id,)) for document in supporting),
        considered_doc_ids=tuple(document.document_id for document in supporting),
    )
    return run_task(
        task, CodingAgent(llm, coding), None, DecisionAgent(llm, decision), judge=judge, n_windows=n_windows, fixed_evidence=evidence
    )


def mean_metrics(results) -> dict[str, float]:
    """Average each metric across tasks, skipping None exactly as dr_cik.pipeline does."""
    collected: dict[str, list[float]] = {}
    for result in results:
        for name, value in (result.metrics or {}).items():
            if value is not None:
                collected.setdefault(name, []).append(value)
    return {name: round(statistics.fmean(values), 4) for name, values in sorted(collected.items())}
