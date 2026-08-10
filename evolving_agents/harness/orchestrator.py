"""Wires Coding -> Retrieval -> Decision for one task, enforcing the one-revision cap."""

from __future__ import annotations

import logging

from dr_cik.evaluation import development_metrics
from dr_cik.llm import LLMClient
from dr_cik.models import Forecast

from ..agents.coding import CodingAgent
from ..agents.decision import DecisionAgent
from ..agents.retrieval import RetrievalAgent
from ..models import (
    CodingAgentResult,
    DecisionOutput,
    RetrievalEvidenceOutput,
    TaskTrace,
    to_numeric_view,
)
from .hindcast import carve_hindcast_windows
from .trace import TraceEvent, emit

logger = logging.getLogger(__name__)


def _fallback_forecast(view) -> Forecast:
    """Last-value persistence, used only when no candidate ran at all."""
    last = view.history_values[-1] if view.history_values else 0.0
    mean = tuple(float(last) for _ in range(view.prediction_length))
    return Forecast(mean=mean, samples=(mean,), method="fallback:last-value-persistence")


def run_task(
    task,
    coding_agent: CodingAgent,
    retrieval_agent: RetrievalAgent | None,
    decision_agent: DecisionAgent | None,
    judge: LLMClient | None = None,
    n_windows: int = 3,
    backbone: tuple[float, ...] | None = None,
    generation: int | None = None,
    fixed_evidence: RetrievalEvidenceOutput | None = None,
) -> TaskTrace:
    """Run the three agents over one task and score the result with dr_cik's proxy metrics.

    The one-revision cap lives here, not in the Decision Agent, which has no memory across calls:
    a second revision request is logged and ignored rather than obeyed.
    """
    task_id = task.benchmark_id
    agent_view = task.agent_view()
    numeric_view = to_numeric_view(agent_view)
    windows = carve_hindcast_windows(numeric_view.history_values, numeric_view.prediction_length, numeric_view.frequency, n_windows=n_windows)

    coding_result = coding_agent.run(numeric_view, windows, backbone=backbone, generation=generation)

    if fixed_evidence is not None:
        retrieval_result = fixed_evidence
    elif retrieval_agent is not None:
        retrieval_result = retrieval_agent.run(agent_view, generation=generation)
    else:
        retrieval_result = RetrievalEvidenceOutput(kept=(), considered_doc_ids=())

    revised = False
    if decision_agent is not None and coding_result.candidates:
        decision_result = decision_agent.decide(coding_result.candidates, retrieval_result.kept, task_id=task_id, generation=generation)
        if decision_result.revision_request:
            logger.info("task %s: honoring the single allowed revision request", task_id)
            revised = True
            revised_coding = coding_agent.run(
                numeric_view, windows, backbone=backbone, revision_request=decision_result.revision_request, generation=generation
            )
            merged = _merge(coding_result, revised_coding)
            if merged.candidates:
                second = decision_agent.decide(
                    merged.candidates, retrieval_result.kept, task_id=task_id, generation=generation, allow_revision=False
                )
                if second.revision_request:
                    logger.warning("task %s: ignoring a second revision request; one round trip is the cap", task_id)
                coding_result, decision_result = merged, second
    else:
        decision_result = None

    forecast = decision_result.final_forecast if decision_result is not None else _best_forecast(coding_result, numeric_view)
    metrics = development_metrics(task, forecast, retrieval_result.kept, set(retrieval_result.considered_doc_ids), judge)
    emit(TraceEvent(task_id=task_id, agent="task", event_type="agent_end", generation=generation, detail={"revised": revised, "smae": metrics.get("smae")}))

    return TaskTrace(
        benchmark_id=task_id,
        coding_result=coding_result,
        retrieval_result=retrieval_result,
        decision_result=decision_result or DecisionOutput(final_forecast=forecast, weights={}),
        forecast=forecast,
        metrics=metrics,
        revised=revised,
    )


def _merge(first: CodingAgentResult, second: CodingAgentResult) -> CodingAgentResult:
    """Combine an original and a revised generation, re-ranking their candidates together."""
    combined = [candidate for candidate in (*first.candidates, *second.candidates) if candidate.forecast is not None]
    combined.sort(key=lambda candidate: candidate.hindcast_score if candidate.hindcast_score is not None else float("inf"))
    return CodingAgentResult(
        candidates=tuple(combined),
        all_candidates=(*first.all_candidates, *second.all_candidates),
        steps=(*first.steps, *second.steps),
        llm_call_count=first.llm_call_count + second.llm_call_count,
    )


def _best_forecast(coding_result: CodingAgentResult, view) -> Forecast:
    """Return the top candidate's forecast, or a persistence fallback if none ran."""
    if coding_result.candidates and coding_result.candidates[0].forecast is not None:
        return coding_result.candidates[0].forecast
    return _fallback_forecast(view)
