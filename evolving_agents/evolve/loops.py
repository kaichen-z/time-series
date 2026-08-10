"""The three loop-specific score functions: A (coding), B (retrieval), C (whole system)."""

from __future__ import annotations

import logging

from dr_cik.llm import LLMClient
from dr_cik.models import ForecastTask

from ..agents.coding import CodingAgent, CodingAgentConfig
from ..harness.hindcast import carve_hindcast_windows
from ..models import Bundle, BundleTriple, to_numeric_view
from .evaluate import TaskResult

logger = logging.getLogger(__name__)

# A bundle whose every hypothesis failed still has to be ranked against ones that worked;
# a large finite penalty keeps it last without turning the population mean into -inf.
FAILED_PENALTY = 10.0


def loop_a_score_fn(
    bundle: Bundle,
    task: ForecastTask,
    llm: LLMClient,
    config: CodingAgentConfig | None = None,
    n_windows: int = 3,
    generation: int | None = None,
) -> TaskResult:
    """Score the Coding Agent alone by hindcasting its best candidate; needs no labels at all."""
    view = to_numeric_view(task.agent_view())
    windows = carve_hindcast_windows(view.history_values, view.prediction_length, view.frequency, n_windows=n_windows)
    result = CodingAgent(llm, bundle, config).run(view, windows, generation=generation)

    best = result.candidates[0] if result.candidates else None
    if best is None or best.hindcast_score is None:
        score = -FAILED_PENALTY
        trace = {
            "reason": "no candidate produced a hindcastable forecast",
            "attempted": len(result.all_candidates),
            "errors": [item.sandbox_result.error for item in result.all_candidates if item.sandbox_result.error][:3],
            "assumptions": [item.hypothesis.assumption_text for item in result.all_candidates][:3],
        }
    else:
        score = -best.hindcast_score
        trace = {
            "best_assumption": best.hypothesis.assumption_text,
            "best_code": best.hypothesis.code[:600],
            "hindcast_error": best.hindcast_score,
            "windows": len(windows),
            "usable_candidates": len(result.candidates),
            "attempted": len(result.all_candidates),
            "errors": [item.sandbox_result.error for item in result.all_candidates if item.sandbox_result.error][:3],
        }
    return TaskResult(task_id=task.benchmark_id, score=score, trace=trace)


def loop_b_score_fn(
    bundle: Bundle,
    task: ForecastTask,
    llm: LLMClient,
    frozen_coding: Bundle,
    frozen_decision: Bundle,
    bonus_weight: float = 0.2,
    n_windows: int = 2,
    generation: int | None = None,
) -> TaskResult:
    """Score the Retrieval Agent by label F1, plus a bonus when its evidence improves the forecast.

    Loop B evolves on the evolve split like every other loop; the labels it needs are the document
    roles, which the whole labeled pool carries -- not a licence to evolve against the dev split.
    """
    from ..agents.decision import DecisionAgent
    from ..agents.retrieval import RetrievalAgent
    from ..harness.metrics import retrieval_f1
    from ..harness.orchestrator import run_task

    agent_view = task.agent_view()
    evidence = RetrievalAgent(llm, bundle).run(agent_view, generation=generation)
    f1 = retrieval_f1(task, evidence.kept)

    bonus = 0.0
    downstream: dict[str, float | None] = {}
    if bonus_weight and evidence.kept:
        coding_agent = CodingAgent(llm, frozen_coding)
        decision_agent = DecisionAgent(llm, frozen_decision)
        with_evidence = run_task(task, coding_agent, None, decision_agent, n_windows=n_windows, fixed_evidence=evidence)
        without = run_task(
            task, CodingAgent(llm, frozen_coding), None, DecisionAgent(llm, frozen_decision),
            n_windows=n_windows, fixed_evidence=_empty_evidence(),
        )
        improvement = (without.metrics.get("smae") or 0.0) - (with_evidence.metrics.get("smae") or 0.0)
        bonus = max(0.0, improvement) * bonus_weight
        downstream = {"smae_with": with_evidence.metrics.get("smae"), "smae_without": without.metrics.get("smae")}

    score = (f1 if f1 is not None else 0.0) + bonus
    supporting = {document.document_id for document in task.documents if document.role == "supporting"}
    cited = {doc_id for item in evidence.kept for doc_id in item.source_doc_ids}
    return TaskResult(
        task_id=task.benchmark_id,
        score=score,
        trace={
            "f1": f1,
            "bonus": bonus,
            "kept_claims": [item.claim[:200] for item in evidence.kept][:5],
            "cited": sorted(cited),
            "missed_supporting": sorted(supporting - cited),
            "cited_distractors": sorted(cited - supporting),
            "considered": len(evidence.considered_doc_ids),
            **downstream,
        },
    )


def loop_c_score_fn(
    triple: BundleTriple,
    task: ForecastTask,
    llm: LLMClient,
    judge: LLMClient | None = None,
    n_windows: int = 2,
    generation: int | None = None,
) -> TaskResult:
    """Score the whole system end to end with dr_cik's proxy metrics; see PROXY_NOTE in run_log."""
    from ..agents.decision import DecisionAgent
    from ..agents.retrieval import RetrievalAgent
    from ..harness.metrics import loop_c_score
    from ..harness.orchestrator import run_task

    trace = run_task(
        task,
        CodingAgent(llm, triple.coding),
        RetrievalAgent(llm, triple.retrieval),
        DecisionAgent(llm, triple.decision),
        judge=judge,
        n_windows=n_windows,
        generation=generation,
    )
    discarded = [entry for entry in trace.decision_result.audit if not entry.kept]
    return TaskResult(
        task_id=task.benchmark_id,
        score=loop_c_score(trace.metrics),
        trace={
            "smae": trace.metrics.get("smae"),
            "srmse": trace.metrics.get("srmse"),
            "scrps": trace.metrics.get("scrps"),
            "evidence_recall": trace.metrics.get("evidence_recall"),
            "kept_evidence": [item.claim[:200] for item in trace.retrieval_result.kept][:3],
            "winning_assumption": trace.coding_result.candidates[0].hypothesis.assumption_text if trace.coding_result.candidates else None,
            "discarded": [{"candidate": entry.candidate_id, "why": entry.reason[:200]} for entry in discarded][:3],
            "revised": trace.revised,
        },
    )


def _empty_evidence():
    """An evidence output carrying nothing, for the no-context arm of Loop B's bonus."""
    from ..models import RetrievalEvidenceOutput

    return RetrievalEvidenceOutput(kept=(), considered_doc_ids=())
