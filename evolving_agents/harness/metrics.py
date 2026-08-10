"""Retrieval scoring and the composite end-to-end score, adapting dr_cik's proxy metrics."""

from __future__ import annotations

from dataclasses import dataclass

from dr_cik.evaluation import cited_document_ids
from dr_cik.models import EvidenceItem, ForecastTask


@dataclass(frozen=True)
class RetrievalScores:
    """Precision/recall/F1 of kept documents against the supporting/distractor labels."""

    precision: float | None
    recall: float | None
    f1: float | None
    kept: int
    supporting_total: int


def supporting_ids(task: ForecastTask) -> set[str]:
    """Return the ids of this task's supporting documents."""
    return {document.document_id for document in task.documents if document.role == "supporting"}


def retrieval_scores(task: ForecastTask, kept: tuple[EvidenceItem, ...]) -> RetrievalScores:
    """Score kept evidence against the role labels; degenerate cases return None, never 0.0 or 1.0.

    Following dr_cik.evaluation's discipline: a task with no supporting documents, or an agent that
    kept nothing, is unscoreable rather than perfect or worthless, and is skipped when averaging.
    """
    supporting = supporting_ids(task)
    cited = cited_document_ids(kept)
    if not supporting:
        return RetrievalScores(precision=None, recall=None, f1=None, kept=len(cited), supporting_total=0)

    hits = len(cited & supporting)
    recall = hits / len(supporting)
    precision = hits / len(cited) if cited else None
    if precision is None or precision + recall == 0:
        f1 = 0.0 if cited else None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return RetrievalScores(precision=precision, recall=recall, f1=f1, kept=len(cited), supporting_total=len(supporting))


def retrieval_f1(task: ForecastTask, kept: tuple[EvidenceItem, ...]) -> float | None:
    """Return just the F1, or None when the task or the result is unscoreable."""
    return retrieval_scores(task, kept).f1


def loop_c_score(metrics: dict[str, float | None], weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.5)) -> float:
    """Combine dr_cik's proxy forecast metrics and evidence recall into one higher-is-better score.

    These are local development proxies from dr_cik.evaluation, not Dr-CiK's private official scorer.
    """
    w_mae, w_rmse, w_crps, w_evidence = weights
    error = w_mae * (metrics.get("smae") or 0.0) + w_rmse * (metrics.get("srmse") or 0.0) + w_crps * (metrics.get("scrps") or 0.0)
    return -error + w_evidence * (metrics.get("evidence_recall") or 0.0)
