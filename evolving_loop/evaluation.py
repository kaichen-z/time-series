"""Immutable resolved-label scoring host for all harness implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from evolving_loop.data import ContextTask
from common.metrics import scaled_mae, scaled_rmse, spearman_rank_correlation

if TYPE_CHECKING:
    from evolving_loop.harness import HarnessResult


@dataclass(frozen=True)
class ResolvedOutcome:
    task_id: str
    final_smae: float
    final_srmse: float
    coding_oracle_smae: float
    coding_coverage_regret: float
    retrieval_precision: float
    supporting_recall: float
    distractor_avoidance: float
    decision_selection_regret: float
    candidate_count: int
    hindcast_future_rank_correlation: float


def score_after_resolution(task: ContextTask, result: "HarnessResult") -> ResolvedOutcome:
    """Score a label-free inference result after public labels resolve."""
    if not task.labels_public:
        raise ValueError("resolved-outcome learning is forbidden for hidden/unreleased labels")
    truth = list(task.numeric.future_values)
    candidate_scores = {
        candidate.candidate_id: scaled_mae(truth, list(candidate.forecast))
        for candidate in result.candidates
    }
    hindcast_scores = [candidate.hindcast_smae for candidate in result.candidates]
    resolved_scores = list(candidate_scores.values())
    oracle = min(candidate_scores.values())
    selected_score = candidate_scores[result.decision.selected.candidate_id]
    retrieved_ids = set(result.retrieval.selected_document_ids)
    supporting = {item.document_id for item in task.documents if item.role == "supporting"}
    distractors = {item.document_id for item in task.documents if item.role == "distractor"}
    precision = len(retrieved_ids & supporting) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(retrieved_ids & supporting) / len(supporting) if supporting else 1.0
    avoidance = 1.0 - (len(retrieved_ids & distractors) / len(distractors)) if distractors else 1.0
    return ResolvedOutcome(
        task_id=task.numeric.task_id,
        final_smae=scaled_mae(truth, list(result.forecast)),
        final_srmse=scaled_rmse(truth, list(result.forecast)),
        coding_oracle_smae=oracle,
        coding_coverage_regret=oracle,
        retrieval_precision=precision,
        supporting_recall=recall,
        distractor_avoidance=avoidance,
        decision_selection_regret=selected_score - oracle,
        candidate_count=len(result.candidates),
        hindcast_future_rank_correlation=spearman_rank_correlation(
            hindcast_scores,
            resolved_scores,
        ),
    )
