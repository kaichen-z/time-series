"""Immutable resolved-label scoring host for all harness implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from evolving_loop.data import ContextTask
from common.metrics import score_forecast, spearman_rank_correlation

if TYPE_CHECKING:
    from evolving_loop.harness import HarnessResult


@dataclass(frozen=True)
class ResolvedOutcome:
    task_id: str
    final_smape: float
    final_mae: float
    coding_oracle_smape: float
    coding_coverage_regret: float
    retrieval_precision: float
    supporting_recall: float
    distractor_avoidance: float
    decision_selection_regret: float
    candidate_count: int
    hindcast_future_rank_correlation: float
    coding_oracle_mae: float = 0.0
    decision_selection_mae_regret: float = 0.0
    contextual_oracle_smape: float = 0.0
    contextual_oracle_mae: float = 0.0
    retrieval_candidate_gain_mae: float = 0.0


def score_after_resolution(task: ContextTask, result: "HarnessResult") -> ResolvedOutcome:
    """Score a label-free inference result after public labels resolve."""
    if not task.labels_public:
        raise ValueError("resolved-outcome learning is forbidden for hidden/unreleased labels")
    truth = list(task.numeric.future_values)
    final = score_forecast(truth, list(result.forecast))
    candidate_scores = {
        candidate.candidate_id: score_forecast(truth, list(candidate.forecast))
        for candidate in result.candidates
    }
    coding_scores = [
        score_forecast(truth, list(candidate.forecast))
        for candidate in result.coding.candidates
    ] or list(candidate_scores.values())
    hindcast_scores = [candidate.hindcast_smape for candidate in result.candidates]
    resolved_scores = [
        candidate_scores[candidate.candidate_id]["smape"]
        for candidate in result.candidates
    ]
    oracle_smape = min(score["smape"] for score in coding_scores)
    oracle_mae = min(score["mae"] for score in coding_scores)
    contextual_oracle_smape = min(
        score["smape"] for score in candidate_scores.values()
    )
    contextual_oracle_mae = min(score["mae"] for score in candidate_scores.values())
    selected_score = candidate_scores[result.decision.selected.candidate_id]
    retrieved_ids = set(result.retrieval.selected_document_ids)
    supporting = {item.document_id for item in task.documents if item.role == "supporting"}
    distractors = {item.document_id for item in task.documents if item.role == "distractor"}
    precision = len(retrieved_ids & supporting) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(retrieved_ids & supporting) / len(supporting) if supporting else 1.0
    avoidance = 1.0 - (len(retrieved_ids & distractors) / len(distractors)) if distractors else 1.0
    return ResolvedOutcome(
        task_id=task.numeric.task_id,
        final_smape=final["smape"],
        final_mae=final["mae"],
        coding_oracle_smape=oracle_smape,
        coding_coverage_regret=oracle_smape,
        retrieval_precision=precision,
        supporting_recall=recall,
        distractor_avoidance=avoidance,
        decision_selection_regret=(
            selected_score["smape"] - contextual_oracle_smape
        ),
        candidate_count=len(result.candidates),
        hindcast_future_rank_correlation=spearman_rank_correlation(
            hindcast_scores,
            resolved_scores,
        ),
        coding_oracle_mae=oracle_mae,
        decision_selection_mae_regret=(
            selected_score["mae"] - contextual_oracle_mae
        ),
        contextual_oracle_smape=contextual_oracle_smape,
        contextual_oracle_mae=contextual_oracle_mae,
        retrieval_candidate_gain_mae=oracle_mae - contextual_oracle_mae,
    )
