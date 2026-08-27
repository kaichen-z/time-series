"""Immutable resolved-label scoring host for all harness implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from common.metrics import (
    score_forecast,
    spearman_rank_correlation,
)
from evolving_loop.data import ContextTask
from evolving_loop.retrieval_agent.credit import (
    RetrievalTaskDiagnostics,
    _score_forecast_drcik,
    _selected_candidate_matches_pool,
    assign_chain_credit,
)

if TYPE_CHECKING:
    from evolving_loop.harness import HarnessResult


@dataclass(frozen=True)
class ResolvedOutcome:
    task_id: str
    final_smae: float | None = None
    final_srmse: float | None = None
    coding_oracle_smae: float | None = None
    coding_oracle_srmse: float | None = None
    contextual_oracle_smae: float | None = None
    contextual_oracle_srmse: float | None = None
    decision_selection_smae_regret: float | None = None
    decision_selection_srmse_regret: float | None = None
    candidate_count: int = 0
    retrieval_diagnostics: RetrievalTaskDiagnostics | None = None
    # Legacy diagnostics remain readable so old checkpoints and downstream
    # reports survive the metric migration.  They never drive the new Pareto
    # acceptance rule.
    final_smape: float = 0.0
    final_mae: float = 0.0
    coding_oracle_smape: float = 0.0
    coding_coverage_regret: float = 0.0
    retrieval_precision: float = 0.0
    supporting_recall: float = 0.0
    distractor_avoidance: float = 0.0
    decision_selection_regret: float = 0.0
    hindcast_future_rank_correlation: float = 0.0
    coding_oracle_mae: float = 0.0
    decision_selection_mae_regret: float = 0.0
    contextual_oracle_smape: float = 0.0
    contextual_oracle_mae: float = 0.0
    retrieval_candidate_gain_mae: float = 0.0


def score_after_resolution(
    task: ContextTask, result: "HarnessResult"
) -> ResolvedOutcome:
    """Score label-free inference with only official Dr-CiK point metrics."""
    if not task.labels_public:
        raise ValueError("resolved-outcome learning is forbidden for hidden/unreleased labels")
    truth = task.numeric.future_values
    retrieval_credit = assign_chain_credit(task, result)
    candidates = {
        candidate.candidate_id: _score_forecast_drcik(truth, candidate.forecast)[0]
        for candidate in result.candidates
    }
    scale = sum(abs(float(value)) for value in truth) / len(truth) if truth else 1.0

    def legacy_score(forecast) -> dict[str, float]:
        try:
            return score_forecast(list(truth), list(forecast))
        except (TypeError, ValueError, OverflowError):
            return {"smape": 200.0, "mae": 5.0 * max(scale, 1e-12), "primary": 200.0}

    selection_valid = _selected_candidate_matches_pool(result)
    invalid_legacy = {
        "smape": 200.0,
        "mae": 5.0 * max(scale, 1e-12),
        "primary": 200.0,
    }
    legacy_final = legacy_score(result.forecast) if selection_valid else invalid_legacy
    legacy_candidates = {
        candidate.candidate_id: legacy_score(candidate.forecast)
        for candidate in result.candidates
    }
    legacy_coding = {
        candidate.program.name: legacy_score(candidate.forecast)
        for candidate in result.coding.candidates
    } or legacy_candidates
    legacy_selected = (
        legacy_candidates[result.decision.selected.candidate_id]
        if selection_valid
        else invalid_legacy
    )
    legacy_contextual_smape = min(row["smape"] for row in legacy_candidates.values())
    legacy_contextual_mae = min(row["mae"] for row in legacy_candidates.values())
    legacy_coding_smape = min(row["smape"] for row in legacy_coding.values())
    legacy_coding_mae = min(row["mae"] for row in legacy_coding.values())
    retrieved_ids = set(result.retrieval.selected_document_ids)
    supporting = {item.document_id for item in task.documents if item.role == "supporting"}
    distractors = {item.document_id for item in task.documents if item.role == "distractor"}
    precision = len(retrieved_ids & supporting) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(retrieved_ids & supporting) / len(supporting) if supporting else 1.0
    avoidance = (
        1.0 - len(retrieved_ids & distractors) / len(distractors)
        if distractors
        else 1.0
    )
    hindcast = [float(candidate.hindcast_srmse) for candidate in result.candidates]
    resolved = [
        candidates[candidate.candidate_id]["srmse"] for candidate in result.candidates
    ]
    return ResolvedOutcome(
        task_id=task.numeric.task_id,
        final_smae=retrieval_credit.final_smae,
        final_srmse=retrieval_credit.final_srmse,
        coding_oracle_smae=retrieval_credit.coding_oracle_smae,
        coding_oracle_srmse=retrieval_credit.coding_oracle_srmse,
        contextual_oracle_smae=retrieval_credit.contextual_oracle_smae,
        contextual_oracle_srmse=retrieval_credit.contextual_oracle_srmse,
        decision_selection_smae_regret=retrieval_credit.decision_smae_regret,
        decision_selection_srmse_regret=retrieval_credit.decision_srmse_regret,
        candidate_count=len(result.candidates),
        retrieval_diagnostics=retrieval_credit.diagnostics,
        final_smape=legacy_final["smape"],
        final_mae=legacy_final["mae"],
        coding_oracle_smape=legacy_coding_smape,
        coding_coverage_regret=legacy_coding_smape,
        retrieval_precision=precision,
        supporting_recall=recall,
        distractor_avoidance=avoidance,
        decision_selection_regret=legacy_selected["smape"] - legacy_contextual_smape,
        hindcast_future_rank_correlation=spearman_rank_correlation(hindcast, resolved),
        coding_oracle_mae=legacy_coding_mae,
        decision_selection_mae_regret=legacy_selected["mae"] - legacy_contextual_mae,
        contextual_oracle_smape=legacy_contextual_smape,
        contextual_oracle_mae=legacy_contextual_mae,
        retrieval_candidate_gain_mae=legacy_coding_mae - legacy_contextual_mae,
    )
