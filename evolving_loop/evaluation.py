"""Immutable resolved-label scoring host for all harness implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from common.metrics import drcik_point_metrics
from evolving_loop.data import ContextTask

if TYPE_CHECKING:
    from evolving_loop.harness import HarnessResult


@dataclass(frozen=True)
class ResolvedOutcome:
    task_id: str
    final_smae: float
    final_srmse: float
    coding_oracle_smae: float
    coding_oracle_srmse: float
    contextual_oracle_smae: float
    contextual_oracle_srmse: float
    decision_selection_smae_regret: float
    decision_selection_srmse_regret: float
    candidate_count: int


def score_after_resolution(
    task: ContextTask, result: "HarnessResult"
) -> ResolvedOutcome:
    """Score label-free inference with only official Dr-CiK point metrics."""
    if not task.labels_public:
        raise ValueError("resolved-outcome learning is forbidden for hidden/unreleased labels")
    truth = task.numeric.future_values
    final = drcik_point_metrics(truth, [result.forecast])
    candidates = {
        candidate.candidate_id: drcik_point_metrics(truth, [candidate.forecast])
        for candidate in result.candidates
    }
    coding = {
        candidate.program.name: drcik_point_metrics(truth, [candidate.forecast])
        for candidate in result.coding.candidates
    } or candidates
    selected = candidates[result.decision.selected.candidate_id]
    contextual_id = min(
        candidates,
        key=lambda candidate_id: (
            candidates[candidate_id]["srmse"],
            candidates[candidate_id]["smae"],
            candidate_id,
        ),
    )
    coding_id = min(
        coding,
        key=lambda candidate_id: (
            coding[candidate_id]["srmse"],
            coding[candidate_id]["smae"],
            candidate_id,
        ),
    )
    contextual = candidates[contextual_id]
    coding_oracle = coding[coding_id]
    return ResolvedOutcome(
        task_id=task.numeric.task_id,
        final_smae=final["smae"],
        final_srmse=final["srmse"],
        coding_oracle_smae=coding_oracle["smae"],
        coding_oracle_srmse=coding_oracle["srmse"],
        contextual_oracle_smae=contextual["smae"],
        contextual_oracle_srmse=contextual["srmse"],
        decision_selection_smae_regret=selected["smae"] - contextual["smae"],
        decision_selection_srmse_regret=selected["srmse"] - contextual["srmse"],
        candidate_count=len(result.candidates),
    )
