"""End-to-end Coding -> Retrieval -> Decision forecasting harness."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from evolving_agent.coding_agent.evolution import CodingEvolutionAgent, CodingEvolutionResult
from evolving_agent.data import ContextTask
from evolving_agent.decision_agent.agent import DecisionAgent, DecisionCandidate, DecisionResult
from evolving_agent.metrics import score_forecast
from evolving_agent.retrieval_agent.agent import RetrievalAgent, RetrievalResult


@dataclass(frozen=True)
class HarnessResult:
    task_id: str
    coding: CodingEvolutionResult
    retrieval: RetrievalResult
    decision: DecisionResult
    candidates: tuple[DecisionCandidate, ...]
    forecast: tuple[float, ...]


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


class OutcomeLearner(Protocol):
    def learn(
        self, task: ContextTask, result: HarnessResult, outcome: ResolvedOutcome
    ) -> Any: ...


class EvolvingForecastHarness:
    def __init__(
        self,
        coding: CodingEvolutionAgent,
        retrieval: RetrievalAgent,
        decision: DecisionAgent,
        outcome_learner: OutcomeLearner | None = None,
    ) -> None:
        self.coding = coding
        self.retrieval = retrieval
        self.decision = decision
        self.outcome_learner = outcome_learner

    def run(self, task: ContextTask) -> HarnessResult:
        # Enforce the information boundary structurally, not only by prompt:
        # the object passed to Coding has no realized future labels.
        coding_input = replace(task.numeric, future_values=())
        coding = self.coding.run_task(coding_input)
        retrieval = self.retrieval.run(task, coding.candidates)
        candidates = self._decision_candidates(task, coding, retrieval)
        decision = self.decision.run(candidates, retrieval)
        return HarnessResult(
            task_id=task.numeric.task_id,
            coding=coding,
            retrieval=retrieval,
            decision=decision,
            candidates=candidates,
            forecast=decision.selected.forecast,
        )

    @staticmethod
    def _decision_candidates(
        task: ContextTask,
        coding: CodingEvolutionResult,
        retrieval: RetrievalResult,
    ) -> tuple[DecisionCandidate, ...]:
        candidates = [
            DecisionCandidate(
                candidate_id=item.program.name,
                forecast=item.forecast,
                assumption=item.program.assumption,
                failure_condition=item.program.failure_condition,
                hindcast_smape=item.hindcast_smape,
                tags=(item.program.source,),
            )
            for item in coding.candidates
        ]
        best = min(candidates, key=lambda item: item.hindcast_smape)
        for index, impact in enumerate(retrieval.impacts):
            if impact.temporal_relation != "overlaps_future":
                continue
            if impact.adjustment_kind not in {"multiply", "add"} or impact.adjustment_value is None:
                continue
            start, end = _future_window(task, impact.start_timestamp, impact.end_timestamp)
            if start is None or end is None:
                continue
            values = list(best.forecast)
            for step in range(start, end + 1):
                if impact.adjustment_kind == "multiply":
                    values[step] *= 1.0 + impact.adjustment_value
                else:
                    values[step] += impact.adjustment_value
            candidates.append(
                DecisionCandidate(
                    candidate_id=f"{best.candidate_id}__evidence_{index}",
                    forecast=tuple(values),
                    assumption=f"{best.assumption} plus verified future impact: {impact.rationale}",
                    failure_condition="The cited magnitude or event window does not apply to the target.",
                    hindcast_smape=best.hindcast_smape,
                    source_document_ids=impact.source_document_ids,
                    tags=("evidence_adjusted", impact.mechanism_layer),
                )
            )
        return tuple(candidates)

    @staticmethod
    def score_after_resolution(task: ContextTask, result: HarnessResult) -> ResolvedOutcome:
        """Use delayed labels only after inference; create separate module rewards."""
        if not task.labels_public:
            raise ValueError("resolved-outcome learning is forbidden for hidden/unreleased labels")
        truth = list(task.numeric.future_values)
        final = score_forecast(truth, list(result.forecast))
        candidate_scores = {
            candidate.candidate_id: score_forecast(truth, list(candidate.forecast))["smape"]
            for candidate in result.candidates
        }
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
            final_smape=final["smape"],
            final_mae=final["mae"],
            coding_oracle_smape=oracle,
            coding_coverage_regret=oracle,
            retrieval_precision=precision,
            supporting_recall=recall,
            distractor_avoidance=avoidance,
            decision_selection_regret=selected_score - oracle,
        )

    def record_outcome(
        self, task: ContextTask, result: HarnessResult
    ) -> tuple[ResolvedOutcome, Any | None]:
        """Score a resolved public task, then optionally generate validated reusable skills."""
        outcome = self.score_after_resolution(task, result)
        learning = (
            self.outcome_learner.learn(task, result, outcome)
            if self.outcome_learner is not None
            else None
        )
        return outcome, learning


def _future_window(
    task: ContextTask, start_timestamp: str | None, end_timestamp: str | None
) -> tuple[int | None, int | None]:
    if not start_timestamp or not end_timestamp or not task.future_timestamps:
        return None, None
    indexes = [
        index
        for index, timestamp in enumerate(task.future_timestamps)
        if start_timestamp <= timestamp <= end_timestamp
    ]
    return (min(indexes), max(indexes)) if indexes else (None, None)
