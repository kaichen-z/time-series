"""End-to-end Coding -> Retrieval -> Decision forecasting harness."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Protocol

from evolving_loop.coding_agent.evolution import CodingEvolutionAgent, CodingEvolutionResult
from evolving_loop.data import ContextTask
from evolving_loop.decision_agent.agent import DecisionAgent, DecisionCandidate, DecisionResult
from evolving_loop.evaluation import ResolvedOutcome, score_after_resolution
from evolving_loop.retrieval_agent.agent import RetrievalAgent, RetrievalResult


@dataclass(frozen=True)
class HarnessResult:
    task_id: str
    coding: CodingEvolutionResult
    retrieval: RetrievalResult
    decision: DecisionResult
    candidates: tuple[DecisionCandidate, ...]
    forecast: tuple[float, ...]


@dataclass(frozen=True)
class HarnessRuntimeConfig:
    """Executable topology controls exposed to the outer Meta-Harness."""

    workflow: tuple[str, ...] = ("retrieve", "decide")
    enable_evidence_adjustments: bool = True
    max_evidence_adjustments: int = 3
    decision_aggregation: str = "last"


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
        runtime: HarnessRuntimeConfig | None = None,
    ) -> None:
        self.coding = coding
        self.retrieval = retrieval
        self.decision = decision
        self.outcome_learner = outcome_learner
        self.runtime = runtime or HarnessRuntimeConfig()

    def run(
        self, task: ContextTask, *, allow_skill_writes: bool = True
    ) -> HarnessResult:
        # Keep this local boundary for normal ``run`` calls as well as evaluator calls.
        coding = self.coding.run_task(
            task.numeric_view(), allow_skill_writes=allow_skill_writes
        )
        retrieval_runs = []
        retrieval = _empty_retrieval()
        candidates = self._decision_candidates(task, coding, retrieval)
        decisions = []
        for stage in self.runtime.workflow:
            if stage == "retrieve":
                current = self.retrieval.run(
                    task,
                    coding.candidates,
                    prior=retrieval if retrieval.evidence else None,
                    round_index=len(retrieval_runs),
                )
                retrieval_runs.append(current)
                retrieval = _merge_retrieval(retrieval_runs)
                candidates = self._decision_candidates(
                    task,
                    coding,
                    retrieval,
                    enable_evidence_adjustments=self.runtime.enable_evidence_adjustments,
                    max_evidence_adjustments=self.runtime.max_evidence_adjustments,
                )
            elif stage == "decide":
                decision = self.decision.run(
                    candidates,
                    retrieval,
                    prior_decisions=tuple(decisions),
                    round_index=len(decisions),
                )
                decisions.append(decision)
            else:
                raise ValueError(f"Unknown harness stage: {stage}")
        if not decisions:
            decisions.append(self.decision.run(candidates, retrieval))
        decision = _aggregate_decisions(
            tuple(decisions),
            candidates,
            self.runtime.decision_aggregation,
        )
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
        *,
        enable_evidence_adjustments: bool = True,
        max_evidence_adjustments: int = 3,
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
        if not enable_evidence_adjustments:
            return tuple(candidates)
        if max_evidence_adjustments <= 0:
            return tuple(candidates)
        best = min(candidates, key=lambda item: item.hindcast_smape)
        adjustments = 0
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
            adjustments += 1
            if adjustments >= max_evidence_adjustments:
                break
        return tuple(candidates)

    @staticmethod
    def score_after_resolution(task: ContextTask, result: HarnessResult) -> ResolvedOutcome:
        """Compatibility wrapper around the immutable evaluation host."""
        return score_after_resolution(task, result)

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


def _merge_retrieval(results: list[RetrievalResult]) -> RetrievalResult:
    evidence = []
    evidence_keys = set()
    impacts = []
    impact_keys = set()
    for result in results:
        for item in result.evidence:
            key = (item.document_id, item.exact_quote)
            if key not in evidence_keys:
                evidence_keys.add(key)
                evidence.append(item)
        for item in result.impacts:
            key = (
                item.source_document_ids,
                item.mechanism_layer,
                item.temporal_relation,
                item.adjustment_kind,
                item.adjustment_value,
                item.start_timestamp,
                item.end_timestamp,
            )
            if key not in impact_keys:
                impact_keys.add(key)
                impacts.append(item)
    return RetrievalResult(
        query=" | ".join(item.query for item in results if item.query),
        selected_document_ids=tuple(
            dict.fromkeys(value for item in results for value in item.selected_document_ids)
        ),
        evidence=tuple(evidence),
        impacts=tuple(impacts),
        sufficient=any(item.sufficient for item in results),
        missing_information=tuple(
            dict.fromkeys(value for item in results for value in item.missing_information)
        ),
        rejected=tuple(dict.fromkeys(value for item in results for value in item.rejected)),
        used_skill_names=tuple(
            dict.fromkeys(value for item in results for value in item.used_skill_names)
        ),
    )


def _empty_retrieval() -> RetrievalResult:
    return RetrievalResult(
        query="",
        selected_document_ids=(),
        evidence=(),
        impacts=(),
        sufficient=False,
        missing_information=("No retrieval stage has run.",),
    )


def _aggregate_decisions(
    decisions: tuple[DecisionResult, ...],
    candidates: tuple[DecisionCandidate, ...],
    strategy: str,
) -> DecisionResult:
    if not decisions:
        raise RuntimeError("Harness requires at least one Decision round")
    if strategy == "last" or len(decisions) == 1:
        return decisions[-1]
    votes = Counter(item.selected.candidate_id for item in decisions)
    largest = max(votes.values())
    finalists = {candidate_id for candidate_id, count in votes.items() if count == largest}
    chosen = min(
        (candidate for candidate in candidates if candidate.candidate_id in finalists),
        key=lambda item: item.hindcast_smape,
    )
    source = next(item for item in reversed(decisions) if item.selected.candidate_id == chosen.candidate_id)
    return replace(source, selected=chosen, rationale=f"Panel aggregation ({strategy}): {source.rationale}")
