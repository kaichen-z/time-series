"""End-to-end Coding -> Retrieval -> Decision forecasting harness."""
from __future__ import annotations

import math
import statistics
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
                    host_default_id=coding.selected.program.name,
                    prior_decisions=tuple(decisions),
                    round_index=len(decisions),
                )
                decisions.append(decision)
            else:
                raise ValueError(f"Unknown harness stage: {stage}")
        if decisions and decisions[-1].requested_more_retrieval and len(retrieval_runs) < 2:
            current = self.retrieval.run(
                task,
                coding.candidates,
                prior=retrieval,
                round_index=len(retrieval_runs),
                decision_feedback=decisions[-1].rationale,
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
            decisions.append(
                self.decision.run(
                    candidates,
                    retrieval,
                    host_default_id=coding.selected.program.name,
                    prior_decisions=tuple(decisions),
                    round_index=len(decisions),
                )
            )
        if not decisions:
            decisions.append(
                self.decision.run(
                    candidates,
                    retrieval,
                    host_default_id=coding.selected.program.name,
                )
            )
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

    def _historical_clean_candidate(
        self,
        task: ContextTask,
        coding: CodingEvolutionResult,
        retrieval: RetrievalResult,
    ) -> DecisionCandidate | None:
        impacts = [
            impact
            for impact in retrieval.impacts
            if impact.mechanism_layer == "observation"
            and impact.temporal_relation in {"historical", "ended_before_future"}
            and impact.adjustment_kind in {"history_mask", "history_repair"}
            and impact.start_timestamp is not None
            and impact.end_timestamp is not None
        ]
        kinds = {impact.adjustment_kind for impact in impacts}
        if not impacts or len(kinds) != 1:
            return None

        raw_history = task.numeric.history_values
        history = list(raw_history)
        history_offset = 0
        if kinds == {"history_mask"}:
            masked = {
                index
                for impact in impacts
                for index, timestamp in enumerate(task.history_timestamps)
                if impact.start_timestamp <= timestamp <= impact.end_timestamp
            }
            if not masked:
                return None
            history_offset = max(masked) + 1
            history = history[history_offset:]
        else:
            rates: dict[tuple[str | None, str | None], set[float]] = {}
            for impact in impacts:
                key = (impact.start_timestamp, impact.end_timestamp)
                rates.setdefault(key, set())
                if impact.adjustment_value is not None:
                    rates[key].add(impact.adjustment_value)
            if any(len(values) != 1 for values in rates.values()):
                return None
            used_indexes: set[int] = set()
            for (start, end), values in rates.items():
                indexes = [
                    index
                    for index, timestamp in enumerate(task.history_timestamps)
                    if start <= timestamp <= end
                ]
                if not indexes or used_indexes.intersection(indexes):
                    return None
                used_indexes.update(indexes)
                rate = next(iter(values))
                for step, index in enumerate(indexes, start=1):
                    history[index] += rate * step
            if not all(math.isfinite(value) for value in history):
                return None

        clean_task = replace(task.numeric_view(), history_values=tuple(history))
        folds = self.coding._folds(clean_task)
        if len(folds) < self.coding.config.validation_folds:
            return None
        validated = [
            candidate
            for item in coding.candidates
            if (candidate := self.coding._validate(clean_task, item.program)) is not None
            and not candidate.fold_errors
        ]
        if len(validated) < 3:
            return None
        top = sorted(validated, key=lambda candidate: candidate.hindcast_smape)[:3]
        raw_host = coding.selected.program
        raw_scores = self.coding.score_program_folds(
            raw_host,
            tuple(
                (
                    tuple(raw_history[: history_offset + len(train)]),
                    target,
                )
                for train, target in folds
            ),
            clean_task.frequency,
        )
        ensemble_result = self.coding.validate_median(clean_task, tuple(top))
        if raw_scores is None or ensemble_result is None:
            return None
        ensemble_forecast, ensemble_scores = ensemble_result

        best = top[0]
        selected_forecast = best.forecast
        selected_scores = best.fold_scores
        selected_id = f"history_clean__{best.program.name}"
        selected_tags = ("history_cleaned", "observation", "single")
        distinct_forecasts = {candidate.forecast for candidate in top}
        if (
            len(distinct_forecasts) >= 2
            and statistics.fmean(ensemble_scores) < 0.99 * best.hindcast_smape
            and max(ensemble_scores) <= max(best.fold_scores)
        ):
            selected_forecast = ensemble_forecast
            selected_scores = ensemble_scores
            selected_id = "history_clean_top3_median"
            selected_tags = ("history_cleaned", "observation", "validated_ensemble")

        if not (
            statistics.fmean(selected_scores) < 0.99 * statistics.fmean(raw_scores)
            and max(selected_scores) <= max(raw_scores)
        ):
            return None
        treatment = "post-defect suffix" if history_offset else "exact additive repair"
        return DecisionCandidate(
            candidate_id=selected_id,
            forecast=selected_forecast,
            assumption=(
                f"Verified observation evidence supports a {treatment}. On the same cleaned "
                "validation targets, this route improved mean and worst-fold sMAPE over the "
                "raw-history host replay."
            ),
            failure_condition=(
                "The cited defect window or repair rate is wrong, or fewer than three "
                "clean causal folds remain."
            ),
            hindcast_smape=statistics.fmean(selected_scores),
            source_document_ids=tuple(
                dict.fromkeys(
                    document_id
                    for impact in impacts
                    for document_id in impact.source_document_ids
                )
            ),
            tags=selected_tags,
        )

    def _resolved_regime_consensus_candidate(
        self,
        task: ContextTask,
        coding: CodingEvolutionResult,
        retrieval: RetrievalResult,
    ) -> DecisionCandidate | None:
        """Expose a broad numeric consensus when evidence invalidates a past regime."""
        if len(coding.candidates) < 6:
            return None
        relevant = [
            impact
            for impact in retrieval.impacts
            if impact.source_document_ids
            and impact.direction == "stable"
            and impact.adjustment_kind in {"preserve", "history_downweight"}
            and (
                impact.temporal_relation == "ended_before_future"
                or (
                    impact.temporal_relation == "overlaps_future"
                    and impact.mechanism_layer == "regime"
                )
            )
        ]
        if not relevant:
            return None
        if any(
            impact.adjustment_kind == "history_mask"
            and all(
                impact.start_timestamp <= timestamp <= impact.end_timestamp
                for timestamp in task.history_timestamps
            )
            for impact in retrieval.impacts
            if impact.start_timestamp is not None and impact.end_timestamp is not None
        ):
            return None
        source_ids = tuple(
            dict.fromkeys(
                document_id
                for impact in relevant
                for document_id in impact.source_document_ids
            )
        )
        verified_ids = {item.document_id for item in retrieval.evidence}
        if not source_ids or not set(source_ids).issubset(verified_ids):
            return None
        members = tuple(
            sorted(coding.candidates, key=lambda item: item.hindcast_smape)[:6]
        )
        validated = self.coding.validate_median(task.numeric_view(), members)
        if validated is None:
            return None
        return DecisionCandidate(
            candidate_id="resolved_regime_top6_consensus",
            forecast=validated[0],
            assumption=(
                "Verified evidence says a historical anomaly ended or a stable regime resumed. "
                "Use the pointwise median of six independently executed numeric trajectories "
                "instead of trusting one recurrence-specific host: "
                + ", ".join(item.program.name for item in members)
            ),
            failure_condition=(
                "The cited regime resolution is wrong, or most numeric members share the same "
                "post-origin bias."
            ),
            hindcast_smape=statistics.fmean(validated[1]),
            source_document_ids=source_ids,
            tags=("regime_consensus", "evidence_grounded", "validated_ensemble"),
        )

    def _decision_candidates(
        self,
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
        if enable_evidence_adjustments and max_evidence_adjustments > 0:
            best = min(candidates, key=lambda item: item.hindcast_smape)
            adjustments = 0
            for index, impact in enumerate(retrieval.impacts):
                if impact.temporal_relation != "overlaps_future":
                    continue
                if impact.adjustment_kind not in {"multiply", "add"} or impact.adjustment_value is None:
                    continue
                if _persistent_effect_already_observed(task, impact, retrieval.impacts):
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
        clean = self._historical_clean_candidate(task, coding, retrieval)
        if clean is not None:
            candidates.append(clean)
        else:
            consensus = self._resolved_regime_consensus_candidate(task, coding, retrieval)
            if consensus is not None:
                candidates.append(consensus)
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


def _persistent_effect_already_observed(
    task: ContextTask,
    impact: Any,
    impacts: tuple[Any, ...],
) -> bool:
    if impact.permanence != "permanent" or not task.future_timestamps:
        return False
    origin = task.future_timestamps[0]
    if impact.start_timestamp is not None and impact.start_timestamp < origin:
        return True
    return any(
        other is not impact
        and other.permanence == "permanent"
        and other.adjustment_kind == impact.adjustment_kind
        and other.adjustment_value == impact.adjustment_value
        and set(other.source_document_ids) == set(impact.source_document_ids)
        and other.start_timestamp is not None
        and other.start_timestamp < origin
        and (other.end_timestamp is None or other.end_timestamp >= origin)
        for other in impacts
    )


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
