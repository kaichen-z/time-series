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
from evolving_loop.morphology_adapter import MorphologyProvider
from evolving_loop.retrieval_agent.agent import RetrievalAgent, RetrievalResult
from evolving_loop.retrieval_agent.schemas import (
    FinalRetrievalCard,
    RetrievalAssumption,
)
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from evolving_loop.retrieval_agent.verifier import merge_verified_rounds


@dataclass(frozen=True)
class HarnessResult:
    task_id: str
    coding: CodingEvolutionResult
    retrieval: RetrievalResult
    decision: DecisionResult
    candidates: tuple[DecisionCandidate, ...]
    forecast: tuple[float, ...]
    retrieval_card: FinalRetrievalCard | None = None


@dataclass(frozen=True)
class HarnessRuntimeConfig:
    """Executable topology controls exposed to the outer Meta-Harness."""

    workflow: tuple[str, ...] = ("retrieve", "decide")
    retrieval_mode: str = "single_pass"
    enable_evidence_adjustments: bool = True
    max_evidence_adjustments: int = 3
    decision_aggregation: str = "last"

    def __post_init__(self) -> None:
        if self.retrieval_mode not in {"single_pass", "two_stage"}:
            raise ValueError("retrieval_mode must be single_pass or two_stage")


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
        morphology: MorphologyProvider | None = None,
    ) -> None:
        self.coding = coding
        self.retrieval = retrieval
        self.decision = decision
        self.outcome_learner = outcome_learner
        self.runtime = runtime or HarnessRuntimeConfig()
        self.morphology = morphology
        if self.runtime.retrieval_mode == "two_stage":
            if not isinstance(self.retrieval, TwoStageRetrievalAgent):
                raise ValueError("two_stage retrieval_mode requires TwoStageRetrievalAgent")
            if not isinstance(self.morphology, MorphologyProvider) or not callable(
                getattr(self.morphology, "assumptions", None)
            ):
                raise ValueError("two_stage retrieval_mode requires MorphologyProvider")

    def run(
        self, task: ContextTask, *, allow_skill_writes: bool = True
    ) -> HarnessResult:
        # Keep this local boundary for normal ``run`` calls as well as evaluator calls.
        coding = self.coding.run_task(
            task.numeric_view(), allow_skill_writes=allow_skill_writes
        )
        if self.runtime.retrieval_mode == "two_stage":
            return self._run_two_stage(task, coding)
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
        decision = _guard_raw_override(decision, candidates, coding)
        return HarnessResult(
            task_id=task.numeric.task_id,
            coding=coding,
            retrieval=retrieval,
            decision=decision,
            candidates=candidates,
            forecast=decision.selected.forecast,
        )

    def _run_two_stage(self, task: ContextTask, coding: CodingEvolutionResult) -> HarnessResult:
        """Execute the fixed two-stage topology without interpreting ``workflow``."""
        assert isinstance(self.retrieval, TwoStageRetrievalAgent)
        assert self.morphology is not None
        morphology_failure: str | None = None
        try:
            assumptions = tuple(
                RetrievalAssumption.from_payload(
                    item.to_payload() if isinstance(item, RetrievalAssumption) else item
                )
                for item in self.morphology.assumptions(task)
            )
        except Exception as error:
            assumptions = ()
            morphology_failure = f"morphology_provider_failed:{type(error).__name__}"

        round1 = self.retrieval.run_round1(task)
        round1_card = merge_verified_rounds(round1, None)
        if morphology_failure is not None:
            round1_card = replace(
                round1_card,
                rejected=tuple(dict.fromkeys((*round1_card.rejected, morphology_failure))),
            )
        provisional_retrieval = round1_card.to_legacy_result()
        provisional_candidates = self._decision_candidates(
            task,
            coding,
            provisional_retrieval,
            enable_evidence_adjustments=self.runtime.enable_evidence_adjustments,
            max_evidence_adjustments=self.runtime.max_evidence_adjustments,
        )
        provisional = self.decision.run(
            provisional_candidates,
            provisional_retrieval,
            host_default_id=coding.selected.program.name,
            round_index=0,
            assumptions=assumptions,
        )
        provisional = _guard_raw_override(provisional, provisional_candidates, coding)

        round2 = None
        if morphology_failure is None and assumptions and _should_run_round2(
            self.retrieval.genome.second_round_trigger,
            round1,
            provisional,
        ):
            round2 = self.retrieval.run_round2(
                task,
                round1,
                provisional.gaps,
                assumptions,
            )
        card = merge_verified_rounds(round1, round2)
        if morphology_failure is not None:
            card = replace(
                card,
                rejected=tuple(dict.fromkeys((*card.rejected, morphology_failure))),
            )
        retrieval = card.to_legacy_result()
        candidates = self._decision_candidates(
            task,
            coding,
            retrieval,
            enable_evidence_adjustments=self.runtime.enable_evidence_adjustments,
            max_evidence_adjustments=self.runtime.max_evidence_adjustments,
        )
        decision = self.decision.run(
            candidates,
            retrieval,
            host_default_id=coding.selected.program.name,
            prior_decisions=(provisional,),
            round_index=1,
            assumptions=assumptions,
        )
        decision = _guard_raw_override(decision, candidates, coding)
        return HarnessResult(
            task_id=task.numeric.task_id,
            coding=coding,
            retrieval=retrieval,
            decision=decision,
            candidates=candidates,
            forecast=decision.selected.forecast,
            retrieval_card=card,
        )

    def _historical_clean_candidate(
        self,
        task: ContextTask,
        coding: CodingEvolutionResult,
        retrieval: RetrievalResult,
    ) -> DecisionCandidate | None:
        """Replay executable programs after a citation-grounded observation repair."""
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
        if not impacts or len(kinds) != 1 or coding.selected.program.source == "tsfm":
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
            if item.program.source != "tsfm"
            and (candidate := self.coding._validate(clean_task, item.program)) is not None
            and not candidate.fold_errors
        ]
        if len(validated) < 3:
            return None
        top = sorted(
            validated,
            key=lambda candidate: (candidate.hindcast_srmse, candidate.hindcast_smae),
        )[:3]
        raw_scores = self.coding.score_program_folds(
            coding.selected.program,
            tuple(
                (
                    tuple(raw_history[: history_offset + len(train)]),
                    target,
                )
                for train, target in folds
            ),
            clean_task.frequency,
        )
        ensemble = self.coding.validate_median(clean_task, tuple(top))
        if raw_scores is None or ensemble is None:
            return None
        ensemble_forecast, ensemble_smae, ensemble_srmse = ensemble

        best = top[0]
        selected_forecast = best.forecast
        selected_smae = best.fold_smae
        selected_srmse = best.fold_srmse
        selected_id = f"history_clean__{best.program.name}"
        selected_tags = ("history_cleaned", "observation", "single")
        if (
            len({candidate.forecast for candidate in top}) >= 2
            and _fold_pair_improves(
                ensemble_smae,
                ensemble_srmse,
                best.fold_smae,
                best.fold_srmse,
            )
        ):
            selected_forecast = ensemble_forecast
            selected_smae = ensemble_smae
            selected_srmse = ensemble_srmse
            selected_id = "history_clean_top3_median"
            selected_tags = ("history_cleaned", "observation", "validated_ensemble")

        raw_smae, raw_srmse = raw_scores
        if not _fold_pair_improves(
            selected_smae,
            selected_srmse,
            raw_smae,
            raw_srmse,
        ):
            return None
        treatment = "post-defect suffix" if history_offset else "exact additive repair"
        return DecisionCandidate(
            candidate_id=selected_id,
            forecast=selected_forecast,
            assumption=(
                f"Verified observation evidence supports a {treatment}. On identical cleaned "
                "validation targets, this route passed mean and worst-fold sMAE/sRMSE gates "
                "against the raw-history host replay."
            ),
            failure_condition=(
                "The cited defect window or repair rate is wrong, or fewer than three "
                "clean causal folds remain."
            ),
            hindcast_smae=statistics.fmean(selected_smae),
            hindcast_srmse=statistics.fmean(selected_srmse),
            source_document_ids=tuple(
                dict.fromkeys(
                    document_id
                    for impact in impacts
                    for document_id in impact.source_document_ids
                )
            ),
            tags=selected_tags,
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
                hindcast_smae=item.hindcast_smae,
                hindcast_srmse=item.hindcast_srmse,
                tags=(item.program.source,),
            )
            for item in coding.candidates
        ]
        if not enable_evidence_adjustments:
            return tuple(candidates)
        best = min(
            candidates, key=lambda item: (item.hindcast_srmse, item.hindcast_smae)
        )
        adjustments = 0
        for index, impact in enumerate(retrieval.impacts):
            if adjustments >= max_evidence_adjustments:
                break
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
                    hindcast_smae=best.hindcast_smae,
                    hindcast_srmse=best.hindcast_srmse,
                    source_document_ids=impact.source_document_ids,
                    tags=("evidence_adjusted", impact.mechanism_layer),
                )
            )
            adjustments += 1
        clean = self._historical_clean_candidate(task, coding, retrieval)
        if clean is not None:
            candidates.append(clean)
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


def _fold_pair_improves(
    candidate_smae: tuple[float, ...],
    candidate_srmse: tuple[float, ...],
    reference_smae: tuple[float, ...],
    reference_srmse: tuple[float, ...],
    *,
    minimum_relative_gain: float = 0.01,
) -> bool:
    """Require Pareto-safe mean and worst-fold gains in the two official metrics."""
    candidate_mean_smae = statistics.fmean(candidate_smae)
    candidate_mean_srmse = statistics.fmean(candidate_srmse)
    reference_mean_smae = statistics.fmean(reference_smae)
    reference_mean_srmse = statistics.fmean(reference_srmse)
    return (
        candidate_mean_smae <= reference_mean_smae
        and candidate_mean_srmse <= reference_mean_srmse
        and max(candidate_smae) <= max(reference_smae)
        and max(candidate_srmse) <= max(reference_srmse)
        and (
            candidate_mean_smae
            < (1.0 - minimum_relative_gain) * reference_mean_smae
            or candidate_mean_srmse
            < (1.0 - minimum_relative_gain) * reference_mean_srmse
        )
    )


def _fold_pair_within(
    candidate_smae: tuple[float, ...],
    candidate_srmse: tuple[float, ...],
    reference_smae: tuple[float, ...],
    reference_srmse: tuple[float, ...],
    *,
    relative_slack: float = 0.05,
) -> bool:
    """Keep evidence-selectable alternatives whose two fold metrics are comparable."""
    multiplier = 1.0 + relative_slack
    return (
        statistics.fmean(candidate_smae) <= multiplier * statistics.fmean(reference_smae)
        and statistics.fmean(candidate_srmse)
        <= multiplier * statistics.fmean(reference_srmse)
        and max(candidate_smae) <= multiplier * max(reference_smae)
        and max(candidate_srmse) <= multiplier * max(reference_srmse)
    )


def _guard_raw_override(
    decision: DecisionResult,
    candidates: tuple[DecisionCandidate, ...],
    coding: CodingEvolutionResult,
) -> DecisionResult:
    """Reject only unstable raw overrides, after preserving the full Decision context."""
    host = coding.selected
    if decision.selected.candidate_id == host.program.name:
        return decision
    raw = next(
        (
            item
            for item in coding.candidates
            if item.program.name == decision.selected.candidate_id
        ),
        None,
    )
    if raw is None:
        return decision
    host_is_near_perfect = max(host.hindcast_smae, host.hindcast_srmse) <= 0.01
    if (
        host_is_near_perfect
        or _fold_pair_improves(
            raw.fold_smae,
            raw.fold_srmse,
            host.fold_smae,
            host.fold_srmse,
        )
        or _fold_pair_within(
            raw.fold_smae,
            raw.fold_srmse,
            host.fold_smae,
            host.fold_srmse,
        )
    ):
        return decision
    host_candidate = next(
        item for item in candidates if item.candidate_id == host.program.name
    )
    return replace(
        decision,
        selected=host_candidate,
        requested_more_retrieval=False,
        rationale="Preserve the stable numeric host after the raw override failed its fold gate.",
        supporting_document_ids=(),
        llm_override_accepted=False,
        rejection_reason="raw_override_failed_smae_srmse_fold_gate",
        used_skill_names=(),
    )


def _persistent_effect_already_observed(
    task: ContextTask,
    impact: Any,
    impacts: tuple[Any, ...],
) -> bool:
    """Avoid applying a permanent effect twice when history already contains it."""
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


def _should_run_round2(
    trigger: str,
    round1: Any,
    provisional: DecisionResult,
) -> bool:
    if trigger == "never":
        return False
    if trigger == "always":
        return True
    if trigger == "on_named_gap":
        return provisional.requested_more_retrieval and bool(provisional.gaps)
    if trigger == "on_incomplete_chain":
        return (
            not round1.sufficient
            or not round1.chains
            or any(item.missing_links or not item.numeric_eligible for item in round1.chains)
        )
    raise ValueError(f"Unknown second-round trigger: {trigger}")


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
        key=lambda item: (item.hindcast_srmse, item.hindcast_smae),
    )
    source = next(item for item in reversed(decisions) if item.selected.candidate_id == chosen.candidate_id)
    return replace(source, selected=chosen, rationale=f"Panel aggregation ({strategy}): {source.rationale}")
