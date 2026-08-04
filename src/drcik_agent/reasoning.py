from __future__ import annotations

import statistics
from dataclasses import replace

from .models import (
    AgentBeliefState,
    Diagnosis,
    EvidenceImpact,
    ForecastTask,
    LinguisticBelief,
    MacroOutlook,
    MicroEventOutlook,
    MicroOutlook,
    RevisionAction,
    RevisionDecision,
)


class MacroReasoningAgent:
    """NEXUS-style broad numerical outlook anchored to the baseline model."""

    def analyze(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        baseline_method: str,
    ) -> MacroOutlook:
        seasonal_confidence = diagnosis.seasonal_strength if diagnosis.seasonal_period else 0.0
        trend_scale = max(statistics.fmean(abs(value) for value in task.history_values), 1e-8)
        trend_signal = min(1.0, abs(diagnosis.slope_per_step) / trend_scale * 20.0)
        confidence = min(0.95, 0.45 + 0.35 * seasonal_confidence + 0.20 * trend_signal)
        if diagnosis.seasonal_period:
            pattern = (
                f"a {diagnosis.seasonal_period}-step cycle with strength "
                f"{diagnosis.seasonal_strength:.2f}"
            )
        else:
            pattern = "no sufficiently strong repeating cycle"
        summary = (
            f"The numerical history has a {diagnosis.trend} macro trajectory and {pattern}. "
            f"Use {baseline_method} as the prior unless verified future events justify a local revision."
        )
        return MacroOutlook(
            direction=diagnosis.trend,
            slope_per_step=diagnosis.slope_per_step,
            seasonal_period=diagnosis.seasonal_period,
            seasonal_strength=diagnosis.seasonal_strength,
            baseline_method=baseline_method,
            confidence=confidence,
            summary=summary,
        )


class MicroReasoningAgent:
    """NEXUS-style granular outlook over localized events and catalysts."""

    def analyze(self, impacts: list[EvidenceImpact]) -> MicroOutlook:
        events = tuple(
            MicroEventOutlook(
                event_type=impact.event_type,
                direction=impact.direction,
                start_timestamp=impact.start_timestamp,
                end_timestamp=impact.end_timestamp,
                forecast_relation=impact.forecast_relation,
                adjustment_kind=impact.adjustment_kind,
                confidence=impact.confidence,
                source_document_ids=impact.source_document_ids,
            )
            for impact in impacts
        )
        actionable = [
            event
            for event in events
            if event.forecast_relation in {"overlaps_forecast", "forecast_relevant_undated"}
            and event.adjustment_kind
            not in {"qualitative_only", "return_to_baseline", "already_in_baseline"}
        ]
        confidence = statistics.fmean(event.confidence for event in events) if events else 0.0
        summary = (
            f"Found {len(events)} localized event hypotheses; {len(actionable)} can potentially "
            "revise the requested horizon. Resolved, stale, or unquantified events remain context only."
        )
        return MicroOutlook(events=events, confidence=confidence, summary=summary)


class RevisionUtilityAgent:
    """PostTime-style revise-or-preserve gate using only inference-time signals."""

    EVENT_QUESTION = {
        "anomaly": "anomaly_cause",
        "resolution": "resolution_permanence",
        "promotion": "external_drivers",
        "external_driver": "external_drivers",
        "forecast_regime": "forecast_regime",
        "explicit_future_value": "forecast_regime",
    }

    def __init__(self, threshold: float = 0.60) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("revision threshold must be between 0 and 1")
        self.threshold = threshold

    @staticmethod
    def _belief(state: AgentBeliefState, event_type: str) -> LinguisticBelief | None:
        question_id = RevisionUtilityAgent.EVENT_QUESTION.get(event_type, "external_drivers")
        return state.linguistic_beliefs.get(question_id)

    def evaluate(
        self,
        action: RevisionAction,
        macro: MacroOutlook,
        micro: MicroOutlook,
        state: AgentBeliefState,
    ) -> RevisionDecision:
        if action.action_type == "preserve":
            return RevisionDecision(
                action_id=action.action_id,
                revise=False,
                utility_score=1.0,
                threshold=self.threshold,
                reasons=("The evidence-to-impact stage explicitly recommends preserving the prior.",),
                fallback_action_id=action.action_id,
            )

        belief = self._belief(state, action.event_type)
        belief_score = belief.evidence_sufficiency if belief else 0.5
        source_score = min(1.0, len(action.source_document_ids) / 2.0)
        inferred_magnitude = "No magnitude is stated" in action.rationale
        specificity = (
            0.25
            if inferred_magnitude
            else 1.0 if action.value is not None or action.values else 0.0
        )
        temporal = 1.0 if action.start_index <= action.end_index else 0.0
        memory_support = min(1.0, len(action.memory_entry_ids) / 2.0)
        utility = (
            0.42 * action.confidence
            + 0.18 * belief_score
            + 0.14 * source_score
            + 0.14 * specificity
            + 0.08 * temporal
            + 0.04 * memory_support
        )
        reasons = [
            f"evidence confidence={action.confidence:.2f}",
            f"belief sufficiency={belief_score:.2f}",
            f"source corroboration={source_score:.2f}",
            f"magnitude specificity={specificity:.2f}",
        ]

        matching = [event for event in micro.events if event.event_type == action.event_type]
        if matching:
            event = max(matching, key=lambda item: item.confidence)
            opposite = (
                (macro.direction == "upward" and event.direction == "down")
                or (macro.direction == "downward" and event.direction == "up")
            )
            if opposite and event.adjustment_kind == "standardized_additive" and macro.confidence >= 0.65:
                utility = max(0.0, utility - 0.08)
                reasons.append("weak event direction conflicts with a confident macro trajectory")

        revise = utility >= self.threshold
        if revise:
            reasons.append("predicted revision utility clears the revise threshold")
            fallback_id = None
        else:
            reasons.append("predicted revision utility is too weak; preserve the numerical prior")
            fallback_id = f"{action.action_id}-preserve"
        return RevisionDecision(
            action_id=action.action_id,
            revise=revise,
            utility_score=utility,
            threshold=self.threshold,
            reasons=tuple(reasons),
            fallback_action_id=fallback_id,
        )

    @staticmethod
    def fallback(action: RevisionAction, decision: RevisionDecision) -> RevisionAction:
        if decision.revise:
            return action
        return replace(
            action,
            action_id=decision.fallback_action_id or f"{action.action_id}-preserve",
            action_type="preserve",
            value=None,
            values=(),
            rationale=(
                "PostTime-style fallback: the available context does not justify changing "
                "the numerical prior. " + "; ".join(decision.reasons)
            ),
        )
