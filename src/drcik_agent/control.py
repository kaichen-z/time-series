from __future__ import annotations

from dataclasses import replace

from .models import (
    AgentBeliefState,
    Diagnosis,
    EvidenceVerdict,
    ForecastGap,
    ForecastTask,
    LinguisticBelief,
    QueryAction,
    SufficiencyDecision,
)


class ForecastGapControllerAgent:
    """Judge evidence sufficiency and make the next information need explicit.

    The controller borrows the judge-first interface from S2G-RAG.  This local
    implementation is deterministic so that the benchmark can be run without
    an LLM; a trained judge can replace ``decide`` while preserving the same
    structured state and trace format.
    """

    def initial_gaps(self, task: ForecastTask, diagnosis: Diagnosis) -> dict[str, ForecastGap]:
        horizon = f"{task.future_timestamps[0]} to {task.future_timestamps[-1]}"
        history = f"{task.history_timestamps[0]} to {task.history_timestamps[-1]}"
        gaps = {
            "historical_regime": ForecastGap(
                gap_id="historical_regime",
                category="causal_history",
                target=task.target_name,
                time_scope=history,
                description=(
                    "Explain material historical anomalies or regime changes that must not "
                    "be extrapolated into the forecast."
                ),
                query_terms=(
                    "anomaly", "regime", "cause", "incident", "bug", "error",
                    "spike", "drop", "temporary", "permanent",
                ),
                priority=0.90,
            ),
            "external_drivers": ForecastGap(
                gap_id="external_drivers",
                category="future_catalyst",
                target=task.target_name,
                time_scope=horizon,
                description=(
                    "Identify external events or interventions that overlap the forecast "
                    "horizon, including their direction, dates, and magnitude."
                ),
                query_terms=(
                    "future", "event", "promotion", "policy", "maintenance",
                    "weather", "increase", "decrease", "start", "end", "magnitude",
                ),
                priority=1.00,
            ),
            "forecast_regime": ForecastGap(
                gap_id="forecast_regime",
                category="future_regime",
                target=task.target_name,
                time_scope=horizon,
                description=(
                    "Determine the baseline, trend, and seasonal regime that should govern "
                    "the requested forecast horizon."
                ),
                query_terms=(
                    "forecast", "baseline", "normal", "seasonality", "periodic",
                    "cycle", "trend", "trajectory", diagnosis.trend,
                ),
                priority=0.95,
            ),
        }
        return gaps

    @property
    def question_ids(self) -> list[str]:
        """Initial gap IDs retained for callers of the pre-0.7 planner API."""
        return ["historical_regime", "external_drivers", "forecast_regime"]

    @staticmethod
    def _belief(state: AgentBeliefState, gap_id: str) -> LinguisticBelief:
        return state.linguistic_beliefs.get(gap_id, LinguisticBelief(gap_id, 0.5))

    def decide(
        self,
        task: ForecastTask,
        state: AgentBeliefState,
    ) -> tuple[SufficiencyDecision, QueryAction | None]:
        unresolved = [
            state.forecast_gaps[gap_id]
            for gap_id in state.open_question_ids
            if gap_id in state.forecast_gaps
        ]
        if not unresolved:
            exhausted = tuple(state.exhausted_question_ids)
            evidence_sufficient = not exhausted
            decision = SufficiencyDecision(
                sufficient=evidence_sufficient,
                resolved_gap_ids=tuple(state.answered_question_ids),
                unresolved_gap_ids=exhausted,
                selected_gap_id=None,
                next_query=None,
                expected_information_gain=0.0,
                rationale=(
                    "All active forecast information gaps are supported by grounded evidence."
                    if evidence_sufficient
                    else "No searchable gap remains, but some information needs were exhausted "
                    "without adequate evidence; preserve the numerical prior for those gaps."
                ),
                stop_reason=(
                    "evidence_sufficient"
                    if evidence_sufficient
                    else "unresolved_gaps_exhausted"
                ),
            )
            return decision, None

        selected = max(
            unresolved,
            key=lambda gap: (
                gap.priority
                * (1.0 - self._belief(state, gap.gap_id).evidence_sufficiency)
                / (1.0 + state.attempt_counts.get(gap.gap_id, 0)),
                -state.attempt_counts.get(gap.gap_id, 0),
                gap.gap_id,
            ),
        )
        belief = self._belief(state, selected.gap_id)
        attempts = state.attempt_counts.get(selected.gap_id, 0)
        expected_gain = selected.priority * (1.0 - belief.evidence_sufficiency) / (1.0 + attempts)
        query_parts = [
            task.entity_name,
            task.target_name,
            selected.category,
            selected.time_scope,
            selected.description,
            *selected.query_terms,
        ]
        # Rejected evidence becomes a negative constraint rather than an
        # ever-growing raw context.  This helps reformulate repeated searches.
        rejected = state.rejected_reasons
        if attempts and rejected:
            common_reasons = sorted({reason for reasons in rejected.values() for reason in reasons})
            query_parts.extend(("avoid", *common_reasons[:3]))
        query = " ".join(part for part in query_parts if part)
        action = QueryAction(
            question_id=selected.gap_id,
            question=selected.description,
            query=query,
            rationale=(
                f"Resolve structured gap {selected.gap_id}; expected information gain "
                f"is {expected_gain:.3f}."
            ),
        )
        decision = SufficiencyDecision(
            sufficient=False,
            resolved_gap_ids=tuple(state.answered_question_ids),
            unresolved_gap_ids=tuple(gap.gap_id for gap in unresolved),
            selected_gap_id=selected.gap_id,
            next_query=query,
            expected_information_gain=expected_gain,
            rationale=(
                "The evidence context is incomplete. Retrieve evidence for the highest-value "
                "explicit gap instead of following a fixed query schedule."
            ),
        )
        return decision, action

    def plan(self, task: ForecastTask, state: AgentBeliefState) -> QueryAction:
        """Compatibility wrapper around the structured judge-and-plan call."""
        _decision, action = self.decide(task, state)
        if action is None:
            raise RuntimeError("no unresolved forecast gap remains")
        return action

    @staticmethod
    def _ensure_gap(state: AgentBeliefState, gap: ForecastGap) -> None:
        if gap.gap_id in state.forecast_gaps:
            return
        state.forecast_gaps[gap.gap_id] = gap
        state.open_question_ids.append(gap.gap_id)
        state.linguistic_beliefs[gap.gap_id] = LinguisticBelief(gap.gap_id, 0.5)

    def expand_from_verdicts(
        self,
        task: ForecastTask,
        action: QueryAction,
        verdicts: list[EvidenceVerdict],
        state: AgentBeliefState,
    ) -> None:
        """Create follow-up gaps only when newly grounded evidence requires them."""
        accepted = [verdict for verdict in verdicts if verdict.accepted]
        event_types = {event for verdict in accepted for event in verdict.event_types}
        source_ids = tuple(verdict.document_id for verdict in accepted)
        horizon = f"{task.future_timestamps[0]} to {task.future_timestamps[-1]}"

        if "anomaly" in event_types and "resolution_permanence" not in state.answered_question_ids:
            self._ensure_gap(
                state,
                ForecastGap(
                    gap_id="resolution_permanence",
                    category="resolution_and_recurrence",
                    target=task.target_name,
                    time_scope=horizon,
                    description=(
                        "Determine whether the historical disruption was resolved and whether "
                        "it can recur during the forecast horizon."
                    ),
                    query_terms=(
                        "resolution", "patch", "fixed", "deployment", "recurrence",
                        "permanent", "temporary", "stabilized", "restored",
                    ),
                    priority=1.05,
                    created_from=source_ids,
                ),
            )

        if event_types & {"temporary_event", "external_driver"}:
            claims = " ".join(
                evidence.claim for verdict in accepted for evidence in verdict.evidence
            ).lower()
            has_magnitude = any(
                token in claims
                for token in ("percent", "%", " times", "double", "triple", " units", " points")
            )
            if not has_magnitude and action.question_id == "external_drivers":
                self._ensure_gap(
                    state,
                    ForecastGap(
                        gap_id="event_magnitude",
                        category="effect_magnitude",
                        target=task.target_name,
                        time_scope=horizon,
                        description=(
                            "Find quantitative evidence for the magnitude and duration of the "
                            "future event effect; otherwise preserve the numerical prior."
                        ),
                        query_terms=(
                            "impact", "magnitude", "percent", "change", "increase",
                            "decrease", "duration", "historical analogue",
                        ),
                        priority=0.85,
                        created_from=source_ids,
                    ),
                )

    @staticmethod
    def refresh_gap_priorities(state: AgentBeliefState) -> None:
        """Keep serialized gaps immutable while allowing evidence-aware priority decay."""
        for gap_id, gap in list(state.forecast_gaps.items()):
            belief = state.linguistic_beliefs.get(gap_id)
            if belief and belief.evidence_sufficiency >= 0.8 and gap_id in state.open_question_ids:
                state.forecast_gaps[gap_id] = replace(gap, priority=max(0.1, gap.priority * 0.5))


# Backward-compatible import for callers that used the old class name.  The
# behavior is now gap-driven rather than a fixed four-question scheduler.
QueryPlannerAgent = ForecastGapControllerAgent
