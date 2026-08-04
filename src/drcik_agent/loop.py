from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime

from .agents import EvidenceSynthesisAgent, ProbabilisticForecastAgent, RetrievalAgent, tokenize
from .backbones import TimesFMBackboneConfig, build_forecast_backbone
from .context import ImportanceAwareContextAgent, RetrievalProcessRewardAgent
from .impacts import EvidenceToForecastAgent
from .memory import ForecastMemoryBank
from .metrics import forecast_metrics, retrieval_metrics
from .models import (
    AgentBeliefState,
    Diagnosis,
    Document,
    Evidence,
    EvidenceVerdict,
    Forecast,
    ForecastTask,
    LinguisticBelief,
    QueryAction,
    RetrievedDocument,
    RunResult,
)
from .reasoning import MacroReasoningAgent, MicroReasoningAgent, RevisionUtilityAgent
from .workspace import ForecastWorkspaceExecutor, RevisionPlannerAgent


@dataclass(frozen=True)
class LoopConfig:
    max_steps: int = 10
    documents_per_step: int = 5
    num_samples: int = 100
    context_weight: float = 0.75
    max_no_progress: int = 4
    convergence_tolerance: float = 0.002
    seed: int = 7
    memory_path: str | None = None
    memory_weight: float = 0.25
    learn_from_public_outcomes: bool = False
    retrieval_candidate_multiplier: int = 3
    context_character_budget: int = 12000
    revision_utility_threshold: float = 0.60
    backbone: str = "timesfm"
    timesfm_model_id: str = "google/timesfm-2.5-200m-pytorch"
    timesfm_max_context: int = 4096
    timesfm_max_horizon: int = 1024
    timesfm_cache_dir: str | None = None
    timesfm_local_files_only: bool = False
    allow_statistical_fallback: bool = False

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.documents_per_step <= 0:
            raise ValueError("documents_per_step must be positive")
        if self.max_no_progress <= 0:
            raise ValueError("max_no_progress must be positive")
        if not 0.0 <= self.memory_weight <= 1.0:
            raise ValueError("memory_weight must be between 0 and 1")
        if self.retrieval_candidate_multiplier <= 0:
            raise ValueError("retrieval_candidate_multiplier must be positive")
        if self.context_character_budget <= 0:
            raise ValueError("context_character_budget must be positive")
        if not 0.0 <= self.revision_utility_threshold <= 1.0:
            raise ValueError("revision_utility_threshold must be between 0 and 1")
        if self.backbone not in {"timesfm", "statistical"}:
            raise ValueError("backbone must be 'timesfm' or 'statistical'")


class QueryPlannerAgent:
    """Choose the next unresolved forecast-relevant information need."""

    QUESTIONS = (
        (
            "anomaly_cause",
            "What caused the largest historical anomalies or regime changes?",
            "anomaly spike drop abnormal error bug incident cause malfunction inflated",
        ),
        (
            "resolution_permanence",
            "Was the historical disruption resolved, and is its effect temporary or permanent?",
            "resolution fix patch update deployment ended recurrence permanent temporary stabilized restored",
        ),
        (
            "external_drivers",
            "Which events or interventions changed the target, and over what time window?",
            "event promotion policy maintenance weather intervention impact increase decrease start end",
        ),
        (
            "forecast_regime",
            "Which numerical regime, trend, and seasonal pattern should govern the forecast horizon?",
            "forecast future baseline normal seasonality periodic cycle trend trajectory regime pattern",
        ),
    )

    @property
    def question_ids(self) -> list[str]:
        return [item[0] for item in self.QUESTIONS]

    def plan(self, task: ForecastTask, state: AgentBeliefState) -> QueryAction:
        available = [item for item in self.QUESTIONS if item[0] in state.open_question_ids]
        if not available:
            available = list(self.QUESTIONS)
        question_id, question, keywords = min(
            available,
            key=lambda item: (
                state.attempt_counts.get(item[0], 0),
                state.linguistic_beliefs.get(
                    item[0], LinguisticBelief(item[0], 0.5)
                ).evidence_sufficiency,
                self.question_ids.index(item[0]),
            ),
        )
        query = " ".join(
            (
                task.entity_name,
                task.target_name,
                task.target_description,
                task.history_timestamps[0],
                task.history_timestamps[-1],
                task.future_timestamps[0],
                task.future_timestamps[-1],
                keywords,
            )
        )
        return QueryAction(
            question_id=question_id,
            question=question,
            query=query,
            rationale=f"Resolve {question_id} using evidence not examined in prior steps.",
        )


class EvidenceVerifierAgent:
    """Forecast-aware evidence gate that never reads benchmark role/subtype labels."""

    TARGET_SYNONYMS = {
        "sales": {"transaction", "revenue", "purchase", "customer", "promotion", "discount"},
        "cpu": {"processor", "utilization", "load", "firmware", "compute"},
        "irradiance": {"solar", "radiation", "ghi", "flux", "sun", "watt"},
        "energy": {"electricity", "power", "demand", "consumption", "load"},
    }
    CAUSAL_TERMS = {
        "because", "cause", "caused", "causal", "due", "result", "resulted", "impact",
        "affect", "affected", "following", "after", "before", "trigger", "driven",
        "increase", "decrease", "spike", "drop", "shift", "restore", "resolved",
    }
    EVENT_TERMS = {
        "anomaly": {
            "anomaly", "bug", "error", "malfunction", "inflated", "artificial",
            "abnormal", "incident", "elevated", "surge", "higher",
        },
        "resolution": {"patch", "fix", "fixed", "resolved", "update", "deployment", "removed", "stabilized"},
        "temporary_event": {"promotion", "discount", "temporary", "event", "markdown", "campaign"},
        "external_driver": {
            "promotion", "discount", "campaign", "weather", "cloud", "maintenance",
            "policy", "intervention", "firmware", "pricing", "outage",
        },
        "forecast_regime": {"baseline", "seasonality", "periodic", "cycle", "forecast", "normal", "trajectory"},
    }
    QUESTION_EVENT_TYPES = {
        "anomaly_cause": {"anomaly"},
        "resolution_permanence": {"resolution"},
        "external_drivers": {"temporary_event", "external_driver"},
        "forecast_regime": {"forecast_regime"},
    }
    NON_PERIODIC_PHRASES = (
        "no seasonality",
        "non-seasonal",
        "nonseasonal",
        "non-periodic",
        "nonperiodic",
        "absence of periodic",
        "devoid of inherent periodic",
        "complete absence of periodic",
        "without periodic",
    )
    LINEAR_PHRASES = (
        "linear progression",
        "steady day-over-day growth",
        "smooth and consistent upward trajectory",
        "steady linear",
    )

    def __init__(self) -> None:
        self.synthesis_agent = EvidenceSynthesisAgent()

    @staticmethod
    def _publication_date(text: str) -> datetime | None:
        header = text[:800].replace("**", "")
        match = re.search(
            r"(?im)^\s*date\s*:\s*(\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
            header,
        )
        if not match:
            return None
        raw = match.group(1).replace(",", "")
        for format_string in ("%Y-%m-%d", "%B %d %Y"):
            try:
                return datetime.strptime(raw, format_string)
            except ValueError:
                continue
        return None

    @staticmethod
    def _years(text: str) -> set[int]:
        return {int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", text)}

    @staticmethod
    def _fallback_evidence(document: Document, query: str, score: float) -> Evidence:
        fragments = [
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+|\n+", document.text)
            if len(item.strip()) >= 30
        ]
        query_terms = set(tokenize(query))
        claim = max(
            fragments or [document.text[:800]],
            key=lambda item: len(set(tokenize(item)) & query_terms),
        )
        return Evidence(
            document_id=document.document_id,
            claim=claim[:800],
            matched_terms=tuple(sorted(set(tokenize(claim)) & query_terms)),
            confidence=min(0.95, max(0.35, score)),
            effect_direction="unclear",
            effect_window="unspecified",
        )

    def verify(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        action: QueryAction,
        retrieved: list[RetrievedDocument],
    ) -> list[EvidenceVerdict]:
        task_years = self._years(" ".join((*task.history_timestamps, *task.future_timestamps)))
        cutoff = datetime.fromisoformat(task.history_timestamps[-1])
        entity_terms = set(tokenize(task.entity_name))
        target_terms = set(tokenize(task.target_name))
        for token in list(target_terms):
            target_terms.update(self.TARGET_SYNONYMS.get(token, set()))
        query_terms = set(tokenize(action.query))
        dynamic_diagnosis = replace(diagnosis, retrieval_query=action.query)
        verdicts: list[EvidenceVerdict] = []

        for item in retrieved:
            document = item.document.agent_view()
            lower = document.text.lower()
            document_terms = set(tokenize(document.text))
            exact_entity = task.entity_name.lower() in lower
            entity_match = exact_entity or bool(entity_terms) and entity_terms <= document_terms
            target_overlap = document_terms & target_terms
            target_score = 1.0 if target_overlap else 0.0
            if not target_overlap and document_terms & set().union(*self.EVENT_TERMS.values()):
                target_score = 0.35

            publication_date = self._publication_date(document.text)
            document_years = self._years(document.text)
            if publication_date and publication_date > cutoff:
                temporal_alignment = "published_after_forecast_cutoff"
            elif document_years and not (document_years & task_years):
                temporal_alignment = "outside_task_window"
            elif document_years & task_years:
                temporal_alignment = "task_window"
            else:
                temporal_alignment = "undated"

            contradiction_reasons: list[str] = []
            if diagnosis.seasonal_strength >= 0.45 and any(
                phrase in lower for phrase in self.NON_PERIODIC_PHRASES
            ):
                contradiction_reasons.append("denies_numerically_detected_periodicity")
            if diagnosis.seasonal_strength >= 0.45 and any(
                phrase in lower for phrase in self.LINEAR_PHRASES
            ):
                contradiction_reasons.append("linear_claim_conflicts_with_detected_cycle")
            series_consistency = "conflict" if contradiction_reasons else "not_contradicted"

            query_overlap = len(document_terms & query_terms) / max(math.sqrt(len(query_terms)), 1.0)
            query_score = min(1.0, query_overlap / 2.0)
            causal_score = 1.0 if document_terms & self.CAUSAL_TERMS else 0.0
            temporal_score = {
                "task_window": 1.0,
                "undated": 0.65,
                "outside_task_window": 0.0,
                "published_after_forecast_cutoff": 0.0,
            }[temporal_alignment]
            score = (
                0.30 * float(entity_match)
                + 0.20 * target_score
                + 0.15 * temporal_score
                + 0.20 * query_score
                + 0.15 * causal_score
            )
            event_types = tuple(
                name for name, terms in self.EVENT_TERMS.items() if document_terms & terms
            )
            expected_types = self.QUESTION_EVENT_TYPES.get(action.question_id, set())
            question_alignment = bool(set(event_types) & expected_types)
            reasons: list[str] = []
            if not entity_match:
                reasons.append("entity_mismatch")
            if temporal_alignment in {"outside_task_window", "published_after_forecast_cutoff"}:
                reasons.append(temporal_alignment)
            if target_score == 0:
                reasons.append("no_target_or_event_signal")
            if not question_alignment:
                reasons.append("does_not_answer_current_question")
            reasons.extend(contradiction_reasons)
            accepted = (
                entity_match
                and temporal_alignment not in {"outside_task_window", "published_after_forecast_cutoff"}
                and not contradiction_reasons
                and question_alignment
                and score >= 0.43
            )
            reasons.append("accepted_as_forecast_relevant" if accepted else "rejected_by_verifier")

            evidence = self.synthesis_agent.synthesize(
                task,
                dynamic_diagnosis,
                [item],
                max_claims_per_document=2,
            )
            if accepted and not evidence:
                evidence = [self._fallback_evidence(document, action.query, score)]
            verdicts.append(
                EvidenceVerdict(
                    document_id=document.document_id,
                    accepted=accepted,
                    score=score,
                    entity_match=entity_match,
                    temporal_alignment=temporal_alignment,
                    series_consistency=series_consistency,
                    question_alignment=question_alignment,
                    reasons=tuple(reasons),
                    event_types=event_types,
                    evidence=tuple(evidence if accepted else ()),
                )
            )
        return verdicts


class BeliefUpdaterAgent:
    @staticmethod
    def _update_linguistic_belief(
        state: AgentBeliefState,
        action: QueryAction,
        verdicts: list[EvidenceVerdict],
    ) -> None:
        previous = state.linguistic_beliefs.get(
            action.question_id,
            LinguisticBelief(action.question_id, 0.5),
        )
        probability = min(0.999, max(0.001, previous.evidence_sufficiency))
        log_odds = math.log(probability / (1.0 - probability))
        accepted = [verdict for verdict in verdicts if verdict.accepted]
        if accepted:
            log_odds += sum(0.35 + 1.5 * max(0.0, verdict.score - 0.43) for verdict in accepted)
        else:
            log_odds -= 0.30
        posterior = 1.0 / (1.0 + math.exp(-log_odds))
        evidence_summary = list(previous.evidence_summary)
        counterevidence = list(previous.counterevidence_summary)
        for verdict in accepted:
            for evidence in verdict.evidence:
                if evidence.claim not in evidence_summary:
                    evidence_summary.append(evidence.claim)
        for verdict in verdicts:
            if verdict.accepted:
                continue
            summary = f"{verdict.document_id}: {', '.join(verdict.reasons[:-1])}"
            if summary not in counterevidence:
                counterevidence.append(summary)
        state.linguistic_beliefs[action.question_id] = LinguisticBelief(
            question_id=action.question_id,
            evidence_sufficiency=posterior,
            evidence_summary=tuple(evidence_summary[-8:]),
            counterevidence_summary=tuple(counterevidence[-8:]),
            update_count=previous.update_count + 1,
        )

    def update(
        self,
        state: AgentBeliefState,
        action: QueryAction,
        verdicts: list[EvidenceVerdict],
    ) -> int:
        state.query_history.append(action)
        state.attempt_counts[action.question_id] = state.attempt_counts.get(action.question_id, 0) + 1
        reviewed_for_question = state.reviewed_document_ids_by_question.setdefault(
            action.question_id, []
        )
        accepted_count = 0
        question_supported = False
        existing_evidence = {(item.document_id, item.claim) for item in state.accepted_evidence}

        for verdict in verdicts:
            if verdict.document_id not in state.seen_document_ids:
                state.seen_document_ids.append(verdict.document_id)
            if verdict.document_id not in reviewed_for_question:
                reviewed_for_question.append(verdict.document_id)
            if verdict.accepted:
                question_supported = True
                if verdict.document_id not in state.accepted_document_ids:
                    state.accepted_document_ids.append(verdict.document_id)
                    accepted_count += 1
                for evidence in verdict.evidence:
                    key = (evidence.document_id, evidence.claim)
                    if key not in existing_evidence:
                        state.accepted_evidence.append(evidence)
                        existing_evidence.add(key)
                    for event_type in verdict.event_types or ("general",):
                        bucket = state.beliefs.setdefault(event_type, [])
                        if evidence.claim not in bucket:
                            bucket.append(evidence.claim)
            else:
                # A document that merely answers a different question remains
                # available to later planner actions.  Entity, time, target, or
                # numerical-consistency failures are global rejections.
                soft_question_mismatch = (
                    "does_not_answer_current_question" in verdict.reasons
                    and not any(
                        reason in verdict.reasons
                        for reason in (
                            "entity_mismatch",
                            "outside_task_window",
                            "published_after_forecast_cutoff",
                            "no_target_or_event_signal",
                            "denies_numerically_detected_periodicity",
                            "linear_claim_conflicts_with_detected_cycle",
                        )
                    )
                )
                if not soft_question_mismatch and verdict.document_id not in state.rejected_document_ids:
                    state.rejected_document_ids.append(verdict.document_id)
                state.rejected_reasons[verdict.document_id] = list(verdict.reasons)

        self._update_linguistic_belief(state, action, verdicts)

        if question_supported:
            if action.question_id in state.open_question_ids:
                state.open_question_ids.remove(action.question_id)
            if action.question_id not in state.answered_question_ids:
                state.answered_question_ids.append(action.question_id)
            state.no_progress_steps = 0
        else:
            state.no_progress_steps += 1
            if state.attempt_counts[action.question_id] >= 2:
                if action.question_id in state.open_question_ids:
                    state.open_question_ids.remove(action.question_id)
                if action.question_id not in state.exhausted_question_ids:
                    state.exhausted_question_ids.append(action.question_id)
        return accepted_count


class ForecastCriticAgent:
    @staticmethod
    def relative_change(previous: Forecast | None, current: Forecast) -> float | None:
        if previous is None:
            return None
        scale = max(statistics.fmean(abs(value) for value in previous.mean), 1e-8)
        return statistics.fmean(
            abs(left - right) for left, right in zip(previous.mean, current.mean)
        ) / scale

    def decide(
        self,
        state: AgentBeliefState,
        step: int,
        config: LoopConfig,
        remaining_documents: int,
        forecast_change: float | None,
    ) -> str:
        if not state.open_question_ids:
            return "questions_resolved"
        if remaining_documents <= 0:
            return "document_corpus_exhausted"
        if state.no_progress_steps >= config.max_no_progress:
            return "no_new_verified_evidence"
        if step >= config.max_steps:
            return "max_steps_reached"
        if (
            forecast_change is not None
            and forecast_change <= config.convergence_tolerance
            and len(state.query_history) >= 4
            and len(state.answered_question_ids) >= 2
        ):
            return "forecast_converged"
        return "continue"


class IterativeAgentSystem:
    """Retrieve evidence, then revise an immutable baseline through restricted actions."""

    def __init__(self, config: LoopConfig | None = None) -> None:
        from .agents import TimeSeriesDiagnosisAgent

        self.config = config or LoopConfig()
        self.diagnosis_agent = TimeSeriesDiagnosisAgent()
        self.planner_agent = QueryPlannerAgent()
        self.retrieval_agent = RetrievalAgent()
        self.retrieval_reward_agent = RetrievalProcessRewardAgent()
        self.verifier_agent = EvidenceVerifierAgent()
        self.updater_agent = BeliefUpdaterAgent()
        self.impact_agent = EvidenceToForecastAgent()
        backbone = build_forecast_backbone(
            self.config.backbone,
            timesfm_config=TimesFMBackboneConfig(
                model_id=self.config.timesfm_model_id,
                max_context=self.config.timesfm_max_context,
                max_horizon=self.config.timesfm_max_horizon,
                cache_dir=self.config.timesfm_cache_dir,
                local_files_only=self.config.timesfm_local_files_only,
            ),
            allow_statistical_fallback=self.config.allow_statistical_fallback,
        )
        self.forecast_agent = ProbabilisticForecastAgent(backbone)
        self.context_agent = ImportanceAwareContextAgent(
            total_character_budget=self.config.context_character_budget
        )
        self.macro_agent = MacroReasoningAgent()
        self.micro_agent = MicroReasoningAgent()
        self.revision_utility_agent = RevisionUtilityAgent(
            self.config.revision_utility_threshold
        )
        self.memory = ForecastMemoryBank(self.config.memory_path)
        self.revision_planner = RevisionPlannerAgent(
            self.memory, memory_weight=self.config.memory_weight
        )
        self.workspace_executor = ForecastWorkspaceExecutor()
        self.critic_agent = ForecastCriticAgent()

    def run(self, task: ForecastTask) -> RunResult:
        diagnosis = self.diagnosis_agent.diagnose(task)
        state = AgentBeliefState(
            open_question_ids=self.planner_agent.question_ids,
            linguistic_beliefs={
                question_id: LinguisticBelief(question_id, 0.5)
                for question_id in self.planner_agent.question_ids
            },
        )
        accepted_items: dict[str, RetrievedDocument] = {}
        trace: list[dict[str, object]] = []
        task_seed = self.config.seed + sum(ord(character) for character in task.benchmark_id)
        baseline_values, baseline_method = self.forecast_agent.baseline(task, diagnosis)
        workspace = self.workspace_executor.initialize(
            task, baseline_values, baseline_method
        )
        workspace.macro_outlook = self.macro_agent.analyze(
            task, diagnosis, baseline_method
        )
        final_forecast: Forecast | None = None

        for step in range(1, self.config.max_steps + 1):
            action = self.planner_agent.plan(task, state)
            candidate_pool = self.retrieval_agent.retrieve(
                task,
                diagnosis,
                self.config.documents_per_step * self.config.retrieval_candidate_multiplier,
                query=action.query,
                exclude_ids=(
                    set(state.rejected_document_ids)
                    | set(state.reviewed_document_ids_by_question.get(action.question_id, []))
                ),
            )
            candidates, candidate_assessments = self.retrieval_reward_agent.rank(
                task,
                action,
                candidate_pool,
                state,
                self.config.documents_per_step,
            )
            verdicts = self.verifier_agent.verify(task, diagnosis, action, candidates)
            new_accepted = self.updater_agent.update(state, action, verdicts)
            accepted_ids = {verdict.document_id for verdict in verdicts if verdict.accepted}
            for item in candidates:
                if item.document.document_id in accepted_ids:
                    accepted_items.setdefault(item.document.document_id, item)

            accepted_ranked = [
                RetrievedDocument(document=item.document, score=item.score, rank=index)
                for index, item in enumerate(accepted_items.values(), start=1)
            ]
            compressed_ranked, compression_records = self.context_agent.compress(
                task, diagnosis, accepted_ranked
            )
            workspace.context_compression = compression_records
            state.evidence_impacts = self.impact_agent.translate(
                task,
                diagnosis,
                compressed_ranked,
                state.accepted_evidence,
            )
            workspace.micro_outlook = self.micro_agent.analyze(state.evidence_impacts)
            known_action_ids = {
                record.action.action_id for record in workspace.revision_records
            }
            step_revision_records = []
            step_revision_decisions = []
            for proposal in self.revision_planner.propose(
                task, diagnosis, state.evidence_impacts
            ):
                if proposal.action_id in known_action_ids:
                    continue
                revision_decision = self.revision_utility_agent.evaluate(
                    proposal,
                    workspace.macro_outlook,
                    workspace.micro_outlook,
                    state,
                )
                selected_action = self.revision_utility_agent.fallback(
                    proposal, revision_decision
                )
                if selected_action.action_id in known_action_ids:
                    continue
                workspace.revision_decisions.append(revision_decision)
                step_revision_decisions.append(revision_decision)
                record = self.workspace_executor.apply(workspace, selected_action)
                step_revision_records.append(record)
                known_action_ids.add(selected_action.action_id)

            # Explicit future values in verified documents are represented as
            # point overrides, so every numerical edit still passes through
            # the same restricted action interface.
            context_points = self.forecast_agent._extract_context_points(
                task.future_timestamps, compressed_ranked
            )
            source_ids = tuple(item.document.document_id for item in accepted_ranked)
            for index, timestamp in enumerate(task.future_timestamps):
                if timestamp not in context_points:
                    continue
                proposal = self.revision_planner.point_override(
                    workspace,
                    index,
                    context_points[timestamp],
                    self.config.context_weight,
                    source_ids,
                )
                if proposal.action_id in known_action_ids:
                    continue
                revision_decision = self.revision_utility_agent.evaluate(
                    proposal,
                    workspace.macro_outlook,
                    workspace.micro_outlook,
                    state,
                )
                selected_action = self.revision_utility_agent.fallback(
                    proposal, revision_decision
                )
                if selected_action.action_id in known_action_ids:
                    continue
                workspace.revision_decisions.append(revision_decision)
                step_revision_decisions.append(revision_decision)
                record = self.workspace_executor.apply(workspace, selected_action)
                step_revision_records.append(record)
                known_action_ids.add(selected_action.action_id)

            previous_forecast = state.forecast_history[-1] if state.forecast_history else None
            final_forecast = self.forecast_agent.forecast_from_mean(
                task=task,
                diagnosis=diagnosis,
                mean=tuple(workspace.final_values),
                baseline_mean=workspace.baseline_values,
                baseline_method=workspace.baseline_method,
                num_samples=self.config.num_samples,
                seed=task_seed,
                context_points=context_points,
                impact_adjustments=self.workspace_executor.adjustments(workspace),
                revision_records=workspace.revision_records,
            )
            state.forecast_history.append(final_forecast)
            forecast_change = self.critic_agent.relative_change(previous_forecast, final_forecast)
            remaining_documents = sum(
                1
                for document in task.documents
                if document.document_id not in state.rejected_document_ids
                and any(
                    document.document_id
                    not in state.reviewed_document_ids_by_question.get(question_id, [])
                    for question_id in state.open_question_ids
                )
            )
            decision = self.critic_agent.decide(
                state,
                step,
                self.config,
                remaining_documents,
                forecast_change,
            )
            trace.append(
                {
                    "step": step,
                    "action": asdict(action),
                    "retrieval_candidate_pool_ids": [
                        item.document.document_id for item in candidate_pool
                    ],
                    "retrieval_candidate_assessments": [
                        asdict(item) for item in candidate_assessments
                    ],
                    "candidate_document_ids": [item.document.document_id for item in candidates],
                    "verdicts": [asdict(verdict) for verdict in verdicts],
                    "new_accepted_documents": new_accepted,
                    "accepted_document_ids": list(state.accepted_document_ids),
                    "open_question_ids": list(state.open_question_ids),
                    "evidence_impacts": [
                        asdict(impact) for impact in state.evidence_impacts
                    ],
                    "context_compression": [
                        asdict(item) for item in compression_records
                    ],
                    "macro_outlook": asdict(workspace.macro_outlook),
                    "micro_outlook": asdict(workspace.micro_outlook),
                    "revision_decisions": [
                        asdict(item) for item in step_revision_decisions
                    ],
                    "revision_proposals": [
                        asdict(record.action) for record in step_revision_records
                    ],
                    "revision_results": [
                        asdict(record) for record in step_revision_records
                    ],
                    "forecast": {
                        "baseline_method": final_forecast.baseline_method,
                        "baseline_head": list(workspace.baseline_values[:5]),
                        "mean_head": list(final_forecast.mean[:5]),
                        "mean_tail": list(final_forecast.mean[-5:]),
                        "context_point_count": len(final_forecast.context_points),
                        "impact_adjustments": [
                            asdict(adjustment)
                            for adjustment in final_forecast.impact_adjustments
                        ],
                        "relative_change": forecast_change,
                    },
                    "decision": decision,
                }
            )
            if decision != "continue":
                state.stop_reason = decision
                break

        if final_forecast is None:
            final_forecast = self.forecast_agent.forecast_from_mean(
                task=task,
                diagnosis=diagnosis,
                mean=tuple(workspace.final_values),
                baseline_mean=workspace.baseline_values,
                baseline_method=workspace.baseline_method,
                num_samples=self.config.num_samples,
                seed=task_seed,
                impact_adjustments=self.workspace_executor.adjustments(workspace),
                revision_records=workspace.revision_records,
            )
            state.forecast_history.append(final_forecast)
            state.stop_reason = "no_loop_iteration"
        elif state.stop_reason is None:
            state.stop_reason = "max_steps_reached"

        retrieved = [
            RetrievedDocument(document=item.document, score=item.score, rank=index)
            for index, item in enumerate(accepted_items.values(), start=1)
        ]
        metrics = retrieval_metrics(task, retrieved, state.accepted_evidence)
        metrics.update(forecast_metrics(task, final_forecast))
        if workspace.revision_decisions:
            metrics["revision_accept_rate"] = statistics.fmean(
                float(item.revise) for item in workspace.revision_decisions
            )
            metrics["revision_fallback_rate"] = 1.0 - metrics["revision_accept_rate"]
            metrics["mean_predicted_revision_utility"] = statistics.fmean(
                item.utility_score for item in workspace.revision_decisions
            )
        original_characters = sum(
            item.original_characters for item in workspace.context_compression
        )
        retained_characters = sum(
            item.retained_characters for item in workspace.context_compression
        )
        if original_characters:
            metrics["context_retention_ratio"] = retained_characters / original_characters
        if state.linguistic_beliefs:
            metrics["mean_belief_sufficiency"] = statistics.fmean(
                item.evidence_sufficiency for item in state.linguistic_beliefs.values()
            )
        return RunResult(
            benchmark_id=task.benchmark_id,
            diagnosis=diagnosis,
            retrieved=retrieved,
            evidence=state.accepted_evidence,
            forecast=final_forecast,
            metrics=metrics or None,
            belief_state=state,
            loop_trace=trace,
            workspace=workspace,
        )

    def run_many(self, tasks: list[ForecastTask]) -> list[RunResult]:
        results: list[RunResult] = []
        for task in tasks:
            result = self.run(task)
            results.append(result)
            if self.config.learn_from_public_outcomes and task.future_values is not None:
                self.record_outcome(task, result)
        return results

    def record_outcome(self, task: ForecastTask, result: RunResult):
        """Update memory after resolution; never called inside inference for a task."""
        if result.workspace is None:
            raise ValueError("post-hoc learning requires a forecast workspace")
        return self.memory.record_outcome(task, result.workspace)
