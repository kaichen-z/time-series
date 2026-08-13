from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime

from .agents import EvidenceSynthesisAgent, ProbabilisticForecastAgent, RetrievalAgent, tokenize
from .backbones import ChronosBackboneConfig, TimesFMBackboneConfig, build_forecast_backbone
from .codex_agents import (
    CodexCLIClient,
    CodexCLIConfig,
    CodexEvidenceJudgeAgent,
    CodexEvidenceToForecastAgent,
    CodexQueryPlannerAgent,
)
from .context import ForecastUtilityRetriever, ImportanceAwareContextAgent
from .control import ForecastGapControllerAgent
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
    SufficiencyDecision,
)
from .reasoning import MacroReasoningAgent, MicroReasoningAgent, RevisionUtilityAgent
from .regimes import RegimeNormalizationAgent
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
    min_expected_information_gain: float = 0.05
    backbone: str = "chronos"
    chronos_model_id: str = "amazon/chronos-bolt-small"
    chronos_device_map: str = "cpu"
    chronos_max_context: int = 2048
    chronos_max_horizon: int = 1024
    chronos_cache_dir: str | None = None
    chronos_local_files_only: bool = False
    timesfm_model_id: str = "google/timesfm-2.5-200m-pytorch"
    timesfm_max_context: int = 4096
    timesfm_max_horizon: int = 1024
    timesfm_cache_dir: str | None = None
    timesfm_local_files_only: bool = False
    allow_statistical_fallback: bool = False
    oracle_evidence: bool = False
    allow_unvalidated_event_revisions: bool = False
    reasoning_agent: str = "rules"
    codex_stages: tuple[str, ...] = ("query", "verify", "impact")
    codex_binary: str = "codex"
    codex_model: str | None = None
    codex_cache_dir: str = "outputs/codex-cache"
    codex_timeout_seconds: int = 180
    codex_max_document_characters: int = 12000
    codex_reasoning_effort: str = "low"

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
        if not 0.0 <= self.min_expected_information_gain <= 1.0:
            raise ValueError("min_expected_information_gain must be between 0 and 1")
        if self.backbone not in {"chronos", "timesfm", "statistical"}:
            raise ValueError("backbone must be 'chronos', 'timesfm', or 'statistical'")
        if self.reasoning_agent not in {"rules", "codex"}:
            raise ValueError("reasoning_agent must be 'rules' or 'codex'")
        if not set(self.codex_stages) <= {"query", "verify", "impact"}:
            raise ValueError("codex_stages may contain only query, verify, and impact")
        if self.codex_timeout_seconds <= 0:
            raise ValueError("codex_timeout_seconds must be positive")
        if self.codex_max_document_characters <= 0:
            raise ValueError("codex_max_document_characters must be positive")
        if self.codex_reasoning_effort not in {"none", "low", "medium", "high"}:
            raise ValueError("unsupported codex_reasoning_effort")


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
        "historical_regime": {"anomaly", "resolution", "forecast_regime"},
        "resolution_permanence": {"resolution"},
        "external_drivers": {"temporary_event", "external_driver"},
        "event_magnitude": {"temporary_event", "external_driver", "forecast_regime"},
        "forecast_regime": {"forecast_regime"},
        # The three-agent loop asks a broad question induced by disagreement
        # between numerical candidates.  The verifier still enforces entity,
        # time, provenance, and series-consistency checks below.
        "candidate_disagreement": {
            "anomaly",
            "resolution",
            "temporary_event",
            "external_driver",
            "forecast_regime",
        },
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
    def _magnitude(text: str) -> str | None:
        match = re.search(
            r"\b\d+(?:\.\d+)?\s*(?:%|percent|times?|x|units?|points?)\b",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(0)
        word = re.search(r"\b(?:double|twice|triple)\b", text, re.IGNORECASE)
        return word.group(0) if word else None

    @staticmethod
    def _persistence(text: str) -> str:
        lower = text.lower()
        if any(term in lower for term in ("resolved", "fixed", "ended", "returned to baseline")):
            return "resolved"
        if any(term in lower for term in ("permanent", "persist", "structural", "long-term")):
            return "permanent"
        if any(term in lower for term in ("temporary", "promotion", "campaign", "until")):
            return "temporary"
        return "unspecified"

    @staticmethod
    def _event_dates(text: str) -> tuple[str | None, str | None]:
        dates = re.findall(r"\b(?:19|20)\d{2}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?\b", text)
        if not dates:
            return None, None
        ordered = sorted(dict.fromkeys(dates))
        return ordered[0], ordered[-1] if len(ordered) > 1 else None

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
            if accepted:
                event_start, event_end = self._event_dates(document.text)
                evidence = [
                    replace(
                        item,
                        gap_id=action.question_id,
                        entity=task.entity_name,
                        target_variable=task.target_name,
                        publication_date=(
                            publication_date.date().isoformat() if publication_date else None
                        ),
                        event_start=event_start,
                        event_end=event_end,
                        magnitude=self._magnitude(item.claim),
                        persistence=self._persistence(document.text),
                        evidence_quote=item.claim,
                        provenance_valid=True,
                    )
                    for item in evidence
                ]
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


class CodexEvidenceVerifierAgent:
    """Codex semantic judgment plus deterministic provenance safety checks."""

    def __init__(self, client: CodexCLIClient) -> None:
        self.rules = EvidenceVerifierAgent()
        self.codex = CodexEvidenceJudgeAgent(client)

    def verify(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        action: QueryAction,
        retrieved: list[RetrievedDocument],
    ) -> list[EvidenceVerdict]:
        rule_verdicts = self.rules.verify(task, diagnosis, action, retrieved)
        decisions = self.codex.judge(task, diagnosis, action, retrieved)
        if decisions is None:
            return rule_verdicts

        documents = {
            item.document.document_id: item.document.agent_view() for item in retrieved
        }
        output: list[EvidenceVerdict] = []
        for base in rule_verdicts:
            decision = decisions.get(base.document_id)
            if decision is None:
                output.append(base)
                continue
            document = documents[base.document_id]
            codex_accepts = bool(decision.get("accepted"))
            hard_safe = (
                base.entity_match
                and base.temporal_alignment
                not in {"outside_task_window", "published_after_forecast_cutoff"}
                and base.series_consistency != "conflict"
            )
            exact_quotes = tuple(
                str(quote).strip()
                for quote in decision.get("evidence_quotes", [])
                if str(quote).strip() and str(quote).strip() in document.text
            )
            accepted = codex_accepts and hard_safe and bool(exact_quotes or base.evidence)
            evidence: list[Evidence] = []
            if accepted and exact_quotes:
                publication_date = self.rules._publication_date(document.text)
                event_start, event_end = self.rules._event_dates(document.text)
                confidence = min(0.99, max(0.0, float(decision.get("confidence", 0.5))))
                for quote in exact_quotes:
                    extracted = self.rules._fallback_evidence(
                        document, action.query, confidence
                    )
                    evidence.append(
                        replace(
                            extracted,
                            claim=quote[:1200],
                            matched_terms=tuple(
                                sorted(set(tokenize(quote)) & set(tokenize(action.query)))
                            ),
                            confidence=confidence,
                            gap_id=action.question_id,
                            entity=task.entity_name,
                            target_variable=task.target_name,
                            publication_date=(
                                publication_date.date().isoformat()
                                if publication_date
                                else None
                            ),
                            event_start=event_start,
                            event_end=event_end,
                            magnitude=self.rules._magnitude(quote),
                            persistence=self.rules._persistence(document.text),
                            evidence_quote=quote[:1200],
                            provenance_valid=True,
                        )
                    )
            elif accepted:
                evidence.extend(base.evidence)

            reasons = tuple(
                dict.fromkeys(
                    (
                        *base.reasons,
                        "codex_accepted" if codex_accepts else "codex_rejected",
                        str(decision.get("reason", ""))[:500],
                        *(
                            ("does_not_answer_current_question",)
                            if not codex_accepts and hard_safe
                            else ()
                        ),
                        *(() if exact_quotes or not codex_accepts else ("codex_quote_not_grounded",)),
                    )
                )
            )
            event_types = tuple(
                dict.fromkeys(
                    str(value)
                    for value in decision.get("event_types", [])
                    if str(value) in set().union(*self.rules.QUESTION_EVENT_TYPES.values())
                )
            )
            output.append(
                replace(
                    base,
                    accepted=accepted,
                    score=(
                        0.5 * base.score
                        + 0.5 * min(1.0, max(0.0, float(decision.get("confidence", 0.5))))
                    ),
                    question_alignment=codex_accepts,
                    reasons=reasons,
                    event_types=event_types or base.event_types,
                    evidence=tuple(evidence if accepted else ()),
                )
            )
        return output


class BeliefUpdaterAgent:
    @staticmethod
    def _answers_gap(action: QueryAction, verdict: EvidenceVerdict) -> bool:
        """Require the evidence to contain the field requested by the gap."""
        if not verdict.accepted:
            return False
        if action.question_id == "event_magnitude":
            return any(item.magnitude is not None for item in verdict.evidence)
        if action.question_id == "resolution_permanence":
            return any(
                item.persistence in {"resolved", "permanent", "temporary"}
                for item in verdict.evidence
            )
        return True

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
                question_supported = question_supported or self._answers_gap(action, verdict)
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
        sufficiency: SufficiencyDecision,
    ) -> str:
        if sufficiency.sufficient:
            return sufficiency.stop_reason or "evidence_sufficient"
        if remaining_documents <= 0:
            return "document_corpus_exhausted"
        if state.no_progress_steps >= config.max_no_progress:
            return "no_new_verified_evidence"
        if step >= config.max_steps:
            return "max_steps_reached"
        if (
            step >= 2
            and sufficiency.expected_information_gain
            < config.min_expected_information_gain
        ):
            return "low_expected_information_gain"
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
        self.controller_agent = ForecastGapControllerAgent()
        # Compatibility attribute for external callers. Planning is now done
        # by the structured sufficiency-and-gap controller.
        self.planner_agent = self.controller_agent
        self.codex_client: CodexCLIClient | None = None
        self.codex_query_agent: CodexQueryPlannerAgent | None = None
        if self.config.reasoning_agent == "codex":
            self.codex_client = CodexCLIClient(
                CodexCLIConfig(
                    binary=self.config.codex_binary,
                    model=self.config.codex_model,
                    cache_dir=self.config.codex_cache_dir,
                    timeout_seconds=self.config.codex_timeout_seconds,
                    max_document_characters=self.config.codex_max_document_characters,
                    reasoning_effort=self.config.codex_reasoning_effort,
                )
            )
            if "query" in self.config.codex_stages:
                self.codex_query_agent = CodexQueryPlannerAgent(self.codex_client)
        self.retrieval_agent = RetrievalAgent()
        self.retrieval_reward_agent = ForecastUtilityRetriever()
        self.verifier_agent = (
            CodexEvidenceVerifierAgent(self.codex_client)
            if self.codex_client is not None and "verify" in self.config.codex_stages
            else EvidenceVerifierAgent()
        )
        self.updater_agent = BeliefUpdaterAgent()
        rule_impact_agent = EvidenceToForecastAgent()
        self.impact_agent = (
            CodexEvidenceToForecastAgent(self.codex_client, rule_impact_agent)
            if self.codex_client is not None and "impact" in self.config.codex_stages
            else rule_impact_agent
        )
        backbone = build_forecast_backbone(
            self.config.backbone,
            chronos_config=ChronosBackboneConfig(
                model_id=self.config.chronos_model_id,
                device_map=self.config.chronos_device_map,
                max_context=self.config.chronos_max_context,
                max_horizon=self.config.chronos_max_horizon,
                cache_dir=self.config.chronos_cache_dir,
                local_files_only=self.config.chronos_local_files_only,
            ),
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
            self.config.revision_utility_threshold,
            allow_unvalidated_event_revisions=(
                self.config.allow_unvalidated_event_revisions
            ),
        )
        self.memory = ForecastMemoryBank(self.config.memory_path)
        self.revision_planner = RevisionPlannerAgent(
            self.memory, memory_weight=self.config.memory_weight
        )
        self.regime_agent = RegimeNormalizationAgent()
        self.workspace_executor = ForecastWorkspaceExecutor()
        self.critic_agent = ForecastCriticAgent()

    def _apply_regime_projection(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        state: AgentBeliefState,
        workspace,
        known_action_ids: set[str],
    ):
        projection = self.regime_agent.project(
            task,
            diagnosis,
            workspace.baseline_values,
            state.accepted_evidence,
        )
        if projection is None:
            return None, None
        proposal = self.revision_planner.regime_override(
            workspace,
            projection.values,
            projection.source_document_ids,
            projection.confidence,
            projection.rationale,
        )
        if proposal.action_id in known_action_ids:
            return None, None
        decision = self.revision_utility_agent.evaluate(
            proposal,
            workspace.macro_outlook,
            workspace.micro_outlook,
            state,
        )
        selected = self.revision_utility_agent.fallback(proposal, decision)
        workspace.revision_decisions.append(decision)
        record = self.workspace_executor.apply(workspace, selected)
        known_action_ids.add(selected.action_id)
        return decision, record

    def _run_with_oracle_evidence(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        state: AgentBeliefState,
        workspace,
        task_seed: int,
    ) -> RunResult:
        """Public-development diagnostic that bypasses retrieval with GT evidence.

        This is deliberately separate from the deployable retrieval path.  It
        measures the downstream ceiling when the correct evidence has already
        been found; it must never be enabled for hidden-test inference.
        """
        if not task.labels_public:
            raise ValueError("oracle evidence is forbidden for hidden-label tasks")
        if not task.gt_evidence:
            raise ValueError("oracle evidence requires public gt_evidence annotations")

        retrieved = [
            RetrievedDocument(
                document=Document(
                    document_id=f"oracle_gt_{index}",
                    text=text,
                ),
                score=1.0,
                rank=index,
            )
            for index, text in enumerate(task.gt_evidence, start=1)
        ]
        evidence = [
            Evidence(
                document_id=item.document.document_id,
                claim=item.document.text,
                matched_terms=(),
                confidence=1.0,
                effect_direction="unclear",
                effect_window="oracle_public_development",
                gap_id="oracle_public_development",
                entity=task.entity_name,
                target_variable=task.target_name,
                evidence_quote=item.document.text,
                provenance_valid=True,
            )
            for item in retrieved
        ]
        state.accepted_document_ids = [item.document.document_id for item in retrieved]
        state.seen_document_ids = list(state.accepted_document_ids)
        state.accepted_evidence = evidence
        state.answered_question_ids = list(state.forecast_gaps)
        state.open_question_ids = []
        state.linguistic_beliefs = {
            question_id: replace(
                belief,
                evidence_sufficiency=1.0,
                evidence_summary=task.gt_evidence,
                update_count=belief.update_count + 1,
            )
            for question_id, belief in state.linguistic_beliefs.items()
        }

        state.evidence_impacts = self.impact_agent.translate(
            task, diagnosis, retrieved, evidence
        )
        workspace.micro_outlook = self.micro_agent.analyze(state.evidence_impacts)
        revision_records = []
        revision_decisions = []
        known_action_ids = {
            record.action.action_id for record in workspace.revision_records
        }
        for proposal in self.revision_planner.propose(
            task, diagnosis, state.evidence_impacts
        ):
            if proposal.action_id in known_action_ids:
                continue
            decision = self.revision_utility_agent.evaluate(
                proposal, workspace.macro_outlook, workspace.micro_outlook, state
            )
            selected = self.revision_utility_agent.fallback(proposal, decision)
            workspace.revision_decisions.append(decision)
            revision_decisions.append(decision)
            record = self.workspace_executor.apply(workspace, selected)
            revision_records.append(record)
            known_action_ids.add(selected.action_id)

        regime_decision, regime_record = self._apply_regime_projection(
            task, diagnosis, state, workspace, known_action_ids
        )
        if regime_decision is not None and regime_record is not None:
            revision_decisions.append(regime_decision)
            revision_records.append(regime_record)

        context_points = self.forecast_agent._extract_context_points(
            task.future_timestamps, retrieved
        )
        for index, timestamp in enumerate(task.future_timestamps):
            if timestamp not in context_points:
                continue
            source_ids = tuple(
                item.document.document_id
                for item in retrieved
                if timestamp
                in self.forecast_agent._extract_context_points((timestamp,), [item])
            )
            proposal = self.revision_planner.point_override(
                workspace,
                index,
                context_points[timestamp],
                self.config.context_weight,
                source_ids,
            )
            if proposal.action_id in known_action_ids:
                continue
            decision = self.revision_utility_agent.evaluate(
                proposal, workspace.macro_outlook, workspace.micro_outlook, state
            )
            selected = self.revision_utility_agent.fallback(proposal, decision)
            workspace.revision_decisions.append(decision)
            revision_decisions.append(decision)
            record = self.workspace_executor.apply(workspace, selected)
            revision_records.append(record)
            known_action_ids.add(selected.action_id)

        forecast = self.forecast_agent.forecast_from_mean(
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
        state.forecast_history.append(forecast)
        state.stop_reason = "oracle_evidence_diagnostic_complete"

        metrics = forecast_metrics(task, forecast)
        metrics.update(
            {
                "oracle_evidence_mode": 1.0,
                "oracle_evidence_count": float(len(evidence)),
                "oracle_context_point_count": float(len(context_points)),
                "retrieval_turns": 0.0,
                "documents_inspected": 0.0,
            }
        )
        if revision_decisions:
            metrics["revision_accept_rate"] = statistics.fmean(
                float(item.revise) for item in revision_decisions
            )
            metrics["revision_fallback_rate"] = 1.0 - metrics["revision_accept_rate"]
            metrics["mean_predicted_revision_utility"] = statistics.fmean(
                item.utility_score for item in revision_decisions
            )

        trace = [
            {
                "step": 0,
                "mode": "oracle_evidence_public_development_only",
                "warning": "GT evidence bypasses retrieval and is invalid for hidden-test evaluation.",
                "oracle_document_ids": list(state.accepted_document_ids),
                "evidence_impacts": [asdict(item) for item in state.evidence_impacts],
                "revision_decisions": [asdict(item) for item in revision_decisions],
                "revision_results": [asdict(item) for item in revision_records],
                "context_point_count": len(context_points),
                "forecast": {
                    "baseline_method": forecast.baseline_method,
                    "baseline_head": list(workspace.baseline_values[:5]),
                    "mean_head": list(forecast.mean[:5]),
                    "mean_tail": list(forecast.mean[-5:]),
                },
            }
        ]
        return RunResult(
            benchmark_id=task.benchmark_id,
            diagnosis=diagnosis,
            retrieved=retrieved,
            evidence=evidence,
            forecast=forecast,
            metrics=metrics or None,
            belief_state=state,
            loop_trace=trace,
            workspace=workspace,
        )

    def run(self, task: ForecastTask) -> RunResult:
        codex_before = self.codex_client.stats() if self.codex_client is not None else None
        diagnosis = self.diagnosis_agent.diagnose(task)
        forecast_gaps = self.controller_agent.initial_gaps(task, diagnosis)
        state = AgentBeliefState(
            open_question_ids=list(forecast_gaps),
            linguistic_beliefs={
                question_id: LinguisticBelief(question_id, 0.5)
                for question_id in forecast_gaps
            },
            forecast_gaps=forecast_gaps,
        )
        accepted_items: dict[str, RetrievedDocument] = {}
        trace: list[dict[str, object]] = []
        task_seed = self.config.seed + sum(ord(character) for character in task.benchmark_id)
        baseline_values, baseline_method = self.forecast_agent.baseline(task, diagnosis)
        workspace = self.workspace_executor.initialize(
            task, baseline_values, baseline_method
        )
        workspace.macro_outlook = self.macro_agent.analyze(
            task, diagnosis, workspace.baseline_method
        )
        if self.config.oracle_evidence:
            return self._run_with_oracle_evidence(
                task, diagnosis, state, workspace, task_seed
            )
        final_forecast: Forecast | None = None

        for step in range(1, self.config.max_steps + 1):
            sufficiency_before, action = self.controller_agent.decide(task, state)
            if action is not None and self.codex_query_agent is not None:
                action = self.codex_query_agent.refine(
                    task, diagnosis, state, action
                )
                sufficiency_before = replace(
                    sufficiency_before,
                    next_query=action.query,
                    rationale=(
                        sufficiency_before.rationale
                        + " The lexical query was refined by the Codex query planner."
                    ),
                )
            state.sufficiency_history.append(sufficiency_before)
            if action is None:
                state.stop_reason = sufficiency_before.stop_reason or "evidence_sufficient"
                break
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
            self.controller_agent.expand_from_verdicts(
                task, action, verdicts, state
            )
            self.controller_agent.refresh_gap_priorities(state)
            accepted_ids = {verdict.document_id for verdict in verdicts if verdict.accepted}
            for item in candidates:
                if item.document.document_id in accepted_ids:
                    accepted_items.setdefault(item.document.document_id, item)

            accepted_ranked = [
                RetrievedDocument(document=item.document, score=item.score, rank=index)
                for index, item in enumerate(accepted_items.values(), start=1)
            ]
            pinned_quotes: dict[str, tuple[str, ...]] | None = None
            if self.codex_client is not None:
                pinned_quotes = {
                    document_id: tuple(
                        item.evidence_quote
                        for item in state.accepted_evidence
                        if item.document_id == document_id and item.evidence_quote
                    )
                    for document_id in state.accepted_document_ids
                }
            compressed_ranked, compression_records = self.context_agent.compress(
                task, diagnosis, accepted_ranked, pinned_quotes=pinned_quotes
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

            regime_decision, regime_record = self._apply_regime_projection(
                task, diagnosis, state, workspace, known_action_ids
            )
            if regime_decision is not None and regime_record is not None:
                step_revision_decisions.append(regime_decision)
                step_revision_records.append(regime_record)

            # Explicit future values in verified documents are represented as
            # point overrides, so every numerical edit still passes through
            # the same restricted action interface.
            context_points = self.forecast_agent._extract_context_points(
                task.future_timestamps, compressed_ranked
            )
            for index, timestamp in enumerate(task.future_timestamps):
                if timestamp not in context_points:
                    continue
                source_ids = tuple(
                    item.document.document_id
                    for item in compressed_ranked
                    if timestamp
                    in self.forecast_agent._extract_context_points((timestamp,), [item])
                )
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
            sufficiency_after, _next_action = self.controller_agent.decide(task, state)
            decision = self.critic_agent.decide(
                state,
                step,
                self.config,
                remaining_documents,
                forecast_change,
                sufficiency_after,
            )
            trace.append(
                {
                    "step": step,
                    "sufficiency_before": asdict(sufficiency_before),
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
                    "forecast_gaps": {
                        key: asdict(value) for key, value in state.forecast_gaps.items()
                    },
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
                    "sufficiency_after": asdict(sufficiency_after),
                    "decision": decision,
                    "codex": (
                        self.codex_client.stats()
                        if self.codex_client is not None
                        else None
                    ),
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
            if state.stop_reason is None:
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
        total_gaps = len(state.forecast_gaps)
        if total_gaps:
            metrics["gap_coverage"] = len(state.answered_question_ids) / total_gaps
        if state.sufficiency_history:
            metrics["mean_expected_information_gain"] = statistics.fmean(
                item.expected_information_gain for item in state.sufficiency_history
            )
        metrics["retrieval_turns"] = float(len(trace))
        metrics["documents_inspected"] = float(len(state.seen_document_ids))
        if self.codex_client is not None and codex_before is not None:
            codex_after = self.codex_client.stats()
            metrics["codex_calls"] = float(
                int(codex_after["calls"]) - int(codex_before["calls"])
            )
            metrics["codex_cache_hits"] = float(
                int(codex_after["cache_hits"]) - int(codex_before["cache_hits"])
            )
            metrics["codex_failures"] = float(
                int(codex_after["failures"]) - int(codex_before["failures"])
            )
            metrics["codex_latency_seconds"] = float(codex_after["latency_seconds"]) - float(
                codex_before["latency_seconds"]
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
