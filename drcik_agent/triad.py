from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .agents import ProbabilisticForecastAgent, RetrievalAgent, TimeSeriesDiagnosisAgent, tokenize
from .backbones import (
    ChronosBackboneConfig,
    ForecastBackbone,
    StatisticalForecastBackbone,
    TimesFMBackboneConfig,
    build_forecast_backbone,
)
from .impacts import EvidenceToForecastAgent
from .loop import EvidenceVerifierAgent
from .metrics import forecast_metrics, retrieval_metrics
from .models import (
    AgentFeedback,
    CandidateAssessment,
    CandidateDecision,
    Diagnosis,
    Evidence,
    EvidenceImpact,
    ForecastCandidate,
    ForecastTask,
    QueryAction,
    RetrievedDocument,
    RunResult,
)


@dataclass(frozen=True)
class TriadConfig:
    """Configuration for the Coding/Retrieval/Decision agent loop."""

    max_rounds: int = 3
    documents_per_round: int = 5
    num_samples: int = 100
    seed: int = 7
    decision_margin: float = 0.12
    feedback_path: str | None = None
    evolution_path: str | None = None
    learn_from_public_outcomes: bool = False
    learning_rate: float = 0.05
    validation_folds: int = 3
    validation_horizon: int = 16
    minimum_validation_history: int = 24
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
    reasoning_agent: str = "rules"
    codex_binary: str = "codex"
    codex_model: str | None = None
    codex_cache_dir: str = "outputs/codex-cache"
    codex_timeout_seconds: int = 300
    codex_max_document_characters: int = 12000
    codex_reasoning_effort: str = "low"

    def __post_init__(self) -> None:
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.documents_per_round <= 0:
            raise ValueError("documents_per_round must be positive")
        if self.num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if self.decision_margin < 0:
            raise ValueError("decision_margin must be non-negative")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.validation_folds <= 0:
            raise ValueError("validation_folds must be positive")
        if self.validation_horizon <= 0:
            raise ValueError("validation_horizon must be positive")
        if self.minimum_validation_history <= 1:
            raise ValueError("minimum_validation_history must be greater than one")
        if self.backbone not in {"chronos", "timesfm", "statistical"}:
            raise ValueError("unsupported backbone")
        if self.reasoning_agent not in {"rules", "codex"}:
            raise ValueError("unsupported reasoning agent")


def _mae(truth: tuple[float, ...], prediction: tuple[float, ...]) -> float:
    return statistics.fmean(abs(actual - predicted) for actual, predicted in zip(truth, prediction))


def _timestamp(value: str, end_of_day: bool = False) -> datetime:
    parsed = datetime.fromisoformat(value.replace("T", " "))
    if len(value) == 10 and end_of_day:
        return parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed


def _affected_indices(task: ForecastTask, impact: EvidenceImpact) -> list[int]:
    if impact.forecast_relation in {"ended_before_forecast", "after_forecast"}:
        return []
    start = _timestamp(impact.start_timestamp) if impact.start_timestamp else None
    end = _timestamp(impact.end_timestamp, end_of_day=True) if impact.end_timestamp else None
    indices = []
    for index, value in enumerate(task.future_timestamps):
        current = _timestamp(value)
        if (start is None or current >= start) and (end is None or current <= end):
            indices.append(index)
    return indices


def _is_executable_quantitative_impact(impact: EvidenceImpact) -> bool:
    """Return whether a textual impact is precise enough to change numbers.

    Temporary effects need both temporal boundaries.  Treating a missing start
    or end as the edge of the forecast silently expands a bounded event and can
    be much more harmful than retaining the numerical baseline.
    """

    if impact.adjustment_kind not in {
        "multiplier",
        "percentage",
        "absolute_additive",
        "standardized_additive",
    }:
        return False
    if impact.adjustment_value is None:
        return False
    if impact.permanence == "temporary" and (
        not impact.start_timestamp or not impact.end_timestamp
    ):
        return False
    return True


@dataclass
class TriadEvolutionPolicy:
    """Small persisted policy updated only after outcomes become available.

    This is deliberately interpretable: it learns priors over candidate tags,
    query vocabulary, and decision tags.  It is the executable precursor to a
    learned neural controller, not a claim that foundation-model weights were
    trained by this repository.
    """

    tasks_seen: int = 0
    coding_tag_bias: dict[str, float] = field(default_factory=dict)
    retrieval_term_weights: dict[str, float] = field(default_factory=dict)
    decision_tag_bias: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | None) -> "TriadEvolutionPolicy":
        if not path:
            return cls()
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return cls()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return cls(
            tasks_seen=int(payload.get("tasks_seen", 0)),
            coding_tag_bias={
                str(key): float(value)
                for key, value in payload.get("coding_tag_bias", {}).items()
            },
            retrieval_term_weights={
                str(key): float(value)
                for key, value in payload.get("retrieval_term_weights", {}).items()
            },
            decision_tag_bias={
                str(key): float(value)
                for key, value in payload.get("decision_tag_bias", {}).items()
            },
        )

    def save(self, path: str | None) -> None:
        if not path:
            return
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _bias(values: dict[str, float], tags: tuple[str, ...]) -> float:
        return sum(values.get(tag, 0.0) for tag in tags)

    def coding_bias(self, tags: tuple[str, ...]) -> float:
        return self._bias(self.coding_tag_bias, tags)

    def decision_bias(self, tags: tuple[str, ...]) -> float:
        return self._bias(self.decision_tag_bias, tags)

    def learned_query_terms(self, limit: int = 16) -> tuple[str, ...]:
        ranked = sorted(
            self.retrieval_term_weights.items(), key=lambda item: (-item[1], item[0])
        )
        return tuple(term for term, weight in ranked[:limit] if weight > 0)

    @staticmethod
    def _update(table: dict[str, float], key: str, change: float) -> None:
        table[key] = max(-0.5, min(0.5, table.get(key, 0.0) + change))


@dataclass(frozen=True)
class CandidateValidation:
    historical_score: float
    mae: float | None
    scaled_mae: float | None
    folds: int


class CodingForecastAgent:
    """Generate executable numerical hypotheses from numbers, then impacts.

    Raw documents are intentionally absent from this interface.  Before
    retrieval, the agent only receives the time series.  After verification it
    receives typed impacts, not an unbounded text context.
    """

    PROGRAM_FAMILIES = ("backbone", "statistical", "history_robust", "level")

    def __init__(
        self,
        backbone: ForecastBackbone,
        policy: TriadEvolutionPolicy,
        *,
        validation_folds: int = 3,
        validation_horizon: int = 16,
        minimum_validation_history: int = 24,
    ) -> None:
        self.backbone = backbone
        self.policy = policy
        self.statistical = StatisticalForecastBackbone()
        self.diagnoser = TimeSeriesDiagnosisAgent()
        self.validation_folds = validation_folds
        self.validation_horizon = validation_horizon
        self.minimum_validation_history = minimum_validation_history

    @staticmethod
    def _candidate(
        candidate_id: str,
        round_index: int,
        program_id: str,
        values: tuple[float, ...],
        assumption: str,
        tags: tuple[str, ...],
        validation: CandidateValidation,
        parent_ids: tuple[str, ...] = (),
        source_ids: tuple[str, ...] = (),
    ) -> ForecastCandidate:
        return ForecastCandidate(
            candidate_id=candidate_id,
            round_index=round_index,
            program_id=program_id,
            values=values,
            assumption=assumption,
            tags=tags,
            historical_score=validation.historical_score,
            validation_mae=validation.mae,
            validation_scaled_mae=validation.scaled_mae,
            validation_folds=validation.folds,
            parent_candidate_ids=parent_ids,
            source_document_ids=source_ids,
        )

    @staticmethod
    def _deduplicate(candidates: list[ForecastCandidate]) -> list[ForecastCandidate]:
        unique: dict[tuple[float, ...], ForecastCandidate] = {}
        for candidate in candidates:
            key = tuple(round(value, 8) for value in candidate.values)
            current = unique.get(key)
            if current is None or candidate.historical_score > current.historical_score:
                unique[key] = candidate
        return list(unique.values())

    @staticmethod
    def _robust_task(task: ForecastTask) -> tuple[ForecastTask, bool]:
        values = task.history_values
        recent_count = min(len(values), max(12, (task.seasonal_period or 6) * 3))
        recent = values[-recent_count:]
        center = statistics.median(recent)
        mad = statistics.median(abs(value - center) for value in recent)
        spread = max(1.4826 * mad, statistics.pstdev(recent) * 0.25, abs(center) * 0.02, 1e-6)
        lower, upper = center - 4.0 * spread, center + 4.0 * spread
        cleaned = tuple(min(upper, max(lower, value)) for value in values)
        changed = any(abs(old - new) > 1e-9 for old, new in zip(values, cleaned))
        return replace(task, history_values=cleaned), changed

    @staticmethod
    def _validation_task(
        task: ForecastTask,
        cutoff: int,
        horizon: int,
        frozen_seasonal_period: int | None,
    ) -> ForecastTask:
        return ForecastTask(
            benchmark_id=f"{task.benchmark_id}:validation:{cutoff}",
            entity_name=task.entity_name,
            target_name=task.target_name,
            target_description=task.target_description,
            frequency=task.frequency,
            prediction_length=horizon,
            # Validate the exact executable program that will be used for the
            # final forecast.  Re-inferring a different period at every fold
            # silently changes candidate identity and makes scores incomparable.
            seasonal_period=(
                frozen_seasonal_period
                if frozen_seasonal_period and frozen_seasonal_period < cutoff
                else None
            ),
            history_timestamps=task.history_timestamps[:cutoff],
            history_values=task.history_values[:cutoff],
            future_timestamps=task.history_timestamps[cutoff : cutoff + horizon],
            future_values=None,
            documents=(),
            gt_evidence=(),
            labels_public=False,
        )

    def _forecast_family(
        self,
        family: str,
        task: ForecastTask,
        diagnosis: Diagnosis,
    ) -> tuple[tuple[float, ...], str]:
        if family == "backbone":
            return self.backbone.forecast(task, diagnosis)
        if family == "statistical":
            return self.statistical.forecast(task, diagnosis)
        if family == "history_robust":
            robust_task, _changed = self._robust_task(task)
            robust_diagnosis = self.diagnoser.diagnose(robust_task)
            values, method = self.backbone.forecast(robust_task, robust_diagnosis)
            return values, f"robust_history:{method}"
        if family == "level":
            recent_count = min(
                len(task.history_values), diagnosis.seasonal_period or 12
            )
            level = statistics.fmean(task.history_values[-recent_count:])
            return tuple(level for _ in task.future_timestamps), "recent_level"
        raise ValueError(f"unsupported candidate family: {family}")

    def _rolling_validation(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
    ) -> dict[str, CandidateValidation]:
        period = diagnosis.seasonal_period or 1
        minimum_history = max(self.minimum_validation_history, 2 * period)
        available = len(task.history_values) - minimum_history
        horizon = min(self.validation_horizon, task.prediction_length)
        if available < horizon:
            neutral = CandidateValidation(0.5, None, None, 0)
            return {family: neutral for family in self.PROGRAM_FAMILIES}

        cutoffs = [
            len(task.history_values) - horizon * fold
            for fold in range(1, self.validation_folds + 1)
            if len(task.history_values) - horizon * fold >= minimum_history
        ]
        raw_errors: dict[str, list[float]] = {
            family: [] for family in self.PROGRAM_FAMILIES
        }
        scaled_errors: dict[str, list[float]] = {
            family: [] for family in self.PROGRAM_FAMILIES
        }
        for cutoff in cutoffs:
            validation_task = self._validation_task(
                task, cutoff, horizon, diagnosis.seasonal_period
            )
            validation_diagnosis = self.diagnoser.diagnose(validation_task)
            truth = task.history_values[cutoff : cutoff + horizon]
            differences = [
                abs(
                    validation_task.history_values[index]
                    - validation_task.history_values[index - 1]
                )
                for index in range(1, len(validation_task.history_values))
            ]
            scale = max(statistics.fmean(differences), 1e-8)
            for family in self.PROGRAM_FAMILIES:
                prediction, _method = self._forecast_family(
                    family, validation_task, validation_diagnosis
                )
                error = _mae(truth, prediction)
                raw_errors[family].append(error)
                scaled_errors[family].append(error / scale)

        validations = {}
        for family in self.PROGRAM_FAMILIES:
            mean_mae = statistics.fmean(raw_errors[family])
            mean_scaled_mae = statistics.fmean(scaled_errors[family])
            validations[family] = CandidateValidation(
                historical_score=1.0 / (1.0 + mean_scaled_mae),
                mae=mean_mae,
                scaled_mae=mean_scaled_mae,
                folds=len(cutoffs),
            )
        return validations

    def initial_candidates(
        self, task: ForecastTask, diagnosis: Diagnosis
    ) -> tuple[list[ForecastCandidate], str]:
        validations = self._rolling_validation(task, diagnosis)
        values, method = self._forecast_family("backbone", task, diagnosis)
        candidates = [
            self._candidate(
                "c_backbone",
                0,
                method,
                values,
                "The numerical backbone captures the future pattern without textual intervention.",
                ("numbers_only", "backbone"),
                validations["backbone"],
            )
        ]

        statistical_values, statistical_method = self._forecast_family(
            "statistical", task, diagnosis
        )
        candidates.append(
            self._candidate(
                "c_statistical",
                0,
                statistical_method,
                statistical_values,
                "A transparent seasonal/trend program is sufficient.",
                ("numbers_only", "statistical"),
                validations["statistical"],
            )
        )

        robust_task, changed = self._robust_task(task)
        if changed:
            robust_values, robust_method = self._forecast_family(
                "history_robust", task, diagnosis
            )
            candidates.append(
                self._candidate(
                    "c_robust_history",
                    0,
                    f"robust_history:{robust_method}",
                    robust_values,
                    "Extreme historical values may be observation artifacts; forecast from a winsorized history.",
                    ("numbers_only", "history_robust"),
                    validations["history_robust"],
                )
            )

        recent_values, recent_method = self._forecast_family("level", task, diagnosis)
        candidates.append(
            self._candidate(
                "c_recent_level",
                0,
                recent_method,
                recent_values,
                "The series has entered a local level regime and should mean-revert around it.",
                ("numbers_only", "level"),
                validations["level"],
            )
        )
        return self._deduplicate(candidates), method

    def expand_candidates(
        self,
        task: ForecastTask,
        candidates: list[ForecastCandidate],
        impacts: list[EvidenceImpact],
        round_index: int,
    ) -> list[ForecastCandidate]:
        output = list(candidates)
        active = [
            impact
            for impact in impacts
            if _is_executable_quantitative_impact(impact)
            and impact.forecast_relation in {"overlaps_forecast", "forecast_relevant_undated"}
        ]
        parents = [
            candidate
            for candidate in candidates
            if candidate.round_index == 0 and "level" not in candidate.tags
        ]
        for impact_index, impact in enumerate(active):
            indices = _affected_indices(task, impact)
            if not indices:
                continue
            for parent in parents:
                values = list(parent.values)
                for index in indices:
                    if impact.adjustment_kind == "multiplier" and impact.adjustment_value is not None:
                        values[index] *= impact.adjustment_value
                    elif impact.adjustment_kind == "percentage" and impact.adjustment_value is not None:
                        values[index] *= 1.0 + impact.adjustment_value
                    elif impact.adjustment_kind == "absolute_additive" and impact.adjustment_value is not None:
                        values[index] += impact.adjustment_value
                    elif impact.adjustment_kind == "standardized_additive" and impact.adjustment_value is not None:
                        scale = max(statistics.pstdev(task.history_values), 1e-6)
                        values[index] += impact.adjustment_value * scale
                output.append(
                    self._candidate(
                        f"c_r{round_index}_impact{impact_index}_{parent.candidate_id}",
                        round_index,
                        f"evidence_{impact.adjustment_kind}:{parent.program_id}",
                        tuple(values),
                        f"Apply the verified {impact.event_type} impact to {parent.candidate_id}.",
                        (
                            *parent.tags,
                            "evidence_adjusted",
                            impact.event_type,
                            impact.adjustment_kind,
                        ),
                        CandidateValidation(
                            parent.historical_score,
                            parent.validation_mae,
                            parent.validation_scaled_mae,
                            parent.validation_folds,
                        ),
                        (parent.candidate_id,),
                        impact.source_document_ids,
                    )
                )
        return self._deduplicate(output)


class RetrievalStreamAgent:
    """Retrieve evidence targeted at disagreements among candidate futures."""

    def __init__(self, policy: TriadEvolutionPolicy) -> None:
        self.policy = policy
        self.retriever = RetrievalAgent()
        self.verifier = EvidenceVerifierAgent()
        self.impact_agent = EvidenceToForecastAgent()

    def plan_query(self, task: ForecastTask, candidates: list[ForecastCandidate]) -> QueryAction:
        values_by_step = list(zip(*(candidate.values for candidate in candidates)))
        disagreement = statistics.fmean(
            max(values) - min(values) for values in values_by_step
        ) if values_by_step else 0.0
        assumptions = " ".join(candidate.assumption for candidate in candidates)
        learned_terms = " ".join(self.policy.learned_query_terms())
        query = (
            f"{task.entity_name} {task.target_name} {task.target_description} "
            f"{task.history_timestamps[0]} {task.future_timestamps[-1]} "
            "anomaly bug resolution patch promotion event magnitude baseline seasonality "
            f"{assumptions} {learned_terms}"
        )
        return QueryAction(
            question_id="candidate_disagreement",
            question="Which entity-specific events distinguish the competing numerical futures?",
            query=query,
            rationale=f"Candidate mean absolute spread is {disagreement:.6g}; retrieve evidence that can eliminate or modify hypotheses.",
        )

    def retrieve(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        candidates: list[ForecastCandidate],
        top_k: int,
        seen_ids: set[str],
    ) -> tuple[QueryAction, list[RetrievedDocument], list[RetrievedDocument], list[Evidence]]:
        action = self.plan_query(task, candidates)
        retrieved = self.retriever.retrieve(
            task,
            diagnosis,
            top_k=top_k,
            query=action.query,
            exclude_ids=seen_ids,
        )
        verdicts = self.verifier.verify(task, diagnosis, action, retrieved)
        verdict_by_id = {verdict.document_id: verdict for verdict in verdicts}
        accepted = [
            item
            for item in retrieved
            if verdict_by_id[item.document.document_id].accepted
        ]
        evidence = [
            item
            for verdict in verdicts
            if verdict.accepted
            for item in verdict.evidence
        ]
        return action, retrieved, accepted, evidence


class DecisionForecastAgent:
    """Cross-check, score, select, or ensemble executable candidates."""

    def __init__(
        self,
        decision_margin: float,
        policy: TriadEvolutionPolicy,
    ) -> None:
        self.decision_margin = decision_margin
        self.policy = policy

    def decide(
        self,
        candidates: list[ForecastCandidate],
        impacts: list[EvidenceImpact],
        round_index: int,
        max_rounds: int,
        documents_remaining: bool,
    ) -> CandidateDecision:
        active_impacts = [
            impact
            for impact in impacts
            if _is_executable_quantitative_impact(impact)
            and impact.forecast_relation in {"overlaps_forecast", "forecast_relevant_undated"}
        ]
        resolved_impacts = [
            impact
            for impact in impacts
            if impact.adjustment_kind == "return_to_baseline"
            or impact.forecast_relation == "ended_before_forecast"
        ]
        embedded_impacts = [
            impact for impact in impacts if impact.adjustment_kind == "already_in_baseline"
        ]
        active_source_ids = {
            source_id for impact in active_impacts for source_id in impact.source_document_ids
        }
        has_grounded_adjusted_candidate = any(
            "evidence_adjusted" in candidate.tags
            and bool(active_source_ids & set(candidate.source_document_ids))
            for candidate in candidates
        )
        assessments = []
        for candidate in candidates:
            reasons = []
            evidence_compatible = True
            if active_impacts and has_grounded_adjusted_candidate:
                evidence_compatible = (
                    "evidence_adjusted" in candidate.tags
                    and bool(active_source_ids & set(candidate.source_document_ids))
                )
                reasons.append(
                    "implements_verified_active_effect"
                    if evidence_compatible
                    else "excluded_because_it_ignores_verified_active_effect"
                )
            if resolved_impacts:
                reasons.append("resolved_event_adds_no_manual_candidate_bonus")
            if embedded_impacts:
                reasons.append("embedded_shift_adds_no_manual_candidate_bonus")
            if not impacts:
                reasons.append("ranked_by_rolling_history_validation")
            learned_score = self.policy.decision_bias(candidate.tags)
            if abs(learned_score) > 1e-12:
                reasons.append("delayed_outcome_policy_bias")
            final_score = (
                candidate.historical_score + learned_score
                if evidence_compatible
                else -1_000_000.0
            )
            assessments.append(
                CandidateAssessment(
                    candidate_id=candidate.candidate_id,
                    historical_score=candidate.historical_score,
                    learned_score=learned_score,
                    evidence_compatible=evidence_compatible,
                    final_score=final_score,
                    reasons=tuple(reasons or ("ranked_by_rolling_history_validation",)),
                )
            )
        assessments.sort(key=lambda item: (-item.final_score, item.candidate_id))
        no_validation_signal = not impacts and all(
            candidate.validation_folds == 0 for candidate in candidates
        )
        if no_validation_signal:
            backbone_id = next(
                candidate.candidate_id
                for candidate in candidates
                if "backbone" in candidate.tags
            )
            top = next(
                assessment
                for assessment in assessments
                if assessment.candidate_id == backbone_id
            )
            runner_up = None
        else:
            top = assessments[0]
            runner_up = assessments[1] if len(assessments) > 1 else None
        margin = top.final_score - runner_up.final_score if runner_up else math.inf

        # Do not convert an arbitrary score-distance threshold into ensemble
        # weights.  A future learned stacker may combine candidates only after
        # its weights improve held-out rolling forecasts; until then, select
        # the best validated compatible executable program.
        selected_ids = (top.candidate_id,)
        selected_weights = (1.0,)
        rationale = (
            "Insufficient history for rolling validation; preserve the configured backbone."
            if no_validation_signal
            else "Select the highest historically validated compatible hypothesis; no unvalidated ensemble."
        )

        should_continue = (
            round_index < max_rounds
            and documents_remaining
            and (not impacts or margin < self.decision_margin)
        )
        needs_candidate = bool(active_impacts) and not has_grounded_adjusted_candidate
        return CandidateDecision(
            selected_candidate_ids=selected_ids,
            selected_weights=selected_weights,
            assessments=tuple(assessments),
            request_more_retrieval=should_continue,
            request_new_candidates=needs_candidate,
            rationale=rationale,
        )

    @staticmethod
    def combine(
        candidates: list[ForecastCandidate], decision: CandidateDecision
    ) -> tuple[float, ...]:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        horizon = len(next(iter(by_id.values())).values)
        return tuple(
            sum(
                weight * by_id[candidate_id].values[step]
                for candidate_id, weight in zip(
                    decision.selected_candidate_ids, decision.selected_weights
                )
            )
            for step in range(horizon)
        )


class ThreeAgentForecastSystem:
    """Executable implementation of the Coding/Retrieval/Decision diagram."""

    def __init__(
        self,
        config: TriadConfig | None = None,
        codex_client: Any | None = None,
    ) -> None:
        self.config = config or TriadConfig()
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
        self.policy = TriadEvolutionPolicy.load(self.config.evolution_path)
        self.diagnosis_agent = TimeSeriesDiagnosisAgent()
        self.codex_client = None
        if self.config.reasoning_agent == "codex":
            from .codex_agents import CodexCLIClient, CodexCLIConfig
            from .codex_triad import (
                CodexCodingForecastAgent,
                CodexDecisionForecastAgent,
                CodexRetrievalStreamAgent,
            )

            self.codex_client = codex_client or CodexCLIClient(
                CodexCLIConfig(
                    binary=self.config.codex_binary,
                    model=self.config.codex_model,
                    cache_dir=self.config.codex_cache_dir,
                    timeout_seconds=self.config.codex_timeout_seconds,
                    max_document_characters=self.config.codex_max_document_characters,
                    reasoning_effort=self.config.codex_reasoning_effort,
                )
            )
            self.coding_agent = CodexCodingForecastAgent(
                backbone,
                self.policy,
                validation_folds=self.config.validation_folds,
                validation_horizon=self.config.validation_horizon,
                minimum_validation_history=self.config.minimum_validation_history,
                client=self.codex_client,
            )
            self.retrieval_agent = CodexRetrievalStreamAgent(
                self.codex_client, self.coding_agent
            )
            self.decision_agent = CodexDecisionForecastAgent(
                self.config.decision_margin,
                self.policy,
                client=self.codex_client,
            )
        else:
            self.coding_agent = CodingForecastAgent(
                backbone,
                self.policy,
                validation_folds=self.config.validation_folds,
                validation_horizon=self.config.validation_horizon,
                minimum_validation_history=self.config.minimum_validation_history,
            )
            self.retrieval_agent = RetrievalStreamAgent(self.policy)
            self.decision_agent = DecisionForecastAgent(
                self.config.decision_margin, self.policy
            )
        self.probabilistic_agent = ProbabilisticForecastAgent(backbone)

    def run(self, task: ForecastTask, task_index: int = 0) -> RunResult:
        diagnosis = self.diagnosis_agent.diagnose(task)
        candidates, baseline_method = self.coding_agent.initial_candidates(task, diagnosis)
        baseline = next(candidate for candidate in candidates if "backbone" in candidate.tags)
        seen_ids: set[str] = set()
        accepted_by_id: dict[str, RetrievedDocument] = {}
        evidence_by_key: dict[tuple[str, str], Evidence] = {}
        impacts: list[EvidenceImpact] = []
        decision: CandidateDecision | None = None
        trace: list[dict[str, object]] = []

        for round_index in range(1, self.config.max_rounds + 1):
            action, retrieved, accepted, new_evidence = self.retrieval_agent.retrieve(
                task,
                diagnosis,
                candidates,
                self.config.documents_per_round,
                seen_ids,
            )
            seen_ids.update(item.document.document_id for item in retrieved)
            accepted_by_id.update(
                (item.document.document_id, item) for item in accepted
            )
            for item in new_evidence:
                evidence_by_key[(item.document_id, item.claim)] = item
            accepted_all = list(accepted_by_id.values())
            evidence_all = list(evidence_by_key.values())
            impacts = self.retrieval_agent.impact_agent.translate(
                task, diagnosis, accepted_all, evidence_all
            )
            candidates = self.coding_agent.expand_candidates(
                task, candidates, impacts, round_index
            )
            documents_remaining = len(seen_ids) < len(task.documents)
            decision = self.decision_agent.decide(
                candidates,
                impacts,
                round_index,
                self.config.max_rounds,
                documents_remaining,
            )
            trace.append(
                {
                    "round": round_index,
                    "agent_backend": self.config.reasoning_agent,
                    "coding_candidates": [asdict(candidate) for candidate in candidates],
                    "retrieval_query": asdict(action),
                    "retrieved_document_ids": [item.document.document_id for item in retrieved],
                    "accepted_document_ids": [item.document.document_id for item in accepted],
                    "new_evidence": [asdict(item) for item in new_evidence],
                    "evidence_impacts": [asdict(item) for item in impacts],
                    "decision": asdict(decision),
                    "codex_stats": (
                        self.codex_client.stats() if self.codex_client else None
                    ),
                }
            )
            if not decision.request_more_retrieval and not decision.request_new_candidates:
                break

        if decision is None:
            decision = self.decision_agent.decide(
                candidates, impacts, self.config.max_rounds, self.config.max_rounds, False
            )
        final_mean = self.decision_agent.combine(candidates, decision)
        forecast = self.probabilistic_agent.forecast_from_mean(
            task=task,
            diagnosis=diagnosis,
            mean=final_mean,
            baseline_mean=baseline.values,
            baseline_method=baseline_method,
            num_samples=self.config.num_samples,
            seed=self.config.seed + task_index,
        )
        accepted_all = list(accepted_by_id.values())
        evidence_all = list(evidence_by_key.values())
        metrics = {
            **retrieval_metrics(task, accepted_all, evidence_all),
            **forecast_metrics(task, forecast),
        }
        if self.codex_client is not None:
            codex_stats = self.codex_client.stats()
            metrics.update(
                {
                    "codex_calls": float(codex_stats["calls"]),
                    "codex_cache_hits": float(codex_stats["cache_hits"]),
                    "codex_failures": float(codex_stats["failures"]),
                    "codex_latency_seconds": float(codex_stats["latency_seconds"]),
                }
            )
        if task.future_values is not None:
            candidate_maes = {
                candidate.candidate_id: _mae(task.future_values, candidate.values)
                for candidate in candidates
            }
            oracle_mae = min(candidate_maes.values())
            selected_mae = _mae(task.future_values, final_mean)
            baseline_mae = candidate_maes[baseline.candidate_id]
            metrics.update(
                {
                    "candidate_oracle_mae": oracle_mae,
                    "candidate_coverage_gain": baseline_mae - oracle_mae,
                    "decision_selection_regret": selected_mae - oracle_mae,
                }
            )

        result = RunResult(
            benchmark_id=task.benchmark_id,
            diagnosis=diagnosis,
            retrieved=accepted_all,
            evidence=evidence_all,
            forecast=forecast,
            metrics=metrics,
            loop_trace=trace,
        )
        if self.config.learn_from_public_outcomes:
            self.record_outcome(task, result)
        return result

    def record_outcome(
        self, task: ForecastTask, result: RunResult
    ) -> tuple[AgentFeedback, ...]:
        """Attribute delayed ground truth without using it during inference."""
        if not task.labels_public or task.future_values is None:
            raise ValueError("agent evolution requires a resolved public task")
        baseline_mae = float((result.metrics or {}).get("baseline_mae", 0.0))
        denominator = max(baseline_mae, 1e-8)
        retrieval_values = [
            float((result.metrics or {}).get("retrieval_precision", 0.0)),
            float((result.metrics or {}).get("distractor_avoidance", 0.0)),
            float((result.metrics or {}).get("evidence_token_recall_proxy", 0.0)),
        ]
        feedback = (
            AgentFeedback(
                agent_name="coding_agent",
                reward=float((result.metrics or {}).get("candidate_coverage_gain", 0.0)) / denominator,
                failure_type=(
                    "candidate_set_missed_future"
                    if float((result.metrics or {}).get("candidate_coverage_gain", 0.0)) <= 0
                    else "none"
                ),
                details={
                    "candidate_oracle_mae": (result.metrics or {}).get("candidate_oracle_mae"),
                    "baseline_mae": baseline_mae,
                },
            ),
            AgentFeedback(
                agent_name="retrieval_agent",
                reward=statistics.fmean(retrieval_values),
                failure_type=(
                    "low_evidence_quality"
                    if statistics.fmean(retrieval_values) < 0.5
                    else "none"
                ),
                details={
                    "retrieval_precision": (result.metrics or {}).get("retrieval_precision"),
                    "distractor_avoidance": (result.metrics or {}).get("distractor_avoidance"),
                    "evidence_token_recall_proxy": (result.metrics or {}).get("evidence_token_recall_proxy"),
                },
            ),
            AgentFeedback(
                agent_name="decision_agent",
                reward=-float((result.metrics or {}).get("decision_selection_regret", 0.0)) / denominator,
                failure_type=(
                    "selected_non_oracle_candidate"
                    if float((result.metrics or {}).get("decision_selection_regret", 0.0)) > 1e-9
                    else "none"
                ),
                details={
                    "selection_regret": (result.metrics or {}).get("decision_selection_regret"),
                    "final_mae": (result.metrics or {}).get("mae"),
                },
            ),
        )
        last_step = result.loop_trace[-1]
        candidate_payloads = last_step["coding_candidates"]
        candidate_maes = {
            str(item["candidate_id"]): _mae(
                task.future_values, tuple(float(value) for value in item["values"])
            )
            for item in candidate_payloads
        }
        oracle_id = min(candidate_maes, key=candidate_maes.get)
        oracle = next(item for item in candidate_payloads if item["candidate_id"] == oracle_id)
        selected_ids = set(last_step["decision"]["selected_candidate_ids"])
        rate = self.config.learning_rate
        for tag in oracle["tags"]:
            self.policy._update(self.policy.coding_tag_bias, str(tag), rate)
            self.policy._update(self.policy.decision_tag_bias, str(tag), rate)
        for item in candidate_payloads:
            if item["candidate_id"] in selected_ids and item["candidate_id"] != oracle_id:
                for tag in item["tags"]:
                    self.policy._update(
                        self.policy.decision_tag_bias, str(tag), -rate
                    )
        evidence_recall = float(
            (result.metrics or {}).get("evidence_token_recall_proxy", 0.0)
        )
        retrieval_update = rate * (1.0 - evidence_recall)
        generic_terms = set(tokenize(task.entity_name + " " + task.target_name))
        gold_terms = set(tokenize(" ".join(task.gt_evidence))) - generic_terms
        for term in sorted(gold_terms)[:32]:
            if not term.replace("-", "").isdigit():
                self.policy._update(
                    self.policy.retrieval_term_weights, term, retrieval_update
                )
        self.policy.tasks_seen += 1
        self.policy.save(self.config.evolution_path)
        if self.config.feedback_path:
            path = Path(self.config.feedback_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for item in feedback:
                    handle.write(
                        json.dumps(
                            {"benchmark_id": task.benchmark_id, **asdict(item)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        return feedback

    def run_many(self, tasks: list[ForecastTask]) -> list[RunResult]:
        return [self.run(task, index) for index, task in enumerate(tasks)]
