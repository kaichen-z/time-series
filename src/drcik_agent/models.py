from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    role: str | None = None
    subtype: str | None = None

    def agent_view(self) -> "Document":
        """Return the inference-time view without benchmark labels."""
        return Document(document_id=self.document_id, text=self.text)


@dataclass(frozen=True)
class ForecastTask:
    benchmark_id: str
    entity_name: str
    target_name: str
    target_description: str
    frequency: str
    prediction_length: int
    seasonal_period: int | None
    history_timestamps: tuple[str, ...]
    history_values: tuple[float, ...]
    future_timestamps: tuple[str, ...]
    future_values: tuple[float, ...] | None
    documents: tuple[Document, ...]
    gt_evidence: tuple[str, ...] = ()
    labels_public: bool = True

    def __post_init__(self) -> None:
        if not self.history_values:
            raise ValueError("history_values must not be empty")
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if len(self.future_timestamps) != self.prediction_length:
            raise ValueError("future_timestamps must match prediction_length")
        if self.future_values is not None and len(self.future_values) != self.prediction_length:
            raise ValueError("future_values must match prediction_length")


@dataclass(frozen=True)
class Diagnosis:
    trend: str
    slope_per_step: float
    seasonal_period: int | None
    seasonal_strength: float
    residual_scale: float
    information_needs: tuple[str, ...]
    retrieval_query: str


@dataclass(frozen=True)
class QueryAction:
    question_id: str
    question: str
    query: str
    rationale: str


@dataclass(frozen=True)
class EvidenceVerdict:
    document_id: str
    accepted: bool
    score: float
    entity_match: bool
    temporal_alignment: str
    series_consistency: str
    question_alignment: bool
    reasons: tuple[str, ...]
    event_types: tuple[str, ...]
    evidence: tuple["Evidence", ...] = ()


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float
    rank: int


@dataclass(frozen=True)
class Evidence:
    document_id: str
    claim: str
    matched_terms: tuple[str, ...]
    confidence: float
    effect_direction: str
    effect_window: str


@dataclass(frozen=True)
class EvidenceImpact:
    source_document_ids: tuple[str, ...]
    event_type: str
    start_timestamp: str | None
    end_timestamp: str | None
    direction: str
    permanence: str
    forecast_relation: str
    adjustment_kind: str
    adjustment_value: float | None
    confidence: float
    rationale: str


@dataclass(frozen=True)
class LinguisticBelief:
    """BLF-inspired compact belief about whether an information need is resolved."""

    question_id: str
    evidence_sufficiency: float
    evidence_summary: tuple[str, ...] = ()
    counterevidence_summary: tuple[str, ...] = ()
    update_count: int = 0


@dataclass(frozen=True)
class RetrievalCandidateAssessment:
    document_id: str
    bm25_score: float
    utility_score: float
    relevance_score: float
    causal_score: float
    temporal_score: float
    novelty_score: float
    rationale: str


@dataclass(frozen=True)
class ContextCompressionRecord:
    document_id: str
    utility_score: float
    original_characters: int
    retained_characters: int
    allocated_characters: int
    retained_sentences: int


@dataclass(frozen=True)
class MacroOutlook:
    direction: str
    slope_per_step: float
    seasonal_period: int | None
    seasonal_strength: float
    baseline_method: str
    confidence: float
    summary: str


@dataclass(frozen=True)
class MicroEventOutlook:
    event_type: str
    direction: str
    start_timestamp: str | None
    end_timestamp: str | None
    forecast_relation: str
    adjustment_kind: str
    confidence: float
    source_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class MicroOutlook:
    events: tuple[MicroEventOutlook, ...]
    confidence: float
    summary: str


@dataclass(frozen=True)
class RevisionDecision:
    action_id: str
    revise: bool
    utility_score: float
    threshold: float
    reasons: tuple[str, ...]
    fallback_action_id: str | None = None


@dataclass(frozen=True)
class ForecastAdjustment:
    source_document_ids: tuple[str, ...]
    adjustment_kind: str
    adjustment_value: float | None
    affected_steps: int
    mean_absolute_change: float
    rationale: str


@dataclass(frozen=True)
class RevisionAction:
    """A restricted, evidence-backed edit to the forecast horizon."""

    action_id: str
    action_type: str
    start_index: int
    end_index: int
    value: float | None
    values: tuple[float, ...] = ()
    lower_bound: float | None = None
    upper_bound: float | None = None
    source_document_ids: tuple[str, ...] = ()
    event_type: str = "general"
    evidence: str = ""
    confidence: float = 0.0
    rationale: str = ""
    memory_entry_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RevisionRecord:
    action: RevisionAction
    accepted: bool
    reason: str
    affected_steps: int
    mean_absolute_change: float


@dataclass
class ForecastWorkspace:
    """Shared state with an immutable baseline and an editable final forecast."""

    benchmark_id: str
    history_timestamps: tuple[str, ...]
    history_values: tuple[float, ...]
    future_timestamps: tuple[str, ...]
    baseline_method: str
    baseline_values: tuple[float, ...]
    final_values: list[float]
    evidence_proposals: list[RevisionAction] = field(default_factory=list)
    revision_records: list[RevisionRecord] = field(default_factory=list)
    revision_decisions: list[RevisionDecision] = field(default_factory=list)
    context_compression: list[ContextCompressionRecord] = field(default_factory=list)
    macro_outlook: MacroOutlook | None = None
    micro_outlook: MicroOutlook | None = None
    memory_entry_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        horizon = len(self.future_timestamps)
        if len(self.baseline_values) != horizon or len(self.final_values) != horizon:
            raise ValueError("workspace forecasts must match the future horizon")

    def public_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "history": {
                "timestamps": list(self.history_timestamps),
                "values": list(self.history_values),
            },
            "future_timestamps": list(self.future_timestamps),
            "baseline_method": self.baseline_method,
            "y_baseline": list(self.baseline_values),
            "y_final": list(self.final_values),
            "baseline_immutable": True,
            "evidence_proposals": [asdict(item) for item in self.evidence_proposals],
            "revision_records": [asdict(item) for item in self.revision_records],
            "revision_decisions": [asdict(item) for item in self.revision_decisions],
            "context_compression": [asdict(item) for item in self.context_compression],
            "macro_outlook": asdict(self.macro_outlook) if self.macro_outlook else None,
            "micro_outlook": asdict(self.micro_outlook) if self.micro_outlook else None,
            "memory_entry_ids": list(self.memory_entry_ids),
        }


@dataclass(frozen=True)
class ForecastMemoryEntry:
    entry_id: str
    source_task_id: str
    event_type: str
    action_type: str
    proposed_value: float | None
    recommended_value: float | None
    baseline_mae: float
    revised_mae: float
    lesson: str


@dataclass(frozen=True)
class Forecast:
    mean: tuple[float, ...]
    samples: tuple[tuple[float, ...], ...]
    baseline_method: str
    context_points: dict[str, float] = field(default_factory=dict)
    impact_adjustments: tuple[ForecastAdjustment, ...] = ()
    baseline_mean: tuple[float, ...] = ()
    revision_records: tuple[RevisionRecord, ...] = ()


@dataclass
class AgentBeliefState:
    open_question_ids: list[str]
    answered_question_ids: list[str] = field(default_factory=list)
    exhausted_question_ids: list[str] = field(default_factory=list)
    attempt_counts: dict[str, int] = field(default_factory=dict)
    accepted_document_ids: list[str] = field(default_factory=list)
    rejected_document_ids: list[str] = field(default_factory=list)
    seen_document_ids: list[str] = field(default_factory=list)
    reviewed_document_ids_by_question: dict[str, list[str]] = field(default_factory=dict)
    accepted_evidence: list[Evidence] = field(default_factory=list)
    evidence_impacts: list[EvidenceImpact] = field(default_factory=list)
    rejected_reasons: dict[str, list[str]] = field(default_factory=dict)
    beliefs: dict[str, list[str]] = field(default_factory=dict)
    linguistic_beliefs: dict[str, LinguisticBelief] = field(default_factory=dict)
    query_history: list[QueryAction] = field(default_factory=list)
    forecast_history: list[Forecast] = field(default_factory=list)
    no_progress_steps: int = 0
    stop_reason: str | None = None

    def public_dict(self) -> dict[str, Any]:
        """Serialize the inference state without benchmark answer labels."""
        return {
            "open_question_ids": list(self.open_question_ids),
            "answered_question_ids": list(self.answered_question_ids),
            "exhausted_question_ids": list(self.exhausted_question_ids),
            "attempt_counts": dict(self.attempt_counts),
            "accepted_document_ids": list(self.accepted_document_ids),
            "rejected_document_ids": list(self.rejected_document_ids),
            "seen_document_ids": list(self.seen_document_ids),
            "reviewed_document_ids_by_question": {
                key: list(value) for key, value in self.reviewed_document_ids_by_question.items()
            },
            "accepted_evidence": [asdict(item) for item in self.accepted_evidence],
            "evidence_impacts": [asdict(item) for item in self.evidence_impacts],
            "rejected_reasons": {key: list(value) for key, value in self.rejected_reasons.items()},
            "beliefs": {key: list(value) for key, value in self.beliefs.items()},
            "linguistic_beliefs": {
                key: asdict(value) for key, value in self.linguistic_beliefs.items()
            },
            "query_history": [asdict(item) for item in self.query_history],
            "forecast_count": len(self.forecast_history),
            "no_progress_steps": self.no_progress_steps,
            "stop_reason": self.stop_reason,
        }


@dataclass
class RunResult:
    benchmark_id: str
    diagnosis: Diagnosis
    retrieved: list[RetrievedDocument]
    evidence: list[Evidence]
    forecast: Forecast
    metrics: dict[str, float] | None = None
    belief_state: AgentBeliefState | None = None
    loop_trace: list[dict[str, Any]] = field(default_factory=list)
    workspace: ForecastWorkspace | None = None

    def report_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "diagnosis": asdict(self.diagnosis),
            "retrieval": [
                {
                    "document_id": item.document.document_id,
                    "score": item.score,
                    "rank": item.rank,
                }
                for item in self.retrieved
            ],
            "evidence": [asdict(item) for item in self.evidence],
            "forecast": {
                "mean": list(self.forecast.mean),
                "baseline_mean": list(self.forecast.baseline_mean),
                "baseline_method": self.forecast.baseline_method,
                "context_points": self.forecast.context_points,
                "impact_adjustments": [
                    asdict(item) for item in self.forecast.impact_adjustments
                ],
                "revision_records": [
                    asdict(item) for item in self.forecast.revision_records
                ],
                "num_samples": len(self.forecast.samples),
            },
            "forecast_workspace": self.workspace.public_dict() if self.workspace else None,
            "metrics": self.metrics,
            "belief_state": self.belief_state.public_dict() if self.belief_state else None,
            "loop_steps": len(self.loop_trace),
            "stop_reason": self.belief_state.stop_reason if self.belief_state else "one_pass",
        }

    def forecast_submission(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "samples": [list(sample) for sample in self.forecast.samples],
        }

    def research_submission(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "cited_document_ids": [item.document.document_id for item in self.retrieved],
            "evidence": [item.claim for item in self.evidence],
        }

    def trace_submission(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "steps": self.loop_trace,
            "belief_state": self.belief_state.public_dict() if self.belief_state else None,
            "forecast_workspace": self.workspace.public_dict() if self.workspace else None,
        }
