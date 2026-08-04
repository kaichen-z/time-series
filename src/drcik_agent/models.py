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
class ForecastAdjustment:
    source_document_ids: tuple[str, ...]
    adjustment_kind: str
    adjustment_value: float | None
    affected_steps: int
    mean_absolute_change: float
    rationale: str


@dataclass(frozen=True)
class Forecast:
    mean: tuple[float, ...]
    samples: tuple[tuple[float, ...], ...]
    baseline_method: str
    context_points: dict[str, float] = field(default_factory=dict)
    impact_adjustments: tuple[ForecastAdjustment, ...] = ()


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
                "baseline_method": self.forecast.baseline_method,
                "context_points": self.forecast.context_points,
                "impact_adjustments": [
                    asdict(item) for item in self.forecast.impact_adjustments
                ],
                "num_samples": len(self.forecast.samples),
            },
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
        }
