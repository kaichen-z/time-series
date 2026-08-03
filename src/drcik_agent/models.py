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
class Forecast:
    mean: tuple[float, ...]
    samples: tuple[tuple[float, ...], ...]
    baseline_method: str
    context_points: dict[str, float] = field(default_factory=dict)


@dataclass
class RunResult:
    benchmark_id: str
    diagnosis: Diagnosis
    retrieved: list[RetrievedDocument]
    evidence: list[Evidence]
    forecast: Forecast
    metrics: dict[str, float] | None = None

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
                "num_samples": len(self.forecast.samples),
            },
            "metrics": self.metrics,
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
