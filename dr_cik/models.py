"""Data types for the Dr-CiK reproduction pipeline, split so agent code cannot see labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentDocument:
    """A document as an agent is allowed to see it: id and text only."""

    document_id: str
    text: str


@dataclass(frozen=True)
class Document:
    """A document with its benchmark labels, for loaders and scoring only."""

    document_id: str
    text: str
    role: str | None = None
    subtype: str | None = None

    def agent_view(self) -> AgentDocument:
        """Strip labels before any agent code touches this document."""
        return AgentDocument(document_id=self.document_id, text=self.text)


@dataclass(frozen=True)
class TaskView:
    """Everything a deep-research agent is allowed to see about a task."""

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
    documents: tuple[AgentDocument, ...]


@dataclass(frozen=True)
class ForecastTask:
    """A fully labeled Dr-CiK task, for loaders and post-retrieval scoring only."""

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
    gt_evidence: tuple[dict[str, str], ...] = ()
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

    def agent_view(self) -> TaskView:
        """Return the inference-time view with no labels or future values."""
        return TaskView(
            benchmark_id=self.benchmark_id,
            entity_name=self.entity_name,
            target_name=self.target_name,
            target_description=self.target_description,
            frequency=self.frequency,
            prediction_length=self.prediction_length,
            seasonal_period=self.seasonal_period,
            history_timestamps=self.history_timestamps,
            history_values=self.history_values,
            future_timestamps=self.future_timestamps,
            documents=tuple(document.agent_view() for document in self.documents),
        )


@dataclass(frozen=True)
class EvidenceItem:
    """One synthesized evidence claim with the document IDs it cites."""

    claim: str
    source_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class AgentReport:
    """The final output of a deep-research agent for one task."""

    report_markdown: str
    evidence: tuple[EvidenceItem, ...]


@dataclass(frozen=True)
class AgentStep:
    """One audit-trail entry in an agent's run (plan/search/brief/finish/etc)."""

    step_index: int
    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AgentResult:
    """The full result of running a deep-research agent on one task."""

    report: AgentReport
    steps: tuple[AgentStep, ...]
    stop_reason: str
    llm_call_count: int


@dataclass(frozen=True)
class Forecast:
    """A probabilistic forecast: a point mean plus sample trajectories."""

    mean: tuple[float, ...]
    samples: tuple[tuple[float, ...], ...]
    method: str


class DeepResearchAgent(Protocol):
    """Common interface both OpenDR and DRBench agents implement."""

    def run(self, task_view: TaskView) -> AgentResult: ...


@dataclass(frozen=True)
class RunResult:
    """Everything produced for one task: agent output, forecast, and metrics."""

    benchmark_id: str
    agent_name: str
    agent_result: AgentResult
    forecast: Forecast
    metrics: dict[str, float | None] = field(default_factory=dict)
