"""Dataclasses for bundles, agent views, and the three agents' inputs/outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dr_cik.models import EvidenceItem, Forecast, TaskView


@dataclass(frozen=True)
class FewshotExample:
    """One input/output demonstration spliced into an agent's prompt."""

    input: str
    output: str


@dataclass(frozen=True)
class Bundle:
    """A versioned agent definition: the thing evolution actually mutates."""

    bundle_id: str
    agent: str
    version: str
    parent: str | None
    system_prompt: str
    fewshot_examples: tuple[FewshotExample, ...] = ()
    code_templates: dict[str, str] = field(default_factory=dict)
    notes_from_evolver: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True)
class BundleTriple:
    """One Loop-C individual: the three agents' bundles evaluated together."""

    coding: Bundle
    retrieval: Bundle
    decision: Bundle


@dataclass(frozen=True)
class NumericTaskView:
    """Everything the Coding Agent may see; has no documents field by construction."""

    benchmark_id: str
    history_values: tuple[float, ...]
    prediction_length: int
    frequency: str
    seasonal_period: int | None


def to_numeric_view(view: TaskView) -> NumericTaskView:
    """Drop every textual/document field, leaving only what the Coding Agent may see."""
    return NumericTaskView(
        benchmark_id=view.benchmark_id,
        history_values=view.history_values,
        prediction_length=view.prediction_length,
        frequency=view.frequency,
        seasonal_period=view.seasonal_period,
    )


@dataclass(frozen=True)
class HindcastWindow:
    """A past slice of a series used to score a hypothesis without touching ground truth."""

    train_history: tuple[float, ...]
    held_out_future: tuple[float, ...]
    frequency: str


@dataclass(frozen=True)
class SandboxResult:
    """The outcome of executing one piece of agent-written code."""

    ok: bool
    forecast: tuple[float, ...] | None
    error: str | None
    duration_ms: float
    code_hash: str


@dataclass(frozen=True)
class Hypothesis:
    """One generated assumption plus the code meant to realize it."""

    hypothesis_id: str
    assumption_text: str
    code: str
    reasoning: str | None = None


@dataclass(frozen=True)
class CodingCandidate:
    """A hypothesis after execution and hindcast ranking."""

    hypothesis: Hypothesis
    sandbox_result: SandboxResult
    forecast: Forecast | None
    hindcast_score: float | None
    rank: int | None = None


@dataclass(frozen=True)
class AgentStep:
    """One recorded step in an agent's trace."""

    step_index: int
    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class CodingAgentResult:
    """Ranked candidates plus every attempt, for audit and mutation failure traces."""

    candidates: tuple[CodingCandidate, ...]
    all_candidates: tuple[CodingCandidate, ...]
    steps: tuple[AgentStep, ...] = ()
    llm_call_count: int = 0


@dataclass(frozen=True)
class RetrievalEvidenceOutput:
    """Evidence the Retrieval Agent kept, plus what it considered."""

    kept: tuple[EvidenceItem, ...]
    considered_doc_ids: tuple[str, ...]
    steps: tuple[AgentStep, ...] = ()
    llm_call_count: int = 0


@dataclass(frozen=True)
class DecisionAuditEntry:
    """Why one candidate was kept or discarded."""

    candidate_id: str
    kept: bool
    reason: str
    contradicting_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionOutput:
    """The final forecast plus the audit trail explaining how it was chosen."""

    final_forecast: Forecast
    weights: dict[str, float]
    audit: tuple[DecisionAuditEntry, ...] = ()
    revision_request: str | None = None
    steps: tuple[AgentStep, ...] = ()
    llm_call_count: int = 0


@dataclass(frozen=True)
class TaskTrace:
    """Everything the three agents produced for one task."""

    benchmark_id: str
    coding_result: CodingAgentResult
    retrieval_result: RetrievalEvidenceOutput
    decision_result: DecisionOutput
    forecast: Forecast
    metrics: dict[str, float | None]
    revised: bool = False
