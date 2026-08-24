"""Typed contracts shared by self-evolution adapters."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, Mapping, Protocol, Sequence, TypeVar


ArtifactT = TypeVar("ArtifactT")
ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class MetricSpec:
    """Primary metric and ordering used for screening and acceptance."""

    name: str
    objective: Literal["minimize", "maximize"] = "minimize"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("metric name must not be empty")
        if self.objective not in ("minimize", "maximize"):
            raise ValueError("objective must be 'minimize' or 'maximize'")

    def better(self, candidate: float, parent: float, margin: float = 0.0) -> bool:
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if not math.isfinite(candidate) or not math.isfinite(parent):
            raise ValueError("metric values must be finite")
        if self.objective == "minimize":
            return candidate < parent - margin
        return candidate > parent + margin


@dataclass(frozen=True)
class EvolutionConfig:
    """Domain-independent search budget and acceptance configuration."""

    generations: int = 1
    children_per_generation: int = 2
    seed: int = 20260816
    metric: MetricSpec = field(default_factory=lambda: MetricSpec("smae"))
    acceptance_margin: float = 0.0
    resume: bool = True

    def __post_init__(self) -> None:
        positive = {
            "generations": self.generations,
            "children_per_generation": self.children_per_generation,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.acceptance_margin < 0:
            raise ValueError("acceptance_margin must be non-negative")


@dataclass(frozen=True)
class EvaluationReport:
    """Trusted aggregate for one frozen artifact on one split."""

    artifact_id: str
    split: str
    metrics: Mapping[str, float]
    item_count: int
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MutationContext:
    """Sanitized feedback available to a mutator."""

    generation: int
    parent_train_report: EvaluationReport
    failure_traces: tuple[Mapping[str, object], ...] = ()
    diversity_instruction: str = ""


class ArtifactAdapter(Protocol[ArtifactT]):
    def validate(self, artifact: ArtifactT) -> None: ...

    def artifact_id(self, artifact: ArtifactT) -> str: ...

    def to_payload(self, artifact: ArtifactT) -> dict[str, object]: ...

    def from_payload(self, payload: Mapping[str, object]) -> ArtifactT: ...

    def apply_train_report(
        self, artifact: ArtifactT, report: EvaluationReport
    ) -> ArtifactT: ...


class Mutator(Protocol[ArtifactT]):
    def propose(
        self, parent: ArtifactT, context: MutationContext, count: int
    ) -> Sequence[ArtifactT]: ...


class Executor(Protocol[ArtifactT, ItemT, ResultT]):
    def execute(
        self, artifact: ArtifactT, items: Sequence[ItemT], split: str
    ) -> Sequence[ResultT]: ...


class Evaluator(Protocol[ResultT]):
    def evaluate(
        self, artifact_id: str, results: Sequence[ResultT], split: str
    ) -> EvaluationReport: ...


class AcceptanceGate(Protocol):
    def accept(
        self, parent_report: EvaluationReport, child_report: EvaluationReport
    ) -> bool: ...


class ArtifactStore(Protocol):
    def save_artifact(self, name: str, payload: Mapping[str, object]) -> Path: ...

    def save_checkpoint(self, payload: Mapping[str, object]) -> Path: ...

    def load_checkpoint(self) -> dict[str, object] | None: ...

    def append_trace(self, payload: Mapping[str, object]) -> None: ...


@dataclass(frozen=True)
class EvolutionComponents(Generic[ArtifactT, ItemT, ResultT]):
    artifact_adapter: ArtifactAdapter[ArtifactT]
    mutator: Mutator[ArtifactT]
    executor: Executor[ArtifactT, ItemT, ResultT]
    evaluator: Evaluator[ResultT]
    acceptance_gate: AcceptanceGate
    store: ArtifactStore
