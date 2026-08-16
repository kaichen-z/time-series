"""Reusable, domain-independent Parent/Child self-evolution primitives."""

from .contracts import (
    AcceptanceGate,
    ArtifactAdapter,
    ArtifactStore,
    EvaluationReport,
    Evaluator,
    EvolutionComponents,
    EvolutionConfig,
    Executor,
    MetricSpec,
    MutationContext,
    Mutator,
)
from .acceptance import MetricAcceptanceGate
from .persistence import JsonArtifactStore
from .controller import EvolutionOutcome, EvolutionStep, SelfEvolutionEngine

__all__ = [
    "AcceptanceGate",
    "ArtifactAdapter",
    "ArtifactStore",
    "EvaluationReport",
    "Evaluator",
    "EvolutionComponents",
    "EvolutionConfig",
    "Executor",
    "MetricSpec",
    "MutationContext",
    "Mutator",
    "MetricAcceptanceGate",
    "JsonArtifactStore",
    "EvolutionOutcome",
    "EvolutionStep",
    "SelfEvolutionEngine",
]
