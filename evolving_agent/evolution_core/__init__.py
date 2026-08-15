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
]
