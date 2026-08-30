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
from .acceptance import MetricAcceptanceGate, ScaledPairAcceptanceGate
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
    "ScaledPairAcceptanceGate",
    "JsonArtifactStore",
    "EvolutionOutcome",
    "EvolutionStep",
    "SelfEvolutionEngine",
]
