"""Dependency-injection contracts for externally supplied method providers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .dictionary import MethodCandidate, MethodDefinition


@dataclass(frozen=True)
class ImplementationContext:
    dictionary_id: str
    generation: int
    provider_config: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SanitizedMethodFeedback:
    method_id: str
    metrics: Mapping[str, float]
    failure_categories: tuple[str, ...] = ()
    sample_errors: tuple[str, ...] = ()


class MethodImplementer(Protocol):
    def implement(
        self, method: MethodDefinition, context: ImplementationContext
    ) -> MethodCandidate: ...

    def revise(
        self, parent: MethodCandidate, feedback: SanitizedMethodFeedback
    ) -> MethodCandidate: ...


class MethodRuntime(Protocol):
    def supports(self, candidate: MethodCandidate) -> bool: ...

    def forecast(
        self,
        candidate: MethodCandidate,
        history: Sequence[float],
        horizon: int,
        frequency: str,
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class RuntimeResolution:
    available: bool
    runtime: MethodRuntime | None
    reason: str = ""


class RuntimeRegistry:
    def __init__(self, runtimes: Mapping[str, MethodRuntime] | None = None) -> None:
        self._runtimes = dict(runtimes or {})

    def resolve(self, candidate: MethodCandidate) -> RuntimeResolution:
        runtime = self._runtimes.get(candidate.provider)
        if runtime is None:
            return RuntimeResolution(
                available=False,
                runtime=None,
                reason=f"runtime provider {candidate.provider!r} is not registered",
            )
        if not runtime.supports(candidate):
            return RuntimeResolution(
                available=False,
                runtime=None,
                reason=f"runtime provider {candidate.provider!r} does not support candidate",
            )
        return RuntimeResolution(available=True, runtime=runtime)
