"""Dependency-injection contracts for externally supplied method providers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .dictionary import MethodCandidate, MethodDefinition


class RuntimeUnavailableError(RuntimeError):
    """A method runtime cannot execute because its provider resources are unavailable."""


@dataclass(frozen=True)
class ImplementationContext:
    dictionary_id: str
    generation: int
    provider_config: Mapping[str, object] = field(default_factory=dict)
    child_index: int = 1
    diversity_instruction: str = ""


@dataclass(frozen=True)
class SanitizedMethodFeedback:
    method_id: str
    metrics: Mapping[str, float]
    failure_categories: tuple[str, ...] = ()
    sample_errors: tuple[str, ...] = ()
    child_index: int = 1
    diversity_instruction: str = ""


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

    def close(self) -> None:
        """Close every distinct registered runtime once (e.g. worker subprocesses)."""
        seen: set[int] = set()
        errors: list[BaseException] = []
        for runtime in self._runtimes.values():
            identity = id(runtime)
            if identity in seen:
                continue
            seen.add(identity)
            close = getattr(runtime, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as error:  # noqa: BLE001 - report after closing the rest
                errors.append(error)
        if errors:
            raise errors[0]

    def __enter__(self) -> "RuntimeRegistry":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
