"""Deterministic providers used only to test and smoke the framework."""
from __future__ import annotations

from typing import Sequence

from .dictionary import MethodCandidate, MethodDefinition
from .providers import ImplementationContext, SanitizedMethodFeedback


class FixtureMethodImplementer:
    """Materialize opaque implementation payloads supplied by a fixture dictionary."""

    def implement(
        self, method: MethodDefinition, context: ImplementationContext
    ) -> MethodCandidate:
        return MethodCandidate(
            method_id=method.method_id,
            provider="fake",
            implementation_kind="fixture",
            implementation=dict(method.implementation_spec),
        )

    def revise(
        self, parent: MethodCandidate, feedback: SanitizedMethodFeedback
    ) -> MethodCandidate:
        revised = parent.implementation.get("revision")
        implementation = dict(revised) if isinstance(revised, dict) else dict(parent.implementation)
        return MethodCandidate(
            method_id=parent.method_id,
            provider=parent.provider,
            implementation_kind=parent.implementation_kind,
            implementation=implementation,
            version=parent.version + 1,
            parent_version=parent.version,
        )


class FixtureMethodRuntime:
    def supports(self, candidate: MethodCandidate) -> bool:
        return candidate.provider == "fake" and "prediction" in candidate.implementation

    def forecast(
        self,
        candidate: MethodCandidate,
        history: Sequence[float],
        horizon: int,
        frequency: str,
    ) -> Sequence[float]:
        prediction = float(candidate.implementation["prediction"])
        return [prediction] * horizon
