"""Real providers: an LLM writes each method's forecast() code, the sandbox runs it."""
from __future__ import annotations

from typing import Sequence

from common.llm import LLMClient, parse_json_object
from common.sandbox import run_forecast_code

from .dictionary import MethodCandidate, MethodDefinition
from .prompts import (
    IMPLEMENT_SYSTEM,
    REVISE_SYSTEM,
    render_implement_user,
    render_revise_user,
)
from .providers import ImplementationContext, SanitizedMethodFeedback


SANDBOX_PROVIDER = "sandbox"
IMPLEMENTATION_KIND = "python_code"


class LLMMethodImplementer:
    """Ask an injected LLM to write and repair one named method's forecast() code."""

    def __init__(self, llm: LLMClient, *, timeout_s: float = 10.0) -> None:
        self.llm = llm
        self.timeout_s = timeout_s

    def implement(
        self, method: MethodDefinition, context: ImplementationContext
    ) -> MethodCandidate:
        user = render_implement_user(
            method_id=method.method_id,
            description=method.description,
            assumptions=method.assumptions,
            failure_conditions=method.failure_conditions,
            dictionary_id=context.dictionary_id,
            generation=context.generation,
        )
        code = self._request_code(IMPLEMENT_SYSTEM, user)
        return MethodCandidate(
            method_id=method.method_id,
            provider=SANDBOX_PROVIDER,
            implementation_kind=IMPLEMENTATION_KIND,
            implementation={"code": code},
        )

    def revise(
        self, parent: MethodCandidate, feedback: SanitizedMethodFeedback
    ) -> MethodCandidate:
        previous_code = parent.implementation.get("code", "")
        user = render_revise_user(
            method_id=parent.method_id,
            previous_code=str(previous_code),
            metrics=feedback.metrics,
            failure_categories=feedback.failure_categories,
        )
        code = self._request_code(REVISE_SYSTEM, user)
        return MethodCandidate(
            method_id=parent.method_id,
            provider=parent.provider,
            implementation_kind=parent.implementation_kind,
            implementation={"code": code},
            version=parent.version + 1,
            parent_version=parent.version,
        )

    def _request_code(self, system: str, user: str) -> str:
        """Return the code field of one JSON response, raising when it is absent."""
        response = self.llm.complete(
            system=system,
            messages=[{"role": "user", "content": user}],
            # Deterministic decoding keeps parent/child dictionary evaluation comparable.
            temperature=0.0,
        )
        payload = parse_json_object(response.text)
        code = payload.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("LLM response contained no non-empty 'code' string")
        return code


class SandboxMethodRuntime:
    """Execute an LLM-written candidate through the shared static-checked sandbox."""

    def __init__(self, *, timeout_s: float = 10.0, memory_mb: int = 1024) -> None:
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb

    def supports(self, candidate: MethodCandidate) -> bool:
        code = candidate.implementation.get("code")
        return (
            candidate.provider == SANDBOX_PROVIDER
            and isinstance(code, str)
            and bool(code.strip())
        )

    def forecast(
        self,
        candidate: MethodCandidate,
        history: Sequence[float],
        horizon: int,
        frequency: str,
    ) -> Sequence[float]:
        result = run_forecast_code(
            str(candidate.implementation["code"]),
            [float(value) for value in history],
            horizon,
            frequency,
            timeout_s=self.timeout_s,
            memory_mb=self.memory_mb,
        )
        if not result.ok or result.forecast is None:
            # The executor converts this into an 'invalid' result, which leaves the
            # method quarantined and therefore revisable in the next generation.
            raise RuntimeError(result.error or "sandbox execution failed")
        return result.forecast
