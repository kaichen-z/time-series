"""Real providers: an LLM writes each method's forecast() code, the sandbox runs it."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from common.llm import LLMClient, parse_json_object
from common.sandbox import run_forecast_code
from common.tracing import TraceEvent, emit

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

    def __init__(
        self,
        llm: LLMClient,
        *,
        timeout_s: float = 10.0,
        transcript_dir: str | Path | None = None,
    ) -> None:
        self.llm = llm
        self.timeout_s = timeout_s
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None

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
            child_index=context.child_index,
            diversity_instruction=context.diversity_instruction,
        )
        code = self._request_code(
            IMPLEMENT_SYSTEM, user, method.method_id, "implement", context.generation
        )
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
            sample_errors=feedback.sample_errors,
            child_index=feedback.child_index,
            diversity_instruction=feedback.diversity_instruction,
        )
        code = self._request_code(
            REVISE_SYSTEM,
            user,
            parent.method_id,
            "revise",
            parent.version + 1,
            failure_categories=list(feedback.failure_categories),
            sample_errors=list(feedback.sample_errors),
        )
        return MethodCandidate(
            method_id=parent.method_id,
            provider=parent.provider,
            implementation_kind=parent.implementation_kind,
            implementation={"code": code},
            version=parent.version + 1,
            parent_version=parent.version,
        )

    def _request_code(
        self,
        system: str,
        user: str,
        method_id: str,
        stage: str,
        generation: int,
        **detail: object,
    ) -> str:
        """Return the code field of one JSON response, tracing the call either way."""
        emit(
            TraceEvent(
                task_id=method_id,
                mode="curation",
                event_type="method_start",
                detail={"stage": stage, "generation": generation, **detail},
            )
        )
        start = time.monotonic()
        text = ""
        try:
            response = self.llm.complete(
                system=system,
                messages=[{"role": "user", "content": user}],
                # Deterministic decoding keeps parent/child dictionary evaluation comparable.
                temperature=0.0,
            )
            text = response.text
            payload = parse_json_object(text)
            code = payload.get("code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("LLM response contained no non-empty 'code' string")
        except Exception as exc:
            # The mutator discards this exception, so record it before it is lost.
            self._write_transcript(method_id, stage, system, user, text)
            emit(
                TraceEvent(
                    task_id=method_id,
                    mode="curation",
                    event_type="method_end",
                    detail={
                        "stage": stage,
                        "generation": generation,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "duration_ms": round((time.monotonic() - start) * 1000, 1),
                        "response_chars": len(text),
                    },
                )
            )
            raise
        self._write_transcript(method_id, stage, system, user, text)
        emit(
            TraceEvent(
                task_id=method_id,
                mode="curation",
                event_type="method_end",
                detail={
                    "stage": stage,
                    "generation": generation,
                    "ok": True,
                    "duration_ms": round((time.monotonic() - start) * 1000, 1),
                    "response_chars": len(text),
                    "code_chars": len(code),
                },
            )
        )
        return code

    def _write_transcript(
        self, method_id: str, stage: str, system: str, user: str, response: str
    ) -> None:
        """Persist exactly what was asked and answered so bad code can be diagnosed."""
        if self.transcript_dir is None or Path(method_id).name != method_id:
            return
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        destination = self.transcript_dir / f"{method_id}.{stage}.md"
        destination.write_text(
            f"# system\n\n{system}\n\n# user\n\n{user}\n\n# response\n\n{response}\n",
            encoding="utf-8",
        )


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
            error = result.error or "sandbox execution failed"
            # The executor keeps only the exception class, so record the real reason here.
            emit(
                TraceEvent(
                    task_id=candidate.method_id,
                    mode="curation",
                    event_type="sandbox_failed",
                    detail={
                        "error": error,
                        "version": candidate.version,
                        "horizon": horizon,
                        "history_length": len(history),
                    },
                )
            )
            # This leaves the method quarantined, and therefore revisable next generation.
            raise RuntimeError(error)
        return result.forecast
