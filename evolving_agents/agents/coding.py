"""The Coding Agent: generates coded hypotheses from numbers alone, then ranks them by hindcast."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field, replace

from dr_cik.llm import JsonExtractionError, LLMClient, parse_json_object
from dr_cik.models import Forecast

from ..harness.hindcast import scaled_error
from ..harness.sandbox import SandboxConfig, run_forecast_code
from ..harness.trace import TraceEvent, emit, emit_llm_response
from ..models import (
    AgentStep,
    Bundle,
    CodingAgentResult,
    CodingCandidate,
    HindcastWindow,
    Hypothesis,
    NumericTaskView,
    SandboxResult,
)
from .common import extract_reasoning, render_code_template_block, render_fewshot_block, render_numeric_brief, render_system_prompt

logger = logging.getLogger(__name__)

HYPOTHESIS_SCHEMA = (
    'Respond with exactly one JSON object: {"assumption": "<one sentence naming the driver you are betting on>", '
    '"code": "<python defining def forecast(history, horizon, frequency) -> list[float]>"}\n'
    "The code must be a complete, self-contained function body; it is executed as written, never edited. "
    "A fourth optional parameter named `backbone` receives a precomputed foundation-model forecast "
    "(a list of `horizon` floats) if you declare it -- omit the parameter if you do not want it."
)
FAILED_PENALTY = 1e6


@dataclass(frozen=True)
class CodingAgentConfig:
    """Generation, ranking, and sandbox limits for the Coding Agent."""

    k_hypotheses: int = 6
    m_keep: int = 2
    temperature: float = 0.8
    max_output_tokens: int = 1400
    sandbox_config: SandboxConfig = field(default_factory=SandboxConfig)


class CodingAgent:
    """Writes K coded hypotheses about a series, executes them, and keeps the M that hindcast best."""

    def __init__(self, llm: LLMClient, bundle: Bundle, config: CodingAgentConfig | None = None) -> None:
        self.llm = llm
        self.bundle = bundle
        base = config or CodingAgentConfig()
        settings = bundle.hyperparameters
        self.config = replace(
            base,
            k_hypotheses=int(settings.get("k_hypotheses", base.k_hypotheses)),
            m_keep=int(settings.get("m_keep", base.m_keep)),
            temperature=float(settings.get("temperature", base.temperature)),
        )

    def _build_prompt(self, view: NumericTaskView, revision_request: str | None = None) -> str:
        """Assemble the user message from the bundle's few-shots, templates, and the series brief."""
        parts = [render_numeric_brief(view)]
        templates = render_code_template_block(self.bundle)
        if templates:
            parts.append(templates)
        fewshots = render_fewshot_block(self.bundle)
        if fewshots:
            parts.append(fewshots)
        if revision_request:
            parts.append(f"Revision requested by the decision agent -- your hypothesis must address it:\n{revision_request}")
        parts.append(HYPOTHESIS_SCHEMA)
        return "\n\n".join(parts)

    def _complete_many(self, prompt: str, count: int) -> list[str]:
        """Sample `count` independent hypotheses, using batched generation when the client supports it."""
        system = render_system_prompt(self.bundle)
        messages = [{"role": "user", "content": prompt}]
        batched = getattr(self.llm, "complete_many", None)
        if batched is not None:
            responses = batched(
                system=system,
                messages=messages,
                count=count,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
            return [item.text for item in responses]
        return [
            self.llm.complete(
                system=system, messages=messages, temperature=self.config.temperature, max_output_tokens=self.config.max_output_tokens
            ).text
            for _ in range(count)
        ]

    def _parse_hypothesis(self, text: str, index: int, task_id: str, generation: int | None) -> Hypothesis | None:
        """Turn one raw completion into a Hypothesis, or None when it is unusable."""
        reasoning, answer = extract_reasoning(text)
        emit_llm_response(
            task_id,
            "coding.generate",
            answer,
            reasoning,
            model_id=getattr(self.llm, "model_id", "?"),
            generation=generation,
            prompt_hash=f"draw{index}",
        )
        try:
            parsed = parse_json_object(answer)
        except JsonExtractionError:
            logger.warning("coding[%s]: hypothesis %d was not valid JSON", task_id, index)
            return None
        code = parsed.get("code")
        assumption = parsed.get("assumption")
        if not isinstance(code, str) or not code.strip() or not isinstance(assumption, str):
            logger.warning("coding[%s]: hypothesis %d missing assumption/code", task_id, index)
            return None
        return Hypothesis(
            hypothesis_id=f"h{index}", assumption_text=assumption.strip(), code=code, reasoning=reasoning
        )

    def _retry_once(self, view: NumericTaskView, hypothesis: Hypothesis, error: str) -> Hypothesis | None:
        """Re-prompt once with the sandbox error appended; returns None if the retry is also unusable."""
        prompt = (
            f"{self._build_prompt(view)}\n\nYour previous code failed to run:\n{error}\n\n"
            f"Previous code:\n{hypothesis.code}\n\nReturn corrected JSON keeping the same assumption."
        )
        response = self.llm.complete(
            system=render_system_prompt(self.bundle),
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        _reasoning, answer = extract_reasoning(response.text)
        try:
            parsed = parse_json_object(answer)
        except JsonExtractionError:
            return None
        code = parsed.get("code")
        return replace(hypothesis, code=code) if isinstance(code, str) and code.strip() else None

    def _execute(self, hypothesis: Hypothesis, view: NumericTaskView, backbone, task_id, generation) -> SandboxResult:
        """Run one hypothesis over the real horizon."""
        return run_forecast_code(
            hypothesis.code,
            view.history_values,
            view.prediction_length,
            view.frequency,
            config=self.config.sandbox_config,
            backbone=backbone,
            task_id=task_id,
            generation=generation,
        )

    def _hindcast(self, hypothesis: Hypothesis, windows: tuple[HindcastWindow, ...], task_id, generation) -> float | None:
        """Replay a hypothesis on past windows and return its mean scaled error, or None if unscoreable."""
        errors: list[float] = []
        for window in windows:
            result = run_forecast_code(
                hypothesis.code,
                window.train_history,
                len(window.held_out_future),
                window.frequency,
                config=self.config.sandbox_config,
                task_id=task_id,
                generation=generation,
            )
            if result.ok and result.forecast is not None:
                errors.append(scaled_error(window.held_out_future, result.forecast))
        return statistics.fmean(errors) if errors else None

    def run(
        self,
        view: NumericTaskView,
        hindcast_windows: tuple[HindcastWindow, ...] = (),
        backbone: tuple[float, ...] | None = None,
        revision_request: str | None = None,
        generation: int | None = None,
    ) -> CodingAgentResult:
        """Generate, execute, hindcast-rank, and return the top m_keep candidates plus every attempt."""
        task_id = view.benchmark_id
        emit(TraceEvent(task_id=task_id, agent="coding", event_type="agent_start", generation=generation, detail={"k": self.config.k_hypotheses}))
        prompt = self._build_prompt(view, revision_request)
        logger.info("coding[%s]: sampling %d hypothesis/es (temp=%.2f)", task_id, self.config.k_hypotheses, self.config.temperature)

        texts = self._complete_many(prompt, self.config.k_hypotheses)
        call_count = len(texts)
        steps: list[AgentStep] = []
        candidates: list[CodingCandidate] = []

        for index, text in enumerate(texts):
            hypothesis = self._parse_hypothesis(text, index, task_id, generation)
            if hypothesis is None:
                steps.append(AgentStep(step_index=index, kind="parse_failure", payload={"raw": text[:500]}))
                continue

            result = self._execute(hypothesis, view, backbone, task_id, generation)
            if not result.ok:
                logger.info("coding[%s]: hypothesis %d failed (%s), retrying once", task_id, index, result.error)
                retried = self._retry_once(view, hypothesis, result.error or "unknown error")
                call_count += 1
                if retried is not None:
                    hypothesis = retried
                    result = self._execute(hypothesis, view, backbone, task_id, generation)

            steps.append(
                AgentStep(
                    step_index=index,
                    kind="hypothesis",
                    payload={"assumption": hypothesis.assumption_text, "ok": result.ok, "error": result.error},
                )
            )
            forecast = (
                Forecast(mean=result.forecast, samples=(result.forecast,), method=f"coding:{hypothesis.hypothesis_id}")
                if result.ok and result.forecast is not None
                else None
            )
            score = self._hindcast(hypothesis, hindcast_windows, task_id, generation) if forecast is not None else None
            candidates.append(
                CodingCandidate(hypothesis=hypothesis, sandbox_result=result, forecast=forecast, hindcast_score=score)
            )

        usable = [candidate for candidate in candidates if candidate.forecast is not None]
        # A candidate that ran but could not be hindcast still beats one that never ran at all.
        usable.sort(key=lambda candidate: candidate.hindcast_score if candidate.hindcast_score is not None else FAILED_PENALTY)
        ranked = tuple(replace(candidate, rank=position) for position, candidate in enumerate(usable[: self.config.m_keep], start=1))

        logger.info("coding[%s]: %d/%d hypothesis/es ran, keeping %d", task_id, len(usable), self.config.k_hypotheses, len(ranked))
        emit(
            TraceEvent(
                task_id=task_id,
                agent="coding",
                event_type="agent_end",
                generation=generation,
                detail={"generated": len(texts), "usable": len(usable), "kept": len(ranked)},
            )
        )
        return CodingAgentResult(
            candidates=ranked, all_candidates=tuple(candidates), steps=tuple(steps), llm_call_count=call_count
        )
