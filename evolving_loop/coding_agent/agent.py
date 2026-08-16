"""CodingSkillAgent: decide/retrieve/write a skill, run it safely, and report the outcome."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from evolving_loop.coding_agent.prompts import (
    InvalidAgentResponseError,
    build_system_prompt,
    build_user_message,
    validate_response,
)

from evolving_loop.coding_agent.skill_library import Skill, SkillLibrary
from evolving_loop.data import Task
from evolving_loop.sandbox import SandboxResult, run_forecast_code
from evolving_loop.tracing import TraceEvent, emit
from common.llm import JsonExtractionError, LLMClient, parse_json_object

Mode = Literal["library", "fresh"]

FALLBACK_SKILL_NAME = "__fallback_repeat_last_value__"


@dataclass(frozen=True)
class AgentResult:
    """What one task run produced: the forecast plus which skill (if any) made it."""

    forecast: tuple[float, ...]
    action: str
    skill_name: str | None
    sandbox_result: SandboxResult | None
    error: str | None = None


@dataclass(frozen=True)
class _Attempt:
    """One decide-then-execute attempt; carries an error string when any step failed."""

    ok: bool
    error: str | None
    sandbox_result: SandboxResult | None = None
    action: str | None = None
    skill_name: str | None = None
    description: str | None = None
    code: str | None = None


def _fallback_forecast(task: Task) -> tuple[float, ...]:
    """Deterministic, LLM-free safety net so a bad code draw can never crash the run."""
    last = task.history_values[-1] if task.history_values else 0.0
    return tuple(last for _ in range(task.prediction_length))


class CodingSkillAgent:
    """Given a numeric task, decide/retrieve/write a skill, run it, and report the outcome."""

    def __init__(self, llm: LLMClient, library: SkillLibrary | None, mode: Mode) -> None:
        self.llm = llm
        self.library = library if mode == "library" else None
        self.mode = mode

    def run_task(self, task: Task) -> AgentResult:
        """One task: try once, retry once on any failure with the error appended, else fall back."""
        attempt = self._attempt(task, retry_error=None)
        if not attempt.ok:
            attempt = self._attempt(task, retry_error=attempt.error)

        if not attempt.ok:
            return AgentResult(
                forecast=_fallback_forecast(task),
                action="fallback",
                skill_name=FALLBACK_SKILL_NAME,
                sandbox_result=attempt.sandbox_result,
                error=attempt.error,
            )

        if attempt.action == "write_skill" and self.library is not None:
            self.library.add(
                Skill(
                    skill_id=str(uuid.uuid4()),
                    name=attempt.skill_name,
                    description=attempt.description,
                    code=attempt.code,
                    created_from_task=task.task_id,
                )
            )

        return AgentResult(
            forecast=attempt.sandbox_result.forecast,
            action=attempt.action,
            skill_name=attempt.skill_name,
            sandbox_result=attempt.sandbox_result,
        )

    def _attempt(self, task: Task, retry_error: str | None) -> _Attempt:
        """One LLM turn plus sandbox execution; never raises, failures come back as _Attempt(ok=False)."""

        library_text = self.library.list_for_prompt() if self.library is not None else None
        system = build_system_prompt(has_library=library_text is not None)
        user = build_user_message(task, library_text, retry_error=retry_error)
        response_text = self.llm.complete(system=system, messages=[{"role": "user", "content": user}]).text
        emit(
            TraceEvent(
                task_id=task.task_id,
                mode=self.mode,
                event_type="llm_response",
                detail={"is_retry": retry_error is not None, "response_text": response_text},
            )
        )

        try:
            response = parse_json_object(response_text)
            validate_response(response)
        except (JsonExtractionError, InvalidAgentResponseError) as exc:
            return _Attempt(ok=False, error=str(exc))

        if response["action"] == "use_skill":
            skill = self.library.get(response["skill_name"]) if self.library is not None else None
            if skill is None:
                return _Attempt(ok=False, error=f"referenced unknown skill: {response['skill_name']!r}")
            code, action, skill_name, description = skill.code, "use_skill", skill.name, skill.description
        else:
            new_skill = response["new_skill"]
            code = new_skill["code"]
            action, skill_name, description = "write_skill", new_skill["name"], new_skill["description"]

        sandbox_result = run_forecast_code(code, list(task.history_values), task.prediction_length, task.frequency)
        return _Attempt(
            ok=sandbox_result.ok,
            error=sandbox_result.error,
            sandbox_result=sandbox_result,
            action=action,
            skill_name=skill_name,
            description=description,
            code=code,
        )
