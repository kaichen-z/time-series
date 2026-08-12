"""System prompt, response schema, and message construction for the coding-skill agent."""
from __future__ import annotations

from evolving_agent.data import Task

ACTIONS = ("use_skill", "write_skill")

SYSTEM_PROMPT = """You are a forecasting agent. You are given only a numeric time series \
history and a horizon to predict - no documents, no side information. You cannot forecast \
directly: you must produce Python code that computes the forecast, and a harness will run it.

The code you write or reuse must define exactly one function:
    def forecast(history: list[float], horizon: int, frequency: str) -> list[float]
It must return a list of exactly `horizon` finite numbers. Only these imports are allowed: \
numpy, math, statistics, itertools, functools, collections.

If a skill library is shown below, you may reuse an existing skill instead of writing new code \
whenever one of them fits this task. Otherwise, write a new skill: a short, reusable, named \
piece of code with a one-line description of what it does and when to use it - not throwaway \
code for this task alone.

Respond with exactly one JSON object, nothing else:
{"action": "use_skill", "skill_name": "<existing skill name>"}
or
{"action": "write_skill", "new_skill": {"name": "<short_snake_case_name>", \
"description": "<one line: what it does and when to use it>", "code": "<the Python code>"}}
"""

SYSTEM_PROMPT_NO_LIBRARY = """You are a forecasting agent. You are given only a numeric time \
series history and a horizon to predict - no documents, no side information. You cannot \
forecast directly: you must write Python code that computes the forecast, and a harness will \
run it.

The code you write must define exactly one function:
    def forecast(history: list[float], horizon: int, frequency: str) -> list[float]
It must return a list of exactly `horizon` finite numbers. Only these imports are allowed: \
numpy, math, statistics, itertools, functools, collections.

Respond with exactly one JSON object, nothing else:
{"action": "write_skill", "new_skill": {"name": "<short_snake_case_name>", \
"description": "<one line: what it does and when to use it>", "code": "<the Python code>"}}
"""


def build_system_prompt(has_library: bool) -> str:
    """Pick the library-aware or library-free system prompt (fresh mode never mentions reuse)."""
    return SYSTEM_PROMPT if has_library else SYSTEM_PROMPT_NO_LIBRARY


def build_user_message(task: Task, library_text: str | None, retry_error: str | None = None) -> str:
    """Describe the task numerically, optionally the skill library, optionally a prior failure to fix."""
    parts = [
        f"history_values: {list(task.history_values)}",
        f"horizon: {task.prediction_length}",
        f"frequency: {task.frequency}",
    ]
    if library_text is not None:
        parts.append(f"\nAvailable skills:\n{library_text}")
    if retry_error is not None:
        parts.append(f"\nYour previous code failed with this error - fix it and try again:\n{retry_error}")
    return "\n".join(parts)


class InvalidAgentResponseError(ValueError):
    """Raised when a parsed response doesn't match the required action schema."""


def validate_response(response: dict) -> None:
    """Check a parsed response actually complies with the action it declares, never trust it blindly."""
    action = response.get("action")
    if action not in ACTIONS:
        raise InvalidAgentResponseError(f"action must be one of {ACTIONS}, got {action!r}")

    if action == "use_skill":
        skill_name = response.get("skill_name")
        if not isinstance(skill_name, str) or not skill_name:
            raise InvalidAgentResponseError("use_skill requires a non-empty 'skill_name'")

    if action == "write_skill":
        new_skill = response.get("new_skill")
        if not isinstance(new_skill, dict):
            raise InvalidAgentResponseError("write_skill requires a 'new_skill' object")
        for field in ("name", "description", "code"):
            if not isinstance(new_skill.get(field), str) or not new_skill[field]:
                raise InvalidAgentResponseError(f"write_skill's new_skill requires a non-empty '{field}'")
