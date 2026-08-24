"""System prompt, response schema, and message construction for the coding-skill agent."""
from __future__ import annotations

import ast

from evolving_loop.data import Task

ACTIONS = ("use_skill", "revise_skill", "write_skill")

SYSTEM_PROMPT = """You are a forecasting agent. You are given only a numeric time series \
history and a horizon to predict - no documents, no side information. You cannot forecast \
directly: you must produce Python code that computes the forecast, and a harness will run it.

The code you WRITE, REVISE, or REUSE must define a function named exactly `forecast`:
    def forecast(history: list[float], horizon: int, frequency: str) -> list[float]
The function name in the code is always the literal word `forecast`, never the skill's name. \
It must return a list of exactly `horizon` finite numbers. `forecast` is the entry point the \
harness calls; you MAY define additional helper functions in the same code and call them from \
`forecast`. Only these imports are allowed: numpy, math, statistics, itertools, functools, \
collections.

A skill is a reusable, named piece of code for a CLASS of series, not for one task: infer any \
parameters from the inputs (history, horizon, frequency) instead of hardcoding values that \
only fit this task.

If a skill library is shown below, each skill comes with usage stats: uses, ok_rate (fraction \
of runs without errors), and mean_smae (lower is better). mean_smae is the Dr-CiK scaled MAE: \
each task's error divided by the mean absolute value of that task's truth over the horizon, so \
it reads as a fraction of the series' own magnitude - 0.1 is an average error a tenth the size \
of the series, and 1.0 is an error as large as the series itself. Choose as follows:
- If an existing skill fits this task, reuse it - prefer skills with strong stats.
- If an existing skill ALMOST fits but has a bug or limitation, revise it: keep its name, \
return improved code. NEVER create a new skill that is a variant of an existing one.
- Write a brand-new skill only when nothing in the library covers this kind of series.

Respond with exactly one JSON object, nothing else:
{"action": "use_skill", "skill_name": "<existing skill name>"}
or
{"action": "revise_skill", "skill_name": "<existing skill name>", "new_code": "<improved code>"}
or
{"action": "write_skill", "new_skill": {"name": "<short_snake_case_name>", \
"description": "<one line: what it does and when to use it>", "code": "<the Python code>"}}
"""

SYSTEM_PROMPT_NO_LIBRARY = """You are a forecasting agent. You are given only a numeric time \
series history and a horizon to predict - no documents, no side information. You cannot \
forecast directly: you must write Python code that computes the forecast, and a harness will \
run it.

The code you write must define a function named exactly `forecast`:
    def forecast(history: list[float], horizon: int, frequency: str) -> list[float]
The function name in the code is always the literal word `forecast`, never the skill's name. \
It must return a list of exactly `horizon` finite numbers. `forecast` is the entry point the \
harness calls; you MAY define additional helper functions in the same code and call them from \
`forecast`. Only these imports are allowed: numpy, math, statistics, itertools, functools, \
collections.

A skill is a reusable, named piece of code for a CLASS of series, not for one task: infer any \
parameters from the inputs (history, horizon, frequency) instead of hardcoding values that \
only fit this task.

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


def _defines_forecast_function(code: str) -> bool:
    """Statically check for a top-level def forecast(...), before ever executing the code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == "forecast" for node in tree.body)


def validate_response(response: dict) -> None:
    """Check a parsed response actually complies with the action it declares, never trust it blindly."""
    action = response.get("action")
    if action not in ACTIONS:
        raise InvalidAgentResponseError(f"action must be one of {ACTIONS}, got {action!r}")

    if action == "use_skill":
        skill_name = response.get("skill_name")
        if not isinstance(skill_name, str) or not skill_name:
            raise InvalidAgentResponseError("use_skill requires a non-empty 'skill_name'")

    if action == "revise_skill":
        skill_name = response.get("skill_name")
        if not isinstance(skill_name, str) or not skill_name:
            raise InvalidAgentResponseError("revise_skill requires a non-empty 'skill_name'")
        new_code = response.get("new_code")
        if not isinstance(new_code, str) or not new_code:
            raise InvalidAgentResponseError("revise_skill requires a non-empty 'new_code'")
        if not _defines_forecast_function(new_code):
            raise InvalidAgentResponseError(
                "revise_skill's new_code must define a top-level def forecast(history, horizon, "
                f"frequency) function - the function must be literally named 'forecast', not '{skill_name}'"
            )

    if action == "write_skill":
        new_skill = response.get("new_skill")
        if not isinstance(new_skill, dict):
            raise InvalidAgentResponseError("write_skill requires a 'new_skill' object")
        for field in ("name", "description", "code"):
            if not isinstance(new_skill.get(field), str) or not new_skill[field]:
                raise InvalidAgentResponseError(f"write_skill's new_skill requires a non-empty '{field}'")
        if not _defines_forecast_function(new_skill["code"]):
            raise InvalidAgentResponseError(
                "write_skill's code must define a top-level "
                "def forecast(history, horizon, frequency) function - the function must be "
                f"literally named 'forecast', not '{new_skill['name']}' or any other descriptive name"
            )
