"""Prompts asking one LLM to materialize and revise a single named statistical method."""
from __future__ import annotations

import json
from typing import Mapping, Sequence


ALLOWED_IMPORTS_TEXT = (
    "numpy, math, statistics, itertools, functools, collections, and statsmodels"
)

CONTRACT_TEXT = f"""Every implementation must define exactly:
    def forecast(history: list[float], horizon: int, frequency: str) -> list[float]
It must return exactly horizon finite numbers. Allowed imports are {ALLOWED_IMPORTS_TEXT}.
Do not access files or the network, use randomness, call eval/exec, or hard-code any series.
Handle short histories and degenerate inputs by falling back inside the function rather than
raising."""

IMPLEMENT_SYSTEM = f"""You are the numbers-only Numerical Agent in a time-series harness.
You implement one specific, named classical statistical forecasting method that is given to you.
Implement the method as described; do not substitute a different method you consider stronger.
You may see only historical numbers, horizon, and frequency. You must not request or infer
documents, retrieved evidence, ground-truth evidence, or future values.

{CONTRACT_TEXT}

Return exactly one JSON object:
{{"code": "def forecast(history, horizon, frequency): ..."}}
"""

REVISE_SYSTEM = f"""You are the numbers-only Numerical Agent repairing one failing method
implementation. You receive the previous implementation plus sanitized aggregate diagnostics:
error metrics, failure categories, and up to 3 sample error messages describing how your own
code failed (e.g. an exception raised while it ran). Those messages describe only your code's
own behavior; they never contain future values, per-item labels, or textual context.

Rewrite the implementation so it still implements the same named method, and no longer fails in
the reported way. Use the sample error messages to diagnose the actual bug, not just to guess
again. Keep the method's identity; do not replace it with a different method.

{CONTRACT_TEXT}

Return exactly one JSON object:
{{"code": "def forecast(history, horizon, frequency): ..."}}
"""


def render_implement_user(
    *,
    method_id: str,
    description: str,
    assumptions: Sequence[str],
    failure_conditions: Sequence[str],
    dictionary_id: str,
    generation: int,
    child_index: int = 1,
    diversity_instruction: str = "",
) -> str:
    """Build the user message asking for one named method's implementation."""
    return json.dumps(
        {
            "dictionary_id": dictionary_id,
            "generation": generation,
            "child_index": child_index,
            "diversity_instruction": diversity_instruction,
            "method_id": method_id,
            "description": description,
            "assumptions": list(assumptions),
            "failure_conditions": list(failure_conditions),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def render_revise_user(
    *,
    method_id: str,
    previous_code: str,
    metrics: Mapping[str, float],
    failure_categories: Sequence[str],
    sample_errors: Sequence[str] = (),
    child_index: int = 1,
    diversity_instruction: str = "",
) -> str:
    """Build the user message asking for a repaired implementation of one method."""
    return json.dumps(
        {
            "method_id": method_id,
            "previous_code": previous_code,
            "metrics": {key: float(value) for key, value in metrics.items()},
            "failure_categories": list(failure_categories),
            "sample_errors": list(sample_errors),
            "child_index": child_index,
            "diversity_instruction": diversity_instruction,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
