"""Bundle-to-prompt splicing, numeric series briefs, and reasoning extraction."""

from __future__ import annotations

import re
import statistics

# trend_word/trend_phrase read only .history_values and .seasonal_period, so they accept a
# NumericTaskView as readily as a TaskView; reusing them keeps one copy of the seasonality math.
from dr_cik.agents.common import trend_phrase, trend_word

from ..models import Bundle

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>(.*)", re.DOTALL)

__all__ = [
    "extract_reasoning",
    "get_code_template",
    "render_code_template_block",
    "render_fewshot_block",
    "render_numeric_brief",
    "render_system_prompt",
    "trend_phrase",
    "trend_word",
]


def extract_reasoning(text: str) -> tuple[str | None, str]:
    """Split a <think> block off the front of a response, returning (reasoning, remaining_answer)."""
    match = _THINK_RE.search(text)
    if match:
        return match.group(1).strip(), _THINK_RE.sub("", text, count=1).strip()
    unterminated = _OPEN_THINK_RE.search(text)
    if unterminated:
        # A truncated response can open <think> and never close it; keep the reasoning, answer is empty.
        return unterminated.group(1).strip(), ""
    return None, text


def render_system_prompt(bundle: Bundle) -> str:
    """Return the bundle's system prompt, the primary thing evolution rewrites."""
    return bundle.system_prompt


def render_fewshot_block(bundle: Bundle) -> str:
    """Render the bundle's few-shot examples as plain prompt text, or an empty string if it has none."""
    if not bundle.fewshot_examples:
        return ""
    blocks = [
        f"Example {index}:\nInput:\n{item.input}\nOutput:\n{item.output}"
        for index, item in enumerate(bundle.fewshot_examples, start=1)
    ]
    return "Worked examples:\n" + "\n\n".join(blocks)


def get_code_template(bundle: Bundle, name: str) -> str:
    """Return one named code template, with a clear error if a mutation left a stale reference."""
    try:
        return bundle.code_templates[name]
    except KeyError:
        available = ", ".join(sorted(bundle.code_templates)) or "(none)"
        raise KeyError(f"bundle {bundle.bundle_id} has no code template {name!r}; available: {available}") from None


def render_code_template_block(bundle: Bundle) -> str:
    """Render every code template as adaptable starter code, or an empty string if it has none."""
    if not bundle.code_templates:
        return ""
    blocks = [f"# template: {name}\n{source.strip()}" for name, source in sorted(bundle.code_templates.items())]
    return "Starter code you may adapt or ignore:\n" + "\n\n".join(blocks)


def render_numeric_brief(view) -> str:
    """Summarize a series' shape and horizon; never mentions documents, text, or corpus ids."""
    values = view.history_values
    window = values[-min(len(values), 20) :]
    return "\n".join(
        [
            f"Frequency: {view.frequency}",
            f"History length: {len(values)} points",
            f"Full history: min={min(values):.6g} max={max(values):.6g} mean={statistics.fmean(values):.6g}",
            f"Recent {len(window)}: min={min(window):.6g} max={max(window):.6g} mean={statistics.fmean(window):.6g}",
            f"Last 10 values: {', '.join(f'{value:.6g}' for value in values[-10:])}",
            f"Trend: {trend_phrase(view)}",
            f"Forecast horizon: {view.prediction_length} steps",
        ]
    )
