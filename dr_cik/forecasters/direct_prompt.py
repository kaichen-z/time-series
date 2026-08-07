"""Direct-Prompt LLM baseline: an LLM forecasts directly from history + research context, no numeric foundation model."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from ..agents.common import render_task_brief
from ..llm import JsonExtractionError, LLMClient, parse_json_object
from ..models import Forecast, TaskView

DIRECT_PROMPT_SYSTEM_PREAMBLE = (
    "You are a probabilistic time-series forecaster. You are given a task's history, target "
    "description, forecast horizon, and supporting research context. Forecast the target "
    "variable for every step of the horizon."
)

_RETRY_REMINDER = "Your previous response was not valid JSON. Reply with ONLY the JSON object, no other text."
_TOKENS_PER_NUMBER = 8  # generous: digits + comma can each be separate BPE tokens for dense decimal output
_JSON_OVERHEAD_TOKENS = 128


def _output_token_budget(horizon: int, floor: int) -> int:
    """Scale the generation budget with the horizon so long-horizon tasks don't get truncated mid-array."""
    return max(floor, horizon * _TOKENS_PER_NUMBER + _JSON_OVERHEAD_TOKENS)


@dataclass(frozen=True)
class DirectPromptConfig:
    """Tunables for the Direct-Prompt forecaster."""

    model_id: str
    num_samples: int = 25
    temperature: float = 1.0  # must be >0: each of the S calls needs to sample, or all S draws come out identical
    max_output_tokens: int = 512  # floor only: forecast() scales this up with horizon (one trajectory per call now, not S at once)
    seed: int = 7


def _render_history_table(view: TaskView) -> str:
    """Render the full (timestamp, value) history, since the model must forecast real numbers."""
    lines = [f"{ts}\t{value:.6g}" for ts, value in zip(view.history_timestamps, view.history_values)]
    return "timestamp\tvalue\n" + "\n".join(lines)


def _build_prompt(view: TaskView, context_text: str) -> str:
    """One trajectory per call — diversity across the S calls comes from sampling (temperature), not from asking for variety in one shot."""
    return (
        f"{render_task_brief(view)}\n\n"
        f"Full history:\n{_render_history_table(view)}\n\n"
        f"Research context:\n{context_text or '(no research context provided)'}\n\n"
        f"Forecast the next {view.prediction_length} steps "
        f"({view.future_timestamps[0]} to {view.future_timestamps[-1]}). "
        f'Respond with exactly one JSON object: {{"forecast": [v1, v2, ...]}} with exactly '
        f"{view.prediction_length} numbers. "
        "Round every number to 2 decimal places. "
        "Output compact JSON only: no markdown fence, no whitespace between elements, no commentary before or after."
    )


def _extract_forecast(parsed: dict, horizon: int) -> list[float] | None:
    """Accept a forecast array of exactly horizon length; truncate a near-miss over-long array (a common off-by-one)."""
    raw = parsed.get("forecast")
    if not isinstance(raw, list) or len(raw) < horizon:
        return None
    try:
        return [float(value) for value in raw[:horizon]]
    except (TypeError, ValueError):
        return None


def _stable_rng(benchmark_id: str, seed: int) -> random.Random:
    """A per-task RNG that's deterministic given --seed, unlike Python's randomized str hash()."""
    digest = hashlib.md5(f"{seed}:{benchmark_id}".encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def _jitter_fallback(view: TaskView, num_samples: int, rng: random.Random) -> tuple[tuple[float, ...], ...]:
    """Last-value persistence plus gaussian jitter scaled to recent volatility; a degraded fallback only."""
    window = view.history_values[-min(len(view.history_values), 20) :]
    last = view.history_values[-1]
    spread = statistics.pstdev(window) if len(window) > 1 else (abs(last) * 0.05 or 1.0)
    return tuple(tuple(last + rng.gauss(0, spread) for _ in range(view.prediction_length)) for _ in range(num_samples))


def _complete_many(llm: LLMClient, *, system: str, messages: list[dict[str, str]], count: int, temperature: float, max_output_tokens: int):
    """Use LLMClient.complete_many if the backend has it (e.g. QwenClient batches via num_return_sequences); else loop complete()."""
    batched = getattr(llm, "complete_many", None)
    if batched is not None:
        return batched(system=system, messages=messages, count=count, temperature=temperature, max_output_tokens=max_output_tokens)
    return [llm.complete(system=system, messages=messages, temperature=temperature, max_output_tokens=max_output_tokens) for _ in range(count)]


class DirectPromptForecaster:
    """Asks an LLM to forecast directly from history + research context; no numeric foundation model.

    Matches how the literature (Williams et al. 2025, "Context is Key", which Dr-CiK's own Direct
    Prompt baseline cites) actually samples: S independent temperature-sampled calls, each producing
    one trajectory — not one call asked to invent S varied arrays, which turned out unreliable (see
    git history: thinking-mode token burn, budget truncation on long horizons, miscounted arrays).
    """

    def __init__(self, llm: LLMClient, config: DirectPromptConfig) -> None:
        self.llm = llm
        self.config = config

    def forecast(self, task_view: TaskView, context_text: str, num_samples: int | None = None) -> Forecast:

        sample_count = num_samples or self.config.num_samples
        horizon = task_view.prediction_length
        budget = _output_token_budget(horizon, floor=self.config.max_output_tokens)
        prompt = _build_prompt(task_view, context_text)

        rows = self._collect_rows(prompt, horizon, sample_count, budget)
        missing = sample_count - len(rows)
        if missing > 0:
            rows.extend(self._collect_rows(f"{prompt}\n\n{_RETRY_REMINDER}", horizon, missing, budget))

        rng = _stable_rng(task_view.benchmark_id, self.config.seed)
        if not rows:
            samples = _jitter_fallback(task_view, sample_count, rng)
            method = f"direct-prompt:{self.config.model_id}:degraded-fallback(S={sample_count})"
        elif len(rows) < sample_count:
            padded = rows + [rng.choice(rows) for _ in range(sample_count - len(rows))]
            samples = tuple(tuple(row) for row in padded)
            method = f"direct-prompt:{self.config.model_id}:padded(S={sample_count},model_rows={len(rows)})"
        else:
            samples = tuple(tuple(row) for row in rows[:sample_count])
            method = f"direct-prompt:{self.config.model_id}(S={sample_count})"

        mean = tuple(statistics.fmean(sample[step] for sample in samples) for step in range(horizon))
        return Forecast(mean=mean, samples=samples, method=method)

    def _collect_rows(self, prompt: str, horizon: int, count: int, max_output_tokens: int) -> list[list[float]]:
        responses = _complete_many(
            self.llm,
            system=DIRECT_PROMPT_SYSTEM_PREAMBLE,
            messages=[{"role": "user", "content": prompt}],
            count=count,
            temperature=self.config.temperature,
            max_output_tokens=max_output_tokens,
        )
        rows: list[list[float]] = []
        for response in responses:
            try:
                parsed = parse_json_object(response.text)
            except JsonExtractionError:
                continue
            row = _extract_forecast(parsed, horizon)
            if row is not None:
                rows.append(row)
        return rows


def load_prior_context(run_dir: str | Path) -> dict[str, str]:
    """Read a prior run's run_report.jsonl into benchmark_id -> rendered report+evidence text (the DR-synthesized context condition)."""
    path = Path(run_dir).expanduser().resolve() / "run_report.jsonl"
    context_by_id: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            claims = "\n".join(f"- {item['claim']}" for item in row.get("evidence", []))
            context_by_id[str(row["benchmark_id"])] = f"{row.get('report_markdown', '')}\n\n{claims}".strip()
    return context_by_id
