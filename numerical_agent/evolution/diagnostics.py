"""Deterministic forecast-shape diagnostics for Train-only failure explanation."""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from typing import Mapping, Sequence

from common.llm import parse_json_object

from .execution import Outcome, SUCCESS, Task


FAILURE_JUDGE_SYSTEM = """You are a Train-only forecasting failure analyst. Interpret the
deterministic diagnostic measurements without inventing unseen values. Distinguish level bias,
trend error, periodicity mismatch, phase shift, amplitude error, over-smoothing, horizon error
growth, and recursive explosion. Give general mutation guidance, never Python code, task-specific
constants, a forecast, or a decision to accept a Child. You do not see Dev or holdout labels.

Return exactly one JSON object:
{"failure_types": ["..."], "summary": "...", "evidence": ["..."],
 "mutation_guidance": ["..."], "confidence": 0.0}
"""


def diagnose_forecasts(
    method: str,
    outcomes: Sequence[Outcome],
    tasks: Sequence[Task],
) -> dict[str, object]:
    """Measure error shape without exposing raw forecasts or future values."""
    by_task = {task.task_id: task for task in tasks}
    rows: list[dict[str, object]] = []
    statuses = Counter(outcome.status for outcome in outcomes)
    for outcome in outcomes:
        task = by_task.get(outcome.task_id)
        if (
            task is None
            or outcome.status != SUCCESS
            or len(outcome.forecast) != task.horizon
        ):
            continue
        truth = tuple(float(value) for value in task.future)
        forecast = outcome.forecast
        history = tuple(float(value) for value in task.history)
        split = max(1, len(truth) // 2)
        early = _mean_absolute_error(truth[:split], forecast[:split])
        late = _mean_absolute_error(truth[split:], forecast[split:]) if split < len(truth) else early
        truth_spread = statistics.pstdev(truth) if len(truth) > 1 else 0.0
        forecast_spread = statistics.pstdev(forecast) if len(forecast) > 1 else 0.0
        history_spread = statistics.pstdev(history) if len(history) > 1 else 0.0
        history_radius = max((abs(value - statistics.fmean(history)) for value in history), default=0.0)
        forecast_radius = max(
            (abs(value - statistics.fmean(history)) for value in forecast), default=0.0
        )
        rows.append(
            {
                "series_characteristics": list(task.characteristics()),
                "horizon": task.horizon,
                "mase": _finite_or_none(outcome.mase),
                "mae": _finite_or_none(outcome.mae),
                "smape": _finite_or_none(outcome.smape),
                "mean_bias": _round(statistics.fmean(forecast) - statistics.fmean(truth)),
                "early_mae": _round(early),
                "late_mae": _round(late),
                "late_to_early_error_ratio": _safe_ratio(late, early),
                "amplitude_ratio_to_truth": _safe_ratio(forecast_spread, truth_spread),
                "amplitude_ratio_to_history": _safe_ratio(forecast_spread, history_spread),
                "explosion_ratio_to_history": _safe_ratio(forecast_radius, history_radius),
                "history_trend": _round(_slope(history)),
                "truth_trend": _round(_slope(truth)),
                "forecast_trend": _round(_slope(forecast)),
                "history_dominant_period": _dominant_period(history),
                "truth_dominant_period": _dominant_period(truth),
                "forecast_dominant_period": _dominant_period(forecast),
                "phase_shift_steps": _best_phase_shift(truth, forecast),
            }
        )
    return {
        "method": method,
        "scope": "train_screen_only",
        "evaluated_tasks": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "aggregate": _aggregate(rows),
        "tasks": rows,
    }


def render_failure_judge_user(diagnostics: Mapping[str, object]) -> str:
    """Serialize the bounded diagnostic report supplied to the Judge."""
    return json.dumps(
        {"forecast_diagnostics": dict(diagnostics)},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def parse_failure_diagnosis(text: str) -> dict[str, object]:
    """Validate the Judge's language-only explanation contract."""
    raw = parse_json_object(text)
    failure_types = _string_list(raw.get("failure_types"), "failure_types", allow_empty=True)
    evidence = _string_list(raw.get("evidence"), "evidence")
    guidance = _string_list(raw.get("mutation_guidance"), "mutation_guidance")
    summary = str(raw.get("summary", "")).strip()
    if not summary:
        raise ValueError("failure diagnosis requires a summary")
    confidence = float(raw.get("confidence", -1.0))
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("failure diagnosis confidence must be between zero and one")
    combined = "\n".join((summary, *evidence, *guidance)).lower()
    if any(token in combined for token in ("```", "def ", "import ", "return ")):
        raise ValueError("failure diagnosis must not contain code")
    return {
        "failure_types": failure_types,
        "summary": summary,
        "evidence": evidence,
        "mutation_guidance": guidance,
        "confidence": confidence,
    }


def _string_list(value: object, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    cleaned = [item.strip() for item in value if item.strip()]
    if not cleaned and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return cleaned[:8]


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    names = (
        "mase",
        "mae",
        "smape",
        "mean_bias",
        "early_mae",
        "late_mae",
        "late_to_early_error_ratio",
        "amplitude_ratio_to_truth",
        "amplitude_ratio_to_history",
        "explosion_ratio_to_history",
    )
    result: dict[str, object] = {}
    for name in names:
        values = [float(row[name]) for row in rows if isinstance(row.get(name), (int, float))]
        result[f"median_{name}"] = _round(statistics.median(values)) if values else None
    shifts = [int(row["phase_shift_steps"]) for row in rows if row.get("phase_shift_steps") is not None]
    result["median_absolute_phase_shift_steps"] = (
        _round(statistics.median(abs(value) for value in shifts)) if shifts else None
    )
    result["catastrophic_mase_tasks"] = sum(
        1 for row in rows if isinstance(row.get("mase"), (int, float)) and float(row["mase"]) > 10.0
    )
    return result


def _mean_absolute_error(left: Sequence[float], right: Sequence[float]) -> float:
    return statistics.fmean(abs(a - b) for a, b in zip(left, right, strict=True))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) <= 1e-12:
        return 1.0 if abs(numerator) <= 1e-12 else None
    return _round(numerator / denominator)


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = (len(values) - 1) / 2.0
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    mean = statistics.fmean(values)
    return sum((index - center) * (value - mean) for index, value in enumerate(values)) / denominator


def _dominant_period(values: Sequence[float]) -> int | None:
    if len(values) < 6 or statistics.pstdev(values) <= 1e-12:
        return None
    maximum = min(48, len(values) // 2)
    candidates = [
        (lag, _correlation(values[:-lag], values[lag:]))
        for lag in range(2, maximum + 1)
    ]
    usable = [(lag, score) for lag, score in candidates if score is not None]
    if not usable:
        return None
    lag, score = max(usable, key=lambda item: item[1])
    return lag if score >= 0.3 else None


def _best_phase_shift(truth: Sequence[float], forecast: Sequence[float]) -> int | None:
    if len(truth) < 3 or statistics.pstdev(truth) <= 1e-12 or statistics.pstdev(forecast) <= 1e-12:
        return None
    maximum = min(12, max(1, len(truth) // 2))
    candidates: list[tuple[int, float]] = []
    for shift in range(-maximum, maximum + 1):
        if shift < 0:
            left, right = truth[-shift:], forecast[: len(forecast) + shift]
        elif shift > 0:
            left, right = truth[:-shift], forecast[shift:]
        else:
            left, right = truth, forecast
        score = _correlation(left, right)
        if score is not None:
            candidates.append((shift, score))
    if not candidates:
        return None
    shift, score = max(candidates, key=lambda item: (item[1], -abs(item[0])))
    return shift if score >= 0.3 else None


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator <= 1e-12:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered, strict=True)) / denominator


def _finite_or_none(value: float | None) -> float | None:
    return _round(value) if value is not None and math.isfinite(value) else None


def _round(value: float) -> float:
    return round(float(value), 8)
