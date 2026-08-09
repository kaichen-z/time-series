"""Prompt scaffolding shared by the OpenDR and DRBench agents."""

from __future__ import annotations

import math
import statistics
from typing import Any

from ..models import AgentDocument, EvidenceItem, TaskView

AGENT_SYSTEM_PREAMBLE = (
    "You are a deep-research forecasting assistant. You are given a forecasting task "
    "(entity, target variable, historical time series, forecast horizon) and a corpus of "
    "short documents identified only by document_id. Some documents are relevant context; "
    "others are distractors planted to mislead you, and you are not told which is which. "
    "You may only see document text by using the search tool. Every claim you report must "
    "be traceable to a specific document_id returned by a search. Never invent facts."
)


def _linear_slope(values: tuple[float, ...]) -> float:
    """OLS slope of values against their index; robust to where a window's endpoints happen to fall."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2
    mean_y = statistics.fmean(values)
    denominator = sum((index - mean_x) ** 2 for index in range(n))
    if denominator == 0:
        return 0.0
    return sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values)) / denominator


def _correlation(left: list[float], right: list[float]) -> float:
    """Pearson correlation between two equal-length sequences, 0.0 if either is constant."""
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    scale_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    scale_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if scale_left == 0 or scale_right == 0:
        return 0.0
    return numerator / (scale_left * scale_right)


def _infer_seasonal_period(values: tuple[float, ...]) -> tuple[int | None, float]:
    """Infer a short repeated cycle after detrending.

    Smooth trends have high autocorrelation at lags 2, 3, 4, ... even when they are not
    periodic, so this looks for an interior autocorrelation peak across several recent
    windows instead of blindly selecting the smallest lag.
    """
    if len(values) < 12:
        return None, 0.0
    candidates: list[tuple[float, int]] = []
    for requested_window in (48, 72, 96, 180):
        window = min(len(values), requested_window)
        if window < 12:
            continue
        recent = values[-window:]
        slope = _linear_slope(recent)
        intercept = statistics.fmean(recent) - slope * (len(recent) - 1) / 2
        detrended = [value - (intercept + slope * index) for index, value in enumerate(recent)]
        maximum_lag = min(60, len(detrended) // 2)
        correlations = [
            (lag, _correlation(detrended[lag:], detrended[:-lag]))
            for lag in range(3, maximum_lag + 1)
        ]
        for index in range(1, len(correlations) - 1):
            lag, correlation = correlations[index]
            if correlation >= correlations[index - 1][1] and correlation >= correlations[index + 1][1]:
                candidates.append((correlation, lag))
        if correlations:
            lag, correlation = correlations[-1]
            if correlation >= correlations[-2][1]:
                candidates.append((correlation, lag))
    if not candidates:
        return None, 0.0
    strength, period = max(candidates, key=lambda item: (item[0], -item[1]))
    if strength < 0.45:
        return None, max(0.0, strength)
    return period, max(0.0, strength)


def _declared_seasonal_period(seasonal_period: object) -> int | None:
    """Accept a usable step-count; Dr-CiK sometimes stores a pandas alias like "D"/"5T" here instead."""
    if seasonal_period is None or isinstance(seasonal_period, bool):
        return None
    try:
        parsed = int(seasonal_period)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def trend_word(history_values: tuple[float, ...]) -> str:
    """Describe a series as rising, falling, volatile, or stable using an OLS slope over its full history."""
    if len(history_values) < 2:
        return "stable"
    window = history_values[-min(len(history_values), 20) :]
    mean = statistics.fmean(window)
    spread = statistics.pstdev(window)
    if mean != 0 and spread / abs(mean) > 0.3:
        return "volatile"
    slope = _linear_slope(history_values)
    scale = max(abs(statistics.fmean(history_values)), statistics.pstdev(history_values), 1e-8)
    if slope / scale > 0.01:
        return "rising"
    if slope / scale < -0.01:
        return "falling"
    return "stable"


def trend_phrase(view: TaskView) -> str:
    """Combine the trend word with any declared or detected repeating cycle into one phrase."""
    word = trend_word(view.history_values)
    declared = _declared_seasonal_period(view.seasonal_period)
    inferred_period, inferred_strength = _infer_seasonal_period(view.history_values)
    period = declared or inferred_period
    if period is None or not (0 < period < len(view.history_values)):
        return f"{word} trajectory, no strong cycle"
    strength = max(0.0, _correlation(list(view.history_values[period:]), list(view.history_values[:-period])))
    if declared is None and inferred_period == period:
        strength = max(strength, inferred_strength)
    source = "declared" if declared is not None else "detected"
    return f"{word} trajectory with an approximate {period}-step cycle ({source}, strength {strength:.2f})"


def render_task_brief(view: TaskView) -> str:
    """Summarize a task's entity/target/history and list corpus document IDs, no text."""
    window = view.history_values[-min(len(view.history_values), 20) :]
    lines = [
        f"Entity: {view.entity_name}",
        f"Target variable: {view.target_name}",
        f"Target description: {view.target_description}",
        f"Frequency: {view.frequency}",
        f"History range: {view.history_timestamps[0]} to {view.history_timestamps[-1]} ({len(view.history_values)} points)",
        f"Recent values (last {len(window)}): min={min(window):.4g} max={max(window):.4g} mean={statistics.fmean(window):.4g}",
        f"Trend: {trend_phrase(view)}",
        f"Forecast horizon: {view.future_timestamps[0]} to {view.future_timestamps[-1]} ({view.prediction_length} steps)",
        f"Corpus: {len(view.documents)} documents, ids: {', '.join(document.document_id for document in view.documents)}",
    ]
    return "\n".join(lines)


def parse_evidence_list(raw: list[dict[str, Any]], valid_document_ids: set[str]) -> tuple[EvidenceItem, ...]:
    """Validate evidence items and drop any citations to document IDs outside the corpus."""
    items: list[EvidenceItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        claim = entry.get("claim")
        source_ids = entry.get("source_doc_ids")
        if not isinstance(claim, str) or not claim.strip() or not isinstance(source_ids, list):
            continue
        filtered = tuple(str(doc_id) for doc_id in source_ids if str(doc_id) in valid_document_ids)
        items.append(EvidenceItem(claim=claim.strip(), source_doc_ids=filtered))
    return tuple(items)


def documents_by_id(documents: tuple[AgentDocument, ...]) -> dict[str, AgentDocument]:
    """Index a corpus by document_id for O(1) lookups."""
    return {document.document_id: document for document in documents}
