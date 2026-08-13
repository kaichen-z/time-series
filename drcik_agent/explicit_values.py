from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .agents import ProbabilisticForecastAgent
from .models import Diagnosis, Evidence, ForecastTask, RetrievedDocument


@dataclass(frozen=True)
class ExplicitValueDecision:
    timestamp: str
    value: float
    source_document_ids: tuple[str, ...]
    accepted: bool
    reason: str
    baseline_value: float
    standardized_deviation: float


@dataclass(frozen=True)
class ExplicitValueValidation:
    accepted_points: dict[str, float]
    accepted_sources: dict[str, tuple[str, ...]]
    decisions: tuple[ExplicitValueDecision, ...]


class ExplicitValueValidator:
    """Validate explicit future values without benchmark labels.

    A dated timestamp-value anchor must occur in one grounded cited document,
    be independently corroborated by a second grounded cited document at the
    same local time, and remain plausible under the observed numerical scale.
    This blocks single-document numerical distractors while allowing differently
    formatted reports to corroborate the same schedule.
    """

    def __init__(self, min_sources: int = 2, max_scale_deviation: float = 8.0) -> None:
        if min_sources < 2:
            raise ValueError("explicit values require at least two independent sources")
        if max_scale_deviation <= 0:
            raise ValueError("max_scale_deviation must be positive")
        self.min_sources = min_sources
        self.max_scale_deviation = max_scale_deviation

    @staticmethod
    def _number(cell: str) -> float | None:
        match = re.fullmatch(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", cell.strip())
        return float(match.group(0).replace(",", "")) if match else None

    @classmethod
    def _corroborates_local_value(cls, text: str, timestamp: str, value: float) -> bool:
        clock = timestamp[11:16] if len(timestamp) >= 16 else ""
        if not clock:
            return False
        for raw_line in text.splitlines():
            if "|" not in raw_line:
                continue
            cells = [cell.strip() for cell in raw_line.strip().strip("|").split("|")]
            if not cells or not re.fullmatch(re.escape(clock) + r"(?::\d{2})?", cells[0]):
                continue
            for cell in cells[1:]:
                parsed = cls._number(cell)
                if parsed is not None and math.isclose(parsed, value, rel_tol=1e-6, abs_tol=1e-9):
                    return True
        # Also support prose/list representations containing the full timestamp
        # and value, but never accept an unanchored numeric range.
        escaped = re.escape(timestamp)
        value_pattern = re.escape(f"{value:g}")
        return bool(
            re.search(
                escaped + r"[^\n]{0,80}(?:value\s*[=:]\s*)?" + value_pattern + r"\b",
                text,
                re.IGNORECASE,
            )
        )

    def validate(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        baseline_values: tuple[float, ...],
        retrieved: list[RetrievedDocument],
        evidence: list[Evidence],
    ) -> ExplicitValueValidation:
        grounded_ids = {
            item.document_id
            for item in evidence
            if item.provenance_valid and item.evidence_quote
        }
        eligible = [
            item for item in retrieved if item.document.document_id in grounded_ids
        ]
        index_by_timestamp = {
            timestamp: index for index, timestamp in enumerate(task.future_timestamps)
        }
        anchors: dict[str, list[tuple[str, float]]] = {}
        for item in eligible:
            points = ProbabilisticForecastAgent._extract_context_points(
                task.future_timestamps, [item]
            )
            for timestamp, value in points.items():
                anchors.setdefault(timestamp, []).append(
                    (item.document.document_id, value)
                )

        history_min = min(task.history_values)
        history_max = max(task.history_values)
        history_range = max(history_max - history_min, diagnosis.residual_scale, 1e-9)
        accepted_points: dict[str, float] = {}
        accepted_sources: dict[str, tuple[str, ...]] = {}
        decisions: list[ExplicitValueDecision] = []
        for timestamp, source_values in sorted(anchors.items()):
            anchor_values = [value for _, value in source_values]
            value = sorted(anchor_values)[len(anchor_values) // 2]
            sources = [document_id for document_id, _ in source_values]
            for item in eligible:
                document_id = item.document.document_id
                if document_id in sources:
                    continue
                if self._corroborates_local_value(item.document.text, timestamp, value):
                    sources.append(document_id)
            sources = list(dict.fromkeys(sources))
            baseline_value = baseline_values[index_by_timestamp[timestamp]]
            standardized_deviation = abs(value - baseline_value) / history_range
            scale_floor = history_min - 0.5 * history_range
            scale_ceiling = history_max + 0.5 * history_range
            plausible_scale = scale_floor <= value <= scale_ceiling
            plausible_deviation = standardized_deviation <= self.max_scale_deviation
            accepted = (
                len(sources) >= self.min_sources
                and plausible_scale
                and plausible_deviation
            )
            if len(sources) < self.min_sources:
                reason = "insufficient_independent_corroboration"
            elif not plausible_scale:
                reason = "outside_observed_scale"
            elif not plausible_deviation:
                reason = "inconsistent_with_numerical_baseline"
            else:
                reason = "corroborated_explicit_future_value"
                accepted_points[timestamp] = value
                accepted_sources[timestamp] = tuple(sources)
            decisions.append(
                ExplicitValueDecision(
                    timestamp=timestamp,
                    value=value,
                    source_document_ids=tuple(sources),
                    accepted=accepted,
                    reason=reason,
                    baseline_value=baseline_value,
                    standardized_deviation=standardized_deviation,
                )
            )
        return ExplicitValueValidation(
            accepted_points=accepted_points,
            accepted_sources=accepted_sources,
            decisions=tuple(decisions),
        )
