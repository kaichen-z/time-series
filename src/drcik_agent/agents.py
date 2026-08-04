from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass

from .backbones import ForecastBackbone, StatisticalForecastBackbone
from .models import (
    Diagnosis,
    Document,
    Evidence,
    EvidenceImpact,
    Forecast,
    ForecastAdjustment,
    ForecastTask,
    RetrievedDocument,
    RevisionRecord,
)


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}|\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?")
STOPWORDS = {
    "about", "after", "again", "also", "among", "and", "are", "been", "before",
    "being", "between", "both", "but", "can", "could", "does", "each", "for",
    "from", "had", "has", "have", "into", "its", "may", "more", "most", "not",
    "only", "other", "our", "over", "same", "should", "such", "than", "that",
    "the", "their", "then", "there", "these", "they", "this", "through", "under",
    "using", "was", "were", "what", "when", "where", "which", "while", "will",
    "with", "within", "would", "your",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


def _linear_slope(values: tuple[float, ...]) -> float:
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
    """Infer a short repeated cycle after removing a linear trend.

    Dr-CiK occasionally stores pandas frequency aliases (for example ``D`` or
    ``5T``) in the seasonal-period field.  Those aliases are not step counts,
    so the agent needs a conservative numerical fallback.
    """
    if len(values) < 12:
        return None, 0.0
    candidates: list[tuple[float, int]] = []
    # Smooth trends have high autocorrelation at lags 2, 3, 4, ... even when
    # they are not periodic.  Look for an interior autocorrelation peak across
    # several recent windows instead of blindly selecting the smallest lag.
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


class TimeSeriesDiagnosisAgent:
    def diagnose(self, task: ForecastTask) -> Diagnosis:
        values = task.history_values
        slope = _linear_slope(values)
        value_scale = max(abs(statistics.fmean(values)), statistics.pstdev(values), 1e-8)
        normalized_slope = slope / value_scale
        if normalized_slope > 0.01:
            trend = "upward"
        elif normalized_slope < -0.01:
            trend = "downward"
        else:
            trend = "stable"

        inferred_period, inferred_strength = _infer_seasonal_period(values)
        period = task.seasonal_period or inferred_period
        seasonal_strength = 0.0
        seasonal_errors: list[float] = []
        if period and 0 < period < len(values):
            seasonal_strength = max(
                0.0,
                _correlation(list(values[period:]), list(values[:-period])),
            )
            seasonal_errors = [values[index] - values[index - period] for index in range(period, len(values))]
        if task.seasonal_period is None and inferred_period == period:
            seasonal_strength = max(seasonal_strength, inferred_strength)
        first_differences = [values[index] - values[index - 1] for index in range(1, len(values))]
        residuals = seasonal_errors or first_differences or [0.0]
        recent_count = 2 * period if period else 50
        recent_residuals = residuals[-min(len(residuals), recent_count) :]
        residual_std = statistics.pstdev(recent_residuals)
        residual_median = statistics.median(recent_residuals)
        residual_mad = 1.4826 * statistics.median(
            abs(value - residual_median) for value in recent_residuals
        )
        # Historical anomalies can make the raw standard deviation unusably
        # large.  A recent robust scale retains uncertainty without allowing a
        # resolved incident to dominate every future sample trajectory.
        residual_scale = min(residual_std, max(residual_mad, 0.1 * residual_std))
        if residual_scale <= 1e-12:
            residual_scale = max(statistics.pstdev(values) * 0.05, abs(values[-1]) * 0.01, 1e-6)

        needs = [
            f"Events affecting {task.target_name} during {task.future_timestamps[0]} to {task.future_timestamps[-1]}",
            f"Evidence explaining a {trend} or regime-changing trajectory",
            f"Entity-specific information for {task.entity_name}",
        ]
        query = " ".join(
            [
                task.entity_name,
                task.target_name,
                task.target_description,
                task.frequency,
                *task.future_timestamps,
                *needs,
            ]
        )
        return Diagnosis(
            trend=trend,
            slope_per_step=slope,
            seasonal_period=period,
            seasonal_strength=seasonal_strength,
            residual_scale=residual_scale,
            information_needs=tuple(needs),
            retrieval_query=query,
        )


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75


class RetrievalAgent:
    """A deterministic BM25 retriever that never sees Dr-CiK role labels."""

    def __init__(self, config: BM25Config | None = None) -> None:
        self.config = config or BM25Config()

    def retrieve(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        top_k: int,
        query: str | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[RetrievedDocument]:
        excluded = exclude_ids or set()
        documents = [
            document.agent_view()
            for document in task.documents
            if document.document_id not in excluded
        ]
        if not documents or top_k <= 0:
            return []
        tokenized = [tokenize(document.text) for document in documents]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        query_counts = Counter(tokenize(query or diagnosis.retrieval_query))
        average_length = statistics.fmean([len(tokens) for tokens in tokenized]) or 1.0
        total_documents = len(documents)

        scored: list[tuple[float, Document]] = []
        for document, tokens in zip(documents, tokenized):
            counts = Counter(tokens)
            length_norm = self.config.k1 * (
                1 - self.config.b + self.config.b * len(tokens) / average_length
            )
            score = 0.0
            for token, query_weight in query_counts.items():
                frequency = counts.get(token, 0)
                if frequency == 0:
                    continue
                doc_frequency = document_frequency[token]
                inverse_frequency = math.log(1 + (total_documents - doc_frequency + 0.5) / (doc_frequency + 0.5))
                term_score = inverse_frequency * frequency * (self.config.k1 + 1) / (frequency + length_norm)
                score += min(query_weight, 3) * term_score
            scored.append((score, document))

        scored.sort(key=lambda item: (-item[0], item[1].document_id))
        return [
            RetrievedDocument(document=document, score=score, rank=rank)
            for rank, (score, document) in enumerate(scored[: min(top_k, len(scored))], start=1)
        ]


class EvidenceSynthesisAgent:
    DIRECTION_WORDS = {
        "up": {"increase", "increased", "increasing", "rise", "rising", "growth", "higher", "upward"},
        "down": {"decrease", "decreased", "decreasing", "decline", "fall", "falling", "lower", "downward", "drop"},
    }

    def synthesize(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        retrieved: list[RetrievedDocument],
        max_claims_per_document: int = 2,
    ) -> list[Evidence]:
        query_terms = set(tokenize(diagnosis.retrieval_query))
        evidence: list[Evidence] = []
        for item in retrieved:
            fragments = [
                fragment.strip(" -\t")
                for fragment in re.split(r"(?<=[.!?])\s+|\n+", item.document.text)
                if len(fragment.strip()) >= 30
            ]
            ranked: list[tuple[float, str, set[str]]] = []
            for fragment in fragments:
                terms = set(tokenize(fragment))
                matches = terms & query_terms
                if matches:
                    ranked.append((len(matches) / math.sqrt(max(len(terms), 1)), fragment, matches))
            ranked.sort(key=lambda value: (-value[0], -len(value[2]), value[1]))
            for fragment_score, fragment, matches in ranked[:max_claims_per_document]:
                lower_terms = set(tokenize(fragment))
                up = len(lower_terms & self.DIRECTION_WORDS["up"])
                down = len(lower_terms & self.DIRECTION_WORDS["down"])
                direction = "up" if up > down else "down" if down > up else "unclear"
                date_matches = re.findall(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?\b", fragment)
                window = " to ".join((date_matches[0], date_matches[-1])) if date_matches else "unspecified"
                confidence = min(0.99, 0.35 + 0.08 * len(matches) + 0.05 * fragment_score)
                evidence.append(
                    Evidence(
                        document_id=item.document.document_id,
                        claim=fragment[:800],
                        matched_terms=tuple(sorted(matches)),
                        confidence=confidence,
                        effect_direction=direction,
                        effect_window=window,
                    )
                )
        return evidence


class ProbabilisticForecastAgent:
    def __init__(self, backbone: ForecastBackbone | None = None) -> None:
        # Direct construction remains lightweight for unit tests and the
        # one-pass legacy API. Systems explicitly select their backbone.
        self.backbone = backbone or StatisticalForecastBackbone()

    def _baseline(self, task: ForecastTask, diagnosis: Diagnosis) -> tuple[list[float], str]:
        values, method = self.backbone.forecast(task, diagnosis)
        return list(values), method

    def baseline(self, task: ForecastTask, diagnosis: Diagnosis) -> tuple[tuple[float, ...], str]:
        """Generate the forecasting backbone output exactly once."""
        values, method = self._baseline(task, diagnosis)
        return tuple(values), method

    @staticmethod
    def _extract_context_points(
        future_timestamps: tuple[str, ...], retrieved: list[RetrievedDocument]
    ) -> dict[str, float]:
        values_by_timestamp: dict[str, list[float]] = {timestamp: [] for timestamp in future_timestamps}
        for item in retrieved:
            text = item.document.text
            for timestamp in future_timestamps:
                pattern = re.compile(
                    re.escape(timestamp)
                    + r"\s*[),|:=]\s*(?:value\s*[=:]\s*)?"
                    + r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
                    re.IGNORECASE,
                )
                for match in pattern.finditer(text):
                    values_by_timestamp[timestamp].append(
                        float(match.group(1).replace(",", ""))
                    )
        return {
            timestamp: statistics.median(values)
            for timestamp, values in values_by_timestamp.items()
            if values
        }

    @staticmethod
    def _affected_indices(task: ForecastTask, impact: EvidenceImpact) -> list[int]:
        if impact.forecast_relation == "forecast_relevant_undated":
            return list(range(task.prediction_length))
        if impact.forecast_relation != "overlaps_forecast":
            return []
        start = impact.start_timestamp
        end = impact.end_timestamp
        if impact.permanence == "permanent" and start and end is None:
            end = task.future_timestamps[-1]
        indices: list[int] = []
        for index, timestamp in enumerate(task.future_timestamps):
            date_or_timestamp = timestamp[: len(start)] if start else timestamp
            after_start = start is None or date_or_timestamp >= start
            date_or_timestamp = timestamp[: len(end)] if end else timestamp
            before_end = end is None or date_or_timestamp <= end
            if after_start and before_end:
                indices.append(index)
        return indices

    @staticmethod
    def _apply_impacts(
        task: ForecastTask,
        diagnosis: Diagnosis,
        baseline: list[float],
        impacts: list[EvidenceImpact],
    ) -> tuple[list[float], tuple[ForecastAdjustment, ...]]:
        adjusted = list(baseline)
        audit: list[ForecastAdjustment] = []
        for impact in impacts:
            indices = ProbabilisticForecastAgent._affected_indices(task, impact)
            before = [adjusted[index] for index in indices]
            value = impact.adjustment_value
            if impact.adjustment_kind == "multiplier" and value is not None:
                for index in indices:
                    adjusted[index] *= value
            elif impact.adjustment_kind == "percentage" and value is not None:
                for index in indices:
                    adjusted[index] *= 1.0 + value
            elif impact.adjustment_kind == "standardized_additive" and value is not None:
                for index in indices:
                    adjusted[index] += value * diagnosis.residual_scale
            elif impact.adjustment_kind == "absolute_additive" and value is not None:
                for index in indices:
                    adjusted[index] += value
            mean_change = (
                statistics.fmean(
                    abs(adjusted[index] - old_value)
                    for index, old_value in zip(indices, before)
                )
                if indices
                else 0.0
            )
            audit.append(
                ForecastAdjustment(
                    source_document_ids=impact.source_document_ids,
                    adjustment_kind=impact.adjustment_kind,
                    adjustment_value=impact.adjustment_value,
                    affected_steps=len(indices),
                    mean_absolute_change=mean_change,
                    rationale=impact.rationale,
                )
            )
        return adjusted, tuple(audit)

    def forecast(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        retrieved: list[RetrievedDocument],
        num_samples: int,
        seed: int,
        context_weight: float,
        impacts: list[EvidenceImpact] | None = None,
    ) -> Forecast:
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if not 0.0 <= context_weight <= 1.0:
            raise ValueError("context_weight must be between 0 and 1")
        baseline, method = self._baseline(task, diagnosis)
        mean, impact_adjustments = self._apply_impacts(
            task, diagnosis, baseline, impacts or []
        )
        context_points = self._extract_context_points(task.future_timestamps, retrieved)
        for index, timestamp in enumerate(task.future_timestamps):
            if timestamp in context_points:
                mean[index] = (1 - context_weight) * mean[index] + context_weight * context_points[timestamp]

        return self.forecast_from_mean(
            task=task,
            diagnosis=diagnosis,
            mean=tuple(mean),
            baseline_mean=tuple(baseline),
            baseline_method=method,
            num_samples=num_samples,
            seed=seed,
            context_points=context_points,
            impact_adjustments=impact_adjustments,
        )

    def forecast_from_mean(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        mean: tuple[float, ...],
        baseline_mean: tuple[float, ...],
        baseline_method: str,
        num_samples: int,
        seed: int,
        context_points: dict[str, float] | None = None,
        impact_adjustments: tuple[ForecastAdjustment, ...] = (),
        revision_records: tuple[RevisionRecord, ...] | list[RevisionRecord] = (),
    ) -> Forecast:
        """Attach uncertainty samples to an already audited workspace forecast."""
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if len(mean) != task.prediction_length or len(baseline_mean) != task.prediction_length:
            raise ValueError("forecast means must match prediction_length")

        nonnegative = min(task.history_values) >= 0
        random_generator = random.Random(seed)
        samples: list[tuple[float, ...]] = []
        for _ in range(num_samples):
            trajectory: list[float] = []
            shared_shock = random_generator.gauss(0, diagnosis.residual_scale * 0.35)
            for step, center in enumerate(mean):
                step_scale = diagnosis.residual_scale * math.sqrt(1 + step / max(task.prediction_length, 1))
                value = center + shared_shock + random_generator.gauss(0, step_scale)
                trajectory.append(max(0.0, value) if nonnegative else value)
            samples.append(tuple(trajectory))
        return Forecast(
            mean=tuple(mean),
            samples=tuple(samples),
            baseline_method=baseline_method,
            context_points=context_points or {},
            impact_adjustments=impact_adjustments,
            baseline_mean=tuple(baseline_mean),
            revision_records=tuple(revision_records),
        )
