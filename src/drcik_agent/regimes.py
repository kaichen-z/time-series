from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass

from .models import Diagnosis, Evidence, ForecastTask


@dataclass(frozen=True)
class RegimeProjection:
    """Evidence-gated projection of a recovered normal seasonal regime."""

    values: tuple[float, ...]
    source_document_ids: tuple[str, ...]
    confidence: float
    seasonal_period: int
    validation_mae: float
    seasonal_naive_mae: float
    blend_weight: float
    rationale: str


def _solve_linear_system(
    matrix: list[list[float]], vector: list[float]
) -> tuple[float, ...] | None:
    """Solve a tiny dense system with pivoted Gauss-Jordan elimination."""

    size = len(vector)
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if abs(factor) < 1e-14:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return tuple(augmented[row][-1] for row in range(size))


def _features(index: int, period: int) -> tuple[float, float, float, float]:
    angle = 2.0 * math.pi * index / period
    return 1.0, float(index), math.sin(angle), math.cos(angle)


def _fit_harmonic(values: tuple[float, ...], period: int) -> tuple[float, ...] | None:
    dimension = 4
    gram = [[0.0] * dimension for _ in range(dimension)]
    target = [0.0] * dimension
    for index, value in enumerate(values):
        row = _features(index, period)
        for left in range(dimension):
            target[left] += row[left] * value
            for right in range(dimension):
                gram[left][right] += row[left] * row[right]
    ridge = max(1e-8, sum(gram[index][index] for index in range(dimension)) * 1e-10)
    for index in range(dimension):
        gram[index][index] += ridge
    return _solve_linear_system(gram, target)


def _predict(coefficients: tuple[float, ...], indices: range, period: int) -> tuple[float, ...]:
    return tuple(
        sum(value * feature for value, feature in zip(coefficients, _features(index, period)))
        for index in indices
    )


class RegimeNormalizationAgent:
    """Translate "return to normal" evidence into a backtested numeric trajectory.

    The agent is intentionally conservative.  Text only opens the gate; all
    magnitudes come from the observed series.  It fits a trend-plus-harmonic
    model to the most recent two cycles, validates it on the last cycle, and
    blends it with the immutable forecasting-backbone prior according to the
    measured validation gain.
    """

    STRONG_NORMALIZATION_PHRASES = (
        "return to normal",
        "returned to normal",
        "return to baseline",
        "returned to baseline",
        "reverted to baseline",
        "restored to normal",
        "return to standard",
        "historical baseline",
        "historical norms",
        "historical averages",
        "baseline and seasonality",
        "standard sales cycle",
    )
    WEAK_NORMALIZATION_PHRASES = (
        "normalization",
        "stabiliz",
        "settled into",
    )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        # Lightweight singular normalization is sufficient for pairs such as
        # ``reading/readings`` without adding a stemming dependency.
        return tokens | {token[:-1] for token in tokens if len(token) > 3 and token.endswith("s")}

    @classmethod
    def _is_grounded_normalization_claim(
        cls,
        task: ForecastTask,
        item: Evidence,
        target_terms: set[str],
    ) -> bool:
        lower = item.claim.lower()
        target_match = bool(cls._tokens(item.claim) & target_terms)
        entity_match = bool(task.entity_name) and task.entity_name.lower() in lower
        strong = any(phrase in lower for phrase in cls.STRONG_NORMALIZATION_PHRASES)
        weak = any(phrase in lower for phrase in cls.WEAK_NORMALIZATION_PHRASES)
        return (strong and (target_match or entity_match)) or (weak and target_match)

    @classmethod
    def _normalization_sources(
        cls, task: ForecastTask, evidence: list[Evidence]
    ) -> tuple[str, ...]:
        # Use only the variable name. Descriptions often contain generic words
        # such as "sensor" or "store" that let unrelated stabilization claims
        # pass the gate.
        target_terms = cls._tokens(task.target_name)
        return tuple(
            dict.fromkeys(
                item.document_id
                for item in evidence
                if item.provenance_valid
                and cls._is_grounded_normalization_claim(
                    task,
                    item,
                    target_terms,
                )
            )
        )

    def project(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        baseline_values: tuple[float, ...],
        evidence: list[Evidence],
    ) -> RegimeProjection | None:
        period = diagnosis.seasonal_period
        sources = self._normalization_sources(task, evidence)
        if not sources or period is None or period < 2 or diagnosis.seasonal_strength < 0.75:
            return None
        required = 2 * period + 1
        if len(task.history_values) < required:
            return None

        recent = tuple(task.history_values[-required:])
        training = recent[:-period]
        validation = recent[-period:]
        validation_model = _fit_harmonic(training, period)
        if validation_model is None:
            return None
        validation_prediction = _predict(
            validation_model, range(len(training), len(recent)), period
        )
        validation_mae = statistics.fmean(
            abs(predicted - actual)
            for predicted, actual in zip(validation_prediction, validation)
        )
        seasonal_reference = recent[-2 * period : -period]
        seasonal_naive_mae = statistics.fmean(
            abs(previous - actual)
            for previous, actual in zip(seasonal_reference, validation)
        )
        if seasonal_naive_mae <= 1e-10:
            return None
        validation_gain = 1.0 - validation_mae / seasonal_naive_mae
        if validation_gain < 0.10:
            return None

        full_model = _fit_harmonic(recent, period)
        if full_model is None:
            return None
        harmonic_values = _predict(
            full_model,
            range(len(recent), len(recent) + task.prediction_length),
            period,
        )
        blend_weight = min(0.75, max(0.20, validation_gain))
        values = tuple(
            (1.0 - blend_weight) * baseline + blend_weight * harmonic
            for baseline, harmonic in zip(baseline_values, harmonic_values)
        )
        confidence = min(0.95, 0.65 + 0.30 * blend_weight)
        rationale = (
            f"Verified normalization evidence says the future should follow the recovered normal "
            f"seasonal regime. A {period}-step trend-harmonic model achieved validation MAE "
            f"{validation_mae:.6g} versus {seasonal_naive_mae:.6g} for seasonal naive; blend "
            f"weight={blend_weight:.3f}. Magnitudes use history only."
        )
        return RegimeProjection(
            values=values,
            source_document_ids=sources,
            confidence=confidence,
            seasonal_period=period,
            validation_mae=validation_mae,
            seasonal_naive_mae=seasonal_naive_mae,
            blend_weight=blend_weight,
            rationale=rationale,
        )
