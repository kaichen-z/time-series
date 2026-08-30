"""Trusted Parent/Child acceptance gates."""
from __future__ import annotations

from dataclasses import dataclass

from common.metrics import pareto_scaled_improvement

from .contracts import EvaluationReport, MetricSpec


@dataclass(frozen=True)
class MetricAcceptanceGate:
    """Accept only strict improvement on one configured metric."""

    metric: MetricSpec
    margin: float = 0.0

    def __post_init__(self) -> None:
        if self.margin < 0:
            raise ValueError("margin must be non-negative")

    def accept(
        self, parent_report: EvaluationReport, child_report: EvaluationReport
    ) -> bool:
        try:
            parent_score = parent_report.metrics[self.metric.name]
            child_score = child_report.metrics[self.metric.name]
        except KeyError as exc:
            raise ValueError(
                f"evaluation report is missing primary metric {self.metric.name!r}"
            ) from exc
        return self.metric.better(child_score, parent_score, self.margin)


@dataclass(frozen=True)
class ScaledPairAcceptanceGate:
    """Accept only Pareto improvement of the canonical scaled metric pair."""

    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")

    def accept(
        self, parent_report: EvaluationReport, child_report: EvaluationReport
    ) -> bool:
        try:
            parent_smae = float(parent_report.metrics["smae"])
            parent_srmse = float(parent_report.metrics["srmse"])
            child_smae = float(child_report.metrics["smae"])
            child_srmse = float(child_report.metrics["srmse"])
        except KeyError as error:
            raise ValueError(
                "evaluation report is missing the canonical scaled metric pair"
            ) from error
        return pareto_scaled_improvement(
            parent_smae,
            parent_srmse,
            child_smae,
            child_srmse,
            tolerance=self.tolerance,
        )
