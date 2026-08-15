"""Trusted Parent/Child acceptance gates."""
from __future__ import annotations

from dataclasses import dataclass

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
