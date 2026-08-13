from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastUtilityLabel:
    """Offline supervision for a document or retrieval action.

    This object must only be created from chronologically resolved training or
    validation tasks.  It is deliberately not imported by the online loop.
    """

    task_id: str
    document_id: str
    error_before: float
    error_after: float
    forecast_gain: float
    latency_cost: float
    redundancy_cost: float
    token_cost: float
    net_utility: float
    beneficial: bool


class ForecastUtilityLabeler:
    """Create Agentic-R-style global utility labels from forecast outcomes."""

    @staticmethod
    def mae(prediction: tuple[float, ...], actual: tuple[float, ...]) -> float:
        if len(prediction) != len(actual) or not prediction:
            raise ValueError("prediction and actual must have the same non-zero length")
        return statistics.fmean(abs(left - right) for left, right in zip(prediction, actual))

    def label(
        self,
        *,
        task_id: str,
        document_id: str,
        forecast_before: tuple[float, ...],
        forecast_after: tuple[float, ...],
        actual: tuple[float, ...],
        latency_cost: float = 0.0,
        redundancy_cost: float = 0.0,
        token_cost: float = 0.0,
    ) -> ForecastUtilityLabel:
        for name, value in (
            ("latency_cost", latency_cost),
            ("redundancy_cost", redundancy_cost),
            ("token_cost", token_cost),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        before = self.mae(forecast_before, actual)
        after = self.mae(forecast_after, actual)
        gain = before - after
        net = gain - latency_cost - redundancy_cost - token_cost
        return ForecastUtilityLabel(
            task_id=task_id,
            document_id=document_id,
            error_before=before,
            error_after=after,
            forecast_gain=gain,
            latency_cost=latency_cost,
            redundancy_cost=redundancy_cost,
            token_cost=token_cost,
            net_utility=net,
            beneficial=net > 0.0,
        )
