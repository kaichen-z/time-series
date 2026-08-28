"""The Dr-CiK scaled metrics (sMAE, sRMSE) as a curation MetricFunction.

Each task's error is divided by the mean absolute value of that task's truth over the forecast
horizon, so a large-magnitude series and a small one weigh the same.
"""
from __future__ import annotations

import math
from typing import Sequence

from common.metrics import mean_absolute_truth


class ScaledMetric:
    """sMAE or sRMSE, scaled by the mean absolute truth over the horizon."""

    def __init__(self, kind: str):
        if kind not in ("smae", "srmse"):
            raise ValueError(f"unsupported scaled metric {kind!r}")
        self.kind = kind

    def _raw(self, prediction: Sequence[float], truth: Sequence[float]) -> float:
        if len(prediction) != len(truth):
            raise ValueError(f"length mismatch: {len(truth)} truth vs {len(prediction)} predicted")
        if not truth:
            return 0.0
        if self.kind == "smae":
            return sum(abs(t - p) for t, p in zip(truth, prediction)) / len(truth)
        return (sum((t - p) ** 2 for t, p in zip(truth, prediction)) / len(truth)) ** 0.5

    def __call__(self, prediction: Sequence[float], truth: Sequence[float]) -> float:
        value = self._raw(prediction, truth) / mean_absolute_truth([float(t) for t in truth])
        # A non-finite error is a failed method, not a merely bad one: rank it last outright.
        return value if math.isfinite(value) else math.inf
