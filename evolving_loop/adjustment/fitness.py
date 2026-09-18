"""Robust (CVaR) fitness for the adjustment / controller evolution.

Given per-task deltas (delta = toto_joint - policy_joint, so positive = improvement over
the base), the old objective penalized each regression by a fixed factor. That is ad hoc
and, on sparse data, rewards over-confident policies that win a few tasks while quietly
regressing others. CVaR (Conditional Value at Risk) is the principled alternative:
optimize the mean improvement, but heavily weight the WORST tail -- the tasks that
regress or gain least. This directly targets our two hard constraints (zero regression,
generalization) instead of hand-tuning a penalty factor.
"""
from __future__ import annotations

from statistics import mean
from typing import Sequence


def cvar_downside_fitness(deltas: Sequence[float], *, alpha: float = 0.3,
                          beta: float = 2.0) -> float:
    """Mean improvement plus a CVaR penalty on the worst-``alpha`` fraction.

    - ``alpha``: tail fraction treated as "worst case" (0.3 = worst 30% of tasks).
    - ``beta``: how heavily the downside tail is weighted (larger = more risk-averse).

    Only the *downside* tail penalizes (a positive tail — i.e. no regressions — adds
    nothing), so: (improve, no regression) > identity(all-zero → 0) > (improve, but
    regresses somewhere). Higher is better.
    """
    xs = [float(x) for x in deltas]
    if not xs:
        return 0.0
    m = mean(xs)
    ordered = sorted(xs)                       # ascending: worst first
    k = max(1, round(alpha * len(ordered)))
    tail = mean(ordered[:k])                    # CVaR: average of the worst alpha fraction
    return m + beta * min(0.0, tail)


def worst_case(deltas: Sequence[float]) -> float:
    """The single worst per-task delta (most negative = biggest regression). For audit."""
    xs = [float(x) for x in deltas]
    return min(xs) if xs else 0.0
