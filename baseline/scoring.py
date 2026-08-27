"""Score sampled forecasts the way the Dr-CiK paper does: point metrics on the sample mean."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from common.metrics import ROUND_DIGITS, scaled_mae, scaled_rmse

# The paper winsorizes per-task scaled errors before averaging (appendix H.2). Without it a
# single near-zero-magnitude series dominates the mean and the table stops meaning anything.
WINSOR_CAP = 5.0


@dataclass(frozen=True)
class TaskScore:
    """One task's point metrics, or the reason it could not be scored."""

    benchmark_id: str
    smae: float | None
    srmse: float | None
    failure: str = ""


@dataclass(frozen=True)
class Report:
    """Aggregate over one baseline's run."""

    baseline: str
    tasks: int
    scored: int
    mean_smae: float | None
    mean_srmse: float | None
    coverage: float
    sample_failures: tuple[str, ...] = ()


def sample_mean(samples: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Collapse trajectories to the point forecast the paper scores: the per-step mean."""
    if not samples:
        raise ValueError("no sample trajectories to average")
    horizon = len(samples[0])
    if any(len(path) != horizon for path in samples):
        raise ValueError("sample trajectories differ in length")
    return tuple(statistics.fmean(path[step] for path in samples) for step in range(horizon))


def score_task(
    benchmark_id: str, truth: Sequence[float], samples: Sequence[Sequence[float]]
) -> TaskScore:
    """Score one task, recording a failure rather than raising so one bad task cannot end a run."""
    try:
        point = sample_mean(samples)
        if len(point) != len(truth):
            raise ValueError(f"forecast has {len(point)} steps, expected {len(truth)}")
        truth = [float(value) for value in truth]
        return TaskScore(
            benchmark_id,
            min(scaled_mae(truth, list(point)), WINSOR_CAP),
            min(scaled_rmse(truth, list(point)), WINSOR_CAP),
        )
    except (ValueError, TypeError, OverflowError) as exc:
        return TaskScore(benchmark_id, None, None, f"{type(exc).__name__}: {exc}"[:200])


def summarize(baseline: str, scores: Sequence[TaskScore]) -> Report:
    """Average the scored tasks, keeping coverage visible beside the means.

    A baseline that fails on its hardest tasks looks strong if only its successes are averaged,
    so coverage belongs in the same table as the metric, never in a separate log line.
    """
    scored = [score for score in scores if score.smae is not None]
    failures = tuple(dict.fromkeys(s.failure for s in scores if s.failure))[:3]

    def mean(values) -> float | None:
        collected = list(values)
        return round(statistics.fmean(collected), ROUND_DIGITS) if collected else None

    return Report(
        baseline=baseline,
        tasks=len(scores),
        scored=len(scored),
        mean_smae=mean(score.smae for score in scored),
        mean_srmse=mean(score.srmse for score in scored),
        coverage=round(len(scored) / len(scores), ROUND_DIGITS) if scores else 0.0,
        sample_failures=failures,
    )


def render(reports: Sequence[Report]) -> str:
    """Render reports as a markdown table."""
    lines = [
        "| baseline | sMAE | sRMSE | coverage | scored |",
        "|---|---|---|---|---|",
    ]
    for report in reports:
        smae = "-" if report.mean_smae is None else f"{report.mean_smae:.3f}"
        srmse = "-" if report.mean_srmse is None else f"{report.mean_srmse:.3f}"
        lines.append(
            f"| {report.baseline} | {smae} | {srmse} | "
            f"{report.coverage:.3f} | {report.scored}/{report.tasks} |"
        )
    return "\n".join(lines)
