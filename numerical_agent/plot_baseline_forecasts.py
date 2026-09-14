"""Compare probabilistic baseline forecasts on a shared set of labeled tasks."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import DEFAULT_TASKS_FILE, Task, load_tasks_by_id
from common.metrics import drcik_point_metrics


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
LINESTYLES = ("-", "--", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)))


def _label(path: Path) -> str:
    return path.stem.removesuffix("_dev").replace("_dp", "").replace("_", " ").title()


def _load_forecasts(path: Path) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            samples = np.asarray(payload["samples"], dtype=float)
            if samples.ndim != 2 or samples.size == 0 or not np.isfinite(samples).all():
                continue
            rows[str(payload["benchmark_id"])] = samples
    return rows


def _select_tasks(
    tasks: Sequence[Task], forecasts: dict[str, dict[str, np.ndarray]], count: int
) -> list[Task]:
    """Span the observed range of normalized point-forecast disagreement."""
    ranked: list[tuple[float, Task]] = []
    for task in tasks:
        points = np.stack([rows[task.task_id].mean(axis=0) for rows in forecasts.values()])
        scale = statistics.fmean(abs(value) for value in task.future_values) or 1.0
        disagreement = float(np.mean(np.ptp(points, axis=0)) / scale)
        ranked.append((disagreement, task))
    ranked.sort(key=lambda item: (item[0], item[1].task_id))
    if count >= len(ranked):
        return [task for _, task in ranked]
    indices = np.linspace(0, len(ranked) - 1, count).round().astype(int)
    return [ranked[index][1] for index in indices]


def _aggregate_smae(
    tasks: Sequence[Task], forecasts: dict[str, dict[str, np.ndarray]]
) -> dict[str, tuple[float, float]]:
    summary: dict[str, tuple[float, float]] = {}
    for name, rows in forecasts.items():
        scores = [
            float(drcik_point_metrics(task.future_values, rows[task.task_id].tolist())["smae"])
            for task in tasks
        ]
        mean = statistics.fmean(scores)
        se = statistics.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
        summary[name] = (mean, 1.96 * se)
    return summary


def plot_comparison(
    forecast_paths: Sequence[Path],
    tasks_file: Path,
    output: Path,
    task_count: int = 6,
    excluded_task_ids: Sequence[str] = (),
) -> list[str]:
    """Write one aggregate-and-example comparison figure."""
    named = {_label(path): _load_forecasts(path) for path in forecast_paths}
    common_ids = set.intersection(*(set(rows) for rows in named.values()))
    tasks = load_tasks_by_id(tasks_file, sorted(common_ids))
    tasks = [
        task
        for task in tasks
        if task.future_values
        and all(rows[task.task_id].shape[1] == len(task.future_values) for rows in named.values())
    ]
    if not tasks:
        raise ValueError("no labeled tasks with compatible forecasts are shared by every baseline")

    eligible = [task for task in tasks if task.task_id not in set(excluded_task_ids)]
    selected = _select_tasks(eligible, named, min(task_count, len(eligible)))
    scores = _aggregate_smae(tasks, named)
    labels = list(named)
    colors = {name: COLORS[index % len(COLORS)] for index, name in enumerate(labels)}
    styles = {name: LINESTYLES[index % len(LINESTYLES)] for index, name in enumerate(labels)}

    fig = plt.figure(figsize=(15, 12), facecolor="white")
    grid = fig.add_gridspec(4, 2, height_ratios=(0.95, 1.0, 1.0, 1.0))
    aggregate = fig.add_subplot(grid[0, :])
    x = np.arange(len(labels))
    means = [scores[name][0] for name in labels]
    errors = [scores[name][1] for name in labels]
    aggregate.bar(x, means, color=[colors[name] for name in labels], width=0.65)
    aggregate.errorbar(x, means, yerr=errors, fmt="none", ecolor="#222222", capsize=4, lw=1.2)
    aggregate.set_xticks(x, labels)
    aggregate.set_ylabel("Mean capped sMAE")
    aggregate.set_title(f"Aggregate accuracy on {len(tasks)} shared labeled dev tasks (95% normal CI)")
    aggregate.grid(axis="y", alpha=0.25)
    aggregate.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(means):
        aggregate.text(index, value + errors[index] + 0.02, f"{value:.3f}", ha="center", fontsize=9)

    for ax, task in zip((fig.add_subplot(grid[row, col]) for row in range(1, 4) for col in range(2)), selected):
        horizon = len(task.future_values)
        history = np.asarray(task.history_values, dtype=float)
        context = min(len(history), max(2 * horizon, 100))
        history_x = np.arange(-context, 0)
        forecast_x = np.arange(horizon)
        ax.plot(history_x, history[-context:], color="#666666", lw=1.25, label="History")
        ax.plot(forecast_x, task.future_values, color="#111111", lw=2.1, label="Truth")
        for name, rows in named.items():
            samples = rows[task.task_id]
            point = samples.mean(axis=0)
            if samples.shape[0] > 1 and np.any(np.ptp(samples, axis=0) > 0):
                lower, upper = np.quantile(samples, (0.1, 0.9), axis=0)
                ax.fill_between(forecast_x, lower, upper, color=colors[name], alpha=0.09, linewidth=0)
            score = float(drcik_point_metrics(task.future_values, samples.tolist())["smae"])
            ax.plot(
                forecast_x,
                point,
                color=colors[name],
                linestyle=styles[name],
                lw=1.55,
                label=f"{name} ({score:.2f})",
            )
        ax.axvline(-0.5, color="#999999", linestyle=":", lw=1)
        ax.set_title(f"{task.task_id} · {task.entity_name} · {task.frequency}", fontsize=10)
        ax.set_xlabel("Steps from forecast origin")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    handles, legend_labels = fig.axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.932),
        ncol=min(6, len(legend_labels)),
        frameon=False,
    )
    fig.suptitle(
        "Baseline forecast comparison",
        fontsize=17,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.963,
        "Point forecasts are trajectory means; translucent bands show the 10th–90th sample percentiles. "
        "Panels span quantiles of cross-baseline disagreement; legend values are task sMAE.",
        ha="center",
        va="top",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.89), h_pad=1.35, w_pad=1.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [task.task_id for task in selected]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecasts", nargs="+", type=Path)
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    parser.add_argument("--output", type=Path, default=Path("runs/baselines/baseline_forecast_comparison.png"))
    parser.add_argument("--task-count", type=int, default=6)
    parser.add_argument("--exclude-task", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = plot_comparison(
        args.forecasts, args.tasks_file, args.output, args.task_count, args.exclude_task
    )
    print(f"wrote {args.output}; selected tasks: {', '.join(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
