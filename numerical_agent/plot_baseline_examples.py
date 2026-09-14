"""Plot baseline forecast trajectories for selected Dr-CiK tasks."""
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

from common.data import DEFAULT_TASKS_FILE, Task, load_tasks, load_tasks_by_id


COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#8C564B", "#7F7F7F", "#17BECF", "#BCBD22",
)
LINESTYLES = ("-", "--", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)))


def _label(path: Path) -> str:
    stem = path.stem.removesuffix("_dev").removesuffix("_train").removesuffix("_test")
    names = {
        "qwen35_27b": "Qwen3.5 27B",
        "qwen35_9b": "Qwen3.5 9B",
        "gemini_3.1_flash_lite": "Gemini 3.1 Flash-Lite",
        "seasonal_naive": "Seasonal Naive",
    }
    return names.get(stem, stem.replace("_", " ").title())


def _load_forecasts(path: Path) -> dict[str, np.ndarray]:
    forecasts: dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            samples = np.asarray(row["samples"], dtype=float)
            if samples.ndim == 2 and samples.size and np.isfinite(samples).all():
                forecasts[str(row["benchmark_id"])] = samples
    return forecasts


def _partition_tasks(tasks_file: Path, split_file: Path, split: str) -> list[Task]:
    if split == "all":
        return load_tasks(tasks_file)
    partition = {"val": "dev", "test": "public_test"}.get(split, split)
    manifest = json.loads(split_file.read_text(encoding="utf-8"))
    task_ids = manifest["partitions"][partition]["task_ids"]
    return load_tasks_by_id(tasks_file, task_ids)


def _select_tasks(
    tasks: Sequence[Task], forecasts: dict[str, dict[str, np.ndarray]], count: int
) -> list[Task]:
    ranked: list[tuple[float, Task]] = []
    for task in tasks:
        points = np.stack([rows[task.task_id].mean(axis=0) for rows in forecasts.values()])
        scale = statistics.fmean(abs(value) for value in task.future_values) or 1.0
        ranked.append((float(np.mean(np.ptp(points, axis=0)) / scale), task))
    ranked.sort(key=lambda item: (item[0], item[1].task_id))
    if count >= len(ranked):
        return [task for _, task in ranked]
    indices = np.linspace(0, len(ranked) - 1, count).round().astype(int)
    return [ranked[index][1] for index in indices]


def plot_examples(
    forecast_paths: Sequence[Path],
    tasks: Sequence[Task],
    output: Path,
    *,
    num_plots: int,
    task_ids: Sequence[str] = (),
) -> list[str]:
    named = {_label(path): _load_forecasts(path) for path in forecast_paths}
    if len(named) != len(forecast_paths):
        raise ValueError("baseline labels must be unique")
    common_ids = set.intersection(*(set(rows) for rows in named.values()))
    compatible = [
        task for task in tasks
        if task.task_id in common_ids
        and task.future_values
        and all(rows[task.task_id].shape[1] == task.prediction_length for rows in named.values())
    ]
    if not compatible:
        raise ValueError("no compatible labeled forecast trajectories are shared by every baseline")
    if task_ids:
        wanted = set(task_ids)
        selected = [task for task in compatible if task.task_id in wanted]
        missing = wanted - {task.task_id for task in selected}
        if missing:
            raise ValueError(f"requested task IDs are unavailable: {', '.join(sorted(missing))}")
    else:
        if num_plots < 1:
            raise ValueError("--num-plots must be positive")
        selected = _select_tasks(compatible, named, min(num_plots, len(compatible)))

    columns = min(2, len(selected))
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(8.2 * columns, 4.25 * rows), squeeze=False)
    colors = {name: COLORS[index % len(COLORS)] for index, name in enumerate(named)}
    styles = {name: LINESTYLES[index % len(LINESTYLES)] for index, name in enumerate(named)}

    for axis, task in zip(axes.flat, selected):
        history = np.asarray(task.history_values, dtype=float)
        horizon = task.prediction_length
        context = min(len(history), max(100, 2 * horizon))
        history_x = np.arange(-context, 0)
        forecast_x = np.arange(horizon)
        axis.plot(history_x, history[-context:], color="#777777", lw=1.25, label="History")
        axis.plot(forecast_x, task.future_values, color="#111111", lw=2.1, label="Actual")
        for name, baseline in named.items():
            axis.plot(
                forecast_x,
                baseline[task.task_id].mean(axis=0),
                color=colors[name],
                linestyle=styles[name],
                lw=1.55,
                label=name,
            )
        axis.axvline(-0.5, color="#9a9a9a", linestyle=":", lw=1)
        axis.set_title(f"{task.task_id} · {task.entity_name}\n{task.frequency} · horizon {horizon}", fontsize=10)
        axis.set_xlabel("Steps from forecast origin")
        axis.set_ylabel("Value")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)
    fig.suptitle("Baseline forecast examples", y=1.04, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.91), h_pad=2.2, w_pad=1.7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [task.task_id for task in selected]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecasts", nargs="+", type=Path)
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    parser.add_argument("--split-file", type=Path, default=Path("splits/drcik_public_80_20_99_v1.json"))
    parser.add_argument("--split", default="all", choices=("all", "train", "dev", "val", "test"))
    parser.add_argument("--num-plots", type=int, default=6)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("runs/baselines/baseline_forecast_examples.png"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = plot_examples(
        args.forecasts,
        _partition_tasks(args.tasks_file, args.split_file, args.split),
        args.output,
        num_plots=args.num_plots,
        task_ids=args.task_id,
    )
    print(f"wrote {args.output}; selected tasks: {', '.join(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
