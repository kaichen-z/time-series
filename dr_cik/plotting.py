"""Renders history + sample trajectories + mean forecast per task, from one or more forecasts.jsonl files."""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import ForecastTask

# Colourblind-safe categorical palette, checked for adjacent-pair separation on a light surface.
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_BLUE = "#2a78d6"  # single-run forecast: samples + mean
_RED = "#e34948"  # ground truth (reserved: never assigned to a method in a comparison plot)
_METHOD_HUES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # categorical slots 1-5, fixed order


def _parse_timestamp(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized timestamp format: {value!r}")


def _new_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    return plt, fig, ax


def _style_and_save(plt, fig, ax, task: ForecastTask, title: str, history_x: list[datetime], output_path: str | Path) -> None:
    import matplotlib.dates as mdates

    ax.axvline(history_x[-1], color=_BASELINE, linewidth=1, linestyle=":", zorder=1)
    ax.set_title(title, color=_INK_PRIMARY, fontsize=11)
    ax.set_xlabel("Time", color=_INK_SECONDARY)
    ax.set_ylabel(task.target_name, color=_INK_SECONDARY)
    ax.tick_params(colors=_INK_MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_BASELINE)
    ax.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.legend(frameon=False, fontsize=8, labelcolor=_INK_SECONDARY, loc="upper left")

    fig.tight_layout()
    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved, facecolor=_SURFACE)
    plt.close(fig)


def plot_task_samples(
    task: ForecastTask,
    samples: tuple[tuple[float, ...], ...],
    label: str,
    output_path: str | Path,
) -> None:
    """Draw history, all sample trajectories, the mean forecast, and ground truth (if public) for one task."""
    plt, fig, ax = _new_figure()

    history_x = [_parse_timestamp(t) for t in task.history_timestamps]
    future_x = [_parse_timestamp(t) for t in task.future_timestamps]
    horizon = len(future_x)
    mean = [statistics.fmean(sample[step] for sample in samples) for step in range(horizon)]

    for sample in samples:
        ax.plot(future_x, sample, color=_BLUE, alpha=0.15, linewidth=0.8, zorder=2)
    ax.plot([], [], color=_BLUE, alpha=0.4, linewidth=0.8, label=f"Samples (S={len(samples)})")

    ax.plot(history_x, task.history_values, color=_INK_PRIMARY, linewidth=2, label="History", zorder=3)
    ax.plot(future_x, mean, color=_BLUE, linewidth=2.5, label="Forecast mean", zorder=4)

    if task.future_values is not None:
        ax.plot(future_x, task.future_values, color=_RED, linewidth=2, linestyle="--", label="Ground truth", zorder=4)

    _style_and_save(plt, fig, ax, task, title=f"{task.benchmark_id} — {label}", history_x=history_x, output_path=output_path)


def plot_task_comparison(
    task: ForecastTask,
    series: list[tuple[str, tuple[tuple[float, ...], ...]]],
    output_path: str | Path,
) -> None:
    """Overlay history + each method's mean forecast (with a light min/max band) + ground truth, for direct comparison."""
    if len(series) > len(_METHOD_HUES):
        raise ValueError(f"plot_task_comparison supports at most {len(_METHOD_HUES)} series, got {len(series)}")
    plt, fig, ax = _new_figure()

    history_x = [_parse_timestamp(t) for t in task.history_timestamps]
    future_x = [_parse_timestamp(t) for t in task.future_timestamps]
    horizon = len(future_x)

    ax.plot(history_x, task.history_values, color=_INK_PRIMARY, linewidth=2, label="History", zorder=3)

    for (label, samples), hue in zip(series, _METHOD_HUES):
        mean = [statistics.fmean(sample[step] for sample in samples) for step in range(horizon)]
        lower = [min(sample[step] for sample in samples) for step in range(horizon)]
        upper = [max(sample[step] for sample in samples) for step in range(horizon)]
        ax.fill_between(future_x, lower, upper, color=hue, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(future_x, mean, color=hue, linewidth=2.5, label=label, zorder=4)

    if task.future_values is not None:
        ax.plot(future_x, task.future_values, color=_RED, linewidth=2, linestyle="--", label="Ground truth", zorder=5)

    _style_and_save(plt, fig, ax, task, title=f"{task.benchmark_id} — method comparison", history_x=history_x, output_path=output_path)


def _read_forecasts(forecasts_path: str | Path) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Read a forecasts.jsonl into benchmark_id -> samples."""
    samples_by_id: dict[str, tuple[tuple[float, ...], ...]] = {}
    with Path(forecasts_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            samples = tuple(tuple(float(value) for value in sample) for sample in row["samples"])
            samples_by_id[str(row["benchmark_id"])] = samples
    return samples_by_id


def plot_forecasts_file(tasks: Iterable[ForecastTask], forecasts_path: str | Path, output_dir: str | Path, label: str) -> list[Path]:
    """Read a forecasts.jsonl (benchmark_id -> samples) and render one PNG per task it covers."""
    tasks_by_id = {task.benchmark_id: task for task in tasks}
    output = Path(output_dir).expanduser().resolve()
    written: list[Path] = []
    for benchmark_id, samples in _read_forecasts(forecasts_path).items():
        task = tasks_by_id.get(benchmark_id)
        if task is None:
            continue
        path = output / f"{benchmark_id}.png"
        plot_task_samples(task, samples, label=label, output_path=path)
        written.append(path)
    return written


def plot_comparison_files(tasks: Iterable[ForecastTask], series: list[tuple[str, str | Path]], output_dir: str | Path) -> list[Path]:
    """Overlay several forecasts.jsonl runs (e.g. Chronos vs multiple Direct-Prompt models) per task they all cover."""
    labels = [label for label, _ in series]
    if len(set(labels)) != len(labels):
        raise ValueError(f"--series labels must be unique, got {labels}")
    tasks_by_id = {task.benchmark_id: task for task in tasks}
    samples_by_label = {label: _read_forecasts(path) for label, path in series}
    common_ids = set.intersection(*(set(samples) for samples in samples_by_label.values())) if samples_by_label else set()

    output = Path(output_dir).expanduser().resolve()
    written: list[Path] = []
    for benchmark_id in sorted(common_ids):
        task = tasks_by_id.get(benchmark_id)
        if task is None:
            continue
        row_series = [(label, samples_by_label[label][benchmark_id]) for label, _ in series]
        path = output / f"{benchmark_id}.png"
        plot_task_comparison(task, row_series, output_path=path)
        written.append(path)
    return written
