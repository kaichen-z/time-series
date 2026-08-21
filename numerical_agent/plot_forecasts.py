"""Plot every forecasting method in an evolved methods module against Dr-CiK tasks.

One PNG per task: history, the trusted future, and every method's forecast overlaid on
shared axes so the methods are directly comparable, each method's own subplot annotated
with its MAE and MSE against the trusted future.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator

from common.data import Task, load_tasks
from common.metrics import mae, mse
from common.payload import read_json_object
from numerical_agent.evolution.execution import load_methods

INK = "#1c2333"
PAPER = "#fbfaf7"
PANEL = "#ffffff"
GRID = "#e7e4dc"
HISTORY = "#5b6472"
FUTURE = "#1c8c6f"
FORECAST = "#e0632b"
BAD = "#b23b4a"
MUTED = "#9096a1"

_FONT_CANDIDATES = ("Arimo", "Carlito", "DejaVu Sans")


def _pick_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _FONT_CANDIDATES:
        if name in available:
            return name
    return "sans-serif"


plt.rcParams.update({
    "font.family": _pick_font(),
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1.0,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def _run_method(
    function: Callable, task: Task, not_applicable: type
) -> tuple[str, list[float] | None, str]:
    """Call one method on one task and classify what came back, mirroring the evolution scorer."""
    try:
        raw = function(list(task.history_values), task.prediction_length, task.frequency)
    except not_applicable as exc:
        return "not_applicable", None, str(exc)
    except BaseException as exc:  # noqa: BLE001 - a broken method must not kill the whole plot
        return "crashed", None, f"{type(exc).__name__}: {exc}"
    try:
        forecast = [float(v) for v in raw]
    except (TypeError, ValueError) as exc:
        return "invalid", None, f"unreadable forecast: {exc}"
    if len(forecast) != task.prediction_length or not all(math.isfinite(v) for v in forecast):
        return "invalid", None, "wrong shape or non-finite value"
    return "success", forecast, ""


def _grid_shape(count: int) -> tuple[int, int]:
    """Pick a roughly-square grid, capping width so subplots stay readable."""
    ncols = min(4, max(1, math.ceil(math.sqrt(count))))
    nrows = math.ceil(count / ncols)
    return nrows, ncols


def plot_task_forecasts(
    task: Task,
    functions: Mapping[str, Callable],
    not_applicable: type,
    output_dir: Path,
    *,
    dpi: int = 160,
) -> Path:
    """Render one PNG for one task: every method's forecast in its own panel, shared x-axis."""
    names = sorted(functions)
    nrows, ncols = _grid_shape(len(names))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows), squeeze=False,
    )
    flat_axes = [ax for row in axes for ax in row]

    history = list(task.history_values)
    future = list(task.future_values)
    has_future = bool(future)
    history_x = list(range(len(history)))
    future_x = list(range(len(history), len(history) + len(future))) if has_future \
        else list(range(len(history), len(history) + task.prediction_length))

    for ax, name in zip(flat_axes, names):
        status, forecast, detail = _run_method(functions[name], task, not_applicable)

        ax.plot(history_x, history, color=HISTORY, linewidth=1.4, zorder=2)
        ax.axvline(len(history) - 0.5, color=GRID, linewidth=1.2, linestyle=(0, (2, 2)), zorder=1)

        if has_future:
            ax.plot(
                [history_x[-1], *future_x], [history[-1], *future],
                color=FUTURE, linewidth=1.6, label="actual", zorder=3,
            )

        if status == "success":
            ax.plot(
                [history_x[-1], *future_x], [history[-1], *forecast],
                color=FORECAST, linewidth=1.6, linestyle=(0, (5, 1.5)), label="forecast", zorder=4,
            )
            subtitle = (
                f"MAE {mae(future, forecast):.3g}  ·  MSE {mse(future, forecast):.3g}"
                if has_future else "forecast only"
            )
            title_color = INK
        else:
            label = {"not_applicable": "not applicable", "crashed": "crashed", "invalid": "invalid"}[status]
            ax.text(
                0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center",
                color=BAD, fontsize=10, fontweight="bold", alpha=0.85,
            )
            subtitle = detail[:48] + ("…" if len(detail) > 48 else "") if detail else label
            title_color = BAD

        ax.set_title(name, fontsize=10, fontweight="bold", pad=14)
        ax.text(
            0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color=title_color, alpha=0.75,
        )
        ax.grid(True, alpha=0.6, linewidth=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)

    for ax in flat_axes[len(names):]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig_height = 3.3 * nrows
    footer_in = 0.45
    header_in = 0.85
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=len(labels), frameon=False,
            bbox_to_anchor=(0.5, (footer_in * 0.35) / fig_height), fontsize=10,
        )

    fig.suptitle(
        f"{task.task_id} · {task.entity_name}",
        fontsize=15, fontweight="bold", y=1 - (header_in * 0.35) / fig_height,
    )
    fig.text(
        0.5, 1 - (header_in * 0.68) / fig_height,
        f"{task.frequency} · horizon {task.prediction_length}"
        + (f" · seasonal period {task.seasonal_period}" if task.seasonal_period else ""),
        ha="center", fontsize=10, color=MUTED,
    )

    fig.tight_layout(rect=(0, footer_in / fig_height, 1, 1 - header_in / fig_height))
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{task.task_id}.png"
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_all(
    methods_path: str | Path,
    tasks: Sequence[Task],
    output_dir: str | Path,
    *,
    dpi: int = 160,
) -> list[Path]:
    """Plot every task against every method in the module; return the written PNG paths."""
    _module, functions = load_methods(methods_path)
    not_applicable = _module.NotApplicable
    destination = Path(output_dir)
    return [
        plot_task_forecasts(task, functions, not_applicable, destination, dpi=dpi)
        for task in tasks
    ]


def _select_tasks(
    tasks_file: str, split_file: str | None, partition: str | None, limit: int | None
) -> list[Task]:
    catalog = load_tasks(tasks_file)
    if split_file and partition:
        payload = read_json_object(split_file)
        ids = set(payload["partitions"][partition]["task_ids"])  # type: ignore[index]
        catalog = [task for task in catalog if task.task_id in ids]
    if limit is not None:
        catalog = catalog[:limit]
    return catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", required=True, help="path to an evolved methods.py")
    parser.add_argument(
        "--tasks-file",
        default="/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK/data/tasks/train.jsonl",
    )
    parser.add_argument("--split-file", default="splits/drcik_public_80_20_99_v1.json")
    parser.add_argument("--partition", default="train", choices=("train", "dev", "public_test", "none"))
    parser.add_argument("--output-dir", default="runs/method_evolution/v001/forecast_plots")
    parser.add_argument("--limit", type=int, default=None, help="plot at most this many tasks")
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    partition = None if args.partition == "none" else args.partition
    tasks = _select_tasks(args.tasks_file, args.split_file, partition, args.limit)
    if not tasks:
        print("no tasks selected")
        return 1
    written = plot_all(args.methods, tasks, args.output_dir, dpi=args.dpi)
    print(f"wrote {len(written)} plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
