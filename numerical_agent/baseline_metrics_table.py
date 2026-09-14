"""Score labeled baseline forecasts and write a publication-ready LaTeX table."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Sequence

from common.data import DEFAULT_TASKS_FILE, load_tasks, load_tasks_by_id
from common.metrics import drcik_point_metrics


def _method_name(path: Path) -> str:
    stem = path.stem.removesuffix("_dev").removesuffix("_all")
    return {
        "qwen35_27b": "Qwen3.5 27B",
        "qwen35_9b": "Qwen3.5 9B",
    }.get(stem, stem.replace("_", " ").title())


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def score_file(path: Path, truths: dict[str, tuple[float, ...]]) -> dict[str, object]:
    """Score every compatible labeled row using capped Dr-CiK point metrics."""
    task_metrics: list[dict[str, float | bool]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            task_id = str(payload["benchmark_id"])
            if task_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate benchmark_id {task_id}")
            seen.add(task_id)
            truth = truths.get(task_id)
            if truth is None:
                continue
            samples = payload.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"{path}:{line_number}: samples must be a non-empty list")
            metric = drcik_point_metrics(truth, samples)
            task_metrics.append(metric)
    if not task_metrics:
        raise ValueError(f"{path}: no compatible labeled forecasts")
    smae_values = [float(row["smae"]) for row in task_metrics]
    srmse_values = [float(row["srmse"]) for row in task_metrics]
    return {
        "method": _method_name(path),
        "tasks": len(task_metrics),
        "smae": statistics.fmean(smae_values),
        "smae_se": (
            statistics.stdev(smae_values) / math.sqrt(len(smae_values))
            if len(smae_values) > 1 else 0.0
        ),
        "srmse": statistics.fmean(srmse_values),
        "srmse_se": (
            statistics.stdev(srmse_values) / math.sqrt(len(srmse_values))
            if len(srmse_values) > 1 else 0.0
        ),
        "smae_clipped": sum(bool(row["smae_clipped"]) for row in task_metrics),
        "srmse_clipped": sum(bool(row["srmse_clipped"]) for row in task_metrics),
    }


def render_latex(rows: Sequence[dict[str, object]], split_label: str) -> str:
    """Render rows sorted by sMAE, bolding each metric's best value."""
    if not rows:
        raise ValueError("at least one result row is required")
    ordered = sorted(rows, key=lambda row: (float(row["smae"]), str(row["method"])))
    best_smae = min(float(row["smae"]) for row in ordered)
    best_srmse = min(float(row["srmse"]) for row in ordered)

    def metric(value: float, standard_error: float, best: float) -> str:
        formatted = rf"{value:.3f} \pm {standard_error:.3f}"
        return rf"\mathbf{{{formatted}}}" if math.isclose(value, best, abs_tol=5e-7) else formatted

    lines = [
        r"% Requires \usepackage{booktabs}.",
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{Baseline point-forecast accuracy (mean $\pm$ standard error) on {split_label} tasks. Forecast trajectories are reduced by their step-wise mean; task-level sMAE and sRMSE are clipped at 5 before averaging.}}",
        r"  \label{tab:baseline-point-metrics}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Method & Tasks & sMAE $\downarrow$ & sRMSE $\downarrow$ \\",
        r"    \midrule",
    ]
    for row in ordered:
        lines.append(
            "    "
            + _latex_escape(str(row["method"]))
            + f" & {int(row['tasks'])}"
            + f" & ${metric(float(row['smae']), float(row['smae_se']), best_smae)}$"
            + f" & ${metric(float(row['srmse']), float(row['srmse_se']), best_srmse)}$"
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _default_forecasts(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*_dev.jsonl")
        if "hidden" not in path.name.lower()
    )


def _truths_for_split(tasks_file: Path, split_file: Path, split: str) -> tuple[dict[str, tuple[float, ...]], str]:
    if split == "all":
        tasks = load_tasks(tasks_file)
        return {task.task_id: task.future_values for task in tasks}, "all labeled"
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    partitions = payload.get("partitions", {})
    if not isinstance(partitions, dict):
        raise ValueError(f"{split_file} has no partitions object")

    if split == "train_val":
        names = ("train", "dev") if "dev" in partitions else ("train",)
        split_label = "combined frozen Train and validation"
    elif split in {"dev", "val"}:
        names = ("dev",)
        split_label = "frozen Dev"
    elif split == "test":
        names = ("test",) if "test" in partitions else ("public_test",)
        split_label = "frozen public-test"
    else:
        names = ("train",)
        split_label = (
            "combined frozen Train and validation"
            if "dev" not in partitions
            else "frozen Train"
        )

    task_ids: list[str] = []
    for name in names:
        selected = partitions.get(name, {})
        selected_ids = selected.get("task_ids", []) if isinstance(selected, dict) else []
        if not isinstance(selected_ids, list) or not selected_ids:
            raise ValueError(f"split {name!r} has no task IDs in {split_file}")
        task_ids.extend(str(task_id) for task_id in selected_ids)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"selected partitions overlap in {split_file}")
    tasks = load_tasks_by_id(tasks_file, task_ids)
    return {task.task_id: task.future_values for task in tasks}, split_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecasts", nargs="*", type=Path)
    parser.add_argument("--baseline-dir", type=Path, default=Path("runs/baselines"))
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "train", "train_val", "dev", "val", "test"),
    )
    parser.add_argument(
        "--split-file", type=Path, default=Path("splits/dr_cik_train_100_test_99.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/baselines/baseline_metrics_table.tex")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = args.forecasts or _default_forecasts(args.baseline_dir)
    paths = [
        path
        for path in paths
        if "hidden" not in path.name.lower()
    ]
    if not paths:
        raise ValueError("no non-hidden forecast files selected")
    truths, split_label = _truths_for_split(args.tasks_file, args.split_file, args.split)
    rows = [score_file(path, truths) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_latex(rows, split_label), encoding="utf-8")
    print(f"wrote {args.output} from {len(rows)} methods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
