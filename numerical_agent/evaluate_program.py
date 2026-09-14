"""Score one evolved forecasting program on the Train and Test partitions and plot its forecasts."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from common.data import Task, load_tasks_by_id
from common.metrics import ROUND_DIGITS
from numerical_agent.program_evolution import EvolutionRun, ProgramGrader

DEFAULT_SPLIT_FILE = Path(__file__).parents[1] / "splits" / "dr_cik_train_100_test_99.jsonl"
PARTITIONS = ("train", "test")


@dataclass(frozen=True)
class Scored:
    """One program evaluated on one partition."""

    label: str
    partition: str
    commit_hash: str
    metrics: dict[str, object]
    cases: tuple[dict[str, object], ...]


def _round(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def partition_task_ids(split_file: Path, partition: str) -> tuple[str, ...]:
    """Read one partition's authorized task IDs from the split manifest."""
    manifest = json.loads(Path(split_file).read_text(encoding="utf-8"))
    try:
        task_ids = manifest["partitions"][partition]["task_ids"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"split manifest has no {partition} task IDs") from exc
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError(f"{partition} task IDs must be a non-empty list")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"{partition} task IDs must be unique")
    return tuple(str(task_id) for task_id in task_ids)


def load_partition(split_file: Path, tasks_file: Path, partition: str) -> tuple[Task, ...]:
    """Load exactly the tasks named by one partition, in manifest order."""
    task_ids = partition_task_ids(split_file, partition)
    tasks = tuple(load_tasks_by_id(tasks_file, task_ids))
    if tuple(task.task_id for task in tasks) != task_ids:
        raise ValueError(f"{partition} tasks are missing or misordered")
    return tasks


def resolve_commit(run: EvolutionRun, reference: str) -> tuple[str, str]:
    """Resolve 'best', 'seed', or an explicit hash to (commit_hash, label)."""
    ranked = run.attempts.parent_candidates()
    if not ranked:
        raise ValueError("the run has no evaluated attempt to score")
    if reference == "best":
        return str(ranked[0]["commit_hash"]), "best"
    if reference == "seed":
        seeds = [r for r in ranked if r.get("parent_hash") is None]
        if not seeds:
            raise ValueError("the run has no recorded seed attempt")
        return str(seeds[0]["commit_hash"]), "seed"
    matches = [r for r in ranked if str(r["commit_hash"]).startswith(reference.lower())]
    if len(matches) != 1:
        raise ValueError(f"{reference!r} matches {len(matches)} recorded attempts")
    return str(matches[0]["commit_hash"]), reference


def program_source(run_dir: Path, commit_hash: str) -> str:
    """Read program.py as recorded at one commit of the run repository."""
    completed = subprocess.run(
        ["git", "-C", str(run_dir / "repo"), "show", f"{commit_hash}:program.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def _score_job(job: tuple[str, str, str, str, Path, Path]) -> Scored:
    label, partition, commit_hash, source, split_file, tasks_file = job
    tasks = load_partition(split_file, tasks_file, partition)
    grader = ProgramGrader(tasks, expected_task_count=len(tasks))
    evaluation = grader.grade_with_diagnostics(source)
    cases = tuple(
        {
            "case_id": case.case_id,
            "task_id": task.task_id,
            "frequency": case.frequency,
            "horizon": case.horizon,
            "smae": case.smae,
            "srmse": case.srmse,
            "score": 0.5 * case.smae + 0.5 * case.srmse,
            "failure": case.failure,
            "history": list(case.history),
            "truth": list(case.truth),
            "forecast": None if case.forecast is None else list(case.forecast),
        }
        for case, task in zip(evaluation.cases, tasks)
    )
    return Scored(label, partition, commit_hash, dict(evaluation.metrics.__dict__), cases)


def summary_table(results: list[Scored]) -> str:
    """Render one aggregate row per program and partition, rounded for reporting."""
    header = (
        f"{'program':10s} {'split':6s} {'n':>4s} {'cov':>6s} "
        f"{'sMAE':>7s} {'sRMSE':>7s} {'score':>7s} {'se_sMAE':>8s} {'se_sRMSE':>9s} {'fail':>5s}"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        m = row.metrics
        lines.append(
            f"{row.label:10s} {row.partition:6s} {int(m['task_count']):4d} "
            f"{_round(m['coverage']):6.3f} {_round(m['mean_smae']):7.3f} "
            f"{_round(m['mean_srmse']):7.3f} {_round(m['score']):7.3f} "
            f"{_round(m['se_smae']):8.3f} {_round(m['se_srmse']):9.3f} "
            f"{int(m['failure_count']):5d}"
        )
    return "\n".join(lines)


def paired_table(results: list[Scored], baseline_label: str) -> str:
    """Compare each program against the baseline on the same cases, per partition."""
    lines = [
        f"{'program':10s} {'split':6s} {'delta':>8s} {'se':>7s} {'t':>7s} "
        f"{'better':>7s} {'worse':>6s} {'same':>5s}",
        "-" * 60,
    ]
    by_split: dict[str, dict[str, Scored]] = {}
    for row in results:
        by_split.setdefault(row.partition, {})[row.label] = row
    for partition, rows in by_split.items():
        base = rows.get(baseline_label)
        if base is None:
            continue
        base_scores = {c["case_id"]: c["score"] for c in base.cases}
        for label, row in rows.items():
            if label == baseline_label:
                continue
            deltas = [c["score"] - base_scores[c["case_id"]] for c in row.cases]
            mean = statistics.fmean(deltas)
            se = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
            better = sum(1 for d in deltas if d < -1e-9)
            worse = sum(1 for d in deltas if d > 1e-9)
            lines.append(
                f"{label:10s} {partition:6s} {_round(mean):8.3f} {_round(se):7.3f} "
                f"{_round(mean / se) if se else 0.0:7.3f} "
                f"{better:7d} {worse:6d} {len(deltas) - better - worse:5d}"
            )
    return "\n".join(lines)


def plot_cases(row: Scored, out_dir: Path, count: int, select: str) -> Path:
    """Plot history tail, truth, and forecast for the selected cases."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [c for c in row.cases if c["forecast"] is not None]
    if not usable:
        raise ValueError("no case produced a forecast to plot")
    ordered = sorted(usable, key=lambda c: c["score"], reverse=(select == "worst"))
    if select == "spread":
        step = max(1, len(ordered) // count)
        chosen = ordered[::step][:count]
    else:
        chosen = ordered[:count]

    columns = min(3, len(chosen))
    rows_n = math.ceil(len(chosen) / columns)
    figure, axes = plt.subplots(rows_n, columns, figsize=(5.2 * columns, 3.1 * rows_n), squeeze=False)
    for index, case in enumerate(chosen):
        axis = axes[index // columns][index % columns]
        horizon = case["horizon"]
        tail = case["history"][-min(len(case["history"]), 3 * horizon):]
        axis.plot(range(-len(tail), 0), tail, color="0.55", linewidth=1.0, label="history")
        axis.plot(range(horizon), case["truth"], color="#1b3a6b", linewidth=1.6, label="truth")
        axis.plot(range(horizon), case["forecast"], color="#c2472a", linewidth=1.6,
                  linestyle="--", label="forecast")
        axis.axvline(-0.5, color="0.8", linewidth=0.8)
        axis.set_title(
            f"{case['case_id']}  {case['frequency']}  h={horizon}  score={_round(case['score']):.3f}",
            fontsize=9,
        )
        axis.tick_params(labelsize=8)
    for index in range(len(chosen), rows_n * columns):
        axes[index // columns][index % columns].axis("off")
    axes[0][0].legend(fontsize=8, loc="best")
    figure.suptitle(
        f"{row.label} ({row.commit_hash[:12]}) — {row.partition} — {select} {len(chosen)} cases",
        fontsize=11,
    )
    figure.tight_layout()
    path = out_dir / f"forecasts_{row.label}_{row.partition}_{select}.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_distribution(results: list[Scored], partition: str, out_dir: Path) -> Path:
    """Plot the sorted per-case score curve for every program on one partition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    for row in results:
        if row.partition != partition:
            continue
        scores = sorted(c["score"] for c in row.cases)
        axis.plot(range(1, len(scores) + 1), scores, linewidth=1.7,
                  label=f"{row.label} (mean {_round(row.metrics['score']):.3f})")
    axis.set_xlabel("case, ranked best to worst")
    axis.set_ylabel("0.5·sMAE + 0.5·sRMSE")
    axis.set_title(f"Per-case score distribution — {partition}")
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(fontsize=9)
    figure.tight_layout()
    path = out_dir / f"distribution_{partition}.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--commit", default="best", help="'best', 'seed', or a commit hash prefix")
    parser.add_argument("--baseline", default="seed", help="'seed', 'none', or a commit hash prefix")
    parser.add_argument("--split", choices=(*PARTITIONS, "both"), default="both")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--tasks-file", type=Path, default=None,
                        help="defaults to the tasks file recorded in the run config")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="defaults to <run-dir>/evaluation")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-cases", type=int, default=9)
    parser.add_argument("--plot-select", choices=("worst", "best", "spread"), default="worst")
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = EvolutionRun(args.run_dir)

    tasks_file = args.tasks_file
    if tasks_file is None:
        config = json.loads(
            (args.run_dir / ".evolution" / "private" / "taskdata" / "config.json").read_text("utf-8")
        )
        tasks_file = Path(config["tasks_file"])

    programs = [resolve_commit(run, args.commit)]
    if args.baseline != "none":
        baseline = resolve_commit(run, args.baseline)
        if baseline[0] != programs[0][0]:
            programs.append(baseline)

    partitions = PARTITIONS if args.split == "both" else (args.split,)
    jobs = [
        (label, partition, commit_hash, program_source(args.run_dir, commit_hash),
         args.split_file, tasks_file)
        for commit_hash, label in programs
        for partition in partitions
    ]
    print(f"scoring {len(programs)} program(s) on {len(partitions)} partition(s)…", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
        results = list(pool.map(_score_job, jobs))

    print()
    print(summary_table(results))
    if len(programs) > 1:
        print()
        print(paired_table(results, programs[-1][1]))

    out_dir = args.output_dir or (args.run_dir / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_file": str(args.split_file.resolve()),
        "tasks_file": str(Path(tasks_file).resolve()),
        "programs": [{"label": label, "commit_hash": h} for h, label in programs],
        "results": [
            {
                "label": r.label,
                "partition": r.partition,
                "commit_hash": r.commit_hash,
                "metrics": {
                    k: (_round(v) if isinstance(v, float) else v) for k, v in r.metrics.items()
                },
                "cases": [
                    {k: (_round(v) if isinstance(v, float) else v)
                     for k, v in c.items() if k not in {"history", "truth", "forecast"}}
                    for c in r.cases
                ],
            }
            for r in results
        ],
    }
    results_path = out_dir / "evaluation.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {results_path}")

    if args.plot:
        for row in results:
            print(f"wrote {plot_cases(row, out_dir, args.plot_cases, args.plot_select)}")
        for partition in partitions:
            print(f"wrote {plot_distribution(results, partition, out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
