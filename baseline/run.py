"""Run one baseline over a Dr-CiK split and report sMAE/sRMSE, or write its forecasts."""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

from .classical import ARIMAForecaster, ETSForecaster, NaiveForecaster, SESForecaster
from .data import DEV, HIDDEN, DEFAULT_TASKS_FILE, BenchmarkTask, load_benchmark
from .forecasters import SeasonalNaive
from .scoring import render, score_task, summarize

BASELINES = (
    "seasonal_naive", "naive", "ses", "ets", "arima",
    "moirai", "aurora", "chronos", "gemini_dp",
)


def build_forecaster(name: str, task: BenchmarkTask, device: str, seed: int):
    """Build the forecaster for one task.

    Model-backed baselines are cached process-wide: reloading a checkpoint per task would cost
    far more than the inference itself.
    """
    if name == "seasonal_naive":
        return SeasonalNaive(task.frequency, task.seasonal_period, seed=seed)
    if name == "naive":
        return NaiveForecaster(seed=seed)
    if name == "ses":
        return SESForecaster(seed=seed)
    if name == "ets":
        return ETSForecaster(task.frequency, task.seasonal_period, seed=seed)
    if name == "arima":
        return ARIMAForecaster(seed=seed)
    if name == "moirai":
        return _shared_moirai(device, seed)
    if name == "aurora":
        return _shared_aurora(device, task)
    if name == "chronos":
        return _shared_chronos(device, seed)
    if name == "gemini_dp":
        return _shared_gemini(task)
    raise ValueError(f"unknown baseline {name!r}; choose from {', '.join(BASELINES)}")


_SHARED: dict[str, object] = {}


def _shared_moirai(device: str, seed: int):
    if "moirai" not in _SHARED:
        from common.tsfm import MoiraiConfig

        from .moirai import MoiraiSampleForecaster

        _SHARED["moirai"] = MoiraiSampleForecaster(MoiraiConfig(device=device), seed=seed)
    return _SHARED["moirai"]


def _shared_aurora(device: str, task: BenchmarkTask):
    """One loaded checkpoint, but token length is re-derived per task from its seasonality."""
    from .aurora import AuroraConfig, AuroraForecaster, token_length_for

    if "aurora" not in _SHARED:
        _SHARED["aurora"] = AuroraForecaster(AuroraConfig(device=device))
    forecaster = _SHARED["aurora"]
    forecaster.token_len = token_length_for(task.frequency, task.seasonal_period)
    return forecaster


def _shared_chronos(device: str, seed: int):
    if "chronos" not in _SHARED:
        from common.tsfm import ChronosConfig

        from .chronos_baseline import ChronosSampleForecaster

        _SHARED["chronos"] = ChronosSampleForecaster(ChronosConfig(device_map=device), seed=seed)
    return _SHARED["chronos"]


def _shared_gemini(task: BenchmarkTask):
    """One client and one cache for the run; the series itself is set per task."""
    from .gemini import DirectPromptForecaster

    if "gemini_dp" not in _SHARED:
        _SHARED["gemini_dp"] = DirectPromptForecaster()
    forecaster = _SHARED["gemini_dp"]
    forecaster.timestamps = task.history_timestamps
    forecaster.frequency = task.frequency
    return forecaster


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--split", default=DEV, choices=(DEV, HIDDEN))
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None, help="first N tasks, for a smoke run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks-file", default=str(DEFAULT_TASKS_FILE))
    parser.add_argument("--out", default=None, help="write per-task forecasts as JSONL")
    parser.add_argument("--resume", action="store_true",
                         help="skip tasks already present in --out instead of overwriting it")
    return parser


def _done_benchmark_ids(destination: Path) -> set[str]:
    """Benchmark ids already written to --out, so a resumed sweep does not repeat them."""
    if not destination.exists():
        return set()
    ids = set()
    for line in destination.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["benchmark_id"])
    return ids


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = load_benchmark(args.tasks_file, split=args.split)
    if args.limit:
        tasks = tasks[: args.limit]
    destination = Path(args.out) if args.out else None
    done = _done_benchmark_ids(destination) if args.resume and destination else set()
    if done:
        tasks = [task for task in tasks if task.benchmark_id not in done]
        print(f"resuming: {len(done)} already done, {len(tasks)} remaining", flush=True)
    print(f"{args.baseline}: {len(tasks)} {args.split} tasks, {args.samples} samples each",
          flush=True)

    scores, written, started = [], 0, time.monotonic()
    with _forecast_writer(destination, append=bool(done)) as handle:
        for index, task in enumerate(tasks, start=1):
            task_started = time.monotonic()
            try:
                forecaster = build_forecaster(args.baseline, task, args.device, args.seed)
                samples = forecaster.forecast_samples(
                    task.history_values, task.prediction_length, args.samples
                )
            except Exception as exc:  # one unforecastable task must not end the sweep
                scores.append(score_task(task.benchmark_id, task.future_values, []))
                _log(index, len(tasks), task.benchmark_id, started, task_started,
                     detail=f"FAILED {type(exc).__name__}: {str(exc)[:120]}")
                continue
            if handle is not None:
                _write_record(handle, task.benchmark_id, samples)
                written += 1

            detail = f"H={task.prediction_length} paths={len(samples)}/{args.samples}"
            if task.labels_public:
                score = score_task(task.benchmark_id, task.future_values, samples)
                scores.append(score)
                running = summarize(args.baseline, scores)
                shown = "n/a" if score.smae is None else f"{score.smae:.3f}"
                mean = "n/a" if running.mean_smae is None else f"{running.mean_smae:.3f}"
                detail += f" sMAE={shown} mean={mean}"
            _log(index, len(tasks), task.benchmark_id, started, task_started, detail)

    if args.out:
        print(f"wrote {written} forecasts to {args.out}")
    if scores:
        print()
        print(render([summarize(args.baseline, scores)]))
        for failure in summarize(args.baseline, scores).sample_failures:
            print(f"failure: {failure}")
    return 0


def _log(
    index: int, total: int, benchmark_id: str, started: float, task_started: float, detail: str
) -> None:
    """One line per task, flushed, so a long sweep can be followed while it runs."""
    now = time.monotonic()
    elapsed = now - started
    eta = (elapsed / index) * (total - index)
    print(
        f"{time.strftime('%H:%M:%S')} [{index}/{total}] {benchmark_id} {detail} "
        f"({now - task_started:.1f}s, elapsed {elapsed / 60:.1f}m, eta {eta / 60:.0f}m)",
        flush=True,
    )


@contextmanager
def _forecast_writer(destination: Path | None, append: bool = False):
    """Open the JSONL sink, or yield None when the sweep is scoring only."""
    if destination is None:
        yield None
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a" if append else "w", encoding="utf-8") as handle:
        yield handle


def _write_record(handle, benchmark_id: str, samples) -> None:
    """Append one task and flush: a sweep that dies mid-run keeps everything it already ran."""
    handle.write(json.dumps({
        "benchmark_id": benchmark_id,
        # 6 decimals: the metrics round to 3, so more precision only inflates the file.
        "samples": [[round(value, 6) for value in path] for path in samples],
    }) + "\n")
    handle.flush()


if __name__ == "__main__":
    raise SystemExit(main())
