"""One-off: run a baseline over all 279 tasks (dev + hidden) in one file, not exposed on the CLI."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from baseline.data import DEFAULT_TASKS_FILE, load_benchmark
from baseline.run import _forecast_writer, _log, _write_record, build_forecaster
from baseline.scoring import render, score_task, summarize


def main() -> int:
    baseline, device, samples, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    tasks = load_benchmark(DEFAULT_TASKS_FILE, split=None)
    print(f"{baseline}: {len(tasks)} tasks (dev+hidden), {samples} samples each", flush=True)

    scores, started = [], time.monotonic()
    with _forecast_writer(Path(out)) as handle:
        for index, task in enumerate(tasks, start=1):
            task_started = time.monotonic()
            try:
                forecaster = build_forecaster(baseline, task, device, seed=0)
                samples_out = forecaster.forecast_samples(
                    task.history_values, task.prediction_length, samples
                )
            except Exception as exc:  # one unforecastable task must not end the sweep
                scores.append(score_task(task.benchmark_id, task.future_values, []))
                _log(index, len(tasks), task.benchmark_id, started, task_started,
                     detail=f"FAILED {type(exc).__name__}: {str(exc)[:120]}")
                continue
            _write_record(handle, task.benchmark_id, samples_out)
            detail = f"H={task.prediction_length} paths={len(samples_out)}/{samples}"
            if task.labels_public:
                score = score_task(task.benchmark_id, task.future_values, samples_out)
                scores.append(score)
                detail += f" sMAE={score.smae:.3f}" if score.smae is not None else " sMAE=n/a"
            _log(index, len(tasks), task.benchmark_id, started, task_started, detail)

    if scores:
        print(render([summarize(baseline, scores)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
