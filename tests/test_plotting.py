"""Smoke tests that plot_forecasts_file / plot_comparison_files render non-empty PNGs; no pixel assertions."""

from __future__ import annotations

import json
from pathlib import Path

from dr_cik.plotting import plot_comparison_files, plot_forecasts_file

from .conftest import requires_sample


def _write_forecasts(path: Path, tasks, jitter: float = 0.0) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            base = task.history_values[-1] + jitter
            samples = [[base + row + step for step in range(task.prediction_length)] for row in range(5)]
            handle.write(json.dumps({"benchmark_id": task.benchmark_id, "samples": samples}) + "\n")


@requires_sample
def test_plot_forecasts_file_writes_one_png_per_task(sample_tasks, tmp_path: Path) -> None:
    forecasts_path = tmp_path / "forecasts.jsonl"
    _write_forecasts(forecasts_path, sample_tasks)

    output_dir = tmp_path / "plots"
    written = plot_forecasts_file(sample_tasks, forecasts_path, output_dir, label="fake-model")

    assert len(written) == len(sample_tasks)
    for path in written:
        assert path.is_file()
        assert path.stat().st_size > 0


@requires_sample
def test_plot_comparison_files_writes_one_png_per_common_task(sample_tasks, tmp_path: Path) -> None:
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    _write_forecasts(path_a, sample_tasks, jitter=0.0)
    _write_forecasts(path_b, sample_tasks, jitter=5.0)

    output_dir = tmp_path / "compare"
    written = plot_comparison_files(sample_tasks, [("Method A", path_a), ("Method B", path_b)], output_dir)

    assert len(written) == len(sample_tasks)
    for path in written:
        assert path.is_file()
        assert path.stat().st_size > 0


@requires_sample
def test_plot_comparison_files_only_plots_tasks_common_to_all_series(sample_tasks, tmp_path: Path) -> None:
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    _write_forecasts(path_a, sample_tasks)
    _write_forecasts(path_b, sample_tasks[:1])  # only the first task

    written = plot_comparison_files(sample_tasks, [("Method A", path_a), ("Method B", path_b)], tmp_path / "compare")

    assert len(written) == 1
