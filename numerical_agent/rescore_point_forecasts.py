"""Rescore frozen deterministic forecasts with Dr-CiK-aligned point metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from common.payload import write_json

from .evaluate_frozen_two_stage import (
    ForecastResult,
    _paired_counts,
    _public_test_tasks,
    score_forecast_results,
)
from .evolution.execution import Task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", default="splits/drcik_public_80_20_99_v1.json")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--per-task-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-row", default="E_toto_reference")
    return parser


def rescore_cached_point_forecasts(
    tasks: Sequence[Task],
    artifact_path: str | Path,
    *,
    baseline_row: str,
) -> dict[str, object]:
    """Read frozen trajectories and recompute point metrics without model inference."""
    forecasts = _load_forecast_rows(artifact_path)
    expected_task_ids = {task.task_id for task in tasks}
    if set(forecasts) != expected_task_ids:
        missing = sorted(expected_task_ids - set(forecasts))
        extra = sorted(set(forecasts) - expected_task_ids)
        raise ValueError(f"cached task_id coverage mismatch: missing={missing}, extra={extra}")
    row_names = {name for rows in forecasts.values() for name in rows}
    if baseline_row not in row_names:
        raise ValueError(f"baseline row {baseline_row!r} is absent from cached forecasts")
    by_row = {
        name: tuple(
            forecasts[task.task_id][name]
            for task in tasks
            if name in forecasts[task.task_id]
        )
        for name in sorted(row_names)
    }
    scores = {
        name: score_forecast_results(tasks, results)
        for name, results in by_row.items()
    }
    baseline_by_task = {
        str(row["task_id"]): float(row["smae"])
        for row in scores[baseline_row]["per_task"]
    }
    paired = {
        name: _paired_counts(score["per_task"], baseline_by_task)
        for name, score in scores.items()
    }
    return {
        "schema_version": 1,
        "task_count": len(tasks),
        "source_artifact": str(Path(artifact_path)),
        "source_artifact_sha256": hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest(),
        "baseline_row": baseline_row,
        "metric_contract": {
            "point_forecast_only": True,
            "scrps_computed": False,
            "scale": "mean absolute true future value per task",
            "winsorization_cap": 5.0,
            "aggregation": "task mean plus standard error",
            "official_hidden_score": False,
        },
        "rows": scores,
        "paired_vs_baseline": paired,
        "model_calls": 0,
    }


def render_point_report(payload: Mapping[str, object]) -> str:
    rows = payload["rows"]
    paired = payload["paired_vs_baseline"]
    if not isinstance(rows, Mapping) or not isinstance(paired, Mapping):
        raise ValueError("rescore payload needs row and paired mappings")
    lines = [
        "# Dr-CiK-Aligned Point Forecast Rescore",
        "",
        f"- Tasks: {payload['task_count']}",
        f"- Baseline: `{payload['baseline_row']}`",
        "- sCRPS: **not computed** (no probabilistic trajectories in this phase)",
        "- Model calls: **0**; all forecasts were read from the frozen artifact",
        "- Status: public-label development/regression metrics, not an official hidden-test score",
        "",
        "| Row | Mean sMAE | sMAE SE | Mean sRMSE | sRMSE SE | P90/P95 sMAE | Clipped sMAE/sRMSE | Coverage | W/T/L vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, raw in rows.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"score for {name} must be a mapping")
        comparison = paired[name]
        if not isinstance(comparison, Mapping):
            raise ValueError(f"comparison for {name} must be a mapping")
        lines.append(
            f"| {name} | {_number(raw['mean_smae'])} | {_number(raw['se_smae'])} | "
            f"{_number(raw['mean_srmse'])} | {_number(raw['se_srmse'])} | "
            f"{_number(raw['p90_smae'])}/{_number(raw['p95_smae'])} | "
            f"{raw['smae_clipped_count']}/{raw['srmse_clipped_count']} | "
            f"{_number(raw['coverage'], digits=4)} | "
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']} |"
        )
    return "\n".join(lines) + "\n"


def _load_forecast_rows(path: str | Path) -> dict[str, dict[str, ForecastResult]]:
    result: dict[str, dict[str, ForecastResult]] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"line {line_number} must contain an object")
        task_id = str(raw.get("task_id", ""))
        if not task_id or task_id in result:
            raise ValueError(f"duplicate or empty task_id on line {line_number}")
        raw_rows = raw.get("rows")
        if not isinstance(raw_rows, Mapping) or not raw_rows:
            raise ValueError(f"line {line_number} needs non-empty rows")
        parsed: dict[str, ForecastResult] = {}
        for name, value in raw_rows.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"row {name!r} on line {line_number} must be an object")
            row_task_id = str(value.get("task_id", ""))
            if row_task_id != task_id:
                raise ValueError(
                    f"row {name!r} task_id {row_task_id!r} does not match {task_id!r}"
                )
            forecast = tuple(float(item) for item in _sequence(value.get("forecast"), "forecast"))
            if not all(math.isfinite(item) for item in forecast):
                raise ValueError(f"row {name!r} contains a non-finite forecast")
            oracle = value.get("oracle_mase")
            parsed[str(name)] = ForecastResult(
                task_id,
                forecast,
                tuple(str(item) for item in _sequence(value.get("selected"), "selected")),
                tuple(str(item) for item in _sequence(value.get("families"), "families")),
                str(value.get("mode", "")),
                None if oracle is None else float(oracle),
            )
        result[task_id] = parsed
    return result


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _number(value: object, *, digits: int = 6) -> str:
    number = float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = _public_test_tasks(args.split_file, args.tasks_file)
    payload = rescore_cached_point_forecasts(
        tasks,
        args.per_task_results,
        baseline_row=args.baseline_row,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "point_rescore_results.json", _finite_json(payload))
    (output / "POINT_RESCORE_REPORT.md").write_text(
        render_point_report(payload), encoding="utf-8"
    )
    print(json.dumps(_finite_json(payload), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _finite_json(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
