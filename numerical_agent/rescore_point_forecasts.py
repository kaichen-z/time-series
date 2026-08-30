"""Rescore frozen deterministic forecasts with Dr-CiK-aligned point metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from common.evolution_core.contracts import (
    load_active_release,
    metric_report_metadata,
)
from common.payload import standards_json_value, strict_json_loads, write_json

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
    forecasts = _load_forecast_rows(artifact_path, allow_legacy=True)
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
        str(row["task_id"]): (float(row["smae"]), float(row["srmse"]))
        for row in scores[baseline_row]["per_task"]
    }
    paired = {
        name: _paired_counts(
            score["per_task"], baseline_by_task, tuple(task.task_id for task in tasks)
        )
        for name, score in scores.items()
    }
    return {
        "schema_version": 2,
        **metric_report_metadata(),
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
            "ordering": "paired joint mean of capped sMAE and sRMSE",
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
        f"- Metric policy SHA-256: `{payload['metric_policy_fingerprint']}`",
        "- Primary metrics: sMAE, sRMSE",
        "- Diagnostic only: MASE, MAE, sMAPE, RMSSE",
        "- sCRPS: **not computed** (no probabilistic trajectories in this phase)",
        "- Model calls: **0**; all forecasts were read from the frozen artifact",
        "- Status: public-label development/regression metrics, not an official hidden-test score",
        "",
        "| Row | Mean sMAE | Median sMAE | sMAE SE | Mean sRMSE | Median sRMSE | sRMSE SE | Raw P90/P95 sMAE/sRMSE | Clipped sMAE/sRMSE | Coverage | W/T/L/M/U vs baseline (joint) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, raw in rows.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"score for {name} must be a mapping")
        comparison = paired[name]
        if not isinstance(comparison, Mapping):
            raise ValueError(f"comparison for {name} must be a mapping")
        lines.append(
            f"| {name} | {_number(raw['mean_smae'])} | {_number(raw['median_smae'])} | "
            f"{_number(raw['se_smae'])} | {_number(raw['mean_srmse'])} | "
            f"{_number(raw['median_srmse'])} | {_number(raw['se_srmse'])} | "
            f"{_number(raw['p90_smae_raw'])}/{_number(raw['p95_smae_raw'])} / "
            f"{_number(raw['p90_srmse_raw'])}/{_number(raw['p95_srmse_raw'])} | "
            f"{raw['smae_clipped_count']}/{raw['srmse_clipped_count']} | "
            f"{_number(raw['coverage'], digits=4)} | "
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']}/"
            f"{comparison['missing']}/{comparison['unscored']} |"
        )
    return "\n".join(lines) + "\n"


def _load_forecast_rows(
    path: str | Path, *, allow_legacy: bool = False
) -> dict[str, dict[str, ForecastResult]]:
    """Read a frozen historical forecast artifact for report-only rescoring."""
    result: dict[str, dict[str, ForecastResult]] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = strict_json_loads(
            line, context=f"historical forecast row {line_number}"
        )
        if not isinstance(raw, Mapping):
            raise ValueError(f"line {line_number} must contain an object")
        active = "schema_version" in raw
        if active:
            load_active_release(raw)
            _require_exact_row_fields(raw, {
                "schema_version", "metric_policy", "metric_policy_fingerprint",
                "task_id", "rows",
            }, f"active line {line_number}")
        else:
            if not allow_legacy:
                raise ValueError(
                    "legacy forecast rows require allow_legacy=True report-only parsing"
                )
            _require_exact_row_fields(
                raw, {"task_id", "rows"}, f"legacy line {line_number}"
            )
        task_id = raw["task_id"]
        if type(task_id) is not str or not task_id:
            raise ValueError(f"line {line_number} task_id must be a non-empty string")
        if not task_id or task_id in result:
            raise ValueError(f"duplicate or empty task_id on line {line_number}")
        raw_rows = raw["rows"]
        if type(raw_rows) is not dict or not raw_rows:
            raise ValueError(f"line {line_number} needs non-empty rows")
        parsed: dict[str, ForecastResult] = {}
        for name, value in raw_rows.items():
            if type(name) is not str or not name:
                raise ValueError(f"line {line_number} row names must be non-empty strings")
            if not isinstance(value, Mapping):
                raise ValueError(f"row {name!r} on line {line_number} must be an object")
            expected_fields = {
                "task_id", "forecast", "selected", "families", "mode", "oracle_mase",
            }
            if active:
                expected_fields.add("assumption_ids")
            _require_exact_row_fields(
                value, expected_fields, f"row {name!r} on line {line_number}"
            )
            row_task_id = value["task_id"]
            if type(row_task_id) is not str:
                raise ValueError(
                    f"row {name!r} task_id on line {line_number} must be a string"
                )
            if row_task_id != task_id:
                raise ValueError(
                    f"row {name!r} task_id {row_task_id!r} does not match {task_id!r}"
                )
            forecast = _finite_number_list(value["forecast"], "forecast")
            selected = _string_list(value["selected"], "selected")
            families = _string_list(value["families"], "families")
            mode = value["mode"]
            if type(mode) is not str or not mode:
                raise ValueError(f"row {name!r} mode must be a non-empty string")
            oracle = value["oracle_mase"]
            if oracle is not None:
                oracle = _finite_number(oracle, f"row {name!r} oracle_mase")
            assumption_ids = (
                _string_list(value["assumption_ids"], "assumption_ids")
                if active else ()
            )
            parsed[name] = ForecastResult(
                task_id,
                forecast,
                selected,
                families,
                mode,
                oracle,
                assumption_ids,
            )
        result[task_id] = parsed
    return result


def _require_exact_row_fields(
    payload: Mapping[str, object], expected: set[str], context: str
) -> None:
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise ValueError(
            f"{context} fields mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _finite_number_list(value: object, field_name: str) -> tuple[float, ...]:
    if type(value) is not list:
        raise ValueError(f"{field_name} must be a list")
    return tuple(
        _finite_number(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _string_list(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    return tuple(value)


def _number(value: object, *, digits: int = 6) -> str:
    number = float(value)
    if math.isnan(number):
        raise ValueError("rescore report cannot render NaN")
    if math.isinf(number):
        return "positive_infinity" if number > 0 else "negative_infinity"
    return f"{number:.{digits}f}"


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
    print(json.dumps(
        _finite_json(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


def _finite_json(value: object) -> object:
    return standards_json_value(value)


if __name__ == "__main__":
    raise SystemExit(main())
