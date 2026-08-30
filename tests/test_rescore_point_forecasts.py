from __future__ import annotations

import json

import pytest

from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT
from common.payload import decode_infinity_sentinel, strict_json_loads
from numerical_agent.evolution.execution import Task
from numerical_agent.rescore_point_forecasts import (
    _finite_json,
    _load_forecast_rows,
    _number,
    render_point_report,
    rescore_cached_point_forecasts,
)


def test_rescore_raw_infinity_round_trips_as_strict_explicit_sentinel() -> None:
    encoded = json.dumps(_finite_json({"p95_smae_raw": float("inf")}), allow_nan=False)
    decoded = strict_json_loads(encoded)

    assert decoded == {
        "p95_smae_raw": {"status": "positive_infinity", "value": None}
    }
    assert decode_infinity_sentinel(decoded["p95_smae_raw"], "p95_smae_raw") == float("inf")
    assert _number(float("inf")) == "positive_infinity"
    assert _number(float("-inf")) == "negative_infinity"


def _write_rows(path) -> None:
    rows = (
        {
            "task_id": "t1",
            "rows": {
                "candidate": {
                    "task_id": "t1", "forecast": [3.0, 3.0], "selected": ["a"],
                    "families": ["statistical"], "mode": "single", "oracle_mase": None,
                },
                "baseline": {
                    "task_id": "t1", "forecast": [4.0, 4.0], "selected": ["b"],
                    "families": ["tsfm"], "mode": "single", "oracle_mase": None,
                },
            },
        },
        {
            "task_id": "t2",
            "rows": {
                "candidate": {
                    "task_id": "t2", "forecast": [1.0, 1.0], "selected": ["a"],
                    "families": ["statistical"], "mode": "single", "oracle_mase": None,
                },
                "baseline": {
                    "task_id": "t2", "forecast": [2.0, 2.0], "selected": ["b"],
                    "families": ["tsfm"], "mode": "single", "oracle_mase": None,
                },
            },
        },
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_legacy_forecast_reader_requires_explicit_report_only_opt_in(tmp_path):
    artifact = tmp_path / "legacy.jsonl"
    _write_rows(artifact)

    with pytest.raises(ValueError, match="allow_legacy"):
        _load_forecast_rows(artifact)

    assert set(_load_forecast_rows(artifact, allow_legacy=True)) == {"t1", "t2"}


def test_rescore_uses_cached_forecasts_without_probabilistic_metrics(tmp_path):
    artifact = tmp_path / "per_task_results.jsonl"
    _write_rows(artifact)
    tasks = (
        Task("t1", (1.0, 2.0, 3.0), 2, "D", (2.0, 2.0)),
        Task("t2", (1.0, 2.0, 3.0), 2, "D", (1.0, 1.0)),
    )

    payload = rescore_cached_point_forecasts(tasks, artifact, baseline_row="baseline")

    assert payload["metric_contract"]["point_forecast_only"] is True
    assert payload["metric_contract"]["scrps_computed"] is False
    assert payload["schema_version"] == 2
    assert payload["metric_policy_fingerprint"] == METRIC_POLICY_FINGERPRINT
    assert payload["primary_metrics"] == ["smae", "srmse"]
    assert set(payload["diagnostic_only"]) >= {"mase", "mae", "smape", "rmsse"}
    assert payload["rows"]["candidate"]["mean_smae"] == pytest.approx(0.25)
    assert payload["rows"]["baseline"]["mean_smae"] == pytest.approx(1.0)
    assert payload["paired_vs_baseline"]["candidate"] == {
        "wins": 2, "ties": 0, "losses": 0, "missing": 0,
        "unscored": 0,
    }
    report = render_point_report(payload)
    assert "sCRPS: **not computed**" in report
    assert "Mean sMAE" in report
    assert "Median sMAE" in report
    assert "Mean sRMSE" in report
    assert "Raw P90/P95 sMAE/sRMSE" in report
    assert "W/T/L/M/U vs baseline" in report


def test_rescore_paired_counts_conserve_every_expected_task(tmp_path):
    artifact = tmp_path / "paired.jsonl"
    membership = {
        "win": {"candidate": 1.0, "baseline": 2.0},
        "tie": {"candidate": 2.0, "baseline": 2.0},
        "loss": {"candidate": 3.0, "baseline": 2.0},
        "candidate_only": {"candidate": 2.0},
        "baseline_only": {"baseline": 2.0},
        "both_missing": {"other": 1.0},
    }
    rows = []
    tasks = []
    for task_id, forecasts in membership.items():
        tasks.append(Task(task_id, (1.0,), 1, "D", (1.0,)))
        rows.append({
            "task_id": task_id,
            "rows": {
                name: {
                    "task_id": task_id,
                    "forecast": [forecast],
                    "selected": [name],
                    "families": ["statistical"],
                    "mode": "single",
                    "oracle_mase": None,
                }
                for name, forecast in forecasts.items()
            },
        })
    artifact.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    payload = rescore_cached_point_forecasts(
        tuple(tasks), artifact, baseline_row="baseline"
    )

    counts = payload["paired_vs_baseline"]["candidate"]
    assert counts == {
        "wins": 1, "ties": 1, "losses": 1, "missing": 2, "unscored": 1,
    }
    assert sum(counts.values()) == len(tasks)


def test_rescore_rejects_duplicate_or_mismatched_task_rows(tmp_path):
    artifact = tmp_path / "bad.jsonl"
    row = {
        "task_id": "t1",
        "rows": {
            "candidate": {
                "task_id": "other", "forecast": [1.0], "selected": [],
                "families": [], "mode": "single", "oracle_mase": None,
            }
        },
    }
    artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
    task = Task("t1", (1.0,), 1, "D", (1.0,))

    with pytest.raises(ValueError, match="task_id"):
        rescore_cached_point_forecasts((task,), artifact, baseline_row="candidate")


@pytest.mark.parametrize(
    "source",
    (
        '{"task_id":"t1","task_id":"t1","rows":{"candidate":{"task_id":"t1","forecast":[1.0],"selected":[],"families":[],"mode":"single","oracle_mase":null}}}',
        '{"task_id":"t1","rows":{"candidate":{"task_id":"t1","forecast":[1.0],"forecast":[1.0],"selected":[],"families":[],"mode":"single","oracle_mase":null}}}',
        '{"task_id":"t1","rows":{"candidate":{"task_id":"t1","forecast":[NaN],"selected":[],"families":[],"mode":"single","oracle_mase":null}}}',
        '{"task_id":"t1","rows":{"candidate":{"task_id":"t1","forecast":[Infinity],"selected":[],"families":[],"mode":"single","oracle_mase":null}}}',
        '{"task_id":"t1","rows":{"candidate":{"task_id":"t1","forecast":[1.0],"selected":[],"families":[],"mode":"single","oracle_mase":Infinity}}}',
    ),
)
def test_rescore_rejects_duplicate_keys_and_nonstandard_constants_before_scoring(
    tmp_path, source
):
    artifact = tmp_path / "invalid.jsonl"
    artifact.write_text(source + "\n", encoding="utf-8")
    task = Task("t1", (1.0,), 1, "D", (1.0,))

    with pytest.raises(ValueError, match="duplicate|non-finite"):
        rescore_cached_point_forecasts((task,), artifact, baseline_row="candidate")


@pytest.mark.parametrize(
    "mutation",
    (
        lambda row: {**row, "unexpected": True},
        lambda row: {**row, "rows": {"candidate": {**row["rows"]["candidate"], "unexpected": True}}},
        lambda row: {**row, "rows": {"candidate": {**row["rows"]["candidate"], "forecast": [True]}}},
        lambda row: {**row, "rows": {"candidate": {**row["rows"]["candidate"], "oracle_mase": "1.0"}}},
    ),
)
def test_rescore_legacy_reader_requires_exact_finite_row_schema(tmp_path, mutation):
    artifact = tmp_path / "invalid-legacy.jsonl"
    row = {
        "task_id": "t1",
        "rows": {
            "candidate": {
                "task_id": "t1", "forecast": [1.0], "selected": [],
                "families": [], "mode": "single", "oracle_mase": None,
            }
        },
    }
    artifact.write_text(json.dumps(mutation(row)) + "\n", encoding="utf-8")
    task = Task("t1", (1.0,), 1, "D", (1.0,))

    with pytest.raises(ValueError):
        rescore_cached_point_forecasts((task,), artifact, baseline_row="candidate")
