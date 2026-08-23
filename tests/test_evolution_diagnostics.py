from __future__ import annotations

import json

from numerical_agent.evolution.diagnostics import (
    diagnose_forecasts,
    parse_failure_diagnosis,
)
from numerical_agent.evolution.execution import Outcome, SUCCESS, Task


def test_diagnostics_measure_bias_amplitude_phase_and_horizon_error() -> None:
    task = Task(
        "seasonal",
        (10.0, 20.0, 10.0, 20.0, 10.0, 20.0),
        4,
        "1 day",
        (10.0, 20.0, 10.0, 20.0),
    )
    outcome = Outcome(
        "shifted",
        task.task_id,
        SUCCESS,
        smape=100.0,
        mae=10.0,
        mase=1.0,
        forecast=(20.0, 10.0, 20.0, 10.0),
    )

    report = diagnose_forecasts("shifted", (outcome,), (task,))
    row = report["tasks"][0]

    assert row["mean_bias"] == 0.0
    assert row["amplitude_ratio_to_truth"] == 1.0
    assert abs(row["phase_shift_steps"]) == 1
    assert row["early_mae"] == 10.0
    assert row["late_mae"] == 10.0
    assert "future_values" not in json.dumps(report)
    assert "forecast_values" not in json.dumps(report)


def test_failure_diagnosis_parser_rejects_code_and_requires_grounded_evidence() -> None:
    valid = json.dumps({
        "failure_types": ["phase_shift"],
        "summary": "The periodic forecast is displaced by one step.",
        "evidence": ["median absolute phase shift is one step"],
        "mutation_guidance": ["align the seasonal origin without changing the period"],
        "confidence": 0.8,
    })

    parsed = parse_failure_diagnosis(valid)

    assert parsed["failure_types"] == ["phase_shift"]
    assert parsed["confidence"] == 0.8

    unsafe = json.dumps({
        **json.loads(valid),
        "mutation_guidance": ["def forecast(history): return history"],
    })
    try:
        parse_failure_diagnosis(unsafe)
    except ValueError as exc:
        assert "code" in str(exc)
    else:
        raise AssertionError("code-bearing diagnosis must be rejected")
