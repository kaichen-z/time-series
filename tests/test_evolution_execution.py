from __future__ import annotations

import multiprocessing
import signal
import time
from pathlib import Path

import pytest

from numerical_agent.evolution.execution import (
    CRASHED,
    INVALID,
    NOT_APPLICABLE,
    SUCCESS,
    Task,
    IsolatedForecastRuntime,
    MethodForecastError,
    derive_characteristics,
    load_methods,
    report_payload,
    run_module,
    _stop_worker,
)
from numerical_agent.evolution.module import MODULE_HEADER


FIXTURE = MODULE_HEADER + '''

def perfect_method(history, horizon, frequency):
    """Use when the future simply repeats the last observed value."""
    return [float(history[-1])] * horizon


def picky_method(history, horizon, frequency):
    """Use only for series with at least 100 observations."""
    if len(history) < 100:
        raise NotApplicable(f"needs 100 points, got {len(history)}")
    return [float(history[-1])] * horizon


def broken_method(history, horizon, frequency):
    """Use for nothing; it has an indexing defect."""
    return [history[999] for _ in range(horizon)]


def wrong_length_method(history, horizon, frequency):
    """Use for nothing; it returns the wrong number of values."""
    return [1.0]
'''


def write_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "methods.py"
    destination.write_text(FIXTURE, encoding="utf-8")
    return destination


def tasks() -> tuple[Task, ...]:
    return (
        Task("t1", tuple(float(i) for i in range(20)), 3, "1 day", (19.0, 19.0, 19.0)),
        Task("t2", tuple(float(i) for i in range(30)), 3, "1 day", (29.0, 29.0, 29.0)),
    )


def test_load_methods_returns_the_forecasting_functions(tmp_path: Path) -> None:
    module, functions = load_methods(write_fixture(tmp_path))

    assert set(functions) == {
        "perfect_method", "picky_method", "broken_method", "wrong_length_method"
    }
    # NotApplicable is a class, not a forecasting function, so it must not be collected.
    assert "NotApplicable" not in functions
    assert issubclass(module.NotApplicable, Exception)


def test_not_applicable_is_separated_from_a_crash(tmp_path: Path) -> None:
    _, reports = run_module(write_fixture(tmp_path), tasks())
    by_name = {report.method: report for report in reports}

    picky, broken = by_name["picky_method"], by_name["broken_method"]

    # Declining to apply is coverage, not failure.
    assert (picky.not_applicable, picky.crashed) == (2, 0)
    assert (broken.crashed, broken.not_applicable) == (2, 0)
    assert picky.mean_smape is None and broken.mean_smape is None


def test_a_crash_keeps_its_real_message(tmp_path: Path) -> None:
    _, reports = run_module(write_fixture(tmp_path), tasks())
    broken = next(r for r in reports if r.method == "broken_method")

    assert any("IndexError" in failure for failure in broken.sample_failures)


def test_a_wrong_length_forecast_is_invalid_not_a_crash(tmp_path: Path) -> None:
    outcomes, reports = run_module(write_fixture(tmp_path), tasks())
    wrong = next(r for r in reports if r.method == "wrong_length_method")

    assert (wrong.invalid, wrong.crashed) == (2, 0)
    assert any("expected 3" in o.detail for o in outcomes if o.method == "wrong_length_method")


def test_a_correct_method_is_scored_on_every_task(tmp_path: Path) -> None:
    outcomes, reports = run_module(write_fixture(tmp_path), tasks())
    perfect = next(r for r in reports if r.method == "perfect_method")
    perfect_outcomes = [o for o in outcomes if o.method == "perfect_method"]

    assert (perfect.success, perfect.coverage) == (2, 1.0)
    assert perfect.mean_smape == pytest.approx(0.0)
    assert perfect.mean_mae == pytest.approx(0.0)
    assert [outcome.forecast for outcome in perfect_outcomes] == [
        (19.0, 19.0, 19.0),
        (29.0, 29.0, 29.0),
    ]


def test_successful_outcome_records_capped_and_raw_scaled_metrics(tmp_path: Path) -> None:
    outcomes, reports = run_module(write_fixture(tmp_path), tasks(), isolated=True)
    row = next(outcome for outcome in outcomes if outcome.method == "perfect_method")
    report = next(item for item in reports if item.method == "perfect_method")

    assert row.smae is not None and row.srmse is not None
    assert row.smae_raw is not None and row.srmse_raw is not None
    assert next(item for item in reports if item.method == "picky_method").mean_smae is None
    assert report.mean_smae == row.smae
    assert report.mean_srmse == row.srmse
    assert report.by_characteristic_smae["frequency:1 day"] == row.smae
    assert report.by_characteristic_srmse["frequency:1 day"] == row.srmse


def test_outcome_statuses_cover_every_method_and_task(tmp_path: Path) -> None:
    outcomes, _ = run_module(write_fixture(tmp_path), tasks())

    assert len(outcomes) == 4 * 2
    assert {o.status for o in outcomes} == {SUCCESS, NOT_APPLICABLE, CRASHED, INVALID}


def test_scores_are_grouped_by_series_characteristic(tmp_path: Path) -> None:
    _, reports = run_module(write_fixture(tmp_path), tasks())
    perfect = next(r for r in reports if r.method == "perfect_method")

    assert "frequency:1 day" in perfect.by_characteristic
    assert any(tag.startswith("history:") for tag in perfect.by_characteristic)


def test_characteristics_flag_intermittent_and_trending_series() -> None:
    intermittent = derive_characteristics([0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0], 4, "1 day")
    trending = derive_characteristics([float(i) for i in range(40)], 4, "1 hour")
    flat = derive_characteristics([5.0] * 40, 4, "1 hour")

    assert "intermittent" in intermittent
    assert "trending" in trending
    assert "flat" in flat


def test_characteristics_distinguish_integer_counts_from_continuous_series() -> None:
    counts = derive_characteristics([0.0, 1.0, 0.0, 4.0, 2.0, 0.0], 2, "1 day")
    continuous = derive_characteristics([0.1, 0.4, 1.3, 2.7, 3.2, 4.8], 2, "1 day")
    signed = derive_characteristics([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], 2, "1 day")

    assert {"nonnegative", "integer_valued", "many_zeros"}.issubset(counts)
    assert {"nonnegative", "continuous_valued", "no_zeros"}.issubset(continuous)
    assert "signed" in signed


def test_characteristics_never_read_the_future() -> None:
    task = Task("t", (1.0, 2.0, 3.0, 4.0), 2, "1 day", (999.0, 999.0))

    # Derived only from history/horizon/frequency, so an extreme future cannot leak in.
    assert task.characteristics() == derive_characteristics(task.history, 2, "1 day")


def test_report_payload_is_json_shaped(tmp_path: Path) -> None:
    import json

    _, reports = run_module(write_fixture(tmp_path), tasks())
    payload = report_payload(reports)

    assert json.loads(json.dumps(payload))
    assert {entry["method"] for entry in payload} == {
        "perfect_method", "picky_method", "broken_method", "wrong_length_method"
    }


def test_a_module_without_not_applicable_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "methods.py"
    bad.write_text(
        'def alpha(history, horizon, frequency):\n    """Doc."""\n    return [1.0] * horizon\n',
        encoding="utf-8",
    )

    with pytest.raises(ImportError, match="NotApplicable"):
        run_module(bad, tasks())


def test_isolated_execution_enforces_a_hard_timeout_without_stopping_other_methods(
    tmp_path: Path,
) -> None:
    module = tmp_path / "methods.py"
    module.write_text(
        MODULE_HEADER
        + '''

def slow_method(history, horizon, frequency):
    """Use when a deliberately slow method is under test."""
    import time
    time.sleep(5)
    return [float(history[-1])] * horizon


def perfect_method(history, horizon, frequency):
    """Use when the future repeats the final observation."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )

    outcomes, reports = run_module(
        module,
        tasks(),
        time_budget_s=0.1,
        isolated=True,
        timeout_circuit_breaker=1,
    )

    slow = next(report for report in reports if report.method == "slow_method")
    perfect = next(report for report in reports if report.method == "perfect_method")
    assert (slow.invalid, slow.success) == (2, 0)
    assert all(
        "hard timeout" in outcome.detail
        for outcome in outcomes
        if outcome.method == "slow_method"
    )
    assert (perfect.success, perfect.invalid) == (2, 0)


def test_worker_startup_time_is_not_charged_to_the_method_budget(tmp_path: Path) -> None:
    module = tmp_path / "methods.py"
    module.write_text(
        MODULE_HEADER
        + '''

import time
time.sleep(0.3)

def fast_after_import(history, horizon, frequency):
    """Use when worker startup timing is under test."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )

    outcomes, reports = run_module(
        module,
        tasks()[:1],
        time_budget_s=0.1,
        isolated=True,
    )

    assert outcomes[0].status == SUCCESS
    assert reports[0].success == 1


def test_worker_startup_import_hang_is_bounded_separately(tmp_path: Path) -> None:
    module = tmp_path / "methods.py"
    module.write_text(
        MODULE_HEADER
        + '''

import time
time.sleep(30)

def never_loaded(history, horizon, frequency):
    """Use when startup-hang containment is under test."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )

    outcomes, reports = run_module(
        module,
        tasks()[:1],
        time_budget_s=0.1,
        worker_startup_timeout_s=0.1,
        isolated=True,
    )

    assert outcomes[0].status == INVALID
    assert "startup timeout" in outcomes[0].detail
    assert reports[0].invalid == 1


def test_isolated_execution_contains_a_native_style_worker_exit(tmp_path: Path) -> None:
    module = tmp_path / "methods.py"
    module.write_text(
        MODULE_HEADER
        + '''

def exiting_method(history, horizon, frequency):
    """Use when a native-style worker failure is under test."""
    import os
    os._exit(17)


def perfect_method(history, horizon, frequency):
    """Use when the future repeats the final observation."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )

    outcomes, reports = run_module(module, tasks(), isolated=True, time_budget_s=1.0)

    exiting = next(report for report in reports if report.method == "exiting_method")
    perfect = next(report for report in reports if report.method == "perfect_method")
    assert exiting.crashed == 2
    assert all(
        "worker exited" in outcome.detail
        for outcome in outcomes
        if outcome.method == "exiting_method"
    )
    assert perfect.success == 2


def test_isolated_forecast_runtime_contains_exit_and_continues(tmp_path: Path) -> None:
    module = tmp_path / "methods.py"
    module.write_text(
        MODULE_HEADER
        + '''

def exiting_method(history, horizon, frequency):
    """Use when a native-style worker failure is under test."""
    import os
    os._exit(23)


def perfect_method(history, horizon, frequency):
    """Use when the forecast repeats the final observation."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )

    runtime = IsolatedForecastRuntime(module, time_budget_s=1.0)
    try:
        with pytest.raises(MethodForecastError, match="worker exited"):
            runtime.forecast("exiting_method", (1.0, 2.0), 2, "D")
        assert runtime.forecast("perfect_method", (1.0, 2.0), 2, "D") == (2.0, 2.0)
        with pytest.raises(MethodForecastError, match="previously failed"):
            runtime.forecast("exiting_method", (1.0, 2.0), 2, "D")
    finally:
        runtime.close()


def test_isolated_forecast_runtime_hard_timeout_does_not_block_next_method(
    tmp_path: Path,
) -> None:
    module = tmp_path / "methods.py"
    module.write_text(
        MODULE_HEADER
        + '''

def hanging_method(history, horizon, frequency):
    """Use when hard-timeout containment is under test."""
    import time
    time.sleep(30)
    return [float(history[-1])] * horizon


def perfect_method(history, horizon, frequency):
    """Use when the forecast repeats the final observation."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )

    runtime = IsolatedForecastRuntime(module, time_budget_s=0.1)
    try:
        with pytest.raises(MethodForecastError, match="hard timeout"):
            runtime.forecast("hanging_method", (1.0, 2.0), 2, "D")
        assert runtime.forecast("perfect_method", (1.0, 2.0), 2, "D") == (2.0, 2.0)
    finally:
        runtime.close()


def test_isolated_execution_restarts_a_worker_after_one_task_exits(tmp_path: Path) -> None:
    module = tmp_path / "methods.py"
    module.write_text(
        MODULE_HEADER
        + '''

def sometimes_exiting_method(history, horizon, frequency):
    """Use when worker recovery across tasks is under test."""
    if len(history) == 20:
        import os
        os._exit(19)
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )

    outcomes, reports = run_module(module, tasks(), isolated=True, time_budget_s=1.0)

    report = reports[0]
    assert (report.crashed, report.success) == (1, 1)
    assert outcomes[0].status == CRASHED
    assert outcomes[1].status == SUCCESS


def _ignore_sigterm_forever(ready: object) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.send(True)  # type: ignore[attr-defined]
    while True:
        time.sleep(30)


def test_hard_timeout_cleanup_kills_a_worker_that_ignores_sigterm() -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_ignore_sigterm_forever, args=(child,))
    worker.start()
    child.close()
    assert parent.poll(5.0) and parent.recv() is True

    _stop_worker(worker)

    parent.close()
    assert worker.is_alive() is False
    assert worker.exitcode is not None
