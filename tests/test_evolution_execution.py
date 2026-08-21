from __future__ import annotations

from pathlib import Path

import pytest

from numerical_agent.evolution.execution import (
    CRASHED,
    INVALID,
    NOT_APPLICABLE,
    SUCCESS,
    Task,
    derive_characteristics,
    load_methods,
    report_payload,
    run_module,
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
    assert picky.mean_mae is None and broken.mean_mae is None


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
    _, reports = run_module(write_fixture(tmp_path), tasks())
    perfect = next(r for r in reports if r.method == "perfect_method")

    assert (perfect.success, perfect.coverage) == (2, 1.0)
    assert perfect.mean_mae == pytest.approx(0.0)
    assert perfect.mean_rmse == pytest.approx(0.0)


def test_outcome_statuses_cover_every_method_and_task(tmp_path: Path) -> None:
    outcomes, _ = run_module(write_fixture(tmp_path), tasks())

    assert len(outcomes) == 4 * 2
    assert {o.status for o in outcomes} == {SUCCESS, NOT_APPLICABLE, CRASHED, INVALID}


def test_scores_are_grouped_by_series_characteristic(tmp_path: Path) -> None:
    _, reports = run_module(write_fixture(tmp_path), tasks())
    perfect = next(r for r in reports if r.method == "perfect_method")

    assert "frequency:1 day" in perfect.by_characteristic_mae
    assert any(tag.startswith("history:") for tag in perfect.by_characteristic_mae)


def test_characteristics_flag_intermittent_and_trending_series() -> None:
    intermittent = derive_characteristics([0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0], 4, "1 day")
    trending = derive_characteristics([float(i) for i in range(40)], 4, "1 hour")
    flat = derive_characteristics([5.0] * 40, 4, "1 hour")

    assert "intermittent" in intermittent
    assert "trending" in trending
    assert "flat" in flat


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


def test_an_imported_callable_is_never_measured_as_a_method(tmp_path: Path) -> None:
    """A module-level import must not be scored as a forecasting method.

    Without this filter, `sqrt` would be called as sqrt(history, horizon, frequency) on every
    task, raise TypeError each time, and land in the report as a method that crashed everywhere
    -- evidence the model would then act on for a method that does not exist.
    """
    source = MODULE_HEADER + '''

from math import sqrt


def real_method(history, horizon, frequency):
    """A genuine method."""
    return [float(history[-1])] * horizon
'''
    destination = tmp_path / "methods.py"
    destination.write_text(source, encoding="utf-8")

    _module, functions = load_methods(destination)

    assert set(functions) == {"real_method"}
    assert "sqrt" not in functions
    assert "P" not in functions


def test_methods_can_call_the_frozen_skill_library(tmp_path: Path) -> None:
    source = MODULE_HEADER + '''

def period_repeat(history, horizon, frequency):
    """Repeat the last full seasonal cycle the skill library detects."""
    period = P.infer_period(history, frequency)
    return [float(history[-period + (i % period)]) for i in range(horizon)]
'''
    destination = tmp_path / "methods.py"
    destination.write_text(source, encoding="utf-8")

    _module, functions = load_methods(destination)

    assert set(functions) == {"period_repeat"}
    assert functions["period_repeat"]([1.0, 9.0] * 30, 4, "1 day") == [1.0, 9.0, 1.0, 9.0]


def test_mean_rank_orders_methods_within_a_task_not_by_raw_magnitude(tmp_path: Path) -> None:
    """One large-magnitude series must not decide the whole comparison.

    `steady` is beaten on the small task and wins the huge one; `spiky` is the reverse. Their
    mean MAE is dominated by the huge task, but their mean ranks stay balanced.
    """
    source = MODULE_HEADER + '''

def steady(history, horizon, frequency):
    """Always predicts the last value."""
    return [float(history[-1])] * horizon


def spiky(history, horizon, frequency):
    """Always predicts the last value plus one."""
    return [float(history[-1]) + 1.0] * horizon
'''
    destination = tmp_path / "methods.py"
    destination.write_text(source, encoding="utf-8")
    tasks = (
        Task("small", (1.0, 2.0, 3.0), 2, "1 day", (3.0, 3.0)),
        Task("huge", (0.0, 0.0, 1e6), 2, "1 day", (1e6 + 1.0, 1e6 + 1.0)),
    )

    _outcomes, reports = run_module(destination, tasks)
    by_name = {report.method: report for report in reports}

    assert by_name["steady"].mean_rank == pytest.approx(1.5)
    assert by_name["spiky"].mean_rank == pytest.approx(1.5)


def test_a_flat_method_is_visibly_flat_in_the_report(tmp_path: Path) -> None:
    source = MODULE_HEADER + '''

def flat(history, horizon, frequency):
    """Predicts the mean of the history forever."""
    return [sum(history) / len(history)] * horizon
'''
    destination = tmp_path / "methods.py"
    destination.write_text(source, encoding="utf-8")
    task = Task("wave", (1.0, 5.0, 1.0, 5.0), 4, "1 day", (1.0, 5.0, 1.0, 5.0))

    _outcomes, reports = run_module(destination, (task,))

    assert reports[0].mean_variance_ratio == pytest.approx(0.0)
    assert reports[0].mean_shape_correlation == pytest.approx(0.0)
