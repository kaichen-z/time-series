"""Hindcast window carving and the scale-free error used to rank hypotheses."""

from __future__ import annotations

import pytest

from evolving_agents.harness.hindcast import carve_hindcast_windows, scaled_error

HISTORY = tuple(float(index) for index in range(100))


def test_windows_are_carved_from_the_end_backward() -> None:
    windows = carve_hindcast_windows(HISTORY, horizon=10, frequency="H", n_windows=3)
    assert len(windows) == 3
    assert windows[0].held_out_future == HISTORY[90:100]
    assert windows[0].train_history == HISTORY[:90]
    assert windows[1].held_out_future == HISTORY[80:90]
    assert windows[2].held_out_future == HISTORY[70:80]


def test_windows_never_touch_data_after_their_own_train_split() -> None:
    for window in carve_hindcast_windows(HISTORY, horizon=10, frequency="H", n_windows=3):
        assert len(window.train_history) + len(window.held_out_future) <= len(HISTORY)
        assert window.train_history == HISTORY[: len(window.train_history)]


def test_min_train_length_stops_carving() -> None:
    windows = carve_hindcast_windows(HISTORY, horizon=30, frequency="H", n_windows=5, min_train_length=20)
    assert len(windows) == 2  # a third window would leave only 10 training points
    assert all(len(window.train_history) >= 20 for window in windows)


def test_horizon_longer_than_history_yields_nothing() -> None:
    assert carve_hindcast_windows(HISTORY, horizon=200, frequency="H") == ()


@pytest.mark.parametrize("history,horizon", [((), 5), (HISTORY, 0), (HISTORY, -1)])
def test_degenerate_inputs_yield_nothing(history, horizon) -> None:
    assert carve_hindcast_windows(history, horizon=horizon, frequency="H") == ()


def test_scaled_error_is_zero_for_a_perfect_forecast() -> None:
    assert scaled_error((10.0, 20.0), (10.0, 20.0)) == 0.0


def test_scaled_error_is_scale_free() -> None:
    small = scaled_error((10.0, 20.0), (11.0, 22.0))
    large = scaled_error((1000.0, 2000.0), (1100.0, 2200.0))
    assert small == pytest.approx(large)


def test_scaled_error_handles_an_all_zero_truth() -> None:
    assert scaled_error((0.0, 0.0), (1.0, 1.0)) == 1.0


def test_scaled_error_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError):
        scaled_error((1.0, 2.0), (1.0,))
