"""Tests for common/metrics: MAE/RMSE building blocks, the Dr-CiK scaled metrics, and forecast-shape metrics."""
from __future__ import annotations

import unittest
import pytest
from common.metrics import (
    change_mae,
    change_smae,
    mean_absolute_truth,
    mae,
    rmse,
    scaled_mae,
    scaled_rmse,
    shape_correlation,
    spearman_rank_correlation,
    variance_ratio,
)


class MaeTests(unittest.TestCase):
    def test_perfect_forecast_is_zero(self):
        self.assertEqual(mae([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_known_value(self):
        self.assertEqual(mae([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]), 1.0)


class RmseTests(unittest.TestCase):
    def test_perfect_forecast_is_zero(self):
        self.assertEqual(rmse([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_known_value(self):
        # sqrt((1 + 0 + 4) / 3), rounded to 3 decimal places like every metric in this module.
        self.assertAlmostEqual(rmse([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]), (5.0 / 3.0) ** 0.5, places=3)

    def test_penalizes_large_errors_more_than_mae(self):
        self.assertGreater(rmse([0.0, 0.0], [0.0, 10.0]), mae([0.0, 0.0], [0.0, 10.0]))

    def test_is_in_the_same_units_as_mae_not_squared(self):
        # RMSE of a constant-error series equals that error exactly, unlike raw MSE.
        self.assertAlmostEqual(rmse([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]), 3.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rmse([1.0, 2.0], [1.0])

    def test_empty_inputs(self):
        self.assertEqual(rmse([], []), 0.0)


class SpearmanRankCorrelationTests(unittest.TestCase):
    def test_perfect_and_inverse_order(self):
        self.assertAlmostEqual(
            spearman_rank_correlation([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
            1.0,
        )
        self.assertAlmostEqual(
            spearman_rank_correlation([1.0, 2.0, 3.0], [6.0, 5.0, 4.0]),
            -1.0,
        )

    def test_ties_use_average_ranks(self):
        self.assertAlmostEqual(
            spearman_rank_correlation([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]),
            0.8660254038,
            places=3,
        )

    def test_constant_rank_has_no_information(self):
        self.assertEqual(spearman_rank_correlation([1.0, 1.0], [2.0, 3.0]), 0.0)


class ShapeMetricTests(unittest.TestCase):
    TRUTH = [1.0, 5.0, 2.0, 6.0, 3.0, 7.0]

    def test_a_flat_forecast_has_no_variance_and_no_correlation(self):
        flat = [4.0] * 6

        self.assertEqual(variance_ratio(self.TRUTH, flat), 0.0)
        self.assertEqual(shape_correlation(self.TRUTH, flat), 0.0)

    def test_a_tracking_forecast_scores_near_one_on_both(self):
        tracking = [1.2, 4.8, 2.1, 5.9, 3.1, 6.8]

        self.assertGreater(variance_ratio(self.TRUTH, tracking), 0.85)
        self.assertGreater(shape_correlation(self.TRUTH, tracking), 0.95)

    def test_variance_ratio_exceeds_one_when_the_forecast_overshoots(self):
        exaggerated = [(value - 4.0) * 3.0 + 4.0 for value in self.TRUTH]

        self.assertAlmostEqual(variance_ratio(self.TRUTH, exaggerated), 3.0)

    def test_shape_correlation_is_negative_for_an_inverted_forecast(self):
        inverted = [8.0 - value for value in self.TRUTH]

        self.assertAlmostEqual(shape_correlation(self.TRUTH, inverted), -1.0)

    def test_a_constant_truth_is_matched_only_by_a_constant_forecast(self):
        self.assertEqual(variance_ratio([3.0] * 5, [3.0] * 5), 1.0)
        self.assertEqual(variance_ratio([3.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0]), 0.0)

    def test_change_mae_of_a_flat_forecast_equals_the_series_own_volatility(self):
        flat = [1.0] * 6  # last observed is also 1.0, so every predicted step is zero

        steps = [abs(self.TRUTH[0] - 1.0)] + [
            abs(self.TRUTH[i] - self.TRUTH[i - 1]) for i in range(1, 6)
        ]
        self.assertAlmostEqual(change_mae(self.TRUTH, flat, 1.0), sum(steps) / len(steps))

    def test_change_mae_is_zero_for_a_perfect_forecast(self):
        self.assertEqual(change_mae(self.TRUTH, self.TRUTH, 1.0), 0.0)

    def test_shape_metrics_reject_a_length_mismatch(self):
        for metric in (variance_ratio, shape_correlation):
            with self.assertRaises(ValueError):
                metric([1.0, 2.0], [1.0])


class ChangeSmaeTests(unittest.TestCase):
    TRUTH = [1.0, 5.0, 2.0, 6.0, 3.0, 7.0]

    def test_is_zero_for_a_perfect_forecast(self):
        self.assertEqual(change_smae(self.TRUTH, self.TRUTH, 1.0), 0.0)

    def test_equals_change_mae_divided_by_the_mean_absolute_truth(self):
        forecast = [4.0] * 6
        scale = sum(abs(value) for value in self.TRUTH) / len(self.TRUTH)
        expected = change_mae(self.TRUTH, forecast, 1.0) / scale
        self.assertAlmostEqual(change_smae(self.TRUTH, forecast, 1.0), expected)

    def test_is_not_clamped_for_a_catastrophic_forecast(self):
        forecast = [1000.0] * 6
        self.assertGreater(change_smae(self.TRUTH, forecast, 1.0), 5.0)


def test_the_scale_is_the_mean_absolute_truth_over_the_horizon() -> None:
    # a^-1 = (1/T) sum |y_t|, the Dr-CiK scale factor.
    assert mean_absolute_truth([1.0, -3.0, 2.0, 6.0]) == pytest.approx(3.0)


def test_the_scale_uses_magnitude_so_sign_does_not_cancel() -> None:
    assert mean_absolute_truth([5.0, -5.0]) == pytest.approx(5.0)


def test_an_all_zero_horizon_never_divides_by_zero() -> None:
    """With nothing to scale by, the error stays in its own units rather than blowing up."""
    assert mean_absolute_truth([0.0, 0.0]) == 1.0
    assert scaled_mae([0.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0)


def test_an_empty_horizon_is_still_a_valid_scale() -> None:
    assert mean_absolute_truth([]) == 1.0


def test_smae_is_the_error_as_a_fraction_of_the_series_magnitude() -> None:
    truth, prediction = [10.0, 10.0], [11.0, 11.0]

    # MAE of 1.0 against a mean absolute truth of 10.0.
    assert scaled_mae(truth, prediction) == pytest.approx(0.1)


def test_scaling_makes_two_series_of_different_magnitude_comparable() -> None:
    """The whole point: a big-magnitude series must not dominate a mean over tasks."""
    small_smae = scaled_mae([10.0], [12.0])
    large_smae = scaled_mae([10000.0], [12000.0])

    assert mae([10.0], [12.0]) * 1000 == pytest.approx(mae([10000.0], [12000.0]))
    assert small_smae == pytest.approx(large_smae)


def test_a_catastrophic_forecast_keeps_its_true_magnitude() -> None:
    """Nothing is clamped: how bad a blown task was is itself the evidence for deleting a method."""
    assert scaled_mae([1.0], [1001.0]) == pytest.approx(1000.0)
    assert scaled_mae([1.0], [1e9]) == pytest.approx(1e9 - 1.0)


def test_scaled_rmse_still_punishes_one_large_error_more_than_scaled_mae() -> None:
    truth, prediction = [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 5.0]

    assert rmse(truth, prediction) > mae(truth, prediction)
    assert scaled_rmse(truth, prediction) > scaled_mae(truth, prediction)


def test_the_scale_does_not_depend_on_the_history(tmp_path) -> None:
    """Two tasks with the same horizon score alike however different their histories."""
    assert scaled_mae([10.0, 10.0], [11.0, 11.0]) == pytest.approx(
        scaled_mae([10.0, 10.0], [11.0, 11.0])
    )


def test_the_report_carries_the_scaled_metrics(tmp_path) -> None:
    from numerical_agent.evolution.execution import Task, run_module
    from numerical_agent.evolution.module import METHODS_FILE_HEADER

    path = tmp_path / "methods.py"
    path.write_text(
        METHODS_FILE_HEADER
        + "\n\ndef flat(history, horizon, frequency):\n"
        '    """Repeat the last value."""\n'
        "    return [float(history[-1])] * horizon\n",
        encoding="utf-8",
    )
    history = tuple(float(i % 24) for i in range(240))
    task = Task("t", history, 4, "1 hour", (1.0, 2.0, 3.0, 4.0))

    _outcomes, reports = run_module(path, (task,))
    report = reports[0]

    assert report.mean_smae is not None
    assert report.mean_srmse is not None
    assert report.smae_by_series_type
