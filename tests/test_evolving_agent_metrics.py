from __future__ import annotations

import unittest

from common.metrics import (
    change_mae,
    mae,
    mse,
    score_forecast,
    shape_correlation,
    smape,
    spearman_rank_correlation,
    variance_ratio,
)


class SmapeTests(unittest.TestCase):
    def test_perfect_forecast_is_zero(self):
        self.assertEqual(smape([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0)

    def test_zero_true_and_zero_pred_contributes_zero_not_nan(self):
        self.assertEqual(smape([0.0], [0.0]), 0.0)

    def test_known_value(self):
        # |1-2| / ((1+2)/2) * 100 = 1 / 1.5 * 100 = 66.666...
        self.assertAlmostEqual(smape([1.0], [2.0]), 66.6666, places=3)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            smape([1.0, 2.0], [1.0])

    def test_empty_inputs(self):
        self.assertEqual(smape([], []), 0.0)


class MaeTests(unittest.TestCase):
    def test_perfect_forecast_is_zero(self):
        self.assertEqual(mae([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_known_value(self):
        self.assertEqual(mae([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]), 1.0)


class MseTests(unittest.TestCase):
    def test_perfect_forecast_is_zero(self):
        self.assertEqual(mse([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_known_value(self):
        # (1 + 0 + 4) / 3
        self.assertAlmostEqual(mse([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]), 5.0 / 3.0)

    def test_penalizes_large_errors_more_than_mae(self):
        self.assertGreater(mse([0.0, 0.0], [0.0, 10.0]), mae([0.0, 0.0], [0.0, 10.0]))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mse([1.0, 2.0], [1.0])

    def test_empty_inputs(self):
        self.assertEqual(mse([], []), 0.0)


class ScoreForecastTests(unittest.TestCase):
    def test_returns_both_metrics_and_a_primary(self):
        result = score_forecast([1.0, 2.0], [1.0, 3.0])
        self.assertIn("smape", result)
        self.assertIn("mae", result)
        self.assertEqual(result["primary"], result["smape"])


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
        )

    def test_constant_rank_has_no_information(self):
        self.assertEqual(spearman_rank_correlation([1.0, 1.0], [2.0, 3.0]), 0.0)


if __name__ == "__main__":
    unittest.main()


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
