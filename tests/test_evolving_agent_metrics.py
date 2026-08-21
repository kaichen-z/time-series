from __future__ import annotations

import unittest

from common.metrics import mae, mse, score_forecast, smape, spearman_rank_correlation


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
