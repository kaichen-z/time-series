from __future__ import annotations

import unittest

from common.metrics import (
    aggregate_drcik_metrics,
    drcik_task_metrics,
    mae,
    score_forecast,
    smape,
    spearman_rank_correlation,
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


class ScoreForecastTests(unittest.TestCase):
    def test_returns_both_metrics_and_a_primary(self):
        result = score_forecast([1.0, 2.0], [1.0, 3.0])
        self.assertIn("smape", result)
        self.assertIn("mae", result)
        self.assertEqual(result["primary"], result["smape"])


class DrCikMetricTests(unittest.TestCase):
    def test_matches_appendix_formula_for_two_trajectories(self):
        result = drcik_task_metrics([1.0, 3.0], [[1.0, 1.0], [3.0, 5.0]])

        self.assertAlmostEqual(result["smae"], 0.25)
        self.assertAlmostEqual(result["srmse"], 2**0.5 / 4)
        self.assertAlmostEqual(result["scrps"], 0.375)

    def test_deterministic_crps_equals_scaled_mae_and_metrics_are_capped(self):
        deterministic = drcik_task_metrics([2.0, 4.0], [[3.0, 2.0]])
        capped = drcik_task_metrics([1.0], [[100.0]])

        self.assertAlmostEqual(deterministic["scrps"], deterministic["smae"])
        self.assertEqual(capped["smae"], 5.0)
        self.assertEqual(capped["srmse"], 5.0)
        self.assertEqual(capped["scrps"], 5.0)

    def test_aggregate_reports_sample_standard_error(self):
        result = aggregate_drcik_metrics(
            [
                {"smae": 0.0, "srmse": 1.0, "scrps": 2.0},
                {"smae": 2.0, "srmse": 3.0, "scrps": 4.0},
            ]
        )

        self.assertEqual(result["num_tasks"], 2)
        self.assertEqual(result["smae"], 1.0)
        self.assertAlmostEqual(result["smae_se"], 1.0)


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
