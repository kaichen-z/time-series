from __future__ import annotations

import math
import unittest

from common.metrics import (
    aggregate_drcik_point_metrics,
    drcik_point_metrics,
    joint_scaled_error,
    mae,
    pareto_scaled_improvement,
    rmse,
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


class DrCikPointMetricTests(unittest.TestCase):
    def test_joint_scaled_error_and_pareto_gate_keep_metrics_separate(self):
        self.assertEqual(joint_scaled_error(1.0, 3.0), 2.0)
        self.assertTrue(
            pareto_scaled_improvement(1.0, 1.0, 0.9, 1.0, tolerance=1e-12)
        )
        self.assertFalse(
            pareto_scaled_improvement(1.0, 1.0, 0.5, 1.01, tolerance=1e-12)
        )

    def test_reduces_multiple_trajectories_to_their_stepwise_mean(self):
        result = drcik_point_metrics(
            [2.0, 4.0],
            [[1.0, 3.0], [3.0, 5.0]],
        )

        self.assertEqual(result["smae"], 0.0)
        self.assertEqual(result["srmse"], 0.0)

    def test_aggregate_averages_already_capped_task_metrics(self):
        result = aggregate_drcik_point_metrics(
            [{"smae": 1.0, "srmse": 2.0}, {"smae": 3.0, "srmse": 4.0}]
        )

        self.assertEqual(result, {"smae": 2.0, "srmse": 3.0})

    def test_scales_by_mean_absolute_future_and_winsorizes_each_metric(self):
        result = drcik_point_metrics(
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 25.0],
        )

        self.assertEqual(result["scale"], 1.0)
        self.assertEqual(result["mae"], 6.0)
        self.assertEqual(result["rmse"], 12.0)
        self.assertEqual(result["smae_raw"], 6.0)
        self.assertEqual(result["srmse_raw"], 12.0)
        self.assertEqual(result["smae"], 5.0)
        self.assertEqual(result["srmse"], 5.0)
        self.assertTrue(result["smae_clipped"])
        self.assertTrue(result["srmse_clipped"])

    def test_zero_scale_is_perfect_only_for_an_exact_zero_forecast(self):
        perfect = drcik_point_metrics([0.0, 0.0], [0.0, 0.0])
        wrong = drcik_point_metrics([0.0, 0.0], [1.0, 1.0])

        self.assertEqual(perfect["smae"], 0.0)
        self.assertEqual(perfect["srmse"], 0.0)
        self.assertFalse(perfect["smae_clipped"])
        self.assertEqual(wrong["smae"], 5.0)
        self.assertEqual(wrong["srmse"], 5.0)
        self.assertTrue(wrong["smae_clipped"])
        self.assertTrue(wrong["srmse_clipped"])

    def test_finite_extreme_prediction_is_clipped_instead_of_crashing(self):
        result = drcik_point_metrics([1.0, 1.0], [1e308, 1e308])

        self.assertTrue(math.isfinite(result["rmse"]))
        self.assertEqual(result["srmse"], 5.0)
        self.assertTrue(result["srmse_clipped"])


class RmseTests(unittest.TestCase):
    def test_uses_a_scale_stable_norm_for_large_finite_errors(self):
        self.assertTrue(math.isfinite(rmse([0.0, 0.0], [1e308, 1e308])))


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
