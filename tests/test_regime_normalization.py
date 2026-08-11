from __future__ import annotations

import math
import unittest

from drcik_agent.models import Diagnosis, Document, Evidence, ForecastTask
from drcik_agent.regimes import RegimeNormalizationAgent


class RegimeNormalizationTest(unittest.TestCase):
    @staticmethod
    def _task() -> ForecastTask:
        history = tuple(
            10.0 + 0.3 * index + 2.0 * math.sin(2.0 * math.pi * index / 4)
            for index in range(13)
        )
        return ForecastTask(
            benchmark_id="regime_test",
            entity_name="Example Store",
            target_name="sales",
            target_description="Example Store sales",
            frequency="1 day",
            prediction_length=8,
            seasonal_period=4,
            history_timestamps=tuple(f"2026-01-{index + 1:02d} 00:00:00" for index in range(13)),
            history_values=history,
            future_timestamps=tuple(f"2026-01-{index + 14:02d} 00:00:00" for index in range(8)),
            future_values=None,
            documents=(Document("doc_normal", "Sales returned to normal."),),
        )

    @staticmethod
    def _diagnosis() -> Diagnosis:
        return Diagnosis(
            trend="upward",
            slope_per_step=0.3,
            seasonal_period=4,
            seasonal_strength=0.95,
            residual_scale=0.5,
            information_needs=(),
            retrieval_query="",
        )

    @staticmethod
    def _evidence() -> list[Evidence]:
        return [
            Evidence(
                document_id="doc_normal",
                claim="After the temporary event, sales should return to normal.",
                matched_terms=("normal",),
                confidence=0.9,
                effect_direction="unclear",
                effect_window="future",
            )
        ]

    def test_normalization_evidence_opens_history_only_regime_projection(self) -> None:
        task = self._task()
        baseline = tuple(10.0 for _ in task.future_timestamps)
        projection = RegimeNormalizationAgent().project(
            task, self._diagnosis(), baseline, self._evidence()
        )

        self.assertIsNotNone(projection)
        self.assertEqual(len(projection.values), task.prediction_length)
        self.assertEqual(projection.source_document_ids, ("doc_normal",))
        self.assertLess(projection.validation_mae, 1e-4)
        self.assertGreaterEqual(projection.blend_weight, 0.2)
        self.assertIn("Magnitudes use history only", projection.rationale)

    def test_projection_requires_normalization_evidence_and_strong_seasonality(self) -> None:
        task = self._task()
        baseline = tuple(10.0 for _ in task.future_timestamps)
        agent = RegimeNormalizationAgent()
        self.assertIsNone(agent.project(task, self._diagnosis(), baseline, []))

        weak = Diagnosis(
            trend="stable",
            slope_per_step=0.0,
            seasonal_period=4,
            seasonal_strength=0.4,
            residual_scale=1.0,
            information_needs=(),
            retrieval_query="",
        )
        self.assertIsNone(agent.project(task, weak, baseline, self._evidence()))

    def test_weak_stabilization_claim_must_name_the_target_variable(self) -> None:
        task = self._task()
        baseline = tuple(10.0 for _ in task.future_timestamps)
        unrelated = [
            Evidence(
                document_id="doc_wrong_variable",
                claim="Temperature stabilization was completed after maintenance.",
                matched_terms=("stabilization",),
                confidence=0.9,
                effect_direction="unclear",
                effect_window="future",
            )
        ]
        unrelated_baseline = [
            Evidence(
                document_id="doc_storage",
                claim="The local storage protocol reverted to baseline after maintenance.",
                matched_terms=("baseline",),
                confidence=0.9,
                effect_direction="unclear",
                effect_window="future",
            )
        ]
        related = [
            Evidence(
                document_id="doc_sales_stable",
                claim="Sales stabilized after the temporary event.",
                matched_terms=("sales", "stabilized"),
                confidence=0.9,
                effect_direction="unclear",
                effect_window="future",
            )
        ]

        agent = RegimeNormalizationAgent()
        self.assertIsNone(agent.project(task, self._diagnosis(), baseline, unrelated))
        self.assertIsNone(
            agent.project(task, self._diagnosis(), baseline, unrelated_baseline)
        )
        self.assertIsNotNone(agent.project(task, self._diagnosis(), baseline, related))


if __name__ == "__main__":
    unittest.main()
