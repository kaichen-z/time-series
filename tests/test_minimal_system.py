from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from drcik_agent.metrics import crps_ensemble
from drcik_agent.models import Document, ForecastTask
from drcik_agent.pipeline import MinimalAgentSystem, SystemConfig, write_outputs


def example_task() -> ForecastTask:
    future_timestamps = ("2024-01-03 00:00:00", "2024-01-03 01:00:00")
    return ForecastTask(
        benchmark_id="task_test",
        entity_name="Alpha Station",
        target_name="energy demand",
        target_description="Hourly energy demand at Alpha Station",
        frequency="1 hour",
        prediction_length=2,
        seasonal_period=2,
        history_timestamps=(
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-02 00:00:00",
            "2024-01-02 01:00:00",
        ),
        history_values=(10.0, 20.0, 11.0, 21.0),
        future_timestamps=future_timestamps,
        future_values=(12.0, 22.0),
        documents=(
            Document(
                document_id="doc_support",
                text=(
                    "Alpha Station hourly energy demand forecast. "
                    "The expected values are (2024-01-03 00:00:00, 12.0) and "
                    "(2024-01-03 01:00:00, 22.0)."
                ),
                role="supporting",
            ),
            Document(
                document_id="doc_noise",
                text="A maintenance handbook for an unrelated water pump from 2018.",
                role="distractor",
                subtype="noisy",
            ),
        ),
        gt_evidence=("Expected energy demand is 12.0 and 22.0.",),
    )


class MinimalSystemTest(unittest.TestCase):
    def test_perfect_ensemble_has_zero_crps(self) -> None:
        score = crps_ensemble((1.0, 2.0), ((1.0, 2.0), (1.0, 2.0)))
        self.assertAlmostEqual(score, 0.0)

    def test_end_to_end_uses_context_without_label_leakage(self) -> None:
        task = example_task()
        system = MinimalAgentSystem(
            SystemConfig(top_k=1, num_samples=100, context_weight=1.0, seed=1)
        )
        result = system.run(task)

        self.assertEqual(result.retrieved[0].document.document_id, "doc_support")
        self.assertIsNone(result.retrieved[0].document.role)
        self.assertEqual(result.forecast.mean, (12.0, 22.0))
        self.assertEqual(len(result.forecast.samples), 100)
        self.assertEqual(result.metrics["supporting_document_recall"], 1.0)

    def test_outputs_match_submission_shapes(self) -> None:
        result = MinimalAgentSystem(
            SystemConfig(top_k=1, num_samples=100, context_weight=1.0)
        ).run(example_task())
        with tempfile.TemporaryDirectory() as temporary_directory:
            write_outputs([result], temporary_directory)
            root = Path(temporary_directory)
            forecast = json.loads((root / "forecasts.jsonl").read_text(encoding="utf-8"))
            research = json.loads((root / "deep_research.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(forecast["benchmark_id"], "task_test")
            self.assertEqual(len(forecast["samples"]), 100)
            self.assertEqual(len(forecast["samples"][0]), 2)
            self.assertEqual(research["cited_document_ids"], ["doc_support"])


if __name__ == "__main__":
    unittest.main()
