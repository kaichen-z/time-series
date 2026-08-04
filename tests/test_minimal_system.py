from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from drcik_agent.agents import (
    ProbabilisticForecastAgent,
    TimeSeriesDiagnosisAgent,
    _infer_seasonal_period,
)
from drcik_agent.impacts import EvidenceToForecastAgent
from drcik_agent.metrics import crps_ensemble
from drcik_agent.loop import IterativeAgentSystem, LoopConfig
from drcik_agent.models import Document, ForecastTask, RetrievedDocument
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


def iterative_task() -> ForecastTask:
    base = example_task()
    return ForecastTask(
        benchmark_id="task_loop",
        entity_name=base.entity_name,
        target_name=base.target_name,
        target_description=base.target_description,
        frequency=base.frequency,
        prediction_length=base.prediction_length,
        seasonal_period=base.seasonal_period,
        history_timestamps=base.history_timestamps,
        history_values=base.history_values,
        future_timestamps=base.future_timestamps,
        future_values=base.future_values,
        documents=(
            Document(
                "doc_anomaly",
                "Alpha Station energy demand anomaly was caused by a meter software bug in 2024.",
                "supporting",
            ),
            Document(
                "doc_resolution",
                "Alpha Station energy demand software patch permanently resolved the error and restored normal operation in 2024.",
                "supporting",
            ),
            Document(
                "doc_event",
                "Alpha Station energy demand increased during a temporary 2024 promotion event, which later ended.",
                "supporting",
            ),
            Document(
                "doc_regime",
                "Alpha Station energy demand forecast should follow the normal two-step seasonal cycle and baseline.",
                "supporting",
            ),
            Document(
                "doc_wrong_entity",
                "Beta Harbor energy demand will grow linearly after a permanent policy change.",
                "distractor",
                "profile",
            ),
        ),
        gt_evidence=(
            "A software bug caused an anomaly.",
            "A patch permanently restored normal operation.",
            "A temporary event ended.",
            "The normal seasonal cycle should govern the forecast.",
        ),
    )


def future_impact_task() -> ForecastTask:
    base = example_task()
    return ForecastTask(
        benchmark_id="task_future_impact",
        entity_name=base.entity_name,
        target_name=base.target_name,
        target_description=base.target_description,
        frequency=base.frequency,
        prediction_length=base.prediction_length,
        seasonal_period=base.seasonal_period,
        history_timestamps=base.history_timestamps,
        history_values=base.history_values,
        future_timestamps=base.future_timestamps,
        future_values=base.future_values,
        documents=(
            Document(
                "doc_future_event",
                (
                    "Alpha Station will run a temporary promotion from "
                    "2024-01-03 00:00:00 to 2024-01-03 01:00:00. "
                    "The promotion will increase energy demand by 50 percent throughout the event."
                ),
                "supporting",
            ),
        ),
        gt_evidence=("A future promotion increases energy demand by 50 percent.",),
    )


class MinimalSystemTest(unittest.TestCase):
    def test_explicit_future_percentage_changes_the_numerical_forecast(self) -> None:
        task = future_impact_task()
        document = task.documents[0]
        retrieved = [RetrievedDocument(document=document, score=1.0, rank=1)]
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        impacts = EvidenceToForecastAgent().translate(task, diagnosis, retrieved, [])
        forecaster = ProbabilisticForecastAgent()
        baseline = forecaster.forecast(task, diagnosis, [], 100, 1, 0.75)
        adjusted = forecaster.forecast(task, diagnosis, retrieved, 100, 1, 0.75, impacts)

        self.assertEqual(impacts[0].adjustment_kind, "percentage")
        self.assertAlmostEqual(impacts[0].adjustment_value, 0.5)
        self.assertEqual(adjusted.context_points, {})
        self.assertEqual(
            adjusted.mean,
            tuple(value * 1.5 for value in baseline.mean),
        )
        self.assertEqual(adjusted.impact_adjustments[0].affected_steps, 2)

    def test_iterative_loop_applies_the_translated_future_impact(self) -> None:
        task = future_impact_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        baseline = ProbabilisticForecastAgent().forecast(
            task, diagnosis, [], 100, 1, 0.75
        )
        result = IterativeAgentSystem(
            LoopConfig(max_steps=10, documents_per_step=1, max_no_progress=4, seed=1)
        ).run(task)

        self.assertIn("external_drivers", result.belief_state.answered_question_ids)
        self.assertEqual(
            result.forecast.mean,
            tuple(value * 1.5 for value in baseline.mean),
        )
        self.assertTrue(
            any(
                adjustment.mean_absolute_change > 0
                for adjustment in result.forecast.impact_adjustments
            )
        )

    def test_resolved_historical_event_is_not_extrapolated(self) -> None:
        task = example_task()
        document = Document(
            "doc_resolved_event",
            (
                "Alpha Station energy demand increased by 50 percent during a temporary event "
                "from 2024-01-01 00:00:00 to 2024-01-01 01:00:00. "
                "The event ended and demand returned to baseline."
            ),
        )
        retrieved = [RetrievedDocument(document=document, score=1.0, rank=1)]
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        impacts = EvidenceToForecastAgent().translate(task, diagnosis, retrieved, [])
        forecaster = ProbabilisticForecastAgent()
        baseline = forecaster.forecast(task, diagnosis, [], 100, 1, 0.75)
        adjusted = forecaster.forecast(task, diagnosis, retrieved, 100, 1, 0.75, impacts)

        self.assertEqual(impacts[0].adjustment_kind, "return_to_baseline")
        self.assertEqual(adjusted.mean, baseline.mean)
        self.assertEqual(adjusted.impact_adjustments[0].affected_steps, 0)

    def test_explicit_multiplier_is_translated_without_inventing_a_magnitude(self) -> None:
        task = future_impact_task()
        document = Document(
            "doc_multiplier",
            (
                "From 2024-01-03 00:00:00 to 2024-01-03 01:00:00, "
                "energy demand will be 5 times the usual level."
            ),
        )
        retrieved = [RetrievedDocument(document=document, score=1.0, rank=1)]
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        impacts = EvidenceToForecastAgent().translate(task, diagnosis, retrieved, [])

        self.assertEqual(impacts[0].adjustment_kind, "multiplier")
        self.assertEqual(impacts[0].adjustment_value, 5.0)

    def test_permanent_historical_shift_is_not_double_counted(self) -> None:
        task = future_impact_task()
        document = Document(
            "doc_historical_shift",
            (
                "A permanent policy change started on 2024-01-02 00:00:00 and "
                "increased Alpha Station energy demand by 50 percent."
            ),
        )
        retrieved = [RetrievedDocument(document=document, score=1.0, rank=1)]
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        impacts = EvidenceToForecastAgent().translate(task, diagnosis, retrieved, [])

        self.assertEqual(impacts[0].forecast_relation, "embedded_in_history")
        self.assertEqual(impacts[0].adjustment_kind, "already_in_baseline")

    def test_seasonality_inference_distinguishes_cycle_from_smooth_trend(self) -> None:
        smooth_trend = tuple(float(index * index) for index in range(120))
        repeated_cycle = tuple(float(index % 12) for index in range(120))

        self.assertIsNone(_infer_seasonal_period(smooth_trend)[0])
        self.assertEqual(_infer_seasonal_period(repeated_cycle)[0], 12)

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

    def test_iterative_loop_plans_verifies_updates_and_stops(self) -> None:
        result = IterativeAgentSystem(
            LoopConfig(
                max_steps=10,
                documents_per_step=1,
                num_samples=100,
                max_no_progress=4,
                seed=3,
            )
        ).run(iterative_task())

        self.assertGreaterEqual(len(result.loop_trace), 3)
        self.assertIsNotNone(result.belief_state)
        self.assertGreaterEqual(len(result.belief_state.answered_question_ids), 3)
        self.assertTrue(result.belief_state.stop_reason)
        self.assertIn("doc_event", result.belief_state.accepted_document_ids)
        self.assertNotIn("doc_wrong_entity", result.belief_state.accepted_document_ids)
        self.assertTrue(
            all(item.document.role is None and item.document.subtype is None for item in result.retrieved)
        )

    def test_iterative_outputs_include_auditable_trace(self) -> None:
        result = IterativeAgentSystem(
            LoopConfig(max_steps=4, documents_per_step=1, num_samples=100)
        ).run(iterative_task())
        with tempfile.TemporaryDirectory() as temporary_directory:
            write_outputs([result], temporary_directory)
            trace_path = Path(temporary_directory) / "loop_trace.jsonl"
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(trace["benchmark_id"], "task_loop")
            self.assertGreaterEqual(len(trace["steps"]), 1)
            self.assertIn("evidence_impacts", trace["steps"][0])
            self.assertIn("impact_adjustments", trace["steps"][0]["forecast"])
            self.assertNotIn("future_values", trace["belief_state"])
            self.assertNotIn("gt_evidence", trace["belief_state"])


if __name__ == "__main__":
    unittest.main()
