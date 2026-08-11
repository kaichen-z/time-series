from __future__ import annotations

import unittest

from drcik_agent.agents import TimeSeriesDiagnosisAgent
from drcik_agent.explicit_values import ExplicitValueValidator
from drcik_agent.models import Document, Evidence, ForecastTask, RetrievedDocument


def _task(documents: tuple[Document, ...]) -> ForecastTask:
    return ForecastTask(
        benchmark_id="explicit_test",
        entity_name="Solar Site",
        target_name="irradiance",
        target_description="Hourly irradiance",
        frequency="1 hour",
        prediction_length=2,
        seasonal_period=2,
        history_timestamps=(
            "2026-01-01 00:00:00",
            "2026-01-01 01:00:00",
            "2026-01-01 02:00:00",
            "2026-01-01 03:00:00",
        ),
        history_values=(0.0, 100.0, 0.0, 100.0),
        future_timestamps=("2026-01-02 00:00:00", "2026-01-02 01:00:00"),
        future_values=(0.0, 90.0),
        documents=documents,
    )


def _evidence(document_id: str) -> Evidence:
    return Evidence(
        document_id=document_id,
        claim="A forecast schedule is provided.",
        matched_terms=(),
        confidence=1.0,
        effect_direction="unknown",
        effect_window="forecast",
        evidence_quote="forecast schedule",
        provenance_valid=True,
    )


class ExplicitValueValidatorTest(unittest.TestCase):
    def test_requires_independent_table_corroboration(self) -> None:
        dated = Document(
            "dated",
            "forecast schedule\n| Timestamp | 2026-01-02 |\n| 00:00 | 0 |\n| 01:00 | 90 |",
        )
        cycle = Document(
            "cycle",
            "forecast schedule\n| Local Time | Day 3 Cycle |\n| 00:00 | 0 |\n| 01:00 | 90 |",
        )
        task = _task((dated, cycle))
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        result = ExplicitValueValidator().validate(
            task,
            diagnosis,
            (0.0, 100.0),
            [RetrievedDocument(dated, 1.0, 1), RetrievedDocument(cycle, 1.0, 2)],
            [_evidence("dated"), _evidence("cycle")],
        )
        self.assertEqual(result.accepted_points, {
            "2026-01-02 00:00:00": 0.0,
            "2026-01-02 01:00:00": 90.0,
        })
        self.assertTrue(all(item.accepted for item in result.decisions))

    def test_rejects_single_source_numeric_claim(self) -> None:
        dated = Document(
            "single",
            "forecast schedule\n| Timestamp | 2026-01-02 |\n| 00:00 | 50 |\n| 01:00 | 50 |",
        )
        task = _task((dated,))
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        result = ExplicitValueValidator().validate(
            task,
            diagnosis,
            (0.0, 100.0),
            [RetrievedDocument(dated, 1.0, 1)],
            [_evidence("single")],
        )
        self.assertEqual(result.accepted_points, {})
        self.assertTrue(
            all(
                item.reason == "insufficient_independent_corroboration"
                for item in result.decisions
            )
        )


if __name__ == "__main__":
    unittest.main()
