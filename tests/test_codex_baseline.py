from __future__ import annotations

import tempfile
import unittest

from drcik_agent.codex_baseline import (
    CodexContractSystem,
    CodexDirectBaseline,
    CodexDirectConfig,
)
from drcik_agent.models import Document, ForecastTask


class FakeCodexClient:
    def __init__(self) -> None:
        self.calls = 0

    def stats(self):
        return {
            "calls": self.calls,
            "cache_hits": 0,
            "failures": 0,
            "latency_seconds": float(self.calls),
            "last_error": None,
        }

    def complete(self, stage, prompt, schema, workspace_files=None):
        self.calls += 1
        assert stage == "codex_direct"
        assert workspace_files and "task.json" in workspace_files
        assert any(name.startswith("documents/") for name in workspace_files)
        return {
            "research_report": "A planned intervention changes the forecast.",
            "cited_document_ids": ["doc_support"],
            "evidence": [
                {
                    "claim": "The intervention increases the target.",
                    "document_ids": ["doc_support"],
                    "exact_quotes": ["The intervention increases the target."],
                }
            ],
            "forecast_values": [20.0, 21.0],
        }


class FakeContractCodexClient(FakeCodexClient):
    def complete(self, stage, prompt, schema, workspace_files=None):
        self.calls += 1
        assert stage == "codex_contract"
        assert "Do not produce future numerical values" in prompt
        return {
            "research_report": "The temporary anomaly ended; normal seasonality resumes.",
            "cited_document_ids": ["doc_support"],
            "evidence": [
                {
                    "claim": "The series returns to historical baseline and seasonality.",
                    "document_ids": ["doc_support"],
                    "exact_quotes": ["Return to historical baseline and seasonality."],
                }
            ],
            "forecast_contract": {
                "regime": "normal_seasonal",
                "expected_behavior": "Resume the normal seasonal trajectory.",
                "seasonality": "preserve",
                "anomalous_history_windows": [],
                "future_event_windows": [],
                "confidence": 0.9,
                "rationale": "The supporting document explicitly says so.",
            },
        }

class CodexDirectBaselineTest(unittest.TestCase):
    def test_direct_baseline_uses_codex_forecast_without_hybrid_agents(self) -> None:
        task = ForecastTask(
            benchmark_id="task_test",
            entity_name="Example Store",
            target_name="sales",
            target_description="Daily sales",
            frequency="1 day",
            prediction_length=2,
            seasonal_period=None,
            history_timestamps=("2026-01-01", "2026-01-02", "2026-01-03"),
            history_values=(10.0, 11.0, 12.0),
            future_timestamps=("2026-01-04", "2026-01-05"),
            future_values=(20.0, 21.0),
            documents=(
                Document(
                    "doc_support",
                    "The intervention increases the target.",
                    role="supporting",
                ),
                Document("doc_noise", "Office paint is blue.", role="distractor"),
            ),
            gt_evidence=("The intervention increases the target.",),
        )
        with tempfile.TemporaryDirectory() as cache:
            system = CodexDirectBaseline(
                CodexDirectConfig(
                    backbone="statistical",
                    num_samples=10,
                    codex_cache_dir=cache,
                )
            )
            system.codex = FakeCodexClient()
            result = system.run(task)

        self.assertEqual(result.forecast.mean, (20.0, 21.0))
        self.assertEqual([item.document.document_id for item in result.retrieved], ["doc_support"])
        self.assertEqual(result.metrics["codex_calls"], 1.0)
        self.assertEqual(result.metrics["codex_valid_forecast"], 1.0)
        self.assertEqual(result.metrics["mae"], 0.0)
        self.assertIsNone(result.workspace)

    def test_contract_uses_history_backtest_instead_of_codex_numbers(self) -> None:
        history = (10.0, 20.0, 10.0, 0.0, 12.0, 22.0, 12.0, 2.0, 14.0, 24.0, 14.0, 4.0)
        task = ForecastTask(
            benchmark_id="task_contract",
            entity_name="Example Store",
            target_name="sales",
            target_description="Daily sales",
            frequency="1 day",
            prediction_length=4,
            seasonal_period=4,
            history_timestamps=tuple(f"2026-01-{index:02d}" for index in range(1, 13)),
            history_values=history,
            future_timestamps=tuple(f"2026-01-{index:02d}" for index in range(13, 17)),
            future_values=(16.0, 26.0, 16.0, 6.0),
            documents=(
                Document(
                    "doc_support",
                    "Return to historical baseline and seasonality.",
                    role="supporting",
                ),
            ),
            gt_evidence=("Return to historical baseline and seasonality.",),
        )
        with tempfile.TemporaryDirectory() as cache:
            system = CodexContractSystem(
                CodexDirectConfig(
                    backbone="statistical",
                    num_samples=10,
                    codex_cache_dir=cache,
                )
            )
            system.codex = FakeContractCodexClient()
            result = system.run(task)

        self.assertEqual(result.metrics["codex_calls"], 1.0)
        self.assertEqual(result.metrics["contract_revision_applied"], 1.0)
        self.assertLess(result.metrics["candidate_validation_mae"], result.metrics["candidate_seasonal_naive_mae"])
        self.assertNotEqual(result.forecast.mean, result.forecast.baseline_mean)
        self.assertIsNotNone(result.workspace)
        self.assertTrue(any(record.accepted for record in result.forecast.revision_records))


if __name__ == "__main__":
    unittest.main()
