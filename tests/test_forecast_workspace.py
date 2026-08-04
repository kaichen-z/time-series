from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from drcik_agent.agents import ProbabilisticForecastAgent, TimeSeriesDiagnosisAgent
from drcik_agent.loop import IterativeAgentSystem, LoopConfig
from drcik_agent.memory import ForecastMemoryBank
from drcik_agent.models import RevisionAction
from drcik_agent.workspace import ForecastWorkspaceExecutor

from test_minimal_system import example_task, future_impact_task


class ForecastWorkspaceTest(unittest.TestCase):
    def _workspace(self):
        task = example_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        baseline, method = ProbabilisticForecastAgent().baseline(task, diagnosis)
        workspace = ForecastWorkspaceExecutor().initialize(task, baseline, method)
        return task, workspace

    def test_baseline_remains_immutable_when_final_is_revised(self) -> None:
        _task, workspace = self._workspace()
        original = workspace.baseline_values
        action = RevisionAction(
            action_id="multiply-test",
            action_type="multiply",
            start_index=0,
            end_index=1,
            value=1.5,
            source_document_ids=("doc_support",),
            event_type="promotion",
            evidence="A verified promotion increases demand by 50 percent.",
            confidence=0.9,
        )
        record = ForecastWorkspaceExecutor().apply(workspace, action)

        self.assertTrue(record.accepted)
        self.assertEqual(workspace.baseline_values, original)
        self.assertEqual(tuple(workspace.final_values), tuple(value * 1.5 for value in original))

    def test_duplicate_revision_is_rejected_instead_of_compounded(self) -> None:
        _task, workspace = self._workspace()
        executor = ForecastWorkspaceExecutor()
        action = RevisionAction(
            action_id="same-event",
            action_type="add",
            start_index=0,
            end_index=0,
            value=2.0,
            source_document_ids=("doc_support",),
            event_type="external_driver",
            confidence=0.8,
        )
        first = executor.apply(workspace, action)
        after_first = tuple(workspace.final_values)
        second = executor.apply(workspace, action)

        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertEqual(second.reason, "duplicate_revision")
        self.assertEqual(tuple(workspace.final_values), after_first)

    def test_numerical_revision_without_evidence_is_rejected(self) -> None:
        _task, workspace = self._workspace()
        action = RevisionAction(
            action_id="unsupported-edit",
            action_type="override",
            start_index=0,
            end_index=0,
            value=999.0,
            confidence=1.0,
        )
        record = ForecastWorkspaceExecutor().apply(workspace, action)

        self.assertFalse(record.accepted)
        self.assertEqual(record.reason, "numerical_revision_requires_evidence")
        self.assertEqual(tuple(workspace.final_values), workspace.baseline_values)

    def test_preserve_action_records_reason_without_changing_values(self) -> None:
        _task, workspace = self._workspace()
        action = RevisionAction(
            action_id="resolved-event",
            action_type="preserve",
            start_index=0,
            end_index=1,
            value=0.0,
            source_document_ids=("doc_resolution",),
            event_type="resolution",
            evidence="The disruption ended before the forecast horizon.",
            confidence=0.9,
        )
        record = ForecastWorkspaceExecutor().apply(workspace, action)

        self.assertTrue(record.accepted)
        self.assertEqual(record.reason, "preserved_baseline")
        self.assertEqual(record.affected_steps, 0)
        self.assertEqual(tuple(workspace.final_values), workspace.baseline_values)

    def test_iterative_result_exposes_baseline_final_and_actions(self) -> None:
        result = IterativeAgentSystem(
            LoopConfig(max_steps=5, documents_per_step=1, seed=1)
        ).run(future_impact_task())

        self.assertIsNotNone(result.workspace)
        self.assertEqual(result.forecast.baseline_mean, result.workspace.baseline_values)
        self.assertEqual(result.forecast.mean, tuple(result.workspace.final_values))
        self.assertTrue(result.workspace.public_dict()["baseline_immutable"])
        self.assertIn("baseline_mae", result.metrics)
        self.assertIn("revision_value_mae", result.metrics)
        self.assertTrue(
            any(
                record.accepted and record.action.action_type == "multiply"
                for record in result.workspace.revision_records
            )
        )

    def test_posthoc_memory_is_written_only_after_outcomes_are_recorded(self) -> None:
        task = future_impact_task()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            system = IterativeAgentSystem(
                LoopConfig(max_steps=5, documents_per_step=1, seed=1, memory_path=str(path))
            )
            result = system.run(task)
            self.assertFalse(path.exists())

            entries = system.record_outcome(task, result)
            self.assertTrue(entries)
            self.assertTrue(path.exists())
            loaded = ForecastMemoryBank(path)
            self.assertTrue(loaded.query("promotion", "multiply"))


if __name__ == "__main__":
    unittest.main()
