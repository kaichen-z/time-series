from __future__ import annotations

import unittest
from types import SimpleNamespace

from drcik_agent.agents import TimeSeriesDiagnosisAgent
from drcik_agent.codex_agents import (
    CodexEvidenceToForecastAgent,
    CodexQueryPlannerAgent,
)
from drcik_agent.control import ForecastGapControllerAgent
from drcik_agent.context import ImportanceAwareContextAgent
from drcik_agent.impacts import EvidenceToForecastAgent
from drcik_agent.loop import CodexEvidenceVerifierAgent
from drcik_agent.models import AgentBeliefState, LinguisticBelief, RetrievedDocument

from test_minimal_system import future_impact_task


class FakeCodexClient:
    def __init__(self, responses):
        self.responses = responses
        self.config = SimpleNamespace(max_document_characters=12000)

    def complete(self, stage, _prompt, _schema):
        return self.responses.get(stage)


class CodexAgentTest(unittest.TestCase):
    def _state_and_action(self):
        task = future_impact_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        controller = ForecastGapControllerAgent()
        gaps = controller.initial_gaps(task, diagnosis)
        state = AgentBeliefState(
            open_question_ids=list(gaps),
            linguistic_beliefs={key: LinguisticBelief(key, 0.5) for key in gaps},
            forecast_gaps=gaps,
        )
        _decision, action = controller.decide(task, state)
        return task, diagnosis, state, action

    def test_codex_query_planner_refines_but_preserves_gap_identity(self) -> None:
        task, diagnosis, state, action = self._state_and_action()
        client = FakeCodexClient(
            {
                "query": {
                    "query": '"Alpha Station" "energy demand" promotion January 2024',
                    "rationale": "Use entity, target, event, and forecast dates.",
                }
            }
        )
        refined = CodexQueryPlannerAgent(client).refine(
            task, diagnosis, state, action
        )

        self.assertEqual(refined.question_id, action.question_id)
        self.assertNotEqual(refined.query, action.query)
        self.assertIn("Alpha Station", refined.query)

    def test_codex_verifier_requires_an_exact_grounded_quote(self) -> None:
        task, diagnosis, _state, action = self._state_and_action()
        document = task.documents[0]
        quote = "The promotion will increase energy demand by 50 percent throughout the event."
        client = FakeCodexClient(
            {
                "verify": {
                    "decisions": [
                        {
                            "document_id": document.document_id,
                            "accepted": True,
                            "confidence": 0.95,
                            "reason": "Exact entity, target, date, and quantified effect.",
                            "event_types": ["temporary_event", "external_driver"],
                            "evidence_quotes": [quote],
                        }
                    ]
                }
            }
        )
        verdict = CodexEvidenceVerifierAgent(client).verify(
            task,
            diagnosis,
            action,
            [RetrievedDocument(document, 1.0, 1)],
        )[0]

        self.assertTrue(verdict.accepted)
        self.assertEqual(verdict.evidence[0].claim, quote)
        self.assertEqual(verdict.evidence[0].magnitude, "50 percent")
        self.assertTrue(verdict.evidence[0].provenance_valid)

    def test_codex_impact_translation_is_structured_and_source_bounded(self) -> None:
        task, diagnosis, _state, action = self._state_and_action()
        document = task.documents[0]
        retrieved = [RetrievedDocument(document, 1.0, 1)]
        verifier_client = FakeCodexClient(
            {
                "verify": {
                    "decisions": [
                        {
                            "document_id": document.document_id,
                            "accepted": True,
                            "confidence": 0.95,
                            "reason": "Grounded future event.",
                            "event_types": ["temporary_event"],
                            "evidence_quotes": [
                                "The promotion will increase energy demand by 50 percent throughout the event."
                            ],
                        }
                    ]
                }
            }
        )
        evidence = list(
            CodexEvidenceVerifierAgent(verifier_client)
            .verify(task, diagnosis, action, retrieved)[0]
            .evidence
        )
        impact_client = FakeCodexClient(
            {
                "impact": {
                    "impacts": [
                        {
                            "source_document_ids": [document.document_id],
                            "event_type": "promotion",
                            "start_timestamp": "2024-01-03 00:00:00",
                            "end_timestamp": "2024-01-03 01:00:00",
                            "direction": "up",
                            "permanence": "temporary",
                            "forecast_relation": "overlaps_forecast",
                            "adjustment_kind": "percentage",
                            "adjustment_value": 0.5,
                            "confidence": 0.95,
                            "rationale": "The evidence explicitly states a 50 percent increase.",
                        }
                    ]
                }
            }
        )
        impacts = CodexEvidenceToForecastAgent(
            impact_client, EvidenceToForecastAgent()
        ).translate(task, diagnosis, retrieved, evidence)

        self.assertEqual(len(impacts), 1)
        self.assertEqual(impacts[0].adjustment_kind, "percentage")
        self.assertEqual(impacts[0].adjustment_value, 0.5)
        self.assertEqual(impacts[0].source_document_ids, (document.document_id,))

    def test_codex_failure_falls_back_to_deterministic_impact_agent(self) -> None:
        task, diagnosis, _state, _action = self._state_and_action()
        retrieved = [RetrievedDocument(task.documents[0], 1.0, 1)]
        impacts = CodexEvidenceToForecastAgent(
            FakeCodexClient({"impact": None}), EvidenceToForecastAgent()
        ).translate(task, diagnosis, retrieved, [])

        self.assertEqual(impacts[0].adjustment_kind, "percentage")
        self.assertEqual(impacts[0].adjustment_value, 0.5)

    def test_codex_exact_quote_is_pinned_through_context_compression(self) -> None:
        task, diagnosis, _state, _action = self._state_and_action()
        quote = "2024-01-03 00:00:00 | 123.45"
        document = type(task.documents[0])(
            "doc_table",
            ("Generic operational background without forecast value.\n" * 80) + quote,
        )
        compressed, records = ImportanceAwareContextAgent(
            total_character_budget=500,
            minimum_document_budget=100,
        ).compress(
            task,
            diagnosis,
            [RetrievedDocument(document, 1.0, 1)],
            pinned_quotes={document.document_id: (quote,)},
        )

        self.assertIn(quote, compressed[0].document.text)
        self.assertLessEqual(records[0].retained_characters, records[0].original_characters)


if __name__ == "__main__":
    unittest.main()
