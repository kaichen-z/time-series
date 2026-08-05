from __future__ import annotations

import unittest

from drcik_agent.agents import TimeSeriesDiagnosisAgent
from drcik_agent.context import ForecastUtilityRetriever
from drcik_agent.control import ForecastGapControllerAgent
from drcik_agent.forecast_utility import ForecastUtilityLabeler
from drcik_agent.loop import BeliefUpdaterAgent, IterativeAgentSystem, LoopConfig
from drcik_agent.models import (
    AgentBeliefState,
    Document,
    Evidence,
    EvidenceVerdict,
    ForecastGap,
    LinguisticBelief,
    QueryAction,
    RetrievedDocument,
)

from test_minimal_system import example_task, future_impact_task


class _PreferSecondScorer:
    kind = "test_learned_forecast_utility"

    def score(self, task, action, document, features):
        return 1.0 if document.document_id == "second" else 0.0


class GapControlTest(unittest.TestCase):
    def test_controller_outputs_structured_gap_and_next_query(self) -> None:
        task = example_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        controller = ForecastGapControllerAgent()
        gaps = controller.initial_gaps(task, diagnosis)
        state = AgentBeliefState(
            open_question_ids=list(gaps),
            forecast_gaps=gaps,
            linguistic_beliefs={key: LinguisticBelief(key, 0.5) for key in gaps},
        )

        decision, action = controller.decide(task, state)

        self.assertFalse(decision.sufficient)
        self.assertIn(decision.selected_gap_id, gaps)
        self.assertIsNotNone(action)
        self.assertEqual(action.question_id, decision.selected_gap_id)
        self.assertIn(task.entity_name, decision.next_query)
        self.assertGreater(decision.expected_information_gain, 0.0)

    def test_exhausted_gaps_are_not_reported_as_sufficient(self) -> None:
        task = example_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        controller = ForecastGapControllerAgent()
        gaps = controller.initial_gaps(task, diagnosis)
        state = AgentBeliefState(
            open_question_ids=[],
            answered_question_ids=["historical_regime"],
            exhausted_question_ids=["external_drivers", "forecast_regime"],
            forecast_gaps=gaps,
        )

        decision, action = controller.decide(task, state)

        self.assertFalse(decision.sufficient)
        self.assertEqual(decision.stop_reason, "unresolved_gaps_exhausted")
        self.assertEqual(
            decision.unresolved_gap_ids,
            ("external_drivers", "forecast_regime"),
        )
        self.assertIsNone(action)

    def test_magnitude_gap_requires_quantified_evidence(self) -> None:
        state = AgentBeliefState(
            open_question_ids=["event_magnitude"],
            forecast_gaps={
                "event_magnitude": ForecastGap(
                    gap_id="event_magnitude",
                    category="effect_magnitude",
                    target="sales",
                    time_scope="future",
                    description="quantify the effect",
                    query_terms=("percent",),
                    priority=1.0,
                )
            },
        )
        action = QueryAction("event_magnitude", "quantify", "sales percent", "test")
        qualitative = EvidenceVerdict(
            document_id="d1",
            accepted=True,
            score=0.8,
            entity_match=True,
            temporal_alignment="aligned",
            series_consistency="consistent",
            question_alignment=True,
            reasons=("accepted",),
            event_types=("external_driver",),
            evidence=(
                Evidence(
                    document_id="d1",
                    claim="A campaign will increase sales.",
                    matched_terms=("increase",),
                    confidence=0.8,
                    effect_direction="up",
                    effect_window="future",
                    magnitude=None,
                ),
            ),
        )

        BeliefUpdaterAgent().update(state, action, [qualitative])

        self.assertIn("event_magnitude", state.open_question_ids)
        self.assertNotIn("event_magnitude", state.answered_question_ids)

    def test_anomaly_evidence_creates_resolution_followup_gap(self) -> None:
        task = example_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        controller = ForecastGapControllerAgent()
        gaps = controller.initial_gaps(task, diagnosis)
        state = AgentBeliefState(
            open_question_ids=list(gaps),
            forecast_gaps=gaps,
            linguistic_beliefs={key: LinguisticBelief(key, 0.5) for key in gaps},
        )
        action = QueryAction("historical_regime", "history", "query", "test")
        evidence = Evidence(
            "doc_anomaly",
            "A software bug caused the anomaly.",
            ("bug", "anomaly"),
            0.9,
            "up",
            "historical",
        )
        verdict = EvidenceVerdict(
            document_id="doc_anomaly",
            accepted=True,
            score=0.9,
            entity_match=True,
            temporal_alignment="task_window",
            series_consistency="not_contradicted",
            question_alignment=True,
            reasons=("accepted_as_forecast_relevant",),
            event_types=("anomaly",),
            evidence=(evidence,),
        )

        controller.expand_from_verdicts(task, action, [verdict], state)

        self.assertIn("resolution_permanence", state.forecast_gaps)
        self.assertIn("resolution_permanence", state.open_question_ids)
        self.assertEqual(
            state.forecast_gaps["resolution_permanence"].created_from,
            ("doc_anomaly",),
        )

    def test_learned_scorer_adapter_can_replace_the_proxy(self) -> None:
        task = example_task()
        action = QueryAction(
            "external_drivers", "future drivers", "Alpha Station energy demand", "test"
        )
        candidates = [
            RetrievedDocument(Document("first", "Alpha Station energy demand event."), 2.0, 1),
            RetrievedDocument(Document("second", "Alpha Station energy demand event."), 1.0, 2),
        ]
        ranked, assessments = ForecastUtilityRetriever(_PreferSecondScorer()).rank(
            task,
            action,
            candidates,
            AgentBeliefState(open_question_ids=["external_drivers"]),
            top_k=1,
        )

        self.assertEqual(ranked[0].document.document_id, "second")
        self.assertTrue(
            all(item.scorer_kind == "test_learned_forecast_utility" for item in assessments)
        )

    def test_offline_label_is_forecast_gain_minus_costs(self) -> None:
        label = ForecastUtilityLabeler().label(
            task_id="task",
            document_id="doc",
            forecast_before=(0.0, 0.0),
            forecast_after=(1.0, 1.0),
            actual=(1.0, 1.0),
            latency_cost=0.1,
            redundancy_cost=0.05,
            token_cost=0.05,
        )

        self.assertAlmostEqual(label.forecast_gain, 1.0)
        self.assertAlmostEqual(label.net_utility, 0.8)
        self.assertTrue(label.beneficial)

    def test_loop_trace_contains_gap_judgments_and_grounded_evidence(self) -> None:
        result = IterativeAgentSystem(
            LoopConfig(max_steps=5, documents_per_step=1, backbone="statistical")
        ).run(future_impact_task())

        self.assertIn("sufficiency_before", result.loop_trace[0])
        self.assertIn("sufficiency_after", result.loop_trace[0])
        self.assertIn("forecast_gaps", result.loop_trace[0])
        self.assertTrue(result.evidence)
        self.assertEqual(result.evidence[0].entity, "Alpha Station")
        self.assertEqual(result.evidence[0].target_variable, "energy demand")
        self.assertTrue(result.evidence[0].evidence_quote)
        self.assertIn("gap_coverage", result.metrics)


if __name__ == "__main__":
    unittest.main()
