from __future__ import annotations

import unittest

from drcik_agent.agents import TimeSeriesDiagnosisAgent
from drcik_agent.context import ImportanceAwareContextAgent, RetrievalProcessRewardAgent
from drcik_agent.loop import IterativeAgentSystem, LoopConfig
from drcik_agent.models import (
    AgentBeliefState,
    Document,
    LinguisticBelief,
    MacroOutlook,
    MicroOutlook,
    QueryAction,
    RetrievedDocument,
    RevisionAction,
)
from drcik_agent.reasoning import RevisionUtilityAgent

from test_minimal_system import example_task, future_impact_task


class PaperInspiredAgentTest(unittest.TestCase):
    def test_forecast_utility_ranker_prefers_entity_specific_causal_context(self) -> None:
        task = example_task()
        action = QueryAction(
            question_id="external_drivers",
            question="Which event changes demand?",
            query="Alpha Station energy demand event increase future",
            rationale="test",
        )
        candidates = [
            RetrievedDocument(
                Document(
                    "relevant",
                    "Alpha Station energy demand will increase after a scheduled policy event in 2024.",
                ),
                score=1.0,
                rank=1,
            ),
            RetrievedDocument(
                Document(
                    "distractor",
                    "Beta Harbor future event increase energy demand forecast policy event increase.",
                ),
                score=1.0,
                rank=2,
            ),
        ]
        ranked, assessments = RetrievalProcessRewardAgent().rank(
            task,
            action,
            candidates,
            AgentBeliefState(open_question_ids=["external_drivers"]),
            top_k=1,
        )

        self.assertEqual(ranked[0].document.document_id, "relevant")
        utilities = {item.document_id: item.utility_score for item in assessments}
        self.assertGreater(utilities["relevant"], utilities["distractor"])

    def test_importance_aware_context_retains_causal_dates_and_magnitude(self) -> None:
        task = example_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        noise = " ".join(
            f"Administrative paragraph {index} discusses office furniture and routine supplies."
            for index in range(30)
        )
        important = (
            "Alpha Station energy demand will increase by 50 percent from "
            "2024-01-03 00:00:00 to 2024-01-03 01:00:00 because of a scheduled event."
        )
        document = RetrievedDocument(
            Document("long_doc", noise + " " + important), score=1.0, rank=1
        )
        compressed, records = ImportanceAwareContextAgent(
            total_character_budget=350,
            minimum_document_budget=100,
        ).compress(task, diagnosis, [document])

        self.assertIn("50 percent", compressed[0].document.text)
        self.assertIn("2024-01-03", compressed[0].document.text)
        self.assertLess(records[0].retained_characters, records[0].original_characters)
        self.assertLessEqual(records[0].retained_characters, 350)

    def test_posttime_gate_falls_back_for_an_invented_weak_magnitude(self) -> None:
        state = AgentBeliefState(
            open_question_ids=[],
            linguistic_beliefs={
                "external_drivers": LinguisticBelief("external_drivers", 0.65)
            },
        )
        macro = MacroOutlook(
            direction="stable",
            slope_per_step=0.0,
            seasonal_period=None,
            seasonal_strength=0.0,
            baseline_method="test",
            confidence=0.6,
            summary="stable",
        )
        action = RevisionAction(
            action_id="weak",
            action_type="add",
            start_index=0,
            end_index=1,
            value=1.0,
            source_document_ids=("doc_weak",),
            event_type="external_driver",
            confidence=0.55,
            rationale="No magnitude is stated; apply a conservative inferred adjustment.",
        )
        agent = RevisionUtilityAgent(threshold=0.60)
        decision = agent.evaluate(action, macro, MicroOutlook((), 0.0, "none"), state)
        fallback = agent.fallback(action, decision)

        self.assertFalse(decision.revise)
        self.assertEqual(fallback.action_type, "preserve")

    def test_loop_exposes_blf_beliefs_nexus_outlooks_and_retrieval_scores(self) -> None:
        result = IterativeAgentSystem(
            LoopConfig(max_steps=6, documents_per_step=1, seed=1)
        ).run(future_impact_task())

        self.assertIsNotNone(result.workspace.macro_outlook)
        self.assertIsNotNone(result.workspace.micro_outlook)
        self.assertTrue(result.workspace.revision_decisions)
        belief = result.belief_state.linguistic_beliefs["external_drivers"]
        self.assertGreater(belief.evidence_sufficiency, 0.5)
        self.assertIn("retrieval_candidate_assessments", result.loop_trace[0])
        self.assertIn("context_compression", result.loop_trace[0])
        self.assertIn("revision_accept_rate", result.metrics)


if __name__ == "__main__":
    unittest.main()
