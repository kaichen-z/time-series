from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from evolving_loop.coding_agent.evolution import CodingEvolutionAgent, CodingEvolutionConfig
from evolving_loop.coding_agent.skill_library import SkillLibrary
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.decision_agent.skill_library import DecisionSkillLibrary
from evolving_loop.harness import EvolvingForecastHarness
from evolving_loop.retrieval_agent.agent import RetrievalAgent
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.skill_learning import OutcomeSkillLearner
from common.llm import FakeLLMClient

from test_evolving_harness import TREND_CODE, _program, _retrieval_response, _task


def _decision_response(*, use_skills: bool = False) -> str:
    return json.dumps(
        {
            "selected_candidate_id": "trend__evidence_0",
            "supporting_document_ids": ["doc_event"],
            "rationale": "Use the verified two-day absolute effect.",
            "request_more_retrieval": False,
            "used_skill_names": ["prefer_verified_window"] if use_skills else [],
        }
    )


def _learned_skills_response() -> str:
    return json.dumps(
        {
            "retrieval_skill": {
                "name": "retrieve_explicit_event_window",
                "description": "Find target-specific future events with explicit magnitudes and windows.",
                "applicability": "A numeric hypothesis may fail because an external event overlaps the horizon.",
                "query_strategy": "Search jointly for target, event type, magnitude, start, and end conditions.",
                "verification_rule": "Require one exact quote to establish target, magnitude, and complete window.",
            },
            "decision_skill": {
                "name": "prefer_verified_window",
                "description": "Use an evidence-adjusted candidate only for a completely verified event window.",
                "applicability": "A candidate is derived from a cited quantitative future event.",
                "decision_rule": "Require matching citations and restrict adjustment to the verified window.",
                "failure_condition": "Magnitude, target, or either temporal boundary is missing.",
            },
        }
    )


def _harness(directory: str):
    coding_library = SkillLibrary.load(Path(directory) / "skills.json")
    retrieval_library = RetrievalSkillLibrary.load(Path(directory) / "retrieval_skills.json")
    decision_library = DecisionSkillLibrary.load(Path(directory) / "decision_skills.json")
    coding = CodingEvolutionAgent(
        FakeLLMClient([_program("trend", TREND_CODE)]),
        coding_library,
        CodingEvolutionConfig(
            setting="statistics",
            initial_programs=1,
            mutations=0,
            validation_folds=2,
            validation_horizon=2,
            minimum_validation_history=4,
        ),
    )
    harness = EvolvingForecastHarness(
        coding,
        RetrievalAgent(FakeLLMClient([_retrieval_response()]), retrieval_library),
        DecisionAgent(FakeLLMClient([_decision_response()]), decision_library),
        OutcomeSkillLearner(
            FakeLLMClient([_learned_skills_response()]),
            retrieval_library,
            decision_library,
        ),
    )
    return harness, retrieval_library, decision_library


def test_public_outcome_generates_both_persistent_skill_libraries() -> None:
    with tempfile.TemporaryDirectory() as directory:
        retrieval_path = Path(directory) / "retrieval_skills.json"
        decision_path = Path(directory) / "decision_skills.json"
        harness, retrieval_library, decision_library = _harness(directory)
        result = harness.run(_task())

        assert not retrieval_path.exists()
        assert not decision_path.exists()

        outcome, learning = harness.record_outcome(_task(), result)

        assert outcome.final_smape == 0.0
        assert learning.retrieval_skill_name == "retrieve_explicit_event_window"
        assert learning.decision_skill_name == "prefer_verified_window"
        assert retrieval_path.exists()
        assert decision_path.exists()
        assert len(retrieval_library) == 1
        assert retrieval_library.get("retrieve_explicit_event_window").status == "candidate"
        assert len(decision_library) == 1


def test_generated_skill_cannot_memorize_task_identifiers() -> None:
    bad = json.loads(_learned_skills_response())
    bad["retrieval_skill"]["query_strategy"] = "Always search for doc_event."
    bad["decision_skill"]["decision_rule"] = "Use this only for linear_event."
    with tempfile.TemporaryDirectory() as directory:
        retrieval_library = RetrievalSkillLibrary.load(Path(directory) / "retrieval.json")
        decision_library = DecisionSkillLibrary.load(Path(directory) / "decision.json")
        harness = EvolvingForecastHarness(
            CodingEvolutionAgent(
                FakeLLMClient([_program("trend", TREND_CODE)]),
                SkillLibrary.load(Path(directory) / "coding.json"),
                CodingEvolutionConfig(
                    setting="statistics",
                    initial_programs=1,
                    mutations=0,
                    validation_folds=2,
                    validation_horizon=2,
                    minimum_validation_history=4,
                ),
            ),
            RetrievalAgent(FakeLLMClient([_retrieval_response()]), retrieval_library),
            DecisionAgent(FakeLLMClient([_decision_response()]), decision_library),
            OutcomeSkillLearner(
                FakeLLMClient([json.dumps(bad)]), retrieval_library, decision_library
            ),
        )
        task = _task()
        result = harness.run(task)
        _outcome, learning = harness.record_outcome(task, result)

        assert len(retrieval_library) == 0
        assert len(decision_library) == 0
        assert "retrieval:task_specific_identifier" in learning.rejection_reasons
        assert "decision:task_specific_identifier" in learning.rejection_reasons


def test_decision_skill_gate_uses_mae_regret() -> None:
    with tempfile.TemporaryDirectory() as directory:
        harness, retrieval_library, decision_library = _harness(directory)
        task = _task()
        result = harness.run(task)
        outcome = harness.score_after_resolution(task, result)
        result = replace(
            result,
            retrieval=replace(
                result.retrieval,
                selected_document_ids=(),
                evidence=(),
            ),
        )
        outcome = replace(
            outcome,
            retrieval_precision=0.0,
            supporting_recall=0.0,
            distractor_avoidance=0.0,
            decision_selection_regret=0.0,
            decision_selection_mae_regret=1.0,
        )
        learner = OutcomeSkillLearner(
            FakeLLMClient([]), retrieval_library, decision_library
        )

        learning = learner.learn(task, result, outcome)

        assert learning.decision_eligible is False
        assert len(decision_library) == 0


def test_candidate_retrieval_skills_are_not_projected_into_prompts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        harness, retrieval_library, decision_library = _harness(directory)
        task = _task()
        result = harness.run(task)
        harness.record_outcome(task, result)

        retrieval_llm = FakeLLMClient(
            [
                json.dumps(
                    {
                        **json.loads(_retrieval_response()),
                        "used_skill_names": ["retrieve_explicit_event_window"],
                    }
                )
            ]
        )
        decision_llm = FakeLLMClient([_decision_response(use_skills=True)])
        second = EvolvingForecastHarness(
            CodingEvolutionAgent(
                FakeLLMClient([_program("trend", TREND_CODE)]),
                SkillLibrary.load(Path(directory) / "skills.json"),
                CodingEvolutionConfig(
                    setting="statistics",
                    initial_programs=1,
                    mutations=0,
                    validation_folds=2,
                    validation_horizon=2,
                    minimum_validation_history=4,
                ),
            ),
            RetrievalAgent(retrieval_llm, retrieval_library),
            DecisionAgent(decision_llm, decision_library),
        ).run(task)

        assert second.retrieval.used_skill_names == ("retrieve_explicit_event_window",)
        assert second.decision.used_skill_names == ("prefer_verified_window",)
        assert "retrieve_explicit_event_window" not in retrieval_llm.calls[0]["messages"][0]["content"]
        assert "prefer_verified_window" in decision_llm.calls[0]["messages"][0]["content"]
