from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evolving_loop.coding_agent.evolution import CodingEvolutionAgent, CodingEvolutionConfig
from evolving_loop.coding_agent.skill_library import SkillLibrary
from evolving_loop.co_evolution import HarnessPolicy, evaluate_policy
from evolving_loop.data import ContextTask, Document, Task
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.harness import EvolvingForecastHarness, HarnessRuntimeConfig
from evolving_loop.retrieval_agent.agent import RetrievalAgent
from common.llm import FakeLLMClient


LEVEL_CODE = (
    "def forecast(history, horizon, frequency):\n"
    "    return [history[-1] for _ in range(horizon)]\n"
)
TREND_CODE = (
    "def forecast(history, horizon, frequency):\n"
    "    slope = history[-1] - history[-2]\n"
    "    return [history[-1] + slope * (step + 1) for step in range(horizon)]\n"
)


def _program(name: str, code: str) -> str:
    return json.dumps(
        {
            "programs": [
                {
                    "name": name,
                    "description": f"{name} method",
                    "assumption": "The local numeric pattern persists.",
                    "failure_condition": "A new regime begins after the cutoff.",
                    "code": code,
                }
            ]
        }
    )


def _task() -> ContextTask:
    history = tuple(float(value) for value in range(1, 21))
    numeric = Task(
        task_id="linear_event",
        history_values=history,
        future_values=(26.0, 27.0),
        prediction_length=2,
        frequency="1 day",
        seasonal_period=None,
        entity_name="Alpha Store",
    )
    return ContextTask(
        numeric=numeric,
        target_name="sales",
        target_description="Daily sales",
        history_timestamps=tuple(f"2026-01-{index:02d}" for index in range(1, 21)),
        future_timestamps=("2026-01-21", "2026-01-22"),
        documents=(
            Document(
                "doc_event",
                "Alpha Store will receive an absolute sales boost of 5 units from "
                "2026-01-21 through 2026-01-22.",
                role="supporting",
            ),
            Document("doc_noise", "The office carpet is blue.", role="distractor"),
        ),
        gt_evidence=("A two-day event adds five sales units.",),
    )


def _retrieval_response() -> str:
    return json.dumps(
        {
            "query": "Alpha Store future sales boost",
            "selected_document_ids": ["doc_event"],
            "evidence": [
                {
                    "document_id": "doc_event",
                    "claim": "A future event adds five units.",
                    "exact_quote": (
                        "Alpha Store will receive an absolute sales boost of 5 units from "
                        "2026-01-21 through 2026-01-22."
                    ),
                }
            ],
            "impacts": [
                {
                    "source_document_ids": ["doc_event"],
                    "mechanism_layer": "future_driver",
                    "temporal_relation": "overlaps_future",
                    "direction": "up",
                    "permanence": "temporary",
                    "adjustment_kind": "add",
                    "adjustment_value": 5,
                    "start_timestamp": "2026-01-21",
                    "end_timestamp": "2026-01-22",
                    "rationale": "The quote gives the target, magnitude, and complete window.",
                }
            ],
            "sufficient": True,
            "missing_information": [],
        }
    )


def test_numbers_only_evolution_then_verified_context_decision() -> None:
    with tempfile.TemporaryDirectory() as directory:
        coding_llm = FakeLLMClient([_program("level", LEVEL_CODE), _program("trend", TREND_CODE)])
        coding = CodingEvolutionAgent(
            coding_llm,
            SkillLibrary.load(Path(directory) / "skills.json"),
            CodingEvolutionConfig(
                setting="statistics",
                initial_programs=1,
                mutations=1,
                validation_folds=2,
                validation_horizon=2,
                minimum_validation_history=4,
            ),
        )
        retrieval = RetrievalAgent(FakeLLMClient([_retrieval_response()]))
        decision = DecisionAgent(
            FakeLLMClient(
                [
                    json.dumps(
                        {
                            "selected_candidate_id": "trend__evidence_0",
                            "supporting_document_ids": ["doc_event"],
                            "rationale": "Use the verified two-day five-unit effect.",
                            "request_more_retrieval": False,
                        }
                    )
                ]
            )
        )
        result = EvolvingForecastHarness(
            coding,
            retrieval,
            decision,
            runtime=HarnessRuntimeConfig(retrieval_mode="single_pass"),
        ).run(_task())

        assert result.coding.selected.program.name == "trend"
        assert result.coding.improvement > 0
        assert result.coding.saved_skill_name == "trend"
        assert result.forecast == (26.0, 27.0)
        assert result.decision.llm_override_accepted
        assert result.retrieval.rejected == ()
        assert len(retrieval.llm.calls) == 1
        outcome = EvolvingForecastHarness.score_after_resolution(_task(), result)
        assert outcome.candidate_count == len(result.candidates)
        assert outcome.coding_oracle_mae == 5.0
        assert outcome.contextual_oracle_mae == 0.0
        assert outcome.retrieval_candidate_gain_mae == 5.0
        assert outcome.decision_selection_mae_regret == 0.0
        assert -1.0 <= outcome.hindcast_future_rank_correlation <= 1.0

        # Coding receives the numeric Task only; future labels and documents never enter its prompt.
        coding_text = " ".join(
            call["messages"][0]["content"] for call in coding_llm.calls
        )
        assert "doc_event" not in coding_text
        assert "future_values" not in coding_text
        assert "26.0" not in coding_text


def test_read_only_evaluation_cannot_write_coding_skills() -> None:
    with tempfile.TemporaryDirectory() as directory:
        library = SkillLibrary.load(Path(directory) / "skills.json")
        harness = EvolvingForecastHarness(
            CodingEvolutionAgent(
                FakeLLMClient([_program("trend", TREND_CODE)]),
                library,
                CodingEvolutionConfig(
                    setting="statistics",
                    initial_programs=1,
                    mutations=0,
                    validation_folds=2,
                    validation_horizon=2,
                    minimum_validation_history=4,
                ),
            ),
            RetrievalAgent(FakeLLMClient([_retrieval_response()])),
            DecisionAgent(
                FakeLLMClient(
                    [
                        json.dumps(
                            {
                                "selected_candidate_id": "trend__evidence_0",
                                "supporting_document_ids": ["doc_event"],
                                "rationale": "Use the verified event window.",
                                "request_more_retrieval": False,
                            }
                        )
                    ]
                )
            ),
        )

        evaluation = evaluate_policy(
            HarnessPolicy(),
            (_task(),),
            lambda _policy: harness,
            learn_skills=False,
            harness=harness,
        )

        assert evaluation.system_reward == 0.0
        assert evaluation.outcomes[0].final_mae == 0.0
        assert len(library) == 0
        assert not (Path(directory) / "skills.json").exists()


def test_setting2_adds_a_cited_knowledge_branch_without_removing_plain_candidates() -> None:
    conditioned = json.dumps(
        {
            "programs": [
                {
                    "name": "analogue",
                    "description": "Use a historical continuation when it survives hindcasts.",
                    "assumption": "A prior trajectory has a transferable continuation.",
                    "failure_condition": "Historical neighbours disagree.",
                    "knowledge_ids": ["ANALOG_DIRECT_CONTINUATION", "UNKNOWN_RULE"],
                    "prior_confidence": 0.7,
                    "code": LEVEL_CODE,
                }
            ]
        }
    )
    llm = FakeLLMClient([_program("plain_level", LEVEL_CODE), conditioned])
    result = CodingEvolutionAgent(
        llm,
        config=CodingEvolutionConfig(
            setting="statistics",
            initial_programs=1,
            mutations=0,
            validation_folds=2,
            validation_horizon=2,
            minimum_validation_history=4,
            use_external_knowledge=True,
        ),
    ).run_task(_task().numeric)

    assert result.knowledge_base_version == "setting2-tskb-2026-08-15"
    assert "ANALOG_DIRECT_CONTINUATION" in result.retrieved_knowledge_ids
    assert {item.program.source for item in result.candidates} == {"generated", "knowledge"}
    knowledge_program = next(
        item.program for item in result.candidates if item.program.source == "knowledge"
    )
    assert knowledge_program.knowledge_ids == ("ANALOG_DIRECT_CONTINUATION",)
    assert knowledge_program.prior_confidence == 0.7
    assert "ANALOG_DIRECT_CONTINUATION" in llm.calls[1]["system"]


def test_setting2_evolves_plain_and_knowledge_lineages_independently() -> None:
    def knowledge_program(name: str, code: str, confidence: float) -> str:
        return json.dumps(
            {
                "programs": [
                    {
                        "name": name,
                        "description": "Use a cited analogue only when hindcasts support it.",
                        "assumption": "A historical continuation remains transferable.",
                        "failure_condition": "The analogue breaks on recent folds.",
                        "knowledge_ids": [
                            "ANALOG_DIRECT_CONTINUATION",
                            "UNKNOWN_RULE",
                        ],
                        "prior_confidence": confidence,
                        "code": code,
                    }
                ]
            }
        )

    llm = FakeLLMClient(
        [
            _program("plain_level", LEVEL_CODE),
            knowledge_program("knowledge_level", LEVEL_CODE, 0.7),
            _program("plain_revision", LEVEL_CODE),
            knowledge_program("knowledge_trend", TREND_CODE, 0.55),
        ]
    )
    result = CodingEvolutionAgent(
        llm,
        config=CodingEvolutionConfig(
            setting="statistics",
            initial_programs=1,
            mutations=1,
            mutation_children=1,
            validation_folds=2,
            validation_horizon=2,
            minimum_validation_history=4,
            use_external_knowledge=True,
        ),
    ).run_task(_task().numeric)

    assert len(llm.calls) == 4
    assert {item.program.source for item in result.candidates} == {
        "generated",
        "knowledge",
        "mutation",
        "knowledge_mutation",
    }
    assert result.selected.program.name == "knowledge_trend"
    assert result.selected_knowledge_ids == ("ANALOG_DIRECT_CONTINUATION",)
    knowledge_child = next(
        item.program
        for item in result.candidates
        if item.program.source == "knowledge_mutation"
    )
    assert knowledge_child.knowledge_ids == ("ANALOG_DIRECT_CONTINUATION",)
    assert knowledge_child.prior_confidence == 0.55
    assert "ANALOG_DIRECT_CONTINUATION" not in llm.calls[2]["system"]
    assert "ANALOG_DIRECT_CONTINUATION" in llm.calls[3]["system"]
    revision_payload = llm.calls[3]["messages"][0]["content"]
    assert "knowledge_diagnostics" in revision_payload
    assert "future_values" not in revision_payload
    assert "doc_event" not in revision_payload


def test_decision_rejects_uncited_override() -> None:
    with tempfile.TemporaryDirectory() as directory:
        coding = CodingEvolutionAgent(
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
        )
        retrieval = RetrievalAgent(FakeLLMClient([_retrieval_response()]))
        decision = DecisionAgent(
            FakeLLMClient(
                [
                    json.dumps(
                        {
                            "selected_candidate_id": "trend__evidence_0",
                            "supporting_document_ids": [],
                            "rationale": "Unsupported override.",
                            "request_more_retrieval": False,
                        }
                    )
                ]
            )
        )
        result = EvolvingForecastHarness(coding, retrieval, decision).run(_task())

        assert result.forecast == (21.0, 22.0)
        assert result.decision.rejection_reason == "override_requires_task_evidence"


def test_pure_tsfm_does_not_call_llm_or_save_external_model_as_skill() -> None:
    class FakeTSFM:
        def forecast(self, history, horizon, frequency):
            del frequency
            slope = history[-1] - history[-2]
            return tuple(history[-1] + slope * (step + 1) for step in range(horizon))

    with tempfile.TemporaryDirectory() as directory:
        llm = FakeLLMClient([])
        library = SkillLibrary.load(Path(directory) / "skills.json")
        result = CodingEvolutionAgent(
            llm,
            library,
            CodingEvolutionConfig(
                setting="tsfm",
                mutations=1,
                validation_folds=2,
                validation_horizon=2,
                minimum_validation_history=4,
            ),
            tsfm_forecaster=FakeTSFM(),
        ).run_task(_task().numeric)

        assert result.selected.program.name == "tsfm_backbone"
        assert result.selected.forecast == (21.0, 22.0)
        assert llm.calls == []
        assert len(library) == 0


def test_hidden_labels_cannot_be_used_for_posthoc_evolution() -> None:
    task = _task()
    task = ContextTask(**{**task.__dict__, "labels_public": False})
    with tempfile.TemporaryDirectory() as directory:
        harness = EvolvingForecastHarness(
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
            RetrievalAgent(FakeLLMClient([_retrieval_response()])),
            DecisionAgent(
                FakeLLMClient(
                    [
                        json.dumps(
                            {
                                "selected_candidate_id": "trend",
                                "supporting_document_ids": [],
                                "rationale": "Use the best hindcast.",
                                "request_more_retrieval": False,
                            }
                        )
                    ]
                )
            ),
        )
        result = harness.run(task)
        try:
            harness.score_after_resolution(task, result)
        except ValueError as error:
            assert "forbidden" in str(error)
        else:
            raise AssertionError("hidden labels must not be accepted for evolution")
