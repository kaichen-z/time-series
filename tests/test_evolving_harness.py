from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from evolving_loop.coding_agent.evolution import (
    CodingEvolutionAgent,
    CodingEvolutionConfig,
)
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
        coding_llm = FakeLLMClient(
            [_program("level", LEVEL_CODE), _program("trend", TREND_CODE)]
        )
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
        result = EvolvingForecastHarness(coding, retrieval, decision).run(_task())

        assert result.coding.selected.program.name == "trend"
        assert result.coding.improvement > 0
        assert result.coding.saved_skill_name == "trend"
        assert result.forecast == (26.0, 27.0)
        assert result.decision.llm_override_accepted
        assert result.retrieval.rejected == ()
        outcome = EvolvingForecastHarness.score_after_resolution(_task(), result)
        assert outcome.candidate_count == len(result.candidates)
        assert outcome.coding_oracle_mae == 5.0
        assert outcome.contextual_oracle_mae == 0.0
        assert outcome.retrieval_candidate_gain_mae == 5.0
        assert outcome.decision_selection_mae_regret == 0.0
        assert -1.0 <= outcome.hindcast_future_rank_correlation <= 1.0
        assert outcome.final_drcik_smae == 0.0
        assert outcome.final_drcik_srmse == 0.0
        assert outcome.final_drcik_scrps_deterministic == 0.0

        retrieval_payload = json.loads(retrieval.llm.calls[0]["messages"][0]["content"])
        assert retrieval_payload["history_values"] == list(
            _task().numeric.history_values
        )
        assert "future_values" not in retrieval_payload

        # Coding receives the numeric Task only; future labels and documents never enter its prompt.
        coding_text = " ".join(
            call["messages"][0]["content"] for call in coding_llm.calls
        )
        assert "doc_event" not in coding_text
        assert "future_values" not in coding_text
        assert "26.0" not in coding_text


def test_retrieval_rejects_a_numeric_adjustment_not_present_in_the_quote() -> None:
    response = json.loads(_retrieval_response())
    response["impacts"][0]["adjustment_value"] = 50
    retrieval = RetrievalAgent(FakeLLMClient([json.dumps(response)])).run(_task(), ())

    assert retrieval.impacts[0].adjustment_kind == "none"
    assert retrieval.impacts[0].adjustment_value is None
    assert "quantitative_impact_without_matching_magnitude" in retrieval.rejected
    assert not retrieval.sufficient


def test_every_verified_evidence_document_is_exported_as_a_citation() -> None:
    response = json.loads(_retrieval_response())
    response["selected_document_ids"] = []

    retrieval = RetrievalAgent(FakeLLMClient([json.dumps(response)])).run(_task(), ())

    assert retrieval.selected_document_ids == ("doc_event",)


def test_retrieval_rejects_wrong_direction_duration_number_and_unquoted_window() -> None:
    wrong_direction = json.loads(_retrieval_response())
    wrong_direction["impacts"][0]["direction"] = "down"
    wrong_direction["impacts"][0]["adjustment_value"] = -5
    result = RetrievalAgent(FakeLLMClient([json.dumps(wrong_direction)])).run(
        _task(), ()
    )
    assert result.impacts[0].adjustment_kind == "none"
    assert "quantitative_impact_without_matching_magnitude" in result.rejected

    date_number = json.loads(_retrieval_response())
    date_number["impacts"][0]["adjustment_value"] = 21
    result = RetrievalAgent(FakeLLMClient([json.dumps(date_number)])).run(_task(), ())
    assert result.impacts[0].adjustment_kind == "none"
    assert "quantitative_impact_without_matching_magnitude" in result.rejected

    duration_quote = (
        "Alpha Store promotion is a 5-day event and increases sales by 2 units from "
        "2026-01-21 through 2026-01-22."
    )
    duration_task = replace(
        _task(),
        documents=(Document("doc_event", duration_quote, role="supporting"),),
    )
    unrelated_number = json.loads(_retrieval_response())
    unrelated_number["evidence"][0]["exact_quote"] = duration_quote
    unrelated_number["impacts"][0]["adjustment_value"] = 5
    result = RetrievalAgent(FakeLLMClient([json.dumps(unrelated_number)])).run(
        duration_task, ()
    )
    assert result.impacts[0].adjustment_kind == "none"
    assert "quantitative_impact_without_matching_magnitude" in result.rejected

    fractional_quote = (
        "Alpha Store sales increase by 0.2 units from 2026-01-21 through " "2026-01-22."
    )
    fractional_task = replace(
        _task(),
        documents=(Document("doc_event", fractional_quote, role="supporting"),),
    )
    non_percent_multiplier = json.loads(_retrieval_response())
    non_percent_multiplier["evidence"][0]["exact_quote"] = fractional_quote
    non_percent_multiplier["impacts"][0]["adjustment_kind"] = "multiply"
    non_percent_multiplier["impacts"][0]["adjustment_value"] = 0.2
    result = RetrievalAgent(FakeLLMClient([json.dumps(non_percent_multiplier)])).run(
        fractional_task, ()
    )
    assert result.impacts[0].adjustment_kind == "none"
    assert "quantitative_impact_without_matching_magnitude" in result.rejected

    percent_as_absolute = json.loads(_retrieval_response())
    percent_quote = (
        "Alpha Store sales will increase by 5 percent from 2026-01-21 through "
        "2026-01-22."
    )
    percent_task = replace(
        _task(),
        documents=(Document("doc_event", percent_quote, role="supporting"),),
    )
    percent_as_absolute["evidence"][0]["exact_quote"] = percent_quote
    result = RetrievalAgent(FakeLLMClient([json.dumps(percent_as_absolute)])).run(
        percent_task, ()
    )
    assert result.impacts[0].adjustment_kind == "none"
    assert "quantitative_impact_without_matching_magnitude" in result.rejected

    percent_multiplier = json.loads(_retrieval_response())
    percent_multiplier["evidence"][0]["exact_quote"] = percent_quote
    percent_multiplier["impacts"][0]["adjustment_kind"] = "multiply"
    percent_multiplier["impacts"][0]["adjustment_value"] = 0.05
    result = RetrievalAgent(FakeLLMClient([json.dumps(percent_multiplier)])).run(
        percent_task, ()
    )
    assert result.impacts[0].adjustment_kind == "multiply"
    assert result.impacts[0].adjustment_value == 0.05
    assert result.rejected == ()

    wrong_window = json.loads(_retrieval_response())
    wrong_window["impacts"][0]["end_timestamp"] = "2026-01-23"
    result = RetrievalAgent(FakeLLMClient([json.dumps(wrong_window)])).run(_task(), ())
    assert result.impacts[0].adjustment_kind == "none"
    assert "quantitative_impact_without_quoted_window" in result.rejected

    wrong_date_quote = (
        "Alpha Store sales will increase by 20 percent from 09:00 to 10:00 on "
        "2025-12-01."
    )
    intraday_task = replace(
        _task(),
        future_timestamps=("2026-01-21 09:00:00", "2026-01-21 10:00:00"),
        documents=(Document("doc_event", wrong_date_quote, role="supporting"),),
    )
    wrong_intraday_date = json.loads(_retrieval_response())
    wrong_intraday_date["evidence"][0]["exact_quote"] = wrong_date_quote
    wrong_intraday_date["impacts"][0].update(
        {
            "adjustment_kind": "multiply",
            "adjustment_value": 0.2,
            "start_timestamp": "2026-01-21 09:00:00",
            "end_timestamp": "2026-01-21 10:00:00",
        }
    )
    result = RetrievalAgent(FakeLLMClient([json.dumps(wrong_intraday_date)])).run(
        intraday_task, ()
    )
    assert result.impacts[0].adjustment_kind == "none"
    assert "quantitative_impact_without_quoted_window" in result.rejected


def test_follow_up_is_authoritative_snapshot_and_retracts_prior_evidence() -> None:
    down_quote = (
        "Alpha Store sales will decrease by 2 units from 2026-01-21 through "
        "2026-01-22."
    )
    task = replace(
        _task(),
        documents=_task().documents
        + (Document("doc_down", down_quote, role="supporting"),),
    )
    second = json.loads(_retrieval_response())
    second["query"] = "corrected sales impact"
    second["selected_document_ids"] = ["doc_down"]
    second["evidence"] = [
        {
            "document_id": "doc_down",
            "claim": "Sales decrease by two units for the full horizon.",
            "exact_quote": down_quote,
        }
    ]
    second["impacts"] = [
        {
            "source_document_ids": ["doc_down"],
            "mechanism_layer": "future_driver",
            "temporal_relation": "overlaps_future",
            "direction": "down",
            "permanence": "temporary",
            "adjustment_kind": "add",
            "adjustment_value": -2,
            "start_timestamp": "2026-01-21",
            "end_timestamp": "2026-01-22",
            "rationale": "The corrected quote gives a two-unit decrease and window.",
        }
    ]
    retrieval_llm = FakeLLMClient([_retrieval_response(), json.dumps(second)])
    decision_llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "selected_candidate_id": "trend",
                    "supporting_document_ids": [],
                    "rationale": "Audit the first impact.",
                    "request_more_retrieval": True,
                }
            ),
            json.dumps(
                {
                    "selected_candidate_id": "trend__evidence_0",
                    "supporting_document_ids": ["doc_down"],
                    "rationale": "Use the corrected two-unit decrease.",
                    "request_more_retrieval": False,
                }
            ),
        ]
    )
    harness = EvolvingForecastHarness(
        CodingEvolutionAgent(
            FakeLLMClient([_program("trend", TREND_CODE)]),
            config=CodingEvolutionConfig(
                setting="statistics",
                initial_programs=1,
                mutations=0,
                validation_folds=2,
                validation_horizon=2,
                minimum_validation_history=4,
            ),
        ),
        RetrievalAgent(retrieval_llm),
        DecisionAgent(decision_llm),
        runtime=HarnessRuntimeConfig(
            workflow=("retrieve", "decide", "retrieve", "decide"),
            max_evidence_adjustments=1,
        ),
    )

    result = harness.run(task)

    assert result.retrieval.selected_document_ids == ("doc_down",)
    assert [item.document_id for item in result.retrieval.evidence] == ["doc_down"]
    assert result.retrieval.impacts[0].adjustment_value == -2
    assert result.forecast == (19.0, 20.0)
    follow_up = json.loads(retrieval_llm.calls[1]["messages"][0]["content"])
    assert follow_up["prior_retrieval"]["evidence"][0]["exact_quote"]
    assert follow_up["prior_retrieval"]["impacts"][0]["adjustment_value"] == 5


def test_decision_host_rejects_override_when_retrieval_is_insufficient() -> None:
    retrieval_response = json.loads(_retrieval_response())
    retrieval_response["sufficient"] = False
    retrieval_response["missing_information"] = ["target-unit confirmation"]
    harness = EvolvingForecastHarness(
        CodingEvolutionAgent(
            FakeLLMClient([_program("trend", TREND_CODE)]),
            config=CodingEvolutionConfig(
                setting="statistics",
                initial_programs=1,
                mutations=0,
                validation_folds=2,
                validation_horizon=2,
                minimum_validation_history=4,
            ),
        ),
        RetrievalAgent(FakeLLMClient([json.dumps(retrieval_response)])),
        DecisionAgent(
            FakeLLMClient(
                [
                    json.dumps(
                        {
                            "selected_candidate_id": "trend__evidence_0",
                            "supporting_document_ids": ["doc_event"],
                            "rationale": "Attempt an unresolved override.",
                            "request_more_retrieval": False,
                        }
                    )
                ]
            )
        ),
    )

    result = harness.run(_task())

    assert result.decision.selected.candidate_id == result.decision.host_default_id
    assert not result.decision.llm_override_accepted
    assert result.decision.rejection_reason == "override_requires_sufficient_retrieval"


def test_second_retrieval_receives_failed_first_round_and_decision_feedback() -> None:
    first_retrieval = json.dumps(
        {
            "query": "future event",
            "selected_document_ids": [],
            "evidence": [],
            "impacts": [],
            "sufficient": False,
            "missing_information": ["event magnitude"],
        }
    )
    retrieval_llm = FakeLLMClient([first_retrieval, _retrieval_response()])
    decision_llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "selected_candidate_id": "trend",
                    "supporting_document_ids": [],
                    "rationale": "Need the event magnitude.",
                    "request_more_retrieval": True,
                }
            ),
            json.dumps(
                {
                    "selected_candidate_id": "trend__evidence_0",
                    "supporting_document_ids": ["doc_event"],
                    "rationale": "The follow-up supplies the exact five-unit window.",
                    "request_more_retrieval": False,
                }
            ),
        ]
    )
    harness = EvolvingForecastHarness(
        CodingEvolutionAgent(
            FakeLLMClient([_program("trend", TREND_CODE)]),
            config=CodingEvolutionConfig(
                setting="statistics",
                initial_programs=1,
                mutations=0,
                validation_folds=2,
                validation_horizon=2,
                minimum_validation_history=4,
            ),
        ),
        RetrievalAgent(retrieval_llm),
        DecisionAgent(decision_llm),
        runtime=HarnessRuntimeConfig(
            workflow=("retrieve", "decide", "retrieve", "decide")
        ),
    )

    result = harness.run(_task())

    follow_up = json.loads(retrieval_llm.calls[1]["messages"][0]["content"])
    first_decision = json.loads(decision_llm.calls[0]["messages"][0]["content"])
    assert follow_up["prior_retrieval"]["missing_information"] == ["event magnitude"]
    assert follow_up["prior_decision_feedback"]["request_more_retrieval"] is True
    assert first_decision["retrieval_status"]["sufficient"] is False
    assert result.retrieval.sufficient
    assert result.retrieval.missing_information == ()
    assert result.forecast == (26.0, 27.0)


def test_decision_aggregation_rejects_cross_snapshot_votes() -> None:
    harness = EvolvingForecastHarness(
        CodingEvolutionAgent(FakeLLMClient([])),
        RetrievalAgent(FakeLLMClient([])),
        DecisionAgent(FakeLLMClient([])),
        runtime=HarnessRuntimeConfig(
            workflow=("retrieve", "decide", "retrieve", "decide"),
            decision_aggregation="majority",
        ),
    )

    try:
        harness.run(_task())
    except ValueError as error:
        assert str(error) == (
            "Decision aggregation cannot combine different retrieval snapshots"
        )
    else:
        raise AssertionError("cross-snapshot majority voting must be rejected")


def test_same_snapshot_majority_and_final_decision_boundary() -> None:
    decision_response = json.dumps(
        {
            "selected_candidate_id": "trend",
            "supporting_document_ids": [],
            "rationale": "Keep the best hindcast.",
            "request_more_retrieval": False,
        }
    )
    safe = EvolvingForecastHarness(
        CodingEvolutionAgent(
            FakeLLMClient([_program("trend", TREND_CODE)]),
            config=CodingEvolutionConfig(
                setting="statistics",
                initial_programs=1,
                mutations=0,
                validation_folds=2,
                validation_horizon=2,
                minimum_validation_history=4,
            ),
        ),
        RetrievalAgent(FakeLLMClient([_retrieval_response(), _retrieval_response()])),
        DecisionAgent(FakeLLMClient([decision_response, decision_response])),
        runtime=HarnessRuntimeConfig(
            workflow=("retrieve", "retrieve", "decide", "decide"),
            decision_aggregation="majority",
        ),
    )

    assert safe.run(_task()).forecast == (21.0, 22.0)

    stale = EvolvingForecastHarness(
        CodingEvolutionAgent(FakeLLMClient([])),
        RetrievalAgent(FakeLLMClient([])),
        DecisionAgent(FakeLLMClient([])),
        runtime=HarnessRuntimeConfig(
            workflow=("retrieve", "decide", "retrieve"),
            decision_aggregation="last",
        ),
    )
    try:
        stale.run(_task())
    except ValueError as error:
        assert str(error) == "A workflow with Decision stages must end with decide"
    else:
        raise AssertionError("the final retrieval snapshot requires a final decision")


def test_decision_rejects_override_that_claims_an_unknown_skill() -> None:
    decision_response = json.dumps(
        {
            "selected_candidate_id": "trend__evidence_0",
            "supporting_document_ids": ["doc_event"],
            "rationale": "Use a fabricated decision rule.",
            "request_more_retrieval": False,
            "used_skill_names": ["fabricated_gate"],
        }
    )
    harness = EvolvingForecastHarness(
        CodingEvolutionAgent(
            FakeLLMClient([_program("trend", TREND_CODE)]),
            config=CodingEvolutionConfig(
                setting="statistics",
                initial_programs=1,
                mutations=0,
                validation_folds=2,
                validation_horizon=2,
                minimum_validation_history=4,
            ),
        ),
        RetrievalAgent(FakeLLMClient([_retrieval_response()])),
        DecisionAgent(FakeLLMClient([decision_response])),
    )

    result = harness.run(_task())

    assert result.decision.selected.candidate_id == result.decision.host_default_id
    assert not result.decision.llm_override_accepted
    assert result.decision.rejection_reason == "unknown_decision_skills:fabricated_gate"


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
    assert {item.program.source for item in result.candidates} == {
        "generated",
        "knowledge",
    }
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


def test_duplicate_coding_names_cannot_overwrite_the_host_default() -> None:
    empty_retrieval = json.dumps(
        {
            "query": "no relevant future evidence",
            "selected_document_ids": [],
            "evidence": [],
            "impacts": [],
            "sufficient": False,
            "missing_information": ["No forecast-relevant evidence found."],
        }
    )
    decision = json.dumps(
        {
            "selected_candidate_id": "duplicate__2",
            "supporting_document_ids": [],
            "rationale": "Keep the lowest-hindcast candidate.",
            "request_more_retrieval": False,
        }
    )
    harness = EvolvingForecastHarness(
        CodingEvolutionAgent(
            FakeLLMClient(
                [
                    _program("duplicate", LEVEL_CODE),
                    _program("duplicate", TREND_CODE),
                ]
            ),
            config=CodingEvolutionConfig(
                setting="statistics",
                initial_programs=1,
                mutations=1,
                mutation_children=1,
                validation_folds=2,
                validation_horizon=2,
                minimum_validation_history=4,
            ),
        ),
        RetrievalAgent(FakeLLMClient([empty_retrieval])),
        DecisionAgent(FakeLLMClient([decision])),
    )

    result = harness.run(_task())
    candidate_ids = [item.candidate_id for item in result.candidates]
    outcome = harness.score_after_resolution(_task(), result)

    assert candidate_ids == ["duplicate__2", "duplicate"]
    assert len(candidate_ids) == len(set(candidate_ids))
    assert result.decision.host_default_id == "duplicate__2"
    assert result.decision.selected.candidate_id == "duplicate__2"
    assert result.forecast == (21.0, 22.0)
    assert outcome.contextual_oracle_mae == 5.0
    assert outcome.final_mae == 5.0


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
