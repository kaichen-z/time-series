"""Tests for the evolving_loop harness: end-to-end runs, frozen inference, and outcome-driven skill learning."""
from __future__ import annotations

import json
import tempfile
import pytest
from pathlib import Path
from evolving_loop.coding_agent.evolution import CodingEvolutionAgent, CodingEvolutionConfig
from evolving_loop.coding_agent.skill_library import SkillLibrary
from evolving_loop.data import ContextTask, Document, Task, load_context_tasks
from evolving_loop.decision_agent.agent import DecisionAgent, DecisionCandidate, DecisionResult
from evolving_loop.harness import EvolvingForecastHarness, HarnessResult
from evolving_loop.retrieval_agent.agent import Evidence, RetrievalAgent, RetrievalResult
from common.llm import FakeLLMClient
from evolving_loop.decision_agent.skill_library import DecisionSkillLibrary
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.skill_learning import OutcomeSkillLearner
from dataclasses import replace
from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.frozen_inference import inference_view, run_frozen_inference
from evolving_loop.cli import build_parser, inference_command


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
        result = EvolvingForecastHarness(coding, retrieval, decision).run(_task())

        assert result.coding.selected.program.name == "trend"
        assert result.coding.improvement > 0
        assert result.coding.saved_skill_name == "trend"
        assert result.forecast == (26.0, 27.0)
        assert result.decision.llm_override_accepted
        assert result.retrieval.rejected == ()
        outcome = EvolvingForecastHarness.score_after_resolution(_task(), result)
        assert outcome.candidate_count == len(result.candidates)
        assert -1.0 <= outcome.hindcast_future_rank_correlation <= 1.0

        # Coding receives the numeric Task only; future labels and documents never enter its prompt.
        coding_text = " ".join(
            call["messages"][0]["content"] for call in coding_llm.calls
        )
        assert "doc_event" not in coding_text
        assert "future_values" not in coding_text
        assert "26.0" not in coding_text


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
    assert "ANALOG_DIRECT_CONTINUATION" in result.selected_knowledge_ids
    assert {item.program.source for item in result.candidates} == {"generated", "knowledge"}
    knowledge_program = next(
        item.program for item in result.candidates if item.program.source == "knowledge"
    )
    assert knowledge_program.knowledge_ids == ("ANALOG_DIRECT_CONTINUATION",)
    assert knowledge_program.prior_confidence == 0.7
    assert "ANALOG_DIRECT_CONTINUATION" in llm.calls[1]["system"]


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

        assert outcome.final_smae == 0.0
        assert learning.retrieval_skill_name == "retrieve_explicit_event_window"
        assert learning.decision_skill_name == "prefer_verified_window"
        assert retrieval_path.exists()
        assert decision_path.exists()
        assert len(retrieval_library) == 1
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


def test_skill_summaries_are_available_but_unknown_claims_are_not_trusted() -> None:
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
        assert "retrieve_explicit_event_window" in retrieval_llm.calls[0]["messages"][0]["content"]
        assert "prefer_verified_window" in decision_llm.calls[0]["messages"][0]["content"]

def _frozen_task(*, public: bool) -> ContextTask:
    return ContextTask(
        numeric=Task(
            task_id="task_hidden" if not public else "task_public",
            history_values=(1.0, 2.0, 3.0),
            future_values=(4.0, 5.0) if public else (),
            prediction_length=2,
            frequency="1 day",
            seasonal_period=None,
            entity_name="Entity",
        ),
        target_name="value",
        target_description="A value",
        history_timestamps=("1", "2", "3"),
        future_timestamps=("4", "5"),
        documents=(Document("doc_1", "A relevant sentence.", "supporting", "x"),),
        gt_evidence=("secret evaluator evidence",),
        labels_public=public,
    )


def _result(task_id: str) -> HarnessResult:
    candidate = DecisionCandidate(
        candidate_id="level",
        forecast=(3.0, 3.0),
        assumption="level persists",
        failure_condition="regime changes",
        hindcast_smae=1.0,
    )
    retrieval = RetrievalResult(
        query="q",
        selected_document_ids=("doc_1",),
        evidence=(Evidence("doc_1", "claim", "A relevant sentence."),),
        impacts=(),
        sufficient=True,
        missing_information=(),
    )
    decision = DecisionResult(
        selected=candidate,
        host_default_id="level",
        requested_more_retrieval=False,
        rationale="validated",
        supporting_document_ids=(),
        llm_override_accepted=False,
    )
    return HarnessResult(
        task_id=task_id,
        coding=object(),
        retrieval=retrieval,
        decision=decision,
        candidates=(candidate,),
        forecast=candidate.forecast,
    )


def test_hidden_loader_retains_task_but_strips_evaluator_fields(tmp_path: Path) -> None:
    record = {
        "benchmark_id": "task_hidden",
        "labels_public": False,
        "showcase": {
            "entity": {"name": "Entity"},
            "time_series_variable": {"name": "value"},
        },
        "task_metadata": {"prediction_length": 2, "frequency": "1 day"},
        "series": {
            "history_values": [1, 2, 3],
            "history_timestamps": ["1", "2", "3"],
            "future_timestamps": ["4", "5"],
            "future_values": [4, 5],
        },
        "documents": [
            {
                "document_id": "doc_1",
                "content": "text",
                "role": "supporting",
                "subtype": "leaky",
            }
        ],
        "annotations": {"gt_evidence": [{"evidence": "secret"}]},
    }
    path = tmp_path / "hidden.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_context_tasks(path) == []
    loaded = load_context_tasks(path, include_unlabeled=True)
    assert len(loaded) == 1
    task = loaded[0]
    assert task.numeric.future_values == ()
    assert task.gt_evidence == ()
    assert task.documents[0].role is None
    assert task.documents[0].subtype is None
    assert not task.labels_public


def test_frozen_inference_strips_labels_and_exports_submission(tmp_path: Path) -> None:
    observed = []

    class Harness:
        def run(self, task):
            observed.append(task)
            return _result(task.numeric.task_id)

    summary = run_frozen_inference(
        HarnessPolicy(),
        [_frozen_task(public=False)],
        lambda _: Harness(),
        output_dir=tmp_path,
        samples=3,
        score_public=False,
    )
    assert summary["labels_accessed"] is False
    assert observed[0] == inference_view(_frozen_task(public=False))
    forecast = json.loads((tmp_path / "forecasts.jsonl").read_text(encoding="utf-8"))
    research = json.loads((tmp_path / "deep_research.jsonl").read_text(encoding="utf-8"))
    assert forecast == {
        "benchmark_id": "task_hidden",
        "samples": [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]],
    }
    assert research["cited_document_ids"] == ["doc_1"]


def test_hidden_task_cannot_be_scored_by_frozen_runner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        run_frozen_inference(
            HarnessPolicy(),
            [_frozen_task(public=False)],
            lambda _: object(),
            output_dir=tmp_path,
            score_public=True,
        )


def test_cli_rejects_hidden_scoring_before_loading_data() -> None:
    args = build_parser().parse_args(
        ["--inference", "genome", "--hidden-test", "--score-public"]
    )
    with pytest.raises(ValueError, match="forbidden"):
        inference_command(args)
