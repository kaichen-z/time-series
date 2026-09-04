"""Trusted construction of package task-feedback projections."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from common.evolution_core.task_feedback import TaskFeedbackError
from common.llm import FakeLLMClient
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.numerical_two_stage import run_numerical_two_stage
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from evolving_loop.package_task_feedback import PackageTaskFeedbackLedger
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from tests.test_package_coordinate_evolution import _bundle
from tests.test_package_decision_evolution import (
    _decision_task,
    _final_specialist,
    _request_specialist,
    _round2_response,
    _safe_default_package,
)
from tests.test_package_retrieval_evolution import _frozen_registry, _round_response


def _verified_state_and_result(tmp_path):
    task = _decision_task()
    package = _safe_default_package()
    registry = _frozen_registry(((task, package),))
    _unused, state = _bundle(tmp_path, registry=registry)
    genome = state.bundle.policy.retrieval_genome
    assert genome is not None
    retrieval = TwoStageRetrievalAgent(
        FakeLLMClient([_round_response(), _round2_response(task)]),
        genome,
        RetrievalSkillLibrary(tmp_path / "ledger-skills.json", persist=False),
    )
    decision = DecisionAgent(
        FakeLLMClient([_request_specialist(), _final_specialist()])
    )
    result = run_numerical_two_stage(task, package, retrieval, decision)
    return task, state, result


def test_ledger_projects_verified_train_trace_without_task_or_document_ids(
    tmp_path,
) -> None:
    task, state, result = _verified_state_and_result(tmp_path)
    ledger = PackageTaskFeedbackLedger({task.numeric.task_id: "train"})

    ledger.record(state.bundle, task, result)
    projection = ledger.build_projection(
        state.bundle, (task.numeric.task_id,), generation=3
    )

    assert len(projection.cases) == 1
    case = projection.cases[0]
    assert case.assumption_id == "assumption_001"
    assert case.stance == "supported"
    assert case.target_match == "matched"
    assert case.window_relation == "overlaps"
    assert case.magnitude_status == "present"
    assert case.mechanism == "event_shock"
    assert case.decision_action == "selected"
    encoded = json.dumps(projection.to_payload(), sort_keys=True)
    assert "partition" not in encoded
    assert task.numeric.task_id not in encoded
    payload_strings = {
        value
        for item in projection.to_payload()["cases"]
        for value in (
            item["case_id"],
            item["assumption_id"],
            item["claim"],
            item["failure_condition"],
            item["evidence_chain_sha256"],
        )
    }
    assert all(
        document.document_id not in payload_strings for document in task.documents
    )


def test_ledger_rejects_public_membership_and_cross_bundle_replay(tmp_path) -> None:
    task, state, result = _verified_state_and_result(tmp_path)
    with pytest.raises(TaskFeedbackError, match="Train or Dev"):
        PackageTaskFeedbackLedger({task.numeric.task_id: "public"})

    ledger = PackageTaskFeedbackLedger({task.numeric.task_id: "dev"})
    ledger.record(state.bundle, task, result)
    changed = replace(
        state.bundle,
        policy=replace(state.bundle.policy, decision_prompt="changed decision"),
    )
    with pytest.raises(TaskFeedbackError, match="trace|bundle"):
        ledger.build_projection(changed, (task.numeric.task_id,), generation=3)


def test_pipeline_evaluator_captures_only_successful_verified_inference(
    tmp_path,
) -> None:
    task, state, _result = _verified_state_and_result(tmp_path)
    ledger = PackageTaskFeedbackLedger({task.numeric.task_id: "dev"})
    skills = RetrievalSkillLibrary(tmp_path / "pipeline-skills.json", persist=False)

    def retrieval_factory(policy):
        assert policy.retrieval_genome is not None
        return TwoStageRetrievalAgent(
            FakeLLMClient([_round_response(), _round2_response(task)]),
            policy.retrieval_genome,
            skills,
        )

    def decision_factory(_policy):
        return DecisionAgent(
            FakeLLMClient([_request_specialist(), _final_specialist()])
        )

    evaluator = PackagePipelineEvaluator(
        retrieval_factory,
        decision_factory,
        task_feedback_ledger=ledger,
    )

    evaluation = evaluator.evaluate(
        state.bundle, state.registry, (task,), stage="dev20"
    )
    projection = ledger.build_projection(
        state.bundle, (task.numeric.task_id,), generation=3
    )

    assert evaluation.coverage == 1.0
    assert len(projection.cases) == 1


def test_ledger_does_not_disguise_fallback_inference_as_empty_evidence(
    tmp_path,
) -> None:
    task, state, result = _verified_state_and_result(tmp_path)
    ledger = PackageTaskFeedbackLedger({task.numeric.task_id: "train"})

    ledger.record(state.bundle, task, replace(result, fallback_reason="invalid_round2"))

    with pytest.raises(TaskFeedbackError, match="missing"):
        ledger.build_projection(state.bundle, (task.numeric.task_id,), generation=3)
