from __future__ import annotations

import json
from dataclasses import replace

import pytest

from common.llm import FakeLLMClient
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.package_numerical_supply import build_package_registry
from evolving_loop.package_registry import numerical_package_fingerprint
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from evolving_loop.v2.bundle import EvolutionBundleV2
from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.cooperative.adapters import (
    CooperativeArtifactCatalog,
    CooperativePipelineAdapter,
    sanitize_train_feedback,
)
from evolving_loop.v2.cooperative.contracts import DecisionModuleV2, RetrievalModuleV2
from evolving_loop.v2.numerical_qd.adapters import FrozenNumericalArtifactsV2, import_numerical_seed
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_handoff import task_input_fingerprint
from numerical_agent.evolution.screening import profile_task
from tests.test_package_numerical_supply import (
    _registry_tasks,
)
from tests.test_package_retrieval_evolution import _package, _seed_supply_release
from tests.test_package_stage_runner import _evaluation


def _train_tasks():
    seeds = _registry_tasks()
    return tuple(
        replace(
            seeds[index % len(seeds)],
            numeric=replace(
                seeds[index % len(seeds)].numeric,
                task_id=f"cooperative-train-{index}",
                entity_name=f"Cooperative Entity {index}",
            ),
        )
        for index in range(4)
    )


def _round1_response() -> str:
    return json.dumps(
        {
            "evidence_chains": [],
            "counterevidence": [],
            "missing_information": [],
            "sufficient": True,
        }
    )


def _decision_response() -> str:
    return json.dumps(
        {
            "selected_candidate_id": "safe_anchor",
            "supporting_document_ids": [],
            "rationale": "Preserve the frozen safe anchor.",
            "request_more_retrieval": False,
            "gaps": [],
            "used_skill_names": [],
        }
    )


def _pipeline_package(task, release):
    package = _package()
    profile = profile_task(
        Task(
            task.numeric.task_id,
            task.numeric.history_values,
            task.numeric.prediction_length,
            task.numeric.frequency,
            (),
        )
    )
    return replace(
        package,
        task_profile=profile,
        component_fingerprints={
            **dict(package.component_fingerprints),
            "task_input": task_input_fingerprint(
                task_id=task.numeric.task_id,
                history=task.numeric.history_values,
                frequency=task.numeric.frequency,
                horizon=task.numeric.prediction_length,
            ),
            "numerical_supply_release": release.fingerprint,
        },
    )


def _marked(evaluation, stage):
    return type(evaluation).from_rows(
        evaluation.candidate_sha256,
        evaluation.task_rows,
        evaluation.expected_task_ids,
        {f"cooperative_stage_{stage}": 1.0},
    )


@pytest.fixture
def pipeline_case(tmp_path):
    tasks = _train_tasks()
    release = _seed_supply_release()
    registry = build_package_registry(tasks, release, _pipeline_package)
    envelope = import_numerical_seed(release, registry, tasks=tasks).envelope
    numerical = FrozenNumericalArtifactsV2(release, registry, envelope, ())
    catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
    numerical_ids = catalog.add_numerical(numerical)
    retrieval = RetrievalModuleV2(
        1, "a" * 64, RetrievalGenome.seed().to_payload(), ()
    )
    decision = DecisionModuleV2(1, "safe decision prompt", (), True, 2, "last")
    retrieval_sha = catalog.add_retrieval(retrieval)
    decision_sha = catalog.add_decision(decision)
    bundle = EvolutionBundleV2(
        2,
        0,
        None,
        *numerical_ids,
        retrieval_sha,
        decision_sha,
        "5" * 64,
        "6" * 64,
        "7" * 64,
        "8" * 64,
        {"python": "9" * 64},
        None,
    )
    trace: list[tuple[str, str]] = []
    retrieval_clients: list[FakeLLMClient] = []
    decision_clients: list[FakeLLMClient] = []

    def retrieval_factory(genome, skills):
        task_id = tasks[len(retrieval_clients) % len(tasks)].numeric.task_id
        trace.append((task_id, "numerical"))
        client = FakeLLMClient([_round1_response()])
        retrieval_clients.append(client)
        return TwoStageRetrievalAgent(client, genome, skills)

    def decision_factory(module):
        client = FakeLLMClient([_decision_response(), _decision_response()])
        decision_clients.append(client)
        return DecisionAgent(client, prompt=module.prompt)

    pipeline = CooperativePipelineAdapter(
        catalog,
        retrieval_factory,
        decision_factory,
        empty_skill_path=tmp_path / "empty-retrieval-skills.json",
    )
    return {
        "pipeline": pipeline,
        "bundle": bundle,
        "tasks": tasks,
        "trace": trace,
        "retrieval_clients": retrieval_clients,
        "decision_clients": decision_clients,
        "registry": registry,
        "catalog": catalog,
        "retrieval": retrieval,
        "decision": decision,
    }


def test_pipeline_adapter_runs_all_three_agents_and_binds_module_identities(
    pipeline_case,
):
    result = pipeline_case["pipeline"].evaluate(
        pipeline_case["bundle"], pipeline_case["tasks"], stage="train"
    )

    tasks = pipeline_case["tasks"]
    assert result.task_count == len(tasks) == 4
    assert result.coverage == 1.0
    assert pipeline_case["trace"] == [
        (task.numeric.task_id, "numerical") for task in tasks
    ]
    assert len({id(client) for client in pipeline_case["retrieval_clients"]}) == 4
    assert len({id(client) for client in pipeline_case["decision_clients"]}) == 4
    assert all(len(client.calls) == 1 for client in pipeline_case["retrieval_clients"])
    assert all(len(client.calls) == 2 for client in pipeline_case["decision_clients"])
    expected_package_ids = {
        numerical_package_fingerprint(pipeline_case["registry"].package_for(task))
        for task in tasks
    }
    assert {row.numerical_package_sha256 for row in result.task_rows} == expected_package_ids
    assert result.secondary_diagnostics["cooperative_stage_train"] == 1.0


def test_pipeline_stage_marker_binds_train_and_dev_evaluation_fingerprints(
    pipeline_case,
):
    train = pipeline_case["pipeline"].evaluate(
        pipeline_case["bundle"], pipeline_case["tasks"], stage="train"
    )
    dev = pipeline_case["pipeline"].evaluate(
        pipeline_case["bundle"], pipeline_case["tasks"], stage="dev"
    )

    assert train.task_rows == dev.task_rows
    assert train.fingerprint != dev.fingerprint
    assert train.secondary_diagnostics["cooperative_stage_train"] == 1.0
    assert dev.secondary_diagnostics["cooperative_stage_dev"] == 1.0


@pytest.mark.parametrize("stage", ("public", "test", "TRAIN", ""))
def test_pipeline_rejects_every_stage_except_exact_train_or_dev(
    pipeline_case, stage
):
    with pytest.raises(ValueError, match="train or dev"):
        pipeline_case["pipeline"].evaluate(
            pipeline_case["bundle"], pipeline_case["tasks"], stage=stage
        )


def test_pipeline_rejects_a_factory_that_changes_the_bound_retrieval_module(
    pipeline_case, tmp_path
):
    changed = replace(RetrievalGenome.seed(), round1_strategy="entity_first")
    pipeline = CooperativePipelineAdapter(
        pipeline_case["catalog"],
        lambda _genome, skills: TwoStageRetrievalAgent(
            FakeLLMClient([_round1_response()]), changed, skills
        ),
        lambda module: DecisionAgent(
            FakeLLMClient([_decision_response(), _decision_response()]),
            prompt=module.prompt,
        ),
        empty_skill_path=tmp_path / "identity-skills.json",
    )

    with pytest.raises(ValueError, match="bound Retrieval Genome"):
        pipeline.evaluate(
            pipeline_case["bundle"], pipeline_case["tasks"][:1], stage="train"
        )


def test_pipeline_rejects_a_factory_that_changes_the_bound_decision_prompt(
    pipeline_case, tmp_path
):
    pipeline = CooperativePipelineAdapter(
        pipeline_case["catalog"],
        lambda genome, skills: TwoStageRetrievalAgent(
            FakeLLMClient([_round1_response()]), genome, skills
        ),
        lambda _module: DecisionAgent(
            FakeLLMClient([_decision_response(), _decision_response()]),
            prompt="unbound prompt",
        ),
        empty_skill_path=tmp_path / "decision-identity-skills.json",
    )

    with pytest.raises(ValueError, match="bound Decision prompt"):
        pipeline.evaluate(
            pipeline_case["bundle"], pipeline_case["tasks"][:1], stage="train"
        )


def test_pipeline_requires_a_verified_runtime_library_for_nonempty_skills(
    pipeline_case, tmp_path
):
    retrieval = replace(
        pipeline_case["retrieval"],
        skills_payload=({"skill_id": "inactive-but-material"},),
    )
    retrieval_sha = pipeline_case["catalog"].add_retrieval(retrieval)
    bundle = replace(
        pipeline_case["bundle"], retrieval_release_sha256=retrieval_sha
    )

    with pytest.raises(ValueError, match="verified Retrieval Skill library"):
        pipeline_case["pipeline"].evaluate(bundle, pipeline_case["tasks"][:1], stage="train")

    empty_library = RetrievalSkillLibrary(
        tmp_path / "unverified.json", persist=False
    ).clone(read_only=True)
    with pytest.raises(ValueError, match="does not match"):
        CooperativePipelineAdapter(
            pipeline_case["catalog"],
            lambda genome, skills: TwoStageRetrievalAgent(
                FakeLLMClient([_round1_response()]), genome, skills
            ),
            lambda module: DecisionAgent(
                FakeLLMClient([_decision_response(), _decision_response()]),
                prompt=module.prompt,
            ),
            retrieval_skill_library=empty_library,
        ).evaluate(bundle, pipeline_case["tasks"][:1], stage="train")


def test_train_feedback_contains_only_aggregate_train_values():
    parent = _marked(
        _evaluation("1" * 64, ("private-task-a", "private-task-b"), 2.0),
        "train",
    )
    child = _marked(
        _evaluation("2" * 64, ("private-task-a", "private-task-b"), 1.5),
        "train",
    )

    feedback = sanitize_train_feedback(parent, child, normalized_cost=0.25)
    wire = canonical_v2_bytes(feedback.to_payload()).decode("utf-8").lower()

    assert feedback.train_objectives == {
        "normalized_cost": 0.25,
        "relative_joint_improvement": 0.25,
    }
    assert feedback.train_behavior_descriptors == {
        "catastrophic_count": 0,
        "fallback_count": 0,
        "invalid_count": 0,
    }
    assert feedback.train_evaluation_sha256 == child.fingerprint
    assert "private-task" not in wire
    assert "dev" not in wire
    assert "future" not in wire
    assert "forecast" not in wire


def test_train_feedback_rejects_dev_marked_evaluations():
    parent = _marked(_evaluation("1" * 64, ("task",), 2.0), "train")
    child = _marked(_evaluation("2" * 64, ("task",), 1.0), "dev")

    with pytest.raises(ValueError, match="Train-marked"):
        sanitize_train_feedback(parent, child, normalized_cost=0.25)


@pytest.mark.parametrize("normalized_cost", [-0.1, float("inf"), True])
def test_train_feedback_rejects_invalid_normalized_cost(normalized_cost):
    parent = _marked(_evaluation("1" * 64, ("task",), 2.0), "train")
    child = _marked(_evaluation("2" * 64, ("task",), 1.0), "train")

    with pytest.raises(ValueError, match="normalized_cost"):
        sanitize_train_feedback(parent, child, normalized_cost=normalized_cost)
