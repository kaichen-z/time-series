from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evolving_loop.co_evolution import (
    CoEvolutionEngine,
    HarnessPolicy,
    PolicyEvaluation,
    evaluation_diagnostics,
    snapshot_policy_skills,
)
from evolving_loop.coding_agent.skill_library import Skill, SkillLibrary
from evolving_loop.decision_agent.skill_library import DecisionSkill, DecisionSkillLibrary
from evolving_loop.evaluation import ResolvedOutcome
from evolving_loop.retrieval_agent.skill_library import RetrievalSkill, RetrievalSkillLibrary
from common.llm import FakeLLMClient, TransientLLMError


def _open_genome(**overrides) -> str:
    payload = {
        "coding_generation_prompt": "Generate diverse executable forecasters.",
        "coding_revision_prompt": "Rewrite failed numerical frameworks.",
        "retrieval_prompt": "Retrieve discriminating exact evidence.",
        "decision_prompt": "Select the best grounded candidate.",
        "coding_initial_programs": 5,
        "coding_mutations": 2,
        "coding_mutation_children": 3,
        "coding_validation_folds": 4,
        "coding_validation_horizon": 12,
        "workflow": ["retrieve", "retrieve", "decide", "decide"],
        "enable_evidence_adjustments": False,
        "max_evidence_adjustments": 2,
        "decision_aggregation": "majority",
        "changelog": "Co-evolve search, topology, and all role instructions.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_mutation_can_redesign_the_complete_harness_genome() -> None:
    evaluation = PolicyEvaluation(
        version="v000",
        system_reward=0.2,
        module_rewards={"coding": 0.8, "retrieval": 0.2, "decision": 0.7},
        outcomes=(),
    )
    llm = FakeLLMClient(
        [
            _open_genome()
        ]
    )
    engine = CoEvolutionEngine(llm, harness_factory=lambda _policy: None)
    child = engine.mutate(HarnessPolicy(), evaluation)

    assert child.parent == "v000"
    assert child.retrieval_prompt.startswith("Retrieve discriminating")
    assert child.coding_generation_prompt.startswith("Generate diverse")
    assert child.workflow == ("retrieve", "retrieve", "decide", "decide")
    assert child.coding_initial_programs == 5
    assert child.coding_mutations == 2
    assert child.coding_mutation_children == 3
    assert child.decision_aggregation == "majority"
    assert child.enable_evidence_adjustments is False


def test_multiple_children_receive_distinct_payloads_without_full_skill_source() -> None:
    evaluation = PolicyEvaluation(
        version="v000",
        system_reward=0.2,
        module_rewards={"coding": 0.8, "retrieval": 0.2, "decision": 0.7},
        outcomes=(),
    )
    client = FakeLLMClient([_open_genome(), _open_genome()])
    engine = CoEvolutionEngine(client, harness_factory=lambda _policy: None)
    parent = HarnessPolicy(
        coding_skills=(
            {
                "skill_id": "c1",
                "name": "private_code_skill",
                "description": "A validated numeric skill.",
                "code": "FULL_EXECUTABLE_SOURCE_MUST_NOT_ENTER_MUTATION_PROMPT",
                "created_from_task": "task_train",
            },
        )
    )

    engine.mutate(parent, evaluation, child_index=0)
    engine.mutate(parent, evaluation, child_index=1)

    first = json.loads(client.calls[0]["messages"][0]["content"])
    second = json.loads(client.calls[1]["messages"][0]["content"])
    assert first["child_index"] == 0
    assert second["child_index"] == 1
    assert first["diversity_instruction"] != second["diversity_instruction"]
    assert first["skill_inventory"]["coding"] == ["private_code_skill"]
    assert "FULL_EXECUTABLE_SOURCE" not in client.calls[0]["messages"][0]["content"]


def test_prompt_mode_changes_only_one_prompt() -> None:
    evaluation = PolicyEvaluation(
        version="v000",
        system_reward=0.2,
        module_rewards={"coding": 0.8, "retrieval": 0.2, "decision": 0.7},
        outcomes=(),
    )
    client = FakeLLMClient(
        [
            json.dumps(
                {
                    "prompt_field": "retrieval_prompt",
                    "replacement_prompt": "Retrieve contrastive exact evidence.",
                    "changelog": "Improve retrieval precision.",
                }
            )
        ]
    )
    from evolving_loop.co_evolution import CoEvolutionConfig

    child = CoEvolutionEngine(
        client,
        harness_factory=lambda _policy: None,
        config=CoEvolutionConfig(mode="prompt"),
    ).mutate(HarnessPolicy(), evaluation)
    assert child.retrieval_prompt == "Retrieve contrastive exact evidence."
    assert child.workflow == HarnessPolicy().workflow
    assert child.coding_initial_programs == HarnessPolicy().coding_initial_programs


def test_coding_target_overrides_weakest_role_in_prompt_mode() -> None:
    from evolving_loop.co_evolution import CoEvolutionConfig

    evaluation = PolicyEvaluation(
        version="v000",
        system_reward=0.2,
        module_rewards={"coding": 0.8, "retrieval": 0.1, "decision": 0.7},
        outcomes=(),
    )
    client = FakeLLMClient(
        [
            json.dumps(
                {
                    "prompt_field": "coding_generation_prompt",
                    "replacement_prompt": "Generate falsifiable numeric programs only.",
                    "changelog": "Improve Coding candidate coverage.",
                }
            )
        ]
    )
    child = CoEvolutionEngine(
        client,
        harness_factory=lambda _policy: None,
        config=CoEvolutionConfig(mode="prompt", target="coding"),
    ).mutate(HarnessPolicy(), evaluation)

    assert child.coding_generation_prompt == "Generate falsifiable numeric programs only."
    assert child.retrieval_prompt == HarnessPolicy().retrieval_prompt
    assert child.decision_prompt == HarnessPolicy().decision_prompt
    assert CoEvolutionEngine(
        FakeLLMClient([]),
        harness_factory=lambda _policy: None,
        config=CoEvolutionConfig(mode="prompt", target="coding"),
    ).target_agent(evaluation) == "coding"
    call = client.calls[0]
    payload = json.loads(call["messages"][0]["content"])
    assert "Coding" in payload["instruction"]
    assert "must not mutate Retrieval or Decision" in payload["instruction"]
    assert "diagnosed weakest role" not in call["system"]


def test_coding_target_genome_preserves_other_agents_and_workflow() -> None:
    from evolving_loop.co_evolution import CoEvolutionConfig

    parent = HarnessPolicy()
    evaluation = PolicyEvaluation(
        version=parent.version,
        system_reward=0.2,
        module_rewards={"coding": 0.8, "retrieval": 0.1, "decision": 0.7},
        outcomes=(),
    )
    child = CoEvolutionEngine(
        FakeLLMClient([_open_genome()]),
        harness_factory=lambda _policy: None,
        config=CoEvolutionConfig(mode="genome", target="coding"),
    ).mutate(parent, evaluation)

    assert child.coding_generation_prompt == "Generate diverse executable forecasters."
    assert child.coding_initial_programs == 5
    assert child.retrieval_prompt == parent.retrieval_prompt
    assert child.decision_prompt == parent.decision_prompt
    assert child.workflow == parent.workflow
    assert child.decision_aggregation == parent.decision_aggregation
    assert child.enable_evidence_adjustments == parent.enable_evidence_adjustments
    assert child.max_evidence_adjustments == parent.max_evidence_adjustments


def test_illegal_unbounded_workflow_mutation_is_rejected() -> None:
    evaluation = PolicyEvaluation(
        version="v000",
        system_reward=0.2,
        module_rewards={"coding": 0.9, "retrieval": 0.1, "decision": 0.8},
        outcomes=(),
    )
    llm = FakeLLMClient(
        [
            _open_genome(workflow=["retrieve", "execute_arbitrary_shell", "decide"])
        ]
    )
    child = CoEvolutionEngine(llm, harness_factory=lambda _policy: None).mutate(
        HarnessPolicy(), evaluation
    )
    assert child.workflow == HarnessPolicy().workflow
    assert child.changelog == "Illegal workflow mutation; unchanged."


def test_accepted_genome_round_trips_for_later_evolution() -> None:
    policy = HarnessPolicy(
        version="v007",
        workflow=("retrieve", "retrieve", "decide", "decide"),
        coding_mutations=3,
        coding_mutation_children=2,
        decision_aggregation="majority",
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best_policy.json"
        policy.save(path)
        restored = HarnessPolicy.load(path)
    assert restored == policy


def test_complete_policy_round_trips_all_three_skill_snapshots(tmp_path) -> None:
    policy = HarnessPolicy(
        coding_skills=(
            {
                "skill_id": "coding-1",
                "name": "robust_tail",
                "description": "Forecast a stable robust tail.",
                "code": "def forecast(history, horizon, frequency): return [history[-1]] * horizon",
                "created_from_task": "task_train",
                "assumption": "The tail level persists.",
                "failure_condition": "A regime changes.",
                "validation_score": 1.0,
                "uses": 2,
                "avg_score": 0.9,
            },
        ),
        retrieval_skills=(
            {
                "skill_id": "retrieval-1",
                "name": "future_window_check",
                "description": "Check temporal overlap.",
                "applicability": "Scheduled events.",
                "query_strategy": "Search the forecast window.",
                "verification_rule": "Require an exact quote.",
                "created_from_task": "task_train",
                "validation_score": 0.8,
                "uses": 1,
                "avg_score": 0.8,
            },
        ),
        decision_skills=(
            {
                "skill_id": "decision-1",
                "name": "safe_override",
                "description": "Override only on falsification.",
                "applicability": "Competing candidates.",
                "decision_rule": "Keep the hindcast leader without counterevidence.",
                "failure_condition": "Verified future context falsifies it.",
                "created_from_task": "task_train",
                "validation_score": 0.95,
                "uses": 1,
                "avg_score": 0.95,
            },
        ),
    )
    path = tmp_path / "policy.json"
    policy.save(path)

    assert HarnessPolicy.load(path) == policy


def test_snapshot_policy_skills_captures_every_harness_library(tmp_path) -> None:
    coding = SkillLibrary(tmp_path / "coding.json", persist=False)
    coding.add(
        Skill(
            skill_id="c1",
            name="numeric_skill",
            description="numeric",
            code="def forecast(history, horizon, frequency): return [0.0] * horizon",
            created_from_task="task_train",
        )
    )
    retrieval = RetrievalSkillLibrary(tmp_path / "retrieval.json", persist=False)
    retrieval.add(
        RetrievalSkill(
            skill_id="r1",
            name="retrieval_skill",
            description="retrieval",
            applicability="always",
            query_strategy="search",
            verification_rule="quote",
            created_from_task="task_train",
            validation_score=0.8,
        )
    )
    decision = DecisionSkillLibrary(tmp_path / "decision.json", persist=False)
    decision.add(
        DecisionSkill(
            skill_id="d1",
            name="decision_skill",
            description="decision",
            applicability="always",
            decision_rule="select",
            failure_condition="none",
            created_from_task="task_train",
            validation_score=0.9,
        )
    )

    class Agent:
        def __init__(self, library):
            self.library = library

    class Harness:
        def __init__(self):
            self.coding = Agent(coding)
            self.retrieval = Agent(retrieval)
            self.decision = Agent(decision)

    snapshot = snapshot_policy_skills(HarnessPolicy(), Harness())

    assert snapshot.coding_skills[0]["name"] == "numeric_skill"
    assert snapshot.retrieval_skills[0]["name"] == "retrieval_skill"
    assert snapshot.decision_skills[0]["name"] == "decision_skill"


def test_evaluation_diagnostics_separate_generation_from_selection_failure() -> None:
    outcomes = (
        ResolvedOutcome(
            task_id="task_a",
            final_smae=0.6,
            final_srmse=0.8,
            coding_oracle_smae=0.2,
            coding_coverage_regret=0.2,
            retrieval_precision=0.5,
            supporting_recall=0.5,
            distractor_avoidance=1.0,
            decision_selection_regret=0.4,
            candidate_count=4,
            hindcast_future_rank_correlation=0.8,
        ),
        ResolvedOutcome(
            task_id="task_b",
            final_smae=1.0,
            final_srmse=1.2,
            coding_oracle_smae=0.4,
            coding_coverage_regret=0.4,
            retrieval_precision=1.0,
            supporting_recall=0.0,
            distractor_avoidance=1.0,
            decision_selection_regret=0.6,
            candidate_count=6,
            hindcast_future_rank_correlation=-0.2,
        ),
    )

    assert evaluation_diagnostics(outcomes) == {
        "mean_final_smae": 0.8,
        "mean_best_of_k_smae": 0.30000000000000004,
        "mean_selection_regret": 0.5,
        "mean_candidate_count": 5.0,
        "mean_hindcast_future_rank_correlation": 0.30000000000000004,
    }


def test_mutation_transport_failure_becomes_rejected_candidate() -> None:
    class BrokenClient:
        def complete(self, **_kwargs):
            raise RuntimeError("temporary model failure")

    evaluation = PolicyEvaluation(
        version="v000",
        system_reward=0.2,
        module_rewards={"coding": 0.1, "retrieval": 0.2, "decision": 0.3},
        outcomes=(),
    )
    child = CoEvolutionEngine(BrokenClient(), lambda _policy: None).mutate(
        HarnessPolicy(), evaluation
    )
    assert child.version == "v001"
    assert child.parent == "v000"
    assert "Mutation call failed" in child.changelog


def test_mutation_propagates_transient_infrastructure_failure() -> None:
    class OfflineClient:
        def complete(self, **_kwargs):
            raise TransientLLMError("Connection refused")

    evaluation = PolicyEvaluation(
        version="v000",
        system_reward=0.2,
        module_rewards={"coding": 0.1, "retrieval": 0.2, "decision": 0.3},
        outcomes=(),
    )
    engine = CoEvolutionEngine(OfflineClient(), lambda _policy: None)

    with pytest.raises(TransientLLMError, match="Connection refused"):
        engine.mutate(HarnessPolicy(), evaluation)


def test_checkpoint_round_trip_resumes_next_generation() -> None:
    from evolving_loop.co_evolution import CoEvolutionConfig, EvolutionStep

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.json"
        config = CoEvolutionConfig(
            mode="prompt", checkpoint_path=checkpoint, progress_path=Path(directory) / "progress.jsonl"
        )
        engine = CoEvolutionEngine(FakeLLMClient([]), lambda _policy: None, config)
        policy = HarnessPolicy(version="v004", parent="v003")
        step = EvolutionStep(
            mode="prompt",
            generation=0,
            parent_version="v000",
            child_versions=("v004",),
            target_agent="coding",
            parent_train_reward=0.1,
            child_train_rewards={"v004": 0.2},
            parent_dev_reward=0.1,
            best_child_dev_reward=0.2,
            accepted_version="v004",
            parent_train_module_rewards={"coding": 0.1, "retrieval": 0.2, "decision": 0.3},
            parent_dev_module_rewards={"coding": 0.2, "retrieval": 0.3, "decision": 0.4},
            best_child_train_module_rewards={"coding": 0.4, "retrieval": 0.2, "decision": 0.3},
            best_child_dev_module_rewards={"coding": 0.5, "retrieval": 0.3, "decision": 0.4},
        )
        engine._save_checkpoint(policy, [step], 1)
        restored, history, start = engine._load_checkpoint(HarnessPolicy())

    assert restored == policy
    assert history == [step]
    assert start == 1
    assert engine._version == 5


def test_evaluation_progress_keeps_failure_traces_for_debugging(monkeypatch, tmp_path) -> None:
    from evolving_loop import co_evolution
    from evolving_loop.co_evolution import CoEvolutionConfig

    evaluation = PolicyEvaluation(
        version="v000",
        system_reward=0.5,
        module_rewards={"coding": 0.4, "retrieval": 0.5, "decision": 0.6},
        outcomes=(),
        failure_traces=({"task_id": "task_x", "oracle_candidate_id": "oracle"},),
        diagnostics={"mean_selection_regret": 4.0},
    )
    monkeypatch.setattr(co_evolution, "evaluate_policy", lambda *_args, **_kwargs: evaluation)
    progress = tmp_path / "progress.jsonl"
    engine = CoEvolutionEngine(
        FakeLLMClient([]),
        lambda _policy: None,
        CoEvolutionConfig(progress_path=progress),
    )

    engine._evaluate(
        HarnessPolicy(),
        (),
        stage="child_train",
        generation=0,
        learn_skills=False,
        harness=None,
    )

    completed = json.loads(progress.read_text().splitlines()[-1])
    assert completed["failure_traces"] == [
        {"task_id": "task_x", "oracle_candidate_id": "oracle"}
    ]
    assert completed["diagnostics"] == {"mean_selection_regret": 4.0}


def test_child_must_improve_train_before_dev_is_spent(monkeypatch) -> None:
    from evolving_loop.co_evolution import CoEvolutionConfig

    engine = CoEvolutionEngine(
        FakeLLMClient([]),
        lambda _policy: object(),
        CoEvolutionConfig(generations=1, children_per_generation=1),
    )
    child = HarnessPolicy(version="v001", parent="v000")
    monkeypatch.setattr(
        engine,
        "mutate",
        lambda _parent, _evaluation, **_kwargs: child,
    )
    calls = []

    def fake_evaluate(policy, _tasks, *, stage, **_kwargs):
        calls.append(stage)
        if stage == "parent_train":
            reward = 0.8
        elif stage == "parent_val":
            reward = 0.7
        elif stage == "child_train":
            reward = 0.6
        else:
            raise AssertionError("a train-regressing child must not reach dev")
        return PolicyEvaluation(
            version=policy.version,
            system_reward=reward,
            module_rewards={"coding": reward, "retrieval": reward, "decision": reward},
            outcomes=(),
        )

    monkeypatch.setattr(engine, "_evaluate", fake_evaluate)
    best, trace = engine.evolve(HarnessPolicy(), (object(),), (object(),))

    assert best.version == "v000"
    assert calls == ["parent_train", "parent_val", "child_train"]
    assert trace[0].best_child_dev_reward is None


def test_successive_halving_prunes_screen_failures_and_only_fully_evaluates_top_child(
    monkeypatch,
) -> None:
    from evolving_loop.co_evolution import CoEvolutionConfig

    engine = CoEvolutionEngine(
        FakeLLMClient([]),
        lambda _policy: object(),
        CoEvolutionConfig(
            generations=1,
            children_per_generation=3,
            successive_halving=True,
            screening_train_tasks=2,
            screening_dev_tasks=1,
            screening_promote=1,
            screening_tolerance=0.01,
        ),
    )
    monkeypatch.setattr(
        engine,
        "mutate",
        lambda _parent, _evaluation, *, child_index=0: HarnessPolicy(
            version=f"v{child_index + 1:03d}", parent="v000"
        ),
    )
    calls: list[tuple[str, str, int]] = []

    def fake_evaluate(policy, tasks, *, stage, **_kwargs):
        calls.append((policy.version, stage, len(tasks)))
        if policy.version == "v000":
            reward = {
                "parent_screen_train": 0.8,
                "parent_screen_dev": 0.7,
                "parent_train_remaining": 0.8,
                "parent_val": 0.7,
            }[stage]
        else:
            reward = {
                ("v001", "child_screen_train"): 0.8,
                ("v001", "child_screen_dev"): 0.60,
                ("v002", "child_screen_train"): 0.9,
                ("v002", "child_screen_dev"): 0.75,
                ("v002", "child_train_remaining"): 0.9,
                ("v002", "child_val"): 0.8,
                ("v003", "child_screen_train"): 0.85,
                ("v003", "child_screen_dev"): 0.72,
            }[(policy.version, stage)]
        return PolicyEvaluation(
            version=policy.version,
            system_reward=reward,
            module_rewards={"coding": reward, "retrieval": reward, "decision": reward},
            outcomes=(),
            diagnostics={"mean_final_smae": 1.0 - reward},
        )

    monkeypatch.setattr(engine, "_evaluate", fake_evaluate)
    best, trace = engine.evolve(
        HarnessPolicy(),
        (object(), object(), object(), object()),
        (object(), object()),
    )

    assert best.version == "v002"
    assert ("v001", "child_train_remaining", 2) not in calls
    assert ("v003", "child_train_remaining", 2) not in calls
    assert ("v002", "child_train_remaining", 2) in calls
    assert ("v002", "child_val", 2) in calls
    assert trace[0].successive_halving is True
    assert trace[0].parent_screen_dev_reward == pytest.approx(0.7)
    assert trace[0].child_screen_dev_rewards == {
        "v001": pytest.approx(0.60),
        "v002": pytest.approx(0.75),
        "v003": pytest.approx(0.72),
    }
    assert trace[0].promoted_versions == ("v002",)
    assert trace[0].screen_prune_reasons == {
        "v001": "below_parent_tolerance",
        "v003": "not_top_k",
    }


def test_successive_halving_keeps_parent_when_every_child_fails_screen(
    monkeypatch,
) -> None:
    from evolving_loop.co_evolution import CoEvolutionConfig

    engine = CoEvolutionEngine(
        FakeLLMClient([]),
        lambda _policy: object(),
        CoEvolutionConfig(
            generations=1,
            children_per_generation=2,
            successive_halving=True,
            screening_train_tasks=1,
            screening_dev_tasks=1,
            screening_promote=1,
            screening_tolerance=0.01,
        ),
    )
    monkeypatch.setattr(
        engine,
        "mutate",
        lambda _parent, _evaluation, *, child_index=0: HarnessPolicy(
            version=f"v{child_index + 1:03d}", parent="v000"
        ),
    )
    calls: list[str] = []

    def fake_evaluate(policy, _tasks, *, stage, **_kwargs):
        calls.append(stage)
        reward = 0.8 if policy.version == "v000" else 0.5
        return PolicyEvaluation(
            version=policy.version,
            system_reward=reward,
            module_rewards={"coding": reward, "retrieval": reward, "decision": reward},
            outcomes=(),
        )

    monkeypatch.setattr(engine, "_evaluate", fake_evaluate)
    best, trace = engine.evolve(
        HarnessPolicy(), (object(), object()), (object(), object())
    )

    assert best.version == "v000"
    assert "child_train_remaining" not in calls
    assert "child_val" not in calls
    assert trace[0].promoted_versions == ()
    assert trace[0].screen_prune_reasons == {
        "v001": "below_parent_tolerance",
        "v002": "below_parent_tolerance",
    }


def test_successive_halving_pauses_on_infrastructure_failure_instead_of_failing_candidate(
    monkeypatch, tmp_path
) -> None:
    from evolving_loop import co_evolution
    from evolving_loop.co_evolution import CoEvolutionConfig

    progress = tmp_path / "progress.jsonl"
    engine = CoEvolutionEngine(
        FakeLLMClient([]),
        lambda _policy: object(),
        CoEvolutionConfig(
            generations=1,
            children_per_generation=1,
            successive_halving=True,
            screening_train_tasks=1,
            screening_dev_tasks=1,
            progress_path=progress,
        ),
    )
    child = HarnessPolicy(version="v001", parent="v000")
    monkeypatch.setattr(
        engine,
        "mutate",
        lambda _parent, _evaluation, **_kwargs: child,
    )

    def fake_evaluate_policy(policy, _tasks, _factory, **_kwargs):
        if policy.version == "v001":
            raise TransientLLMError("Connection reset by peer")
        return PolicyEvaluation(
            version=policy.version,
            system_reward=0.7,
            module_rewards={"coding": 0.7, "retrieval": 0.7, "decision": 0.7},
            outcomes=(),
        )

    monkeypatch.setattr(co_evolution, "evaluate_policy", fake_evaluate_policy)

    with pytest.raises(TransientLLMError, match="Connection reset"):
        engine.evolve(
            HarnessPolicy(),
            (object(), object()),
            (object(), object()),
        )

    events = [json.loads(line) for line in progress.read_text().splitlines()]
    event_names = [event["event"] for event in events]
    assert "infrastructure_interrupted" in event_names
    assert "candidate_failed" not in event_names


def test_accepted_child_contains_skills_learned_during_train(monkeypatch, tmp_path) -> None:
    from evolving_loop.co_evolution import CoEvolutionConfig

    class Agent:
        def __init__(self, library):
            self.library = library

    class Harness:
        def __init__(self, version):
            self.version = version
            self.coding = Agent(SkillLibrary(tmp_path / f"{version}-coding.json", persist=False))
            self.retrieval = Agent(
                RetrievalSkillLibrary(tmp_path / f"{version}-retrieval.json", persist=False)
            )
            self.decision = Agent(
                DecisionSkillLibrary(tmp_path / f"{version}-decision.json", persist=False)
            )

    engine = CoEvolutionEngine(
        FakeLLMClient([]),
        lambda policy: Harness(policy.version),
        CoEvolutionConfig(generations=1, children_per_generation=1),
    )
    child = HarnessPolicy(version="v001", parent="v000")
    monkeypatch.setattr(
        engine,
        "mutate",
        lambda _parent, _evaluation, **_kwargs: child,
    )

    def fake_evaluate(policy, _tasks, *, stage, harness, **_kwargs):
        rewards = {
            "parent_train": 0.5,
            "parent_val": 0.5,
            "child_train": 0.7,
            "child_val": 0.7,
        }
        if stage.endswith("train"):
            harness.coding.library.add(
                Skill(
                    skill_id=stage,
                    name=f"{stage}_skill",
                    description=stage,
                    code="def forecast(history, horizon, frequency): return [0.0] * horizon",
                    created_from_task="task_train",
                )
            )
        reward = rewards[stage]
        return PolicyEvaluation(
            version=policy.version,
            system_reward=reward,
            module_rewards={"coding": reward, "retrieval": reward, "decision": reward},
            outcomes=(),
        )

    monkeypatch.setattr(engine, "_evaluate", fake_evaluate)
    best, _trace = engine.evolve(HarnessPolicy(), (object(),), (object(),))

    assert best.version == "v001"
    assert [record["name"] for record in best.coding_skills] == ["child_train_skill"]


def test_checkpoint_rejects_a_different_evolution_target() -> None:
    from evolving_loop.co_evolution import CoEvolutionConfig

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.json"
        writer = CoEvolutionEngine(
            FakeLLMClient([]),
            lambda _policy: None,
            CoEvolutionConfig(mode="prompt", target="coding", checkpoint_path=checkpoint),
        )
        writer._save_checkpoint(HarnessPolicy(), [], 1)
        reader = CoEvolutionEngine(
            FakeLLMClient([]),
            lambda _policy: None,
            CoEvolutionConfig(mode="prompt", target="retrieval", checkpoint_path=checkpoint),
        )
        with pytest.raises(ValueError, match="target"):
            reader._load_checkpoint(HarnessPolicy())


def test_checkpoint_rejects_different_successive_halving_controls() -> None:
    from evolving_loop.co_evolution import CoEvolutionConfig

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.json"
        writer = CoEvolutionEngine(
            FakeLLMClient([]),
            lambda _policy: None,
            CoEvolutionConfig(
                checkpoint_path=checkpoint,
                successive_halving=True,
                screening_train_tasks=6,
                screening_dev_tasks=2,
                screening_promote=1,
                screening_tolerance=0.01,
            ),
        )
        writer._save_checkpoint(HarnessPolicy(), [], 1)
        reader = CoEvolutionEngine(
            FakeLLMClient([]),
            lambda _policy: None,
            CoEvolutionConfig(
                checkpoint_path=checkpoint,
                successive_halving=True,
                screening_train_tasks=4,
                screening_dev_tasks=2,
                screening_promote=1,
                screening_tolerance=0.01,
            ),
        )
        with pytest.raises(ValueError, match="successive-halving"):
            reader._load_checkpoint(HarnessPolicy())
