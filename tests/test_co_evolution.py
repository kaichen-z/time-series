from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evolving_agent.co_evolution import CoEvolutionEngine, HarnessPolicy, PolicyEvaluation
from evolving_agent.llm import FakeLLMClient


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
