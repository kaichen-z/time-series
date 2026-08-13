from __future__ import annotations

import json

from evolving_agent.co_evolution import CoEvolutionEngine, HarnessPolicy, PolicyEvaluation
from evolving_agent.llm import FakeLLMClient


def test_mutation_targets_only_weakest_agent_prompt() -> None:
    evaluation = PolicyEvaluation(
        version="v000",
        system_reward=0.2,
        module_rewards={"coding": 0.8, "retrieval": 0.2, "decision": 0.7},
        outcomes=(),
    )
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "prompt_field": "retrieval_prompt",
                    "replacement_prompt": "Retrieve evidence that falsifies candidate assumptions.",
                    "changelog": "Improve hypothesis-discriminating retrieval.",
                }
            )
        ]
    )
    engine = CoEvolutionEngine(llm, harness_factory=lambda _policy: None)
    child = engine.mutate(HarnessPolicy(), evaluation)

    assert child.parent == "v000"
    assert child.retrieval_prompt.startswith("Retrieve evidence")
    assert child.coding_generation_prompt == HarnessPolicy().coding_generation_prompt
    assert child.decision_prompt == HarnessPolicy().decision_prompt


def test_illegal_cross_agent_mutation_is_rejected() -> None:
    evaluation = PolicyEvaluation(
        version="v000",
        system_reward=0.2,
        module_rewards={"coding": 0.9, "retrieval": 0.1, "decision": 0.8},
        outcomes=(),
    )
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "prompt_field": "decision_prompt",
                    "replacement_prompt": "Leak labels.",
                    "changelog": "Wrong module.",
                }
            )
        ]
    )
    child = CoEvolutionEngine(llm, harness_factory=lambda _policy: None).mutate(
        HarnessPolicy(), evaluation
    )
    assert child.decision_prompt == HarnessPolicy().decision_prompt
    assert child.changelog == "Illegal mutation; unchanged."
