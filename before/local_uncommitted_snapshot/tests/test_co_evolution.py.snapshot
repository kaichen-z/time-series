from __future__ import annotations

from dataclasses import replace

from drcik_agent.co_evolution import (
    AgentPromptBundle,
    BundleEvaluation,
    CoEvolutionConfig,
    PromptCoEvolutionEngine,
    TaskEvolutionOutcome,
)


def _outcome(task_id: str, score: float) -> TaskEvolutionOutcome:
    return TaskEvolutionOutcome(
        benchmark_id=task_id,
        system_score=score,
        coding_reward=-0.5,
        retrieval_reward=0.4,
        decision_reward=-0.1,
        trace={"failure": "candidate set missed the future"},
    )


def _evaluation(bundle: AgentPromptBundle, _tasks) -> BundleEvaluation:
    outcome = _outcome("task_1", 1.0 if bundle.version != "v000" else 0.0)
    return BundleEvaluation(
        bundle_version=bundle.version,
        mean_score=outcome.system_score,
        module_rewards={"coding": -0.5, "retrieval": 0.4, "decision": -0.1},
        outcomes=(outcome,),
        worst=(outcome,),
    )


class FakeMutationClient:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, stage, prompt, schema, workspace_files=None):
        self.calls.append((stage, prompt, schema, workspace_files))
        return {
            "prompt_field": "coding_program_prompt",
            "replacement_prompt": "NUMBERS ONLY: generate robust assumptions and executable code.",
            "changelog": "Require a robust candidate after candidate-coverage failures.",
        }


def test_failure_attribution_mutates_only_the_weakest_agent() -> None:
    client = FakeMutationClient()
    engine = PromptCoEvolutionEngine(client, _evaluation)
    seed = AgentPromptBundle()
    child = engine.mutate(seed, _evaluation(seed, ()))

    assert child.parent == "v000"
    assert child.coding_program_prompt.startswith("NUMBERS ONLY")
    assert child.retrieval_prompt == seed.retrieval_prompt
    assert child.decision_prompt == seed.decision_prompt
    payload = client.calls[0][3]["evolution.json"]
    assert '"target_agent": "coding"' in payload


def test_population_evolution_selects_a_dev_validated_child() -> None:
    engine = PromptCoEvolutionEngine(
        FakeMutationClient(),
        _evaluation,
        CoEvolutionConfig(generations=2, population_size=3, keep_elite=1),
    )
    best, records = engine.evolve(AgentPromptBundle(), (object(),), (object(),))

    assert best.version != "v000"
    assert len(records) == 2
    assert records[0].mutated_agent == "coding"


def test_illegal_cross_agent_mutation_is_rejected() -> None:
    class BadClient(FakeMutationClient):
        def complete(self, *args, **kwargs):
            return {
                "prompt_field": "decision_prompt",
                "replacement_prompt": "Hijack another role.",
                "changelog": "bad",
            }

    seed = AgentPromptBundle()
    child = PromptCoEvolutionEngine(BadClient(), _evaluation).mutate(
        seed, _evaluation(seed, ())
    )
    assert child.decision_prompt == seed.decision_prompt
    assert "Illegal mutation" in child.notes


def test_bundle_round_trip(tmp_path) -> None:
    path = tmp_path / "bundle.json"
    expected = replace(AgentPromptBundle(), version="v012", notes="tested")
    expected.save(path)
    assert AgentPromptBundle.load(path) == expected
