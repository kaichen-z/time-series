"""Delayed-outcome, failure-attributed evolution of the three-agent harness."""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from evolving_agent.coding_agent.evolution import GENERATION_PROMPT, REVISION_PROMPT
from evolving_agent.data import ContextTask
from evolving_agent.decision_agent.agent import DECISION_PROMPT
from evolving_agent.harness import EvolvingForecastHarness, ResolvedOutcome
from evolving_agent.llm import JsonExtractionError, LLMClient, parse_json_object
from evolving_agent.retrieval_agent.agent import RETRIEVAL_PROMPT


@dataclass(frozen=True)
class HarnessPolicy:
    """Versioned prompts; a child may replace exactly one field."""

    version: str = "v000"
    parent: str | None = None
    coding_generation_prompt: str = GENERATION_PROMPT
    coding_revision_prompt: str = REVISION_PROMPT
    retrieval_prompt: str = RETRIEVAL_PROMPT
    decision_prompt: str = DECISION_PROMPT
    changelog: str = "Hand-written seed policy."

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "HarnessPolicy":
        source = Path(path)
        return cls(**json.loads(source.read_text())) if source.exists() else cls()


@dataclass(frozen=True)
class PolicyEvaluation:
    version: str
    system_reward: float
    module_rewards: dict[str, float]
    outcomes: tuple[ResolvedOutcome, ...]
    failure_traces: tuple[dict, ...] = ()


@dataclass(frozen=True)
class EvolutionStep:
    generation: int
    parent_version: str
    child_versions: tuple[str, ...]
    target_agent: str
    parent_train_reward: float
    child_train_rewards: dict[str, float]
    parent_dev_reward: float | None
    best_child_dev_reward: float | None
    accepted_version: str


@dataclass(frozen=True)
class CoEvolutionConfig:
    generations: int = 3
    children_per_generation: int = 2


HarnessFactory = Callable[[HarnessPolicy], EvolvingForecastHarness]


def evaluate_policy(
    policy: HarnessPolicy,
    tasks: Sequence[ContextTask],
    harness_factory: HarnessFactory,
) -> PolicyEvaluation:
    """Run label-free inference first, then expose resolved labels only to scoring."""
    if not tasks:
        raise ValueError("policy evaluation needs at least one resolved task")
    harness = harness_factory(policy)
    outcomes = []
    traces = []
    for task in tasks:
        inference = harness.run(task)
        outcome = harness.score_after_resolution(task, inference)
        outcomes.append(outcome)
        candidate_scores = {
            candidate.candidate_id: harness.score_after_resolution(
                task,
                replace(
                    inference,
                    decision=replace(inference.decision, selected=candidate),
                    forecast=candidate.forecast,
                ),
            ).final_smape
            for candidate in inference.candidates
        }
        oracle_id = min(candidate_scores, key=candidate_scores.get)
        supporting_ids = [
            document.document_id for document in task.documents if document.role == "supporting"
        ]
        distractor_ids = [
            document.document_id for document in task.documents if document.role == "distractor"
        ]
        traces.append(
            {
                "task_id": task.numeric.task_id,
                "final_smape": outcome.final_smape,
                "coding_candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "assumption": candidate.assumption,
                        "hindcast_smape": candidate.hindcast_smape,
                        "resolved_smape": candidate_scores[candidate.candidate_id],
                    }
                    for candidate in inference.candidates
                ],
                "oracle_candidate_id": oracle_id,
                "selected_candidate_id": inference.decision.selected.candidate_id,
                "decision_rejection_reason": inference.decision.rejection_reason,
                "retrieved_document_ids": list(inference.retrieval.selected_document_ids),
                "supporting_document_ids": supporting_ids,
                "distractor_document_ids": distractor_ids,
                "retrieval_rejections": list(inference.retrieval.rejected),
            }
        )
    coding = statistics.fmean(1.0 - min(item.coding_coverage_regret, 200.0) / 200.0 for item in outcomes)
    retrieval = statistics.fmean(
        statistics.fmean(
            (item.retrieval_precision, item.supporting_recall, item.distractor_avoidance)
        )
        for item in outcomes
    )
    decision = statistics.fmean(
        1.0 - min(max(item.decision_selection_regret, 0.0), 200.0) / 200.0
        for item in outcomes
    )
    forecast = statistics.fmean(1.0 - min(item.final_smape, 200.0) / 200.0 for item in outcomes)
    return PolicyEvaluation(
        version=policy.version,
        system_reward=0.8 * forecast + 0.2 * retrieval,
        module_rewards={"coding": coding, "retrieval": retrieval, "decision": decision},
        outcomes=tuple(outcomes),
        failure_traces=tuple(traces),
    )


class CoEvolutionEngine:
    """Evolve one failure-attributed prompt at a time with train/dev elitism."""

    _FIELDS = {
        "coding": {"coding_generation_prompt", "coding_revision_prompt"},
        "retrieval": {"retrieval_prompt"},
        "decision": {"decision_prompt"},
    }

    def __init__(
        self,
        llm: LLMClient,
        harness_factory: HarnessFactory,
        config: CoEvolutionConfig | None = None,
    ) -> None:
        self.llm = llm
        self.harness_factory = harness_factory
        self.config = config or CoEvolutionConfig()
        self._version = 1

    @staticmethod
    def weakest_agent(evaluation: PolicyEvaluation) -> str:
        return min(evaluation.module_rewards, key=evaluation.module_rewards.get)

    def mutate(self, parent: HarnessPolicy, evaluation: PolicyEvaluation) -> HarnessPolicy:
        target = self.weakest_agent(evaluation)
        worst = sorted(
            evaluation.failure_traces,
            key=lambda item: item["final_smape"],
            reverse=True,
        )[:5]
        payload = {
            "target_agent": target,
            "module_rewards": evaluation.module_rewards,
            "worst_failure_trajectories": worst,
            "current_policy": asdict(parent),
            "constraints": {
                "change_exactly_one_prompt": True,
                "coding_sees": "historical numbers and historical hindcast diagnostics only",
                "retrieval_sees": "documents and coding assumptions, never labels",
                "decision_sees": "executed candidates and verified evidence, never labels",
            },
        }
        response = self.llm.complete(
            system=(
                "You are the Harness Evolver. Diagnose the resolved failure metrics and replace "
                f"exactly one full prompt belonging to the {target} agent. Preserve all role "
                "information boundaries. Return JSON with prompt_field, replacement_prompt, "
                "and changelog. Do not patch source code or expose future labels at inference."
            ),
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.4,
        )
        version = f"v{self._version:03d}"
        self._version += 1
        try:
            proposal = parse_json_object(response.text)
        except JsonExtractionError:
            return replace(parent, version=version, parent=parent.version, changelog="Invalid mutation; unchanged.")
        field = str(proposal.get("prompt_field", ""))
        replacement = str(proposal.get("replacement_prompt", "")).strip()
        if field not in self._FIELDS[target] or not replacement:
            return replace(parent, version=version, parent=parent.version, changelog="Illegal mutation; unchanged.")
        return replace(
            parent,
            version=version,
            parent=parent.version,
            changelog=str(proposal.get("changelog", ""))[:500],
            **{field: replacement},
        )

    def evolve(
        self,
        seed: HarnessPolicy,
        train_tasks: Sequence[ContextTask],
        dev_tasks: Sequence[ContextTask],
    ) -> tuple[HarnessPolicy, tuple[EvolutionStep, ...]]:
        if not dev_tasks:
            raise ValueError("a held-out dev split is required to accept harness mutations")
        incumbent = seed
        history = []
        for generation in range(self.config.generations):
            parent_train = evaluate_policy(incumbent, train_tasks, self.harness_factory)
            parent_dev = evaluate_policy(incumbent, dev_tasks, self.harness_factory)
            children = [
                self.mutate(incumbent, parent_train)
                for _ in range(self.config.children_per_generation)
            ]
            train_evaluations = {
                child.version: evaluate_policy(child, train_tasks, self.harness_factory)
                for child in children
            }
            train_best = max(children, key=lambda child: train_evaluations[child.version].system_reward)
            child_dev = evaluate_policy(train_best, dev_tasks, self.harness_factory)
            accepted = (
                train_best
                if child_dev.system_reward > parent_dev.system_reward
                else incumbent
            )
            history.append(
                EvolutionStep(
                    generation=generation,
                    parent_version=incumbent.version,
                    child_versions=tuple(child.version for child in children),
                    target_agent=self.weakest_agent(parent_train),
                    parent_train_reward=parent_train.system_reward,
                    child_train_rewards={
                        version: item.system_reward for version, item in train_evaluations.items()
                    },
                    parent_dev_reward=parent_dev.system_reward,
                    best_child_dev_reward=child_dev.system_reward,
                    accepted_version=accepted.version,
                )
            )
            incumbent = accepted
        return incumbent, tuple(history)
