from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .code_evolution import INITIAL_PROMPT, MUTATION_PROMPT
from .codex_triad import CODING_AGENT_PROMPT, DECISION_AGENT_PROMPT, RETRIEVAL_AGENT_PROMPT
from .models import ForecastTask, RunResult


@dataclass(frozen=True)
class AgentPromptBundle:
    """Versioned, reusable policy for the three-agent forecasting harness.

    The Coding Agent has two nested policies: a triad planning prompt and the
    prompts that generate/revise executable numbers-only programs.  Retrieval
    and Decision receive their own prompts.  Evolution changes one prompt at a
    time, which keeps every child auditable and falsifiable.
    """

    version: str = "v000"
    parent: str | None = None
    coding_plan_prompt: str = CODING_AGENT_PROMPT
    coding_program_prompt: str = INITIAL_PROMPT
    coding_revision_prompt: str = MUTATION_PROMPT
    retrieval_prompt: str = RETRIEVAL_AGENT_PROMPT
    decision_prompt: str = DECISION_AGENT_PROMPT
    notes: str = "Hand-written seed bundle."

    @classmethod
    def load(cls, path: str | Path | None) -> "AgentPromptBundle":
        if path is None:
            return cls()
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return cls()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        seed = cls()
        return cls(**{
            field: payload.get(field, getattr(seed, field))
            for field in asdict(seed)
        })

    def save(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def coding_policy(self):
        from .code_evolution import CodingAgentPolicy

        return CodingAgentPolicy(
            version=self.version,
            parent=self.parent,
            generation_prompt=self.coding_program_prompt,
            revision_prompt=self.coding_revision_prompt,
            notes=self.notes,
        )


@dataclass(frozen=True)
class TaskEvolutionOutcome:
    benchmark_id: str
    system_score: float
    coding_reward: float
    retrieval_reward: float
    decision_reward: float
    trace: dict[str, Any]

    @classmethod
    def from_result(cls, task: ForecastTask, result: RunResult) -> "TaskEvolutionOutcome":
        if not task.labels_public or task.future_values is None:
            raise ValueError("co-evolution requires resolved training tasks")
        metrics = result.metrics or {}
        baseline = max(float(metrics.get("baseline_mae", 0.0)), 1e-8)
        mae = float(metrics.get("mae", baseline))
        coverage_gain = float(metrics.get("candidate_coverage_gain", 0.0))
        regret = float(metrics.get("decision_selection_regret", 0.0))
        retrieval_parts = (
            float(metrics.get("retrieval_precision", 0.0)),
            float(metrics.get("distractor_avoidance", 0.0)),
            float(metrics.get("evidence_token_recall_proxy", 0.0)),
        )
        retrieval_reward = statistics.fmean(retrieval_parts)
        return cls(
            benchmark_id=task.benchmark_id,
            system_score=-(mae / baseline) + 0.2 * retrieval_reward,
            coding_reward=coverage_gain / baseline,
            retrieval_reward=retrieval_reward,
            decision_reward=-(regret / baseline),
            trace={
                "forecast_mae_ratio": mae / baseline,
                "candidate_coverage_gain": coverage_gain,
                "retrieval_quality": retrieval_reward,
                "decision_selection_regret": regret,
            },
        )


@dataclass(frozen=True)
class BundleEvaluation:
    bundle_version: str
    mean_score: float
    module_rewards: dict[str, float]
    outcomes: tuple[TaskEvolutionOutcome, ...]
    worst: tuple[TaskEvolutionOutcome, ...]


class BundleEvaluator(Protocol):
    def __call__(
        self, bundle: AgentPromptBundle, tasks: Sequence[ForecastTask]
    ) -> BundleEvaluation: ...


def evaluate_bundle(
    bundle: AgentPromptBundle,
    tasks: Sequence[ForecastTask],
    system_factory: Callable[[AgentPromptBundle], Any],
    *,
    worst_n: int = 5,
) -> BundleEvaluation:
    """Run inference label-free, then score only after each future is revealed."""

    system = system_factory(bundle)
    outcomes = tuple(
        TaskEvolutionOutcome.from_result(task, system.run(task, index))
        for index, task in enumerate(tasks)
    )
    if not outcomes:
        raise ValueError("at least one resolved task is required")
    return BundleEvaluation(
        bundle_version=bundle.version,
        mean_score=statistics.fmean(item.system_score for item in outcomes),
        module_rewards={
            "coding": statistics.fmean(item.coding_reward for item in outcomes),
            "retrieval": statistics.fmean(item.retrieval_reward for item in outcomes),
            "decision": statistics.fmean(item.decision_reward for item in outcomes),
        },
        outcomes=outcomes,
        worst=tuple(sorted(outcomes, key=lambda item: item.system_score)[:worst_n]),
    )


MUTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prompt_field", "replacement_prompt", "changelog"],
    "properties": {
        "prompt_field": {
            "type": "string",
            "enum": [
                "coding_plan_prompt",
                "coding_program_prompt",
                "coding_revision_prompt",
                "retrieval_prompt",
                "decision_prompt",
            ],
        },
        "replacement_prompt": {"type": "string", "maxLength": 12000},
        "changelog": {"type": "string", "maxLength": 300},
    },
}


@dataclass(frozen=True)
class CoEvolutionConfig:
    generations: int = 3
    population_size: int = 4
    keep_elite: int = 2
    worst_n: int = 5


@dataclass(frozen=True)
class CoEvolutionGeneration:
    generation: int
    population: tuple[str, ...]
    train_scores: dict[str, float]
    dev_scores: dict[str, float]
    elite: tuple[str, ...]
    mutated_agent: str | None


class PromptCoEvolutionEngine:
    """Failure-attributed population evolution for Coding/Retrieval/Decision.

    A child changes exactly one prompt.  The weakest module reward determines
    which role is eligible for mutation; unlike a random topology search, this
    makes the edit traceable to an observed forecasting failure.
    """

    def __init__(
        self,
        client: Any,
        evaluator: BundleEvaluator,
        config: CoEvolutionConfig | None = None,
    ) -> None:
        self.client = client
        self.evaluator = evaluator
        self.config = config or CoEvolutionConfig()
        self._next_version = 1

    @staticmethod
    def target_agent(evaluation: BundleEvaluation) -> str:
        return min(evaluation.module_rewards, key=evaluation.module_rewards.get)

    @staticmethod
    def _allowed_fields(agent: str) -> set[str]:
        return {
            "coding": {
                "coding_plan_prompt",
                "coding_program_prompt",
                "coding_revision_prompt",
            },
            "retrieval": {"retrieval_prompt"},
            "decision": {"decision_prompt"},
        }[agent]

    def mutate(
        self, parent: AgentPromptBundle, evaluation: BundleEvaluation
    ) -> AgentPromptBundle:
        target = self.target_agent(evaluation)
        failures = [
            {
                "benchmark_id": item.benchmark_id,
                "system_score": item.system_score,
                "trace": item.trace,
            }
            for item in evaluation.worst
        ]
        prompt = (
            "You are the Harness Evolver. Read evolution.json and make exactly one "
            f"targeted prompt change to the {target} agent. Preserve the agent's information "
            "boundary. Coding may see only historical numbers and numerical diagnostics; "
            "Retrieval may see candidate assumptions and documents but no labels; Decision may "
            "see executed candidates and verified evidence but no future values. Return a full "
            "replacement prompt, not a patch."
        )
        result = self.client.complete(
            f"co_evolve_{parent.version}_{target}",
            prompt,
            MUTATION_SCHEMA,
            workspace_files={
                "evolution.json": json.dumps(
                    {
                        "target_agent": target,
                        "module_rewards": evaluation.module_rewards,
                        "worst_failures": failures,
                        "current_bundle": asdict(parent),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            },
        )
        version = f"v{self._next_version:03d}"
        self._next_version += 1
        if not result:
            return replace(parent, version=version, parent=parent.version, notes="Mutation failed; unchanged.")
        field = str(result.get("prompt_field", ""))
        replacement = str(result.get("replacement_prompt", "")).strip()
        if field not in self._allowed_fields(target) or not replacement:
            return replace(parent, version=version, parent=parent.version, notes="Illegal mutation; unchanged.")
        return replace(
            parent,
            version=version,
            parent=parent.version,
            notes=str(result.get("changelog", ""))[:300],
            **{field: replacement},
        )

    def evolve(
        self,
        seed: AgentPromptBundle,
        train_tasks: Sequence[ForecastTask],
        dev_tasks: Sequence[ForecastTask] = (),
    ) -> tuple[AgentPromptBundle, tuple[CoEvolutionGeneration, ...]]:
        population = [seed]
        records = []
        best_bundle = seed
        best_dev = float("-inf")
        for generation in range(self.config.generations):
            evaluations = {
                bundle.version: self.evaluator(bundle, train_tasks)
                for bundle in population
            }
            ranked = sorted(
                population,
                key=lambda bundle: evaluations[bundle.version].mean_score,
                reverse=True,
            )
            elite = ranked[: self.config.keep_elite]
            dev_scores = {
                bundle.version: self.evaluator(bundle, dev_tasks).mean_score
                for bundle in elite
            } if dev_tasks else {}
            selected = max(
                elite,
                key=lambda bundle: dev_scores.get(
                    bundle.version, evaluations[bundle.version].mean_score
                ),
            )
            selected_score = dev_scores.get(
                selected.version, evaluations[selected.version].mean_score
            )
            if selected_score > best_dev:
                best_bundle, best_dev = selected, selected_score
            mutated_agent = self.target_agent(evaluations[elite[0].version])
            records.append(CoEvolutionGeneration(
                generation=generation,
                population=tuple(bundle.version for bundle in population),
                train_scores={key: value.mean_score for key, value in evaluations.items()},
                dev_scores=dev_scores,
                elite=tuple(bundle.version for bundle in elite),
                mutated_agent=mutated_agent,
            ))
            if generation == self.config.generations - 1:
                break
            children = []
            while len(elite) + len(children) < self.config.population_size:
                parent = elite[len(children) % len(elite)]
                children.append(self.mutate(parent, evaluations[parent.version]))
            population = elite + children
        return best_bundle, tuple(records)
