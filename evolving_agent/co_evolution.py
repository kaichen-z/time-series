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


META_HARNESS_PROMPT = """You are an open-ended Meta-Harness Engineer for contextual time-series
forecasting. Propose a complete child Harness Genome, not merely a local prompt tweak. You may
redesign all mutable inference and search components together: Coding hypothesis generation and
revision prompts, Retrieval and Decision prompts, candidate/search budgets, historical hindcast
configuration, evidence-adjustment policy, multi-round workflow, and decision aggregation.

The child must remain executable by the supplied host primitives. The workflow is a list containing
only "retrieve" and "decide", must include both, and may contain at most 8 stages. You may create
multiple instances of these roles by repeating stages. Return every field, even if unchanged:
{"coding_generation_prompt": "...", "coding_revision_prompt": "...",
"retrieval_prompt": "...", "decision_prompt": "...",
"coding_initial_programs": 3, "coding_mutations": 1, "coding_mutation_children": 1,
"coding_validation_folds": 3, "coding_validation_horizon": 8,
"workflow": ["retrieve", "decide"], "enable_evidence_adjustments": true,
"max_evidence_adjustments": 3, "decision_aggregation": "last|majority",
"changelog": "testable rationale for this child"}

Immutable scientific and safety boundaries: Coding sees only historical numbers and historical
hindcast diagnostics; Retrieval and Decision never see future values or GT evidence during
inference; only verified quotes may support contextual changes; generated code keeps the existing
forecast(history, horizon, frequency) contract and sandbox restrictions; no agent may edit the
scorer, data split, label boundary, sandbox, acceptance test, or resource caps. The child will be
executed on train tasks and accepted only if it improves a disjoint held-out development split.
"""


@dataclass(frozen=True)
class HarnessPolicy:
    """A complete, inheritable genome for both Coding search and the whole harness."""

    version: str = "v000"
    parent: str | None = None
    coding_generation_prompt: str = GENERATION_PROMPT
    coding_revision_prompt: str = REVISION_PROMPT
    retrieval_prompt: str = RETRIEVAL_PROMPT
    decision_prompt: str = DECISION_PROMPT
    coding_initial_programs: int = 3
    coding_mutations: int = 1
    coding_mutation_children: int = 1
    coding_validation_folds: int = 3
    coding_validation_horizon: int = 8
    workflow: tuple[str, ...] = ("retrieve", "decide")
    enable_evidence_adjustments: bool = True
    max_evidence_adjustments: int = 3
    decision_aggregation: str = "last"
    changelog: str = "Hand-written seed policy."

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "HarnessPolicy":
        source = Path(path)
        if not source.exists():
            return cls()
        payload = json.loads(source.read_text())
        if "workflow" in payload:
            payload["workflow"] = tuple(payload["workflow"])
        return cls(**payload)


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
    child_changelogs: dict[str, str] | None = None


@dataclass(frozen=True)
class CoEvolutionConfig:
    generations: int = 3
    children_per_generation: int = 2
    max_workflow_stages: int = 8


HarnessFactory = Callable[[HarnessPolicy], EvolvingForecastHarness]


def evaluate_policy(
    policy: HarnessPolicy,
    tasks: Sequence[ContextTask],
    harness_factory: HarnessFactory,
    *,
    learn_skills: bool,
    harness: EvolvingForecastHarness | None = None,
) -> PolicyEvaluation:
    """Run label-free inference first, then expose resolved labels only to scoring."""
    if not tasks:
        raise ValueError("policy evaluation needs at least one resolved task")
    harness = harness or harness_factory(policy)
    outcomes = []
    traces = []
    for task in tasks:
        inference = harness.run(task)
        if learn_skills:
            outcome, _learning = harness.record_outcome(task, inference)
        else:
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
    """Evolve complete, inheritable Harness Genomes with train/dev elitism."""

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
            "instruction": (
                "The weakest observed module is a diagnosis, not a mutation restriction. "
                "Redesign any mutually dependent genome fields needed to improve the whole system."
            ),
        }
        response = self.llm.complete(
            system=META_HARNESS_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.4,
        )
        version = f"v{self._version:03d}"
        self._version += 1
        try:
            proposal = parse_json_object(response.text)
        except JsonExtractionError:
            return replace(parent, version=version, parent=parent.version, changelog="Invalid mutation; unchanged.")
        candidate, reason = self._proposal(parent, proposal, version)
        if candidate is None:
            return replace(parent, version=version, parent=parent.version, changelog=reason)
        return candidate

    def _proposal(
        self,
        parent: HarnessPolicy,
        proposal: dict,
        version: str,
    ) -> tuple[HarnessPolicy | None, str]:
        prompt_fields = (
            "coding_generation_prompt",
            "coding_revision_prompt",
            "retrieval_prompt",
            "decision_prompt",
        )
        prompts = {}
        for field in prompt_fields:
            value = str(proposal.get(field, getattr(parent, field))).strip()
            if not value:
                return None, f"Illegal empty field: {field}."
            prompts[field] = value

        workflow = tuple(str(item) for item in proposal.get("workflow", parent.workflow))
        if (
            not workflow
            or len(workflow) > self.config.max_workflow_stages
            or set(workflow) - {"retrieve", "decide"}
            or "retrieve" not in workflow
            or "decide" not in workflow
        ):
            return None, "Illegal workflow mutation; unchanged."
        aggregation = str(proposal.get("decision_aggregation", parent.decision_aggregation))
        if aggregation not in {"last", "majority"}:
            return None, "Illegal decision aggregation; unchanged."

        integer_ranges = {
            "coding_initial_programs": (1, 12),
            "coding_mutations": (0, 6),
            "coding_mutation_children": (1, 6),
            "coding_validation_folds": (1, 8),
            "coding_validation_horizon": (1, 64),
            "max_evidence_adjustments": (0, 12),
        }
        integers = {}
        for field, (lower, upper) in integer_ranges.items():
            try:
                value = int(proposal.get(field, getattr(parent, field)))
            except (TypeError, ValueError):
                return None, f"Illegal integer field: {field}."
            if value < lower or value > upper:
                return None, f"Out-of-budget field: {field}."
            integers[field] = value

        return (
            replace(
                parent,
                version=version,
                parent=parent.version,
                changelog=str(proposal.get("changelog", "Open-ended genome mutation."))[:500],
                workflow=workflow,
                decision_aggregation=aggregation,
                enable_evidence_adjustments=bool(
                    proposal.get(
                        "enable_evidence_adjustments",
                        parent.enable_evidence_adjustments,
                    )
                ),
                **prompts,
                **integers,
            ),
            "",
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
            parent_harness = self.harness_factory(incumbent)
            parent_train = evaluate_policy(
                incumbent,
                train_tasks,
                self.harness_factory,
                learn_skills=True,
                harness=parent_harness,
            )
            parent_dev = evaluate_policy(
                incumbent,
                dev_tasks,
                self.harness_factory,
                learn_skills=False,
                harness=parent_harness,
            )
            children = [
                self.mutate(incumbent, parent_train)
                for _ in range(self.config.children_per_generation)
            ]
            child_harnesses = {
                child.version: self.harness_factory(child) for child in children
            }
            train_evaluations = {
                child.version: evaluate_policy(
                    child,
                    train_tasks,
                    self.harness_factory,
                    learn_skills=True,
                    harness=child_harnesses[child.version],
                )
                for child in children
            }
            train_best = max(children, key=lambda child: train_evaluations[child.version].system_reward)
            child_dev = evaluate_policy(
                train_best,
                dev_tasks,
                self.harness_factory,
                learn_skills=False,
                harness=child_harnesses[train_best.version],
            )
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
                    child_changelogs={child.version: child.changelog for child in children},
                )
            )
            incumbent = accepted
        return incumbent, tuple(history)
