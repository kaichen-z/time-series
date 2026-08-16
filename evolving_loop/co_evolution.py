"""Delayed-outcome, failure-attributed evolution of the three-agent harness."""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Sequence

from evolving_loop.coding_agent.evolution import GENERATION_PROMPT, REVISION_PROMPT
from evolving_loop.data import ContextTask, Document
from evolving_loop.decision_agent.agent import DECISION_PROMPT
from evolving_loop.evaluation import ResolvedOutcome, score_after_resolution
from evolving_loop.harness import EvolvingForecastHarness
from evolving_loop.retrieval_agent.agent import RETRIEVAL_PROMPT
from common.llm import (
    JsonExtractionError,
    LLMClient,
    TransientLLMError,
    parse_json_object,
)


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

PROMPT_ONLY_EVOLVER_PROMPT = """You are a constrained Prompt Evolver for a time-series agent
harness. Use resolved training failures to replace exactly one complete prompt owned by the
diagnosed weakest role. You may not change another role, any numeric/search budget, topology,
source code, scorer, data boundary, or safety mechanism. Return exactly:
{"prompt_field": "coding_generation_prompt|coding_revision_prompt|retrieval_prompt|decision_prompt",
"replacement_prompt": "complete replacement prompt", "changelog": "testable rationale"}
"""

EvolutionMode = Literal["prompt", "genome"]
EvolutionTarget = Literal["auto", "coding", "retrieval", "decision"]


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
    coding_skills: tuple[dict, ...] = ()
    retrieval_skills: tuple[dict, ...] = ()
    decision_skills: tuple[dict, ...] = ()
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
        for field in ("coding_skills", "retrieval_skills", "decision_skills"):
            if field in payload:
                payload[field] = tuple(payload[field])
        return cls(**payload)


def snapshot_policy_skills(
    policy: HarnessPolicy,
    harness: EvolvingForecastHarness,
) -> HarnessPolicy:
    """Freeze all evaluation-local validated skills into an inheritable policy artifact."""

    def records(agent: object | None, current: tuple[dict, ...]) -> tuple[dict, ...]:
        if agent is None:
            return current
        library = getattr(agent, "library", None)
        if library is None:
            return current
        return tuple(
            asdict(skill)
            for skill in sorted(library.all(), key=lambda item: item.name)
        )

    return replace(
        policy,
        coding_skills=records(getattr(harness, "coding", None), policy.coding_skills),
        retrieval_skills=records(
            getattr(harness, "retrieval", None), policy.retrieval_skills
        ),
        decision_skills=records(
            getattr(harness, "decision", None), policy.decision_skills
        ),
    )


@dataclass(frozen=True)
class PolicyEvaluation:
    version: str
    system_reward: float
    module_rewards: dict[str, float]
    outcomes: tuple[ResolvedOutcome, ...]
    failure_traces: tuple[dict, ...] = ()
    diagnostics: dict[str, float] = field(default_factory=dict)


def evaluation_diagnostics(
    outcomes: Sequence[ResolvedOutcome],
) -> dict[str, float]:
    """Aggregate diagnostics that distinguish candidate coverage from final selection."""
    if not outcomes:
        return {
            "mean_final_smape": 0.0,
            "mean_best_of_k_smape": 0.0,
            "mean_selection_regret": 0.0,
            "mean_candidate_count": 0.0,
            "mean_hindcast_future_rank_correlation": 0.0,
        }
    return {
        "mean_final_smape": statistics.fmean(item.final_smape for item in outcomes),
        "mean_best_of_k_smape": statistics.fmean(
            item.coding_oracle_smape for item in outcomes
        ),
        "mean_selection_regret": statistics.fmean(
            item.decision_selection_regret for item in outcomes
        ),
        "mean_candidate_count": statistics.fmean(item.candidate_count for item in outcomes),
        "mean_hindcast_future_rank_correlation": statistics.fmean(
            item.hindcast_future_rank_correlation for item in outcomes
        ),
    }


@dataclass(frozen=True)
class EvolutionStep:
    mode: str
    generation: int
    parent_version: str
    child_versions: tuple[str, ...]
    target_agent: str
    parent_train_reward: float
    child_train_rewards: dict[str, float]
    parent_dev_reward: float | None
    best_child_dev_reward: float | None
    accepted_version: str
    parent_train_module_rewards: dict[str, float] | None = None
    parent_dev_module_rewards: dict[str, float] | None = None
    best_child_train_module_rewards: dict[str, float] | None = None
    best_child_dev_module_rewards: dict[str, float] | None = None
    parent_train_diagnostics: dict[str, float] | None = None
    parent_dev_diagnostics: dict[str, float] | None = None
    best_child_train_diagnostics: dict[str, float] | None = None
    best_child_dev_diagnostics: dict[str, float] | None = None
    child_changelogs: dict[str, str] | None = None
    successive_halving: bool = False
    parent_screen_train_reward: float | None = None
    parent_screen_dev_reward: float | None = None
    child_screen_train_rewards: dict[str, float] | None = None
    child_screen_dev_rewards: dict[str, float] | None = None
    promoted_versions: tuple[str, ...] = ()
    screen_prune_reasons: dict[str, str] | None = None


@dataclass(frozen=True)
class CoEvolutionConfig:
    generations: int = 3
    children_per_generation: int = 2
    max_workflow_stages: int = 8
    mode: EvolutionMode = "genome"
    target: EvolutionTarget = "auto"
    checkpoint_path: str | Path | None = None
    progress_path: str | Path | None = None
    resume: bool = True
    successive_halving: bool = False
    screening_train_tasks: int = 6
    screening_dev_tasks: int = 2
    screening_promote: int = 1
    screening_tolerance: float = 0.01


HarnessFactory = Callable[[HarnessPolicy], EvolvingForecastHarness]


def evaluate_policy(
    policy: HarnessPolicy,
    tasks: Sequence[ContextTask],
    harness_factory: HarnessFactory,
    *,
    learn_skills: bool,
    harness: EvolvingForecastHarness | None = None,
    progress: Callable[[str, dict], None] | None = None,
) -> PolicyEvaluation:
    """Run label-free inference first, then expose resolved labels only to scoring."""
    if not tasks:
        raise ValueError("policy evaluation needs at least one resolved task")
    harness = harness or harness_factory(policy)
    outcomes = []
    traces = []
    for task in tasks:
        if progress:
            progress("task_started", {"task_id": task.numeric.task_id})
        # This firewall lives outside all source-mutable agent/orchestration files.
        # A source-evolved child receives no labels even if it rewrites harness.py.
        inference_task = replace(
            task,
            numeric=replace(task.numeric, future_values=()),
            documents=tuple(
                Document(document.document_id, document.content)
                for document in task.documents
            ),
            gt_evidence=(),
            labels_public=False,
        )
        inference = harness.run(inference_task)
        if learn_skills:
            outcome, _learning = harness.record_outcome(task, inference)
        else:
            outcome = score_after_resolution(task, inference)
        outcomes.append(outcome)
        candidate_scores = {
            candidate.candidate_id: score_after_resolution(
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
        if progress:
            progress(
                "task_completed",
                {"task_id": task.numeric.task_id, "final_smape": outcome.final_smape},
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
        diagnostics=evaluation_diagnostics(outcomes),
    )


def combine_policy_evaluations(
    version: str,
    pieces: Sequence[tuple[PolicyEvaluation, int]],
) -> PolicyEvaluation:
    """Combine disjoint evaluation segments without rerunning screened train tasks."""
    usable = [(evaluation, count) for evaluation, count in pieces if count > 0]
    total = sum(count for _evaluation, count in usable)
    if total <= 0:
        raise ValueError("combined policy evaluation needs at least one task")

    def weighted(values: Sequence[tuple[float, int]]) -> float:
        return sum(value * count for value, count in values) / total

    module_keys = {
        key for evaluation, _count in usable for key in evaluation.module_rewards
    }
    diagnostic_keys = {
        key for evaluation, _count in usable for key in evaluation.diagnostics
    }
    return PolicyEvaluation(
        version=version,
        system_reward=weighted(
            [(evaluation.system_reward, count) for evaluation, count in usable]
        ),
        module_rewards={
            key: weighted(
                [(evaluation.module_rewards.get(key, 0.0), count) for evaluation, count in usable]
            )
            for key in module_keys
        },
        outcomes=tuple(
            outcome
            for evaluation, _count in usable
            for outcome in evaluation.outcomes
        ),
        failure_traces=tuple(
            trace
            for evaluation, _count in usable
            for trace in evaluation.failure_traces
        ),
        diagnostics={
            key: weighted(
                [(evaluation.diagnostics.get(key, 0.0), count) for evaluation, count in usable]
            )
            for key in diagnostic_keys
        },
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

    def target_agent(self, evaluation: PolicyEvaluation) -> str:
        return (
            self.weakest_agent(evaluation)
            if self.config.target == "auto"
            else self.config.target
        )

    def mutate(
        self,
        parent: HarnessPolicy,
        evaluation: PolicyEvaluation,
        *,
        child_index: int = 0,
    ) -> HarnessPolicy:
        target = self.target_agent(evaluation)
        worst = sorted(
            evaluation.failure_traces,
            key=lambda item: item["final_smape"],
            reverse=True,
        )[:5]
        if self.config.target == "auto":
            mutation_instruction = (
                "The weakest observed module is a diagnosis, not a mutation restriction. "
                "Redesign any mutually dependent genome fields needed to improve the whole system."
            )
        else:
            mutation_instruction = (
                f"This is a targeted {target.title()} evolution stage. Mutate only fields owned "
                f"by the {target.title()} Agent; you must not mutate "
                + {
                    "coding": "Retrieval or Decision fields.",
                    "retrieval": "Coding or Decision fields.",
                    "decision": "Coding or Retrieval fields.",
                }[target]
            )
        current_policy = asdict(parent)
        for field_name in ("coding_skills", "retrieval_skills", "decision_skills"):
            current_policy.pop(field_name, None)
        diversity_modes = (
            "Prefer a minimal, conservative genome whose improvement can be attributed clearly.",
            "Prefer a structurally different workflow or cross-agent coordination strategy.",
            "Prefer a different validation, search-budget, or evidence-to-decision tradeoff.",
        )
        diversity_instruction = (
            f"Child {child_index}: "
            f"{diversity_modes[child_index % len(diversity_modes)]} "
            "Do not duplicate another child from this generation."
        )
        payload = {
            "target_agent": target,
            "child_index": child_index,
            "diversity_instruction": diversity_instruction,
            "module_rewards": evaluation.module_rewards,
            "worst_failure_trajectories": worst,
            "current_policy": current_policy,
            "skill_inventory": {
                "coding": [record.get("name", "") for record in parent.coding_skills],
                "retrieval": [record.get("name", "") for record in parent.retrieval_skills],
                "decision": [record.get("name", "") for record in parent.decision_skills],
            },
            "instruction": mutation_instruction,
        }
        version = f"v{self._version:03d}"
        self._version += 1
        try:
            system_prompt = (
                    PROMPT_ONLY_EVOLVER_PROMPT
                    if self.config.mode == "prompt"
                    else META_HARNESS_PROMPT
                )
            if self.config.target != "auto":
                system_prompt = system_prompt.replace(
                    "diagnosed weakest role", f"explicitly targeted {target.title()} role"
                )
                system_prompt += (
                    f"\nThis run explicitly targets the {target.title()} Agent. Follow the target "
                    "in the user payload even if another module has the lowest diagnostic score."
                )
            response = self.llm.complete(
                system=system_prompt,
                messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                temperature=0.4,
            )
        except TransientLLMError:
            raise
        except Exception as exc:
            return replace(
                parent,
                version=version,
                parent=parent.version,
                changelog=f"Mutation call failed; unchanged: {type(exc).__name__}: {exc}"[:500],
            )
        try:
            proposal = parse_json_object(response.text)
        except JsonExtractionError:
            return replace(parent, version=version, parent=parent.version, changelog="Invalid mutation; unchanged.")
        if self.config.mode == "prompt":
            candidate, reason = self._prompt_proposal(parent, proposal, version, target)
        else:
            candidate, reason = self._proposal(parent, proposal, version)
            if candidate is not None and self.config.target != "auto":
                candidate = self._scope_genome_candidate(parent, candidate, target)
        if candidate is None:
            return replace(parent, version=version, parent=parent.version, changelog=reason)
        return candidate

    @staticmethod
    def _scope_genome_candidate(
        parent: HarnessPolicy,
        candidate: HarnessPolicy,
        target: str,
    ) -> HarnessPolicy:
        """Freeze genome fields that are owned by roles outside a targeted mutation."""
        if target == "coding":
            return replace(
                candidate,
                retrieval_prompt=parent.retrieval_prompt,
                decision_prompt=parent.decision_prompt,
                workflow=parent.workflow,
                enable_evidence_adjustments=parent.enable_evidence_adjustments,
                max_evidence_adjustments=parent.max_evidence_adjustments,
                decision_aggregation=parent.decision_aggregation,
            )
        if target == "retrieval":
            return replace(
                candidate,
                coding_generation_prompt=parent.coding_generation_prompt,
                coding_revision_prompt=parent.coding_revision_prompt,
                decision_prompt=parent.decision_prompt,
                coding_initial_programs=parent.coding_initial_programs,
                coding_mutations=parent.coding_mutations,
                coding_mutation_children=parent.coding_mutation_children,
                coding_validation_folds=parent.coding_validation_folds,
                coding_validation_horizon=parent.coding_validation_horizon,
                enable_evidence_adjustments=parent.enable_evidence_adjustments,
                max_evidence_adjustments=parent.max_evidence_adjustments,
                decision_aggregation=parent.decision_aggregation,
            )
        if target == "decision":
            return replace(
                candidate,
                coding_generation_prompt=parent.coding_generation_prompt,
                coding_revision_prompt=parent.coding_revision_prompt,
                retrieval_prompt=parent.retrieval_prompt,
                coding_initial_programs=parent.coding_initial_programs,
                coding_mutations=parent.coding_mutations,
                coding_mutation_children=parent.coding_mutation_children,
                coding_validation_folds=parent.coding_validation_folds,
                coding_validation_horizon=parent.coding_validation_horizon,
                workflow=parent.workflow,
            )
        return candidate

    @staticmethod
    def _prompt_proposal(
        parent: HarnessPolicy,
        proposal: dict,
        version: str,
        target: str,
    ) -> tuple[HarnessPolicy | None, str]:
        allowed = {
            "coding": {"coding_generation_prompt", "coding_revision_prompt"},
            "retrieval": {"retrieval_prompt"},
            "decision": {"decision_prompt"},
        }
        field = str(proposal.get("prompt_field", ""))
        replacement_prompt = str(proposal.get("replacement_prompt", "")).strip()
        if field not in allowed[target] or not replacement_prompt:
            return None, "Illegal prompt-only mutation; unchanged."
        return (
            replace(
                parent,
                version=version,
                parent=parent.version,
                changelog=str(proposal.get("changelog", "Prompt-only mutation."))[:500],
                **{field: replacement_prompt},
            ),
            "",
        )

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
        incumbent, history, start_generation = self._load_checkpoint(seed)
        for generation in range(start_generation, self.config.generations):
            self._progress("generation_started", generation=generation, parent=incumbent.version)
            if self.config.successive_halving:
                incumbent, step, child_dev_reward = self._successive_halving_generation(
                    incumbent,
                    train_tasks,
                    dev_tasks,
                    generation=generation,
                )
                history.append(step)
                self._save_checkpoint(incumbent, history, generation + 1)
                self._progress(
                    "generation_completed",
                    generation=generation,
                    accepted=incumbent.version,
                    parent_dev_reward=step.parent_dev_reward,
                    child_dev_reward=child_dev_reward,
                )
                continue
            parent_harness = self.harness_factory(incumbent)
            parent_train = self._evaluate(
                incumbent,
                train_tasks,
                stage="parent_train",
                generation=generation,
                learn_skills=True,
                harness=parent_harness,
            )
            trained_parent = snapshot_policy_skills(incumbent, parent_harness)
            parent_dev = self._evaluate(
                trained_parent,
                dev_tasks,
                stage="parent_dev",
                generation=generation,
                learn_skills=False,
                harness=parent_harness,
            )
            children: list[HarnessPolicy] = []
            child_harnesses: dict[str, EvolvingForecastHarness] = {}
            train_evaluations: dict[str, PolicyEvaluation] = {}
            for child_index in range(self.config.children_per_generation):
                self._progress("candidate_started", generation=generation, child=child_index)
                child = self.mutate(
                    incumbent,
                    parent_train,
                    child_index=child_index,
                )
                children.append(child)
                try:
                    child_harness = self.harness_factory(child)
                    child_harnesses[child.version] = child_harness
                    train_evaluations[child.version] = self._evaluate(
                        child,
                        train_tasks,
                        stage="child_train",
                        generation=generation,
                        learn_skills=True,
                        harness=child_harness,
                    )
                    children[-1] = snapshot_policy_skills(child, child_harness)
                    self._progress(
                        "candidate_completed",
                        generation=generation,
                        child=child.version,
                        train_reward=train_evaluations[child.version].system_reward,
                    )
                except TransientLLMError:
                    raise
                except Exception as exc:
                    self._progress(
                        "candidate_failed",
                        generation=generation,
                        child=child.version,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            valid_children = [child for child in children if child.version in train_evaluations]
            improving_children = [
                child
                for child in valid_children
                if train_evaluations[child.version].system_reward > parent_train.system_reward
            ]
            for child in valid_children:
                if child not in improving_children:
                    self._progress(
                        "candidate_pruned_before_dev",
                        generation=generation,
                        child=child.version,
                        parent_train_reward=parent_train.system_reward,
                        child_train_reward=train_evaluations[child.version].system_reward,
                    )
            train_best = (
                max(
                    improving_children,
                    key=lambda item: train_evaluations[item.version].system_reward,
                )
                if improving_children
                else None
            )
            child_dev = None
            if train_best is not None:
                try:
                    child_dev = self._evaluate(
                        train_best,
                        dev_tasks,
                        stage="child_dev",
                        generation=generation,
                        learn_skills=False,
                        harness=child_harnesses[train_best.version],
                    )
                except TransientLLMError:
                    raise
                except Exception as exc:
                    self._progress(
                        "candidate_dev_failed",
                        generation=generation,
                        child=train_best.version,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            accepted = (
                train_best
                if train_best is not None
                and child_dev is not None
                and child_dev.system_reward > parent_dev.system_reward
                else trained_parent
            )
            history.append(
                EvolutionStep(
                    mode=self.config.mode,
                    generation=generation,
                    parent_version=incumbent.version,
                    child_versions=tuple(child.version for child in children),
                    target_agent=self.target_agent(parent_train),
                    parent_train_reward=parent_train.system_reward,
                    child_train_rewards={
                        version: item.system_reward for version, item in train_evaluations.items()
                    },
                    parent_dev_reward=parent_dev.system_reward,
                    best_child_dev_reward=(child_dev.system_reward if child_dev else None),
                    accepted_version=accepted.version,
                    parent_train_module_rewards=parent_train.module_rewards,
                    parent_dev_module_rewards=parent_dev.module_rewards,
                    best_child_train_module_rewards=(
                        train_evaluations[train_best.version].module_rewards
                        if train_best is not None
                        else None
                    ),
                    best_child_dev_module_rewards=(
                        child_dev.module_rewards if child_dev is not None else None
                    ),
                    parent_train_diagnostics=parent_train.diagnostics,
                    parent_dev_diagnostics=parent_dev.diagnostics,
                    best_child_train_diagnostics=(
                        train_evaluations[train_best.version].diagnostics
                        if train_best is not None
                        else None
                    ),
                    best_child_dev_diagnostics=(
                        child_dev.diagnostics if child_dev is not None else None
                    ),
                    child_changelogs={child.version: child.changelog for child in children},
                )
            )
            incumbent = accepted
            self._save_checkpoint(incumbent, history, generation + 1)
            self._progress(
                "generation_completed",
                generation=generation,
                accepted=incumbent.version,
                parent_dev_reward=parent_dev.system_reward,
                child_dev_reward=(child_dev.system_reward if child_dev else None),
            )
        return incumbent, tuple(history)

    def _successive_halving_generation(
        self,
        incumbent: HarnessPolicy,
        train_tasks: Sequence[ContextTask],
        dev_tasks: Sequence[ContextTask],
        *,
        generation: int,
    ) -> tuple[HarnessPolicy, EvolutionStep, float | None]:
        """Screen every child cheaply, then fully evaluate only the best survivors."""
        screen_train_count = min(
            max(1, self.config.screening_train_tasks), len(train_tasks)
        )
        screen_dev_count = min(max(1, self.config.screening_dev_tasks), len(dev_tasks))
        screen_train_tasks = train_tasks[:screen_train_count]
        remaining_train_tasks = train_tasks[screen_train_count:]
        screen_dev_tasks = dev_tasks[:screen_dev_count]
        self._progress(
            "screening_started",
            generation=generation,
            parent=incumbent.version,
            screen_train_tasks=screen_train_count,
            screen_dev_tasks=screen_dev_count,
            promote=self.config.screening_promote,
            tolerance=self.config.screening_tolerance,
        )

        parent_harness = self.harness_factory(incumbent)
        parent_screen_train = self._evaluate(
            incumbent,
            screen_train_tasks,
            stage="parent_screen_train",
            generation=generation,
            learn_skills=True,
            harness=parent_harness,
        )
        screen_trained_parent = snapshot_policy_skills(incumbent, parent_harness)
        parent_screen_dev = self._evaluate(
            screen_trained_parent,
            screen_dev_tasks,
            stage="parent_screen_dev",
            generation=generation,
            learn_skills=False,
            harness=parent_harness,
        )
        if remaining_train_tasks:
            parent_train_remaining = self._evaluate(
                screen_trained_parent,
                remaining_train_tasks,
                stage="parent_train_remaining",
                generation=generation,
                learn_skills=True,
                harness=parent_harness,
            )
            parent_train = combine_policy_evaluations(
                incumbent.version,
                (
                    (parent_screen_train, len(screen_train_tasks)),
                    (parent_train_remaining, len(remaining_train_tasks)),
                ),
            )
        else:
            parent_train = parent_screen_train
        trained_parent = snapshot_policy_skills(incumbent, parent_harness)
        parent_dev = self._evaluate(
            trained_parent,
            dev_tasks,
            stage="parent_dev",
            generation=generation,
            learn_skills=False,
            harness=parent_harness,
        )

        children: list[HarnessPolicy] = []
        child_harnesses: dict[str, EvolvingForecastHarness] = {}
        child_screen_train: dict[str, PolicyEvaluation] = {}
        child_screen_dev: dict[str, PolicyEvaluation] = {}
        child_policies: dict[str, HarnessPolicy] = {}
        for child_index in range(self.config.children_per_generation):
            self._progress("candidate_started", generation=generation, child=child_index)
            child = self.mutate(incumbent, parent_train, child_index=child_index)
            children.append(child)
            try:
                child_harness = self.harness_factory(child)
                child_harnesses[child.version] = child_harness
                child_screen_train[child.version] = self._evaluate(
                    child,
                    screen_train_tasks,
                    stage="child_screen_train",
                    generation=generation,
                    learn_skills=True,
                    harness=child_harness,
                )
                screen_child = snapshot_policy_skills(child, child_harness)
                child_policies[child.version] = screen_child
                child_screen_dev[child.version] = self._evaluate(
                    screen_child,
                    screen_dev_tasks,
                    stage="child_screen_dev",
                    generation=generation,
                    learn_skills=False,
                    harness=child_harness,
                )
                self._progress(
                    "candidate_screen_completed",
                    generation=generation,
                    child=child.version,
                    train_reward=child_screen_train[child.version].system_reward,
                    dev_reward=child_screen_dev[child.version].system_reward,
                )
            except TransientLLMError:
                raise
            except Exception as exc:
                self._progress(
                    "candidate_failed",
                    generation=generation,
                    child=child.version,
                    stage="screen",
                    error=f"{type(exc).__name__}: {exc}",
                )

        threshold = parent_screen_dev.system_reward - self.config.screening_tolerance
        eligible = sorted(
            (
                child
                for child in children
                if child.version in child_screen_dev
                and child_screen_dev[child.version].system_reward >= threshold
            ),
            key=lambda child: child_screen_dev[child.version].system_reward,
            reverse=True,
        )
        promote_count = max(0, min(self.config.screening_promote, len(eligible)))
        promoted = eligible[:promote_count]
        promoted_versions = {child.version for child in promoted}
        prune_reasons: dict[str, str] = {}
        for child in children:
            if child.version not in child_screen_dev:
                prune_reasons[child.version] = "screen_failed"
            elif child_screen_dev[child.version].system_reward < threshold:
                prune_reasons[child.version] = "below_parent_tolerance"
            elif child.version not in promoted_versions:
                prune_reasons[child.version] = "not_top_k"
            else:
                self._progress(
                    "candidate_promoted",
                    generation=generation,
                    child=child.version,
                    screen_dev_reward=child_screen_dev[child.version].system_reward,
                )
                continue
            self._progress(
                "candidate_pruned_after_screen",
                generation=generation,
                child=child.version,
                reason=prune_reasons[child.version],
                parent_screen_dev_reward=parent_screen_dev.system_reward,
                child_screen_dev_reward=(
                    child_screen_dev[child.version].system_reward
                    if child.version in child_screen_dev
                    else None
                ),
            )

        train_evaluations: dict[str, PolicyEvaluation] = {}
        dev_evaluations: dict[str, PolicyEvaluation] = {}
        full_policies: dict[str, HarnessPolicy] = {}
        for child in promoted:
            screen_child = child_policies[child.version]
            child_harness = child_harnesses[child.version]
            if remaining_train_tasks:
                remaining = self._evaluate(
                    screen_child,
                    remaining_train_tasks,
                    stage="child_train_remaining",
                    generation=generation,
                    learn_skills=True,
                    harness=child_harness,
                )
                train_evaluations[child.version] = combine_policy_evaluations(
                    child.version,
                    (
                        (child_screen_train[child.version], len(screen_train_tasks)),
                        (remaining, len(remaining_train_tasks)),
                    ),
                )
            else:
                train_evaluations[child.version] = child_screen_train[child.version]
            full_child = snapshot_policy_skills(screen_child, child_harness)
            full_policies[child.version] = full_child
            if train_evaluations[child.version].system_reward <= parent_train.system_reward:
                prune_reasons[child.version] = "full_train_not_improved"
                self._progress(
                    "candidate_pruned_before_dev",
                    generation=generation,
                    child=child.version,
                    parent_train_reward=parent_train.system_reward,
                    child_train_reward=train_evaluations[child.version].system_reward,
                )
                continue
            try:
                dev_evaluations[child.version] = self._evaluate(
                    full_child,
                    dev_tasks,
                    stage="child_dev",
                    generation=generation,
                    learn_skills=False,
                    harness=child_harness,
                )
            except TransientLLMError:
                raise
            except Exception as exc:
                prune_reasons[child.version] = "full_dev_failed"
                self._progress(
                    "candidate_dev_failed",
                    generation=generation,
                    child=child.version,
                    error=f"{type(exc).__name__}: {exc}",
                )

        train_best = (
            max(
                (full_policies[version] for version in dev_evaluations),
                key=lambda policy: dev_evaluations[policy.version].system_reward,
            )
            if dev_evaluations
            else None
        )
        child_dev = dev_evaluations.get(train_best.version) if train_best else None
        accepted = (
            train_best
            if train_best is not None
            and child_dev is not None
            and child_dev.system_reward > parent_dev.system_reward
            else trained_parent
        )
        step = EvolutionStep(
            mode=self.config.mode,
            generation=generation,
            parent_version=incumbent.version,
            child_versions=tuple(child.version for child in children),
            target_agent=self.target_agent(parent_train),
            parent_train_reward=parent_train.system_reward,
            child_train_rewards={
                child.version: (
                    train_evaluations.get(child.version)
                    or child_screen_train.get(child.version)
                ).system_reward
                for child in children
                if child.version in train_evaluations or child.version in child_screen_train
            },
            parent_dev_reward=parent_dev.system_reward,
            best_child_dev_reward=(child_dev.system_reward if child_dev else None),
            accepted_version=accepted.version,
            parent_train_module_rewards=parent_train.module_rewards,
            parent_dev_module_rewards=parent_dev.module_rewards,
            best_child_train_module_rewards=(
                train_evaluations[train_best.version].module_rewards
                if train_best is not None
                else None
            ),
            best_child_dev_module_rewards=(
                child_dev.module_rewards if child_dev is not None else None
            ),
            parent_train_diagnostics=parent_train.diagnostics,
            parent_dev_diagnostics=parent_dev.diagnostics,
            best_child_train_diagnostics=(
                train_evaluations[train_best.version].diagnostics
                if train_best is not None
                else None
            ),
            best_child_dev_diagnostics=(
                child_dev.diagnostics if child_dev is not None else None
            ),
            child_changelogs={child.version: child.changelog for child in children},
            successive_halving=True,
            parent_screen_train_reward=parent_screen_train.system_reward,
            parent_screen_dev_reward=parent_screen_dev.system_reward,
            child_screen_train_rewards={
                version: evaluation.system_reward
                for version, evaluation in child_screen_train.items()
            },
            child_screen_dev_rewards={
                version: evaluation.system_reward
                for version, evaluation in child_screen_dev.items()
            },
            promoted_versions=tuple(child.version for child in promoted),
            screen_prune_reasons=prune_reasons,
        )
        return accepted, step, child_dev.system_reward if child_dev else None

    def _evaluate(
        self,
        policy: HarnessPolicy,
        tasks: Sequence[ContextTask],
        *,
        stage: str,
        generation: int,
        learn_skills: bool,
        harness: EvolvingForecastHarness,
    ) -> PolicyEvaluation:
        self._progress(
            "evaluation_started",
            generation=generation,
            stage=stage,
            policy=policy.version,
            task_count=len(tasks),
        )

        def task_progress(event: str, payload: dict) -> None:
            self._progress(
                event,
                generation=generation,
                stage=stage,
                policy=policy.version,
                **payload,
            )

        try:
            result = evaluate_policy(
                policy,
                tasks,
                self.harness_factory,
                learn_skills=learn_skills,
                harness=harness,
                progress=task_progress,
            )
        except TransientLLMError as exc:
            self._progress(
                "infrastructure_interrupted",
                generation=generation,
                stage=stage,
                policy=policy.version,
                action="run_paused_without_scoring_candidate",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._progress(
            "evaluation_completed",
            generation=generation,
            stage=stage,
            policy=policy.version,
            reward=result.system_reward,
            module_rewards=result.module_rewards,
            diagnostics=result.diagnostics,
            failure_traces=result.failure_traces,
        )
        return result

    def _progress(self, event: str, **payload) -> None:
        if self.config.progress_path is None:
            return
        destination = Path(self.config.progress_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _save_checkpoint(
        self,
        incumbent: HarnessPolicy,
        history: list[EvolutionStep],
        next_generation: int,
    ) -> None:
        if self.config.checkpoint_path is None:
            return
        destination = Path(self.config.checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "mode": self.config.mode,
                    "target": self.config.target,
                    "successive_halving": self._successive_halving_signature(),
                    "next_generation": next_generation,
                    "incumbent": asdict(incumbent),
                    "history": [asdict(item) for item in history],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(destination)

    def _load_checkpoint(
        self, seed: HarnessPolicy
    ) -> tuple[HarnessPolicy, list[EvolutionStep], int]:
        if (
            not self.config.resume
            or self.config.checkpoint_path is None
            or not Path(self.config.checkpoint_path).exists()
        ):
            return seed, [], 0
        payload = json.loads(Path(self.config.checkpoint_path).read_text(encoding="utf-8"))
        if payload.get("mode") != self.config.mode:
            raise ValueError("checkpoint evolution mode does not match this run")
        if payload.get("target", "auto") != self.config.target:
            raise ValueError("checkpoint evolution target does not match this run")
        stored_halving = payload.get("successive_halving")
        current_halving = self._successive_halving_signature()
        if stored_halving is None:
            if current_halving["enabled"]:
                raise ValueError(
                    "checkpoint successive-halving controls do not match this run"
                )
        elif stored_halving != current_halving:
            raise ValueError("checkpoint successive-halving controls do not match this run")
        incumbent_payload = dict(payload["incumbent"])
        incumbent_payload["workflow"] = tuple(incumbent_payload["workflow"])
        for field in ("coding_skills", "retrieval_skills", "decision_skills"):
            incumbent_payload[field] = tuple(incumbent_payload.get(field, ()))
        incumbent = HarnessPolicy(**incumbent_payload)
        history = []
        for raw in payload.get("history", []):
            item = dict(raw)
            item["child_versions"] = tuple(item["child_versions"])
            item["promoted_versions"] = tuple(item.get("promoted_versions", ()))
            history.append(EvolutionStep(**item))
        versions = [incumbent.version]
        for item in history:
            versions.extend(item.child_versions)
        numeric_versions = [
            int(value[1:])
            for value in versions
            if value.startswith("v") and value[1:].isdigit()
        ]
        self._version = max(numeric_versions, default=0) + 1
        start = int(payload.get("next_generation", len(history)))
        self._progress("checkpoint_resumed", next_generation=start, incumbent=incumbent.version)
        return incumbent, history, start

    def _successive_halving_signature(self) -> dict[str, bool | int | float]:
        return {
            "enabled": self.config.successive_halving,
            "screening_train_tasks": self.config.screening_train_tasks,
            "screening_dev_tasks": self.config.screening_dev_tasks,
            "screening_promote": self.config.screening_promote,
            "screening_tolerance": self.config.screening_tolerance,
        }
