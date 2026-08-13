"""Delayed-outcome, failure-attributed evolution of the three-agent harness."""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Sequence

from evolving_agent.coding_agent.evolution import GENERATION_PROMPT, REVISION_PROMPT
from evolving_agent.data import ContextTask, Document
from evolving_agent.decision_agent.agent import DECISION_PROMPT
from evolving_agent.evaluation import ResolvedOutcome, score_after_resolution
from evolving_agent.harness import EvolvingForecastHarness
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

PROMPT_ONLY_EVOLVER_PROMPT = """You are a constrained Prompt Evolver for a time-series agent
harness. Use resolved training failures to replace exactly one complete prompt owned by the
diagnosed weakest role. You may not change another role, any numeric/search budget, topology,
source code, scorer, data boundary, or safety mechanism. Return exactly:
{"prompt_field": "coding_generation_prompt|coding_revision_prompt|retrieval_prompt|decision_prompt",
"replacement_prompt": "complete replacement prompt", "changelog": "testable rationale"}
"""

EvolutionMode = Literal["prompt", "genome"]


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
    child_changelogs: dict[str, str] | None = None


@dataclass(frozen=True)
class CoEvolutionConfig:
    generations: int = 3
    children_per_generation: int = 2
    max_workflow_stages: int = 8
    mode: EvolutionMode = "genome"
    checkpoint_path: str | Path | None = None
    progress_path: str | Path | None = None
    resume: bool = True


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
        version = f"v{self._version:03d}"
        self._version += 1
        try:
            response = self.llm.complete(
                system=(
                    PROMPT_ONLY_EVOLVER_PROMPT
                    if self.config.mode == "prompt"
                    else META_HARNESS_PROMPT
                ),
                messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                temperature=0.4,
            )
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
        if candidate is None:
            return replace(parent, version=version, parent=parent.version, changelog=reason)
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
            parent_harness = self.harness_factory(incumbent)
            parent_train = self._evaluate(
                incumbent,
                train_tasks,
                stage="parent_train",
                generation=generation,
                learn_skills=True,
                harness=parent_harness,
            )
            parent_dev = self._evaluate(
                incumbent,
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
                child = self.mutate(incumbent, parent_train)
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
                    self._progress(
                        "candidate_completed",
                        generation=generation,
                        child=child.version,
                        train_reward=train_evaluations[child.version].system_reward,
                    )
                except Exception as exc:
                    self._progress(
                        "candidate_failed",
                        generation=generation,
                        child=child.version,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            valid_children = [child for child in children if child.version in train_evaluations]
            train_best = (
                max(valid_children, key=lambda item: train_evaluations[item.version].system_reward)
                if valid_children
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
                else incumbent
            )
            history.append(
                EvolutionStep(
                    mode=self.config.mode,
                    generation=generation,
                    parent_version=incumbent.version,
                    child_versions=tuple(child.version for child in children),
                    target_agent=self.weakest_agent(parent_train),
                    parent_train_reward=parent_train.system_reward,
                    child_train_rewards={
                        version: item.system_reward for version, item in train_evaluations.items()
                    },
                    parent_dev_reward=parent_dev.system_reward,
                    best_child_dev_reward=(child_dev.system_reward if child_dev else None),
                    accepted_version=accepted.version,
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

        result = evaluate_policy(
            policy,
            tasks,
            self.harness_factory,
            learn_skills=learn_skills,
            harness=harness,
            progress=task_progress,
        )
        self._progress(
            "evaluation_completed",
            generation=generation,
            stage=stage,
            policy=policy.version,
            reward=result.system_reward,
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
        incumbent_payload = dict(payload["incumbent"])
        incumbent_payload["workflow"] = tuple(incumbent_payload["workflow"])
        incumbent = HarnessPolicy(**incumbent_payload)
        history = []
        for raw in payload.get("history", []):
            item = dict(raw)
            item["child_versions"] = tuple(item["child_versions"])
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
