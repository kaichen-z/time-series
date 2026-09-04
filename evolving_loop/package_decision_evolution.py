"""Decision-only evolution over frozen Champion Numerical packages."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from common.llm import LLMClient, TransientLLMError
from evolving_loop.co_evolution import (
    CoEvolutionConfig,
    CoEvolutionEngine,
    EvolutionStep,
    HarnessPolicy,
    PolicyEvaluation,
    evaluation_diagnostics,
)
from evolving_loop.coordinate_evolution import principal_module_fingerprints
from evolving_loop.data import ContextTask
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.evaluation import ResolvedOutcome
from evolving_loop.package_candidate_proposal import (
    PackageCandidate,
    PackageProposalFeedback,
    proposal_fingerprint,
)
from evolving_loop.package_coordinate_evolution import (
    PackageCoordinateState,
    package_principal_fingerprints,
)
from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DEPENDENCY_KEYS = frozenset(
    {"retrieval_factory", "decision_factory", "bridge_runtime"}
)


class PackageDecisionEvolutionError(ValueError):
    """Raised when Decision evolution crosses a frozen package boundary."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


class PackageDecisionEvaluator:
    """Score Decision policies without constructing an old harness result."""

    def __init__(
        self,
        registry: FrozenNumericalPackageRegistry,
        retrieval_factory: Callable[[HarnessPolicy], TwoStageRetrievalAgent],
        decision_factory: Callable[[HarnessPolicy], DecisionAgent],
        *,
        dependency_fingerprints: Mapping[str, str],
        metric_cap: float = 5.0,
    ) -> None:
        if not isinstance(registry, FrozenNumericalPackageRegistry):
            raise PackageDecisionEvolutionError(
                "package Decision evaluator requires a frozen package registry"
            )
        if not callable(retrieval_factory) or not callable(decision_factory):
            raise PackageDecisionEvolutionError(
                "package Decision factories must be callable"
            )
        dependencies = dict(dependency_fingerprints)
        if set(dependencies) != _DEPENDENCY_KEYS or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in dependencies.values()
        ):
            raise PackageDecisionEvolutionError(
                "package Decision dependency fingerprints must be canonical"
            )
        if not math.isfinite(metric_cap) or metric_cap <= 0.0:
            raise PackageDecisionEvolutionError(
                "package Decision metric cap must be positive and finite"
            )
        self.registry = registry
        self.retrieval_factory = retrieval_factory
        self.decision_factory = decision_factory
        self.dependency_fingerprints = MappingProxyType(
            dict(sorted(dependencies.items()))
        )
        self.metric_cap = float(metric_cap)
        self.evaluator_hash = _digest(
            {
                "schema_version": 1,
                "registry": registry.fingerprint,
                "dependencies": dict(self.dependency_fingerprints),
                "metric_cap": self.metric_cap,
            }
        )

    def evaluate_package(
        self,
        policy: HarnessPolicy,
        tasks: Sequence[ContextTask],
        *,
        stage: str,
    ) -> PackageEvaluation:
        if not isinstance(policy, HarnessPolicy):
            raise PackageDecisionEvolutionError(
                "package Decision evaluation requires a HarnessPolicy"
            )
        if not policy.has_accepted_retrieval_release:
            raise PackageDecisionEvolutionError(
                "package Decision requires a non-v000 accepted Retrieval release"
            )
        return PackagePipelineEvaluator._evaluate_components(
            candidate_sha256=hashlib.sha256(policy.canonical_bytes()).hexdigest(),
            registry=self.registry,
            tasks=tasks,
            stage=stage,
            retrieval_factory=lambda: self.retrieval_factory(policy),
            decision_factory=lambda: self.decision_factory(policy),
            metric_cap=self.metric_cap,
            expected_retrieval_sha256=policy.retrieval_genome.fingerprint(),
            expected_decision_prompt_sha256=hashlib.sha256(
                policy.decision_prompt.encode("utf-8")
            ).hexdigest(),
        )

    def evaluate(
        self,
        policy: HarnessPolicy,
        tasks: Sequence[ContextTask],
    ) -> PolicyEvaluation:
        if not isinstance(policy, HarnessPolicy):
            raise PackageDecisionEvolutionError(
                "package Decision evaluation requires a HarnessPolicy"
            )
        if not policy.has_accepted_retrieval_release:
            raise PackageDecisionEvolutionError(
                "package Decision requires a non-v000 accepted Retrieval release"
            )
        resolved = tuple(tasks)
        package = self.evaluate_package(policy, resolved, stage="compatibility")
        rows = {row.task_id: row for row in package.task_rows}
        outcomes: list[ResolvedOutcome] = []
        for task in resolved:
            row = rows[task.numeric.task_id]
            oracle_smae = row.numerical_oracle_smae
            oracle_srmse = row.numerical_oracle_srmse
            outcomes.append(
                ResolvedOutcome(
                    task_id=task.numeric.task_id,
                    final_smae=row.final_smae,
                    final_srmse=row.final_srmse,
                    coding_oracle_smae=oracle_smae,
                    coding_oracle_srmse=oracle_srmse,
                    contextual_oracle_smae=oracle_smae,
                    contextual_oracle_srmse=oracle_srmse,
                    decision_selection_smae_regret=row.final_smae - oracle_smae,
                    decision_selection_srmse_regret=row.final_srmse - oracle_srmse,
                    candidate_count=row.numerical_candidate_count,
                )
            )
        outcomes_tuple = tuple(outcomes)
        diagnostics = evaluation_diagnostics(outcomes_tuple)
        diagnostics.update(
            {
                "p90_smae": package.p90_smae,
                "p95_smae": package.p95_smae,
                "p90_srmse": package.p90_srmse,
                "p95_srmse": package.p95_srmse,
                "invalid_count": float(package.invalid_count),
                "catastrophic_count": float(package.catastrophic_count),
                "clipped_count": float(package.clipped_count),
                "fallback_count": float(package.fallback_count),
                "coverage": package.coverage,
                "public_test_accessed": 0.0,
            }
        )
        reward = -float(diagnostics["mean_srmse"])
        return PolicyEvaluation(
            version=policy.version,
            system_reward=reward,
            module_rewards={"coding": 0.0, "retrieval": 0.0, "decision": reward},
            outcomes=outcomes_tuple,
            failure_traces=tuple(row.to_payload() for row in package.task_rows),
            diagnostics=diagnostics,
        )


def package_decision_gate_failures(
    child: PolicyEvaluation,
    parent: PolicyEvaluation,
    tolerance: float,
    *,
    require_strict: bool,
) -> tuple[str, ...]:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise PackageDecisionEvolutionError("Decision tolerance must be non-negative")
    failures: list[str] = []
    if child.mean_smae > parent.mean_smae + tolerance:
        failures.append("mean_smae")
    if child.mean_srmse > parent.mean_srmse + tolerance:
        failures.append("mean_srmse")
    if require_strict and not (
        child.mean_smae < parent.mean_smae - tolerance
        or child.mean_srmse < parent.mean_srmse - tolerance
    ):
        failures.append("strict_final_gain")
    for field_name in (
        "p90_smae",
        "p95_smae",
        "invalid_count",
        "catastrophic_count",
        "fallback_count",
    ):
        if child.diagnostics.get(field_name, math.inf) > parent.diagnostics.get(
            field_name, math.inf
        ) + tolerance:
            failures.append(field_name)
    if child.diagnostics.get("public_test_accessed") != 0.0:
        failures.append("public_test_accessed")
    return tuple(failures)


def _unreachable_legacy_harness_factory(_policy: HarnessPolicy):
    raise AssertionError("package Decision evolution must not build a legacy harness")


class PackageDecisionEvolutionEngine(CoEvolutionEngine):
    """Reuse typed Decision mutation while replacing all harness evaluation."""

    def __init__(
        self,
        llm: LLMClient,
        evaluator: PackageDecisionEvaluator,
        config: CoEvolutionConfig | None = None,
    ) -> None:
        super().__init__(llm, _unreachable_legacy_harness_factory, config)
        if not isinstance(evaluator, PackageDecisionEvaluator):
            raise PackageDecisionEvolutionError(
                "package Decision engine requires PackageDecisionEvaluator"
            )
        if self.config.mode != "genome" or self.config.target != "decision":
            raise PackageDecisionEvolutionError(
                "package Decision engine must use genome mode and target=decision"
            )
        if self.config.checkpoint_path is not None or self.config.successive_halving:
            raise PackageDecisionEvolutionError(
                "package Decision engine does not accept legacy checkpoint/screen modes"
            )
        self.package_evaluator = evaluator

    @staticmethod
    def _decision_coordinate_only(
        parent: HarnessPolicy,
        child: HarnessPolicy,
    ) -> bool:
        if child.parent != parent.version or child.version == parent.version:
            return False
        parent_fingerprints = principal_module_fingerprints(parent)
        child_fingerprints = principal_module_fingerprints(child)
        changed = tuple(
            name
            for name in ("numerical_morphology", "retrieval", "decision")
            if child_fingerprints[name] != parent_fingerprints[name]
        )
        return changed == ("decision",)

    def evolve(
        self,
        seed: HarnessPolicy,
        train_tasks: Sequence[ContextTask],
        dev_tasks: Sequence[ContextTask],
    ) -> tuple[HarnessPolicy, tuple[EvolutionStep, ...]]:
        if not isinstance(seed, HarnessPolicy) or not seed.has_accepted_retrieval_release:
            raise PackageDecisionEvolutionError(
                "package Decision requires a non-v000 accepted Retrieval release"
            )
        train = tuple(train_tasks)
        dev = tuple(dev_tasks)
        if not train or not dev:
            raise PackageDecisionEvolutionError(
                "package Decision requires non-empty Train and Dev splits"
            )
        if not (
            seed.version.startswith("v") and seed.version[1:].isdigit()
        ):
            raise PackageDecisionEvolutionError(
                "package Decision bundle versions must use vNNN identities"
            )
        self._version = max(self._version, int(seed.version[1:]) + 1)
        incumbent = seed
        history: list[EvolutionStep] = []
        for generation in range(self.config.generations):
            parent_train = self.package_evaluator.evaluate(incumbent, train)
            children: list[HarnessPolicy] = []
            child_train: dict[str, PolicyEvaluation] = {}
            for child_index in range(self.config.children_per_generation):
                child = self.mutate(
                    incumbent,
                    parent_train,
                    child_index=child_index,
                )
                children.append(child)
                if not self._decision_coordinate_only(incumbent, child):
                    continue
                try:
                    evaluation = self.package_evaluator.evaluate(child, train)
                except TransientLLMError:
                    raise
                except Exception:
                    continue
                child_train[child.version] = evaluation
            eligible = tuple(
                child
                for child in children
                if child.version in child_train
                and not package_decision_gate_failures(
                    child_train[child.version],
                    parent_train,
                    self.config.screening_tolerance,
                    require_strict=True,
                )
            )
            train_winner = (
                min(
                    eligible,
                    key=lambda child: (
                        child_train[child.version].mean_srmse,
                        child_train[child.version].mean_smae,
                        child.version,
                    ),
                )
                if eligible
                else None
            )
            parent_dev = self.package_evaluator.evaluate(incumbent, dev)
            child_dev = (
                self.package_evaluator.evaluate(train_winner, dev)
                if train_winner is not None
                else None
            )
            accepted = bool(
                train_winner is not None
                and child_dev is not None
                and not package_decision_gate_failures(
                    child_dev,
                    parent_dev,
                    self.config.screening_tolerance,
                    require_strict=True,
                )
            )
            selected = train_winner if accepted and train_winner is not None else incumbent
            winner_train = (
                child_train[train_winner.version]
                if train_winner is not None
                else None
            )
            history.append(
                EvolutionStep(
                    mode=self.config.mode,
                    generation=generation,
                    parent_version=incumbent.version,
                    child_versions=tuple(child.version for child in children),
                    target_agent="decision",
                    parent_train_reward=parent_train.system_reward,
                    child_train_rewards={
                        version: evaluation.system_reward
                        for version, evaluation in child_train.items()
                    },
                    parent_dev_reward=parent_dev.system_reward,
                    best_child_dev_reward=(
                        child_dev.system_reward if child_dev is not None else None
                    ),
                    accepted_version=selected.version,
                    parent_train_module_rewards=parent_train.module_rewards,
                    parent_dev_module_rewards=parent_dev.module_rewards,
                    best_child_train_module_rewards=(
                        winner_train.module_rewards if winner_train is not None else None
                    ),
                    best_child_dev_module_rewards=(
                        child_dev.module_rewards if child_dev is not None else None
                    ),
                    parent_train_diagnostics=parent_train.diagnostics,
                    parent_dev_diagnostics=parent_dev.diagnostics,
                    best_child_train_diagnostics=(
                        winner_train.diagnostics if winner_train is not None else None
                    ),
                    best_child_dev_diagnostics=(
                        child_dev.diagnostics if child_dev is not None else None
                    ),
                    child_changelogs={
                        child.version: child.changelog for child in children
                    },
                )
            )
            incumbent = selected
        return incumbent, tuple(history)


def _proposal_evaluation(
    policy: HarnessPolicy,
    summary: Mapping[str, float | int],
) -> PolicyEvaluation:
    """Project aggregate package feedback onto the legacy mutation boundary."""
    diagnostics = {
        key: float(value)
        for key, value in summary.items()
        if type(value) in {int, float}
    }
    mean_srmse = diagnostics.get("mean_srmse", 0.0)
    return PolicyEvaluation(
        version=policy.version,
        system_reward=-mean_srmse,
        module_rewards={
            "coding": 0.0,
            "retrieval": 0.0,
            "decision": -mean_srmse,
        },
        outcomes=(),
        failure_traces=(),
        diagnostics=diagnostics,
    )


class DecisionCandidateProposer:
    """Adapt Decision-only mutation to isolated package Child records."""

    def __init__(self, engine: PackageDecisionEvolutionEngine) -> None:
        if not isinstance(engine, PackageDecisionEvolutionEngine):
            raise PackageDecisionEvolutionError(
                "package Decision adapter requires PackageDecisionEvolutionEngine"
            )
        self.engine = engine

    def propose(
        self,
        parent: PackageCoordinateState,
        feedback: PackageProposalFeedback,
        *,
        generation: int,
        child_count: int,
    ) -> tuple[PackageCandidate, ...]:
        if not isinstance(parent, PackageCoordinateState) or not isinstance(
            feedback, PackageProposalFeedback
        ):
            raise PackageDecisionEvolutionError(
                "package Decision proposal requires typed state and feedback"
            )
        if type(generation) is not int or generation < 0:
            raise PackageDecisionEvolutionError(
                "package Decision generation must be non-negative"
            )
        if type(child_count) is not int or child_count != 3:
            raise PackageDecisionEvolutionError(
                "formal package Decision proposal requires exactly three slots"
            )
        policy = parent.bundle.policy
        if not policy.has_accepted_retrieval_release:
            raise PackageDecisionEvolutionError(
                "package Decision requires an accepted Retrieval Parent"
            )
        if re.fullmatch(r"v[0-9]{3}", policy.version) is None:
            raise PackageDecisionEvolutionError(
                "package Decision Parent version must use vNNN"
            )
        first_version = int(policy.version[1:]) + generation * 3 + 1
        if first_version + 2 > 999:
            raise PackageDecisionEvolutionError(
                "package Decision proposal version namespace exhausted"
            )
        self.engine._version = first_version
        evaluation = _proposal_evaluation(policy, feedback.parent_summary)
        parent_fingerprints = package_principal_fingerprints(parent.bundle)
        parent_policy_fingerprints = principal_module_fingerprints(policy)
        seen_decision: set[str] = set()
        children: list[PackageCandidate] = []
        for slot in range(3):
            child_policy = self.engine.mutate(
                policy,
                evaluation,
                child_index=slot,
            )
            expected_version = f"v{first_version + slot:03d}"
            reason: str | None = None
            child_state = parent
            if (
                not isinstance(child_policy, HarnessPolicy)
                or child_policy.version != expected_version
                or child_policy.parent != policy.version
            ):
                reason = "invalid_schema"
            else:
                child_policy_fingerprints = principal_module_fingerprints(child_policy)
                if any(
                    child_policy_fingerprints[name]
                    != parent_policy_fingerprints[name]
                    for name in ("numerical_morphology", "retrieval")
                ):
                    reason = "cross_coordinate_change"
                elif (
                    child_policy_fingerprints["decision"]
                    == parent_policy_fingerprints["decision"]
                ):
                    reason = (
                        "unknown_candidate"
                        if "unknown_candidate" in child_policy.changelog
                        else "invalid_schema"
                    )
                else:
                    candidate_state = parent.with_policy(
                        child_policy,
                        target="decision",
                    )
                    fingerprints = package_principal_fingerprints(
                        candidate_state.bundle
                    )
                    decision_identity = fingerprints["decision"]
                    if decision_identity in seen_decision:
                        reason = "duplicate_child"
                    else:
                        child_state = candidate_state
                        seen_decision.add(decision_identity)
            raw_payload = (
                child_policy.to_payload()
                if isinstance(child_policy, HarnessPolicy)
                else {"invalid": type(child_policy).__name__}
            )
            identity = proposal_fingerprint(
                target="decision",
                generation=generation,
                slot=slot,
                payload={
                    "policy": raw_payload,
                    "invalid_reason": reason,
                },
            )
            children.append(
                PackageCandidate(
                    slot=slot,
                    target="decision",
                    state=child_state,
                    proposal_sha256=identity,
                    invalid_reason=reason,
                )
            )
        return tuple(children)


__all__ = [
    "DecisionCandidateProposer",
    "PackageDecisionEvaluator",
    "PackageDecisionEvolutionEngine",
    "PackageDecisionEvolutionError",
    "package_decision_gate_failures",
]
