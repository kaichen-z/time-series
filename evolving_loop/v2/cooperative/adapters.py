"""Typed coordinate adapters for cooperative Evolution V2."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.data import ContextTask
from evolving_loop.package_numerical_supply import NumericalSupplyRelease
from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from evolving_loop.package_registry import FrozenNumericalPackageRegistry, _digest
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent

from ..bundle import EvolutionBundleV2
from ..contracts import SanitizedEvolutionFeedback, fingerprint_payload, require_sha256
from ..numerical_qd.adapters import FrozenNumericalArtifactsV2
from ..numerical_qd.contracts import FrozenNumericalRegistryEnvelopeV2
from .contracts import DecisionModuleV2, RetrievalModuleV2


ROUND1_NEXT = MappingProxyType({
    "timeline_first": "entity_first",
    "entity_first": "contrastive",
    "contrastive": "timeline_first",
})
ROUND2_NEXT = MappingProxyType({
    "counterevidence_first": "gap_first",
    "gap_first": "causal_chain_first",
    "causal_chain_first": "counterevidence_first",
})
TRIGGER_NEXT = MappingProxyType({
    "on_named_gap": "on_incomplete_chain",
    "on_incomplete_chain": "always",
    "always": "never",
    "never": "on_named_gap",
})
_COOPERATIVE_STAGE_PREFIX = "cooperative_stage_"
_COOPERATIVE_STAGES = frozenset({"train", "dev"})


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


class CooperativeArtifactCatalog:
    """Keep exact runtime objects beside their canonical persisted payloads."""

    def __init__(self, write_object: Callable[[str, object], object]) -> None:
        if not callable(write_object):
            raise ValueError("write_object must be callable")
        self._write_object = write_object
        self._numerical: dict[tuple[str, str], FrozenNumericalArtifactsV2] = {}
        self._retrieval: dict[str, RetrievalModuleV2] = {}
        self._decision: dict[str, DecisionModuleV2] = {}

    def add_numerical(
        self, artifacts: FrozenNumericalArtifactsV2
    ) -> tuple[str, str]:
        if type(artifacts) is not FrozenNumericalArtifactsV2:
            raise ValueError("Numerical artifact must be a frozen pair")
        if (
            type(artifacts.release) is not NumericalSupplyRelease
            or type(artifacts.registry) is not FrozenNumericalPackageRegistry
            or type(artifacts.envelope) is not FrozenNumericalRegistryEnvelopeV2
        ):
            raise ValueError("exact Numerical release, registry, and envelope required")
        selected = artifacts.selected_genome_sha256s
        if type(selected) is not tuple or selected != tuple(sorted(set(selected))):
            raise ValueError("exact Numerical selected Genome SHA tuple required")
        for identity in selected:
            require_sha256(identity, "selected Numerical Genome SHA")
        release_sha = artifacts.release.fingerprint
        registry_sha = artifacts.registry.fingerprint
        if release_sha != artifacts.registry.release_sha256:
            raise ValueError("Numerical release/registry mismatch")
        if artifacts.envelope.release_sha256 != release_sha:
            raise ValueError("Numerical release/envelope mismatch")
        if registry_sha != artifacts.envelope.registry_sha256:
            raise ValueError("Numerical registry/envelope mismatch")
        registry_payload = _plain_json(artifacts.registry.manifest)
        if _digest(registry_payload) != registry_sha:
            raise ValueError("Numerical registry content mismatch")
        key = (release_sha, registry_sha)
        existing = self._numerical.get(key)
        if existing is not None and (
            existing.envelope.fingerprint() != artifacts.envelope.fingerprint()
            or existing.selected_genome_sha256s != selected
        ):
            raise ValueError("conflicting Numerical artifact pair")
        release_payload = artifacts.release.to_payload()
        self._write_object(fingerprint_payload(release_payload), release_payload)
        self._write_object(fingerprint_payload(registry_payload), registry_payload)
        self._write_object(
            artifacts.envelope.fingerprint(), artifacts.envelope.to_payload()
        )
        if existing is None:
            self._numerical[key] = artifacts
        return release_sha, registry_sha

    def add_retrieval(self, artifact: RetrievalModuleV2) -> str:
        if type(artifact) is not RetrievalModuleV2:
            raise ValueError("Retrieval artifact must be a RetrievalModuleV2")
        identity = artifact.fingerprint()
        self._write_object(identity, artifact.to_payload())
        self._retrieval[identity] = artifact
        return identity

    def add_decision(self, artifact: DecisionModuleV2) -> str:
        if type(artifact) is not DecisionModuleV2:
            raise ValueError("Decision artifact must be a DecisionModuleV2")
        identity = artifact.fingerprint()
        self._write_object(identity, artifact.to_payload())
        self._decision[identity] = artifact
        return identity

    def resolve_numerical(
        self, release_sha256: str, registry_sha256: str
    ) -> FrozenNumericalArtifactsV2:
        require_sha256(release_sha256, "Numerical release SHA")
        require_sha256(registry_sha256, "Numerical registry SHA")
        try:
            artifact = self._numerical[(release_sha256, registry_sha256)]
        except KeyError as error:
            raise ValueError("Numerical artifact pair is missing") from error
        if (
            artifact.release.fingerprint != release_sha256
            or artifact.registry.fingerprint != registry_sha256
            or artifact.registry.release_sha256 != release_sha256
            or artifact.envelope.registry_sha256 != registry_sha256
        ):
            raise ValueError("Numerical artifact pair is mismatched")
        return artifact

    def resolve_retrieval(self, identity: str) -> RetrievalModuleV2:
        require_sha256(identity, "Retrieval artifact SHA")
        try:
            artifact = self._retrieval[identity]
        except KeyError as error:
            raise ValueError("Retrieval artifact is missing") from error
        if artifact.fingerprint() != identity:
            raise ValueError("Retrieval artifact is mismatched")
        return artifact

    def resolve_decision(self, identity: str) -> DecisionModuleV2:
        require_sha256(identity, "Decision artifact SHA")
        try:
            artifact = self._decision[identity]
        except KeyError as error:
            raise ValueError("Decision artifact is missing") from error
        if artifact.fingerprint() != identity:
            raise ValueError("Decision artifact is mismatched")
        return artifact


class NumericalCoordinateAdapter:
    """Select the first canonical frozen pair that differs from the Parent."""

    def __init__(self, alternatives: Sequence[FrozenNumericalArtifactsV2]) -> None:
        values = tuple(alternatives)
        if any(type(value) is not FrozenNumericalArtifactsV2 for value in values):
            raise ValueError("Numerical alternatives must be frozen pairs")
        self._alternatives = tuple(
            sorted(
                values,
                key=lambda value: (
                    value.release.fingerprint,
                    value.registry.fingerprint,
                ),
            )
        )

    def propose(
        self, parent: FrozenNumericalArtifactsV2
    ) -> FrozenNumericalArtifactsV2 | None:
        if type(parent) is not FrozenNumericalArtifactsV2:
            raise ValueError("Numerical Parent must be a frozen pair")
        parent_identity = (parent.release.fingerprint, parent.registry.fingerprint)
        return next(
            (
                value
                for value in self._alternatives
                if (value.release.fingerprint, value.registry.fingerprint)
                != parent_identity
            ),
            None,
        )


class RetrievalCoordinateAdapter:
    """Apply one deterministic, bounded mutation to a Retrieval Genome."""

    def propose(self, parent: RetrievalModuleV2, step: int) -> RetrievalModuleV2:
        if type(parent) is not RetrievalModuleV2:
            raise ValueError("Retrieval Parent must be a RetrievalModuleV2")
        if type(step) is not int or step < 0:
            raise ValueError("Retrieval step must be a non-negative integer")
        genome = parent.genome
        changes: dict[str, object] = {
            "version": f"v{int(genome.version[1:]) + 1:03d}",
            "parent": genome.version,
        }
        coordinate = step % 4
        if coordinate == 0:
            changes["round1_strategy"] = ROUND1_NEXT[genome.round1_strategy]
        elif coordinate == 1:
            changes["round2_strategy"] = ROUND2_NEXT[genome.round2_strategy]
        elif coordinate == 2:
            changes["second_round_trigger"] = TRIGGER_NEXT[genome.second_round_trigger]
        else:
            changes["max_selected_documents"] = (
                genome.max_selected_documents % 20
            ) + 1
        child = replace(genome, **changes)
        return RetrievalModuleV2(
            schema_version=1,
            source_release_sha256=parent.source_release_sha256,
            genome_payload=child.to_payload(),
            skills_payload=parent.skills_payload,
        )


class DecisionCoordinateAdapter:
    """Change one Decision-owned prompt or bounded setting."""

    def __init__(
        self,
        prompts: Sequence[str],
        *,
        settings_cycle: bool = False,
    ) -> None:
        values = tuple(prompts)
        if any(type(value) is not str or not value.strip() for value in values):
            raise ValueError("Decision prompts must be non-empty strings")
        if type(settings_cycle) is not bool:
            raise ValueError("settings_cycle must be a boolean")
        if not values and not settings_cycle:
            raise ValueError("Decision adapter requires prompts or settings")
        self._prompts = values
        self._settings_cycle = settings_cycle

    def propose(
        self, parent: DecisionModuleV2, step: int
    ) -> DecisionModuleV2 | None:
        if type(parent) is not DecisionModuleV2:
            raise ValueError("Decision Parent must be a DecisionModuleV2")
        if type(step) is not int or step < 0:
            raise ValueError("Decision step must be a non-negative integer")
        if self._settings_cycle:
            if step % 2 == 0:
                return replace(
                    parent,
                    max_evidence_adjustments=(parent.max_evidence_adjustments + 1) % 4,
                )
            return replace(
                parent,
                aggregation="mean" if parent.aggregation == "last" else "last",
            )
        index = step % len(self._prompts)
        prompt = self._prompts[index]
        if prompt == parent.prompt:
            prompt = self._prompts[(index + 1) % len(self._prompts)]
        return None if prompt == parent.prompt else replace(parent, prompt=prompt)


class CooperativePipelineAdapter:
    """Resolve one Bundle and execute the existing complete package pipeline."""

    def __init__(
        self,
        catalog: CooperativeArtifactCatalog,
        retrieval_factory: Callable[
            [RetrievalGenome, RetrievalSkillLibrary], TwoStageRetrievalAgent
        ],
        decision_factory: Callable[[DecisionModuleV2], DecisionAgent],
        *,
        metric_cap: float = 5.0,
        retrieval_skill_library: RetrievalSkillLibrary | None = None,
        empty_skill_path: str | Path = "unused-cooperative-retrieval-skills.json",
    ) -> None:
        if type(catalog) is not CooperativeArtifactCatalog:
            raise TypeError("pipeline catalog must be CooperativeArtifactCatalog")
        if not callable(retrieval_factory) or not callable(decision_factory):
            raise ValueError("pipeline agent factories must be callable")
        if (
            isinstance(metric_cap, bool)
            or not isinstance(metric_cap, (int, float))
            or not math.isfinite(metric_cap)
            or metric_cap <= 0.0
        ):
            raise ValueError("pipeline metric_cap must be positive and finite")
        if retrieval_skill_library is not None and type(
            retrieval_skill_library
        ) is not RetrievalSkillLibrary:
            raise TypeError("runtime Skill library must be RetrievalSkillLibrary")
        self.catalog = catalog
        self.retrieval_factory = retrieval_factory
        self.decision_factory = decision_factory
        self.metric_cap = float(metric_cap)
        self.retrieval_skill_library = retrieval_skill_library
        self.empty_skill_path = Path(empty_skill_path)

    def _skills_for(self, module: RetrievalModuleV2) -> RetrievalSkillLibrary:
        source = self.retrieval_skill_library
        if module.skills_payload and source is None:
            raise ValueError(
                "nonempty Retrieval Skills require a verified Retrieval Skill library"
            )
        if source is None:
            return RetrievalSkillLibrary(
                self.empty_skill_path, persist=False
            ).clone(read_only=True)

        snapshot = source.frozen_execution_snapshot()
        payloads = tuple(skill.to_payload() for skill in snapshot.all())
        if payloads != tuple(dict(row) for row in module.skills_payload):
            raise ValueError(
                "verified Retrieval Skill library does not match module skills_payload"
            )
        if tuple(skill.skill_id for skill in snapshot.active_skills()) != tuple(
            module.genome.active_skill_ids
        ):
            raise ValueError(
                "verified Retrieval Skill library active IDs do not match module Genome"
            )
        return snapshot

    def evaluate(
        self,
        bundle: EvolutionBundleV2,
        tasks: Sequence[ContextTask],
        stage: str,
    ) -> PackageEvaluation:
        if type(bundle) is not EvolutionBundleV2:
            raise TypeError("pipeline Bundle must be EvolutionBundleV2")
        if type(stage) is not str or stage not in _COOPERATIVE_STAGES:
            raise ValueError("cooperative pipeline stage must be exactly train or dev")
        numerical = self.catalog.resolve_numerical(
            bundle.numerical_release_sha256,
            bundle.numerical_registry_sha256,
        )
        retrieval = self.catalog.resolve_retrieval(bundle.retrieval_release_sha256)
        decision = self.catalog.resolve_decision(bundle.decision_policy_sha256)
        skills = self._skills_for(retrieval)

        evaluation = PackagePipelineEvaluator._evaluate_components(
            candidate_sha256=bundle.fingerprint(),
            registry=numerical.registry,
            tasks=tuple(tasks),
            stage=stage,
            retrieval_factory=lambda: self.retrieval_factory(
                retrieval.genome, skills
            ),
            decision_factory=lambda: self.decision_factory(decision),
            metric_cap=self.metric_cap,
            expected_retrieval_sha256=retrieval.genome.fingerprint(),
            expected_decision_prompt_sha256=hashlib.sha256(
                decision.prompt.encode("utf-8")
            ).hexdigest(),
        )
        diagnostics = dict(evaluation.secondary_diagnostics)
        if any(name.startswith(_COOPERATIVE_STAGE_PREFIX) for name in diagnostics):
            raise ValueError("package evaluator returned a reserved stage marker")
        diagnostics[f"{_COOPERATIVE_STAGE_PREFIX}{stage}"] = 1.0
        return PackageEvaluation.from_rows(
            evaluation.candidate_sha256,
            evaluation.task_rows,
            evaluation.expected_task_ids,
            diagnostics,
        )


def sanitize_train_feedback(
    parent_eval: PackageEvaluation,
    child_eval: PackageEvaluation,
    normalized_cost: float,
) -> SanitizedEvolutionFeedback:
    """Reduce complete Train results to aggregate proposer-visible feedback."""
    if type(parent_eval) is not PackageEvaluation or type(
        child_eval
    ) is not PackageEvaluation:
        raise TypeError("Train feedback requires PackageEvaluation values")
    if parent_eval.public_test_accessed or child_eval.public_test_accessed:
        raise ValueError("Train feedback cannot contain Public evaluation")
    for evaluation in (parent_eval, child_eval):
        stage_markers = {
            name: value
            for name, value in evaluation.secondary_diagnostics.items()
            if name.startswith(_COOPERATIVE_STAGE_PREFIX)
        }
        if stage_markers != {"cooperative_stage_train": 1.0}:
            raise ValueError("Train feedback requires exact Train-marked evaluations")
    if parent_eval.expected_task_ids != child_eval.expected_task_ids:
        raise ValueError("Train feedback evaluations must cover the same task universe")
    if (
        isinstance(normalized_cost, bool)
        or not isinstance(normalized_cost, (int, float))
        or not math.isfinite(normalized_cost)
        or normalized_cost < 0.0
    ):
        raise ValueError("normalized_cost must be finite and non-negative")

    denominator = max(abs(parent_eval.mean_joint), 1e-12)
    improvement = (parent_eval.mean_joint - child_eval.mean_joint) / denominator
    counts = {
        "catastrophic_count": child_eval.catastrophic_count,
        "fallback_count": child_eval.fallback_count,
        "invalid_count": child_eval.invalid_count,
    }
    categories = tuple(
        name.removesuffix("_count") for name, count in counts.items() if count > 0
    )
    return SanitizedEvolutionFeedback(
        parent_sha256=parent_eval.candidate_sha256,
        train_evaluation_sha256=child_eval.fingerprint,
        train_objectives={
            "normalized_cost": float(normalized_cost),
            "relative_joint_improvement": float(improvement),
        },
        train_behavior_descriptors=counts,
        failure_categories=categories,
        remaining_proposal_budget={},
    )


__all__ = [
    "CooperativeArtifactCatalog",
    "DecisionCoordinateAdapter",
    "CooperativePipelineAdapter",
    "NumericalCoordinateAdapter",
    "RetrievalCoordinateAdapter",
    "ROUND1_NEXT",
    "ROUND2_NEXT",
    "sanitize_train_feedback",
]
