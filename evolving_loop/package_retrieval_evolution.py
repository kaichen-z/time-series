"""Trusted Retrieval evaluation over frozen Champion Numerical packages."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from evolving_loop.data import ContextTask
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from evolving_loop.retrieval_agent.evolution import (
    RetrievalEvaluation,
    RetrievalEvolutionError,
    RetrievalInferenceCacheKey,
)
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DEPENDENCY_KEYS = frozenset(
    {"retrieval_factory", "decision_factory", "bridge_runtime"}
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


class PackageRetrievalEvaluator:
    """Implement `RetrievalEvaluator` without a legacy harness or skill writes."""

    def __init__(
        self,
        registry: FrozenNumericalPackageRegistry,
        retrieval_factory: Callable[
            [RetrievalGenome, RetrievalSkillLibrary], TwoStageRetrievalAgent
        ],
        decision_factory: Callable[[], DecisionAgent],
        *,
        dependency_fingerprints: Mapping[str, str],
    ) -> None:
        if not isinstance(registry, FrozenNumericalPackageRegistry):
            raise ValueError("package evaluator requires a frozen package registry")
        if not callable(retrieval_factory) or not callable(decision_factory):
            raise ValueError("package evaluator factories must be callable")
        dependencies = dict(dependency_fingerprints)
        if set(dependencies) != _DEPENDENCY_KEYS or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in dependencies.values()
        ):
            raise ValueError(
                "package evaluator dependency fingerprints must be canonical"
            )
        self.registry = registry
        self.retrieval_factory = retrieval_factory
        self.decision_factory = decision_factory
        self.dependency_fingerprints = MappingProxyType(
            dict(sorted(dependencies.items()))
        )
        self.evaluator_hash = _digest(
            {
                "schema_version": 1,
                "registry": registry.fingerprint,
                "dependencies": dict(self.dependency_fingerprints),
            }
        )
        self.verifier_hash = _digest(
            {
                "bridge_runtime": dependencies["bridge_runtime"],
                "contract": "package-final-retrieval-card-v1",
            }
        )

    def evaluate_package(
        self,
        genome: RetrievalGenome,
        tasks: Sequence[ContextTask],
        *,
        stage: str,
        skill_library: RetrievalSkillLibrary,
    ) -> PackageEvaluation:
        return self._evaluate_package(
            genome,
            tasks,
            stage=stage,
            skill_library=skill_library,
            metric_cap=5.0,
        )

    def _evaluate_package(
        self,
        genome: RetrievalGenome,
        tasks: Sequence[ContextTask],
        *,
        stage: str,
        skill_library: RetrievalSkillLibrary,
        metric_cap: float,
    ) -> PackageEvaluation:
        if not isinstance(genome, RetrievalGenome):
            raise TypeError("package evaluator requires a RetrievalGenome")
        if not isinstance(skill_library, RetrievalSkillLibrary) or not getattr(
            skill_library, "_read_only", False
        ):
            raise ValueError("package evaluator requires a read-only Skill library")
        before = tuple(skill.to_payload() for skill in skill_library.all())
        evaluation = PackagePipelineEvaluator._evaluate_components(
            candidate_sha256=genome.fingerprint(),
            registry=self.registry,
            tasks=tasks,
            stage=stage,
            retrieval_factory=lambda: self.retrieval_factory(genome, skill_library),
            decision_factory=self.decision_factory,
            metric_cap=metric_cap,
            expected_retrieval_sha256=genome.fingerprint(),
        )
        after = tuple(skill.to_payload() for skill in skill_library.all())
        if after != before:
            raise RetrievalEvolutionError(
                "package evaluator mutated the accepted Retrieval Skill snapshot"
            )
        return evaluation

    def evaluate(
        self,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        *,
        stage: str,
        skill_library: RetrievalSkillLibrary | None,
        harness_factory: Callable[..., object] | None,
        persist: bool,
        writers_enabled: bool,
        evolver_enabled: bool,
        cache_keys: tuple[RetrievalInferenceCacheKey, ...],
        metric_cap: float,
    ) -> RetrievalEvaluation:
        if not isinstance(genome, RetrievalGenome):
            raise TypeError("package evaluator requires a RetrievalGenome")
        if not isinstance(tasks, tuple) or not tasks:
            raise ValueError("package evaluator requires a non-empty task tuple")
        if not isinstance(stage, str) or not stage:
            raise ValueError("package evaluator requires an evaluation stage")
        if harness_factory is not None:
            raise ValueError("package evaluator forbids a legacy harness factory")
        if persist or writers_enabled or evolver_enabled:
            raise ValueError("package evaluator is read-only and forbids writers")
        if not isinstance(skill_library, RetrievalSkillLibrary) or not getattr(
            skill_library, "_read_only", False
        ):
            raise ValueError("package evaluator requires a read-only Skill library")
        if len(cache_keys) != len(tasks) or any(
            not isinstance(cache_key, RetrievalInferenceCacheKey)
            or cache_key.task_id != task.numeric.task_id
            or cache_key.genome_sha256 != genome.fingerprint()
            or not math.isclose(cache_key.metric_cap, metric_cap)
            for cache_key, task in zip(cache_keys, tasks, strict=True)
        ):
            raise ValueError("package evaluator cache keys do not bind the task batch")
        if not math.isfinite(metric_cap) or metric_cap <= 0.0:
            raise ValueError("metric_cap must be a positive finite number")
        if any(
            not isinstance(task, ContextTask)
            or not task.labels_public
            or not task.numeric.future_values
            for task in tasks
        ):
            raise ValueError("package evaluator requires resolved labeled tasks")
        package = self._evaluate_package(
            genome,
            tasks,
            stage=stage,
            skill_library=skill_library,
            metric_cap=metric_cap,
        )
        diagnostics = package.secondary_diagnostics
        return RetrievalEvaluation(
            version=genome.version,
            task_count=package.task_count,
            mean_final_smae=package.mean_smae,
            mean_final_srmse=package.mean_srmse,
            mean_contextual_oracle_smae=(
                package.mean_smae - diagnostics["numerical_oracle_smae_gap"]
            ),
            mean_contextual_oracle_srmse=(
                package.mean_srmse - diagnostics["numerical_oracle_srmse_gap"]
            ),
            p90_smae=package.p90_smae,
            p95_smae=package.p95_smae,
            supporting_recall=diagnostics["retrieval_supporting_recall"],
            distractor_avoidance=diagnostics["retrieval_distractor_avoidance"],
            exact_quote_validity=diagnostics["retrieval_exact_quote_validity"],
            complete_chain_rate=diagnostics["retrieval_complete_chain_rate"],
            invalid_count=package.invalid_count,
            catastrophic_count=package.catastrophic_count,
            task_traces=tuple(row.to_payload() for row in package.task_rows),
            promotion_evidence=(),
            promotion_replays=(),
        )
