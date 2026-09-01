"""Trusted Retrieval evaluation over frozen Champion Numerical packages."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from common.metrics import drcik_point_metrics, linear_quantile
from evolving_loop.data import ContextTask
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.numerical_two_stage import run_numerical_two_stage
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from evolving_loop.retrieval_agent.evolution import (
    RetrievalEvaluation,
    RetrievalEvolutionError,
    RetrievalInferenceCacheKey,
)
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.quality import score_retrieval_card_quality
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


@dataclass(frozen=True)
class _PackageRetrievalRow:
    task_id: str
    entity_name: str
    final_smae: float
    final_srmse: float
    oracle_smae: float
    oracle_srmse: float
    supporting_recall: float
    distractor_avoidance: float
    exact_quote_validity: float
    complete_chain_rate: float
    invalid_count: int
    catastrophic_count: int
    numerical_package_sha256: str
    final_retrieval_sha256: str
    final_decision_sha256: str


def _score_metrics(
    truth: tuple[float, ...],
    forecast: tuple[float, ...],
    *,
    cap: float,
) -> dict[str, float | bool]:
    return drcik_point_metrics(truth, forecast, cap=cap)


def _score_task(
    task: ContextTask,
    package,
    retrieval: TwoStageRetrievalAgent,
    decision: DecisionAgent,
    *,
    metric_cap: float,
) -> _PackageRetrievalRow:
    result = run_numerical_two_stage(task, package, retrieval, decision)
    truth = tuple(task.numeric.future_values)
    final = _score_metrics(truth, result.forecast, cap=metric_cap)
    alternatives = tuple(
        (
            item.name,
            _score_metrics(truth, item.forecast, cap=metric_cap),
        )
        for item in package.ranked_alternatives
    )
    if not alternatives:
        raise RetrievalEvolutionError(
            "frozen Numerical package has no materialized alternatives"
        )
    oracle_name, oracle = min(
        alternatives,
        key=lambda item: (
            float(item[1]["srmse"]),
            float(item[1]["smae"]),
            item[0],
        ),
    )
    del oracle_name
    quality = score_retrieval_card_quality(task, result.retrieval_card)
    invalid_count = quality.rejection_count + int(result.fallback_reason is not None)
    catastrophic_count = int(
        bool(final["smae_clipped"]) or bool(final["srmse_clipped"])
    )
    return _PackageRetrievalRow(
        task_id=task.numeric.task_id,
        entity_name=task.numeric.entity_name,
        final_smae=float(final["smae"]),
        final_srmse=float(final["srmse"]),
        oracle_smae=float(oracle["smae"]),
        oracle_srmse=float(oracle["srmse"]),
        supporting_recall=quality.supporting_recall,
        distractor_avoidance=quality.distractor_avoidance,
        exact_quote_validity=quality.exact_quote_validity,
        complete_chain_rate=quality.complete_chain_rate,
        invalid_count=invalid_count,
        catastrophic_count=catastrophic_count,
        numerical_package_sha256=result.fingerprints["numerical_package"],
        final_retrieval_sha256=result.fingerprints["final_retrieval_artifact"],
        final_decision_sha256=result.fingerprints["final_decision_artifact"],
    )


def _aggregate(version: str, rows: tuple[_PackageRetrievalRow, ...]) -> RetrievalEvaluation:
    if not rows:
        raise RetrievalEvolutionError("package evaluation requires at least one task")

    def mean(field_name: str) -> float:
        return statistics.fmean(float(getattr(row, field_name)) for row in rows)

    final_smae = [row.final_smae for row in rows]
    return RetrievalEvaluation(
        version=version,
        task_count=len(rows),
        mean_final_smae=mean("final_smae"),
        mean_final_srmse=mean("final_srmse"),
        mean_contextual_oracle_smae=mean("oracle_smae"),
        mean_contextual_oracle_srmse=mean("oracle_srmse"),
        p90_smae=linear_quantile(final_smae, 0.90),
        p95_smae=linear_quantile(final_smae, 0.95),
        supporting_recall=mean("supporting_recall"),
        distractor_avoidance=mean("distractor_avoidance"),
        exact_quote_validity=mean("exact_quote_validity"),
        complete_chain_rate=mean("complete_chain_rate"),
        invalid_count=sum(row.invalid_count for row in rows),
        catastrophic_count=sum(row.catastrophic_count for row in rows),
        task_traces=tuple(
            {
                "task_id": row.task_id,
                "entity_name": row.entity_name,
                "final_smae": row.final_smae,
                "final_srmse": row.final_srmse,
                "contextual_oracle_smae": row.oracle_smae,
                "contextual_oracle_srmse": row.oracle_srmse,
                "numerical_package_sha256": row.numerical_package_sha256,
                "final_retrieval_sha256": row.final_retrieval_sha256,
                "final_decision_sha256": row.final_decision_sha256,
            }
            for row in rows
        ),
        promotion_evidence=(),
        promotion_replays=(),
    )


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
        before = tuple(skill.to_payload() for skill in skill_library.all())
        rows = tuple(
            _score_task(
                task,
                self.registry.package_for(task),
                self.retrieval_factory(genome, skill_library),
                self.decision_factory(),
                metric_cap=metric_cap,
            )
            for task in tasks
        )
        after = tuple(skill.to_payload() for skill in skill_library.all())
        if after != before:
            raise RetrievalEvolutionError(
                "package evaluator mutated the accepted Retrieval Skill snapshot"
            )
        return _aggregate(genome.version, rows)
