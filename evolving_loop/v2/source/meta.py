"""Host-only meta-CV for selecting audited Source V2 policies.

This deliberately invokes the Project 3 primitives directly.  In particular it
does not use the cooperative runner, whose per-candidate Dev evaluation breaks
the sealed-validation protocol used here.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from evolving_loop.data import ContextTask
from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.v2.bundle import EvolutionBundleV2
from evolving_loop.v2.contracts import fingerprint_payload
from evolving_loop.v2.cooperative.adapters import (
    CooperativeArtifactCatalog,
    CooperativePipelineAdapter,
    sanitize_train_feedback,
)
from evolving_loop.v2.cooperative.contracts import ARM_ORDER
from evolving_loop.v2.cooperative.proposals import propose_bundle_candidate

from .contracts import SourceRequestV2, SourceVariantV2
from .runtime import run_policy


_TOLERANCE = 1e-12
_FOLDS = ((0, 1), (2, 3))


@dataclass(frozen=True, slots=True)
class SourceTrainResultV2:
    source_sha256: str
    fold_gains: tuple[float, ...]
    mean_gain: float
    mean_capped_smae: float
    mean_capped_srmse: float
    invalid_count: int
    catastrophic_count: int
    task_cost: int
    source_invocations: int
    actual_task_cost: int
    actual_source_invocations: int
    feasible: bool
    execution_fingerprint: str


@dataclass(frozen=True, slots=True)
class SourceValidationV2:
    parent_source_sha256: str
    finalist_source_sha256: str
    passed: bool
    reason: str
    parent_outcome: PackageEvaluation
    finalist_outcome: PackageEvaluation
    commitment_sha256: str
    replay_fingerprint: str
    evaluation_fingerprints: tuple[str, str]
    task_cost: int = 0
    source_invocations: int = 0
    actual_task_cost: int = 0
    actual_source_invocations: int = 0


@dataclass(frozen=True, slots=True)
class _TrainingEpisode:
    result: SourceTrainResultV2
    selected_bundle: EvolutionBundleV2
    selected_evaluations: tuple[PackageEvaluation, ...]
    selected_pipeline: CooperativePipelineAdapter


class SourceMetaEvaluatorV2:
    """Evaluate a SourceVariant via complementary Train folds and sealed Dev."""

    def __init__(self, case: object) -> None:
        self.case = case
        self.train_tasks = tuple(getattr(case, "train_tasks"))
        self.dev_tasks = tuple(getattr(case, "dev_tasks"))
        self.seed_bundle = getattr(case, "seed_bundle")
        self.enabled_arms = tuple(getattr(case, "enabled_arms", ARM_ORDER))
        self._episodes: dict[tuple[str, int], _TrainingEpisode] = {}
        self._aggregate_cache: dict[str, PackageEvaluation] = {}
        if type(self.seed_bundle) is not EvolutionBundleV2:
            raise TypeError("source meta case requires an EvolutionBundleV2 seed_bundle")
        if self.enabled_arms != tuple(arm for arm in ARM_ORDER if arm in self.enabled_arms):
            raise ValueError("enabled arms must use canonical SourceRequestV2 order")
        self._validate_case()

    def _validate_case(self) -> None:
        if len(self.train_tasks) != 4 or len(self.dev_tasks) != 1:
            raise ValueError("source meta case requires exactly Train4 and Dev1")
        all_tasks = (*self.train_tasks, *self.dev_tasks)
        if any(type(task) is not ContextTask for task in all_tasks):
            raise TypeError("source meta case tasks must be ContextTask values")
        ids = tuple(task.numeric.task_id for task in all_tasks)
        entities = tuple(task.numeric.entity_name for task in all_tasks)
        if len(ids) != len(set(ids)):
            raise ValueError("source meta task IDs must be unique")
        if len(entities) != len(set(entities)):
            raise ValueError("source meta folds and Dev must have entity-disjoint tasks")
        if any("public" in value.casefold() for value in (*ids, *entities)):
            raise ValueError("Public task membership is forbidden in source meta evaluation")

    def _task_identity(self, tasks: Sequence[ContextTask], *, split: str, seed: int) -> str:
        return fingerprint_payload({
            "task_ids": [task.numeric.task_id for task in tasks],
            "entities": [task.numeric.entity_name for task in tasks],
            "split": split,
            "seed": seed,
            "bundle_protocol": self.seed_bundle.protocol_fingerprint,
            "bundle_runtime": dict(self.seed_bundle.runtime_fingerprints),
        })

    def _evaluate(
        self, pipeline: CooperativePipelineAdapter, bundle: EvolutionBundleV2,
        tasks: Sequence[ContextTask], stage: str, *, split: str, seed: int,
    ) -> tuple[PackageEvaluation, int]:
        key = fingerprint_payload({
            "bundle": bundle.fingerprint(), "tasks": self._task_identity(tasks, split=split, seed=seed),
            "stage": stage, "source_protocol": self.seed_bundle.protocol_fingerprint,
            "source_runtime": dict(self.seed_bundle.runtime_fingerprints),
        })
        cached = self._aggregate_cache.get(key)
        if cached is not None:
            return cached, 0
        evaluation = pipeline.evaluate(bundle, tasks, stage)
        if evaluation.candidate_sha256 != bundle.fingerprint():
            raise ValueError("pipeline aggregate does not bind evaluated Bundle")
        if evaluation.public_test_accessed:
            raise ValueError("Public evaluation is forbidden in source meta evaluation")
        self._aggregate_cache[key] = evaluation
        return evaluation, len(tasks)

    @staticmethod
    def _complete(evaluation: PackageEvaluation) -> bool:
        return (
            evaluation.coverage == 1.0
            and not evaluation.missing_task_ids
            and all(math.isfinite(value) for value in (evaluation.mean_smae, evaluation.mean_srmse, evaluation.mean_joint))
        )

    @classmethod
    def _fold_passes(cls, parent: PackageEvaluation, child: PackageEvaluation) -> bool:
        return (
            cls._complete(parent) and cls._complete(child)
            and child.invalid_count <= parent.invalid_count
            and child.catastrophic_count <= parent.catastrophic_count
            and child.mean_smae <= parent.mean_smae + _TOLERANCE
            and child.mean_srmse <= parent.mean_srmse + _TOLERANCE
        )

    @staticmethod
    def _gain(parent: PackageEvaluation, child: PackageEvaluation) -> float:
        parent_joint = (parent.mean_smae + parent.mean_srmse) / 2.0
        child_joint = (child.mean_smae + child.mean_srmse) / 2.0
        return (parent_joint - child_joint) / max(abs(parent_joint), _TOLERANCE)

    def _fresh_runtime(self) -> tuple[CooperativeArtifactCatalog, EvolutionBundleV2, Mapping[str, object], CooperativePipelineAdapter]:
        catalog, seed = getattr(self.case, "catalog_factory")()
        if type(catalog) is not CooperativeArtifactCatalog or type(seed) is not EvolutionBundleV2:
            raise TypeError("catalog_factory must return (CooperativeArtifactCatalog, EvolutionBundleV2)")
        if seed.fingerprint() != self.seed_bundle.fingerprint():
            raise ValueError("catalog_factory must reconstruct the frozen seed Bundle")
        adapters = getattr(self.case, "adapters_factory")(catalog)
        if not isinstance(adapters, Mapping):
            raise TypeError("adapters_factory must return a mapping")
        pipeline = getattr(self.case, "pipeline_factory")(catalog)
        if type(pipeline) is not CooperativePipelineAdapter:
            raise TypeError("pipeline_factory must return CooperativePipelineAdapter")
        return catalog, seed, adapters, pipeline

    def _request(self, variant: SourceVariantV2, rewards: Mapping[str, float], *, step: int, seed: int) -> str:
        request = SourceRequestV2(1, self.enabled_arms, step, seed, rewards)
        sink = getattr(self.case, "policy_requests", None)
        if sink is not None:
            sink.append(request.to_payload())
        return run_policy(variant, request)

    def _train(self, variant: SourceVariantV2, *, epoch_seed: int, fresh: bool = False) -> _TrainingEpisode:
        if type(variant) is not SourceVariantV2:
            raise TypeError("source meta train requires a SourceVariantV2")
        cache_key = (variant.fingerprint(), epoch_seed)
        if not fresh and cache_key in self._episodes:
            return self._episodes[cache_key]
        fold_gains: list[float] = []
        children: list[EvolutionBundleV2] = []
        outcomes: list[PackageEvaluation] = []
        pipelines: list[CooperativePipelineAdapter] = []
        logical_task_cost = 0
        actual_task_cost = 0
        feasible = True
        for fold_index, held_indices in enumerate(_FOLDS):
            complement = tuple(task for index, task in enumerate(self.train_tasks) if index not in held_indices)
            held_out = tuple(self.train_tasks[index] for index in held_indices)
            catalog, seed, adapters, pipeline = self._fresh_runtime()
            parent_scan, cost = self._evaluate(pipeline, seed, complement, "train", split=f"scan-{fold_index}", seed=epoch_seed)
            logical_task_cost += len(complement)
            actual_task_cost += cost
            feedback_by_arm = {}
            proposal_feedback = {}
            for arm in self.enabled_arms:
                # The initial materialization is deliberately neutral: a child
                # must exist before its real aggregate-only feedback can be
                # computed.  The source-selected proposal below always uses
                # that real sanitized feedback, never held-out evidence.
                candidate = propose_bundle_candidate(
                    seed, arm, catalog, adapters,
                    sanitize_train_feedback(parent_scan, parent_scan, 0.0), fold_index,
                )
                if candidate is None:
                    feedback_by_arm[arm] = 0.0
                    proposal_feedback[arm] = sanitize_train_feedback(parent_scan, parent_scan, 0.0)
                    continue
                child_scan, cost = self._evaluate(pipeline, candidate.to_child(seed), complement, "train", split=f"scan-{fold_index}", seed=epoch_seed)
                logical_task_cost += len(complement)
                actual_task_cost += cost
                proposal_feedback[arm] = sanitize_train_feedback(
                    parent_scan, child_scan, float(len(complement))
                )
                feedback_by_arm[arm] = self._gain(parent_scan, child_scan) if self._fold_passes(parent_scan, child_scan) else 0.0
            selected = self._request(variant, feedback_by_arm, step=fold_index, seed=epoch_seed + fold_index)
            # Fresh catalog per fold is intentionally retained through the selected
            # proposal; source selection never observes held-out measurements.
            selected_candidate = propose_bundle_candidate(
                seed, selected, catalog, adapters, proposal_feedback[selected], fold_index
            )
            child = seed if selected_candidate is None else selected_candidate.to_child(seed)
            parent, cost = self._evaluate(pipeline, seed, held_out, "train", split=f"held-{fold_index}", seed=epoch_seed)
            logical_task_cost += len(held_out)
            actual_task_cost += cost
            child_eval, cost = self._evaluate(pipeline, child, held_out, "train", split=f"held-{fold_index}", seed=epoch_seed)
            logical_task_cost += len(held_out)
            actual_task_cost += cost
            fold_gains.append(self._gain(parent, child_eval))
            feasible = feasible and self._fold_passes(parent, child_eval)
            children.append(child)
            outcomes.append(child_eval)
            pipelines.append(pipeline)
        # Every fold child is first sealed on the same complete Train universe.
        # This full-Train comparison, rather than either held-out fold result,
        # decides which Bundle may proceed to Dev.
        sealed_outcomes: list[PackageEvaluation] = []
        for index, child in enumerate(children):
            sealed_eval, cost = self._evaluate(
                pipelines[index], child, self.train_tasks,
                "train", split="seal-all-train", seed=epoch_seed,
            )
            logical_task_cost += len(self.train_tasks)
            actual_task_cost += cost
            sealed_outcomes.append(sealed_eval)
        selected_index = min(
            range(len(children)),
            key=lambda index: (
                sealed_outcomes[index].mean_joint, children[index].fingerprint()
            ),
        )
        sealed_eval = sealed_outcomes[selected_index]
        result = SourceTrainResultV2(
            variant.fingerprint(), tuple(fold_gains), sum(fold_gains) / len(fold_gains),
            sealed_eval.mean_smae, sealed_eval.mean_srmse,
            sealed_eval.invalid_count, sealed_eval.catastrophic_count,
            logical_task_cost, len(_FOLDS), actual_task_cost, len(_FOLDS), feasible,
            fingerprint_payload({"source": variant.fingerprint(), "epoch_seed": epoch_seed, "fold_gains": fold_gains,
                "children": [child.fingerprint() for child in children], "evaluations": [value.fingerprint for value in outcomes],
                "sealed_evaluations": [value.fingerprint for value in sealed_outcomes],
                "case": self._task_identity(self.train_tasks, split="train", seed=epoch_seed)}),
        )
        episode = _TrainingEpisode(
            result, children[selected_index], (*outcomes, *sealed_outcomes), pipelines[selected_index]
        )
        self._episodes[cache_key] = episode
        return episode

    def train(self, variant: SourceVariantV2) -> SourceTrainResultV2:
        cache_key = (variant.fingerprint(), 0)
        was_cached = cache_key in self._episodes
        result = self._train(variant, epoch_seed=0).result
        return replace(result, actual_task_cost=0, actual_source_invocations=0) if was_cached else result

    def _validate(self, parent_source: SourceVariantV2, finalist_source: SourceVariantV2, *, epoch_seed: int, strict: bool) -> SourceValidationV2:
        parent_cached = (parent_source.fingerprint(), epoch_seed) in self._episodes
        parent_episode = self._train(parent_source, epoch_seed=epoch_seed)
        finalist_cached = (finalist_source.fingerprint(), epoch_seed) in self._episodes
        finalist_episode = self._train(finalist_source, epoch_seed=epoch_seed)
        parent_eval, parent_dev_cost = self._evaluate(parent_episode.selected_pipeline, parent_episode.selected_bundle, self.dev_tasks, "dev", split="dev", seed=epoch_seed)
        finalist_eval, finalist_dev_cost = self._evaluate(finalist_episode.selected_pipeline, finalist_episode.selected_bundle, self.dev_tasks, "dev", split="dev", seed=epoch_seed)
        parent_joint = (parent_eval.mean_smae + parent_eval.mean_srmse) / 2.0
        finalist_joint = (finalist_eval.mean_smae + finalist_eval.mean_srmse) / 2.0
        train_better = finalist_episode.result.feasible and finalist_episode.result.mean_gain > _TOLERANCE and finalist_episode.result.mean_gain > parent_episode.result.mean_gain + _TOLERANCE
        dev_better = finalist_joint < parent_joint - _TOLERANCE if strict else finalist_joint <= parent_joint + _TOLERANCE
        passed = train_better and self._fold_passes(parent_eval, finalist_eval) and dev_better
        reason = "accepted" if passed else ("held_out_not_strictly_better" if strict and not dev_better else "train_or_nonregression_gate_failed")
        replay = fingerprint_payload({"parent_train": parent_episode.result.execution_fingerprint, "finalist_train": finalist_episode.result.execution_fingerprint, "epoch_seed": epoch_seed})
        commitment = fingerprint_payload({"parent": parent_source.fingerprint(), "finalist": finalist_source.fingerprint(), "parent_bundle": parent_episode.selected_bundle.fingerprint(), "finalist_bundle": finalist_episode.selected_bundle.fingerprint(), "replay": replay, "evaluations": [parent_eval.fingerprint, finalist_eval.fingerprint]})
        # Account the logical work of this stage, independent of in-process
        # evaluator caches. Validation reuses the already charged finalist
        # Train episode; a fresh-seed canary evaluates both sources.
        trained = (parent_episode,) if strict else (parent_episode, finalist_episode)
        task_cost = sum(episode.result.task_cost for episode in trained) + 2 * len(self.dev_tasks)
        source_invocations = sum(episode.result.source_invocations for episode in trained)
        actual_task_cost = parent_dev_cost + finalist_dev_cost
        actual_source_invocations = 0
        for cached, episode in ((parent_cached, parent_episode), (finalist_cached, finalist_episode)):
            if not cached:
                actual_task_cost += episode.result.actual_task_cost
                actual_source_invocations += episode.result.actual_source_invocations
        return SourceValidationV2(
            parent_source.fingerprint(), finalist_source.fingerprint(), passed, reason,
            parent_eval, finalist_eval, commitment, replay,
            (parent_eval.fingerprint, finalist_eval.fingerprint),
            task_cost, source_invocations, actual_task_cost, actual_source_invocations,
        )

    def validate(self, parent_source: SourceVariantV2, finalist_source: SourceVariantV2) -> SourceValidationV2:
        return self._validate(parent_source, finalist_source, epoch_seed=0, strict=True)

    def canary(self, parent_source: SourceVariantV2, finalist_source: SourceVariantV2, *, epoch_seed: int) -> SourceValidationV2:
        if type(epoch_seed) is not int:
            raise ValueError("epoch_seed must be an integer")
        return self._validate(parent_source, finalist_source, epoch_seed=epoch_seed, strict=False)

    def replay(self, variant: SourceVariantV2, train_result: SourceTrainResultV2) -> bool:
        if type(train_result) is not SourceTrainResultV2 or train_result.source_sha256 != variant.fingerprint():
            return False
        return self._train(variant, epoch_seed=0, fresh=True).result.execution_fingerprint == train_result.execution_fingerprint


__all__ = ["SourceMetaEvaluatorV2", "SourceTrainResultV2", "SourceValidationV2"]
