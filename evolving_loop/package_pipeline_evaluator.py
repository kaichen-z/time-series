"""Authoritative final Numerical→Retrieval→Decision package scorer."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from common.llm import TransientLLMError
from common.metrics import drcik_point_metrics
from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.data import ContextTask
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.numerical_two_stage import (
    NumericalTwoStageResult,
    numerical_package_fingerprint,
    run_numerical_two_stage,
)
from evolving_loop.package_coordinate_evolution import PackageCoordinateBundle
from evolving_loop.package_metrics import PackageEvaluation, PackageTaskScore
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from evolving_loop.package_task_feedback import PackageTaskFeedbackLedger
from evolving_loop.package_task_store import PackageTaskStore
from evolving_loop.retrieval_agent.quality import score_retrieval_card_quality
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


@dataclass(frozen=True)
class _ScoredTask:
    row: PackageTaskScore
    diagnostics: Mapping[str, float]


class PackagePipelineEvaluator:
    """The sole scorer for package-native full-pipeline inference."""

    def __init__(
        self,
        retrieval_factory: Callable[[HarnessPolicy], TwoStageRetrievalAgent],
        decision_factory: Callable[[HarnessPolicy], DecisionAgent],
        *,
        metric_cap: float = 5.0,
        task_feedback_ledger: PackageTaskFeedbackLedger | None = None,
    ) -> None:
        if not callable(retrieval_factory) or not callable(decision_factory):
            raise ValueError("package pipeline factories must be callable")
        if not math.isfinite(metric_cap) or metric_cap <= 0.0:
            raise ValueError("package pipeline metric cap must be positive and finite")
        if task_feedback_ledger is not None and type(
            task_feedback_ledger
        ) is not PackageTaskFeedbackLedger:
            raise ValueError("package pipeline task feedback ledger is invalid")
        self.retrieval_factory = retrieval_factory
        self.decision_factory = decision_factory
        self.metric_cap = float(metric_cap)
        self.task_feedback_ledger = task_feedback_ledger

    def evaluate(
        self,
        bundle: PackageCoordinateBundle,
        registry: FrozenNumericalPackageRegistry,
        tasks: Sequence[ContextTask],
        *,
        stage: str,
        cache_only: bool = False,
    ) -> PackageEvaluation:
        if not isinstance(bundle, PackageCoordinateBundle):
            raise TypeError("package pipeline requires a PackageCoordinateBundle")
        if not isinstance(registry, FrozenNumericalPackageRegistry):
            raise TypeError("package pipeline requires a frozen Numerical registry")
        if registry.fingerprint != bundle.numerical_manifest_sha256:
            raise ValueError("package bundle Numerical manifest does not match registry")
        if type(cache_only) is not bool:
            raise ValueError("cache_only must be a boolean")
        if cache_only:
            raise RuntimeError(
                "cache-only package evaluation is unavailable until the immutable "
                "cache backend is configured"
            )
        retrieval_genome = bundle.policy.retrieval_genome
        if retrieval_genome is None:
            raise ValueError("package bundle requires a bound Retrieval Genome")
        return self._evaluate_components(
            candidate_sha256=bundle.fingerprint(),
            registry=registry,
            tasks=tasks,
            stage=stage,
            retrieval_factory=lambda: self.retrieval_factory(bundle.policy),
            decision_factory=lambda: self.decision_factory(bundle.policy),
            metric_cap=self.metric_cap,
            expected_retrieval_sha256=retrieval_genome.fingerprint(),
            expected_decision_prompt_sha256=hashlib.sha256(
                bundle.policy.decision_prompt.encode("utf-8")
            ).hexdigest(),
            trace_sink=(
                None
                if self.task_feedback_ledger is None
                else lambda task, result: self.task_feedback_ledger.record(
                    bundle, task, result
                )
            ),
        )

    @classmethod
    def _evaluate_components(
        cls,
        *,
        candidate_sha256: str,
        registry: FrozenNumericalPackageRegistry,
        tasks: Sequence[ContextTask],
        stage: str,
        retrieval_factory: Callable[[], TwoStageRetrievalAgent],
        decision_factory: Callable[[], DecisionAgent],
        metric_cap: float,
        expected_retrieval_sha256: str | None = None,
        expected_decision_prompt_sha256: str | None = None,
        trace_sink: Callable[[ContextTask, NumericalTwoStageResult], None] | None = None,
        task_store: PackageTaskStore | None = None,
    ) -> PackageEvaluation:
        """Shared compatibility boundary; callers supply already-bound factories."""
        if not isinstance(registry, FrozenNumericalPackageRegistry):
            raise TypeError("package pipeline requires a frozen Numerical registry")
        resolved = tuple(tasks)
        if not resolved or any(
            not isinstance(task, ContextTask)
            or not task.labels_public
            or not task.numeric.future_values
            for task in resolved
        ):
            raise ValueError("package pipeline requires resolved labeled tasks")
        task_ids = tuple(task.numeric.task_id for task in resolved)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("package pipeline task IDs must be unique")
        if not isinstance(stage, str) or not stage:
            raise ValueError("package pipeline requires an evaluation stage")
        if not math.isfinite(metric_cap) or metric_cap <= 0.0:
            raise ValueError("package pipeline metric cap must be positive and finite")
        if task_store is not None and trace_sink is not None:
            raise ValueError("task score reuse cannot replay a full-result trace sink")

        scored: list[_ScoredTask] = []
        for task in resolved:
            package = registry.package_for(task)
            identity = None
            if task_store is not None:
                identity = task_store.identity(
                    task=task, package=package, candidate_sha256=candidate_sha256,
                    stage=stage, metric_cap=metric_cap,
                    expected_retrieval_sha256=expected_retrieval_sha256,
                    expected_decision_prompt_sha256=expected_decision_prompt_sha256,
                )
                cached = task_store.load(identity)
                if cached is not None:
                    scored.append(_ScoredTask(*cached))
                    continue
                task_store.write(identity, 'started')
            try:
                item, artifacts = cls._evaluate_task(
                    task=task, package=package,
                    retrieval_factory=retrieval_factory,
                    decision_factory=decision_factory, metric_cap=metric_cap,
                    expected_retrieval_sha256=expected_retrieval_sha256,
                    expected_decision_prompt_sha256=expected_decision_prompt_sha256,
                    trace_sink=trace_sink,
                )
                if task_store is not None:
                    task_store.write(identity, 'completed', score=item.row,
                                     diagnostics=item.diagnostics, artifacts=artifacts)
                scored.append(item)
            except BaseException as error:
                if task_store is not None:
                    task_store.write(identity, 'error', error=error)
                raise

        diagnostic_names = tuple(
            sorted({name for item in scored for name in item.diagnostics})
        )
        diagnostics = {
            name: _mean([float(item.diagnostics[name]) for item in scored])
            for name in diagnostic_names
        }
        return PackageEvaluation.from_rows(
            candidate_sha256,
            tuple(item.row for item in scored),
            expected_task_ids=task_ids,
            secondary_diagnostics=diagnostics,
        )

    @classmethod
    def _evaluate_task(cls, *, task, package, retrieval_factory, decision_factory,
                       metric_cap, expected_retrieval_sha256,
                       expected_decision_prompt_sha256, trace_sink):
        retrieval = retrieval_factory()
        decision = decision_factory()
        if not isinstance(retrieval, TwoStageRetrievalAgent) or not isinstance(decision, DecisionAgent):
            raise TypeError("package pipeline factories returned invalid agents")
        if (expected_retrieval_sha256 is not None
                and retrieval.genome.fingerprint() != expected_retrieval_sha256):
            raise ValueError("package pipeline changed the bound Retrieval Genome")
        if (expected_decision_prompt_sha256 is not None
                and hashlib.sha256(decision.prompt.encode('utf-8')).hexdigest()
                != expected_decision_prompt_sha256):
            raise ValueError("package pipeline changed the bound Decision prompt")
        try:
            result = run_numerical_two_stage(
                task, package, retrieval, decision,
                preserve_round1_on_round2_failure=True,
            )
        except TransientLLMError:
            raise
        except (TypeError, ValueError) as error:
            return cls._score_contract_fallback(task, package, error, metric_cap=metric_cap), {
                'contract_error': {'type': type(error).__name__, 'message': str(error)},
            }
        if (expected_retrieval_sha256 is not None
                and result.fingerprints.get('retrieval_genome') != expected_retrieval_sha256):
            raise ValueError("package pipeline changed the bound Retrieval Genome")
        if (expected_decision_prompt_sha256 is not None
                and result.fingerprints.get('decision_prompt') != expected_decision_prompt_sha256):
            raise ValueError("package pipeline changed the bound Decision prompt")
        if trace_sink is not None:
            trace_sink(task, result)
        artifacts = {
            'fallback_reason': result.fallback_reason,
            'round2_failure_reason': result.round2_failure_reason,
            'provisional_decision_rejection': result.provisional_decision.rejection_reason,
            'final_decision_rejection': result.final_decision.rejection_reason,
            'dictionary_traces': result.dictionary_traces,
            'fingerprints': dict(result.fingerprints),
        }
        return cls._score_result(task, package, result, metric_cap=metric_cap), artifacts

    @staticmethod
    def _score_result(task, package, result, *, metric_cap: float) -> _ScoredTask:
        truth = tuple(task.numeric.future_values)
        final = drcik_point_metrics(truth, result.forecast, cap=metric_cap)
        alternatives = tuple(
            (
                item.name,
                drcik_point_metrics(truth, item.forecast, cap=metric_cap),
            )
            for item in package.ranked_alternatives
        )
        if not alternatives:
            raise ValueError("package pipeline requires materialized alternatives")
        _oracle_name, oracle = min(
            alternatives,
            key=lambda item: (
                float(item[1]["srmse"]),
                float(item[1]["smae"]),
                item[0],
            ),
        )

        retrieval_card = result.retrieval_card
        fallback_count = int(result.fallback_reason is not None)
        invalid_round2_count = int(result.round2_failure_reason is not None)
        quality = score_retrieval_card_quality(task, retrieval_card)
        final_retrieval_sha256 = _canonical_digest(retrieval_card.to_payload())
        invalid_count = (
            quality.rejection_count
            + invalid_round2_count
            + fallback_count
            + int(
                result.final_decision.rejection_reason is not None
                and result.fallback_reason is None
            )
        )
        final_smae = float(final["smae"])
        final_srmse = float(final["srmse"])
        final_smae_raw = float(final["smae_raw"])
        final_srmse_raw = float(final["srmse_raw"])
        oracle_smae = float(oracle["smae"])
        oracle_srmse = float(oracle["srmse"])
        chains = tuple(retrieval_card.chains)
        temporal_match = _mean(
            [float(chain.temporal_relation == "overlaps_future") for chain in chains]
        )
        return _ScoredTask(
            PackageTaskScore(
                task_id=task.numeric.task_id,
                entity_name=task.numeric.entity_name,
                final_smae=final_smae,
                final_srmse=final_srmse,
                final_smae_raw=final_smae_raw,
                final_srmse_raw=final_srmse_raw,
                final_forecast=result.forecast,
                numerical_oracle_smae=oracle_smae,
                numerical_oracle_srmse=oracle_srmse,
                numerical_candidate_count=len(alternatives),
                smae_clipped=bool(final["smae_clipped"]),
                srmse_clipped=bool(final["srmse_clipped"]),
                invalid_count=invalid_count,
                catastrophic_count=int(
                    final_smae_raw > 10.0 or final_srmse_raw > 10.0
                ),
                fallback_count=fallback_count,
                selected_candidate_id=result.final_decision.selected.candidate_id,
                numerical_package_sha256=result.fingerprints["numerical_package"],
                final_retrieval_sha256=final_retrieval_sha256,
                final_decision_sha256=result.fingerprints["final_decision_artifact"],
            ),
            {
                "numerical_oracle_smae_gap": final_smae - oracle_smae,
                "numerical_oracle_srmse_gap": final_srmse - oracle_srmse,
                "numerical_oracle_joint_gap": (
                    (final_smae + final_srmse) - (oracle_smae + oracle_srmse)
                )
                / 2.0,
                "retrieval_supporting_recall": quality.supporting_recall,
                "retrieval_distractor_avoidance": quality.distractor_avoidance,
                "retrieval_exact_quote_validity": quality.exact_quote_validity,
                "retrieval_temporal_match": temporal_match,
                "retrieval_complete_chain_rate": quality.complete_chain_rate,
                "decision_selection_smae_regret": final_smae - oracle_smae,
                "decision_selection_srmse_regret": final_srmse - oracle_srmse,
                "decision_selection_joint_regret": (
                    (final_smae + final_srmse) - (oracle_smae + oracle_srmse)
                )
                / 2.0,
                "candidate_count": float(len(alternatives)),
            },
        )

    @staticmethod
    def _score_contract_fallback(
        task,
        package,
        error: Exception,
        *,
        metric_cap: float,
    ) -> _ScoredTask:
        anchor = package.protected_baseline
        metrics = drcik_point_metrics(
            tuple(task.numeric.future_values),
            anchor.forecast,
            cap=metric_cap,
        )
        alternatives = tuple(
            drcik_point_metrics(
                tuple(task.numeric.future_values),
                item.forecast,
                cap=metric_cap,
            )
            for item in package.ranked_alternatives
        )
        oracle = min(
            alternatives,
            key=lambda item: (float(item["srmse"]), float(item["smae"])),
        )
        error_payload = {
            "schema_version": 1,
            "task_id": task.numeric.task_id,
            "fallback": "deterministic_contract_error",
            "error_type": type(error).__name__,
        }
        final_smae = float(metrics["smae"])
        final_srmse = float(metrics["srmse"])
        final_smae_raw = float(metrics["smae_raw"])
        final_srmse_raw = float(metrics["srmse_raw"])
        oracle_smae = float(oracle["smae"])
        oracle_srmse = float(oracle["srmse"])
        return _ScoredTask(
            PackageTaskScore(
                task_id=task.numeric.task_id,
                entity_name=task.numeric.entity_name,
                final_smae=final_smae,
                final_srmse=final_srmse,
                final_smae_raw=final_smae_raw,
                final_srmse_raw=final_srmse_raw,
                final_forecast=anchor.forecast,
                numerical_oracle_smae=oracle_smae,
                numerical_oracle_srmse=oracle_srmse,
                numerical_candidate_count=len(alternatives),
                smae_clipped=bool(metrics["smae_clipped"]),
                srmse_clipped=bool(metrics["srmse_clipped"]),
                invalid_count=1,
                catastrophic_count=int(
                    final_smae_raw > 10.0 or final_srmse_raw > 10.0
                ),
                fallback_count=1,
                selected_candidate_id=anchor.name,
                numerical_package_sha256=numerical_package_fingerprint(package),
                final_retrieval_sha256=_canonical_digest(
                    {**error_payload, "artifact": "retrieval"}
                ),
                final_decision_sha256=_canonical_digest(
                    {**error_payload, "artifact": "decision", "selected": anchor.name}
                ),
            ),
            {
                "numerical_oracle_smae_gap": final_smae - oracle_smae,
                "numerical_oracle_srmse_gap": final_srmse - oracle_srmse,
                "numerical_oracle_joint_gap": (
                    (final_smae + final_srmse) - (oracle_smae + oracle_srmse)
                )
                / 2.0,
                "retrieval_supporting_recall": 0.0,
                "retrieval_distractor_avoidance": 0.0,
                "retrieval_exact_quote_validity": 0.0,
                "retrieval_temporal_match": 0.0,
                "retrieval_complete_chain_rate": 0.0,
                "decision_selection_smae_regret": final_smae - oracle_smae,
                "decision_selection_srmse_regret": final_srmse - oracle_srmse,
                "decision_selection_joint_regret": (
                    (final_smae + final_srmse) - (oracle_smae + oracle_srmse)
                )
                / 2.0,
                "candidate_count": float(len(package.ranked_alternatives)),
            },
        )


__all__ = ["PackagePipelineEvaluator"]
