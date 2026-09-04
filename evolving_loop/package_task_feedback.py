"""Trusted construction of task-level Retrieval-to-Numerical feedback."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from common.evolution_core.task_feedback import (
    TaskEvidenceCase,
    TaskEvidenceProjection,
    TaskFeedbackError,
    TaskMorphologyProjection,
)
from evolving_loop.data import ContextTask
from evolving_loop.numerical_two_stage import NumericalTwoStageResult
from evolving_loop.package_coordinate_evolution import (
    PackageCoordinateBundle,
    package_principal_fingerprints,
)
from evolving_loop.retrieval_agent.schemas import EvidenceChain, RetrievalAssumption


Partition = Literal["train", "dev"]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _principal_key(bundle: PackageCoordinateBundle) -> tuple[str, str, str]:
    fingerprints = package_principal_fingerprints(bundle)
    genome = bundle.policy.retrieval_genome
    if genome is None:
        raise TaskFeedbackError("task feedback bundle has no Retrieval genome")
    return (
        fingerprints["numerical"],
        genome.fingerprint(),
        fingerprints["decision"],
    )


def _frequency_bucket(value: str) -> str:
    compact = value.strip().casefold()
    if compact in {"s", "sec", "t", "min", "h"}:
        return "subdaily"
    if compact in {"d", "b", "day", "daily"}:
        return "daily"
    if compact in {"w", "week", "weekly"}:
        return "weekly"
    if compact in {"m", "ms", "month", "monthly"}:
        return "monthly"
    if compact in {"q", "quarter", "quarterly"}:
        return "quarterly"
    if compact in {"y", "a", "year", "yearly", "annual"}:
        return "yearly"
    return "other"


def _length_bucket(value: int, *, medium: int, long: int) -> str:
    if value < medium:
        return "short"
    if value < long:
        return "medium"
    return "long"


def _morphology(result: NumericalTwoStageResult) -> TaskMorphologyProjection:
    profile = result.numerical.task_profile
    if profile.trend_strength < 0.2 or profile.trend_direction not in {"up", "down"}:
        trend = "flat"
    else:
        prefix = "strong" if profile.trend_strength >= 0.6 else "weak"
        trend = f"{prefix}_{profile.trend_direction}"
    periodicity = (
        "strong"
        if profile.periodicity_strength >= 0.6
        else "weak" if profile.periodicity_strength >= 0.2 else "none"
    )
    intermittency = (
        "dense"
        if profile.intermittency_adi < 1.32
        else "intermittent" if profile.intermittency_adi < 2.0 else "sparse"
    )
    return TaskMorphologyProjection(
        frequency=cast(str, _frequency_bucket(profile.frequency)),
        history=cast(str, _length_bucket(profile.history_length, medium=48, long=168)),
        horizon=cast(str, _length_bucket(profile.horizon, medium=12, long=36)),
        trend=cast(str, trend),
        periodicity=cast(str, periodicity),
        intermittency=cast(str, intermittency),
        recent_regime=(
            "recent_shift"
            if profile.recent_regime_start is not None
            and profile.recent_regime_confidence >= 0.5
            else "stable"
        ),
    )


def _stance(chains: tuple[EvidenceChain, ...]) -> str:
    values = {
        (
            "supported"
            if item.stance in {"support", "supports"}
            else (
                "falsified"
                if item.stance in {"challenge", "challenges"}
                else "uncertain"
            )
        )
        for item in chains
    }
    return next(iter(values)) if len(values) == 1 else "uncertain"


def _window_relation(chains: tuple[EvidenceChain, ...]) -> str:
    relations = {item.temporal_relation for item in chains}
    if "overlaps_future" in relations:
        return "overlaps"
    if relations and relations <= {"historical", "ended_before_future"}:
        return "precedes"
    return "unknown"


def _magnitude_status(chains: tuple[EvidenceChain, ...]) -> str:
    if not chains:
        return "not_applicable"
    values = tuple(
        (item.magnitude_kind, item.magnitude_value)
        for item in chains
        if item.magnitude_value is not None
    )
    if not values:
        return "missing"
    if len(set(values)) > 1:
        return "conflicting"
    return "present"


def _mechanism(chains: tuple[EvidenceChain, ...]) -> str:
    translated = {
        {
            "future_driver": "event_shock",
            "regime": "regime_change",
        }.get(item.mechanism, "unknown")
        for item in chains
    }
    return next(iter(translated)) if len(translated) == 1 else "unknown"


@dataclass(frozen=True)
class _VerifiedTrace:
    principal_key: tuple[str, str, str]
    task: ContextTask
    result: NumericalTwoStageResult
    partition: Partition
    trace_sha256: str


class PackageTaskFeedbackLedger:
    """Hold verified inference traces and emit all-or-nothing safe projections."""

    def __init__(self, partition_by_task: Mapping[str, str]) -> None:
        if not isinstance(partition_by_task, Mapping) or not partition_by_task:
            raise TaskFeedbackError("task feedback requires Train or Dev membership")
        partitions = dict(partition_by_task)
        if any(
            type(task_id) is not str or not task_id or partition not in {"train", "dev"}
            for task_id, partition in partitions.items()
        ):
            raise TaskFeedbackError(
                "task feedback membership must be Train or Dev only"
            )
        self.partition_by_task = MappingProxyType(dict(sorted(partitions.items())))
        self._traces: dict[tuple[tuple[str, str, str], str], _VerifiedTrace] = {}

    def record(
        self,
        bundle: PackageCoordinateBundle,
        task: ContextTask,
        result: NumericalTwoStageResult,
    ) -> None:
        if type(bundle) is not PackageCoordinateBundle:
            raise TaskFeedbackError(
                "task feedback trace requires an exact package bundle"
            )
        if type(task) is not ContextTask or type(result) is not NumericalTwoStageResult:
            raise TaskFeedbackError(
                "task feedback trace requires verified inference types"
            )
        task_id = task.numeric.task_id
        partition = self.partition_by_task.get(task_id)
        if partition not in {"train", "dev"}:
            raise TaskFeedbackError("task feedback trace is outside Train or Dev")
        if result.fallback_reason is not None:
            return
        if result.numerical.task_profile.task_id != task_id:
            raise TaskFeedbackError("task feedback trace task identity mismatch")
        genome = bundle.policy.retrieval_genome
        if (
            genome is None
            or result.fingerprints.get("retrieval_genome") != genome.fingerprint()
        ):
            raise TaskFeedbackError("task feedback trace Retrieval bundle mismatch")
        decision_sha256 = hashlib.sha256(
            bundle.policy.decision_prompt.encode("utf-8")
        ).hexdigest()
        if result.fingerprints.get("decision_prompt") != decision_sha256:
            raise TaskFeedbackError("task feedback trace Decision bundle mismatch")
        if (
            result.numerical.component_fingerprints.get("numerical_supply_release")
            != bundle.numerical_release_sha256
        ):
            raise TaskFeedbackError("task feedback trace Numerical bundle mismatch")
        trace_sha256 = _digest(
            {
                "task": task_id,
                "partition": partition,
                "principals": list(_principal_key(bundle)),
                "retrieval": result.retrieval_card.to_payload(),
                "decision": result.final_decision.selected.candidate_id,
                "result_fingerprints": dict(result.fingerprints),
            }
        )
        key = (_principal_key(bundle), task_id)
        existing = self._traces.get(key)
        if existing is not None and existing.trace_sha256 != trace_sha256:
            raise TaskFeedbackError("task feedback trace changed for the same bundle")
        self._traces[key] = _VerifiedTrace(
            principal_key=key[0],
            task=task,
            result=result,
            partition=cast(Partition, partition),
            trace_sha256=trace_sha256,
        )

    def build_projection(
        self,
        bundle: PackageCoordinateBundle,
        task_ids: Sequence[str],
        *,
        generation: int,
    ) -> TaskEvidenceProjection:
        if type(bundle) is not PackageCoordinateBundle:
            raise TaskFeedbackError("task feedback projection requires an exact bundle")
        resolved = tuple(task_ids)
        if (
            not resolved
            or len(resolved) != len(set(resolved))
            or any(
                type(task_id) is not str or task_id not in self.partition_by_task
                for task_id in resolved
            )
        ):
            raise TaskFeedbackError("task feedback projection membership is invalid")
        if type(generation) is not int or generation < 1:
            raise TaskFeedbackError("task feedback generation must be positive")
        principal = _principal_key(bundle)
        traces: list[_VerifiedTrace] = []
        for task_id in sorted(resolved):
            trace = self._traces.get((principal, task_id))
            if trace is None:
                raise TaskFeedbackError(
                    "task feedback trace is missing for this bundle"
                )
            traces.append(trace)
        namespace = _digest(
            {
                "source_bundle_sha256": bundle.fingerprint(),
                "generation": generation,
                "trace_sha256s": [item.trace_sha256 for item in traces],
            }
        )
        pending: list[tuple[_VerifiedTrace, RetrievalAssumption, frozenset[str]]] = []
        for trace in traces:
            handoff = tuple(
                RetrievalAssumption.from_payload(dict(item))
                for item in trace.result.numerical.retrieval_handoff
            )
            groundings = tuple(trace.result.numerical.accepted_assumptions)
            if len(handoff) != len(groundings):
                raise TaskFeedbackError(
                    "task feedback assumption mapping is incomplete"
                )
            allowed = {item.assumption_id for item in handoff}
            if any(
                assumption_id not in allowed
                for chain in trace.result.retrieval_card.chains
                for assumption_id in chain.addressed_assumption_ids
            ):
                raise TaskFeedbackError(
                    "task feedback contains an unknown assumption ID"
                )
            pending.extend(
                (trace, assumption, frozenset(grounding.candidate_names))
                for assumption, grounding in zip(handoff, groundings, strict=True)
            )
        cases: list[TaskEvidenceCase] = []
        for index, (trace, assumption, targets) in enumerate(pending):
            chains = tuple(
                item
                for item in trace.result.retrieval_card.chains
                if assumption.assumption_id in item.addressed_assumption_ids
            )
            chain_sha256 = _digest(
                [
                    item.to_payload()
                    for item in sorted(chains, key=lambda item: item.chain_id)
                ]
            )
            selected = trace.result.final_decision.selected.candidate_id
            action = (
                "unresolved"
                if not targets
                else "selected" if selected in targets else "rejected"
            )
            cases.append(
                TaskEvidenceCase(
                    case_id=f"case_{index:03d}_{namespace[:8]}",
                    morphology=_morphology(trace.result),
                    assumption_id=assumption.assumption_id,
                    claim=assumption.claim,
                    failure_condition=assumption.failure_condition,
                    stance=cast(str, _stance(chains)),
                    target_match=(
                        "matched"
                        if any(
                            item.entity_match and item.target_match for item in chains
                        )
                        else "unmatched"
                    ),
                    window_relation=cast(str, _window_relation(chains)),
                    magnitude_status=cast(str, _magnitude_status(chains)),
                    mechanism=cast(str, _mechanism(chains)),
                    decision_action=cast(str, action),
                    evidence_chain_sha256=chain_sha256,
                )
            )
        return TaskEvidenceProjection(
            source_bundle_sha256=bundle.fingerprint(),
            request_namespace_sha256=namespace,
            cases=tuple(cases),
        )


__all__ = ["PackageTaskFeedbackLedger"]
