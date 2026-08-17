"""Dictionary-curation adapter with all forecasting behavior injected."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence, cast

from common.evolution_core.acceptance import MetricAcceptanceGate
from common.evolution_core.contracts import (
    EvaluationReport,
    EvolutionComponents,
    MetricSpec,
    MutationContext,
)
from common.evolution_core.persistence import JsonArtifactStore

from ..config import DictionaryCurationConfig
from ..dictionary import (
    MethodCandidate,
    MethodRecord,
    MethodStatus,
    ToolDictionary,
)
from ..providers import (
    ImplementationContext,
    MethodImplementer,
    RuntimeRegistry,
    SanitizedMethodFeedback,
)


MetricFunction = Callable[[Sequence[float], Sequence[float]], float]


@dataclass(frozen=True)
class NumericalTaskItem:
    """Label-free numerical input passed to method runtimes."""

    item_id: str
    history: tuple[float, ...]
    horizon: int
    frequency: str
    characteristics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.item_id or not self.history:
            raise ValueError("numerical task item needs an ID and non-empty history")
        if self.horizon <= 0:
            raise ValueError("numerical task horizon must be positive")


@dataclass(frozen=True)
class MethodExecutionResult:
    dictionary_id: str
    method_id: str
    item_id: str
    status: str
    forecast: tuple[float, ...] = ()
    error: str = ""


class DictionaryArtifactAdapter:
    def __init__(self, config: DictionaryCurationConfig) -> None:
        self.config = config

    def validate(self, artifact: ToolDictionary) -> None:
        for record in artifact.methods:
            typed = cast(MethodRecord, record)
            if typed.definition.family not in self.config.allowed_families:
                raise ValueError(
                    f"method family {typed.definition.family!r} is disabled by task config"
                )
            if typed.status not in self.config.method_statuses:
                raise ValueError(f"method status {typed.status!r} is disabled by task config")

    def artifact_id(self, artifact: ToolDictionary) -> str:
        return artifact.dictionary_id

    def to_payload(self, artifact: ToolDictionary) -> dict[str, object]:
        return artifact.to_payload()

    def from_payload(self, payload: Mapping[str, object]) -> ToolDictionary:
        return ToolDictionary.from_payload(payload)

    def apply_train_report(
        self, artifact: ToolDictionary, report: EvaluationReport
    ) -> ToolDictionary:
        if report.split != "train":
            raise ValueError("dictionary artifacts may be updated only from Train reports")
        per_method = report.diagnostics.get("per_method", {})
        if not isinstance(per_method, Mapping):
            raise ValueError("Train report per_method diagnostics must be an object")
        updated = []
        for record in artifact.methods:
            typed = cast(MethodRecord, record)
            if typed.candidate is None and typed.status == "unimplemented":
                updated.append(typed)
                continue
            summary = per_method.get(typed.definition.method_id, {})
            if not isinstance(summary, Mapping):
                summary = {}
            numeric_summary = {
                str(key): float(value)
                for key, value in summary.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            status = self._classify(summary)
            updated.append(
                replace(typed, status=status, train_summary=numeric_summary)
            )
        return replace(artifact, methods=tuple(updated))

    def _classify(self, summary: Mapping[str, object]) -> MethodStatus:
        total_count = int(summary.get("total_count", 0))
        success_count = int(summary.get("success_count", 0))
        if int(summary.get("unsafe_count", 0)) > 0:
            return "discarded"
        if total_count > 0 and int(summary.get("unavailable_count", 0)) == total_count:
            return "unavailable"
        if success_count <= 0:
            return "quarantined"
        mean_error = float(summary.get("mean_error", math.inf))
        subset_win_rate = float(summary.get("subset_win_rate", 0.0))
        if mean_error <= self.config.accepted_max_error:
            return "accepted"
        if subset_win_rate > 0 and mean_error <= self.config.specialized_max_error:
            return "specialized"
        dominated = bool(summary.get("dominated", False))
        if dominated and self.config.discard_requires_dominance_evidence:
            return "discarded"
        return "quarantined"


class DictionaryMutator:
    def __init__(
        self,
        config: DictionaryCurationConfig,
        implementer: MethodImplementer,
    ) -> None:
        self.config = config
        self.implementer = implementer

    def propose(
        self,
        parent: ToolDictionary,
        context: MutationContext,
        count: int,
    ) -> Sequence[ToolDictionary]:
        children = []
        for child_index in range(1, count + 1):
            records = tuple(
                self._mutate_record(record, parent, context)
                for record in parent.methods
            )
            children.append(
                ToolDictionary(
                    dictionary_id=(
                        f"{parent.dictionary_id}.g{context.generation:03d}.c{child_index:02d}"
                    ),
                    parent_dictionary_id=parent.dictionary_id,
                    generation=context.generation,
                    methods=records,
                )
            )
        return tuple(children)

    def _mutate_record(
        self,
        raw_record: MethodRecord | object,
        parent: ToolDictionary,
        context: MutationContext,
    ) -> MethodRecord:
        record = cast(MethodRecord, raw_record)
        if record.status == "unimplemented" and record.candidate is None:
            try:
                candidate = self.implementer.implement(
                    record.definition,
                    ImplementationContext(
                        dictionary_id=parent.dictionary_id,
                        generation=context.generation,
                    ),
                )
            except Exception:
                return replace(record, status="quarantined")
            if candidate.method_id != record.definition.method_id:
                return replace(record, status="quarantined")
            return replace(record, candidate=candidate, status="unimplemented")

        if (
            record.status == "quarantined"
            and record.candidate is not None
            and record.revision_count < self.config.max_revisions_per_method
        ):
            feedback = self._feedback(record.definition.method_id, context)
            try:
                candidate = self.implementer.revise(record.candidate, feedback)
            except Exception:
                return record
            if candidate.method_id != record.definition.method_id:
                return record
            return replace(
                record,
                candidate=candidate,
                status="unimplemented",
                revision_count=record.revision_count + 1,
            )
        return record

    def _feedback(
        self, method_id: str, context: MutationContext
    ) -> SanitizedMethodFeedback:
        per_method = context.parent_train_report.diagnostics.get("per_method", {})
        summary: Mapping[str, object] = {}
        if isinstance(per_method, Mapping):
            raw_summary = per_method.get(method_id, {})
            if isinstance(raw_summary, Mapping):
                summary = raw_summary
        metrics = {
            str(key): float(value)
            for key, value in summary.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        categories = []
        if int(summary.get("unavailable_count", 0)) > 0:
            categories.append("unavailable")
        if int(summary.get("invalid_count", 0)) > 0:
            categories.append("invalid")
        if float(summary.get("mean_error", 0.0)) > self.config.accepted_max_error:
            categories.append("high_error")
        # A tuple of strings, so it must bypass the float-only metrics filter above.
        raw_errors = summary.get("sample_errors", ())
        sample_errors = (
            tuple(str(item) for item in raw_errors)
            if isinstance(raw_errors, Sequence) and not isinstance(raw_errors, (str, bytes))
            else ()
        )
        return SanitizedMethodFeedback(method_id, metrics, tuple(categories), sample_errors)


class DictionaryExecutor:
    def __init__(self, runtimes: RuntimeRegistry) -> None:
        self.runtimes = runtimes

    def execute(
        self,
        artifact: ToolDictionary,
        items: Sequence[NumericalTaskItem],
        split: str,
    ) -> Sequence[MethodExecutionResult]:
        results = []
        for raw_record in artifact.methods:
            record = cast(MethodRecord, raw_record)
            if record.status == "discarded":
                continue
            for item in items:
                results.append(self._execute_one(artifact, record, item))
        return tuple(results)

    def _execute_one(
        self,
        artifact: ToolDictionary,
        record: MethodRecord,
        item: NumericalTaskItem,
    ) -> MethodExecutionResult:
        candidate = record.candidate
        if candidate is None:
            return MethodExecutionResult(
                artifact.dictionary_id,
                record.definition.method_id,
                item.item_id,
                "invalid",
                error="method has no implementation candidate",
            )
        if bool(candidate.implementation.get("unsafe", False)):
            return MethodExecutionResult(
                artifact.dictionary_id,
                record.definition.method_id,
                item.item_id,
                "unsafe",
                error="candidate failed common safety validation",
            )
        resolution = self.runtimes.resolve(candidate)
        if not resolution.available or resolution.runtime is None:
            return MethodExecutionResult(
                artifact.dictionary_id,
                record.definition.method_id,
                item.item_id,
                "unavailable",
                error=resolution.reason,
            )
        try:
            raw_forecast = resolution.runtime.forecast(
                candidate, item.history, item.horizon, item.frequency
            )
            forecast = tuple(float(value) for value in raw_forecast)
        except Exception as exc:
            return MethodExecutionResult(
                artifact.dictionary_id,
                record.definition.method_id,
                item.item_id,
                "invalid",
                error=str(exc) or type(exc).__name__,
            )
        if len(forecast) != item.horizon or any(
            not math.isfinite(value) for value in forecast
        ):
            return MethodExecutionResult(
                artifact.dictionary_id,
                record.definition.method_id,
                item.item_id,
                "invalid",
                error="forecast shape or values are invalid",
            )
        return MethodExecutionResult(
            artifact.dictionary_id,
            record.definition.method_id,
            item.item_id,
            "success",
            forecast=forecast,
        )


class DictionaryEvaluator:
    def __init__(
        self,
        config: DictionaryCurationConfig,
        labels: Mapping[str, Mapping[str, Sequence[float]]],
        metric: MetricFunction,
    ) -> None:
        self.config = config
        self.labels = {
            split: {item_id: tuple(float(value) for value in truth) for item_id, truth in values.items()}
            for split, values in labels.items()
        }
        self.metric = metric

    def evaluate(
        self,
        artifact_id: str,
        results: Sequence[MethodExecutionResult],
        split: str,
    ) -> EvaluationReport:
        split_labels = self.labels.get(split)
        if split_labels is None:
            raise ValueError(f"labels are unavailable for split {split!r}")
        by_method: dict[str, list[MethodExecutionResult]] = {}
        by_item_errors: dict[str, list[float]] = {}
        for result in results:
            by_method.setdefault(result.method_id, []).append(result)
            if result.status == "success":
                truth = split_labels.get(result.item_id)
                if truth is None:
                    raise ValueError(f"missing trusted label for item {result.item_id!r}")
                error = float(self.metric(result.forecast, truth))
                if not math.isfinite(error):
                    raise ValueError("method metric must be finite")
                by_item_errors.setdefault(result.item_id, []).append(error)

        per_method = {
            method_id: self._method_summary(method_results, split_labels)
            for method_id, method_results in by_method.items()
        }
        penalty = self.config.specialized_max_error * 10.0
        dictionary_errors = [
            min(by_item_errors.get(item_id, (penalty,))) for item_id in split_labels
        ]
        dictionary_score = statistics.fmean(dictionary_errors)
        failure_traces = [
            {
                "method_id": method_id,
                "mean_error": summary.get("mean_error"),
                "failure": self._failure_category(summary),
            }
            for method_id, summary in per_method.items()
            if self._failure_category(summary)
        ]
        return EvaluationReport(
            artifact_id=artifact_id,
            split=split,
            metrics={self.config.dictionary_metric: dictionary_score},
            item_count=len(split_labels),
            diagnostics={"per_method": per_method, "failure_traces": failure_traces},
        )

    def _method_summary(
        self,
        results: Sequence[MethodExecutionResult],
        labels: Mapping[str, Sequence[float]],
    ) -> dict[str, object]:
        errors = []
        for result in results:
            if result.status == "success":
                truth = labels.get(result.item_id)
                if truth is None:
                    raise ValueError(f"missing trusted label for item {result.item_id!r}")
                errors.append(float(self.metric(result.forecast, truth)))
        total_count = len(results)
        summary: dict[str, object] = {
            "total_count": total_count,
            "success_count": len(errors),
            "unavailable_count": sum(result.status == "unavailable" for result in results),
            "unsafe_count": sum(result.status == "unsafe" for result in results),
            "invalid_count": sum(result.status == "invalid" for result in results),
            "success_rate": len(errors) / total_count if total_count else 0.0,
            "sample_errors": self._sample_errors(results),
        }
        if errors:
            summary.update(
                {
                    "mean_error": statistics.fmean(errors),
                    "median_error": statistics.median(errors),
                    "worst_error": max(errors),
                    "subset_win_rate": sum(
                        error <= self.config.accepted_max_error for error in errors
                    )
                    / len(errors),
                    "dominated": all(
                        error > self.config.specialized_max_error for error in errors
                    ),
                }
            )
        return summary

    @staticmethod
    def _sample_errors(results: Sequence[MethodExecutionResult]) -> tuple[str, ...]:
        """Up to 3 distinct failure messages, first-seen order, for the repair prompt.

        Deduplicated with a dict, not a set: set iteration order isn't reproducible
        across runs since Python randomizes string hashing per process.
        """
        distinct = dict.fromkeys(
            result.error[:200] for result in results if result.error and result.status != "success"
        )
        return tuple(distinct)[:3]

    @staticmethod
    def _failure_category(summary: Mapping[str, object]) -> str:
        if int(summary.get("unsafe_count", 0)) > 0:
            return "unsafe"
        if int(summary.get("unavailable_count", 0)) > 0:
            return "unavailable"
        if int(summary.get("invalid_count", 0)) > 0:
            return "invalid"
        if bool(summary.get("dominated", False)):
            return "high_error"
        return ""


@dataclass(frozen=True)
class DictionaryCurationTask:
    """Compose generic evolution components from externally supplied providers."""

    base_dictionary: ToolDictionary
    config: DictionaryCurationConfig
    implementer: MethodImplementer
    runtimes: RuntimeRegistry
    labels: Mapping[str, Mapping[str, Sequence[float]]]
    metric: MetricFunction
    store: JsonArtifactStore

    def components(
        self,
    ) -> EvolutionComponents[ToolDictionary, NumericalTaskItem, MethodExecutionResult]:
        return EvolutionComponents(
            artifact_adapter=DictionaryArtifactAdapter(self.config),
            mutator=DictionaryMutator(self.config, self.implementer),
            executor=DictionaryExecutor(self.runtimes),
            evaluator=DictionaryEvaluator(self.config, self.labels, self.metric),
            acceptance_gate=MetricAcceptanceGate(
                MetricSpec(self.config.dictionary_metric, "minimize")
            ),
            store=self.store,
        )
