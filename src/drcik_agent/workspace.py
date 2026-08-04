from __future__ import annotations

import hashlib
import math
import statistics

from .agents import ProbabilisticForecastAgent
from .memory import ForecastMemoryBank
from .models import (
    Diagnosis,
    EvidenceImpact,
    ForecastAdjustment,
    ForecastTask,
    ForecastWorkspace,
    RevisionAction,
    RevisionRecord,
)


def _action_id(
    event_type: str,
    action_type: str,
    start_index: int,
    end_index: int,
    value: float | None,
    values: tuple[float, ...] = (),
) -> str:
    # Source IDs are intentionally excluded: corroborating documents must not
    # cause the same event effect to be applied a second time.
    payload = repr((event_type, action_type, start_index, end_index, value, values))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class RevisionPlannerAgent:
    """Turn evidence impacts into the small action language allowed by the workspace."""

    PRESERVE_KINDS = {
        "return_to_baseline",
        "outside_horizon",
        "already_in_baseline",
        "qualitative_only",
    }

    def __init__(self, memory: ForecastMemoryBank | None = None, memory_weight: float = 0.25) -> None:
        if not 0.0 <= memory_weight <= 1.0:
            raise ValueError("memory_weight must be between 0 and 1")
        self.memory = memory or ForecastMemoryBank()
        self.memory_weight = memory_weight

    @staticmethod
    def _indices(task: ForecastTask, impact: EvidenceImpact) -> list[int]:
        return ProbabilisticForecastAgent._affected_indices(task, impact)

    def propose(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        impacts: list[EvidenceImpact],
    ) -> list[RevisionAction]:
        proposals: list[RevisionAction] = []
        for impact in impacts:
            indices = self._indices(task, impact)
            start_index = min(indices) if indices else 0
            end_index = max(indices) if indices else task.prediction_length - 1
            action_type = "preserve"
            value = impact.adjustment_value
            if impact.adjustment_kind == "multiplier":
                action_type = "multiply"
            elif impact.adjustment_kind == "percentage":
                action_type = "multiply"
                value = 1.0 + (value or 0.0)
            elif impact.adjustment_kind == "standardized_additive":
                action_type = "add"
                value = (value or 0.0) * diagnosis.residual_scale
            elif impact.adjustment_kind == "absolute_additive":
                action_type = "add"
            elif impact.adjustment_kind not in self.PRESERVE_KINDS:
                action_type = "preserve"

            memory_hits = self.memory.query(impact.event_type, action_type)
            memory_ids = tuple(item.entry_id for item in memory_hits)
            usable = [item.recommended_value for item in memory_hits if item.recommended_value is not None]
            if action_type in {"multiply", "add"} and value is not None and usable:
                remembered = statistics.fmean(usable)
                value = (1.0 - self.memory_weight) * value + self.memory_weight * remembered

            evidence = (
                f"{impact.event_type}; relation={impact.forecast_relation}; "
                f"direction={impact.direction}; sources={','.join(impact.source_document_ids)}"
            )
            proposals.append(
                RevisionAction(
                    action_id=_action_id(
                        impact.event_type,
                        action_type,
                        start_index,
                        end_index,
                        value,
                    ),
                    action_type=action_type,
                    start_index=start_index,
                    end_index=end_index,
                    value=value,
                    source_document_ids=impact.source_document_ids,
                    event_type=impact.event_type,
                    evidence=evidence,
                    confidence=impact.confidence,
                    rationale=impact.rationale,
                    memory_entry_ids=memory_ids,
                )
            )
        return proposals

    def point_override(
        self,
        workspace: ForecastWorkspace,
        index: int,
        point_value: float,
        context_weight: float,
        source_document_ids: tuple[str, ...],
    ) -> RevisionAction:
        value = (
            (1.0 - context_weight) * workspace.final_values[index]
            + context_weight * point_value
        )
        return RevisionAction(
            action_id=_action_id("explicit_future_value", "override", index, index, value),
            action_type="override",
            start_index=index,
            end_index=index,
            value=value,
            source_document_ids=source_document_ids,
            event_type="explicit_future_value",
            evidence=f"Explicit value for {workspace.future_timestamps[index]}",
            confidence=max(0.5, context_weight),
            rationale="Blend the accepted explicit future value with the current workspace forecast.",
        )


class ForecastWorkspaceExecutor:
    """Validate and execute only restricted edits; never mutate the baseline."""

    ALLOWED_ACTIONS = {"initialize", "preserve", "multiply", "add", "clip", "override"}

    def initialize(
        self,
        task: ForecastTask,
        baseline_values: tuple[float, ...],
        baseline_method: str,
    ) -> ForecastWorkspace:
        workspace = ForecastWorkspace(
            benchmark_id=task.benchmark_id,
            history_timestamps=task.history_timestamps,
            history_values=task.history_values,
            future_timestamps=task.future_timestamps,
            baseline_method=baseline_method,
            baseline_values=tuple(baseline_values),
            final_values=list(baseline_values),
        )
        initialize = RevisionAction(
            action_id=_action_id("workspace", "initialize", 0, task.prediction_length - 1, None),
            action_type="initialize",
            start_index=0,
            end_index=task.prediction_length - 1,
            value=None,
            event_type="workspace",
            evidence="Forecast backbone output",
            confidence=1.0,
            rationale="Initialize y_final as an exact copy of immutable y_baseline.",
        )
        workspace.revision_records.append(
            RevisionRecord(initialize, True, "initialized_from_baseline", task.prediction_length, 0.0)
        )
        return workspace

    @staticmethod
    def _reject(action: RevisionAction, reason: str) -> RevisionRecord:
        return RevisionRecord(action, False, reason, 0, 0.0)

    def apply(self, workspace: ForecastWorkspace, action: RevisionAction) -> RevisionRecord:
        workspace.evidence_proposals.append(action)
        if action.action_type not in self.ALLOWED_ACTIONS or action.action_type == "initialize":
            record = self._reject(action, "action_not_allowed")
            workspace.revision_records.append(record)
            return record
        if any(item.action.action_id == action.action_id for item in workspace.revision_records):
            record = self._reject(action, "duplicate_revision")
            workspace.revision_records.append(record)
            return record
        horizon = len(workspace.final_values)
        if not (0 <= action.start_index <= action.end_index < horizon):
            record = self._reject(action, "range_outside_forecast_horizon")
            workspace.revision_records.append(record)
            return record
        if action.action_type != "preserve" and not action.source_document_ids:
            record = self._reject(action, "numerical_revision_requires_evidence")
            workspace.revision_records.append(record)
            return record
        if action.action_type != "preserve" and action.confidence < 0.5:
            record = self._reject(action, "insufficient_evidence_confidence")
            workspace.revision_records.append(record)
            return record
        if action.value is not None and not math.isfinite(action.value):
            record = self._reject(action, "non_finite_revision_value")
            workspace.revision_records.append(record)
            return record
        if action.action_type == "multiply" and (
            action.value is None or not 0.1 <= action.value <= 10.0
        ):
            record = self._reject(action, "unsafe_multiplier")
            workspace.revision_records.append(record)
            return record

        indices = list(range(action.start_index, action.end_index + 1))
        before = [workspace.final_values[index] for index in indices]
        nonnegative = min(workspace.history_values) >= 0
        if action.action_type == "multiply":
            for index in indices:
                workspace.final_values[index] *= action.value or 1.0
        elif action.action_type == "add":
            for index in indices:
                workspace.final_values[index] += action.value or 0.0
        elif action.action_type == "override":
            if action.values:
                if len(action.values) != len(indices):
                    record = self._reject(action, "override_length_mismatch")
                    workspace.revision_records.append(record)
                    return record
                for index, value in zip(indices, action.values):
                    workspace.final_values[index] = value
            elif action.value is not None:
                for index in indices:
                    workspace.final_values[index] = action.value
            else:
                record = self._reject(action, "override_requires_value")
                workspace.revision_records.append(record)
                return record
        elif action.action_type == "clip":
            if action.lower_bound is None and action.upper_bound is None:
                record = self._reject(action, "clip_requires_bound")
                workspace.revision_records.append(record)
                return record
            for index in indices:
                value = workspace.final_values[index]
                if action.lower_bound is not None:
                    value = max(action.lower_bound, value)
                if action.upper_bound is not None:
                    value = min(action.upper_bound, value)
                workspace.final_values[index] = value

        if nonnegative:
            for index in indices:
                workspace.final_values[index] = max(0.0, workspace.final_values[index])
        mean_change = statistics.fmean(
            abs(workspace.final_values[index] - old)
            for index, old in zip(indices, before)
        )
        preserved = action.action_type == "preserve"
        reason = "preserved_baseline" if preserved else "revision_applied"
        record = RevisionRecord(action, True, reason, 0 if preserved else len(indices), mean_change)
        workspace.revision_records.append(record)
        for entry_id in action.memory_entry_ids:
            if entry_id not in workspace.memory_entry_ids:
                workspace.memory_entry_ids.append(entry_id)
        # The immutable tuple is checked after every mutation so accidental
        # baseline edits fail immediately during development.
        if not isinstance(workspace.baseline_values, tuple):
            raise RuntimeError("y_baseline must remain immutable")
        return record

    @staticmethod
    def adjustments(workspace: ForecastWorkspace) -> tuple[ForecastAdjustment, ...]:
        return tuple(
            ForecastAdjustment(
                source_document_ids=record.action.source_document_ids,
                adjustment_kind=record.action.action_type,
                adjustment_value=record.action.value,
                affected_steps=record.affected_steps,
                mean_absolute_change=record.mean_absolute_change,
                rationale=record.action.rationale,
            )
            for record in workspace.revision_records
            if record.action.action_type != "initialize"
        )
