from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from .models import ForecastMemoryEntry, ForecastTask, ForecastWorkspace


class ForecastMemoryBank:
    """Post-hoc memory populated only after actual future values are available."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser().resolve() if path else None
        self.entries: list[ForecastMemoryEntry] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(ForecastMemoryEntry(**json.loads(line)))

    def query(self, event_type: str, action_type: str, limit: int = 5) -> list[ForecastMemoryEntry]:
        matches = [
            item
            for item in self.entries
            if item.event_type == event_type and item.action_type == action_type
        ]
        return matches[-limit:]

    def record_outcome(
        self,
        task: ForecastTask,
        workspace: ForecastWorkspace,
    ) -> list[ForecastMemoryEntry]:
        if task.future_values is None:
            raise ValueError("post-hoc learning requires outcomes that were unavailable at inference time")
        baseline_mae = statistics.fmean(
            abs(actual - predicted)
            for actual, predicted in zip(task.future_values, workspace.baseline_values)
        )
        revised_mae = statistics.fmean(
            abs(actual - predicted)
            for actual, predicted in zip(task.future_values, workspace.final_values)
        )
        created: list[ForecastMemoryEntry] = []
        for record in workspace.revision_records:
            action = record.action
            if not record.accepted or action.action_type not in {"multiply", "add"}:
                continue
            neutral = 1.0 if action.action_type == "multiply" else 0.0
            recommended = action.value if revised_mae < baseline_mae else neutral
            lesson = (
                "The evidence-backed revision improved MAE; retain this magnitude as a prior."
                if revised_mae < baseline_mae
                else "The baseline was at least as accurate; shrink this event adjustment toward neutral."
            )
            raw_id = repr((task.benchmark_id, action.action_id, baseline_mae, revised_mae))
            entry = ForecastMemoryEntry(
                entry_id=hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16],
                source_task_id=task.benchmark_id,
                event_type=action.event_type,
                action_type=action.action_type,
                proposed_value=action.value,
                recommended_value=recommended,
                baseline_mae=baseline_mae,
                revised_mae=revised_mae,
                lesson=lesson,
            )
            if not any(item.entry_id == entry.entry_id for item in self.entries):
                self.entries.append(entry)
                created.append(entry)
        if created and self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in self.entries),
                encoding="utf-8",
            )
        return created
