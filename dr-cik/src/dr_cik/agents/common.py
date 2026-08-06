"""Prompt scaffolding shared by the OpenDR and DRBench agents."""

from __future__ import annotations

import statistics
from typing import Any

from ..models import AgentDocument, EvidenceItem, TaskView

AGENT_SYSTEM_PREAMBLE = (
    "You are a deep-research forecasting assistant. You are given a forecasting task "
    "(entity, target variable, historical time series, forecast horizon) and a corpus of "
    "short documents identified only by document_id. Some documents are relevant context; "
    "others are distractors planted to mislead you, and you are not told which is which. "
    "You may only see document text by using the search tool. Every claim you report must "
    "be traceable to a specific document_id returned by a search. Never invent facts."
)


def trend_word(history_values: tuple[float, ...]) -> str:
    """Describe a series as rising, falling, volatile, or stable from its recent values."""
    if len(history_values) < 2:
        return "stable"
    window = history_values[-min(len(history_values), 20) :]
    mean = statistics.fmean(window)
    spread = statistics.pstdev(window)
    if mean != 0 and spread / abs(mean) > 0.3:
        return "volatile"
    slope = window[-1] - window[0]
    scale = max(abs(mean), spread, 1e-8)
    if slope / scale > 0.15:
        return "rising"
    if slope / scale < -0.15:
        return "falling"
    return "stable"


def render_task_brief(view: TaskView) -> str:
    """Summarize a task's entity/target/history and list corpus document IDs, no text."""
    window = view.history_values[-min(len(view.history_values), 20) :]
    lines = [
        f"Entity: {view.entity_name}",
        f"Target variable: {view.target_name}",
        f"Target description: {view.target_description}",
        f"Frequency: {view.frequency}",
        f"History range: {view.history_timestamps[0]} to {view.history_timestamps[-1]} ({len(view.history_values)} points)",
        f"Recent values (last {len(window)}): min={min(window):.4g} max={max(window):.4g} mean={statistics.fmean(window):.4g} trend={trend_word(view.history_values)}",
        f"Forecast horizon: {view.future_timestamps[0]} to {view.future_timestamps[-1]} ({view.prediction_length} steps)",
        f"Corpus: {len(view.documents)} documents, ids: {', '.join(document.document_id for document in view.documents)}",
    ]
    return "\n".join(lines)


def parse_evidence_list(raw: list[dict[str, Any]], valid_document_ids: set[str]) -> tuple[EvidenceItem, ...]:
    """Validate evidence items and drop any citations to document IDs outside the corpus."""
    items: list[EvidenceItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        claim = entry.get("claim")
        source_ids = entry.get("source_doc_ids")
        if not isinstance(claim, str) or not claim.strip() or not isinstance(source_ids, list):
            continue
        filtered = tuple(str(doc_id) for doc_id in source_ids if str(doc_id) in valid_document_ids)
        items.append(EvidenceItem(claim=claim.strip(), source_doc_ids=filtered))
    return tuple(items)


def documents_by_id(documents: tuple[AgentDocument, ...]) -> dict[str, AgentDocument]:
    """Index a corpus by document_id for O(1) lookups."""
    return {document.document_id: document for document in documents}
