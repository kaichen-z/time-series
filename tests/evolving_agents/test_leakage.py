"""Structural checks that each agent can only see what its role permits."""

from __future__ import annotations

import ast
from pathlib import Path

from dr_cik.models import Document, ForecastTask

from evolving_agents.models import NumericTaskView, to_numeric_view

SRC = Path(__file__).resolve().parent.parent.parent / "evolving_agents"
CODING = SRC / "agents" / "coding.py"
# Modules that run at inference time and must never reach a labeled type.
AGENT_FACING_FILES = [
    SRC / "agents" / "common.py",
    SRC / "agents" / "coding.py",
    SRC / "harness" / "sandbox.py",
    SRC / "harness" / "hindcast.py",
    SRC / "harness" / "trace.py",
    SRC / "llm_cache.py",
]


def _task() -> ForecastTask:
    return ForecastTask(
        benchmark_id="t1",
        entity_name="Entity",
        target_name="target",
        target_description="desc",
        frequency="D",
        prediction_length=2,
        seasonal_period=None,
        history_timestamps=("a", "b"),
        history_values=(1.0, 2.0),
        future_timestamps=("c", "d"),
        future_values=(3.0, 4.0),
        documents=(Document("doc1", "secret text", role="supporting", subtype=None),),
        gt_evidence=({"id": "E1", "evidence": "fact"},),
        labels_public=True,
    )


def _referenced_names(source: str) -> set[str]:
    """Every identifier used as code (imports, annotations, calls) -- ignores string/comment text."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
    return names


def test_agent_facing_modules_never_reference_labeled_types() -> None:
    for path in AGENT_FACING_FILES:
        names = _referenced_names(path.read_text(encoding="utf-8"))
        assert "Document" not in names, f"{path} references the labeled Document type"
        assert "ForecastTask" not in names, f"{path} references the labeled ForecastTask type"


def test_coding_agent_never_references_any_document_type() -> None:
    names = _referenced_names(CODING.read_text(encoding="utf-8"))
    for forbidden in ("TaskView", "AgentDocument", "Document", "ForecastTask"):
        assert forbidden not in names, f"the Coding Agent references {forbidden}; it must see numbers only"


def test_numeric_view_has_no_textual_or_label_attributes() -> None:
    view = to_numeric_view(_task().agent_view())
    for forbidden in ("documents", "future_values", "gt_evidence", "labels_public", "target_description", "entity_name"):
        assert not hasattr(view, forbidden), f"NumericTaskView exposes {forbidden}"


def test_numeric_view_keeps_only_the_numeric_fields() -> None:
    view = to_numeric_view(_task().agent_view())
    assert view == NumericTaskView(
        benchmark_id="t1", history_values=(1.0, 2.0), prediction_length=2, frequency="D", seasonal_period=None
    )


def test_document_text_cannot_reach_the_numeric_view() -> None:
    assert "secret text" not in repr(to_numeric_view(_task().agent_view()))
