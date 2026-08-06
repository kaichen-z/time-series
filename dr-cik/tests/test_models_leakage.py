"""Structural checks that agent-facing code can never see benchmark labels."""

from __future__ import annotations

import ast
from pathlib import Path

from dr_cik.models import Document, ForecastTask

SRC = Path(__file__).resolve().parent.parent / "src" / "dr_cik"
AGENT_FACING_FILES = [
    SRC / "agents" / "common.py",
    SRC / "agents" / "opendr.py",
    SRC / "agents" / "drbench.py",
    SRC / "retrieval.py",
    SRC / "llm.py",
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
        documents=(Document("doc1", "text", role="supporting", subtype=None),),
        gt_evidence=({"id": "E1", "evidence": "fact"},),
        labels_public=True,
    )


def _referenced_names(source: str) -> set[str]:
    """Every identifier used as code (imports, annotations, calls) — ignores string/comment text."""
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


def test_task_view_has_no_leakage_attributes() -> None:
    view = _task().agent_view()
    assert not hasattr(view, "future_values")
    assert not hasattr(view, "gt_evidence")
    assert not hasattr(view, "labels_public")


def test_agent_document_has_no_leakage_attributes() -> None:
    view = _task().agent_view()
    document_view = view.documents[0]
    assert not hasattr(document_view, "role")
    assert not hasattr(document_view, "subtype")
