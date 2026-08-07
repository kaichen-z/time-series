"""Loader correctness against the real official sample tasks."""

from __future__ import annotations

from .conftest import requires_sample


@requires_sample
def test_load_sample_tasks_returns_three_real_tasks(sample_tasks) -> None:
    ids = {task.benchmark_id for task in sample_tasks}
    assert ids == {"task_42", "task_163", "task_201"}


@requires_sample
def test_loaded_task_has_supporting_and_distractor_documents(sample_tasks) -> None:
    task = next(task for task in sample_tasks if task.benchmark_id == "task_163")
    roles = {document.role for document in task.documents}
    assert roles == {"supporting", "distractor"}
    assert task.future_values is not None
    assert len(task.future_values) == task.prediction_length
    assert task.gt_evidence


@requires_sample
def test_agent_view_strips_labels_but_keeps_document_text(sample_tasks) -> None:
    task = sample_tasks[0]
    view = task.agent_view()
    assert len(view.documents) == len(task.documents)
    assert view.documents[0].text == task.documents[0].text
