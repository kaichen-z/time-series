from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.data import ContextTask, Document, Task, load_context_tasks
from evolving_loop.decision_agent.agent import DecisionCandidate, DecisionResult
from evolving_loop.frozen_inference import inference_view, run_frozen_inference
from evolving_loop.cli import build_parser, inference_command
from evolving_loop.harness import HarnessResult
from evolving_loop.retrieval_agent.agent import Evidence, RetrievalResult


def _task(*, public: bool) -> ContextTask:
    return ContextTask(
        numeric=Task(
            task_id="task_hidden" if not public else "task_public",
            history_values=(1.0, 2.0, 3.0),
            future_values=(4.0, 5.0) if public else (),
            prediction_length=2,
            frequency="1 day",
            seasonal_period=None,
            entity_name="Entity",
        ),
        target_name="value",
        target_description="A value",
        history_timestamps=("1", "2", "3"),
        future_timestamps=("4", "5"),
        documents=(Document("doc_1", "A relevant sentence.", "supporting", "x"),),
        gt_evidence=("secret evaluator evidence",),
        labels_public=public,
    )


def _result(task_id: str) -> HarnessResult:
    candidate = DecisionCandidate(
        candidate_id="level",
        forecast=(3.0, 3.0),
        assumption="level persists",
        failure_condition="regime changes",
        hindcast_smape=1.0,
    )
    retrieval = RetrievalResult(
        query="q",
        selected_document_ids=("doc_1",),
        evidence=(Evidence("doc_1", "claim", "A relevant sentence."),),
        impacts=(),
        sufficient=True,
        missing_information=(),
    )
    decision = DecisionResult(
        selected=candidate,
        host_default_id="level",
        requested_more_retrieval=False,
        rationale="validated",
        supporting_document_ids=(),
        llm_override_accepted=False,
    )
    return HarnessResult(
        task_id=task_id,
        coding=object(),
        retrieval=retrieval,
        decision=decision,
        candidates=(candidate,),
        forecast=candidate.forecast,
    )


def test_hidden_loader_retains_task_but_strips_evaluator_fields(tmp_path: Path) -> None:
    record = {
        "benchmark_id": "task_hidden",
        "labels_public": False,
        "showcase": {
            "entity": {"name": "Entity"},
            "time_series_variable": {"name": "value"},
        },
        "task_metadata": {"prediction_length": 2, "frequency": "1 day"},
        "series": {
            "history_values": [1, 2, 3],
            "history_timestamps": ["1", "2", "3"],
            "future_timestamps": ["4", "5"],
            "future_values": [4, 5],
        },
        "documents": [
            {
                "document_id": "doc_1",
                "content": "text",
                "role": "supporting",
                "subtype": "leaky",
            }
        ],
        "annotations": {"gt_evidence": [{"evidence": "secret"}]},
    }
    path = tmp_path / "hidden.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_context_tasks(path) == []
    loaded = load_context_tasks(path, include_unlabeled=True)
    assert len(loaded) == 1
    task = loaded[0]
    assert task.numeric.future_values == ()
    assert task.gt_evidence == ()
    assert task.documents[0].role is None
    assert task.documents[0].subtype is None
    assert not task.labels_public


def test_frozen_inference_strips_labels_and_exports_submission(tmp_path: Path) -> None:
    observed = []

    class Harness:
        def run(self, task):
            observed.append(task)
            return _result(task.numeric.task_id)

    summary = run_frozen_inference(
        HarnessPolicy(),
        [_task(public=False)],
        lambda _: Harness(),
        output_dir=tmp_path,
        samples=3,
        score_public=False,
    )
    assert summary["labels_accessed"] is False
    assert observed[0] == inference_view(_task(public=False))
    forecast = json.loads((tmp_path / "forecasts.jsonl").read_text(encoding="utf-8"))
    research = json.loads((tmp_path / "deep_research.jsonl").read_text(encoding="utf-8"))
    assert forecast == {
        "benchmark_id": "task_hidden",
        "samples": [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]],
    }
    assert research["cited_document_ids"] == ["doc_1"]


def test_hidden_task_cannot_be_scored_by_frozen_runner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        run_frozen_inference(
            HarnessPolicy(),
            [_task(public=False)],
            lambda _: object(),
            output_dir=tmp_path,
            score_public=True,
        )


def test_cli_rejects_hidden_scoring_before_loading_data() -> None:
    args = build_parser().parse_args(
        ["--inference", "genome", "--hidden-test", "--score-public"]
    )
    with pytest.raises(ValueError, match="forbidden"):
        inference_command(args)
