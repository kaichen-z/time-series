from __future__ import annotations

import json

import pytest

from evolving_loop.benchmark_numeric_adapter import BenchmarkNumericWindow
from evolving_loop.official_benchmark_loader import (
    OfficialBenchmarkLoaderError,
    OfficialBenchmarkPartitions,
    load_anchor_cache,
    write_official_engine_inputs,
)
from evolving_loop.data import Document, load_context_tasks_by_ids
from evolving_loop.run_package_coevolution import _validated_split


def _window(task_id: str) -> BenchmarkNumericWindow:
    return BenchmarkNumericWindow(
        task_id=task_id,
        group_id="evaluator-only",
        history=(1.0, 2.0),
        truth=(3.0,),
        frequency="D",
        anchor_forecasts={"anchor": (2.5,)},
    )


def test_anchor_cache_schema_is_explicit_and_frozen(tmp_path) -> None:
    path = tmp_path / "anchors.json"
    path.write_text(
        json.dumps(
            {
                "schema": "official-anchor-cache-v1",
                "entries": {"external:a": {"timesfm": [1, 2], "moirai": [3, 4]}},
            }
        ),
        encoding="utf-8",
    )
    cache = load_anchor_cache(path)
    assert cache["external:a"]["timesfm"] == (1.0, 2.0)
    with pytest.raises(TypeError):
        cache["external:a"] = {}  # type: ignore[index]


def test_anchor_cache_rejects_unversioned_payload(tmp_path) -> None:
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps({"entries": {"a": {"x": [1]}}}), encoding="utf-8")
    with pytest.raises(OfficialBenchmarkLoaderError, match="schema"):
        load_anchor_cache(path)


def test_official_partitions_require_nonempty_disjoint_identities() -> None:
    with pytest.raises(OfficialBenchmarkLoaderError, match="disjoint"):
        OfficialBenchmarkPartitions(
            train=(_window("same"),),
            dev=(_window("dev"),),
            sealed_test_task_ids=("same",),
            source_fingerprint="f" * 64,
        )


def test_writer_binds_runner_tasks_split_and_anchors_without_test_labels(tmp_path) -> None:
    partitions = OfficialBenchmarkPartitions(
        train=(_window("train_a"),),
        dev=(_window("dev_a"),),
        sealed_test_task_ids=("test_a",),
        source_fingerprint="f" * 64,
    )
    files = write_official_engine_inputs(
        partitions, dataset="synthetic-official", destination=tmp_path / "inputs"
    )
    _payload, train_ids, dev_ids = _validated_split(files.split_manifest)
    tasks = load_context_tasks_by_ids(files.tasks_file, (*train_ids, *dev_ids))
    anchors = json.loads(files.anchor_bindings.read_text(encoding="utf-8"))
    receipt = json.loads(files.receipt.read_text(encoding="utf-8"))

    assert train_ids == ("train_a",)
    assert dev_ids == ("dev_a",)
    assert tuple(task.numeric.task_id for task in tasks) == ("train_a", "dev_a")
    assert set(anchors["entries"]) == {"train_a", "dev_a"}
    assert "test_a" not in files.tasks_file.read_text(encoding="utf-8")
    assert receipt["test_labels_materialized"] is False
    with pytest.raises(OfficialBenchmarkLoaderError, match="nonempty"):
        OfficialBenchmarkPartitions(
            train=(),
            dev=(_window("dev"),),
            sealed_test_task_ids=("test",),
            source_fingerprint="f" * 64,
        )


def test_writer_preserves_public_context_without_evaluator_labels(tmp_path) -> None:
    contextual = BenchmarkNumericWindow(
        task_id="train_context",
        group_id="evaluator-only",
        history=(1.0, 2.0),
        truth=(3.0,),
        frequency="D",
        anchor_forecasts={"anchor": (2.5,)},
        history_timestamps=("2026-01-01", "2026-01-02"),
        future_timestamps=("2026-01-03",),
        documents=(Document("scenario", "A public event occurs."),),
        target_description="daily target",
    )
    files = write_official_engine_inputs(
        OfficialBenchmarkPartitions(
            train=(contextual,),
            dev=(_window("dev_context"),),
            sealed_test_task_ids=("sealed_context",),
            source_fingerprint="f" * 64,
        ),
        dataset="synthetic-context",
        destination=tmp_path / "context-inputs",
    )
    tasks = load_context_tasks_by_ids(
        files.tasks_file, ("train_context", "dev_context")
    )
    assert tasks[0].history_timestamps == ("2026-01-01", "2026-01-02")
    assert tasks[0].future_timestamps == ("2026-01-03",)
    assert tasks[0].target_description == "daily target"
    assert tasks[0].documents == (
        Document("scenario", "A public event occurs."),
    )
    assert tasks[0].gt_evidence == ()
