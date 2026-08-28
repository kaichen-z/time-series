"""Tests for evolving_loop/data: task loading, ContextTask conversion, and split-manifest generation."""
from __future__ import annotations

import json
import tempfile
import unittest
import pytest
from pathlib import Path
from evolving_loop.data import Task, load_context_tasks, load_tasks, split_tasks
from evolving_loop.split_manifest import (
    RECOMMENDED_PUBLIC_SPLIT_SIZES,
    build_split_manifest,
    load_public_records,
    write_split_manifest,
)


def _record(task_id: str, entity: str, future=(1.0, 2.0), history=(1.0, 2.0, 3.0)) -> dict:
    return {
        "benchmark_id": task_id,
        "entity_name": entity,
        "history_values": list(history),
        "future_values": list(future),
        "prediction_length": len(future),
        "frequency": "1 hour",
        "seasonal_period": "1h",
        "document_ids": ["doc_1"],
        "profile_details": {"secret": "should never be read"},
    }


def _write_jsonl(records: list[dict]) -> str:
    fd = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        fd.write(json.dumps(r) + "\n")
    fd.close()
    return fd.name


class LoadTasksTests(unittest.TestCase):
    def test_loads_only_labeled_records(self):
        records = [
            _record("task_1", "entity_a"),
            _record("task_2", "entity_a", future=(None,)),
            _record("task_3", "entity_b", future=[]),
        ]
        path = _write_jsonl(records)
        tasks = load_tasks(path)
        self.assertEqual([t.task_id for t in tasks], ["task_1"])

    def test_task_has_no_document_field(self):
        path = _write_jsonl([_record("task_1", "entity_a")])
        task = load_tasks(path)[0]
        self.assertIsInstance(task, Task)
        self.assertNotIn("documents", task.__dataclass_fields__)
        self.assertNotIn("profile_details", task.__dataclass_fields__)

    def test_numeric_fields_round_trip(self):
        path = _write_jsonl([_record("task_1", "entity_a", history=(1.0, 2.0, 3.0), future=(4.0, 5.0))])
        task = load_tasks(path)[0]
        self.assertEqual(task.history_values, (1.0, 2.0, 3.0))
        self.assertEqual(task.future_values, (4.0, 5.0))
        self.assertEqual(task.prediction_length, 2)
        self.assertEqual(task.frequency, "1 hour")


class SplitTasksTests(unittest.TestCase):
    def test_no_entity_straddles_the_split(self):
        records = [_record(f"task_{i}", f"entity_{i % 5}") for i in range(20)]
        path = _write_jsonl(records)
        tasks = load_tasks(path)
        train, test = split_tasks(tasks, seed=7, test_fraction=0.3)
        train_entities = {t.entity_name for t in train}
        test_entities = {t.entity_name for t in test}
        self.assertEqual(train_entities & test_entities, set())
        self.assertEqual(len(train) + len(test), len(tasks))

    def test_split_is_deterministic_for_a_fixed_seed(self):
        records = [_record(f"task_{i}", f"entity_{i % 7}") for i in range(30)]
        path = _write_jsonl(records)
        tasks = load_tasks(path)
        train_a, test_a = split_tasks(tasks, seed=7)
        train_b, test_b = split_tasks(tasks, seed=7)
        self.assertEqual([t.task_id for t in train_a], [t.task_id for t in train_b])
        self.assertEqual([t.task_id for t in test_a], [t.task_id for t in test_b])

    def test_empty_task_list(self):
        train, test = split_tasks([])
        self.assertEqual(train, [])
        self.assertEqual(test, [])


if __name__ == "__main__":
    unittest.main()

def test_nested_drcik_record_loads_numeric_and_context_views() -> None:
    record = {
        "benchmark_id": "task_42",
        "showcase": {
            "entity": {"name": "Nuance Cosmetic Lab"},
            "time_series_variable": {"name": "sales volume"},
        },
        "task_metadata": {
            "frequency": "1 day",
            "prediction_length": 2,
            "seasonal_period": "D",
            "target_description": "Daily sales volume.",
        },
        "series": {
            "history_timestamps": ["2026-01-01", "2026-01-02"],
            "history_values": [1, 2],
            "future_timestamps": ["2026-01-03", "2026-01-04"],
            "future_values": [3, 4],
        },
        "documents": [
            {
                "document_id": "doc_1",
                "content": "A relevant report.",
                "role": "supporting",
                "subtype": None,
            }
        ],
        "annotations": {"gt_evidence": [{"id": "E1", "evidence": "Relevant fact."}]},
        "labels_public": True,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tasks.jsonl"
        path.write_text(json.dumps(record) + "\n")
        numeric = load_tasks(path)[0]
        contextual = load_context_tasks(path)[0]

    assert numeric.entity_name == "Nuance Cosmetic Lab"
    assert numeric.history_values == (1.0, 2.0)
    assert contextual.target_name == "sales volume"
    assert contextual.documents[0].document_id == "doc_1"
    assert contextual.gt_evidence == ("Relevant fact.",)
    retrieval_view = contextual.retrieval_view()
    assert "future_values" not in retrieval_view
    assert "gt_evidence" not in retrieval_view
    assert "role" not in retrieval_view["documents"][0]


def test_context_loader_accepts_one_json_object() -> None:
    record = {
        "benchmark_id": "one",
        "entity_name": "entity",
        "target_name": "target",
        "frequency": "1 day",
        "prediction_length": 1,
        "history_values": [1, 2],
        "future_values": [3],
        "documents": [],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "task.json"
        path.write_text(json.dumps(record))
        contextual = load_context_tasks(path)
    assert len(contextual) == 1
    assert contextual[0].numeric.task_id == "one"


def test_context_loader_accepts_task_directory() -> None:
    template = {
        "entity_name": "entity",
        "target_name": "target",
        "frequency": "1 day",
        "prediction_length": 1,
        "history_values": [1, 2],
        "future_values": [3],
        "documents": [],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        for task_id in ("b", "a"):
            (path / f"{task_id}.json").write_text(
                json.dumps({**template, "benchmark_id": task_id})
            )
        contextual = load_context_tasks(path)
    assert [task.numeric.task_id for task in contextual] == ["a", "b"]

def _split_record(task_id: str, entity: str, *, frequency: str, hops: int, origin: str) -> dict:
    return {
        "benchmark_id": task_id,
        "origin": origin,
        "reasoning_hops": hops,
        "showcase": {"entity": {"name": entity}},
        "task_metadata": {
            "frequency": frequency,
            "prediction_length": 24 if frequency == "1 hour" else 100,
        },
        # Evaluator-only fields deliberately appear in the fixture. The split
        # manifest must never copy them into its output.
        "series": {"future_values": [999.0]},
        "annotations": {"gt_evidence": [{"evidence": "secret label"}]},
    }


def _records() -> list[dict]:
    rows = []
    entity_sizes = (2, 2, 2, 2, 1, 1, 1, 1)
    for entity_index, size in enumerate(entity_sizes):
        for offset in range(size):
            rows.append(
                _split_record(
                    f"task_{entity_index}_{offset}",
                    f"entity_{entity_index}",
                    frequency="1 hour" if (entity_index + offset) % 2 == 0 else "1 day",
                    hops=2 if entity_index % 2 == 0 else 4,
                    origin="synthetic" if offset == 0 else "human",
                )
            )
    return rows


def test_build_split_manifest_is_exact_deterministic_and_entity_disjoint() -> None:
    """Catches task loss, unstable assignment, or one entity leaking across partitions."""
    first = build_split_manifest(
        _records(), seed=17, train_size=6, val_size=3, public_test_size=3
    )
    second = build_split_manifest(
        list(reversed(_records())), seed=17, train_size=6, val_size=3, public_test_size=3
    )

    assert first == second
    assert first["actual_sizes"] == {"train": 6, "val": 3, "public_test": 3}

    partitions = first["partitions"]
    task_sets = [set(partitions[name]["task_ids"]) for name in ("train", "val", "public_test")]
    entity_sets = [set(partitions[name]["entities"]) for name in ("train", "val", "public_test")]
    assert set.union(*task_sets) == {
        "task_0_0", "task_0_1", "task_1_0", "task_1_1",
        "task_2_0", "task_2_1", "task_3_0", "task_3_1",
        "task_4_0", "task_5_0", "task_6_0", "task_7_0",
    }
    assert sum(len(items) for items in task_sets) == 12
    assert task_sets[0].isdisjoint(task_sets[1])
    assert task_sets[0].isdisjoint(task_sets[2])
    assert task_sets[1].isdisjoint(task_sets[2])
    assert entity_sets[0].isdisjoint(entity_sets[1])
    assert entity_sets[0].isdisjoint(entity_sets[2])
    assert entity_sets[1].isdisjoint(entity_sets[2])
    serialized = json.dumps(first)
    assert "999.0" not in serialized
    assert "secret label" not in serialized
    assert first["selection_uses_future_values"] is False
    assert first["selection_uses_gt_evidence"] is False
    assert first["manifest_sha256"]


def test_build_split_manifest_rejects_invalid_requested_total() -> None:
    """Catches silently dropping tasks when requested split sizes do not cover the dataset."""
    with pytest.raises(ValueError, match="must sum to the number of public tasks"):
        build_split_manifest(
            _records(), seed=17, train_size=5, val_size=3, public_test_size=3
        )


def test_recommended_public_split_reserves_dev_and_large_test() -> None:
    """Catches the formal protocol drifting away from 80/20 development and 99 test."""
    assert RECOMMENDED_PUBLIC_SPLIT_SIZES == {
        "train": 80,
        "val": 20,
        "public_test": 99,
    }


def test_write_split_manifest_round_trips_json(tmp_path) -> None:
    """Catches writing a different artifact than the manifest that was validated."""
    manifest = build_split_manifest(
        _records(), seed=17, train_size=6, val_size=3, public_test_size=3
    )
    destination = tmp_path / "drcik_public_v1.json"

    write_split_manifest(manifest, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == manifest
    assert destination.read_text(encoding="utf-8").endswith("\n")


def test_load_public_records_excludes_hidden_rows_from_mixed_directory(tmp_path) -> None:
    """Catches the official mixed export accidentally placing Hidden 80 into Public 199."""
    public = _split_record("task_public", "entity_public", frequency="1 day", hops=2, origin="human")
    public["labels_public"] = True
    hidden = _split_record("task_hidden", "entity_hidden", frequency="1 day", hops=2, origin="human")
    hidden["labels_public"] = False
    hidden["series"]["future_values"] = [None]
    (tmp_path / "task_public.json").write_text(json.dumps(public), encoding="utf-8")
    (tmp_path / "task_hidden.json").write_text(json.dumps(hidden), encoding="utf-8")

    rows = load_public_records(tmp_path)

    assert [row["benchmark_id"] for row in rows] == ["task_public"]
