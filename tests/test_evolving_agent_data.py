from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolving_agent.data import Task, load_tasks, split_tasks


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
