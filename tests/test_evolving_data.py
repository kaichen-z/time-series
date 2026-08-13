from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evolving_agent.data import load_context_tasks, load_tasks


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
