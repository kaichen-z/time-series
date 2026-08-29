"""Executable contract for the one-task morphology smoke command."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

import numerical_agent.run_morphology_smoke as smoke
from numerical_agent.run_morphology_smoke import main


def _record(task_id: str, *, future: list[float] | None = None) -> dict[str, object]:
    return {
        "benchmark_id": task_id,
        "series": {
            "history_values": [10.0, 11.0, 10.0, 11.0] * 12,
            "future_values": future if future is not None else [11.0, 10.0, 11.0],
        },
        "task_metadata": {"prediction_length": 3, "frequency": "D"},
        "labels_public": True,
    }


def _write_tasks(path, *records: dict[str, object]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_fake_smoke_selects_one_task_freezes_then_writes_complete_result(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "nested" / "smoke.json"
    _write_tasks(tasks, _record("one"))

    completed = subprocess.run(
        [
            sys.executable, "-m", "numerical_agent.run_morphology_smoke",
            "--task-file", str(tasks), "--results-path", str(result), "--llm-backend", "fake",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["task_id"] == "one"
    assert len(payload["final_forecast"]) == 3
    assert set(payload) >= {
        "task_id", "selected", "final_forecast", "protected_baseline",
        "accepted_assumptions", "rejected_assumption_reason_counts",
        "selected_history_only_diagnostics", "baseline_history_only_diagnostics",
        "candidates", "morphology", "component_fingerprints", "freeze",
    }
    assert payload["freeze"]["forecast_frozen_before_labels"] is True
    assert set(payload["freeze"]["post_freeze_trusted_diagnostics"]) == {
        "mae", "mase", "smae", "srmse"
    }
    assert payload["morphology"]["call_status"] == "completed"
    assert all(
        set(item) == {"assumption_id", "kind", "claim", "failure_condition"}
        for item in payload["accepted_assumptions"]
    )
    assert payload["candidates"]["unavailable"]
    assert "toto_2_0" in {item["name"] for item in payload["candidates"]["unavailable"]}


def test_ambiguous_input_fails_before_result_or_model_work(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    _write_tasks(tasks, _record("one"), _record("two"))

    with pytest.raises(ValueError, match="--task-id"):
        main([
            "--task-file", str(tasks), "--results-path", str(result), "--llm-backend", "fake",
        ])
    assert not result.exists()


def test_labels_are_withheld_from_the_numerical_path_until_post_freeze_scoring(tmp_path, monkeypatch) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    _write_tasks(tasks, _record("one", future=[99.0, 98.0, 97.0]))
    original = smoke.run_numerical_loop

    def history_only(task, **kwargs):
        assert task.future == ()
        return original(task, **kwargs)

    monkeypatch.setattr(smoke, "run_numerical_loop", history_only)
    assert main([
        "--task-file", str(tasks), "--results-path", str(result), "--llm-backend", "fake",
    ]) == 0
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["freeze"]["post_freeze_trusted_diagnostics"]["mae"] > 0.0


def test_task_id_selects_exactly_one_record_and_overwrite_is_explicit(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    _write_tasks(tasks, _record("one"), _record("two"))

    assert main([
        "--task-file", str(tasks), "--task-id", "two", "--results-path", str(result),
        "--llm-backend", "fake",
    ]) == 0
    assert json.loads(result.read_text(encoding="utf-8"))["task_id"] == "two"
    with pytest.raises(FileExistsError, match="--overwrite"):
        main([
            "--task-file", str(tasks), "--task-id", "two", "--results-path", str(result),
            "--llm-backend", "fake",
        ])
    assert main([
        "--task-file", str(tasks), "--task-id", "two", "--results-path", str(result),
        "--llm-backend", "fake", "--overwrite",
    ]) == 0


@pytest.mark.parametrize("task_file, results_path", [("missing.jsonl", "out.json"), ("bad.json", "")])
def test_malformed_paths_fail_cleanly(tmp_path, task_file, results_path) -> None:
    if task_file == "bad.json":
        (tmp_path / task_file).write_text("not-json\n", encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError), match="task|result|JSON|path"):
        main([
            "--task-file", str(tmp_path / task_file),
            "--results-path", str(tmp_path / results_path),
            "--llm-backend", "fake",
        ])
