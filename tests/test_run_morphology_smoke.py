"""Executable contract for the one-task morphology smoke command."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from argparse import Namespace

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


def test_task_id_scans_jsonl_ids_without_decoding_an_unselected_task_body(tmp_path, monkeypatch) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    _write_tasks(tasks, _record("unselected"), _record("selected"))
    decode = smoke.json.loads

    def selected_only(value, *args, **kwargs):
        if '"benchmark_id": "unselected"' in value:
            raise AssertionError("unselected task body was decoded")
        return decode(value, *args, **kwargs)

    monkeypatch.setattr(smoke.json, "loads", selected_only)
    assert main([
        "--task-file", str(tasks), "--task-id", "selected", "--results-path", str(result),
        "--llm-backend", "fake",
    ]) == 0


def test_real_mode_requires_an_explicit_reviewed_artifact_bundle(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks, _record("one"))

    with pytest.raises(smoke.SmokeError, match="--methods-path"):
        main(["--task-file", str(tasks), "--results-path", str(tmp_path / "result.json")])


def test_reviewed_artifacts_are_content_hashed_and_decision_binds_screening(tmp_path) -> None:
    screening = tmp_path / "screening.py"
    decision = tmp_path / "decision.py"
    methods = tmp_path / "methods.py"
    skills = tmp_path / "skills.py"
    policies = tmp_path / "policies.py"
    for path in (methods, skills, policies):
        path.write_text("reviewed\n", encoding="utf-8")
    screening.write_text("CANDIDATES = ()\nFALLBACK_NAMES = ()\n", encoding="utf-8")
    decision.write_text(
        f"SCREENING_POLICY_HASH = {'0' * 64!r}\nDECISION_POLICY = {{}}\n", encoding="utf-8"
    )
    args = Namespace(
        llm_backend="fake", methods_path=str(methods), skills_path=str(skills),
        policies_path=str(policies), screening_path=str(screening), decision_path=str(decision),
    )

    first = smoke._validate_artifact_bundle(args)
    methods.write_text("reviewed changed\n", encoding="utf-8")
    assert smoke._validate_artifact_bundle(args)["reviewed_methods"] != first["reviewed_methods"]
    with pytest.raises(smoke.SmokeError, match="SCREENING_POLICY_HASH"):
        smoke._decision_policy(args, smoke._sha256(screening))


def test_non_overwrite_result_creation_has_one_winner_under_a_race(tmp_path) -> None:
    path = tmp_path / "nested" / "result.json"
    start = threading.Barrier(2)
    results: list[object] = []

    def write() -> None:
        start.wait()
        try:
            smoke._write_result(path, {"value": 1}, overwrite=False)
        except Exception as error:  # the losing race is the assertion target
            results.append(error)
        else:
            results.append("written")

    threads = [threading.Thread(target=write), threading.Thread(target=write)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count("written") == 1
    assert sum(isinstance(value, FileExistsError) for value in results) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_nonfinite_post_freeze_metrics_are_rejected_before_json_encoding() -> None:
    with pytest.raises(smoke.SmokeError, match="non-finite"):
        smoke._post_freeze_metrics(
            (1e308, -1e308, 1e308),
            (1e308, -1e308),
            (-1e308, 1e308),
        )


def test_labels_are_withheld_from_the_numerical_path_until_post_freeze_scoring(tmp_path, monkeypatch) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    _write_tasks(tasks, _record("one", future=[99.0, 98.0, 97.0]))
    original = smoke.run_numerical_loop
    original_future = smoke._future_values_after_freeze
    events: list[str] = []

    def history_only(task, **kwargs):
        assert task.future == ()
        package = original(task, **kwargs)
        events.append("frozen")
        return package

    def labels_after_freeze(record, horizon):
        assert events == ["frozen"]
        return original_future(record, horizon)

    monkeypatch.setattr(smoke, "run_numerical_loop", history_only)
    monkeypatch.setattr(smoke, "_future_values_after_freeze", labels_after_freeze)
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
