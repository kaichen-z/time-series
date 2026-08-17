from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from numerical_agent.main import _curation_config, _evolution_config, _labels, _task_items


ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "splits/drcik_public_80_20_99_v1.json"
RUNNER = ROOT / "scripts/run_dictionary_curation.sh"


def write_tasks(path: Path, task_ids, *, history=8, horizon=2) -> Path:
    """Write a minimal Dr-CiK style tasks file covering the given ids."""
    with path.open("w", encoding="utf-8") as handle:
        for index, task_id in enumerate(task_ids):
            record = {
                "benchmark_id": task_id,
                "entity_name": f"entity_{index}",
                "series": {
                    "history_values": [float(step) for step in range(history)],
                    "future_values": [float(step) for step in range(horizon)],
                },
                "task_metadata": {"frequency": "1 day", "prediction_length": horizon},
            }
            handle.write(json.dumps(record) + "\n")
    return path


def split_ids(partition: str) -> list[str]:
    payload = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    return list(payload["partitions"][partition]["task_ids"])


def build(tmp_path: Path, *extra: str) -> tuple[subprocess.CompletedProcess, Path]:
    tasks_file = write_tasks(
        tmp_path / "tasks.jsonl", split_ids("train") + split_ids("dev")
    )
    output = tmp_path / "experiment.json"
    completed = subprocess.run(
        [
            "python3",
            "-m",
            "numerical_agent",
            "build-experiment",
            "--tasks-file",
            str(tasks_file),
            "--split-file",
            str(SPLIT_FILE),
            "--output",
            str(output),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, output


def test_builder_emits_the_frozen_split_sizes(tmp_path: Path) -> None:
    completed, output = build(tmp_path)

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["train_tasks"] == 80
    assert summary["dev_tasks"] == 20
    experiment = json.loads(output.read_text(encoding="utf-8"))
    assert set(experiment) == {"evolution", "curation", "tasks", "labels"}


def test_builder_output_loads_through_the_curate_config_readers(tmp_path: Path) -> None:
    _, output = build(tmp_path)
    experiment = json.loads(output.read_text(encoding="utf-8"))

    curation = _curation_config(experiment)
    _evolution_config(experiment, curation)
    train_items, dev_items = _task_items(experiment)
    labels = _labels(experiment)

    assert len(train_items) == 80
    assert len(dev_items) == 20
    # Every scored item needs a label trajectory exactly as long as its horizon.
    assert all(len(labels["train"][item.item_id]) == item.horizon for item in train_items)
    assert all(len(labels["dev"][item.item_id]) == item.horizon for item in dev_items)


def test_builder_honours_task_limits(tmp_path: Path) -> None:
    completed, _ = build(tmp_path, "--train-limit", "4", "--dev-limit", "2")

    summary = json.loads(completed.stdout)
    assert summary["train_tasks"] == 4
    assert summary["dev_tasks"] == 2


def test_builder_persists_selector_coverage_and_retry_parameters(tmp_path: Path) -> None:
    completed, output = build(
        tmp_path,
        "--max-implementation-attempts",
        "4",
        "--min-success-rate",
        "0.75",
        "--selection-folds",
        "4",
        "--selection-horizon",
        "6",
    )

    assert completed.returncode == 0, completed.stderr
    curation = json.loads(output.read_text(encoding="utf-8"))["curation"]
    assert curation["max_implementation_attempts"] == 4
    assert curation["min_success_rate"] == 0.75
    assert curation["selection_folds"] == 4
    assert curation["selection_horizon"] == 6
    assert curation["allowed_families"] == ["statistical"]


def test_builder_keeps_train_and_dev_disjoint(tmp_path: Path) -> None:
    _, output = build(tmp_path)
    experiment = json.loads(output.read_text(encoding="utf-8"))

    train = {item["item_id"] for item in experiment["tasks"]["train"]}
    dev = {item["item_id"] for item in experiment["tasks"]["dev"]}
    assert not train & dev


def test_builder_rejects_a_task_missing_from_the_tasks_file(tmp_path: Path) -> None:
    tasks_file = write_tasks(tmp_path / "tasks.jsonl", split_ids("train")[:5])
    completed = subprocess.run(
        [
            "python3",
            "-m",
            "numerical_agent",
            "build-experiment",
            "--tasks-file",
            str(tasks_file),
            "--split-file",
            str(SPLIT_FILE),
            "--output",
            str(tmp_path / "experiment.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "absent from the tasks file" in completed.stderr


def run_script(tmp_path: Path, mode: str, **overrides: str) -> subprocess.CompletedProcess:
    tasks_file = write_tasks(tmp_path / "tasks.jsonl", split_ids("train")[:1])
    environment = {
        **os.environ,
        "NA_DRY_RUN": "1",
        "NA_TASKS_FILE": str(tasks_file),
        "NA_RUNS_DIR": str(tmp_path / "runs"),
        "PYTHON": "python3",
        **overrides,
    }
    return subprocess.run(
        ["bash", str(RUNNER), mode],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_runner_renders_both_stages_for_each_mode(tmp_path: Path, mode: str) -> None:
    completed = run_script(tmp_path, mode)

    assert completed.returncode == 0, completed.stderr
    assert "-m numerical_agent build-experiment" in completed.stdout
    assert "-m numerical_agent curate" in completed.stdout
    assert "--provider llm" in completed.stdout
    assert "forecast_method_dataset_v001.json" in completed.stdout


def test_runner_limits_tasks_only_in_smoke_mode(tmp_path: Path) -> None:
    smoke = run_script(tmp_path, "smoke").stdout
    full = run_script(tmp_path, "full").stdout

    assert "--train-limit" in smoke
    assert "--train-limit" not in full


def test_runner_passes_codex_flags_only_for_the_codex_backend(tmp_path: Path) -> None:
    codex = run_script(tmp_path, "smoke", NA_LLM_BACKEND="codex").stdout
    qwen = run_script(tmp_path, "smoke", NA_LLM_BACKEND="qwen").stdout

    assert "--codex-model" in codex
    assert "--llm-backend qwen" in qwen
    assert "--codex-model" not in qwen


def test_runner_rejects_an_unknown_mode(tmp_path: Path) -> None:
    completed = run_script(tmp_path, "bogus")

    assert completed.returncode != 0
    assert "mode must be smoke or full" in completed.stderr


def test_runner_fails_when_the_tasks_file_is_absent(tmp_path: Path) -> None:
    completed = run_script(tmp_path, "smoke", NA_TASKS_FILE=str(tmp_path / "missing.jsonl"))

    assert completed.returncode != 0
    assert "tasks file does not exist" in completed.stderr
