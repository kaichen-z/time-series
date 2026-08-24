"""Subprocess-level tests for the numerical_agent CLI: build-dataset, dictionary curation, and evaluate-frozen."""
from __future__ import annotations

import json
import os
import subprocess
import pytest
import hashlib
import sys
from pathlib import Path
from numerical_agent.main import (
    _curation_config,
    _evolution_config,
    _labels,
    _task_items,
    build_parser,
    main,
)


METHOD_COLLECTION_FIXTURES = Path(__file__).parent / "fixtures" / "method_collection"


def write_query_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "collection_cutoff": "2026-08-17",
                "query_templates": ["time series forecasting {term} original paper"],
                "source_tiers": ["paper", "textbook"],
                "taxonomy": {
                    "statistical": {
                        "baseline": ["naive forecast baseline"],
                        "seasonal": ["seasonal naive"],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_collection_journal(
    path: Path,
    counts: list[int],
    duplicate_resolutions: list[dict[str, str]] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "saturation_base_count": 2,
                "collection_batches": [
                    {
                        "batch_id": f"batch_{index + 1}",
                        "reviewed_source_count": 1,
                        "candidate_count": count,
                        "new_canonical_methods": count,
                        "duplicate_count": 0,
                        "rejected_count": 0,
                    }
                    for index, count in enumerate(counts)
                ],
                "duplicate_resolutions": duplicate_resolutions or [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dataset_subcommands_parse_required_inputs() -> None:
    parser = build_parser()

    collect = parser.parse_args(
        [
            "collect-methods",
            "--sources",
            "sources.jsonl",
            "--methods",
            "methods.jsonl",
            "--output-dir",
            "raw",
        ]
    )
    verify = parser.parse_args(
        [
            "verify-methods",
            "--sources",
            "sources.jsonl",
            "--methods",
            "methods.jsonl",
            "--queries",
            "queries.json",
            "--output",
            "audit.json",
        ]
    )
    build = parser.parse_args(
        [
            "build-dataset",
            "--sources",
            "sources.jsonl",
            "--methods",
            "methods.jsonl",
            "--queries",
            "queries.json",
            "--collection-journal",
            "journal.json",
            "--output",
            "release.json",
            "--audit-output",
            "audit.json",
            "--sha256-output",
            "release.sha256",
        ]
    )

    assert collect.command == "collect-methods"
    assert verify.command == "verify-methods"
    assert build.command == "build-dataset"


def test_collect_methods_writes_raw_registry_and_duplicate_report(
    tmp_path: Path, capsys
) -> None:
    output_dir = tmp_path / "raw"

    code = main(
        [
            "collect-methods",
            "--sources",
            str(METHOD_COLLECTION_FIXTURES / "valid_sources.jsonl"),
            "--methods",
            str(METHOD_COLLECTION_FIXTURES / "valid_methods.jsonl"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    assert (output_dir / "raw_method_registry.json").exists()
    duplicate_payload = json.loads(
        (output_dir / "duplicate_candidates.json").read_text(encoding="utf-8")
    )
    assert duplicate_payload == {"duplicate_candidates": []}
    summary = json.loads(capsys.readouterr().out)
    assert summary["method_count"] == 2


def test_verify_methods_returns_two_for_unpublishable_registry(
    tmp_path: Path, capsys
) -> None:
    queries = tmp_path / "queries.json"
    write_query_manifest(queries)
    methods = tmp_path / "methods.jsonl"
    payload = json.loads(
        (METHOD_COLLECTION_FIXTURES / "valid_methods.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    payload["verification_status"] = "unverified"
    payload["definition_source_ids"] = []
    methods.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    audit = tmp_path / "audit.json"

    code = main(
        [
            "verify-methods",
            "--sources",
            str(METHOD_COLLECTION_FIXTURES / "valid_sources.jsonl"),
            "--methods",
            str(methods),
            "--queries",
            str(queries),
            "--output",
            str(audit),
        ]
    )

    assert code == 2
    report = json.loads(audit.read_text(encoding="utf-8"))
    assert "method_not_verified" in report["verification"]["issue_codes"]
    assert json.loads(capsys.readouterr().out)["publishable"] is False


def test_build_dataset_requires_saturation_and_writes_verified_release(
    tmp_path: Path, capsys
) -> None:
    queries = tmp_path / "queries.json"
    journal = tmp_path / "journal.json"
    write_query_manifest(queries)
    write_collection_journal(journal, [0, 0, 0])
    release = tmp_path / "forecast_method_dataset_v001.json"
    audit = tmp_path / "audit.json"
    sidecar = tmp_path / "forecast_method_dataset_v001.sha256"

    code = main(
        [
            "build-dataset",
            "--sources",
            str(METHOD_COLLECTION_FIXTURES / "valid_sources.jsonl"),
            "--methods",
            str(METHOD_COLLECTION_FIXTURES / "valid_methods.jsonl"),
            "--queries",
            str(queries),
            "--collection-journal",
            str(journal),
            "--output",
            str(release),
            "--audit-output",
            str(audit),
            "--sha256-output",
            str(sidecar),
        ]
    )

    assert code == 0
    payload = json.loads(release.read_text(encoding="utf-8"))
    assert payload["dataset_id"] == "forecast_method_dataset_v001"
    assert len(payload["methods"]) == 2
    assert sidecar.exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary["saturated"] is True
    assert summary["method_count"] == 2


def test_build_dataset_rejects_unsaturated_collection(tmp_path: Path, capsys) -> None:
    queries = tmp_path / "queries.json"
    journal = tmp_path / "journal.json"
    write_query_manifest(queries)
    write_collection_journal(journal, [1, 0, 0])

    code = main(
        [
            "build-dataset",
            "--sources",
            str(METHOD_COLLECTION_FIXTURES / "valid_sources.jsonl"),
            "--methods",
            str(METHOD_COLLECTION_FIXTURES / "valid_methods.jsonl"),
            "--queries",
            str(queries),
            "--collection-journal",
            str(journal),
            "--output",
            str(tmp_path / "release.json"),
            "--audit-output",
            str(tmp_path / "audit.json"),
            "--sha256-output",
            str(tmp_path / "release.sha256"),
        ]
    )

    assert code == 2
    assert not (tmp_path / "release.json").exists()
    assert json.loads(capsys.readouterr().out)["saturated"] is False


def test_build_dataset_accepts_a_manually_resolved_distinct_wrapper_pair(
    tmp_path: Path, capsys
) -> None:
    duplicate_lines = (
        METHOD_COLLECTION_FIXTURES / "duplicate_methods.jsonl"
    ).read_text(encoding="utf-8").splitlines()[:2]
    methods = tmp_path / "methods.jsonl"
    methods.write_text("\n".join(duplicate_lines) + "\n", encoding="utf-8")
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "collection_cutoff": "2026-08-17",
                "query_templates": ["time series forecasting {term} original paper"],
                "source_tiers": ["paper", "textbook"],
                "taxonomy": {
                    "statistical": {
                        "automatic_selection": ["automatic ARIMA"],
                        "autoregressive": ["ARIMA"],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    journal = tmp_path / "journal.json"
    write_collection_journal(
        journal,
        [0, 0, 0],
        [
            {
                "left_method_uid": "method_arima",
                "right_method_uid": "method_auto_arima",
                "decision": "distinct_wrapper",
            }
        ],
    )

    code = main(
        [
            "build-dataset",
            "--sources",
            str(METHOD_COLLECTION_FIXTURES / "valid_sources.jsonl"),
            "--methods",
            str(methods),
            "--queries",
            str(queries),
            "--collection-journal",
            str(journal),
            "--output",
            str(tmp_path / "release.json"),
            "--audit-output",
            str(tmp_path / "audit.json"),
            "--sha256-output",
            str(tmp_path / "release.sha256"),
        ]
    )

    assert code == 0
    assert (tmp_path / "release.json").exists()
    assert json.loads(capsys.readouterr().out)["unresolved_duplicate_count"] == 0

ROOT = Path(__file__).resolve().parents[1]


SPLIT_FILE = ROOT / "splits/drcik_public_80_20_99_v1.json"


RUNNER = ROOT / "scripts/run_dictionary_curation.sh"


FROZEN_RUNNER = ROOT / "scripts/run_dictionary_frozen_test.sh"


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
    qwen = run_script(
        tmp_path,
        "smoke",
        NA_LLM_BACKEND="qwen",
        NA_MODEL_ID="fixture/qwen",
        NA_DEVICE="cuda:7",
    ).stdout

    assert "--codex-model" in codex
    assert "--llm-backend qwen" in qwen
    assert "--model-id fixture/qwen" in qwen
    assert "--device cuda:7" in qwen
    assert "--codex-model" not in qwen


def test_runner_rejects_an_unknown_mode(tmp_path: Path) -> None:
    completed = run_script(tmp_path, "bogus")

    assert completed.returncode != 0
    assert "mode must be smoke or full" in completed.stderr


def test_runner_fails_when_the_tasks_file_is_absent(tmp_path: Path) -> None:
    completed = run_script(tmp_path, "smoke", NA_TASKS_FILE=str(tmp_path / "missing.jsonl"))

    assert completed.returncode != 0
    assert "tasks file does not exist" in completed.stderr


def test_frozen_runner_wires_the_sealed_test_command_without_an_llm(
    tmp_path: Path,
) -> None:
    tasks_file = write_tasks(tmp_path / "tasks.jsonl", ["fixture"])
    dictionary = tmp_path / "working_dictionary.json"
    dictionary.write_text("{}", encoding="utf-8")
    experiment = tmp_path / "experiment.json"
    experiment.write_text("{}", encoding="utf-8")
    environment = {
        **os.environ,
        "NA_DRY_RUN": "1",
        "NA_TASKS_FILE": str(tasks_file),
        "NA_DICTIONARY": str(dictionary),
        "NA_EXPERIMENT_CONFIG": str(experiment),
        "NA_FROZEN_OUTPUT_DIR": str(tmp_path / "frozen"),
        "PYTHON": "python3",
    }

    completed = subprocess.run(
        ["bash", str(FROZEN_RUNNER)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "-m numerical_agent evaluate-frozen" in completed.stdout
    assert "--dictionary" in completed.stdout
    assert "--split-file" in completed.stdout
    assert "--provider" not in completed.stdout
    assert "--llm-backend" not in completed.stdout

NUMERICAL_AGENT_FIXTURES = Path(__file__).parent / "fixtures" / "numerical_agent"


def test_curate_cli_runs_with_injected_fake_provider(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "numerical_agent",
            "curate",
            "--experiment-config",
            str(NUMERICAL_AGENT_FIXTURES / "experiment.json"),
            "--base-methods",
            str(NUMERICAL_AGENT_FIXTURES / "base_methods.json"),
            "--provider",
            "fake",
            "--output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["accepted_dictionary_id"].startswith("fixture_raw_v000.g001")
    assert (tmp_path / "best_artifact.json").exists()
    assert (tmp_path / "working_dictionary.json").exists()
    assert (tmp_path / "method_evaluations.jsonl").exists()
    assert (tmp_path / "quarantine.json").exists()
    assert (tmp_path / "checkpoint.json").exists()

    working = json.loads((tmp_path / "working_dictionary.json").read_text())
    assert working["methods"][0]["status"] == "accepted"


def test_curate_cli_rejects_unregistered_provider(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "numerical_agent",
            "curate",
            "--experiment-config",
            str(NUMERICAL_AGENT_FIXTURES / "experiment.json"),
            "--base-methods",
            str(NUMERICAL_AGENT_FIXTURES / "base_methods.json"),
            "--provider",
            "arbitrary.import.path",
            "--output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "approved provider" in completed.stderr


def test_evaluate_frozen_scores_public_test_without_mutating_dictionary(
    tmp_path: Path,
) -> None:
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        json.dumps(
            {
                "benchmark_id": "test_1",
                "series": {
                    "history_values": [1.0, 2.0, 3.0],
                    "future_values": [3.0, 3.0],
                },
                "task_metadata": {
                    "frequency": "1 day",
                    "prediction_length": 2,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "fixture",
                "manifest_sha256": "fixture-manifest",
                "partitions": {
                    "train": {"task_ids": []},
                    "dev": {"task_ids": []},
                    "public_test": {"task_ids": ["test_1"]},
                },
            }
        ),
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.json"
    experiment.write_text(
        json.dumps(
            {
                "curation": {
                    "allowed_families": ["statistical"],
                    "dictionary_metric": "smae",
                    "method_metric": "smae",
                    "selection_folds": 2,
                    "selection_horizon": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    dictionary = tmp_path / "working_dictionary.json"
    dictionary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dictionary_id": "fixture.g001",
                "parent_dictionary_id": "fixture.v000",
                "generation": 1,
                "methods": [
                    {
                        "definition": {
                            "method_id": "last_value",
                            "family": "statistical",
                            "description": "Repeat the last value.",
                            "assumptions": [],
                            "failure_conditions": [],
                            "implementation_spec": {},
                            "dependencies": [],
                            "status": "unimplemented",
                        },
                        "candidate": {
                            "method_id": "last_value",
                            "provider": "sandbox",
                            "implementation_kind": "python_code",
                            "implementation": {
                                "code": (
                                    "def forecast(history, horizon, frequency):\n"
                                    "    return [history[-1]] * horizon\n"
                                )
                            },
                            "version": 1,
                            "parent_version": None,
                        },
                        "status": "accepted",
                        "revision_count": 0,
                        "implementation_attempts": 1,
                        "train_summary": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    before = hashlib.sha256(dictionary.read_bytes()).hexdigest()
    output_dir = tmp_path / "frozen_test"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "numerical_agent",
            "evaluate-frozen",
            "--tasks-file",
            str(tasks_file),
            "--split-file",
            str(split_file),
            "--experiment-config",
            str(experiment),
            "--dictionary",
            str(dictionary),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary == {
        "artifact_id": "fixture.g001",
        "metric": "smae",
        "public_test_tasks": 1,
        "score": 0.0,
    }
    report = json.loads((output_dir / "frozen_test_report.json").read_text())
    assert report["split"] == "public_test"
    assert report["item_count"] == 1
    assert report["metrics"] == {"smae": 0.0}
    assert report["manifest_sha256"] == "fixture-manifest"
    assert report["dictionary_sha256"] == before
    forecasts = [
        json.loads(line)
        for line in (output_dir / "frozen_test_forecasts.jsonl").read_text().splitlines()
    ]
    assert forecasts == [
        {
            "forecast": [3.0, 3.0],
            "item_id": "test_1",
            "method_id": "last_value",
            "selected": True,
            # Fold error of 1.0 over a mean absolute truth of 3.0.
            "selection_score": 0.3333333333333333,
            "status": "success",
        }
    ]
    assert hashlib.sha256(dictionary.read_bytes()).hexdigest() == before
    assert not (output_dir / "checkpoint.json").exists()
    assert not (output_dir / "best_artifact.json").exists()
    assert not (output_dir / "working_dictionary.json").exists()

    first_report = (output_dir / "frozen_test_report.json").read_bytes()
    repeated = subprocess.run(
        completed.args,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr
    assert (output_dir / "frozen_test_report.json").read_bytes() == first_report
