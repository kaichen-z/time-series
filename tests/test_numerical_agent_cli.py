from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import common.evolution_core.contracts as evolution_contracts
import pytest
from numerical_agent.config import DictionaryCurationConfig
from numerical_agent.main import _curation_config, _evolution_config


FIXTURES = Path(__file__).parent / "fixtures" / "numerical_agent"


def test_active_curation_defaults_use_the_joint_scaled_metric_policy() -> None:
    config = DictionaryCurationConfig()

    assert config.method_metric == "smae"
    assert config.dictionary_metric == "smae"
    assert config.metric_policy == evolution_contracts.METRIC_POLICY
    assert config.metric_policy_fingerprint == evolution_contracts.METRIC_POLICY_FINGERPRINT


def test_active_curation_config_load_fails_closed_without_metric_policy() -> None:
    with pytest.raises(ValueError, match="missing metric policy"):
        _curation_config(
            {
                "curation": {
                    "method_metric": "smae",
                    "dictionary_metric": "smae",
                }
            }
        )


def test_active_evolution_config_load_fails_closed_without_metric_policy() -> None:
    curation = DictionaryCurationConfig()
    with pytest.raises(ValueError, match="missing metric policy"):
        _evolution_config({"evolution": {"generations": 1}}, curation)


def test_active_evolution_config_round_trips_exact_metric_policy() -> None:
    curation = DictionaryCurationConfig()
    config = _evolution_config(
        {"evolution": {"generations": 1, **evolution_contracts.metric_policy_metadata()}},
        curation,
    )

    assert config.metric_policy == evolution_contracts.METRIC_POLICY
    assert config.metric_policy_fingerprint == evolution_contracts.METRIC_POLICY_FINGERPRINT


@pytest.mark.parametrize("legacy_metric", ["smape", "mae"])
def test_active_curation_config_load_rejects_legacy_metric_defaults(
    legacy_metric: str,
) -> None:
    with pytest.raises(ValueError, match="legacy metric policy"):
        _curation_config(
            {
                "curation": {
                    "method_metric": legacy_metric,
                    "dictionary_metric": legacy_metric,
                    **evolution_contracts.metric_policy_metadata(),
                }
            }
        )


def test_curate_cli_runs_with_injected_fake_provider(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "numerical_agent",
            "curate",
            "--experiment-config",
            str(FIXTURES / "experiment.json"),
            "--base-methods",
            str(FIXTURES / "base_methods.json"),
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
            str(FIXTURES / "experiment.json"),
            "--base-methods",
            str(FIXTURES / "base_methods.json"),
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
                    **evolution_contracts.metric_policy_metadata(),
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
    assert summary["artifact_id"] == "fixture.g001"
    assert summary["public_test_tasks"] == 1
    assert summary["smae"] == summary["srmse"] == 0.0
    assert summary["primary_metrics"] == ["smae", "srmse"]
    report = json.loads((output_dir / "frozen_test_report.json").read_text())
    assert report["split"] == "public_test"
    assert report["item_count"] == 1
    assert report["metrics"] == {"smae": 0.0, "srmse": 0.0}
    assert report["manifest_sha256"] == "fixture-manifest"
    assert report["dictionary_sha256"] == before
    assert report["metric_policy_fingerprint"] == (
        evolution_contracts.METRIC_POLICY_FINGERPRINT
    )
    for field in (
        "mean_smae", "mean_srmse", "p95_smae_raw", "p95_srmse_raw",
        "smae_clipped_count", "srmse_clipped_count", "coverage", "paired_joint_wtl",
    ):
        assert field in report
    forecasts = [
        json.loads(line)
        for line in (output_dir / "frozen_test_forecasts.jsonl").read_text().splitlines()
    ]
    assert len(forecasts) == 1
    assert {
        key: forecasts[0][key]
        for key in (
            "forecast", "item_id", "method_id", "selected", "selection_score", "status"
        )
    } == {
        "forecast": [3.0, 3.0],
        "item_id": "test_1",
        "method_id": "last_value",
        "selected": True,
        "selection_score": 1.0 / 3.0,
        "status": "success",
    }
    assert forecasts[0]["metric_policy_fingerprint"] == (
        evolution_contracts.METRIC_POLICY_FINGERPRINT
    )
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
