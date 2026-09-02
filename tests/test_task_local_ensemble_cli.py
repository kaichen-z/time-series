"""Executable lifecycle boundaries for task-local Numerical evolution."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess

import pytest

from common.payload import read_json_object
from numerical_agent.evaluate_frozen_task_local_ensemble import main as public_main
from numerical_agent.run_task_local_ensemble_evolution import main


def test_smoke_runs_eight_train_two_dev_and_freezes_after_acceptance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "accepted"

    assert main(["--smoke", "--output-dir", str(output)]) == 0

    complete = read_json_object(output / "evaluation_complete.json")
    assert complete["train_oof_tasks"] == 8
    assert complete["dev_tasks"] == 2
    assert complete["status"] == "accepted"
    assert (output / "task_local_release.json").is_file()
    assert (output / "oof_report.json").is_file()
    assert (output / "dev_report.json").is_file()


def test_failed_dev_publishes_no_task_local_release(tmp_path: Path) -> None:
    output = tmp_path / "rejected"

    assert main(
        ["--smoke", "--smoke-dev-regression", "--output-dir", str(output)]
    ) == 0

    complete = read_json_object(output / "evaluation_complete.json")
    assert complete["status"] == "dev_rejected"
    assert not (output / "task_local_release.json").exists()


def test_completed_smoke_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "once"
    assert main(["--smoke", "--output-dir", str(output)]) == 0

    with pytest.raises(ValueError, match="already completed"):
        main(["--smoke", "--output-dir", str(output)])


def test_shell_runner_dry_run_names_the_formal_train_dev_command() -> None:
    env = {
        **os.environ,
        "TASK_LOCAL_REPO": "runs/method_evolution/v001",
        "TASK_LOCAL_SPLIT_FILE": "splits/drcik_public_80_20_99_v1.json",
        "TASK_LOCAL_TASKS_FILE": "/tmp/tasks.jsonl",
        "TASK_LOCAL_ANCHOR_RELEASE_DIR": "runs/champion_evolution/latest",
        "TASK_LOCAL_FORECAST_STORE": "runs/champion_forecasts/latest",
        "TASK_LOCAL_OUTPUT_DIR": "runs/task_local_ensemble/v001",
    }
    result = subprocess.run(
        ["bash", "scripts/run_task_local_ensemble_evolution.sh", "--dry-run"],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "run_task_local_ensemble_evolution" in result.stdout
    assert "--anchor-release-dir runs/champion_evolution/latest" in result.stdout


def test_public99_runs_once_after_acceptance_and_compares_toto(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    output = tmp_path / "public"
    assert main(["--smoke", "--output-dir", str(release_dir)]) == 0

    assert public_main(
        ["--smoke", "--release-dir", str(release_dir), "--output-dir", str(output)]
    ) == 0

    report = read_json_object(output / "public_regression_report.json")
    comparison = report["comparison"]
    assert report["public_task_count"] == 99
    assert comparison["baseline"] == "toto_2_0"
    assert {
        "child_mean_smae",
        "child_mean_srmse",
        "child_p90_smae_raw",
        "child_p95_srmse_raw",
        "wins",
        "ties",
        "losses",
    } <= set(comparison)
    assert (output / "evaluation_complete.json").is_file()

    with pytest.raises(ValueError, match="already completed"):
        public_main(
            ["--smoke", "--release-dir", str(release_dir), "--output-dir", str(output)]
        )


def test_public99_rejects_a_dev_rejected_release_before_scoring(tmp_path: Path) -> None:
    release_dir = tmp_path / "rejected"
    assert main(
        ["--smoke", "--smoke-dev-regression", "--output-dir", str(release_dir)]
    ) == 0

    with pytest.raises(ValueError, match="accepted"):
        public_main(
            [
                "--smoke",
                "--release-dir",
                str(release_dir),
                "--output-dir",
                str(tmp_path / "public"),
            ]
        )
