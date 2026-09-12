"""Executable lifecycle boundaries for task-local Numerical evolution."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess

import pytest

from common.payload import read_json_object
from numerical_agent.evaluate_frozen_task_local_ensemble import main as public_main
from numerical_agent.evolution.task_local_ensemble import parse_task_local_release
from numerical_agent.run_task_local_ensemble_evolution import main
from numerical_agent.run_task_local_ensemble_evolution import (
    _CONFIDENCE_HINDCAST_CONFIG,
    _adaptive_hindcast_config,
    materialize_task_shortlist_rows,
    _load_candidate_priors_bundle,
)
from numerical_agent.evolution.execution import Task as RuntimeTask
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.task_shortlist import TaskCandidateShortlistV1


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
    release = parse_task_local_release(
        read_json_object(output / "task_local_release.json")
    )
    assert release.schema_version == 2
    assert release.confidence_evidence is not None


def test_confidence_hindcast_uses_three_origins_for_long_horizon_history() -> None:
    task = RuntimeTask(
        "long_horizon", tuple(float(i) for i in range(112)), 168, "D", ()
    )

    config = _adaptive_hindcast_config(task, _CONFIDENCE_HINDCAST_CONFIG)

    assert config.folds == 3
    assert config.min_successful_folds == 3


def test_shortlist_materializer_never_executes_excluded_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    class Store:
        identity_hash = "store"
        def forecast(self, name: str, history: tuple[float, ...], horizon: int, frequency: str) -> tuple[float, ...]:
            calls.append(name)
            if name == "bad":
                raise RuntimeError("boom")
            return (1.0,) * horizon
    def diagnose(task: RuntimeTask, name: str, family: str, forecast, config, **_kwargs):
        calls.append(f"diagnose:{name}")
        return CandidateDiagnostics.synthetic(name=name, family=family, median_mase=1.0,
            fold_forecasts=((1.0,),) * 3, fold_truths=((1.0,),) * 3,
            median_smae=1.0, median_srmse=1.0)
    monkeypatch.setattr("numerical_agent.run_task_local_ensemble_evolution.diagnose_candidate", diagnose)
    selected = ("toto_2_0", *(f"good{index}" for index in range(7)))
    excluded = tuple((name, "ranked_out") for name in sorted(f"method{index}" for index in range(8, 20)))
    shortlist = TaskCandidateShortlistV1(1, "a" * 64, "b" * 64, "c" * 64,
        selected, excluded, False, False)
    rows = materialize_task_shortlist_rows(Store(), RuntimeTask("task", (1.0, 2.0, 3.0, 4.0), 1, "D", (5.0,)), shortlist,
        {"toto_2_0": "tsfm", **{name: "statistical" for name in selected[1:]}}, split="dev",
        hindcast_config=_CONFIDENCE_HINDCAST_CONFIG)
    assert calls == [item for name in selected for item in (name, f"diagnose:{name}")]
    assert {row.candidate_name for row in rows} == set(selected)
    assert not {name for name in calls if name.startswith("method")}


def test_prior_bundle_requires_exact_complement_and_final_prior_authority(tmp_path: Path) -> None:
    prior = {"candidate_name": "toto_2_0", "family": "tsfm", "success_rate": 1.0,
             "mean_joint": 0.1, "p90_joint": 0.1, "morphology_scores": []}
    unsigned = {"schema_version": 1, "grouping_fingerprint": "a" * 64,
                "dictionary_sha256": "b" * 64, "fold_priors": {"0": [prior], "1": [prior]},
                "final_priors": [prior]}
    unsigned["payload_fingerprint"] = __import__("hashlib").sha256(
        __import__("common.payload", fromlist=["canonical_json_bytes"]).canonical_json_bytes(unsigned)
    ).hexdigest()
    path = tmp_path / "priors.json"
    path.write_text(__import__("json").dumps(unsigned), encoding="utf-8")
    folds, final = _load_candidate_priors_bundle(path, grouping_fingerprint="a" * 64,
        dictionary_sha256="b" * 64, families={"toto_2_0": "tsfm"}, fold_count=2)
    assert folds[0] == final


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
        "TASK_LOCAL_CANDIDATE_PRIORS_FILE": "runs/task_local_priors/v001.json",
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
    assert "--candidate-priors-file runs/task_local_priors/v001.json" in result.stdout


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
