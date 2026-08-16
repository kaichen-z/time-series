from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


FIXTURES = Path(__file__).parent / "fixtures" / "numerical_agent"


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
