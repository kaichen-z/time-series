from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from numerical_agent.run_task_conditioned_screening import build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_screening_cli_has_train_dev_but_no_public_test_option():
    parser = build_parser()
    args = parser.parse_args([
        "--repo", "repo",
        "--tasks-file", "tasks",
        "--outcome-cache-dir", "method-cache",
        "--policy-outcome-cache-dir", "policy-cache",
        "--output-dir", "output",
        "--target-batches-file", "batches.json",
        "--train-limit", "80",
        "--dev-limit", "20",
    ])
    assert args.train_limit == 80
    assert args.dev_limit == 20
    assert args.seed_policy == "all"
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--repo", "repo", "--tasks-file", "tasks",
            "--outcome-cache-dir", "cache", "--policy-outcome-cache-dir", "cache2",
            "--output-dir", "out", "--target-batches-file", "batch",
            "--public-test-limit", "99",
        ])


def test_screening_shell_forwards_formal_configuration():
    script = ROOT / "scripts" / "run_task_conditioned_screening.sh"
    completed = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    source = script.read_text(encoding="utf-8")
    for option in (
        "--repo", "--split-file", "--tasks-file", "--outcome-cache-dir",
        "--policy-outcome-cache-dir", "--train-limit", "--dev-limit",
        "--target-batches-file", "--output-dir", "--codex-model",
    ):
        assert option in source
