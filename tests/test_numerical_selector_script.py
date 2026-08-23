from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.selector_evolution import DecisionCase
from numerical_agent.run_selector_evolution import _write_cases, build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_selector_cli_requires_frozen_screen_and_has_no_test_option():
    parser = build_parser()
    args = parser.parse_args([
        "--repo", "repo", "--screening-dir", "screen", "--tasks-file", "tasks",
        "--outcome-cache-dir", "cache", "--policy-outcome-cache-dir", "pcache",
        "--hindcast-cache-dir", "hcache", "--output-dir", "out",
        "--train-limit", "80", "--dev-limit", "20",
    ])
    assert args.train_limit == 80 and args.dev_limit == 20
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--repo", "repo", "--screening-dir", "screen", "--tasks-file", "tasks",
            "--outcome-cache-dir", "cache", "--policy-outcome-cache-dir", "pcache",
            "--hindcast-cache-dir", "hcache", "--output-dir", "out",
            "--public-test-limit", "99",
        ])


def test_selector_shell_forwards_runtime_and_freeze_inputs():
    script = ROOT / "scripts" / "run_numerical_selector_evolution.sh"
    completed = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    source = script.read_text(encoding="utf-8")
    for option in (
        "--screening-dir", "--hindcast-cache-dir", "--train-limit", "--dev-limit",
        "--generations", "--codex-model", "--tsfm-workers-config",
    ):
        assert option in source


def test_case_artifact_serializes_ineligible_infinite_diagnostics_as_null(tmp_path):
    diagnostic = CandidateDiagnostics.synthetic(
        name="failed", family="statistical", median_mase=float("inf"), eligible=False
    )
    case = DecisionCase(
        Task("t", (1.0, 2.0, 3.0), 1, "D", (4.0,)),
        ("failed",),
        {"failed": diagnostic},
        {},
        {"failed": "statistical"},
    )
    target = tmp_path / "cases.jsonl"
    _write_cases(target, (case,))
    assert '"median_mase": null' in target.read_text(encoding="utf-8")
