from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "run_retrieval_evolution.sh"


def test_retrieval_runner_is_executable_bash_and_dispatches_exact_command(tmp_path) -> None:
    assert RUNNER.exists()
    assert RUNNER.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    binary = tmp_path / "bin"
    binary.mkdir()
    capture = tmp_path / "argv"
    fake_python = binary / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    authority_path = tmp_path / "authority with spaces" / "checkpoint.json"
    authority_head_path = (
        tmp_path / "authority with spaces" / "checkpoint.head.json"
    )
    authority_key = "task-8-runner-operator-authority-key-32-bytes"
    environment = {
        **os.environ,
        "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "TASKS_FILE": "tasks with spaces.jsonl",
        "SPLIT_FILE": "split with spaces.json",
        "MODEL": "model-test",
        "EFFORT": "medium",
        "RUN_DIR": "run with spaces",
        "AUTHORITY_PATH": str(authority_path),
        "AUTHORITY_HEAD_PATH": str(authority_head_path),
        "RETRIEVAL_CHECKPOINT_AUTHORITY_KEY": authority_key,
    }
    completed = subprocess.run(
        [str(RUNNER)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "evolving_loop.cli",
        "--evolution",
        "retrieval",
        "--tasks-file",
        "tasks with spaces.jsonl",
        "--split-manifest",
        "split with spaces.json",
        "--retrieval-mode",
        "two-stage",
        "--llm-backend",
        "codex",
        "--codex-model",
        "model-test",
        "--codex-reasoning-effort",
        "medium",
        "--generations",
        "3",
        "--screen-train-tasks",
        "8",
        "--screen-promote",
        "2",
        "--run-root",
        "run with spaces",
        "--checkpoint-authority-path",
        str(authority_path),
        "--checkpoint-authority-head-path",
        str(authority_head_path),
        "--checkpoint-path",
        "run with spaces/checkpoint.json",
        "--progress-path",
        "run with spaces/progress.jsonl",
        "--policy-path",
        "run with spaces/best_policy.json",
        "--trace-path",
        "run with spaces/evolution_trace.json",
    ]
    assert "python -m evolving_loop.cli" in completed.stdout
    assert "--evolution retrieval" in completed.stdout
    assert authority_key not in completed.stdout
    assert authority_key not in capture.read_text(encoding="utf-8")
    assert authority_path.parent.is_dir()
    assert stat.S_IMODE(authority_path.parent.stat().st_mode) == 0o700


def test_retrieval_runner_fails_closed_without_operator_authority_key(
    tmp_path: Path,
) -> None:
    environment = {
        **os.environ,
        "AUTHORITY_PATH": str(tmp_path / "authority" / "checkpoint.json"),
    }
    environment.pop("RETRIEVAL_CHECKPOINT_AUTHORITY_KEY", None)

    completed = subprocess.run(
        [str(RUNNER)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "authority key" in completed.stderr.lower()
