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
    environment = {
        **os.environ,
        "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "TASKS_FILE": "tasks with spaces.jsonl",
        "SPLIT_FILE": "split with spaces.json",
        "MODEL": "model-test",
        "EFFORT": "medium",
        "RUN_DIR": "run with spaces",
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
