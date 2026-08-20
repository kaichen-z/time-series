from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_method_evolution.sh"


def seeded_repo(tmp_path: Path) -> Path:
    """A minimal repo shaped like a real evolution run, without invoking any model."""
    repo = tmp_path / "v001"
    repo.mkdir()
    (repo / "methods.py").write_text(
        'def naive_last(history, horizon, frequency):\n'
        '    """Use when nothing better applies."""\n'
        "    return [float(history[-1])] * horizon\n",
        encoding="utf-8",
    )
    for args in (["init", "--quiet"], ["config", "user.email", "t@localhost"],
                 ["config", "user.name", "t"], ["add", "methods.py"],
                 ["commit", "--quiet", "-m", "seed"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return repo


def run(repo: Path, **overrides: str) -> subprocess.CompletedProcess:
    environment = {**os.environ, "ME_DRY_RUN": "1", "ME_REPO": str(repo), "PYTHON": "python3", **overrides}
    return subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, env=environment, capture_output=True, text=True, check=False
    )


def test_runner_renders_the_evolution_command(tmp_path: Path) -> None:
    completed = run(seeded_repo(tmp_path))

    assert completed.returncode == 0, completed.stderr
    assert "-m numerical_agent.run_evolution" in completed.stdout
    assert "--llm-backend qwen" in completed.stdout
    assert "--generations " in completed.stdout


def test_runner_passes_codex_flags_only_for_codex(tmp_path: Path) -> None:
    repo = seeded_repo(tmp_path)

    codex = run(repo, ME_LLM_BACKEND="codex").stdout
    qwen = run(repo, ME_LLM_BACKEND="qwen").stdout

    assert "--codex-model" in codex and "--codex-cache-dir" in codex
    assert "--codex-model" not in qwen


def test_runner_forwards_the_generation_count(tmp_path: Path) -> None:
    completed = run(seeded_repo(tmp_path), ME_GENERATIONS="4")

    assert "--generations 4" in completed.stdout


def test_runner_reports_the_current_commit_and_method_count(tmp_path: Path) -> None:
    completed = run(seeded_repo(tmp_path))

    assert "methods:     1" in completed.stdout
    assert "at commit:   " in completed.stdout


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"ME_TASKS_FILE": "/tmp/definitely-missing.jsonl"}, "tasks file does not exist"),
        ({"ME_SPLIT_FILE": "/tmp/definitely-missing.json"}, "split file does not exist"),
    ],
)
def test_runner_refuses_missing_inputs(tmp_path: Path, override: dict, expected: str) -> None:
    completed = run(seeded_repo(tmp_path), **override)

    assert completed.returncode != 0
    assert expected in completed.stderr


def test_runner_refuses_an_unseeded_repo(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    completed = run(empty)

    # Re-running must never re-seed; an absent module is an error, not a fresh start.
    assert completed.returncode != 0
    assert "no seeded module" in completed.stderr


def test_runner_refuses_a_directory_that_is_not_a_git_repo(tmp_path: Path) -> None:
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "methods.py").write_text("x = 1\n", encoding="utf-8")

    completed = run(loose)

    assert completed.returncode != 0
    assert "not a git repository" in completed.stderr
