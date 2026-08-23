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


def add_policies(repo: Path) -> None:
    from numerical_agent.evolution.portfolio import PolicyPortfolio, write_policy_file

    write_policy_file(repo / "policies.py", PolicyPortfolio.flagship5())
    subprocess.run(["git", "add", "policies.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "add policies"], cwd=repo, check=True
    )


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


def test_runner_forwards_two_stage_models_and_train_limit(tmp_path: Path) -> None:
    completed = run(
        seeded_repo(tmp_path),
        ME_LLM_BACKEND="codex",
        ME_CODEX_MODEL="gpt-5.6-terra",
        ME_REASONING_EFFORT="medium",
        ME_SELECTOR_CODEX_MODEL="gpt-5.6-luna",
        ME_SELECTOR_REASONING_EFFORT="medium",
        ME_TRAIN_LIMIT="16",
        ME_VALIDATION_TAIL="4",
    )

    assert completed.returncode == 0, completed.stderr
    assert "--codex-model gpt-5.6-terra" in completed.stdout
    assert "--codex-reasoning-effort medium" in completed.stdout
    assert "--selector-codex-model gpt-5.6-luna" in completed.stdout
    assert "--selector-codex-reasoning-effort medium" in completed.stdout
    assert "--train-limit 16" in completed.stdout
    assert "--validation-tail 4" in completed.stdout


def test_runner_forwards_the_generation_count(tmp_path: Path) -> None:
    completed = run(seeded_repo(tmp_path), ME_GENERATIONS="4")

    assert "--generations 4" in completed.stdout


def test_runner_forwards_targetwise_evolution_options(tmp_path: Path) -> None:
    completed = run(
        seeded_repo(tmp_path),
        ME_EVOLUTION_STRATEGY="targetwise",
        ME_OUTCOME_CACHE_DIR="/tmp/method-outcomes",
        ME_MAX_TARGETS="8",
        ME_SCREEN_TASKS="3",
        ME_FULL_EVALUATION_CANDIDATES="2",
        ME_FAILURE_JUDGE="1",
    )

    assert completed.returncode == 0, completed.stderr
    assert "--evolution-strategy targetwise" in completed.stdout
    assert "--outcome-cache-dir /tmp/method-outcomes" in completed.stdout
    assert "--max-targets 8" in completed.stdout
    assert "--screen-tasks 3" in completed.stdout
    assert "--full-evaluation-candidates 2" in completed.stdout
    assert "--failure-judge" in completed.stdout


def test_runner_forwards_flagship_tsfm_and_combined_portfolio(tmp_path: Path) -> None:
    repo = seeded_repo(tmp_path)
    add_policies(repo)
    deployment = tmp_path / "workers.json"
    deployment.write_text("{}", encoding="utf-8")

    completed = run(
        repo,
        ME_EVOLUTION_STRATEGY="targetwise",
        ME_FOUNDATION_PORTFOLIO="flagship5",
        ME_TSFM_RUNTIMES="chronos,timesfm",
        ME_TSFM_WORKERS_CONFIG=str(deployment),
        ME_ACKNOWLEDGED_MODEL_LICENSES="CC-BY-NC-4.0",
        ME_POLICY_MAX_TARGETS="2",
    )

    assert completed.returncode == 0, completed.stderr
    assert "--foundation-portfolio flagship5" in completed.stdout
    assert "--tsfm-runtimes chronos\\,timesfm" in completed.stdout
    assert f"--tsfm-workers-config {deployment}" in completed.stdout
    assert "--acknowledged-model-licenses CC-BY-NC-4.0" in completed.stdout
    assert "--policy-max-targets 2" in completed.stdout


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
