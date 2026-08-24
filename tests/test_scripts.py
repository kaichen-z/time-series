"""Subprocess-level tests for the shell scripts under scripts/: dry runs and smoke invocations."""
from __future__ import annotations

import os
import subprocess
import pytest
from pathlib import Path


def test_llm_only_script_forwards_target_and_explicit_python() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/run_llm_only_evolutions.sh"
    environment = {
        **os.environ,
        "EA_DRY_RUN": "1",
        "EA_EVOLVE_TARGET": "coding",
        "EA_RUNS_DIR": "/tmp/llm-only-targeted-dry-run",
        "EA_SEED_POLICY_PATH": "/tmp/accepted-coding-policy.json",
        "EA_CODEX_CACHE_DIR": "/tmp/shared-codex-cache",
        "PYTHON": "python",
    }
    completed = subprocess.run(
        ["bash", str(script), "/tmp/nonexistent-drcik-tasks", "prompt"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "python -m evolving_loop" in completed.stdout
    assert "--evolve-target coding" in completed.stdout
    assert "evolution target:    coding" in completed.stdout
    assert "--seed-policy-path /tmp/accepted-coding-policy.json" in completed.stdout
    assert "--codex-cache-dir /tmp/shared-codex-cache" in completed.stdout


def test_pilot30_script_renders_auto_genome_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/run_coevolution_pilot30.sh"
    completed = subprocess.run(
        ["bash", str(script), "/tmp/nonexistent-drcik-tasks"],
        cwd=root,
        env={
            **os.environ,
            "EA_DRY_RUN": "1",
            "EA_RUNS_DIR": "/tmp/coevolution-pilot30-dry-run",
            "EA_SUCCESSIVE_HALVING": "1",
            "EA_SCREEN_TRAIN_TASKS": "6",
            "EA_SCREEN_DEV_TASKS": "2",
            "EA_SCREEN_PROMOTE": "1",
            "EA_SCREEN_TOLERANCE": "0.01",
            "PYTHON": "python",
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert "modes:              genome" in completed.stdout
    assert "--evolve-target auto" in completed.stdout
    assert "--limit 30" in completed.stdout
    assert "--generations 1" in completed.stdout
    assert "--children 4" in completed.stdout
    assert "successive halving:  1" in completed.stdout
    assert "--successive-halving" in completed.stdout
    assert "--screen-train-tasks 6" in completed.stdout
    assert "--screen-dev-tasks 2" in completed.stdout
    assert "--screen-promote 1" in completed.stdout
    assert "--screen-tolerance 0.01" in completed.stdout


def test_fresh30_launcher_dry_run_renders_four_methods_for_one_smoke_task() -> None:
    """A smoke launch must cover all methods without leaking into task two."""
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/run_fresh30_four_method_eval.sh"
    completed = subprocess.run(
        ["bash", str(script), "smoke"],
        cwd=root,
        env={
            **os.environ,
            "EA_DRY_RUN": "1",
            "EA_EVAL_ROOT": "/tmp/fresh30-four-method-dry-run",
            "PYTHON": "python",
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.count("python -m evolving_loop") == 4
    assert completed.stdout.count("--task-id task_157") == 4
    assert "--task-id task_47" not in completed.stdout
    assert "--inference genome" in completed.stdout
    assert "--baseline codex-direct" in completed.stdout
    assert "--baseline codex-contract" in completed.stdout
    expected_sample_dir = root / "external/Dr-CiK/full-download/Dr-CiK_public"
    assert completed.stdout.count(f"--sample-dir {expected_sample_dir}") == 2
    assert "--public-dev" not in completed.stdout

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

ROOT = Path(__file__).resolve().parents[1]


def test_build_method_dataset_script_exposes_reproducible_release_pipeline() -> None:
    script = ROOT / "scripts/build_method_dataset.sh"

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "write_catalog_manifests" in text
    assert "build-dataset" in text
    assert "forecast_method_dataset_v001.json" in text
    assert "pytest" in text
