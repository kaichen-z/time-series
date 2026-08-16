from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_llm_only_script_forwards_target_and_explicit_python() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "evolving_loop/scripts/run_llm_only_evolutions.sh"
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
    script = root / "evolving_loop/scripts/run_coevolution_pilot30.sh"
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
    script = root / "evolving_loop/scripts/run_fresh30_four_method_eval.sh"
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
