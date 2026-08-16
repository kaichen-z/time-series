"""Execute one accepted source patch in an isolated Git worktree."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from evolving_loop.source_evolution import SourceEvolutionEngine


def run_source_inference(
    *,
    repo_root: str | Path,
    patch_path: str | Path,
    config: dict,
    timeout_seconds: int,
) -> dict:
    root = Path(repo_root).resolve()
    patch = Path(patch_path).read_text(encoding="utf-8")
    if not patch.strip():
        # An empty accepted patch means the seed source remained best.
        patch = ""
    with tempfile.TemporaryDirectory(prefix="source-inference-") as directory:
        worktree = Path(directory) / "worktree"
        completed = subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"could not create source inference worktree: {completed.stderr[-2000:]}")
        try:
            if patch:
                SourceEvolutionEngine._apply_patch(worktree, patch)
                changed = SourceEvolutionEngine(root, lambda _: None)._changed_files(worktree)
                audit_error = SourceEvolutionEngine.audit(worktree, changed)
                if audit_error:
                    raise RuntimeError(f"source inference patch failed audit: {audit_error}")
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            environment = dict(os.environ)
            existing = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = str(worktree) + (os.pathsep + existing if existing else "")
            result = subprocess.run(
                [sys.executable, "-m", "evolving_loop.frozen_runner", "--config", str(config_path)],
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=environment,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout)[-3000:])
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError("source frozen runner produced no summary")
            return json.loads(lines[-1])
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=root,
                capture_output=True,
                check=False,
            )
