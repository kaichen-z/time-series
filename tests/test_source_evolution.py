from __future__ import annotations

import tempfile
import subprocess
from pathlib import Path

from evolving_agent.source_evolution import (
    SourceEvaluation,
    SourceEvolutionConfig,
    SourceEvolutionEngine,
)


def test_source_audit_accepts_new_safe_agent_module() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "evolving_agent/generated/candidate_auditor.py"
        path.parent.mkdir(parents=True)
        path.write_text(
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class CandidateAuditor:\n"
            "    threshold: float = 1.0\n"
        )
        result = SourceEvolutionEngine.audit(
            root, ("evolving_agent/generated/candidate_auditor.py",)
        )
    assert result is None


def test_source_audit_rejects_protected_file_changes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assert SourceEvolutionEngine.audit(root, ("evolving_agent/metrics.py",)) == (
            "protected_paths_changed:evolving_agent/metrics.py"
        )


def test_source_audit_rejects_label_access_even_in_mutable_file() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=root, check=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
        path = root / "evolving_agent/harness.py"
        path.parent.mkdir(parents=True)
        path.write_text("def safe(task):\n    return task.numeric.history_values\n")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "seed"], cwd=root, capture_output=True, check=True)
        path.write_text("def cheat(task):\n    return task.numeric.future_values\n")
        result = SourceEvolutionEngine.audit(root, ("evolving_agent/harness.py",))
    assert result == "forbidden_added_code:future_values"


def test_source_audit_rejects_filesystem_import() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "evolving_agent/generated/unsafe.py"
        path.parent.mkdir(parents=True)
        path.write_text("from pathlib import Path\n")
        result = SourceEvolutionEngine.audit(root, ("evolving_agent/generated/unsafe.py",))
    assert result == "forbidden_import:evolving_agent/generated/unsafe.py:pathlib"


def test_source_engine_accepts_better_child_and_returns_inheritable_patch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=root, check=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
        path = root / "evolving_agent/harness.py"
        path.parent.mkdir(parents=True)
        path.write_text("SEED = True\n")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "seed"], cwd=root, capture_output=True, check=True)

        def evaluator(worktree: Path) -> SourceEvaluation:
            improved = "SOURCE_CHILD = True" in (worktree / "evolving_agent/harness.py").read_text()
            reward = 0.8 if improved else 0.5
            return SourceEvaluation(reward, reward, {"coding": reward}, {"coding": reward}, ())

        class StubSourceEngine(SourceEvolutionEngine):
            def _run_engineer(self, worktree, candidate_id, parent):
                candidate = worktree / "evolving_agent/harness.py"
                candidate.write_text(candidate.read_text() + "SOURCE_CHILD = True\n")
                return '{"summary":"added child","hypothesis":"improves reward"}'

            def _tests(self, worktree):
                return True, None

        engine = StubSourceEngine(
            root,
            evaluator,
            SourceEvolutionConfig(generations=1, children_per_generation=1),
        )
        patch, trace = engine.evolve()

    assert "SOURCE_CHILD = True" in patch
    assert trace[0].accepted_candidate == "source_g000_c000"
    assert trace[0].parent_dev_reward == 0.5
    assert trace[0].accepted_dev_reward == 0.8


def test_source_engine_audits_an_external_seed_patch_before_evaluation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=root, check=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
        protected = root / "evolving_agent/metrics.py"
        protected.parent.mkdir(parents=True)
        protected.write_text("SCORE = 1\n")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "seed"], cwd=root, capture_output=True, check=True)
        protected.write_text("SCORE = 999\n")
        patch = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        subprocess.run(["git", "checkout", "--", "evolving_agent/metrics.py"], cwd=root, check=True)

        engine = SourceEvolutionEngine(root, lambda _worktree: None)
        try:
            engine.evolve(seed_patch=patch)
        except RuntimeError as exc:
            assert "seed patch failed audit: protected_paths_changed" in str(exc)
        else:
            raise AssertionError("protected seed patch should have been rejected")


def test_source_checkpoint_round_trip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.json"
        progress = Path(directory) / "progress.jsonl"
        engine = SourceEvolutionEngine(
            directory,
            lambda _worktree: None,
            SourceEvolutionConfig(checkpoint_path=checkpoint, progress_path=progress),
        )
        evaluation = SourceEvaluation(0.4, 0.5, {"coding": 0.4}, {"coding": 0.5}, ())
        engine._save_checkpoint("patch-data", evaluation, [], 2)
        restored = engine._load_checkpoint()

    assert restored is not None
    patch, restored_evaluation, trace, start = restored
    assert patch == "patch-data"
    assert restored_evaluation == evaluation
    assert trace == []
    assert start == 2
