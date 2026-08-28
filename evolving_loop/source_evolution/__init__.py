"""Source-level harness evolution in isolated Git worktrees."""
from __future__ import annotations

import json
import ast
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


SOURCE_ENGINEER_PROMPT = """You are the Source Harness Engineer for a contextual time-series
forecasting system. Work directly in the provided isolated Git worktree. Inspect the current
implementation and the supplied resolved training failure summary, then implement one coherent,
testable child architecture. You may rewrite the mutable Coding, Retrieval, Decision, and Harness
implementation, create new Python agent modules under evolving_loop/generated/, and wire them into
the runtime. You may invent new agent roles, validation/ranking algorithms, communication patterns,
memory use, stopping rules, or orchestration rather than merely tuning constants.

Act autonomously: do not stop after proposing a design. Do not ask for confirmation or permission.
You must make at least one concrete source change in the isolated worktree and validate it. If no
safe, testable change can be implemented, return an explicit failure instead of a proposal.

Do not edit tests, the CLI, data loading, metrics/scorer, LLM transport, code sandbox, outcome skill
learner, co-evolution/source-evolution host, or any file outside the explicit mutable paths. Never
read or introduce future_values, gt_evidence, role/subtype labels, or resolved outcomes into
inference. Do not weaken citation verification or execute shell/network/file operations from
generated forecasting code. Do not commit. Keep public interfaces compatible and finish with a
brief JSON final message:
{"summary": "...", "hypothesis": "...", "changed_files": ["evolving_loop/..."]}.
"""

_EXACT_MUTABLE = frozenset(
    {
        "evolving_loop/harness.py",
        "evolving_loop/coding_agent/evolution.py",
        "evolving_loop/retrieval_agent/agent.py",
        "evolving_loop/decision_agent/agent.py",
    }
)
_GENERATED_PREFIX = "evolving_loop/generated/"
_FORBIDDEN_ADDED_TEXT = (
    "future_values",
    "gt_evidence",
    "document.role",
    "document.subtype",
    "import subprocess",
    "from subprocess",
    "import os",
    "from os",
    "import pathlib",
    "from pathlib",
    "import sys",
    "from sys",
    "import shutil",
    "from shutil",
    "os.system",
    "import socket",
    "import requests",
    "import urllib",
    "eval(",
    "exec(",
    "__import__(",
    "open(",
    "getattr(",
    "setattr(",
    "evolving_loop.data",
    "evolving_loop.evaluation",
    "common.metrics",
    "evolving_loop.co_evolution",
    "evolving_loop.source_evolution",
    "evolving_loop.skill_learning",
    "evolving_loop.cli",
)
_FORBIDDEN_IMPORTS = frozenset(
    {"os", "pathlib", "subprocess", "socket", "requests", "urllib", "sys", "shutil"}
)
_FORBIDDEN_NAMES = frozenset(
    {"open", "eval", "exec", "compile", "__import__", "getattr", "setattr", "vars", "globals", "locals"}
)
_FORBIDDEN_ATTRIBUTES = frozenset(
    {"future_values", "gt_evidence", "annotations", "role", "subtype", "__dict__"}
)


@dataclass(frozen=True)
class SourceEvaluation:
    train_reward: float
    val_reward: float
    train_module_rewards: dict[str, float]
    val_module_rewards: dict[str, float]
    failure_traces: tuple[dict, ...]

    @classmethod
    def from_dict(cls, payload: dict) -> "SourceEvaluation":
        return cls(
            train_reward=float(payload["train"]["system_reward"]),
            val_reward=float(payload["val"]["system_reward"]),
            train_module_rewards=dict(payload["train"]["module_rewards"]),
            val_module_rewards=dict(payload["val"]["module_rewards"]),
            failure_traces=tuple(payload["train"].get("failure_traces", ())),
        )


@dataclass(frozen=True)
class SourceCandidate:
    candidate_id: str
    patch: str
    evaluation: SourceEvaluation | None
    audit_ok: bool
    tests_ok: bool
    rejection_reason: str | None
    changed_files: tuple[str, ...] = ()
    engineer_summary: str = ""


@dataclass(frozen=True)
class SourceEvolutionStep:
    generation: int
    parent_train_reward: float
    parent_dev_reward: float
    candidates: tuple[dict, ...]
    accepted_candidate: str | None
    accepted_dev_reward: float


@dataclass(frozen=True)
class SourceEvolutionConfig:
    generations: int = 1
    children_per_generation: int = 1
    model: str | None = None
    reasoning_effort: str = "high"
    codex_timeout_seconds: int = 1800
    test_timeout_seconds: int = 300
    checkpoint_path: str | Path | None = None
    progress_path: str | Path | None = None
    resume: bool = True


EvaluationCallback = Callable[[Path], SourceEvaluation]


class SourceEvolutionEngine:
    """Generate, audit, test, and evaluate source patches without touching the parent checkout."""

    def __init__(
        self,
        repo_root: str | Path,
        evaluator: EvaluationCallback,
        config: SourceEvolutionConfig | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.evaluator = evaluator
        self.config = config or SourceEvolutionConfig()

    def evolve(self, seed_patch: str = "") -> tuple[str, tuple[SourceEvolutionStep, ...]]:
        restored = self._load_checkpoint()
        if restored is None:
            incumbent_patch = seed_patch
            self._progress("seed_evaluation_started")
            incumbent_evaluation = self._evaluate_patch(incumbent_patch)
            self._progress(
                "seed_evaluation_completed",
                train_reward=incumbent_evaluation.train_reward,
                val_reward=incumbent_evaluation.val_reward,
            )
            trace: list[SourceEvolutionStep] = []
            start_generation = 0
        else:
            incumbent_patch, incumbent_evaluation, trace, start_generation = restored
        for generation in range(start_generation, self.config.generations):
            self._progress("generation_started", generation=generation)
            parent_train_reward = incumbent_evaluation.train_reward
            parent_dev_reward = incumbent_evaluation.val_reward
            candidates = [
                self._child(generation, index, incumbent_patch, incumbent_evaluation)
                for index in range(self.config.children_per_generation)
            ]
            valid = [item for item in candidates if item.evaluation is not None]
            train_best = (
                max(valid, key=lambda item: item.evaluation.train_reward)
                if valid
                else None
            )
            accepted = (
                train_best
                if train_best is not None
                and train_best.evaluation.val_reward > incumbent_evaluation.val_reward
                else None
            )
            if accepted is not None:
                incumbent_patch = accepted.patch
                incumbent_evaluation = accepted.evaluation
            trace.append(
                SourceEvolutionStep(
                    generation=generation,
                    parent_train_reward=parent_train_reward,
                    parent_dev_reward=parent_dev_reward,
                    candidates=tuple(self._candidate_summary(item) for item in candidates),
                    accepted_candidate=accepted.candidate_id if accepted else None,
                    accepted_dev_reward=incumbent_evaluation.val_reward,
                )
            )
            self._save_checkpoint(
                incumbent_patch,
                incumbent_evaluation,
                trace,
                generation + 1,
            )
            self._progress(
                "generation_completed",
                generation=generation,
                accepted=(accepted.candidate_id if accepted else None),
                val_reward=incumbent_evaluation.val_reward,
            )
        return incumbent_patch, tuple(trace)

    def _evaluate_patch(self, patch: str) -> SourceEvaluation:
        with self._worktree("source-parent-") as worktree:
            if patch:
                self._apply_patch(worktree, patch)
                changed = self._changed_files(worktree)
                audit_error = self.audit(worktree, changed)
                if audit_error:
                    raise RuntimeError(f"seed patch failed audit: {audit_error}")
                tests_ok, test_error = self._tests(worktree)
                if not tests_ok:
                    raise RuntimeError(f"seed patch failed tests: {test_error}")
            return self.evaluator(worktree)

    def _child(
        self,
        generation: int,
        index: int,
        incumbent_patch: str,
        parent: SourceEvaluation,
    ) -> SourceCandidate:
        candidate_id = f"source_g{generation:03d}_c{index:03d}"
        self._progress("candidate_started", generation=generation, candidate=candidate_id)
        with self._worktree(f"{candidate_id}-") as worktree:
            if incumbent_patch:
                self._apply_patch(worktree, incumbent_patch)
            baseline_patch = self._diff(worktree)
            try:
                self._progress("engineer_started", generation=generation, candidate=candidate_id)
                summary = self._run_engineer(worktree, candidate_id, parent)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                self._progress(
                    "engineer_failed",
                    generation=generation,
                    candidate=candidate_id,
                    error=str(exc),
                )
                return SourceCandidate(
                    candidate_id,
                    baseline_patch,
                    None,
                    False,
                    False,
                    f"source_engineer_failed:{exc}",
                )
            self._progress("engineer_completed", generation=generation, candidate=candidate_id)
            changed = self._changed_files(worktree)
            self._progress("audit_started", generation=generation, candidate=candidate_id)
            audit_error = self.audit(worktree, changed)
            if audit_error:
                self._progress(
                    "audit_failed",
                    generation=generation,
                    candidate=candidate_id,
                    error=audit_error,
                )
                return SourceCandidate(
                    candidate_id, baseline_patch, None, False, False, audit_error, changed, summary
                )
            self._progress("audit_completed", generation=generation, candidate=candidate_id)
            self._progress("tests_started", generation=generation, candidate=candidate_id)
            tests_ok, test_error = self._tests(worktree)
            if not tests_ok:
                self._progress(
                    "tests_failed",
                    generation=generation,
                    candidate=candidate_id,
                    error=test_error,
                )
                return SourceCandidate(
                    candidate_id, baseline_patch, None, True, False, test_error, changed, summary
                )
            self._progress("tests_completed", generation=generation, candidate=candidate_id)
            patch = self._diff(worktree)
            if not patch.strip() or patch == incumbent_patch:
                return SourceCandidate(
                    candidate_id,
                    incumbent_patch,
                    None,
                    True,
                    True,
                    "no_source_change",
                    changed,
                    summary,
                )
            try:
                self._progress("evaluation_started", generation=generation, candidate=candidate_id)
                evaluation = self.evaluator(worktree)
            except Exception as exc:
                self._progress(
                    "evaluation_failed",
                    generation=generation,
                    candidate=candidate_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return SourceCandidate(
                    candidate_id,
                    patch,
                    None,
                    True,
                    True,
                    f"evaluation_failed:{exc}",
                    changed,
                    summary,
                )
            self._progress(
                "candidate_completed",
                generation=generation,
                candidate=candidate_id,
                train_reward=evaluation.train_reward,
                val_reward=evaluation.val_reward,
            )
            return SourceCandidate(
                candidate_id, patch, evaluation, True, True, None, changed, summary
            )

    def _run_engineer(
        self,
        worktree: Path,
        candidate_id: str,
        parent: SourceEvaluation,
    ) -> str:
        payload = {
            "candidate_id": candidate_id,
            "parent_train_reward": parent.train_reward,
            "parent_dev_reward": parent.val_reward,
            "module_rewards": parent.train_module_rewards,
            "worst_training_failures": [
                self._sanitized_failure(item)
                for item in sorted(
                    parent.failure_traces,
                    key=lambda item: item.get("final_smae", 0.0),
                    reverse=True,
                )[:5]
            ],
            "mutable_files": sorted(_EXACT_MUTABLE),
            "new_module_directory": _GENERATED_PREFIX,
        }
        prompt = SOURCE_ENGINEER_PROMPT + "\n\nFailure payload:\n" + json.dumps(
            payload, ensure_ascii=False
        )
        with tempfile.TemporaryDirectory(prefix="source-engineer-output-") as directory:
            output = Path(directory) / "final.json"
            command = [
                "codex",
                "exec",
                "--ephemeral",
                "--sandbox",
                "workspace-write",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--ignore-rules",
                "--color",
                "never",
                "--output-last-message",
                str(output),
                "--cd",
                str(worktree),
                "-c",
                f'model_reasoning_effort="{self.config.reasoning_effort}"',
            ]
            if self.config.model:
                command.extend(["--model", self.config.model])
            command.append("-")
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.config.codex_timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout)[-2000:]
                raise RuntimeError(f"source engineer failed: {details}")
            return output.read_text(encoding="utf-8") if output.exists() else ""

    @staticmethod
    def audit(worktree: Path, changed_files: tuple[str, ...]) -> str | None:
        if not changed_files:
            return "no_changed_files"
        illegal = [
            path
            for path in changed_files
            if path not in _EXACT_MUTABLE
            and not (path.startswith(_GENERATED_PREFIX) and path.endswith(".py"))
        ]
        if illegal:
            return "protected_paths_changed:" + ",".join(illegal)
        git_diff = subprocess.run(
            ["git", "diff", "--unified=0", "HEAD", "--", *changed_files],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=False,
        )
        added_lines = "\n".join(
            line[1:]
            for line in git_diff.stdout.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ).lower()
        forbidden_text = [
            value for value in _FORBIDDEN_ADDED_TEXT if value.lower() in added_lines
        ]
        if forbidden_text:
            return "forbidden_added_code:" + ",".join(forbidden_text)

        for relative in changed_files:
            path = worktree / relative
            if not path.exists() or path.suffix != ".py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                return f"invalid_python:{relative}:{exc}"
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                            return f"forbidden_import:{relative}:{alias.name}"
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    if root in _FORBIDDEN_IMPORTS:
                        return f"forbidden_import:{relative}:{node.module}"
                elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
                    return f"forbidden_name:{relative}:{node.id}"
                elif isinstance(node, ast.Attribute) and (
                    node.attr in _FORBIDDEN_ATTRIBUTES
                    or (node.attr.startswith("__") and node.attr.endswith("__"))
                ):
                    return f"forbidden_attribute:{relative}:{node.attr}"
        return None

    @staticmethod
    def _sanitized_failure(failure: dict) -> dict:
        """Give the engineer useful error structure without task/document identifiers."""
        hidden = {
            "task_id",
            "retrieved_document_ids",
            "supporting_document_ids",
            "distractor_document_ids",
        }
        return {key: value for key, value in failure.items() if key not in hidden}

    def _tests(self, worktree: Path) -> tuple[bool, str | None]:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=self.config.test_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "tests_timed_out"
        if completed.returncode != 0:
            return False, "tests_failed:" + (completed.stdout + completed.stderr)[-2000:]
        return True, None

    def _changed_files(self, worktree: Path) -> tuple[str, ...]:
        subprocess.run(
            ["git", "add", "-N", "evolving_loop"],
            cwd=worktree,
            capture_output=True,
            check=False,
        )
        output = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return tuple(line.strip() for line in output.splitlines() if line.strip())

    @staticmethod
    def _diff(worktree: Path) -> str:
        subprocess.run(
            ["git", "add", "-N", "evolving_loop"],
            cwd=worktree,
            capture_output=True,
            check=False,
        )
        return subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    @staticmethod
    def _apply_patch(worktree: Path, patch: str) -> None:
        completed = subprocess.run(
            ["git", "apply", "--binary", "-"],
            cwd=worktree,
            input=patch,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"could not apply incumbent patch: {completed.stderr[-2000:]}")

    def _worktree(self, prefix: str):
        return _Worktree(self.repo_root, prefix)

    @staticmethod
    def _candidate_summary(candidate: SourceCandidate) -> dict:
        return {
            "candidate_id": candidate.candidate_id,
            "audit_ok": candidate.audit_ok,
            "tests_ok": candidate.tests_ok,
            "rejection_reason": candidate.rejection_reason,
            "changed_files": list(candidate.changed_files),
            "train_reward": candidate.evaluation.train_reward if candidate.evaluation else None,
            "val_reward": candidate.evaluation.val_reward if candidate.evaluation else None,
            "engineer_summary": candidate.engineer_summary[:1000],
        }

    def _progress(self, event: str, **payload) -> None:
        if self.config.progress_path is None:
            return
        destination = Path(self.config.progress_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "event": event,
                        **payload,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    def _save_checkpoint(
        self,
        incumbent_patch: str,
        evaluation: SourceEvaluation,
        trace: list[SourceEvolutionStep],
        next_generation: int,
    ) -> None:
        if self.config.checkpoint_path is None:
            return
        destination = Path(self.config.checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "next_generation": next_generation,
                    "incumbent_patch": incumbent_patch,
                    "incumbent_evaluation": asdict(evaluation),
                    "trace": [asdict(item) for item in trace],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(destination)

    def _load_checkpoint(
        self,
    ) -> tuple[str, SourceEvaluation, list[SourceEvolutionStep], int] | None:
        if (
            not self.config.resume
            or self.config.checkpoint_path is None
            or not Path(self.config.checkpoint_path).exists()
        ):
            return None
        payload = json.loads(Path(self.config.checkpoint_path).read_text(encoding="utf-8"))
        raw_evaluation = payload["incumbent_evaluation"]
        evaluation = SourceEvaluation(
            train_reward=float(raw_evaluation["train_reward"]),
            val_reward=float(raw_evaluation["val_reward"]),
            train_module_rewards=dict(raw_evaluation["train_module_rewards"]),
            val_module_rewards=dict(raw_evaluation["val_module_rewards"]),
            failure_traces=tuple(raw_evaluation.get("failure_traces", ())),
        )
        trace = []
        for raw in payload.get("trace", []):
            item = dict(raw)
            item["candidates"] = tuple(item["candidates"])
            trace.append(SourceEvolutionStep(**item))
        start = int(payload.get("next_generation", len(trace)))
        self._progress("checkpoint_resumed", next_generation=start)
        return str(payload.get("incumbent_patch", "")), evaluation, trace, start


class _Worktree:
    def __init__(self, repo_root: Path, prefix: str) -> None:
        self.repo_root = repo_root
        self.prefix = prefix
        self._temporary: tempfile.TemporaryDirectory | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self._temporary = tempfile.TemporaryDirectory(prefix=self.prefix)
        self.path = Path(self._temporary.name)
        completed = subprocess.run(
            ["git", "worktree", "add", "--detach", str(self.path), "HEAD"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            self._temporary.cleanup()
            raise RuntimeError(f"could not create source worktree: {completed.stderr[-2000:]}")
        return self.path

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.path is not None:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(self.path)],
                cwd=self.repo_root,
                capture_output=True,
                check=False,
            )
        if self._temporary is not None:
            self._temporary.cleanup()


def save_source_trace(path: str | Path, trace: tuple[SourceEvolutionStep, ...]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps([asdict(item) for item in trace], indent=2))
