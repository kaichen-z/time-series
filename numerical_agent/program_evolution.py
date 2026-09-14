"""Private grading primitives for whole-program forecast evolution."""
from __future__ import annotations

import ast
import fcntl
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from common.data import Task, load_tasks_by_id
from common.llm import QwenClient, VLLMClient, VLLMRequestError, parse_json_object
from common.metrics import drcik_point_metrics, standard_error
from common.sandbox import ALLOWED_IMPORTS, UnsafeCodeError, check_code


TRAIN_TASK_COUNT = 100
FAILURE_SCORE = 5.0
PROGRAM_IMPORTS = ALLOWED_IMPORTS | frozenset({"__future__"})

_RUNNER = Path(__file__).parents[1] / "common" / "sandbox" / "_sandbox_runner.py"
_PASSTHROUGH_ENV = ("PATH", "HOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL")
_SINGLE_THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_COMMIT_HASH_LENGTHS = frozenset({40, 64})
_AGENT_ID = re.compile(r"agent-[a-z0-9][a-z0-9_-]*\Z")
_CASE_REFERENCE = re.compile(r"\bcase_[a-zA-Z0-9_-]+\b")
_IDENTITY_REFERENCE = re.compile(
    r"\b(?:task|entity)[_-]?id\b|\btest[_ -]?(?:case|task)[_-]?\d+\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProgramMetrics:
    """Aggregate evidence returned by the private Train grader."""

    source_valid: bool
    task_count: int
    success_count: int
    failure_count: int
    coverage: float
    mean_smae: float
    mean_srmse: float
    se_smae: float
    se_srmse: float
    score: float
    runtime_seconds: float
    failures: Mapping[str, int]

    def public_payload(self) -> dict[str, object]:
        """Return aggregate-only feedback suitable for shared attempt records."""
        payload = asdict(self)
        payload["failures"] = dict(sorted(self.failures.items()))
        return payload


@dataclass(frozen=True)
class CaseDiagnostic:
    """Private anonymous evidence for one Train case."""

    case_id: str
    history: tuple[float, ...]
    truth: tuple[float, ...]
    forecast: tuple[float, ...] | None
    frequency: str
    horizon: int
    smae: float
    srmse: float
    failure: str | None


@dataclass(frozen=True)
class ProgramEvaluation:
    """Aggregate public metrics plus private anonymous Train diagnostics."""

    metrics: ProgramMetrics
    cases: tuple[CaseDiagnostic, ...]


def validate_program_source(source: str) -> None:
    """Require a passive module containing one three-argument forecast entry point."""
    check_code(source, allowed=PROGRAM_IMPORTS)
    tree = ast.parse(source)
    forecasts = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forecast"
    ]
    if len(forecasts) != 1 or isinstance(forecasts[0], ast.AsyncFunctionDef):
        raise UnsafeCodeError("program must define exactly one synchronous forecast function")
    function = forecasts[0]
    if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        raise UnsafeCodeError("forecast must use only history, horizon, frequency")
    names = tuple(argument.arg for argument in function.args.posonlyargs + function.args.args)
    if names != ("history", "horizon", "frequency") or function.args.defaults:
        raise UnsafeCodeError("forecast signature must be (history, horizon, frequency)")

    passive = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, passive):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None or not any(isinstance(child, ast.Call) for child in ast.walk(value)):
                continue
        raise UnsafeCodeError("program may not execute code while the module is loaded")


def load_training_tasks(split_file: str | Path, tasks_file: str | Path) -> tuple[Task, ...]:
    """Load exactly the authorized 100-task Train partition."""
    manifest = json.loads(Path(split_file).read_text(encoding="utf-8"))
    try:
        raw_ids = manifest["partitions"]["train"]["task_ids"]
    except (KeyError, TypeError) as exc:
        raise ValueError("split manifest has no Train task IDs") from exc
    if not isinstance(raw_ids, list) or any(not isinstance(task_id, str) for task_id in raw_ids):
        raise ValueError("Train task IDs must be a list of strings")
    task_ids = tuple(raw_ids)
    if len(task_ids) != TRAIN_TASK_COUNT or len(set(task_ids)) != TRAIN_TASK_COUNT:
        raise ValueError("Train partition must contain exactly 100 unique tasks")
    tasks = tuple(load_tasks_by_id(tasks_file, task_ids))
    if len(tasks) != TRAIN_TASK_COUNT or tuple(task.task_id for task in tasks) != task_ids:
        raise ValueError("all 100 Train tasks must exist and contain public labels")
    return tasks


def _sandbox_environment() -> dict[str, str]:
    env = {name: os.environ[name] for name in _PASSTHROUGH_ENV if name in os.environ}
    env.update({name: "1" for name in _SINGLE_THREAD_ENV})
    return env


def _run_program(
    source: str,
    task: Task,
    timeout_seconds: float | None,
    memory_mb: int,
) -> tuple[tuple[float, ...] | None, str | None, float]:
    request = {
        "code": source,
        "history": list(task.history_values),
        "horizon": task.prediction_length,
        "frequency": task.frequency,
        "memory_mb": memory_mb,
    }
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, str(_RUNNER)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=_sandbox_environment(),
        )
    except subprocess.TimeoutExpired:
        return None, "timeout", time.monotonic() - started
    duration = time.monotonic() - started
    if completed.returncode != 0 or not completed.stdout.strip():
        return None, "process_error", duration
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, TypeError):
        return None, "invalid_output", duration
    if not payload.get("ok"):
        # Keep the sandbox's own diagnosis: without it an agent sees only "exception".
        reported = str(payload.get("error") or "").strip().replace("\n", " ")
        return None, f"exception: {reported[:200]}" if reported else "exception", duration
    try:
        forecast = tuple(float(value) for value in payload["forecast"])
    except (KeyError, TypeError, ValueError):
        return None, "invalid_output", duration
    if len(forecast) != task.prediction_length or not all(math.isfinite(value) for value in forecast):
        return None, "invalid_output", duration
    return forecast, None, duration


class ProgramGrader:
    """Evaluate untrusted programs privately on an authorized task collection."""

    def __init__(
        self,
        tasks: Sequence[Task],
        *,
        expected_task_count: int = TRAIN_TASK_COUNT,
        timeout_seconds: float | None = None,
        memory_mb: int = 2048,
    ) -> None:
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0.0
        ):
            raise ValueError("timeout_seconds must be None or a positive finite number")
        if isinstance(expected_task_count, bool) or expected_task_count <= 0:
            raise ValueError("expected_task_count must be positive")
        if len(tasks) != expected_task_count:
            raise ValueError(f"grader requires exactly {expected_task_count} tasks")
        if memory_mb <= 0:
            raise ValueError("memory_mb must be positive")
        self._tasks = tuple(tasks)
        self._timeout_seconds = None if timeout_seconds is None else float(timeout_seconds)
        self._memory_mb = int(memory_mb)

    def grade(self, source: str) -> ProgramMetrics:
        """Grade one source replacement and return aggregate-only evidence."""
        return self.grade_with_diagnostics(source).metrics

    def grade_with_diagnostics(self, source: str) -> ProgramEvaluation:
        """Grade one source replacement while retaining private Train evidence."""
        started = time.monotonic()
        try:
            validate_program_source(source)
        except (SyntaxError, UnsafeCodeError):
            metrics = self._failed_metrics(
                "unsafe_source", time.monotonic() - started, source_valid=False
            )
            cases = tuple(
                self._failed_case(index, task, "unsafe_source")
                for index, task in enumerate(self._tasks, start=1)
            )
            return ProgramEvaluation(metrics, cases)

        smae_values: list[float] = []
        srmse_values: list[float] = []
        cases: list[CaseDiagnostic] = []
        failures: Counter[str] = Counter()
        successes = 0
        for index, task in enumerate(self._tasks, start=1):
            prediction, failure, _ = _run_program(
                source,
                task,
                self._timeout_seconds,
                self._memory_mb,
            )
            if failure is not None or prediction is None:
                failures[failure or "invalid_output"] += 1
                smae_values.append(FAILURE_SCORE)
                srmse_values.append(FAILURE_SCORE)
                cases.append(self._failed_case(index, task, failure or "invalid_output"))
                continue
            try:
                row = drcik_point_metrics(task.future_values, prediction, cap=FAILURE_SCORE)
            except (ArithmeticError, TypeError, ValueError):
                failures["invalid_output"] += 1
                smae_values.append(FAILURE_SCORE)
                srmse_values.append(FAILURE_SCORE)
                cases.append(self._failed_case(index, task, "invalid_output"))
                continue
            successes += 1
            smae = float(row["smae"])
            srmse = float(row["srmse"])
            smae_values.append(smae)
            srmse_values.append(srmse)
            cases.append(CaseDiagnostic(
                case_id=f"case_{index:03d}",
                history=task.history_values,
                truth=task.future_values,
                forecast=prediction,
                frequency=task.frequency,
                horizon=task.prediction_length,
                smae=smae,
                srmse=srmse,
                failure=None,
            ))

        task_count = len(self._tasks)
        mean_smae = statistics.fmean(smae_values)
        mean_srmse = statistics.fmean(srmse_values)
        metrics = ProgramMetrics(
            source_valid=True,
            task_count=task_count,
            success_count=successes,
            failure_count=task_count - successes,
            coverage=successes / task_count,
            mean_smae=mean_smae,
            mean_srmse=mean_srmse,
            se_smae=standard_error(smae_values),
            se_srmse=standard_error(srmse_values),
            score=0.5 * mean_smae + 0.5 * mean_srmse,
            runtime_seconds=time.monotonic() - started,
            failures=dict(sorted(failures.items())),
        )
        return ProgramEvaluation(metrics, tuple(cases))

    def _failed_metrics(self, category: str, runtime_seconds: float, *, source_valid: bool) -> ProgramMetrics:
        count = len(self._tasks)
        return ProgramMetrics(
            source_valid=source_valid,
            task_count=count,
            success_count=0,
            failure_count=count,
            coverage=0.0,
            mean_smae=FAILURE_SCORE,
            mean_srmse=FAILURE_SCORE,
            se_smae=0.0,
            se_srmse=0.0,
            score=FAILURE_SCORE,
            runtime_seconds=runtime_seconds,
            failures={category: count},
        )

    @staticmethod
    def _failed_case(index: int, task: Task, category: str) -> CaseDiagnostic:
        return CaseDiagnostic(
            case_id=f"case_{index:03d}",
            history=task.history_values,
            truth=task.future_values,
            forecast=None,
            frequency=task.frequency,
            horizon=task.prediction_length,
            smae=FAILURE_SCORE,
            srmse=FAILURE_SCORE,
            failure=category,
        )


class AttemptStore:
    """Atomically persist aggregate attempt records by commit hash."""

    def __init__(self, public_root: str | Path) -> None:
        self._public_root = Path(public_root)
        self._attempts = self._public_root / "attempts"
        self._attempts.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        commit_hash: str,
        parent_hash: str | None,
        agent_id: str,
        backend: str,
        hypothesis: str,
        metrics: ProgramMetrics,
    ) -> Path:
        """Write one immutable JSON record without exposing task-level evidence."""
        _validate_commit_hash(commit_hash)
        if parent_hash is not None:
            _validate_commit_hash(parent_hash)
        payload: dict[str, object] = {
            "schema_version": 1,
            "commit_hash": commit_hash,
            "parent_hash": parent_hash,
            "agent_id": str(agent_id),
            "backend": str(backend),
            "hypothesis": str(hypothesis),
            "metrics": metrics.public_payload(),
        }
        target = self._attempts / f"{commit_hash}.json"
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            comparable = dict(existing)
            comparable.pop("status", None)
            if comparable != payload:
                raise FileExistsError(f"attempt already exists for commit {commit_hash}")
            return target
        previous = [
            record
            for record in self.records()
            if record["agent_id"] == agent_id
            and bool(record["metrics"]["source_valid"])
            and int(record["metrics"]["success_count"]) > 0
        ]
        if not metrics.source_valid or metrics.success_count == 0:
            payload["status"] = "failed"
        elif not previous or metrics.score < min(float(record["metrics"]["score"]) for record in previous):
            payload["status"] = "personal_best"
        else:
            payload["status"] = "evaluated"
        _atomic_write_json(target, payload)
        self._refresh_leaderboard()
        return target

    def records(self) -> list[dict[str, object]]:
        """Read all immutable attempts in stable commit order."""
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self._attempts.glob("*.json"))
        ]

    def get(self, commit_hash: str) -> dict[str, object] | None:
        """Return one recorded attempt when it exists."""
        _validate_commit_hash(commit_hash)
        path = self._attempts / f"{commit_hash}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def parent_candidates(self) -> list[dict[str, object]]:
        """Rank valid parent candidates by the aggregate composite score."""
        eligible = [
            record
            for record in self.records()
            if bool(record["metrics"]["source_valid"])
            and int(record["metrics"]["success_count"]) > 0
        ]
        return sorted(
            eligible,
            key=lambda record: (float(record["metrics"]["score"]), str(record["commit_hash"])),
        )

    def lineage(self) -> dict[str, tuple[str, ...]]:
        """Reconstruct and validate the attempt lineage DAG."""
        parents = {
            str(record["commit_hash"]): (
                None if record["parent_hash"] is None else str(record["parent_hash"])
            )
            for record in self.records()
        }
        for start in parents:
            seen: set[str] = set()
            current: str | None = start
            while current in parents:
                if current in seen:
                    raise ValueError("attempt lineage contains a cycle")
                seen.add(current)
                current = parents[current]
        children: dict[str, list[str]] = {commit: [] for commit in parents}
        for commit, parent in parents.items():
            if parent is not None:
                children.setdefault(parent, []).append(commit)
        return {commit: tuple(sorted(values)) for commit, values in sorted(children.items())}

    def _refresh_leaderboard(self) -> None:
        rows = [
            {
                "rank": rank,
                "commit_hash": record["commit_hash"],
                "agent_id": record["agent_id"],
                "score": record["metrics"]["score"],
                "status": record["status"],
            }
            for rank, record in enumerate(self.parent_candidates(), start=1)
        ]
        _atomic_write_json(
            self._public_root / "leaderboard.json",
            {"schema_version": 1, "entries": rows},
        )


def _validate_commit_hash(value: str) -> None:
    if len(value) not in _COMMIT_HASH_LENGTHS or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("commit hash must be a full lowercase hexadecimal digest")


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _git(directory: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _validate_agent_id(agent_id: str) -> None:
    if _AGENT_ID.fullmatch(agent_id) is None:
        raise ValueError("agent ID must match agent-[a-z0-9_-]+")


class EvolutionRun:
    """Own isolated worktrees, shared evidence, and concurrent grading state."""

    PROGRAM_FILE = "program.py"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.repo = self.root / "repo"
        self.agents = self.root / "agents"
        self.public = self.root / ".evolution" / "public"
        self.private = self.root / ".evolution" / "private"
        self.sessions_file = self.public / "sessions.json"
        self.eval_count_file = self.public / "eval_count"
        self.queue_file = self.private / "grade_queue.json"
        self.heartbeat_queue_file = self.private / "heartbeat_queue.json"
        self.diagnostics = self.private / "diagnostics"
        self.agent_state_dir = self.private / "agent_state"
        self.lock_file = self.private / "grading.lock"
        self.heartbeat_lock_file = self.private / "heartbeat.lock"
        self.state_lock_file = self.private / "state.lock"
        self.attempts = AttemptStore(self.public)

    @classmethod
    def create(
        cls,
        root: str | Path,
        seed_source: str,
        agent_ids: Sequence[str],
        evaluation_budget: int,
    ) -> "EvolutionRun":
        """Create a run repository and one persistent branch worktree per agent."""
        validate_program_source(seed_source)
        if isinstance(evaluation_budget, bool) or evaluation_budget <= 0:
            raise ValueError("evaluation_budget must be positive")
        unique_agents = tuple(agent_ids)
        if not unique_agents or len(set(unique_agents)) != len(unique_agents):
            raise ValueError("agent IDs must be nonempty and unique")
        for agent_id in unique_agents:
            _validate_agent_id(agent_id)
        root_path = Path(root)
        if root_path.exists() and any(root_path.iterdir()):
            raise FileExistsError(f"run directory is not empty: {root_path}")
        run = cls(root_path)
        run.repo.mkdir(parents=True, exist_ok=True)
        run.agents.mkdir(parents=True, exist_ok=True)
        for directory in (
            run.public / "attempts",
            run.public / "notes",
            run.public / "skills",
            run.public / "logs",
            run.public / "heartbeat" / "events",
            run.public / "synthesis",
            run.private / "taskdata",
            run.private / "grader_checkouts",
            run.private / "diagnostics",
            run.private / "agent_state",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(run.repo / cls.PROGRAM_FILE, seed_source)
        _git(run.repo, "init")
        _git(run.repo, "config", "user.name", "Program Evolution Host")
        _git(run.repo, "config", "user.email", "program-evolution@localhost")
        _git(run.repo, "add", cls.PROGRAM_FILE)
        _git(run.repo, "commit", "-m", "seed forecasting program")
        _git(run.repo, "branch", "-M", "main")
        seed_commit = _git(run.repo, "rev-parse", "HEAD")
        exclude = run.repo / ".git" / "info" / "exclude"
        with exclude.open("a", encoding="utf-8") as handle:
            handle.write("\n.evolution\n")

        sessions: dict[str, object] = {
            "schema_version": 1,
            "status": "running",
            "evaluation_budget": evaluation_budget,
            "backend": None,
            "model": None,
            "agents": {},
        }
        agent_state = sessions["agents"]
        assert isinstance(agent_state, dict)
        for agent_id in unique_agents:
            worktree = run.agents / agent_id
            branch = f"agents/{agent_id}"
            _git(run.repo, "worktree", "add", "-b", branch, str(worktree), seed_commit)
            relative_public = os.path.relpath(run.public, worktree)
            (worktree / ".evolution").symlink_to(relative_public, target_is_directory=True)
            (run.public / "notes" / agent_id).mkdir(parents=True, exist_ok=True)
            agent_state[agent_id] = {
                "branch": branch,
                "worktree": str(Path("agents") / agent_id),
                "head": seed_commit,
                "session_id": None,
                "phase": "idle",
                "outstanding_commit": None,
                "last_error": None,
                "last_heartbeat": None,
                "heartbeat_count": 0,
                "current_hypothesis": None,
            }
        _atomic_write_json(run.sessions_file, sessions)
        _atomic_write_json(run.queue_file, {"schema_version": 2, "next_sequence": 1, "entries": []})
        _atomic_write_json(
            run.heartbeat_queue_file,
            {"schema_version": 1, "next_sequence": 1, "entries": []},
        )
        _atomic_write_text(run.public / "connections.md", "# Connections\n\nNone yet.\n")
        _atomic_write_text(run.public / "open-questions.md", "# Open questions\n\nNone yet.\n")
        _atomic_write_text(run.eval_count_file, "0\n")
        return run

    def sessions(self) -> dict[str, object]:
        """Read persistent run and agent state."""
        return json.loads(self.sessions_file.read_text(encoding="utf-8"))

    def agent_head(self, agent_id: str) -> str:
        """Return an agent branch's current persistent head."""
        worktree = self._agent_worktree(agent_id)
        return _git(worktree, "rev-parse", "HEAD")

    def checkout_parent(self, agent_id: str, commit_hash: str) -> None:
        """Move a clean agent branch to a parent chosen from recorded attempts."""
        _validate_commit_hash(commit_hash)
        if self.attempts.get(commit_hash) not in self.attempts.parent_candidates():
            raise ValueError("parent must be a valid recorded attempt")
        worktree = self._agent_worktree(agent_id)
        if _git(worktree, "status", "--porcelain"):
            raise RuntimeError("agent worktree contains uncommitted changes")
        branch = str(self._agent_state(agent_id)["branch"])
        _git(worktree, "checkout", "-B", branch, commit_hash)
        self._update_agent_head(agent_id, commit_hash)

    def commit_candidate(self, agent_id: str, source: str, message: str) -> str:
        """Validate and commit one host-mediated source replacement."""
        validate_program_source(source)
        worktree = self._agent_worktree(agent_id)
        if _git(worktree, "status", "--porcelain"):
            raise RuntimeError("agent worktree contains uncommitted changes")
        _atomic_write_text(worktree / self.PROGRAM_FILE, source)
        _git(worktree, "add", self.PROGRAM_FILE)
        _git(worktree, "commit", "-m", message)
        head = _git(worktree, "rev-parse", "HEAD")
        self._update_agent_head(agent_id, head)
        return head

    def commit_edited_candidate(self, agent_id: str, message: str) -> str:
        """Validate and commit the sole program edit made by a coding session."""
        worktree = self._agent_worktree(agent_id)
        changes = tuple(
            line.strip()
            for line in _git(worktree, "status", "--porcelain", "--untracked-files=all").splitlines()
            if line.strip()
        )
        if changes != (f"M {self.PROGRAM_FILE}",):
            raise RuntimeError("coding session must modify only program.py")
        validate_program_source((worktree / self.PROGRAM_FILE).read_text(encoding="utf-8"))
        _git(worktree, "add", self.PROGRAM_FILE)
        _git(worktree, "commit", "-m", message)
        head = _git(worktree, "rev-parse", "HEAD")
        self._update_agent_head(agent_id, head)
        return head

    def configure_backend(self, backend: str, model: str) -> None:
        """Select one immutable backend and model for the run."""
        if backend not in {"qwen", "qwen-vllm", "codex", "claude"}:
            raise ValueError(f"unsupported backend: {backend}")
        if not model.strip():
            raise ValueError("model must be nonempty")
        with _file_lock(self.state_lock_file):
            sessions = self.sessions()
            configured = (sessions.get("backend"), sessions.get("model"))
            requested = (backend, model)
            if configured != (None, None) and configured != requested:
                raise RuntimeError("a run cannot mix backends or models")
            sessions["backend"], sessions["model"] = requested
            _atomic_write_json(self.sessions_file, sessions)

    def configure_heartbeat(self, actions: int, idle_seconds: float, curator_model: str) -> None:
        """Persist immutable heartbeat and curator settings."""
        if isinstance(actions, bool) or actions <= 0:
            raise ValueError("heartbeat actions must be positive")
        if isinstance(idle_seconds, bool) or not math.isfinite(idle_seconds) or idle_seconds <= 0:
            raise ValueError("heartbeat idle seconds must be positive and finite")
        requested = {
            "actions": int(actions),
            "idle_seconds": float(idle_seconds),
            "curator_model": curator_model,
        }
        with _file_lock(self.state_lock_file):
            sessions = self.sessions()
            existing = sessions.get("heartbeat")
            if existing is not None and existing != requested:
                raise RuntimeError("heartbeat configuration cannot change within a run")
            sessions["heartbeat"] = requested
            _atomic_write_json(self.sessions_file, sessions)
        for directory in (
            self.public / "heartbeat" / "events",
            self.public / "synthesis",
            self.agent_state_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if not self.heartbeat_queue_file.is_file():
            _atomic_write_json(
                self.heartbeat_queue_file,
                {"schema_version": 1, "next_sequence": 1, "entries": []},
            )
        for path, content in (
            (self.public / "connections.md", "# Connections\n\nNone yet.\n"),
            (self.public / "open-questions.md", "# Open questions\n\nNone yet.\n"),
        ):
            if not path.is_file():
                _atomic_write_text(path, content)

    def agent_checkpoint(self, agent_id: str) -> dict[str, object]:
        """Read one private resumable research checkpoint."""
        _validate_agent_id(agent_id)
        path = self.agent_state_dir / f"{agent_id}.json"
        return {} if not path.is_file() else json.loads(path.read_text(encoding="utf-8"))

    def save_agent_checkpoint(self, agent_id: str, checkpoint: Mapping[str, object]) -> None:
        """Atomically persist one private research checkpoint."""
        _validate_agent_id(agent_id)
        _atomic_write_json(self.agent_state_dir / f"{agent_id}.json", dict(checkpoint))
        with _file_lock(self.state_lock_file):
            sessions = self.sessions()
            sessions["agents"][agent_id]["current_hypothesis"] = checkpoint.get("hypothesis")
            _atomic_write_json(self.sessions_file, sessions)

    def record_heartbeat(
        self,
        agent_id: str,
        trigger: str,
        checkpoint: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Publish one heartbeat and enqueue asynchronous synthesis."""
        _validate_agent_id(agent_id)
        if not trigger.strip():
            raise ValueError("heartbeat trigger must be nonempty")
        if checkpoint is not None:
            self.save_agent_checkpoint(agent_id, checkpoint)
        with _file_lock(self.heartbeat_lock_file):
            queue = self._read_heartbeat_queue()
            sequence = int(queue["next_sequence"])
            event = {
                "sequence": sequence,
                "agent_id": agent_id,
                "trigger": trigger,
                "created_at": time.time(),
            }
            queue["entries"].append({**event, "status": "queued", "started_at": None})
            queue["next_sequence"] = sequence + 1
            _atomic_write_json(self.heartbeat_queue_file, queue)
            _atomic_write_json(
                self.public / "heartbeat" / "events" / f"{sequence:06d}.json",
                event,
            )
        with _file_lock(self.state_lock_file):
            sessions = self.sessions()
            state = sessions["agents"][agent_id]
            state["last_heartbeat"] = event["created_at"]
            state["heartbeat_count"] = int(state.get("heartbeat_count", 0)) + 1
            _atomic_write_json(self.sessions_file, sessions)
        return event

    def latest_heartbeat(self, agent_id: str) -> dict[str, object] | None:
        """Return the latest public heartbeat for one agent."""
        events = []
        for path in (self.public / "heartbeat" / "events").glob("*.json"):
            event = json.loads(path.read_text(encoding="utf-8"))
            if event["agent_id"] == agent_id:
                events.append(event)
        return None if not events else max(events, key=lambda event: int(event["sequence"]))

    def shared_memory_stamp(self) -> tuple[tuple[str, int, int], ...]:
        """Return a stable stamp for attempts and agent-authored memory."""
        roots = (self.public / "attempts", self.public / "notes", self.public / "skills")
        rows = []
        for root in roots:
            for path in root.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    rows.append((str(path.relative_to(self.public)), stat.st_mtime_ns, stat.st_size))
        return tuple(sorted(rows))

    def agent_session(self, agent_id: str) -> str | None:
        """Return the persisted coding-session ID for one agent."""
        value = self._agent_state(agent_id).get("session_id")
        return None if value is None else str(value)

    def save_agent_session(self, agent_id: str, session_id: str | None) -> None:
        """Persist a resumable coding-session ID without exposing model state."""
        if session_id is None:
            return
        if not session_id.strip():
            raise ValueError("session ID must be nonempty")
        with _file_lock(self.state_lock_file):
            sessions = self.sessions()
            if agent_id not in sessions["agents"]:
                raise KeyError(f"unknown agent: {agent_id}")
            sessions["agents"][agent_id]["session_id"] = session_id
            _atomic_write_json(self.sessions_file, sessions)

    def set_agent_activity(
        self,
        agent_id: str,
        phase: str,
        *,
        outstanding_commit: str | None = None,
        error: str | None = None,
    ) -> None:
        """Persist one agent's current asynchronous activity."""
        _validate_agent_id(agent_id)
        if phase not in {
            "idle", "planning", "researching", "waiting_for_grade", "reflecting",
            "complete", "failed", "stopped"
        }:
            raise ValueError("unsupported agent phase")
        if outstanding_commit is not None:
            _validate_commit_hash(outstanding_commit)
        with _file_lock(self.state_lock_file):
            sessions = self.sessions()
            if agent_id not in sessions["agents"]:
                raise KeyError(f"unknown agent: {agent_id}")
            state = sessions["agents"][agent_id]
            state["phase"] = phase
            state["outstanding_commit"] = outstanding_commit
            state["last_error"] = error
            _atomic_write_json(self.sessions_file, sessions)

    def submit(
        self,
        *,
        agent_id: str,
        commit_hash: str,
        backend: str,
        hypothesis: str,
    ) -> int:
        """Append one committed candidate to the persistent FIFO grading queue."""
        _validate_commit_hash(commit_hash)
        with _file_lock(self.lock_file):
            sessions = self.sessions()
            if sessions["status"] != "running":
                raise RuntimeError("run is stopped")
            worktree = self._agent_worktree(agent_id)
            _git(worktree, "cat-file", "-e", f"{commit_hash}^{{commit}}")
            parent_hash = _git(worktree, "rev-parse", f"{commit_hash}^")
            if self.attempts.get(parent_hash) is None:
                raise ValueError("candidate parent must have a recorded evaluation")
            queue = self._read_queue()
            entries = queue["entries"]
            assert isinstance(entries, list)
            if any(entry["commit_hash"] == commit_hash for entry in entries):
                raise ValueError("candidate is already queued")
            count = self._sync_eval_count(entries)
            reserved = sum(entry["status"] in {"queued", "running"} for entry in entries)
            if count + reserved >= int(sessions["evaluation_budget"]):
                raise RuntimeError("global evaluation budget is fully reserved")
            sequence = int(queue["next_sequence"])
            entries.append({
                "sequence": sequence,
                "agent_id": agent_id,
                "commit_hash": commit_hash,
                "parent_hash": parent_hash,
                "backend": str(backend),
                "hypothesis": str(hypothesis),
                "status": "queued",
                "grader_id": None,
                "started_at": None,
            })
            queue["next_sequence"] = sequence + 1
            _atomic_write_json(self.queue_file, queue)
            return sequence

    def grade_next(
        self,
        grader: ProgramGrader,
        grader_id: str = "grader-1",
    ) -> dict[str, object] | None:
        """Claim and grade one FIFO candidate without holding the queue lock."""
        entry = self._claim_next(grader_id)
        if entry is None:
            return None
        commit_hash = str(entry["commit_hash"])
        try:
            existing = self.attempts.get(commit_hash)
            metrics = None
            if existing is None:
                source = _git(self.repo, "show", f"{commit_hash}:{self.PROGRAM_FILE}") + "\n"
                if hasattr(grader, "grade_with_diagnostics"):
                    evaluation = grader.grade_with_diagnostics(source)
                    metrics = evaluation.metrics
                    self.save_diagnostics(commit_hash, str(entry["parent_hash"]), evaluation)
                else:
                    metrics = grader.grade(source)
            with _file_lock(self.lock_file):
                if existing is None:
                    assert metrics is not None
                    self.attempts.record(
                        commit_hash=commit_hash,
                        parent_hash=str(entry["parent_hash"]),
                        agent_id=str(entry["agent_id"]),
                        backend=str(entry["backend"]),
                        hypothesis=str(entry["hypothesis"]),
                        metrics=metrics,
                    )
                queue = self._read_queue()
                entries = queue["entries"]
                assert isinstance(entries, list)
                current = next(item for item in entries if item["commit_hash"] == commit_hash)
                if current["status"] != "running" or current["grader_id"] != grader_id:
                    raise RuntimeError("grading lease ownership changed during evaluation")
                current.update(status="complete", grader_id=None, started_at=None)
                _atomic_write_json(self.queue_file, queue)
                self._sync_eval_count(entries)
            record = self.attempts.get(commit_hash)
            assert record is not None
            return record
        except Exception:
            self._release_grading_lease(commit_hash, grader_id)
            raise

    def recover_grading_leases(self) -> int:
        """Return abandoned running evaluations to the FIFO queue on startup."""
        with _file_lock(self.lock_file):
            stored = json.loads(self.queue_file.read_text(encoding="utf-8"))
            needs_migration = int(stored.get("schema_version", 1)) != 2
            queue = self._read_queue()
            entries = queue["entries"]
            assert isinstance(entries, list)
            recovered = 0
            for entry in entries:
                if entry["status"] == "running":
                    entry.update(status="queued", grader_id=None, started_at=None)
                    recovered += 1
            if recovered or needs_migration:
                _atomic_write_json(self.queue_file, queue)
            return recovered

    def _claim_next(self, grader_id: str) -> dict[str, object] | None:
        if not grader_id.strip():
            raise ValueError("grader ID must be nonempty")
        with _file_lock(self.lock_file):
            sessions = self.sessions()
            if sessions["status"] != "running":
                return None
            queue = self._read_queue()
            entries = queue["entries"]
            assert isinstance(entries, list)
            count = self._sync_eval_count(entries)
            if count >= int(sessions["evaluation_budget"]):
                return None
            entry = next((item for item in entries if item["status"] == "queued"), None)
            if entry is None:
                return None
            entry.update(
                status="running",
                grader_id=grader_id,
                started_at=time.time(),
            )
            _atomic_write_json(self.queue_file, queue)
            return dict(entry)

    def _release_grading_lease(self, commit_hash: str, grader_id: str) -> None:
        with _file_lock(self.lock_file):
            queue = self._read_queue()
            entries = queue["entries"]
            assert isinstance(entries, list)
            current = next(item for item in entries if item["commit_hash"] == commit_hash)
            if current["status"] == "running" and current["grader_id"] == grader_id:
                current.update(status="queued", grader_id=None, started_at=None)
                _atomic_write_json(self.queue_file, queue)

    def save_diagnostics(
        self,
        commit_hash: str,
        parent_hash: str | None,
        evaluation: ProgramEvaluation,
    ) -> Path:
        """Persist anonymous parent-child Train diagnostics outside agent-visible memory."""
        _validate_commit_hash(commit_hash)
        if parent_hash is not None:
            _validate_commit_hash(parent_hash)
        parent_cases: dict[str, dict[str, object]] = {}
        if parent_hash is not None:
            parent_path = self.diagnostics / f"{parent_hash}.json"
            if not parent_path.is_file():
                raise ValueError("parent diagnostics are missing")
            parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
            parent_cases = {
                str(case["case_id"]): case for case in parent_payload["cases"]
            }
        cases = []
        for case in evaluation.cases:
            parent = parent_cases.get(case.case_id)
            parent_smae = None if parent is None else float(parent["child_smae"])
            parent_srmse = None if parent is None else float(parent["child_srmse"])
            cases.append({
                "case_id": case.case_id,
                "frequency": case.frequency,
                "horizon": case.horizon,
                "history": list(case.history),
                "truth": list(case.truth),
                "parent_forecast": None if parent is None else parent["child_forecast"],
                "child_forecast": None if case.forecast is None else list(case.forecast),
                "parent_smae": parent_smae,
                "parent_srmse": parent_srmse,
                "child_smae": case.smae,
                "child_srmse": case.srmse,
                "delta_smae": None if parent_smae is None else case.smae - parent_smae,
                "delta_srmse": None if parent_srmse is None else case.srmse - parent_srmse,
                "failure": case.failure,
            })
        payload = {
            "schema_version": 1,
            "commit_hash": commit_hash,
            "parent_hash": parent_hash,
            "metrics": evaluation.metrics.public_payload(),
            "cases": cases,
        }
        path = self.diagnostics / f"{commit_hash}.json"
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"diagnostics already exist for {commit_hash}")
        _atomic_write_text(path, encoded)
        return path

    def diagnostic_summary(self, commit_hash: str) -> dict[str, object]:
        """Return aggregate parent-child evidence for one private diagnostic record."""
        payload = self._diagnostic_payload(commit_hash)
        parent = self.attempts.get(str(payload["parent_hash"])) if payload["parent_hash"] else None
        return {
            "commit_hash": commit_hash,
            "parent_hash": payload["parent_hash"],
            "child_metrics": payload["metrics"],
            "parent_metrics": None if parent is None else parent["metrics"],
            "case_count": len(payload["cases"]),
            "delta_definition": "child minus parent; negative improves error and positive regresses",
        }

    def diagnose_portfolio(self, commit_hash: str) -> dict[str, object]:
        """Summarize anonymous cases and groups dominating portfolio error."""
        payload = self._diagnostic_payload(commit_hash)
        cases = list(payload["cases"])

        def score(case: Mapping[str, object]) -> float:
            return 0.5 * float(case["child_smae"]) + 0.5 * float(case["child_srmse"])

        def delta(case: Mapping[str, object]) -> float | None:
            if case["delta_smae"] is None or case["delta_srmse"] is None:
                return None
            return 0.5 * float(case["delta_smae"]) + 0.5 * float(case["delta_srmse"])

        total_score = sum(score(case) for case in cases) or 1.0
        groups: dict[tuple[str, int], list[Mapping[str, object]]] = {}
        for case in cases:
            key = (str(case["frequency"]), int(case["horizon"]))
            groups.setdefault(key, []).append(case)
        group_rows = []
        for (frequency, horizon), rows in groups.items():
            deltas = [value for row in rows if (value := delta(row)) is not None]
            group_rows.append({
                "frequency": frequency,
                "horizon": horizon,
                "case_count": len(rows),
                "mean_case_score": statistics.fmean(score(row) for row in rows),
                "portfolio_score_contribution": sum(score(row) for row in rows) / len(cases),
                "mean_parent_delta": None if not deltas else statistics.fmean(deltas),
            })
        group_rows.sort(key=lambda row: float(row["portfolio_score_contribution"]), reverse=True)

        def case_row(case: Mapping[str, object]) -> dict[str, object]:
            case_score = score(case)
            return {
                "case_id": case["case_id"],
                "frequency": case["frequency"],
                "horizon": case["horizon"],
                "case_score": case_score,
                "share_of_total_error": case_score / total_score,
                "parent_delta": delta(case),
                "failure": case["failure"],
            }

        comparable = [case for case in cases if delta(case) is not None]
        return {
            "commit_hash": commit_hash,
            "case_count": len(cases),
            "delta_definition": "child minus parent; negative improves error and positive regresses",
            "highest_error_cases": [case_row(case) for case in sorted(cases, key=score, reverse=True)[:10]],
            "largest_regressions": [
                case_row(case) for case in sorted(comparable, key=lambda row: float(delta(row)), reverse=True)[:10]
            ],
            "largest_improvements": [
                case_row(case) for case in sorted(comparable, key=lambda row: float(delta(row)))[:10]
            ],
            "dominant_frequency_horizon_groups": group_rows[:10],
        }

    def list_results(
        self,
        commit_hash: str,
        *,
        page: int = 1,
        page_size: int = 10,
        order: str = "worst_delta",
        frequency: str | None = None,
    ) -> dict[str, object]:
        """Page through anonymous Train outcomes without exposing dataset identity."""
        if page <= 0 or page_size <= 0:
            raise ValueError("page and page_size must be positive")
        page_size = min(page_size, 20)
        order = {
            "timestamp": "case_id",
            "timestamp_asc": "case_id",
            "timestamp_desc": "case_id_desc",
            "asc": "best_delta",
            "ascending": "best_delta",
            "best": "best_delta",
            "desc": "worst_delta",
            "descending": "worst_delta",
            "worst": "worst_delta",
        }.get(str(order).strip().lower(), order)
        if order not in {"worst_delta", "best_delta", "worst_child", "case_id", "case_id_desc"}:
            raise ValueError(
                "order must be worst_delta, best_delta, worst_child, or case_id"
            )
        if frequency is not None and frequency.lower() in {"", "all", "*"}:
            frequency = None
        payload = self._diagnostic_payload(commit_hash)
        cases = [
            case for case in payload["cases"]
            if frequency is None or str(case["frequency"]) == frequency
        ]

        def delta(case: Mapping[str, object]) -> float:
            if case["delta_smae"] is None or case["delta_srmse"] is None:
                return 0.0
            return 0.5 * float(case["delta_smae"]) + 0.5 * float(case["delta_srmse"])

        if order in {"case_id", "case_id_desc"}:
            cases.sort(
                key=lambda case: str(case["case_id"]), reverse=order == "case_id_desc"
            )
        elif order == "worst_child":
            cases.sort(
                key=lambda case: 0.5 * float(case["child_smae"]) + 0.5 * float(case["child_srmse"]),
                reverse=True,
            )
        else:
            cases.sort(key=delta, reverse=order == "worst_delta")
        start = (page - 1) * page_size
        rows = [{
            "case_id": case["case_id"],
            "frequency": case["frequency"],
            "horizon": case["horizon"],
            "child_smae": case["child_smae"],
            "child_srmse": case["child_srmse"],
            "delta_smae": case["delta_smae"],
            "delta_srmse": case["delta_srmse"],
            "failure": case["failure"],
        } for case in cases[start : start + page_size]]
        pages = math.ceil(len(cases) / page_size)
        return {
            "page": page,
            "page_size": page_size,
            "total": len(cases),
            "total_pages": pages,
            "has_more": page < pages,
            "next_page": page + 1 if page < pages else None,
            "order": order,
            "delta_definition": "child minus parent; negative improves error and positive regresses",
            "rows": rows,
        }

    def read_case(self, commit_hash: str, case_id: str) -> dict[str, object]:
        """Read complete numeric evidence for one anonymous Train case."""
        payload = self._diagnostic_payload(commit_hash)
        for case in payload["cases"]:
            if case["case_id"] == case_id:
                return dict(case)
        raise KeyError(f"unknown diagnostic case: {case_id}")

    def write_note(self, agent_id: str, commit_hash: str, title: str, body: str) -> Path:
        """Write one immutable Qwen-authored reflection note into shared memory."""
        _validate_agent_id(agent_id)
        _validate_commit_hash(commit_hash)
        if self.attempts.get(commit_hash) is None:
            raise ValueError("finding must cite a recorded attempt")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48] or "finding"
        path = self.public / "notes" / agent_id / f"{commit_hash[:12]}-{slug}.md"
        content = (
            "---\n"
            f"agent_id: {json.dumps(agent_id)}\n"
            f"commit_hash: {json.dumps(commit_hash)}\n"
            f"title: {json.dumps(title)}\n"
            "---\n\n"
            f"{body.strip()}\n"
        )
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"note already exists: {path.name}")
        _atomic_write_text(path, content)
        return path

    def write_working_note(
        self,
        agent_id: str,
        heartbeat_sequence: int,
        title: str,
        body: str,
    ) -> Path:
        """Write one immutable identity-free research note before evaluation."""
        _validate_agent_id(agent_id)
        if heartbeat_sequence <= 0 or not title.strip() or not body.strip():
            raise ValueError("working note requires a heartbeat, title, and body")
        if _IDENTITY_REFERENCE.search(title) or _IDENTITY_REFERENCE.search(body):
            raise ValueError("working note contains a forbidden identity reference")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48] or "research"
        content = (
            "---\n"
            f"agent_id: {json.dumps(agent_id)}\n"
            f"heartbeat: {heartbeat_sequence}\n"
            f"title: {json.dumps(title.strip())}\n"
            "---\n\n"
            f"{body.strip()}\n"
        )
        working = self.public / "notes" / agent_id / "working"
        stem = f"{heartbeat_sequence:06d}-{slug}"
        path = working / f"{stem}.md"
        revision = 2
        while path.exists() and path.read_text(encoding="utf-8") != content:
            path = working / f"{stem}-{revision}.md"
            revision += 1
        _atomic_write_text(path, content)
        return path

    def has_note(self, commit_hash: str) -> bool:
        """Return whether any agent recorded a reflection for one commit."""
        _validate_commit_hash(commit_hash)
        return any((self.public / "notes").glob(f"agent-*/*{commit_hash[:12]}-*.md"))

    def has_skill(self, commit_hash: str) -> bool:
        """Return whether an improvement skill exists for one commit."""
        _validate_commit_hash(commit_hash)
        return any((self.public / "skills").glob(f"{commit_hash[:12]}-*/SKILL.md"))

    def reflection_complete(self, record: Mapping[str, object]) -> bool:
        """Check whether one child has its required evidence note."""
        commit_hash = str(record["commit_hash"])
        return record.get("parent_hash") is None or self.has_note(commit_hash)

    def write_skill(self, agent_id: str, commit_hash: str, name: str, body: str) -> Path:
        """Write one immutable improvement-backed forecasting skill."""
        _validate_agent_id(agent_id)
        _validate_commit_hash(commit_hash)
        record = self.attempts.get(commit_hash)
        if record is None or record["parent_hash"] is None:
            raise ValueError("skill must cite an evaluated child")
        parent = self.attempts.get(str(record["parent_hash"]))
        if parent is None or float(record["metrics"]["score"]) >= float(parent["metrics"]["score"]):
            raise ValueError("skill requires strict improvement over the parent")
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "skill"
        directory = self.public / "skills" / f"{commit_hash[:12]}-{slug}"
        path = directory / "SKILL.md"
        content = (
            "---\n"
            f"name: {json.dumps(slug)}\n"
            f"creator: {json.dumps(agent_id)}\n"
            f"commit_hash: {json.dumps(commit_hash)}\n"
            "---\n\n"
            f"{body.strip()}\n"
        )
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"skill already exists: {directory.name}")
        _atomic_write_text(path, content)
        return path

    def memory_catalog(self, kind: str) -> list[dict[str, object]]:
        """List shared notes or skills without loading their bodies."""
        if kind not in {"notes", "skills", "synthesis"}:
            raise ValueError("memory kind must be notes, skills, or synthesis")
        root = self.public / kind
        return [
            {
                "path": str(path.relative_to(self.public)),
                "bytes": path.stat().st_size,
            }
            for path in sorted(root.rglob("*.md"))
            if path.is_file()
        ]

    def read_memory(self, relative_path: str) -> str:
        """Read one guarded public note or skill."""
        candidate = (self.public / relative_path).resolve()
        roots = tuple((self.public / name).resolve() for name in ("notes", "skills", "synthesis"))
        standalone = {
            (self.public / "connections.md").resolve(),
            (self.public / "open-questions.md").resolve(),
        }
        if (
            not any(candidate.is_relative_to(root) for root in roots)
            and candidate not in standalone
        ) or not candidate.is_file():
            raise ValueError("memory path must name public scientific memory")
        return candidate.read_text(encoding="utf-8")

    def stop(self) -> None:
        """Persistently prevent new submissions and grading."""
        self._set_status("stopped")

    def resume(self) -> None:
        """Resume a previously stopped run without resetting its state."""
        self._set_status("running")
        self.recover_grading_leases()

    def agent_progress(self, agent_id: str, window: int = 4) -> dict[str, object]:
        """Compare one agent's recent evaluated scores against its own and the global best."""
        _validate_agent_id(agent_id)
        ranked = self.attempts.parent_candidates()
        by_hash = {str(record["commit_hash"]): record for record in ranked}
        mine = [
            by_hash[str(entry["commit_hash"])]
            for entry in self._read_queue()["entries"]
            if entry["agent_id"] == agent_id and str(entry["commit_hash"]) in by_hash
        ]
        scores = [float(record["metrics"]["score"]) for record in mine]
        personal_best = min(scores) if scores else None
        recent = scores[-window:]
        return {
            "personal_best": personal_best,
            "personal_best_commit": (
                None if not scores else str(min(mine, key=lambda r: float(r["metrics"]["score"]))["commit_hash"])
            ),
            "recent_scores": recent,
            "stagnating": bool(
                personal_best is not None
                and len(recent) >= window
                and all(score > personal_best + 1e-9 for score in recent)
            ),
            "global_best": None if not ranked else float(ranked[0]["metrics"]["score"]),
            "global_best_commit": None if not ranked else str(ranked[0]["commit_hash"]),
        }

    def status(self) -> dict[str, object]:
        """Return compact persistent progress without exposing private queue content."""
        queue = self._read_queue()
        entries = queue["entries"]
        assert isinstance(entries, list)
        count = sum(
            self.attempts.get(str(entry["commit_hash"])) is not None
            for entry in entries
        )
        sessions = self.sessions()
        heartbeat_queue = self._read_heartbeat_queue()
        heartbeat_entries = heartbeat_queue["entries"]
        child_records = [
            record for record in self.attempts.records() if record["parent_hash"] is not None
        ]
        reflected = sum(self.reflection_complete(record) for record in child_records)
        return {
            "status": sessions["status"],
            "evaluation_budget": sessions["evaluation_budget"],
            "evaluation_count": count,
            "queued": sum(entry["status"] == "queued" for entry in entries),
            "running": sum(entry["status"] == "running" for entry in entries),
            "complete": sum(entry["status"] == "complete" for entry in entries),
            "reflections_complete": reflected,
            "reflections_incomplete": len(child_records) - reflected,
            "heartbeat": {
                "configuration": sessions.get("heartbeat"),
                "queued": sum(row["status"] == "queued" for row in heartbeat_entries),
                "running": sum(row["status"] == "running" for row in heartbeat_entries),
                "complete": sum(row["status"] == "complete" for row in heartbeat_entries),
                "failed": sum(row["status"] == "failed" for row in heartbeat_entries),
                "curator_activity": next(
                    (dict(row) for row in heartbeat_entries if row["status"] == "running"),
                    None,
                ),
            },
            "grader_activity": [
                {
                    "grader_id": entry["grader_id"],
                    "sequence": entry["sequence"],
                    "agent_id": entry["agent_id"],
                    "commit_hash": entry["commit_hash"],
                    "started_at": entry["started_at"],
                }
                for entry in entries
                if entry["status"] == "running"
            ],
            "agent_heads": {
                agent_id: state["head"]
                for agent_id, state in sessions["agents"].items()
            },
            "agent_activity": {
                agent_id: {
                    "phase": state.get("phase", "idle"),
                    "outstanding_commit": state.get("outstanding_commit"),
                    "last_error": state.get("last_error"),
                    "last_heartbeat": state.get("last_heartbeat"),
                    "heartbeat_count": state.get("heartbeat_count", 0),
                    "current_hypothesis": state.get("current_hypothesis"),
                }
                for agent_id, state in sessions["agents"].items()
            },
        }

    def show_source(self, commit_hash: str | None = None) -> str:
        """Read a recorded program, defaulting to the best aggregate score."""
        if commit_hash is None:
            candidates = self.attempts.parent_candidates()
            if not candidates:
                raise RuntimeError("no valid recorded program exists")
            commit_hash = str(candidates[0]["commit_hash"])
        _validate_commit_hash(commit_hash)
        if self.attempts.get(commit_hash) is None:
            raise ValueError("program must have a recorded evaluation")
        return _git(self.repo, "show", f"{commit_hash}:{self.PROGRAM_FILE}") + "\n"

    def export_branch(self, branch: str, commit_hash: str | None = None) -> str:
        """Create a local run-repository branch without checking it out."""
        if not branch.strip():
            raise ValueError("export branch must be nonempty")
        _git(self.repo, "check-ref-format", "--branch", branch)
        if commit_hash is None:
            candidates = self.attempts.parent_candidates()
            if not candidates:
                raise RuntimeError("no valid recorded program exists")
            commit_hash = str(candidates[0]["commit_hash"])
        _validate_commit_hash(commit_hash)
        if self.attempts.get(commit_hash) is None:
            raise ValueError("export commit must have a recorded evaluation")
        _git(self.repo, "branch", branch, commit_hash)
        return commit_hash

    def _set_status(self, status: str) -> None:
        with _file_lock(self.state_lock_file):
            sessions = self.sessions()
            sessions["status"] = status
            _atomic_write_json(self.sessions_file, sessions)

    def claim_heartbeat(self) -> dict[str, object] | None:
        """Claim the earliest curator heartbeat job."""
        with _file_lock(self.heartbeat_lock_file):
            queue = self._read_heartbeat_queue()
            entry = next((row for row in queue["entries"] if row["status"] == "queued"), None)
            if entry is None:
                return None
            entry.update(status="running", started_at=time.time())
            _atomic_write_json(self.heartbeat_queue_file, queue)
            return dict(entry)

    def release_heartbeat(self, sequence: int) -> None:
        """Return an interrupted curator job to the queue."""
        with _file_lock(self.heartbeat_lock_file):
            queue = self._read_heartbeat_queue()
            entry = next(row for row in queue["entries"] if int(row["sequence"]) == sequence)
            if entry["status"] == "running":
                entry.update(status="queued", started_at=None)
                _atomic_write_json(self.heartbeat_queue_file, queue)

    def fail_heartbeat(self, sequence: int, error: str) -> None:
        """Record a curator failure without blocking the research run."""
        with _file_lock(self.heartbeat_lock_file):
            queue = self._read_heartbeat_queue()
            entry = next(row for row in queue["entries"] if int(row["sequence"]) == sequence)
            entry.update(status="failed", started_at=None, error=error[-1000:])
            _atomic_write_json(self.heartbeat_queue_file, queue)

    def publish_synthesis(
        self,
        sequence: int,
        synthesis: str,
        connections: str,
        open_questions: str,
    ) -> Path:
        """Atomically publish one curator synthesis and complete its job."""
        for name, value in {
            "synthesis": synthesis,
            "connections": connections,
            "open_questions": open_questions,
        }.items():
            if not value.strip() or _IDENTITY_REFERENCE.search(value):
                raise ValueError(f"{name} must be nonempty and identity-free")
        path = self.public / "synthesis" / f"{sequence:06d}.md"
        _atomic_write_text(path, f"# Heartbeat synthesis {sequence}\n\n{synthesis.strip()}\n")
        _atomic_write_text(self.public / "connections.md", f"# Connections\n\n{connections.strip()}\n")
        _atomic_write_text(self.public / "open-questions.md", f"# Open questions\n\n{open_questions.strip()}\n")
        with _file_lock(self.heartbeat_lock_file):
            queue = self._read_heartbeat_queue()
            entry = next(row for row in queue["entries"] if int(row["sequence"]) == sequence)
            if entry["status"] != "running":
                raise RuntimeError("curator no longer owns the heartbeat job")
            entry.update(status="complete", started_at=None)
            _atomic_write_json(self.heartbeat_queue_file, queue)
        return path

    def recover_heartbeat_leases(self) -> int:
        """Return abandoned curator jobs to the queue."""
        with _file_lock(self.heartbeat_lock_file):
            queue = self._read_heartbeat_queue()
            recovered = 0
            for entry in queue["entries"]:
                if entry["status"] == "running":
                    entry.update(status="queued", started_at=None)
                    recovered += 1
            if recovered:
                _atomic_write_json(self.heartbeat_queue_file, queue)
            return recovered

    def _read_heartbeat_queue(self) -> dict[str, object]:
        if not self.heartbeat_queue_file.is_file():
            return {"schema_version": 1, "next_sequence": 1, "entries": []}
        return json.loads(self.heartbeat_queue_file.read_text(encoding="utf-8"))

    def _read_queue(self) -> dict[str, object]:
        queue = json.loads(self.queue_file.read_text(encoding="utf-8"))
        if int(queue.get("schema_version", 1)) == 1:
            queue["schema_version"] = 2
        for entry in queue["entries"]:
            entry.setdefault("grader_id", None)
            entry.setdefault("started_at", None)
        return queue

    def _diagnostic_payload(self, commit_hash: str) -> dict[str, object]:
        _validate_commit_hash(commit_hash)
        path = self.diagnostics / f"{commit_hash}.json"
        if not path.is_file():
            raise ValueError(f"diagnostics are missing for {commit_hash}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _agent_state(self, agent_id: str) -> dict[str, object]:
        _validate_agent_id(agent_id)
        agents = self.sessions()["agents"]
        if agent_id not in agents:
            raise KeyError(f"unknown agent: {agent_id}")
        return agents[agent_id]

    def _agent_worktree(self, agent_id: str) -> Path:
        state = self._agent_state(agent_id)
        return self.root / str(state["worktree"])

    def _update_agent_head(self, agent_id: str, head: str) -> None:
        with _file_lock(self.state_lock_file):
            sessions = self.sessions()
            sessions["agents"][agent_id]["head"] = head
            _atomic_write_json(self.sessions_file, sessions)

    def _sync_eval_count(self, entries: Sequence[Mapping[str, object]]) -> int:
        count = sum(
            self.attempts.get(str(entry["commit_hash"])) is not None
            for entry in entries
        )
        _atomic_write_text(self.eval_count_file, f"{count}\n")
        return count


@dataclass(frozen=True)
class BackendReply:
    """Text and optional resumable session ID returned by a model runtime."""

    text: str
    session_id: str | None = None


@dataclass(frozen=True)
class ParentChoice:
    """A model-selected recorded parent and falsifiable editing hypothesis."""

    commit_hash: str
    hypothesis: str
    session_id: str | None


@dataclass(frozen=True)
class EditChoice:
    """A completed edit with optional replacement source and session ID."""

    source: str | None
    session_id: str | None


@dataclass(frozen=True)
class EvolutionProposal:
    """The host-committed candidate placed into the grading queue."""

    parent_hash: str
    commit_hash: str
    hypothesis: str
    queue_sequence: int


SessionRuntime = Callable[[str, Path, str | None, bool], BackendReply]


def _candidate_summary(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "commit_hash": record["commit_hash"],
            "parent_hash": record.get("parent_hash"),
            "hypothesis": record.get("hypothesis", ""),
            "score": record["metrics"]["score"],
            "mean_smae": record["metrics"]["mean_smae"],
            "mean_srmse": record["metrics"]["mean_srmse"],
            "coverage": record["metrics"]["coverage"],
            "status": record["status"],
        }
        for record in records
    ]


def _program_functions(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _replace_function(source: str, name: str, replacement: str) -> str:
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    parsed = ast.parse(replacement)
    if len(matches) != 1 or len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef):
        raise ValueError("replace_function requires one existing function and one function replacement")
    if parsed.body[0].name != name:
        raise ValueError("replacement function name must match the requested name")
    lines = source.splitlines(keepends=True)
    node = matches[0]
    candidate = "".join(lines[: node.lineno - 1]) + replacement.rstrip() + "\n\n" + "".join(lines[node.end_lineno :])
    validate_program_source(candidate)
    return candidate


def _insert_function(source: str, function_source: str, before: str = "forecast") -> str:
    parsed = ast.parse(function_source)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef):
        raise ValueError("insert_function requires exactly one function definition")
    name = parsed.body[0].name
    functions = _program_functions(source)
    if name in functions:
        raise ValueError("inserted function name already exists")
    target = next(
        (node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == before),
        None,
    )
    if target is None:
        raise ValueError("before must name an existing function")
    lines = source.splitlines(keepends=True)
    candidate = "".join(lines[: target.lineno - 1]) + function_source.rstrip() + "\n\n" + "".join(lines[target.lineno - 1 :])
    validate_program_source(candidate)
    return candidate


def _register_skill(source: str, name: str, function_name: str) -> str:
    if not name.strip() or not name.replace("_", "").isalnum():
        raise ValueError("skill name must be alphanumeric with optional underscores")
    if function_name not in _program_functions(source):
        raise ValueError("registered skill function must exist")
    tree = ast.parse(source)
    assignment = None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Dict):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == "SKILLS" for target in targets):
            assignment = node
            break
    if assignment is None:
        raise ValueError("program has no literal SKILLS registry")
    if any(isinstance(key, ast.Constant) and key.value == name for key in assignment.value.keys):
        raise ValueError("skill name is already registered")
    lines = source.splitlines(keepends=True)
    closing = assignment.end_lineno - 1
    indent = " " * 4
    candidate = "".join(lines[:closing]) + f'{indent}{json.dumps(name)}: {function_name},\n' + "".join(lines[closing:])
    validate_program_source(candidate)
    return candidate


def _markdown_items(items: Sequence[str]) -> str:
    return "None observed." if not items else "\n".join(f"- {item.strip()}" for item in items)


def _reflection_text(value: object, field: str, *, numeric: bool = False) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if numeric and not isinstance(value, bool) and isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return str(value)
    if isinstance(value, list) and value:
        items = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{field} list entries must be nonempty strings")
            items.append(item.strip())
        return "\n".join(f"- {item}" for item in items)
    expected = "a nonempty string, string array"
    if numeric:
        expected += ", or finite number"
    raise ValueError(f"{field} must be {expected}")


def _trajectory_items(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    items = []
    for index, item in enumerate(value):
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"{field}[{index}] must be a string or object")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{field}[{index}].case_id must be a nonempty string")
        observations = {str(key): value for key, value in item.items() if key != "case_id"}
        if not observations:
            raise ValueError(f"{field}[{index}] must include an observation")
        items.append(f"{case_id.strip()}: {json.dumps(observations, sort_keys=True)}")
    return items


def _validate_reflection_evidence(
    field: str,
    text: str,
    allowed_case_ids: set[str],
    inspected_case_ids: set[str],
) -> None:
    if _IDENTITY_REFERENCE.search(text):
        raise ValueError(f"{field} contains a forbidden task or entity identity reference")
    references = set(_CASE_REFERENCE.findall(text))
    invalid = sorted(references - allowed_case_ids)
    if invalid:
        raise ValueError(f"{field} references unknown anonymous cases: {', '.join(invalid)}")
    uninspected = sorted(references - inspected_case_ids)
    if uninspected:
        raise ValueError(f"{field} cites uninspected anonymous cases: {', '.join(uninspected)}")


def _leaderboard_prompt(records: Sequence[Mapping[str, object]], limit: int = 5) -> str:
    """Render the best recorded attempts so an agent can see beyond its own lineage."""
    rows = [
        f"  {rank}. {float(record['metrics']['score']):.5f}  {record['agent_id']}  {record['commit_hash']}"
        for rank, record in enumerate(records[:limit], start=1)
    ]
    return "\n".join(rows) if rows else "  (no evaluated attempt yet)"


def _program_shape(source: str) -> str:
    """Return a formatting-insensitive fingerprint of a program's behaviour."""
    return ast.dump(ast.parse(source))


_ACTION_PARAMETER_KEYS = ("params", "args", "arguments", "parameters")
_MINIMUM_COMMIT_PREFIX = 7


def _resolve_commit(reference: str, eligible: Collection[str]) -> str:
    """Expand an abbreviated commit reference to the one eligible hash it names."""
    reference = reference.strip().lower()
    if reference in eligible:
        return reference
    known = sorted(eligible)
    if len(reference) >= _MINIMUM_COMMIT_PREFIX and re.fullmatch(r"[0-9a-f]+", reference):
        matches = [commit_hash for commit_hash in known if commit_hash.startswith(reference)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError(
                f"commit {reference} is ambiguous; eligible: {', '.join(matches)}"
            )
    raise ValueError(
        f"commit {reference or '<missing>'} is not an eligible recorded program; "
        f"eligible: {', '.join(known) or 'none'}"
    )


def _action_payload(raw: str) -> dict[str, object]:
    """Flatten one action object, accepting any spelling of its nested parameter key."""
    payload = parse_json_object(raw)
    batch = payload.get("actions")
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        if len(batch) != 1 or not isinstance(batch[0], Mapping):
            raise ValueError("return exactly one action object per turn, not a list of actions")
        payload = dict(batch[0])
    for key in _ACTION_PARAMETER_KEYS:
        nested = payload.get(key)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise ValueError(f"action {key} must be an object")
        payload = {**nested, **{name: value for name, value in payload.items() if name != key}}
    return payload


def _parent_prompt(records: Sequence[Mapping[str, object]]) -> str:
    candidates = json.dumps(_candidate_summary(records), indent=2, sort_keys=True)
    return (
        "Choose one recorded parent for the next forecasting-program experiment. "
        "Use only aggregate evidence. Return only JSON with keys parent_commit and hypothesis. "
        "The hypothesis must describe one concrete, testable code change. Do not edit files yet.\n\n"
        f"CANDIDATES\n{candidates}"
    )


def _parse_parent(reply: BackendReply) -> ParentChoice:
    payload = parse_json_object(reply.text)
    commit_hash = str(payload.get("parent_commit", ""))
    hypothesis = str(payload.get("hypothesis", "")).strip()
    _validate_commit_hash(commit_hash)
    if not hypothesis:
        raise ValueError("backend returned an empty hypothesis")
    return ParentChoice(commit_hash, hypothesis, reply.session_id)


class QwenProgramAdapter:
    """Run guarded Qwen planning, editing, and scientific reflection loops."""

    backend = "qwen"
    edits_worktree = True

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        temperature: float = 0.3,
        max_actions: int = 24,
        heartbeat_idle_seconds: float = 300.0,
        min_actions_before_yield: int = 4,
        submit_after_heartbeats: int = 4,
        curator_model: str | None = None,
        curator_client: Any | None = None,
    ) -> None:
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be a finite nonnegative number")
        if isinstance(max_actions, bool) or max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if (
            isinstance(heartbeat_idle_seconds, bool)
            or not math.isfinite(heartbeat_idle_seconds)
            or heartbeat_idle_seconds <= 0
        ):
            raise ValueError("heartbeat_idle_seconds must be positive and finite")
        if isinstance(min_actions_before_yield, bool) or min_actions_before_yield <= 0:
            raise ValueError("min_actions_before_yield must be positive")
        if isinstance(submit_after_heartbeats, bool) or submit_after_heartbeats <= 0:
            raise ValueError("submit_after_heartbeats must be positive")
        self.model = model
        self.temperature = float(temperature)
        self.max_actions = int(max_actions)
        self.heartbeat_idle_seconds = float(heartbeat_idle_seconds)
        self.min_actions_before_yield = int(min_actions_before_yield)
        self.submit_after_heartbeats = int(submit_after_heartbeats)
        self.client = client or QwenClient(model_id=model)
        self.curator_model = curator_model or model
        self.curator_client = curator_client or self.client

    def _validated_candidate(
        self,
        run: EvolutionRun,
        worktree: Path,
        current_records: Sequence[Mapping[str, object]],
    ) -> str:
        """Return the edited worktree program once it is valid and not already recorded."""
        candidate_source = (worktree / run.PROGRAM_FILE).read_text(encoding="utf-8")
        validate_program_source(candidate_source)
        candidate_shape = _program_shape(candidate_source)
        for record in current_records:
            recorded = run.show_source(str(record["commit_hash"]))
            if candidate_source == recorded or candidate_shape == _program_shape(recorded):
                raise ValueError(
                    "candidate is behaviourally identical to a recorded program; "
                    "grading it would spend the budget on no change"
                )
        return candidate_source

    def plan_and_edit(
        self,
        run: EvolutionRun,
        agent_id: str,
        records: Sequence[Mapping[str, object]],
    ) -> ParentChoice:
        """Research and edit autonomously across resumable heartbeat contexts."""
        worktree = run._agent_worktree(agent_id)
        if run.latest_heartbeat(agent_id) is None:
            run.record_heartbeat(agent_id, "initial", {})
        barren_episodes = 0
        while True:
            if barren_episodes:
                time.sleep(min(2.0 ** barren_episodes, self.heartbeat_idle_seconds))
            current_records = run.attempts.parent_candidates()
            eligible = {str(record["commit_hash"]) for record in current_records}
            checkpoint = run.agent_checkpoint(agent_id)
            parent_hash = checkpoint.get("parent_hash")
            choice = None
            if isinstance(parent_hash, str) and parent_hash in eligible:
                choice = ParentChoice(parent_hash, str(checkpoint.get("hypothesis", "")), None)
            edited = bool(checkpoint.get("edited", False))
            edit_heartbeats = int(checkpoint.get("edit_heartbeats", 0)) if edited else 0
            inspected_cases = set(map(str, checkpoint.get("inspected_cases", [])))
            heartbeat = run.latest_heartbeat(agent_id)
            pending = ""
            if edited and edit_heartbeats:
                pending = (
                    f"\nPENDING EDIT: your worktree has held an unsubmitted program edit for "
                    f"{edit_heartbeats} heartbeat(s). An edit earns evidence only once the evaluator "
                    "grades it. Finish it now, or abandon_hypothesis if you no longer believe it. "
                    "Further research on an unsubmitted edit produces nothing."
                )
            progress = run.agent_progress(agent_id)
            redirect = ""
            if progress["stagnating"] and not edited:
                recent = ", ".join(f"{score:.5f}" for score in progress["recent_scores"])
                redirect = (
                    f"\nREDIRECT: your last evaluated candidates ({recent}) all scored worse than "
                    f"your own best ({progress['personal_best']:.5f}, "
                    f"{str(progress['personal_best_commit'])[:12]}). Further variations on the current "
                    "direction are not paying off. Reassess: adopt a different parent, or pursue a "
                    "structurally different hypothesis such as a missing forecasting skill."
                )
            initial_message = {
                "role": "user",
                "content": (
                    "HEARTBEAT: continue autonomous scientific evolution of program.py. Explore before "
                    "committing when useful, but choose your own research direction. Possible directions "
                    "include routing, trajectory classification, conditional ensembles, hindcast design, "
                    "new forecasting helpers, parameter studies, and diagnosing dominant errors; these are "
                    "examples, not assignments. You may write a working note, abandon a weak hypothesis, "
                    "or checkpoint without submitting. Return one JSON object per turn.\n"
                    "Actions: list_attempts; list_memory(kind); read_memory(path); summary_results(commit_hash); "
                    "diagnose_portfolio(commit_hash); list_results(commit_hash,page,page_size,order,frequency); "
                    "read_case(commit_hash,case_id); show_program(commit_hash); "
                    "write_working_note(title,body); select_parent(commit_hash,hypothesis); read_program; "
                    "replace_text(old,new); replace_function(name,source); insert_function(source,before); "
                    "register_skill(name,function_name); checkpoint(summary,next_actions); "
                    "abandon_hypothesis(reason); finish. Use finish only after a valid program edit. "
                    "Negative deltas improve error; positive deltas regress. Memory kinds are notes, skills, "
                    "and synthesis. Do not request identities, textual context, Test data, shell access, or "
                    "whole-file replacement. The evaluator—not a fixed strategy scheduler—provides selection "
                    "pressure. Avoid repeating saturated tweaks unless evidence makes them worthwhile.\n"
                    f"{pending}{redirect}\n"
                    "LEADERBOARD (best first; lower is better). Any of these is an eligible parent — "
                    "you are not restricted to your own lineage. Inspect one with "
                    "show_program(commit_hash) and adopt it with select_parent(commit_hash,hypothesis).\n"
                    f"{_leaderboard_prompt(current_records)}\n"
                    f"HEARTBEAT EVENT\n{json.dumps(heartbeat, sort_keys=True)}\n"
                    f"PRIVATE CHECKPOINT\n{json.dumps(checkpoint, sort_keys=True)}"
                ),
            }
            if (
                edited
                and choice is not None
                and edit_heartbeats >= self.submit_after_heartbeats
            ):
                try:
                    self._validated_candidate(run, worktree, current_records)
                except Exception as exc:
                    self._log_failure(run, agent_id, "research", 0, "<stranded edit>", exc)
                else:
                    run.save_agent_checkpoint(agent_id, {})
                    return choice
            messages: list[dict[str, str]] = [initial_message]
            successful_actions = list(map(str, checkpoint.get("successful_actions", [])))[-24:]
            consecutive_failures = 0
            episode_actions = 0
            memory_stamp = run.shared_memory_stamp()
            trigger = "action_budget"
            for action_number in range(1, self.max_actions + 1):
                raw = "<model call failed>"
                failed = False
                try:
                    raw = self._complete(messages)
                    action = _action_payload(raw)
                    name = str(action.get("action", ""))
                    if name == "list_attempts":
                        result: object = _candidate_summary(current_records)
                    elif name == "list_memory":
                        result = run.memory_catalog(str(action.get("kind", "")))
                    elif name == "read_memory":
                        result = {"content": run.read_memory(str(action.get("path", "")))}
                    elif name in {"summary_results", "diagnose_portfolio", "list_results", "read_case"}:
                        commit_hash = _resolve_commit(str(action.get("commit_hash", "")), eligible)
                        if name == "summary_results":
                            result = run.diagnostic_summary(commit_hash)
                        elif name == "diagnose_portfolio":
                            result = run.diagnose_portfolio(commit_hash)
                        elif name == "read_case":
                            result = run.read_case(commit_hash, str(action.get("case_id", "")))
                            inspected_cases.add(str(result["case_id"]))
                        else:
                            frequency = action.get("frequency")
                            result = run.list_results(
                                commit_hash,
                                page=int(action.get("page", 1)),
                                page_size=int(action.get("page_size", 10)),
                                order=str(action.get("order", "worst_delta")),
                                frequency=None if frequency is None else str(frequency),
                            )
                            inspected_cases.update(str(row["case_id"]) for row in result["rows"])
                    elif name == "show_program":
                        commit_hash = _resolve_commit(str(action.get("commit_hash", "")), eligible)
                        result = {"commit_hash": commit_hash, "source": run.show_source(commit_hash)}
                    elif name == "write_working_note":
                        if heartbeat is None:
                            raise RuntimeError("working note requires an active heartbeat")
                        path = run.write_working_note(
                            agent_id,
                            int(heartbeat["sequence"]),
                            str(action.get("title", "")),
                            str(action.get("body", "")),
                        )
                        result = {"path": str(path.relative_to(run.public))}
                        memory_stamp = run.shared_memory_stamp()
                    elif name == "select_parent":
                        if edited:
                            raise ValueError("abandon or finish the current edit before changing parent")
                        commit_hash = _resolve_commit(str(action.get("commit_hash", "")), eligible)
                        hypothesis = str(action.get("hypothesis", "")).strip()
                        if not hypothesis:
                            raise ValueError("select_parent requires a nonempty hypothesis")
                        run.checkout_parent(agent_id, commit_hash)
                        choice = ParentChoice(commit_hash, hypothesis, None)
                        result = {"selected": commit_hash, "hypothesis": hypothesis}
                    elif name == "read_program":
                        if choice is None:
                            raise ValueError("select a parent before reading the editable program")
                        result = {"source": (worktree / run.PROGRAM_FILE).read_text(encoding="utf-8")}
                    elif name in {"replace_text", "replace_function", "insert_function", "register_skill"}:
                        if choice is None:
                            raise ValueError("select a parent before editing")
                        path = worktree / run.PROGRAM_FILE
                        source = path.read_text(encoding="utf-8")
                        if name == "replace_text":
                            old, new = action.get("old"), action.get("new")
                            if not isinstance(old, str) or not isinstance(new, str) or not old or old == new:
                                raise ValueError("replace_text requires distinct nonempty old and new strings")
                            if source.count(old) != 1 or old.strip() == source.strip():
                                raise ValueError("old text must occur once and cannot be the whole program")
                            candidate = source.replace(old, new, 1)
                        elif name == "replace_function":
                            candidate = _replace_function(source, str(action.get("name", "")), str(action.get("source", "")))
                        elif name == "insert_function":
                            candidate = _insert_function(
                                source,
                                str(action.get("source", "")),
                                str(action.get("before", "forecast")),
                            )
                        else:
                            candidate = _register_skill(
                                source,
                                str(action.get("name", "")),
                                str(action.get("function_name", "")),
                            )
                        validate_program_source(candidate)
                        _atomic_write_text(path, candidate)
                        edited = True
                        result = {"edited": True, "source_bytes": len(candidate.encode("utf-8"))}
                    elif name == "checkpoint":
                        checkpoint["research_summary"] = _reflection_text(action.get("summary"), "summary")
                        checkpoint["next_actions"] = _reflection_text(action.get("next_actions"), "next_actions")
                        result = {"checkpointed": True}
                        trigger = "model_checkpoint"
                        successful_actions.append(name)
                        episode_actions += 1
                        break
                    elif name == "abandon_hypothesis":
                        if edited:
                            _git(worktree, "restore", run.PROGRAM_FILE)
                        choice = None
                        edited = False
                        checkpoint = {"abandoned_reason": str(action.get("reason", "")).strip()}
                        result = {"abandoned": True}
                        trigger = "hypothesis_abandoned"
                        successful_actions.append(name)
                        episode_actions += 1
                        break
                    elif name == "finish":
                        if choice is None or not edited:
                            raise ValueError("finish requires a selected parent and a valid edit")
                        self._validated_candidate(run, worktree, current_records)
                        run.save_agent_checkpoint(agent_id, {})
                        return choice
                    else:
                        raise ValueError(f"unsupported research action: {name or '<missing>'}")
                    successful_actions.append(name)
                    consecutive_failures = 0
                    episode_actions += 1
                except Exception as exc:
                    failed = True
                    consecutive_failures += 1
                    self._log_failure(run, agent_id, "research", action_number, raw, exc)
                    result = {"error": str(exc), "actions_remaining": self.max_actions - action_number}
                    if isinstance(exc, VLLMRequestError) and exc.context_length_exceeded:
                        trigger = "context_recovery"
                    elif consecutive_failures >= 3:
                        trigger = "malformed_actions"
                    else:
                        self._add_tool_result(messages, None, result)
                        continue
                    break
                self._add_tool_result(messages, None if failed else raw, result)
                current_stamp = run.shared_memory_stamp()
                if current_stamp != memory_stamp:
                    memory_stamp = current_stamp
                    if edited or episode_actions < self.min_actions_before_yield:
                        self._add_notice(messages, {
                            "shared_memory_changed": True,
                            "guidance": (
                                "Peers published new attempts, notes, or skills. Read them when they "
                                "bear on your hypothesis; otherwise continue the work in progress."
                            ),
                        })
                    else:
                        trigger = "shared_memory_changed"
                        break
            checkpoint.update({
                "parent_hash": None if choice is None else choice.commit_hash,
                "hypothesis": None if choice is None else choice.hypothesis,
                "edited": edited,
                "edit_heartbeats": edit_heartbeats + 1 if edited else 0,
                "inspected_cases": sorted(inspected_cases),
                "successful_actions": successful_actions[-24:],
                "redirected": bool(redirect),
            })
            barren_episodes = 0 if episode_actions else barren_episodes + 1
            if episode_actions and trigger != "model_checkpoint":
                checkpoint.update(self._summarize_checkpoint(run, agent_id, checkpoint, messages))
            run.record_heartbeat(agent_id, trigger, checkpoint)

    def _summarize_checkpoint(
        self,
        run: EvolutionRun,
        agent_id: str,
        checkpoint: Mapping[str, object],
        messages: Sequence[Mapping[str, str]],
    ) -> dict[str, str]:
        """Ask Qwen for a compact scientific checkpoint at an action boundary."""
        recent = "\n".join(str(message.get("content", "")) for message in messages[-8:])[-12000:]
        prompt = [{
            "role": "user",
            "content": (
                "Compact this research context for the next heartbeat. Return only JSON with nonempty "
                "research_summary and next_actions. Preserve evidence, rejected hypotheses, uncertainty, "
                "and the current editing intention; do not invent identities.\n"
                f"HOST STATE\n{json.dumps(dict(checkpoint), sort_keys=True)}\nRECENT CONTEXT\n{recent}"
            ),
        }]
        raw = "<checkpoint summary failed>"
        try:
            raw = self._complete(prompt)
            payload = parse_json_object(raw)
            return {
                "research_summary": _reflection_text(payload.get("research_summary"), "research_summary"),
                "next_actions": _reflection_text(payload.get("next_actions"), "next_actions"),
            }
        except Exception as exc:
            self._log_failure(run, agent_id, "checkpoint", 1, raw, exc)
            return {}

    def reflect(
        self,
        run: EvolutionRun,
        agent_id: str,
        record: Mapping[str, object],
    ) -> bool:
        """Analyze one evaluation and write a structured note and optional skill."""
        commit_hash = str(record["commit_hash"])
        parent_hash = record.get("parent_hash")
        parent = None if parent_hash is None else run.attempts.get(str(parent_hash))
        improved = parent is not None and (
            float(record["metrics"]["score"]) < float(parent["metrics"]["score"])
        )
        initial_message = {
            "role": "user",
            "content": (
                f"Reflect scientifically on evaluated commit {commit_hash}. The strict parent-relative "
                f"improvement flag is {str(improved).lower()}. Inspect aggregate and anonymous case "
                "evidence before saving memory. Deltas are child minus parent: negative improves error "
                "and positive regresses. Return exactly one JSON object per turn.\n"
                "Actions: summary_results; diagnose_portfolio; list_results(page,page_size,order,frequency); "
                "read_case(case_id); list_memory(kind); read_memory(path); save_reflection.\n"
                "For list_results, page starts at 1, page_size is at most 20, order is one of "
                "worst_delta, best_delta, worst_child, or case_id, and frequency may be 'all'. "
                "Arguments may be top-level or inside a params, args, or arguments object.\n"
                "save_reflection requires title, result_summary, evidence, explanation, and next_question. "
                "Evidence must cite only inspected anonymous cases. An optional reusable skill may contain "
                "name, technique, evidence, applicability, failure_conditions, and implementation_guidance; "
                "skills are accepted only after strict parent-relative improvement. Do not infer identities.\n"
                "Exact save example: {\"action\":\"save_reflection\",\"title\":\"Drift experiment\","
                "\"result_summary\":\"Score improved from 1.00 to 0.95.\","
                "\"evidence\":\"case_001 improved while case_002 regressed.\","
                "\"explanation\":\"The change helped stable drift but hurt reversals.\","
                "\"next_question\":\"Can a causal reversal test gate the change?\"}"
            ),
        }
        messages: list[dict[str, str]] = [initial_message]
        diagnostic = run._diagnostic_payload(commit_hash)
        allowed_case_ids = {str(case["case_id"]) for case in diagnostic["cases"]}
        inspected_case_ids: set[str] = set()
        evidence_inspected = False
        successful_actions: list[str] = []
        for action_number in range(1, self.max_actions + 1):
            raw = "<model call failed>"
            failed = False
            try:
                raw = self._complete(messages)
                action = _action_payload(raw)
                name = str(action.get("action", ""))
                if name == "summary_results":
                    result: object = run.diagnostic_summary(commit_hash)
                    evidence_inspected = True
                elif name == "diagnose_portfolio":
                    result = run.diagnose_portfolio(commit_hash)
                    for field in (
                        "highest_error_cases", "largest_regressions", "largest_improvements"
                    ):
                        inspected_case_ids.update(
                            str(row["case_id"]) for row in result[field]
                        )
                    evidence_inspected = True
                elif name == "list_results":
                    frequency = action.get("frequency")
                    result = run.list_results(
                        commit_hash,
                        page=int(action.get("page", 1)),
                        page_size=int(action.get("page_size", 10)),
                        order=str(action.get("order", "worst_delta")),
                        frequency=None if frequency is None else str(frequency),
                    )
                    inspected_case_ids.update(str(row["case_id"]) for row in result["rows"])
                    evidence_inspected = True
                elif name == "read_case":
                    result = run.read_case(commit_hash, str(action.get("case_id", "")))
                    inspected_case_ids.add(str(result["case_id"]))
                    evidence_inspected = True
                elif name == "list_memory":
                    result = run.memory_catalog(str(action.get("kind", "")))
                elif name == "read_memory":
                    result = {"content": run.read_memory(str(action.get("path", "")))}
                elif name == "save_reflection":
                    if not evidence_inspected:
                        raise ValueError("save_reflection requires inspected evaluation evidence")
                    note_title, note_body, skill = self._reflection_artifacts(
                        action,
                        improved,
                        allowed_case_ids,
                        inspected_case_ids,
                    )
                    if not run.has_note(commit_hash):
                        run.write_note(agent_id, commit_hash, note_title, note_body)
                    if skill is not None and not run.has_skill(commit_hash):
                        run.write_skill(agent_id, commit_hash, skill[0], skill[1])
                    return run.reflection_complete(record)
                else:
                    raise ValueError(f"unsupported reflection action: {name or '<missing>'}")
                successful_actions.append(name)
            except Exception as exc:
                failed = True
                self._log_failure(run, agent_id, "reflection", action_number, raw, exc)
                if isinstance(exc, VLLMRequestError) and exc.context_length_exceeded:
                    run.record_heartbeat(
                        agent_id,
                        "context_recovery",
                        {
                            "last_commit": commit_hash,
                            "evidence_note_pending": True,
                            "inspected_cases": sorted(inspected_case_ids),
                        },
                    )
                    return False
                error = str(exc)
                result = {"error": error, "actions_remaining": self.max_actions - action_number}
            self._add_tool_result(messages, None if failed else raw, result)
        return False

    def curate(self, run: EvolutionRun, event: Mapping[str, object]) -> bool:
        """Consolidate shared scientific memory for one heartbeat."""
        sequence = int(event["sequence"])
        messages: list[dict[str, str]] = [{
            "role": "user",
            "content": (
                "You are the asynchronous scientific-memory curator. Process this heartbeat without "
                "choosing work for agents. Read relevant public attempts, notes, skills, synthesis, "
                "connections, and open questions, then consolidate agreements, contradictions, and gaps. "
                "Return one JSON object per turn. Actions: list_attempts; list_memory(kind); "
                "read_memory(path); save_synthesis(synthesis,connections,open_questions). "
                "Memory kinds are notes, skills, and synthesis. Do not request private diagnostics, task "
                "identities, context, Test data, program edits, or evaluations.\n"
                f"HEARTBEAT EVENT\n{json.dumps(dict(event), sort_keys=True)}"
            ),
        }]
        for action_number in range(1, self.max_actions + 1):
            raw = "<model call failed>"
            try:
                raw = self._complete(messages, curator=True)
                action = _action_payload(raw)
                name = str(action.get("action", ""))
                if name == "list_attempts":
                    result: object = _candidate_summary(run.attempts.records())
                elif name == "list_memory":
                    result = run.memory_catalog(str(action.get("kind", "")))
                elif name == "read_memory":
                    result = {"content": run.read_memory(str(action.get("path", "")))}
                elif name == "save_synthesis":
                    run.publish_synthesis(
                        sequence,
                        _reflection_text(action.get("synthesis"), "synthesis"),
                        _reflection_text(action.get("connections"), "connections"),
                        _reflection_text(action.get("open_questions"), "open_questions"),
                    )
                    return True
                else:
                    raise ValueError(f"unsupported curator action: {name or '<missing>'}")
            except Exception as exc:
                self._log_failure(run, "curator", "curation", action_number, raw, exc)
                result = {"error": str(exc), "actions_remaining": self.max_actions - action_number}
                self._add_tool_result(messages, None, result)
                continue
            self._add_tool_result(messages, raw, result)
        return False

    def _complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        curator: bool = False,
    ) -> str:
        client = self.curator_client if curator else self.client
        arguments: dict[str, object] = {
            "system": "You are a careful scientific forecasting-program evolution agent.",
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if isinstance(client, QwenClient):
            arguments.update(enable_thinking=False, max_new_tokens=4096)
        response = client.complete(**arguments)
        return str(response.text)

    @staticmethod
    def _add_tool_result(
        messages: list[dict[str, str]], raw: str | None, result: object
    ) -> None:
        messages.append({
            "role": "assistant",
            "content": raw if raw is not None else "(unusable response withheld)",
        })
        messages.append({
            "role": "user",
            "content": "TOOL RESULT\n" + json.dumps(result, sort_keys=True),
        })

    @staticmethod
    def _add_notice(messages: list[dict[str, str]], notice: object) -> None:
        """Append host information to the transcript without faking a model turn."""
        messages.append({
            "role": "user",
            "content": "HOST NOTICE\n" + json.dumps(notice, sort_keys=True),
        })

    @staticmethod
    def _log_failure(
        run: EvolutionRun,
        agent_id: str,
        phase: str,
        action_number: int,
        raw: str,
        error: Exception,
    ) -> None:
        path = run.public / "logs" / f"{agent_id}-{phase}.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"action={action_number} error={type(error).__name__}: {error}\n"
                f"raw={raw}\n\n"
            )

    @staticmethod
    def _reflection_artifacts(
        action: Mapping[str, object],
        improved: bool,
        allowed_case_ids: set[str],
        inspected_case_ids: set[str],
    ) -> tuple[str, str, tuple[str, str] | None]:
        title = action.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a nonempty string")
        _validate_reflection_evidence(
            "title", title, allowed_case_ids, inspected_case_ids
        )
        concise_fields = ("result_summary", "evidence", "explanation", "next_question")
        if all(action.get(field) is not None for field in concise_fields):
            concise = {
                field: _reflection_text(action.get(field), field) for field in concise_fields
            }
            for field, value in concise.items():
                _validate_reflection_evidence(field, value, allowed_case_ids, inspected_case_ids)
            body = (
                f"## Outcome\n\n{concise['result_summary']}\n\n"
                f"## Evidence\n\n{concise['evidence']}\n\n"
                f"## Explanation\n\n{concise['explanation']}\n\n"
                f"## Next question\n\n{concise['next_question']}"
            )
            raw_skill = action.get("skill")
            if raw_skill is None or not improved:
                return title.strip(), body, None
            return title.strip(), body, QwenProgramAdapter._skill_artifact(
                raw_skill, allowed_case_ids, inspected_case_ids
            )
        fields = (
            "result_summary", "mechanism", "surprises", "failure_modes",
            "confidence", "limitations", "next_experiment",
        )
        values = {
            field: _reflection_text(action.get(field), field, numeric=field == "confidence")
            for field in fields
        }
        trajectory_fields = ("improved_trajectories", "regressed_trajectories")
        trajectories = {
            field: _trajectory_items(action.get(field), field) for field in trajectory_fields
        }
        for field, text in {**values, **{
            key: "\n".join(value) for key, value in trajectories.items()
        }}.items():
            _validate_reflection_evidence(
                field, text, allowed_case_ids, inspected_case_ids
            )
        body = (
            f"## Change and measured outcome\n\n{values['result_summary'].strip()}\n\n"
            "## Trajectory evidence\n\n"
            f"Improved: {_markdown_items(trajectories['improved_trajectories'])}\n\n"
            f"Regressed: {_markdown_items(trajectories['regressed_trajectories'])}\n\n"
            f"## Likely mechanism\n\n{values['mechanism'].strip()}\n\n"
            f"## Surprises\n\n{values['surprises'].strip()}\n\n"
            f"## Failure modes\n\n{values['failure_modes'].strip()}\n\n"
            f"## Confidence and limitations\n\n{values['confidence'].strip()} "
            f"{values['limitations'].strip()}\n\n"
            f"## Next experiment\n\n{values['next_experiment'].strip()}"
        )
        raw_skill = action.get("skill")
        if not improved or raw_skill is None:
            return title.strip(), body, None
        return title.strip(), body, QwenProgramAdapter._skill_artifact(
            raw_skill, allowed_case_ids, inspected_case_ids
        )

    @staticmethod
    def _skill_artifact(
        raw_skill: object,
        allowed_case_ids: set[str],
        inspected_case_ids: set[str],
    ) -> tuple[str, str]:
        if not isinstance(raw_skill, Mapping):
            raise ValueError("skill must be an object")
        skill_fields = (
            "name", "technique", "evidence", "applicability",
            "failure_conditions", "implementation_guidance",
        )
        skill_values = {
            field: _reflection_text(raw_skill.get(field), f"skill.{field}")
            for field in skill_fields
        }
        for field, text in skill_values.items():
            _validate_reflection_evidence(
                f"skill.{field}", text, allowed_case_ids, inspected_case_ids
            )
        skill_body = (
            f"# {skill_values['name'].strip()}\n\n"
            f"## Technique\n\n{skill_values['technique'].strip()}\n\n"
            f"## Evidence\n\n{skill_values['evidence'].strip()}\n\n"
            f"## Applicability conditions\n\n{skill_values['applicability'].strip()}\n\n"
            f"## Failure conditions\n\n{skill_values['failure_conditions'].strip()}\n\n"
            f"## Implementation guidance\n\n{skill_values['implementation_guidance'].strip()}"
        )
        return skill_values["name"], skill_body


class VLLMQwenProgramAdapter(QwenProgramAdapter):
    """Use the guarded Qwen agent through a validated vLLM server."""

    backend = "qwen-vllm"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        client: VLLMClient | None = None,
        temperature: float = 0.3,
        max_actions: int = 24,
        heartbeat_idle_seconds: float = 300.0,
        min_actions_before_yield: int = 4,
        submit_after_heartbeats: int = 4,
        curator_model: str | None = None,
    ) -> None:
        runtime = client or VLLMClient(model, base_url)
        runtime.validate_server()
        curator_runtime = runtime
        if curator_model is not None and curator_model != model:
            curator_runtime = VLLMClient(curator_model, base_url)
            curator_runtime.validate_server()
        super().__init__(
            model,
            client=runtime,
            temperature=temperature,
            max_actions=max_actions,
            heartbeat_idle_seconds=heartbeat_idle_seconds,
            min_actions_before_yield=min_actions_before_yield,
            submit_after_heartbeats=submit_after_heartbeats,
            curator_model=curator_model,
            curator_client=curator_runtime,
        )


class _CodingProgramAdapter:
    """Share two-stage behavior for resumable worktree-editing runtimes."""

    edits_worktree = True

    def __init__(self, model: str, runtime: SessionRuntime) -> None:
        self.model = model
        self._runtime = runtime

    def choose_parent(
        self,
        records: Sequence[Mapping[str, object]],
        worktree: Path,
        session_id: str | None,
    ) -> ParentChoice:
        """Plan without authorizing a file edit."""
        return _parse_parent(self._runtime(_parent_prompt(records), worktree, session_id, False))

    def edit(
        self,
        worktree: Path,
        choice: ParentChoice,
        session_id: str | None,
    ) -> EditChoice:
        """Resume the same session and permit only the program source edit."""
        prompt = (
            "Now implement this hypothesis: "
            f"{choice.hypothesis}\n"
            "Edit only program.py. Do not use Git, run grading, inspect private paths, or create "
            "other files. Preserve forecast(history, horizon, frequency). Finish with a short summary."
        )
        reply = self._runtime(prompt, worktree, session_id, True)
        return EditChoice(source=None, session_id=reply.session_id)


class CodexProgramAdapter(_CodingProgramAdapter):
    """Run a resumable Codex coding session inside one agent worktree."""

    backend = "codex"

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str = "high",
        timeout_seconds: float = 900.0,
        binary: str = "codex",
        runtime: SessionRuntime | None = None,
    ) -> None:
        command = runtime or _CodexSessionRuntime(
            binary=binary,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
        )
        super().__init__(model, command)


class ClaudeProgramAdapter(_CodingProgramAdapter):
    """Run Claude with guarded Read/Edit/Write access and no shell tools."""

    backend = "claude"

    def __init__(
        self,
        model: str,
        *,
        timeout_seconds: float = 900.0,
        binary: str = "claude",
        runtime: SessionRuntime | None = None,
    ) -> None:
        command = runtime or _ClaudeSessionRuntime(
            binary=binary,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        super().__init__(model, command)


class _CodexSessionRuntime:
    def __init__(self, *, binary: str, model: str, reasoning_effort: str, timeout_seconds: float) -> None:
        self.binary = binary
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds

    def __call__(self, prompt: str, worktree: Path, session_id: str | None, allow_edits: bool) -> BackendReply:
        sandbox_mode = "workspace-write" if allow_edits else "read-only"
        with tempfile.TemporaryDirectory(prefix="program-evolution-codex-") as directory:
            result_path = Path(directory) / "result.txt"
            if session_id is None:
                command = [
                    self.binary,
                    "exec",
                    "--sandbox",
                    sandbox_mode,
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--color",
                    "never",
                    "--json",
                    "--output-last-message",
                    str(result_path),
                    "--cd",
                    str(worktree),
                    "-c",
                    f'model_reasoning_effort="{self.reasoning_effort}"',
                    "--model",
                    self.model,
                    "-",
                ]
            else:
                command = [
                    self.binary,
                    "exec",
                    "resume",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--json",
                    "--output-last-message",
                    str(result_path),
                    "--model",
                    self.model,
                    "-c",
                    f'sandbox_mode="{sandbox_mode}"',
                    session_id,
                    "-",
                ]
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=worktree,
            )
            if completed.returncode != 0 or not result_path.exists():
                detail = (completed.stderr or completed.stdout).strip()[-2000:]
                raise RuntimeError(f"Codex coding session failed: {detail}")
            resumed_id = session_id or _codex_session_id(completed.stdout)
            if resumed_id is None:
                raise RuntimeError("Codex returned no resumable session ID")
            return BackendReply(result_path.read_text(encoding="utf-8").strip(), resumed_id)


def _codex_session_id(output: str) -> str | None:
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started" and event.get("thread_id"):
            return str(event["thread_id"])
    return None


class _ClaudeSessionRuntime:
    def __init__(self, *, binary: str, model: str, timeout_seconds: float) -> None:
        self.binary = binary
        self.model = model
        self.timeout_seconds = timeout_seconds

    def __call__(self, prompt: str, worktree: Path, session_id: str | None, allow_edits: bool) -> BackendReply:
        tools = "Read,Edit,Write" if allow_edits else "Read"
        command = [
            self.binary,
            "--print",
            "--output-format",
            "json",
            "--safe-mode",
            "--disable-slash-commands",
            "--setting-sources",
            "",
            "--tools",
            tools,
            "--allowedTools",
            tools,
            "--disallowedTools",
            "Bash",
            "--permission-mode",
            "acceptEdits",
            "--system-prompt",
            "Work only on the supplied statistical forecasting program and obey the host boundaries.",
            "--model",
            self.model,
        ]
        if session_id is not None:
            command.extend(["--resume", session_id])
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("CLAUDE")
        }
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            cwd=worktree,
            env=environment,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise RuntimeError(f"Claude coding session failed: {detail}")
        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Claude coding session returned invalid JSON") from exc
        if envelope.get("is_error"):
            raise RuntimeError("Claude coding session returned an error")
        resumed_id = envelope.get("session_id") or session_id
        if not resumed_id:
            raise RuntimeError("Claude returned no resumable session ID")
        return BackendReply(str(envelope.get("result", "")).strip(), str(resumed_id))


class ProgramEvolutionAgent:
    """Coordinate one host-controlled two-stage evolution action."""

    def __init__(self, run: EvolutionRun, adapter: Any) -> None:
        self.run = run
        self.adapter = adapter
        self.run.configure_backend(adapter.backend, adapter.model)

    def propose(self, agent_id: str) -> EvolutionProposal:
        """Choose a parent, request one edit, commit it, and enqueue grading."""
        records = self.run.attempts.parent_candidates()
        if not records:
            raise RuntimeError("at least one valid recorded parent is required")
        worktree = self.run._agent_worktree(agent_id)
        shared_memory = worktree / ".evolution"
        if not shared_memory.is_symlink() or shared_memory.resolve() != self.run.public.resolve():
            raise RuntimeError("agent shared-memory link is invalid")
        if hasattr(self.adapter, "plan_and_edit"):
            try:
                choice = self.adapter.plan_and_edit(self.run, agent_id, records)
            except Exception:
                changes = _git(worktree, "status", "--porcelain", "--untracked-files=all")
                if changes.strip() == f"M {self.run.PROGRAM_FILE}":
                    _git(worktree, "restore", self.run.PROGRAM_FILE)
                raise
            eligible = {
                str(record["commit_hash"]) for record in self.run.attempts.parent_candidates()
            }
            if choice.commit_hash not in eligible:
                raise ValueError("backend selected an ineligible parent")
            if not shared_memory.is_symlink() or shared_memory.resolve() != self.run.public.resolve():
                raise RuntimeError("backend replaced the shared-memory boundary")
            message = "evolve: " + " ".join(choice.hypothesis.split())[:120]
            commit_hash = self.run.commit_edited_candidate(agent_id, message)
            sequence = self.run.submit(
                agent_id=agent_id,
                commit_hash=commit_hash,
                backend=self.adapter.backend,
                hypothesis=choice.hypothesis,
            )
            return EvolutionProposal(choice.commit_hash, commit_hash, choice.hypothesis, sequence)
        session_id = self.run.agent_session(agent_id)
        choice = self.adapter.choose_parent(records, worktree, session_id)
        eligible = {str(record["commit_hash"]) for record in records}
        if choice.commit_hash not in eligible:
            raise ValueError("backend selected an ineligible parent")
        self.run.save_agent_session(agent_id, choice.session_id)
        self.run.checkout_parent(agent_id, choice.commit_hash)
        shared_memory.unlink()
        try:
            edit = self.adapter.edit(worktree, choice, choice.session_id)
        finally:
            if not shared_memory.exists():
                relative_public = os.path.relpath(self.run.public, worktree)
                shared_memory.symlink_to(relative_public, target_is_directory=True)
        if not shared_memory.is_symlink() or shared_memory.resolve() != self.run.public.resolve():
            raise RuntimeError("backend replaced the shared-memory boundary")
        self.run.save_agent_session(agent_id, edit.session_id)
        message = "evolve: " + " ".join(choice.hypothesis.split())[:120]
        if self.adapter.edits_worktree:
            commit_hash = self.run.commit_edited_candidate(agent_id, message)
        else:
            if edit.source is None:
                raise ValueError("structured backend returned no source")
            commit_hash = self.run.commit_candidate(agent_id, edit.source, message)
        sequence = self.run.submit(
            agent_id=agent_id,
            commit_hash=commit_hash,
            backend=self.adapter.backend,
            hypothesis=choice.hypothesis,
        )
        return EvolutionProposal(choice.commit_hash, commit_hash, choice.hypothesis, sequence)


class ConcurrentEvolutionController:
    """Run persistent agents and FIFO-claimed candidate graders concurrently."""

    def __init__(
        self,
        run: EvolutionRun,
        adapter: Any,
        grader: ProgramGrader,
        event_logger: Callable[[Mapping[str, object]], None],
        grader_workers: int = 1,
    ) -> None:
        if isinstance(grader_workers, bool) or grader_workers <= 0:
            raise ValueError("grader_workers must be positive")
        self.run = run
        self.adapter = adapter
        self.grader = grader
        self.agent = ProgramEvolutionAgent(run, adapter)
        self.event_logger = event_logger
        self.grader_workers = int(grader_workers)
        self._condition = threading.Condition()
        self._event_lock = threading.Lock()
        self._shutdown = threading.Event()

    def run_until_complete(self) -> None:
        """Run every agent until the shared budget is exhausted or the run stops."""
        self.run.recover_grading_leases()
        self.run.recover_heartbeat_leases()
        self._shutdown.clear()
        agent_ids = tuple(sorted(self.run.sessions()["agents"]))
        agent_workers = [
            threading.Thread(
                target=self._agent_loop,
                args=(agent_id,),
                name=f"program-evolution-{agent_id}",
            )
            for agent_id in agent_ids
        ]
        grader_workers = [
            threading.Thread(
                target=self._grader_loop,
                args=(f"grader-{index}",),
                name=f"program-evolution-grader-{index}",
            )
            for index in range(1, self.grader_workers + 1)
        ]
        curator_workers = []
        heartbeat_monitors = []
        if hasattr(self.adapter, "curate"):
            curator_workers.append(threading.Thread(
                target=self._curator_loop,
                name="program-evolution-curator",
            ))
            heartbeat_monitors.append(threading.Thread(
                target=self._heartbeat_monitor,
                name="program-evolution-heartbeat-monitor",
            ))
        workers = [*agent_workers, *grader_workers, *curator_workers, *heartbeat_monitors]
        for worker in workers:
            worker.start()
        try:
            while True:
                progress = self.run.status()
                pending = int(progress["queued"]) + int(progress["running"])
                if progress["status"] != "running":
                    break
                budget_done = int(progress["evaluation_count"]) >= int(progress["evaluation_budget"])
                agents_done = not any(worker.is_alive() for worker in agent_workers)
                curator_pending = 0
                if curator_workers:
                    curator_pending = (
                        int(progress["heartbeat"]["queued"])
                        + int(progress["heartbeat"]["running"])
                    )
                if budget_done and agents_done and curator_pending == 0:
                    break
                if not any(worker.is_alive() for worker in agent_workers) and pending == 0:
                    if not budget_done:
                        raise RuntimeError("all asynchronous agents stopped before the budget was exhausted")
                with self._condition:
                    self._condition.wait(timeout=0.05)
        finally:
            self._shutdown.set()
            with self._condition:
                self._condition.notify_all()
            for worker in workers:
                worker.join()

    def _grader_loop(self, grader_id: str) -> None:
        while not self._shutdown.is_set():
            try:
                record = self.run.grade_next(self.grader, grader_id)
            except Exception as exc:
                self._emit({
                    "event": "grader_error",
                    "grader_id": grader_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                with self._condition:
                    self._condition.wait(timeout=0.1)
                continue
            if record is not None:
                metrics = record["metrics"]
                self._emit({
                    "event": "graded",
                    "grader_id": grader_id,
                    "agent_id": record["agent_id"],
                    "commit_hash": record["commit_hash"],
                    "score": metrics["score"],
                    "status": record["status"],
                })
                with self._condition:
                    self._condition.notify_all()
                continue
            with self._condition:
                self._condition.wait(timeout=0.05)

    def _curator_loop(self) -> None:
        while not self._shutdown.is_set():
            event = self.run.claim_heartbeat()
            if event is None:
                with self._condition:
                    self._condition.wait(timeout=0.05)
                continue
            sequence = int(event["sequence"])
            try:
                completed = bool(self.adapter.curate(self.run, event))
                if not completed:
                    self.run.fail_heartbeat(sequence, "curator action budget exhausted")
                self._emit({
                    "event": "heartbeat_synthesized" if completed else "heartbeat_synthesis_failed",
                    "sequence": sequence,
                    "agent_id": event["agent_id"],
                })
            except Exception as exc:
                self.run.fail_heartbeat(sequence, f"{type(exc).__name__}: {exc}")
                self._emit({
                    "event": "heartbeat_synthesis_failed",
                    "sequence": sequence,
                    "agent_id": event["agent_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                })
            with self._condition:
                self._condition.notify_all()

    def _heartbeat_monitor(self) -> None:
        while not self._shutdown.is_set():
            sessions = self.run.sessions()
            configuration = sessions.get("heartbeat") or {}
            idle_seconds = float(configuration.get("idle_seconds", 300.0))
            now = time.time()
            for agent_id, state in sessions["agents"].items():
                last = state.get("last_heartbeat")
                if (
                    state.get("phase") == "idle"
                    and state.get("outstanding_commit") is None
                    and last is not None
                    and now - float(last) >= idle_seconds
                ):
                    self.run.record_heartbeat(str(agent_id), "idle", self.run.agent_checkpoint(str(agent_id)))
                    with self._condition:
                        self._condition.notify_all()
            with self._condition:
                self._condition.wait(timeout=min(1.0, idle_seconds))

    def _agent_loop(self, agent_id: str) -> None:
        try:
            if self.run.latest_heartbeat(agent_id) is None:
                self.run.record_heartbeat(agent_id, "initial", {})
                with self._condition:
                    self._condition.notify_all()
            while not self._capacity_full():
                if self.run.sessions()["status"] != "running":
                    self.run.set_agent_activity(agent_id, "stopped")
                    return
                self.run.set_agent_activity(agent_id, "researching")
                proposal = self.agent.propose(agent_id)
                self.run.set_agent_activity(
                    agent_id, "waiting_for_grade", outstanding_commit=proposal.commit_hash
                )
                self._emit({
                    "event": "proposed",
                    "agent_id": agent_id,
                    "commit_hash": proposal.commit_hash,
                    "parent_hash": proposal.parent_hash,
                })
                with self._condition:
                    self._condition.notify_all()
                    while self.run.attempts.get(proposal.commit_hash) is None:
                        if self.run.sessions()["status"] != "running":
                            self.run.set_agent_activity(
                                agent_id, "stopped", outstanding_commit=proposal.commit_hash
                            )
                            return
                        self._condition.wait(timeout=0.1)
                record = self.run.attempts.get(proposal.commit_hash)
                assert record is not None
                self.run.set_agent_activity(
                    agent_id, "reflecting", outstanding_commit=proposal.commit_hash
                )
                reflected = False
                reflection_attempts = 0
                while (
                    not reflected
                    and reflection_attempts < 3
                    and self.run.sessions()["status"] == "running"
                ):
                    reflection_attempts += 1
                    heartbeat_before = self.run.latest_heartbeat(agent_id)
                    try:
                        reflected = bool(self.adapter.reflect(self.run, agent_id, record))
                        reflected = reflected and self.run.reflection_complete(record)
                    except Exception as exc:
                        path = self.run.public / "logs" / f"{agent_id}-reflection.log"
                        with path.open("a", encoding="utf-8") as handle:
                            handle.write(f"incomplete reflection: {type(exc).__name__}: {exc}\n")
                    if not reflected:
                        heartbeat_after = self.run.latest_heartbeat(agent_id)
                        before_sequence = None if heartbeat_before is None else heartbeat_before["sequence"]
                        after_sequence = None if heartbeat_after is None else heartbeat_after["sequence"]
                        if before_sequence == after_sequence:
                            self.run.record_heartbeat(
                                agent_id,
                                "evidence_note_retry",
                                {
                                    "last_commit": proposal.commit_hash,
                                    "last_hypothesis": proposal.hypothesis,
                                    "evidence_note_pending": True,
                                },
                            )
                        with self._condition:
                            self._condition.notify_all()
                self._emit({
                    "event": "reflected" if reflected else "reflection_incomplete",
                    "agent_id": agent_id,
                    "commit_hash": proposal.commit_hash,
                })
                if not reflected:
                    error = "required evidence note failed after three attempts"
                    self.run.set_agent_activity(
                        agent_id,
                        "failed",
                        outstanding_commit=proposal.commit_hash,
                        error=error,
                    )
                    self._emit({"event": "agent_failed", "agent_id": agent_id, "error": error})
                    return
                self.run.record_heartbeat(
                    agent_id,
                    "post_evaluation",
                    {
                        "last_commit": proposal.commit_hash,
                        "last_parent": proposal.parent_hash,
                        "last_hypothesis": proposal.hypothesis,
                        "parent_hash": None,
                        "hypothesis": None,
                        "edited": False,
                    },
                )
                self.run.set_agent_activity(agent_id, "idle")
                with self._condition:
                    self._condition.notify_all()
            self.run.set_agent_activity(agent_id, "complete")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.run.set_agent_activity(agent_id, "failed", error=message)
            self._emit({"event": "agent_failed", "agent_id": agent_id, "error": message})
            with self._condition:
                self._condition.notify_all()

    def _capacity_full(self) -> bool:
        progress = self.run.status()
        reserved = int(progress["evaluation_count"]) + int(progress["queued"]) + int(progress["running"])
        return reserved >= int(progress["evaluation_budget"])

    def _emit(self, payload: Mapping[str, object]) -> None:
        with self._event_lock:
            self.event_logger(payload)
