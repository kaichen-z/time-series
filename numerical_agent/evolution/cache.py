"""Content-addressed cache for deterministic forecasting-method outcomes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .execution import (
    CRASHED,
    INVALID,
    NOT_APPLICABLE,
    SUCCESS,
    Outcome,
    Task,
    run_method,
)
from .module import Method
from .analysis_skills import DEFAULT_SKILLS_PATH


CACHE_SCHEMA = 1
_STATUSES = {SUCCESS, NOT_APPLICABLE, CRASHED, INVALID}


class CacheMissError(RuntimeError):
    """A cache-only evaluation could not resolve every requested outcome."""


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


class OutcomeCache:
    """Evaluate only cache misses while preserving task order."""

    def __init__(
        self,
        root: str | Path,
        *,
        time_budget_s: float = 20.0,
        timeout_circuit_breaker: int = 2,
        worker_startup_timeout_s: float = 10.0,
        skills_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.time_budget_s = float(time_budget_s)
        self.timeout_circuit_breaker = int(timeout_circuit_breaker)
        self.worker_startup_timeout_s = float(worker_startup_timeout_s)
        self.skills_path = Path(skills_path) if skills_path is not None else DEFAULT_SKILLS_PATH
        self.stats = CacheStats()

    def evaluate_method(
        self,
        method: Method,
        tasks: Sequence[Task],
        *,
        isolated: bool,
        require_forecasts: bool = False,
    ) -> tuple[Outcome, ...]:
        keys = [self._key(method, task, isolated=isolated) for task in tasks]
        outcomes: list[Outcome | None] = []
        missing_tasks: list[Task] = []
        missing_indices: list[int] = []
        for index, (task, key) in enumerate(zip(tasks, keys, strict=True)):
            cached = self._read(
                key, method.name, task.task_id,
                expected_horizon=task.horizon,
                require_forecast=require_forecasts,
            )
            outcomes.append(cached)
            if cached is None:
                self.stats.misses += 1
                missing_tasks.append(task)
                missing_indices.append(index)
            else:
                self.stats.hits += 1

        if missing_tasks:
            computed, _ = run_method(
                method,
                missing_tasks,
                time_budget_s=self.time_budget_s,
                isolated=isolated,
                timeout_circuit_breaker=self.timeout_circuit_breaker,
                worker_startup_timeout_s=self.worker_startup_timeout_s,
                skills_path=self.skills_path,
            )
            by_task = {outcome.task_id: outcome for outcome in computed}
            for index, task in zip(missing_indices, missing_tasks, strict=True):
                outcome = by_task[task.task_id]
                outcomes[index] = outcome
                self._write(keys[index], outcome)

        if any(outcome is None for outcome in outcomes):
            raise RuntimeError(f"outcome cache failed to resolve {method.name}")
        return tuple(outcome for outcome in outcomes if outcome is not None)

    def require_cached_method(
        self,
        method: Method,
        tasks: Sequence[Task],
        *,
        isolated: bool,
        require_forecasts: bool = False,
    ) -> tuple[Outcome, ...]:
        """Read exact content-addressed outcomes without executing cache misses."""
        outcomes: list[Outcome] = []
        missing: list[str] = []
        for task in tasks:
            key = self._key(method, task, isolated=isolated)
            cached = self._read(
                key,
                method.name,
                task.task_id,
                expected_horizon=task.horizon,
                require_forecast=require_forecasts,
            )
            if cached is None:
                self.stats.misses += 1
                missing.append(task.task_id)
            else:
                self.stats.hits += 1
                outcomes.append(cached)
        if missing:
            raise CacheMissError(
                f"cache-only lookup for {method.name} is missing tasks: {', '.join(missing)}"
            )
        return tuple(outcomes)

    def cache_key(self, method: Method, task: Task, *, isolated: bool) -> str:
        """Return the content key used for one method/task execution."""
        return self._key(method, task, isolated=isolated)

    def _key(self, method: Method, task: Task, *, isolated: bool) -> str:
        payload = {
            "schema": CACHE_SCHEMA,
            "method": {"name": method.name, "source": method.source},
            "analysis_skills": self.skills_path.read_text(encoding="utf-8"),
            "task": asdict(task),
            "execution": {
                "isolated": bool(isolated),
                "time_budget_s": self.time_budget_s,
                "timeout_circuit_breaker": self.timeout_circuit_breaker,
                "worker_startup_timeout_s": self.worker_startup_timeout_s,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _read(
        self,
        key: str,
        method: str,
        task_id: str,
        *,
        expected_horizon: int,
        require_forecast: bool,
    ) -> Outcome | None:
        try:
            payload = json.loads((self.root / f"{key}.json").read_text(encoding="utf-8"))
            if payload.get("schema") != CACHE_SCHEMA or payload.get("key") != key:
                return None
            raw = payload["outcome"]
            outcome = Outcome(
                method=str(raw["method"]),
                task_id=str(raw["task_id"]),
                status=str(raw["status"]),
                smape=_optional_float(raw.get("smape")),
                mae=_optional_float(raw.get("mae")),
                mase=_optional_float(raw.get("mase")),
                detail=str(raw.get("detail", "")),
                forecast=tuple(_finite_float(value) for value in raw.get("forecast", ())),
            )
            if outcome.method != method or outcome.task_id != task_id or outcome.status not in _STATUSES:
                return None
            if outcome.status == SUCCESS and any(
                metric is None for metric in (outcome.smape, outcome.mae, outcome.mase)
            ):
                return None
            if (
                require_forecast
                and outcome.status == SUCCESS
                and len(outcome.forecast) != expected_horizon
            ):
                return None
            return outcome
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _write(self, key: str, outcome: Outcome) -> None:
        payload = json.dumps(
            {"schema": CACHE_SCHEMA, "key": key, "outcome": asdict(outcome)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.root, prefix=f".{key}.", delete=False
            ) as handle:
                temporary = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.root / f"{key}.json")
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("cached metrics must be finite")
    return number


def _finite_float(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("cached forecast values must be finite")
    return number
