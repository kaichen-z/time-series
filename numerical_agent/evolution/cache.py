"""Content-addressed cache for deterministic forecasting-method outcomes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

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


CACHE_SCHEMA = 2
SCALED_METRIC_SCHEMA = 1
SCALED_METRIC_CAP = 5.0
_STATUSES = {SUCCESS, NOT_APPLICABLE, CRASHED, INVALID}


class CacheError(ValueError):
    """A serialized outcome cannot satisfy the active cache contract."""


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

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Outcome:
        """Reconstruct one active cache outcome under the scaled-metric contract."""
        return cls._outcome_from_payload(payload, require_scaled_metrics=True)

    @classmethod
    def to_payload(cls, outcome: Outcome) -> dict[str, object]:
        """Serialize one active cache outcome without losing raw infinite tail risk."""
        return cls._outcome_payload(outcome)

    @classmethod
    def from_legacy_report_payload(cls, payload: Mapping[str, object]) -> Outcome:
        """Read a diagnostic-only legacy report row; never use it as an active outcome."""
        return cls._outcome_from_payload(payload, require_scaled_metrics=False)

    def _key(self, method: Method, task: Task, *, isolated: bool) -> str:
        payload = {
            "schema": CACHE_SCHEMA,
            "scaled_metric_schema": SCALED_METRIC_SCHEMA,
            "scaled_metric_cap": SCALED_METRIC_CAP,
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
            if (
                payload.get("schema") != CACHE_SCHEMA
                or payload.get("scaled_metric_schema") != SCALED_METRIC_SCHEMA
                or payload.get("scaled_metric_cap") != SCALED_METRIC_CAP
                or payload.get("key") != key
            ):
                return None
            raw = _mapping(payload["outcome"])
            outcome = self.from_payload(raw)
            if outcome.method != method or outcome.task_id != task_id or outcome.status not in _STATUSES:
                return None
            if (
                require_forecast
                and outcome.status == SUCCESS
                and len(outcome.forecast) != expected_horizon
            ):
                return None
            return outcome
        except (OSError, CacheError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _write(self, key: str, outcome: Outcome) -> None:
        payload = json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "scaled_metric_schema": SCALED_METRIC_SCHEMA,
                "scaled_metric_cap": SCALED_METRIC_CAP,
                "key": key,
                "outcome": self.to_payload(outcome),
            },
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

    @classmethod
    def _outcome_from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        require_scaled_metrics: bool,
    ) -> Outcome:
        try:
            outcome = Outcome(
                method=str(payload["method"]),
                task_id=str(payload["task_id"]),
                status=str(payload["status"]),
                smae=_optional_scaled_float(payload.get("smae")),
                srmse=_optional_scaled_float(payload.get("srmse")),
                smae_raw=_optional_raw_scaled_float(payload.get("smae_raw")),
                srmse_raw=_optional_raw_scaled_float(payload.get("srmse_raw")),
                smae_clipped=_optional_bool(payload.get("smae_clipped")),
                srmse_clipped=_optional_bool(payload.get("srmse_clipped")),
                smape=_optional_float(payload.get("smape")),
                mae=_optional_float(payload.get("mae")),
                mase=_optional_float(payload.get("mase")),
                detail=str(payload.get("detail", "")),
                forecast=tuple(_finite_float(value) for value in payload.get("forecast", ())),
            )
        except (TypeError, ValueError, KeyError) as error:
            raise CacheError("invalid cached outcome") from error
        if outcome.status not in _STATUSES:
            raise CacheError("cached outcome has an unknown status")
        if outcome.status != SUCCESS:
            return outcome
        scaled = (
            outcome.smae,
            outcome.srmse,
            outcome.smae_raw,
            outcome.srmse_raw,
            outcome.smae_clipped,
            outcome.srmse_clipped,
        )
        if require_scaled_metrics and any(value is None for value in scaled):
            raise CacheError("successful cached outcome requires complete scaled metrics")
        if not require_scaled_metrics:
            return outcome
        assert outcome.smae is not None and outcome.srmse is not None
        assert outcome.smae_raw is not None and outcome.srmse_raw is not None
        assert outcome.smae_clipped is not None and outcome.srmse_clipped is not None
        if (
            outcome.smae != min(SCALED_METRIC_CAP, outcome.smae_raw)
            or outcome.srmse != min(SCALED_METRIC_CAP, outcome.srmse_raw)
            or outcome.smae_clipped != (outcome.smae_raw > SCALED_METRIC_CAP)
            or outcome.srmse_clipped != (outcome.srmse_raw > SCALED_METRIC_CAP)
        ):
            raise CacheError("cached scaled metrics do not match the active cap policy")
        if any(metric is None for metric in (outcome.smape, outcome.mae, outcome.mase)):
            raise CacheError("successful cached outcome requires complete diagnostic metrics")
        return outcome

    @staticmethod
    def _outcome_payload(outcome: Outcome) -> dict[str, object]:
        payload = asdict(outcome)
        for name in ("smae_raw", "srmse_raw"):
            value = payload[name]
            if value == math.inf:
                payload[name] = "inf"
        return payload


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("cached metrics must be finite")
    return number


def _optional_scaled_float(value: object) -> float | None:
    number = _optional_float(value)
    if number is not None and (number < 0.0 or number > SCALED_METRIC_CAP):
        raise ValueError("cached scaled metrics must be in the capped range")
    return number


def _optional_raw_scaled_float(value: object) -> float | None:
    if value is None:
        return None
    if value == "inf":
        return math.inf
    number = float(value)
    if math.isnan(number) or number < 0.0:
        raise ValueError("cached raw scaled metrics must be nonnegative")
    return number


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("cached clipping flags must be booleans")
    return value


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CacheError("cached outcome payload must be an object")
    return value


def _finite_float(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("cached forecast values must be finite")
    return number
