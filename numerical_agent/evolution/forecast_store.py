"""Shared content-addressed forecasts for numerical selector evolution."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import multiprocessing
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from common.evolution_core.contracts import (
    METRIC_POLICY_FINGERPRINT,
    metric_policy_metadata,
    require_active_metric_policy,
)
from common.payload import strict_json_loads

from .execution import CRASHED, INVALID, NOT_APPLICABLE, SUCCESS, Outcome, load_methods
from .module import read_module
from .portfolio import (
    CombinedPolicy,
    InvalidTSFMForecastError,
    PolicyNotApplicable,
    PolicyPortfolio,
    combine_materialized_outcome,
    forecast_tsfm,
)


class CacheIntegrityError(ValueError):
    """An existing active cache row is corrupt and must abort the lifecycle."""


class ForecastStore:
    """Content-addressed history-only forecast cache shared across selector generations."""

    def __init__(
        self,
        root: str | Path,
        module_path: Path,
        skills_path: Path | None,
        portfolio: PolicyPortfolio,
        runtimes,
        *,
        screening_hash: str,
        runtime_identity: Mapping[str, object] | None,
        statistical_time_budget_s: float = 20.0,
        statistical_failure_limit: int = 2,
    ) -> None:
        if statistical_time_budget_s <= 0:
            raise ValueError("statistical_time_budget_s must be positive")
        if statistical_failure_limit < 1:
            raise ValueError("statistical_failure_limit must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.not_applicable = _HistoryOnlyNotApplicable
        module = read_module(module_path)
        self.statistical_names = frozenset(module.names())
        self._statistical = _IsolatedStatisticalRuntime(
            module_path,
            skills_path,
            time_budget_s=statistical_time_budget_s,
            failure_limit=statistical_failure_limit,
        )
        self.module = module
        self.portfolio = portfolio
        self.runtimes = runtimes
        self.screening_hash = screening_hash
        self.tsfm = {policy.name: policy for policy in portfolio.tsfm}
        self.combined = {policy.name: policy for policy in portfolio.combined}
        identity = {
            "module_source": module_path.read_text(encoding="utf-8"),
            "skills_source": (
                skills_path.read_text(encoding="utf-8") if skills_path is not None else None
            ),
            "portfolio": asdict(portfolio),
            "reviewed_manifests_sha256": _sha256(
                Path(__file__).parent.parent / "tsfm" / "runtime_manifests.json"
            ),
            "runtime": dict(runtime_identity or {}),
        }
        self.identity_hash = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        ).hexdigest()
        self.hits = 0
        self.misses = 0
        self._materialized_leaf_outcomes: dict[str, Outcome] = {}

    def forecast(
        self, name: str, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        key = self._key(name, history, horizon, frequency)
        path = self.root / f"{key}.json"
        if not path.exists():
            payload = None
        else:
            try:
                payload = strict_json_loads(
                    path.read_text(encoding="utf-8"),
                    context="active hindcast cache row",
                )
            except (OSError, TypeError, ValueError) as error:
                raise CacheIntegrityError(
                    "active hindcast cache row is malformed"
                ) from error
        if payload is not None:
            try:
                if not isinstance(payload, Mapping):
                    raise CacheIntegrityError(
                        "active hindcast cache row must be an object"
                    )
                require_active_metric_policy(payload, context="active hindcast cache row")
                if (
                    type(payload.get("cache_schema")) is not int
                    or payload["cache_schema"] != 3
                ):
                    raise CacheIntegrityError(
                        "active hindcast cache row schema mismatch"
                    )
                if payload.get("key") != key:
                    raise CacheIntegrityError("active hindcast cache row key mismatch")
                if payload.get("status") == SUCCESS:
                    values = tuple(float(value) for value in payload["forecast"])
                    if len(values) != horizon or not all(map(math.isfinite, values)):
                        raise CacheIntegrityError(
                            "active hindcast cache row forecast is noncanonical"
                        )
                    self.hits += 1
                    return values
                if payload.get("status") == NOT_APPLICABLE:
                    self.hits += 1
                    raise self.not_applicable(
                        str(payload.get("detail", "not applicable"))
                    )
                raise CacheIntegrityError(
                    "active hindcast cache row status is noncanonical"
                )
            except CacheIntegrityError:
                raise
            except self.not_applicable:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise CacheIntegrityError(
                    str(error) or "active hindcast cache row is noncanonical"
                ) from error
        self.misses += 1
        try:
            values = self._execute(name, history, horizon, frequency)
        except self.not_applicable as error:
            self._write(path, {
                "cache_schema": 3,
                **metric_policy_metadata(),
                "key": key,
                "status": NOT_APPLICABLE,
                "detail": str(error)[:200],
            })
            raise
        self._write(path, {
            "cache_schema": 3,
            **metric_policy_metadata(),
            "key": key,
            "status": SUCCESS,
            "forecast": list(values),
        })
        return values

    def _execute(
        self, name: str, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        if name in self.statistical_names:
            return self._statistical.forecast(name, history, horizon, frequency)
        if policy := self.tsfm.get(name):
            try:
                return forecast_tsfm(
                    policy,
                    history=history,
                    horizon=horizon,
                    frequency=frequency,
                    runtimes=self.runtimes,
                )
            except PolicyNotApplicable as error:
                raise self.not_applicable(str(error)) from None
            except InvalidTSFMForecastError as error:
                raise _StructuralForecastInvalid(str(error)) from None
        if policy := self.combined.get(name):
            return self._combined(policy, history, horizon, frequency)
        raise KeyError(f"unknown numerical candidate {name}")

    def close(self) -> None:
        self._statistical.close()

    def _combined(
        self,
        policy: CombinedPolicy,
        history: tuple[float, ...],
        horizon: int,
        frequency: str,
    ) -> tuple[float, ...]:
        parent_outcomes = {
            parent: self._materialized_leaf_outcome(parent, history, horizon, frequency)
            for parent in policy.parents
        }
        combined = combine_materialized_outcome(
            policy,
            parent_outcomes,
            task_id="history-only",
            history=history,
            horizon=horizon,
            frequency=frequency,
        )
        if combined.status == SUCCESS:
            return _valid_forecast(combined.forecast, horizon)
        if combined.status == NOT_APPLICABLE:
            raise self.not_applicable(combined.detail)
        if combined.status == INVALID:
            raise ValueError(combined.detail or combined.status)
        raise RuntimeError(combined.detail or combined.status)

    def _materialized_leaf_outcome(
        self,
        name: str,
        history: tuple[float, ...],
        horizon: int,
        frequency: str,
    ) -> Outcome:
        """Resolve one cached leaf; Combined policies cannot be parents in this graph."""
        key = self._key(name, history, horizon, frequency)
        if key in self._materialized_leaf_outcomes:
            return self._materialized_leaf_outcomes[key]
        try:
            outcome = Outcome(
                name,
                "history-only",
                SUCCESS,
                forecast=self.forecast(name, history, horizon, frequency),
            )
        except CacheIntegrityError:
            raise
        except self.not_applicable as error:
            outcome = Outcome(
                name, "history-only", NOT_APPLICABLE, detail=str(error)[:200]
            )
        except _StructuralForecastInvalid as error:
            outcome = Outcome(name, "history-only", INVALID, detail=str(error)[:200])
        except ValueError as error:
            if name in self.tsfm:
                outcome = Outcome(
                    name,
                    "history-only",
                    CRASHED,
                    detail=f"{type(error).__name__}: {error}"[:200],
                )
            else:
                outcome = Outcome(
                    name, "history-only", INVALID, detail=str(error)[:200]
                )
        except Exception as error:
            outcome = Outcome(
                name,
                "history-only",
                CRASHED,
                detail=f"{type(error).__name__}: {error}"[:200],
            )
        self._materialized_leaf_outcomes[key] = outcome
        return outcome

    def _key(self, name, history, horizon, frequency) -> str:
        payload = json.dumps({
            "schema": 3,
            "metric_policy_fingerprint": METRIC_POLICY_FINGERPRINT,
            "identity": self.identity_hash,
            "name": name,
            "history": history,
            "horizon": horizon,
            "frequency": frequency,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _write(path: Path, payload: Mapping[str, object]) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, prefix=f".{path.stem}.", delete=False
            ) as handle:
                temporary = handle.name
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)


class _HistoryOnlyNotApplicable(Exception):
    """Transport-safe applicability signal from the statistical worker."""


class _StructuralForecastInvalid(ValueError):
    """A known finite/horizon failure that maps to the canonical INVALID status."""


class _IsolatedStatisticalRuntime:
    """Execute generated statistical methods behind a restartable hard boundary."""

    def __init__(
        self,
        module_path: str | Path,
        skills_path: str | Path | None,
        *,
        time_budget_s: float,
        failure_limit: int,
        startup_timeout_s: float = 10.0,
    ) -> None:
        self.module_path = str(Path(module_path).resolve())
        self.skills_path = str(Path(skills_path).resolve()) if skills_path else None
        self.time_budget_s = time_budget_s
        self.failure_limit = failure_limit
        self.startup_timeout_s = startup_timeout_s
        self.context = multiprocessing.get_context("spawn")
        self.connection = None
        self.worker = None
        self.hard_failures: dict[str, int] = {}

    def forecast(
        self, name: str, history: Sequence[float], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        failures = self.hard_failures.get(name, 0)
        if failures >= self.failure_limit:
            raise RuntimeError(
                f"hard failure circuit breaker after {failures} failures for {name}"
            )
        self._ensure_worker()
        try:
            self.connection.send((name, tuple(history), horizon, frequency))
        except (BrokenPipeError, EOFError, OSError) as error:
            self._hard_failure(name)
            raise RuntimeError(f"statistical worker crashed: {type(error).__name__}") from None
        if not self.connection.poll(self.time_budget_s):
            self._hard_failure(name)
            raise RuntimeError(f"hard timeout after {self.time_budget_s:g}s for {name}")
        try:
            status, payload = self.connection.recv()
        except (EOFError, OSError):
            self._hard_failure(name)
            raise RuntimeError(f"statistical worker crashed while running {name}") from None
        if status == SUCCESS:
            return _valid_forecast(payload, horizon)
        if status == NOT_APPLICABLE:
            raise _HistoryOnlyNotApplicable(str(payload))
        raise RuntimeError(str(payload))

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.connection.close()
        _stop_statistical_worker(self.worker)
        self.connection = None
        self.worker = None

    def _ensure_worker(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self.close()
        parent, child = self.context.Pipe()
        worker = self.context.Process(
            target=_statistical_worker,
            args=(self.module_path, self.skills_path, child),
            daemon=True,
        )
        worker.start()
        child.close()
        if not parent.poll(self.startup_timeout_s):
            parent.close()
            _stop_statistical_worker(worker)
            raise RuntimeError(
                f"statistical worker startup timeout after {self.startup_timeout_s:g}s"
            )
        try:
            status, detail = parent.recv()
        except (EOFError, OSError):
            parent.close()
            _stop_statistical_worker(worker)
            raise RuntimeError("statistical worker crashed during startup") from None
        if status != "ready":
            parent.close()
            _stop_statistical_worker(worker)
            raise RuntimeError(str(detail))
        self.connection = parent
        self.worker = worker

    def _hard_failure(self, name: str) -> None:
        self.hard_failures[name] = self.hard_failures.get(name, 0) + 1
        self.close()


def _statistical_worker(
    module_path: str, skills_path: str | None, connection: object
) -> None:
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stderr(sink):
        try:
            loaded, functions = load_methods(module_path, skills_path=skills_path)
            not_applicable = loaded.NotApplicable
            connection.send(("ready", ""))  # type: ignore[attr-defined]
            while True:
                request = connection.recv()  # type: ignore[attr-defined]
                if request is None:
                    return
                name, history, horizon, frequency = request
                try:
                    raw = functions[name](list(history), horizon, frequency)
                    connection.send((SUCCESS, _valid_forecast(raw, horizon)))  # type: ignore[attr-defined]
                except not_applicable as error:
                    connection.send((NOT_APPLICABLE, str(error)[:200]))  # type: ignore[attr-defined]
                except BaseException as error:
                    connection.send(  # type: ignore[attr-defined]
                        ("failed", f"{type(error).__name__}: {error}"[:200])
                    )
        except BaseException as error:
            try:
                connection.send(("startup_failed", f"{type(error).__name__}: {error}"[:200]))  # type: ignore[attr-defined]
            except BaseException:
                pass


def _stop_statistical_worker(worker: multiprocessing.Process | None) -> None:
    if worker is None:
        return
    if worker.is_alive():
        worker.terminate()
    worker.join(timeout=1.0)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=1.0)


def _valid_forecast(raw: Sequence[float], horizon: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw)
    if len(values) != horizon or not all(map(math.isfinite, values)):
        raise ValueError("candidate returned an invalid forecast")
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
