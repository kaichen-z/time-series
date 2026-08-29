"""Import an evolved methods module once and score every method against every task."""
from __future__ import annotations

import importlib.util
import math
import multiprocessing
import statistics
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from common.metrics import drcik_point_metrics, mae, mase, smape

from .analysis_skills import DEFAULT_SKILLS_PATH, skill_namespace


NOT_APPLICABLE = "not_applicable"
CRASHED = "crashed"
INVALID = "invalid"
SUCCESS = "success"


def require_unique_task_ids(tasks: Sequence["Task"]) -> None:
    """Reject duplicate task IDs before an evaluation can collapse outcome rows."""
    duplicates = _duplicates(task.task_id for task in tasks)
    if duplicates:
        raise ValueError("duplicate task IDs are not allowed: " + ", ".join(duplicates))


def require_unique_outcome_keys(outcomes: Sequence["Outcome"]) -> None:
    """Reject duplicate method/task rows before constructing an outcome map."""
    duplicates = _duplicates(f"{row.method}/{row.task_id}" for row in outcomes)
    if duplicates:
        raise ValueError("duplicate outcome keys are not allowed: " + ", ".join(duplicates))


def _duplicates(values: Sequence[str] | object) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:  # type: ignore[union-attr]
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


@dataclass(frozen=True)
class Task:
    """One label-free forecasting input plus the trusted future used only for scoring."""

    task_id: str
    history: tuple[float, ...]
    horizon: int
    frequency: str
    future: tuple[float, ...]

    def characteristics(self) -> tuple[str, ...]:
        """Describe the series from history only, never from the future values."""
        return derive_characteristics(self.history, self.horizon, self.frequency)


@dataclass(frozen=True)
class Outcome:
    """What one method did on one task."""

    method: str
    task_id: str
    status: str
    smape: float | None = None
    mae: float | None = None
    mase: float | None = None
    detail: str = ""
    forecast: tuple[float, ...] = ()
    smae: float | None = None
    srmse: float | None = None
    smae_raw: float | None = None
    srmse_raw: float | None = None
    smae_clipped: bool | None = None
    srmse_clipped: bool | None = None


@dataclass(frozen=True)
class MethodReport:
    """Aggregate behavior of one method, with skipped tasks kept apart from failures."""

    method: str
    total: int
    success: int
    not_applicable: int
    crashed: int
    invalid: int
    mean_smae: float | None
    mean_srmse: float | None
    mean_smape: float | None
    mean_mae: float | None
    mean_mase: float | None
    coverage: float
    by_characteristic: Mapping[str, float] = field(default_factory=dict)
    by_characteristic_smae: Mapping[str, float] = field(default_factory=dict)
    by_characteristic_srmse: Mapping[str, float] = field(default_factory=dict)
    by_characteristic_mae: Mapping[str, float] = field(default_factory=dict)
    by_characteristic_mase: Mapping[str, float] = field(default_factory=dict)
    sample_failures: tuple[str, ...] = ()


def derive_characteristics(
    history: Sequence[float], horizon: int, frequency: str
) -> tuple[str, ...]:
    """Bucket a series by traits the model can use to write 'when to use' docstrings."""
    values = [float(value) for value in history]
    length = len(values)
    tags = [f"frequency:{frequency}", f"history:{_bucket(length, (48, 168, 512))}"]
    tags.append(f"horizon:{_bucket(horizon, (8, 24, 96))}")
    zeros = sum(1 for value in values if value == 0.0)
    zero_ratio = zeros / length
    tags.append("intermittent" if zero_ratio > 0.3 else "dense")
    tags.append("many_zeros" if zero_ratio > 0.3 else ("no_zeros" if zeros == 0 else "some_zeros"))
    tags.append("nonnegative" if min(values) >= 0.0 else "signed")
    integer_ratio = sum(abs(value - round(value)) <= 1e-8 for value in values) / length
    tags.append("integer_valued" if integer_ratio >= 0.98 else "continuous_valued")
    if length >= 8:
        first, second = values[: length // 2], values[length // 2 :]
        spread = statistics.pstdev(values) or 1.0
        drift = abs(statistics.fmean(second) - statistics.fmean(first)) / spread
        tags.append("trending" if drift > 0.5 else "flat")
    return tuple(tags)


def _bucket(value: int, edges: Sequence[int]) -> str:
    for edge in edges:
        if value <= edge:
            return f"le_{edge}"
    return f"gt_{edges[-1]}"


def _skills_path(methods_path: Path, requested: str | Path | None) -> Path:
    if requested is not None:
        return Path(requested).resolve()
    sibling = methods_path.with_name("skills.py")
    return sibling if sibling.is_file() else DEFAULT_SKILLS_PATH


def load_methods(
    path: str | Path, *, skills_path: str | Path | None = None
) -> tuple[object, dict[str, object]]:
    """Import the module once and return it with its forecasting functions by name."""
    source = Path(path).resolve()
    from .module import read_module

    method_names = read_module(source).names()
    spec = importlib.util.spec_from_file_location(f"evolved_methods_{source.stem}", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import a methods module from {source}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(skill_namespace(_skills_path(source, skills_path)))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    functions = {name: getattr(module, name) for name in method_names}
    if not functions:
        raise ImportError(f"{source} defines no forecasting functions")
    return module, functions


def run_module(
    path: str | Path,
    tasks: Sequence[Task],
    *,
    time_budget_s: float = 20.0,
    isolated: bool = False,
    timeout_circuit_breaker: int = 2,
    worker_startup_timeout_s: float = 10.0,
    skills_path: str | Path | None = None,
) -> tuple[tuple[Outcome, ...], tuple[MethodReport, ...]]:
    """Run every method over every task in this process and summarize each method."""
    if isolated:
        # Parse without importing in the parent. Import can execute arbitrary top-level work, so
        # it belongs inside the bounded worker-startup phase rather than the trusted evaluator.
        from .module import read_module

        names = read_module(path).names()
        outcomes = _run_isolated(
            Path(path).resolve(),
            names,
            tasks,
            time_budget_s=time_budget_s,
            timeout_circuit_breaker=timeout_circuit_breaker,
            worker_startup_timeout_s=worker_startup_timeout_s,
            skills_path=_skills_path(Path(path).resolve(), skills_path),
        )
    else:
        module, functions = load_methods(path, skills_path=skills_path)
        not_applicable = getattr(module, "NotApplicable", None)
        if not isinstance(not_applicable, type) or not issubclass(not_applicable, Exception):
            raise ImportError("methods module must define a NotApplicable exception")
        names = tuple(functions)
        outcomes = []
        for name, function in functions.items():
            for task in tasks:
                outcomes.append(
                    _run_one(name, function, task, not_applicable, time_budget_s)
                )
    reports = tuple(
        _report(name, [o for o in outcomes if o.method == name], tasks)
        for name in names
    )
    return tuple(outcomes), reports


def run_method(
    method: object,
    tasks: Sequence[Task],
    *,
    time_budget_s: float = 20.0,
    isolated: bool = False,
    timeout_circuit_breaker: int = 2,
    worker_startup_timeout_s: float = 10.0,
    skills_path: str | Path | None = None,
) -> tuple[tuple[Outcome, ...], MethodReport]:
    """Execute one parsed method through the same trusted module evaluator."""
    from .module import MODULE_HEADER, Method

    if not isinstance(method, Method):
        raise TypeError("method must be a parsed Method")
    with tempfile.TemporaryDirectory(prefix="method-outcome-") as directory:
        path = Path(directory) / "methods.py"
        path.write_text(
            MODULE_HEADER.rstrip("\n") + "\n\n\n" + method.source.strip("\n") + "\n",
            encoding="utf-8",
        )
        selected_skills = Path(skills_path) if skills_path is not None else DEFAULT_SKILLS_PATH
        temporary_skills = Path(directory) / "skills.py"
        temporary_skills.write_text(selected_skills.read_text(encoding="utf-8"), encoding="utf-8")
        outcomes, reports = run_module(
            path,
            tasks,
            time_budget_s=time_budget_s,
            isolated=isolated,
            timeout_circuit_breaker=timeout_circuit_breaker,
            worker_startup_timeout_s=worker_startup_timeout_s,
            skills_path=temporary_skills,
        )
    if len(reports) != 1 or reports[0].method != method.name:
        raise RuntimeError(f"single-method execution returned unexpected reports for {method.name}")
    return outcomes, reports[0]


def reports_from_outcomes(
    method_names: Sequence[str],
    outcomes: Sequence[Outcome],
    tasks: Sequence[Task],
) -> tuple[MethodReport, ...]:
    """Aggregate an outcome matrix reconstructed from cached method executions."""
    return tuple(
        _report(name, [outcome for outcome in outcomes if outcome.method == name], tasks)
        for name in method_names
    )


def _run_isolated(
    path: Path,
    names: Sequence[str],
    tasks: Sequence[Task],
    *,
    time_budget_s: float,
    timeout_circuit_breaker: int,
    worker_startup_timeout_s: float,
    skills_path: Path,
) -> list[Outcome]:
    """Run one persistent subprocess per method, containing native crashes and hangs."""
    if time_budget_s <= 0:
        raise ValueError("time_budget_s must be positive")
    if timeout_circuit_breaker < 1:
        raise ValueError("timeout_circuit_breaker must be at least one")
    if worker_startup_timeout_s <= 0:
        raise ValueError("worker_startup_timeout_s must be positive")

    context = multiprocessing.get_context("spawn")
    outcomes: list[Outcome] = []
    for name in names:
        parent, worker, startup_status, startup_detail = _start_isolated_worker(
            context,
            path,
            name,
            worker_startup_timeout_s,
            skills_path,
        )
        if startup_detail:
            outcomes.extend(
                Outcome(name, task.task_id, startup_status, detail=startup_detail)
                for task in tasks
            )
            continue
        assert parent is not None
        timeouts = 0
        try:
            for index, task in enumerate(tasks):
                try:
                    parent.send(task)
                except (BrokenPipeError, EOFError, OSError):
                    outcomes.append(
                        Outcome(name, task.task_id, CRASHED, detail=_worker_exit(worker))
                    )
                    _stop_worker(worker)
                    parent.close()
                    if index + 1 < len(tasks):
                        parent, worker, status, detail = _start_isolated_worker(
                            context, path, name, worker_startup_timeout_s, skills_path
                        )
                        if detail:
                            outcomes.extend(
                                Outcome(name, pending.task_id, status, detail=detail)
                                for pending in tasks[index + 1 :]
                            )
                            break
                        assert parent is not None
                    continue

                if parent.poll(time_budget_s):
                    try:
                        outcome = parent.recv()
                    except (EOFError, OSError):
                        outcome = Outcome(
                            name, task.task_id, CRASHED, detail=_worker_exit(worker)
                        )
                        _stop_worker(worker)
                        parent.close()
                        if index + 1 < len(tasks):
                            parent, worker, status, detail = _start_isolated_worker(
                                context, path, name, worker_startup_timeout_s, skills_path
                            )
                            if detail:
                                outcomes.append(outcome)
                                outcomes.extend(
                                    Outcome(name, pending.task_id, status, detail=detail)
                                    for pending in tasks[index + 1 :]
                                )
                                break
                            assert parent is not None
                    outcomes.append(outcome)
                    if outcome.status != INVALID or "hard timeout" not in outcome.detail:
                        timeouts = 0
                    continue

                if worker.exitcode is not None:
                    outcomes.append(
                        Outcome(name, task.task_id, CRASHED, detail=_worker_exit(worker))
                    )
                else:
                    timeouts += 1
                    outcomes.append(
                        Outcome(
                            name,
                            task.task_id,
                            INVALID,
                            detail=f"hard timeout after {time_budget_s:g}s",
                        )
                    )
                _stop_worker(worker)
                parent.close()

                remaining = tasks[index + 1 :]
                if timeouts >= timeout_circuit_breaker:
                    outcomes.extend(
                        Outcome(
                            name,
                            pending.task_id,
                            INVALID,
                            detail=(
                                "hard timeout circuit breaker after "
                                f"{timeouts} timeouts"
                            ),
                        )
                        for pending in remaining
                    )
                    break
                if remaining:
                    parent, worker, status, detail = _start_isolated_worker(
                        context, path, name, worker_startup_timeout_s, skills_path
                    )
                    if detail:
                        outcomes.extend(
                            Outcome(name, pending.task_id, status, detail=detail)
                            for pending in remaining
                        )
                        break
                    assert parent is not None
        finally:
            if parent is not None:
                try:
                    parent.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                parent.close()
            _stop_worker(worker)
    return outcomes


def _start_isolated_worker(
    context: object,
    path: Path,
    name: str,
    startup_timeout_s: float,
    skills_path: Path,
) -> tuple[object | None, multiprocessing.Process, str, str]:
    """Start one worker and wait separately for its bounded import/ready phase."""
    parent, child = context.Pipe()  # type: ignore[attr-defined]
    worker = context.Process(  # type: ignore[attr-defined]
        target=_isolated_method_worker,
        args=(str(path), str(skills_path), name, child),
        daemon=True,
    )
    worker.start()
    child.close()
    if not parent.poll(startup_timeout_s):
        if worker.exitcode is None:
            status = INVALID
            detail = f"worker startup timeout after {startup_timeout_s:g}s"
        else:
            status = CRASHED
            detail = _worker_exit(worker)
        parent.close()
        _stop_worker(worker)
        return None, worker, status, detail
    try:
        ready = parent.recv()
    except (EOFError, OSError):
        detail = _worker_exit(worker)
        parent.close()
        _stop_worker(worker)
        return None, worker, CRASHED, detail
    if ready != "worker_ready":
        parent.close()
        _stop_worker(worker)
        return None, worker, CRASHED, "worker returned an invalid startup handshake"
    return parent, worker, "", ""


def _isolated_method_worker(
    path: str, skills_path: str, name: str, connection: object
) -> None:
    """Load and repeatedly execute exactly one method inside a disposable process."""
    module, functions = load_methods(path, skills_path=skills_path)
    not_applicable = getattr(module, "NotApplicable")
    function = functions[name]
    connection.send("worker_ready")  # type: ignore[attr-defined]
    while True:
        task = connection.recv()  # type: ignore[attr-defined]
        if task is None:
            return
        connection.send(  # type: ignore[attr-defined]
            _run_one(name, function, task, not_applicable, float("inf"))
        )


def _worker_exit(worker: multiprocessing.Process) -> str:
    worker.join(timeout=0.05)
    return f"worker exited with code {worker.exitcode}"


def _stop_worker(worker: multiprocessing.Process) -> None:
    if worker.is_alive():
        worker.terminate()
    worker.join(timeout=1.0)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=1.0)


def _run_one(
    name: str,
    function: object,
    task: Task,
    not_applicable: type,
    time_budget_s: float,
) -> Outcome:
    start = time.monotonic()
    try:
        raw = function(list(task.history), task.horizon, task.frequency)  # type: ignore[operator]
    except not_applicable as exc:  # type: ignore[misc]
        return Outcome(name, task.task_id, NOT_APPLICABLE, detail=str(exc)[:200])
    except BaseException as exc:
        # Anything other than NotApplicable is a defect, not modesty about applicability.
        return Outcome(name, task.task_id, CRASHED, detail=f"{type(exc).__name__}: {exc}"[:200])
    if time.monotonic() - start > time_budget_s:
        return Outcome(name, task.task_id, INVALID, detail=f"exceeded {time_budget_s}s")
    try:
        forecast = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        return Outcome(name, task.task_id, INVALID, detail=f"unreadable forecast: {exc}"[:200])
    if len(forecast) != task.horizon:
        return Outcome(
            name, task.task_id, INVALID,
            detail=f"returned {len(forecast)} values, expected {task.horizon}",
        )
    if not all(math.isfinite(value) for value in forecast):
        return Outcome(name, task.task_id, INVALID, detail="returned a non-finite value")
    truth = list(task.future)
    history = list(task.history)
    point = drcik_point_metrics(truth, forecast)
    return Outcome(
        name, task.task_id, SUCCESS,
        smae=float(point["smae"]), srmse=float(point["srmse"]),
        smae_raw=float(point["smae_raw"]), srmse_raw=float(point["srmse_raw"]),
        smae_clipped=bool(point["smae_clipped"]), srmse_clipped=bool(point["srmse_clipped"]),
        smape=smape(truth, forecast), mae=mae(truth, forecast),
        mase=mase(truth, forecast, history),
        forecast=tuple(forecast),
    )


def _report(
    name: str, outcomes: Sequence[Outcome], tasks: Sequence[Task]
) -> MethodReport:
    by_id = {task.task_id: task for task in tasks}
    scored = [o for o in outcomes if o.status == SUCCESS]
    counts = Counter(o.status for o in outcomes)

    grouped: dict[str, list[float]] = {}
    grouped_smae: dict[str, list[float]] = {}
    grouped_srmse: dict[str, list[float]] = {}
    grouped_mae: dict[str, list[float]] = {}
    grouped_mase: dict[str, list[float]] = {}
    for outcome in scored:
        for tag in by_id[outcome.task_id].characteristics():
            grouped.setdefault(tag, []).append(float(outcome.smape))
            grouped_smae.setdefault(tag, []).append(float(outcome.smae))
            grouped_srmse.setdefault(tag, []).append(float(outcome.srmse))
            grouped_mae.setdefault(tag, []).append(float(outcome.mae))
            grouped_mase.setdefault(tag, []).append(float(outcome.mase))

    # Deduplicated with a dict, not a set: set order is not reproducible across runs.
    ordered_failures = tuple(
        dict.fromkeys(
            o.detail for o in outcomes if o.status in (CRASHED, INVALID) and o.detail
        )
    )[:3]

    total = len(outcomes)
    return MethodReport(
        method=name,
        total=total,
        success=len(scored),
        not_applicable=counts.get(NOT_APPLICABLE, 0),
        crashed=counts.get(CRASHED, 0),
        invalid=counts.get(INVALID, 0),
        mean_smae=statistics.fmean(o.smae for o in scored) if scored else None,
        mean_srmse=statistics.fmean(o.srmse for o in scored) if scored else None,
        mean_smape=statistics.fmean(o.smape for o in scored) if scored else None,
        mean_mae=statistics.fmean(o.mae for o in scored) if scored else None,
        mean_mase=statistics.fmean(o.mase for o in scored) if scored else None,
        coverage=len(scored) / total if total else 0.0,
        by_characteristic={
            tag: statistics.fmean(values) for tag, values in sorted(grouped.items())
        },
        by_characteristic_smae={
            tag: statistics.fmean(values) for tag, values in sorted(grouped_smae.items())
        },
        by_characteristic_srmse={
            tag: statistics.fmean(values) for tag, values in sorted(grouped_srmse.items())
        },
        by_characteristic_mae={
            tag: statistics.fmean(values) for tag, values in sorted(grouped_mae.items())
        },
        by_characteristic_mase={
            tag: statistics.fmean(values) for tag, values in sorted(grouped_mase.items())
        },
        sample_failures=ordered_failures,
    )


def report_payload(reports: Sequence[MethodReport]) -> list[dict[str, object]]:
    """Render reports as the JSON the evolution prompt and the metrics file both use."""
    return [
        {
            "method": report.method,
            "mean_smae": report.mean_smae,
            "mean_srmse": report.mean_srmse,
            "mean_smape": report.mean_smape,
            "mean_mae": report.mean_mae,
            "mean_mase": report.mean_mase,
            "success": report.success,
            "total": report.total,
            "coverage": round(report.coverage, 4),
            "not_applicable": report.not_applicable,
            "crashed": report.crashed,
            "invalid": report.invalid,
            "by_characteristic": {
                tag: round(value, 4) for tag, value in report.by_characteristic.items()
            },
            "by_characteristic_smae": {
                tag: round(value, 4) for tag, value in report.by_characteristic_smae.items()
            },
            "by_characteristic_srmse": {
                tag: round(value, 4) for tag, value in report.by_characteristic_srmse.items()
            },
            "by_characteristic_mae": {
                tag: round(value, 4) for tag, value in report.by_characteristic_mae.items()
            },
            "by_characteristic_mase": {
                tag: round(value, 4) for tag, value in report.by_characteristic_mase.items()
            },
            "sample_failures": list(report.sample_failures),
        }
        for report in reports
    ]
