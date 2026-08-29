"""Run one immutable, history-only Numerical morphology forecast as a local smoke test."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from common.data import Task as DrCiKTask
from common.llm import CodexCLIClient, CodexCLIConfig
from common.metrics import drcik_point_metrics, mase
from numerical_agent.evolution import MorphologyReasoner, run_numerical_loop
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_selector import DecisionPolicy, HindcastConfig
from numerical_agent.evolution.portfolio import (
    CombinedPolicy,
    PolicyPortfolio,
    TSFMPolicy,
    forecast_tsfm,
    read_policy_file,
)
from numerical_agent.evolution.screening import (
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
)
from numerical_agent.evolution.screening_evolution import parse_screening_source
from numerical_agent.evolution.selector_evolution import parse_decision_source
from numerical_agent.main import _add_tsfm_runtime_options, _runtime_registry
from numerical_agent.providers import RuntimeUnavailableError


class SmokeError(ValueError):
    """A user-facing smoke input cannot safely produce a forecast."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--llm-backend", choices=("codex", "fake"), default="codex")
    parser.add_argument("--codex-model", default="gpt-5.6-luna")
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--codex-timeout", type=int, default=900)
    parser.add_argument("--codex-cache-dir", default=None)
    parser.add_argument(
        "--methods-path",
        help="explicit reviewed statistical methods.py; omit only for the deterministic smoke leaves",
    )
    parser.add_argument("--skills-path")
    parser.add_argument("--policies-path", help="explicit reviewed policies.py")
    parser.add_argument("--screening-path", help="explicit frozen screening policy source")
    parser.add_argument("--decision-path", help="explicit frozen Decision policy source")
    parser.add_argument("--hindcast-folds", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    _add_tsfm_runtime_options(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_path = _result_path(args.results_path, overwrite=args.overwrite)
    source = Path(args.task_file)
    task = _select_one_task(source, args.task_id)
    package_task = Task(
        task.task_id,
        task.history_values,
        task.prediction_length,
        task.frequency,
        (),
    )
    portfolio = _portfolio(args)
    runner, close_runner = _candidate_runner(args, portfolio)
    try:
        screening, policies = _policies(args, runner.statistical_names, portfolio)
        decision = _decision_policy(args)
        morphology = _morphology_reasoner(args)
        package = run_numerical_loop(
            package_task,
            screening_policy=screening,
            candidate_runner=runner,
            combined_policies=policies,
            decision_policy=decision,
            hindcast_config=HindcastConfig(folds=args.hindcast_folds, min_successful_folds=2),
            morphology_reasoner=morphology,
            component_fingerprints=runner.fingerprints,
        )
    finally:
        close_runner()

    payload = _result_payload(task, package, runner)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def _result_path(value: str, *, overwrite: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SmokeError("results path must be a non-empty file path")
    path = Path(value)
    if path.exists() and path.is_dir():
        raise SmokeError("results path must name a file, not a directory")
    if path.exists() and not overwrite:
        raise FileExistsError(f"results path already exists; pass --overwrite: {path}")
    return path


def _select_one_task(path: Path, task_id: str | None) -> DrCiKTask:
    if not path.exists():
        raise FileNotFoundError(f"task path does not exist: {path}")
    records = _records(path)
    tasks = tuple(_to_drcik_task(record) for record in records)
    if task_id is not None:
        selected = tuple(task for task in tasks if task.task_id == task_id)
        if len(selected) != 1:
            raise SmokeError(f"--task-id {task_id!r} selected {len(selected)} tasks")
        return selected[0]
    if len(tasks) != 1:
        raise SmokeError("task input contains multiple tasks; pass --task-id before model/runtime work")
    return tasks[0]


def _records(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        if path.is_dir():
            raw = [json.loads(item.read_text(encoding="utf-8")) for item in sorted(path.glob("task_*.json"))]
        elif path.suffix.lower() == ".json":
            raw = [json.loads(path.read_text(encoding="utf-8"))]
        else:
            raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise SmokeError(f"task JSON is malformed: {path}") from error
    if not raw or any(not isinstance(item, Mapping) for item in raw):
        raise SmokeError("task path must contain one or more Dr-CiK task objects")
    return tuple(raw)


def _to_drcik_task(record: Mapping[str, object]) -> DrCiKTask:
    try:
        series = record.get("series", record)
        metadata = record.get("task_metadata", record)
        if not isinstance(series, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("missing series or task_metadata")
        history = tuple(float(value) for value in series["history_values"])
        raw_future = series.get("future_values") or ()
        future = () if record.get("labels_public", True) is False or raw_future[:1] == [None] else tuple(float(value) for value in raw_future)
        task = DrCiKTask(
            task_id=str(record["benchmark_id"]),
            history_values=history,
            future_values=future,
            prediction_length=int(metadata["prediction_length"]),
            frequency=str(metadata["frequency"]),
            seasonal_period=(str(metadata["seasonal_period"]) if metadata.get("seasonal_period") is not None else None),
            entity_name="unknown",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SmokeError("task does not match the numeric Dr-CiK task model") from error
    if not task.task_id or not task.history_values or task.prediction_length < 1 or not task.frequency:
        raise SmokeError("task has an empty ID, history, horizon, or frequency")
    if not all(math.isfinite(value) for value in (*task.history_values, *task.future_values)):
        raise SmokeError("task contains non-finite numeric values")
    if task.future_values and len(task.future_values) != task.prediction_length:
        raise SmokeError("task future values must match prediction_length")
    return task


class _CandidateRunner:
    def __init__(self, args: argparse.Namespace, portfolio: PolicyPortfolio) -> None:
        self._unavailable: dict[str, str] = {}
        self._runtimes = _runtime_registry(args)
        self._statistics = _statistical_functions(args.methods_path, args.skills_path)
        self.statistical_names = tuple(self._statistics)
        self._tsfm = {policy.name: policy for policy in portfolio.tsfm}
        self.fingerprints = {"smoke_statistical_source": _source_fingerprint(args.methods_path)}

    def __call__(self, name: str, history: tuple[float, ...], horizon: int, frequency: str) -> tuple[float, ...]:
        function = self._statistics.get(name)
        if function is not None:
            return tuple(float(value) for value in function(history, horizon, frequency))
        policy = self._tsfm.get(name)
        if policy is None:
            raise ValueError(f"unknown smoke leaf candidate {name!r}")
        try:
            return forecast_tsfm(policy, history=history, horizon=horizon, frequency=frequency, runtimes=self._runtimes)
        except RuntimeUnavailableError as error:
            self._unavailable[name] = str(error)
            raise

    def unavailable_reason(self, name: str) -> str:
        return self._unavailable.get(name, "candidate did not materialize")

    def close(self) -> None:
        self._runtimes.close()


def _candidate_runner(args: argparse.Namespace, portfolio: PolicyPortfolio) -> tuple[_CandidateRunner, Callable[[], None]]:
    runner = _CandidateRunner(args, portfolio)
    return runner, runner.close


def _statistical_functions(
    path: str | None, skills_path: str | None
) -> dict[str, Callable[[Sequence[float], int, str], Sequence[float]]]:
    if path is None:
        return {
            "naive_last": lambda history, horizon, _frequency: (float(history[-1]),) * horizon,
            "seasonal_naive": _seasonal_naive,
            "holt_damped_trend": _damped_drift,
            "croston_sba": lambda history, horizon, _frequency: (statistics.fmean(history[-min(8, len(history)):]),) * horizon,
            "robust_loess_trend": _damped_drift,
            "median_seasonal_profile_forecast": lambda history, horizon, _frequency: (statistics.median(history[-min(8, len(history)):]),) * horizon,
        }
    from numerical_agent.evolution.execution import load_methods

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"methods path does not exist: {source}")
    _module, functions = load_methods(source, skills_path=skills_path)
    return {
        name: (lambda history, horizon, frequency, function=function: function(list(history), horizon, frequency))
        for name, function in functions.items()
    }


def _seasonal_naive(history: Sequence[float], horizon: int, _frequency: str) -> tuple[float, ...]:
    period = min(7, len(history))
    return tuple(float(history[-period + index % period]) for index in range(horizon))


def _damped_drift(history: Sequence[float], horizon: int, _frequency: str) -> tuple[float, ...]:
    slope = (float(history[-1]) - float(history[0])) / max(1, len(history) - 1)
    return tuple(float(history[-1]) + slope * (index + 1) * 0.8 for index in range(horizon))


def _portfolio(args: argparse.Namespace) -> PolicyPortfolio:
    if not args.policies_path:
        return PolicyPortfolio.flagship5()
    source = Path(args.policies_path)
    if not source.is_file():
        raise FileNotFoundError(f"policies path does not exist: {source}")
    return read_policy_file(source)


def _policies(
    args: argparse.Namespace,
    statistical_names: Sequence[str],
    portfolio: PolicyPortfolio,
) -> tuple[ScreeningPolicy, tuple[CombinedPolicy, ...]]:
    leaves = tuple(dict.fromkeys((*statistical_names, *(item.name for item in portfolio.tsfm))))
    combined = tuple(policy for policy in portfolio.combined if set(policy.parents) <= set(leaves))
    if args.screening_path:
        source = Path(args.screening_path)
        if not source.is_file():
            raise FileNotFoundError(f"screening path does not exist: {source}")
        screening = parse_screening_source(source.read_text(encoding="utf-8"))
        unknown = {entry.name for entry in screening.entries} - set((*leaves, *(item.name for item in combined)))
        if unknown:
            raise SmokeError(f"screening references candidates unavailable to supplied artifacts: {sorted(unknown)!r}")
        return screening, combined
    entries = tuple(
        ScreeningEntry(name, "statistical", "keep", ApplicabilityPolicy(), "deterministic smoke statistical leaf")
        for name in statistical_names
    ) + tuple(
        ScreeningEntry(item.name, "tsfm", "keep", ApplicabilityPolicy(), "reviewed TSFM leaf")
        for item in portfolio.tsfm
    ) + tuple(
        ScreeningEntry(item.name, "combined", "keep", ApplicabilityPolicy(), "reviewed Combined policy")
        for item in combined
    )
    return ScreeningPolicy(entries, (statistical_names[0],)), combined


def _decision_policy(args: argparse.Namespace) -> DecisionPolicy:
    if not args.decision_path:
        return DecisionPolicy(assumption_guidance_enabled=True)
    source = Path(args.decision_path)
    if not source.is_file():
        raise FileNotFoundError(f"decision path does not exist: {source}")
    return parse_decision_source(source.read_text(encoding="utf-8"))


class _FakeMorphologyClient:
    def __init__(self) -> None:
        self._turn = 0

    def complete(self, *, system: str, messages: list[dict], temperature: float = 0.0):
        del system, temperature
        from common.llm import LLMResponse

        initial = json.loads(messages[0]["content"])
        history = initial["history"]
        active = initial["active_candidates"]
        recent_start = max(1, len(history) - max(2, len(history) // 3))
        actions = (
            {"action": "tool", "call_id": "full", "tool": "detect_trend", "window": {"start": 0, "end": len(history)}},
            {"action": "tool", "call_id": "recent", "tool": "detect_trend", "window": {"start": recent_start, "end": len(history)}},
            {"action": "final", "short_term": "history-only local level", "long_term": "history-only level", "assumptions": [{"assumption_id": "smoke_level", "kind": "level", "claim": "level is stable", "failure_condition": "level shifts", "supporting_call_ids": ["full", "recent"], "candidate_names": [active[0]["name"]], "prior_confidence": 0.8}]},
        )
        response = actions[self._turn]
        self._turn += 1
        return LLMResponse(json.dumps(response, sort_keys=True))


def _morphology_reasoner(args: argparse.Namespace) -> MorphologyReasoner:
    if args.llm_backend == "fake":
        return MorphologyReasoner(_FakeMorphologyClient())
    return MorphologyReasoner(CodexCLIClient(CodexCLIConfig(
        model=args.codex_model,
        reasoning_effort=args.codex_reasoning_effort,
        timeout_seconds=args.codex_timeout,
        cache_dir=args.codex_cache_dir,
    )))


def _result_payload(task: DrCiKTask, package, runner: _CandidateRunner) -> dict[str, object]:
    alternatives = {item.name: item for item in package.ranked_alternatives}
    active = []
    available = []
    unavailable = []
    for name in package.active_candidate_names:
        diagnostic = package.candidate_diagnostics.get(name)
        summary = {
            "name": name,
            "family": diagnostic.family if diagnostic is not None else "unknown",
            "history_only_diagnostics": _diagnostic_payload(diagnostic),
        }
        active.append(summary)
        if name in alternatives:
            available.append({**summary, "forecast": list(alternatives[name].forecast)})
        else:
            unavailable.append({**summary, "reason": runner.unavailable_reason(name)})
    selected = package.selection_decision
    metrics = _post_freeze_metrics(task, package.final_forecast)
    rejected_counts = dict(sorted(Counter(package.rejected_assumptions.values()).items()))
    return {
        "task_id": task.task_id,
        "selected": {
            "recipe": asdict(selected.arithmetic) if selected.arithmetic is not None else None,
            "methods": list(selected.selected),
            "weights": list(selected.weights),
            "reason_codes": list(selected.reason_codes),
        },
        "final_forecast": list(package.final_forecast),
        "protected_baseline": _ranked_payload(package.protected_baseline),
        "accepted_assumptions": [dict(item) for item in package.retrieval_handoff],
        "rejected_assumption_reason_counts": rejected_counts,
        "selected_history_only_diagnostics": _diagnostic_payload(package.candidate_diagnostics.get(selected.selected[0])),
        "baseline_history_only_diagnostics": _diagnostic_payload(package.protected_baseline.diagnostics),
        "candidates": {"active": active, "available": available, "unavailable": unavailable},
        "morphology": {
            "call_status": "completed" if package.morphology_card is not None else "fallback",
            "fallback_reason": package.fallback_reason,
            "tool_calls": (len(package.morphology_card.tool_calls) if package.morphology_card else 0),
        },
        "component_fingerprints": dict(package.component_fingerprints),
        "freeze": {
            "forecast_frozen_before_labels": True,
            "post_freeze_trusted_diagnostics": metrics,
        },
    }


def _diagnostic_payload(value) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "eligible": value.eligible,
        "reason_code": value.reason_code,
        "successful_folds": value.successful_folds,
        "median_mase": _finite_or_none(value.median_mase),
        "recent_mase": _finite_or_none(value.recent_mase),
        "worst_mase": _finite_or_none(value.worst_mase),
        "fold_statuses": [item.status for item in value.folds],
    }


def _ranked_payload(value) -> dict[str, object]:
    return {"name": value.name, "family": value.family, "forecast": list(value.forecast), "rank": value.rank}


def _post_freeze_metrics(task: DrCiKTask, forecast: Sequence[float]) -> dict[str, float] | None:
    if not task.future_values:
        return None
    point = drcik_point_metrics(task.future_values, forecast)
    return {
        "mae": float(point["mae"]),
        "mase": mase(list(task.future_values), list(forecast), list(task.history_values)),
        "smae": float(point["smae"]),
        "srmse": float(point["srmse"]),
    }


def _source_fingerprint(path: str | None) -> str:
    if path is None:
        return "builtin_deterministic_smoke_statistics"
    return str(Path(path).resolve())


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SmokeError, ValueError) as error:
        sys.stderr.write(f"error: {error}\n")
        raise SystemExit(2)
