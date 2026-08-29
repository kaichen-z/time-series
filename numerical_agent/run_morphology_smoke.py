"""Run one immutable, history-only Numerical morphology forecast as a local smoke test."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

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


_BENCHMARK_ID = re.compile(r'"benchmark_id"\s*:\s*("(?:[^"\\]|\\.)*")')


@dataclass(frozen=True)
class _LoadedTask:
    """One selected task with its history decoded and future left unread until freeze."""

    task_id: str
    history_values: tuple[float, ...]
    prediction_length: int
    frequency: str
    record: Mapping[str, object]


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
    artifact_fingerprints = _validate_artifact_bundle(args)
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
        decision = _decision_policy(args, artifact_fingerprints.get("reviewed_screening"))
        morphology = _morphology_reasoner(args)
        package = run_numerical_loop(
            package_task,
            screening_policy=screening,
            candidate_runner=runner,
            combined_policies=policies,
            decision_policy=decision,
            hindcast_config=HindcastConfig(folds=args.hindcast_folds, min_successful_folds=2),
            morphology_reasoner=morphology,
            component_fingerprints={**runner.fingerprints, **artifact_fingerprints},
        )
    finally:
        close_runner()

    future = _future_values_after_freeze(task.record, task.prediction_length)
    metrics = _post_freeze_metrics(task.history_values, future, package.final_forecast)
    payload = _result_payload(task, package, runner, metrics)
    _write_result(result_path, payload, overwrite=args.overwrite)
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


def _write_result(path: Path, payload: Mapping[str, object], *, overwrite: bool) -> None:
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _validate_artifact_bundle(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "reviewed_methods": args.methods_path,
        "reviewed_skills": args.skills_path,
        "reviewed_policies": args.policies_path,
        "reviewed_screening": args.screening_path,
        "reviewed_decision": args.decision_path,
    }
    if args.llm_backend != "fake":
        missing = [f"--{key.removeprefix('reviewed_')}-path" for key, value in paths.items() if not value]
        if missing:
            raise SmokeError("real mode requires explicit reviewed artifacts: " + ", ".join(missing))
    present = {
        key: Path(value)
        for key, value in paths.items()
        if value is not None
    }
    for key, path in present.items():
        if not path.is_file():
            raise FileNotFoundError(f"{key.replace('_', ' ')} path does not exist: {path}")
    if args.llm_backend == "fake" and not present:
        return {"smoke_artifact_mode": "fake_synthetic_only"}
    return {key: _sha256(path) for key, path in sorted(present.items())}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select_one_task(path: Path, task_id: str | None) -> _LoadedTask:
    return _to_history_task(_select_one_record(path, task_id))


def _select_one_record(path: Path, task_id: str | None) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"task path does not exist: {path}")
    if path.is_dir():
        records = tuple(sorted(path.glob("task_*.json")))
        if task_id is not None:
            candidate = path / f"{task_id}.json"
            if not candidate.is_file():
                raise SmokeError(f"--task-id {task_id!r} selected 0 tasks")
            return _read_record(candidate)
        if len(records) != 1:
            raise SmokeError("task input contains multiple tasks; pass --task-id before model/runtime work")
        return _read_record(records[0])
    if path.suffix.lower() == ".json":
        record = _read_record(path)
        if task_id is not None and record.get("benchmark_id") != task_id:
            raise SmokeError(f"--task-id {task_id!r} selected 0 tasks")
        return record
    if task_id is None:
        only: str | None = None
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
                    only = line
                    if count > 1:
                        break
        if count == 1 and only is not None:
            return _decode_record(only, path)
        raise SmokeError("task input contains multiple tasks; pass --task-id before model/runtime work")
    selected: str | None = None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = _BENCHMARK_ID.search(line)
            if match is not None and json.loads(match.group(1)) == task_id:
                count += 1
                selected = line
    if count != 1 or selected is None:
        raise SmokeError(f"--task-id {task_id!r} selected {count} tasks")
    return _decode_record(selected, path)


def _read_record(path: Path) -> Mapping[str, object]:
    try:
        return _decode_record(path.read_text(encoding="utf-8"), path)
    except OSError:
        raise


def _decode_record(value: str, source: Path) -> Mapping[str, object]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise SmokeError(f"task JSON is malformed: {source}") from error
    if not isinstance(raw, Mapping):
        raise SmokeError("task path must contain a Dr-CiK task object")
    return raw


def _to_history_task(record: Mapping[str, object]) -> _LoadedTask:
    try:
        series = record.get("series", record)
        metadata = record.get("task_metadata", record)
        if not isinstance(series, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("missing series or task_metadata")
        history = tuple(float(value) for value in series["history_values"])
        task = _LoadedTask(
            task_id=str(record["benchmark_id"]),
            history_values=history,
            prediction_length=int(metadata["prediction_length"]),
            frequency=str(metadata["frequency"]),
            record=record,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SmokeError("task does not match the numeric Dr-CiK task model") from error
    if not task.task_id or not task.history_values or task.prediction_length < 1 or not task.frequency:
        raise SmokeError("task has an empty ID, history, horizon, or frequency")
    if not all(math.isfinite(value) for value in task.history_values):
        raise SmokeError("task history contains non-finite numeric values")
    return task


def _future_values_after_freeze(
    record: Mapping[str, object], prediction_length: int
) -> tuple[float, ...]:
    if record.get("labels_public", True) is False:
        return ()
    series = record.get("series", record)
    if not isinstance(series, Mapping):
        raise SmokeError("task does not match the numeric Dr-CiK task model")
    raw = series.get("future_values") or ()
    if not raw or tuple(raw[:1]) == (None,):
        return ()
    try:
        future = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as error:
        raise SmokeError("task future contains non-numeric values") from error
    if len(future) != prediction_length or not all(math.isfinite(value) for value in future):
        raise SmokeError("task future must be finite and match prediction_length")
    return future


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


def _decision_policy(args: argparse.Namespace, screening_hash: str | None) -> DecisionPolicy:
    if not args.decision_path:
        return DecisionPolicy(assumption_guidance_enabled=True)
    source = Path(args.decision_path)
    if not source.is_file():
        raise FileNotFoundError(f"decision path does not exist: {source}")
    declared = _screening_hash_from_decision(source.read_text(encoding="utf-8"))
    if screening_hash is None or declared != screening_hash:
        raise SmokeError("Decision SCREENING_POLICY_HASH does not bind the supplied screening artifact")
    return parse_decision_source(source.read_text(encoding="utf-8"))


def _screening_hash_from_decision(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise SmokeError("Decision source does not parse") from error
    values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "SCREENING_POLICY_HASH"
    ]
    if len(values) != 1:
        raise SmokeError("Decision source must define exactly one SCREENING_POLICY_HASH")
    try:
        value = ast.literal_eval(values[0])
    except (TypeError, ValueError) as error:
        raise SmokeError("Decision SCREENING_POLICY_HASH must be a literal") from error
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SmokeError("Decision SCREENING_POLICY_HASH must be a SHA-256 digest")
    return value


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


def _result_payload(
    task: _LoadedTask, package, runner: _CandidateRunner, metrics: dict[str, float] | None
) -> dict[str, object]:
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


def _post_freeze_metrics(
    history: Sequence[float], future: Sequence[float], forecast: Sequence[float]
) -> dict[str, float] | None:
    if not future:
        return None
    try:
        point = drcik_point_metrics(future, forecast)
        values = {
            "mae": float(point["mae"]),
            "mase": mase(list(future), list(forecast), list(history)),
            "smae": float(point["smae"]),
            "srmse": float(point["srmse"]),
        }
    except (ArithmeticError, ValueError) as error:
        raise SmokeError("post-freeze metrics contain non-finite values") from error
    if not all(math.isfinite(value) for value in values.values()):
        raise SmokeError("post-freeze metrics contain non-finite values")
    return values


def _source_fingerprint(path: str | None) -> str:
    if path is None:
        return "builtin_deterministic_smoke_statistics"
    return _sha256(Path(path))


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SmokeError, ValueError) as error:
        sys.stderr.write(f"error: {error}\n")
        raise SystemExit(2)
