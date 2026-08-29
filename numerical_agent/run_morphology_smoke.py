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


_BENCHMARK_ID = re.compile(r'"benchmark_id"\s*:\s*("(?:[^"\\]|\\.)*")')


@dataclass(frozen=True)
class _LoadedTask:
    """One normal history-only Dr-CiK task plus deferred raw task bytes."""

    task: DrCiKTask
    raw_record: bytes
    source: Path


_ARTIFACT_OPTIONS = {
    "reviewed_methods": ("methods_path", "methods.py"),
    "reviewed_skills": ("skills_path", "skills.py"),
    "reviewed_policies": ("policies_path", "policies.py"),
    "reviewed_screening": ("screening_path", "screening.py"),
    "reviewed_decision": ("decision_path", "decision.py"),
}


@dataclass
class _ArtifactSnapshots:
    """Immutable local copies whose hashes exactly bind the executed artifact bytes."""

    fingerprints: Mapping[str, str]
    paths: Mapping[str, Path]
    _texts: Mapping[str, str]
    _temporary: tempfile.TemporaryDirectory

    @classmethod
    def capture(cls, args: argparse.Namespace) -> "_ArtifactSnapshots":
        configured = {
            name: getattr(args, option)
            for name, (option, _filename) in _ARTIFACT_OPTIONS.items()
        }
        if args.llm_backend != "fake":
            missing = [
                f"--{option.replace('_', '-')}"
                for name, (option, _filename) in _ARTIFACT_OPTIONS.items()
                if not configured[name]
            ]
            if missing:
                raise SmokeError(
                    "real mode requires explicit reviewed artifacts: " + ", ".join(missing)
                )
        temporary = tempfile.TemporaryDirectory(prefix="numerical-smoke-artifacts-")
        root = Path(temporary.name)
        paths: dict[str, Path] = {}
        texts: dict[str, str] = {}
        fingerprints: dict[str, str] = {}
        try:
            for name, (option, filename) in _ARTIFACT_OPTIONS.items():
                configured_path = configured[name]
                if configured_path is None:
                    continue
                source = Path(configured_path)
                if not source.is_file():
                    raise FileNotFoundError(
                        f"{name.replace('_', ' ')} path does not exist: {source}"
                    )
                content = source.read_bytes()
                destination = root / filename
                destination.write_bytes(content)
                paths[name] = destination
                texts[name] = content.decode("utf-8")
                fingerprints[name] = hashlib.sha256(content).hexdigest()
        except BaseException:
            temporary.cleanup()
            raise
        if args.llm_backend == "fake" and not paths:
            fingerprints["smoke_artifact_mode"] = "fake_synthetic_only"
        return cls(fingerprints, paths, texts, temporary)

    def text(self, name: str) -> str:
        return self._texts[name]

    def path(self, name: str) -> Path | None:
        return self.paths.get(name)

    def close(self) -> None:
        self._temporary.cleanup()


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
    _reject_output_input_aliases(result_path, task.source, args)
    package_task = Task(
        task.task.task_id,
        task.task.history_values,
        task.task.prediction_length,
        task.task.frequency,
        (),
    )
    artifacts = _ArtifactSnapshots.capture(args)
    runner: _CandidateRunner | None = None
    try:
        portfolio = _portfolio(artifacts)
        runner = _CandidateRunner(args, portfolio, artifacts)
        screening, policies = _policies(runner.statistical_names, portfolio, artifacts)
        decision = _decision_policy(artifacts)
        morphology = _morphology_reasoner(args)
        package = run_numerical_loop(
            package_task,
            screening_policy=screening,
            candidate_runner=runner,
            combined_policies=policies,
            decision_policy=decision,
            hindcast_config=HindcastConfig(folds=args.hindcast_folds, min_successful_folds=2),
            morphology_reasoner=morphology,
            component_fingerprints={**runner.fingerprints, **artifacts.fingerprints},
        )
    finally:
        if runner is not None:
            runner.close()
        artifacts.close()

    future = _future_values_after_freeze(task.raw_record, task.task.prediction_length)
    metrics = _post_freeze_metrics(task.task.history_values, future, package.final_forecast)
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


def _reject_output_input_aliases(
    result_path: Path, task_source: Path, args: argparse.Namespace
) -> None:
    """Keep an overwrite target disjoint from every caller-owned input file."""
    protected = [("task source", task_source)]
    for name, (option, _filename) in _ARTIFACT_OPTIONS.items():
        configured = getattr(args, option, None)
        if configured:
            protected.append((name.replace("reviewed_", "reviewed "), Path(configured)))
    worker_config = getattr(args, "tsfm_workers_config", None)
    if worker_config:
        protected.append(("TSFM worker config", Path(worker_config)))
    for label, source in protected:
        if _paths_alias(result_path, source):
            raise SmokeError(f"results path aliases protected {label}: {source}")


def _paths_alias(left: Path, right: Path) -> bool:
    """Compare resolved entries and inode identity, including symlink/hardlink aliases."""
    try:
        if left.samefile(right):
            return True
    except OSError:
        pass
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return False


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select_one_task(path: Path, task_id: str | None) -> _LoadedTask:
    raw, source = _select_one_record(path, task_id)
    return _to_history_task(raw, source, task_id)


def _select_one_record(path: Path, task_id: str | None) -> tuple[bytes, Path]:
    if not path.exists():
        raise FileNotFoundError(f"task path does not exist: {path}")
    _validate_task_id(task_id)
    if path.is_dir():
        records = tuple(sorted(path.glob("task_*.json")))
        if task_id is not None:
            candidate = path / f"{task_id}.json"
            if not candidate.is_file():
                raise SmokeError(f"--task-id {task_id!r} selected 0 tasks")
            return candidate.read_bytes(), candidate
        if len(records) != 1:
            raise SmokeError("task input contains multiple tasks; pass --task-id before model/runtime work")
        return records[0].read_bytes(), records[0]
    if path.suffix.lower() == ".json":
        return path.read_bytes(), path
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
            return only.encode("utf-8"), path
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
    return selected.encode("utf-8"), path


def _validate_task_id(task_id: str | None) -> None:
    if task_id is None:
        return
    if not task_id or Path(task_id).name != task_id or task_id in {".", ".."}:
        raise SmokeError("--task-id must be an exact basename component")


def _decode_history_record(raw: bytes, source: Path) -> Mapping[str, object]:
    try:
        record = json.loads(_mask_future_values(raw.decode("utf-8")))
    except json.JSONDecodeError as error:
        raise SmokeError(f"task JSON is malformed: {source}") from error
    if not isinstance(record, Mapping):
        raise SmokeError("task path must contain a Dr-CiK task object")
    return record


def _to_history_task(raw: bytes, source: Path, expected_task_id: str | None) -> _LoadedTask:
    record = _decode_history_record(raw, source)
    try:
        series = record.get("series", record)
        metadata = record.get("task_metadata", record)
        if not isinstance(series, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("missing series or task_metadata")
        history = tuple(float(value) for value in series["history_values"])
        task = DrCiKTask(
            task_id=str(record["benchmark_id"]),
            history_values=history,
            prediction_length=int(metadata["prediction_length"]),
            frequency=str(metadata["frequency"]),
            future_values=(),
            seasonal_period=(
                str(metadata["seasonal_period"])
                if metadata.get("seasonal_period") is not None
                else None
            ),
            entity_name="unknown",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SmokeError("task does not match the numeric Dr-CiK task model") from error
    if not task.task_id or not task.history_values or task.prediction_length < 1 or not task.frequency:
        raise SmokeError("task has an empty ID, history, horizon, or frequency")
    if expected_task_id is not None and task.task_id != expected_task_id:
        raise SmokeError("decoded benchmark_id does not match --task-id")
    if not all(math.isfinite(value) for value in task.history_values):
        raise SmokeError("task history contains non-finite numeric values")
    return _LoadedTask(task=task, raw_record=raw, source=source)


def _future_values_after_freeze(
    raw: bytes, prediction_length: int
) -> tuple[float, ...]:
    try:
        record = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise SmokeError("task JSON is malformed") from error
    if not isinstance(record, Mapping):
        raise SmokeError("task path must contain a Dr-CiK task object")
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


def _mask_future_values(raw: str) -> str:
    """Replace future-value spans while decoding property names only."""
    _, spans = _scan_json_value(raw, _skip_json_whitespace(raw, 0))
    parts: list[str] = []
    cursor = 0
    for value_start, value_end in spans:
        parts.extend((raw[cursor:value_start], "null"))
        cursor = value_end
    parts.append(raw[cursor:])
    return "".join(parts)


def _scan_json_value(raw: str, start: int) -> tuple[int, list[tuple[int, int]]]:
    """Return a JSON value's end plus raw spans for decoded future-value keys."""
    start = _skip_json_whitespace(raw, start)
    if start >= len(raw):
        raise SmokeError("task JSON is malformed")
    if raw[start] == "{":
        return _scan_json_object(raw, start)
    if raw[start] == "[":
        return _scan_json_array(raw, start)
    return _json_value_end(raw, start), []


def _scan_json_object(raw: str, start: int) -> tuple[int, list[tuple[int, int]]]:
    index = _skip_json_whitespace(raw, start + 1)
    spans: list[tuple[int, int]] = []
    if index < len(raw) and raw[index] == "}":
        return index + 1, spans
    while True:
        if index >= len(raw) or raw[index] != '"':
            raise SmokeError("task JSON is malformed")
        key_end = _json_string_end(raw, index)
        try:
            key = json.loads(raw[index:key_end])
        except json.JSONDecodeError as error:
            raise SmokeError("task JSON is malformed") from error
        if not isinstance(key, str):
            raise SmokeError("task JSON is malformed")
        index = _skip_json_whitespace(raw, key_end)
        if index >= len(raw) or raw[index] != ":":
            raise SmokeError("task JSON is malformed")
        value_start = _skip_json_whitespace(raw, index + 1)
        if key == "future_values":
            value_end = _json_value_end(raw, value_start)
            spans.append((value_start, value_end))
        else:
            value_end, child_spans = _scan_json_value(raw, value_start)
            spans.extend(child_spans)
        index = _skip_json_whitespace(raw, value_end)
        if index >= len(raw):
            raise SmokeError("task JSON is malformed")
        if raw[index] == "}":
            return index + 1, spans
        if raw[index] != ",":
            raise SmokeError("task JSON is malformed")
        index = _skip_json_whitespace(raw, index + 1)


def _scan_json_array(raw: str, start: int) -> tuple[int, list[tuple[int, int]]]:
    index = _skip_json_whitespace(raw, start + 1)
    spans: list[tuple[int, int]] = []
    if index < len(raw) and raw[index] == "]":
        return index + 1, spans
    while True:
        value_end, child_spans = _scan_json_value(raw, index)
        spans.extend(child_spans)
        index = _skip_json_whitespace(raw, value_end)
        if index >= len(raw):
            raise SmokeError("task JSON is malformed")
        if raw[index] == "]":
            return index + 1, spans
        if raw[index] != ",":
            raise SmokeError("task JSON is malformed")
        index = _skip_json_whitespace(raw, index + 1)


def _skip_json_whitespace(raw: str, index: int) -> int:
    while index < len(raw) and raw[index] in " \r\n\t":
        index += 1
    return index


def _json_string_end(raw: str, start: int) -> int:
    if start >= len(raw) or raw[start] != '"':
        raise SmokeError("task JSON is malformed")
    index = start + 1
    while index < len(raw):
        if raw[index] == "\\":
            index += 2
            continue
        if raw[index] == '"':
            return index + 1
        index += 1
    raise SmokeError("task JSON is malformed")


def _json_value_end(raw: str, start: int) -> int:
    if start >= len(raw):
        raise SmokeError("task JSON is malformed")
    if raw[start] == '"':
        return _json_string_end(raw, start)
    if raw[start] not in "[{":
        index = start
        while index < len(raw) and raw[index] not in ",}]\r\n\t ":
            index += 1
        return index
    stack = [raw[start]]
    index = start + 1
    while index < len(raw) and stack:
        current = raw[index]
        if current == '"':
            index = _json_string_end(raw, index)
            continue
        if current in "[{":
            stack.append(current)
        elif current in "]}":
            opener = stack.pop()
            if (opener, current) not in {("[", "]"), ("{", "}")}:
                raise SmokeError("task JSON is malformed")
        index += 1
    if stack:
        raise SmokeError("task JSON is malformed")
    return index


class _CandidateRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        portfolio: PolicyPortfolio,
        artifacts: _ArtifactSnapshots,
    ) -> None:
        self._unavailable: dict[str, str] = {}
        self._runtimes = _smoke_runtime_registry(args)
        self._statistics = _statistical_functions(
            artifacts.path("reviewed_methods"), artifacts.path("reviewed_skills")
        )
        self.statistical_names = tuple(self._statistics)
        self._tsfm = {policy.name: policy for policy in portfolio.tsfm}
        self.fingerprints = {
            "smoke_statistical_source": artifacts.fingerprints.get(
                "reviewed_methods", "builtin_deterministic_smoke_statistics"
            )
        }

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


def _smoke_runtime_registry(args: argparse.Namespace):
    """Build smoke runtimes while treating absent optional worker venvs as unavailable."""
    worker_config = getattr(args, "tsfm_workers_config", None)
    if not worker_config:
        return _runtime_registry(args)
    try:
        config_bytes = Path(worker_config).read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot load TSFM deployment ({type(error).__name__})"
        ) from None
    with tempfile.TemporaryDirectory(prefix="numerical-smoke-workers-") as temporary:
        snapshot = Path(temporary) / "workers.json"
        snapshot.write_bytes(config_bytes)
        snapshot_args = _args_with_worker_config(args, str(snapshot))
        try:
            return _runtime_registry(snapshot_args)
        except ValueError as error:
            if " interpreter does not exist" not in str(error):
                raise
            missing_interpreter_error = error
        filtered, removed = _filter_absent_worker_environments(config_bytes)
        if not removed:
            raise missing_interpreter_error
        if filtered is None:
            return _runtime_registry(
                _args_with_worker_config(args, None, clear_acknowledgements=True)
            )
        snapshot.write_bytes(filtered)
        return _runtime_registry(snapshot_args)


def _args_with_worker_config(
    args: argparse.Namespace,
    worker_config: str | None,
    *,
    clear_acknowledgements: bool = False,
) -> argparse.Namespace:
    values = vars(args).copy()
    values["tsfm_workers_config"] = worker_config
    if clear_acknowledgements:
        values["acknowledged_model_licenses"] = ""
    return argparse.Namespace(**values)


def _filter_absent_worker_environments(config_bytes: bytes) -> tuple[bytes | None, bool]:
    """Remove only absent interpreter paths; shared deployment validation handles the rest."""
    try:
        payload = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return config_bytes, False
    if not isinstance(payload, dict) or not isinstance(payload.get("environments"), dict):
        return config_bytes, False
    environments = payload["environments"]
    filtered = {
        name: entry
        for name, entry in environments.items()
        if not (
            isinstance(name, str)
            and isinstance(entry, dict)
            and isinstance(entry.get("interpreter"), str)
            and Path(entry["interpreter"]).expanduser().is_absolute()
            and not Path(entry["interpreter"]).expanduser().exists()
        )
    }
    if len(filtered) == len(environments):
        return config_bytes, False
    if not filtered:
        return None, True
    copied = dict(payload)
    copied["environments"] = filtered
    return json.dumps(copied, sort_keys=True, separators=(",", ":")).encode("utf-8"), True


def _statistical_functions(
    path: Path | None, skills_path: Path | None
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

    _module, functions = load_methods(path, skills_path=skills_path)
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


def _portfolio(artifacts: _ArtifactSnapshots) -> PolicyPortfolio:
    source = artifacts.path("reviewed_policies")
    if source is None:
        return PolicyPortfolio.flagship5()
    return read_policy_file(source)


def _policies(
    statistical_names: Sequence[str],
    portfolio: PolicyPortfolio,
    artifacts: _ArtifactSnapshots,
) -> tuple[ScreeningPolicy, tuple[CombinedPolicy, ...]]:
    leaves = tuple(dict.fromkeys((*statistical_names, *(item.name for item in portfolio.tsfm))))
    combined = tuple(policy for policy in portfolio.combined if set(policy.parents) <= set(leaves))
    if artifacts.path("reviewed_screening") is not None:
        screening = parse_screening_source(artifacts.text("reviewed_screening"))
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


def _decision_policy(artifacts: _ArtifactSnapshots) -> DecisionPolicy:
    source = artifacts.path("reviewed_decision")
    if source is None:
        return DecisionPolicy(assumption_guidance_enabled=True)
    declared = _screening_hash_from_decision(artifacts.text("reviewed_decision"))
    if declared != artifacts.fingerprints.get("reviewed_screening"):
        raise SmokeError("Decision SCREENING_POLICY_HASH does not bind the supplied screening artifact")
    return parse_decision_source(artifacts.text("reviewed_decision"))


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

        _instruction, _separator, context = messages[0]["content"].rpartition("\n")
        initial = json.loads(context)
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
        "task_id": task.task.task_id,
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


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SmokeError, ValueError) as error:
        sys.stderr.write(f"error: {error}\n")
        raise SystemExit(2)
