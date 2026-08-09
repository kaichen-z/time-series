from __future__ import annotations

import ast
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agents import TimeSeriesDiagnosisAgent
from .models import ForecastTask


@dataclass(frozen=True)
class CodeEvolutionCLIConfig:
    binary: str = "codex"
    model: str | None = None
    cache_dir: str = "outputs/codex-cache-code-evolution"
    timeout_seconds: int = 600
    reasoning_effort: str = "high"


class CodeEvolutionCLIClient:
    """Minimal schema-constrained Codex CLI adapter for this experiment."""

    def __init__(self, config: CodeEvolutionCLIConfig | None = None) -> None:
        self.config = config or CodeEvolutionCLIConfig()
        self.cache_dir = Path(self.config.cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.total_latency_seconds = 0.0
        self.last_error: str | None = None

    def stats(self) -> dict[str, float | int | str | None]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "latency_seconds": round(self.total_latency_seconds, 3),
            "last_error": self.last_error,
        }

    def _cache_path(
        self,
        stage: str,
        prompt: str,
        schema: dict[str, Any],
        workspace_files: dict[str, str] | None,
    ) -> Path:
        material = json.dumps(
            {
                "stage": stage,
                "model": self.config.model,
                "reasoning_effort": self.config.reasoning_effort,
                "prompt": prompt,
                "schema": schema,
                "workspace_files": workspace_files or {},
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        digest = hashlib.sha256(material).hexdigest()
        return self.cache_dir / f"{stage}-{digest}.json"

    def complete(
        self,
        stage: str,
        prompt: str,
        schema: dict[str, Any],
        workspace_files: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        cache_path = self._cache_path(stage, prompt, schema, workspace_files)
        if cache_path.exists():
            try:
                self.cache_hits += 1
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

        self.calls += 1
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="drcik-code-evolve-codex-") as directory:
                temporary = Path(directory)
                schema_path = temporary / "schema.json"
                output_path = temporary / "result.json"
                for filename, contents in (workspace_files or {}).items():
                    relative = Path(filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"unsafe Codex workspace path: {filename}")
                    destination = temporary / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(contents, encoding="utf-8")
                schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
                command = [
                    self.config.binary,
                    "exec",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--color",
                    "never",
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "--cd",
                    directory,
                    "-c",
                    f'model_reasoning_effort="{self.config.reasoning_effort}"',
                ]
                if self.config.model:
                    command.extend(("--model", self.config.model))
                command.append("-")
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
                if completed.returncode != 0 or not output_path.exists():
                    detail = completed.stderr.strip().splitlines()
                    raise RuntimeError(detail[-1][:500] if detail else "codex exec failed")
                result = json.loads(output_path.read_text(encoding="utf-8"))
                cache_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self.last_error = None
                return result
        except (
            OSError,
            ValueError,
            subprocess.TimeoutExpired,
            RuntimeError,
            json.JSONDecodeError,
        ) as error:
            self.failures += 1
            self.last_error = f"{type(error).__name__}: {error}"[:700]
            return None
        finally:
            self.total_latency_seconds += time.monotonic() - started


PROGRAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["programs"],
    "properties": {
        "programs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "program_id",
                    "assumption",
                    "failure_condition",
                    "code",
                ],
                "properties": {
                    "program_id": {"type": "string", "maxLength": 80},
                    "assumption": {"type": "string", "maxLength": 1200},
                    "failure_condition": {"type": "string", "maxLength": 1200},
                    "code": {"type": "string", "maxLength": 10000},
                },
            },
        }
    },
}


INITIAL_PROMPT = """You are a numbers-only Coding Agent for time-series forecasting.
Read task.json. Generate diverse, deterministic Python forecast programs from the historical
numbers only. Each program must define exactly:

def forecast(history, horizon, seasonal_period):
    ...
    return values

The result must be a list of exactly horizon finite numbers. Use only the injected math and
statistics modules plus basic Python builtins; do not import anything, access files, use network,
use randomness, call eval/exec, or hard-code the supplied history. Prefer genuinely different
statistical assumptions such as robust local trend, damped seasonality, harmonic structure, or
regime-local level. The assumption must say what must remain true in the future, and the
failure_condition must say when it should fail. Do not output future values directly; output code.
"""


MUTATION_PROMPT = """You are evolving one executable time-series forecasting program.
Read evolution.json. The parent was evaluated on rolling historical holdouts. Produce revised
programs that directly address the reported failure pattern while preserving the required function
signature. Do not copy holdout values, hard-code dates or task values, import modules, access files,
use network, randomness, eval, or exec. Each revision must remain a general forecasting algorithm.
Return code, its updated falsifiable assumption, and its failure condition.
"""


@dataclass(frozen=True)
class CodeEvolutionConfig:
    initial_programs: int = 3
    mutations: int = 2
    validation_folds: int = 3
    validation_horizon: int = 16
    minimum_validation_history: int = 32
    execution_timeout_seconds: float = 3.0
    max_code_characters: int = 10000


@dataclass(frozen=True)
class GeneratedProgram:
    program_id: str
    assumption: str
    failure_condition: str
    code: str
    generation: int
    parent_program_id: str | None = None


@dataclass(frozen=True)
class FoldEvaluation:
    fold: int
    train_end: str
    validation_start: str
    validation_end: str
    mae: float
    scaled_mae: float
    mean_bias: float
    predicted_scale: float
    actual_scale: float


@dataclass(frozen=True)
class ProgramEvaluation:
    program: GeneratedProgram
    valid: bool
    mean_mae: float | None
    mean_scaled_mae: float | None
    folds: tuple[FoldEvaluation, ...]
    forecast: tuple[float, ...] | None
    error: str | None = None


@dataclass(frozen=True)
class CodeEvolutionResult:
    benchmark_id: str
    initial_evaluations: tuple[ProgramEvaluation, ...]
    mutation_evaluations: tuple[ProgramEvaluation, ...]
    selected: ProgramEvaluation
    initial_best: ProgramEvaluation
    backtest_improvement: float
    initial_future_mae: float | None
    selected_future_mae: float | None
    future_mae_improvement: float | None
    codex_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnsafeForecastProgram(ValueError):
    pass


class ForecastProgramSandbox:
    """Execute a deliberately small Python subset in an isolated subprocess."""

    _BANNED_NODES = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.AsyncFunctionDef,
        ast.Global,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.Delete,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
    )
    _SAFE_CALLS = {
        "abs",
        "all",
        "any",
        "bool",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "round",
        "sorted",
        "sum",
        "tuple",
        "zip",
    }
    _SAFE_MODULE_ATTRIBUTES = {
        "math": {
            "cos",
            "exp",
            "floor",
            "isfinite",
            "log",
            "pi",
            "sin",
            "sqrt",
            "tanh",
        },
        "statistics": {
            "fmean",
            "mean",
            "median",
            "pstdev",
            "stdev",
        },
    }
    _SAFE_METHODS = {"append", "copy", "count", "extend", "index", "reverse", "sort"}

    _RUNNER = r'''import json
import math
import statistics
import sys

SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range, "round": round,
    "sorted": sorted, "sum": sum, "tuple": tuple, "zip": zip,
}
payload = json.load(sys.stdin)
namespace = {"__builtins__": SAFE_BUILTINS, "math": math, "statistics": statistics}
exec(payload["code"], namespace, namespace)
result = namespace["forecast"](
    [float(value) for value in payload["history"]],
    int(payload["horizon"]),
    int(payload["seasonal_period"] or 1),
)
if not isinstance(result, (list, tuple)):
    raise TypeError("forecast must return a list or tuple")
values = [float(value) for value in result]
if len(values) != int(payload["horizon"]):
    raise ValueError("forecast length does not match horizon")
if not all(math.isfinite(value) for value in values):
    raise ValueError("forecast contains a non-finite value")
json.dump({"forecast": values}, sys.stdout)
'''

    def __init__(self, timeout_seconds: float = 3.0, max_code_characters: int = 10000) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_code_characters = max_code_characters

    @staticmethod
    def _strip_fence(code: str) -> str:
        stripped = code.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines)
        return stripped.strip()

    def validate(self, code: str) -> str:
        code = self._strip_fence(code)
        if not code or len(code) > self.max_code_characters:
            raise UnsafeForecastProgram("code is empty or too large")
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            raise UnsafeForecastProgram(f"syntax error: {error.msg}") from error
        top_level_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        non_docstring_nodes = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        ]
        if len(top_level_functions) != 1 or top_level_functions[0].name != "forecast":
            raise UnsafeForecastProgram("program must define exactly one forecast function")
        if non_docstring_nodes != top_level_functions:
            raise UnsafeForecastProgram("only the forecast function is allowed at top level")
        arguments = [item.arg for item in top_level_functions[0].args.args]
        if arguments != ["history", "horizon", "seasonal_period"]:
            raise UnsafeForecastProgram("forecast signature must be (history, horizon, seasonal_period)")
        local_function_names = {
            node.name for node in ast.walk(top_level_functions[0]) if isinstance(node, ast.FunctionDef)
        }
        for node in ast.walk(tree):
            if isinstance(node, self._BANNED_NODES):
                raise UnsafeForecastProgram(f"unsupported syntax: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise UnsafeForecastProgram("dunder names are forbidden")
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__"):
                    raise UnsafeForecastProgram("dunder attributes are forbidden")
                if isinstance(node.value, ast.Name) and node.value.id in self._SAFE_MODULE_ATTRIBUTES:
                    if node.attr not in self._SAFE_MODULE_ATTRIBUTES[node.value.id]:
                        raise UnsafeForecastProgram(f"unsafe module attribute: {node.value.id}.{node.attr}")
                elif node.attr not in self._SAFE_METHODS:
                    raise UnsafeForecastProgram(f"unsafe method: {node.attr}")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in self._SAFE_CALLS and node.func.id not in local_function_names:
                        raise UnsafeForecastProgram(f"unsafe call: {node.func.id}")
                elif not isinstance(node.func, ast.Attribute):
                    raise UnsafeForecastProgram("indirect calls are forbidden")
            if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) > 32:
                if all(isinstance(item, ast.Constant) for item in node.elts):
                    raise UnsafeForecastProgram("large literal tables are forbidden")
        return code

    def run(
        self,
        code: str,
        history: tuple[float, ...],
        horizon: int,
        seasonal_period: int | None,
    ) -> tuple[float, ...]:
        code = self.validate(code)
        payload = json.dumps(
            {
                "code": code,
                "history": history,
                "horizon": horizon,
                "seasonal_period": seasonal_period,
            }
        )
        with tempfile.TemporaryDirectory(prefix="drcik-code-program-") as directory:
            runner = Path(directory) / "runner.py"
            runner.write_text(self._RUNNER, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(runner)],
                    input=payload,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout_seconds,
                    check=False,
                    cwd=directory,
                )
            except subprocess.TimeoutExpired as error:
                raise UnsafeForecastProgram("execution timed out") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()
            raise UnsafeForecastProgram(detail[-1][:500] if detail else "execution failed")
        try:
            result = json.loads(completed.stdout)
            return tuple(float(value) for value in result["forecast"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise UnsafeForecastProgram(f"invalid runner output: {error}") from error


class CodexCodeEvolutionAgent:
    """One-generation numbers-only code evolution for a single forecast task."""

    def __init__(
        self,
        client: CodeEvolutionCLIClient,
        config: CodeEvolutionConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or CodeEvolutionConfig()
        self.diagnoser = TimeSeriesDiagnosisAgent()
        self.sandbox = ForecastProgramSandbox(
            timeout_seconds=self.config.execution_timeout_seconds,
            max_code_characters=self.config.max_code_characters,
        )

    def _task_workspace(self, task: ForecastTask) -> dict[str, str]:
        diagnosis = self.diagnoser.diagnose(task)
        return {
            "task.json": json.dumps(
                {
                    "benchmark_id": task.benchmark_id,
                    "frequency": task.frequency,
                    "prediction_length": task.prediction_length,
                    "history_timestamps": task.history_timestamps,
                    "history_values": task.history_values,
                    "numeric_diagnostics": {
                        "trend": diagnosis.trend,
                        "slope_per_step": diagnosis.slope_per_step,
                        "candidate_seasonal_period": diagnosis.seasonal_period,
                        "candidate_seasonal_strength": diagnosis.seasonal_strength,
                        "residual_scale": diagnosis.residual_scale,
                    },
                    "requested_programs": self.config.initial_programs,
                },
                ensure_ascii=False,
                indent=2,
            )
        }

    @staticmethod
    def _parse_programs(
        result: dict[str, Any] | None,
        generation: int,
        parent_program_id: str | None = None,
    ) -> list[GeneratedProgram]:
        if not result:
            return []
        programs = []
        seen = set()
        for index, raw in enumerate(result.get("programs", [])):
            program_id = str(raw.get("program_id", f"g{generation}_{index}"))[:80]
            code = str(raw.get("code", "")).strip()
            if not code or code in seen:
                continue
            seen.add(code)
            programs.append(
                GeneratedProgram(
                    program_id=program_id,
                    assumption=str(raw.get("assumption", "")),
                    failure_condition=str(raw.get("failure_condition", "")),
                    code=code,
                    generation=generation,
                    parent_program_id=parent_program_id,
                )
            )
        return programs

    def generate(self, task: ForecastTask) -> list[GeneratedProgram]:
        result = self.client.complete(
            f"code_evolve_generate_{task.benchmark_id}",
            INITIAL_PROMPT,
            PROGRAM_SCHEMA,
            workspace_files=self._task_workspace(task),
        )
        return self._parse_programs(result, generation=0)[: self.config.initial_programs]

    def _cutoffs(self, task: ForecastTask) -> list[int]:
        horizon = min(self.config.validation_horizon, task.prediction_length)
        return [
            len(task.history_values) - horizon * fold
            for fold in range(1, self.config.validation_folds + 1)
            if len(task.history_values) - horizon * fold
            >= self.config.minimum_validation_history
        ]

    def evaluate(self, task: ForecastTask, program: GeneratedProgram) -> ProgramEvaluation:
        folds = []
        horizon = min(self.config.validation_horizon, task.prediction_length)
        try:
            for fold, cutoff in enumerate(self._cutoffs(task), start=1):
                history = task.history_values[:cutoff]
                truth = task.history_values[cutoff : cutoff + horizon]
                fold_task = ForecastTask(
                    benchmark_id=f"{task.benchmark_id}:code-evolve:{cutoff}",
                    entity_name=task.entity_name,
                    target_name=task.target_name,
                    target_description=task.target_description,
                    frequency=task.frequency,
                    prediction_length=horizon,
                    seasonal_period=None,
                    history_timestamps=task.history_timestamps[:cutoff],
                    history_values=history,
                    future_timestamps=task.history_timestamps[cutoff : cutoff + horizon],
                    future_values=None,
                    documents=(),
                    labels_public=False,
                )
                period = self.diagnoser.diagnose(fold_task).seasonal_period
                prediction = self.sandbox.run(program.code, history, horizon, period)
                mae = statistics.fmean(abs(actual - predicted) for actual, predicted in zip(truth, prediction))
                differences = [abs(history[index] - history[index - 1]) for index in range(1, len(history))]
                scale = max(statistics.fmean(differences), 1e-8)
                folds.append(
                    FoldEvaluation(
                        fold=fold,
                        train_end=task.history_timestamps[cutoff - 1],
                        validation_start=task.history_timestamps[cutoff],
                        validation_end=task.history_timestamps[cutoff + horizon - 1],
                        mae=mae,
                        scaled_mae=mae / scale,
                        mean_bias=statistics.fmean(prediction) - statistics.fmean(truth),
                        predicted_scale=statistics.pstdev(prediction),
                        actual_scale=statistics.pstdev(truth),
                    )
                )
            if not folds:
                raise UnsafeForecastProgram("insufficient history for rolling validation")
            full_period = self.diagnoser.diagnose(task).seasonal_period
            forecast = self.sandbox.run(
                program.code,
                task.history_values,
                task.prediction_length,
                full_period,
            )
            return ProgramEvaluation(
                program=program,
                valid=True,
                mean_mae=statistics.fmean(item.mae for item in folds),
                mean_scaled_mae=statistics.fmean(item.scaled_mae for item in folds),
                folds=tuple(folds),
                forecast=forecast,
            )
        except (UnsafeForecastProgram, OverflowError, ValueError, ZeroDivisionError) as error:
            return ProgramEvaluation(
                program=program,
                valid=False,
                mean_mae=None,
                mean_scaled_mae=None,
                folds=tuple(folds),
                forecast=None,
                error=f"{type(error).__name__}: {error}",
            )

    @staticmethod
    def _best(evaluations: list[ProgramEvaluation]) -> ProgramEvaluation:
        valid = [item for item in evaluations if item.valid and item.mean_scaled_mae is not None]
        if not valid:
            errors = "; ".join(item.error or "invalid" for item in evaluations)
            raise RuntimeError(f"no valid forecast programs: {errors}")
        return min(valid, key=lambda item: (float(item.mean_scaled_mae), item.program.program_id))

    def mutate(
        self,
        task: ForecastTask,
        parent: ProgramEvaluation,
    ) -> list[GeneratedProgram]:
        payload = {
            "task_summary": {
                "history_length": len(task.history_values),
                "frequency": task.frequency,
                "prediction_length": task.prediction_length,
                "detected_period": self.diagnoser.diagnose(task).seasonal_period,
            },
            "parent": {
                "program_id": parent.program.program_id,
                "assumption": parent.program.assumption,
                "failure_condition": parent.program.failure_condition,
                "code": parent.program.code,
            },
            "rolling_evaluation": {
                "mean_mae": parent.mean_mae,
                "mean_scaled_mae": parent.mean_scaled_mae,
                "folds": [asdict(item) for item in parent.folds],
            },
            "requested_mutations": self.config.mutations,
        }
        result = self.client.complete(
            f"code_evolve_mutate_{task.benchmark_id}_{parent.program.program_id}",
            MUTATION_PROMPT,
            PROGRAM_SCHEMA,
            workspace_files={
                "evolution.json": json.dumps(payload, ensure_ascii=False, indent=2)
            },
        )
        return self._parse_programs(
            result,
            generation=parent.program.generation + 1,
            parent_program_id=parent.program.program_id,
        )[: self.config.mutations]

    @staticmethod
    def _future_mae(task: ForecastTask, evaluation: ProgramEvaluation) -> float | None:
        if task.future_values is None or evaluation.forecast is None:
            return None
        return statistics.fmean(
            abs(actual - predicted)
            for actual, predicted in zip(task.future_values, evaluation.forecast)
        )

    def run(self, task: ForecastTask) -> CodeEvolutionResult:
        initial_programs = self.generate(task)
        if not initial_programs:
            raise RuntimeError("Codex returned no initial programs")
        initial_evaluations = [self.evaluate(task, item) for item in initial_programs]
        initial_best = self._best(initial_evaluations)
        mutations = self.mutate(task, initial_best)
        mutation_evaluations = [self.evaluate(task, item) for item in mutations]
        valid_mutations = [item for item in mutation_evaluations if item.valid]
        selected = initial_best
        if valid_mutations:
            mutation_best = self._best(valid_mutations)
            if float(mutation_best.mean_scaled_mae) < float(initial_best.mean_scaled_mae):
                selected = mutation_best
        initial_future_mae = self._future_mae(task, initial_best)
        selected_future_mae = self._future_mae(task, selected)
        return CodeEvolutionResult(
            benchmark_id=task.benchmark_id,
            initial_evaluations=tuple(initial_evaluations),
            mutation_evaluations=tuple(mutation_evaluations),
            selected=selected,
            initial_best=initial_best,
            backtest_improvement=float(initial_best.mean_scaled_mae)
            - float(selected.mean_scaled_mae),
            initial_future_mae=initial_future_mae,
            selected_future_mae=selected_future_mae,
            future_mae_improvement=(
                initial_future_mae - selected_future_mae
                if initial_future_mae is not None and selected_future_mae is not None
                else None
            ),
            codex_stats=self.client.stats(),
        )
