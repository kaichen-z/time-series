"""Numbers-only, falsifiable evolution of executable forecasting skills."""
from __future__ import annotations

import json
import statistics
import uuid
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from evolving_loop.coding_agent.skill_library import Skill, SkillLibrary
from evolving_loop.data import Task
from common.sandbox import SandboxResult, run_forecast_code
from common.llm import JsonExtractionError, LLMClient, parse_json_object
from common.metrics import smape

CodingSetting = Literal["llm_only", "statistics", "tsfm", "combined"]


class NumericForecaster(Protocol):
    def forecast(
        self, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]: ...

STATISTICAL_SKILL_DICTIONARY = """Available statistical ideas (choose only when justified):
- robust local level: median/recent trimmed level for outlier resistance
- local linear trend: recent slope with damping for non-stationary series
- seasonal naive: repeat an empirically supported lag
- seasonal trend: combine a repeated seasonal profile with a damped level drift
- moving-average residual: forecast a smoothed level and decay the last residual
- Fourier/harmonic extrapolation: use only when repeated cycles survive holdouts
Every method is a hypothesis, not a guaranteed rule. State how historical holdouts could falsify it.
"""

GENERATION_PROMPT = """You are the open-ended numbers-only Coding Meta-Agent in a time-series harness.
You may see historical numbers, horizon, frequency, and optionally a reusable skill summary.
You must not request or infer documents, retrieved evidence, ground-truth evidence, or future values.

Design any numerical forecasting framework that can be expressed inside the executable contract:
single models, adaptive model selection, decomposition, change-point logic, ensembles, internal
validation, or a newly invented method are all allowed. Do not assume the supplied method dictionary
is exhaustive. Generate multiple structurally diverse, falsifiable implementations. Every program
must define exactly:
    def forecast(history: list[float], horizon: int, frequency: str) -> list[float]
It must return exactly horizon finite numbers. Allowed imports are numpy, math, statistics,
itertools, functools, and collections. Do not access files/network, use randomness, eval/exec,
or hard-code the supplied series.

Return exactly one JSON object:
{"programs": [{"name": "short_snake_case", "description": "when to use it",
"assumption": "a falsifiable statement about the numeric process",
"failure_condition": "an observable condition under which it should fail",
"code": "def forecast(...): ..."}]}
"""

REVISION_PROMPT = """You are the inner Coding Harness Engineer.
The parent numerical framework was evaluated on rolling historical holdouts. You may rewrite the
entire algorithm, create a different internal model-selection or ensemble framework, or generate
multiple competing descendants. Use only the fold scores, execution errors, and historical numbers;
do not use future labels or textual context. Each descendant must state a falsifiable assumption and
why it addresses the observed failure. Preserve
the required forecast(history, horizon, frequency) signature and return exactly one JSON object:
{"programs": [{"name": "short_snake_case", "description": "when to use it",
"assumption": "falsifiable numeric assumption", "failure_condition": "when it fails",
"code": "def forecast(...): ..."}]}
"""


@dataclass(frozen=True)
class ForecastProgram:
    name: str
    description: str
    assumption: str
    failure_condition: str
    code: str
    generation: int = 0
    source: str = "generated"


@dataclass(frozen=True)
class ValidatedProgram:
    program: ForecastProgram
    forecast: tuple[float, ...]
    hindcast_smape: float
    fold_scores: tuple[float, ...]
    fold_errors: tuple[str, ...]
    sandbox_result: SandboxResult


@dataclass(frozen=True)
class CodingEvolutionResult:
    candidates: tuple[ValidatedProgram, ...]
    selected: ValidatedProgram
    initial_best: ValidatedProgram
    repeat_last_hindcast_smape: float
    improvement: float
    saved_skill_name: str | None


@dataclass(frozen=True)
class CodingEvolutionConfig:
    setting: CodingSetting = "statistics"
    initial_programs: int = 3
    mutations: int = 1
    mutation_children: int = 1
    validation_folds: int = 3
    validation_horizon: int = 8
    minimum_validation_history: int = 16
    minimum_library_improvement: float = 0.0


class CodingEvolutionAgent:
    """Generate, hindcast, revise, and retain only validated numbers-only skills."""

    def __init__(
        self,
        llm: LLMClient,
        library: SkillLibrary | None = None,
        config: CodingEvolutionConfig | None = None,
        tsfm_forecaster: NumericForecaster | None = None,
        *,
        generation_prompt: str = GENERATION_PROMPT,
        revision_prompt: str = REVISION_PROMPT,
    ) -> None:
        self.llm = llm
        self.library = library
        self.config = config or CodingEvolutionConfig()
        self.tsfm_forecaster = tsfm_forecaster
        self.generation_prompt = generation_prompt
        self.revision_prompt = revision_prompt

    def run_task(self, task: Task) -> CodingEvolutionResult:
        programs = [] if self.config.setting == "tsfm" else self._library_programs()
        if self.config.setting != "tsfm":
            programs.extend(self._generate(task))
        validated = [candidate for program in programs if (candidate := self._validate(task, program))]
        if self.config.setting in {"tsfm", "combined"}:
            if self.tsfm_forecaster is None:
                raise ValueError(f"setting={self.config.setting!r} requires a TSFM forecaster")
            tsfm_candidate = self._validate_tsfm(task)
            if tsfm_candidate is not None:
                validated.append(tsfm_candidate)
        if not validated:
            fallback = self._fallback_program()
            validated = [self._validate(task, fallback)]
        validated = [item for item in validated if item is not None]
        initial_best = min(validated, key=lambda item: item.hindcast_smape)
        all_candidates = list(validated)

        generated_candidates = [
            item for item in validated if item.program.source not in {"tsfm", "fallback"}
        ]
        parent = min(generated_candidates, key=lambda item: item.hindcast_smape) if generated_candidates else None
        for generation in range(1, self.config.mutations + 1):
            if parent is None:
                break
            mutations = self._mutate(task, parent, generation)
            children = [candidate for program in mutations if (candidate := self._validate(task, program))]
            all_candidates.extend(children)
            if children:
                child = min(children, key=lambda item: item.hindcast_smape)
                if child.hindcast_smape < parent.hindcast_smape:
                    parent = child

        selected = min(all_candidates, key=lambda item: item.hindcast_smape)
        baseline_score = self._repeat_last_hindcast(task)
        saved_name = None
        if (
            self.library is not None
            and selected.program.source in {"generated", "mutation"}
            and selected.hindcast_smape + self.config.minimum_library_improvement < baseline_score
        ):
            saved_name = selected.program.name
            self.library.add(
                Skill(
                    skill_id=str(uuid.uuid4()),
                    name=selected.program.name,
                    description=selected.program.description,
                    code=selected.program.code,
                    created_from_task=task.task_id,
                    assumption=selected.program.assumption,
                    failure_condition=selected.program.failure_condition,
                    validation_score=selected.hindcast_smape,
                )
            )
        return CodingEvolutionResult(
            candidates=tuple(sorted(all_candidates, key=lambda item: item.hindcast_smape)),
            selected=selected,
            initial_best=initial_best,
            repeat_last_hindcast_smape=baseline_score,
            improvement=initial_best.hindcast_smape - selected.hindcast_smape,
            saved_skill_name=saved_name,
        )

    def _library_programs(self) -> list[ForecastProgram]:
        if self.library is None:
            return []
        return [
            ForecastProgram(
                name=skill.name,
                description=skill.description,
                assumption=skill.assumption or skill.description,
                failure_condition=skill.failure_condition or "Historical holdout performance degrades.",
                code=skill.code,
                source="library",
            )
            for skill in self.library.all()
        ]

    def _generate(self, task: Task) -> list[ForecastProgram]:
        setting = self.config.setting
        guidance = ""
        if setting in {"statistics", "combined"}:
            guidance += "\n" + STATISTICAL_SKILL_DICTIONARY
        if setting in {"tsfm", "combined"}:
            guidance += (
                "\nA separately executed TSFM candidate may be supplied by the harness. "
                "Your code should add transparent alternatives, not imitate neural weights."
            )
        user = self._numeric_payload(task)
        if self.library is not None:
            user += "\nReusable skill summaries:\n" + self.library.list_for_prompt()
        return self._call_programs(
            self.generation_prompt + guidance,
            user,
            generation=0,
            limit=self.config.initial_programs,
        )

    def _mutate(
        self, task: Task, parent: ValidatedProgram, generation: int
    ) -> list[ForecastProgram]:
        payload = {
            "task": json.loads(self._numeric_payload(task)),
            "parent": {
                "name": parent.program.name,
                "description": parent.program.description,
                "assumption": parent.program.assumption,
                "failure_condition": parent.program.failure_condition,
                "code": parent.program.code,
            },
            "historical_validation": {
                "mean_smape": parent.hindcast_smape,
                "fold_scores": list(parent.fold_scores),
                "execution_errors": list(parent.fold_errors),
            },
        }
        programs = self._call_programs(
            self.revision_prompt,
            json.dumps(payload, ensure_ascii=False),
            generation=generation,
            limit=self.config.mutation_children,
        )
        return [replace(program, source="mutation") for program in programs]

    @staticmethod
    def _numeric_payload(task: Task) -> str:
        return json.dumps(
            {
                "task_id": task.task_id,
                "history_values": list(task.history_values),
                "horizon": task.prediction_length,
                "frequency": task.frequency,
                "seasonal_period": task.seasonal_period,
            },
            ensure_ascii=False,
        )

    def _call_programs(
        self, system: str, user: str, *, generation: int, limit: int
    ) -> list[ForecastProgram]:
        response = self.llm.complete(
            system=system,
            messages=[{"role": "user", "content": user}],
            # Candidate diversity comes from the required multi-program output.
            # Deterministic decoding keeps parent/child policy evaluation comparable.
            temperature=0.0,
        )
        try:
            payload = parse_json_object(response.text)
        except JsonExtractionError:
            return []
        records = payload.get("programs", [])
        if not isinstance(records, list):
            return []
        programs = []
        for index, record in enumerate(records[:limit]):
            if not isinstance(record, dict):
                continue
            fields = ("name", "description", "assumption", "failure_condition", "code")
            if not all(isinstance(record.get(field), str) and record[field].strip() for field in fields):
                continue
            programs.append(
                ForecastProgram(
                    name=record["name"],
                    description=record["description"],
                    assumption=record["assumption"],
                    failure_condition=record["failure_condition"],
                    code=record["code"],
                    generation=generation,
                )
            )
        return programs

    def _validate(self, task: Task, program: ForecastProgram) -> ValidatedProgram | None:
        result = run_forecast_code(
            program.code,
            list(task.history_values),
            task.prediction_length,
            task.frequency,
        )
        if not result.ok or result.forecast is None:
            return None
        fold_scores = []
        errors = []
        for train, target in self._folds(task):
            fold = run_forecast_code(
                program.code, list(train), len(target), task.frequency
            )
            if not fold.ok or fold.forecast is None:
                errors.append(fold.error or "unknown sandbox failure")
                fold_scores.append(200.0)
            else:
                fold_scores.append(smape(list(target), list(fold.forecast)))
        if not fold_scores:
            fold_scores = [200.0]
            errors.append("insufficient_history_for_hindcast")
        return ValidatedProgram(
            program=program,
            forecast=result.forecast,
            hindcast_smape=statistics.fmean(fold_scores),
            fold_scores=tuple(fold_scores),
            fold_errors=tuple(errors),
            sandbox_result=result,
        )

    def _validate_tsfm(self, task: Task) -> ValidatedProgram | None:
        try:
            forecast = self.tsfm_forecaster.forecast(
                task.history_values, task.prediction_length, task.frequency
            )
            fold_scores = []
            for train, target in self._folds(task):
                prediction = self.tsfm_forecaster.forecast(train, len(target), task.frequency)
                fold_scores.append(smape(list(target), list(prediction)))
        except Exception:
            return None
        if len(forecast) != task.prediction_length:
            return None
        result = SandboxResult(ok=True, forecast=forecast, error=None, duration_ms=0.0)
        return ValidatedProgram(
            program=ForecastProgram(
                name="tsfm_backbone",
                description="Zero-shot time-series foundation-model forecast.",
                assumption="Patterns learned across many series transfer to this numeric history.",
                failure_condition="The task contains a regime or mechanism absent from the numeric input.",
                code="# External TSFM adapter; not a generated reusable Python skill.",
                source="tsfm",
            ),
            forecast=forecast,
            hindcast_smape=statistics.fmean(fold_scores) if fold_scores else 200.0,
            fold_scores=tuple(fold_scores),
            fold_errors=(() if fold_scores else ("insufficient_history_for_hindcast",)),
            sandbox_result=result,
        )

    def _folds(self, task: Task) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
        history = task.history_values
        horizon = min(self.config.validation_horizon, task.prediction_length)
        horizon = max(1, horizon)
        minimum = max(self.config.minimum_validation_history, horizon * 2)
        possible = max(0, (len(history) - minimum) // horizon + 1)
        count = min(self.config.validation_folds, possible)
        folds = []
        for offset in range(count, 0, -1):
            cutoff = len(history) - offset * horizon
            if cutoff < minimum:
                continue
            folds.append((history[:cutoff], history[cutoff : cutoff + horizon]))
        return folds

    def _repeat_last_hindcast(self, task: Task) -> float:
        scores = []
        for train, target in self._folds(task):
            prediction = [train[-1]] * len(target)
            scores.append(smape(list(target), prediction))
        return statistics.fmean(scores) if scores else 200.0

    @staticmethod
    def _fallback_program() -> ForecastProgram:
        return ForecastProgram(
            name="repeat_last",
            description="Repeat the latest observed value when no generated program is valid.",
            assumption="The local level persists over the forecast horizon.",
            failure_condition="The process changes level, trend, or regime after the cutoff.",
            code=(
                "def forecast(history, horizon, frequency):\n"
                "    value = history[-1] if history else 0.0\n"
                "    return [value for _ in range(horizon)]\n"
            ),
            source="fallback",
        )
