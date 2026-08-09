from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from drcik_agent.code_evolution import (
    CodeEvolutionConfig,
    CodexCodeEvolutionAgent,
    ForecastProgramSandbox,
    UnsafeForecastProgram,
)
from drcik_agent.models import Document, ForecastTask


class FakeCodeClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(max_document_characters=12000)
        self.workspaces: list[dict[str, str]] = []

    def complete(self, stage, _prompt, _schema, workspace_files=None):
        self.workspaces.append(workspace_files or {})
        if stage.startswith("code_evolve_generate"):
            return {
                "programs": [
                    {
                        "program_id": "level",
                        "assumption": "The latest level persists.",
                        "failure_condition": "A local trend continues.",
                        "code": (
                            "def forecast(history, horizon, seasonal_period):\n"
                            "    return [history[-1] for _ in range(horizon)]\n"
                        ),
                    }
                ]
            }
        return {
            "programs": [
                {
                    "program_id": "local_trend",
                    "assumption": "The latest one-step trend persists.",
                    "failure_condition": "The trend reverses after the cutoff.",
                    "code": (
                        "def forecast(history, horizon, seasonal_period):\n"
                        "    slope = history[-1] - history[-2]\n"
                        "    return [history[-1] + slope * (step + 1) "
                        "for step in range(horizon)]\n"
                    ),
                }
            ]
        }

    def stats(self):
        return {"calls": 2, "cache_hits": 0, "failures": 0}


def _linear_task() -> ForecastTask:
    history = tuple(float(value) for value in range(1, 41))
    future = tuple(float(value) for value in range(41, 45))
    return ForecastTask(
        benchmark_id="linear",
        entity_name="Synthetic",
        target_name="value",
        target_description="Linear synthetic series",
        frequency="1 day",
        prediction_length=4,
        seasonal_period=None,
        history_timestamps=tuple(f"2024-01-{value:02d}" for value in range(1, 41)),
        history_values=history,
        future_timestamps=tuple(f"2024-02-{value:02d}" for value in range(1, 5)),
        future_values=future,
        documents=(Document("secret", "future truth and context", role="supporting"),),
        gt_evidence=("secret ground-truth evidence",),
    )


def test_one_generation_selects_mutation_using_backtest() -> None:
    client = FakeCodeClient()
    result = CodexCodeEvolutionAgent(
        client,
        CodeEvolutionConfig(
            initial_programs=1,
            mutations=1,
            validation_folds=2,
            validation_horizon=4,
            minimum_validation_history=16,
        ),
    ).run(_linear_task())

    assert result.initial_best.program.program_id == "level"
    assert result.selected.program.program_id == "local_trend"
    assert result.selected.program.generation == 1
    assert result.backtest_improvement > 0
    assert result.selected_future_mae == 0.0
    assert result.future_mae_improvement is not None
    assert result.future_mae_improvement > 0

    task_payload = json.loads(client.workspaces[0]["task.json"])
    assert "documents" not in task_payload
    assert "future_values" not in task_payload
    assert "gt_evidence" not in task_payload
    mutation_payload = json.loads(client.workspaces[1]["evolution.json"])
    assert "future_values" not in mutation_payload


def test_sandbox_rejects_imports() -> None:
    sandbox = ForecastProgramSandbox()
    with pytest.raises(UnsafeForecastProgram, match="only the forecast function"):
        sandbox.run(
            "import os\n\ndef forecast(history, horizon, seasonal_period):\n"
            "    return [0.0] * horizon\n",
            (1.0, 2.0),
            2,
            None,
        )


def test_sandbox_executes_allowed_forecast() -> None:
    values = ForecastProgramSandbox().run(
        "def forecast(history, horizon, seasonal_period):\n"
        "    return [statistics.mean(history) for _ in range(horizon)]\n",
        (1.0, 3.0),
        3,
        None,
    )
    assert values == (2.0, 2.0, 2.0)


def test_sandbox_allows_local_pure_helpers_and_sort_lambdas() -> None:
    values = ForecastProgramSandbox().run(
        "def forecast(history, horizon, seasonal_period):\n"
        "    def middle(values):\n"
        "        ordered = sorted(values, key=lambda value: abs(value))\n"
        "        return ordered[len(ordered) // 2]\n"
        "    return [middle(history) for _ in range(horizon)]\n",
        (-3.0, 1.0, 2.0),
        2,
        None,
    )
    assert values == (2.0, 2.0)
