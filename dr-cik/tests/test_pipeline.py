"""Both orchestration paths end-to-end against the real sample tasks, with every dependency faked.

Asserts the output files match the shapes SUBMISSION.md requires.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import torch

from dr_cik.forecasters.chronos import ChronosConfig, ChronosForecaster
from dr_cik.forecasters.direct_prompt import DirectPromptConfig, DirectPromptForecaster
from dr_cik.llm import FakeLLMClient
from dr_cik.pipeline import RunConfig, build_pipeline, run_direct_prompt, write_direct_prompt_outputs, write_outputs

from .conftest import requires_sample

_OVERSIZED_HORIZON = 500  # comfortably >= every sample task's prediction_length; _extract_forecast truncates to fit


class _FT(Enum):
    SAMPLES = "s"
    QUANTILES = "q"


class _FakePipeline:
    forecast_type = _FT.QUANTILES

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()

    def predict_quantiles(self, context, prediction_length, quantile_levels):
        count = len(quantile_levels)
        quantiles = torch.randn(1, prediction_length, count) * 5 + 400.0
        return quantiles, quantiles[:, :, count // 2]


def _fake_forecaster(num_samples: int) -> ChronosForecaster:
    fake_module = SimpleNamespace(BaseChronosPipeline=_FakePipeline, ForecastType=_FT)
    return ChronosForecaster(ChronosConfig(num_samples=num_samples), runtime_module=fake_module)


def _drbench_responder(system: str, messages: list[dict[str, str]]) -> str:
    content = messages[0]["content"]
    if content.startswith("Document"):
        document_id = content.split("Document ")[1].split(":")[0]
        return json.dumps({"document_id": document_id, "relevant": True, "brief": "b", "key_claims": []})
    return json.dumps({"report": "a report", "evidence": [{"claim": "a claim", "source_doc_ids": []}]})


def _forecast_json(horizon: int = _OVERSIZED_HORIZON) -> str:
    return json.dumps({"forecast": [10.0 + step for step in range(horizon)]})


def _direct_prompt_forecaster() -> DirectPromptForecaster:
    llm = FakeLLMClient(responses=lambda system, messages: _forecast_json())
    return DirectPromptForecaster(llm, DirectPromptConfig(model_id="fake-model", num_samples=25))


# --- agent + Chronos path ---------------------------------------------------------------


@requires_sample
def test_agent_pipeline_produces_submission_shaped_files(sample_tasks, tmp_path: Path) -> None:
    config = RunConfig(agent="drbench", num_samples=12, crps_sample_size=6, drbench_top_k=2, judge_enabled=False)
    llm = FakeLLMClient(responses=_drbench_responder)
    pipeline = build_pipeline(config, llm=llm, judge=None, forecaster=_fake_forecaster(12))

    results = pipeline.run_many(sample_tasks)
    write_outputs(results, tmp_path)

    forecasts = [json.loads(line) for line in (tmp_path / "forecasts.jsonl").read_text().splitlines()]
    deep_research = [json.loads(line) for line in (tmp_path / "deep_research.jsonl").read_text().splitlines()]
    run_report = [json.loads(line) for line in (tmp_path / "run_report.jsonl").read_text().splitlines()]
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert len(forecasts) == len(sample_tasks) == len(deep_research) == len(run_report)
    for row, task in zip(forecasts, sample_tasks):
        assert row["benchmark_id"] == task.benchmark_id
        assert len(row["samples"]) == 12
        assert all(len(sample) == task.prediction_length for sample in row["samples"])
    for row in deep_research:
        assert set(row.keys()) == {"benchmark_id", "cited_document_ids", "evidence"}
    assert summary["num_tasks"] == len(sample_tasks)
    assert summary["agent"] == "drbench"
    assert "smae" in summary["mean_metrics"]
    assert "evidence_recall" not in summary["mean_metrics"]  # judge_enabled=False -> always None -> excluded


# --- Direct-Prompt path -----------------------------------------------------------------


@requires_sample
def test_direct_prompt_scores_labeled_tasks(sample_tasks) -> None:
    context_by_id = {task.benchmark_id: "some context" for task in sample_tasks}

    results = run_direct_prompt(sample_tasks, _direct_prompt_forecaster(), context_by_id, crps_sample_size=10)

    assert len(results) == len(sample_tasks)
    for result, task in zip(results, sample_tasks):
        assert result.benchmark_id == task.benchmark_id
        assert len(result.forecast.samples) == 25
        if task.future_values is not None:
            assert result.metrics["smae"] is not None
            assert result.metrics["scrps"] is not None


@requires_sample
def test_direct_prompt_output_schema(sample_tasks, tmp_path: Path) -> None:
    results = run_direct_prompt(sample_tasks, _direct_prompt_forecaster(), {}, crps_sample_size=10)

    write_direct_prompt_outputs(results, tmp_path, model_id="fake-model", from_run_dir="some/run/dir")

    forecasts = [json.loads(line) for line in (tmp_path / "forecasts.jsonl").read_text().splitlines()]
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert len(forecasts) == len(sample_tasks)
    for row in forecasts:
        assert set(row.keys()) == {"benchmark_id", "samples"}
        assert len(row["samples"]) == 25
    assert summary["model_id"] == "fake-model"
    assert summary["from_run_dir"] == "some/run/dir"
    assert "smae" in summary["mean_metrics"]
    # `methods` is how a reader spots degraded/padded runs; a clean run must say so.
    assert summary["methods"] == ["direct-prompt:fake-model(S=25)"]
