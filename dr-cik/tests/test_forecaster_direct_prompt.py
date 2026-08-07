"""DirectPromptForecaster: independent per-sample calls, partial-failure retry, degraded fallback, and padding."""

from __future__ import annotations

import json

from dr_cik.forecasters.direct_prompt import DirectPromptConfig, DirectPromptForecaster, load_prior_context
from dr_cik.llm import FakeLLMClient

from .conftest import requires_sample


def _forecast_json(horizon: int, base: float = 10.0) -> str:
    return json.dumps({"forecast": [base + step for step in range(horizon)]})


@requires_sample
def test_direct_prompt_issues_one_call_per_sample(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    horizon = view.prediction_length
    responses = [json.dumps({"forecast": [10.0 + i + step for step in range(horizon)]}) for i in range(25)]
    llm = FakeLLMClient(responses=responses)
    forecaster = DirectPromptForecaster(llm, DirectPromptConfig(model_id="fake-model", num_samples=25))

    forecast = forecaster.forecast(view, context_text="some DR-synthesized context")

    assert len(forecast.samples) == 25
    assert all(len(sample) == horizon for sample in forecast.samples)
    assert forecast.method == "direct-prompt:fake-model(S=25)"
    assert len(llm.calls) == 25  # one independent sampled call per trajectory, no retry needed


@requires_sample
def test_direct_prompt_retries_only_the_missing_count(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    horizon = view.prediction_length
    good = _forecast_json(horizon)
    bad = "not json at all"
    # first pass (25 calls): 20 good, 5 bad -> missing=5 -> retry pass issues exactly 5 more calls, all good
    llm = FakeLLMClient(responses=[good] * 20 + [bad] * 5 + [good] * 5)
    forecaster = DirectPromptForecaster(llm, DirectPromptConfig(model_id="fake-model", num_samples=25))

    forecast = forecaster.forecast(view, context_text="")

    assert len(forecast.samples) == 25
    assert forecast.method == "direct-prompt:fake-model(S=25)"
    assert len(llm.calls) == 30  # 25 first pass + 5 retry (not another full 25)
    assert "ONLY the JSON object" in llm.calls[25]["messages"][0]["content"]


@requires_sample
def test_direct_prompt_degrades_when_no_call_parses(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    llm = FakeLLMClient(responses=lambda system, messages: "not json")
    forecaster = DirectPromptForecaster(llm, DirectPromptConfig(model_id="fake-model", num_samples=25))

    forecast = forecaster.forecast(view, context_text="")

    assert len(forecast.samples) == 25
    assert all(len(sample) == view.prediction_length for sample in forecast.samples)
    assert forecast.method == "direct-prompt:fake-model:degraded-fallback(S=25)"


@requires_sample
def test_direct_prompt_truncates_off_by_one_long_forecast(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    horizon = view.prediction_length
    oversized = json.dumps({"forecast": [1.0 + step for step in range(horizon + 1)]})
    llm = FakeLLMClient(responses=lambda system, messages: oversized)
    forecaster = DirectPromptForecaster(llm, DirectPromptConfig(model_id="fake-model", num_samples=25))

    forecast = forecaster.forecast(view, context_text="")

    assert len(forecast.samples) == 25
    assert all(len(sample) == horizon for sample in forecast.samples)
    assert forecast.method == "direct-prompt:fake-model(S=25)"


@requires_sample
def test_direct_prompt_drops_short_forecast(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    horizon = view.prediction_length
    too_short = json.dumps({"forecast": [1.0] * (horizon - 1)})
    llm = FakeLLMClient(responses=lambda system, messages: too_short)
    forecaster = DirectPromptForecaster(llm, DirectPromptConfig(model_id="fake-model", num_samples=25))

    forecast = forecaster.forecast(view, context_text="")

    assert forecast.method == "direct-prompt:fake-model:degraded-fallback(S=25)"


@requires_sample
def test_direct_prompt_pads_when_still_short_after_retry(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    horizon = view.prediction_length
    good = _forecast_json(horizon)
    calls = {"n": 0}

    def responder(system: str, messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        return good if calls["n"] <= 3 else "nope"

    llm = FakeLLMClient(responses=responder)
    forecaster = DirectPromptForecaster(llm, DirectPromptConfig(model_id="fake-model", num_samples=25))

    forecast = forecaster.forecast(view, context_text="")

    assert len(forecast.samples) == 25
    assert forecast.method == "direct-prompt:fake-model:padded(S=25,model_rows=3)"


@requires_sample
def test_direct_prompt_is_deterministic_given_same_seed(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    llm_a = FakeLLMClient(responses=lambda system, messages: "nope")
    forecaster_a = DirectPromptForecaster(llm_a, DirectPromptConfig(model_id="fake-model", num_samples=25, seed=7))
    llm_b = FakeLLMClient(responses=lambda system, messages: "nope")
    forecaster_b = DirectPromptForecaster(llm_b, DirectPromptConfig(model_id="fake-model", num_samples=25, seed=7))

    forecast_a = forecaster_a.forecast(view, context_text="")
    forecast_b = forecaster_b.forecast(view, context_text="")

    assert forecast_a.samples == forecast_b.samples


def test_load_prior_context_reads_run_report(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {"benchmark_id": "t1", "report_markdown": "History rose sharply.", "evidence": [{"claim": "claim one", "source_doc_ids": ["d1"]}]},
        {"benchmark_id": "t2", "report_markdown": "No strong signal.", "evidence": []},
    ]
    (run_dir / "run_report.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    context = load_prior_context(run_dir)

    assert set(context) == {"t1", "t2"}
    assert "History rose sharply." in context["t1"]
    assert "claim one" in context["t1"]
    assert context["t2"].startswith("No strong signal.")
