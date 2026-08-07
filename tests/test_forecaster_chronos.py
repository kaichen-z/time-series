"""Chronos forecaster against fake pipelines covering both the SAMPLES and QUANTILES APIs."""

from __future__ import annotations

import sys
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

from dr_cik.forecasters.chronos import ChronosUnavailableError, ChronosConfig, ChronosForecaster

from .conftest import requires_sample


class _FakeForecastType(Enum):
    SAMPLES = "samples"
    QUANTILES = "quantiles"


class _FakeSamplesPipeline:
    forecast_type = _FakeForecastType.SAMPLES

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()

    def predict(self, context, prediction_length, num_samples):
        return torch.randn(1, num_samples, prediction_length) + 500.0


class _FakeQuantilesPipeline:
    forecast_type = _FakeForecastType.QUANTILES

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()

    def predict_quantiles(self, context, prediction_length, quantile_levels):
        count = len(quantile_levels)
        quantiles = torch.randn(1, prediction_length, count) + 500.0
        mean = quantiles[:, :, count // 2]
        return quantiles, mean


@requires_sample
def test_samples_branch_returns_correct_shape(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    fake_module = SimpleNamespace(BaseChronosPipeline=_FakeSamplesPipeline, ForecastType=_FakeForecastType)
    forecaster = ChronosForecaster(ChronosConfig(num_samples=10), runtime_module=fake_module)
    forecast = forecaster.forecast(view)
    assert len(forecast.mean) == view.prediction_length
    assert len(forecast.samples) == 10
    assert all(len(sample) == view.prediction_length for sample in forecast.samples)
    assert forecast.method.startswith("chronos:")
    assert "mc-samples" in forecast.method


@requires_sample
def test_quantiles_branch_returns_correct_shape(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()
    fake_module = SimpleNamespace(BaseChronosPipeline=_FakeQuantilesPipeline, ForecastType=_FakeForecastType)
    forecaster = ChronosForecaster(ChronosConfig(num_samples=25), runtime_module=fake_module)
    forecast = forecaster.forecast(view)
    assert len(forecast.mean) == view.prediction_length
    assert len(forecast.samples) == 25
    assert "quantile-ensemble" in forecast.method


@requires_sample
def test_forecast_never_calls_a_fit_or_train_method(sample_tasks) -> None:
    view = sample_tasks[0].agent_view()

    class _StrictPipeline(_FakeSamplesPipeline):
        def fit(self, *args, **kwargs):
            raise AssertionError("forecaster must never fine-tune")

        def train(self, *args, **kwargs):
            raise AssertionError("forecaster must never fine-tune")

    fake_module = SimpleNamespace(BaseChronosPipeline=_StrictPipeline, ForecastType=_FakeForecastType)
    forecaster = ChronosForecaster(ChronosConfig(num_samples=5), runtime_module=fake_module)
    forecaster.forecast(view)  # would raise via the overrides above if fit/train were ever called


def test_missing_chronos_package_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "chronos", None)  # forces ImportError regardless of what's actually installed
    forecaster = ChronosForecaster(ChronosConfig())
    with pytest.raises(ChronosUnavailableError):
        forecaster._load_runtime()
