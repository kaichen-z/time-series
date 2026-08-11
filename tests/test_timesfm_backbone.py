from __future__ import annotations

import unittest

from drcik_agent.agents import TimeSeriesDiagnosisAgent
from drcik_agent.backbones import (
    FallbackForecastBackbone,
    StatisticalForecastBackbone,
    TimesFMBackboneConfig,
    TimesFMForecastBackbone,
)
from test_minimal_system import example_task


class _FakeForecastConfig:
    def __init__(self, **kwargs) -> None:
        self.values = kwargs


class _FakeModel:
    def __init__(self) -> None:
        self.compile_config = None
        self.forecast_call = None

    def compile(self, config) -> None:
        self.compile_config = config

    def forecast(self, *, horizon, inputs):
        self.forecast_call = {"horizon": horizon, "inputs": inputs}
        point = [[30.0 + index for index in range(horizon)]]
        quantiles = [[[value] * 10 for value in point[0]]]
        return point, quantiles


class _FakeModelClass:
    model = None
    load_call = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.load_call = {"model_id": model_id, **kwargs}
        cls.model = _FakeModel()
        return cls.model


class _FakeTimesFM:
    ForecastConfig = _FakeForecastConfig
    TimesFM_2p5_200M_torch = _FakeModelClass


class _BrokenModelClass:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        raise OSError("checkpoint unavailable")


class _BrokenTimesFM:
    ForecastConfig = _FakeForecastConfig
    TimesFM_2p5_200M_torch = _BrokenModelClass


class TimesFMBackboneTest(unittest.TestCase):
    def test_official_timesfm_2p5_api_is_used_for_the_baseline(self) -> None:
        task = example_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        backbone = TimesFMForecastBackbone(
            TimesFMBackboneConfig(max_context=128, max_horizon=16),
            runtime_module=_FakeTimesFM,
        )
        forecast, method = backbone.forecast(task, diagnosis)

        self.assertEqual(forecast, (30.0, 31.0))
        self.assertTrue(method.startswith("timesfm-2.5-200m-pytorch:"))
        self.assertEqual(_FakeModelClass.load_call["model_id"], "google/timesfm-2.5-200m-pytorch")
        self.assertEqual(_FakeModelClass.model.forecast_call["horizon"], 2)
        self.assertEqual(
            _FakeModelClass.model.forecast_call["inputs"],
            [list(task.history_values)],
        )
        self.assertEqual(_FakeModelClass.model.compile_config.values["max_context"], 128)
        self.assertTrue(
            _FakeModelClass.model.compile_config.values["use_continuous_quantile_head"]
        )

    def test_statistical_fallback_is_explicit_and_visible_in_method(self) -> None:
        task = example_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        primary = TimesFMForecastBackbone(
            TimesFMBackboneConfig(max_context=128, max_horizon=16),
            runtime_module=_BrokenTimesFM,
        )
        fallback = FallbackForecastBackbone(primary, StatisticalForecastBackbone())
        forecast, method = fallback.forecast(task, diagnosis)

        self.assertEqual(len(forecast), task.prediction_length)
        self.assertTrue(method.startswith("statistical_fallback:"))
        self.assertIn("checkpoint unavailable", fallback.last_error)

    def test_horizon_larger_than_compiled_limit_is_rejected(self) -> None:
        task = example_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        backbone = TimesFMForecastBackbone(
            TimesFMBackboneConfig(max_context=128, max_horizon=1),
            runtime_module=_FakeTimesFM,
        )
        with self.assertRaisesRegex(ValueError, "exceeds TimesFM max_horizon"):
            backbone.forecast(task, diagnosis)


if __name__ == "__main__":
    unittest.main()
