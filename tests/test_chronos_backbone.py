from __future__ import annotations

import unittest

from drcik_agent.agents import TimeSeriesDiagnosisAgent
from drcik_agent.backbones import (
    ChronosBackboneConfig,
    ChronosForecastBackbone,
    FallbackForecastBackbone,
    StatisticalForecastBackbone,
)
from drcik_agent.loop import IterativeAgentSystem, LoopConfig

from test_minimal_system import example_task


class _FakeTorch:
    float32 = "float32"

    @staticmethod
    def tensor(values, dtype=None):
        return {"values": tuple(values), "dtype": dtype}


class _FakePipeline:
    load_call = None
    predict_call = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.load_call = {"model_id": model_id, **kwargs}
        return cls()

    def predict_quantiles(self, *, inputs, prediction_length, quantile_levels):
        type(self).predict_call = {
            "inputs": inputs,
            "prediction_length": prediction_length,
            "quantile_levels": quantile_levels,
        }
        quantiles = None
        point = [[30.0 + index for index in range(prediction_length)]]
        return quantiles, point


class _FakeChronos:
    BaseChronosPipeline = _FakePipeline


class _BrokenPipeline:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        raise OSError("checkpoint unavailable")


class _BrokenChronos:
    BaseChronosPipeline = _BrokenPipeline


class ChronosBackboneTest(unittest.TestCase):
    def test_chronos_is_the_default_system_backbone(self) -> None:
        self.assertEqual(LoopConfig().backbone, "chronos")
        system = IterativeAgentSystem()
        self.assertIsInstance(system.forecast_agent.backbone, ChronosForecastBackbone)

    def test_official_chronos_api_is_used_for_the_baseline(self) -> None:
        task = example_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        backbone = ChronosForecastBackbone(
            ChronosBackboneConfig(max_context=128, max_horizon=16),
            runtime_module=_FakeChronos,
            tensor_module=_FakeTorch,
        )
        forecast, method = backbone.forecast(task, diagnosis)

        self.assertEqual(forecast, (30.0, 31.0))
        self.assertEqual(method, "chronos-bolt:amazon/chronos-bolt-small")
        self.assertEqual(_FakePipeline.load_call["model_id"], "amazon/chronos-bolt-small")
        self.assertEqual(_FakePipeline.load_call["device_map"], "cpu")
        self.assertEqual(_FakePipeline.predict_call["prediction_length"], 2)
        self.assertEqual(_FakePipeline.predict_call["quantile_levels"], [0.1, 0.5, 0.9])
        self.assertEqual(
            _FakePipeline.predict_call["inputs"]["values"],
            task.history_values,
        )

    def test_statistical_fallback_is_explicit_and_visible_in_method(self) -> None:
        task = example_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        primary = ChronosForecastBackbone(
            ChronosBackboneConfig(max_context=128, max_horizon=16),
            runtime_module=_BrokenChronos,
            tensor_module=_FakeTorch,
        )
        fallback = FallbackForecastBackbone(primary, StatisticalForecastBackbone())
        forecast, method = fallback.forecast(task, diagnosis)

        self.assertEqual(len(forecast), task.prediction_length)
        self.assertTrue(method.startswith("statistical_fallback:"))
        self.assertIn("checkpoint unavailable", fallback.last_error)

    def test_horizon_larger_than_configured_limit_is_rejected(self) -> None:
        task = example_task()
        diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
        backbone = ChronosForecastBackbone(
            ChronosBackboneConfig(max_context=128, max_horizon=1),
            runtime_module=_FakeChronos,
            tensor_module=_FakeTorch,
        )
        with self.assertRaisesRegex(ValueError, "exceeds Chronos max_horizon"):
            backbone.forecast(task, diagnosis)


if __name__ == "__main__":
    unittest.main()
