from __future__ import annotations

import dataclasses
import importlib
import math

import pytest

from numerical_agent.dictionary import MethodCandidate
from numerical_agent.providers import RuntimeUnavailableError
from numerical_agent.tsfm import ManifestRegistry, TimesFMRuntime


def _candidate(
    model_id: str = "google/timesfm-2.5-200m-pytorch",
    *,
    provider: str = "timesfm",
    implementation_kind: str = "tsfm_checkpoint",
    method_id: str = "method_tsfm_0031",
    runtime_options: object = None,
    runtime_family: str = "timesfm",
    worker_environment: str = "",
) -> MethodCandidate:
    return MethodCandidate(
        method_id=method_id,
        provider=provider,
        implementation_kind=implementation_kind,
        implementation={
            "model_id": model_id,
            "checkpoint": model_id,
            "runtime_family": runtime_family,
            "worker_environment": worker_environment,
            "manifest_id": method_id,
            "runtime_options": {} if runtime_options is None else runtime_options,
            "point_reduction": "direct",
            "license_id": "Apache-2.0",
            "license_acknowledgement_required": False,
        },
    )


class _FakeModel:
    def __init__(self, point_forecast: object | None = None) -> None:
        self.point_forecast = point_forecast or [[10.0, 11.5]]
        self.compile_calls: list[object] = []
        self.forecast_calls: list[dict[str, object]] = []

    def compile(self, config: object) -> None:
        self.compile_calls.append(config)

    def forecast(self, *, horizon: int, inputs: list[list[float]]):
        self.forecast_calls.append({"horizon": horizon, "inputs": inputs})
        return self.point_forecast, [[[0.0]]]


class _FakeOfficialObjects:
    def __init__(self, model: _FakeModel | None = None) -> None:
        self.model = model or _FakeModel()
        self.load_calls: list[str] = []
        self.config_calls: list[dict[str, object]] = []

    def load(self, model_id: str) -> _FakeModel:
        self.load_calls.append(model_id)
        return self.model

    def config(self, **kwargs: object) -> dict[str, object]:
        self.config_calls.append(dict(kwargs))
        return {"official_config": dict(kwargs)}


def test_forecast_compiles_once_and_returns_first_point_forecast() -> None:
    official = _FakeOfficialObjects()
    runtime = TimesFMRuntime(
        model_loader=official.load,
        forecast_config_factory=official.config,
        max_context=3,
        max_horizon=4,
    )

    first = runtime.forecast(_candidate(), [1.0, 2.0, 3.0, 4.0], 2, "D")
    second = runtime.forecast(_candidate(), [5.0, 6.0], 2, "H")

    assert first == (10.0, 11.5)
    assert second == (10.0, 11.5)
    assert official.load_calls == ["google/timesfm-2.5-200m-pytorch"]
    assert len(official.model.compile_calls) == 1
    assert official.model.compile_calls == [
        {
            "official_config": {
                "max_context": 3,
                "max_horizon": 4,
                "normalize_inputs": True,
                "use_continuous_quantile_head": True,
                "force_flip_invariance": True,
                "infer_is_positive": True,
                "fix_quantile_crossing": True,
            }
        }
    ]
    assert official.config_calls == [official.model.compile_calls[0]["official_config"]]
    assert official.model.forecast_calls == [
        {"horizon": 2, "inputs": [[2.0, 3.0, 4.0]]},
        {"horizon": 2, "inputs": [[5.0, 6.0]]},
    ]


def test_missing_official_package_is_runtime_unavailable(monkeypatch) -> None:
    real_import_module = importlib.import_module

    def import_module_for_test(name: str) -> object:
        if name == "timesfm":
            raise ImportError("missing timesfm")
        return real_import_module(name)

    monkeypatch.setattr(
        "numerical_agent.tsfm.timesfm.importlib.import_module", import_module_for_test
    )

    with pytest.raises(RuntimeUnavailableError, match="TimesFM is not installed"):
        TimesFMRuntime().forecast(_candidate(), [1.0, 2.0], 2, "D")


def test_checkpoint_loader_failure_is_runtime_unavailable() -> None:
    def load(model_id: str) -> object:
        raise OSError(f"cannot load {model_id}")

    runtime = TimesFMRuntime(
        model_loader=load,
        forecast_config_factory=lambda **kwargs: kwargs,
    )

    with pytest.raises(RuntimeUnavailableError) as caught:
        runtime.forecast(_candidate(), [1.0, 2.0], 2, "D")

    assert "cannot load google/timesfm-2.5-200m-pytorch" == str(caught.value)


def test_checkpoint_compile_failure_is_runtime_unavailable() -> None:
    class BrokenCompileModel(_FakeModel):
        def compile(self, config: object) -> None:
            del config
            raise RuntimeError("unsupported torch runtime")

    runtime = TimesFMRuntime(
        model_loader=lambda model_id: BrokenCompileModel(),
        forecast_config_factory=lambda **kwargs: kwargs,
    )

    with pytest.raises(RuntimeUnavailableError) as caught:
        runtime.forecast(_candidate(), [1.0, 2.0], 2, "D")

    assert str(caught.value) == "unsupported torch runtime"


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate("google/timesfm-1.0-200m"),
        _candidate("google/timesfm-2.5-200m-jax"),
        _candidate(provider="chronos"),
        _candidate(implementation_kind="python_source"),
    ],
)
def test_supports_only_timesfm_2p5_pytorch_candidates(
    candidate: MethodCandidate,
) -> None:
    runtime = TimesFMRuntime(model_loader=lambda _model_id: None)

    assert runtime.supports(_candidate()) is True
    assert runtime.supports(candidate) is False


@pytest.mark.parametrize(
    ("field", "substitute"),
    [
        ("checkpoint", "attacker/timesfm"),
        ("model_id", "attacker/timesfm"),
        ("runtime_family", "attacker_adapter"),
        ("worker_environment", "attacker_environment"),
        ("runtime_options", {"revision": "attacker"}),
        ("point_reduction", "median"),
        ("license_id", "attacker-license"),
        ("license_acknowledgement_required", True),
        ("manifest_id", "method_tsfm_0001"),
    ],
)
def test_direct_timesfm_rejects_candidate_manifest_substitution(
    field: str, substitute: object,
) -> None:
    runtime = TimesFMRuntime(model_loader=lambda _model_id: None)
    implementation = dict(_candidate().implementation)
    implementation[field] = substitute
    candidate = dataclasses.replace(_candidate(), implementation=implementation)

    assert not runtime.supports(candidate)


def test_direct_timesfm_rejects_provider_substitution() -> None:
    runtime = TimesFMRuntime(model_loader=lambda _model_id: None)

    assert not runtime.supports(_candidate(provider="tsfm_worker"))


def test_direct_timesfm_honors_an_explicit_empty_registry() -> None:
    runtime = TimesFMRuntime(
        model_loader=lambda _model_id: None,
        manifests=ManifestRegistry({}),
    )

    assert not runtime.supports(_candidate())


@pytest.mark.parametrize("history", [[], [1.0, math.nan], [1.0, math.inf]])
def test_forecast_rejects_empty_or_non_finite_history(history: list[float]) -> None:
    official = _FakeOfficialObjects()
    runtime = TimesFMRuntime(
        model_loader=official.load,
        forecast_config_factory=official.config,
    )

    with pytest.raises(ValueError, match="history"):
        runtime.forecast(_candidate(), history, 2, "D")

    assert official.load_calls == []


@pytest.mark.parametrize("horizon", [0, 3])
def test_forecast_rejects_horizons_outside_the_compiled_limit(horizon: int) -> None:
    official = _FakeOfficialObjects()
    runtime = TimesFMRuntime(
        model_loader=official.load,
        forecast_config_factory=official.config,
        max_horizon=2,
    )

    with pytest.raises(ValueError, match="horizon"):
        runtime.forecast(_candidate(), [1.0], horizon, "D")

    assert official.load_calls == []


@pytest.mark.parametrize(
    ("point_forecast", "message"),
    [
        ([[10.0]], "horizon"),
        ([[10.0, math.nan]], "finite"),
        ([10.0, 11.0], "shape"),
    ],
)
def test_forecast_rejects_malformed_point_output(
    point_forecast: object,
    message: str,
) -> None:
    official = _FakeOfficialObjects(_FakeModel(point_forecast))
    runtime = TimesFMRuntime(
        model_loader=official.load,
        forecast_config_factory=official.config,
    )

    with pytest.raises(ValueError, match=message):
        runtime.forecast(_candidate(), [1.0, 2.0], 2, "D")


def test_constructor_rejects_non_positive_compile_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        TimesFMRuntime(max_context=0)
    with pytest.raises(ValueError, match="positive"):
        TimesFMRuntime(max_horizon=0)
