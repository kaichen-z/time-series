from __future__ import annotations

import dataclasses
import math

import pytest

from numerical_agent.dictionary import MethodCandidate
from numerical_agent.providers import RuntimeUnavailableError
from numerical_agent.tsfm import ChronosRuntime, ManifestRegistry


def chronos_candidate(
    checkpoint: str = "amazon/chronos-bolt-base",
    *,
    provider: str = "chronos",
    implementation_kind: str = "tsfm_checkpoint",
    runtime_family: str = "chronos",
    worker_environment: str = "",
    method_id: str | None = None,
    runtime_options: object = None,
) -> MethodCandidate:
    if method_id is None:
        method_id = {
            "amazon/chronos-t5-base": "method_tsfm_0002",
            "amazon/chronos-2": "method_tsfm_0016",
            "amazon/chronos-bolt-base": "method_tsfm_0018",
        }.get(checkpoint, "method_tsfm_0018")
    return MethodCandidate(
        method_id=method_id,
        provider=provider,
        implementation_kind=implementation_kind,
        implementation={
            "checkpoint": checkpoint,
            "model_id": checkpoint,
            "worker_environment": worker_environment,
            "runtime_family": runtime_family,
            "release": "2024",
            "context_limit": "checkpoint_config",
            "prediction_limit": "direct_multi_step",
            "manifest_id": method_id,
            "runtime_options": {} if runtime_options is None else runtime_options,
            "point_reduction": "median",
            "license_id": "Apache-2.0",
            "license_acknowledgement_required": False,
        },
    )


class FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = values

    def tolist(self) -> object:
        return self.values


class FakePipeline:
    def __init__(
        self,
        quantiles: object | None = None,
        mean: object | None = None,
        *,
        chronos2_output: bool = False,
    ) -> None:
        self.quantiles = (
            quantiles
            if quantiles is not None
            else [[[1.25], [2.5], [3.75]]]
        )
        self.mean = mean if mean is not None else [[10.0, 11.5, 12.25]]
        self.chronos2_output = chronos2_output
        self.predict_calls: list[dict[str, object]] = []

    def predict_quantiles(
        self, *, inputs: object, prediction_length: int, quantile_levels: list[float]
    ) -> tuple[object, object]:
        self.predict_calls.append(
            {
                "inputs": inputs,
                "prediction_length": prediction_length,
                "quantile_levels": quantile_levels,
            }
        )
        quantiles: object = FakeTensor(self.quantiles)
        if self.chronos2_output:
            quantiles = [quantiles]
        return quantiles, self.mean


def test_supports_only_chronos_checkpoint_candidates() -> None:
    runtime = ChronosRuntime(model_loader=lambda *args, **kwargs: FakePipeline())

    assert runtime.supports(chronos_candidate())
    assert not runtime.supports(chronos_candidate(provider="sandbox"))
    assert not runtime.supports(chronos_candidate(implementation_kind="python_code"))
    assert not runtime.supports(chronos_candidate(runtime_family="timesfm"))
    assert not runtime.supports(chronos_candidate(checkpoint="google/timesfm-2.5"))
    assert not runtime.supports(chronos_candidate(checkpoint=""))


@pytest.mark.parametrize(
    ("field", "substitute"),
    [
        ("checkpoint", "attacker/chronos-model"),
        ("model_id", "attacker/chronos-model"),
        ("runtime_family", "attacker_adapter"),
        ("worker_environment", "attacker_environment"),
        ("runtime_options", {"revision": "attacker"}),
        ("point_reduction", "mean"),
        ("license_id", "attacker-license"),
        ("license_acknowledgement_required", True),
        ("manifest_id", "method_tsfm_0031"),
    ],
)
def test_direct_chronos_rejects_candidate_manifest_substitution(
    field: str, substitute: object,
) -> None:
    runtime = ChronosRuntime(model_loader=lambda *args, **kwargs: FakePipeline())
    implementation = dict(chronos_candidate().implementation)
    implementation[field] = substitute
    candidate = dataclasses.replace(chronos_candidate(), implementation=implementation)

    assert not runtime.supports(candidate)


def test_direct_chronos_rejects_provider_substitution() -> None:
    runtime = ChronosRuntime(model_loader=lambda *args, **kwargs: FakePipeline())

    assert not runtime.supports(chronos_candidate(provider="tsfm_worker"))


def test_direct_chronos_honors_an_explicit_empty_registry() -> None:
    runtime = ChronosRuntime(
        model_loader=lambda *args, **kwargs: FakePipeline(),
        manifests=ManifestRegistry({}),
    )

    assert not runtime.supports(chronos_candidate())


def test_forecast_uses_official_p50_and_caches_each_checkpoint() -> None:
    pipelines: dict[str, FakePipeline] = {}
    load_calls: list[tuple[str, str]] = []

    def load(checkpoint: str, *, device_map: str) -> FakePipeline:
        load_calls.append((checkpoint, device_map))
        return pipelines.setdefault(checkpoint, FakePipeline())

    runtime = ChronosRuntime(model_loader=load, device_map="cpu")
    candidate = chronos_candidate()

    first = runtime.forecast(candidate, [1, 2.5, 4], 3, "D")
    second = runtime.forecast(candidate, [5, 6, 7], 3, "H")
    other = runtime.forecast(
        chronos_candidate("amazon/chronos-t5-base"), [8, 9, 10], 3, "M"
    )

    assert first == (1.25, 2.5, 3.75)
    assert second == first
    assert other == first
    assert load_calls == [
        ("amazon/chronos-bolt-base", "cpu"),
        ("amazon/chronos-t5-base", "cpu"),
    ]
    assert pipelines["amazon/chronos-bolt-base"].predict_calls[0] == {
        "inputs": [1.0, 2.5, 4.0],
        "prediction_length": 3,
        "quantile_levels": [0.5],
    }


@pytest.mark.parametrize(
    ("checkpoint", "chronos2_output"),
    [
        ("amazon/chronos-t5-base", False),
        ("amazon/chronos-2", True),
        ("amazon/chronos-bolt-base", False),
    ],
)
def test_all_direct_chronos_manifests_reduce_to_requested_median(
    checkpoint: str, chronos2_output: bool
) -> None:
    pipeline = FakePipeline(chronos2_output=chronos2_output)
    runtime = ChronosRuntime(model_loader=lambda *args, **kwargs: pipeline)

    assert runtime.forecast(chronos_candidate(checkpoint), [1.0, 2.0], 3, "D") == (
        1.25,
        2.5,
        3.75,
    )
    assert pipeline.predict_calls[0]["quantile_levels"] == [0.5]


def test_official_dependencies_are_imported_only_on_first_forecast(monkeypatch) -> None:
    pipeline = FakePipeline(chronos2_output=True)
    imports: list[str] = []

    class FakeBaseChronosPipeline:
        @staticmethod
        def from_pretrained(checkpoint: str, *, device_map: str) -> FakePipeline:
            assert checkpoint == "amazon/chronos-2"
            assert device_map == "cpu"
            return pipeline

    class FakeChronos:
        BaseChronosPipeline = FakeBaseChronosPipeline

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def tensor(values: object, *, dtype: object) -> dict[str, object]:
            return {"values": list(values), "dtype": dtype}

    def import_module(name: str) -> object:
        imports.append(name)
        return {"chronos": FakeChronos, "torch": FakeTorch}[name]

    monkeypatch.setattr("numerical_agent.tsfm.chronos.importlib.import_module", import_module)
    runtime = ChronosRuntime()

    assert imports == []
    assert runtime.forecast(
        chronos_candidate("amazon/chronos-2"), [1.0, 2.0], 3, "D"
    ) == (
        1.25,
        2.5,
        3.75,
    )
    assert imports == ["chronos", "torch"]
    assert pipeline.predict_calls[0]["inputs"] == [
        {"values": [1.0, 2.0], "dtype": "float32"}
    ]


def test_missing_official_package_is_runtime_unavailable(monkeypatch) -> None:
    def import_module(name: str) -> object:
        raise ImportError(f"missing {name}")

    monkeypatch.setattr("numerical_agent.tsfm.chronos.importlib.import_module", import_module)

    with pytest.raises(RuntimeUnavailableError, match="Chronos runtime is unavailable"):
        ChronosRuntime().forecast(chronos_candidate(), [1.0, 2.0], 3, "D")


def test_checkpoint_loader_failure_is_runtime_unavailable() -> None:
    def load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError("checkpoint cache is unavailable")

    with pytest.raises(RuntimeUnavailableError) as caught:
        ChronosRuntime(model_loader=load).forecast(
            chronos_candidate(), [1.0, 2.0], 3, "D"
        )

    assert str(caught.value) == "checkpoint cache is unavailable"


@pytest.mark.parametrize(
    ("history", "horizon", "message"),
    [
        ([], 3, "history must not be empty"),
        ([1.0, math.nan], 3, "history must contain only finite values"),
        ([[1.0], [2.0]], 3, "history must be one-dimensional"),
        ([1.0], 0, "horizon must be positive"),
    ],
)
def test_forecast_rejects_invalid_inputs(
    history: list[object], horizon: int, message: str
) -> None:
    runtime = ChronosRuntime(model_loader=lambda *args, **kwargs: FakePipeline())

    with pytest.raises(ValueError, match=message):
        runtime.forecast(chronos_candidate(), history, horizon, "D")


@pytest.mark.parametrize(
    ("quantiles", "message"),
    [
        ([[[1.0], [2.0]]], "wrong horizon length"),
        ([[[1.0], [math.inf], [3.0]]], "non-finite"),
        ([[[1.0], [10**1000], [3.0]]], "scalar values"),
        ([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], "batch shape"),
        ([[[1.0, 9.0], [2.0, 9.0], [3.0, 9.0]]], "quantile shape"),
    ],
)
def test_forecast_rejects_malformed_official_p50(
    quantiles: object, message: str
) -> None:
    runtime = ChronosRuntime(
        model_loader=lambda *args, **kwargs: FakePipeline(quantiles=quantiles)
    )

    with pytest.raises(RuntimeError, match=message):
        runtime.forecast(chronos_candidate(), [1.0, 2.0], 3, "D")


def test_forecast_rejects_an_unsupported_candidate_before_loading() -> None:
    loaded = False

    def load(*args, **kwargs) -> FakePipeline:
        nonlocal loaded
        loaded = True
        return FakePipeline()

    runtime = ChronosRuntime(model_loader=load)

    with pytest.raises(ValueError, match="does not support candidate"):
        runtime.forecast(chronos_candidate(provider="timesfm"), [1.0], 3, "D")
    assert not loaded
