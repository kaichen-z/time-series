from __future__ import annotations

from contextlib import nullcontext
import builtins
import importlib
import io
import json
import math
import sys
from typing import Any

import pytest

from numerical_agent.tsfm import ManifestRegistry
from numerical_agent.tsfm.protocol import WorkerRequest


TIMESFM_ID = "method_tsfm_0001"
LAG_LLAMA_ID = "method_tsfm_0004"
TEMPO_ID = "method_tsfm_0011"
REVISION_A = "a" * 40
REVISION_B = "b" * 40


def _request(
    method_id: str,
    *,
    history: tuple[float, ...] = (1.0, 2.0, 3.0),
    horizon: int = 2,
    frequency: str = "D",
    runtime_options: dict[str, object] | None = None,
) -> WorkerRequest:
    manifest = ManifestRegistry.load_default()[method_id]
    return WorkerRequest(
        request_id=f"request-{method_id}",
        provider=manifest.adapter,
        checkpoint=manifest.checkpoint,
        history=history,
        horizon=horizon,
        frequency=frequency,
        runtime_options=(
            dict(manifest.runtime_options)
            if runtime_options is None
            else runtime_options
        ),
    )


def _attested_request(method_id: str, revision: str) -> WorkerRequest:
    request = _request(method_id)
    object.__setattr__(request, "checkpoint_revision", revision)
    return request


class _Constructor:
    def __init__(self, factory: Any) -> None:
        self.factory = factory
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self.factory(**kwargs)


class _TimesFMModel:
    def __init__(self, point: object = None) -> None:
        self.point = [[10.0, 11.0, 12.0]] if point is None else point
        self.calls: list[tuple[object, dict[str, object]]] = []

    def forecast(self, inputs: object, **kwargs: object) -> object:
        self.calls.append((inputs, dict(kwargs)))
        return self.point, [[[0.0]]]


class _TimesFMBackend:
    def __init__(self, point: object = None) -> None:
        self.model = _TimesFMModel(point)
        self.hparams_class = _Constructor(lambda **kwargs: {"hparams": kwargs})
        self.checkpoint_class = _Constructor(lambda **kwargs: {"checkpoint": kwargs})
        self.model_class = _Constructor(lambda **kwargs: self.model)
        self.frequency_calls: list[str] = []
        self.download_calls: list[tuple[str, str, str]] = []

    def download_checkpoint(
        self, repo_id: str, filename: str, *, revision: str
    ) -> str:
        self.download_calls.append((repo_id, filename, revision))
        return f"/hub/{revision}/{filename}"

    def frequency_class(self, frequency: str) -> int:
        self.frequency_calls.append(frequency)
        return {
            "ms": 0,
            "5ms": 0,
            "MS": 1,
            "5MS": 1,
            "15T": 0,
            "30S": 0,
            "H": 0,
            "D": 0,
            "W": 1,
            "M": 1,
            "Q": 2,
            "Y": 2,
        }[frequency]


class _LagForecast:
    def __init__(self, values: object = None) -> None:
        self.values = [10.0, 11.0] if values is None else values
        self.quantile_calls: list[float] = []

    def quantile(self, quantile: float) -> object:
        self.quantile_calls.append(quantile)
        return self.values


class _LagPredictor:
    def __init__(self, forecast: _LagForecast) -> None:
        self.forecast = forecast
        self.calls: list[tuple[object, dict[str, object]]] = []

    def predict(self, dataset: object, **kwargs: object) -> object:
        self.calls.append((dataset, dict(kwargs)))
        return iter([self.forecast])


class _LagModule:
    def __init__(self) -> None:
        self.to_calls: list[object] = []

    def to(self, device: object) -> "_LagModule":
        self.to_calls.append(device)
        return self


class _LagEstimator:
    def __init__(self, forecast: _LagForecast) -> None:
        self.forecast = forecast
        self.module = _LagModule()
        self.predictor = _LagPredictor(forecast)
        self.create_module_calls = 0
        self.create_transformation_calls = 0
        self.create_predictor_calls: list[tuple[object, object]] = []

    def create_lightning_module(self) -> _LagModule:
        self.create_module_calls += 1
        return self.module

    def create_transformation(self) -> object:
        self.create_transformation_calls += 1
        return "official-transformation"

    def create_predictor(self, transformation: object, module: object) -> _LagPredictor:
        self.create_predictor_calls.append((transformation, module))
        return self.predictor


class _LagEstimatorClass:
    def __init__(self, forecast: _LagForecast) -> None:
        self.forecast = forecast
        self.calls: list[dict[str, object]] = []
        self.estimators: list[_LagEstimator] = []

    def __call__(self, **kwargs: object) -> _LagEstimator:
        self.calls.append(dict(kwargs))
        estimator = _LagEstimator(self.forecast)
        self.estimators.append(estimator)
        return estimator


class _LagBackend:
    device = "fake-device"

    def __init__(self, values: object = None) -> None:
        self.forecast = _LagForecast(values)
        self.estimator_class = _LagEstimatorClass(self.forecast)
        self.download_calls: list[tuple[object, ...]] = []
        self.load_calls: list[str] = []
        self.dataset_calls: list[tuple[tuple[float, ...], str]] = []

    def download_checkpoint(
        self, repo_id: str, filename: str, *, revision: str = ""
    ) -> str:
        if revision:
            self.download_calls.append((repo_id, filename, revision))
            return f"/hub/{revision}/{filename}"
        self.download_calls.append((repo_id, filename))
        return "/official-cache/lag-llama.ckpt"

    def load_checkpoint(self, path: str) -> dict[str, object]:
        self.load_calls.append(path)
        return {
            "hyper_parameters": {
                "model_kwargs": {
                    "input_size": 1,
                    "context_length": 32,
                    "n_layer": 8,
                    "n_embd_per_head": 16,
                    "n_head": 9,
                    "scaling": "robust",
                    "time_feat": True,
                }
            }
        }

    def dataset(self, history: tuple[float, ...], frequency: str) -> object:
        self.dataset_calls.append((tuple(history), frequency))
        return {"history": tuple(history), "frequency": frequency}


class _TempoModel:
    def __init__(self, output: object = None) -> None:
        self.output = [10.0, 11.0] if output is None else output
        self.predict_calls: list[tuple[tuple[float, ...], dict[str, object]]] = []

    def predict(self, history: object, **kwargs: object) -> object:
        copied = tuple(history)  # type: ignore[arg-type]
        self.predict_calls.append((copied, dict(kwargs)))
        if callable(self.output):
            return self.output(kwargs["pred_length"])
        return self.output


class _TempoModelClass:
    def __init__(self, model: _TempoModel) -> None:
        self.model = model
        self.calls: list[dict[str, object]] = []

    def load_pretrained_model(self, **kwargs: object) -> _TempoModel:
        self.calls.append(dict(kwargs))
        return self.model


class _TempoBackend:
    device = "fake-device"

    def __init__(self, output: object = None) -> None:
        self.model = _TempoModel(output)
        self.model_class = _TempoModelClass(self.model)
        self.exact_load_calls: list[tuple[str, str, str]] = []

    def load_exact_model(
        self, repo_id: str, filename: str, revision: str
    ) -> _TempoModel:
        self.exact_load_calls.append((repo_id, filename, revision))
        return self.model

    def no_grad(self) -> object:
        return nullcontext()


def test_timesfm_backend_corrects_archived_millisecond_frequency_bug() -> None:
    from numerical_agent.tsfm.workers.legacy import _TimesFMBackend

    archived_calls: list[str] = []

    def archived_freq_map(frequency: str) -> int:
        archived_calls.append(frequency)
        return 1 if frequency.upper().endswith("MS") else 0

    backend = _TimesFMBackend(
        model_class=object(),
        hparams_class=object(),
        checkpoint_class=object(),
        frequency_map=archived_freq_map,
    )

    assert backend.frequency_class("5ms") == 0
    assert backend.frequency_class("5MS") == 1
    assert archived_calls == ["5MS"]


def test_timesfm_v1_forces_archived_pytorch_api_and_frequency_class() -> None:
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    backend = _TimesFMBackend()
    history = tuple(float(index) for index in range(520))
    adapter = LegacyAdapter(loader=lambda method_id: backend)

    assert adapter.forecast(
        _request(TIMESFM_ID, history=history, frequency="1 day")
    ) == (10.0, 11.0)
    assert backend.hparams_class.calls == [
        {
            "context_len": 512,
            "horizon_len": 512,
            "input_patch_len": 32,
            "output_patch_len": 128,
            "per_core_batch_size": 1,
            "backend": "cpu",
        }
    ]
    assert backend.checkpoint_class.calls == [
        {
            "version": "torch",
            "huggingface_repo_id": "google/timesfm-1.0-200m-pytorch",
        }
    ]
    assert backend.model_class.calls == [
        {
            "hparams": {"hparams": backend.hparams_class.calls[0]},
            "checkpoint": {"checkpoint": backend.checkpoint_class.calls[0]},
        }
    ]
    assert backend.frequency_calls == ["D"]
    assert backend.model.calls == [([history[-512:]], {"freq": [0]})]


@pytest.mark.parametrize(
    ("frequency", "normalized", "frequency_class"),
    [
        ("15 minutes", "15T", 0),
        ("15T", "15T", 0),
        ("30 seconds", "30S", 0),
        ("ms", "ms", 0),
        ("5ms", "5ms", 0),
        ("MS", "MS", 1),
        ("5MS", "5MS", 1),
        ("H", "H", 0),
        ("hourly", "H", 0),
        ("D", "D", 0),
        ("daily", "D", 0),
        ("1 week", "W", 1),
        ("weekly", "W", 1),
        ("M", "M", 1),
        ("monthly", "M", 1),
        ("1 quarter", "Q", 2),
        ("quarterly", "Q", 2),
        ("Y", "Y", 2),
        ("yearly", "Y", 2),
    ],
)
def test_timesfm_uses_official_frequency_classes(
    frequency: str, normalized: str, frequency_class: int
) -> None:
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    backend = _TimesFMBackend()
    LegacyAdapter(loader=lambda method_id: backend).forecast(
        _request(TIMESFM_ID, frequency=frequency)
    )

    assert backend.frequency_calls == [normalized]
    assert backend.model.calls[0][1] == {"freq": [frequency_class]}


def test_timesfm_rejects_unknown_frequency_and_horizon_limit_before_loading() -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    loader_calls = 0

    def load(method_id: str) -> object:
        nonlocal loader_calls
        loader_calls += 1
        return _TimesFMBackend()

    adapter = LegacyAdapter(loader=load)
    with pytest.raises(RequestUnavailableError, match="frequency"):
        adapter.forecast(_request(TIMESFM_ID, frequency="sometimes"))
    with pytest.raises(RequestUnavailableError, match="512"):
        adapter.forecast(_request(TIMESFM_ID, horizon=513))
    assert loader_calls == 0


@pytest.mark.parametrize("frequency", ["5mS", "0ms", "05ms"])
@pytest.mark.parametrize("method_id", [TIMESFM_ID, LAG_LLAMA_ID])
def test_legacy_rejects_ambiguous_millisecond_case_before_loading(
    method_id: str, frequency: str
) -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    loader_calls = 0

    def load(selected: str) -> object:
        nonlocal loader_calls
        loader_calls += 1
        return _TimesFMBackend()

    with pytest.raises(RequestUnavailableError, match="frequency"):
        LegacyAdapter(loader=load).forecast(
            _request(method_id, frequency=frequency)
        )
    assert loader_calls == 0


def test_lag_llama_downloads_exact_checkpoint_builds_estimator_and_uses_p50() -> None:
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    backend = _LagBackend()
    adapter = LegacyAdapter(loader=lambda method_id: backend)

    assert adapter.forecast(_request(LAG_LLAMA_ID)) == (10.0, 11.0)
    assert backend.download_calls == [
        ("time-series-foundation-models/Lag-Llama", "lag-llama.ckpt")
    ]
    assert backend.load_calls == ["/official-cache/lag-llama.ckpt"]
    assert backend.estimator_class.calls == [
        {
            "ckpt_path": "/official-cache/lag-llama.ckpt",
            "prediction_length": 2,
            "context_length": 3,
            "device": "fake-device",
            "input_size": 1,
            "n_layer": 8,
            "n_embd_per_head": 16,
            "n_head": 9,
            "scaling": "robust",
            "time_feat": True,
            "rope_scaling": {"type": "linear", "factor": 1.0},
            "batch_size": 1,
            "num_parallel_samples": 100,
        }
    ]
    estimator = backend.estimator_class.estimators[0]
    assert estimator.create_module_calls == 1
    assert estimator.module.to_calls == ["fake-device"]
    assert estimator.create_transformation_calls == 1
    assert estimator.create_predictor_calls == [
        ("official-transformation", estimator.module)
    ]
    assert backend.dataset_calls == [((1.0, 2.0, 3.0), "D")]
    assert estimator.predictor.calls == [
        ({"history": (1.0, 2.0, 3.0), "frequency": "D"}, {"num_samples": 100})
    ]
    assert backend.forecast.quantile_calls == [0.5]


def test_lag_llama_caps_context_at_official_2048_and_preserves_cadence() -> None:
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    backend = _LagBackend()
    history = tuple(float(index) for index in range(2_100))
    LegacyAdapter(loader=lambda method_id: backend).forecast(
        _request(LAG_LLAMA_ID, history=history, frequency="1 hour")
    )

    assert backend.dataset_calls == [(history[-2_048:], "h")]
    estimator_call = backend.estimator_class.calls[0]
    assert estimator_call["context_length"] == 2_048
    assert estimator_call["rope_scaling"] == {
        "type": "linear",
        "factor": (2_048 + 2) / 32,
    }


@pytest.mark.parametrize("frequency", ["ms", "5ms", "MS", "5MS"])
def test_lag_llama_preserves_case_sensitive_millisecond_timestamps(
    frequency: str,
) -> None:
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    backend = _LagBackend()
    LegacyAdapter(loader=lambda method_id: backend).forecast(
        _request(LAG_LLAMA_ID, frequency=frequency)
    )

    assert backend.dataset_calls == [((1.0, 2.0, 3.0), frequency)]


@pytest.mark.parametrize(
    ("frequency", "period_frequency"),
    [("5ms", "5ms"), ("5MS", "5M")],
)
def test_lag_backend_converts_month_start_to_period_compatible_frequency(
    frequency: str, period_frequency: str
) -> None:
    from numerical_agent.tsfm.workers.legacy import _LagBackend

    class Periods:
        freqstr = period_frequency

    class Timestamps:
        def __init__(self) -> None:
            self.to_period_calls: list[str] = []

        def to_period(self, frequency: str) -> Periods:
            self.to_period_calls.append(frequency)
            return Periods()

    class Pandas:
        def __init__(self) -> None:
            self.timestamps = Timestamps()
            self.date_range_calls: list[dict[str, object]] = []
            self.series_calls: list[tuple[object, object, str]] = []

        def date_range(self, **kwargs: object) -> Timestamps:
            self.date_range_calls.append(dict(kwargs))
            return self.timestamps

        def Series(self, values: object, *, index: object, name: str) -> object:
            self.series_calls.append((values, index, name))
            return {"values": values, "index": index, "name": name}

    pandas = Pandas()
    dataset_calls: list[tuple[object, str]] = []

    def pandas_dataset(values: object, *, freq: str) -> object:
        dataset_calls.append((values, freq))
        return {"values": values, "freq": freq}

    backend = _LagBackend(
        torch=object(),
        pandas=pandas,
        pandas_dataset=pandas_dataset,
        estimator_class=object(),
        hf_hub_download=lambda **kwargs: "unused",
        device="fake-device",
    )
    history = (1.0, 2.0, 3.0)

    backend.dataset(history, frequency)

    assert pandas.date_range_calls == [
        {"start": "2000-01-01", "periods": 3, "freq": frequency}
    ]
    assert pandas.timestamps.to_period_calls == [period_frequency]
    assert dataset_calls[0][1] == period_frequency


def test_lag_llama_context_cap_is_owned_by_the_immutable_manifest() -> None:
    manifest = ManifestRegistry.load_default()[LAG_LLAMA_ID]

    assert dict(manifest.runtime_options) == {
        "checkpoint_file": "lag-llama.ckpt",
        "max_context": 2_048,
        "num_samples": 100,
    }


def test_lag_llama_reuses_checkpoint_metadata_and_matching_predictor() -> None:
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    backend = _LagBackend()
    loader_calls = 0

    def load(method_id: str) -> _LagBackend:
        nonlocal loader_calls
        loader_calls += 1
        return backend

    adapter = LegacyAdapter(loader=load)
    adapter.forecast(_request(LAG_LLAMA_ID))
    adapter.forecast(_request(LAG_LLAMA_ID, history=(4.0, 5.0, 6.0)))

    assert loader_calls == 1
    assert len(backend.download_calls) == 1
    assert len(backend.load_calls) == 1
    assert len(backend.estimator_class.calls) == 1
    assert len(backend.estimator_class.estimators[0].predictor.calls) == 2


def test_tempo_loads_exact_filename_crops_native_context_and_passes_horizon() -> None:
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    backend = _TempoBackend()
    history = tuple(float(index) for index in range(340))
    adapter = LegacyAdapter(loader=lambda method_id: backend)

    assert adapter.forecast(_request(TEMPO_ID, history=history)) == (10.0, 11.0)
    assert backend.model_class.calls == [
        {
            "device": "fake-device",
            "repo_id": "Melady/TEMPO",
            "filename": "TEMPO-80M_v1.pth",
        }
    ]
    assert backend.model.predict_calls == [
        (history[-336:], {"pred_length": 2})
    ]


def test_tempo_supports_official_autoregression_beyond_native_96() -> None:
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    backend = _TempoBackend(output=lambda horizon: list(range(horizon)))
    result = LegacyAdapter(loader=lambda method_id: backend).forecast(
        _request(TEMPO_ID, horizon=97)
    )

    assert len(result) == 97
    assert result[-1] == 96.0
    assert backend.model.predict_calls == [
        ((1.0, 2.0, 3.0), {"pred_length": 97})
    ]


def test_tempo_reuses_one_backend_and_model() -> None:
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    backend = _TempoBackend()
    loader_calls = 0

    def load(method_id: str) -> _TempoBackend:
        nonlocal loader_calls
        loader_calls += 1
        return backend

    adapter = LegacyAdapter(loader=load)
    adapter.forecast(_request(TEMPO_ID))
    adapter.forecast(_request(TEMPO_ID))

    assert loader_calls == 1
    assert len(backend.model_class.calls) == 1
    assert len(backend.model.predict_calls) == 2


@pytest.mark.parametrize("method_id", [TIMESFM_ID, LAG_LLAMA_ID, TEMPO_ID])
def test_smoke_revision_reaches_each_legacy_checkpoint_loader(
    method_id: str,
) -> None:
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    backend: object
    if method_id == TIMESFM_ID:
        backend = _TimesFMBackend()
    elif method_id == LAG_LLAMA_ID:
        backend = _LagBackend()
    else:
        backend = _TempoBackend()
    adapter = LegacyAdapter(loader=lambda selected: backend)
    request = _attested_request(method_id, REVISION_A)

    adapter.forecast(request)

    if method_id == TIMESFM_ID:
        assert backend.download_calls == [  # type: ignore[attr-defined]
            (
                "google/timesfm-1.0-200m-pytorch",
                "torch_model.ckpt",
                REVISION_A,
            )
        ]
        assert backend.checkpoint_class.calls == [  # type: ignore[attr-defined]
            {"version": "torch", "path": f"/hub/{REVISION_A}/torch_model.ckpt"}
        ]
    elif method_id == LAG_LLAMA_ID:
        assert backend.download_calls == [  # type: ignore[attr-defined]
            (
                "time-series-foundation-models/Lag-Llama",
                "lag-llama.ckpt",
                REVISION_A,
            )
        ]
        assert backend.load_calls == [  # type: ignore[attr-defined]
            f"/hub/{REVISION_A}/lag-llama.ckpt"
        ]
    else:
        assert backend.exact_load_calls == [  # type: ignore[attr-defined]
            ("Melady/TEMPO", "TEMPO-80M_v1.pth", REVISION_A)
        ]
    assert adapter.loaded_checkpoint_revision(request) == REVISION_A


@pytest.mark.parametrize("method_id", [TIMESFM_ID, LAG_LLAMA_ID, TEMPO_ID])
def test_legacy_checkpoint_caches_are_scoped_to_immutable_revision(
    method_id: str,
) -> None:
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    backend: object
    if method_id == TIMESFM_ID:
        backend = _TimesFMBackend()
    elif method_id == LAG_LLAMA_ID:
        backend = _LagBackend()
    else:
        backend = _TempoBackend()
    adapter = LegacyAdapter(loader=lambda selected: backend)

    adapter.forecast(_attested_request(method_id, REVISION_A))
    adapter.forecast(_attested_request(method_id, REVISION_B))

    if method_id in {TIMESFM_ID, LAG_LLAMA_ID}:
        calls = backend.download_calls  # type: ignore[attr-defined]
    else:
        calls = backend.exact_load_calls  # type: ignore[attr-defined]
    assert len(calls) == 2


def test_rejects_non_manifest_binding_before_loading() -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    loader_calls = 0

    def load(method_id: str) -> object:
        nonlocal loader_calls
        loader_calls += 1
        return _TempoBackend()

    with pytest.raises(RequestUnavailableError, match="reviewed"):
        LegacyAdapter(loader=load).forecast(
            _request(TEMPO_ID, runtime_options={"checkpoint_file": "attacker.pth"})
        )
    assert loader_calls == 0


def test_dependency_and_checkpoint_failures_remain_distinct() -> None:
    from numerical_agent.tsfm.workers.common import (
        CheckpointUnavailableError,
        DependencyUnavailableError,
    )
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    def missing_dependency(method_id: str) -> object:
        raise ImportError("legacy dependency missing")

    with pytest.raises(DependencyUnavailableError, match="dependency missing"):
        LegacyAdapter(loader=missing_dependency).forecast(_request(TEMPO_ID))

    backend = _TempoBackend()

    def broken_checkpoint(**kwargs: object) -> object:
        raise OSError("weights unavailable")

    backend.model_class.load_pretrained_model = broken_checkpoint  # type: ignore[method-assign]
    with pytest.raises(CheckpointUnavailableError, match="weights unavailable"):
        LegacyAdapter(loader=lambda method_id: backend).forecast(_request(TEMPO_ID))


@pytest.mark.parametrize(
    ("method_id", "backend"),
    [
        (TIMESFM_ID, _TimesFMBackend(point=[[10.0, math.inf]])),
        (LAG_LLAMA_ID, _LagBackend(values=[10.0])),
        (TEMPO_ID, _TempoBackend(output=[[10.0, 11.0]])),
    ],
)
def test_malformed_or_non_finite_output_is_runtime_error(
    method_id: str, backend: object
) -> None:
    from numerical_agent.tsfm.workers.common import ModelOutputError
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    with pytest.raises(ModelOutputError):
        LegacyAdapter(loader=lambda selected: backend).forecast(_request(method_id))


def test_worker_main_maps_request_and_model_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.workers.legacy import LegacyAdapter

    output = io.StringIO()
    monkeypatch.setattr(
        worker_main,
        "_load_adapter",
        lambda name: LegacyAdapter(loader=lambda method_id: _TimesFMBackend()),
    )
    worker_main.serve(
        "legacy",
        io.StringIO(_request(TIMESFM_ID, frequency="invalid").to_json() + "\n"),
        output,
    )
    response = json.loads(output.getvalue())
    assert response["status"] == "unavailable"
    assert response["reason_code"] == "request_unavailable"

    output = io.StringIO()
    monkeypatch.setattr(
        worker_main,
        "_load_adapter",
        lambda name: LegacyAdapter(
            loader=lambda method_id: _TempoBackend(output=[math.nan, 2.0])
        ),
    )
    worker_main.serve(
        "legacy", io.StringIO(_request(TEMPO_ID).to_json() + "\n"), output
    )
    response = json.loads(output.getvalue())
    assert response["status"] == "runtime_error"
    assert response["reason_code"] == "adapter_runtime_error"


def test_module_import_is_lazy_for_all_optional_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional_roots = {
        "torch",
        "timesfm",
        "pandas",
        "gluonts",
        "huggingface_hub",
        "lag_llama",
        "tempo",
    }
    imported: list[str] = []
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> Any:
        root = name.split(".", 1)[0]
        if root in optional_roots:
            imported.append(name)
            raise AssertionError(f"optional import during module load: {name}")
        return real_import(name, globals, locals, fromlist, level)

    sys.modules.pop("numerical_agent.tsfm.workers.legacy", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.import_module("numerical_agent.tsfm.workers.legacy")
    assert imported == []


def test_worker_main_has_reviewed_legacy_target() -> None:
    from numerical_agent.tsfm import worker_main

    assert worker_main._ADAPTER_TARGETS["legacy"] == (
        "numerical_agent.tsfm.workers.legacy",
        "LegacyAdapter",
    )
