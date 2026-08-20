from __future__ import annotations

import builtins
import dataclasses
import importlib
import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from numerical_agent.tsfm import ManifestRegistry
from numerical_agent.tsfm.protocol import WorkerRequest


MOIRAI_ID = "method_tsfm_0003"
MOIRAI2_ID = "method_tsfm_0017"
MOIRAI_MOE_ID = "method_tsfm_0019"
REVISION_A = "a" * 40
REVISION_B = "b" * 40


def _request(
    method_id: str,
    *,
    history: tuple[float, ...] = (1.0, 2.0, 3.0),
    horizon: int = 2,
    frequency: str = "1 hour",
) -> WorkerRequest:
    manifest = ManifestRegistry.load_default()[method_id]
    return WorkerRequest(
        request_id=f"request-{method_id}",
        provider=manifest.adapter,
        checkpoint=manifest.checkpoint,
        history=history,
        horizon=horizon,
        frequency=frequency,
        runtime_options=dict(manifest.runtime_options),
    )


def _attested_request(method_id: str, revision: str) -> WorkerRequest:
    request = _request(method_id)
    object.__setattr__(request, "checkpoint_revision", revision)
    return request


class _FakeModule:
    pass


class _FakeModuleClass:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.module = _FakeModule()

    def from_pretrained(self, checkpoint: str) -> _FakeModule:
        self.calls.append(checkpoint)
        return self.module


class _RevisionModuleClass:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.module = _FakeModule()

    def from_pretrained(
        self, checkpoint: str, **kwargs: object
    ) -> _FakeModule:
        self.calls.append((checkpoint, dict(kwargs)))
        return self.module


class _SampleForecast:
    def __init__(
        self,
        samples: object | None = None,
        quantile_value: object | None = None,
    ) -> None:
        self.samples = (
            [[1.0, 10.0], [3.0, 30.0], [100.0, 50.0]]
            if samples is None
            else samples
        )
        if quantile_value is not None:
            self.quantile_value = quantile_value
        elif samples is None:
            self.quantile_value = [3.0, 30.0]
        else:
            self.quantile_value = None
        self.quantile_calls: list[float] = []

    def quantile(self, level: float) -> object:
        self.quantile_calls.append(level)
        if self.quantile_value is not None:
            return self.quantile_value
        sample_rows = list(self.samples)
        sample_index = round((len(sample_rows) - 1) * level)
        return [sorted(column)[sample_index] for column in zip(*sample_rows)]


class _QuantileForecast:
    def __init__(self, median: object | None = None) -> None:
        self.median = [20.0, 21.0] if median is None else median
        self.quantile_calls: list[float] = []

    def quantile(self, level: float) -> object:
        self.quantile_calls.append(level)
        return self.median


class _FakePredictor:
    def __init__(self, forecast: object) -> None:
        self.forecast = forecast
        self.predict_calls: list[object] = []

    def predict(self, dataset: object):
        self.predict_calls.append(dataset)
        return iter((self.forecast,))


class _FakeForecastClass:
    def __init__(self, forecast: object) -> None:
        self.forecast = forecast
        self.calls: list[dict[str, object]] = []
        self.predictor_calls: list[dict[str, object]] = []
        self.predictors: list[_FakePredictor] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        owner = self

        class ForecastModel:
            def create_predictor(self, **predictor_kwargs: object) -> _FakePredictor:
                owner.predictor_calls.append(dict(predictor_kwargs))
                predictor = _FakePredictor(owner.forecast)
                owner.predictors.append(predictor)
                return predictor

        return ForecastModel()


class _FakeBackend:
    def __init__(
        self,
        *,
        sample_forecast: object | None = None,
        quantile_forecast: object | None = None,
    ) -> None:
        self.moirai_module_class = _FakeModuleClass()
        self.moirai2_module_class = _FakeModuleClass()
        self.moirai_moe_module_class = _FakeModuleClass()
        self.moirai_forecast_class = _FakeForecastClass(
            _SampleForecast() if sample_forecast is None else sample_forecast
        )
        self.moirai2_forecast_class = _FakeForecastClass(
            _QuantileForecast() if quantile_forecast is None else quantile_forecast
        )
        self.moirai_moe_forecast_class = _FakeForecastClass(
            _SampleForecast() if sample_forecast is None else sample_forecast
        )


class _DatasetFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[float, ...], str]] = []
        self.datasets: list[object] = []

    def __call__(self, history: tuple[float, ...], frequency: str) -> object:
        self.calls.append((history, frequency))
        dataset = {"dataset": len(self.datasets)}
        self.datasets.append(dataset)
        return dataset


@pytest.mark.parametrize(
    ("method_id", "checkpoint", "module_attr", "forecast_attr", "expected"),
    [
        (
            MOIRAI_ID,
            "Salesforce/moirai-1.1-R-base",
            "moirai_module_class",
            "moirai_forecast_class",
            (3.0, 30.0),
        ),
        (
            MOIRAI2_ID,
            "Salesforce/moirai-2.0-R-small",
            "moirai2_module_class",
            "moirai2_forecast_class",
            (20.0, 21.0),
        ),
        (
            MOIRAI_MOE_ID,
            "Salesforce/moirai-moe-1.0-R-small",
            "moirai_moe_module_class",
            "moirai_moe_forecast_class",
            (3.0, 30.0),
        ),
    ],
)
def test_dispatches_each_checkpoint_to_its_exact_official_class_pair(
    method_id: str,
    checkpoint: str,
    module_attr: str,
    forecast_attr: str,
    expected: tuple[float, ...],
) -> None:
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    backend = _FakeBackend()
    datasets = _DatasetFactory()

    assert Uni2TSAdapter(loader=lambda: backend, dataset_factory=datasets).forecast(
        _request(method_id)
    ) == expected

    assert getattr(backend, module_attr).calls == [checkpoint]
    assert len(getattr(backend, forecast_attr).calls) == 1
    for other_attr in {
        "moirai_module_class",
        "moirai2_module_class",
        "moirai_moe_module_class",
    } - {module_attr}:
        assert getattr(backend, other_attr).calls == []


def test_builds_univariate_no_covariate_forecast_and_propagates_official_options() -> None:
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    backend = _FakeBackend()
    datasets = _DatasetFactory()
    request = _request(MOIRAI_ID, history=(4.0, 5.0), horizon=2)

    assert Uni2TSAdapter(loader=lambda: backend, dataset_factory=datasets).forecast(
        request
    ) == (3.0, 30.0)

    assert datasets.calls == [((4.0, 5.0), "h")]
    assert backend.moirai_forecast_class.calls == [
        {
            "module": backend.moirai_module_class.module,
            "prediction_length": 2,
            "context_length": 2,
            "patch_size": "auto",
            "num_samples": 100,
            "target_dim": 1,
            "feat_dynamic_real_dim": 0,
            "past_feat_dynamic_real_dim": 0,
        }
    ]
    assert backend.moirai_forecast_class.predictor_calls == [
        {"batch_size": 1, "device": "auto"}
    ]
    assert backend.moirai_forecast_class.predictors[0].predict_calls == [
        datasets.datasets[0]
    ]


def test_moirai_moe_uses_manifest_patch_size_and_sample_count() -> None:
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    backend = _FakeBackend()
    adapter = Uni2TSAdapter(loader=lambda: backend, dataset_factory=_DatasetFactory())

    assert adapter.forecast(_request(MOIRAI_MOE_ID)) == (3.0, 30.0)
    assert backend.moirai_moe_forecast_class.calls == [
        {
            "module": backend.moirai_moe_module_class.module,
            "prediction_length": 2,
            "context_length": 3,
            "patch_size": 16,
            "num_samples": 100,
            "target_dim": 1,
            "feat_dynamic_real_dim": 0,
            "past_feat_dynamic_real_dim": 0,
        }
    ]


def test_moirai2_uses_its_quantile_api_without_unsupported_sample_arguments() -> None:
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    forecast = _QuantileForecast()
    backend = _FakeBackend(quantile_forecast=forecast)
    adapter = Uni2TSAdapter(loader=lambda: backend, dataset_factory=_DatasetFactory())

    assert adapter.forecast(_request(MOIRAI2_ID)) == (20.0, 21.0)
    assert backend.moirai2_forecast_class.calls == [
        {
            "module": backend.moirai2_module_class.module,
            "prediction_length": 2,
            "context_length": 3,
            "target_dim": 1,
            "feat_dynamic_real_dim": 0,
            "past_feat_dynamic_real_dim": 0,
        }
    ]
    assert backend.moirai2_forecast_class.predictor_calls == [
        {"batch_size": 1, "device": "auto"}
    ]
    assert forecast.quantile_calls == [0.5]


@pytest.mark.parametrize("method_id", [MOIRAI_ID, MOIRAI_MOE_ID])
def test_sample_models_use_official_p50_for_even_manifest_sample_count(
    method_id: str,
) -> None:
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    samples = [[float(value), float(value + 100)] for value in range(100)]
    forecast = _SampleForecast(samples, quantile_value=[50.0, 150.0])
    backend = _FakeBackend(sample_forecast=forecast)
    adapter = Uni2TSAdapter(loader=lambda: backend, dataset_factory=_DatasetFactory())

    assert adapter.forecast(_request(method_id)) == (50.0, 150.0)
    assert forecast.quantile_calls == [0.5]


@pytest.mark.parametrize(
    ("method_id", "history_length", "horizon"),
    [
        (MOIRAI_ID, 4_081, 16),
        (MOIRAI2_ID, 8_177, 16),
        (MOIRAI_MOE_ID, 8_177, 16),
    ],
)
def test_rejects_requests_above_the_manifest_patch_token_budget_before_loading(
    method_id: str,
    history_length: int,
    horizon: int,
) -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    loader_calls = 0
    dataset_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _FakeBackend()

    def dataset_factory(history: tuple[float, ...], frequency: str) -> object:
        nonlocal dataset_calls
        del history, frequency
        dataset_calls += 1
        return object()

    with pytest.raises(RequestUnavailableError, match="512 patch tokens"):
        Uni2TSAdapter(loader=load, dataset_factory=dataset_factory).forecast(
            _request(method_id, history=(1.0,) * history_length, horizon=horizon)
        )

    assert loader_calls == 0
    assert dataset_calls == 0


@pytest.mark.parametrize(
    ("frequency", "normalized"),
    [
        ("D", "D"),
        ("1 day", "D"),
        ("H", "h"),
        ("1 hour", "h"),
        ("15 minutes", "15min"),
        ("W", "W"),
        ("1 week", "W"),
        ("M", "M"),
        ("1 month", "M"),
        ("Q", "Q"),
        ("1 quarter", "Q"),
        ("Y", "Y"),
        ("1 year", "Y"),
    ],
)
def test_normalizes_supported_protocol_frequencies(
    frequency: str, normalized: str
) -> None:
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    datasets = _DatasetFactory()
    adapter = Uni2TSAdapter(loader=_FakeBackend, dataset_factory=datasets)

    assert adapter.forecast(_request(MOIRAI2_ID, frequency=frequency)) == (20.0, 21.0)
    assert datasets.calls == [((1.0, 2.0, 3.0), normalized)]


def test_rejects_unsupported_frequency_before_loading_dependencies() -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _FakeBackend()

    with pytest.raises(RequestUnavailableError, match="frequency"):
        Uni2TSAdapter(loader=load, dataset_factory=_DatasetFactory()).forecast(
            _request(MOIRAI_ID, frequency="business whenever")
        )

    assert loader_calls == 0


def test_default_dataset_factory_builds_one_timestamped_series_without_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.uni2ts import _create_pandas_dataset

    calls: dict[str, object] = {}
    fake_pandas = ModuleType("pandas")

    period_index = SimpleNamespace(freqstr="D")

    class TimestampIndex:
        def to_period(self) -> object:
            calls["to_period"] = True
            return period_index

    timestamp_index = TimestampIndex()

    def date_range(*, start: str, periods: int, freq: str) -> object:
        calls["date_range"] = {"start": start, "periods": periods, "freq": freq}
        return timestamp_index

    def series(values: object, *, index: object, name: str) -> object:
        calls["series"] = {"values": values, "index": index, "name": name}
        target = SimpleNamespace(values=values, index=index, name=name)
        calls["target"] = target
        return target

    fake_pandas.date_range = date_range  # type: ignore[attr-defined]
    fake_pandas.Series = series  # type: ignore[attr-defined]

    fake_gluonts = ModuleType("gluonts")
    fake_dataset_package = ModuleType("gluonts.dataset")
    fake_dataset_module = ModuleType("gluonts.dataset.pandas")

    def pandas_dataset(data: object, *, freq: str) -> object:
        calls["dataset"] = {"data": data, "freq": freq}
        return {"official-dataset": data}

    fake_dataset_module.PandasDataset = pandas_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)
    monkeypatch.setitem(sys.modules, "gluonts", fake_gluonts)
    monkeypatch.setitem(sys.modules, "gluonts.dataset", fake_dataset_package)
    monkeypatch.setitem(sys.modules, "gluonts.dataset.pandas", fake_dataset_module)

    result = _create_pandas_dataset((1.0, 2.0), "D")

    assert calls["date_range"] == {
        "start": "2000-01-01",
        "periods": 2,
        "freq": "D",
    }
    assert calls["series"] == {
        "values": (1.0, 2.0),
        "index": period_index,
        "name": "target",
    }
    assert calls["to_period"] is True
    assert calls["dataset"] == {"data": {"target": calls["target"]}, "freq": "D"}
    assert result == {"official-dataset": {"target": calls["target"]}}


@pytest.mark.parametrize("accepted_date_alias", ["ME", "M"])
def test_default_dataset_factory_handles_modern_and_legacy_month_aliases(
    monkeypatch: pytest.MonkeyPatch,
    accepted_date_alias: str,
) -> None:
    from numerical_agent.tsfm.workers.uni2ts import _create_pandas_dataset

    date_range_calls: list[str] = []
    period_index = SimpleNamespace(freqstr="M")

    class TimestampIndex:
        def to_period(self) -> object:
            return period_index

    fake_pandas = ModuleType("pandas")

    def date_range(*, start: str, periods: int, freq: str) -> object:
        del start, periods
        date_range_calls.append(freq)
        if freq != accepted_date_alias:
            raise ValueError(f"unsupported alias {freq}")
        return TimestampIndex()

    def series(values: object, *, index: object, name: str) -> object:
        return SimpleNamespace(values=values, index=index, name=name)

    fake_pandas.date_range = date_range  # type: ignore[attr-defined]
    fake_pandas.Series = series  # type: ignore[attr-defined]
    fake_gluonts = ModuleType("gluonts")
    fake_dataset_package = ModuleType("gluonts.dataset")
    fake_dataset_module = ModuleType("gluonts.dataset.pandas")
    dataset_calls: list[tuple[object, str]] = []

    def pandas_dataset(data: object, *, freq: str) -> object:
        dataset_calls.append((data, freq))
        return object()

    fake_dataset_module.PandasDataset = pandas_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)
    monkeypatch.setitem(sys.modules, "gluonts", fake_gluonts)
    monkeypatch.setitem(sys.modules, "gluonts.dataset", fake_dataset_package)
    monkeypatch.setitem(sys.modules, "gluonts.dataset.pandas", fake_dataset_module)

    _create_pandas_dataset((1.0, 2.0), "M")

    assert date_range_calls == (
        ["ME"] if accepted_date_alias == "ME" else ["ME", "M"]
    )
    assert dataset_calls[0][1] == "M"


def test_default_dataset_rejects_invalid_pandas_frequency_before_checkpoint_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    fake_pandas = ModuleType("pandas")

    def invalid_date_range(**kwargs: object) -> object:
        del kwargs
        raise ValueError("invalid pandas frequency")

    fake_pandas.date_range = invalid_date_range  # type: ignore[attr-defined]
    fake_pandas.Series = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    fake_gluonts = ModuleType("gluonts")
    fake_dataset_package = ModuleType("gluonts.dataset")
    fake_dataset_module = ModuleType("gluonts.dataset.pandas")
    fake_dataset_module.PandasDataset = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: object()
    )
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)
    monkeypatch.setitem(sys.modules, "gluonts", fake_gluonts)
    monkeypatch.setitem(sys.modules, "gluonts.dataset", fake_dataset_package)
    monkeypatch.setitem(sys.modules, "gluonts.dataset.pandas", fake_dataset_module)
    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _FakeBackend()

    with pytest.raises(RequestUnavailableError, match="frequency"):
        Uni2TSAdapter(loader=load).forecast(_request(MOIRAI_ID, frequency="FOO"))

    assert loader_calls == 0


def test_reuses_backend_and_pretrained_module_inside_the_worker_process() -> None:
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    backend = _FakeBackend()
    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return backend

    adapter = Uni2TSAdapter(loader=load, dataset_factory=_DatasetFactory())

    assert adapter.forecast(_request(MOIRAI_ID)) == (3.0, 30.0)
    assert adapter.forecast(_request(MOIRAI_ID, history=(7.0, 8.0))) == (3.0, 30.0)
    assert loader_calls == 1
    assert backend.moirai_module_class.calls == ["Salesforce/moirai-1.1-R-base"]
    assert len(backend.moirai_forecast_class.calls) == 2


@pytest.mark.parametrize(
    ("method_id", "module_attr", "checkpoint"),
    [
        (MOIRAI_ID, "moirai_module_class", "Salesforce/moirai-1.1-R-base"),
        (MOIRAI2_ID, "moirai2_module_class", "Salesforce/moirai-2.0-R-small"),
        (
            MOIRAI_MOE_ID,
            "moirai_moe_module_class",
            "Salesforce/moirai-moe-1.0-R-small",
        ),
    ],
)
def test_smoke_revision_is_passed_to_each_uni2ts_hub_mixin(
    method_id: str, module_attr: str, checkpoint: str
) -> None:
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    backend = _FakeBackend()
    revision_class = _RevisionModuleClass()
    setattr(backend, module_attr, revision_class)
    adapter = Uni2TSAdapter(loader=lambda: backend, dataset_factory=_DatasetFactory())
    request = _attested_request(method_id, REVISION_A)

    adapter.forecast(request)

    assert revision_class.calls == [
        (checkpoint, {"revision": REVISION_A})
    ]
    assert adapter.loaded_checkpoint_revision(request) == REVISION_A


def test_uni2ts_module_cache_is_scoped_to_immutable_revision() -> None:
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    backend = _FakeBackend()
    revision_class = _RevisionModuleClass()
    backend.moirai2_module_class = revision_class
    adapter = Uni2TSAdapter(loader=lambda: backend, dataset_factory=_DatasetFactory())

    adapter.forecast(_attested_request(MOIRAI2_ID, REVISION_A))
    adapter.forecast(_attested_request(MOIRAI2_ID, REVISION_B))

    assert revision_class.calls == [
        ("Salesforce/moirai-2.0-R-small", {"revision": REVISION_A}),
        ("Salesforce/moirai-2.0-R-small", {"revision": REVISION_B}),
    ]


@pytest.mark.parametrize(
    "substitute",
    [
        {"provider": "attacker"},
        {"checkpoint": "attacker/checkpoint"},
        {"runtime_options": {"patch_size": 128, "max_patch_tokens": 512}},
    ],
)
def test_rejects_requests_outside_the_immutable_manifest_binding(
    substitute: dict[str, object],
) -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    backend = _FakeBackend()
    request = dataclasses.replace(_request(MOIRAI_ID), **substitute)

    with pytest.raises(RequestUnavailableError, match="reviewed Uni2TS manifest"):
        Uni2TSAdapter(loader=lambda: backend, dataset_factory=_DatasetFactory()).forecast(
            request
        )

    assert backend.moirai_module_class.calls == []


def test_dependency_checkpoint_and_dataset_failures_are_typed() -> None:
    from numerical_agent.tsfm.workers.common import (
        CheckpointUnavailableError,
        DependencyUnavailableError,
    )
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    def missing_dependency() -> object:
        raise ImportError("uni2ts missing")

    with pytest.raises(DependencyUnavailableError, match="Uni2TS dependencies"):
        Uni2TSAdapter(loader=missing_dependency, dataset_factory=_DatasetFactory()).forecast(
            _request(MOIRAI_ID)
        )

    backend = _FakeBackend()

    def missing_checkpoint(checkpoint: str) -> object:
        del checkpoint
        raise OSError("checkpoint absent")

    backend.moirai_module_class.from_pretrained = missing_checkpoint  # type: ignore[method-assign]
    with pytest.raises(CheckpointUnavailableError, match="checkpoint absent"):
        Uni2TSAdapter(loader=lambda: backend, dataset_factory=_DatasetFactory()).forecast(
            _request(MOIRAI_ID)
        )

    def missing_pandas(history: tuple[float, ...], frequency: str) -> object:
        del history, frequency
        raise ImportError("pandas missing")

    with pytest.raises(DependencyUnavailableError, match="pandas missing"):
        Uni2TSAdapter(loader=_FakeBackend, dataset_factory=missing_pandas).forecast(
            _request(MOIRAI2_ID)
        )


@pytest.mark.parametrize(
    ("method_id", "backend", "message"),
    [
        (
            MOIRAI_ID,
            _FakeBackend(sample_forecast=_SampleForecast([[1.0], [2.0]])),
            "horizon",
        ),
        (
            MOIRAI_MOE_ID,
            _FakeBackend(sample_forecast=_SampleForecast([[1.0, math.nan]])),
            "finite",
        ),
        (
            MOIRAI2_ID,
            _FakeBackend(quantile_forecast=_QuantileForecast([[1.0], [2.0]])),
            "shape",
        ),
    ],
)
def test_forecasts_must_have_a_finite_exact_univariate_horizon(
    method_id: str,
    backend: _FakeBackend,
    message: str,
) -> None:
    from numerical_agent.tsfm.workers.common import ModelOutputError
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    with pytest.raises(ModelOutputError, match=message):
        Uni2TSAdapter(loader=lambda: backend, dataset_factory=_DatasetFactory()).forecast(
            _request(method_id)
        )


def test_injected_execution_never_imports_optional_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("numerical_agent.tsfm.workers.uni2ts")
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pandas" or name.startswith(("gluonts", "uni2ts")):
            raise AssertionError(f"unexpected optional import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    adapter = module.Uni2TSAdapter(
        loader=_FakeBackend,
        dataset_factory=_DatasetFactory(),
    )
    assert adapter.forecast(_request(MOIRAI_MOE_ID)) == (3.0, 30.0)


def test_worker_main_reviewed_map_loads_uni2ts_without_loading_a_checkpoint() -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter

    assert isinstance(worker_main._load_adapter("uni2ts"), Uni2TSAdapter)
