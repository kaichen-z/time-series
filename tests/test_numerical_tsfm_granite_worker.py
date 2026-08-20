from __future__ import annotations

from contextlib import nullcontext
import builtins
import dataclasses
import io
import json
import math
from typing import Any

import pytest

from numerical_agent.tsfm import ManifestRegistry
from numerical_agent.tsfm.protocol import WorkerRequest


TTM_ID = "method_tsfm_0006"
FLOWSTATE_ID = "method_tsfm_0020"
PATCHTST_ID = "method_tsfm_0030"
REVISION_A = "a" * 40
REVISION_B = "b" * 40


def _request(
    method_id: str,
    *,
    history: tuple[float, ...] = (1.0, 2.0, 3.0),
    horizon: int = 2,
    frequency: str = "H",
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


class _FakeModel:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []
        self.eval_calls = 0

    def eval(self) -> "_FakeModel":
        self.eval_calls += 1
        return self

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self.output


class _FakeFromPretrained:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, checkpoint: str, **kwargs: object) -> _FakeModel:
        self.calls.append((checkpoint, dict(kwargs)))
        return self.model


class _FakeBackend:
    def __init__(
        self,
        *,
        ttm_output: object | None = None,
        flowstate_output: object | None = None,
        patchtst_output: object | None = None,
    ) -> None:
        self.ttm_model = _FakeModel(
            _Output(
                prediction_outputs=[
                    [[10.0], [11.0]] + [[0.0] for _ in range(94)]
                ]
            )
            if ttm_output is None
            else ttm_output
        )
        self.flowstate_model = _FakeModel(
            _Output(prediction_outputs=[[[20.0], [21.0]]])
            if flowstate_output is None
            else flowstate_output
        )
        self.patchtst_model = _FakeModel(
            _Output(quantile_outputs=[[[[30.0], [31.0]]]])
            if patchtst_output is None
            else patchtst_output
        )
        self.flowstate_class = _FakeFromPretrained(self.flowstate_model)
        self.patchtst_class = _FakeFromPretrained(self.patchtst_model)
        self.get_model_calls: list[dict[str, object]] = []
        self.tensor_calls: list[tuple[tuple[float, ...], str]] = []
        self.observed_mask_calls: list[object] = []
        self.frequency_token_calls: list[int] = []

    def get_model(self, **kwargs: object) -> _FakeModel:
        self.get_model_calls.append(dict(kwargs))
        return self.ttm_model

    def tensor(self, values: tuple[float, ...], *, layout: str) -> object:
        copied = tuple(values)
        self.tensor_calls.append((copied, layout))
        return {"values": copied, "layout": layout}

    @staticmethod
    def no_grad():
        return nullcontext()

    def observed_mask(self, tensor: object) -> object:
        self.observed_mask_calls.append(tensor)
        return {"observed": tensor}

    def frequency_token(self, value: int) -> object:
        self.frequency_token_calls.append(value)
        return {"frequency_token": value}


@dataclasses.dataclass
class _Output:
    prediction_outputs: object | None = None
    quantile_outputs: object | None = None


def test_ttm_selects_the_audited_branch_and_reuses_the_model() -> None:
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    backend = _FakeBackend()
    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return backend

    adapter = GraniteAdapter(loader=load)
    history = tuple(float(value) for value in range(520))

    assert adapter.forecast(_request(TTM_ID, history=history)) == (10.0, 11.0)
    assert adapter.forecast(_request(TTM_ID, history=(5.0, 6.0))) == (10.0, 11.0)

    assert loader_calls == 1
    assert backend.get_model_calls == [
        {
            "model_path": "ibm-granite/granite-timeseries-ttm-r2",
            "context_length": 512,
            "prediction_length": 96,
            "model_revision": "512-96-ft-r2.1",
            "freq_prefix_tuning": True,
            "freq": "h",
        }
    ]
    assert backend.ttm_model.eval_calls == 1
    assert backend.tensor_calls == [
        (history[-512:], "batch_time_channel"),
        ((5.0, 6.0), "batch_time_channel"),
    ]
    assert backend.ttm_model.calls == [
        {
            "past_values": {
                "values": history[-512:],
                "layout": "batch_time_channel",
            },
            "past_observed_mask": {
                "observed": {
                    "values": history[-512:],
                    "layout": "batch_time_channel",
                }
            },
            "freq_token": {"frequency_token": 7},
        },
        {
            "past_values": {
                "values": (5.0, 6.0),
                "layout": "batch_time_channel",
            },
            "past_observed_mask": {
                "observed": {
                    "values": (5.0, 6.0),
                    "layout": "batch_time_channel",
                }
            },
            "freq_token": {"frequency_token": 7},
        },
    ]
    assert backend.frequency_token_calls == [7, 7]


@pytest.mark.parametrize(
    ("frequency", "normalized", "token"),
    [
        ("1 hour", "h", 7),
        ("1 day", "d", 8),
        ("1 week", "W", 9),
    ],
)
def test_ttm_uses_the_audited_revision_and_frequency_token(
    frequency: str,
    normalized: str,
    token: int,
) -> None:
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    backend = _FakeBackend()

    assert GraniteAdapter(loader=lambda: backend).forecast(
        _request(TTM_ID, frequency=frequency)
    ) == (10.0, 11.0)

    assert backend.get_model_calls == [
        {
            "model_path": "ibm-granite/granite-timeseries-ttm-r2",
            "context_length": 512,
            "prediction_length": 96,
            "model_revision": "512-96-ft-r2.1",
            "freq_prefix_tuning": True,
            "freq": normalized,
        }
    ]
    assert backend.frequency_token_calls == [token]
    assert backend.ttm_model.calls[0]["freq_token"] == {
        "frequency_token": token
    }


def test_ttm_rejects_unreviewed_frequency_before_loading() -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _FakeBackend()

    with pytest.raises(RequestUnavailableError, match="frequency"):
        GraniteAdapter(loader=load).forecast(
            _request(TTM_ID, frequency="15min")
        )

    assert loader_calls == 0


def test_flowstate_loads_r1p1_and_requests_the_documented_median() -> None:
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    backend = _FakeBackend()
    adapter = GraniteAdapter(loader=lambda: backend)

    assert adapter.forecast(_request(FLOWSTATE_ID)) == (20.0, 21.0)

    assert backend.flowstate_class.calls == [
        ("ibm-research/flowstate", {"revision": "r1.1"})
    ]
    assert backend.flowstate_model.eval_calls == 1
    assert backend.tensor_calls == [
        ((1.0, 2.0, 3.0), "time_batch_channel")
    ]
    assert backend.flowstate_model.calls == [
        {
            "past_values": {
                "values": (1.0, 2.0, 3.0),
                "layout": "time_batch_channel",
            },
            "scale_factor": 1.0,
            "prediction_length": 2,
            "batch_first": False,
            "prediction_type": "median",
        }
    ]


@pytest.mark.parametrize(
    ("frequency", "scale_factor"),
    [
        ("15min", 0.25),
        ("30min", 0.5),
        ("H", 1.0),
        ("1 hour", 1.0),
        ("W", 0.46),
        ("1 week", 0.46),
        ("M", 2.0),
        ("1 month", 2.0),
    ],
)
def test_flowstate_uses_documented_unambiguous_frequency_scales(
    frequency: str,
    scale_factor: float,
) -> None:
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    backend = _FakeBackend()

    assert GraniteAdapter(loader=lambda: backend).forecast(
        _request(FLOWSTATE_ID, frequency=frequency)
    ) == (20.0, 21.0)

    assert backend.flowstate_model.calls[0]["scale_factor"] == scale_factor


def test_flowstate_rejects_ambiguous_daily_scaling_before_loading() -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _FakeBackend()

    with pytest.raises(RequestUnavailableError, match="frequency"):
        GraniteAdapter(loader=load).forecast(
            _request(FLOWSTATE_ID, frequency="D")
        )

    assert loader_calls == 0


def test_flowstate_preserves_scaled_15_minute_history_up_to_16384() -> None:
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    backend = _FakeBackend()
    adapter = GraniteAdapter(loader=lambda: backend)
    ten_thousand = tuple(float(value) for value in range(10_000))
    boundary_plus_one = tuple(float(value) for value in range(16_385))

    assert adapter.forecast(
        _request(FLOWSTATE_ID, history=ten_thousand, frequency="15min")
    ) == (20.0, 21.0)
    assert adapter.forecast(
        _request(FLOWSTATE_ID, history=boundary_plus_one, frequency="15min")
    ) == (20.0, 21.0)

    assert backend.tensor_calls == [
        (ten_thousand, "time_batch_channel"),
        (boundary_plus_one[-16_384:], "time_batch_channel"),
    ]


def test_patchtst_loads_r1_and_extracts_only_the_median_quantile() -> None:
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    backend = _FakeBackend()
    adapter = GraniteAdapter(loader=lambda: backend)

    assert adapter.forecast(_request(PATCHTST_ID)) == (30.0, 31.0)

    assert backend.patchtst_class.calls == [
        ("ibm-research/patchtst-fm-r1", {"revision": "main"})
    ]
    assert backend.patchtst_model.eval_calls == 1
    assert backend.tensor_calls == [
        ((1.0, 2.0, 3.0), "batch_time_channel")
    ]
    assert backend.patchtst_model.calls == [
        {
            "past_values": {
                "values": (1.0, 2.0, 3.0),
                "layout": "batch_time_channel",
            },
            "prediction_length": 2,
            "quantile_levels": [0.5],
        }
    ]


@pytest.mark.parametrize("method_id", [TTM_ID, FLOWSTATE_ID, PATCHTST_ID])
def test_smoke_revision_overrides_mutable_granite_loader_revision(
    method_id: str,
) -> None:
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    backend = _FakeBackend()
    adapter = GraniteAdapter(loader=lambda: backend)
    request = _attested_request(method_id, REVISION_A)

    assert adapter.forecast(request) in {
        (10.0, 11.0),
        (20.0, 21.0),
        (30.0, 31.0),
    }
    if method_id == TTM_ID:
        assert backend.get_model_calls[0]["model_revision"] == REVISION_A
    elif method_id == FLOWSTATE_ID:
        assert backend.flowstate_class.calls == [
            ("ibm-research/flowstate", {"revision": REVISION_A})
        ]
    else:
        assert backend.patchtst_class.calls == [
            ("ibm-research/patchtst-fm-r1", {"revision": REVISION_A})
        ]
    assert adapter.loaded_checkpoint_revision(request) == REVISION_A


def test_granite_model_cache_is_scoped_to_immutable_revision() -> None:
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    backend = _FakeBackend()
    adapter = GraniteAdapter(loader=lambda: backend)

    adapter.forecast(_attested_request(FLOWSTATE_ID, REVISION_A))
    adapter.forecast(_attested_request(FLOWSTATE_ID, REVISION_B))

    assert backend.flowstate_class.calls == [
        ("ibm-research/flowstate", {"revision": REVISION_A}),
        ("ibm-research/flowstate", {"revision": REVISION_B}),
    ]


def test_patchtst_rejects_context_plus_horizon_above_8192_before_loading() -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    backend = _FakeBackend()
    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return backend

    adapter = GraniteAdapter(loader=load)

    with pytest.raises(RequestUnavailableError, match="8192"):
        adapter.forecast(
            _request(PATCHTST_ID, history=(1.0,) * 8191, horizon=2)
        )

    assert loader_calls == 0
    assert backend.patchtst_class.calls == []


def test_ttm_rejects_horizons_beyond_the_audited_branch_before_loading() -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _FakeBackend()

    with pytest.raises(RequestUnavailableError, match="96"):
        GraniteAdapter(loader=load).forecast(_request(TTM_ID, horizon=97))

    assert loader_calls == 0


@pytest.mark.parametrize(
    ("method_id", "backend_kwargs", "message"),
    [
        (
            TTM_ID,
            {"ttm_output": _Output(prediction_outputs=[10.0, 11.0])},
            "shape",
        ),
        (
            FLOWSTATE_ID,
            {
                "flowstate_output": _Output(
                    prediction_outputs=[[[20.0], [math.nan]]]
                )
            },
            "finite",
        ),
        (
            PATCHTST_ID,
            {
                "patchtst_output": _Output(
                    quantile_outputs=[[[[30.0]]]]
                )
            },
            "horizon",
        ),
    ],
)
def test_granite_models_must_return_a_finite_one_dimensional_horizon(
    method_id: str,
    backend_kwargs: dict[str, object],
    message: str,
) -> None:
    from numerical_agent.tsfm.workers.common import ModelOutputError
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    adapter = GraniteAdapter(loader=lambda: _FakeBackend(**backend_kwargs))

    with pytest.raises(ModelOutputError, match=message):
        adapter.forecast(_request(method_id))


@pytest.mark.parametrize(
    "substitute",
    [
        {"provider": "attacker"},
        {"checkpoint": "attacker/checkpoint"},
        {"runtime_options": {"revision": "attacker"}},
    ],
)
def test_granite_rejects_requests_outside_the_immutable_manifest_binding(
    substitute: dict[str, object],
) -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    backend = _FakeBackend()
    request = dataclasses.replace(_request(FLOWSTATE_ID), **substitute)

    with pytest.raises(RequestUnavailableError, match="reviewed Granite manifest"):
        GraniteAdapter(loader=lambda: backend).forecast(request)

    assert backend.flowstate_class.calls == []


def test_dependency_and_checkpoint_failures_have_distinct_unavailability_types() -> None:
    from numerical_agent.tsfm.workers.common import (
        CheckpointUnavailableError,
        DependencyUnavailableError,
    )
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    def missing_dependency() -> object:
        raise ImportError("torch missing")

    with pytest.raises(DependencyUnavailableError, match="Granite TSFM dependencies"):
        GraniteAdapter(loader=missing_dependency).forecast(_request(TTM_ID))

    backend = _FakeBackend()

    def missing_checkpoint(checkpoint: str, **kwargs: object) -> object:
        del checkpoint, kwargs
        raise OSError("checkpoint absent")

    backend.flowstate_class.from_pretrained = missing_checkpoint  # type: ignore[method-assign]
    with pytest.raises(CheckpointUnavailableError, match="checkpoint absent"):
        GraniteAdapter(loader=lambda: backend).forecast(_request(FLOWSTATE_ID))

    def broken_dependency(checkpoint: str, **kwargs: object) -> object:
        del checkpoint, kwargs
        raise ImportError("incompatible transformers")

    backend.patchtst_class.from_pretrained = broken_dependency  # type: ignore[method-assign]
    with pytest.raises(DependencyUnavailableError, match="incompatible transformers"):
        GraniteAdapter(loader=lambda: backend).forecast(_request(PATCHTST_ID))


def test_injected_granite_execution_never_imports_optional_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.granite import GraniteAdapter

    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("tsfm_public"):
            raise AssertionError(f"unexpected optional import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    adapter = GraniteAdapter(loader=lambda: _FakeBackend())
    assert adapter.forecast(_request(FLOWSTATE_ID)) == (20.0, 21.0)


def test_worker_main_exposes_typed_granite_unavailability() -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.workers.common import DependencyUnavailableError

    class _UnavailableAdapter:
        def forecast(self, request: WorkerRequest) -> tuple[float, ...]:
            del request
            raise DependencyUnavailableError("missing granite-tsfm")

    request = _request(TTM_ID)
    output = io.StringIO()
    original = worker_main._load_adapter
    worker_main._load_adapter = lambda name: _UnavailableAdapter()
    try:
        worker_main.serve("granite", io.StringIO(request.to_json() + "\n"), output)
    finally:
        worker_main._load_adapter = original

    response = json.loads(output.getvalue())
    assert response["status"] == "unavailable"
    assert response["reason_code"] == "dependency_unavailable"


def test_worker_main_reports_model_output_validation_as_runtime_error() -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.workers.common import ModelOutputError

    class _BadOutputAdapter:
        def forecast(self, request: WorkerRequest) -> tuple[float, ...]:
            del request
            raise ModelOutputError("model returned NaN")

    request = _request(TTM_ID)
    output = io.StringIO()
    original = worker_main._load_adapter
    worker_main._load_adapter = lambda name: _BadOutputAdapter()
    try:
        worker_main.serve("granite", io.StringIO(request.to_json() + "\n"), output)
    finally:
        worker_main._load_adapter = original

    response = json.loads(output.getvalue())
    assert response["status"] == "runtime_error"
    assert response["reason_code"] == "adapter_runtime_error"


def test_worker_main_only_uses_typed_adapter_invalid_request() -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.workers.common import InvalidRequestError

    class _InvalidRequestAdapter:
        def forecast(self, request: WorkerRequest) -> tuple[float, ...]:
            del request
            raise InvalidRequestError("invalid adapter request")

    request = _request(TTM_ID)
    output = io.StringIO()
    original = worker_main._load_adapter
    worker_main._load_adapter = lambda name: _InvalidRequestAdapter()
    try:
        worker_main.serve("granite", io.StringIO(request.to_json() + "\n"), output)
    finally:
        worker_main._load_adapter = original

    response = json.loads(output.getvalue())
    assert response["status"] == "invalid_request"
    assert response["reason_code"] == "adapter_rejected_request"


def test_worker_main_reports_official_model_value_error_as_runtime_error() -> None:
    from numerical_agent.tsfm import worker_main

    class _OfficialValueErrorAdapter:
        def forecast(self, request: WorkerRequest) -> tuple[float, ...]:
            del request
            raise ValueError("official forward rejected tensor")

    request = _request(TTM_ID)
    output = io.StringIO()
    original = worker_main._load_adapter
    worker_main._load_adapter = lambda name: _OfficialValueErrorAdapter()
    try:
        worker_main.serve("granite", io.StringIO(request.to_json() + "\n"), output)
    finally:
        worker_main._load_adapter = original

    response = json.loads(output.getvalue())
    assert response["status"] == "runtime_error"
    assert response["reason_code"] == "adapter_runtime_error"
