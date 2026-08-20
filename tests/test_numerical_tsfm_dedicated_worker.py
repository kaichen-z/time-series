from __future__ import annotations

from contextlib import nullcontext
import builtins
import importlib
import io
import json
import math
import os
import sys
from typing import Any

import pytest

from numerical_agent.tsfm import ManifestRegistry
from numerical_agent.tsfm.protocol import WorkerRequest


TOTO_ID = "method_tsfm_0014"
TIREX_ID = "method_tsfm_0027"
TABPFN_ID = "method_tsfm_0029"
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


class _FromPretrained:
    def __init__(self, model: object) -> None:
        self.model = model
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, checkpoint: str, **kwargs: object) -> object:
        self.calls.append((checkpoint, dict(kwargs)))
        return self.model


class _TiRexModel:
    def __init__(self, mean: object = None) -> None:
        self.mean = [[10.0, 11.0]] if mean is None else mean
        self.calls: list[dict[str, object]] = []

    def forecast(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return [[[-1.0]]], self.mean


class _TiRexBackend:
    def __init__(self, mean: object = None) -> None:
        self.model = _TiRexModel(mean)
        self.load_calls: list[tuple[str, dict[str, object]]] = []
        self.tensor_calls: list[tuple[tuple[float, ...], str]] = []

    def load_model(self, checkpoint: str, **kwargs: object) -> _TiRexModel:
        self.load_calls.append((checkpoint, dict(kwargs)))
        return self.model

    def tensor(self, values: tuple[float, ...], *, layout: str) -> object:
        self.tensor_calls.append((tuple(values), layout))
        return {"values": tuple(values), "layout": layout}


class _TotoModel:
    def __init__(self, output: object = None) -> None:
        self.output = output if output is not None else [
            [[[float(level * 10 + step) for step in range(2)]]]
            for level in range(9)
        ]
        self.to_calls: list[object] = []
        self.eval_calls = 0
        self.forecast_calls: list[tuple[object, dict[str, object]]] = []

    def to(self, device: object) -> "_TotoModel":
        self.to_calls.append(device)
        return self

    def eval(self) -> "_TotoModel":
        self.eval_calls += 1
        return self

    def forecast(self, inputs: object, **kwargs: object) -> object:
        self.forecast_calls.append((inputs, dict(kwargs)))
        return self.output


class _TotoBackend:
    device = "fake-device"

    def __init__(self, output: object = None) -> None:
        self.model = _TotoModel(output)
        self.model_class = _FromPretrained(self.model)
        self.tensor_calls: list[tuple[tuple[float, ...], str, object]] = []

    def tensor(
        self, values: tuple[float, ...], *, layout: str, device: object
    ) -> object:
        self.tensor_calls.append((tuple(values), layout, device))
        return {"values": tuple(values), "layout": layout, "device": device}

    def observed_mask(self, tensor: object) -> object:
        return {"ones_like": tensor, "dtype": "bool"}

    def series_ids(self, *, device: object) -> object:
        return {"shape": (1, 1), "device": device, "dtype": "long"}

    def no_grad(self) -> object:
        return nullcontext()


class _Column:
    def __init__(self, values: object) -> None:
        self._values = values

    def tolist(self) -> object:
        return self._values


class _Predictions:
    def __init__(self, values: object) -> None:
        self._target = _Column(values)

    def __getitem__(self, name: str) -> _Column:
        assert name == "target"
        return self._target


class _TabPFNLicenseError(Exception):
    pass


class _TabPFNGatedRepoError(Exception):
    pass


class _TabPFNPipeline:
    def __init__(
        self, output: object = None, predict_error: Exception | None = None
    ) -> None:
        self.output = [10.0, 11.0] if output is None else output
        self.predict_error = predict_error
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.no_browser_values: list[str | None] = []

    def predict_df(self, context: object, **kwargs: object) -> _Predictions:
        self.calls.append((context, dict(kwargs)))
        self.no_browser_values.append(os.environ.get("TABPFN_NO_BROWSER"))
        if self.predict_error is not None:
            raise self.predict_error
        return _Predictions(self.output)


class _TabPFNPipelineClass:
    def __init__(
        self, pipeline: _TabPFNPipeline, load_error: Exception | None = None
    ) -> None:
        self.pipeline = pipeline
        self.load_error = load_error
        self.calls: list[dict[str, object]] = []
        self.no_browser_values: list[str | None] = []

    def __call__(self, **kwargs: object) -> _TabPFNPipeline:
        self.calls.append(dict(kwargs))
        self.no_browser_values.append(os.environ.get("TABPFN_NO_BROWSER"))
        if self.load_error is not None:
            raise self.load_error
        return self.pipeline


class _TabPFNBackend:
    local_mode = "LOCAL-enum"
    license_error_classes = (_TabPFNLicenseError, _TabPFNGatedRepoError)
    device = "cpu"

    def __init__(
        self,
        output: object = None,
        *,
        load_error: Exception | None = None,
        predict_error: Exception | None = None,
    ) -> None:
        self.pipeline = _TabPFNPipeline(output, predict_error)
        self.pipeline_class = _TabPFNPipelineClass(self.pipeline, load_error)
        self.date_range_calls: list[dict[str, object]] = []
        self.future_date_range_calls: list[dict[str, object]] = []
        self.frame_calls: list[dict[str, object]] = []
        self.resolve_calls: list[str] = []
        self.download_calls: list[tuple[str, str, str]] = []

    def resolve_checkpoint(self, filename: str) -> str:
        self.resolve_calls.append(filename)
        return f"/official-cache/{filename}"

    def download_checkpoint(
        self, repo_id: str, filename: str, *, revision: str
    ) -> str:
        self.download_calls.append((repo_id, filename, revision))
        return f"/hub/{revision}/{filename}"

    def date_range(self, *, start: str, periods: int, frequency: str) -> object:
        call = {"start": start, "periods": periods, "frequency": frequency}
        self.date_range_calls.append(call)
        return ("timestamps", start, periods, frequency)

    def dataframe(self, columns: dict[str, object]) -> object:
        self.frame_calls.append(columns)
        return {"official-frame": columns}

    def future_date_range(
        self, context_timestamps: object, *, periods: int, frequency: str
    ) -> object:
        call = {
            "context_timestamps": context_timestamps,
            "periods": periods,
            "frequency": frequency,
        }
        self.future_date_range_calls.append(call)
        return ("future-timestamps", context_timestamps, periods, frequency)


def test_tirex_uses_official_point_output_and_exact_checkpoint() -> None:
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    backend = _TiRexBackend()
    adapter = DedicatedAdapter(loader=lambda method_id: backend)

    assert adapter.forecast(_request(TIREX_ID)) == (10.0, 11.0)
    assert backend.load_calls == [("NX-AI/TiRex", {})]
    assert backend.tensor_calls == [((1.0, 2.0, 3.0), "batch_time")]
    assert backend.model.calls == [
        {
            "context": {"values": (1.0, 2.0, 3.0), "layout": "batch_time"},
            "prediction_length": 2,
        }
    ]


def test_toto_uses_rank_three_inputs_and_returns_p50_quantile() -> None:
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    backend = _TotoBackend()
    adapter = DedicatedAdapter(loader=lambda method_id: backend)

    assert adapter.forecast(_request(TOTO_ID)) == (40.0, 41.0)
    assert backend.model_class.calls == [("Datadog/Toto-2.0-22m", {})]
    assert backend.model.to_calls == ["fake-device"]
    assert backend.model.eval_calls == 1
    assert backend.tensor_calls == [
        ((1.0, 2.0, 3.0), "batch_variate_time", "fake-device")
    ]
    target = {
        "values": (1.0, 2.0, 3.0),
        "layout": "batch_variate_time",
        "device": "fake-device",
    }
    assert backend.model.forecast_calls == [
        (
            {
                "target": target,
                "target_mask": {"ones_like": target, "dtype": "bool"},
                "series_ids": {
                    "shape": (1, 1),
                    "device": "fake-device",
                    "dtype": "long",
                },
            },
            {"horizon": 2, "decode_block_size": None, "has_missing_values": False},
        )
    ]


def test_tabpfn_cached_local_weights_do_not_require_environment_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    backend = _TabPFNBackend()

    assert DedicatedAdapter(loader=lambda method_id: backend).forecast(
        _request(TABPFN_ID)
    ) == (10.0, 11.0)


@pytest.mark.parametrize(
    ("frequency", "normalized"),
    [
        ("15 minutes", "15min"),
        ("15min", "15min"),
        ("30 seconds", "30s"),
        ("5ms", "5ms"),
        ("5MS", "5MS"),
        ("H", "h"),
        ("2H", "2h"),
        ("hourly", "h"),
        ("1 day", "D"),
        ("daily", "D"),
        ("W", "W"),
        ("W-MON", "W-MON"),
        ("weekly", "W"),
        ("1 month", "ME"),
        ("monthly", "ME"),
        ("Q", "QE"),
        ("quarterly", "QE"),
        ("1 year", "YE"),
        ("yearly", "YE"),
    ],
)
def test_tabpfn_forces_local_mode_and_builds_timestamped_context(
    monkeypatch: pytest.MonkeyPatch, frequency: str, normalized: str
) -> None:
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    monkeypatch.setenv("HF_TOKEN", "secret-never-forwarded")
    backend = _TabPFNBackend()
    adapter = DedicatedAdapter(loader=lambda method_id: backend)

    assert adapter.forecast(_request(TABPFN_ID, frequency=frequency)) == (10.0, 11.0)
    assert backend.resolve_calls == [
        "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"
    ]
    assert backend.pipeline_class.calls == [
        {
            "tabpfn_mode": "LOCAL-enum",
            "tabpfn_output_selection": "median",
            "tabpfn_model_config": {
                "model_path": (
                    "/official-cache/"
                    "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"
                ),
                "device": "cpu",
            },
        }
    ]
    assert backend.date_range_calls == [
        {"start": "2000-01-01", "periods": 3, "frequency": normalized}
    ]
    context_timestamps = ("timestamps", "2000-01-01", 3, normalized)
    assert backend.future_date_range_calls == [
        {
            "context_timestamps": context_timestamps,
            "periods": 2,
            "frequency": normalized,
        }
    ]
    expected_columns = {
        "item_id": ("series", "series", "series"),
        "timestamp": context_timestamps,
        "target": (1.0, 2.0, 3.0),
    }
    expected_future_columns = {
        "item_id": ("series", "series"),
        "timestamp": ("future-timestamps", context_timestamps, 2, normalized),
    }
    assert backend.frame_calls == [expected_columns, expected_future_columns]
    assert backend.pipeline.calls == [
        (
            {"official-frame": expected_columns},
            {
                "future_df": {"official-frame": expected_future_columns},
                "quantiles": [0.5],
            },
        )
    ]


def test_tabpfn_supplies_explicit_future_timestamps_for_single_point_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    monkeypatch.setenv("HF_TOKEN", "present")
    backend = _TabPFNBackend(output=[7.0, 8.0])

    assert DedicatedAdapter(loader=lambda method_id: backend).forecast(
        _request(TABPFN_ID, history=(3.0,), frequency="D")
    ) == (7.0, 8.0)
    assert backend.future_date_range_calls == [
        {
            "context_timestamps": ("timestamps", "2000-01-01", 1, "D"),
            "periods": 2,
            "frequency": "D",
        }
    ]
    assert "prediction_length" not in backend.pipeline.calls[0][1]


def test_tabpfn_rejects_unknown_frequency_before_optional_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    monkeypatch.setenv("HF_TOKEN", "present")
    loader_calls = 0

    def load(method_id: str) -> _TabPFNBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _TabPFNBackend()

    with pytest.raises(RequestUnavailableError, match="frequency"):
        DedicatedAdapter(loader=load).forecast(
            _request(TABPFN_ID, frequency="whenever")
        )
    assert loader_calls == 0


@pytest.mark.parametrize("frequency", ["W-XYZ", "QE-XYZ"])
def test_tabpfn_rejects_invalid_frequency_anchors_before_optional_imports(
    monkeypatch: pytest.MonkeyPatch, frequency: str
) -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    monkeypatch.setenv("HF_TOKEN", "present")
    loader_calls = 0

    def load(method_id: str) -> _TabPFNBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _TabPFNBackend()

    with pytest.raises(RequestUnavailableError, match="frequency"):
        DedicatedAdapter(loader=load).forecast(
            _request(TABPFN_ID, frequency=frequency)
        )
    assert loader_calls == 0


def test_tabpfn_never_selects_client_or_forwards_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("TABPFN_TOKEN", "do-not-leak")
    backend = _TabPFNBackend()
    DedicatedAdapter(loader=lambda method_id: backend).forecast(_request(TABPFN_ID))

    constructor = backend.pipeline_class.calls[0]
    assert constructor["tabpfn_mode"] == backend.local_mode
    assert "token" not in repr(constructor).lower()
    assert "client" not in repr(constructor).lower()
    assert "do-not-leak" not in repr(constructor)
    assert backend.pipeline_class.no_browser_values == ["1"]
    assert backend.pipeline.no_browser_values == ["1"]


@pytest.mark.parametrize("stage", ["load", "predict"])
def test_tabpfn_priorlabs_license_failures_are_typed_and_sanitized(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from numerical_agent.tsfm.workers.common import LicenseUnavailableError
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    secret = "invalid-priorlabs-secret"
    monkeypatch.setenv("HF_TOKEN", "irrelevant-huggingface-token")
    monkeypatch.setenv("TABPFN_TOKEN", secret)
    error = _TabPFNLicenseError(f"invalid token {secret}")
    backend = _TabPFNBackend(
        load_error=error if stage == "load" else None,
        predict_error=error if stage == "predict" else None,
    )

    with pytest.raises(LicenseUnavailableError) as captured:
        DedicatedAdapter(loader=lambda method_id: backend).forecast(
            _request(TABPFN_ID)
        )
    assert "TABPFN_TOKEN" in str(captured.value)
    assert secret not in str(captured.value)


def test_tabpfn_hf_token_does_not_replace_missing_priorlabs_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.common import LicenseUnavailableError
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    monkeypatch.setenv("HF_TOKEN", "irrelevant-huggingface-token")
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    backend = _TabPFNBackend(
        load_error=_TabPFNLicenseError("PriorLabs credential missing")
    )

    with pytest.raises(LicenseUnavailableError, match="TABPFN_TOKEN"):
        DedicatedAdapter(loader=lambda method_id: backend).forecast(
            _request(TABPFN_ID)
        )


def test_tabpfn_nested_gated_repo_failure_is_typed_license_unavailability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.common import LicenseUnavailableError
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    wrapped = RuntimeError("download failed")
    wrapped.__cause__ = _TabPFNGatedRepoError("terms not accepted")
    backend = _TabPFNBackend(load_error=wrapped)

    with pytest.raises(LicenseUnavailableError, match="TABPFN_TOKEN"):
        DedicatedAdapter(loader=lambda method_id: backend).forecast(
            _request(TABPFN_ID)
        )


def test_tabpfn_unchained_v3_gated_download_is_typed_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.common import LicenseUnavailableError
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    upstream_error = RuntimeError(
        "Failed to download TabPFN ModelVersion.V3 model "
        "'tabpfn-v3-regressor-v3_20260506_timeseries.ckpt'.\n\n"
        "Details and instructions:\n"
        "HuggingFace authentication error downloading from "
        "'Prior-Labs/tabpfn_3'.\n"
        "This model is gated and requires you to accept its terms.\n\n"
        "Please follow these steps:\n"
        "1. Visit https://huggingface.co/Prior-Labs/tabpfn_3 in your browser "
        "and accept the terms of use.\n"
        "2. Log in to your Hugging Face account via the command line by running:\n"
        "   hf auth login\n"
        "   (Alternatively, you can set the HF_TOKEN environment variable with "
        "a read token.)\n\n"
        "For detailed instructions, see "
        "https://docs.priorlabs.ai/how-to-access-gated-models\n\n"
        "For commercial usage, we provide alternative download options for "
        "TabPFN ModelVersion.V3; please reach out to us at sales@priorlabs.ai."
    )
    assert upstream_error.__cause__ is None
    assert upstream_error.__context__ is None
    backend = _TabPFNBackend(predict_error=upstream_error)
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)

    with pytest.raises(LicenseUnavailableError) as captured:
        DedicatedAdapter(loader=lambda method_id: backend).forecast(
            _request(TABPFN_ID)
        )
    assert captured.value.reason_code == "license_not_acknowledged"
    assert str(upstream_error) not in str(captured.value)
    assert "sales@priorlabs.ai" not in str(captured.value)


@pytest.mark.parametrize("stage", ["load", "predict"])
def test_tabpfn_arbitrary_runtime_error_is_not_reclassified_as_license(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from numerical_agent.tsfm.workers.common import CheckpointUnavailableError
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    error = RuntimeError(
        "Failed to download TabPFN v3 model: gated license was mentioned"
    )
    backend = _TabPFNBackend(
        load_error=error if stage == "load" else None,
        predict_error=error if stage == "predict" else None,
    )
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)

    with pytest.raises(RuntimeError) as captured:
        DedicatedAdapter(loader=lambda method_id: backend).forecast(
            _request(TABPFN_ID)
        )
    expected_type = CheckpointUnavailableError if stage == "load" else RuntimeError
    assert type(captured.value) is expected_type


def test_reuses_family_backends_and_checkpoint_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    monkeypatch.setenv("HF_TOKEN", "present")
    backends: dict[str, object] = {
        TIREX_ID: _TiRexBackend(),
        TOTO_ID: _TotoBackend(),
        TABPFN_ID: _TabPFNBackend(),
    }
    loader_calls: list[str] = []

    def load(method_id: str) -> object:
        loader_calls.append(method_id)
        return backends[method_id]

    adapter = DedicatedAdapter(loader=load)
    for method_id in (TIREX_ID, TOTO_ID, TABPFN_ID):
        adapter.forecast(_request(method_id))
        adapter.forecast(_request(method_id))

    assert loader_calls == [TIREX_ID, TOTO_ID, TABPFN_ID]
    assert len(backends[TIREX_ID].load_calls) == 1  # type: ignore[attr-defined]
    assert len(backends[TOTO_ID].model_class.calls) == 1  # type: ignore[attr-defined]
    assert len(backends[TABPFN_ID].pipeline_class.calls) == 1  # type: ignore[attr-defined]


@pytest.mark.parametrize("method_id", [TIREX_ID, TOTO_ID, TABPFN_ID])
def test_smoke_revision_reaches_each_dedicated_checkpoint_loader(
    method_id: str,
) -> None:
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    backend: object
    if method_id == TIREX_ID:
        backend = _TiRexBackend()
    elif method_id == TOTO_ID:
        backend = _TotoBackend()
    else:
        backend = _TabPFNBackend()
    adapter = DedicatedAdapter(loader=lambda selected: backend)
    request = _attested_request(method_id, REVISION_A)

    adapter.forecast(request)

    if method_id == TIREX_ID:
        assert backend.load_calls == [  # type: ignore[attr-defined]
            ("NX-AI/TiRex", {"hf_kwargs": {"revision": REVISION_A}})
        ]
    elif method_id == TOTO_ID:
        assert backend.model_class.calls == [  # type: ignore[attr-defined]
            ("Datadog/Toto-2.0-22m", {"revision": REVISION_A})
        ]
    else:
        assert backend.download_calls == [  # type: ignore[attr-defined]
            (
                "Prior-Labs/tabpfn_3",
                "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt",
                REVISION_A,
            )
        ]
        assert backend.pipeline_class.calls[0]["tabpfn_model_config"][  # type: ignore[attr-defined,index]
            "model_path"
        ] == (
            "/hub/"
            + REVISION_A
            + "/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"
        )
    assert adapter.loaded_checkpoint_revision(request) == REVISION_A


@pytest.mark.parametrize("method_id", [TIREX_ID, TOTO_ID, TABPFN_ID])
def test_dedicated_model_cache_is_scoped_to_immutable_revision(
    method_id: str,
) -> None:
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    backend: object
    if method_id == TIREX_ID:
        backend = _TiRexBackend()
    elif method_id == TOTO_ID:
        backend = _TotoBackend()
    else:
        backend = _TabPFNBackend()
    adapter = DedicatedAdapter(loader=lambda selected: backend)

    adapter.forecast(_attested_request(method_id, REVISION_A))
    adapter.forecast(_attested_request(method_id, REVISION_B))

    if method_id == TIREX_ID:
        calls = backend.load_calls  # type: ignore[attr-defined]
    elif method_id == TOTO_ID:
        calls = backend.model_class.calls  # type: ignore[attr-defined]
    else:
        calls = backend.pipeline_class.calls  # type: ignore[attr-defined]
    assert len(calls) == 2


def test_tabpfn_attested_request_cannot_reuse_generic_cached_file() -> None:
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    backend = _TabPFNBackend()
    adapter = DedicatedAdapter(loader=lambda selected: backend)

    adapter.forecast(_request(TABPFN_ID))
    adapter.forecast(_attested_request(TABPFN_ID, REVISION_A))

    assert backend.resolve_calls == [
        "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"
    ]
    assert backend.download_calls == [
        (
            "Prior-Labs/tabpfn_3",
            "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt",
            REVISION_A,
        )
    ]
    assert len(backend.pipeline_class.calls) == 2


def test_rejects_non_manifest_binding_before_loading() -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    loader_calls = 0

    def load(method_id: str) -> object:
        nonlocal loader_calls
        loader_calls += 1
        return _TiRexBackend()

    with pytest.raises(RequestUnavailableError, match="reviewed"):
        DedicatedAdapter(loader=load).forecast(
            _request(TIREX_ID, runtime_options={"backend": "unreviewed"})
        )
    assert loader_calls == 0


def test_dependency_and_checkpoint_failures_remain_distinct() -> None:
    from numerical_agent.tsfm.workers.common import (
        CheckpointUnavailableError,
        DependencyUnavailableError,
    )
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    def missing_dependency(method_id: str) -> object:
        raise ImportError("tirex missing")

    with pytest.raises(DependencyUnavailableError, match="tirex missing"):
        DedicatedAdapter(loader=missing_dependency).forecast(_request(TIREX_ID))

    backend = _TiRexBackend()

    def broken_checkpoint(checkpoint: str, **kwargs: object) -> object:
        raise OSError("checkpoint denied")

    backend.load_model = broken_checkpoint  # type: ignore[method-assign]
    with pytest.raises(CheckpointUnavailableError, match="checkpoint denied"):
        DedicatedAdapter(loader=lambda method_id: backend).forecast(_request(TIREX_ID))


@pytest.mark.parametrize(
    ("method_id", "backend"),
    [
        (TIREX_ID, _TiRexBackend(mean=[[10.0, math.nan]])),
        (TOTO_ID, _TotoBackend(output=[[[[1.0, 2.0]]]] * 8)),
    ],
)
def test_malformed_outputs_are_runtime_errors(method_id: str, backend: object) -> None:
    from numerical_agent.tsfm.workers.common import ModelOutputError
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    with pytest.raises(ModelOutputError):
        DedicatedAdapter(loader=lambda selected: backend).forecast(_request(method_id))


def test_worker_main_maps_license_and_model_output_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.workers.dedicated import DedicatedAdapter

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    output = io.StringIO()
    monkeypatch.setattr(
        worker_main,
        "_load_adapter",
        lambda name: DedicatedAdapter(
            loader=lambda method_id: _TabPFNBackend(
                load_error=_TabPFNLicenseError("missing credential")
            )
        ),
    )
    worker_main.serve(
        "dedicated", io.StringIO(_request(TABPFN_ID).to_json() + "\n"), output
    )
    response = json.loads(output.getvalue())
    assert response["status"] == "unavailable"
    assert response["reason_code"] == "license_not_acknowledged"

    output = io.StringIO()
    monkeypatch.setattr(
        worker_main,
        "_load_adapter",
        lambda name: DedicatedAdapter(
            loader=lambda method_id: _TiRexBackend(mean=[[1.0]])
        ),
    )
    worker_main.serve(
        "dedicated", io.StringIO(_request(TIREX_ID).to_json() + "\n"), output
    )
    response = json.loads(output.getvalue())
    assert response["status"] == "runtime_error"
    assert response["reason_code"] == "adapter_runtime_error"


def test_module_import_is_lazy_for_all_optional_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional_roots = {
        "torch",
        "tirex",
        "toto2",
        "pandas",
        "tabpfn",
        "tabpfn_time_series",
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

    sys.modules.pop("numerical_agent.tsfm.workers.dedicated", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.import_module("numerical_agent.tsfm.workers.dedicated")
    assert imported == []


def test_worker_main_has_reviewed_dedicated_target() -> None:
    from numerical_agent.tsfm import worker_main

    assert worker_main._ADAPTER_TARGETS["dedicated"] == (
        "numerical_agent.tsfm.workers.dedicated",
        "DedicatedAdapter",
    )
