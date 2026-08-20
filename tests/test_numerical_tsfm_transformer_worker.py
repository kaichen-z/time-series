from __future__ import annotations

from contextlib import nullcontext
import builtins
import importlib
import io
import json
import math
import sys
from types import ModuleType
from typing import Any

import pytest

from numerical_agent.tsfm import ManifestRegistry
from numerical_agent.tsfm.protocol import WorkerRequest


TIMER_ID = "method_tsfm_0007"
TIME_MOE_ID = "method_tsfm_0008"
SUNDIAL_ID = "method_tsfm_0013"
TIMER_S1_ID = "method_tsfm_0015"
KAIROS_ID = "method_tsfm_0022"
REVISION_A = "a" * 40
REVISION_B = "b" * 40


def _request(
    method_id: str,
    *,
    history: tuple[float, ...] = (1.0, 2.0, 3.0),
    horizon: int = 2,
) -> WorkerRequest:
    manifest = ManifestRegistry.load_default()[method_id]
    return WorkerRequest(
        request_id=f"request-{method_id}",
        provider=manifest.adapter,
        checkpoint=manifest.checkpoint,
        history=history,
        horizon=horizon,
        frequency="H",
        runtime_options=dict(manifest.runtime_options),
    )


def _attested_request(method_id: str, revision: str) -> WorkerRequest:
    history = (1.0,) * 96 if method_id == TIMER_ID else (1.0, 2.0, 3.0)
    request = _request(method_id, history=history)
    object.__setattr__(request, "checkpoint_revision", revision)
    return request


class _FakeModel:
    def __init__(self, output: object, *, device: str = "fake-device") -> None:
        self.output = output
        self.device = device
        self.generate_calls: list[tuple[object, dict[str, object]]] = []
        self.call_calls: list[dict[str, object]] = []
        self.eval_calls = 0

    def eval(self) -> "_FakeModel":
        self.eval_calls += 1
        return self

    def generate(self, inputs: object, **kwargs: object) -> object:
        self.generate_calls.append((inputs, dict(kwargs)))
        return self.output

    def __call__(self, **kwargs: object) -> object:
        self.call_calls.append(dict(kwargs))
        return self.output


class _FakeFromPretrained:
    def __init__(self, models: dict[str, _FakeModel]) -> None:
        self.models = models
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, checkpoint: str, **kwargs: object) -> _FakeModel:
        self.calls.append((checkpoint, dict(kwargs)))
        return self.models[checkpoint]


class _FakeBackend:
    def __init__(self, outputs: dict[str, object] | None = None) -> None:
        configured = {
            "thuml/timer-base-84m": [[10.0, 11.0]],
            "Maple728/TimeMoE-200M": [[-1.0, 0.0, 1.0, 10.0, 20.0]],
            "thuml/sundial-base-128m": [
                [[float(sample), float(sample + 100)] for sample in range(20)]
            ],
            "thuml/Timer-S1": [
                [[float(q * 10 + step) for step in range(2)] for q in range(9)]
            ],
            "mldi-lab/Kairos_50m": {
                "prediction_outputs": [
                    [[float(q * 10 + step) for step in range(2)] for q in range(9)]
                ]
            },
        }
        if outputs:
            configured.update(outputs)
        self.models = {
            checkpoint: _FakeModel(output)
            for checkpoint, output in configured.items()
        }
        self.causal_lm_class = _FakeFromPretrained(self.models)
        self.kairos_class = _FakeFromPretrained(self.models)
        self.tensor_calls: list[tuple[tuple[float, ...], str, object]] = []
        self.normalize_calls: list[object] = []
        self.inverse_scale_calls: list[tuple[object, object, object]] = []
        self.no_grad_entries = 0

    def tensor(
        self,
        values: tuple[float, ...],
        *,
        layout: str,
        device: object = None,
    ) -> object:
        copied = tuple(values)
        self.tensor_calls.append((copied, layout, device))
        return {"values": copied, "layout": layout, "device": device, "dtype": "float32"}

    def normalize_time_moe(self, tensor: object) -> tuple[object, object, object]:
        self.normalize_calls.append(tensor)
        values = tensor["values"]  # type: ignore[index]
        mean = sum(values) / len(values)
        std = math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )
        normalized = {
            "values": tuple((value - mean) / std for value in values),
            "layout": "batch_time",
            "device": tensor["device"],  # type: ignore[index]
            "dtype": "float32",
        }
        return normalized, mean, std

    def inverse_scale(self, tensor: object, mean: object, std: object) -> object:
        self.inverse_scale_calls.append((tensor, mean, std))
        return [[value * float(std) + float(mean) for value in tensor[0]]]  # type: ignore[index]

    def require_kairos_class(self) -> _FakeFromPretrained:
        return self.kairos_class

    def no_grad(self):
        backend = self

        class Guard:
            def __enter__(self) -> None:
                backend.no_grad_entries += 1

            def __exit__(self, *args: object) -> None:
                return None

        return Guard()


def test_timer_uses_official_generation_call_and_returns_exact_forecast() -> None:
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    backend = _FakeBackend()
    adapter = TransformerGenerationAdapter(loader=lambda: backend)
    history = tuple(float(index) for index in range(96))

    assert adapter.forecast(_request(TIMER_ID, history=history)) == (10.0, 11.0)

    assert backend.causal_lm_class.calls == [
        ("thuml/timer-base-84m", {"trust_remote_code": True})
    ]
    assert backend.tensor_calls == [
        (history, "batch_time", None)
    ]
    assert backend.models["thuml/timer-base-84m"].generate_calls == [
        (
            {
                "values": history,
                "layout": "batch_time",
                "device": None,
                "dtype": "float32",
            },
            {"max_new_tokens": 2},
        )
    ]


def test_timer_rejects_overlong_forecast_instead_of_tail_slicing() -> None:
    from numerical_agent.tsfm.workers.common import ModelOutputError
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    backend = _FakeBackend(
        {"thuml/timer-base-84m": [[90.0, 91.0, 10.0, 11.0]]}
    )

    with pytest.raises(ModelOutputError, match="horizon"):
        TransformerGenerationAdapter(loader=lambda: backend).forecast(
            _request(TIMER_ID, history=(1.0,) * 96)
        )


def test_worker_main_reports_overlong_timer_forecast_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    backend = _FakeBackend(
        {"thuml/timer-base-84m": [[90.0, 91.0, 10.0, 11.0]]}
    )
    adapter = TransformerGenerationAdapter(loader=lambda: backend)
    monkeypatch.setattr(worker_main, "_load_adapter", lambda name: adapter)
    request = _request(TIMER_ID, history=(1.0,) * 96)
    output = io.StringIO()

    worker_main.serve(
        "transformer_generation",
        io.StringIO(request.to_json() + "\n"),
        output,
    )

    response = json.loads(output.getvalue())
    assert response["status"] == "runtime_error"
    assert response["reason_code"] == "adapter_runtime_error"
    assert "horizon" in response["message"]


def test_sundial_generates_manifest_samples_and_returns_the_sample_mean() -> None:
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    samples = [
        [[float(sample), float(sample + 100)] for sample in range(20)]
    ]
    backend = _FakeBackend({"thuml/sundial-base-128m": samples})
    adapter = TransformerGenerationAdapter(loader=lambda: backend)

    assert adapter.forecast(_request(SUNDIAL_ID)) == (9.5, 109.5)

    assert backend.causal_lm_class.calls == [
        ("thuml/sundial-base-128m", {"trust_remote_code": True})
    ]
    assert backend.models["thuml/sundial-base-128m"].generate_calls == [
        (
            {
                "values": (1.0, 2.0, 3.0),
                "layout": "batch_time",
                "device": None,
                "dtype": "float32",
            },
            {"max_new_tokens": 2, "num_samples": 20},
        )
    ]


def test_time_moe_normalizes_generates_slices_and_inverse_scales() -> None:
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    backend = _FakeBackend()
    adapter = TransformerGenerationAdapter(loader=lambda: backend)

    assert adapter.forecast(_request(TIME_MOE_ID)) == (12.0, 22.0)

    assert backend.causal_lm_class.calls == [
        (
            "Maple728/TimeMoE-200M",
            {"device_map": "cpu", "trust_remote_code": True},
        )
    ]
    assert backend.tensor_calls == [
        ((1.0, 2.0, 3.0), "batch_time", None)
    ]
    assert backend.normalize_calls == [
        {
            "values": (1.0, 2.0, 3.0),
            "layout": "batch_time",
            "device": None,
            "dtype": "float32",
        }
    ]
    normalized = {
        "values": (-1.0, 0.0, 1.0),
        "layout": "batch_time",
        "device": None,
        "dtype": "float32",
    }
    assert backend.models["Maple728/TimeMoE-200M"].generate_calls == [
        (normalized, {"max_new_tokens": 2})
    ]
    assert backend.inverse_scale_calls == [([[10.0, 20.0]], 2.0, 1.0)]


def test_timer_s1_moves_float32_input_to_model_and_selects_p50() -> None:
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    backend = _FakeBackend()
    adapter = TransformerGenerationAdapter(loader=lambda: backend)

    assert adapter.forecast(_request(TIMER_S1_ID)) == (40.0, 41.0)

    assert backend.causal_lm_class.calls == [
        (
            "thuml/Timer-S1",
            {"device_map": "auto", "trust_remote_code": True},
        )
    ]
    assert backend.tensor_calls == [
        ((1.0, 2.0, 3.0), "batch_time", "fake-device")
    ]
    assert backend.models["thuml/Timer-S1"].generate_calls == [
        (
            {
                "values": (1.0, 2.0, 3.0),
                "layout": "batch_time",
                "device": "fake-device",
                "dtype": "float32",
            },
            {"max_new_tokens": 2, "revin": True},
        )
    ]


def test_kairos_uses_supported_generation_flags_and_median_quantile() -> None:
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    quantile_rows = [
        [90.0, 190.0],
        [10.0, 110.0],
        [70.0, 170.0],
        [20.0, 120.0],
        [80.0, 180.0],
        [30.0, 130.0],
        [60.0, 160.0],
        [40.0, 140.0],
        [50.0, 150.0],
    ]
    backend = _FakeBackend(
        {
            "mldi-lab/Kairos_50m": {
                "prediction_outputs": [quantile_rows]
            }
        }
    )
    adapter = TransformerGenerationAdapter(loader=lambda: backend)

    assert adapter.forecast(_request(KAIROS_ID)) == (50.0, 150.0)

    assert backend.kairos_class.calls == [
        ("mldi-lab/Kairos_50m", {"trust_remote_code": True})
    ]
    assert backend.models["mldi-lab/Kairos_50m"].call_calls == [
        {
            "past_target": {
                "values": (1.0, 2.0, 3.0),
                "layout": "batch_time",
                "device": None,
                "dtype": "float32",
            },
            "prediction_length": 2,
            "generation": True,
            "infer_is_positive": True,
            "force_flip_invariance": True,
        }
    ]


@pytest.mark.parametrize(
    ("method_id", "history_length", "message"),
    [
        (TIMER_ID, 2_881, "2880"),
        (SUNDIAL_ID, 2_881, "2880"),
        (TIMER_S1_ID, 11_521, "11520"),
        (KAIROS_ID, 2_049, "2048"),
    ],
)
def test_rejects_context_above_exact_manifest_limit_before_loading(
    method_id: str,
    history_length: int,
    message: str,
) -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _FakeBackend()

    with pytest.raises(RequestUnavailableError, match=message):
        TransformerGenerationAdapter(loader=load).forecast(
            _request(method_id, history=(1.0,) * history_length)
        )

    assert loader_calls == 0


@pytest.mark.parametrize(
    ("method_id", "history_length"),
    [
        (TIMER_ID, 2_880),
        (SUNDIAL_ID, 2_880),
        (TIMER_S1_ID, 11_520),
        (KAIROS_ID, 2_048),
    ],
)
def test_accepts_history_at_exact_manifest_context_limit(
    method_id: str,
    history_length: int,
) -> None:
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    backend = _FakeBackend()
    result = TransformerGenerationAdapter(loader=lambda: backend).forecast(
        _request(method_id, history=tuple(float(index % 2) for index in range(history_length)))
    )

    assert len(result) == 2


def test_timer_rejects_less_than_one_input_patch_before_loading() -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _FakeBackend()

    adapter = TransformerGenerationAdapter(loader=load)
    with pytest.raises(RequestUnavailableError, match="at least 96"):
        adapter.forecast(_request(TIMER_ID, history=(1.0,) * 95))
    assert loader_calls == 0

    assert len(adapter.forecast(_request(TIMER_ID, history=(1.0,) * 96))) == 2
    assert loader_calls == 1


@pytest.mark.parametrize("history", [(1.0,), (3.0, 3.0, 3.0)])
def test_time_moe_rejects_history_with_undefined_official_scale_before_loading(
    history: tuple[float, ...],
) -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _FakeBackend()

    with pytest.raises(RequestUnavailableError, match="sample standard deviation"):
        TransformerGenerationAdapter(loader=load).forecast(
            _request(TIME_MOE_ID, history=history)
        )

    assert loader_calls == 0


def test_time_moe_rejects_context_plus_horizon_above_4096_before_loading() -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _FakeBackend()

    adapter = TransformerGenerationAdapter(loader=load)
    boundary_history = tuple(float(index % 2) for index in range(4_094))
    assert len(
        adapter.forecast(
            _request(TIME_MOE_ID, history=boundary_history, horizon=2)
        )
    ) == 2

    with pytest.raises(RequestUnavailableError, match="4096"):
        adapter.forecast(
            _request(
                TIME_MOE_ID,
                history=tuple(float(index % 2) for index in range(4_095)),
                horizon=2,
            )
        )

    assert loader_calls == 1


def test_reuses_one_backend_and_one_model_per_checkpoint() -> None:
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    backend = _FakeBackend()
    loader_calls = 0

    def load() -> _FakeBackend:
        nonlocal loader_calls
        loader_calls += 1
        return backend

    adapter = TransformerGenerationAdapter(loader=load)
    adapter.forecast(_request(TIMER_ID, history=(1.0,) * 96))
    adapter.forecast(_request(TIMER_ID, history=(4.0,) * 96))
    adapter.forecast(_request(SUNDIAL_ID))
    adapter.forecast(_request(SUNDIAL_ID, history=(4.0, 5.0, 6.0)))

    assert loader_calls == 1
    assert backend.causal_lm_class.calls == [
        ("thuml/timer-base-84m", {"trust_remote_code": True}),
        ("thuml/sundial-base-128m", {"trust_remote_code": True}),
    ]
    assert backend.models["thuml/timer-base-84m"].eval_calls == 1
    assert backend.models["thuml/sundial-base-128m"].eval_calls == 1
    assert backend.no_grad_entries == 4


@pytest.mark.parametrize(
    "method_id",
    [TIMER_ID, TIME_MOE_ID, SUNDIAL_ID, TIMER_S1_ID, KAIROS_ID],
)
def test_smoke_revision_is_passed_to_transformer_checkpoint_loader(
    method_id: str,
) -> None:
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    backend = _FakeBackend()
    adapter = TransformerGenerationAdapter(loader=lambda: backend)
    request = _attested_request(method_id, REVISION_A)

    adapter.forecast(request)

    loader = backend.kairos_class if method_id == KAIROS_ID else backend.causal_lm_class
    assert loader.calls[0][1]["revision"] == REVISION_A
    assert adapter.loaded_checkpoint_revision(request) == REVISION_A


def test_transformer_model_cache_is_scoped_to_immutable_revision() -> None:
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    backend = _FakeBackend()
    adapter = TransformerGenerationAdapter(loader=lambda: backend)

    adapter.forecast(_attested_request(SUNDIAL_ID, REVISION_A))
    adapter.forecast(_attested_request(SUNDIAL_ID, REVISION_B))

    assert [kwargs["revision"] for _checkpoint, kwargs in backend.causal_lm_class.calls] == [
        REVISION_A,
        REVISION_B,
    ]


def test_rejects_checkpoint_provider_and_options_substitution_before_loading() -> None:
    from numerical_agent.tsfm.workers.common import RequestUnavailableError
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    backend = _FakeBackend()
    adapter = TransformerGenerationAdapter(loader=lambda: backend)
    valid = _request(TIMER_ID)
    mutations = [
        {"checkpoint": "attacker/arbitrary-remote-code"},
        {"provider": "granite"},
        {"runtime_options": {**valid.runtime_options, "max_context": 9_999}},
        {"runtime_options": {**valid.runtime_options, "trust_remote_code": False}},
    ]

    for mutation in mutations:
        payload = valid.to_payload()
        payload.update(mutation)
        with pytest.raises(RequestUnavailableError, match="reviewed"):
            adapter.forecast(WorkerRequest.from_payload(payload))

    assert backend.causal_lm_class.calls == []


@pytest.mark.parametrize(
    ("method_id", "output", "message"),
    [
        (TIMER_ID, [10.0, 11.0], "shape"),
        (SUNDIAL_ID, [[[1.0, 2.0]]], "20 samples"),
        (TIME_MOE_ID, [[1.0]], "horizon"),
        (TIMER_S1_ID, [[[1.0, 2.0]]], "9 quantiles"),
        (
            KAIROS_ID,
            {"prediction_outputs": [[[1.0, math.nan] for _ in range(9)]]},
            "finite",
        ),
    ],
)
def test_rejects_malformed_or_nonfinite_official_outputs(
    method_id: str,
    output: object,
    message: str,
) -> None:
    from numerical_agent.tsfm.workers.common import ModelOutputError
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    manifest = ManifestRegistry.load_default()[method_id]
    backend = _FakeBackend({manifest.checkpoint: output})
    history = (1.0,) * 96 if method_id == TIMER_ID else (1.0, 2.0, 3.0)

    with pytest.raises(ModelOutputError, match=message):
        TransformerGenerationAdapter(loader=lambda: backend).forecast(
            _request(method_id, history=history)
        )


def test_maps_dependency_and_checkpoint_load_failures_to_typed_unavailability() -> None:
    from numerical_agent.tsfm.workers.common import (
        CheckpointUnavailableError,
        DependencyUnavailableError,
    )
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    def missing_dependency() -> object:
        raise ImportError("transformers missing")

    with pytest.raises(DependencyUnavailableError, match="dependencies"):
        TransformerGenerationAdapter(loader=missing_dependency).forecast(
            _request(TIMER_ID, history=(1.0,) * 96)
        )

    backend = _FakeBackend()

    def broken_checkpoint(checkpoint: str, **kwargs: object) -> object:
        del checkpoint, kwargs
        raise OSError("checkpoint unavailable")

    backend.causal_lm_class.from_pretrained = broken_checkpoint  # type: ignore[method-assign]
    with pytest.raises(CheckpointUnavailableError, match="checkpoint"):
        TransformerGenerationAdapter(loader=lambda: backend).forecast(
            _request(TIMER_ID, history=(1.0,) * 96)
        )


def test_module_import_does_not_import_optional_model_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "numerical_agent.tsfm.workers.transformer_generation"
    sys.modules.pop(module_name, None)
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> Any:
        if name == "torch" or name == "transformers" or name.startswith("tsfm"):
            raise AssertionError(f"optional package imported eagerly: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    imported = importlib.import_module(module_name)

    assert imported.TransformerGenerationAdapter is not None


def test_default_loader_does_not_require_kairos_in_timer_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.transformer_generation import (
        _load_official_backend,
    )

    fake_torch = ModuleType("torch")
    fake_torch.float32 = object()  # type: ignore[attr-defined]
    fake_torch.tensor = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    fake_torch.no_grad = nullcontext  # type: ignore[attr-defined]
    fake_transformers = ModuleType("transformers")
    causal_class = object()
    fake_transformers.AutoModelForCausalLM = causal_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.delitem(sys.modules, "tsfm", raising=False)
    monkeypatch.delitem(sys.modules, "tsfm.model", raising=False)
    monkeypatch.delitem(sys.modules, "tsfm.model.kairos", raising=False)

    backend = _load_official_backend()

    assert backend.causal_lm_class is causal_class


def test_default_backend_imports_official_kairos_class_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.transformer_generation import (
        _load_official_backend,
    )

    fake_torch = ModuleType("torch")
    fake_torch.float32 = object()  # type: ignore[attr-defined]
    fake_torch.tensor = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    fake_torch.no_grad = nullcontext  # type: ignore[attr-defined]
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object()  # type: ignore[attr-defined]
    fake_tsfm = ModuleType("tsfm")
    fake_tsfm_model = ModuleType("tsfm.model")
    fake_kairos = ModuleType("tsfm.model.kairos")
    kairos_class = object()
    fake_kairos.AutoModel = kairos_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "tsfm", fake_tsfm)
    monkeypatch.setitem(sys.modules, "tsfm.model", fake_tsfm_model)
    monkeypatch.setitem(sys.modules, "tsfm.model.kairos", fake_kairos)

    backend = _load_official_backend()

    assert backend.require_kairos_class() is kairos_class


def test_worker_main_constructs_registered_adapter_without_loading_checkpoint() -> None:
    from numerical_agent.tsfm.worker_main import _load_adapter
    from numerical_agent.tsfm.workers.transformer_generation import (
        TransformerGenerationAdapter,
    )

    adapter = _load_adapter("transformer_generation")

    assert isinstance(adapter, TransformerGenerationAdapter)
