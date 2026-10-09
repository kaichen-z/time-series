from __future__ import annotations

import pytest

from evolving_loop.benchmark_numeric_adapter import (
    BenchmarkNumericAdapterError,
    BenchmarkNumericWindow,
    build_benchmark_numeric_batch,
)


def _window(task_id: str = "opaque_001") -> BenchmarkNumericWindow:
    return BenchmarkNumericWindow(
        task_id=task_id,
        group_id="held_by_evaluator",
        history=(1.0, 2.0, 3.0),
        truth=(4.0, 5.0),
        frequency="D",
        anchor_forecasts={
            "anchor": (3.5, 4.5),
            "specialist": (4.2, 5.2),
        },
    )


def test_adapter_seals_labels_and_group_identity_outside_inference_view() -> None:
    batch = build_benchmark_numeric_batch((_window(),))
    labeled = batch.labeled_tasks[0]
    inference = batch.inference_tasks[0]

    assert labeled.numeric.future_values == (4.0, 5.0)
    assert labeled.labels_public is True
    assert inference.numeric.future_values == ()
    assert inference.labels_public is False
    assert inference.documents == ()
    assert "held_by_evaluator" not in repr(inference)
    assert batch.group_by_task == {"opaque_001": "held_by_evaluator"}
    assert batch.anchor_forecasts[("opaque_001", "anchor")] == (3.5, 4.5)


def test_adapter_fingerprint_is_order_sensitive_and_deterministic() -> None:
    first = build_benchmark_numeric_batch((_window("a"), _window("b")))
    replay = build_benchmark_numeric_batch((_window("a"), _window("b")))
    reversed_batch = build_benchmark_numeric_batch((_window("b"), _window("a")))

    assert replay.fingerprint == first.fingerprint
    assert reversed_batch.fingerprint != first.fingerprint


def test_adapter_rejects_bad_anchor_horizon_and_duplicate_tasks() -> None:
    with pytest.raises(BenchmarkNumericAdapterError, match="anchor horizon"):
        BenchmarkNumericWindow(
            task_id="bad",
            group_id="g",
            history=(1.0, 2.0),
            truth=(3.0, 4.0),
            frequency="D",
            anchor_forecasts={"anchor": (3.0,)},
        )
    with pytest.raises(BenchmarkNumericAdapterError, match="identities"):
        build_benchmark_numeric_batch((_window(), _window()))
