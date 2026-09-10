from __future__ import annotations

import inspect
import math

import pytest

from evolving_loop.v2.numerical_qd.descriptors import (
    DescriptorPolicyV2,
    describe_history,
)


def policy_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "trend_low_threshold": 0.15,
        "trend_high_threshold": 0.75,
        "intermittency_high_threshold": 0.30,
        "regime_shift_threshold": 0.50,
        "horizon_short_threshold": 0.10,
        "horizon_medium_threshold": 0.30,
        "seasonality_threshold": 0.30,
        "short_seasonal_lag_max": 24,
        "variance_floor": 1e-12,
        "seasonal_lags": {
            "H": [24, 168],
            "D": [7, 30],
            "W": [52],
            "M": [12],
            "Q": [4],
        },
    }


def policy() -> DescriptorPolicyV2:
    return DescriptorPolicyV2.from_payload(policy_payload())


HISTORY = [float((index % 7) + index / 20) for index in range(40)]


def test_descriptor_uses_history_not_future_or_labels():
    first = describe_history(HISTORY, 12, "D", "statistical", policy())
    second = describe_history(tuple(HISTORY), 12, "D", "statistical", policy())
    assert first == second
    assert set(first.to_payload()) == {
        "trend", "seasonality", "intermittency", "regime", "horizon", "family"
    }
    assert not any("score" in key or "history" in key for key in first.to_payload())


def test_future_and_label_inputs_are_not_part_of_descriptor_boundary():
    assert tuple(inspect.signature(describe_history).parameters) == (
        "history", "horizon", "frequency", "family", "policy"
    )
    with pytest.raises(TypeError):
        describe_history(HISTORY, 4, "D", "program", policy(), future=[999])
    with pytest.raises(TypeError):
        describe_history(HISTORY, 4, "D", "program", policy(), labels={"winner": True})


def test_descriptor_threshold_edges_are_closed_and_deterministic():
    constant = [5.0] * 40
    intermittent = [0.0] * 3 + [1.0] * 7
    assert describe_history(constant, 4, "D", "program", policy()).trend == "low"
    assert describe_history(intermittent, 4, "D", "program", policy()).intermittency == "high"
    assert describe_history([1.0] * 40, 4, "D", "program", policy()).horizon == "short"
    assert describe_history([1.0] * 40, 12, "D", "program", policy()).horizon == "medium"


@pytest.mark.parametrize("history", [[], [1.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
def test_empty_or_too_short_history_is_rejected(history):
    with pytest.raises(ValueError, match="history"):
        describe_history(history, 1, "D", "statistical", policy())


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1", None])
def test_nonfinite_or_non_numeric_history_is_rejected(value):
    with pytest.raises((TypeError, ValueError), match="history"):
        describe_history([1.0, 2.0, 3.0, value], 1, "D", "statistical", policy())


@pytest.mark.parametrize("horizon", [0, -1, True, 1.0, "1"])
def test_invalid_horizon_is_rejected(horizon):
    with pytest.raises(ValueError, match="horizon"):
        describe_history(HISTORY, horizon, "D", "statistical", policy())


@pytest.mark.parametrize("frequency", ["", " ", 1, True, None])
def test_invalid_frequency_is_rejected(frequency):
    with pytest.raises(ValueError, match="frequency"):
        describe_history(HISTORY, 4, frequency, "statistical", policy())


@pytest.mark.parametrize("family", ["", "neural", 1, None])
def test_invalid_family_is_rejected(family):
    with pytest.raises(ValueError, match="family"):
        describe_history(HISTORY, 4, "D", family, policy())


def test_trend_regime_horizon_and_intermittency_cover_all_bins():
    low = [0.0, 1.0] * 20
    medium = [0.0] * 10 + [-2.0, 2.0] * 10 + [1.0] * 10
    high = [float(value) for value in range(40)]
    assert describe_history(low, 4, "unknown", "tsfm", policy()).trend == "low"
    assert describe_history(medium, 5, "unknown", "combined", policy()).trend == "medium"
    assert describe_history(high, 20, "unknown", "program", policy()).trend == "high"
    assert describe_history(low, 4, "unknown", "tsfm", policy()).regime == "stable"
    assert describe_history(high, 20, "unknown", "program", policy()).regime == "shift"
    assert describe_history([0.0, 1.0, 1.0, 1.0, 1.0] * 8, 4, "unknown", "program", policy()).intermittency == "low"
    assert describe_history(high, 20, "unknown", "program", policy()).horizon == "long"


def test_configured_seasonal_lags_cover_none_short_and_long():
    short = [0.0, 1.0, 0.0, -1.0] * 4
    long_period = [float(value) for value in range(30)]
    long = long_period * 2
    assert describe_history([5.0] * 60, 4, "D", "program", policy()).seasonality == "none"
    assert describe_history(short, 4, "Q", "program", policy()).seasonality == "short"
    assert describe_history(long, 4, "D", "program", policy()).seasonality == "long"
    assert describe_history(long, 4, "X", "program", policy()).seasonality == "none"


def test_seasonal_lag_requires_two_complete_repeats():
    almost_two_periods = [float(value) for value in range(59)]
    assert describe_history(almost_two_periods, 4, "D", "program", policy()).seasonality != "long"


def test_descriptor_policy_is_exact_finite_canonical_and_deeply_immutable():
    payload = policy_payload()
    descriptor_policy = DescriptorPolicyV2.from_payload(payload)
    assert descriptor_policy.to_payload() == payload
    assert DescriptorPolicyV2.from_payload(descriptor_policy.to_payload()) == descriptor_policy
    assert descriptor_policy.canonical_bytes().endswith(b"\n")
    with pytest.raises(TypeError):
        descriptor_policy.seasonal_lags["D"] = (1,)
    payload["seasonal_lags"]["D"].append(365)
    assert descriptor_policy.seasonal_lags["D"] == (7, 30)


@pytest.mark.parametrize("field", list(policy_payload()))
def test_descriptor_policy_rejects_missing_fields(field):
    payload = policy_payload()
    del payload[field]
    with pytest.raises(ValueError, match="exact schema"):
        DescriptorPolicyV2.from_payload(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("trend_low_threshold", math.nan),
        ("trend_low_threshold", 0.75),
        ("trend_high_threshold", 0.15),
        ("intermittency_high_threshold", 0.0),
        ("regime_shift_threshold", math.inf),
        ("horizon_short_threshold", 0.30),
        ("horizon_medium_threshold", 0.10),
        ("seasonality_threshold", 1.01),
        ("short_seasonal_lag_max", 0),
        ("variance_floor", 0.0),
        ("seasonal_lags", {"D": [7, 7]}),
        ("seasonal_lags", {"D": [30, 7]}),
        ("seasonal_lags", {"": [7]}),
    ],
)
def test_descriptor_policy_rejects_invalid_thresholds_or_lags(field, value):
    with pytest.raises(ValueError, match=field):
        DescriptorPolicyV2.from_payload(policy_payload() | {field: value})
