from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace

import pytest

from numerical_agent.evolution.champion import (
    ChampionRecipe,
    EvolutionAssumption,
    FittedChampionPolicy,
)
from numerical_agent.evolution.champion_runtime import execute_champion
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.screening import TaskProfile


HISTORY = (1.0, 2.0, 3.0, 4.0)
PROFILE = TaskProfile(
    task_id="history_only",
    frequency="D",
    history_length=4,
    horizon=2,
    zero_fraction=0.0,
    signed=False,
    integer_valued=False,
    trend_direction="flat",
    trend_strength=0.1,
    periodicity_periods=(),
    periodicity_strength=0.8,
    periodicity_confidence=0.7,
    outlier_fraction=0.0,
    noise_relative_scale=0.2,
    likely_stationary=True,
    stationarity_score=0.8,
    recent_regime_start=None,
    recent_regime_confidence=0.0,
    intermittency_adi=1.0,
    intermittency_cv2=0.1,
)


def _diagnostic(name: str, *, eligible: bool = True) -> CandidateDiagnostics:
    return CandidateDiagnostics.synthetic(
        name=name,
        family="combined" if name.startswith("combined") else "statistical",
        median_mase=0.5,
        eligible=eligible,
    )


def _policy(
    *,
    kind: str,
    parents: tuple[str, ...],
    fallback: str,
    candidate: str | None = None,
    split: float = 0.5,
    weights: tuple[float, ...] = (),
    alpha: float = 0.0,
    cap: float = 0.0,
    feature: str = "periodicity_strength",
    direction: str = "above",
    threshold: float = 0.5,
) -> FittedChampionPolicy:
    candidate = candidate or parents[-1]
    assumption = EvolutionAssumption(
        assumption_id="supported_history",
        candidate_name=candidate,
        feature=feature,
        direction=direction,
        horizon_region="full",
        operator=kind,
        rationale="The historical signal supports this frozen policy.",
        failure_condition="The historical signal is below its fitted threshold.",
    )
    recipe = ChampionRecipe(
        name=f"{kind}_champion",
        kind=kind,
        parents=parents,
        fallback_parent=fallback,
        assumptions=(assumption,),
    )
    return FittedChampionPolicy(
        recipe=recipe,
        thresholds=((assumption.assumption_id, threshold),),
        weights=weights,
        overlay_alpha=alpha,
        correction_cap=cap,
        horizon_split=split,
    )


def _diagnostics(*names: str, unsafe: str | None = None):
    return {name: _diagnostic(name, eligible=name != unsafe) for name in names}


def test_independent_combined_can_win_without_toto() -> None:
    name = "combined_timesfm_seasonal"
    policy = _policy(kind="select", parents=(name,), fallback=name)

    result = execute_champion(
        policy,
        forecasts={name: (3.0, 4.0), "toto_2_0": (9.0, 9.0)},
        diagnostics=_diagnostics(name),
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast == (3.0, 4.0)


def test_failed_assumption_returns_exact_fallback_forecast() -> None:
    fallback = (10.0, 11.0)
    policy = _policy(
        kind="bounded_overlay",
        parents=("toto_2_0", "specialist"),
        fallback="toto_2_0",
        candidate="specialist",
        alpha=0.5,
        cap=0.25,
    )

    result = execute_champion(
        policy,
        forecasts={"toto_2_0": fallback, "specialist": (99.0, 99.0)},
        diagnostics=_diagnostics("toto_2_0", "specialist", unsafe="specialist"),
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "assumption_not_satisfied"


def test_horizon_route_uses_each_parent_only_in_its_segment() -> None:
    profile = TaskProfile(**{**PROFILE.__dict__, "horizon": 4})
    policy = _policy(
        kind="horizon_route",
        parents=("early", "late"),
        fallback="early",
        candidate="late",
        split=0.5,
    )

    result = execute_champion(
        policy,
        forecasts={
            "early": (1.0, 2.0, 3.0, 4.0),
            "late": (5.0, 6.0, 7.0, 8.0),
        },
        diagnostics=_diagnostics("early", "late"),
        profile=profile,
        history=HISTORY,
        horizon=4,
    )

    assert result.forecast == (1.0, 2.0, 7.0, 8.0)


def test_route_uses_assumption_candidate_not_parent_position() -> None:
    policy = _policy(
        kind="route",
        parents=("specialist", "safe_parent"),
        fallback="safe_parent",
        candidate="specialist",
    )

    result = execute_champion(
        policy,
        forecasts={"specialist": (8.0, 9.0), "safe_parent": (1.0, 2.0)},
        diagnostics=_diagnostics("specialist", "safe_parent"),
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast == (8.0, 9.0)
    assert result.selected_names == ("specialist",)
    assert result.activated_assumptions == ("supported_history",)
    assert result.fallback_reason is None


def test_below_route_is_strict_at_the_fitted_threshold() -> None:
    fallback = (1.0, 2.0)
    policy = _policy(
        kind="route",
        parents=("safe_parent", "specialist"),
        fallback="safe_parent",
        candidate="specialist",
        direction="below",
        threshold=0.8,
    )

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback, "specialist": (8.0, 9.0)},
        diagnostics=_diagnostics("safe_parent", "specialist"),
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "assumption_not_satisfied"


def test_weighted_operator_uses_frozen_normalized_weights() -> None:
    policy = _policy(
        kind="weighted",
        parents=("first", "second"),
        fallback="first",
        weights=(0.25, 0.75),
    )

    result = execute_champion(
        policy,
        forecasts={"first": (0.0, 8.0), "second": (4.0, 0.0)},
        diagnostics=_diagnostics("first", "second"),
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast == (3.0, 2.0)
    assert result.selected_names == ("first", "second")


def test_median_operator_averages_two_parents_without_overflow() -> None:
    policy = _policy(
        kind="median",
        parents=("first", "second"),
        fallback="first",
    )

    result = execute_champion(
        policy,
        forecasts={"first": (1e308, 2.0), "second": (1e308, 6.0)},
        diagnostics=_diagnostics("first", "second"),
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast == (1e308, 4.0)


def test_bounded_overlay_uses_deterministic_odd_history_median_scale() -> None:
    policy = _policy(
        kind="bounded_overlay",
        parents=("safe_parent", "specialist"),
        fallback="safe_parent",
        candidate="specialist",
        alpha=1.0,
        cap=0.1,
    )

    result = execute_champion(
        policy,
        forecasts={"safe_parent": (100.0, 100.0), "specialist": (200.0, 50.0)},
        diagnostics=_diagnostics("safe_parent", "specialist"),
        profile=replace(PROFILE, history_length=3),
        history=(90.0, 100.0, 110.0),
        horizon=2,
    )

    assert result.forecast == (110.0, 90.0)


def test_bounded_overlay_even_history_scale_is_overflow_safe() -> None:
    policy = _policy(
        kind="bounded_overlay",
        parents=("safe_parent", "specialist"),
        fallback="safe_parent",
        candidate="specialist",
        alpha=1.0,
        cap=0.1,
    )

    result = execute_champion(
        policy,
        forecasts={"safe_parent": (0.0, 0.0), "specialist": (1e308, -1e308)},
        diagnostics=_diagnostics("safe_parent", "specialist"),
        profile=replace(PROFILE, history_length=2),
        history=(1e308, -1e308),
        horizon=2,
    )

    assert result.forecast == pytest.approx((1e307, -1e307))


class _ExplodingSequence(Sequence[float]):
    def __len__(self) -> int:
        raise AssertionError("hostile sequence was inspected")

    def __getitem__(self, index: int) -> float:
        raise AssertionError("hostile sequence was inspected")


class _DirectOnlyMapping(Mapping[str, object]):
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("mapping must not be iterated")

    def __len__(self) -> int:
        raise AssertionError("mapping length must not be read")

    def get(self, key: str, default: object = None) -> object:
        raise AssertionError("mapping.get must not be trusted")


class _OneReadMapping(Mapping[str, object]):
    def __init__(self, key: str, value: object) -> None:
        self._key = key
        self._value = value
        self._reads = 0

    def __getitem__(self, key: str) -> object:
        if key != self._key or self._reads:
            raise AssertionError("parent forecast was read more than once")
        self._reads += 1
        return self._value

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("mapping must not be iterated")

    def __len__(self) -> int:
        raise AssertionError("mapping length must not be read")


@pytest.mark.parametrize(
    "invalid",
    [
        [9.0, 9.0],
        (9.0,),
        (9.0, 9.0, 9.0),
        ("9", 9.0),
        (True, 9.0),
        (float("nan"), 9.0),
        (float("inf"), 9.0),
        _ExplodingSequence(),
    ],
)
def test_malformed_parent_forecast_falls_back_without_inspecting_hostile_sequences(
    invalid: object,
) -> None:
    fallback = (1.0, 2.0)
    policy = _policy(
        kind="weighted",
        parents=("safe_parent", "specialist"),
        fallback="safe_parent",
        weights=(0.5, 0.5),
    )

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback, "specialist": invalid},
        diagnostics=_diagnostics("safe_parent", "specialist"),
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "invalid_parent_forecast"


def test_missing_nonfallback_parent_returns_exact_fallback() -> None:
    fallback = (1.0, 2.0)
    policy = _policy(
        kind="median",
        parents=("safe_parent", "missing"),
        fallback="safe_parent",
    )

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback},
        diagnostics=_diagnostics("safe_parent", "missing"),
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "missing_parent"


def test_zero_or_boolean_horizon_returns_exact_fallback() -> None:
    fallback: tuple[float, ...] = ()
    policy = _policy(kind="select", parents=("safe_parent",), fallback="safe_parent")

    for horizon in (0, True):
        result = execute_champion(
            policy,
            forecasts={"safe_parent": fallback},
            diagnostics=_diagnostics("safe_parent"),
            profile=PROFILE,
            history=HISTORY,
            horizon=horizon,
        )
        assert result.forecast is fallback
        assert result.fallback_reason == "invalid_horizon"


@pytest.mark.parametrize("history", [(), (1.0, float("nan")), (True, 1.0), _ExplodingSequence()])
def test_invalid_history_returns_exact_fallback(history: object) -> None:
    fallback = (1.0, 2.0)
    policy = _policy(kind="select", parents=("safe_parent",), fallback="safe_parent")

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback},
        diagnostics=_diagnostics("safe_parent"),
        profile=PROFILE,
        history=history,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "invalid_history"


@pytest.mark.parametrize(
    "field,value",
    [
        ("eligible", 1),
        ("explosion", 0),
        ("median_mase", math.nan),
        ("median_mase", -0.1),
    ],
)
def test_invalid_diagnostic_fields_return_exact_fallback(field: str, value: object) -> None:
    fallback = (1.0, 2.0)
    policy = _policy(kind="select", parents=("safe_parent",), fallback="safe_parent")
    diagnostic = _diagnostic("safe_parent")
    object.__setattr__(diagnostic, field, value)

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback},
        diagnostics={"safe_parent": diagnostic},
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "invalid_diagnostics"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_invalid_profile_measurement_returns_exact_fallback(value: object) -> None:
    fallback = (1.0, 2.0)
    policy = _policy(kind="select", parents=("safe_parent",), fallback="safe_parent")
    profile = replace(PROFILE)
    object.__setattr__(profile, "periodicity_strength", value)

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback},
        diagnostics=_diagnostics("safe_parent"),
        profile=profile,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "invalid_profile"


def test_unknown_assumption_feature_fails_closed() -> None:
    fallback = (1.0, 2.0)
    policy = _policy(
        kind="select",
        parents=("safe_parent",),
        fallback="safe_parent",
        feature="unknown_feature",
    )

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback},
        diagnostics=_diagnostics("safe_parent"),
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "invalid_profile"


def test_hostile_mappings_are_read_only_by_explicit_parent_key() -> None:
    policy = _policy(
        kind="weighted",
        parents=("first", "second"),
        fallback="first",
        weights=(0.5, 0.5),
    )
    forecasts = _DirectOnlyMapping({"first": (2.0, 4.0), "second": (4.0, 8.0)})
    diagnostics = _DirectOnlyMapping(
        {"first": _diagnostic("first"), "second": _diagnostic("second")}
    )

    result = execute_champion(
        policy,
        forecasts=forecasts,
        diagnostics=diagnostics,
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast == (3.0, 6.0)


def test_fallback_parent_is_snapshotted_from_mapping_exactly_once() -> None:
    fallback = (1.0, 2.0)
    policy = _policy(kind="select", parents=("safe_parent",), fallback="safe_parent")

    result = execute_champion(
        policy,
        forecasts=_OneReadMapping("safe_parent", fallback),
        diagnostics=_diagnostics("safe_parent"),
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason is None


def test_missing_diagnostic_field_returns_exact_fallback() -> None:
    fallback = (1.0, 2.0)
    policy = _policy(kind="select", parents=("safe_parent",), fallback="safe_parent")
    diagnostic = _diagnostic("safe_parent")
    object.__delattr__(diagnostic, "median_mase")

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback},
        diagnostics={"safe_parent": diagnostic},
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "invalid_diagnostics"


def test_missing_profile_field_returns_exact_fallback() -> None:
    fallback = (1.0, 2.0)
    policy = _policy(kind="select", parents=("safe_parent",), fallback="safe_parent")
    profile = replace(PROFILE)
    object.__delattr__(profile, "periodicity_strength")

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback},
        diagnostics=_diagnostics("safe_parent"),
        profile=profile,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "invalid_profile"


@pytest.mark.parametrize(
    "field,value",
    [("zero_fraction", -0.1), ("periodicity_confidence", 1.1), ("intermittency_adi", -1.0)],
)
def test_out_of_bounds_profile_measurement_returns_exact_fallback(
    field: str, value: float
) -> None:
    fallback = (1.0, 2.0)
    policy = _policy(kind="select", parents=("safe_parent",), fallback="safe_parent")
    profile = replace(PROFILE)
    object.__setattr__(profile, field, value)

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback},
        diagnostics=_diagnostics("safe_parent"),
        profile=profile,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "invalid_profile"


def test_out_of_bounds_diagnostic_measurement_returns_exact_fallback() -> None:
    fallback = (1.0, 2.0)
    policy = _policy(kind="select", parents=("safe_parent",), fallback="safe_parent")
    diagnostic = _diagnostic("safe_parent")
    object.__setattr__(diagnostic, "long_horizon_coverage", 1.1)

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback},
        diagnostics={"safe_parent": diagnostic},
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "invalid_diagnostics"


def test_arithmetic_overflow_returns_exact_fallback() -> None:
    fallback = (1e308, 1e308)
    policy = _policy(
        kind="weighted",
        parents=("safe_parent", "specialist"),
        fallback="safe_parent",
        weights=(1.0, 0.0),
    )
    object.__setattr__(policy, "weights", (1e308, 1e308))

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback, "specialist": (1e308, 1e308)},
        diagnostics=_diagnostics("safe_parent", "specialist"),
        profile=PROFILE,
        history=HISTORY,
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "invalid_policy"


def test_overlay_rounding_cannot_realize_correction_above_cap() -> None:
    baseline = math.nextafter(1e16, math.inf)
    fallback = (baseline, baseline)
    policy = _policy(
        kind="bounded_overlay",
        parents=("safe_parent", "specialist"),
        fallback="safe_parent",
        candidate="specialist",
        alpha=1.0,
        cap=1.0,
    )

    result = execute_champion(
        policy,
        forecasts={"safe_parent": fallback, "specialist": (baseline + 4.0,) * 2},
        diagnostics=_diagnostics("safe_parent", "specialist"),
        profile=replace(PROFILE, history_length=1),
        history=(1.0,),
        horizon=2,
    )

    assert result.forecast is fallback
    assert result.fallback_reason == "invalid_arithmetic"
