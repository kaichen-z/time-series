from __future__ import annotations

import math

import pytest

from numerical_agent.evolution.primite_ts_skills import (
    AR_METHODS,
    COSTS,
    DENOISE_METHODS,
    DICTIONARY_KINDS,
    DISTANCE_METRICS,
    FEATURE_GROUPS,
    FREQUENCY_PERIODS,
    INTERPOLATE_METHODS,
    LEARN_METHODS,
    OUTLIER_METHODS,
    PURSUIT_METHODS,
    SEASONAL_METHODS,
    SINUSOIDAL_METHODS,
    TREND_METHODS,
    PENALTIES,
    SEARCHES,
    NotApplicable,
    acf,
    decompose,
    denoise,
    deseasonalize,
    detect_changepoints,
    detrend,
    dominant_frequencies,
    fit_ar,
    fit_arma,
    fit_hmm,
    fit_sinusoidal,
    fit_state_space,
    fit_trend_seasonal,
    barycenter,
    distance,
    features,
    learn_dictionary,
    make_dictionary,
    matrix_profile,
    nearest_windows,
    reconstruct,
    sax,
    sparse_code,
    infer_period,
    interpolate_missing,
    last_regime,
    pacf,
    remove_outliers,
    segment_cost,
    spectrum,
)


def sinusoid(period: int, length: int, amplitude: float = 1.0, phase: float = 0.0) -> list[float]:
    return [amplitude * math.sin(2 * math.pi * (i / period) + phase) for i in range(length)]


def ar1(coefficient: float, length: int, seed: int = 0) -> list[float]:
    """A deterministic AR(1) built from a fixed pseudo-random sequence, so tests never flake."""
    state = seed or 1
    values = [0.0]
    for _ in range(length - 1):
        state = (1103515245 * state + 12345) % 2147483648
        noise = (state / 2147483648.0) - 0.5
        values.append(coefficient * values[-1] + noise)
    return values


# --------------------------------------------------------------------------------------
# acf
# --------------------------------------------------------------------------------------


def test_acf_is_one_at_lag_zero() -> None:
    assert acf(sinusoid(12, 120), 5)[0] == pytest.approx(1.0)


def test_acf_of_a_sinusoid_peaks_at_its_period() -> None:
    correlations = acf(sinusoid(24, 240), 60)
    assert correlations[24] == pytest.approx(max(correlations[2:]), abs=1e-9)


def test_acf_of_a_constant_series_reports_no_structure_instead_of_nan() -> None:
    correlations = acf([7.0] * 50, 10)

    assert correlations[0] == 1.0
    assert all(value == 0.0 for value in correlations[1:])


def test_acf_rejects_a_lag_the_series_cannot_support() -> None:
    with pytest.raises(NotApplicable):
        acf([1.0, 2.0, 3.0], 3)


def test_acf_rejects_a_non_finite_series() -> None:
    with pytest.raises(NotApplicable):
        acf([1.0, float("nan"), 3.0, 4.0], 2)


# --------------------------------------------------------------------------------------
# pacf
# --------------------------------------------------------------------------------------


def test_pacf_of_an_ar1_process_cuts_off_after_lag_one() -> None:
    partial = pacf(ar1(0.8, 400), 6)

    assert partial[1] > 0.5
    assert all(abs(value) < 0.25 for value in partial[2:])


def test_pacf_is_one_at_lag_zero() -> None:
    assert pacf(ar1(0.5, 200), 4)[0] == pytest.approx(1.0)


def test_pacf_needs_more_than_twice_the_lag_in_points() -> None:
    with pytest.raises(NotApplicable):
        pacf(list(range(10)), 5)


# --------------------------------------------------------------------------------------
# spectrum
# --------------------------------------------------------------------------------------


def test_dft_spectrum_recovers_the_frequency_of_a_pure_sinusoid() -> None:
    period = 20
    components = spectrum(sinusoid(period, 200), method="dft")
    strongest = max(components, key=lambda component: component[1])

    assert strongest[0] == pytest.approx(1.0 / period, abs=1e-9)


def test_dft_spectrum_recovers_the_amplitude_of_a_pure_sinusoid() -> None:
    components = spectrum(sinusoid(20, 200, amplitude=3.0), method="dft")
    strongest = max(components, key=lambda component: component[1])

    assert strongest[1] == pytest.approx(3.0, rel=1e-6)


def test_dft_spectrum_carries_a_real_phase_but_power_methods_do_not() -> None:
    signal = sinusoid(20, 200, phase=0.7)

    assert any(phase != 0.0 for _, _, phase in spectrum(signal, method="dft"))
    assert all(phase == 0.0 for _, _, phase in spectrum(signal, method="welch"))
    assert all(phase == 0.0 for _, _, phase in spectrum(signal, method="periodogram"))


def test_every_spectrum_method_drops_the_zero_frequency() -> None:
    for method in ("dft", "periodogram", "welch"):
        assert all(freq > 0.0 for freq, _, _ in spectrum(sinusoid(16, 128), method=method))


def test_spectrum_rejects_an_unknown_method() -> None:
    with pytest.raises(NotApplicable):
        spectrum(sinusoid(8, 64), method="wavelet")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# dominant_frequencies
# --------------------------------------------------------------------------------------


def test_dominant_frequencies_returns_k_components_strongest_first() -> None:
    # 280 is a multiple of both periods, so both land exactly on a DFT bin.
    signal = [a + b for a, b in zip(sinusoid(20, 280, 3.0), sinusoid(7, 280, 1.0))]
    components = dominant_frequencies(signal, k=2)

    assert len(components) == 2
    assert components[0][1] >= components[1][1]
    assert components[0][0] == pytest.approx(1.0 / 20, abs=1e-3)
    assert components[1][0] == pytest.approx(1.0 / 7, abs=1e-3)


def test_dominant_frequencies_rejects_a_non_positive_k() -> None:
    with pytest.raises(NotApplicable):
        dominant_frequencies(sinusoid(8, 64), k=0)


# --------------------------------------------------------------------------------------
# infer_period
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("period", [7, 12, 24, 30])
def test_infer_period_recovers_a_planted_period(period: int) -> None:
    assert infer_period(sinusoid(period, period * 12)) == period


def test_infer_period_prefers_the_data_over_a_contradicting_frequency_hint() -> None:
    # The hint for "1 day" is 7, but the series genuinely cycles every 24 samples.
    assert FREQUENCY_PERIODS["1 day"] == 7
    assert infer_period(sinusoid(24, 288), frequency="1 day") == 24


def test_infer_period_falls_back_to_the_frequency_hint_without_periodic_structure() -> None:
    assert infer_period([7.0] * 200, frequency="1 hour") == FREQUENCY_PERIODS["1 hour"]


def test_infer_period_returns_one_when_there_is_no_structure_and_no_usable_hint() -> None:
    assert infer_period([7.0] * 200) == 1
    assert infer_period([7.0] * 200, frequency="not a frequency") == 1


def test_infer_period_honours_a_max_period_ceiling() -> None:
    assert infer_period(sinusoid(24, 288), max_period=10) <= 10


def test_infer_period_rejects_a_series_too_short_to_judge() -> None:
    with pytest.raises(NotApplicable):
        infer_period([1.0, 2.0])


# --------------------------------------------------------------------------------------
# Axis D: segmentation
# --------------------------------------------------------------------------------------

def mean_shift(levels: list[float], run: int = 60) -> list[float]:
    """Piecewise-constant signal with a small deterministic wobble so variance is non-zero."""
    values: list[float] = []
    state = 7
    for level in levels:
        for _ in range(run):
            state = (1103515245 * state + 12345) % 2147483648
            values.append(level + ((state / 2147483648.0) - 0.5) * 0.2)
    return values


def variance_shift(run: int = 120) -> list[float]:
    """Constant mean, and the second half is ten times noisier."""
    values, state = [], 3
    for index in range(2 * run):
        state = (1103515245 * state + 12345) % 2147483648
        noise = (state / 2147483648.0) - 0.5
        values.append(noise * (0.1 if index < run else 1.0))
    return values


def near(found: list[int], expected: int, tolerance: int) -> bool:
    return any(abs(point - expected) <= tolerance for point in found)


@pytest.mark.parametrize("search", SEARCHES)
def test_every_search_recovers_a_planted_mean_shift(search: str) -> None:
    found = detect_changepoints(mean_shift([0.0, 10.0]), search=search, n_breaks=1)

    assert near(found, 60, tolerance=2), f"{search} found {found}"


@pytest.mark.parametrize("search", SEARCHES)
def test_every_search_recovers_two_planted_breaks(search: str) -> None:
    found = detect_changepoints(mean_shift([0.0, 10.0, -5.0]), search=search, n_breaks=2)

    assert len(found) == 2
    assert near(found, 60, tolerance=3) and near(found, 120, tolerance=3), f"{search} found {found}"


@pytest.mark.parametrize("cost", COSTS)
def test_every_cost_recovers_a_planted_mean_shift(cost: str) -> None:
    found = detect_changepoints(mean_shift([0.0, 10.0]), cost=cost, search="optimal", n_breaks=1)

    assert near(found, 60, tolerance=3), f"{cost} found {found}"


def test_the_normal_cost_sees_a_pure_variance_shift_that_l2_misses() -> None:
    signal = variance_shift()

    normal = detect_changepoints(signal, cost="normal", search="optimal", n_breaks=1)
    l2 = detect_changepoints(signal, cost="l2", search="optimal", n_breaks=1)

    assert near(normal, 120, tolerance=10), f"normal found {normal}"
    assert not near(l2, 120, tolerance=10), f"l2 unexpectedly found {l2}"


def test_breaks_are_ascending_and_strictly_inside_the_series() -> None:
    signal = mean_shift([0.0, 5.0, 0.0, 5.0])
    found = detect_changepoints(signal, search="binseg", n_breaks=3)

    assert found == sorted(found)
    assert all(0 < point < len(signal) for point in found)
    assert len(set(found)) == len(found)


@pytest.mark.parametrize("penalty", [p for p in PENALTIES if p != "none"])
def test_every_penalty_finds_the_break_in_an_obvious_two_regime_series(penalty: str) -> None:
    found = detect_changepoints(mean_shift([0.0, 20.0]), search="pelt", penalty=penalty)

    assert near(found, 60, tolerance=3), f"{penalty} found {found}"


def test_a_weaker_penalty_never_finds_fewer_breaks_than_a_stronger_one() -> None:
    signal = mean_shift([0.0, 3.0, 0.0, 3.0], run=40)
    counts = {
        penalty: len(detect_changepoints(signal, search="pelt", penalty=penalty))
        for penalty in ("bic", "aic", "linear")
    }

    assert counts["linear"] >= counts["aic"] >= counts["bic"], counts


def test_pelt_agrees_with_exhaustive_optimal_search() -> None:
    signal = mean_shift([0.0, 8.0, 2.0], run=40)

    assert detect_changepoints(signal, search="pelt", penalty="bic") == detect_changepoints(
        signal, search="optimal", penalty="bic"
    )


def test_a_homogeneous_series_yields_no_breaks() -> None:
    state, noise = 11, []
    for _ in range(200):
        state = (1103515245 * state + 12345) % 2147483648
        noise.append((state / 2147483648.0) - 0.5)

    assert detect_changepoints(noise, search="pelt", penalty="bic") == []


def test_the_penalty_is_scaled_so_detection_is_invariant_to_series_magnitude() -> None:
    signal = mean_shift([0.0, 4.0])
    scaled = [value * 1000.0 for value in signal]

    assert detect_changepoints(signal, search="pelt") == detect_changepoints(scaled, search="pelt")


def test_n_breaks_zero_returns_no_breaks() -> None:
    assert detect_changepoints(mean_shift([0.0, 10.0]), n_breaks=0) == []


def test_min_size_is_honoured() -> None:
    signal = mean_shift([0.0, 5.0, 0.0, 5.0], run=30)
    found = detect_changepoints(signal, search="binseg", n_breaks=3, min_size=20)
    bounds = [0] + found + [len(signal)]

    assert all(end - start >= 20 for start, end in zip(bounds[:-1], bounds[1:]))


def test_penalty_none_requires_an_explicit_n_breaks() -> None:
    with pytest.raises(NotApplicable):
        detect_changepoints(mean_shift([0.0, 10.0]), penalty="none")


def test_unknown_cost_search_and_penalty_are_rejected() -> None:
    signal = mean_shift([0.0, 10.0])
    for kwargs in ({"cost": "wibble"}, {"search": "wibble"}, {"penalty": "wibble"}):
        with pytest.raises(NotApplicable):
            detect_changepoints(signal, **kwargs)  # type: ignore[arg-type]


def test_asking_for_more_breaks_than_the_series_can_hold_is_rejected() -> None:
    with pytest.raises(NotApplicable):
        detect_changepoints([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], n_breaks=5, min_size=2)


# --------------------------------------------------------------------------------------
# segment_cost and last_regime
# --------------------------------------------------------------------------------------


def test_segment_cost_is_zero_on_a_constant_segment_and_positive_otherwise() -> None:
    assert segment_cost([4.0] * 20, 0, 20) == pytest.approx(0.0, abs=1e-9)
    assert segment_cost([0.0, 1.0, 0.0, 1.0] * 5, 0, 20) > 0.0


def test_splitting_at_a_real_break_lowers_the_total_cost() -> None:
    signal = mean_shift([0.0, 10.0])
    whole = segment_cost(signal, 0, len(signal))
    split = segment_cost(signal, 0, 60) + segment_cost(signal, 60, len(signal))

    assert split < whole


def test_segment_cost_rejects_an_invalid_range() -> None:
    with pytest.raises(NotApplicable):
        segment_cost([1.0, 2.0, 3.0, 4.0], 3, 2)


def test_last_regime_returns_the_final_segment() -> None:
    signal = mean_shift([0.0, 10.0])

    assert last_regime(signal, [60]) == pytest.approx(signal[60:])
    assert last_regime(signal, [30, 60]) == pytest.approx(signal[60:])


def test_last_regime_without_breaks_returns_the_whole_series() -> None:
    assert last_regime([1.0, 2.0, 3.0], []) == [1.0, 2.0, 3.0]


def test_last_regime_rejects_a_break_outside_the_series() -> None:
    with pytest.raises(NotApplicable):
        last_regime([1.0, 2.0, 3.0], [9])


# --------------------------------------------------------------------------------------
# Axis B: cleaning
# --------------------------------------------------------------------------------------


def noisy(period: int, length: int, scale: float = 0.4, seed: int = 5) -> tuple[list[float], list[float]]:
    """A clean sinusoid and the same signal with deterministic noise added."""
    clean = sinusoid(period, length)
    state, dirty = seed, []
    for value in clean:
        state = (1103515245 * state + 12345) % 2147483648
        dirty.append(value + ((state / 2147483648.0) - 0.5) * scale)
    return clean, dirty


def mae(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


@pytest.mark.parametrize("method", DENOISE_METHODS)
def test_denoise_preserves_length(method: str) -> None:
    _clean, dirty = noisy(20, 200)

    assert len(denoise(dirty, method)) == len(dirty)


@pytest.mark.parametrize("method", DENOISE_METHODS)
def test_denoise_moves_a_noisy_signal_closer_to_the_truth(method: str) -> None:
    clean, dirty = noisy(20, 200)

    assert mae(denoise(dirty, method), clean) < mae(dirty, clean), method


def test_hampel_removes_a_spike_without_smoothing_everything_else() -> None:
    clean = sinusoid(20, 120)
    spiked = list(clean)
    spiked[60] = 50.0

    hampel = denoise(spiked, "hampel", window=7)
    average = denoise(spiked, "moving_average", window=7)

    # Index 15 is a trough, not a zero crossing, so "changed" is measurable there.
    assert abs(hampel[60] - clean[60]) < 1.0
    assert hampel[15] == pytest.approx(clean[15])       # untouched away from the spike
    assert abs(average[15] - clean[15]) > 1e-3          # smoothed everywhere


def test_denoise_rejects_bad_arguments() -> None:
    signal = sinusoid(20, 100)
    with pytest.raises(NotApplicable):
        denoise(signal, "wibble")  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        denoise(signal, "moving_average", window=500)
    with pytest.raises(NotApplicable):
        denoise(signal, "butterworth", cutoff=0.9)
    with pytest.raises(NotApplicable):
        denoise(signal, "savgol", window=5, polyorder=9)


@pytest.mark.parametrize("method", INTERPOLATE_METHODS)
def test_every_interpolation_recovers_a_linear_ramp_exactly(method: str) -> None:
    truth = [2.0 * i + 1.0 for i in range(40)]
    gapped: list[float | None] = list(truth)
    for index in (7, 8, 21, 33):
        gapped[index] = None

    filled = interpolate_missing(gapped, method)

    assert len(filled) == len(truth)
    assert all(math.isfinite(value) for value in filled)
    assert filled == pytest.approx(truth, abs=1e-6)


def test_interpolate_missing_leaves_a_complete_series_untouched() -> None:
    truth = sinusoid(12, 60)

    assert interpolate_missing(truth) == pytest.approx(truth)


def test_interpolate_missing_rejects_infinities_and_too_few_known_points() -> None:
    with pytest.raises(NotApplicable):
        interpolate_missing([1.0, float("inf"), 3.0])
    with pytest.raises(NotApplicable):
        interpolate_missing([1.0, None, None, None])


@pytest.mark.parametrize("method", OUTLIER_METHODS)
def test_every_outlier_method_finds_a_planted_spike(method: str) -> None:
    signal = sinusoid(20, 120)
    spiked = list(signal)
    spiked[60] = 50.0

    cleaned, indices = remove_outliers(spiked, method)

    assert 60 in indices, f"{method} found {indices}"
    assert len(cleaned) == len(spiked)
    assert abs(cleaned[60]) < 5.0


def test_remove_outliers_leaves_a_clean_series_alone() -> None:
    signal = sinusoid(20, 120)
    cleaned, indices = remove_outliers(signal)

    assert indices == []
    assert cleaned == pytest.approx(signal)


def test_remove_outliers_finds_nothing_in_a_flat_series() -> None:
    assert remove_outliers([5.0] * 40)[1] == []


def test_contiguous_catches_a_run_of_outliers_beside_a_flat_stretch() -> None:
    signal = [1.0, 2.0] * 30
    for index in (40, 41, 42):
        signal[index] = 80.0

    plain = remove_outliers(signal, "hampel", contiguous=False)[1]
    widened = remove_outliers(signal, "hampel", contiguous=True)[1]

    assert set(plain) <= set(widened)
    assert len(widened) >= len(plain)


def test_remove_outliers_rejects_bad_arguments() -> None:
    with pytest.raises(NotApplicable):
        remove_outliers(sinusoid(10, 50), "wibble")  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        remove_outliers(sinusoid(10, 50), threshold=0.0)


# --------------------------------------------------------------------------------------
# Axis C: decomposition
# --------------------------------------------------------------------------------------


def trended_seasonal(length: int = 120, period: int = 12, slope: float = 0.1) -> list[float]:
    return [10.0 + slope * i + 3.0 * math.sin(2 * math.pi * i / period) for i in range(length)]


@pytest.mark.parametrize("method", TREND_METHODS)
def test_trend_plus_detrended_reconstructs_the_input(method: str) -> None:
    signal = trended_seasonal()
    trend, detrended = detrend(signal, method)

    assert len(trend) == len(detrended) == len(signal)
    assert [t + d for t, d in zip(trend, detrended)] == pytest.approx(signal, abs=1e-9)


def test_least_squares_detrending_recovers_a_known_slope() -> None:
    signal = [5.0 + 2.5 * i for i in range(60)]
    trend, detrended = detrend(signal, "least_squares", degree=1)

    assert trend == pytest.approx(signal, abs=1e-6)
    assert all(abs(value) < 1e-6 for value in detrended)


def test_detrend_rejects_bad_arguments() -> None:
    with pytest.raises(NotApplicable):
        detrend(trended_seasonal(), "wibble")  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        detrend(trended_seasonal(), "least_squares", degree=-1)


@pytest.mark.parametrize("method", SEASONAL_METHODS)
def test_the_seasonal_part_has_zero_mean_and_reconstructs(method: str) -> None:
    signal = trended_seasonal()
    seasonal, deseasonalized = deseasonalize(signal, 12, method)

    assert sum(seasonal) / len(seasonal) == pytest.approx(0.0, abs=1e-8)
    assert [s + d for s, d in zip(seasonal, deseasonalized)] == pytest.approx(signal, abs=1e-9)


@pytest.mark.parametrize("method", SEASONAL_METHODS)
def test_deseasonalizing_a_pure_cycle_recovers_its_amplitude(method: str) -> None:
    signal = [3.0 * math.sin(2 * math.pi * i / 12) for i in range(120)]
    seasonal, _ = deseasonalize(signal, 12, method)

    assert max(seasonal) - min(seasonal) == pytest.approx(6.0, rel=0.05), method


def test_deseasonalize_rejects_an_impossible_period() -> None:
    with pytest.raises(NotApplicable):
        deseasonalize(trended_seasonal(), 1)
    with pytest.raises(NotApplicable):
        deseasonalize(sinusoid(12, 20), 12)  # fewer than two full cycles


@pytest.mark.parametrize("model", ["additive", "multiplicative"])
def test_decomposition_recombines_to_the_input(model: str) -> None:
    signal = (
        trended_seasonal() if model == "additive"
        else [(10.0 + 0.1 * i) * (1 + 0.3 * math.sin(2 * math.pi * i / 12)) for i in range(120)]
    )
    parts = decompose(signal, 12, model)  # type: ignore[arg-type]

    assert parts.model == model
    assert len(parts.trend) == len(parts.seasonal) == len(parts.residual) == len(signal)
    assert parts.recombine() == pytest.approx(signal, rel=1e-8)


def test_multiplicative_decomposition_rejects_non_positive_values() -> None:
    with pytest.raises(NotApplicable):
        decompose([0.0] + trended_seasonal(), 12, "multiplicative")


def test_decompose_rejects_an_unknown_model() -> None:
    with pytest.raises(NotApplicable):
        decompose(trended_seasonal(), 12, "exponential")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Axis F: models
# --------------------------------------------------------------------------------------


def trended_cycle(length: int = 160, period: int = 12) -> list[float]:
    return [10.0 + 0.05 * i + 3.0 * math.sin(2 * math.pi * i / period) for i in range(length)]


MODEL_NAMES = (
    "ar_yule_walker", "ar_burg", "ar_levinson", "ar_ols", "arma",
    "state_space_level", "state_space_bsm", "trend_seasonal",
    "sinusoidal_periodogram", "sinusoidal_music", "sinusoidal_prony", "hmm",
)

_MODEL_CACHE: dict[str, object] = {}


def models() -> dict[str, object]:
    """Fit each model once; refitting per parametrisation costs 20s of test time."""
    if not _MODEL_CACHE:
        _MODEL_CACHE.update(_build_models(trended_cycle()))
    return _MODEL_CACHE


def _build_models(signal: list[float]) -> dict[str, object]:
    return {
        "ar_yule_walker": fit_ar(signal),
        "ar_burg": fit_ar(signal, method="burg"),
        "ar_levinson": fit_ar(signal, method="levinson"),
        "ar_ols": fit_ar(signal, method="ols"),
        "arma": fit_arma(signal, 1, 1),
        "state_space_level": fit_state_space(signal),
        "state_space_bsm": fit_state_space(signal, "basic_structural", period=12),
        "trend_seasonal": fit_trend_seasonal(signal, 12),
        "sinusoidal_periodogram": fit_sinusoidal(signal),
        "sinusoidal_music": fit_sinusoidal(signal, 2, "music"),
        "sinusoidal_prony": fit_sinusoidal(signal, 2, "prony"),
        "hmm": fit_hmm(signal, 2),
    }


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fitted_plus_residuals_reconstructs_the_series(name: str) -> None:
    signal = trended_cycle()
    model = models()[name]

    assert [f + r for f, r in zip(model.fitted(), model.residuals())] == pytest.approx(signal, abs=1e-8)
    assert len(model.fitted()) == len(signal)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_extrapolate_returns_exactly_horizon_finite_values(name: str) -> None:
    model = models()[name]

    for horizon in (1, 7, 48):
        future = model.extrapolate(horizon)
        assert len(future) == horizon
        assert all(math.isfinite(value) for value in future)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_rejects_a_non_positive_horizon(name: str) -> None:
    model = models()[name]

    with pytest.raises(NotApplicable):
        model.extrapolate(0)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_beats_a_flat_mean_forecast_in_sample(name: str) -> None:
    signal = trended_cycle()
    model = models()[name]
    flat = [sum(signal) / len(signal)] * len(signal)

    assert mae(model.fitted(), signal) < mae(flat, signal), name


@pytest.mark.parametrize("method", AR_METHODS)
def test_ar_recovers_the_coefficient_of_a_known_ar1(method: str) -> None:
    model = fit_ar(ar1(0.8, 600), order=1, method=method)

    assert model.params["phi_1"] == pytest.approx(0.8, abs=0.15), method


def test_ar_order_selection_prefers_a_small_order_for_an_ar1() -> None:
    assert fit_ar(ar1(0.7, 400), order_criterion="bic").params["order"] <= 3


def test_ar_extrapolation_of_a_stationary_series_decays_towards_the_mean() -> None:
    signal = ar1(0.5, 400)
    model = fit_ar(signal, order=1)
    future = model.extrapolate(60)
    level = sum(signal) / len(signal)

    assert abs(future[-1] - level) < abs(future[0] - level) + 1e-9


def test_trend_seasonal_extrapolation_continues_the_trend_and_the_cycle() -> None:
    period = 12
    signal = trended_cycle(240, period)
    future = fit_trend_seasonal(signal, period).extrapolate(2 * period)

    assert future[period] > future[0]                       # the trend keeps rising
    assert future[0] == pytest.approx(future[period], abs=1.0)  # one cycle apart, same phase


def test_prony_reproduces_a_pure_sinusoid_almost_exactly() -> None:
    signal = sinusoid(20, 200)
    model = fit_sinusoidal(signal, 1, "prony")

    assert mae(model.fitted(), signal) < 1e-6
    assert model.extrapolate(20) == pytest.approx(sinusoid(20, 220)[200:], abs=1e-3)


def test_prony_bounds_an_explosive_series_instead_of_diverging() -> None:
    # Roots outside the unit circle are reflected onto it, so extrapolation stays finite.
    explosive = [1.5 ** i for i in range(40)]
    future = fit_sinusoidal(explosive, 2, "prony").extrapolate(30)

    assert all(math.isfinite(value) for value in future)
    assert max(abs(value) for value in future) < 1e6 * max(explosive)


@pytest.mark.parametrize("method", SINUSOIDAL_METHODS)
def test_every_sinusoidal_method_finds_the_dominant_cycle(method: str) -> None:
    signal = sinusoid(20, 240)
    model = fit_sinusoidal(signal, 1 if method != "periodogram" else 2, method)

    assert mae(model.fitted(), signal) < mae([0.0] * len(signal), signal), method


def test_hmm_separates_two_regimes() -> None:
    signal = mean_shift([0.0, 20.0], run=80)
    model = fit_hmm(signal, 2)
    means = sorted(value for key, value in model.params.items() if key.startswith("mean_"))

    assert means[0] == pytest.approx(0.0, abs=1.0)
    assert means[1] == pytest.approx(20.0, abs=1.0)


def test_hmm_forecast_stays_between_its_state_means() -> None:
    model = fit_hmm(mean_shift([0.0, 20.0], run=80), 2)
    future = model.extrapolate(30)

    assert all(-1.0 <= value <= 21.0 for value in future)


def test_models_reject_arguments_they_cannot_honour() -> None:
    signal = trended_cycle()
    with pytest.raises(NotApplicable):
        fit_ar(signal, method="wibble")  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        fit_ar(signal, order=0)
    with pytest.raises(NotApplicable):
        fit_ar(signal, order=None, order_criterion=None)
    with pytest.raises(NotApplicable):
        fit_state_space(signal, "wibble")  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        fit_state_space(signal, "basic_structural", period=None)
    with pytest.raises(NotApplicable):
        fit_sinusoidal(signal, 0)
    with pytest.raises(NotApplicable):
        fit_sinusoidal(signal, 2, "wibble")  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        fit_hmm(signal, 1)
    with pytest.raises(NotApplicable):
        fit_arma(signal, 0, 0)


def test_models_reject_a_series_too_short_to_fit() -> None:
    with pytest.raises(NotApplicable):
        fit_arma([1.0, 2.0, 3.0], 1, 1)
    with pytest.raises(NotApplicable):
        fit_hmm([1.0, 2.0, 3.0], 2)


# --------------------------------------------------------------------------------------
# Axis E: representation
# --------------------------------------------------------------------------------------


def two_tone(length: int = 64, period: int = 32) -> list[float]:
    return [
        math.sin(2 * math.pi * i / period) + 0.5 * math.cos(2 * math.pi * 3 * i / period)
        for i in range(length)
    ]


@pytest.mark.parametrize("kind", DICTIONARY_KINDS)
def test_every_dictionary_has_unit_norm_atoms_of_the_requested_length(kind: str) -> None:
    atoms = make_dictionary(kind, 64)

    assert atoms
    for atom in atoms:
        assert len(atom) == 64
        assert math.sqrt(sum(v * v for v in atom)) == pytest.approx(1.0, abs=1e-9)


def test_dictionary_size_can_be_capped() -> None:
    assert len(make_dictionary("dct", 32, n_atoms=5)) == 5


def test_make_dictionary_rejects_bad_arguments() -> None:
    with pytest.raises(NotApplicable):
        make_dictionary("wibble", 32)  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        make_dictionary("dct", 1)


def test_omp_recovers_a_two_tone_signal_exactly_from_the_dft_dictionary() -> None:
    signal = two_tone()
    atoms = make_dictionary("dft", 64)
    codes = sparse_code(signal, atoms, "omp", sparsity=4)

    assert sum(1 for v in codes if abs(v) > 1e-9) <= 4
    assert reconstruct(codes, atoms) == pytest.approx(signal, abs=1e-8)


@pytest.mark.parametrize("pursuit", PURSUIT_METHODS)
def test_every_pursuit_produces_a_sparse_code_that_reduces_error(pursuit: str) -> None:
    signal = two_tone()
    atoms = make_dictionary("dft", 64)
    codes = sparse_code(signal, atoms, pursuit, sparsity=6)
    rebuilt = reconstruct(codes, atoms)

    assert len(codes) == len(atoms)
    assert sum(1 for v in codes if abs(v) > 1e-9) < len(atoms)
    assert mae(rebuilt, signal) < mae([0.0] * len(signal), signal), pursuit


def test_the_lasso_penalty_is_scale_free() -> None:
    small = two_tone()
    large = [value * 1000.0 for value in small]
    atoms = make_dictionary("dft", 64)

    small_codes = sparse_code(small, atoms, "lasso", alpha=0.1)
    large_codes = sparse_code(large, atoms, "lasso", alpha=0.1)

    assert sum(1 for v in small_codes if abs(v) > 1e-9) == sum(1 for v in large_codes if abs(v) > 1e-6)
    assert [v * 1000.0 for v in small_codes] == pytest.approx(large_codes, rel=1e-3)


def test_sparse_code_rejects_a_dictionary_of_the_wrong_length() -> None:
    with pytest.raises(NotApplicable):
        sparse_code(two_tone(), make_dictionary("dft", 32), "omp")
    with pytest.raises(NotApplicable):
        sparse_code(two_tone(), make_dictionary("dft", 64), "wibble")  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        sparse_code(two_tone(), make_dictionary("dft", 64), "lasso", alpha=5.0)


def test_reconstruct_rejects_a_code_of_the_wrong_size() -> None:
    with pytest.raises(NotApplicable):
        reconstruct([1.0, 2.0], make_dictionary("dft", 64))


@pytest.mark.parametrize("method", LEARN_METHODS)
def test_a_learned_dictionary_reconstructs_the_windows_it_was_trained_on(method: str) -> None:
    period, width = 16, 16
    signal = [math.sin(2 * math.pi * i / period) for i in range(width * 12)]
    windows = [signal[start:start + width] for start in range(0, len(signal) - width, 4)]

    atoms = learn_dictionary(windows, n_atoms=8, sparsity=2, method=method)

    assert len(atoms) == 8
    assert all(len(atom) == width for atom in atoms)
    errors = [
        mae(reconstruct(sparse_code(window, atoms, "omp", sparsity=2), atoms), window)
        for window in windows
    ]
    assert sum(errors) / len(errors) < 0.2, method


def test_learn_dictionary_rejects_bad_arguments() -> None:
    windows = [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]
    with pytest.raises(NotApplicable):
        learn_dictionary(windows, 4, method="wibble")  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        learn_dictionary(windows, 4, sparsity=9)
    with pytest.raises(NotApplicable):
        learn_dictionary([[1.0, 2.0]], 4)


# --------------------------------------------------------------------------------------
# Axis G: matching
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("metric", DISTANCE_METRICS)
def test_every_distance_is_zero_for_identical_series_and_positive_otherwise(metric: str) -> None:
    a = sinusoid(20, 60)
    b = sinusoid(26, 60)

    assert distance(a, a, metric) == pytest.approx(0.0, abs=1e-6), metric
    assert distance(a, b, metric) > 0.0, metric


@pytest.mark.parametrize("metric", DISTANCE_METRICS)
def test_every_distance_is_symmetric(metric: str) -> None:
    a, b = sinusoid(20, 40), sinusoid(26, 40)

    assert distance(a, b, metric) == pytest.approx(distance(b, a, metric), rel=1e-9), metric


def test_dtw_beats_euclidean_on_a_time_warped_pair() -> None:
    reference = sinusoid(20, 60)
    stretched = sinusoid(26, 60)

    assert distance(reference, stretched, "dtw") < distance(reference, stretched, "euclidean")


def test_dtw_tolerates_different_lengths_but_euclidean_does_not() -> None:
    short, long = sinusoid(20, 40), sinusoid(20, 60)

    assert distance(short, long, "dtw") >= 0.0
    with pytest.raises(NotApplicable):
        distance(short, long, "euclidean")


def test_normalized_euclidean_ignores_offset_and_scale() -> None:
    signal = sinusoid(20, 60)
    shifted = [3.0 * value + 100.0 for value in signal]

    assert distance(signal, shifted, "normalized_euclidean") == pytest.approx(0.0, abs=1e-9)
    assert distance(signal, shifted, "euclidean") > 1.0


def test_distance_rejects_bad_arguments() -> None:
    a = sinusoid(20, 40)
    with pytest.raises(NotApplicable):
        distance(a, a, "wibble")  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        distance(a, a, "dtw", radius=0)
    with pytest.raises(NotApplicable):
        distance(a, a, "soft_dtw", gamma=0.0)


def test_nearest_windows_finds_the_repeat_of_a_motif() -> None:
    motif = [1.0, 5.0, 2.0, 8.0]
    signal = [0.0] * 10 + motif + [0.0] * 10 + motif + [0.0] * 6

    found = nearest_windows(signal, motif, k=1)

    assert found[0] in (10, 24)


def test_nearest_windows_excludes_the_tail_by_default() -> None:
    signal = sinusoid(12, 60)
    query = signal[-12:]

    assert 48 not in nearest_windows(signal, query, k=3)
    assert 48 in nearest_windows(signal, query, k=1, exclude_tail=False)


def test_nearest_windows_rejects_an_oversized_query() -> None:
    with pytest.raises(NotApplicable):
        nearest_windows(sinusoid(12, 20), sinusoid(12, 40))
    with pytest.raises(NotApplicable):
        nearest_windows(sinusoid(12, 40), sinusoid(12, 8), k=0)


def test_matrix_profile_flags_a_planted_discord() -> None:
    repeated = [1.0, 2.0, 3.0, 2.0] * 6
    signal = repeated + [9.0, -9.0, 9.0, -9.0] + repeated
    profile, indices = matrix_profile(signal, 4)

    assert len(profile) == len(indices) == len(signal) - 3
    discord = max(range(len(profile)), key=lambda i: profile[i])
    assert 21 <= discord <= 28, discord


def test_matrix_profile_rejects_an_impossible_window() -> None:
    with pytest.raises(NotApplicable):
        matrix_profile(sinusoid(12, 40), 1)
    with pytest.raises(NotApplicable):
        matrix_profile(sinusoid(12, 40), 30)


@pytest.mark.parametrize("metric", ["euclidean", "dtw"])
def test_barycenter_of_identical_windows_is_that_window(metric: str) -> None:
    window = [1.0, 4.0, 2.0, 7.0, 3.0]

    assert barycenter([window] * 4, metric) == pytest.approx(window, abs=1e-6), metric


def test_euclidean_barycenter_is_the_pointwise_mean() -> None:
    windows = [[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]]

    assert barycenter(windows) == pytest.approx([1.0, 3.0, 5.0])


def test_barycenter_rejects_bad_arguments() -> None:
    with pytest.raises(NotApplicable):
        barycenter([[1.0, 2.0]], "wibble")  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        barycenter([])


# --------------------------------------------------------------------------------------
# Axis H: features
# --------------------------------------------------------------------------------------


def test_features_returns_every_requested_group_and_only_finite_numbers() -> None:
    result = features(sinusoid(20, 120), FEATURE_GROUPS)

    assert result
    assert all(isinstance(value, float) and math.isfinite(value) for value in result.values())
    for expected in ("mean", "spectral_centroid", "approximate_entropy", "symbolic_entropy", "trend_slope"):
        assert expected in result


def test_the_default_groups_are_a_subset_of_everything() -> None:
    assert set(features(sinusoid(20, 120))) < set(features(sinusoid(20, 120), FEATURE_GROUPS))


def test_shape_features_separate_a_flat_line_from_a_trend() -> None:
    flat = features([5.0] * 100, ("shape",))
    rising = features([0.5 * i for i in range(100)], ("shape",))

    assert flat["trend_slope"] == pytest.approx(0.0, abs=1e-9)
    assert rising["trend_slope"] == pytest.approx(0.5, abs=1e-9)
    assert flat["flat_fraction"] == pytest.approx(1.0)
    assert rising["flat_fraction"] == pytest.approx(0.0)


def test_spectral_entropy_is_low_for_a_pure_tone_and_high_for_noise() -> None:
    tone = features(sinusoid(20, 256), ("spectral",))["spectral_entropy"]
    state, noise = 17, []
    for _ in range(256):
        state = (1103515245 * state + 12345) % 2147483648
        noise.append((state / 2147483648.0) - 0.5)

    assert tone < features(noise, ("spectral",))["spectral_entropy"]


def test_zero_fraction_detects_an_intermittent_series() -> None:
    intermittent = [0.0, 0.0, 5.0, 0.0, 0.0, 3.0] * 10

    assert features(intermittent, ("statistical",))["zero_fraction"] == pytest.approx(4 / 6)


def test_features_rejects_an_unknown_group_or_an_empty_request() -> None:
    with pytest.raises(NotApplicable):
        features(sinusoid(20, 60), ("wibble",))  # type: ignore[arg-type]
    with pytest.raises(NotApplicable):
        features(sinusoid(20, 60), ())


def test_sax_produces_a_word_of_the_requested_length_from_the_requested_alphabet() -> None:
    word = sax(sinusoid(20, 120), word_length=10, alphabet_size=4)

    assert len(word) == 10
    assert set(word) <= set("abcd")


def test_sax_is_invariant_to_offset_and_scale() -> None:
    signal = sinusoid(20, 120)
    rescaled = [3.0 * value + 50.0 for value in signal]

    assert sax(signal, 12, 4) == sax(rescaled, 12, 4)


def test_sax_rejects_an_unsupported_alphabet_or_word_length() -> None:
    with pytest.raises(NotApplicable):
        sax(sinusoid(20, 60), alphabet_size=99)
    with pytest.raises(NotApplicable):
        sax(sinusoid(20, 60), word_length=999)
