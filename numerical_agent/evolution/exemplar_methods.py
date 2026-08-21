"""Hand-written forecasting methods that seed generation 0.

These exist to demonstrate composition: every one builds its forecast out of the frozen skill
library rather than implementing statistics inline. They are ordinary methods, so the evolution
loop is free to rewrite or delete any of them once it has measurements. Their docstrings argue
from first principles because nothing has been measured yet; the loop replaces that with
evidence.
"""

import numerical_agent.evolution.primite_ts_skills as P
from numerical_agent.evolution.primite_ts_skills import NotApplicable


def regime_aware_ar(history, horizon, frequency):
    """Use when the series changes level or volatility partway through and only the latest
    regime is informative, such as a sensor that was recalibrated or a series with a step
    change. Segments the history with a mean-and-variance cost, keeps the final regime, and
    fits an autoregressive model to that alone, so older regimes cannot drag the forecast
    toward a level the series has left. Caveat: it discards data, so on a genuinely
    homogeneous series it is a weaker AR model than one fitted to the whole history."""
    if len(history) < 40:
        raise NotApplicable(f"needs at least 40 points to segment, got {len(history)}")
    breaks = P.detect_changepoints(
        history, cost="normal", search="pelt", penalty="bic", min_size=10
    )
    recent = P.last_regime(history, breaks)
    if len(recent) < 12:
        raise NotApplicable(f"the final regime has only {len(recent)} points, needs 12")
    return P.fit_ar(recent, order_criterion="bic").extrapolate(horizon)


def decomposed_trend_seasonal(history, horizon, frequency):
    """Use when the series has both a trend and a stable repeating cycle, which is the common
    case for hourly and daily operational data. Fits a polynomial trend plus a seasonal
    profile, then fits an autoregressive model to what that leaves behind, so short-range
    correlation in the residual is carried into the forecast instead of being thrown away.
    Caveat: the trend is extrapolated as a straight line, so it drifts badly over long horizons
    on a series whose growth is levelling off."""
    period = P.infer_period(history, frequency)
    if period < 2:
        raise NotApplicable("no seasonal period could be detected in this series")
    if len(history) < 4 * period:
        raise NotApplicable(f"needs {4 * period} points for period {period}, got {len(history)}")

    shape = P.fit_trend_seasonal(history, period, trend_degree=1)
    residual = shape.residuals()
    base = shape.extrapolate(horizon)
    noise = P.fit_ar(residual, order_criterion="bic").extrapolate(horizon)
    return [b + n for b, n in zip(base, noise)]


def analogue_continuation(history, horizon, frequency):
    """Use when the series repeats characteristic shapes at irregular intervals, so what
    happened after a similar-looking stretch is better evidence than any fitted model. Matches
    the most recent window against earlier ones under dynamic time warping, which tolerates the
    same shape unfolding at a slightly different speed, and averages what followed the three
    closest matches. Caveat: it can only reproduce patterns already present in the history, so
    it cannot forecast a level the series has never visited."""
    period = P.infer_period(history, frequency)
    width = max(8, min(period, len(history) // 4))
    if len(history) < 2 * width + 2 * horizon:
        raise NotApplicable(
            f"needs {2 * width + 2 * horizon} points for width {width} and horizon {horizon}, "
            f"got {len(history)}"
        )

    # Searching the history minus one horizon guarantees every match has a full continuation.
    searchable = history[: len(history) - horizon]
    query = history[-width:]
    starts = P.nearest_windows(searchable, query, k=3, metric="dtw")
    continuations = [history[start + width: start + width + horizon] for start in starts]
    return P.barycenter(continuations, "euclidean")


def denoised_local_trend(history, horizon, frequency):
    """Use when the series is contaminated by isolated spikes that would otherwise dominate a
    fitted model, such as meter readings with occasional transmission errors. Replaces spikes
    that sit far from a rolling median, then fits a local linear trend by Kalman filtering, so
    the level and slope both adapt to the end of the series rather than being averaged over all
    of it. Caveat: spike removal is indiscriminate about cause, so a real and important jump is
    smoothed away exactly like an error."""
    if len(history) < 24:
        raise NotApplicable(f"needs at least 24 points, got {len(history)}")
    cleaned, _replaced = P.remove_outliers(history, method="hampel", threshold=4.0, window=7)
    return P.fit_state_space(cleaned, "local_linear_trend").extrapolate(horizon)


def spectral_harmonics(history, horizon, frequency):
    """Use when the series is dominated by a few strong periodicities that need not be
    harmonics of one another, such as a signal carrying both a daily and a weekly rhythm.
    Fits the three strongest frequencies found in the periodogram as sinusoids by least
    squares and continues them forward. Caveat: it models the series as strictly periodic, so
    it captures no trend at all and will sit at the wrong level on a drifting series."""
    if len(history) < 32:
        raise NotApplicable(f"needs at least 32 points to resolve frequencies, got {len(history)}")
    return P.fit_sinusoidal(history, n_components=3, method="periodogram").extrapolate(horizon)
