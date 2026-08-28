"""Hand-written forecasting methods that seed generation 0.

Every one is self-contained: it implements its own algorithm from the allowed libraries,
imported inside the function body, because that is what the method contract now requires.
They are ordinary methods, so the evolution loop is free to rewrite, merge or delete any of
them once it has measurements. Their docstrings argue from first principles because nothing
has been measured yet; the loop replaces that with evidence.
"""


class NotApplicable(Exception):
    """Raised by a method whose stated preconditions the series does not meet."""


def regime_aware_ar(history, horizon, frequency):
    """Use when the series changes level partway through and only the latest regime is
    informative, such as a sensor that was recalibrated or a series with a step change. Finds
    the single strongest mean shift, keeps what follows it, and fits an autoregressive model to
    that alone, so an abandoned level cannot drag the forecast back toward it. Caveat: it
    discards data, so on a genuinely homogeneous series it is a weaker AR model than one fitted
    to the whole history."""
    import numpy as np
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order

    if len(history) < 40:
        raise NotApplicable(f"needs at least 40 points to segment, got {len(history)}")
    values = np.asarray(history, dtype=float)
    total = len(values)

    best_split, best_score = 0, 0.0
    for split in range(12, total - 12):
        left, right = values[:split], values[split:]
        spread = np.sqrt(left.var() / len(left) + right.var() / len(right))
        if spread <= 0.0:
            continue
        score = abs(left.mean() - right.mean()) / spread
        if score > best_score:
            best_score, best_split = float(score), split

    # Below this the split is noise, and cutting the history would only cost the model data.
    recent = values[best_split:] if best_score > 3.0 else values
    if len(recent) < 12:
        raise NotApplicable(f"the final regime has only {len(recent)} points, needs 12")

    selection = ar_select_order(recent, maxlag=max(1, min(12, len(recent) // 4)), ic="bic")
    lags = selection.ar_lags if selection.ar_lags else 1
    fitted = AutoReg(recent, lags=lags, old_names=False).fit()
    return [float(value) for value in fitted.forecast(horizon)]


def decomposed_trend_seasonal(history, horizon, frequency):
    """Use when the series has both a trend and a stable repeating cycle, which is the common
    case for hourly and daily operational data. Fits a linear trend plus a seasonal profile by
    least squares, then fits an autoregressive model to what that leaves behind, so short-range
    correlation in the residual is carried into the forecast instead of being thrown away.
    Caveat: the trend is extrapolated as a straight line, so it drifts badly over long horizons
    on a series whose growth is levelling off."""
    import numpy as np
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order

    period = 0
    label = frequency.lower()
    for token, candidate in (("hour", 24), ("day", 7), ("week", 52), ("month", 12), ("quarter", 4)):
        if token in label:
            period = candidate
            break
    if period < 2:
        raise NotApplicable(f"no seasonal period is defined for frequency {frequency!r}")
    if len(history) < 4 * period:
        raise NotApplicable(f"needs {4 * period} points for period {period}, got {len(history)}")

    values = np.asarray(history, dtype=float)
    total = len(values)
    steps = np.arange(total, dtype=float)
    season = np.eye(period)[np.arange(total) % period]
    design = np.column_stack([steps, season])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)

    future_steps = np.arange(total, total + horizon, dtype=float)
    future_season = np.eye(period)[np.arange(total, total + horizon) % period]
    base = np.column_stack([future_steps, future_season]) @ coefficients

    residual = values - design @ coefficients
    selection = ar_select_order(residual, maxlag=max(1, min(period, len(residual) // 4)), ic="bic")
    lags = selection.ar_lags if selection.ar_lags else 1
    noise = AutoReg(residual, lags=lags, old_names=False).fit().forecast(horizon)
    return [float(b + n) for b, n in zip(base, noise)]


def analogue_continuation(history, horizon, frequency):
    """Use when the series repeats characteristic shapes at irregular intervals, so what
    happened after a similar-looking stretch is better evidence than any fitted model. Matches
    the most recent window against earlier ones under dynamic time warping, which tolerates the
    same shape unfolding at a slightly different speed, and averages what followed the three
    closest matches. Caveat: it can only reproduce patterns already present in the history, so
    it cannot forecast a level the series has never visited."""
    import numpy as np

    values = np.asarray(history, dtype=float)
    width = max(8, min(24, len(values) // 8))
    if len(values) < 2 * width + 2 * horizon:
        raise NotApplicable(
            f"needs {2 * width + 2 * horizon} points for width {width} and horizon {horizon}, "
            f"got {len(values)}"
        )

    def warped_distance(left, right):
        """Dynamic time warping cost between two equal-length windows."""
        cost = np.full((len(left) + 1, len(right) + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, len(left) + 1):
            for j in range(1, len(right) + 1):
                step = abs(left[i - 1] - right[j - 1])
                cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        return float(cost[-1, -1])

    query = values[-width:]
    # Stopping one horizon short of the end guarantees every match has a full continuation.
    last_start = len(values) - horizon - width
    scored = [
        (warped_distance(values[start : start + width], query), start)
        for start in range(0, last_start)
    ]
    if not scored:
        raise NotApplicable(f"no earlier window of width {width} has a full continuation")

    scored.sort()
    continuations = [
        values[start + width : start + width + horizon] for _distance, start in scored[:3]
    ]
    return [float(value) for value in np.mean(continuations, axis=0)]


def denoised_local_trend(history, horizon, frequency):
    """Use when the series is contaminated by isolated spikes that would otherwise dominate a
    fitted model, such as meter readings with occasional transmission errors. Replaces points
    far from a rolling median, then fits a local linear trend state-space model, so the level
    and slope both adapt to the end of the series rather than being averaged over all of it.
    Caveat: spike removal is indiscriminate about cause, so a real and important jump is
    smoothed away exactly like an error."""
    import numpy as np
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    if len(history) < 24:
        raise NotApplicable(f"needs at least 24 points, got {len(history)}")
    values = np.asarray(history, dtype=float)

    half = 3
    cleaned = values.copy()
    for index in range(len(values)):
        window = values[max(0, index - half) : index + half + 1]
        centre = np.median(window)
        # 1.4826 scales the median absolute deviation to a standard deviation for normal data.
        deviation = 1.4826 * np.median(np.abs(window - centre))
        if deviation > 0.0 and abs(values[index] - centre) > 4.0 * deviation:
            cleaned[index] = centre

    fitted = UnobservedComponents(cleaned, level="local linear trend").fit(disp=0)
    return [float(value) for value in fitted.forecast(horizon)]


def spectral_harmonics(history, horizon, frequency):
    """Use when the series is dominated by a few strong periodicities that need not be
    harmonics of one another, such as a signal carrying both a daily and a weekly rhythm.
    Fits the three strongest frequencies found in the periodogram as sinusoids by least squares
    and continues them forward. Caveat: it models the series as strictly periodic, so it
    captures no trend at all and will sit at the wrong level on a drifting series."""
    import numpy as np

    if len(history) < 32:
        raise NotApplicable(f"needs at least 32 points to resolve frequencies, got {len(history)}")
    values = np.asarray(history, dtype=float)
    total = len(values)

    spectrum = np.abs(np.fft.rfft(values - values.mean())) ** 2
    # Bin 0 is the mean, which the intercept already carries.
    strongest = np.argsort(spectrum[1:])[::-1][:3] + 1
    frequencies = strongest / total

    def design(steps):
        columns = [np.ones(len(steps))]
        for cycles in frequencies:
            columns.append(np.sin(2.0 * np.pi * cycles * steps))
            columns.append(np.cos(2.0 * np.pi * cycles * steps))
        return np.column_stack(columns)

    steps = np.arange(total, dtype=float)
    coefficients, *_ = np.linalg.lstsq(design(steps), values, rcond=None)
    future = design(np.arange(total, total + horizon, dtype=float)) @ coefficients
    return [float(value) for value in future]
