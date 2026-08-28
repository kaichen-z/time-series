"""Frozen time-series analysis skills that evolved forecasting methods compose.

This module is written and tested by hand and is never mutated by the evolution loop. Methods
import it as ``P`` and build forecasts by combining these skills, so the only thing an evolved
method can get wrong is the composition.
"""
from __future__ import annotations

import math
from typing import Literal, Protocol, Sequence

import numpy as np


# Type vocabulary

Series = list[float]
Breaks = list[int]                          # segment boundaries, ascending, exclusive of 0 and N
Spectrum = list[tuple[float, float, float]]  # (cycles_per_sample, amplitude, phase_radians)
Features = dict[str, float]


class NotApplicable(Exception):
    """Raised when a series does not meet a skill's stated requirements.

    Evolved methods import this same class, so a skill declining a series is classified as
    not-applicable rather than as a crash.
    """


class Model(Protocol):
    """A fitted model that can extrapolate; the bridge from analysis to forecasting."""

    params: dict[str, float]

    def extrapolate(self, horizon: int) -> Series:
        """Return exactly horizon future values."""

    def fitted(self) -> Series:
        """Return the in-sample reconstruction, same length as the fitted series."""

    def residuals(self) -> Series:
        """Return the fitted series minus its reconstruction."""


# Seasonal period in samples for each frequency the Dr-CiK tasks use. Only a hint: skills
# estimate the period from the data first and fall back to this when the data is uninformative.
FREQUENCY_PERIODS: dict[str, int] = {
    "1 second": 60,
    "1 minute": 60,
    "5 minutes": 12,
    "15 minutes": 4,
    "30 minutes": 48,
    "1 hour": 24,
    "1 day": 7,
    "1 week": 52,
    "1 month": 12,
    "1 year": 1,
}

_MIN_PERIOD = 2
_PEAK_THRESHOLD = 0.1


def _as_array(x: Sequence[float], *, minimum: int, what: str) -> np.ndarray:
    values = np.asarray(list(x), dtype=float)
    if values.size < minimum:
        raise NotApplicable(f"{what} needs at least {minimum} points, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise NotApplicable(f"{what} needs finite values; the series contains nan or inf")
    return values


# Structure inference


def acf(x: Sequence[float], max_lag: int) -> Series:
    """Autocorrelation at lags 0..max_lag; a constant series correlates 1.0 at lag 0 and 0.0 after."""
    values = _as_array(x, minimum=2, what="acf")
    if max_lag < 0:
        raise NotApplicable(f"max_lag must be non-negative, got {max_lag}")
    if max_lag >= values.size:
        raise NotApplicable(f"max_lag {max_lag} needs more than {max_lag} points, got {values.size}")

    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        # A constant series has no variance to correlate; report no structure rather than nan.
        return [1.0] + [0.0] * max_lag
    return [
        1.0 if lag == 0 else float(np.dot(centered[lag:], centered[:-lag]) / denominator)
        for lag in range(max_lag + 1)
    ]


def pacf(x: Sequence[float], max_lag: int) -> Series:
    """Partial autocorrelation at lags 0..max_lag by Durbin-Levinson, for choosing an AR order."""
    if max_lag < 0:
        raise NotApplicable(f"max_lag must be non-negative, got {max_lag}")
    values = _as_array(x, minimum=2, what="pacf")
    if max_lag >= values.size // 2:
        raise NotApplicable(
            f"pacf at lag {max_lag} needs more than {2 * max_lag} points, got {values.size}"
        )

    rho = acf(x, max_lag)
    partial = [1.0] + [0.0] * max_lag
    phi = np.zeros((max_lag + 1, max_lag + 1))
    for k in range(1, max_lag + 1):
        numerator = rho[k] - sum(phi[k - 1, j] * rho[k - j] for j in range(1, k))
        denominator = 1.0 - sum(phi[k - 1, j] * rho[j] for j in range(1, k))
        phi[k, k] = 0.0 if abs(denominator) < 1e-12 else numerator / denominator
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        partial[k] = float(phi[k, k])
    return partial


def spectrum(
    x: Sequence[float],
    method: Literal["dft", "periodogram", "welch"] = "dft",
) -> Spectrum:
    """Frequency content as (cycles_per_sample, amplitude, phase), zero frequency dropped.

    Only ``dft`` carries a real phase; ``periodogram`` and ``welch`` estimate power, so their
    amplitude is the square root of power density and their phase is always 0.0.
    """
    values = _as_array(x, minimum=4, what="spectrum")

    if method == "dft":
        coefficients = np.fft.rfft(values)
        frequencies = np.fft.rfftfreq(values.size, d=1.0)
        amplitudes = 2.0 * np.abs(coefficients) / values.size
        phases = np.angle(coefficients)
        return [
            (float(f), float(a), float(p))
            for f, a, p in zip(frequencies[1:], amplitudes[1:], phases[1:])
        ]

    if method in ("periodogram", "welch"):
        from scipy import signal as _signal

        if method == "periodogram":
            frequencies, power = _signal.periodogram(values, fs=1.0)
        else:
            segment = min(values.size, max(8, values.size // 4))
            frequencies, power = _signal.welch(values, fs=1.0, nperseg=segment)
        return [
            (float(f), float(np.sqrt(max(p, 0.0))), 0.0)
            for f, p in zip(frequencies[1:], power[1:])
        ]

    raise NotApplicable(f"unknown spectrum method {method!r}")


def dominant_frequencies(
    x: Sequence[float],
    k: int = 3,
    method: Literal["dft", "periodogram", "welch"] = "dft",
) -> Spectrum:
    """The k strongest frequency components, highest amplitude first."""
    if k < 1:
        raise NotApplicable(f"k must be at least 1, got {k}")
    components = spectrum(x, method=method)
    return sorted(components, key=lambda component: component[1], reverse=True)[:k]


def infer_period(
    x: Sequence[float],
    frequency: str | None = None,
    max_period: int | None = None,
) -> int:
    """Seasonal period in samples, estimated from autocorrelation peaks.

    Falls back to the frequency hint in FREQUENCY_PERIODS when the series shows no periodic
    structure, and to 1 when there is no usable hint either.
    """
    values = _as_array(x, minimum=4, what="infer_period")
    ceiling = values.size // 2 if max_period is None else min(max_period, values.size // 2)
    hint = FREQUENCY_PERIODS.get(frequency or "")

    if ceiling >= _MIN_PERIOD:
        correlations = acf(values, ceiling)
        best_lag, best_value = 0, _PEAK_THRESHOLD
        for lag in range(_MIN_PERIOD, ceiling):
            neighbours = correlations[lag] > correlations[lag - 1] and correlations[lag] > correlations[lag + 1]
            if neighbours and correlations[lag] > best_value:
                best_lag, best_value = lag, correlations[lag]
        if best_lag:
            return best_lag

    if hint is not None and _MIN_PERIOD <= hint <= max(ceiling, 1):
        return hint
    return 1


# Segmentation
#
# Change-point detection is not one algorithm but three orthogonal choices: a cost function
# that scores how homogeneous a segment is, a search method that decides where to cut, and a
# penalty that decides how many cuts are worth making. Every combination below is valid.

Cost = Literal["l2", "normal", "ar", "kernel_rbf", "rank"]
Search = Literal["optimal", "window", "binseg", "bottomup", "pelt"]
Penalty = Literal["bic", "aic", "linear", "none"]

COSTS: tuple[str, ...] = ("l2", "normal", "ar", "kernel_rbf", "rank")
SEARCHES: tuple[str, ...] = ("optimal", "window", "binseg", "bottomup", "pelt")
PENALTIES: tuple[str, ...] = ("bic", "aic", "linear", "none")

_KERNEL_MAX_POINTS = 3000


class _SegmentCost:
    """Segment cost c(start, end) with O(1) evaluation after an O(n) or O(n^2) setup."""

    def __init__(self, values: np.ndarray, kind: str) -> None:
        if kind not in COSTS:
            raise NotApplicable(f"unknown cost {kind!r}; expected one of {list(COSTS)}")
        self.kind = kind
        self.size = values.size
        self.min_size = 3 if kind == "ar" else 2
        self.n_params = {"l2": 1, "normal": 2, "ar": 2, "kernel_rbf": 1, "rank": 1}[kind]

        source = _rank_transform(values) if kind == "rank" else values
        if kind == "kernel_rbf":
            self._setup_kernel(values)
        else:
            self._setup_sums(source)
            if kind == "ar":
                self._setup_ar(source)

        # Scale-dependent costs are measured in squared data units, so an unscaled log(n)
        # penalty would be meaningless on large-magnitude series. Log-likelihood and kernel
        # costs are already dimensionless.
        if kind in ("normal", "kernel_rbf"):
            self.penalty_scale = 1.0
        else:
            spread = float(np.var(source))
            self.penalty_scale = spread if spread > 1e-12 else 1.0

    def _setup_sums(self, source: np.ndarray) -> None:
        self._sum = np.concatenate(([0.0], np.cumsum(source)))
        self._sumsq = np.concatenate(([0.0], np.cumsum(source * source)))

    def _setup_ar(self, source: np.ndarray) -> None:
        lagged, current = source[:-1], source[1:]
        self._ar_x = np.concatenate(([0.0], np.cumsum(lagged)))
        self._ar_y = np.concatenate(([0.0], np.cumsum(current)))
        self._ar_xx = np.concatenate(([0.0], np.cumsum(lagged * lagged)))
        self._ar_yy = np.concatenate(([0.0], np.cumsum(current * current)))
        self._ar_xy = np.concatenate(([0.0], np.cumsum(lagged * current)))

    def _setup_kernel(self, values: np.ndarray) -> None:
        if values.size > _KERNEL_MAX_POINTS:
            raise NotApplicable(
                f"kernel_rbf needs at most {_KERNEL_MAX_POINTS} points, got {values.size}"
            )
        differences = values[:, None] - values[None, :]
        squared = differences * differences
        # Median heuristic for the bandwidth, ignoring the zero diagonal.
        off_diagonal = squared[~np.eye(values.size, dtype=bool)]
        median = float(np.median(off_diagonal)) if off_diagonal.size else 1.0
        bandwidth = median if median > 1e-12 else 1.0
        gram = np.exp(-squared / bandwidth)
        self._gram_cumulative = np.pad(gram.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))

    def __call__(self, start: int, end: int) -> float:
        length = end - start
        if length < self.min_size:
            return math.inf

        if self.kind == "kernel_rbf":
            block = float(
                self._gram_cumulative[end, end]
                - self._gram_cumulative[start, end]
                - self._gram_cumulative[end, start]
                + self._gram_cumulative[start, start]
            )
            return length - block / length

        total = self._sum[end] - self._sum[start]
        squares = self._sumsq[end] - self._sumsq[start]
        deviation = max(squares - total * total / length, 0.0)

        if self.kind in ("l2", "rank"):
            return float(deviation)
        if self.kind == "normal":
            variance = deviation / length
            return float(length * math.log(variance + 1e-12))
        return self._ar_cost(start, end)

    def _ar_cost(self, start: int, end: int) -> float:
        """Residual sum of squares of an AR(1) fitted on the segment."""
        lo, hi = start, end - 1  # index range into the lagged/current arrays
        count = hi - lo
        if count < 2:
            return math.inf
        sx = self._ar_x[hi] - self._ar_x[lo]
        sy = self._ar_y[hi] - self._ar_y[lo]
        sxx = self._ar_xx[hi] - self._ar_xx[lo]
        syy = self._ar_yy[hi] - self._ar_yy[lo]
        sxy = self._ar_xy[hi] - self._ar_xy[lo]
        centered_xx = sxx - sx * sx / count
        centered_xy = sxy - sx * sy / count
        centered_yy = syy - sy * sy / count
        if centered_xx <= 1e-12:
            return float(max(centered_yy, 0.0))
        slope = centered_xy / centered_xx
        return float(max(centered_yy - slope * centered_xy, 0.0))


def _rank_transform(values: np.ndarray) -> np.ndarray:
    """Replace values by their ranks, so the cost stops depending on the marginal distribution."""
    order = np.argsort(np.argsort(values))
    return order.astype(float)


def _penalty_value(kind: str, cost: _SegmentCost, size: int) -> float:
    if kind not in PENALTIES:
        raise NotApplicable(f"unknown penalty {kind!r}; expected one of {list(PENALTIES)}")
    if kind == "none":
        return 0.0
    if kind == "bic":
        return cost.n_params * math.log(size) * cost.penalty_scale
    if kind == "aic":
        return 2.0 * cost.n_params * cost.penalty_scale
    return cost.n_params * cost.penalty_scale


def _search_optimal(cost: _SegmentCost, size: int, beta: float, k: int | None, min_size: int) -> Breaks:
    """Exact dynamic programming: penalised when k is None, otherwise exactly k breaks."""
    if k is None:
        best = [0.0] + [math.inf] * size
        previous = [0] * (size + 1)
        for end in range(min_size, size + 1):
            for start in range(0, end - min_size + 1):
                if best[start] == math.inf:
                    continue
                candidate = best[start] + cost(start, end) + beta
                if candidate < best[end]:
                    best[end], previous[end] = candidate, start
        return _walk_back(previous, size)

    if (k + 1) * min_size > size:
        raise NotApplicable(f"{k} breaks need at least {(k + 1) * min_size} points, got {size}")
    best = np.full((k + 2, size + 1), math.inf)
    previous = np.zeros((k + 2, size + 1), dtype=int)
    best[0, 0] = 0.0
    for segments in range(1, k + 2):
        for end in range(segments * min_size, size + 1):
            for start in range((segments - 1) * min_size, end - min_size + 1):
                if best[segments - 1, start] == math.inf:
                    continue
                candidate = best[segments - 1, start] + cost(start, end)
                if candidate < best[segments, end]:
                    best[segments, end], previous[segments, end] = candidate, start
    if best[k + 1, size] == math.inf:
        raise NotApplicable(f"no admissible segmentation into {k + 1} segments of size {min_size}")
    breaks, end = [], size
    for segments in range(k + 1, 0, -1):
        start = int(previous[segments, end])
        if start:
            breaks.append(start)
        end = start
    return sorted(breaks)


def _search_pelt(cost: _SegmentCost, size: int, beta: float, min_size: int) -> Breaks:
    """Same recursion as penalised optimal, with pruning that removes hopeless candidates."""
    best = [0.0] + [math.inf] * size
    previous = [0] * (size + 1)
    candidates = [0]
    for end in range(min_size, size + 1):
        scores = []
        for start in candidates:
            if end - start < min_size:
                scores.append(math.inf)
                continue
            scores.append(best[start] + cost(start, end) + beta)
        if not scores:
            continue
        position = int(np.argmin(scores))
        if scores[position] < best[end]:
            best[end], previous[end] = scores[position], candidates[position]
        # Pruning: a start that already exceeds the best total here can never win later.
        candidates = [
            start for start, score in zip(candidates, scores)
            if score != math.inf and score - beta <= best[end]
        ]
        candidates.append(end - min_size + 1)
    return _walk_back(previous, size)


def _walk_back(previous: Sequence[int], size: int) -> Breaks:
    breaks, end = [], size
    while end > 0:
        start = int(previous[end])
        if start:
            breaks.append(start)
        if start >= end:
            break
        end = start
    return sorted(breaks)


def _search_binseg(cost: _SegmentCost, size: int, beta: float, k: int | None, min_size: int) -> Breaks:
    """Greedy: repeatedly split the segment whose best single cut gains the most."""
    breaks: list[int] = []
    while k is None or len(breaks) < k:
        bounds = [0] + breaks + [size]
        best_gain, best_point = -math.inf, None
        for start, end in zip(bounds[:-1], bounds[1:]):
            whole = cost(start, end)
            if whole == math.inf:
                continue
            for middle in range(start + min_size, end - min_size + 1):
                gain = whole - cost(start, middle) - cost(middle, end)
                if gain > best_gain:
                    best_gain, best_point = gain, middle
        if best_point is None:
            break
        if k is None and best_gain <= beta:
            break
        breaks = sorted(breaks + [best_point])
    return breaks


def _search_bottomup(cost: _SegmentCost, size: int, beta: float, k: int | None, min_size: int) -> Breaks:
    """Start from a fine grid and repeatedly remove the break whose removal costs least."""
    breaks = list(range(min_size, size - min_size + 1, min_size))
    target = k if k is not None else 0
    while len(breaks) > target:
        bounds = [0] + breaks + [size]
        best_increase, best_index = math.inf, None
        for index in range(len(breaks)):
            start, middle, end = bounds[index], bounds[index + 1], bounds[index + 2]
            increase = cost(start, end) - cost(start, middle) - cost(middle, end)
            if increase < best_increase:
                best_increase, best_index = increase, index
        if best_index is None:
            break
        if k is None and best_increase > beta:
            break
        breaks.pop(best_index)
    return breaks


def _search_window(cost: _SegmentCost, size: int, beta: float, k: int | None, min_size: int) -> Breaks:
    """Slide a two-sided window and keep the peaks of the left/right discrepancy."""
    width = max(2 * min_size, size // 10)
    if 2 * width >= size:
        width = max(min_size, size // 4)
    scores = np.zeros(size)
    for centre in range(width, size - width + 1):
        whole = cost(centre - width, centre + width)
        if whole == math.inf:
            continue
        scores[centre] = whole - cost(centre - width, centre) - cost(centre, centre + width)

    peaks = [
        centre for centre in range(width + 1, size - width)
        if scores[centre] > scores[centre - 1] and scores[centre] >= scores[centre + 1]
    ]
    peaks.sort(key=lambda centre: scores[centre], reverse=True)
    if k is not None:
        chosen = _spread_out(peaks, min_size)[:k]
    else:
        chosen = _spread_out([centre for centre in peaks if scores[centre] > beta], min_size)
    return sorted(chosen)


def _spread_out(peaks: Sequence[int], min_size: int) -> list[int]:
    """Keep peaks in score order, dropping any that crowd an already-kept one."""
    kept: list[int] = []
    for centre in peaks:
        if all(abs(centre - other) >= min_size for other in kept):
            kept.append(centre)
    return kept


def detect_changepoints(
    x: Sequence[float],
    *,
    cost: Cost = "l2",
    search: Search = "pelt",
    penalty: Penalty = "bic",
    n_breaks: int | None = None,
    min_size: int = 2,
) -> Breaks:
    """Indices where the series changes regime, ascending and exclusive of 0 and len(x).

    Pick a cost for what kind of change to notice (mean, mean and variance, autoregressive
    dynamics, arbitrary distribution), a search for how to look, and a penalty for how many
    cuts to accept. Pass n_breaks to fix the number of cuts and ignore the penalty.
    """
    values = _as_array(x, minimum=4, what="detect_changepoints")
    if search not in SEARCHES:
        raise NotApplicable(f"unknown search {search!r}; expected one of {list(SEARCHES)}")
    if min_size < 1:
        raise NotApplicable(f"min_size must be at least 1, got {min_size}")
    if n_breaks is not None and n_breaks < 0:
        raise NotApplicable(f"n_breaks must be non-negative, got {n_breaks}")
    if penalty == "none" and n_breaks is None:
        raise NotApplicable("penalty 'none' needs an explicit n_breaks")
    if n_breaks == 0:
        return []

    scorer = _SegmentCost(values, cost)
    floor = max(min_size, scorer.min_size)
    if values.size < 2 * floor:
        raise NotApplicable(f"needs at least {2 * floor} points for min_size {floor}, got {values.size}")
    beta = _penalty_value(penalty, scorer, values.size)

    if search == "pelt":
        if n_breaks is not None:
            # PELT prunes using the penalty, so a fixed count is served by exact DP instead.
            return _search_optimal(scorer, values.size, beta, n_breaks, floor)
        return _search_pelt(scorer, values.size, beta, floor)
    if search == "optimal":
        return _search_optimal(scorer, values.size, beta, n_breaks, floor)
    if search == "binseg":
        return _search_binseg(scorer, values.size, beta, n_breaks, floor)
    if search == "bottomup":
        return _search_bottomup(scorer, values.size, beta, n_breaks, floor)
    return _search_window(scorer, values.size, beta, n_breaks, floor)


def segment_cost(x: Sequence[float], start: int, end: int, cost: Cost = "l2") -> float:
    """Cost of treating x[start:end] as one homogeneous segment; lower is more homogeneous."""
    values = _as_array(x, minimum=2, what="segment_cost")
    if not 0 <= start < end <= values.size:
        raise NotApplicable(f"invalid segment [{start}, {end}) for {values.size} points")
    return float(_SegmentCost(values, cost)(start, end))


def last_regime(x: Sequence[float], breaks: Sequence[int]) -> Series:
    """The final segment of x, or all of x when there are no breaks."""
    values = _as_array(x, minimum=1, what="last_regime")
    if not breaks:
        return [float(v) for v in values]
    start = int(max(breaks))
    if not 0 < start < values.size:
        raise NotApplicable(f"break {start} is outside [1, {values.size})")
    return [float(v) for v in values[start:]]


# Cleaning

DenoiseMethod = Literal["moving_average", "median", "hampel", "savgol", "butterworth", "svd"]
InterpolateMethod = Literal["linear", "spline", "polynomial", "low_rank"]
OutlierMethod = Literal["zscore", "hampel", "iqr"]

DENOISE_METHODS: tuple[str, ...] = (
    "moving_average", "median", "hampel", "savgol", "butterworth", "svd",
)
INTERPOLATE_METHODS: tuple[str, ...] = ("linear", "spline", "polynomial", "low_rank")
OUTLIER_METHODS: tuple[str, ...] = ("zscore", "hampel", "iqr")

_MAD_TO_SIGMA = 1.4826


def _with_gaps(x: Sequence[float], *, minimum: int, what: str) -> np.ndarray:
    """Like _as_array but tolerates nan and None, which mark the gaps to be filled."""
    values = np.asarray([np.nan if v is None else float(v) for v in x], dtype=float)
    if values.size < minimum:
        raise NotApplicable(f"{what} needs at least {minimum} points, got {values.size}")
    if np.any(np.isinf(values)):
        raise NotApplicable(f"{what} cannot fill infinities; only nan marks a gap")
    return values


def _odd_window(window: int, size: int, what: str) -> int:
    if window < 1:
        raise NotApplicable(f"{what} window must be at least 1, got {window}")
    if window > size:
        raise NotApplicable(f"{what} window {window} exceeds the {size} points available")
    return window if window % 2 else window - 1


def _rolling_median_and_scale(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    half = window // 2
    padded = np.pad(values, half, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    median = np.median(windows, axis=1)
    deviation = np.median(np.abs(windows - median[:, None]), axis=1)
    return median, _MAD_TO_SIGMA * deviation


def _fallback_scale(local: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Replace a zero rolling scale with a global one.

    A perfectly flat window has a MAD of exactly zero, and treating that as "no scale" would
    silently accept a spike sitting beside a flat run -- the clearest outlier there is.
    """
    global_scale = _MAD_TO_SIGMA * float(np.median(np.abs(values - np.median(values))))
    if global_scale <= 1e-12:
        global_scale = float(np.std(values))
    return np.where(local > 1e-12, local, max(global_scale, 1e-12))


def _hankel(values: np.ndarray, width: int) -> np.ndarray:
    rows = values.size - width + 1
    return np.lib.stride_tricks.sliding_window_view(values, width)[:rows]


def _average_antidiagonals(matrix: np.ndarray) -> np.ndarray:
    rows, columns = matrix.shape
    size = rows + columns - 1
    totals = np.zeros(size)
    counts = np.zeros(size)
    for row in range(rows):
        totals[row:row + columns] += matrix[row]
        counts[row:row + columns] += 1.0
    return totals / counts


def _low_rank_series(
    values: np.ndarray, rank: int | None, width: int | None, energy: float = 0.99
) -> np.ndarray:
    """Singular-spectrum reconstruction: embed, truncate the SVD, average back.

    Rank is chosen by cumulative spectral energy rather than by a ratio to the largest
    singular value, because a linear ramp is exactly rank two yet its second component is
    only 7% of its first -- a ratio rule silently discards the slope.
    """
    span = width or max(2, min(values.size // 2, 50))
    if span < 2 or span >= values.size:
        raise NotApplicable(f"low-rank needs an embedding width in [2, {values.size}), got {span}")
    if not 0.0 < energy <= 1.0:
        raise NotApplicable(f"energy must be in (0, 1], got {energy}")
    matrix = _hankel(values, span)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    if rank is not None:
        keep = rank
    else:
        total = float(np.sum(singular ** 2))
        if total <= 1e-24:
            keep = 1
        else:
            cumulative = np.cumsum(singular ** 2) / total
            keep = int(np.searchsorted(cumulative, energy)) + 1
    keep = max(1, min(keep, singular.size))
    reduced = (left[:, :keep] * singular[:keep]) @ right[:keep]
    return _average_antidiagonals(reduced)


def denoise(
    x: Sequence[float],
    method: DenoiseMethod = "moving_average",
    *,
    window: int = 5,
    threshold: float = 3.0,
    polyorder: int = 2,
    cutoff: float = 0.1,
    rank: int | None = None,
    energy: float = 0.99,
) -> Series:
    """Smooth a series, returning the same number of points.

    ``hampel`` replaces only points more than ``threshold`` robust deviations from a rolling
    median, so it removes spikes while leaving genuine structure alone; the others smooth
    everywhere. ``svd`` is a singular-spectrum reconstruction rather than a filter.
    """
    values = _as_array(x, minimum=3, what="denoise")
    if method not in DENOISE_METHODS:
        raise NotApplicable(f"unknown denoise method {method!r}; expected one of {list(DENOISE_METHODS)}")

    if method == "moving_average":
        span = _odd_window(window, values.size, "denoise")
        padded = np.pad(values, span // 2, mode="reflect")
        kernel = np.ones(span) / span
        return [float(v) for v in np.convolve(padded, kernel, mode="valid")]

    if method == "median":
        from scipy import signal as _signal

        return [float(v) for v in _signal.medfilt(values, _odd_window(window, values.size, "denoise"))]

    if method == "hampel":
        span = _odd_window(max(window, 3), values.size, "denoise")
        median, scale = _rolling_median_and_scale(values, span)
        spikes = np.abs(values - median) > threshold * _fallback_scale(scale, values)
        return [float(v) for v in np.where(spikes, median, values)]

    if method == "savgol":
        from scipy import signal as _signal

        span = _odd_window(window, values.size, "denoise")
        if polyorder >= span:
            raise NotApplicable(f"savgol polyorder {polyorder} must be below the window {span}")
        return [float(v) for v in _signal.savgol_filter(values, span, polyorder)]

    if method == "butterworth":
        from scipy import signal as _signal

        if not 0.0 < cutoff < 0.5:
            raise NotApplicable(f"butterworth cutoff must be in (0, 0.5), got {cutoff}")
        if values.size <= 9:
            raise NotApplicable(f"butterworth needs more than 9 points, got {values.size}")
        numerator, denominator = _signal.butter(2, cutoff / 0.5, btype="low")
        return [float(v) for v in _signal.filtfilt(numerator, denominator, values)]

    return [float(v) for v in _low_rank_series(values, rank, None, energy)]


def interpolate_missing(
    x: Sequence[float],
    method: InterpolateMethod = "linear",
    *,
    degree: int = 3,
    rank: int | None = None,
    energy: float = 0.9999,
) -> Series:
    """Fill nan or None gaps, returning a fully finite series of the same length.

    ``low_rank`` keeps far more spectral energy than ``denoise`` does, because filling a gap
    calls for faithful reconstruction rather than smoothing.
    """
    values = _with_gaps(x, minimum=2, what="interpolate_missing")
    if method not in INTERPOLATE_METHODS:
        raise NotApplicable(
            f"unknown interpolate method {method!r}; expected one of {list(INTERPOLATE_METHODS)}"
        )
    gaps = np.isnan(values)
    if not gaps.any():
        return [float(v) for v in values]
    known = ~gaps
    if known.sum() < 2:
        raise NotApplicable(f"needs at least 2 known points, got {int(known.sum())}")

    positions = np.arange(values.size, dtype=float)
    filled = values.copy()

    if method == "linear":
        filled[gaps] = np.interp(positions[gaps], positions[known], values[known])
        return [float(v) for v in filled]

    if method == "spline":
        from scipy.interpolate import CubicSpline

        if known.sum() < 4:
            raise NotApplicable(f"cubic spline needs at least 4 known points, got {int(known.sum())}")
        spline = CubicSpline(positions[known], values[known], extrapolate=True)
        filled[gaps] = spline(positions[gaps])
        return [float(v) for v in filled]

    if method == "polynomial":
        if degree < 1:
            raise NotApplicable(f"polynomial degree must be at least 1, got {degree}")
        order = min(degree, int(known.sum()) - 1)
        coefficients = np.polyfit(positions[known], values[known], order)
        filled[gaps] = np.polyval(coefficients, positions[gaps])
        return [float(v) for v in filled]

    # Low-rank: seed the gaps linearly, then iterate embed / truncate / re-impute.
    filled[gaps] = np.interp(positions[gaps], positions[known], values[known])
    for _ in range(10):
        reconstructed = _low_rank_series(filled, rank, None, energy)
        previous = filled[gaps]
        filled[gaps] = reconstructed[gaps]
        if np.max(np.abs(filled[gaps] - previous)) < 1e-9:
            break
    return [float(v) for v in filled]


def remove_outliers(
    x: Sequence[float],
    method: OutlierMethod = "hampel",
    *,
    threshold: float = 3.0,
    window: int = 7,
    contiguous: bool = False,
) -> tuple[Series, list[int]]:
    """Return the series with outliers replaced by interpolation, plus the indices replaced.

    Replacing rather than dropping keeps the series aligned with its timestamps. Set
    ``contiguous`` to also catch runs of adjacent outliers, which point estimators miss.
    """
    values = _as_array(x, minimum=3, what="remove_outliers")
    if method not in OUTLIER_METHODS:
        raise NotApplicable(f"unknown outlier method {method!r}; expected one of {list(OUTLIER_METHODS)}")
    if threshold <= 0:
        raise NotApplicable(f"threshold must be positive, got {threshold}")

    if method == "zscore":
        spread = float(values.std())
        flagged = np.abs(values - values.mean()) > threshold * spread if spread > 1e-12 \
            else np.zeros(values.size, dtype=bool)
    elif method == "iqr":
        first, third = np.percentile(values, [25, 75])
        spread = third - first
        flagged = (values < first - threshold * spread) | (values > third + threshold * spread) \
            if spread > 1e-12 else np.zeros(values.size, dtype=bool)
    else:
        span = _odd_window(max(window, 3), values.size, "remove_outliers")
        median, scale = _rolling_median_and_scale(values, span)
        flagged = np.abs(values - median) > threshold * _fallback_scale(scale, values)

    if contiguous:
        # A run of outliers drags a rolling median with it, so widen each flag to its neighbours
        # that also sit far from the overall level.
        level = float(np.median(values))
        spread = float(np.median(np.abs(values - level))) * _MAD_TO_SIGMA
        if spread > 1e-12:
            far = np.abs(values - level) > threshold * spread
            flagged = flagged | (far & (np.roll(flagged, 1) | np.roll(flagged, -1)))

    indices = [int(i) for i in np.flatnonzero(flagged)]
    if not indices or len(indices) == values.size:
        return [float(v) for v in values], indices
    gapped = values.copy()
    gapped[flagged] = np.nan
    return interpolate_missing(gapped, "linear"), indices


# Decomposition

TrendMethod = Literal["least_squares", "spline", "hodrick_prescott", "moving_average"]
SeasonalMethod = Literal["seasonal_means", "dft", "stl"]

TREND_METHODS: tuple[str, ...] = ("least_squares", "spline", "hodrick_prescott", "moving_average")
SEASONAL_METHODS: tuple[str, ...] = ("seasonal_means", "dft", "stl")


class Decomposition:
    """A series split into trend, seasonal and residual parts, each the original length."""

    __slots__ = ("trend", "seasonal", "residual", "model")

    def __init__(self, trend: Series, seasonal: Series, residual: Series, model: str) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.residual = residual
        self.model = model

    def recombine(self) -> Series:
        """Put the parts back together, additively or multiplicatively as fitted."""
        if self.model == "multiplicative":
            return [t * s * r for t, s, r in zip(self.trend, self.seasonal, self.residual)]
        return [t + s + r for t, s, r in zip(self.trend, self.seasonal, self.residual)]

    def __repr__(self) -> str:
        return f"Decomposition(model={self.model!r}, length={len(self.trend)})"


def detrend(
    x: Sequence[float],
    method: TrendMethod = "least_squares",
    *,
    degree: int = 1,
    window: int | None = None,
    smoothing: float | None = None,
    lamb: float = 1600.0,
) -> tuple[Series, Series]:
    """Return (trend, detrended); trend + detrended reconstructs the input exactly."""
    values = _as_array(x, minimum=3, what="detrend")
    if method not in TREND_METHODS:
        raise NotApplicable(f"unknown trend method {method!r}; expected one of {list(TREND_METHODS)}")
    positions = np.arange(values.size, dtype=float)

    if method == "least_squares":
        if degree < 0:
            raise NotApplicable(f"degree must be non-negative, got {degree}")
        if degree >= values.size:
            raise NotApplicable(f"degree {degree} needs more than {degree} points, got {values.size}")
        trend = np.polyval(np.polyfit(positions, values, degree), positions)
    elif method == "spline":
        from scipy.interpolate import UnivariateSpline

        if values.size < 5:
            raise NotApplicable(f"spline detrending needs at least 5 points, got {values.size}")
        factor = smoothing if smoothing is not None else values.size * float(np.var(values))
        trend = UnivariateSpline(positions, values, s=factor)(positions)
    elif method == "hodrick_prescott":
        from statsmodels.tsa.filters.hp_filter import hpfilter

        if values.size < 4:
            raise NotApplicable(f"Hodrick-Prescott needs at least 4 points, got {values.size}")
        _cycle, trend = hpfilter(values, lamb=lamb)
        trend = np.asarray(trend, dtype=float)
    else:
        span = _odd_window(window or max(3, values.size // 10), values.size, "detrend")
        trend = np.asarray(denoise(values, "moving_average", window=span), dtype=float)

    return [float(v) for v in trend], [float(v) for v in values - trend]


def deseasonalize(
    x: Sequence[float],
    period: int,
    method: SeasonalMethod = "seasonal_means",
    *,
    harmonics: int = 3,
) -> tuple[Series, Series]:
    """Return (seasonal, deseasonalized); the seasonal part averages to zero."""
    values = _as_array(x, minimum=4, what="deseasonalize")
    if method not in SEASONAL_METHODS:
        raise NotApplicable(
            f"unknown seasonal method {method!r}; expected one of {list(SEASONAL_METHODS)}"
        )
    if period < 2:
        raise NotApplicable(f"period must be at least 2, got {period}")
    if values.size < 2 * period:
        raise NotApplicable(f"needs at least {2 * period} points for period {period}, got {values.size}")

    if method == "seasonal_means":
        phases = np.arange(values.size) % period
        centered = values - values.mean()
        profile = np.array([centered[phases == phase].mean() for phase in range(period)])
        profile -= profile.mean()
        seasonal = profile[phases]
    elif method == "dft":
        if harmonics < 1:
            raise NotApplicable(f"harmonics must be at least 1, got {harmonics}")
        coefficients = np.fft.rfft(values - values.mean())
        keep = np.zeros_like(coefficients)
        for multiple in range(1, harmonics + 1):
            bin_index = int(round(multiple * values.size / period))
            if 0 < bin_index < coefficients.size:
                keep[bin_index] = coefficients[bin_index]
        seasonal = np.fft.irfft(keep, n=values.size)
        seasonal -= seasonal.mean()
    else:
        from statsmodels.tsa.seasonal import STL

        seasonal = np.asarray(STL(values, period=period).fit().seasonal, dtype=float)
        seasonal = seasonal - seasonal.mean()

    return [float(v) for v in seasonal], [float(v) for v in values - seasonal]


def decompose(
    x: Sequence[float],
    period: int,
    model: Literal["additive", "multiplicative"] = "additive",
    *,
    trend_method: TrendMethod = "moving_average",
    seasonal_method: SeasonalMethod = "seasonal_means",
) -> Decomposition:
    """Split a series into trend, seasonal and residual parts that recombine to the input."""
    values = _as_array(x, minimum=4, what="decompose")
    if model not in ("additive", "multiplicative"):
        raise NotApplicable(f"unknown model {model!r}; expected 'additive' or 'multiplicative'")
    if model == "multiplicative" and np.any(values <= 0):
        raise NotApplicable("multiplicative decomposition needs strictly positive values")

    working = np.log(values) if model == "multiplicative" else values
    trend, detrended = detrend(working, trend_method, window=period if period > 1 else None)
    seasonal, residual = deseasonalize(detrended, period, seasonal_method)

    if model == "multiplicative":
        return Decomposition(
            [float(np.exp(v)) for v in trend],
            [float(np.exp(v)) for v in seasonal],
            [float(np.exp(v)) for v in residual],
            model,
        )
    return Decomposition(trend, seasonal, residual, model)


# Models
#
# Every fit_* returns a Model. Extrapolation lives behind Model.extrapolate(horizon), so a
# method can never train a direct h-step model and then apply it recursively by accident.

ARMethod = Literal["yule_walker", "burg", "levinson", "ols"]
StateSpaceKind = Literal["local_level", "local_linear_trend", "basic_structural"]
SinusoidalMethod = Literal["periodogram", "prony", "music"]

AR_METHODS: tuple[str, ...] = ("yule_walker", "burg", "levinson", "ols")
STATE_SPACE_KINDS: tuple[str, ...] = ("local_level", "local_linear_trend", "basic_structural")
SINUSOIDAL_METHODS: tuple[str, ...] = ("periodogram", "prony", "music")


class _Fitted:
    """Concrete Model: holds the fitted values and a callable that produces the future."""

    __slots__ = ("name", "params", "_values", "_fitted", "_forward")

    def __init__(self, name, values, fitted, forward, params):  # type: ignore[no-untyped-def]
        self.name = name
        self.params = params
        self._values = np.asarray(values, dtype=float)
        self._fitted = np.asarray(fitted, dtype=float)
        if self._fitted.size != self._values.size:
            raise NotApplicable(
                f"{name} fitted {self._fitted.size} values for {self._values.size} points"
            )
        self._fitted = np.where(np.isfinite(self._fitted), self._fitted, float(self._values.mean()))
        self._forward = forward

    def fitted(self) -> Series:
        return [float(v) for v in self._fitted]

    def residuals(self) -> Series:
        return [float(v) for v in self._values - self._fitted]

    def extrapolate(self, horizon: int) -> Series:
        if horizon < 1:
            raise NotApplicable(f"horizon must be at least 1, got {horizon}")
        future = np.asarray(self._forward(horizon), dtype=float).ravel()
        if future.size != horizon:
            raise NotApplicable(f"{self.name} produced {future.size} values for horizon {horizon}")
        if not np.all(np.isfinite(future)):
            # A diverging fit is a statement that this model does not suit the series, not a
            # crash; report it as inapplicable rather than emitting infinities downstream.
            raise NotApplicable(f"{self.name} extrapolation diverged over horizon {horizon}")
        return [float(v) for v in future]

    def __repr__(self) -> str:
        return f"Model({self.name!r}, points={self._values.size})"


def _levinson_coefficients(values: np.ndarray, order: int) -> np.ndarray:
    """AR coefficients by the Durbin-Levinson recursion on the autocorrelation."""
    rho = np.asarray(acf(values, order), dtype=float)
    phi = np.zeros((order + 1, order + 1))
    for k in range(1, order + 1):
        numerator = rho[k] - sum(phi[k - 1, j] * rho[k - j] for j in range(1, k))
        denominator = 1.0 - sum(phi[k - 1, j] * rho[j] for j in range(1, k))
        phi[k, k] = 0.0 if abs(denominator) < 1e-12 else numerator / denominator
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
    return phi[order, 1:order + 1]


def _ar_coefficients(values: np.ndarray, order: int, method: str) -> np.ndarray:
    centered = values - values.mean()
    if method == "levinson":
        return _levinson_coefficients(values, order)
    if method == "yule_walker":
        from statsmodels.regression.linear_model import yule_walker

        coefficients, _sigma = yule_walker(centered, order=order, method="mle")
        return np.asarray(coefficients, dtype=float)
    if method == "burg":
        from statsmodels.regression.linear_model import burg

        coefficients, _sigma = burg(centered, order=order)
        return np.asarray(coefficients, dtype=float)
    design = np.column_stack([centered[order - lag - 1:-lag - 1] for lag in range(order)])
    target = centered[order:]
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(solution, dtype=float)


def _ar_one_step(values: np.ndarray, coefficients: np.ndarray, mean: float) -> np.ndarray:
    """In-sample one-step-ahead predictions; the first `order` points fall back to the mean."""
    order = coefficients.size
    fitted = np.full(values.size, mean)
    for t in range(order, values.size):
        window = values[t - order:t][::-1] - mean
        fitted[t] = mean + float(np.dot(coefficients, window))
    return fitted


def fit_ar(
    x: Sequence[float],
    order: int | None = None,
    *,
    method: ARMethod = "yule_walker",
    order_criterion: Literal["aic", "bic"] | None = "bic",
    max_order: int = 20,
) -> Model:
    """Fit an autoregressive model, choosing the order by AIC or BIC when it is not given."""
    values = _as_array(x, minimum=4, what="fit_ar")
    if method not in AR_METHODS:
        raise NotApplicable(f"unknown AR method {method!r}; expected one of {list(AR_METHODS)}")

    ceiling = max(1, min(max_order, values.size // 3))
    if order is None:
        if order_criterion not in ("aic", "bic"):
            raise NotApplicable("fit_ar needs either an order or an order_criterion of 'aic'/'bic'")
        best_order, best_score = 1, math.inf
        for candidate in range(1, ceiling + 1):
            coefficients = _ar_coefficients(values, candidate, method)
            residual = values - _ar_one_step(values, coefficients, float(values.mean()))
            variance = float(np.mean(residual[candidate:] ** 2))
            if variance <= 1e-24:
                score = -math.inf
            else:
                penalty = 2.0 * candidate if order_criterion == "aic" else candidate * math.log(values.size)
                score = values.size * math.log(variance) + penalty
            if score < best_score:
                best_order, best_score = candidate, score
        order = best_order
    if order < 1:
        raise NotApplicable(f"AR order must be at least 1, got {order}")
    if order >= values.size:
        raise NotApplicable(f"AR order {order} needs more than {order} points, got {values.size}")

    mean = float(values.mean())
    coefficients = _ar_coefficients(values, order, method)
    fitted = _ar_one_step(values, coefficients, mean)

    def forward(horizon: int) -> np.ndarray:
        history = list(values)
        future = []
        for _ in range(horizon):
            window = np.asarray(history[-order:][::-1]) - mean
            step = mean + float(np.dot(coefficients, window))
            future.append(step)
            history.append(step)
        return np.asarray(future)

    params = {"order": float(order), "mean": mean}
    params.update({f"phi_{i + 1}": float(c) for i, c in enumerate(coefficients)})
    return _Fitted(f"ar[{method}]", values, fitted, forward, params)


def fit_arma(x: Sequence[float], p: int = 1, q: int = 0, d: int = 0) -> Model:
    """Fit an ARIMA(p, d, q) by maximum likelihood."""
    values = _as_array(x, minimum=8, what="fit_arma")
    if min(p, q, d) < 0 or p + q < 1:
        raise NotApplicable(f"invalid ARIMA order ({p}, {d}, {q})")
    if values.size <= p + q + d + 2:
        raise NotApplicable(f"ARIMA({p},{d},{q}) needs more than {p + q + d + 2} points, got {values.size}")

    from statsmodels.tsa.arima.model import ARIMA

    result = ARIMA(values, order=(p, d, q)).fit()
    fitted = np.asarray(result.fittedvalues, dtype=float)
    return _Fitted(
        f"arima({p},{d},{q})", values, fitted,
        lambda horizon: np.asarray(result.forecast(steps=horizon), dtype=float),
        {"p": float(p), "d": float(d), "q": float(q), "aic": float(result.aic)},
    )


def fit_state_space(
    x: Sequence[float],
    kind: StateSpaceKind = "local_level",
    period: int | None = None,
) -> Model:
    """Fit an unobserved-components model by Kalman filtering."""
    values = _as_array(x, minimum=8, what="fit_state_space")
    if kind not in STATE_SPACE_KINDS:
        raise NotApplicable(f"unknown state-space kind {kind!r}; expected one of {list(STATE_SPACE_KINDS)}")
    if kind == "basic_structural":
        if period is None or period < 2:
            raise NotApplicable("basic_structural needs a seasonal period of at least 2")
        if values.size < 2 * period:
            raise NotApplicable(f"needs at least {2 * period} points for period {period}, got {values.size}")

    from statsmodels.tsa.statespace.structural import UnobservedComponents

    level = "local level" if kind == "local_level" else "local linear trend"
    seasonal = period if kind == "basic_structural" else None
    result = UnobservedComponents(values, level=level, seasonal=seasonal).fit(disp=False)
    fitted = np.asarray(result.fittedvalues, dtype=float)
    return _Fitted(
        f"state_space[{kind}]", values, fitted,
        lambda horizon: np.asarray(result.forecast(steps=horizon), dtype=float),
        {"aic": float(result.aic), "period": float(period or 0)},
    )


def fit_trend_seasonal(x: Sequence[float], period: int, *, trend_degree: int = 1) -> Model:
    """Fit a polynomial trend plus a repeating seasonal profile."""
    values = _as_array(x, minimum=4, what="fit_trend_seasonal")
    if period < 1:
        raise NotApplicable(f"period must be at least 1, got {period}")
    if trend_degree < 0:
        raise NotApplicable(f"trend_degree must be non-negative, got {trend_degree}")
    if values.size <= trend_degree:
        raise NotApplicable(f"degree {trend_degree} needs more than {trend_degree} points")

    positions = np.arange(values.size, dtype=float)
    coefficients = np.polyfit(positions, values, trend_degree)
    trend = np.polyval(coefficients, positions)
    phases = np.arange(values.size) % period
    detrended = values - trend
    profile = np.array([detrended[phases == phase].mean() for phase in range(period)])
    profile -= profile.mean()
    fitted = trend + profile[phases]

    def forward(horizon: int) -> np.ndarray:
        future_positions = np.arange(values.size, values.size + horizon, dtype=float)
        future_phases = np.arange(values.size, values.size + horizon) % period
        return np.polyval(coefficients, future_positions) + profile[future_phases]

    return _Fitted(
        "trend_seasonal", values, fitted, forward,
        {"period": float(period), "trend_degree": float(trend_degree)},
    )


def _music_frequencies(values: np.ndarray, n_components: int) -> np.ndarray:
    """Frequencies from the noise subspace of the autocorrelation matrix (MUSIC)."""
    width = min(max(2 * n_components + 1, 8), values.size // 2)
    matrix = _hankel(values - values.mean(), width)
    covariance = (matrix.T @ matrix) / matrix.shape[0]
    _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    noise = eigenvectors[:, : max(1, width - 2 * n_components)]
    grid = np.linspace(1e-4, 0.5 - 1e-4, 2048)
    steering = np.exp(-2j * np.pi * np.outer(grid, np.arange(width)))
    spectrum_values = 1.0 / (np.abs(steering @ noise) ** 2).sum(axis=1)
    peaks = [
        index for index in range(1, grid.size - 1)
        if spectrum_values[index] > spectrum_values[index - 1]
        and spectrum_values[index] >= spectrum_values[index + 1]
    ]
    peaks.sort(key=lambda index: spectrum_values[index], reverse=True)
    return grid[peaks[:n_components]] if peaks else np.array([])


def _least_squares_sinusoids(
    values: np.ndarray, frequencies: np.ndarray
) -> tuple[np.ndarray, float]:
    positions = np.arange(values.size, dtype=float)
    columns = [np.ones(values.size)]
    for frequency in frequencies:
        columns.append(np.cos(2 * np.pi * frequency * positions))
        columns.append(np.sin(2 * np.pi * frequency * positions))
    design = np.column_stack(columns)
    solution, *_ = np.linalg.lstsq(design, values, rcond=None)
    return solution, float(design.shape[1])


def fit_sinusoidal(
    x: Sequence[float],
    n_components: int = 3,
    method: SinusoidalMethod = "periodogram",
) -> Model:
    """Fit a sum of sinusoids; ``prony`` also allows each component to grow or decay."""
    values = _as_array(x, minimum=8, what="fit_sinusoidal")
    if method not in SINUSOIDAL_METHODS:
        raise NotApplicable(
            f"unknown sinusoidal method {method!r}; expected one of {list(SINUSOIDAL_METHODS)}"
        )
    if n_components < 1:
        raise NotApplicable(f"n_components must be at least 1, got {n_components}")

    if method == "prony":
        return _fit_prony(values, n_components)

    if method == "music":
        frequencies = _music_frequencies(values, n_components)
        if frequencies.size == 0:
            raise NotApplicable("MUSIC found no spectral peaks in this series")
    else:
        frequencies = np.array([f for f, _a, _p in dominant_frequencies(values, n_components)])

    solution, _width = _least_squares_sinusoids(values, frequencies)
    positions = np.arange(values.size, dtype=float)

    def evaluate(at: np.ndarray) -> np.ndarray:
        total = np.full(at.size, solution[0])
        for index, frequency in enumerate(frequencies):
            total = total + solution[1 + 2 * index] * np.cos(2 * np.pi * frequency * at)
            total = total + solution[2 + 2 * index] * np.sin(2 * np.pi * frequency * at)
        return total

    params = {"n_components": float(frequencies.size), "offset": float(solution[0])}
    params.update({f"freq_{i + 1}": float(f) for i, f in enumerate(frequencies)})
    return _Fitted(
        f"sinusoidal[{method}]", values, evaluate(positions),
        lambda horizon: evaluate(np.arange(values.size, values.size + horizon, dtype=float)),
        params,
    )


def _fit_prony(values: np.ndarray, n_components: int) -> Model:
    """Prony: damped complex exponentials from a linear recurrence, then amplitudes by least squares."""
    order = 2 * n_components
    if values.size < 2 * order + 2:
        raise NotApplicable(f"Prony with {n_components} components needs at least {2 * order + 2} points")
    centered = values - values.mean()
    design = np.column_stack([centered[order - lag - 1:-lag - 1] for lag in range(order)])
    coefficients, *_ = np.linalg.lstsq(design, centered[order:], rcond=None)
    roots = np.roots(np.concatenate(([1.0], -coefficients)))
    if roots.size == 0:
        raise NotApplicable("Prony found no modes in this series")
    # Reflect explosive modes onto the unit circle so the extrapolation stays bounded.
    magnitudes = np.abs(roots)
    roots = np.where(magnitudes > 1.0, roots / np.where(magnitudes > 0, magnitudes, 1.0), roots)

    positions = np.arange(values.size, dtype=float)
    design = np.power(roots[None, :], positions[:, None])
    amplitudes, *_ = np.linalg.lstsq(design, centered.astype(complex), rcond=None)
    mean = float(values.mean())

    def evaluate(at: np.ndarray) -> np.ndarray:
        return mean + np.real(np.power(roots[None, :], at[:, None]) @ amplitudes)

    return _Fitted(
        "sinusoidal[prony]", values, evaluate(positions),
        lambda horizon: evaluate(np.arange(values.size, values.size + horizon, dtype=float)),
        {"n_components": float(n_components), "modes": float(roots.size)},
    )


def fit_hmm(x: Sequence[float], n_states: int = 2, *, iterations: int = 50) -> Model:
    """Fit a Gaussian hidden Markov model by Baum-Welch; forecasts blend the state means."""
    values = _as_array(x, minimum=10, what="fit_hmm")
    if n_states < 2:
        raise NotApplicable(f"n_states must be at least 2, got {n_states}")
    if values.size < 3 * n_states:
        raise NotApplicable(f"{n_states} states need at least {3 * n_states} points, got {values.size}")

    size = values.size
    means = np.quantile(values, np.linspace(0.1, 0.9, n_states))
    variances = np.full(n_states, max(float(values.var()), 1e-8))
    transition = np.full((n_states, n_states), 1.0 / n_states)
    initial = np.full(n_states, 1.0 / n_states)

    posterior = np.full((size, n_states), 1.0 / n_states)
    for _ in range(iterations):
        emission = np.exp(-0.5 * (values[:, None] - means) ** 2 / variances) / np.sqrt(2 * np.pi * variances)
        emission = np.maximum(emission, 1e-300)

        alpha = np.zeros((size, n_states))
        scale = np.zeros(size)
        alpha[0] = initial * emission[0]
        scale[0] = max(alpha[0].sum(), 1e-300)
        alpha[0] /= scale[0]
        for t in range(1, size):
            alpha[t] = (alpha[t - 1] @ transition) * emission[t]
            scale[t] = max(alpha[t].sum(), 1e-300)
            alpha[t] /= scale[t]

        beta = np.zeros((size, n_states))
        beta[-1] = 1.0
        for t in range(size - 2, -1, -1):
            beta[t] = transition @ (emission[t + 1] * beta[t + 1]) / scale[t + 1]

        posterior = alpha * beta
        posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-300)

        joint = np.zeros((n_states, n_states))
        for t in range(size - 1):
            joint += (
                np.outer(alpha[t], emission[t + 1] * beta[t + 1]) * transition / scale[t + 1]
            )
        transition = joint / np.maximum(joint.sum(axis=1, keepdims=True), 1e-300)
        initial = posterior[0] / max(posterior[0].sum(), 1e-300)
        weights = np.maximum(posterior.sum(axis=0), 1e-300)
        means = (posterior * values[:, None]).sum(axis=0) / weights
        variances = np.maximum(
            (posterior * (values[:, None] - means) ** 2).sum(axis=0) / weights, 1e-8
        )

    fitted = posterior @ means
    last = posterior[-1]

    def forward(horizon: int) -> np.ndarray:
        state = last.copy()
        future = []
        for _ in range(horizon):
            state = state @ transition
            future.append(float(state @ means))
        return np.asarray(future)

    params = {"n_states": float(n_states)}
    params.update({f"mean_{i + 1}": float(m) for i, m in enumerate(means)})
    return _Fitted("hmm", values, fitted, forward, params)


# Matching

DistanceMetric = Literal["euclidean", "normalized_euclidean", "dtw", "soft_dtw"]
DISTANCE_METRICS: tuple[str, ...] = ("euclidean", "normalized_euclidean", "dtw", "soft_dtw")


def _standardize(values: np.ndarray) -> np.ndarray:
    spread = float(values.std())
    return (values - values.mean()) / (spread if spread > 1e-12 else 1.0)


def _dtw_matrix(left: np.ndarray, right: np.ndarray, radius: int | None) -> np.ndarray:
    rows, columns = left.size, right.size
    band = radius if radius is not None else max(rows, columns)
    accumulated = np.full((rows + 1, columns + 1), np.inf)
    accumulated[0, 0] = 0.0
    for i in range(1, rows + 1):
        low = max(1, i - band)
        high = min(columns, i + band)
        for j in range(low, high + 1):
            local = (left[i - 1] - right[j - 1]) ** 2
            accumulated[i, j] = local + min(
                accumulated[i - 1, j], accumulated[i, j - 1], accumulated[i - 1, j - 1]
            )
    return accumulated


def distance(
    a: Sequence[float],
    b: Sequence[float],
    metric: DistanceMetric = "euclidean",
    *,
    radius: int | None = None,
    gamma: float = 1.0,
) -> float:
    """Dissimilarity between two series; only the DTW metrics tolerate different lengths.

    ``soft_dtw`` is the divergence form, so like the others it is zero for identical series and
    never negative, unlike raw soft-DTW.
    """
    if metric not in DISTANCE_METRICS:
        raise NotApplicable(f"unknown metric {metric!r}; expected one of {list(DISTANCE_METRICS)}")
    left = _as_array(a, minimum=1, what="distance")
    right = _as_array(b, minimum=1, what="distance")

    if metric in ("euclidean", "normalized_euclidean"):
        if left.size != right.size:
            raise NotApplicable(f"{metric} needs equal lengths, got {left.size} and {right.size}")
        if metric == "normalized_euclidean":
            left, right = _standardize(left), _standardize(right)
        return float(np.linalg.norm(left - right))

    if metric == "dtw":
        if radius is not None and radius < 1:
            raise NotApplicable(f"radius must be at least 1, got {radius}")
        accumulated = _dtw_matrix(left, right, radius)
        total = accumulated[left.size, right.size]
        if not math.isfinite(total):
            raise NotApplicable(f"radius {radius} is too narrow to align {left.size} and {right.size}")
        return float(math.sqrt(total))

    if gamma <= 0:
        raise NotApplicable(f"gamma must be positive, got {gamma}")
    # Raw soft-DTW is negative and non-zero on identical inputs; the divergence form subtracts
    # the self-terms so this behaves like a distance.
    cross = _soft_dtw(left, right, gamma)
    divergence = cross - 0.5 * (_soft_dtw(left, left, gamma) + _soft_dtw(right, right, gamma))
    return float(max(divergence, 0.0))


def _soft_dtw(left: np.ndarray, right: np.ndarray, gamma: float) -> float:
    rows, columns = left.size, right.size
    accumulated = np.full((rows + 1, columns + 1), np.inf)
    accumulated[0, 0] = 0.0
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            candidates = np.array([
                accumulated[i - 1, j], accumulated[i, j - 1], accumulated[i - 1, j - 1]
            ])
            smallest = candidates.min()
            soft_min = smallest - gamma * math.log(np.sum(np.exp(-(candidates - smallest) / gamma)))
            accumulated[i, j] = (left[i - 1] - right[j - 1]) ** 2 + soft_min
    return float(accumulated[rows, columns])


def nearest_windows(
    x: Sequence[float],
    query: Sequence[float],
    k: int = 3,
    metric: DistanceMetric = "euclidean",
    *,
    exclude_tail: bool = True,
) -> list[int]:
    """Start indices of the k windows of x most similar to query, closest first.

    ``exclude_tail`` drops the window that ends at the series end, which is the query itself
    when the query is the most recent stretch of x.
    """
    values = _as_array(x, minimum=2, what="nearest_windows")
    pattern = _as_array(query, minimum=1, what="nearest_windows")
    width = pattern.size
    if width > values.size:
        raise NotApplicable(f"query of {width} points does not fit in {values.size}")
    if k < 1:
        raise NotApplicable(f"k must be at least 1, got {k}")

    last = values.size - width
    starts = range(0, last if exclude_tail else last + 1)
    scored = [(distance(values[s:s + width], pattern, metric), s) for s in starts]
    if not scored:
        raise NotApplicable("no candidate windows once the tail is excluded")
    scored.sort()
    return [start for _score, start in scored[:k]]


def matrix_profile(x: Sequence[float], window: int) -> tuple[Series, list[int]]:
    """Distance from every window to its nearest non-trivial match, and that match's index.

    Low values mark motifs (repeated shapes); high values mark discords (anomalies).
    """
    values = _as_array(x, minimum=4, what="matrix_profile")
    if window < 2 or window > values.size // 2:
        raise NotApplicable(f"window must be in [2, {values.size // 2}], got {window}")

    count = values.size - window + 1
    windows = np.lib.stride_tricks.sliding_window_view(values, window)[:count]
    standardized = np.array([_standardize(w) for w in windows])
    exclusion = max(1, window // 2)

    profile = np.full(count, np.inf)
    indices = np.zeros(count, dtype=int)
    for i in range(count):
        deltas = standardized - standardized[i]
        distances = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
        low, high = max(0, i - exclusion), min(count, i + exclusion + 1)
        distances[low:high] = np.inf
        best = int(np.argmin(distances))
        profile[i], indices[i] = distances[best], best
    return [float(v) for v in profile], [int(v) for v in indices]


def barycenter(
    windows: Sequence[Sequence[float]],
    metric: Literal["euclidean", "dtw"] = "euclidean",
    *,
    iterations: int = 10,
) -> Series:
    """An average shape for a set of equal-length windows; DTW averaging follows DBA."""
    matrix = np.asarray([[float(v) for v in window] for window in windows], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise NotApplicable(f"need at least one window of equal length, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise NotApplicable("windows must be finite")
    if metric == "euclidean":
        return [float(v) for v in matrix.mean(axis=0)]
    if metric != "dtw":
        raise NotApplicable(f"barycenter supports 'euclidean' or 'dtw', got {metric!r}")

    centre = matrix.mean(axis=0)
    width = centre.size
    for _ in range(iterations):
        totals = np.zeros(width)
        counts = np.zeros(width)
        for row in matrix:
            accumulated = _dtw_matrix(centre, row, None)
            i, j = width, row.size
            while i > 0 and j > 0:
                totals[i - 1] += row[j - 1]
                counts[i - 1] += 1.0
                step = int(np.argmin([
                    accumulated[i - 1, j - 1], accumulated[i - 1, j], accumulated[i, j - 1]
                ]))
                if step == 0:
                    i, j = i - 1, j - 1
                elif step == 1:
                    i -= 1
                else:
                    j -= 1
        updated = totals / np.maximum(counts, 1.0)
        if float(np.max(np.abs(updated - centre))) < 1e-9:
            centre = updated
            break
        centre = updated
    return [float(v) for v in centre]


# Features

FeatureGroup = Literal["statistical", "spectral", "entropy", "symbolic", "shape"]
FEATURE_GROUPS: tuple[str, ...] = ("statistical", "spectral", "entropy", "symbolic", "shape")

_SAX_BREAKPOINTS: dict[int, list[float]] = {
    2: [0.0],
    3: [-0.43, 0.43],
    4: [-0.67, 0.0, 0.67],
    5: [-0.84, -0.25, 0.25, 0.84],
    6: [-0.97, -0.43, 0.0, 0.43, 0.97],
    7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
    8: [-1.15, -0.67, -0.32, 0.0, 0.32, 0.67, 1.15],
}


def sax(x: Sequence[float], word_length: int = 8, alphabet_size: int = 4) -> str:
    """Symbolic Aggregate approXimation: standardize, average into segments, map to letters."""
    values = _as_array(x, minimum=2, what="sax")
    if word_length < 1 or word_length > values.size:
        raise NotApplicable(f"word_length must be in [1, {values.size}], got {word_length}")
    if alphabet_size not in _SAX_BREAKPOINTS:
        raise NotApplicable(f"alphabet_size must be in {sorted(_SAX_BREAKPOINTS)}, got {alphabet_size}")

    standardized = _standardize(values)
    edges = np.linspace(0, values.size, word_length + 1).astype(int)
    letters = []
    for start, end in zip(edges[:-1], edges[1:]):
        chunk = standardized[start:max(end, start + 1)]
        level = float(chunk.mean())
        index = int(np.searchsorted(_SAX_BREAKPOINTS[alphabet_size], level))
        letters.append(chr(ord("a") + index))
    return "".join(letters)


def _shannon_entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def features(
    x: Sequence[float],
    groups: Sequence[FeatureGroup] = ("statistical", "spectral", "shape"),
) -> Features:
    """Named scalar descriptors of a series, for saying which series a method suits."""
    values = _as_array(x, minimum=4, what="features")
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown:
        raise NotApplicable(f"unknown feature groups {unknown}; expected from {list(FEATURE_GROUPS)}")
    if not groups:
        raise NotApplicable("at least one feature group is required")

    result: Features = {}
    spread = float(values.std())

    if "statistical" in groups:
        centered = values - values.mean()
        result.update({
            "mean": float(values.mean()),
            "std": spread,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "median": float(np.median(values)),
            "iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
            "skewness": float(np.mean(centered ** 3) / spread ** 3) if spread > 1e-12 else 0.0,
            "kurtosis": float(np.mean(centered ** 4) / spread ** 4) if spread > 1e-12 else 0.0,
            "zero_fraction": float(np.mean(values == 0.0)),
        })

    if "spectral" in groups:
        components = spectrum(values, method="periodogram")
        frequencies = np.array([f for f, _a, _p in components])
        power = np.array([a ** 2 for _f, a, _p in components])
        total = float(power.sum())
        if total > 1e-24:
            normalized = power / total
            centroid = float(np.sum(frequencies * normalized))
            result.update({
                "spectral_centroid": centroid,
                "spectral_spread": float(math.sqrt(np.sum(((frequencies - centroid) ** 2) * normalized))),
                "spectral_entropy": _shannon_entropy(power) / math.log(power.size) if power.size > 1 else 0.0,
                "spectral_flatness": float(
                    np.exp(np.mean(np.log(power + 1e-30))) / (np.mean(power) + 1e-30)
                ),
            })
        else:
            result.update({
                "spectral_centroid": 0.0, "spectral_spread": 0.0,
                "spectral_entropy": 0.0, "spectral_flatness": 0.0,
            })

    if "entropy" in groups:
        result["approximate_entropy"] = _approximate_entropy(values)
        result["permutation_entropy"] = _permutation_entropy(values)

    if "symbolic" in groups:
        word = sax(values, word_length=min(16, values.size), alphabet_size=4)
        _letters, counts = np.unique(list(word), return_counts=True)
        result["symbolic_entropy"] = _shannon_entropy(counts.astype(float))
        result["symbolic_distinct"] = float(len(set(word)))

    if "shape" in groups:
        differences = np.diff(values)
        result.update({
            "trend_slope": float(np.polyfit(np.arange(values.size), values, 1)[0]),
            "lag1_autocorrelation": float(acf(values, 1)[1]),
            "turning_point_rate": float(
                np.mean(np.sign(differences[1:]) != np.sign(differences[:-1]))
            ) if differences.size > 1 else 0.0,
            "mean_absolute_change": float(np.mean(np.abs(differences))),
            "flat_fraction": float(np.mean(np.abs(differences) < 1e-12)),
        })
    return result


def _approximate_entropy(values: np.ndarray, dimension: int = 2, tolerance: float | None = None) -> float:
    size = values.size
    if size < dimension + 2:
        return 0.0
    threshold = tolerance if tolerance is not None else 0.2 * float(values.std())
    if threshold <= 1e-12:
        return 0.0

    def correlation(length: int) -> float:
        count = size - length + 1
        blocks = np.lib.stride_tricks.sliding_window_view(values, length)[:count]
        totals = [
            float(np.mean(np.max(np.abs(blocks - block), axis=1) <= threshold)) for block in blocks
        ]
        return float(np.mean(np.log(np.maximum(totals, 1e-300))))

    return abs(correlation(dimension) - correlation(dimension + 1))


def _permutation_entropy(values: np.ndarray, order: int = 3) -> float:
    if values.size < order + 1:
        return 0.0
    blocks = np.lib.stride_tricks.sliding_window_view(values, order)
    patterns = [tuple(np.argsort(block)) for block in blocks]
    _unique, counts = np.unique(np.array(patterns), axis=0, return_counts=True)
    return _shannon_entropy(counts.astype(float)) / math.log(math.factorial(order))
