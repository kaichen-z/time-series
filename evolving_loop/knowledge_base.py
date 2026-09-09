"""Curated, source-backed time-series knowledge for Setting 2."""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evolving_loop.data import Task


@dataclass(frozen=True)
class TrendEvidence:
    window_lengths: tuple[int, ...] = ()
    signed_horizon_effects: tuple[float, ...] = ()
    median_signed_horizon_effect: float = 0.0
    sign_agreement: float = 0.0
    relative_effect_dispersion: float = 0.0
    median_detrending_gain: float = 0.0
    assessment: str = "insufficient"


@dataclass(frozen=True)
class SeasonalLagEvidence:
    lag: int = 0
    complete_cycles: int = 0
    supporting_windows: int = 0
    detrended_acf_median: float = 0.0
    detrended_acf_min: float = 0.0
    local_peak_margin_min: float = 0.0
    phase_error_ratio_to_level: float = 0.0
    phase_replay_origin_count: int = 0
    cycle_amplitude_dispersion: float = 0.0
    support_mode: str = "none"
    assessment: str = "insufficient"


@dataclass(frozen=True)
class RegimeOODEvidence:
    candidate_suffix_length: int | None = None
    signed_shift_over_scale: float = 0.0
    supporting_width_fraction: float = 0.0
    suffix_persistence_fraction: float = 0.0
    suffix_dispersion_to_shift: float = 0.0
    boundary_jump_to_shift: float = 0.0
    seasonal_phase_explained: bool = False
    terminal_ood_width: int | None = None
    assessment: str = "insufficient"


@dataclass(frozen=True)
class DiagnosticProfile:
    history_length: int
    horizon: int
    horizon_ratio: float
    seasonal_period: int | None
    lag1_autocorrelation: float
    seasonal_autocorrelation: float | None
    trend_effect_over_horizon: float
    recent_level_shift: float
    recent_trend_change: float
    outlier_fraction: float
    zero_fraction: float
    variance_ratio_recent_to_early: float
    candidate_lags: tuple[int, ...]
    tags: tuple[str, ...]
    # Appended defaults preserve construction by every legacy positional/keyword caller.
    profile_version: str = "legacy-v3"
    trend_evidence: TrendEvidence = field(default_factory=TrendEvidence)
    seasonality_evidence: tuple[SeasonalLagEvidence, ...] = ()
    regime_ood_evidence: RegimeOODEvidence = field(default_factory=RegimeOODEvidence)
    risk_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeEntry:
    entry_id: str
    category: str
    title: str
    principle: str
    use_when: str
    avoid_when: str
    implementation: str
    applicability: tuple[str, ...]
    source_ids: tuple[str, ...]
    priority: int


@dataclass(frozen=True)
class KnowledgeSelection:
    profile: DiagnosticProfile
    entries: tuple[KnowledgeEntry, ...]

    @property
    def entry_ids(self) -> tuple[str, ...]:
        return tuple(item.entry_id for item in self.entries)

    def prompt_text(
        self, sources: dict[str, dict], *, include_profile: bool = True
    ) -> str:
        blocks = [
            "External time-series domain knowledge selected from the curated Setting 2 library.",
            "Treat every entry as a falsifiable prior, not as a command. Cite entry IDs in each hypothesis.",
        ]
        if include_profile:
            profile = json.dumps(asdict(self.profile), ensure_ascii=False)
            blocks.append(f"Deterministic numeric diagnostic profile: {profile}")
        else:
            blocks.append(
                "Use the numeric_evidence already supplied in the task payload; it is not repeated here."
            )
        for item in self.entries:
            citations = "; ".join(
                f"{sources[source_id]['citation']} ({sources[source_id]['url']})"
                for source_id in item.source_ids
            )
            blocks.append(
                f"[{item.entry_id}] {item.title}\n"
                f"Principle: {item.principle}\n"
                f"Use when: {item.use_when}\n"
                f"Avoid when: {item.avoid_when}\n"
                f"Executable guidance: {item.implementation}\n"
                f"Sources: {citations}"
            )
        return "\n\n".join(blocks)


class TimeSeriesKnowledgeBase:
    """Load the full library and retrieve a small diagnostic-matched subset."""

    def __init__(self, path: str | Path | None = None) -> None:
        source = Path(path) if path else Path(__file__).with_name("knowledge") / "time_series.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        self.version = str(payload["version"])
        self.sources = {str(item["source_id"]): item for item in payload["sources"]}
        self.entries = tuple(
            KnowledgeEntry(
                entry_id=str(item["entry_id"]),
                category=str(item["category"]),
                title=str(item["title"]),
                principle=str(item["principle"]),
                use_when=str(item["use_when"]),
                avoid_when=str(item["avoid_when"]),
                implementation=str(item["implementation"]),
                applicability=tuple(str(tag) for tag in item["applicability"]),
                source_ids=tuple(str(source_id) for source_id in item["source_ids"]),
                priority=int(item.get("priority", 50)),
            )
            for item in payload["entries"]
        )
        self._validate()

    def _validate(self) -> None:
        entry_ids = [item.entry_id for item in self.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("time-series knowledge entry IDs must be unique")
        missing = {
            source_id
            for item in self.entries
            for source_id in item.source_ids
            if source_id not in self.sources
        }
        if missing:
            raise ValueError(f"unknown knowledge sources: {sorted(missing)}")

    def retrieve(
        self,
        task: Task,
        *,
        limit: int = 10,
        include_tsfm: bool = False,
        profile: DiagnosticProfile | None = None,
    ) -> KnowledgeSelection:
        profile = profile or diagnose(task)
        tags = set(profile.tags)
        if include_tsfm:
            tags.add("tsfm_available")
        ranked = sorted(
            self.entries,
            key=lambda item: (
                -(item.priority + 20 * len(tags.intersection(item.applicability))),
                item.entry_id,
            ),
        )
        selected: list[KnowledgeEntry] = []
        category_counts: dict[str, int] = {}
        for item in ranked:
            if item.category in {"tsfm", "neural_prior"} and not include_tsfm:
                continue
            if not tags.intersection(item.applicability):
                continue
            if category_counts.get(item.category, 0) >= 2:
                continue
            selected.append(item)
            category_counts[item.category] = category_counts.get(item.category, 0) + 1
            if len(selected) == limit:
                break
        if len(selected) < limit:
            for item in ranked:
                if item.category in {"tsfm", "neural_prior"} and not include_tsfm:
                    continue
                if item in selected or not tags.intersection(item.applicability):
                    continue
                selected.append(item)
                if len(selected) == limit:
                    break
        return KnowledgeSelection(profile=profile, entries=tuple(selected))


def diagnose(task: Task) -> DiagnosticProfile:
    values = _interpolate_nonfinite(tuple(float(value) for value in task.history_values))
    n = len(values)
    horizon = task.prediction_length
    period = _positive_int(task.seasonal_period)
    median = statistics.median(values) if values else 0.0
    mad = statistics.median(abs(value - median) for value in values) if values else 0.0
    scale = max(1.4826 * mad, _standard_deviation(values), 1e-9)
    full_slope = _linear_slope(values)
    window = max(4, min(n // 3, max(horizon, period or 0, 8)))
    recent = values[-window:]
    previous = values[-2 * window : -window]
    recent_slope = _linear_slope(recent)
    previous_slope = _linear_slope(previous) if len(previous) >= 3 else full_slope
    shift = (
        abs(statistics.median(recent) - statistics.median(previous)) / scale
        if previous and recent
        else 0.0
    )
    trend_change = abs(recent_slope - previous_slope) * max(1, horizon) / scale
    trend_effect = abs(recent_slope) * max(1, horizon) / scale
    lag1 = _autocorrelation(values, 1)
    seasonal_acf = _autocorrelation(values, period) if period and n > period else None
    diff = tuple(values[index] - values[index - 1] for index in range(1, n))
    diff_lag1 = _autocorrelation(diff, 1)
    split = max(2, len(diff) // 3)
    early_scale = _standard_deviation(diff[:split])
    recent_scale = _standard_deviation(diff[-split:])
    variance_ratio = recent_scale / max(early_scale, 1e-9)
    outliers = sum(abs(value - median) > 4.5 * scale for value in values)
    zero_fraction = sum(abs(value) <= 1e-12 for value in values) / max(n, 1)
    max_lag = min(60, max(1, n // 3))
    lag_scores = sorted(
        ((_autocorrelation(values, lag), lag) for lag in range(2, max_lag + 1)),
        reverse=True,
    )
    candidate_lags = []
    if period:
        candidate_lags.append(period)
    for score, lag in lag_scores:
        if score < 0.3:
            break
        if any(_same_cycle_family(lag, other) for other in candidate_lags):
            continue
        candidate_lags.append(lag)
        if len(candidate_lags) >= 4:
            break

    tags = {"always"}
    ratio = horizon / max(n, 1)
    if ratio >= 0.5:
        tags.add("long_horizon")
    if ratio >= 0.8:
        tags.add("very_long_horizon")
    if ratio <= 0.2:
        tags.add("short_horizon")
    if n >= max(128, 4 * max(horizon, 1)):
        tags.add("long_history")
        tags.add("enough_hindcasts")
    if n < 64 or (period and n < 3 * period):
        tags.add("short_history")
    if period:
        tags.add("declared_seasonality")
        tags.add("seasonality_supported" if n >= 2 * period and (seasonal_acf or 0) >= 0.2 else "weak_seasonal_evidence")
        if horizon >= period:
            tags.add("horizon_exceeds_season")
    if lag1 >= 0.65:
        tags.add("high_persistence")
    if lag1 >= 0.85 and abs(diff_lag1) < 0.2:
        tags.add("possible_random_walk")
    if abs(lag1) >= 0.25:
        tags.add("autocorrelated")
    if trend_effect >= 0.75:
        tags.add("trend")
    if trend_effect >= 2.0:
        tags.add("strong_trend")
    if shift >= 1.25:
        tags.add("recent_level_shift")
    if trend_change >= 1.0:
        tags.add("recent_trend_change")
    if outliers / max(n, 1) >= 0.01:
        tags.add("outliers")
    if variance_ratio >= 1.8 or variance_ratio <= 0.55:
        tags.add("heteroscedastic")
    if diff and _standard_deviation(diff) >= 0.9 * max(_standard_deviation(values), 1e-9):
        tags.add("volatile")
    if zero_fraction >= 0.25 and values and min(values) >= 0:
        tags.add("intermittent")
    if values and min(values) >= 0:
        tags.add("nonnegative")
        if all(abs(value - round(value)) <= 1e-6 for value in values):
            tags.add("count_like")
        positives = [value for value in values if value > 0]
        if positives and max(positives) / max(min(positives), 1e-9) >= 20:
            tags.add("high_dynamic_range")
    if values and min(values) >= 0 and max(values) <= 100:
        tags.add("bounded_0_100")
    if lag_scores and lag_scores[0][0] >= 0.45 and (not period or lag_scores[0][1] != period):
        tags.add("empirical_cycle")
    multiple_cycles = False
    if period and n >= 3 * period:
        residual = _seasonal_phase_residual(values, period)
        residual_slope = _linear_slope(residual)
        residual = tuple(
            value - residual_slope * index for index, value in enumerate(residual)
        )
        residual_by_lag = {
            lag: _autocorrelation(residual, lag) for lag in range(2, max_lag + 1)
        }
        residual_scores = sorted(
            (
                (score, lag)
                for lag, score in residual_by_lag.items()
                if score >= residual_by_lag.get(lag - 1, -1.0)
                and score >= residual_by_lag.get(lag + 1, -1.0)
            ),
            reverse=True,
        )
        for score, lag in residual_scores:
            if score < 0.45:
                break
            if n >= 4 * lag and not _same_cycle_family(lag, period):
                multiple_cycles = True
                if lag not in candidate_lags:
                    candidate_lags.append(lag)
                break
    elif not period:
        independent_cycles: list[int] = []
        for lag in sorted(candidate_lags):
            if n < 3 * lag:
                continue
            if any(_same_cycle_family(lag, other) for other in independent_cycles):
                continue
            independent_cycles.append(lag)
        multiple_cycles = len(independent_cycles) >= 2
    if multiple_cycles:
        tags.add("multiple_cycles")
    if values and _standard_deviation(values) <= 0.02 * max(abs(median), 1.0):
        tags.add("nearly_constant")
    if abs(lag1) < 0.2 and trend_effect < 0.75:
        tags.add("weak_structure")
    if abs(lag1) < 0.5 and trend_effect < 0.75:
        tags.add("mean_reverting")

    normalized_values, finite_count = _normalized_diagnostic_values(
        tuple(task.history_values)
    )
    trend_evidence = _trend_evidence(normalized_values, horizon)
    seasonality_evidence = _seasonality_evidence(
        normalized_values, horizon, period
    )
    regime_ood_evidence = _regime_ood_evidence(
        normalized_values, horizon, seasonality_evidence
    )
    risk_flags: set[str] = set()
    if finite_count < len(task.history_values):
        risk_flags.add("nonfinite_history_interpolated_for_diagnostics")
    if trend_evidence.assessment == "ambiguous":
        risk_flags.add("trend_evidence_ambiguous")
    if any(card.assessment == "ambiguous" for card in seasonality_evidence):
        risk_flags.add("seasonality_evidence_ambiguous")
    if period and not any(
        card.lag == period and card.assessment == "supported"
        for card in seasonality_evidence
    ):
        risk_flags.add("declared_period_not_supported")
    if regime_ood_evidence.assessment == "ambiguous_terminal":
        risk_flags.add("terminal_change_ambiguous")
    if regime_ood_evidence.terminal_ood_width is not None:
        risk_flags.add("terminal_numeric_ood")
    if "nearly_constant" in tags:
        risk_flags.add("near_constant_acf_unreliable")

    return DiagnosticProfile(
        history_length=n,
        horizon=horizon,
        horizon_ratio=round(ratio, 6),
        seasonal_period=period,
        lag1_autocorrelation=round(lag1, 6),
        seasonal_autocorrelation=round(seasonal_acf, 6) if seasonal_acf is not None else None,
        trend_effect_over_horizon=round(trend_effect, 6),
        recent_level_shift=round(shift, 6),
        recent_trend_change=round(trend_change, 6),
        outlier_fraction=round(outliers / max(n, 1), 6),
        zero_fraction=round(zero_fraction, 6),
        variance_ratio_recent_to_early=round(variance_ratio, 6),
        candidate_lags=tuple(candidate_lags),
        tags=tuple(sorted(tags)),
        profile_version="v4.1-causal-phase-replay",
        trend_evidence=trend_evidence,
        seasonality_evidence=seasonality_evidence,
        regime_ood_evidence=regime_ood_evidence,
        risk_flags=tuple(sorted(risk_flags)),
    )


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 1 else None


def _interpolate_nonfinite(values: tuple[float, ...]) -> tuple[float, ...]:
    """Fill diagnostic-only gaps linearly without changing the forecast input series."""
    finite = [index for index, value in enumerate(values) if math.isfinite(value)]
    if not finite:
        return (0.0,) * len(values)
    filled = list(values)
    first, last = finite[0], finite[-1]
    filled[:first] = [values[first]] * first
    filled[last + 1 :] = [values[last]] * (len(values) - last - 1)
    for left, right in zip(finite, finite[1:]):
        if right == left + 1:
            continue
        step = (values[right] - values[left]) / (right - left)
        for index in range(left + 1, right):
            filled[index] = values[left] + step * (index - left)
    return tuple(filled)


def _linear_slope(values: tuple[float, ...]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    center = (n - 1) / 2
    denominator = sum((index - center) ** 2 for index in range(n))
    return sum((index - center) * value for index, value in enumerate(values)) / max(denominator, 1e-12)


def _autocorrelation(values: tuple[float, ...], lag: int | None) -> float:
    if lag is None or lag <= 0 or len(values) <= lag + 1:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator > 1e-12 else 0.0


def _standard_deviation(values: tuple[float, ...]) -> float:
    return statistics.pstdev(values) if len(values) >= 2 else 0.0


def _seasonal_phase_residual(values: tuple[float, ...], period: int) -> tuple[float, ...]:
    phase_levels = []
    for phase in range(period):
        observations = values[phase::period]
        phase_levels.append(statistics.median(observations) if observations else 0.0)
    return tuple(value - phase_levels[index % period] for index, value in enumerate(values))


def _same_cycle_family(left: int, right: int) -> bool:
    """Treat adjacent peaks and integer harmonics as one underlying cycle."""
    shorter, longer = sorted((left, right))
    if longer - shorter <= max(2, round(0.2 * shorter)):
        return True
    ratio = longer / shorter
    return abs(ratio - round(ratio)) <= 0.12


def _clip(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value):
        if math.isnan(value):
            return 0.0
        return upper if value > 0 else lower
    return min(upper, max(lower, value))


def _rounded(value: float, *, nonnegative: bool = False) -> float:
    lower = 0.0 if nonnegative else -20.0
    return round(_clip(value, lower, 20.0), 6)


def _mad(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def _normalized_diagnostic_values(values: tuple[object, ...]) -> tuple[tuple[float, ...], int]:
    """Return an interpolated [-1, 1] copy used only by diagnostic cards."""
    parsed: list[float] = []
    finite_values: list[float] = []
    for raw in values:
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            value = math.nan
        parsed.append(value)
        if math.isfinite(value):
            finite_values.append(value)
    if not finite_values:
        return (tuple(0.0 for _ in parsed), 0)
    normalizer = max(abs(value) for value in finite_values) or 1.0
    scaled = tuple(
        value / normalizer if math.isfinite(value) else math.nan for value in parsed
    )
    return _interpolate_nonfinite(scaled), len(finite_values)


def _robust_scale(values: tuple[float, ...]) -> float:
    differences = tuple(
        values[index] - values[index - 1] for index in range(1, len(values))
    )
    median_abs = statistics.median(abs(value) for value in values) if values else 0.0
    return max(1.4826 * _mad(differences), 0.01 * median_abs, 1e-9)


def _evenly_spaced_indices(length: int, limit: int = 64) -> tuple[int, ...]:
    if length <= 0:
        return ()
    count = min(length, limit)
    if count == 1:
        return (0,)
    return tuple(
        sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})
    )


def _theil_sen_fit(values: tuple[float, ...]) -> tuple[float, float, tuple[float, ...]]:
    if not values:
        return 0.0, 0.0, ()
    indices = _evenly_spaced_indices(len(values))
    slopes = tuple(
        (values[right] - values[left]) / (right - left)
        for position, left in enumerate(indices)
        for right in indices[position + 1 :]
    )
    slope = statistics.median(slopes) if slopes else 0.0
    intercept = statistics.median(
        value - slope * index for index, value in enumerate(values)
    )
    residuals = tuple(
        value - (intercept + slope * index) for index, value in enumerate(values)
    )
    return slope, intercept, residuals


def _trend_evidence(values: tuple[float, ...], horizon: int) -> TrendEvidence:
    n = len(values)
    h = max(1, horizon)
    lengths = tuple(
        sorted(
            {
                min(n, max(8, min(h, n // 4))),
                min(n, max(16, min(2 * h, n // 2))),
                min(n, max(32, min(4 * h, n))),
            }
        )
    )
    lengths = tuple(length for length in lengths if length >= 8)
    effects: list[float] = []
    gains: list[float] = []
    for length in lengths:
        window = values[-length:]
        slope, _intercept, residuals = _theil_sen_fit(window)
        residual_scale = max(
            1.4826 * _mad(residuals),
            1.4826
            * _mad(
                tuple(
                    window[index] - window[index - 1]
                    for index in range(1, len(window))
                )
            ),
            0.01 * statistics.median(abs(value) for value in window),
            1e-9,
        )
        effects.append(_clip(slope * h / residual_scale, -20.0, 20.0))
        baseline_mad = _mad(window)
        gains.append(
            _clip(1.0 - _mad(residuals) / max(baseline_mad, 1e-9), -1.0, 1.0)
            if baseline_mad > 1e-9
            else 0.0
        )
    if not effects:
        return TrendEvidence()
    median_effect = statistics.median(effects)
    material_signs = [
        1.0 if effect > 0 else -1.0 for effect in effects if abs(effect) >= 0.5
    ]
    sign_agreement = (
        abs(statistics.fmean(material_signs)) if material_signs else 0.0
    )
    dispersion = _mad(tuple(effects)) / max(abs(median_effect), 0.5)
    median_gain = statistics.median(gains)
    if len(effects) < 2:
        assessment = "insufficient"
    elif (
        sign_agreement >= 1.0 - 1e-12
        and abs(median_effect) >= 0.75
        and dispersion <= 0.75
        and median_gain > 0.0
    ):
        assessment = "supported"
    elif all(abs(effect) < 0.5 for effect in effects):
        assessment = "absent"
    else:
        assessment = "ambiguous"
    return TrendEvidence(
        window_lengths=lengths,
        signed_horizon_effects=tuple(_rounded(value) for value in effects),
        median_signed_horizon_effect=_rounded(median_effect),
        sign_agreement=_rounded(sign_agreement, nonnegative=True),
        relative_effect_dispersion=_rounded(dispersion, nonnegative=True),
        median_detrending_gain=round(_clip(median_gain, -1.0, 1.0), 6),
        assessment=assessment,
    )


def _correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_center = statistics.fmean(left)
    right_center = statistics.fmean(right)
    centered_left = tuple(value - left_center for value in left)
    centered_right = tuple(value - right_center for value in right)
    normalizer = max(
        *(abs(value) for value in centered_left),
        *(abs(value) for value in centered_right),
        1e-12,
    )
    scaled_left = tuple(value / normalizer for value in centered_left)
    scaled_right = tuple(value / normalizer for value in centered_right)
    variance_left = statistics.fmean(value * value for value in scaled_left)
    variance_right = statistics.fmean(value * value for value in scaled_right)
    if variance_left <= 1e-24 or variance_right <= 1e-24:
        return None
    covariance = statistics.fmean(
        left_value * right_value
        for left_value, right_value in zip(scaled_left, scaled_right, strict=True)
    )
    return _clip(covariance / math.sqrt(variance_left * variance_right), -1.0, 1.0)


def _residual_autocorrelation(values: tuple[float, ...], lag: int) -> float | None:
    if lag <= 0 or len(values) <= lag + 1:
        return None
    return _correlation(values[:-lag], values[lag:])


def _quantile(values: tuple[float, ...], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def _seasonal_card(
    values: tuple[float, ...], horizon: int, lag: int
) -> SeasonalLagEvidence:
    n = len(values)
    cycles = n // lag if lag > 0 else 0
    window_lengths = tuple(
        sorted(
            {
                length
                for length in (3 * lag, 5 * lag, min(n, 8 * lag))
                if lag >= 2 and 3 * lag <= length <= n
            }
        )
    )
    acfs: list[float] = []
    margins: list[float] = []
    for length in window_lengths:
        _slope, _intercept, residuals = _theil_sen_fit(values[-length:])
        center = _residual_autocorrelation(residuals, lag)
        left = _residual_autocorrelation(residuals, lag - 1)
        right = _residual_autocorrelation(residuals, lag + 1)
        if center is None or left is None or right is None:
            continue
        acfs.append(center)
        margins.append(center - max(left, right))

    replay_horizon = min(max(1, horizon), lag) if lag > 0 else 0
    ratios: list[float] = []
    if replay_horizon:
        origins = tuple(
            n - replay_horizon - offset * lag for offset in range(3)
        )
        for origin in origins:
            if origin < 2 * lag or origin + replay_horizon > n:
                continue
            train = values[:origin]
            target = values[origin : origin + replay_horizon]
            phase_start = len(train) - lag
            phase = train[phase_start : phase_start + replay_horizon]
            if len(phase) != replay_horizon:
                continue
            phase_mae = statistics.fmean(
                abs(prediction - actual)
                for prediction, actual in zip(phase, target, strict=True)
            )
            level_mae = statistics.fmean(abs(train[-1] - actual) for actual in target)
            ratios.append(
                _clip(phase_mae / max(level_mae, _robust_scale(train)), 0.0, 20.0)
            )

    amplitudes: list[float] = []
    if cycles:
        complete = values[-cycles * lag :]
        for start in range(0, len(complete), lag):
            cycle = complete[start : start + lag]
            amplitudes.append(_quantile(cycle, 0.9) - _quantile(cycle, 0.1))
    amplitude_dispersion = (
        1.4826 * _mad(tuple(amplitudes))
        / max(statistics.median(amplitudes), _robust_scale(values))
        if amplitudes
        else 0.0
    )
    acf_median = statistics.median(acfs) if acfs else 0.0
    acf_min = min(acfs) if acfs else 0.0
    margin_min = min(margins) if margins else 0.0
    phase_ratio = statistics.median(ratios) if ratios else 20.0
    near_constant = bool(values) and max(values) - min(values) <= 1e-9
    classic_support = (
        cycles >= 3
        and len(acfs) >= 2
        and bool(ratios)
        and acf_median >= 0.45
        and acf_min >= 0.25
        and margin_min > 0.0
        and phase_ratio <= 0.90
        and amplitude_dispersion <= 0.75
    )
    # Piecewise-constant and discrete-state cycles can form a broad ACF
    # plateau instead of a sharp local maximum. Repeated causal phase replay
    # is more direct predictive evidence in that case, but this route remains
    # deliberately narrow so a coincidental lag is not promoted.
    plateau_replay_support = (
        cycles >= 5
        and len(acfs) >= 2
        and len(ratios) >= 3
        and acf_median >= 0.45
        and acf_min >= 0.15
        and margin_min >= -0.03
        and phase_ratio <= 0.20
        and amplitude_dispersion <= 0.50
    )
    if near_constant:
        assessment = "absent"
    elif cycles < 3 or len(acfs) < 2 or not ratios:
        assessment = "insufficient"
    elif classic_support or plateau_replay_support:
        assessment = "supported"
    elif acf_median < 0.25 or phase_ratio > 1.25:
        assessment = "absent"
    else:
        assessment = "ambiguous"
    support_mode = (
        "local_acf_peak"
        if classic_support
        else "causal_replay_plateau"
        if plateau_replay_support
        else "none"
    )
    return SeasonalLagEvidence(
        lag=lag,
        complete_cycles=cycles,
        supporting_windows=len(acfs),
        detrended_acf_median=round(_clip(acf_median, -1.0, 1.0), 6),
        detrended_acf_min=round(_clip(acf_min, -1.0, 1.0), 6),
        local_peak_margin_min=round(_clip(margin_min, -2.0, 2.0), 6),
        phase_error_ratio_to_level=_rounded(phase_ratio, nonnegative=True),
        phase_replay_origin_count=len(ratios),
        cycle_amplitude_dispersion=_rounded(amplitude_dispersion, nonnegative=True),
        support_mode=support_mode,
        assessment=assessment,
    )


def _seasonality_evidence(
    values: tuple[float, ...], horizon: int, declared_period: int | None
) -> tuple[SeasonalLagEvidence, ...]:
    n = len(values)
    if n < 6:
        return ()
    _slope, _intercept, residuals = _theil_sen_fit(values)
    max_lag = min(60, n // 3)
    scores = {
        lag: _residual_autocorrelation(residuals, lag)
        for lag in range(1, max_lag + 2)
    }
    peaks = sorted(
        (
            (score, lag)
            for lag in range(2, max_lag + 1)
            if (score := scores.get(lag)) is not None
            and score > 0.0
            and scores.get(lag - 1) is not None
            and scores.get(lag + 1) is not None
            and score >= scores[lag - 1]  # type: ignore[operator]
            and score >= scores[lag + 1]  # type: ignore[operator]
            and (score > scores[lag - 1] or score > scores[lag + 1])  # type: ignore[operator]
        ),
        key=lambda item: (-item[0], item[1]),
    )
    candidate_lags: list[int] = []
    if declared_period and 3 * declared_period <= n:
        candidate_lags.append(declared_period)
    for _score, lag in peaks:
        if any(_same_cycle_family(lag, other) for other in candidate_lags):
            continue
        candidate_lags.append(lag)
        if len(candidate_lags) >= 3:
            break
    cards = [_seasonal_card(values, horizon, lag) for lag in candidate_lags[:3]]
    if any(card.assessment == "supported" for card in cards):
        return tuple(cards)

    # A broad autocorrelation plateau may not yield a strict local maximum.
    # Search only for the same narrow causal-replay support defined above and
    # surface at most one such card; weak extra lags stay hidden.
    plateau_scores = sorted(
        (
            (score, lag)
            for lag, score in scores.items()
            if 2 <= lag <= max_lag
            and score is not None
            and score >= 0.35
            and n >= 5 * lag
            and lag not in candidate_lags
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for _score, lag in plateau_scores:
        card = _seasonal_card(values, horizon, lag)
        if card.support_mode != "causal_replay_plateau":
            continue
        related = [
            index
            for index, existing in enumerate(cards)
            if _same_cycle_family(existing.lag, lag)
        ]
        if related:
            cards[related[0]] = card
        elif len(cards) < 3:
            cards.append(card)
        else:
            replaceable = [
                index
                for index, existing in enumerate(cards)
                if existing.lag != declared_period
            ]
            if replaceable:
                weakest = min(
                    replaceable,
                    key=lambda index: cards[index].detrended_acf_median,
                )
                cards[weakest] = card
        break
    return tuple(cards)


def _terminal_ood_width(values: tuple[float, ...], seasonal_lag: int | None) -> int | None:
    if len(values) < 8:
        return None
    for width in range(min(3, len(values) - 7), 0, -1):
        start = len(values) - width
        prior = values[start - 7 : start]
        suffix = values[start:]
        prior_differences = tuple(
            prior[index] - prior[index - 1] for index in range(1, len(prior))
        )
        difference_center = statistics.median(prior_differences)
        difference_scale = max(
            1.4826 * _mad(prior_differences),
            0.01 * abs(statistics.median(prior)),
            1e-9,
        )
        level_center = statistics.median(prior)
        level_scale = max(1.4826 * _mad(prior), 0.01 * abs(level_center), 1e-9)
        suffix_center = statistics.median(suffix)
        shift = abs(suffix_center - level_center)
        coherent = max(abs(value - suffix_center) for value in suffix) <= max(
            6.0 * max(difference_scale, level_scale), 0.1 * shift
        )
        seasonal = bool(
            seasonal_lag
            and all(
                index >= seasonal_lag
                and abs(values[index] - values[index - seasonal_lag])
                <= 3.0 * max(difference_scale, level_scale)
                for index in range(start, len(values))
            )
        )
        boundary = values[start] - values[start - 1]
        if (
            coherent
            and not seasonal
            and abs(boundary - difference_center) > 6.0 * difference_scale
            and shift > 6.0 * level_scale
        ):
            return width
    return None


def _seasonal_phase_explains(
    values: tuple[float, ...], width: int, lag: int | None, shift: float, scale: float
) -> bool:
    if lag is None:
        return False
    start = len(values) - width
    count = min(width, lag)
    if start < lag or count <= 0:
        return False
    phase_mae = statistics.fmean(
        abs(values[index] - values[index - lag])
        for index in range(start, start + count)
    )
    return phase_mae <= 0.75 * max(abs(shift), scale)


def _regime_ood_evidence(
    values: tuple[float, ...],
    horizon: int,
    seasonality: tuple[SeasonalLagEvidence, ...],
) -> RegimeOODEvidence:
    n = len(values)
    h = max(1, horizon)
    base = max(3, min(8, max(3, h // 4)))
    widths = tuple(
        sorted(
            {
                width
                for width in (base, 2 * base, 4 * base, min(h, n // 3))
                if width >= 3 and 2 * width <= n
            }
        )
    )
    supported = [card for card in seasonality if card.assessment == "supported"]
    seasonal_lag = (
        max(
            supported,
            key=lambda card: (
                card.detrended_acf_median,
                -card.phase_error_ratio_to_level,
                -card.lag,
            ),
        ).lag
        if supported
        else None
    )
    terminal_width = _terminal_ood_width(values, seasonal_lag)
    rows: list[tuple[int, float, float, float, float, float, float, float]] = []
    for width in widths:
        pre = values[-2 * width : -width]
        post = values[-width:]
        pre_center = statistics.median(pre)
        post_center = statistics.median(post)
        within_differences = tuple(
            [pre[index] - pre[index - 1] for index in range(1, len(pre))]
            + [post[index] - post[index - 1] for index in range(1, len(post))]
        )
        pooled_scale = max(
            1.4826 * _mad(within_differences),
            0.01 * statistics.median(abs(value) for value in (*pre, *post)),
            1e-9,
        )
        shift = post_center - pre_center
        shift_z = _clip(shift / pooled_scale, -20.0, 20.0)
        persistence = statistics.fmean(
            abs(value - post_center) < abs(value - pre_center) for value in post
        )
        dispersion = 1.4826 * _mad(post) / max(abs(shift), pooled_scale)
        boundary = abs(values[-width] - values[-width - 1]) / max(
            abs(shift), pooled_scale
        )
        score = abs(shift_z) * persistence / (1.0 + dispersion)
        rows.append(
            (
                width,
                shift_z,
                persistence,
                dispersion,
                boundary,
                score,
                shift,
                pooled_scale,
            )
        )
    if not rows:
        return RegimeOODEvidence(terminal_ood_width=terminal_width)
    best = max(rows, key=lambda row: (row[5], -row[0]))
    width, shift_z, persistence, dispersion, boundary, _score, shift, pooled_scale = best
    sign = 1 if shift_z > 0 else -1 if shift_z < 0 else 0
    supporting_fraction = statistics.fmean(
        (1 if row[1] > 0 else -1 if row[1] < 0 else 0) == sign
        and abs(row[1]) >= 2.0
        for row in rows
    )
    seasonal_explained = _seasonal_phase_explains(
        values, width, seasonal_lag, shift, pooled_scale
    )
    if (
        width >= 4
        and abs(shift_z) >= 3.0
        and persistence >= 0.8
        and dispersion <= 1.0
        and supporting_fraction >= 0.5
        and not seasonal_explained
    ):
        assessment = "persistent_regime"
    elif terminal_width is not None or abs(shift_z) >= 2.0 or seasonal_explained:
        assessment = "ambiguous_terminal"
    else:
        assessment = "no_regime"
    return RegimeOODEvidence(
        candidate_suffix_length=width,
        signed_shift_over_scale=_rounded(shift_z),
        supporting_width_fraction=_rounded(supporting_fraction, nonnegative=True),
        suffix_persistence_fraction=_rounded(persistence, nonnegative=True),
        suffix_dispersion_to_shift=_rounded(dispersion, nonnegative=True),
        boundary_jump_to_shift=_rounded(boundary, nonnegative=True),
        seasonal_phase_explained=seasonal_explained,
        terminal_ood_width=terminal_width,
        assessment=assessment,
    )
