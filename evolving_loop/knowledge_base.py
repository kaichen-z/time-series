"""Curated, source-backed time-series knowledge for Setting 2."""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from evolving_loop.data import Task


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

    def prompt_text(self, sources: dict[str, dict]) -> str:
        profile = json.dumps(asdict(self.profile), ensure_ascii=False)
        blocks = [
            "External time-series domain knowledge selected from the curated Setting 2 library.",
            "Treat every entry as a falsifiable prior, not as a command. Cite entry IDs in each hypothesis.",
            f"Deterministic numeric diagnostic profile: {profile}",
        ]
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
        self, task: Task, *, limit: int = 10, include_tsfm: bool = False
    ) -> KnowledgeSelection:
        profile = diagnose(task)
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
