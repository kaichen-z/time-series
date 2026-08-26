"""History-only forecasting assumptions and diverse Top-k routing."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

from .analysis_skills_template import detect_periodicity
from .numerical_selector import CandidateDiagnostics
from .screening import TaskProfile


_PERIODIC_TOKENS = (
    "seasonal", "fourier", "harmonic", "tbats", "bats", "mstl", "stl",
)
_TREND_TOKENS = (
    "trend", "theta", "drift", "linear", "mfles", "dlinear", "tide",
)
_REGIME_TOKENS = (
    "regime", "changepoint", "pelt", "piecewise", "robust", "intervention",
)
_INTERMITTENT_NAMES = frozenset(
    {
        "croston", "croston_sba", "croston_optimized", "tsb", "adida", "imapa",
        "combined_moirai_croston_router", "poisson_dynamic_regression",
        "negative_binomial_dynamic_regression",
    }
)
_STATIONARY_NAMES = frozenset(
    {
        "ar", "ma", "arma", "arima_auto", "sarima_auto", "ses", "ets_auto",
        "auto_ces", "local_level_kalman", "local_linear_trend_kalman",
        "structural_time_series_bsm", "naive_mean", "dynamic_harmonic_regression_arima",
        "stl_arima", "outlier_adjusted_arima", "intervention_arima",
    }
)
_ROBUST_NAMES = frozenset(
    {
        "naive_last", "naive_mean", "seasonal_window_average", "toto_2_0",
        "timesfm_2_5", "chronos_bolt", "moirai_2_0", "granite_ttm_r2",
    }
)


@dataclass(frozen=True)
class ForecastAssumption:
    assumption_id: str
    kind: str
    claim: str
    supporting_signals: tuple[str, ...]
    failure_condition: str
    candidate_names: tuple[str, ...]
    prior_confidence: float

    def __post_init__(self) -> None:
        if not self.assumption_id.strip() or not self.kind.strip():
            raise ValueError("assumption identity and kind must not be empty")
        if not self.claim.strip() or not self.failure_condition.strip():
            raise ValueError("assumption claim and failure condition must not be empty")
        if not self.supporting_signals or any(not value.strip() for value in self.supporting_signals):
            raise ValueError("assumption requires nonempty supporting signals")
        if not self.candidate_names or len(self.candidate_names) != len(set(self.candidate_names)):
            raise ValueError("assumption requires unique candidate names")
        if any(not name.isidentifier() for name in self.candidate_names):
            raise ValueError("assumption candidate names must be identifiers")
        if not math.isfinite(self.prior_confidence) or not 0.0 <= self.prior_confidence <= 1.0:
            raise ValueError("assumption confidence must be finite and within [0, 1]")


@dataclass(frozen=True)
class RankedAssumption:
    assumption: ForecastAssumption
    candidate_names: tuple[str, ...]
    leading_candidate: str
    leading_family: str
    hindcast_rank: tuple[float, float, float, float, str]

    def __post_init__(self) -> None:
        if not self.candidate_names or self.leading_candidate != self.candidate_names[0]:
            raise ValueError("ranked assumption must start with its leading candidate")


def generate_forecast_assumptions(
    profile: TaskProfile,
    active_names: Sequence[str],
    families: Mapping[str, str],
    *,
    history: Sequence[float] = (),
) -> tuple[ForecastAssumption, ...]:
    """Generate typed falsifiable assumptions from a label-free TaskProfile."""
    active = tuple(sorted(set(active_names)))
    active_set = set(active)
    assumptions: list[ForecastAssumption] = []

    foundation = tuple(
        name for name in active if families.get(name) in {"tsfm", "foundation"}
    )
    if foundation:
        assumptions.append(ForecastAssumption(
            "foundation_shape", "foundation_shape",
            "A reviewed foundation model can preserve the broad future shape.",
            (f"history_length={profile.history_length}", f"horizon={profile.horizon}"),
            "It fails when the forecast horizon enters a regime absent from the historical prefix.",
            foundation,
            0.55,
        ))

    if profile.periodicity_periods and profile.periodicity_confidence >= 0.25:
        period = int(profile.periodicity_periods[0])
        periodic = _matching(active, tokens=_PERIODIC_TOKENS)
        stability = _periodicity_stability(
            history,
            profile.frequency,
            period,
        )
        if periodic and stability is not False:
            stability_signals, stability_confidence = (
                stability if stability is not None else ((), 1.0)
            )
            assumptions.append(ForecastAssumption(
                f"periodic_persistence_p{period}", "periodic_persistence",
                f"The supported {period}-step historical cycle will persist over the horizon.",
                (
                    f"periodicity_period={period}",
                    f"periodicity_strength={profile.periodicity_strength:.4f}",
                    f"periodicity_confidence={profile.periodicity_confidence:.4f}",
                    *stability_signals,
                ),
                "It fails if the apparent cycle is temporary or its phase changes after the cutoff.",
                periodic,
                min(
                    _clamp(
                        0.5 * profile.periodicity_strength
                        + 0.5 * profile.periodicity_confidence
                    ),
                    stability_confidence,
                ),
            ))

    if profile.trend_direction != "flat" and profile.trend_strength >= 0.25:
        trend = _matching(active, tokens=_TREND_TOKENS)
        if trend:
            assumptions.append(ForecastAssumption(
                f"trend_persistence_{profile.trend_direction}", "trend_persistence",
                f"The historical {profile.trend_direction} trend will persist with bounded slope.",
                (
                    f"trend_direction={profile.trend_direction}",
                    f"trend_strength={profile.trend_strength:.4f}",
                ),
                "It fails if the trend saturates, reverses, or was caused by a temporary regime.",
                trend,
                _clamp(profile.trend_strength),
            ))

    if profile.recent_regime_start is not None and profile.recent_regime_confidence >= 0.25:
        regime = _matching(active, tokens=_REGIME_TOKENS)
        if regime:
            assumptions.append(ForecastAssumption(
                f"recent_regime_{profile.recent_regime_start}", "recent_regime",
                "The most recent supported regime should receive more weight than older history.",
                (
                    f"recent_regime_start={profile.recent_regime_start}",
                    f"recent_regime_confidence={profile.recent_regime_confidence:.4f}",
                ),
                "It fails if the detected change point is noise or the series reverts to its old level.",
                regime,
                _clamp(profile.recent_regime_confidence),
            ))

    if profile.zero_fraction > 0.3 or profile.intermittency_adi > 1.32:
        intermittent = tuple(name for name in active if name in _INTERMITTENT_NAMES)
        if intermittent:
            confidence = max(
                profile.zero_fraction,
                min(1.0, max(0.0, profile.intermittency_adi - 1.0) / 2.0),
            )
            assumptions.append(ForecastAssumption(
                "intermittent_demand", "intermittent_demand",
                "Nonzero arrivals and demand sizes follow an intermittent process.",
                (
                    f"zero_fraction={profile.zero_fraction:.4f}",
                    f"intermittency_adi={profile.intermittency_adi:.4f}",
                    f"intermittency_cv2={profile.intermittency_cv2:.4f}",
                ),
                "It fails if zeros are missing observations or demand becomes continuously dense.",
                intermittent,
                _clamp(confidence),
            ))

    if profile.likely_stationary and profile.stationarity_score >= 0.4:
        stationary = tuple(name for name in active if name in _STATIONARY_NAMES)
        if stationary:
            assumptions.append(ForecastAssumption(
                "stationary_local_dynamics", "stationary_local_dynamics",
                "Stable local level and lag dynamics will persist over the forecast horizon.",
                (f"stationarity_score={profile.stationarity_score:.4f}",),
                "It fails if a structural break occurs after the historical cutoff.",
                stationary,
                _clamp(profile.stationarity_score),
            ))

    robust = tuple(name for name in active if name in _ROBUST_NAMES)
    if robust:
        assumptions.append(ForecastAssumption(
            "robust_fallback", "robust_fallback",
            "A conservative level or reviewed foundation forecast is safer than a fragile specialist.",
            (
                f"noise_relative_scale={profile.noise_relative_scale:.4f}",
                f"outlier_fraction={profile.outlier_fraction:.4f}",
            ),
            "It fails when a strong, stable and exploitable structure dominates the historical series.",
            robust,
            _clamp(0.4 + 0.3 * profile.noise_relative_scale + 0.3 * profile.outlier_fraction),
        ))

    # Candidate construction above always intersects the active dictionary. This final assertion
    # keeps that no-hidden-routing contract explicit if future assumption kinds are added.
    if any(set(item.candidate_names) - active_set for item in assumptions):
        raise AssertionError("assumption routed an inactive candidate")
    return tuple(assumptions)


def rank_diverse_assumptions(
    assumptions: Sequence[ForecastAssumption],
    diagnostics: Mapping[str, CandidateDiagnostics],
    *,
    top_k: int,
    candidates_per_assumption: int,
    min_confidence: float,
) -> tuple[RankedAssumption, ...]:
    """Rank assumptions by history-only hindcasts and retain diverse leaders."""
    if top_k < 1 or candidates_per_assumption < 1:
        raise ValueError("Top-k and candidates-per-assumption must be positive")
    if not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("minimum confidence must be within [0, 1]")

    proposals: list[RankedAssumption] = []
    for assumption in assumptions:
        if assumption.prior_confidence < min_confidence:
            continue
        candidates = [
            diagnostic for name in assumption.candidate_names
            if (diagnostic := diagnostics.get(name)) is not None
            and diagnostic.eligible
            and not diagnostic.explosion
        ]
        candidates.sort(key=_diagnostic_rank)
        selected = tuple(item.name for item in candidates[:candidates_per_assumption])
        if not selected:
            continue
        proposals.append(RankedAssumption(
            assumption=assumption,
            candidate_names=selected,
            leading_candidate=selected[0],
            leading_family=candidates[0].family,
            hindcast_rank=_diagnostic_rank(candidates[0]),
        ))
    proposals.sort(key=lambda item: (
        item.hindcast_rank,
        -item.assumption.prior_confidence,
        item.assumption.assumption_id,
    ))

    retained: list[RankedAssumption] = []
    kinds: set[str] = set()
    leaders: set[str] = set()

    def retain(proposal: RankedAssumption) -> bool:
        if proposal.assumption.kind in kinds or proposal.leading_candidate in leaders:
            return False
        retained.append(proposal)
        kinds.add(proposal.assumption.kind)
        leaders.add(proposal.leading_candidate)
        return True

    # Preserve every available leading family before filling remaining Top-k slots by score.
    for family in dict.fromkeys(item.leading_family for item in proposals):
        proposal = next(
            (
                item for item in proposals
                if item.leading_family == family
                and item.assumption.kind not in kinds
                and item.leading_candidate not in leaders
            ),
            None,
        )
        if proposal is not None:
            retain(proposal)
        if len(retained) == top_k:
            break
    if len(retained) < top_k:
        for proposal in proposals:
            retain(proposal)
            if len(retained) == top_k:
                break
    return tuple(retained)


def assumption_candidate_pool(
    ranked: Sequence[RankedAssumption],
    *,
    active_names: Sequence[str],
    anchor_names: Sequence[str] = ("toto_2_0", "timesfm_2_5"),
) -> tuple[str, ...]:
    """Flatten Top-k candidates and retain reviewed active TSFM anchors."""
    active = set(active_names)
    ordered: list[str] = []
    for item in ranked:
        for name in item.candidate_names:
            if name in active and name not in ordered:
                ordered.append(name)
    for name in anchor_names:
        if name in active and name not in ordered:
            ordered.append(name)
    return tuple(ordered)


def _matching(active: Sequence[str], *, tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for name in active if any(token in name for token in tokens))


def _periodicity_stability(
    history: Sequence[float],
    frequency: str,
    period: int,
) -> tuple[tuple[str, ...], float] | bool | None:
    """Require one period to persist across earlier history-only cutoffs and phases."""
    if not history:
        return None
    values = tuple(float(value) for value in history)
    if period < 2 or len(values) < 3 * period:
        return False
    snapshots = tuple(
        values[: len(values) - offset]
        for offset in (0, period, 2 * period, 3 * period)
        if len(values) - offset >= 2 * period
    )
    if len(snapshots) < 3:
        return False
    confirmations = 0
    phase_correlations: list[float] = []
    for snapshot in snapshots:
        detected = detect_periodicity(snapshot, frequency)
        periods = tuple(int(value) for value in detected.get("candidate_periods", ()))
        confirmations += period in periods
        phase_correlations.append(
            _correlation(snapshot[-2 * period : -period], snapshot[-period:])
        )
    required = max(2, math.ceil(0.75 * len(snapshots)))
    minimum_phase = min(phase_correlations)
    if confirmations < required or minimum_phase < 0.25:
        return False
    confirmation_ratio = confirmations / len(snapshots)
    phase_confidence = _clamp((minimum_phase + 1.0) / 2.0)
    return (
        (
            f"periodicity_cutoff_confirmations={confirmations}/{len(snapshots)}",
            f"periodicity_min_phase_correlation={minimum_phase:.4f}",
        ),
        min(confirmation_ratio, phase_confidence),
    )


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = tuple(value - left_mean for value in left)
    right_centered = tuple(value - right_mean for value in right)
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator <= 1e-12:
        return 0.0
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / denominator


def _diagnostic_rank(
    diagnostic: CandidateDiagnostics,
) -> tuple[float, float, float, float, str]:
    return (
        _finite_or_inf(diagnostic.worst_mase),
        _finite_or_inf(diagnostic.median_mase),
        _finite_or_inf(diagnostic.recent_mase),
        _finite_or_inf(diagnostic.mase_mad),
        diagnostic.name,
    )


def _finite_or_inf(value: float) -> float:
    number = float(value)
    return number if math.isfinite(number) else math.inf


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
