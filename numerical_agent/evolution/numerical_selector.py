"""History-only hindcasting and task-conditioned numerical forecast selection."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

from common.metrics import mae, mase, smape

from .execution import Task


CandidateRunner = Callable[
    [str, tuple[float, ...], int, str], Sequence[float]
]


@dataclass(frozen=True)
class HindcastConfig:
    folds: int = 3
    min_successful_folds: int = 2
    catastrophic_mase: float = 10.0

    def __post_init__(self) -> None:
        if self.folds < 1:
            raise ValueError("folds must be positive")
        if not 1 <= self.min_successful_folds <= self.folds:
            raise ValueError("min_successful_folds must be within folds")
        if self.catastrophic_mase <= 0:
            raise ValueError("catastrophic_mase must be positive")


@dataclass(frozen=True)
class HindcastFold:
    train_end: int
    validation_end: int
    status: str
    forecast: tuple[float, ...] = ()
    truth: tuple[float, ...] = ()
    mase: float | None = None
    mae: float | None = None
    smape: float | None = None
    rmsse: float | None = None
    normalized_bias: float | None = None
    slope_error: float | None = None
    phase_error: float | None = None
    amplitude_ratio: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class CandidateDiagnostics:
    name: str
    family: str
    folds: tuple[HindcastFold, ...]
    successful_folds: int
    eligible: bool
    reason_code: str
    median_mase: float
    recent_mase: float
    worst_mase: float
    mase_mad: float
    median_mae: float
    median_smape: float
    median_rmsse: float
    normalized_bias: float
    slope_error: float
    phase_error: float | None
    amplitude_ratio: float | None
    explosion: bool
    fold_forecasts: tuple[tuple[float, ...], ...]
    fold_truths: tuple[tuple[float, ...], ...]
    cache_key: str = ""

    @classmethod
    def synthetic(
        cls,
        *,
        name: str,
        family: str,
        median_mase: float,
        recent_mase: float | None = None,
        worst_mase: float | None = None,
        mase_mad: float = 0.0,
        eligible: bool = True,
        fold_forecasts: Sequence[Sequence[float]] = (),
        fold_truths: Sequence[Sequence[float]] = (),
    ) -> "CandidateDiagnostics":
        forecasts = tuple(tuple(map(float, fold)) for fold in fold_forecasts)
        truths = tuple(tuple(map(float, fold)) for fold in fold_truths)
        count = min(len(forecasts), len(truths)) or (3 if eligible else 0)
        return cls(
            name=name,
            family=family,
            folds=(),
            successful_folds=count,
            eligible=eligible,
            reason_code="ok" if eligible else "insufficient_successful_folds",
            median_mase=float(median_mase),
            recent_mase=float(median_mase if recent_mase is None else recent_mase),
            worst_mase=float(median_mase if worst_mase is None else worst_mase),
            mase_mad=float(mase_mad),
            median_mae=float(median_mase),
            median_smape=float(median_mase),
            median_rmsse=float(median_mase),
            normalized_bias=0.0,
            slope_error=0.0,
            phase_error=0.0,
            amplitude_ratio=1.0,
            explosion=False,
            fold_forecasts=forecasts,
            fold_truths=truths,
        )


@dataclass(frozen=True)
class DecisionPolicy:
    min_successful_folds: int = 2
    catastrophic_mase: float = 10.0
    ranking_order: tuple[str, ...] = (
        "median_mase",
        "recent_mase",
        "worst_mase",
        "mase_mad",
        "normalized_bias",
    )
    recent_regime_first: bool = False
    ensemble_enabled: bool = True
    ensemble_max_members: int = 3
    ensemble_min_diversity: float = 0.05
    ensemble_min_improvement: float = 0.01
    fallback_to_best_available: bool = True

    def __post_init__(self) -> None:
        allowed = {
            "median_mase", "recent_mase", "worst_mase", "mase_mad",
            "normalized_bias", "median_rmsse", "slope_error",
        }
        if not self.ranking_order or not set(self.ranking_order) <= allowed:
            raise ValueError("ranking_order contains unsupported fields")
        if self.min_successful_folds < 1:
            raise ValueError("min_successful_folds must be positive")
        if not 1 <= self.ensemble_max_members <= 3:
            raise ValueError("ensemble_max_members must be between one and three")
        if self.ensemble_min_diversity < 0 or self.ensemble_min_improvement < 0:
            raise ValueError("ensemble thresholds must be nonnegative")


@dataclass(frozen=True)
class SelectionDecision:
    mode: str
    selected: tuple[str, ...]
    weights: tuple[float, ...]
    forecast: tuple[float, ...]
    confidence: float
    reason_codes: tuple[str, ...]
    rejected: Mapping[str, str]


def hindcast_cache_key(
    task: Task,
    candidate_name: str,
    family: str,
    config: HindcastConfig,
    *,
    screening_policy_hash: str = "",
    runtime_settings: Mapping[str, object] | None = None,
) -> str:
    payload = {
        "history": task.history,
        "horizon": task.horizon,
        "frequency": task.frequency,
        "candidate": candidate_name,
        "family": family,
        "config": asdict(config),
        "screening_policy_hash": screening_policy_hash,
        "runtime_settings": dict(runtime_settings or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def diagnose_candidate(
    task: Task,
    name: str,
    family: str,
    runner: CandidateRunner,
    config: HindcastConfig = HindcastConfig(),
    *,
    screening_policy_hash: str = "",
    runtime_settings: Mapping[str, object] | None = None,
) -> CandidateDiagnostics:
    """Diagnose one candidate using only earlier prefixes of ``task.history``."""
    history = tuple(float(value) for value in task.history)
    n = len(history)
    fold_horizon = min(task.horizon, max(1, n // 4))
    if n < fold_horizon * (config.folds + 1):
        return _empty_diagnostics(name, family, "insufficient_history", config, task,
                                  screening_policy_hash, runtime_settings)

    first_end = n - fold_horizon * (config.folds - 1)
    ends = tuple(first_end + index * fold_horizon for index in range(config.folds))
    folds: list[HindcastFold] = []
    for validation_end in ends:
        train_end = validation_end - fold_horizon
        prefix = history[:train_end]
        truth = history[train_end:validation_end]
        try:
            raw = runner(name, prefix, fold_horizon, task.frequency)
            forecast = tuple(float(value) for value in raw)
            if len(forecast) != fold_horizon or not all(map(math.isfinite, forecast)):
                raise ValueError("candidate returned an invalid forecast")
            folds.append(_score_fold(prefix, truth, forecast, train_end, validation_end))
        except BaseException as error:
            folds.append(HindcastFold(
                train_end=train_end,
                validation_end=validation_end,
                status="failed",
                truth=truth,
                detail=f"{type(error).__name__}: {error}"[:200],
            ))

    successful = tuple(fold for fold in folds if fold.status == "success")
    if len(successful) < config.min_successful_folds:
        return _summarize(
            name, family, tuple(folds), False, "insufficient_successful_folds",
            config, task, screening_policy_hash, runtime_settings,
        )
    return _summarize(
        name, family, tuple(folds), True, "ok", config, task,
        screening_policy_hash, runtime_settings,
    )


def diagnose_active_candidates(
    task: Task,
    active: Sequence[tuple[str, str]],
    runner: CandidateRunner,
    config: HindcastConfig = HindcastConfig(),
    *,
    screening_policy_hash: str = "",
    runtime_settings: Mapping[str, object] | None = None,
) -> Mapping[str, CandidateDiagnostics]:
    return {
        name: diagnose_candidate(
            task, name, family, runner, config,
            screening_policy_hash=screening_policy_hash,
            runtime_settings=runtime_settings,
        )
        for name, family in active
    }


def pairwise_diversity(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> float:
    values: list[float] = []
    for left_fold, right_fold in zip(left, right):
        if len(left_fold) != len(right_fold) or not left_fold:
            continue
        scale_values = tuple(float(value) for value in left_fold) + tuple(
            float(value) for value in right_fold
        )
        scale = max(1.0, max(scale_values) - min(scale_values))
        mse = statistics.fmean(
            (float(a) - float(b)) ** 2 for a, b in zip(left_fold, right_fold)
        )
        values.append(math.sqrt(mse) / scale)
    return statistics.fmean(values) if values else 0.0


def select_numerical_forecast(
    policy: DecisionPolicy,
    *,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
) -> SelectionDecision:
    active = set(active_names)
    rejected: dict[str, str] = {}
    eligible: list[CandidateDiagnostics] = []
    for name in sorted(active):
        diagnostic = diagnostics.get(name)
        if diagnostic is None or name not in forecasts:
            rejected[name] = "missing_diagnostics_or_forecast"
        elif not diagnostic.eligible or diagnostic.successful_folds < policy.min_successful_folds:
            rejected[name] = "insufficient_hindcast_reliability"
        elif diagnostic.explosion or diagnostic.worst_mase > policy.catastrophic_mase:
            rejected[name] = "catastrophic_hindcast_tail"
        else:
            eligible.append(diagnostic)
    if not eligible:
        if not policy.fallback_to_best_available:
            raise ValueError("no active candidate passed the reliability gate")
        fallback = [
            diagnostic
            for name in sorted(active)
            if name in forecasts and (diagnostic := diagnostics.get(name)) is not None
        ]
        if not fallback:
            raise ValueError("no active candidate has both diagnostics and a final forecast")
        chosen = min(
            fallback,
            key=lambda item: (
                -item.successful_folds,
                not math.isfinite(item.median_mase),
                item.median_mase,
                item.recent_mase,
                item.name,
            ),
        )
        return SelectionDecision(
            mode="single",
            selected=(chosen.name,),
            weights=(1.0,),
            forecast=tuple(float(value) for value in forecasts[chosen.name]),
            confidence=0.0,
            reason_codes=("conservative_best_available_fallback",),
            rejected=rejected,
        )

    front = _pareto_front(eligible)
    order = policy.ranking_order
    if policy.recent_regime_first:
        order = ("recent_mase",) + tuple(field for field in order if field != "recent_mase")
    ranked = sorted(front, key=lambda item: _rank_key(item, order))
    best = ranked[0]
    best_forecast = tuple(float(value) for value in forecasts[best.name])
    chosen = (best.name,)
    weights = (1.0,)
    forecast = best_forecast
    mode = "single"
    reasons = ["reliability_gate", "pareto_front", f"best_{order[0]}"]

    if policy.ensemble_enabled:
        proposal = _best_validated_ensemble(policy, eligible, forecasts, best)
        if proposal is not None:
            chosen, weights, forecast = proposal
            mode = "ensemble"
            reasons.extend(("diverse_members", "hindcast_blend_improvement"))

    all_ranked = sorted(eligible, key=lambda item: _rank_key(item, order))
    gap = 1.0
    if len(all_ranked) > 1:
        numerator = max(0.0, all_ranked[1].median_mase - all_ranked[0].median_mase)
        gap = numerator / (1.0 + abs(all_ranked[0].median_mase))
    confidence = max(0.0, min(1.0, gap))
    return SelectionDecision(
        mode=mode,
        selected=chosen,
        weights=weights,
        forecast=forecast,
        confidence=confidence,
        reason_codes=tuple(reasons),
        rejected=rejected,
    )


def _score_fold(
    prefix: tuple[float, ...],
    truth: tuple[float, ...],
    forecast: tuple[float, ...],
    train_end: int,
    validation_end: int,
) -> HindcastFold:
    absolute_scale = _naive_absolute_scale(prefix)
    squared_scale = _naive_squared_scale(prefix)
    errors = tuple(prediction - actual for prediction, actual in zip(forecast, truth))
    rmsse = math.sqrt(statistics.fmean(error * error for error in errors) / squared_scale)
    normalized_bias = statistics.fmean(errors) / absolute_scale
    slope_error = abs(_slope(forecast) - _slope(truth)) / absolute_scale
    phase = _phase_error(forecast, truth)
    truth_sd = statistics.pstdev(truth) if len(truth) > 1 else 0.0
    forecast_sd = statistics.pstdev(forecast) if len(forecast) > 1 else 0.0
    amplitude = forecast_sd / truth_sd if truth_sd > 1e-8 else (1.0 if forecast_sd <= 1e-8 else None)
    return HindcastFold(
        train_end=train_end,
        validation_end=validation_end,
        status="success",
        forecast=forecast,
        truth=truth,
        mase=mase(list(truth), list(forecast), list(prefix)),
        mae=mae(list(truth), list(forecast)),
        smape=smape(list(truth), list(forecast)),
        rmsse=rmsse,
        normalized_bias=normalized_bias,
        slope_error=slope_error,
        phase_error=phase,
        amplitude_ratio=amplitude,
    )


def _summarize(
    name: str,
    family: str,
    folds: tuple[HindcastFold, ...],
    eligible: bool,
    reason: str,
    config: HindcastConfig,
    task: Task,
    screening_policy_hash: str,
    runtime_settings: Mapping[str, object] | None,
) -> CandidateDiagnostics:
    successful = tuple(fold for fold in folds if fold.status == "success")

    def values(field: str) -> list[float]:
        return [float(value) for fold in successful if (value := getattr(fold, field)) is not None]

    mases = values("mase")
    median_mase = _median_or_inf(mases)
    med = median_mase
    mad = statistics.median(abs(value - med) for value in mases) if mases else math.inf
    phases = values("phase_error")
    amplitudes = values("amplitude_ratio")
    return CandidateDiagnostics(
        name=name,
        family=family,
        folds=folds,
        successful_folds=len(successful),
        eligible=eligible,
        reason_code=reason,
        median_mase=median_mase,
        recent_mase=float(successful[-1].mase) if successful else math.inf,
        worst_mase=max(mases, default=math.inf),
        mase_mad=mad,
        median_mae=_median_or_inf(values("mae")),
        median_smape=_median_or_inf(values("smape")),
        median_rmsse=_median_or_inf(values("rmsse")),
        normalized_bias=statistics.median(values("normalized_bias")) if successful else math.inf,
        slope_error=_median_or_inf(values("slope_error")),
        phase_error=statistics.median(phases) if phases else None,
        amplitude_ratio=statistics.median(amplitudes) if amplitudes else None,
        explosion=any(value > config.catastrophic_mase for value in mases),
        fold_forecasts=tuple(fold.forecast for fold in successful),
        fold_truths=tuple(fold.truth for fold in successful),
        cache_key=hindcast_cache_key(
            task, name, family, config,
            screening_policy_hash=screening_policy_hash,
            runtime_settings=runtime_settings,
        ),
    )


def _empty_diagnostics(
    name: str,
    family: str,
    reason: str,
    config: HindcastConfig,
    task: Task,
    screening_policy_hash: str,
    runtime_settings: Mapping[str, object] | None,
) -> CandidateDiagnostics:
    return _summarize(
        name, family, (), False, reason, config, task,
        screening_policy_hash, runtime_settings,
    )


def _pareto_front(candidates: Sequence[CandidateDiagnostics]) -> list[CandidateDiagnostics]:
    dimensions = ("median_mase", "recent_mase", "worst_mase", "mase_mad")
    front = []
    for candidate in candidates:
        dominated = any(
            all(getattr(other, field) <= getattr(candidate, field) for field in dimensions)
            and any(getattr(other, field) < getattr(candidate, field) for field in dimensions)
            for other in candidates
            if other.name != candidate.name
        )
        if not dominated:
            front.append(candidate)
    return front


def _rank_key(candidate: CandidateDiagnostics, order: Sequence[str]) -> tuple[object, ...]:
    values = []
    for field in order:
        value = float(getattr(candidate, field))
        values.append(abs(value) if field == "normalized_bias" else value)
    return (*values, candidate.name)


def _best_validated_ensemble(
    policy: DecisionPolicy,
    eligible: Sequence[CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    best: CandidateDiagnostics,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]] | None:
    pool = sorted(eligible, key=lambda item: _rank_key(item, policy.ranking_order))[:5]
    best_score = _fold_score(best.fold_forecasts, best.fold_truths)
    best_proposal = None
    for size in range(2, min(policy.ensemble_max_members, len(pool)) + 1):
        for members in itertools.combinations(pool, size):
            if not _aligned_folds(members):
                continue
            if min(
                pairwise_diversity(left.fold_forecasts, right.fold_forecasts)
                for left, right in itertools.combinations(members, 2)
            ) < policy.ensemble_min_diversity:
                continue
            weights = tuple(1.0 / size for _ in members)
            fold_forecasts = _blend_folds(tuple(member.fold_forecasts for member in members), weights)
            score = _fold_score(fold_forecasts, members[0].fold_truths)
            required = best_score * (1.0 - policy.ensemble_min_improvement)
            if not score < required:
                continue
            names_and_forecasts = sorted(
                ((member.name, tuple(map(float, forecasts[member.name]))) for member in members),
                key=lambda item: item[0],
            )
            names = tuple(item[0] for item in names_and_forecasts)
            final = tuple(
                statistics.fmean(values)
                for values in zip(*(item[1] for item in names_and_forecasts), strict=True)
            )
            proposal = (score, names, weights, final)
            if best_proposal is None or proposal[:2] < best_proposal[:2]:
                best_proposal = proposal
    if best_proposal is None:
        return None
    _, names, weights, final = best_proposal
    return names, weights, final


def _aligned_folds(members: Sequence[CandidateDiagnostics]) -> bool:
    first = members[0]
    return bool(first.fold_forecasts) and all(
        member.fold_truths == first.fold_truths
        and len(member.fold_forecasts) == len(first.fold_forecasts)
        for member in members[1:]
    )


def _blend_folds(
    all_folds: Sequence[Sequence[Sequence[float]]], weights: Sequence[float]
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            sum(weight * value for weight, value in zip(weights, values))
            for values in zip(*(all_folds[member][fold] for member in range(len(all_folds))))
        )
        for fold in range(len(all_folds[0]))
    )


def _fold_score(forecasts: Sequence[Sequence[float]], truths: Sequence[Sequence[float]]) -> float:
    if not forecasts or len(forecasts) != len(truths):
        return math.inf
    scores = []
    for forecast, truth in zip(forecasts, truths):
        if len(forecast) != len(truth) or not truth:
            return math.inf
        scale = max(1.0, max(truth) - min(truth))
        scores.append(statistics.fmean(abs(a - b) for a, b in zip(forecast, truth)) / scale)
    return statistics.median(scores)


def _naive_absolute_scale(history: Sequence[float]) -> float:
    diffs = [abs(history[index] - history[index - 1]) for index in range(1, len(history))]
    scale = statistics.fmean(diffs) if diffs else 0.0
    if scale <= 1e-8:
        spread = max(history) - min(history)
        return spread if spread > 1e-8 else 1.0
    return scale


def _naive_squared_scale(history: Sequence[float]) -> float:
    diffs = [(history[index] - history[index - 1]) ** 2 for index in range(1, len(history))]
    scale = statistics.fmean(diffs) if diffs else 0.0
    return scale if scale > 1e-8 else 1.0


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return (float(values[-1]) - float(values[0])) / (len(values) - 1)


def _phase_error(forecast: Sequence[float], truth: Sequence[float]) -> float | None:
    if len(truth) < 3 or statistics.pstdev(truth) <= 1e-8:
        return None
    max_lag = min(3, len(truth) - 1)
    scored = []
    for lag in range(-max_lag, max_lag + 1):
        left = forecast[max(0, lag): min(len(forecast), len(forecast) + lag)]
        right = truth[max(0, -lag): min(len(truth), len(truth) - lag)]
        if len(left) < 2:
            continue
        error = statistics.fmean((a - b) ** 2 for a, b in zip(left, right))
        scored.append((error, abs(lag)))
    return float(min(scored)[1]) if scored else None


def _median_or_inf(values: Sequence[float]) -> float:
    return statistics.median(values) if values else math.inf
