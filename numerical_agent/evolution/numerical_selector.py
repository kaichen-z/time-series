"""History-only hindcasting and task-conditioned numerical forecast selection."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from typing import Callable, Mapping, Sequence

from common.metrics import (
    drcik_point_metrics, joint_scaled_error, pareto_scaled_improvement, mae, mase, smape,
)

from .execution import Task


CandidateRunner = Callable[
    [str, tuple[float, ...], int, str], Sequence[float]
]


@dataclass(frozen=True)
class HindcastConfig:
    folds: int = 3
    min_successful_folds: int = 2
    catastrophic_smae_raw: float = 10.0
    catastrophic_srmse_raw: float = 10.0
    # Compatibility-only: retained for frozen callers; never read by selector gates.
    catastrophic_mase: float = 10.0
    long_horizon_audit: bool = False

    def __post_init__(self) -> None:
        if self.folds < 1:
            raise ValueError("folds must be positive")
        if not 1 <= self.min_successful_folds <= self.folds:
            raise ValueError("min_successful_folds must be within folds")
        if (
            not math.isfinite(self.catastrophic_smae_raw)
            or not math.isfinite(self.catastrophic_srmse_raw)
            or self.catastrophic_smae_raw <= 0
            or self.catastrophic_srmse_raw <= 0
        ):
            raise ValueError("raw scaled catastrophe thresholds must be positive")
        if not math.isfinite(self.catastrophic_mase) or self.catastrophic_mase <= 0:
            raise ValueError("legacy catastrophic_mase must be positive")


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
    mase_scale: float | None = None
    smae: float | None = None
    srmse: float | None = None
    smae_raw: float | None = None
    srmse_raw: float | None = None
    smae_clipped: bool | None = None
    srmse_clipped: bool | None = None
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
    long_horizon_fold: HindcastFold | None = None
    long_horizon_coverage: float = 0.0
    median_joint_scaled_error: float = math.inf
    recent_joint_scaled_error: float = math.inf
    worst_joint_scaled_error: float = math.inf
    median_smae: float = math.inf
    recent_smae: float = math.inf
    worst_smae: float = math.inf
    smae_mad: float = math.inf
    median_srmse: float = math.inf
    recent_srmse: float = math.inf
    worst_srmse: float = math.inf
    srmse_mad: float = math.inf
    worst_smae_raw: float = math.inf
    worst_srmse_raw: float = math.inf

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
        median_smae: float | None = None,
        recent_smae: float | None = None,
        worst_smae: float | None = None,
        smae_mad: float = 0.0,
        median_srmse: float | None = None,
        recent_srmse: float | None = None,
        worst_srmse: float | None = None,
        srmse_mad: float = 0.0,
        worst_smae_raw: float | None = None,
        worst_srmse_raw: float | None = None,
    ) -> "CandidateDiagnostics":
        forecasts = tuple(tuple(map(float, fold)) for fold in fold_forecasts)
        truths = tuple(tuple(map(float, fold)) for fold in fold_truths)
        count = min(len(forecasts), len(truths)) or (3 if eligible else 0)
        def legacy_scaled(value: float) -> float:
            return min(5.0, max(0.0, float(value))) if math.isfinite(float(value)) else 5.0

        legacy_recent = float(median_mase if recent_mase is None else recent_mase)
        legacy_worst = float(median_mase if worst_mase is None else worst_mase)
        legacy_median = float(median_mase)
        uses_scaled_inputs = any(
            value is not None
            for value in (
                median_smae, recent_smae, worst_smae,
                median_srmse, recent_srmse, worst_srmse,
            )
        )
        scaled_smae = float(legacy_scaled(legacy_median) if median_smae is None else median_smae)
        scaled_srmse = float(legacy_scaled(legacy_median) if median_srmse is None else median_srmse)
        scaled_recent_smae = float(legacy_scaled(legacy_recent) if recent_smae is None else recent_smae)
        scaled_recent_srmse = float(legacy_scaled(legacy_recent) if recent_srmse is None else recent_srmse)
        scaled_worst_smae = float(legacy_scaled(legacy_worst) if worst_smae is None else worst_smae)
        scaled_worst_srmse = float(legacy_scaled(legacy_worst) if worst_srmse is None else worst_srmse)
        return cls(
            name=name,
            family=family,
            folds=(),
            successful_folds=count,
            eligible=eligible,
            reason_code="ok" if eligible else "insufficient_successful_folds",
            median_mase=float(median_mase),
            recent_mase=legacy_recent,
            worst_mase=legacy_worst,
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
            median_joint_scaled_error=joint_scaled_error(scaled_smae, scaled_srmse),
            recent_joint_scaled_error=joint_scaled_error(
                scaled_recent_smae, scaled_recent_srmse
            ),
            worst_joint_scaled_error=joint_scaled_error(
                scaled_worst_smae, scaled_worst_srmse
            ),
            median_smae=scaled_smae,
            recent_smae=scaled_recent_smae,
            worst_smae=scaled_worst_smae,
            smae_mad=float(smae_mad),
            median_srmse=scaled_srmse,
            recent_srmse=scaled_recent_srmse,
            worst_srmse=scaled_worst_srmse,
            srmse_mad=float(srmse_mad),
            worst_smae_raw=float(
                legacy_worst if worst_smae_raw is None and not uses_scaled_inputs else (
                    scaled_worst_smae if worst_smae_raw is None else worst_smae_raw
                )
            ),
            worst_srmse_raw=float(
                legacy_worst if worst_srmse_raw is None and not uses_scaled_inputs else (
                    scaled_worst_srmse if worst_srmse_raw is None else worst_srmse_raw
                )
            ),
        )


@dataclass(frozen=True)
class DecisionPolicy:
    min_successful_folds: int = 3
    catastrophic_smae_raw: float = 10.0
    catastrophic_srmse_raw: float = 10.0
    # Compatibility-only: downstream frozen payloads may still supply this field.
    catastrophic_mase: float = 10.0
    max_smae_fold_regret: float = 0.02
    max_srmse_fold_regret: float = 0.02
    baseline_strategy: str = "toto_first"
    tsfm_router_min_improvement: float = 0.02
    tsfm_router_blend_weight: float = 0.0
    assumption_guidance_enabled: bool = False
    assumption_top_k: int = 5
    assumption_candidates_per_hypothesis: int = 2
    assumption_min_confidence: float = 0.25
    ranking_order: tuple[str, ...] = (
        "median_joint_scaled_error",
        "recent_joint_scaled_error",
        "worst_joint_scaled_error",
        "median_smae",
        "median_srmse",
        "normalized_bias",
    )
    recent_regime_first: bool = True
    ensemble_enabled: bool = True
    ensemble_max_members: int = 2
    ensemble_min_diversity: float = 0.1
    ensemble_min_improvement: float = 0.05
    ensemble_weight_grid: tuple[float, ...] = (0.8, 0.9)
    ensemble_residual_strengths: tuple[float, ...] = (0.1, 0.25)
    ensemble_correction_clip: float = 1.0
    ensemble_min_fold_wins: int = 2
    ensemble_max_worst_fold_regret: float = 0.02
    long_horizon_audit_enabled: bool = False
    long_horizon_penalty_weight: float = 0.0
    long_horizon_route_feature: str = "audit_coverage"
    long_horizon_route_operator: str = "at_least"
    long_horizon_route_threshold: float = 1.0
    long_horizon_guard_enabled: bool = False
    long_horizon_min_coverage: float = 0.75
    long_horizon_max_regret: float = 0.0
    fallback_to_best_available: bool = True

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object], *, allow_legacy: bool = False
    ) -> "DecisionPolicy":
        """Parse active scaled policies; legacy MASE migration is opt-in only."""
        raw = dict(payload)
        legacy_keys = {"catastrophic_mase", "median_mase", "recent_mase", "worst_mase", "mase_mad"}
        if legacy_keys & set(raw) and not allow_legacy:
            raise ValueError("legacy MASE policy fields require allow_legacy=True")
        if "ranking_order" in raw:
            ranking = raw["ranking_order"]
            if isinstance(ranking, (str, bytes)):
                raise ValueError("ranking_order must be a sequence")
            ranking = tuple(ranking)  # type: ignore[arg-type]
            if any(
                str(field) in {
                    "median_mase", "recent_mase", "worst_mase", "mase_mad",
                    "median_rmsse", "median_smape",
                }
                for field in ranking
            ) and not allow_legacy:
                raise ValueError("legacy MASE policy fields require allow_legacy=True")
            raw["ranking_order"] = ranking
        allowed = set(asdict(cls()))
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unsupported DecisionPolicy fields: {sorted(unknown)}")
        return cls(**raw)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        legacy_ranking = {
            "median_mase": "median_joint_scaled_error",
            "recent_mase": "recent_joint_scaled_error",
            "worst_mase": "worst_joint_scaled_error",
            "mase_mad": "smae_mad",
            "median_rmsse": "median_srmse",
        }
        normalized_order = tuple(
            legacy_ranking.get(field, field) for field in self.ranking_order
        )
        # Old frozen payloads are read-only inputs; the live object never carries
        # an MASE ranking field into selection or override logic.
        object.__setattr__(self, "ranking_order", normalized_order)
        allowed = {
            "median_joint_scaled_error", "recent_joint_scaled_error",
            "worst_joint_scaled_error", "median_smae", "recent_smae",
            "worst_smae", "smae_mad", "median_srmse", "recent_srmse",
            "worst_srmse", "srmse_mad", "normalized_bias", "slope_error",
        }
        if not normalized_order or not set(normalized_order) <= allowed:
            raise ValueError("ranking_order contains unsupported fields")
        if self.min_successful_folds < 1:
            raise ValueError("min_successful_folds must be positive")
        safety_values = (
            self.catastrophic_smae_raw,
            self.catastrophic_srmse_raw,
            self.catastrophic_mase,
            self.max_smae_fold_regret,
            self.max_srmse_fold_regret,
        )
        if (
            not all(math.isfinite(value) for value in safety_values)
            or self.catastrophic_smae_raw <= 0
            or self.catastrophic_srmse_raw <= 0
            or self.catastrophic_mase <= 0
            or self.max_smae_fold_regret < 0
            or self.max_srmse_fold_regret < 0
        ):
            raise ValueError("scaled safety thresholds must be nonnegative")
        if self.baseline_strategy not in {
            "toto_first",
            "minimax_tsfm",
            "conservative_tsfm",
            "conservative_combined",
            "conservative_single_tsfm",
            "conservative_tsfm_portfolio",
            "conservative_tsfm_statistical",
            "conservative_joint_portfolio",
            "protected_single_tsfm",
            "protected_tsfm_portfolio",
            "protected_joint_residual",
            "protected_topk_single_tsfm",
            "protected_topk_tsfm_portfolio",
            "protected_topk_joint_residual",
        }:
            raise ValueError(
                "baseline_strategy must be toto_first, minimax_tsfm, "
                "conservative_tsfm, conservative_combined, "
                "conservative_single_tsfm, conservative_tsfm_portfolio, "
                "conservative_tsfm_statistical, conservative_joint_portfolio, "
                "protected_single_tsfm, protected_tsfm_portfolio, "
                "protected_joint_residual, protected_topk_single_tsfm, "
                "protected_topk_tsfm_portfolio, or protected_topk_joint_residual"
            )
        if (
            not math.isfinite(self.tsfm_router_min_improvement)
            or not 0.0 <= self.tsfm_router_min_improvement <= 1.0
        ):
            raise ValueError("TSFM router minimum improvement must be within [0, 1]")
        if (
            not math.isfinite(self.tsfm_router_blend_weight)
            or not 0.0 <= self.tsfm_router_blend_weight <= 0.5
        ):
            raise ValueError("TSFM router blend weight must be within [0, 0.5]")
        if not 1 <= self.assumption_top_k <= 7:
            raise ValueError("assumption_top_k must be between one and seven")
        if not 1 <= self.assumption_candidates_per_hypothesis <= 3:
            raise ValueError("assumption candidates per hypothesis must be between one and three")
        if (
            not math.isfinite(self.assumption_min_confidence)
            or not 0.0 <= self.assumption_min_confidence <= 1.0
        ):
            raise ValueError("assumption minimum confidence must be within [0, 1]")
        if not 1 <= self.ensemble_max_members <= 3:
            raise ValueError("ensemble_max_members must be between one and three")
        if self.ensemble_min_diversity < 0 or self.ensemble_min_improvement < 0:
            raise ValueError("ensemble thresholds must be nonnegative")
        if any(not 0.5 < weight < 1.0 for weight in self.ensemble_weight_grid):
            raise ValueError("ensemble weights must be strictly between 0.5 and 1.0")
        if any(not 0.0 < strength <= 0.5 for strength in self.ensemble_residual_strengths):
            raise ValueError("residual strengths must be in (0, 0.5]")
        if self.ensemble_correction_clip <= 0:
            raise ValueError("ensemble_correction_clip must be positive")
        if self.ensemble_min_fold_wins < 1:
            raise ValueError("ensemble_min_fold_wins must be positive")
        if self.ensemble_max_worst_fold_regret < 0:
            raise ValueError("ensemble_max_worst_fold_regret must be nonnegative")
        if self.long_horizon_penalty_weight not in {0.0, 0.25, 0.5, 1.0}:
            raise ValueError("long-horizon penalty must be 0, 0.25, 0.5, or 1.0")
        route_features = {
            "audit_coverage",
            "horizon_ratio",
            "history_length",
            "horizon",
            "trend_strength",
            "periodicity_strength",
            "recent_regime_confidence",
            "noise_relative_scale",
            "intermittency_adi",
            "zero_fraction",
        }
        if self.long_horizon_route_feature not in route_features:
            raise ValueError("unsupported long-horizon route feature")
        if self.long_horizon_route_operator not in {"at_least", "at_most"}:
            raise ValueError("long-horizon route operator must be at_least or at_most")
        if not math.isfinite(self.long_horizon_route_threshold):
            raise ValueError("long-horizon route threshold must be finite")
        if self.long_horizon_audit_enabled != bool(self.long_horizon_penalty_weight):
            raise ValueError("long-horizon audit must be enabled exactly for nonzero penalty")
        if not 0.0 <= self.long_horizon_min_coverage <= 1.0:
            raise ValueError("long-horizon minimum coverage must be within [0, 1]")
        if not 0.0 <= self.long_horizon_max_regret <= 1.0:
            raise ValueError("long-horizon maximum regret must be within [0, 1]")


@dataclass(frozen=True)
class SelectionArithmetic:
    """Immutable, selector-owned recipe for replaying a materialized decision."""

    operation: str
    candidate_name: str | None = None
    inputs: tuple["SelectionArithmetic", ...] = ()
    weights: tuple[float, ...] = ()
    strength: float | None = None
    clip_multiplier: float | None = None
    scale: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str):
            raise ValueError("selection arithmetic operation must be a string")
        inputs = tuple(self.inputs)
        raw_weights = tuple(self.weights)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_weights
        ):
            raise ValueError("selection arithmetic weights must be numerical")
        weights = tuple(float(value) for value in raw_weights)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "weights", weights)
        if self.operation == "leaf":
            if not isinstance(self.candidate_name, str) or not self.candidate_name:
                raise ValueError("selection arithmetic leaf requires a candidate")
            if inputs or weights or any(
                value is not None
                for value in (self.strength, self.clip_multiplier, self.scale)
            ):
                raise ValueError("selection arithmetic leaf has unexpected fields")
            return
        if self.candidate_name is not None or len(inputs) < 2:
            raise ValueError("selection arithmetic operation requires two or more inputs")
        if any(not isinstance(item, SelectionArithmetic) for item in inputs):
            raise ValueError("selection arithmetic inputs must be immutable recipes")
        if self.operation in {"blend", "weighted"}:
            if (
                len(weights) != len(inputs)
                or (self.operation == "blend" and len(inputs) != 2)
                or any(not math.isfinite(value) or value < 0.0 for value in weights)
                or math.fsum(weights) != 1.0
            ):
                raise ValueError(
                    f"{self.operation} arithmetic requires normalized finite weights"
                )
            if any(
                value is not None
                for value in (self.strength, self.clip_multiplier, self.scale)
            ):
                raise ValueError(
                    f"{self.operation} arithmetic has unexpected residual fields"
                )
            return
        if self.operation in {"fmean", "median"}:
            if weights or any(
                value is not None
                for value in (self.strength, self.clip_multiplier, self.scale)
            ):
                raise ValueError(f"{self.operation} arithmetic has unexpected fields")
            return
        if self.operation == "residual":
            if len(inputs) != 2 or weights:
                raise ValueError("residual arithmetic requires exactly two inputs")
            parameters = (self.strength, self.clip_multiplier, self.scale)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in parameters
            ):
                raise ValueError("residual arithmetic requires finite parameters")
            assert self.strength is not None
            assert self.clip_multiplier is not None
            assert self.scale is not None
            strength = float(self.strength)
            clip_multiplier = float(self.clip_multiplier)
            scale = float(self.scale)
            if not 0.0 < strength <= 0.5:
                raise ValueError("residual strength must be within (0, 0.5]")
            if clip_multiplier <= 0.0 or scale <= 0.0:
                raise ValueError("residual clip and scale must be positive")
            object.__setattr__(self, "strength", strength)
            object.__setattr__(self, "clip_multiplier", clip_multiplier)
            object.__setattr__(self, "scale", scale)
            return
        raise ValueError("unsupported selection arithmetic operation")


@dataclass(frozen=True)
class SelectionDecision:
    mode: str
    selected: tuple[str, ...]
    weights: tuple[float, ...]
    forecast: tuple[float, ...]
    confidence: float
    reason_codes: tuple[str, ...]
    rejected: Mapping[str, str]
    combination_type: str | None = None
    baseline_name: str | None = None
    assumption_ids: tuple[str, ...] = ()
    assumption_kinds: tuple[str, ...] = ()
    considered_candidates: tuple[str, ...] = ()
    arithmetic: SelectionArithmetic | None = None


_WEIGHTED_SELECTION_TYPES = frozenset(
    {
        "protected_tsfm_weighted_portfolio",
        "tsfm_weighted_portfolio",
    }
)
_BLEND_SELECTION_TYPES = frozenset(
    {
        "joint_tsfm_statistical_portfolio",
        "statistical_shrinkage_overlay",
        "tsfm_shrinkage_overlay",
        "weighted_blend",
    }
)
_MEDIAN_SELECTION_TYPES = frozenset(
    {"protected_tsfm_median_portfolio", "tsfm_median_portfolio"}
)
_RESIDUAL_SELECTION_TYPES = frozenset(
    {
        "protected_joint_tsfm_statistical_residual",
        "protected_statistical_residual",
        "residual_correction",
    }
)


def replay_selection_forecast(
    decision: SelectionDecision,
    forecasts: Mapping[str, Sequence[float]],
) -> tuple[float, ...]:
    """Replay a selector decision using only its trusted arithmetic and leaves."""
    if not isinstance(decision, SelectionDecision):
        raise ValueError("selection arithmetic replay requires a decision")
    arithmetic = decision.arithmetic or _implicit_selection_arithmetic(decision)
    expected_operation = _expected_arithmetic_operation(decision)
    if arithmetic.operation != expected_operation:
        raise ValueError("selection mode does not match its arithmetic replay")
    if _arithmetic_leaf_names(arithmetic) != tuple(decision.selected):
        raise ValueError("selection names do not match its arithmetic replay")
    if _arithmetic_attribution_weights(arithmetic) != tuple(decision.weights):
        raise ValueError("selection weights do not match its arithmetic replay")
    return _replay_arithmetic(arithmetic, forecasts)


def _implicit_selection_arithmetic(
    decision: SelectionDecision,
) -> SelectionArithmetic:
    leaves = tuple(
        SelectionArithmetic("leaf", candidate_name=name)
        for name in decision.selected
    )
    operation = _expected_arithmetic_operation(decision)
    if operation == "leaf":
        return leaves[0]
    if operation == "residual":
        raise ValueError("residual selection requires trusted arithmetic replay")
    if operation in {"blend", "weighted"}:
        return SelectionArithmetic(operation, inputs=leaves, weights=decision.weights)
    return SelectionArithmetic(operation, inputs=leaves)


def _expected_arithmetic_operation(decision: SelectionDecision) -> str:
    if decision.mode == "single" and decision.combination_type is None:
        return "leaf"
    if decision.mode == "ensemble" and decision.combination_type is None:
        if len(set(decision.weights)) == 1:
            return "fmean"
        return "weighted"
    if decision.mode == "combined":
        if decision.combination_type in _BLEND_SELECTION_TYPES:
            return "blend"
        if decision.combination_type in _WEIGHTED_SELECTION_TYPES:
            return "weighted"
        if decision.combination_type in _MEDIAN_SELECTION_TYPES:
            return "median"
        if decision.combination_type in _RESIDUAL_SELECTION_TYPES:
            return "residual"
    raise ValueError("selection mode lacks a supported arithmetic replay")


def _arithmetic_leaf_names(arithmetic: SelectionArithmetic) -> tuple[str, ...]:
    if arithmetic.operation == "leaf":
        assert arithmetic.candidate_name is not None
        return (arithmetic.candidate_name,)
    return tuple(
        name for item in arithmetic.inputs for name in _arithmetic_leaf_names(item)
    )


def _arithmetic_attribution_weights(
    arithmetic: SelectionArithmetic,
) -> tuple[float, ...]:
    if arithmetic.operation == "leaf":
        return (1.0,)
    if arithmetic.operation == "residual":
        assert arithmetic.strength is not None
        parent, specialist = arithmetic.inputs
        return (
            *(value * (1.0 - arithmetic.strength)
              for value in _arithmetic_attribution_weights(parent)),
            *(value * arithmetic.strength
              for value in _arithmetic_attribution_weights(specialist)),
        )
    factors = (
        arithmetic.weights
        if arithmetic.operation in {"blend", "weighted"}
        else (1.0 / len(arithmetic.inputs),) * len(arithmetic.inputs)
    )
    return tuple(
        value * factor
        for item, factor in zip(arithmetic.inputs, factors, strict=True)
        for value in _arithmetic_attribution_weights(item)
    )


def _replay_arithmetic(
    arithmetic: SelectionArithmetic,
    forecasts: Mapping[str, Sequence[float]],
) -> tuple[float, ...]:
    if arithmetic.operation == "leaf":
        assert arithmetic.candidate_name is not None
        try:
            values = forecasts[arithmetic.candidate_name]
        except (KeyError, TypeError) as error:
            raise ValueError("selection arithmetic references an unavailable forecast") from error
        return tuple(float(value) for value in values)
    values = tuple(_replay_arithmetic(item, forecasts) for item in arithmetic.inputs)
    if arithmetic.operation == "blend":
        return _blend_values(values[0], values[1], arithmetic.weights[0])
    if arithmetic.operation == "weighted":
        return _weighted_values(values, arithmetic.weights)
    if arithmetic.operation == "fmean":
        return _fmean_values(values)
    if arithmetic.operation == "median":
        return _median_values(values)
    assert arithmetic.operation == "residual"
    assert arithmetic.strength is not None
    assert arithmetic.clip_multiplier is not None
    assert arithmetic.scale is not None
    return _residual_values(
        values[0], values[1], arithmetic.strength,
        arithmetic.clip_multiplier, arithmetic.scale,
    )


def select_assumption_guided_forecast(
    policy: DecisionPolicy,
    *,
    profile,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    families: Mapping[str, str],
    history: Sequence[float] = (),
    conditioned_names: Sequence[str] = (),
) -> SelectionDecision:
    """Route a diverse Top-k of history-only assumptions into the safe Verifier."""
    if not policy.assumption_guidance_enabled:
        return select_numerical_forecast(
            policy,
            profile=profile,
            active_names=active_names,
            diagnostics=diagnostics,
            forecasts=forecasts,
            history=history,
            conditioned_names=conditioned_names,
        )

    # Local import avoids making the assumption schema depend cyclically on selector loading.
    from .assumptions import (
        assumption_candidate_pool,
        generate_forecast_assumptions,
        rank_diverse_assumptions,
    )

    assumptions = generate_forecast_assumptions(
        profile,
        active_names,
        families,
        history=history,
    )
    ranked = rank_diverse_assumptions(
        assumptions,
        diagnostics,
        top_k=policy.assumption_top_k,
        candidates_per_assumption=policy.assumption_candidates_per_hypothesis,
        min_confidence=policy.assumption_min_confidence,
    )
    pool = assumption_candidate_pool(ranked, active_names=active_names)
    if not pool:
        pool = tuple(active_names)
    justified = tuple(
        dict.fromkeys(name for item in ranked for name in item.candidate_names)
    )
    decision = select_numerical_forecast(
        policy,
        profile=profile,
        active_names=pool,
        diagnostics=diagnostics,
        forecasts=forecasts,
        history=history,
        conditioned_names=tuple(dict.fromkeys((*conditioned_names, *justified))),
    )
    return replace(
        decision,
        reason_codes=(*decision.reason_codes, "assumption_top_k_verifier"),
        assumption_ids=tuple(item.assumption.assumption_id for item in ranked),
        assumption_kinds=tuple(item.assumption.kind for item in ranked),
        considered_candidates=pool,
    )


def select_grounded_morphology_forecast(
    policy: DecisionPolicy,
    *,
    assumptions: Sequence[object],
    protected_anchor: SelectionDecision,
    horizon: int,
    profile,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    history: Sequence[float] = (),
    conditioned_names: Sequence[str] = (),
) -> SelectionDecision:
    """Apply validated grounded assumptions without granting them numerical authority.

    The caller owns card validation and supplies the exact protected Safe-Anchor computed
    for the loop.  This boundary uses only typed candidate names, then requires any proposed
    override to pass the ordinary numerical gates.  Assumption prose is deliberately unread.
    """
    from .morphology import AssumptionGrounding

    grounded = tuple(assumptions)
    if not grounded or any(not isinstance(item, AssumptionGrounding) for item in grounded):
        raise ValueError("grounded morphology selection requires validated assumptions")
    if len({item.assumption_id for item in grounded}) != len(grounded):
        raise ValueError("grounded assumptions must have unique ids")

    active = tuple(dict.fromkeys(active_names))
    active_set = set(active)
    pool = tuple(
        dict.fromkeys(
            name
            for item in grounded
            for name in item.candidate_names
            if name in active_set
        )
    )
    if not is_materialized_single_selection(
        protected_anchor,
        active_names=active,
        forecasts=forecasts,
        horizon=horizon,
    ):
        raise ValueError("protected anchor must be one active materialized forecast")
    anchor_name = protected_anchor.selected[0]
    considered = tuple(
        dict.fromkeys((*pool, anchor_name))
    )
    if not pool:
        raise ValueError("grounded assumptions contain no active candidate")

    proposal = select_numerical_forecast(
        policy,
        profile=profile,
        active_names=considered,
        diagnostics=diagnostics,
        forecasts=forecasts,
        history=history,
        conditioned_names=tuple(
            dict.fromkeys((*conditioned_names, *pool))
        ),
    )
    decision = proposal
    protected = False
    if proposal.selected != (anchor_name,):
        reference = diagnostics.get(anchor_name)
        challenger = (
            diagnostics.get(proposal.selected[0])
            if is_materialized_single_selection(
                proposal,
                active_names=considered,
                forecasts=forecasts,
                horizon=horizon,
            )
            else None
        )
        if (
            reference is None
            or challenger is None
            or not _passes_reliability_gate(policy, reference)
            or not _passes_reliability_gate(policy, challenger)
            or not _passes_single_override(
                policy, challenger, reference, profile=profile
            )
        ):
            decision = protected_anchor
            protected = True
    return replace(
        decision,
        reason_codes=(
            *decision.reason_codes,
            "grounded_morphology_top_k_verifier",
            *(("exact_safe_anchor_protection",) if protected else ()),
        ),
        baseline_name=anchor_name,
        assumption_ids=tuple(item.assumption_id for item in grounded),
        assumption_kinds=tuple(item.kind for item in grounded),
        considered_candidates=considered,
    )


def is_materialized_single_selection(
    decision: SelectionDecision,
    *,
    active_names: Sequence[str],
    forecasts: Mapping[str, Sequence[float]],
    horizon: int,
) -> bool:
    """Validate the single executed-forecast boundary used by Morphology selection."""
    if (
        not isinstance(decision, SelectionDecision)
        or decision.mode != "single"
        or len(decision.selected) != 1
    ):
        return False
    name = decision.selected[0]
    materialized = forecasts.get(name)
    return bool(
        name in set(active_names)
        and materialized is not None
        and _valid_exact_forecast(materialized, horizon)
        and tuple(float(value) for value in materialized) == tuple(decision.forecast)
    )


def _passes_reliability_gate(
    policy: DecisionPolicy, diagnostic: CandidateDiagnostics
) -> bool:
    return bool(
        diagnostic.eligible
        and diagnostic.successful_folds >= policy.min_successful_folds
        and not diagnostic.explosion
        and _passes_scaled_tail_gate(policy, diagnostic)
    )


def _passes_scaled_tail_gate(
    policy: DecisionPolicy, diagnostic: CandidateDiagnostics
) -> bool:
    """Use uncapped scaled errors for catastrophe discrimination."""
    return bool(
        math.isfinite(diagnostic.worst_smae_raw)
        and math.isfinite(diagnostic.worst_srmse_raw)
        and diagnostic.worst_smae_raw <= policy.catastrophic_smae_raw
        and diagnostic.worst_srmse_raw <= policy.catastrophic_srmse_raw
    )


def select_protected_safe_anchor(
    policy: DecisionPolicy,
    *,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    horizon: int,
    fallback_reason: str,
) -> SelectionDecision:
    """Return one finite materialized anchor when advisory morphology fails closed."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("safe-anchor horizon must be positive")
    active = set(active_names)
    available = [
        diagnostic
        for name in sorted(active)
        if (diagnostic := diagnostics.get(name)) is not None
        and name in forecasts
        and _valid_exact_forecast(forecasts[name], horizon)
    ]
    anchor = _stable_baseline(available, policy.baseline_strategy)
    if anchor is not None and not _passes_reliability_gate(policy, anchor):
        anchor = None
    if anchor is None:
        reliable = [
            item
            for item in available
            if item.eligible
            and item.successful_folds >= policy.min_successful_folds
            and not item.explosion
            and _passes_scaled_tail_gate(policy, item)
        ]
        if reliable:
            anchor = min(reliable, key=lambda item: _rank_key(item, policy.ranking_order))
    if anchor is None and policy.fallback_to_best_available and available:
        anchor = min(
            available,
            key=lambda item: (
                -item.successful_folds,
                not math.isfinite(item.median_joint_scaled_error),
                item.median_joint_scaled_error,
                item.recent_joint_scaled_error,
                item.name,
            ),
        )
    if anchor is None:
        raise ValueError("no finite protected Safe-Anchor forecast is available")
    return SelectionDecision(
        mode="single",
        selected=(anchor.name,),
        weights=(1.0,),
        forecast=tuple(float(value) for value in forecasts[anchor.name]),
        confidence=0.0,
        reason_codes=("protected_safe_anchor", fallback_reason),
        rejected={},
        baseline_name=anchor.name,
        considered_candidates=(anchor.name,),
    )


def _valid_exact_forecast(values: Sequence[float], horizon: int) -> bool:
    try:
        return (
            not isinstance(values, (str, bytes))
            and len(values) == horizon
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in values
            )
        )
    except (ArithmeticError, TypeError, ValueError):
        return False


def hindcast_cache_key(
    task: Task,
    candidate_name: str,
    family: str,
    config: HindcastConfig,
    *,
    screening_policy_hash: str = "",
    runtime_settings: Mapping[str, object] | None = None,
) -> str:
    # Screening decides whether a candidate is eligible for a task; it cannot
    # change that candidate's history-only forecast or hindcast diagnostics.
    # Keep the argument for artifact/API compatibility, but deliberately leave
    # it out of the reusable numerical identity.
    del screening_policy_hash
    payload = {
        "schema": 2,
        "history": task.history,
        "horizon": task.horizon,
        "frequency": task.frequency,
        "candidate": candidate_name,
        "family": family,
        "config": asdict(config),
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
        except Exception as error:
            folds.append(HindcastFold(
                train_end=train_end,
                validation_end=validation_end,
                status="failed",
                truth=truth,
                detail=f"{type(error).__name__}: {error}"[:200],
            ))

    long_horizon_fold = None
    long_horizon_coverage = 0.0
    if config.long_horizon_audit:
        long_horizon_fold, long_horizon_coverage = _long_horizon_audit_fold(
            task, name, runner, history, tuple(folds)
        )

    successful = tuple(fold for fold in folds if fold.status == "success")
    if len(successful) < config.min_successful_folds:
        return _summarize(
            name, family, tuple(folds), False, "insufficient_successful_folds",
            config, task, screening_policy_hash, runtime_settings,
            long_horizon_fold=long_horizon_fold,
            long_horizon_coverage=long_horizon_coverage,
        )
    return _summarize(
        name, family, tuple(folds), True, "ok", config, task,
        screening_policy_hash, runtime_settings,
        long_horizon_fold=long_horizon_fold,
        long_horizon_coverage=long_horizon_coverage,
    )


def _long_horizon_audit_fold(
    task: Task,
    name: str,
    runner: CandidateRunner,
    history: tuple[float, ...],
    ranking_folds: tuple[HindcastFold, ...],
) -> tuple[HindcastFold, float]:
    target_horizon = int(task.horizon)
    audit_horizon = min(target_horizon, max(1, len(history) // 3))
    train_end = len(history) - audit_horizon
    coverage = audit_horizon / target_horizon
    for fold in ranking_folds:
        if (
            fold.train_end == train_end
            and fold.validation_end == len(history)
            and len(fold.truth) == audit_horizon
        ):
            return fold, coverage
    prefix = history[:train_end]
    truth = history[train_end:]
    try:
        raw = runner(name, prefix, audit_horizon, task.frequency)
        forecast = tuple(float(value) for value in raw)
        if len(forecast) != audit_horizon or not all(map(math.isfinite, forecast)):
            raise ValueError("candidate returned an invalid long-horizon forecast")
        return _score_fold(prefix, truth, forecast, train_end, len(history)), coverage
    except Exception as error:
        return HindcastFold(
            train_end=train_end,
            validation_end=len(history),
            status="failed",
            truth=truth,
            detail=f"{type(error).__name__}: {error}"[:200],
        ), coverage


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
    profile=None,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    history: Sequence[float] = (),
    conditioned_names: Sequence[str] = (),
) -> SelectionDecision:
    if policy.baseline_strategy in {
        "protected_single_tsfm",
        "protected_tsfm_portfolio",
        "protected_joint_residual",
        "protected_topk_single_tsfm",
        "protected_topk_tsfm_portfolio",
        "protected_topk_joint_residual",
    }:
        parent_strategy = (
            "minimax_tsfm"
            if policy.baseline_strategy.startswith("protected_topk_")
            else "toto_first"
        )
        parent = select_numerical_forecast(
            replace(policy, baseline_strategy=parent_strategy),
            profile=profile,
            active_names=active_names,
            diagnostics=diagnostics,
            forecasts=forecasts,
            history=history,
            conditioned_names=conditioned_names,
        )
        reference = _selection_diagnostic(parent, diagnostics, policy=policy)
        if reference is None:
            return _protected_parent(parent)
        if policy.baseline_strategy in {
            "protected_single_tsfm",
            "protected_topk_single_tsfm",
        }:
            return _protected_single_tsfm_challenger(
                policy,
                parent,
                reference,
                active_names=active_names,
                diagnostics=diagnostics,
                forecasts=forecasts,
            )
        portfolio = _protected_multi_tsfm_challenger(
            policy,
            parent,
            reference,
            active_names=active_names,
            diagnostics=diagnostics,
            forecasts=forecasts,
        )
        if policy.baseline_strategy in {
            "protected_tsfm_portfolio",
            "protected_topk_tsfm_portfolio",
        }:
            return portfolio
        portfolio_reference = _selection_diagnostic(
            portfolio, diagnostics, policy=policy
        )
        if portfolio_reference is None:
            return portfolio
        return _protected_statistical_residual_overlay(
            policy,
            portfolio,
            portfolio_reference,
            active_names=active_names,
            diagnostics=diagnostics,
            forecasts=forecasts,
            history=history,
            conditioned_names=conditioned_names,
        )
    if policy.baseline_strategy in {
        "conservative_single_tsfm",
        "conservative_tsfm_portfolio",
        "conservative_tsfm_statistical",
        "conservative_joint_portfolio",
    }:
        anchor = _conservative_single_tsfm_decision(
            policy,
            active_names=active_names,
            diagnostics=diagnostics,
            forecasts=forecasts,
        )
        if policy.baseline_strategy == "conservative_single_tsfm":
            return anchor
        if policy.baseline_strategy == "conservative_tsfm_statistical":
            return _conservative_statistical_soft_overlay(
                policy,
                anchor,
                active_names=active_names,
                diagnostics=diagnostics,
                forecasts=forecasts,
                conditioned_names=conditioned_names,
            )
        portfolio = _conservative_multi_tsfm_portfolio(
            policy,
            anchor,
            active_names=active_names,
            diagnostics=diagnostics,
            forecasts=forecasts,
        )
        if policy.baseline_strategy == "conservative_tsfm_portfolio":
            return portfolio
        return _conservative_statistical_soft_overlay(
            policy,
            portfolio,
            active_names=active_names,
            diagnostics=diagnostics,
            forecasts=forecasts,
            conditioned_names=conditioned_names,
        )
    if (
        policy.baseline_strategy in {"conservative_tsfm", "conservative_combined"}
        and policy.tsfm_router_blend_weight > 0.0
    ):
        parent = select_numerical_forecast(
            replace(
                policy,
                baseline_strategy="toto_first",
                tsfm_router_blend_weight=0.0,
            ),
            profile=profile,
            active_names=active_names,
            diagnostics=diagnostics,
            forecasts=forecasts,
            history=history,
            conditioned_names=conditioned_names,
        )
        if policy.baseline_strategy == "conservative_tsfm":
            return _conservative_tsfm_soft_overlay(
                policy,
                parent,
                active_names=active_names,
                diagnostics=diagnostics,
                forecasts=forecasts,
            )
        return _conservative_statistical_soft_overlay(
            policy,
            parent,
            active_names=active_names,
            diagnostics=diagnostics,
            forecasts=forecasts,
            conditioned_names=conditioned_names,
        )

    active = set(active_names)
    rejected: dict[str, str] = {}
    available = [
        diagnostic
        for name in sorted(active)
        if name in forecasts and (diagnostic := diagnostics.get(name)) is not None
    ]
    baseline = _stable_baseline(available, policy.baseline_strategy)
    eligible: list[CandidateDiagnostics] = []
    for name in sorted(active):
        diagnostic = diagnostics.get(name)
        if diagnostic is None or name not in forecasts:
            rejected[name] = "missing_diagnostics_or_forecast"
        elif not diagnostic.eligible or diagnostic.successful_folds < policy.min_successful_folds:
            rejected[name] = "insufficient_hindcast_reliability"
        elif diagnostic.explosion or not _passes_scaled_tail_gate(policy, diagnostic):
            rejected[name] = "catastrophic_hindcast_tail"
        else:
            eligible.append(diagnostic)
    if not eligible:
        if baseline is not None:
            return SelectionDecision(
                mode="single",
                selected=(baseline.name,),
                weights=(1.0,),
                forecast=tuple(float(value) for value in forecasts[baseline.name]),
                confidence=0.0,
                reason_codes=("unverified_baseline_fallback",),
                rejected=rejected,
                baseline_name=baseline.name,
            )
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
                not math.isfinite(item.median_joint_scaled_error),
                item.median_joint_scaled_error,
                item.recent_joint_scaled_error,
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

    routed_tsfm_anchor = False
    if policy.baseline_strategy == "conservative_tsfm":
        baseline, routed_tsfm_anchor = _conservative_tsfm_anchor(
            policy,
            eligible,
            baseline,
            profile=profile,
        )

    front = _pareto_front(eligible)
    order = policy.ranking_order
    if policy.recent_regime_first:
        order = ("recent_joint_scaled_error",) + tuple(
            field for field in order if field != "recent_joint_scaled_error"
        )
    ranked = sorted(front, key=lambda item: _rank_key(item, order))
    best = ranked[0]
    baseline_protected = False
    baseline_is_eligible = baseline is not None and any(
        candidate.name == baseline.name for candidate in eligible
    )
    if baseline is not None and baseline_is_eligible and best.name != baseline.name:
        if not _passes_single_override(
            policy, best, baseline, profile=profile
        ):
            best = baseline
            baseline_protected = True
    best_forecast = tuple(float(value) for value in forecasts[best.name])
    chosen = (best.name,)
    weights = (1.0,)
    forecast = best_forecast
    mode = "single"
    arithmetic = SelectionArithmetic("leaf", candidate_name=best.name)
    reasons = ["reliability_gate", "pareto_front", f"best_{order[0]}"]
    if baseline_protected:
        reasons.append(
            "stable_baseline_protection"
            if baseline_is_eligible
            else "unverified_baseline_fallback"
        )
    if routed_tsfm_anchor:
        reasons.append("conservative_tsfm_router")

    combination_type = None
    if policy.ensemble_enabled and (baseline is None or baseline_is_eligible):
        proposal = _best_guarded_combination(
            policy,
            eligible,
            forecasts,
            history=history,
            baseline=baseline,
            conditioned_names=conditioned_names,
            profile=profile,
        )
        if proposal is not None:
            chosen, weights, forecast, combination_type, arithmetic = proposal
            mode = "combined"
            reasons.extend((
                "cross_family_combination",
                "majority_fold_improvement",
                "worst_fold_regret_gate",
                "baseline_protection",
            ))
            if set(chosen) & set(conditioned_names):
                reasons.append("task_conditioned_specialist")
        elif not _has_cross_family_pair(eligible):
            legacy = _best_validated_ensemble(policy, eligible, forecasts, best)
            if legacy is not None:
                chosen, weights, forecast = legacy
                mode = "ensemble"
                arithmetic = SelectionArithmetic(
                    "fmean",
                    inputs=tuple(
                        SelectionArithmetic("leaf", candidate_name=name)
                        for name in chosen
                    ),
                )
                reasons.extend(("diverse_members", "hindcast_blend_improvement"))

    all_ranked = sorted(eligible, key=lambda item: _rank_key(item, order))
    gap = 1.0
    if len(all_ranked) > 1:
        numerator = max(
            0.0,
            all_ranked[1].median_joint_scaled_error
            - all_ranked[0].median_joint_scaled_error,
        )
        gap = numerator / (1.0 + abs(all_ranked[0].median_joint_scaled_error))
    confidence = max(0.0, min(1.0, gap))
    return SelectionDecision(
        mode=mode,
        selected=chosen,
        weights=weights,
        forecast=forecast,
        confidence=confidence,
        reason_codes=tuple(reasons),
        rejected=rejected,
        combination_type=combination_type,
        baseline_name=baseline.name if baseline is not None else None,
        arithmetic=arithmetic,
    )


def _conservative_tsfm_soft_overlay(
    policy: DecisionPolicy,
    parent: SelectionDecision,
    *,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
) -> SelectionDecision:
    """Add a small reviewed TimesFM correction without replacing the parent decision."""
    if parent.mode != "single" or len(parent.selected) != 1:
        return parent
    active = set(active_names)
    anchor_name = parent.selected[0]
    if anchor_name == "timesfm_2_5":
        return parent
    anchor = diagnostics.get(anchor_name)
    toto = diagnostics.get("toto_2_0")
    timesfm = diagnostics.get("timesfm_2_5")
    if any(item is None for item in (anchor, toto, timesfm)):
        return parent
    assert anchor is not None and toto is not None and timesfm is not None
    if not {anchor_name, "toto_2_0", "timesfm_2_5"} <= active:
        return parent
    if any(
        not item.eligible
        or item.successful_folds < policy.min_successful_folds
        or item.explosion
        or not _passes_scaled_tail_gate(policy, item)
        for item in (anchor, toto, timesfm)
    ):
        return parent
    if not _passes_conservative_override(policy, timesfm, toto):
        return parent
    if not _strictly_aligned_successful_folds(
        (anchor, toto, timesfm), minimum_folds=policy.min_successful_folds
    ):
        return parent

    selected = None
    for challenger_weight in _adaptive_overlay_weights(
        policy.tsfm_router_blend_weight
    ):
        anchor_weight = 1.0 - challenger_weight
        blended_folds = _blend_folds(
            (anchor.fold_forecasts, timesfm.fold_forecasts),
            (anchor_weight, challenger_weight),
        )
        if not _passes_scaled_fold_acceptance(
            policy,
            blended_folds,
            anchor.fold_forecasts,
            anchor.fold_truths,
            required_wins=len(anchor.fold_forecasts),
            maximum_regret=0.0,
        ):
            continue

        audit_fold, audit_coverage = _combined_long_horizon_fold(
            anchor,
            timesfm,
            kind="weighted_blend",
            parameter=anchor_weight,
            clip_multiplier=policy.ensemble_correction_clip,
        )
        anchor_audit = anchor.long_horizon_fold
        if (
            audit_fold is None
            or anchor_audit is None
            or not _passes_scaled_fold_acceptance(
                policy,
                (audit_fold.forecast,),
                (anchor_audit.forecast,),
                (anchor_audit.truth,),
                required_wins=1,
                maximum_regret=0.0,
            )
        ):
            continue
        selected = (anchor_weight, challenger_weight)
        break

    if selected is None:
        return parent
    anchor_weight, challenger_weight = selected

    final = _blend_values(
        forecasts[anchor_name],
        forecasts["timesfm_2_5"],
        anchor_weight,
    )
    arithmetic = SelectionArithmetic(
        "blend",
        inputs=(
            parent.arithmetic or _implicit_selection_arithmetic(parent),
            SelectionArithmetic("leaf", candidate_name="timesfm_2_5"),
        ),
        weights=(anchor_weight, challenger_weight),
    )
    return SelectionDecision(
        mode="combined",
        selected=(anchor_name, "timesfm_2_5"),
        weights=(anchor_weight, challenger_weight),
        forecast=final,
        confidence=parent.confidence,
        reason_codes=(
            *parent.reason_codes,
            "conservative_tsfm_soft_overlay",
            "adaptive_strict_fold_weight",
        ),
        rejected=parent.rejected,
        combination_type="tsfm_shrinkage_overlay",
        baseline_name=parent.baseline_name,
        assumption_ids=parent.assumption_ids,
        assumption_kinds=parent.assumption_kinds,
        considered_candidates=parent.considered_candidates,
        arithmetic=arithmetic,
    )


def _adaptive_overlay_weights(maximum_weight: float) -> tuple[float, ...]:
    """Try the largest predeclared safe correction no greater than the policy cap."""
    return tuple(
        weight
        for weight in (0.25, 0.1, 0.05)
        if weight <= maximum_weight + 1e-12
    )


def _conservative_single_tsfm_decision(
    policy: DecisionPolicy,
    *,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
) -> SelectionDecision:
    """Choose only among TSFMs, with Toto as the protected reviewed anchor."""
    active = set(active_names)
    available = tuple(
        item
        for name in active
        if name in forecasts
        and (item := diagnostics.get(name)) is not None
        and item.family in {"tsfm", "foundation"}
    )
    if not available:
        raise ValueError("no active TSFM has a final forecast")
    anchor = _stable_baseline(available, "toto_first")
    if anchor is None:
        raise ValueError("no reviewed TSFM anchor is available")
    candidates = tuple(
        item
        for item in available
        if item.eligible
        and item.successful_folds >= policy.min_successful_folds
        and not item.explosion
        and _passes_scaled_tail_gate(policy, item)
    )
    anchor_reliable = anchor in candidates
    safe = [anchor]
    if anchor_reliable:
        safe.extend(
            item
            for item in candidates
            if item.name != anchor.name
            and _passes_conservative_override(policy, item, anchor)
        )
    selected = (
        min(safe, key=lambda item: _rank_key(item, policy.ranking_order))
        if anchor_reliable
        else anchor
    )
    reasons = [
        "conservative_single_tsfm",
        "reviewed_anchor_protection",
    ]
    reasons.append(
        "history_only_dual_metric_gate"
        if anchor_reliable
        else "unverified_tsfm_fallback"
    )
    return SelectionDecision(
        mode="single",
        selected=(selected.name,),
        weights=(1.0,),
        forecast=tuple(float(value) for value in forecasts[selected.name]),
        confidence=0.0,
        reason_codes=tuple(reasons),
        rejected={},
        baseline_name=anchor.name,
    )


def _protected_parent(parent: SelectionDecision) -> SelectionDecision:
    if "protected_parent_reference" in parent.reason_codes:
        return parent
    return replace(
        parent,
        reason_codes=(*parent.reason_codes, "protected_parent_reference"),
    )


def _protected_single_tsfm_challenger(
    policy: DecisionPolicy,
    parent: SelectionDecision,
    reference: CandidateDiagnostics,
    *,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
) -> SelectionDecision:
    """Replace Parent only with a TSFM that passes every history-only audit."""
    active = set(active_names)
    safe = tuple(
        item
        for name in active
        if name in forecasts
        and (item := diagnostics.get(name)) is not None
        and item.family in {"tsfm", "foundation"}
        and item.eligible
        and item.successful_folds >= policy.min_successful_folds
        and not item.explosion
        and _passes_scaled_tail_gate(policy, item)
        and _passes_conservative_override(policy, item, reference)
    )
    if not safe:
        return _protected_parent(parent)
    selected = min(safe, key=lambda item: _rank_key(item, policy.ranking_order))
    return SelectionDecision(
        mode="single",
        selected=(selected.name,),
        weights=(1.0,),
        forecast=tuple(float(value) for value in forecasts[selected.name]),
        confidence=parent.confidence,
        reason_codes=(
            *parent.reason_codes,
            "protected_parent_reference",
            "protected_single_tsfm_challenger",
            "dual_metric_tail_gate",
        ),
        rejected=parent.rejected,
        baseline_name=parent.baseline_name,
        assumption_ids=parent.assumption_ids,
        assumption_kinds=parent.assumption_kinds,
        considered_candidates=parent.considered_candidates,
    )


def _conservative_multi_tsfm_portfolio(
    policy: DecisionPolicy,
    parent: SelectionDecision,
    *,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    reference: CandidateDiagnostics | None = None,
) -> SelectionDecision:
    """Search a bounded Top-3 TSFM portfolio and keep the single-model parent on doubt."""
    if reference is None:
        if parent.mode != "single" or len(parent.selected) != 1:
            return parent
        anchor = diagnostics.get(parent.selected[0])
        if anchor is None or anchor.family not in {"tsfm", "foundation"}:
            return parent
    else:
        anchor = reference
    active = set(active_names)
    pool = sorted(
        (
            item
            for name in active
            if name in forecasts
            and (item := diagnostics.get(name)) is not None
            and item.family in {"tsfm", "foundation"}
            and item.eligible
            and item.successful_folds >= policy.min_successful_folds
            and not item.explosion
            and _passes_scaled_tail_gate(policy, item)
        ),
        key=lambda item: _rank_key(item, policy.ranking_order),
    )[:3]
    if len(pool) < 2:
        return parent
    if not _strictly_aligned_successful_folds(
        (anchor,), minimum_folds=policy.min_successful_folds
    ):
        return parent

    proposals: list[
        tuple[
            float,
            float,
            float,
            str,
            tuple[str, ...],
            tuple[float, ...],
            tuple[tuple[float, ...], ...],
            tuple[float, ...],
            HindcastFold,
            float,
        ]
    ] = []
    for left, right in itertools.combinations(pool, 2):
        if not _strictly_aligned_successful_folds(
            (anchor, left, right), minimum_folds=policy.min_successful_folds
        ):
            continue
        if pairwise_diversity(left.fold_forecasts, right.fold_forecasts) \
                < policy.ensemble_min_diversity:
            continue
        for left_weight in (0.5, 0.75, 0.9):
            weights = (left_weight, 1.0 - left_weight)
            candidate_folds = _blend_folds(
                (left.fold_forecasts, right.fold_forecasts), weights
            )
            audit, coverage = _multi_member_long_horizon_fold(
                (left, right), weights=weights, kind="weighted"
            )
            if audit is None or not _passes_tsfm_portfolio_gate(
                policy, candidate_folds, audit, coverage, anchor
            ):
                continue
            final = _weighted_values(
                (forecasts[left.name], forecasts[right.name]), weights
            )
            proposals.append(_tsfm_portfolio_proposal(
                candidate_folds,
                audit,
                coverage,
                anchor,
                kind="tsfm_weighted_portfolio",
                names=(left.name, right.name),
                weights=weights,
                final=final,
            ))

    if len(pool) == 3 and _strictly_aligned_successful_folds(
        (anchor, *pool), minimum_folds=policy.min_successful_folds
    ):
        diversity = max(
            pairwise_diversity(left.fold_forecasts, right.fold_forecasts)
            for left, right in itertools.combinations(pool, 2)
        )
        if diversity >= policy.ensemble_min_diversity:
            candidate_folds = _median_folds(
                tuple(item.fold_forecasts for item in pool)
            )
            weights = (1.0 / 3.0,) * 3
            audit, coverage = _multi_member_long_horizon_fold(
                tuple(pool), weights=weights, kind="median"
            )
            if audit is not None and _passes_tsfm_portfolio_gate(
                policy, candidate_folds, audit, coverage, anchor
            ):
                final = _median_values(tuple(forecasts[item.name] for item in pool))
                proposals.append(_tsfm_portfolio_proposal(
                    candidate_folds,
                    audit,
                    coverage,
                    anchor,
                    kind="tsfm_median_portfolio",
                    names=tuple(item.name for item in pool),
                    weights=weights,
                    final=final,
                ))

    if not proposals:
        return parent
    proposal = min(proposals, key=lambda item: item[:6])
    _, _, _, kind, names, weights, _, final, _, _ = proposal
    operation = "median" if kind == "tsfm_median_portfolio" else "weighted"
    arithmetic = SelectionArithmetic(
        operation,
        inputs=tuple(
            SelectionArithmetic("leaf", candidate_name=name) for name in names
        ),
        weights=weights if operation == "weighted" else (),
    )
    return SelectionDecision(
        mode="combined",
        selected=names,
        weights=weights,
        forecast=final,
        confidence=parent.confidence,
        reason_codes=(
            *parent.reason_codes,
            "conservative_multi_tsfm_portfolio",
            "top3_history_only_search",
            "dual_metric_tail_gate",
        ),
        rejected=parent.rejected,
        combination_type=kind,
        baseline_name=parent.baseline_name,
        assumption_ids=parent.assumption_ids,
        assumption_kinds=parent.assumption_kinds,
        considered_candidates=parent.considered_candidates,
        arithmetic=arithmetic,
    )


def _protected_multi_tsfm_challenger(
    policy: DecisionPolicy,
    parent: SelectionDecision,
    reference: CandidateDiagnostics,
    *,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
) -> SelectionDecision:
    proposal = _conservative_multi_tsfm_portfolio(
        policy,
        parent,
        active_names=active_names,
        diagnostics=diagnostics,
        forecasts=forecasts,
        reference=reference,
    )
    if proposal == parent:
        return _protected_parent(parent)
    kind = {
        "tsfm_weighted_portfolio": "protected_tsfm_weighted_portfolio",
        "tsfm_median_portfolio": "protected_tsfm_median_portfolio",
    }.get(proposal.combination_type, proposal.combination_type)
    return replace(
        proposal,
        combination_type=kind,
        reason_codes=(
            *proposal.reason_codes,
            "protected_parent_reference",
            "protected_multi_tsfm_challenger",
        ),
    )


def _tsfm_portfolio_proposal(
    candidate_folds: tuple[tuple[float, ...], ...],
    audit: HindcastFold,
    coverage: float,
    anchor: CandidateDiagnostics,
    *,
    kind: str,
    names: tuple[str, ...],
    weights: tuple[float, ...],
    final: tuple[float, ...],
):
    scores = _fold_scores(
        candidate_folds, anchor.fold_truths, _fold_mase_scales(anchor)
    )
    srmses = tuple(
        float(drcik_point_metrics(list(truth), list(forecast))["srmse"])
        for forecast, truth in zip(candidate_folds, anchor.fold_truths, strict=True)
    )
    return (
        statistics.median(scores),
        statistics.median(srmses),
        max(scores),
        kind,
        names,
        weights,
        candidate_folds,
        final,
        audit,
        coverage,
    )


def _passes_tsfm_portfolio_gate(
    policy: DecisionPolicy,
    candidate_folds: Sequence[Sequence[float]],
    audit: HindcastFold,
    coverage: float,
    anchor: CandidateDiagnostics,
) -> bool:
    if coverage + 1e-12 < policy.long_horizon_min_coverage:
        return False
    reference_audit = anchor.long_horizon_fold
    if (
        reference_audit is None
        or reference_audit.status != "success"
        or audit.status != "success"
        or audit.train_end != reference_audit.train_end
        or audit.validation_end != reference_audit.validation_end
        or audit.truth != reference_audit.truth
    ):
        return False
    minimum = policy.tsfm_router_min_improvement
    maximum_regret = policy.long_horizon_max_regret
    if not _passes_scaled_fold_acceptance(
        policy,
        candidate_folds,
        anchor.fold_forecasts,
        anchor.fold_truths,
        minimum_improvement=minimum,
    ):
        return False
    pairs = tuple(zip(
        candidate_folds,
        anchor.fold_forecasts,
        anchor.fold_truths,
        strict=True,
    ))
    if not pairs:
        return False
    candidate_smae = []
    reference_smae = []
    candidate_srmse = []
    reference_srmse = []
    for candidate, reference, truth in pairs:
        candidate_metric = drcik_point_metrics(list(truth), list(candidate))
        reference_metric = drcik_point_metrics(list(truth), list(reference))
        if (
            bool(candidate_metric["smae_clipped"])
            and not bool(reference_metric["smae_clipped"])
        ) or (
            bool(candidate_metric["srmse_clipped"])
            and not bool(reference_metric["srmse_clipped"])
        ):
            return False
        c_smae = float(candidate_metric["smae"])
        r_smae = float(reference_metric["smae"])
        c_srmse = float(candidate_metric["srmse"])
        r_srmse = float(reference_metric["srmse"])
        if (
            c_smae > r_smae * (1.0 + maximum_regret) + 1e-12
            or c_srmse > r_srmse * (1.0 + maximum_regret) + 1e-12
        ):
            return False
        candidate_smae.append(c_smae)
        reference_smae.append(r_smae)
        candidate_srmse.append(c_srmse)
        reference_srmse.append(r_srmse)
    if not (
        statistics.median(candidate_smae)
        < statistics.median(reference_smae) * (1.0 - minimum)
        and statistics.median(candidate_srmse)
        < statistics.median(reference_srmse) * (1.0 - minimum)
    ):
        return False
    audit_candidate = drcik_point_metrics(list(audit.truth), list(audit.forecast))
    audit_reference = drcik_point_metrics(
        list(reference_audit.truth), list(reference_audit.forecast)
    )
    return bool(
        _passes_scaled_fold_acceptance(
            policy,
            (audit.forecast,),
            (reference_audit.forecast,),
            (reference_audit.truth,),
            required_wins=1,
            maximum_regret=maximum_regret,
            minimum_improvement=minimum,
        )
        and float(audit_candidate["smae"])
        < float(audit_reference["smae"]) * (1.0 - minimum)
        and float(audit_candidate["srmse"])
        < float(audit_reference["srmse"]) * (1.0 - minimum)
        and not (
            bool(audit_candidate["smae_clipped"])
            and not bool(audit_reference["smae_clipped"])
        )
        and not (
            bool(audit_candidate["srmse_clipped"])
            and not bool(audit_reference["srmse_clipped"])
        )
    )


def _multi_member_long_horizon_fold(
    members: Sequence[CandidateDiagnostics],
    *,
    weights: Sequence[float],
    kind: str,
) -> tuple[HindcastFold | None, float]:
    audits = tuple(member.long_horizon_fold for member in members)
    coverage = min((member.long_horizon_coverage for member in members), default=0.0)
    if not audits or any(audit is None for audit in audits):
        return None, coverage
    reference = audits[0]
    assert reference is not None
    if any(
        audit is None
        or audit.status != "success"
        or audit.train_end != reference.train_end
        or audit.validation_end != reference.validation_end
        or audit.truth != reference.truth
        or len(audit.forecast) != len(reference.forecast)
        for audit in audits
    ):
        return None, coverage
    all_forecasts = tuple(audit.forecast for audit in audits if audit is not None)
    forecast = (
        _weighted_values(all_forecasts, weights)
        if kind == "weighted"
        else _median_values(all_forecasts)
    )
    return replace(
        reference, forecast=forecast, **_scaled_fold_fields(reference.truth, forecast)
    ), coverage


def _weighted_values(
    values: Sequence[Sequence[float]], weights: Sequence[float]
) -> tuple[float, ...]:
    if not values or len(values) != len(weights):
        raise ValueError("portfolio values and weights must be nonempty and aligned")
    if any(len(item) != len(values[0]) for item in values):
        raise ValueError("portfolio members returned different forecast horizons")
    return tuple(
        sum(weight * float(value) for weight, value in zip(weights, timestep, strict=True))
        for timestep in zip(*values, strict=True)
    )


def _fmean_values(values: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not values or any(len(item) != len(values[0]) for item in values):
        raise ValueError("mean portfolio members must be nonempty and aligned")
    return tuple(statistics.fmean(timestep) for timestep in zip(*values, strict=True))


def _median_values(values: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not values or any(len(item) != len(values[0]) for item in values):
        raise ValueError("median portfolio members must be nonempty and aligned")
    return tuple(statistics.median(timestep) for timestep in zip(*values, strict=True))


def _median_folds(
    all_folds: Sequence[Sequence[Sequence[float]]],
) -> tuple[tuple[float, ...], ...]:
    if not all_folds or any(len(folds) != len(all_folds[0]) for folds in all_folds):
        return ()
    return tuple(
        _median_values(tuple(folds[index] for folds in all_folds))
        for index in range(len(all_folds[0]))
    )


def _conservative_statistical_soft_overlay(
    policy: DecisionPolicy,
    parent: SelectionDecision,
    *,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    conditioned_names: Sequence[str],
) -> SelectionDecision:
    """Add one bounded statistical specialist only when every history audit improves."""
    if not parent.selected:
        return parent
    anchor = _selection_diagnostic(parent, diagnostics)
    if (
        anchor is None
        or anchor.family not in {"tsfm", "foundation"}
        or any(name not in forecasts for name in parent.selected)
    ):
        return parent
    if (
        not anchor.eligible
        or anchor.successful_folds < policy.min_successful_folds
        or anchor.explosion
        or not _passes_scaled_tail_gate(policy, anchor)
    ):
        return parent

    active = set(active_names)
    conditioned = set(conditioned_names)
    ranked = sorted(
        (
            item
            for name in active
            if name in forecasts
            and (item := diagnostics.get(name)) is not None
            and item.family == "statistical"
            and item.name in conditioned
            and item.eligible
            and item.successful_folds >= policy.min_successful_folds
            and not item.explosion
            and _passes_scaled_tail_gate(policy, item)
        ),
        key=lambda item: _rank_key(item, policy.ranking_order),
    )
    specialists = list(ranked[:6])
    if not specialists:
        return parent
    if not _strictly_aligned_successful_folds(
        (anchor,), minimum_folds=policy.min_successful_folds
    ):
        return parent

    proposals = []
    for specialist in specialists:
        if not _strictly_aligned_successful_folds(
            (anchor, specialist), minimum_folds=policy.min_successful_folds
        ):
            continue
        for challenger_weight in _adaptive_overlay_weights(
            policy.tsfm_router_blend_weight
        ):
            anchor_weight = 1.0 - challenger_weight
            blended_folds = _blend_folds(
                (anchor.fold_forecasts, specialist.fold_forecasts),
                (anchor_weight, challenger_weight),
            )
            if not _passes_scaled_fold_acceptance(
                policy,
                blended_folds,
                anchor.fold_forecasts,
                anchor.fold_truths,
                required_wins=len(anchor.fold_forecasts),
                maximum_regret=0.0,
                minimum_improvement=policy.tsfm_router_min_improvement,
            ):
                continue

            audit_fold, audit_coverage = _combined_long_horizon_fold(
                anchor,
                specialist,
                kind="weighted_blend",
                parameter=anchor_weight,
                clip_multiplier=policy.ensemble_correction_clip,
            )
            anchor_audit = anchor.long_horizon_fold
            if (
                audit_fold is None
                or anchor_audit is None
                or not _passes_scaled_fold_acceptance(
                    policy,
                    (audit_fold.forecast,),
                    (anchor_audit.forecast,),
                    (anchor_audit.truth,),
                    required_wins=1,
                    maximum_regret=0.0,
                    minimum_improvement=policy.tsfm_router_min_improvement,
                )
            ):
                continue
                blended_srmse = tuple(
                float(drcik_point_metrics(list(truth), list(forecast))["srmse"])
                    for forecast, truth in zip(
                        blended_folds, anchor.fold_truths, strict=True
                    )
                )
                blended_scores = _fold_scores(
                    blended_folds, anchor.fold_truths
                )
                proposals.append((
                statistics.median(blended_scores),
                statistics.median(blended_srmse),
                max(blended_scores),
                challenger_weight,
                specialist.name,
                anchor_weight,
            ))

    if not proposals:
        return parent
    _, _, _, challenger_weight, specialist_name, anchor_weight = min(proposals)
    final = _blend_values(
        parent.forecast, forecasts[specialist_name], anchor_weight
    )
    arithmetic = SelectionArithmetic(
        "blend",
        inputs=(
            parent.arithmetic or _implicit_selection_arithmetic(parent),
            SelectionArithmetic("leaf", candidate_name=specialist_name),
        ),
        weights=(anchor_weight, challenger_weight),
    )
    reasons = [
        *parent.reason_codes,
        "conservative_statistical_soft_overlay",
        "strict_fold_and_long_audit",
    ]
    if specialist_name in conditioned:
        reasons.append("task_conditioned_statistical_specialist")
    selected = (*parent.selected, specialist_name)
    weights = (
        *(float(weight) * anchor_weight for weight in parent.weights),
        challenger_weight,
    )
    joint = len(parent.selected) > 1
    return SelectionDecision(
        mode="combined",
        selected=selected,
        weights=weights,
        forecast=final,
        confidence=parent.confidence,
        reason_codes=tuple(reasons),
        rejected=parent.rejected,
        combination_type=(
            "joint_tsfm_statistical_portfolio"
            if joint
            else "statistical_shrinkage_overlay"
        ),
        baseline_name=parent.baseline_name,
        assumption_ids=parent.assumption_ids,
        assumption_kinds=parent.assumption_kinds,
        considered_candidates=parent.considered_candidates,
        arithmetic=arithmetic,
    )


def _protected_statistical_residual_overlay(
    policy: DecisionPolicy,
    parent: SelectionDecision,
    reference: CandidateDiagnostics,
    *,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    history: Sequence[float],
    conditioned_names: Sequence[str],
) -> SelectionDecision:
    """Apply one clipped Statistical residual after strict 3+1 fold evidence."""
    active = set(active_names)
    conditioned = set(conditioned_names)
    specialists = sorted(
        (
            item
            for name in active
            if name in forecasts
            and name not in parent.selected
            and name in conditioned
            and (item := diagnostics.get(name)) is not None
            and item.family == "statistical"
            and item.eligible
            and item.successful_folds >= policy.min_successful_folds
            and not item.explosion
            and _passes_scaled_tail_gate(policy, item)
        ),
        key=lambda item: _rank_key(item, policy.ranking_order),
    )[:6]
    if not specialists or not _strictly_aligned_successful_folds(
        (reference,), minimum_folds=policy.min_successful_folds
    ):
        return _protected_parent(parent)

    proposals = []
    scales = _fold_correction_scales(reference)
    for specialist in specialists:
        if not _strictly_aligned_successful_folds(
            (reference, specialist), minimum_folds=policy.min_successful_folds
        ):
            continue
        for strength in policy.ensemble_residual_strengths:
            folds = _residual_folds(
                reference.fold_forecasts,
                specialist.fold_forecasts,
                strength,
                policy.ensemble_correction_clip,
                scales,
            )
            audit, coverage = _combined_long_horizon_fold(
                reference,
                specialist,
                kind="residual_correction",
                parameter=strength,
                clip_multiplier=policy.ensemble_correction_clip,
            )
            if audit is None or not _passes_tsfm_portfolio_gate(
                policy, folds, audit, coverage, reference
            ):
                continue
            scores = _fold_scores(folds, reference.fold_truths, scales)
            srmses = tuple(
                float(drcik_point_metrics(list(truth), list(forecast))["srmse"])
                for forecast, truth in zip(folds, reference.fold_truths, strict=True)
            )
            proposals.append((
                statistics.median(scores),
                statistics.median(srmses),
                max(scores),
                strength,
                specialist.name,
            ))
    if not proposals:
        return _protected_parent(parent)

    _, _, _, strength, specialist_name = min(proposals)
    final_scale = (
        _naive_absolute_scale(tuple(float(value) for value in history))
        if history
        else _forecast_scale(parent.forecast)
    )
    final = _residual_values(
        parent.forecast,
        forecasts[specialist_name],
        strength,
        policy.ensemble_correction_clip,
        final_scale,
    )
    arithmetic = SelectionArithmetic(
        "residual",
        inputs=(
            parent.arithmetic or _implicit_selection_arithmetic(parent),
            SelectionArithmetic("leaf", candidate_name=specialist_name),
        ),
        strength=strength,
        clip_multiplier=policy.ensemble_correction_clip,
        scale=final_scale,
    )
    selected = (*parent.selected, specialist_name)
    weights = (
        *(float(weight) * (1.0 - strength) for weight in parent.weights),
        strength,
    )
    return SelectionDecision(
        mode="combined",
        selected=selected,
        weights=weights,
        forecast=final,
        confidence=parent.confidence,
        reason_codes=(
            *parent.reason_codes,
            "protected_parent_reference",
            "bounded_statistical_residual",
            "dual_metric_tail_gate",
        ),
        rejected=parent.rejected,
        combination_type=(
            "protected_joint_tsfm_statistical_residual"
            if len(parent.selected) > 1
            else "protected_statistical_residual"
        ),
        baseline_name=parent.baseline_name,
        assumption_ids=parent.assumption_ids,
        assumption_kinds=parent.assumption_kinds,
        considered_candidates=parent.considered_candidates,
        arithmetic=arithmetic,
    )


def _selection_diagnostic(
    decision: SelectionDecision,
    diagnostics: Mapping[str, CandidateDiagnostics],
    *,
    policy: DecisionPolicy | None = None,
) -> CandidateDiagnostics | None:
    """Materialize a TSFM SelectionDecision as a history-only diagnostic view."""
    members = tuple(diagnostics.get(name) for name in decision.selected)
    if not members or any(member is None for member in members):
        return None
    concrete = tuple(member for member in members if member is not None)
    if len(concrete) == 1:
        return concrete[0]
    if len(decision.weights) != len(concrete) or not _strictly_aligned_successful_folds(
        concrete, minimum_folds=1
    ):
        return None
    if decision.combination_type in {
        "tsfm_median_portfolio", "protected_tsfm_median_portfolio"
    }:
        fold_forecasts = _median_folds(
            tuple(member.fold_forecasts for member in concrete)
        )
        audit, coverage = _multi_member_long_horizon_fold(
            concrete, weights=decision.weights, kind="median"
        )
    elif decision.combination_type in {
        "residual_correction",
        "protected_statistical_residual",
        "protected_joint_tsfm_statistical_residual",
    }:
        if policy is None or len(concrete) != 2:
            return None
        strength = float(decision.weights[-1])
        fold_forecasts = _residual_folds(
            concrete[0].fold_forecasts,
            concrete[1].fold_forecasts,
            strength,
            policy.ensemble_correction_clip,
            _fold_correction_scales(concrete[0]),
        )
        audit, coverage = _combined_long_horizon_fold(
            concrete[0],
            concrete[1],
            kind="residual_correction",
            parameter=strength,
            clip_multiplier=policy.ensemble_correction_clip,
        )
    else:
        fold_forecasts = _blend_folds(
            tuple(member.fold_forecasts for member in concrete), decision.weights
        )
        audit, coverage = _multi_member_long_horizon_fold(
            concrete, weights=decision.weights, kind="weighted"
        )
    if not fold_forecasts or audit is None:
        return None
    reference = concrete[0]
    folds = tuple(
        replace(
            fold,
            forecast=fold_forecasts[index],
            **_scaled_fold_fields(fold.truth, fold_forecasts[index]),
        )
        for index, fold in enumerate(reference.folds)
    )
    scales = _fold_mase_scales(reference)
    scores = _fold_scores(fold_forecasts, reference.fold_truths, scales)
    if not scores:
        return None
    return replace(
        reference,
        name="portfolio:" + "+".join(decision.selected),
        family=(
            "foundation"
            if all(member.family in {"tsfm", "foundation"} for member in concrete)
            else "combined"
        ),
        folds=folds,
        successful_folds=len(folds),
        eligible=True,
        reason_code="ok",
        median_mase=statistics.median(scores),
        recent_mase=scores[-1],
        worst_mase=max(scores),
        mase_mad=statistics.median(abs(score - statistics.median(scores)) for score in scores),
        fold_forecasts=fold_forecasts,
        fold_truths=reference.fold_truths,
        explosion=any(not math.isfinite(value) for fold in fold_forecasts for value in fold),
        long_horizon_fold=audit,
        long_horizon_coverage=coverage,
        **_scaled_candidate_summary(folds),
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
        mase_scale=absolute_scale,
        **_scaled_fold_fields(truth, forecast),
    )


def _scaled_fold_fields(
    truth: Sequence[float], forecast: Sequence[float]
) -> dict[str, float | bool]:
    metrics = drcik_point_metrics(list(truth), list(forecast))
    return {
        "smae": float(metrics["smae"]),
        "srmse": float(metrics["srmse"]),
        "smae_raw": float(metrics["smae_raw"]),
        "srmse_raw": float(metrics["srmse_raw"]),
        "smae_clipped": bool(metrics["smae_clipped"]),
        "srmse_clipped": bool(metrics["srmse_clipped"]),
    }


def _scaled_candidate_summary(
    folds: Sequence[HindcastFold],
) -> dict[str, float]:
    successful = tuple(fold for fold in folds if fold.status == "success")

    def values(field: str) -> list[float]:
        return [float(value) for fold in successful if (value := getattr(fold, field)) is not None]

    smaes = values("smae")
    srmses = values("srmse")
    joints = [joint_scaled_error(smae, srmse) for smae, srmse in zip(smaes, srmses)]
    def median_mad(items: list[float]) -> tuple[float, float]:
        if not items:
            return math.inf, math.inf
        median = statistics.median(items)
        return median, statistics.median(abs(value - median) for value in items)

    median_smae, smae_mad = median_mad(smaes)
    median_srmse, srmse_mad = median_mad(srmses)
    return {
        "median_joint_scaled_error": _median_or_inf(joints),
        "recent_joint_scaled_error": joints[-1] if joints else math.inf,
        "worst_joint_scaled_error": max(joints, default=math.inf),
        "median_smae": median_smae,
        "recent_smae": smaes[-1] if smaes else math.inf,
        "worst_smae": max(smaes, default=math.inf),
        "smae_mad": smae_mad,
        "median_srmse": median_srmse,
        "recent_srmse": srmses[-1] if srmses else math.inf,
        "worst_srmse": max(srmses, default=math.inf),
        "srmse_mad": srmse_mad,
        "worst_smae_raw": max(values("smae_raw"), default=math.inf),
        "worst_srmse_raw": max(values("srmse_raw"), default=math.inf),
    }


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
    *,
    long_horizon_fold: HindcastFold | None = None,
    long_horizon_coverage: float = 0.0,
) -> CandidateDiagnostics:
    successful = tuple(fold for fold in folds if fold.status == "success")

    def values(field: str) -> list[float]:
        return [float(value) for fold in successful if (value := getattr(fold, field)) is not None]

    mases = values("mase")
    smaes = values("smae")
    srmses = values("srmse")
    smaes_raw = values("smae_raw")
    srmses_raw = values("srmse_raw")
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
        explosion=any(
            value > config.catastrophic_smae_raw
            for value in smaes_raw
        ) or any(
            value > config.catastrophic_srmse_raw
            for value in srmses_raw
        ),
        fold_forecasts=tuple(fold.forecast for fold in successful),
        fold_truths=tuple(fold.truth for fold in successful),
        cache_key=hindcast_cache_key(
            task, name, family, config,
            screening_policy_hash=screening_policy_hash,
            runtime_settings=runtime_settings,
        ),
        long_horizon_fold=long_horizon_fold,
        long_horizon_coverage=long_horizon_coverage,
        median_joint_scaled_error=_median_or_inf([
            joint_scaled_error(float(fold.smae), float(fold.srmse))
            for fold in successful
            if fold.smae is not None and fold.srmse is not None
        ]),
        recent_joint_scaled_error=(
            joint_scaled_error(float(successful[-1].smae), float(successful[-1].srmse))
            if successful and successful[-1].smae is not None and successful[-1].srmse is not None
            else math.inf
        ),
        worst_joint_scaled_error=max((
            joint_scaled_error(float(fold.smae), float(fold.srmse))
            for fold in successful
            if fold.smae is not None and fold.srmse is not None
        ), default=math.inf),
        median_smae=_median_or_inf(smaes),
        recent_smae=float(successful[-1].smae) if successful and successful[-1].smae is not None else math.inf,
        worst_smae=max(smaes, default=math.inf),
        smae_mad=(
            statistics.median(abs(value - statistics.median(smaes)) for value in smaes)
            if smaes else math.inf
        ),
        median_srmse=_median_or_inf(srmses),
        recent_srmse=float(successful[-1].srmse) if successful and successful[-1].srmse is not None else math.inf,
        worst_srmse=max(srmses, default=math.inf),
        srmse_mad=(
            statistics.median(abs(value - statistics.median(srmses)) for value in srmses)
            if srmses else math.inf
        ),
        worst_smae_raw=max(smaes_raw, default=math.inf),
        worst_srmse_raw=max(srmses_raw, default=math.inf),
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
    dimensions = (
        "median_smae", "median_srmse", "recent_smae", "recent_srmse",
        "worst_smae", "worst_srmse",
    )
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
        raw = getattr(candidate, field)
        value = float(raw) if raw is not None else math.inf
        if not math.isfinite(value):
            value = math.inf
        values.append(abs(value) if field == "normalized_bias" else value)
    return (*values, candidate.name)


def _stable_baseline(
    eligible: Sequence[CandidateDiagnostics],
    strategy: str = "toto_first",
) -> CandidateDiagnostics | None:
    by_name = {candidate.name: candidate for candidate in eligible}
    if strategy == "minimax_tsfm":
        reviewed = [
            by_name[name]
            for name in ("toto_2_0", "timesfm_2_5")
            if name in by_name
        ]
        if reviewed:
            return min(
                reviewed,
                key=lambda item: (
                    item.worst_joint_scaled_error,
                    item.median_joint_scaled_error,
                    item.recent_joint_scaled_error,
                    item.name,
                ),
            )
    for name in ("toto_2_0", "timesfm_2_5"):
        if name in by_name:
            return by_name[name]
    tsfm = [candidate for candidate in eligible if candidate.family in {"tsfm", "foundation"}]
    return min(tsfm, key=lambda item: _rank_key(item, (
        "median_joint_scaled_error", "worst_joint_scaled_error"
    ))) \
        if tsfm else None


def _conservative_tsfm_anchor(
    policy: DecisionPolicy,
    eligible: Sequence[CandidateDiagnostics],
    fallback: CandidateDiagnostics | None,
    *,
    profile=None,
) -> tuple[CandidateDiagnostics | None, bool]:
    """Keep Toto unless TimesFM passes the same four-comparison override gate."""
    by_name = {candidate.name: candidate for candidate in eligible}
    toto = by_name.get("toto_2_0")
    timesfm = by_name.get("timesfm_2_5")
    if toto is None:
        return (timesfm or fallback), timesfm is not None
    if timesfm is None:
        return toto, False
    del profile
    if _passes_conservative_override(policy, timesfm, toto):
        return timesfm, True
    return toto, False


def _has_cross_family_pair(eligible: Sequence[CandidateDiagnostics]) -> bool:
    families = {candidate.family for candidate in eligible}
    return bool(families & {"tsfm", "foundation"}) and "statistical" in families


def _passes_single_override(
    policy: DecisionPolicy,
    challenger: CandidateDiagnostics,
    baseline: CandidateDiagnostics,
    *,
    profile=None,
) -> bool:
    if (
        policy.baseline_strategy == "conservative_tsfm"
        and (
            baseline.name == "timesfm_2_5"
            or (baseline.name == "toto_2_0" and challenger.name == "timesfm_2_5")
        )
    ):
        return _passes_conservative_override(policy, challenger, baseline)
    if not _aligned_folds((challenger, baseline)):
        return False
    minimum_improvement = policy.ensemble_min_improvement
    if challenger.family == "combined":
        minimum_improvement = max(0.05, minimum_improvement)
    if not _passes_scaled_fold_acceptance(
        policy,
        challenger.fold_forecasts,
        baseline.fold_forecasts,
        baseline.fold_truths,
        minimum_improvement=minimum_improvement,
    ):
        return False
    if policy.long_horizon_guard_enabled and not _passes_long_horizon_override_guard(
        policy, challenger, baseline
    ):
        return False
    return True


def _passes_conservative_override(
    policy: DecisionPolicy,
    challenger: CandidateDiagnostics,
    baseline: CandidateDiagnostics,
) -> bool:
    """Require stable absolute and squared-error evidence across 3+1 folds."""
    if not _aligned_folds((challenger, baseline)):
        return False
    required_ordinary_wins = math.ceil(
        0.75 * (len(challenger.fold_forecasts) + 1)
    ) - 1
    if not _passes_scaled_fold_acceptance(
        policy,
        challenger.fold_forecasts,
        baseline.fold_forecasts,
        baseline.fold_truths,
        required_wins=required_ordinary_wins,
        maximum_regret=0.0,
    ):
        return False
    baseline_smae, baseline_srmse = _median_scaled_metrics(
        baseline.fold_forecasts, baseline.fold_truths
    )
    challenger_smae, challenger_srmse = _median_scaled_metrics(
        challenger.fold_forecasts, challenger.fold_truths
    )
    if not (
        challenger_smae < baseline_smae * (1.0 - policy.tsfm_router_min_improvement)
        and challenger_srmse < baseline_srmse * (1.0 - policy.tsfm_router_min_improvement)
    ):
        return False
    audit_fold = challenger.long_horizon_fold
    baseline_audit = baseline.long_horizon_fold
    return bool(
        audit_fold is not None
        and baseline_audit is not None
        and _passes_scaled_fold_acceptance(
            policy,
            (audit_fold.forecast,),
            (baseline_audit.forecast,),
            (baseline_audit.truth,),
            required_wins=1,
            maximum_regret=policy.long_horizon_max_regret,
        )
    )


def _long_horizon_penalty(
    policy: DecisionPolicy,
    challenger: CandidateDiagnostics,
    baseline: CandidateDiagnostics,
    *,
    profile,
    challenger_fold: HindcastFold | None = None,
    coverage: float | None = None,
) -> float:
    if not policy.long_horizon_audit_enabled:
        return 0.0
    audit = challenger.long_horizon_fold if challenger_fold is None else challenger_fold
    reference = baseline.long_horizon_fold
    audit_coverage = challenger.long_horizon_coverage if coverage is None else coverage
    if not _long_horizon_route_matches(policy, profile, audit_coverage):
        return 0.0
    if (
        audit is None
        or reference is None
        or audit.status != "success"
        or reference.status != "success"
        or audit.truth != reference.truth
        or not audit.truth
    ):
        return 0.0
    scale = reference.mase_scale
    if scale is None or not math.isfinite(scale) or scale <= 0:
        return 0.0
    candidate_scores = _fold_scores((audit.forecast,), (audit.truth,), (float(scale),))
    reference_scores = _fold_scores(
        (reference.forecast,), (reference.truth,), (float(scale),)
    )
    if not candidate_scores or not reference_scores:
        return 0.0
    regret = (candidate_scores[0] - reference_scores[0]) / (1.0 + reference_scores[0])
    return policy.long_horizon_penalty_weight * audit_coverage * max(0.0, regret)


def _passes_long_horizon_override_guard(
    policy: DecisionPolicy,
    challenger: CandidateDiagnostics,
    baseline: CandidateDiagnostics,
    *,
    challenger_fold: HindcastFold | None = None,
    coverage: float | None = None,
) -> bool:
    comparison = _long_horizon_comparison(
        policy,
        challenger,
        baseline,
        challenger_fold=challenger_fold,
        coverage=coverage,
    )
    if comparison is None:
        return False
    audit = challenger.long_horizon_fold if challenger_fold is None else challenger_fold
    reference = baseline.long_horizon_fold
    if audit is None or reference is None:
        return False
    return _passes_independent_scaled_regret(
        (audit.forecast,),
        (reference.forecast,),
        (reference.truth,),
        max_smae_regret=policy.long_horizon_max_regret,
        max_srmse_regret=policy.long_horizon_max_regret,
    )


def _passes_independent_scaled_regret(
    candidate_forecasts: Sequence[Sequence[float]],
    reference_forecasts: Sequence[Sequence[float]],
    truths: Sequence[Sequence[float]],
    *,
    max_smae_regret: float,
    max_srmse_regret: float,
) -> bool:
    """Fail closed unless each scaled metric respects every aligned fold guard."""
    if (
        not candidate_forecasts
        or len(candidate_forecasts) != len(reference_forecasts)
        or len(candidate_forecasts) != len(truths)
    ):
        return False
    for candidate, reference, truth in zip(
        candidate_forecasts, reference_forecasts, truths, strict=True
    ):
        if len(candidate) != len(reference) or len(candidate) != len(truth) or not truth:
            return False
        candidate_metrics = drcik_point_metrics(list(truth), list(candidate))
        reference_metrics = drcik_point_metrics(list(truth), list(reference))
        for metric, raw_metric, clipped_metric, limit in (
            ("smae", "smae_raw", "smae_clipped", max_smae_regret),
            ("srmse", "srmse_raw", "srmse_clipped", max_srmse_regret),
        ):
            if bool(candidate_metrics[clipped_metric]) and not bool(
                reference_metrics[clipped_metric]
            ):
                return False
            candidate_raw = float(candidate_metrics[raw_metric])
            reference_raw = float(reference_metrics[raw_metric])
            if not math.isfinite(candidate_raw) or not math.isfinite(reference_raw):
                return False
            if (candidate_raw - reference_raw) / (1.0 + reference_raw) > limit + 1e-12:
                return False
    return True


def _scaled_fold_pairs(
    forecasts: Sequence[Sequence[float]], truths: Sequence[Sequence[float]]
) -> tuple[tuple[float, float], ...]:
    if len(forecasts) != len(truths):
        return ()
    pairs = []
    for forecast, truth in zip(forecasts, truths, strict=True):
        if len(forecast) != len(truth) or not truth:
            return ()
        metrics = drcik_point_metrics(list(truth), list(forecast))
        pairs.append((float(metrics["smae"]), float(metrics["srmse"])))
    return tuple(pairs)


def _median_scaled_metrics(
    forecasts: Sequence[Sequence[float]], truths: Sequence[Sequence[float]]
) -> tuple[float, float]:
    pairs = _scaled_fold_pairs(forecasts, truths)
    if not pairs:
        return math.inf, math.inf
    return (
        statistics.median(pair[0] for pair in pairs),
        statistics.median(pair[1] for pair in pairs),
    )


def _long_horizon_comparison(
    policy: DecisionPolicy,
    challenger: CandidateDiagnostics,
    baseline: CandidateDiagnostics,
    *,
    challenger_fold: HindcastFold | None = None,
    coverage: float | None = None,
) -> tuple[float, float] | None:
    audit = challenger.long_horizon_fold if challenger_fold is None else challenger_fold
    reference = baseline.long_horizon_fold
    audit_coverage = challenger.long_horizon_coverage if coverage is None else coverage
    if audit_coverage + 1e-12 < policy.long_horizon_min_coverage:
        return None
    if (
        audit is None
        or reference is None
        or audit.status != "success"
        or reference.status != "success"
        or audit.truth != reference.truth
        or not audit.truth
    ):
        return None
    candidate_scores = _fold_scores((audit.forecast,), (audit.truth,))
    baseline_scores = _fold_scores((reference.forecast,), (reference.truth,))
    if not candidate_scores or not baseline_scores:
        return None
    return candidate_scores[0], baseline_scores[0]


def _long_horizon_route_matches(policy: DecisionPolicy, profile, coverage: float) -> bool:
    if profile is None:
        return False
    if policy.long_horizon_route_feature == "audit_coverage":
        value = coverage
    elif policy.long_horizon_route_feature == "horizon_ratio":
        value = profile.horizon / max(1, profile.history_length)
    else:
        value = float(getattr(profile, policy.long_horizon_route_feature))
    if policy.long_horizon_route_operator == "at_least":
        return value >= policy.long_horizon_route_threshold
    return value <= policy.long_horizon_route_threshold


def _best_guarded_combination(
    policy: DecisionPolicy,
    eligible: Sequence[CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    *,
    history: Sequence[float],
    baseline: CandidateDiagnostics | None,
    conditioned_names: Sequence[str],
    profile=None,
) -> tuple[
    tuple[str, ...],
    tuple[float, ...],
    tuple[float, ...],
    str,
    SelectionArithmetic,
] | None:
    """Search a bounded TSFM+statistical portfolio using history-only folds."""
    ranked = sorted(eligible, key=lambda item: _rank_key(item, policy.ranking_order))
    anchors = [item for item in ranked if item.family in {"tsfm", "foundation"}][:2]
    ranked_specialists = [item for item in ranked if item.family == "statistical"]
    specialists = ranked_specialists[:3]
    already_selected = {item.name for item in specialists}
    conditioned = set(conditioned_names)
    specialists.extend(
        item
        for item in ranked_specialists
        if item.name in conditioned and item.name not in already_selected
    )
    specialists = specialists[:6]
    proposals: list[
        tuple[
            float,
            float,
            str,
            str,
            str,
            float,
            tuple[float, ...],
            tuple[float, ...],
            SelectionArithmetic,
        ]
    ] = []
    for anchor, specialist in itertools.product(anchors, specialists):
        if not _aligned_folds((anchor, specialist)):
            continue
        if pairwise_diversity(anchor.fold_forecasts, specialist.fold_forecasts) \
                < policy.ensemble_min_diversity:
            continue
        reference = min(
            (anchor, specialist),
            key=lambda item: (_fold_score(item.fold_forecasts, item.fold_truths), item.name),
        )
        references = [reference]
        if baseline is not None and baseline.name not in {anchor.name, specialist.name}:
            if _aligned_folds((anchor, baseline)):
                references.append(baseline)

        for anchor_weight in policy.ensemble_weight_grid:
            fold_forecasts = _blend_folds(
                (anchor.fold_forecasts, specialist.fold_forecasts),
                (anchor_weight, 1.0 - anchor_weight),
            )
            if not _passes_combination_gates(policy, fold_forecasts, references):
                continue
            audit_fold, audit_coverage = _combined_long_horizon_fold(
                anchor,
                specialist,
                kind="weighted_blend",
                parameter=anchor_weight,
                clip_multiplier=policy.ensemble_correction_clip,
            )
            if not _passes_combination_long_horizon_gate(
                policy,
                fold_forecasts,
                baseline,
                audit_fold,
                audit_coverage,
                profile,
            ):
                continue
            final = _blend_values(
                forecasts[anchor.name],
                forecasts[specialist.name],
                anchor_weight,
            )
            score = _fold_score(
                fold_forecasts, anchor.fold_truths, _fold_mase_scales(anchor)
            )
            proposals.append((
                score,
                _worst_fold_score(
                    fold_forecasts, anchor.fold_truths, _fold_mase_scales(anchor)
                ),
                "weighted_blend",
                anchor.name,
                specialist.name,
                anchor_weight,
                (anchor_weight, 1.0 - anchor_weight),
                final,
                SelectionArithmetic(
                    "blend",
                    inputs=(
                        SelectionArithmetic("leaf", candidate_name=anchor.name),
                        SelectionArithmetic("leaf", candidate_name=specialist.name),
                    ),
                    weights=(anchor_weight, 1.0 - anchor_weight),
                ),
            ))

        scales = _fold_correction_scales(anchor)
        for strength in policy.ensemble_residual_strengths:
            fold_forecasts = _residual_folds(
                anchor.fold_forecasts,
                specialist.fold_forecasts,
                strength,
                policy.ensemble_correction_clip,
                scales,
            )
            if not _passes_combination_gates(policy, fold_forecasts, references):
                continue
            audit_fold, audit_coverage = _combined_long_horizon_fold(
                anchor,
                specialist,
                kind="residual_correction",
                parameter=strength,
                clip_multiplier=policy.ensemble_correction_clip,
            )
            if not _passes_combination_long_horizon_gate(
                policy,
                fold_forecasts,
                baseline,
                audit_fold,
                audit_coverage,
                profile,
            ):
                continue
            final_scale = _naive_absolute_scale(tuple(float(value) for value in history)) \
                if history else _forecast_scale(forecasts[anchor.name])
            final = _residual_values(
                forecasts[anchor.name],
                forecasts[specialist.name],
                strength,
                policy.ensemble_correction_clip,
                final_scale,
            )
            score = _fold_score(
                fold_forecasts, anchor.fold_truths, _fold_mase_scales(anchor)
            )
            proposals.append((
                score,
                _worst_fold_score(
                    fold_forecasts, anchor.fold_truths, _fold_mase_scales(anchor)
                ),
                "residual_correction",
                anchor.name,
                specialist.name,
                strength,
                (1.0 - strength, strength),
                final,
                SelectionArithmetic(
                    "residual",
                    inputs=(
                        SelectionArithmetic("leaf", candidate_name=anchor.name),
                        SelectionArithmetic("leaf", candidate_name=specialist.name),
                    ),
                    strength=strength,
                    clip_multiplier=policy.ensemble_correction_clip,
                    scale=final_scale,
                ),
            ))

    if not proposals:
        return None
    proposal = min(proposals, key=lambda item: item[:6])
    _, _, kind, anchor_name, specialist_name, _, weights, final, arithmetic = proposal
    return (anchor_name, specialist_name), weights, final, kind, arithmetic


def _passes_combination_gates(
    policy: DecisionPolicy,
    candidate_folds: Sequence[Sequence[float]],
    references: Sequence[CandidateDiagnostics],
) -> bool:
    for reference in references:
        if not _passes_scaled_fold_acceptance(
            policy,
            candidate_folds,
            reference.fold_forecasts,
            reference.fold_truths,
        ):
            return False
    return True


def _passes_scaled_fold_acceptance(
    policy: DecisionPolicy,
    candidate_forecasts: Sequence[Sequence[float]],
    reference_forecasts: Sequence[Sequence[float]],
    truths: Sequence[Sequence[float]],
    *,
    required_wins: int | None = None,
    maximum_regret: float | None = None,
    minimum_improvement: float = 0.0,
) -> bool:
    """Accept only independently safe capped sMAE and sRMSE fold evidence."""
    if not _passes_independent_scaled_regret(
        candidate_forecasts, reference_forecasts, truths,
        max_smae_regret=(
            policy.max_smae_fold_regret if maximum_regret is None else maximum_regret
        ),
        max_srmse_regret=(
            policy.max_srmse_fold_regret if maximum_regret is None else maximum_regret
        ),
    ):
        return False
    candidate = _scaled_fold_pairs(candidate_forecasts, truths)
    reference = _scaled_fold_pairs(reference_forecasts, truths)
    if not candidate or len(candidate) != len(reference):
        return False
    candidate_smae, candidate_srmse = _median_scaled_metrics(candidate_forecasts, truths)
    reference_smae, reference_srmse = _median_scaled_metrics(reference_forecasts, truths)
    if not pareto_scaled_improvement(
        reference_smae, reference_srmse, candidate_smae, candidate_srmse
    ):
        return False
    if not (
        candidate_smae < reference_smae * (1.0 - minimum_improvement)
        and candidate_srmse < reference_srmse * (1.0 - minimum_improvement)
    ):
        return False
    wins_required = min(
        policy.ensemble_min_fold_wins if required_wins is None else required_wins,
        len(candidate),
    )
    for metric_index in (0, 1):
        wins = sum(
            candidate_fold[metric_index] < reference_fold[metric_index] - 1e-12
            for candidate_fold, reference_fold in zip(candidate, reference, strict=True)
        )
        if wins < wins_required:
            return False
        regret_limit = (
            policy.ensemble_max_worst_fold_regret
            if maximum_regret is None else maximum_regret
        )
        capped_regrets = tuple(
            (candidate_fold[metric_index] - reference_fold[metric_index])
            / (1.0 + reference_fold[metric_index])
            for candidate_fold, reference_fold in zip(candidate, reference, strict=True)
        )
        if max(capped_regrets) > regret_limit + 1e-12:
            return False
        if capped_regrets[-1] > regret_limit + 1e-12:
            return False
    return True


def _combined_long_horizon_fold(
    anchor: CandidateDiagnostics,
    specialist: CandidateDiagnostics,
    *,
    kind: str,
    parameter: float,
    clip_multiplier: float,
) -> tuple[HindcastFold | None, float]:
    left = anchor.long_horizon_fold
    right = specialist.long_horizon_fold
    coverage = min(anchor.long_horizon_coverage, specialist.long_horizon_coverage)
    if (
        left is None
        or right is None
        or left.status != "success"
        or right.status != "success"
        or left.train_end != right.train_end
        or left.validation_end != right.validation_end
        or left.truth != right.truth
        or len(left.forecast) != len(right.forecast)
    ):
        return None, coverage
    if kind == "weighted_blend":
        forecast = _blend_values(left.forecast, right.forecast, parameter)
    elif kind == "residual_correction":
        scale = left.mase_scale
        if scale is None or not math.isfinite(scale) or scale <= 0:
            scale = _forecast_scale(left.truth)
        forecast = _residual_values(
            left.forecast,
            right.forecast,
            parameter,
            clip_multiplier,
            float(scale),
        )
    else:
        raise ValueError(f"unsupported Combined audit kind: {kind}")
    return replace(left, forecast=forecast, **_scaled_fold_fields(left.truth, forecast)), coverage


def _passes_combination_long_horizon_gate(
    policy: DecisionPolicy,
    candidate_folds: Sequence[Sequence[float]],
    baseline: CandidateDiagnostics | None,
    audit_fold: HindcastFold | None,
    audit_coverage: float,
    profile,
) -> bool:
    if baseline is None:
        return True
    if not _passes_scaled_fold_acceptance(
        policy,
        candidate_folds,
        baseline.fold_forecasts,
        baseline.fold_truths,
    ):
        return False
    del profile
    if (
        policy.baseline_strategy == "conservative_tsfm"
        and baseline.name == "timesfm_2_5"
    ):
        baseline_audit = baseline.long_horizon_fold
        if audit_fold is None or baseline_audit is None:
            return False
        return _passes_scaled_fold_acceptance(
            policy,
            (audit_fold.forecast,),
            (baseline_audit.forecast,),
            (baseline_audit.truth,),
            required_wins=1,
            maximum_regret=policy.long_horizon_max_regret,
        )
    return not policy.long_horizon_guard_enabled or _passes_long_horizon_override_guard(
        policy, baseline, baseline, challenger_fold=audit_fold, coverage=audit_coverage
    )


def _fold_scores(
    forecasts: Sequence[Sequence[float]],
    truths: Sequence[Sequence[float]],
    scales: Sequence[float] = (),
) -> tuple[float, ...]:
    """Capped Dr-CiK joint diagnostics; ``scales`` is legacy-compatible only."""
    del scales
    if len(forecasts) != len(truths):
        return ()
    scores = []
    for forecast, truth in zip(forecasts, truths):
        if len(forecast) != len(truth) or not truth:
            return ()
        metrics = drcik_point_metrics(list(truth), list(forecast))
        scores.append(joint_scaled_error(float(metrics["smae"]), float(metrics["srmse"])))
    return tuple(scores)


def _passes_srmse_noninferiority(
    candidate_forecasts: Sequence[Sequence[float]],
    reference_forecasts: Sequence[Sequence[float]],
    truths: Sequence[Sequence[float]],
) -> bool:
    """Fail closed if any fold worsens Dr-CiK clipped sRMSE or clipping status."""
    if (
        not candidate_forecasts
        or len(candidate_forecasts) != len(reference_forecasts)
        or len(candidate_forecasts) != len(truths)
    ):
        return False
    for candidate, reference, truth in zip(
        candidate_forecasts, reference_forecasts, truths, strict=True
    ):
        if len(candidate) != len(reference) or len(candidate) != len(truth) or not truth:
            return False
        candidate_metric = drcik_point_metrics(list(truth), list(candidate))
        reference_metric = drcik_point_metrics(list(truth), list(reference))
        if bool(candidate_metric["srmse_clipped"]) and not bool(
            reference_metric["srmse_clipped"]
        ):
            return False
        if float(candidate_metric["srmse"]) > float(reference_metric["srmse"]) + 1e-12:
            return False
    return True


def _passes_srmse_strict_improvement(
    candidate_forecasts: Sequence[Sequence[float]],
    reference_forecasts: Sequence[Sequence[float]],
    truths: Sequence[Sequence[float]],
) -> bool:
    """Require lower Dr-CiK sRMSE on every aligned fold, without new clipping."""
    if (
        not candidate_forecasts
        or len(candidate_forecasts) != len(reference_forecasts)
        or len(candidate_forecasts) != len(truths)
    ):
        return False
    for candidate, reference, truth in zip(
        candidate_forecasts, reference_forecasts, truths, strict=True
    ):
        if len(candidate) != len(reference) or len(candidate) != len(truth) or not truth:
            return False
        candidate_metric = drcik_point_metrics(list(truth), list(candidate))
        reference_metric = drcik_point_metrics(list(truth), list(reference))
        if bool(candidate_metric["srmse_clipped"]) and not bool(
            reference_metric["srmse_clipped"]
        ):
            return False
        if not float(candidate_metric["srmse"]) + 1e-12 < float(
            reference_metric["srmse"]
        ):
            return False
    return True


def _passes_srmse_relative_improvement(
    candidate_forecasts: Sequence[Sequence[float]],
    reference_forecasts: Sequence[Sequence[float]],
    truths: Sequence[Sequence[float]],
    *,
    minimum_improvement: float,
) -> bool:
    """Require a relative Dr-CiK sRMSE margin on every aligned fold."""
    if not 0.0 <= minimum_improvement <= 1.0:
        return False
    if (
        not candidate_forecasts
        or len(candidate_forecasts) != len(reference_forecasts)
        or len(candidate_forecasts) != len(truths)
    ):
        return False
    for candidate, reference, truth in zip(
        candidate_forecasts, reference_forecasts, truths, strict=True
    ):
        if len(candidate) != len(reference) or len(candidate) != len(truth) or not truth:
            return False
        candidate_metric = drcik_point_metrics(list(truth), list(candidate))
        reference_metric = drcik_point_metrics(list(truth), list(reference))
        if bool(candidate_metric["srmse_clipped"]) and not bool(
            reference_metric["srmse_clipped"]
        ):
            return False
        if not float(candidate_metric["srmse"]) + 1e-12 < float(
            reference_metric["srmse"]
        ) * (1.0 - minimum_improvement):
            return False
    return True


def _worst_fold_score(
    forecasts: Sequence[Sequence[float]],
    truths: Sequence[Sequence[float]],
    scales: Sequence[float] = (),
) -> float:
    return max(_fold_scores(forecasts, truths, scales), default=math.inf)


def _blend_values(
    anchor: Sequence[float], specialist: Sequence[float], anchor_weight: float
) -> tuple[float, ...]:
    if len(anchor) != len(specialist):
        raise ValueError("Combined parents returned different forecast horizons")
    return tuple(
        anchor_weight * float(left) + (1.0 - anchor_weight) * float(right)
        for left, right in zip(anchor, specialist, strict=True)
    )


def _fold_correction_scales(candidate: CandidateDiagnostics) -> tuple[float, ...]:
    successful = tuple(fold for fold in candidate.folds if fold.status == "success")
    if len(successful) == len(candidate.fold_forecasts):
        scales = tuple(
            float(fold.mase_scale)
            if fold.mase_scale is not None and fold.mase_scale > 0
            else max(1e-8, float(fold.mae) / float(fold.mase))
            if fold.mae is not None and fold.mase is not None and fold.mase > 0
            else _forecast_scale(fold.truth)
            for fold in successful
        )
        if scales:
            return scales
    return tuple(_forecast_scale(truth) for truth in candidate.fold_truths)


def _fold_mase_scales(candidate: CandidateDiagnostics) -> tuple[float, ...]:
    return _fold_correction_scales(candidate)


def _forecast_scale(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    spread = max(values) - min(values)
    return float(spread) if spread > 1e-8 else 1.0


def _residual_values(
    anchor: Sequence[float],
    specialist: Sequence[float],
    strength: float,
    clip_multiplier: float,
    scale: float,
) -> tuple[float, ...]:
    if len(anchor) != len(specialist):
        raise ValueError("Combined parents returned different forecast horizons")
    bound = max(1e-8, clip_multiplier * scale)
    return tuple(
        float(left) + strength * max(-bound, min(bound, float(right) - float(left)))
        for left, right in zip(anchor, specialist, strict=True)
    )


def _residual_folds(
    anchors: Sequence[Sequence[float]],
    specialists: Sequence[Sequence[float]],
    strength: float,
    clip_multiplier: float,
    scales: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    if len(anchors) != len(specialists) or len(anchors) != len(scales):
        return ()
    return tuple(
        _residual_values(anchor, specialist, strength, clip_multiplier, scale)
        for anchor, specialist, scale in zip(anchors, specialists, scales, strict=True)
    )


def _best_validated_ensemble(
    policy: DecisionPolicy,
    eligible: Sequence[CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    best: CandidateDiagnostics,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]] | None:
    pool = sorted(eligible, key=lambda item: _rank_key(item, policy.ranking_order))[:5]
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
            if not _passes_combination_gates(policy, fold_forecasts, members):
                continue
            audit, coverage = _multi_member_long_horizon_fold(
                members, weights=weights, kind="weighted"
            )
            if not _passes_combination_long_horizon_gate(
                policy, fold_forecasts, best, audit, coverage, profile=None
            ):
                continue
            names_and_forecasts = sorted(
                ((member.name, tuple(map(float, forecasts[member.name]))) for member in members),
                key=lambda item: item[0],
            )
            names = tuple(item[0] for item in names_and_forecasts)
            final = _fmean_values(tuple(item[1] for item in names_and_forecasts))
            proposal = (names, weights, final)
            if best_proposal is None or proposal[0] < best_proposal[0]:
                best_proposal = proposal
    if best_proposal is None:
        return None
    names, weights, final = best_proposal
    return names, weights, final


def _aligned_folds(members: Sequence[CandidateDiagnostics]) -> bool:
    first = members[0]
    return bool(first.fold_forecasts) and all(
        member.fold_truths == first.fold_truths
        and len(member.fold_forecasts) == len(first.fold_forecasts)
        for member in members[1:]
    )


def _strictly_aligned_successful_folds(
    members: Sequence[CandidateDiagnostics], *, minimum_folds: int
) -> bool:
    """Require complete, boundary-aligned fold records for a strict overlay audit."""
    if not members:
        return False
    reference = members[0].folds
    if len(reference) < minimum_folds:
        return False
    for member in members:
        if (
            len(member.folds) != len(reference)
            or member.successful_folds != len(member.folds)
            or len(member.fold_forecasts) != len(member.folds)
            or len(member.fold_truths) != len(member.folds)
            or any(fold.status != "success" for fold in member.folds)
        ):
            return False
        for index, fold in enumerate(member.folds):
            expected = reference[index]
            if (
                fold.train_end != expected.train_end
                or fold.validation_end != expected.validation_end
                or fold.truth != expected.truth
                or fold.forecast != member.fold_forecasts[index]
                or fold.truth != member.fold_truths[index]
            ):
                return False
    return True


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


def _fold_score(
    forecasts: Sequence[Sequence[float]],
    truths: Sequence[Sequence[float]],
    scales: Sequence[float] = (),
) -> float:
    if not forecasts or len(forecasts) != len(truths):
        return math.inf
    scores = _fold_scores(forecasts, truths, scales)
    if not scores:
        return math.inf
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
