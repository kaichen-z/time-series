"""Train-only specialist discovery and history-only routing features."""

from __future__ import annotations

import math
import statistics
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from common.metrics import drcik_point_metrics, joint_scaled_error

from .numerical_selector import CandidateDiagnostics
from .task_local_evolution import TaskLocalTaskRow


def _canonical_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 value")
    return value


@dataclass(frozen=True)
class AtlasPolicy:
    """Closed host-owned candidate discovery and calibration limits."""

    schema_version: int = 1
    maximum_pool_size: int = 12
    maximum_task_candidates: int = 8
    neighbor_grid: tuple[int, ...] = (3, 5, 7, 9)
    minimum_independent_groups: int = 2
    minimum_win_probability: float = 0.60
    minimum_effect_margin: float = 0.0
    maximum_predicted_regret: float = 0.25

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Atlas policy schema must be exactly one")
        if (
            type(self.maximum_pool_size) is not int
            or not 2 <= self.maximum_pool_size <= 12
        ):
            raise ValueError("Atlas pool size must be within [2, 12]")
        if (
            type(self.maximum_task_candidates) is not int
            or not 2 <= self.maximum_task_candidates <= 8
            or self.maximum_task_candidates > self.maximum_pool_size
        ):
            raise ValueError("Atlas task supply must be within the pool and [2, 8]")
        if (
            type(self.neighbor_grid) is not tuple
            or not self.neighbor_grid
            or self.neighbor_grid != tuple(sorted(set(self.neighbor_grid)))
            or any(type(value) is not int or not 2 <= value <= 31 for value in self.neighbor_grid)
        ):
            raise ValueError("Atlas neighbor grid must contain sorted unique integers")
        if (
            type(self.minimum_independent_groups) is not int
            or not 2 <= self.minimum_independent_groups <= 80
        ):
            raise ValueError("Atlas group support is outside its closed range")
        for name in (
            "minimum_win_probability",
            "minimum_effect_margin",
            "maximum_predicted_regret",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if not 0.5 < self.minimum_win_probability <= 1.0:
            raise ValueError("Atlas win probability must be within (0.5, 1]")


@dataclass(frozen=True)
class AtlasFeature:
    """Task-identity-free, history-only features used for routing distance."""

    categorical: tuple[str, ...]
    numeric: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            type(self.categorical) is not tuple
            or len(self.categorical) != 9
            or any(type(value) is not str or not value for value in self.categorical)
        ):
            raise ValueError("Atlas categorical features are noncanonical")
        if (
            type(self.numeric) is not tuple
            or len(self.numeric) != 15
            or any(type(value) is not float or not math.isfinite(value) for value in self.numeric)
        ):
            raise ValueError("Atlas numeric features are noncanonical")


@dataclass(frozen=True)
class AtlasTrainingRecord:
    """One fitting target paired with a history-only Atlas feature."""

    candidate_name: str
    family: str
    group_sha256: str
    feature: AtlasFeature
    improvement_smae: float
    improvement_srmse: float
    regret_smae_raw: float
    regret_srmse_raw: float

    def __post_init__(self) -> None:
        if type(self.candidate_name) is not str or not self.candidate_name.isidentifier():
            raise ValueError("Atlas record candidate must be an identifier")
        if type(self.family) is not str or self.family not in {
            "statistical",
            "tsfm",
            "combined",
        }:
            raise ValueError("Atlas record family is unsupported")
        _require_sha256(self.group_sha256, "Atlas record group")
        if type(self.feature) is not AtlasFeature:
            raise ValueError("Atlas record requires an exact feature")
        AtlasFeature.__post_init__(self.feature)
        for name in (
            "improvement_smae",
            "improvement_srmse",
            "regret_smae_raw",
            "regret_srmse_raw",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        if self.regret_smae_raw < 0.0 or self.regret_srmse_raw < 0.0:
            raise ValueError("Atlas record regret must be nonnegative")


def _history_scale(history: tuple[float, ...]) -> float:
    differences = tuple(abs(right - left) for left, right in zip(history, history[1:]))
    finite_differences = tuple(value for value in differences if math.isfinite(value))
    if finite_differences:
        scale = float(statistics.median(finite_differences))
        if scale > 1e-12:
            return scale
    level = float(statistics.median(abs(value) for value in history))
    return level if math.isfinite(level) and level > 1e-12 else 1.0


def _diagnostics_pair(
    candidate: TaskLocalTaskRow,
    anchor: TaskLocalTaskRow,
) -> tuple[CandidateDiagnostics, CandidateDiagnostics]:
    if (
        type(candidate) is not TaskLocalTaskRow
        or type(anchor) is not TaskLocalTaskRow
        or candidate.task_id != anchor.task_id
        or candidate.history != anchor.history
        or candidate.profile != anchor.profile
        or candidate.forecast is None
        or anchor.forecast is None
        or type(candidate.diagnostic) is not CandidateDiagnostics
        or type(anchor.diagnostic) is not CandidateDiagnostics
    ):
        raise ValueError("Atlas feature requires aligned successful task rows")
    left = candidate.diagnostic
    right = anchor.diagnostic
    if (
        left.fold_truths != right.fold_truths
        or not left.fold_truths
        or left.successful_folds < 1
        or right.successful_folds < 1
    ):
        raise ValueError("Atlas feature requires aligned paired hindcasts")
    return left, right


def atlas_feature(candidate: TaskLocalTaskRow, anchor: TaskLocalTaskRow) -> AtlasFeature:
    """Construct one candidate-specific feature without using task identity or truth."""
    diagnostic, anchor_diagnostic = _diagnostics_pair(candidate, anchor)
    profile = candidate.profile
    history_bucket = (
        "short" if profile.history_length < 64 else "medium" if profile.history_length < 256 else "long"
    )
    ratio = float(profile.horizon / profile.history_length)
    horizon_bucket = "short" if ratio <= 0.1 else "medium" if ratio <= 0.3 else "long"
    trend = (
        profile.trend_direction
        if profile.trend_strength >= 0.35 and profile.trend_direction != "flat"
        else "flat"
    )
    periodic = bool(
        profile.periodicity_periods and profile.periodicity_confidence >= 0.5
    )
    intermittent = bool(
        profile.zero_fraction >= 0.5 or profile.intermittency_adi >= 1.32
    )
    recent_regime = bool(
        profile.recent_regime_start is not None
        and profile.recent_regime_confidence >= 0.5
    )
    categorical = (
        _canonical_name(profile.frequency),
        history_bucket,
        horizon_bucket,
        trend,
        "periodic" if periodic else "aperiodic",
        "intermittent" if intermittent else "dense",
        "recent_regime" if recent_regime else "stable_regime",
        "signed" if profile.signed else "nonnegative",
        candidate.family,
    )
    paired_margins = (
        anchor_diagnostic.median_smae - diagnostic.median_smae,
        anchor_diagnostic.recent_smae - diagnostic.recent_smae,
        anchor_diagnostic.worst_smae - diagnostic.worst_smae,
        anchor_diagnostic.median_srmse - diagnostic.median_srmse,
        anchor_diagnostic.recent_srmse - diagnostic.recent_srmse,
        anchor_diagnostic.worst_srmse - diagnostic.worst_srmse,
    )
    fold_coverage = min(
        diagnostic.successful_folds, anchor_diagnostic.successful_folds
    ) / max(len(diagnostic.folds), len(anchor_diagnostic.folds), 1)
    disagreement = statistics.fmean(
        abs(left - right)
        for left, right in zip(candidate.forecast, anchor.forecast, strict=True)
    ) / _history_scale(candidate.history)
    numeric = tuple(
        float(value)
        for value in (
            profile.trend_strength,
            profile.periodicity_confidence,
            profile.zero_fraction,
            profile.noise_relative_scale,
            profile.stationarity_score,
            profile.recent_regime_confidence,
            ratio,
            *paired_margins,
            fold_coverage,
            min(disagreement, 100.0),
        )
    )
    return AtlasFeature(categorical, numeric)


def _rows_by_task(
    rows: Sequence[TaskLocalTaskRow], task_ids: Sequence[str]
) -> dict[str, dict[str, TaskLocalTaskRow]]:
    requested = tuple(task_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Atlas discovery requires unique task identifiers")
    result = {task_id: {} for task_id in requested}
    for row in rows:
        if type(row) is not TaskLocalTaskRow:
            raise ValueError("Atlas discovery requires exact task rows")
        if row.task_id not in result:
            continue
        if row.candidate_name in result[row.task_id]:
            raise ValueError("Atlas discovery contains a duplicate candidate/task row")
        result[row.task_id][row.candidate_name] = row
    return result


def _joint(row: TaskLocalTaskRow | None) -> float:
    if row is None or row.forecast is None:
        return 5.0
    point = drcik_point_metrics(row.truth, row.forecast)
    return joint_scaled_error(float(point["smae"]), float(point["srmse"]))


def select_specialist_pool(
    rows: Sequence[TaskLocalTaskRow],
    *,
    task_ids: Sequence[str],
    group_ids: Mapping[str, str],
    anchor_name: str,
    policy: AtlasPolicy,
) -> tuple[str, ...]:
    """Greedily retain candidates that add complementary fitting-fold coverage."""
    if type(policy) is not AtlasPolicy:
        raise TypeError("Atlas discovery requires an exact policy")
    AtlasPolicy.__post_init__(policy)
    if type(anchor_name) is not str or not anchor_name.isidentifier():
        raise ValueError("Atlas anchor must be an identifier")
    requested = tuple(task_ids)
    if set(group_ids) != set(requested):
        raise ValueError("Atlas discovery requires one group for every task")
    for group in group_ids.values():
        _require_sha256(group, "Atlas discovery group")
    by_task = _rows_by_task(rows, requested)
    if any(
        anchor_name not in by_task[task_id]
        or by_task[task_id][anchor_name].forecast is None
        for task_id in requested
    ):
        raise ValueError("Atlas discovery requires a successful anchor on every task")
    candidates = sorted(
        {
            name
            for task_id in requested
            for name in by_task[task_id]
            if name != anchor_name
        },
        key=_canonical_name,
    )
    supported = [
        name
        for name in candidates
        if len(
            {
                group_ids[task_id]
                for task_id in requested
                if (row := by_task[task_id].get(name)) is not None
                and row.forecast is not None
            }
        )
        >= policy.minimum_independent_groups
    ]
    current = {
        task_id: _joint(by_task[task_id][anchor_name]) for task_id in requested
    }
    selected = [anchor_name]
    while supported and len(selected) < policy.maximum_pool_size:
        ranked: list[tuple[float, str, dict[str, float]]] = []
        for name in supported:
            proposed = {
                task_id: min(current[task_id], _joint(by_task[task_id].get(name)))
                for task_id in requested
            }
            gain = statistics.fmean(current.values()) - statistics.fmean(
                proposed.values()
            )
            ranked.append((-gain, _canonical_name(name), proposed))
        negative_gain, _canonical, proposed = min(ranked)
        if -negative_gain <= 1e-12:
            break
        chosen = next(
            name
            for name in supported
            if _canonical_name(name) == _canonical
        )
        selected.append(chosen)
        supported.remove(chosen)
        current = proposed
    return tuple(selected)
