"""Trusted label-bearing scoring and anonymous Build evidence.

Raw futures, materialized forecasts, task identities, and runtime failures stop
at this module.  The only value intended for a structural proposer is the
``ProposerEvidence`` projection built by :func:`sanitize_build_evidence`.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Literal

from common.metrics import drcik_point_metrics, joint_scaled_error, linear_quantile

from .cache import SCALED_METRIC_CAP
from .champion import FittedChampionPolicy
from .screening import TaskProfile


_SPLITS = frozenset({"build", "calibration", "dev", "public"})
_FORBIDDEN_PROPOSER_NAME = re.compile(
    r"(?:^|_)(?:dev|public|hidden|entity|truth|exception)(?:_|$)", re.IGNORECASE
)
_MAX_ROWS = 1_000_000
_TIE_EPSILON = 1e-12
_PROFILE_FLOAT_FIELDS = (
    "zero_fraction",
    "trend_strength",
    "periodicity_strength",
    "periodicity_confidence",
    "outlier_fraction",
    "noise_relative_scale",
    "stationarity_score",
    "recent_regime_confidence",
    "intermittency_adi",
    "intermittency_cv2",
)
_UNIT_PROFILE_FIELDS = frozenset({
    "zero_fraction",
    "trend_strength",
    "periodicity_strength",
    "periodicity_confidence",
    "outlier_fraction",
    "stationarity_score",
    "recent_regime_confidence",
})


class ChampionEvidenceError(ValueError):
    """Trusted Champion evidence is malformed, incomplete, or unsafe."""


def _fail(message: str) -> None:
    raise ChampionEvidenceError(message)


def _public_identifier(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or not value.isidentifier()
        or value.startswith("_")
    ):
        _fail(f"{label} must be a public Python identifier")
    return value


def _finite_number(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        _fail(f"{label} must have an exact numeric type")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} must be finite")
    return number


def _finite_float(value: object, label: str, *, lower: float | None = None) -> float:
    if type(value) is not float or not math.isfinite(value):
        _fail(f"{label} must be a finite float")
    if lower is not None and value < lower:
        _fail(f"{label} must be at least {lower}")
    return value


def _nonnegative_tail(value: object, label: str) -> float:
    if type(value) is not float or math.isnan(value) or value < 0.0:
        _fail(f"{label} must be a nonnegative float or positive infinity")
    return value


def _validate_profile(profile: object) -> TaskProfile:
    if type(profile) is not TaskProfile:
        _fail("profile must be an exact TaskProfile")
    try:
        TaskProfile.__post_init__(profile)
    except (TypeError, ValueError) as error:
        raise ChampionEvidenceError("profile contains invalid measurements") from error
    if type(profile.task_id) is not str or not profile.task_id:
        _fail("profile task identity must be a non-empty string")
    if type(profile.frequency) is not str or not profile.frequency.strip():
        _fail("profile frequency must be a non-empty exact string")
    if type(profile.trend_direction) is not str or not profile.trend_direction:
        _fail("profile trend_direction must be a non-empty exact string")
    if type(profile.horizon) is not int or type(profile.history_length) is not int:
        _fail("profile lengths must have exact integer types")
    if profile.horizon <= 0 or profile.history_length <= 0:
        _fail("profile lengths must be positive")
    for name in ("signed", "integer_valued", "likely_stationary"):
        if type(getattr(profile, name)) is not bool:
            _fail(f"profile {name} must be an exact bool")
    for name in _PROFILE_FLOAT_FIELDS:
        value = _finite_float(getattr(profile, name), f"profile {name}", lower=0.0)
        if name in _UNIT_PROFILE_FIELDS and value > 1.0:
            _fail(f"profile {name} must not exceed one")
    if type(profile.periodicity_periods) is not tuple or any(
        type(period) is not int or not 1 <= period <= _MAX_ROWS
        for period in profile.periodicity_periods
    ):
        _fail("profile periodicity_periods must be an exact positive-integer tuple")
    if profile.recent_regime_start is not None and (
        type(profile.recent_regime_start) is not int
        or not 0 <= profile.recent_regime_start < profile.history_length
    ):
        _fail("profile recent_regime_start must be a valid exact history index")
    return profile


def _validate_array(value: object, label: str, *, length: int) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        _fail(f"{label} must be an exact tuple forming a complete horizon")
    converted = tuple(_finite_number(item, f"{label} value") for item in value)
    return converted


@dataclass(frozen=True)
class ChampionTaskRow:
    """One trusted materialized policy outcome for one labeled task."""

    task_id: str
    candidate_name: str
    profile: TaskProfile
    truth: tuple[float, ...] | None
    forecast: tuple[float, ...] | None
    failure_reason: str | None = None
    fold: int = 0
    split: Literal["build", "calibration", "dev", "public"] = "build"

    def __post_init__(self) -> None:
        _validate_task_row(self)


def _validate_task_row(row: object) -> ChampionTaskRow:
    if type(row) is not ChampionTaskRow:
        _fail("rows must contain exact ChampionTaskRow records")
    if type(row.task_id) is not str or not row.task_id:
        _fail("task identity must be a non-empty string")
    _public_identifier(row.candidate_name, "candidate_name")
    profile = _validate_profile(row.profile)
    if row.task_id != profile.task_id:
        _fail("mislabeled row task identity does not match its profile")
    if type(row.fold) is not int or not 0 <= row.fold <= _MAX_ROWS:
        _fail("fold must be a bounded exact nonnegative integer")
    if type(row.split) is not str or row.split not in _SPLITS:
        _fail("split label is unsupported")

    complete = row.truth is not None and row.forecast is not None
    failed = row.truth is None and row.forecast is None
    if not complete and not failed:
        _fail("truth and forecast must form a complete pair")
    if complete:
        if row.failure_reason is not None:
            _fail("successful rows cannot carry raw runtime failures")
        _validate_array(row.truth, "truth", length=profile.horizon)
        _validate_array(row.forecast, "forecast", length=profile.horizon)
    else:
        if (
            type(row.failure_reason) is not str
            or not row.failure_reason.strip()
            or len(row.failure_reason) > 10_000
        ):
            _fail("failed rows require one bounded raw runtime failure")
    return row


@dataclass(frozen=True)
class _TaskMetric:
    task_id: str
    profile: TaskProfile
    fold: int
    split: str
    successful: bool
    smae: float | None = None
    srmse: float | None = None
    smae_raw: float | None = None
    srmse_raw: float | None = None
    smae_clipped: bool = False
    srmse_clipped: bool = False


@dataclass(frozen=True)
class ChampionScore:
    """Canonical aggregate score for one policy over a fixed task universe."""

    policy_name: str
    total_tasks: int
    successful_tasks: int
    failure_count: int
    coverage: float
    failure_rate: float
    mean_smae: float
    mean_srmse: float
    median_smae: float
    median_srmse: float
    p90_smae: float
    p95_smae: float
    max_smae: float
    p90_srmse: float
    p95_srmse: float
    max_srmse: float
    mean_smae_raw: float
    mean_srmse_raw: float
    median_smae_raw: float
    median_srmse_raw: float
    p90_smae_raw: float
    p95_smae_raw: float
    max_smae_raw: float
    p90_srmse_raw: float
    p95_srmse_raw: float
    max_srmse_raw: float
    smae_clipped_count: int
    srmse_clipped_count: int
    smae_clipped_rate: float
    srmse_clipped_rate: float
    fold_count: int
    _task_metrics: tuple[_TaskMetric, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_score(self)


def _validate_score(score: object) -> ChampionScore:
    if type(score) is not ChampionScore:
        _fail("comparison scores must be exact ChampionScore records")
    _public_identifier(score.policy_name, "policy_name")
    for name in (
        "total_tasks",
        "successful_tasks",
        "failure_count",
        "smae_clipped_count",
        "srmse_clipped_count",
        "fold_count",
    ):
        value = getattr(score, name)
        if type(value) is not int or not 0 <= value <= _MAX_ROWS:
            _fail(f"{name} must be a bounded exact nonnegative integer")
    if score.total_tasks <= 0 or score.successful_tasks <= 0:
        _fail("scores require at least one task and one successful task")
    if score.successful_tasks + score.failure_count != score.total_tasks:
        _fail("score success and failure counts must cover all tasks")
    if score.fold_count <= 0:
        _fail("scores require at least one fold")
    for name in ("coverage", "failure_rate", "smae_clipped_rate", "srmse_clipped_rate"):
        value = _finite_float(getattr(score, name), name, lower=0.0)
        if value > 1.0:
            _fail(f"{name} must not exceed one")
    for name in (
        "mean_smae", "mean_srmse", "median_smae", "median_srmse",
        "p90_smae", "p95_smae", "max_smae",
        "p90_srmse", "p95_srmse", "max_srmse",
    ):
        value = _finite_float(getattr(score, name), name, lower=0.0)
        if value > SCALED_METRIC_CAP:
            _fail(f"{name} exceeds the active scaled-metric cap")
    for name in (
        "mean_smae_raw", "mean_srmse_raw", "median_smae_raw", "median_srmse_raw",
        "p90_smae_raw", "p95_smae_raw", "max_smae_raw",
        "p90_srmse_raw", "p95_srmse_raw", "max_srmse_raw",
    ):
        _nonnegative_tail(getattr(score, name), name)
    if type(score._task_metrics) is not tuple or len(score._task_metrics) != score.total_tasks:
        _fail("score task metrics must cover the complete task universe")
    if any(type(metric) is not _TaskMetric for metric in score._task_metrics):
        _fail("score task metrics contain an invalid record")
    metric_ids = tuple(metric.task_id for metric in score._task_metrics)
    if len(metric_ids) != len(set(metric_ids)):
        _fail("score task metrics contain duplicate task IDs")
    return score


def _policy_name(policy: object) -> str:
    if type(policy) is str:
        return _public_identifier(policy, "policy")
    if type(policy) is FittedChampionPolicy:
        try:
            FittedChampionPolicy.__post_init__(policy)
        except (TypeError, ValueError) as error:
            raise ChampionEvidenceError("policy is not a valid fitted Champion") from error
        return policy.recipe.name
    _fail("policy must be a public name or exact FittedChampionPolicy")
    raise AssertionError("unreachable")


def _quantile(values: list[float], probability: float) -> float:
    return float(linear_quantile(values, probability))


def score_policy(
    rows: tuple[ChampionTaskRow, ...] | list[ChampionTaskRow],
    policy: str | FittedChampionPolicy,
) -> ChampionScore:
    """Score one policy with canonical capped and raw Dr-CiK metrics."""
    if type(rows) not in {tuple, list} or not rows or len(rows) > _MAX_ROWS:
        _fail("rows must be a nonempty bounded tuple or list")
    snapshot = tuple(_validate_task_row(row) for row in rows)
    name = _policy_name(policy)

    # Detect identity collisions before constructing any lookup map.
    keys = tuple((row.candidate_name, row.task_id) for row in snapshot)
    if len(keys) != len(set(keys)):
        _fail("duplicate candidate/task key")
    task_ids = tuple(dict.fromkeys(row.task_id for row in snapshot))
    selected_ids = tuple(row.task_id for row in snapshot if row.candidate_name == name)
    if not selected_ids:
        _fail("policy has no trusted task rows")
    if len(selected_ids) != len(set(selected_ids)):
        _fail("duplicate task IDs for policy")
    if set(selected_ids) != set(task_ids):
        _fail("policy rows do not provide complete task coverage")

    # Every repeated task identity must retain its exact profile, fold, and split.
    identity_values: dict[str, tuple[TaskProfile, int, str]] = {}
    for row in snapshot:
        identity = (row.profile, row.fold, row.split)
        prior = identity_values.setdefault(row.task_id, identity)
        if identity != prior:
            _fail("mislabeled task rows disagree on profile, fold, or split")

    metrics: list[_TaskMetric] = []
    for row in snapshot:
        if row.candidate_name != name:
            continue
        if row.truth is None:
            metrics.append(_TaskMetric(row.task_id, row.profile, row.fold, row.split, False))
            continue
        assert row.forecast is not None
        try:
            point = drcik_point_metrics(
                row.truth, row.forecast, cap=SCALED_METRIC_CAP
            )
        except (OverflowError, TypeError, ValueError) as error:
            raise ChampionEvidenceError(
                "canonical scoring rejected a complete finite pair"
            ) from error
        metrics.append(
            _TaskMetric(
                task_id=row.task_id,
                profile=row.profile,
                fold=row.fold,
                split=row.split,
                successful=True,
                smae=float(point["smae"]),
                srmse=float(point["srmse"]),
                smae_raw=float(point["smae_raw"]),
                srmse_raw=float(point["srmse_raw"]),
                smae_clipped=bool(point["smae_clipped"]),
                srmse_clipped=bool(point["srmse_clipped"]),
            )
        )
    successful = [metric for metric in metrics if metric.successful]
    if not successful:
        _fail("policy has no complete successful pairs to score")

    def values(field_name: str) -> list[float]:
        return [float(getattr(metric, field_name)) for metric in successful]

    smae = values("smae")
    srmse = values("srmse")
    smae_raw = values("smae_raw")
    srmse_raw = values("srmse_raw")
    total = len(metrics)
    success_count = len(successful)
    failure_count = total - success_count
    smae_clipped_count = sum(metric.smae_clipped for metric in successful)
    srmse_clipped_count = sum(metric.srmse_clipped for metric in successful)

    def summaries(prefix: str, metric_values: list[float]) -> dict[str, float]:
        return {
            f"mean_{prefix}": float(statistics.fmean(metric_values)),
            f"median_{prefix}": float(statistics.median(metric_values)),
            f"p90_{prefix}": _quantile(metric_values, 0.90),
            f"p95_{prefix}": _quantile(metric_values, 0.95),
            f"max_{prefix}": max(metric_values),
        }

    aggregate: dict[str, object] = {
        "policy_name": name,
        "total_tasks": total,
        "successful_tasks": success_count,
        "failure_count": failure_count,
        "coverage": float(success_count / total),
        "failure_rate": float(failure_count / total),
        "smae_clipped_count": smae_clipped_count,
        "srmse_clipped_count": srmse_clipped_count,
        "smae_clipped_rate": float(smae_clipped_count / success_count),
        "srmse_clipped_rate": float(srmse_clipped_count / success_count),
        "fold_count": len({metric.fold for metric in metrics}),
        "_task_metrics": tuple(metrics),
    }
    aggregate.update(summaries("smae", smae))
    aggregate.update(summaries("srmse", srmse))
    aggregate.update(summaries("smae_raw", smae_raw))
    aggregate.update(summaries("srmse_raw", srmse_raw))
    return ChampionScore(**aggregate)  # type: ignore[arg-type]


@dataclass(frozen=True)
class WinTieLoss:
    wins: int
    ties: int
    losses: int

    def __post_init__(self) -> None:
        for name in ("wins", "ties", "losses"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= _MAX_ROWS:
                _fail(f"{name} must be a bounded exact nonnegative integer")


@dataclass(frozen=True)
class ChampionGateConfig:
    """Pre-registered independent gates for one paired comparison."""

    minimum_joint_improvement: float = 0.005
    primary_regression_tolerance: float = 0.0
    tail_regression_tolerance: float = 0.0
    minimum_coverage: float = 1.0
    maximum_failure_rate: float = 0.0
    maximum_coverage_regression: float = 0.0
    maximum_failure_rate_increase: float = 0.0
    maximum_clipped_count_increase: int = 0
    maximum_task_regret_smae: float = 0.25
    maximum_task_regret_srmse: float = 0.25
    minimum_improved_folds: int = 0
    tie_tolerance: float = _TIE_EPSILON

    def __post_init__(self) -> None:
        for name in (
            "minimum_joint_improvement", "primary_regression_tolerance",
            "tail_regression_tolerance", "minimum_coverage", "maximum_failure_rate",
            "maximum_coverage_regression", "maximum_failure_rate_increase",
            "maximum_task_regret_smae", "maximum_task_regret_srmse", "tie_tolerance",
        ):
            _finite_float(getattr(self, name), name, lower=0.0)
        if self.minimum_joint_improvement > 1.0:
            _fail("minimum_joint_improvement must not exceed one")
        if self.minimum_coverage > 1.0 or self.maximum_failure_rate > 1.0:
            _fail("coverage and failure gates must be rates")
        for name in ("maximum_clipped_count_increase", "minimum_improved_folds"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= _MAX_ROWS:
                _fail(f"{name} must be a bounded exact nonnegative integer")


@dataclass(frozen=True)
class ChampionComparison:
    """Paired metric authority plus explicit diagnostic and safety results."""

    parent_name: str
    child_name: str
    accepted: bool
    failures: tuple[str, ...]
    mean_delta_smae: float
    mean_delta_srmse: float
    joint_improvement: float
    parent_coverage: float
    child_coverage: float
    parent_failure_rate: float
    child_failure_rate: float
    parent_p90_smae: float
    child_p90_smae: float
    parent_p95_smae: float
    child_p95_smae: float
    parent_p90_srmse: float
    child_p90_srmse: float
    parent_p95_srmse: float
    child_p95_srmse: float
    parent_p90_smae_raw: float
    child_p90_smae_raw: float
    parent_p95_smae_raw: float
    child_p95_smae_raw: float
    parent_p90_srmse_raw: float
    child_p90_srmse_raw: float
    parent_p95_srmse_raw: float
    child_p95_srmse_raw: float
    parent_smae_clipped_count: int
    child_smae_clipped_count: int
    parent_srmse_clipped_count: int
    child_srmse_clipped_count: int
    max_regret_smae: float
    max_regret_srmse: float
    improved_folds: int
    total_folds: int
    wtl: WinTieLoss

    def __post_init__(self) -> None:
        _public_identifier(self.parent_name, "parent_name")
        _public_identifier(self.child_name, "child_name")
        if self.parent_name == self.child_name:
            _fail("comparison requires distinct Parent and Child")
        if type(self.accepted) is not bool:
            _fail("accepted must be an exact bool")
        if type(self.failures) is not tuple or any(
            type(item) is not str for item in self.failures
        ):
            _fail("failures must be an exact tuple of gate identifiers")
        if len(self.failures) != len(set(self.failures)) or any(
            not item or not item.isidentifier() for item in self.failures
        ):
            _fail("failures must contain unique gate identifiers")
        if self.accepted != (not self.failures):
            _fail("accepted must agree with the independent gate failures")
        for name in ("mean_delta_smae", "mean_delta_srmse", "joint_improvement"):
            _finite_float(getattr(self, name), name)
        for name in ("max_regret_smae", "max_regret_srmse"):
            value = _finite_float(getattr(self, name), name, lower=0.0)
            if value > SCALED_METRIC_CAP:
                _fail(f"{name} exceeds the active scaled-metric cap")
        for prefix in ("parent", "child"):
            coverage = _finite_float(
                getattr(self, f"{prefix}_coverage"),
                f"{prefix}_coverage",
                lower=0.0,
            )
            failure_rate = _finite_float(
                getattr(self, f"{prefix}_failure_rate"),
                f"{prefix}_failure_rate",
                lower=0.0,
            )
            if coverage > 1.0 or failure_rate > 1.0 or not math.isclose(
                coverage + failure_rate, 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                _fail(f"{prefix} coverage and failure rate must partition support")
            for metric in ("p90_smae", "p95_smae", "p90_srmse", "p95_srmse"):
                value = _finite_float(
                    getattr(self, f"{prefix}_{metric}"),
                    f"{prefix}_{metric}",
                    lower=0.0,
                )
                if value > SCALED_METRIC_CAP:
                    _fail(f"{prefix}_{metric} exceeds the active scaled-metric cap")
            for metric in (
                "p90_smae_raw", "p95_smae_raw", "p90_srmse_raw", "p95_srmse_raw"
            ):
                _nonnegative_tail(
                    getattr(self, f"{prefix}_{metric}"), f"{prefix}_{metric}"
                )
            for metric in ("smae_clipped_count", "srmse_clipped_count"):
                value = getattr(self, f"{prefix}_{metric}")
                if type(value) is not int or not 0 <= value <= _MAX_ROWS:
                    _fail(f"{prefix}_{metric} must be a bounded exact integer")
        for name in ("improved_folds", "total_folds"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= _MAX_ROWS:
                _fail(f"{name} must be a bounded exact nonnegative integer")
        if self.total_folds <= 0 or self.improved_folds > self.total_folds:
            _fail("fold stability counts are inconsistent")
        if type(self.wtl) is not WinTieLoss:
            _fail("wtl must be an exact WinTieLoss")
        WinTieLoss.__post_init__(self.wtl)


def _metric_map(score: ChampionScore) -> dict[str, _TaskMetric]:
    return {metric.task_id: metric for metric in score._task_metrics}


def _relative_improvement(parent: float, child: float) -> float:
    if parent == 0.0:
        return 0.0
    if child <= parent:
        return float((parent - child) / parent)
    return float(-((child - parent) / child))


def compare_champion(
    parent: ChampionScore,
    child: ChampionScore,
    config: ChampionGateConfig,
) -> ChampionComparison:
    """Apply paired primary and independent safety gates; W/T/L is diagnostic."""
    _validate_score(parent)
    _validate_score(child)
    if type(config) is not ChampionGateConfig:
        _fail("config must be an exact ChampionGateConfig")
    ChampionGateConfig.__post_init__(config)
    if parent.policy_name == child.policy_name:
        _fail("Parent and Child policy names must differ")
    parent_metrics = _metric_map(parent)
    child_metrics = _metric_map(child)
    if set(parent_metrics) != set(child_metrics):
        _fail("Parent and Child scores must contain complete paired task IDs")
    for task_id in parent_metrics:
        left = parent_metrics[task_id]
        right = child_metrics[task_id]
        if (left.profile, left.fold, left.split) != (right.profile, right.fold, right.split):
            _fail("mislabeled Parent/Child rows do not describe the same task")

    paired = [
        (parent_metrics[task_id], child_metrics[task_id])
        for task_id in parent_metrics
        if parent_metrics[task_id].successful and child_metrics[task_id].successful
    ]
    if not paired:
        _fail("comparison has no complete successful task pairs")
    smae_regrets = [max(0.0, float(right.smae) - float(left.smae)) for left, right in paired]
    srmse_regrets = [max(0.0, float(right.srmse) - float(left.srmse)) for left, right in paired]
    max_regret_smae = max(smae_regrets)
    max_regret_srmse = max(srmse_regrets)

    wins = ties = losses = 0
    tolerance = config.tie_tolerance
    for left, right in paired:
        parent_joint = joint_scaled_error(float(left.smae), float(left.srmse))
        child_joint = joint_scaled_error(float(right.smae), float(right.srmse))
        if child_joint < parent_joint - tolerance:
            wins += 1
        elif child_joint > parent_joint + tolerance:
            losses += 1
        else:
            ties += 1
    for task_id in parent_metrics:
        left = parent_metrics[task_id]
        right = child_metrics[task_id]
        if left.successful and not right.successful:
            losses += 1
        elif right.successful and not left.successful:
            wins += 1
        elif not left.successful and not right.successful:
            ties += 1

    folds = sorted({metric.fold for metric in parent_metrics.values()})
    improved_folds = 0
    for fold in folds:
        fold_pairs = [(left, right) for left, right in paired if left.fold == fold]
        if not fold_pairs:
            continue
        parent_fold = statistics.fmean(
            joint_scaled_error(float(left.smae), float(left.srmse))
            for left, _ in fold_pairs
        )
        child_fold = statistics.fmean(
            joint_scaled_error(float(right.smae), float(right.srmse))
            for _, right in fold_pairs
        )
        if child_fold < parent_fold - tolerance:
            improved_folds += 1

    failures: list[str] = []
    primary_tolerance = config.primary_regression_tolerance
    if child.mean_smae > parent.mean_smae + primary_tolerance:
        failures.append("mean_smae")
    if child.mean_srmse > parent.mean_srmse + primary_tolerance:
        failures.append("mean_srmse")
    if not (
        child.mean_smae < parent.mean_smae - tolerance
        or child.mean_srmse < parent.mean_srmse - tolerance
    ):
        failures.append("paired_primary_improvement")
    parent_joint = joint_scaled_error(parent.mean_smae, parent.mean_srmse)
    child_joint = joint_scaled_error(child.mean_smae, child.mean_srmse)
    joint_improvement = _relative_improvement(parent_joint, child_joint)
    if joint_improvement + tolerance < config.minimum_joint_improvement:
        failures.append("joint_improvement")

    for metric in (
        "p90_smae", "p95_smae", "p90_srmse", "p95_srmse",
        "p90_smae_raw", "p95_smae_raw", "p90_srmse_raw", "p95_srmse_raw",
    ):
        if getattr(child, metric) > getattr(parent, metric) + config.tail_regression_tolerance:
            failures.append(metric)
    if child.coverage < config.minimum_coverage or (
        child.coverage + config.maximum_coverage_regression < parent.coverage
    ):
        failures.append("coverage")
    if child.failure_rate > config.maximum_failure_rate or (
        child.failure_rate > parent.failure_rate + config.maximum_failure_rate_increase
    ):
        failures.append("failure_rate")
    for metric in ("smae_clipped_count", "srmse_clipped_count"):
        if (
            getattr(child, metric)
            > getattr(parent, metric) + config.maximum_clipped_count_increase
        ):
            failures.append(metric)
    if max_regret_smae > config.maximum_task_regret_smae + tolerance:
        failures.append("max_regret_smae")
    if max_regret_srmse > config.maximum_task_regret_srmse + tolerance:
        failures.append("max_regret_srmse")
    if improved_folds < config.minimum_improved_folds:
        failures.append("fold_stability")
    unique_failures = tuple(dict.fromkeys(failures))

    values: dict[str, object] = {
        "parent_name": parent.policy_name,
        "child_name": child.policy_name,
        "accepted": not unique_failures,
        "failures": unique_failures,
        "mean_delta_smae": float(child.mean_smae - parent.mean_smae),
        "mean_delta_srmse": float(child.mean_srmse - parent.mean_srmse),
        "joint_improvement": joint_improvement,
        "parent_coverage": parent.coverage,
        "child_coverage": child.coverage,
        "parent_failure_rate": parent.failure_rate,
        "child_failure_rate": child.failure_rate,
        "parent_smae_clipped_count": parent.smae_clipped_count,
        "child_smae_clipped_count": child.smae_clipped_count,
        "parent_srmse_clipped_count": parent.srmse_clipped_count,
        "child_srmse_clipped_count": child.srmse_clipped_count,
        "max_regret_smae": max_regret_smae,
        "max_regret_srmse": max_regret_srmse,
        "improved_folds": improved_folds,
        "total_folds": len(folds),
        "wtl": WinTieLoss(wins, ties, losses),
    }
    for metric in (
        "p90_smae", "p95_smae", "p90_srmse", "p95_srmse",
        "p90_smae_raw", "p95_smae_raw", "p90_srmse_raw", "p95_srmse_raw",
    ):
        values[f"parent_{metric}"] = getattr(parent, metric)
        values[f"child_{metric}"] = getattr(child, metric)
    return ChampionComparison(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class MorphologyAggregate:
    """Anonymous candidate evidence for one reviewed morphology group."""

    group_id: str
    support: int
    candidate_name: str
    mean_delta_smae: float
    mean_delta_srmse: float
    coverage: float
    p95_regret_smae: float
    p95_regret_srmse: float

    def __post_init__(self) -> None:
        if type(self.group_id) is not str or ":" not in self.group_id:
            _fail("morphology group_id must be a reviewed anonymous group")
        _public_identifier(self.candidate_name, "candidate_name")
        if type(self.support) is not int or not 1 <= self.support <= _MAX_ROWS:
            _fail("morphology support must be a positive bounded exact integer")
        for name in ("mean_delta_smae", "mean_delta_srmse"):
            _finite_float(getattr(self, name), name)
        for name in ("coverage", "p95_regret_smae", "p95_regret_srmse"):
            value = _finite_float(getattr(self, name), name, lower=0.0)
            if name == "coverage" and value > 1.0:
                _fail("morphology coverage must not exceed one")


@dataclass(frozen=True)
class _ProposerComparison:
    candidate_name: str
    accepted: bool
    failures: tuple[str, ...]
    mean_delta_smae: float
    mean_delta_srmse: float
    joint_improvement: float
    coverage: float
    failure_rate: float
    p90_smae: float
    p95_smae: float
    p90_srmse: float
    p95_srmse: float
    p90_smae_raw: float
    p95_smae_raw: float
    p90_srmse_raw: float
    p95_srmse_raw: float
    smae_clipped_count: int
    srmse_clipped_count: int
    max_regret_smae: float
    max_regret_srmse: float
    improved_folds: int
    total_folds: int
    wins: int
    ties: int
    losses: int


@dataclass(frozen=True)
class ProposerEvidence:
    """The complete recursively sanitized projection allowed into a proposer."""

    label: Literal["adaptive_train_build_diagnostic"]
    independent_generalization_claim: Literal[False]
    morphology: tuple[MorphologyAggregate, ...]
    comparisons: tuple[_ProposerComparison, ...]

    def __post_init__(self) -> None:
        if self.label != "adaptive_train_build_diagnostic":
            _fail("Build evidence requires its non-independent diagnostic label")
        if self.independent_generalization_claim is not False:
            _fail("Build evidence cannot claim independent generalization")
        if type(self.morphology) is not tuple or any(
            type(item) is not MorphologyAggregate for item in self.morphology
        ):
            _fail("morphology evidence must be an exact aggregate tuple")
        if type(self.comparisons) is not tuple or any(
            type(item) is not _ProposerComparison for item in self.comparisons
        ):
            _fail("comparison evidence must be an exact aggregate tuple")

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-safe payload with no scorer-internal identities or arrays."""
        payload: dict[str, object] = {
            "label": self.label,
            "independent_generalization_claim": False,
            "morphology": [
                {
                    "group_id": item.group_id,
                    "support": item.support,
                    "candidate_name": item.candidate_name,
                    "mean_delta_smae": item.mean_delta_smae,
                    "mean_delta_srmse": item.mean_delta_srmse,
                    "coverage": item.coverage,
                    "p95_regret_smae": item.p95_regret_smae,
                    "p95_regret_srmse": item.p95_regret_srmse,
                }
                for item in self.morphology
            ],
            "comparisons": [
                {field_name: _json_safe(getattr(item, field_name))
                 for field_name in item.__dataclass_fields__}
                for item in self.comparisons
            ],
        }
        _assert_sanitized(payload)
        return payload


def _json_safe(value: object) -> object:
    if type(value) is float and value == math.inf:
        return "positive_infinity"
    if type(value) is tuple:
        return [_json_safe(item) for item in value]
    return value


def _assert_safe_candidate_name(name: str) -> None:
    _public_identifier(name, "candidate_name")
    if _FORBIDDEN_PROPOSER_NAME.search(name):
        _fail("candidate_name contains a forbidden proposer marker")


def _assert_sanitized(value: object) -> None:
    forbidden_keys = frozenset({
        "task_id", "task_ids", "truth", "truths", "forecast", "forecasts",
        "entity", "entity_name", "entity_names", "split", "exception",
    })
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or key.lower() in forbidden_keys:
                _fail("proposer evidence contains a forbidden field")
            _assert_sanitized(item)
    elif type(value) is list:
        for item in value:
            _assert_sanitized(item)
    elif type(value) is str and re.search(r"(?:^|[_\s])(?:dev|public)(?:[_\s]|$)", value, re.I):
        _fail("proposer evidence contains a Dev/Public marker")
    elif type(value) is float and not math.isfinite(value):
        _fail("proposer evidence contains a non-JSON numeric value")


def _profile_groups(profile: TaskProfile) -> tuple[str, ...]:
    reviewed = (
        ("periodicity_strength", profile.periodicity_strength >= 0.6),
        ("trend_strength", profile.trend_strength >= 0.6),
        ("intermittency_adi", profile.intermittency_adi >= 1.32),
        ("zero_fraction", profile.zero_fraction >= 0.3),
        ("recent_regime_confidence", profile.recent_regime_confidence >= 0.5),
        ("outlier_fraction", profile.outlier_fraction >= 0.05),
        ("noise_relative_scale", profile.noise_relative_scale >= 1.0),
        ("history_length", profile.history_length >= 168),
        ("horizon", profile.horizon >= 24),
        ("horizon_ratio", profile.horizon / profile.history_length >= 0.25),
    )
    groups = [f"{feature}:{'high' if high else 'low'}" for feature, high in reviewed]
    if re.search(r"(?:^|[^A-Za-z0-9])(?:dev|public)(?:[^A-Za-z0-9]|$)", profile.frequency, re.I):
        _fail("profile frequency contains a Dev/Public marker")
    frequency = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile.frequency).strip("_")
    if not frequency or len(frequency) > 100:
        _fail("profile frequency cannot form anonymous morphology evidence")
    groups.append(f"frequency:{frequency}")
    return tuple(groups)


def _comparison_projection(comparison: ChampionComparison) -> _ProposerComparison:
    return _ProposerComparison(
        candidate_name=comparison.child_name,
        accepted=comparison.accepted,
        failures=comparison.failures,
        mean_delta_smae=comparison.mean_delta_smae,
        mean_delta_srmse=comparison.mean_delta_srmse,
        joint_improvement=comparison.joint_improvement,
        coverage=comparison.child_coverage,
        failure_rate=comparison.child_failure_rate,
        p90_smae=comparison.child_p90_smae,
        p95_smae=comparison.child_p95_smae,
        p90_srmse=comparison.child_p90_srmse,
        p95_srmse=comparison.child_p95_srmse,
        p90_smae_raw=comparison.child_p90_smae_raw,
        p95_smae_raw=comparison.child_p95_smae_raw,
        p90_srmse_raw=comparison.child_p90_srmse_raw,
        p95_srmse_raw=comparison.child_p95_srmse_raw,
        smae_clipped_count=comparison.child_smae_clipped_count,
        srmse_clipped_count=comparison.child_srmse_clipped_count,
        max_regret_smae=comparison.max_regret_smae,
        max_regret_srmse=comparison.max_regret_srmse,
        improved_folds=comparison.improved_folds,
        total_folds=comparison.total_folds,
        wins=comparison.wtl.wins,
        ties=comparison.wtl.ties,
        losses=comparison.wtl.losses,
    )


def sanitize_build_evidence(
    rows: tuple[ChampionTaskRow, ...] | list[ChampionTaskRow],
    comparisons: tuple[ChampionComparison, ...] | list[ChampionComparison],
) -> ProposerEvidence:
    """Aggregate reviewed Build morphology and discard all label-bearing values."""
    if type(rows) not in {tuple, list} or not rows or len(rows) > _MAX_ROWS:
        _fail("Build rows must be a nonempty bounded tuple or list")
    snapshot = tuple(_validate_task_row(row) for row in rows)
    if any(row.split != "build" for row in snapshot):
        _fail("only exact Build rows may enter proposer evidence")
    keys = tuple((row.candidate_name, row.task_id) for row in snapshot)
    if len(keys) != len(set(keys)):
        _fail("duplicate candidate/task key")
    if type(comparisons) not in {tuple, list} or not comparisons:
        _fail("comparisons must be a nonempty tuple or list")
    comparison_snapshot = tuple(comparisons)
    if any(type(item) is not ChampionComparison for item in comparison_snapshot):
        _fail("comparisons must contain exact ChampionComparison records")
    child_names = tuple(item.child_name for item in comparison_snapshot)
    if len(child_names) != len(set(child_names)):
        _fail("Build comparisons contain duplicate Child names")
    for comparison in comparison_snapshot:
        ChampionComparison.__post_init__(comparison)
        _assert_safe_candidate_name(comparison.parent_name)
        _assert_safe_candidate_name(comparison.child_name)

    row_map = {(row.candidate_name, row.task_id): row for row in snapshot}
    task_ids = tuple(dict.fromkeys(row.task_id for row in snapshot))
    morphology: list[MorphologyAggregate] = []
    for comparison in comparison_snapshot:
        grouped: dict[str, list[tuple[ChampionTaskRow, ChampionTaskRow]]] = {}
        for task_id in task_ids:
            parent = row_map.get((comparison.parent_name, task_id))
            child = row_map.get((comparison.child_name, task_id))
            if parent is None or child is None:
                _fail("Build comparison rows require complete Parent/Child pairs")
            if parent.profile != child.profile:
                _fail("mislabeled Build Parent/Child profiles disagree")
            for group_id in _profile_groups(parent.profile):
                grouped.setdefault(group_id, []).append((parent, child))
        for group_id, pairs in grouped.items():
            paired_values: list[tuple[float, float]] = []
            for parent, child in pairs:
                if parent.truth is None or child.truth is None:
                    continue
                if parent.truth != child.truth:
                    _fail("mislabeled Build rows contain different truths for one task")
                assert parent.forecast is not None and child.forecast is not None
                parent_point = drcik_point_metrics(
                    parent.truth, parent.forecast, cap=SCALED_METRIC_CAP
                )
                child_point = drcik_point_metrics(
                    child.truth, child.forecast, cap=SCALED_METRIC_CAP
                )
                paired_values.append((
                    float(child_point["smae"]) - float(parent_point["smae"]),
                    float(child_point["srmse"]) - float(parent_point["srmse"]),
                ))
            if not paired_values:
                continue
            smae_deltas = [value[0] for value in paired_values]
            srmse_deltas = [value[1] for value in paired_values]
            morphology.append(MorphologyAggregate(
                group_id=group_id,
                support=len(pairs),
                candidate_name=comparison.child_name,
                mean_delta_smae=float(statistics.fmean(smae_deltas)),
                mean_delta_srmse=float(statistics.fmean(srmse_deltas)),
                coverage=float(len(paired_values) / len(pairs)),
                p95_regret_smae=_quantile([max(0.0, value) for value in smae_deltas], 0.95),
                p95_regret_srmse=_quantile([max(0.0, value) for value in srmse_deltas], 0.95),
            ))

    evidence = ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=tuple(
            sorted(morphology, key=lambda item: (item.group_id, item.candidate_name))
        ),
        comparisons=tuple(
            _comparison_projection(item)
            for item in sorted(comparison_snapshot, key=lambda item: item.child_name)
        ),
    )
    evidence.to_payload()
    return evidence


__all__ = [
    "ChampionComparison",
    "ChampionEvidenceError",
    "ChampionGateConfig",
    "ChampionScore",
    "ChampionTaskRow",
    "MorphologyAggregate",
    "ProposerEvidence",
    "WinTieLoss",
    "compare_champion",
    "sanitize_build_evidence",
    "score_policy",
]
