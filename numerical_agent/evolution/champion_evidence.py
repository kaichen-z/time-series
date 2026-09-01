"""Trusted label-bearing scoring and anonymous Build evidence.

Raw futures, materialized forecasts, task identities, and runtime failures stop
at this module.  The only value intended for a structural proposer is the
``ProposerEvidence`` projection built by :func:`sanitize_build_evidence`.
"""
from __future__ import annotations

import hashlib
import json
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
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
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


def _truth_fingerprint(truth: tuple[float, ...]) -> str:
    payload = json.dumps(
        list(truth), separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _universe_fingerprint(metrics: tuple[_TaskMetric, ...]) -> str:
    records = [
        {
            "task_id": metric.task_id,
            "truth": metric.truth_fingerprint,
            "profile": metric.profile.to_public_payload(),
            "fold": metric.fold,
            "split": metric.split,
        }
        for metric in sorted(metrics, key=lambda item: item.task_id)
    ]
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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

    if row.truth is None:
        _fail("every task row requires a complete finite truth array")
    _validate_array(row.truth, "truth", length=profile.horizon)
    if row.forecast is not None:
        if row.failure_reason is not None:
            _fail("successful rows cannot carry raw runtime failures")
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
    truth_fingerprint: str
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
    _universe_fingerprint: str = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_score(self)


def _metric_values(metrics: list[_TaskMetric], field_name: str) -> list[float]:
    return [float(getattr(metric, field_name)) for metric in metrics]


def _summaries(prefix: str, values: list[float]) -> dict[str, float]:
    return {
        f"mean_{prefix}": float(statistics.fmean(values)),
        f"median_{prefix}": float(statistics.median(values)),
        f"p90_{prefix}": _quantile(values, 0.90),
        f"p95_{prefix}": _quantile(values, 0.95),
        f"max_{prefix}": max(values),
    }


def _same_number(left: object, right: object) -> bool:
    if type(left) is not type(right) or type(left) not in {int, float}:
        return False
    return left == right


def _validate_task_metric(metric: object) -> _TaskMetric:
    if type(metric) is not _TaskMetric:
        _fail("score task metrics contain an invalid record")
    if type(metric.task_id) is not str or not metric.task_id:
        _fail("score task metric has an invalid task identity")
    profile = _validate_profile(metric.profile)
    if profile.task_id != metric.task_id:
        _fail("score task metric is mislabeled")
    if type(metric.fold) is not int or not 0 <= metric.fold <= _MAX_ROWS:
        _fail("score task metric has an invalid fold")
    if type(metric.split) is not str or metric.split not in _SPLITS:
        _fail("score task metric has an invalid split")
    if type(metric.truth_fingerprint) is not str or _SHA256.fullmatch(
        metric.truth_fingerprint
    ) is None:
        _fail("score task metric has an invalid truth fingerprint")
    if type(metric.successful) is not bool:
        _fail("score task metric successful must be an exact bool")
    if metric.successful:
        for name in ("smae", "srmse"):
            capped = _finite_float(getattr(metric, name), name, lower=0.0)
            raw = _nonnegative_tail(getattr(metric, f"{name}_raw"), f"{name}_raw")
            clipped = getattr(metric, f"{name}_clipped")
            if capped > SCALED_METRIC_CAP or type(clipped) is not bool:
                _fail("score task metric violates the active cap contract")
            if capped != min(SCALED_METRIC_CAP, raw) or clipped != (
                raw > SCALED_METRIC_CAP
            ):
                _fail("score task metric is not canonically capped")
    elif (
        any(
            getattr(metric, name) is not None
            for name in ("smae", "srmse", "smae_raw", "srmse_raw")
        )
        or metric.smae_clipped is not False
        or metric.srmse_clipped is not False
    ):
        _fail("failed score task metrics cannot contain authored metrics")
    return metric


def _derived_score_values(metrics: tuple[_TaskMetric, ...]) -> dict[str, object]:
    successful = [metric for metric in metrics if metric.successful]
    if not successful:
        _fail("policy has no complete successful pairs to score")
    total = len(metrics)
    success_count = len(successful)
    failure_count = total - success_count
    smae_clipped_count = sum(metric.smae_clipped for metric in successful)
    srmse_clipped_count = sum(metric.srmse_clipped for metric in successful)
    values: dict[str, object] = {
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
    }
    for field_name in ("smae", "srmse", "smae_raw", "srmse_raw"):
        values.update(_summaries(field_name, _metric_values(successful, field_name)))
    return values


def _validate_score(score: object) -> ChampionScore:
    if type(score) is not ChampionScore:
        _fail("comparison scores must be exact ChampionScore records")
    _public_identifier(score.policy_name, "policy_name")
    if type(score._task_metrics) is not tuple or not score._task_metrics:
        _fail("score task metrics must cover the complete task universe")
    for metric in score._task_metrics:
        _validate_task_metric(metric)
    metric_ids = tuple(metric.task_id for metric in score._task_metrics)
    if len(metric_ids) != len(set(metric_ids)):
        _fail("score task metrics contain duplicate task IDs")
    expected = _derived_score_values(score._task_metrics)
    for name, value in expected.items():
        if not _same_number(getattr(score, name), value):
            _fail(f"ChampionScore {name} is not coherent with its task metrics")
    expected_fingerprint = _universe_fingerprint(score._task_metrics)
    if (
        type(score._universe_fingerprint) is not str
        or score._universe_fingerprint != expected_fingerprint
    ):
        _fail("ChampionScore truth universe fingerprint is not coherent")
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

    # Every repeated task identity must retain its exact profile, truth, fold, and split.
    identity_values: dict[str, tuple[TaskProfile, str, int, str]] = {}
    for row in snapshot:
        assert row.truth is not None
        truth = _validate_array(row.truth, "truth", length=row.profile.horizon)
        identity = (row.profile, _truth_fingerprint(truth), row.fold, row.split)
        prior = identity_values.setdefault(row.task_id, identity)
        if identity != prior:
            _fail("mislabeled task rows disagree on profile, truth, fold, or split")

    metrics: list[_TaskMetric] = []
    for row in snapshot:
        if row.candidate_name != name:
            continue
        assert row.truth is not None
        truth = _validate_array(row.truth, "truth", length=row.profile.horizon)
        truth_fingerprint = _truth_fingerprint(truth)
        if row.forecast is None:
            metrics.append(_TaskMetric(
                task_id=row.task_id,
                profile=row.profile,
                fold=row.fold,
                split=row.split,
                truth_fingerprint=truth_fingerprint,
                successful=False,
            ))
            continue
        try:
            point = drcik_point_metrics(
                truth, row.forecast, cap=SCALED_METRIC_CAP
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
                truth_fingerprint=truth_fingerprint,
                successful=True,
                smae=float(point["smae"]),
                srmse=float(point["srmse"]),
                smae_raw=float(point["smae_raw"]),
                srmse_raw=float(point["srmse_raw"]),
                smae_clipped=bool(point["smae_clipped"]),
                srmse_clipped=bool(point["srmse_clipped"]),
            )
        )
    metric_tuple = tuple(metrics)
    aggregate: dict[str, object] = {
        "policy_name": name,
        "_task_metrics": metric_tuple,
        "_universe_fingerprint": _universe_fingerprint(metric_tuple),
    }
    aggregate.update(_derived_score_values(metric_tuple))
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
            "primary_regression_tolerance", "tail_regression_tolerance",
            "minimum_coverage", "maximum_failure_rate",
            "maximum_coverage_regression", "maximum_failure_rate_increase",
            "maximum_task_regret_smae", "maximum_task_regret_srmse", "tie_tolerance",
        ):
            _finite_float(getattr(self, name), name, lower=0.0)
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


def _paired_diagnostics(
    parent: ChampionScore,
    child: ChampionScore,
    *,
    tolerance: float,
) -> dict[str, object]:
    if parent._universe_fingerprint != child._universe_fingerprint:
        _fail("Parent and Child truth universe fingerprints differ")
    parent_metrics = _metric_map(parent)
    child_metrics = _metric_map(child)
    if set(parent_metrics) != set(child_metrics):
        _fail("Parent and Child scores must contain the same truth universe")
    for task_id in parent_metrics:
        left = parent_metrics[task_id]
        right = child_metrics[task_id]
        if (
            left.profile,
            left.truth_fingerprint,
            left.fold,
            left.split,
        ) != (
            right.profile,
            right.truth_fingerprint,
            right.fold,
            right.split,
        ):
            _fail("mislabeled Parent/Child rows do not describe the same truth")

    paired = [
        (parent_metrics[task_id], child_metrics[task_id])
        for task_id in sorted(parent_metrics)
        if parent_metrics[task_id].successful and child_metrics[task_id].successful
    ]
    if not paired:
        _fail("comparison has no common successful task pairs")

    def paired_values(side: int, metric: str) -> list[float]:
        return [float(getattr(pair[side], metric)) for pair in paired]

    values: dict[str, object] = {}
    for metric in ("smae", "srmse", "smae_raw", "srmse_raw"):
        for side, prefix in ((0, "parent"), (1, "child")):
            metric_values = paired_values(side, metric)
            values[f"{prefix}_mean_{metric}"] = float(
                statistics.fmean(metric_values)
            )
            values[f"{prefix}_p90_{metric}"] = _quantile(metric_values, 0.90)
            values[f"{prefix}_p95_{metric}"] = _quantile(metric_values, 0.95)

    for side, prefix in ((0, "parent"), (1, "child")):
        values[f"{prefix}_smae_clipped_count"] = sum(
            pair[side].smae_clipped for pair in paired
        )
        values[f"{prefix}_srmse_clipped_count"] = sum(
            pair[side].srmse_clipped for pair in paired
        )

    smae_regrets = [
        max(0.0, float(right.smae) - float(left.smae))
        for left, right in paired
    ]
    srmse_regrets = [
        max(0.0, float(right.srmse) - float(left.srmse))
        for left, right in paired
    ]
    values["max_regret_smae"] = max(smae_regrets)
    values["max_regret_srmse"] = max(srmse_regrets)

    wins = ties = losses = 0
    for left, right in paired:
        parent_joint = joint_scaled_error(float(left.smae), float(left.srmse))
        child_joint = joint_scaled_error(float(right.smae), float(right.srmse))
        if child_joint < parent_joint - tolerance:
            wins += 1
        elif child_joint > parent_joint + tolerance:
            losses += 1
        else:
            ties += 1
    values["wtl"] = WinTieLoss(wins, ties, losses)

    folds = sorted({left.fold for left, _ in paired})
    improved_folds = 0
    for fold in folds:
        fold_pairs = [(left, right) for left, right in paired if left.fold == fold]
        parent_smae = statistics.fmean(float(left.smae) for left, _ in fold_pairs)
        child_smae = statistics.fmean(float(right.smae) for _, right in fold_pairs)
        parent_srmse = statistics.fmean(float(left.srmse) for left, _ in fold_pairs)
        child_srmse = statistics.fmean(float(right.srmse) for _, right in fold_pairs)
        if (
            child_smae <= parent_smae + tolerance
            and child_srmse <= parent_srmse + tolerance
            and (
                child_smae < parent_smae - tolerance
                or child_srmse < parent_srmse - tolerance
            )
        ):
            improved_folds += 1
    values["improved_folds"] = improved_folds
    values["total_folds"] = len(folds)
    parent_joint = joint_scaled_error(
        float(values["parent_mean_smae"]),
        float(values["parent_mean_srmse"]),
    )
    child_joint = joint_scaled_error(
        float(values["child_mean_smae"]),
        float(values["child_mean_srmse"]),
    )
    values["joint_improvement"] = _relative_improvement(parent_joint, child_joint)
    return values


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
    tolerance = config.tie_tolerance
    diagnostics = _paired_diagnostics(parent, child, tolerance=tolerance)

    failures: list[str] = []
    primary_tolerance = config.primary_regression_tolerance
    if diagnostics["child_mean_smae"] > diagnostics["parent_mean_smae"] + primary_tolerance:
        failures.append("mean_smae")
    if diagnostics["child_mean_srmse"] > diagnostics["parent_mean_srmse"] + primary_tolerance:
        failures.append("mean_srmse")
    if not (
        diagnostics["child_mean_smae"] < diagnostics["parent_mean_smae"] - tolerance
        or diagnostics["child_mean_srmse"] < diagnostics["parent_mean_srmse"] - tolerance
    ):
        failures.append("paired_primary_improvement")

    for metric in (
        "p90_smae", "p95_smae", "p90_srmse", "p95_srmse",
        "p90_smae_raw", "p95_smae_raw", "p90_srmse_raw", "p95_srmse_raw",
    ):
        if (
            diagnostics[f"child_{metric}"]
            > diagnostics[f"parent_{metric}"] + config.tail_regression_tolerance
        ):
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
            diagnostics[f"child_{metric}"]
            > diagnostics[f"parent_{metric}"] + config.maximum_clipped_count_increase
        ):
            failures.append(metric)
    if diagnostics["max_regret_smae"] > config.maximum_task_regret_smae + tolerance:
        failures.append("max_regret_smae")
    if diagnostics["max_regret_srmse"] > config.maximum_task_regret_srmse + tolerance:
        failures.append("max_regret_srmse")
    if diagnostics["improved_folds"] < config.minimum_improved_folds:
        failures.append("fold_stability")
    unique_failures = tuple(dict.fromkeys(failures))

    values: dict[str, object] = {
        "parent_name": parent.policy_name,
        "child_name": child.policy_name,
        "accepted": not unique_failures,
        "failures": unique_failures,
        "mean_delta_smae": float(
            diagnostics["child_mean_smae"] - diagnostics["parent_mean_smae"]
        ),
        "mean_delta_srmse": float(
            diagnostics["child_mean_srmse"] - diagnostics["parent_mean_srmse"]
        ),
        "joint_improvement": diagnostics["joint_improvement"],
        "parent_coverage": parent.coverage,
        "child_coverage": child.coverage,
        "parent_failure_rate": parent.failure_rate,
        "child_failure_rate": child.failure_rate,
        "parent_smae_clipped_count": diagnostics["parent_smae_clipped_count"],
        "child_smae_clipped_count": diagnostics["child_smae_clipped_count"],
        "parent_srmse_clipped_count": diagnostics["parent_srmse_clipped_count"],
        "child_srmse_clipped_count": diagnostics["child_srmse_clipped_count"],
        "max_regret_smae": diagnostics["max_regret_smae"],
        "max_regret_srmse": diagnostics["max_regret_srmse"],
        "improved_folds": diagnostics["improved_folds"],
        "total_folds": diagnostics["total_folds"],
        "wtl": diagnostics["wtl"],
    }
    for metric in (
        "p90_smae", "p95_smae", "p90_srmse", "p95_srmse",
        "p90_smae_raw", "p95_smae_raw", "p90_srmse_raw", "p95_srmse_raw",
    ):
        values[f"parent_{metric}"] = diagnostics[f"parent_{metric}"]
        values[f"child_{metric}"] = diagnostics[f"child_{metric}"]
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
    if _FORBIDDEN_PROPOSER_NAME.search(name) or _contains_split_marker(name):
        _fail("candidate_name contains a forbidden proposer marker")


def _assert_sanitized(value: object) -> None:
    forbidden_keys = frozenset({
        "accepted", "rejected", "passed", "failures", "gate", "gates",
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
    elif type(value) is str and _contains_split_marker(value):
        _fail("proposer evidence contains a Dev/Public marker")
    elif type(value) is float and not math.isfinite(value):
        _fail("proposer evidence contains a non-JSON numeric value")


def _tokens(value: str) -> tuple[str, ...]:
    camel_split = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", camel_split)
    return tuple(
        token.casefold()
        for token in re.split(r"[^A-Za-z0-9]+", camel_split)
        if token
    )


def _contains_split_marker(value: str) -> bool:
    return bool({"dev", "public"} & set(_tokens(value)))


def _frequency_bucket(frequency: str) -> str:
    if _contains_split_marker(frequency):
        _fail("profile frequency contains a Dev/Public marker")
    compact = frequency.strip().casefold()
    tokens = set(_tokens(frequency))
    if compact in {"s", "sec", "t", "min", "h"} or tokens & {
        "second", "seconds", "minute", "minutes", "hour", "hours",
        "hourly", "subdaily",
    }:
        return "subdaily"
    if compact in {"d", "b"} or tokens & {"day", "days", "daily", "business"}:
        return "daily"
    if compact == "w" or tokens & {"week", "weeks", "weekly"}:
        return "weekly"
    if compact in {"m", "ms"} or tokens & {"month", "months", "monthly"}:
        return "monthly"
    if compact == "q" or tokens & {"quarter", "quarters", "quarterly"}:
        return "quarterly"
    if compact in {"y", "a"} or tokens & {"year", "years", "yearly", "annual"}:
        return "yearly"
    return "other"


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
    groups.append(f"frequency:{_frequency_bucket(profile.frequency)}")
    return tuple(groups)


def _comparison_projection(
    child_name: str,
    child: ChampionScore,
    diagnostics: dict[str, object],
) -> _ProposerComparison:
    return _ProposerComparison(
        candidate_name=child_name,
        mean_delta_smae=float(
            diagnostics["child_mean_smae"] - diagnostics["parent_mean_smae"]
        ),
        mean_delta_srmse=float(
            diagnostics["child_mean_srmse"] - diagnostics["parent_mean_srmse"]
        ),
        joint_improvement=float(diagnostics["joint_improvement"]),
        coverage=child.coverage,
        failure_rate=child.failure_rate,
        p90_smae=float(diagnostics["child_p90_smae"]),
        p95_smae=float(diagnostics["child_p95_smae"]),
        p90_srmse=float(diagnostics["child_p90_srmse"]),
        p95_srmse=float(diagnostics["child_p95_srmse"]),
        p90_smae_raw=float(diagnostics["child_p90_smae_raw"]),
        p95_smae_raw=float(diagnostics["child_p95_smae_raw"]),
        p90_srmse_raw=float(diagnostics["child_p90_srmse_raw"]),
        p95_srmse_raw=float(diagnostics["child_p95_srmse_raw"]),
        smae_clipped_count=int(diagnostics["child_smae_clipped_count"]),
        srmse_clipped_count=int(diagnostics["child_srmse_clipped_count"]),
        max_regret_smae=float(diagnostics["max_regret_smae"]),
        max_regret_srmse=float(diagnostics["max_regret_srmse"]),
        improved_folds=int(diagnostics["improved_folds"]),
        total_folds=int(diagnostics["total_folds"]),
        wins=diagnostics["wtl"].wins,
        ties=diagnostics["wtl"].ties,
        losses=diagnostics["wtl"].losses,
    )


def _reject_identity_projection(value: str, task_ids: tuple[str, ...]) -> None:
    emitted = value.casefold()
    suffix = emitted.split(":", 1)[-1]
    identities = {task_id.casefold() for task_id in task_ids}
    if emitted in identities or suffix in identities:
        _fail("proposer evidence string matches a known task identity")


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
        _public_identifier(comparison.parent_name, "parent_name")
        _public_identifier(comparison.child_name, "child_name")
        if comparison.parent_name == comparison.child_name:
            _fail("Build comparisons require distinct Parent and Child names")
        _assert_safe_candidate_name(comparison.parent_name)
        _assert_safe_candidate_name(comparison.child_name)

    row_map = {(row.candidate_name, row.task_id): row for row in snapshot}
    task_ids = tuple(dict.fromkeys(row.task_id for row in snapshot))
    projections: list[_ProposerComparison] = []
    morphology: list[MorphologyAggregate] = []
    for comparison in comparison_snapshot:
        _reject_identity_projection(comparison.parent_name, task_ids)
        _reject_identity_projection(comparison.child_name, task_ids)
        parent_score = score_policy(snapshot, comparison.parent_name)
        child_score = score_policy(snapshot, comparison.child_name)
        diagnostics = _paired_diagnostics(
            parent_score, child_score, tolerance=_TIE_EPSILON
        )
        projections.append(
            _comparison_projection(comparison.child_name, child_score, diagnostics)
        )
        grouped: dict[str, list[tuple[ChampionTaskRow, ChampionTaskRow]]] = {}
        for task_id in task_ids:
            parent = row_map.get((comparison.parent_name, task_id))
            child = row_map.get((comparison.child_name, task_id))
            if parent is None or child is None:
                _fail("Build comparison rows require complete Parent/Child pairs")
            if parent.profile != child.profile:
                _fail("mislabeled Build Parent/Child profiles disagree")
            for group_id in _profile_groups(parent.profile):
                _reject_identity_projection(group_id, task_ids)
                grouped.setdefault(group_id, []).append((parent, child))
        for group_id, pairs in grouped.items():
            paired_values: list[tuple[float, float]] = []
            for parent, child in pairs:
                if parent.forecast is None or child.forecast is None:
                    continue
                assert parent.truth is not None and child.truth is not None
                if parent.truth != child.truth:
                    _fail("mislabeled Build rows contain different truths for one task")
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
            sorted(projections, key=lambda item: item.candidate_name)
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
