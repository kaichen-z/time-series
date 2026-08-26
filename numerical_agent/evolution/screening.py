"""History-only task profiles and task-conditioned candidate screening."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .analysis_skills_template import analyze_series
from .execution import CRASHED, INVALID, NOT_APPLICABLE, SUCCESS, Outcome, Task


_FEATURE_FIELDS = frozenset(
    {
        "frequency",
        "history_length",
        "horizon",
        "zero_fraction",
        "signed",
        "integer_valued",
        "trend_direction",
        "trend_strength",
        "periodicity_periods",
        "periodicity_strength",
        "periodicity_confidence",
        "outlier_fraction",
        "noise_relative_scale",
        "likely_stationary",
        "stationarity_score",
        "recent_regime_start",
        "recent_regime_confidence",
        "intermittency_adi",
        "intermittency_cv2",
    }
)
_FEATURE_OPERATORS = frozenset({"<", "<=", "==", ">=", ">", "in"})
_SELECTABLE_STATUSES = frozenset({"keep", "specialized"})
_STATUSES = frozenset({"keep", "specialized", "repair", "quarantine", "discard"})
_FAMILIES = frozenset({"statistical", "tsfm", "combined"})


@dataclass(frozen=True)
class TaskProfile:
    """Deterministic numerical characteristics available before the future exists."""

    task_id: str
    frequency: str
    history_length: int
    horizon: int
    zero_fraction: float
    signed: bool
    integer_valued: bool
    trend_direction: str
    trend_strength: float
    periodicity_periods: tuple[int, ...]
    periodicity_strength: float
    periodicity_confidence: float
    outlier_fraction: float
    noise_relative_scale: float
    likely_stationary: bool
    stationarity_score: float
    recent_regime_start: int | None
    recent_regime_confidence: float
    intermittency_adi: float
    intermittency_cv2: float

    def __post_init__(self) -> None:
        if self.history_length <= 0:
            raise ValueError("history_length must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        numerical = (
            self.zero_fraction,
            self.trend_strength,
            self.periodicity_strength,
            self.periodicity_confidence,
            self.outlier_fraction,
            self.noise_relative_scale,
            self.stationarity_score,
            self.recent_regime_confidence,
            self.intermittency_adi,
            self.intermittency_cv2,
        )
        if any(not math.isfinite(value) for value in numerical):
            raise ValueError("task profile contains a non-finite measurement")

    def to_public_payload(self) -> dict[str, object]:
        """Return the inference payload without task identity or future labels."""
        return {
            "frequency": self.frequency,
            "history_length": self.history_length,
            "horizon": self.horizon,
            "zero_fraction": self.zero_fraction,
            "signed": self.signed,
            "integer_valued": self.integer_valued,
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "periodicity_periods": list(self.periodicity_periods),
            "periodicity_strength": self.periodicity_strength,
            "periodicity_confidence": self.periodicity_confidence,
            "outlier_fraction": self.outlier_fraction,
            "noise_relative_scale": self.noise_relative_scale,
            "likely_stationary": self.likely_stationary,
            "stationarity_score": self.stationarity_score,
            "recent_regime_start": self.recent_regime_start,
            "recent_regime_confidence": self.recent_regime_confidence,
            "intermittency_adi": self.intermittency_adi,
            "intermittency_cv2": self.intermittency_cv2,
        }


def profile_task(task: Task) -> TaskProfile:
    """Build one label-free profile from a task's historical prefix."""
    values = tuple(float(value) for value in task.history)
    analysis = analyze_series(values, task.frequency)
    periodicity = _mapping(analysis, "periodicity")
    outliers = _mapping(analysis, "outliers")
    trend = _mapping(analysis, "trend")
    intermittency = _mapping(analysis, "intermittency")
    noise = _mapping(analysis, "noise")
    stationarity = _mapping(analysis, "stationarity")
    regime = _mapping(analysis, "recent_regime")
    periods = periodicity.get("candidate_periods", ())
    indices = outliers.get("indices", ())
    integer_ratio = sum(abs(value - round(value)) <= 1e-8 for value in values) / len(values)
    return TaskProfile(
        task_id=task.task_id,
        frequency=task.frequency,
        history_length=len(values),
        horizon=task.horizon,
        zero_fraction=sum(abs(value) <= 1e-12 for value in values) / len(values),
        signed=any(value < 0.0 for value in values),
        integer_valued=integer_ratio >= 0.98,
        trend_direction=str(trend["direction"]),
        trend_strength=float(trend["strength"]),
        periodicity_periods=tuple(int(value) for value in periods),
        periodicity_strength=float(periodicity["strength"]),
        periodicity_confidence=float(periodicity["confidence"]),
        outlier_fraction=len(tuple(indices)) / len(values),
        noise_relative_scale=float(noise["relative_scale"]),
        likely_stationary=bool(stationarity["likely_stationary"]),
        stationarity_score=float(stationarity["score"]),
        recent_regime_start=(
            int(regime["regime_start"])
            if regime.get("regime_start") is not None
            else None
        ),
        recent_regime_confidence=float(regime["confidence"]),
        intermittency_adi=float(intermittency["average_nonzero_gap"]),
        intermittency_cv2=float(intermittency["nonzero_cv2"]),
    )


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"analysis profile is missing {key}")
    return nested


@dataclass(frozen=True)
class FeatureTest:
    field: str
    operator: str
    value: object

    def __post_init__(self) -> None:
        if self.field not in _FEATURE_FIELDS:
            raise ValueError(f"unsupported profile field {self.field!r}")
        if self.operator not in _FEATURE_OPERATORS:
            raise ValueError(f"unsupported feature operator {self.operator!r}")
        values = self.value if self.operator == "in" else (self.value,)
        if self.operator == "in" and not isinstance(values, tuple):
            raise ValueError("the 'in' feature operator requires a tuple literal")
        for value in values:
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                raise ValueError("feature values must be finite literals")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("feature values must be finite")

    def matches(self, profile: TaskProfile) -> bool:
        current = getattr(profile, self.field)
        if self.operator == "<":
            return bool(current < self.value)  # type: ignore[operator]
        if self.operator == "<=":
            return bool(current <= self.value)  # type: ignore[operator]
        if self.operator == "==":
            return current == self.value
        if self.operator == ">=":
            return bool(current >= self.value)  # type: ignore[operator]
        if self.operator == ">":
            return bool(current > self.value)  # type: ignore[operator]
        assert self.operator == "in" and isinstance(self.value, tuple)
        if isinstance(current, tuple):
            return any(value in self.value for value in current)
        return current in self.value

    def reason_code(self) -> str:
        value = self.value
        if isinstance(value, tuple):
            rendered = "_or_".join(str(item) for item in value)
        else:
            rendered = str(value)
        operator = {
            "<": "lt",
            "<=": "le",
            "==": "eq",
            ">=": "ge",
            ">": "gt",
            "in": "in",
        }[self.operator]
        return f"{self.field}_{operator}_{rendered}"


@dataclass(frozen=True)
class ApplicabilityClause:
    all_tags: tuple[str, ...] = ()
    feature_tests: tuple[FeatureTest, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.all_tags)) != len(self.all_tags):
            raise ValueError("applicability clause contains duplicate tags")
        if any(not tag.strip() for tag in self.all_tags):
            raise ValueError("applicability tags must not be empty")

    def matches(self, profile: TaskProfile) -> bool:
        tags = profile_tags(profile)
        return set(self.all_tags).issubset(tags) and all(
            test.matches(profile) for test in self.feature_tests
        )

    def reason_codes(self) -> tuple[str, ...]:
        return self.all_tags + tuple(test.reason_code() for test in self.feature_tests)


@dataclass(frozen=True)
class ApplicabilityPolicy:
    any_of: tuple[ApplicabilityClause, ...] = ()

    def match(self, profile: TaskProfile) -> int | None:
        if not self.any_of:
            return -1
        return next(
            (index for index, clause in enumerate(self.any_of) if clause.matches(profile)),
            None,
        )


@dataclass(frozen=True)
class ScreeningEntry:
    name: str
    family: str
    status: str
    applicability: ApplicabilityPolicy
    reason: str

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid screening candidate name {self.name!r}")
        if self.family not in _FAMILIES:
            raise ValueError(f"invalid screening family {self.family!r}")
        if self.status not in _STATUSES:
            raise ValueError(f"invalid screening status {self.status!r}")
        if self.status == "specialized" and not self.applicability.any_of:
            raise ValueError("specialized candidates require an applicability policy")
        if not self.reason.strip():
            raise ValueError("screening entries require a reason")


@dataclass(frozen=True)
class ScreeningPolicy:
    entries: tuple[ScreeningEntry, ...]
    fallback_names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(entry.name for entry in self.entries)
        if not names or len(names) != len(set(names)):
            raise ValueError("screening policy requires unique candidate names")
        if len(self.fallback_names) != len(set(self.fallback_names)):
            raise ValueError("screening fallback names must be unique")
        if unknown := set(self.fallback_names) - set(names):
            raise ValueError(f"unknown screening fallback candidates: {sorted(unknown)!r}")

    def get(self, name: str) -> ScreeningEntry | None:
        return next((entry for entry in self.entries if entry.name == name), None)

    def fingerprint(self) -> str:
        return _hash_payload(_policy_payload(self))


@dataclass(frozen=True)
class ActiveCandidate:
    name: str
    family: str
    matched_clause: int
    screen_confidence: float
    reason_codes: tuple[str, ...]
    fallback: bool = False


@dataclass(frozen=True)
class ExcludedCandidate:
    name: str
    family: str
    reason_code: str


@dataclass(frozen=True)
class ActiveDictionary:
    task_profile_hash: str
    screening_policy_hash: str
    active: tuple[ActiveCandidate, ...]
    excluded: tuple[ExcludedCandidate, ...]
    fallback_applied: bool


@dataclass(frozen=True)
class ScreeningScore:
    task_count: int
    coverage: float
    active_success_rate: float
    failure_exposure: float
    not_applicable_exposure: float
    mean_active_failures: float
    mean_active_not_applicable: float
    compression: float
    mean_active_candidates: float
    median_active_candidates: float
    global_oracle_retention: float
    mean_active_oracle_regret: float
    mean_active_families: float
    fallback_rate: float
    active_counts: Mapping[str, int]
    min_active_candidates: int
    max_active_candidates: int
    unique_active_dictionaries: int
    mean_pairwise_jaccard: float
    conditioned_entries_by_family: Mapping[str, int]


@dataclass(frozen=True)
class ScreeningGateResult:
    accepted: bool
    reason: str
    improved_dimensions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScreeningConstraints:
    """Acceptance limits for one task-conditioned screening experiment."""

    baseline_method: str = "toto_2_0"
    min_active_candidates: int = 12
    max_active_candidates: int = 103
    min_unique_active_dictionaries: int = 3
    max_mean_pairwise_jaccard: float = 0.995
    min_group_support: int = 4
    min_dev_oracle_retention: float = 0.9
    required_conditioned_families: tuple[str, ...] = (
        "statistical",
        "tsfm",
        "combined",
    )

    def __post_init__(self) -> None:
        if not self.baseline_method.isidentifier():
            raise ValueError("baseline_method must be a Python identifier")
        if self.min_active_candidates < 1:
            raise ValueError("min_active_candidates must be positive")
        if self.max_active_candidates < self.min_active_candidates:
            raise ValueError("max_active_candidates must not be smaller than the minimum")
        if self.min_unique_active_dictionaries < 1:
            raise ValueError("min_unique_active_dictionaries must be positive")
        if not 0.0 <= self.max_mean_pairwise_jaccard <= 1.0:
            raise ValueError("max_mean_pairwise_jaccard must be between zero and one")
        if self.min_group_support < 1:
            raise ValueError("min_group_support must be positive")
        if not 0.0 <= self.min_dev_oracle_retention <= 1.0:
            raise ValueError("min_dev_oracle_retention must be between zero and one")
        if (
            len(self.required_conditioned_families)
            != len(set(self.required_conditioned_families))
            or set(self.required_conditioned_families) - _FAMILIES
        ):
            raise ValueError("required_conditioned_families must be unique known families")


def profile_tags(profile: TaskProfile) -> frozenset[str]:
    """Return the backward-compatible coarse tags derived from a rich profile."""
    tags = {
        f"frequency:{profile.frequency}",
        f"history:{_bucket(profile.history_length, (48, 168, 512))}",
        f"horizon:{_bucket(profile.horizon, (8, 24, 96))}",
        "signed" if profile.signed else "nonnegative",
        "integer_valued" if profile.integer_valued else "continuous_valued",
        "trending" if profile.trend_direction != "flat" else "flat",
    }
    intermittent = profile.zero_fraction > 0.3 or profile.intermittency_adi > 1.32
    tags.add("intermittent" if intermittent else "dense")
    tags.add(
        "many_zeros"
        if profile.zero_fraction > 0.3
        else ("no_zeros" if profile.zero_fraction <= 1e-12 else "some_zeros")
    )
    tags.update(task_group_tags(profile))
    return frozenset(tags)


def task_group_tags(profile: TaskProfile) -> frozenset[str]:
    """Assign one history-only stratum for each approved screening dimension."""
    if profile.periodicity_strength >= 0.6 and profile.periodicity_confidence >= 0.5:
        periodicity = "strong"
    elif profile.periodicity_periods or profile.periodicity_strength >= 0.25:
        periodicity = "weak"
    else:
        periodicity = "none"

    if profile.trend_direction == "flat" or profile.trend_strength < 0.25:
        trend = "flat"
    elif profile.trend_strength >= 0.6:
        trend = f"strong_{profile.trend_direction}"
    else:
        trend = f"weak_{profile.trend_direction}"

    intermittent = profile.zero_fraction > 0.3 or profile.intermittency_adi > 1.32
    if intermittent and profile.intermittency_cv2 > 0.49:
        intermittency = "lumpy"
    else:
        intermittency = "intermittent" if intermittent else "dense"

    regime = "recent_shift" if profile.recent_regime_confidence >= 0.5 else "stable"
    return frozenset(
        {
            f"periodicity:{periodicity}",
            f"trend:{trend}",
            f"intermittency:{intermittency}",
            f"regime:{regime}",
            f"frequency:{profile.frequency}",
            f"history:{_bucket(profile.history_length, (48, 168, 512))}",
            f"horizon:{_bucket(profile.horizon, (8, 24, 96))}",
        }
    )


def materialize_active_dictionary(
    policy: ScreeningPolicy,
    profile: TaskProfile,
) -> ActiveDictionary:
    """Apply a frozen screening policy to one label-free task profile."""
    active: list[ActiveCandidate] = []
    excluded_by_name: dict[str, ExcludedCandidate] = {}
    for entry in policy.entries:
        if entry.status not in _SELECTABLE_STATUSES:
            excluded_by_name[entry.name] = ExcludedCandidate(
                entry.name, entry.family, f"status_{entry.status}"
            )
            continue
        matched = entry.applicability.match(profile)
        if matched is None:
            excluded_by_name[entry.name] = ExcludedCandidate(
                entry.name, entry.family, "applicability_not_matched"
            )
            continue
        clause = entry.applicability.any_of[matched] if matched >= 0 else None
        active.append(
            ActiveCandidate(
                entry.name,
                entry.family,
                matched,
                1.0 if clause is not None else 0.5,
                clause.reason_codes() if clause is not None else ("broad_applicability",),
            )
        )

    fallback_applied = False
    available_families = {
        entry.family for entry in policy.entries if entry.status in _SELECTABLE_STATUSES
    }
    while _needs_fallback(active, available_families):
        name = next(
            (
                fallback
                for fallback in policy.fallback_names
                if all(candidate.name != fallback for candidate in active)
                and (entry := policy.get(fallback)) is not None
                and entry.status in _SELECTABLE_STATUSES
            ),
            None,
        )
        if name is None:
            break
        entry = policy.get(name)
        assert entry is not None
        active.append(
            ActiveCandidate(
                entry.name,
                entry.family,
                -1,
                0.0,
                ("reviewed_fallback",),
                fallback=True,
            )
        )
        excluded_by_name.pop(entry.name, None)
        fallback_applied = True

    active_names = {candidate.name for candidate in active}
    excluded = tuple(
        excluded_by_name[entry.name]
        for entry in policy.entries
        if entry.name not in active_names
    )
    return ActiveDictionary(
        task_profile_hash=_hash_payload(profile.to_public_payload()),
        screening_policy_hash=policy.fingerprint(),
        active=tuple(active),
        excluded=excluded,
        fallback_applied=fallback_applied,
    )


def evaluate_screening(
    policy: ScreeningPolicy,
    tasks: Sequence[Task],
    outcomes: Sequence[Outcome],
) -> ScreeningScore:
    """Score candidate eligibility without choosing a final forecast."""
    by_key = {(row.method, row.task_id): row for row in outcomes}
    active_counts: dict[str, int] = {}
    active_attempts = 0
    active_successes = 0
    active_failures = 0
    active_not_applicable = 0
    covered = 0
    oracle_tasks = 0
    oracle_retained = 0
    regrets: list[float] = []
    family_counts: list[int] = []
    fallback_count = 0
    policy_names = {entry.name for entry in policy.entries}
    active_signatures: list[frozenset[str]] = []
    profiles = {task.task_id: profile_task(task) for task in tasks}

    for task in tasks:
        active_dictionary = materialize_active_dictionary(policy, profiles[task.task_id])
        active_names = {candidate.name for candidate in active_dictionary.active}
        active_signatures.append(frozenset(active_names))
        active_counts[task.task_id] = len(active_names)
        active_attempts += len(active_names)
        family_counts.append(len({candidate.family for candidate in active_dictionary.active}))
        fallback_count += int(active_dictionary.fallback_applied)
        active_rows = []
        for name in active_names:
            row = by_key.get((name, task.task_id))
            if row is not None and row.status == SUCCESS and _finite_mase(row):
                active_successes += 1
                active_rows.append(row)
            elif row is not None and row.status == NOT_APPLICABLE:
                active_not_applicable += 1
            else:
                active_failures += 1
        covered += int(bool(active_rows))

        global_rows = [
            row
            for row in outcomes
            if row.task_id == task.task_id
            and row.method in policy_names
            and row.status == SUCCESS
            and _finite_mase(row)
        ]
        if not global_rows:
            regrets.append(10.0)
            continue
        oracle_tasks += 1
        best_global = min(float(row.mase) for row in global_rows)  # type: ignore[arg-type]
        global_oracles = {
            row.method
            for row in global_rows
            if abs(float(row.mase) - best_global) <= 1e-12  # type: ignore[arg-type]
        }
        oracle_retained += int(bool(global_oracles.intersection(active_names)))
        if active_rows:
            best_active = min(float(row.mase) for row in active_rows)  # type: ignore[arg-type]
            regrets.append(max(0.0, (best_active - best_global) / (1.0 + best_global)))
        else:
            regrets.append(10.0)

    denominator = max(1, active_attempts)
    task_denominator = max(1, len(tasks))
    counts = tuple(active_counts.values())
    conditioned = {
        family: sum(
            entry.family == family
            and entry.status == "specialized"
            and any(entry.applicability.match(profile) is not None for profile in profiles.values())
            for entry in policy.entries
        )
        for family in sorted(_FAMILIES)
    }
    return ScreeningScore(
        task_count=len(tasks),
        coverage=covered / task_denominator,
        active_success_rate=active_successes / denominator,
        failure_exposure=active_failures / denominator,
        not_applicable_exposure=active_not_applicable / denominator,
        mean_active_failures=active_failures / task_denominator,
        mean_active_not_applicable=active_not_applicable / task_denominator,
        compression=active_attempts / max(1, len(policy.entries) * len(tasks)),
        mean_active_candidates=sum(counts) / task_denominator,
        median_active_candidates=_median(counts),
        global_oracle_retention=oracle_retained / max(1, oracle_tasks),
        mean_active_oracle_regret=sum(regrets) / task_denominator,
        mean_active_families=sum(family_counts) / task_denominator,
        fallback_rate=fallback_count / task_denominator,
        active_counts=active_counts,
        min_active_candidates=min(counts, default=0),
        max_active_candidates=max(counts, default=0),
        unique_active_dictionaries=len(set(active_signatures)),
        mean_pairwise_jaccard=_mean_pairwise_jaccard(active_signatures),
        conditioned_entries_by_family=conditioned,
    )


def compare_screening(
    train_parent: ScreeningScore,
    train_child: ScreeningScore,
    dev_parent: ScreeningScore,
    dev_child: ScreeningScore,
    *,
    constraints: ScreeningConstraints | None = None,
    enforce_final_constraints: bool = False,
) -> ScreeningGateResult:
    """Apply the frozen Train-improvement and read-only Dev acceptance gate."""
    tolerance = 1e-12
    if train_child.coverage < 1.0 - tolerance or dev_child.coverage < 1.0 - tolerance:
        return ScreeningGateResult(False, "rejected: screening coverage is below 100%")
    if train_child.global_oracle_retention < 1.0 - tolerance:
        return ScreeningGateResult(
            False, "rejected: Train oracle retention must remain 100%"
        )
    required_dev_oracle = (
        constraints.min_dev_oracle_retention if constraints is not None else 1.0
    )
    if dev_child.global_oracle_retention < required_dev_oracle - tolerance:
        return ScreeningGateResult(
            False,
            "rejected: Dev oracle retention is below the bounded safety floor",
        )
    if dev_child.mean_active_oracle_regret > dev_parent.mean_active_oracle_regret + 0.01 + tolerance:
        return ScreeningGateResult(False, "rejected: Dev active-oracle regret regressed")
    if dev_child.mean_active_failures > dev_parent.mean_active_failures + tolerance:
        return ScreeningGateResult(
            False, "rejected: Dev failure exposure count increased"
        )
    if train_child.mean_active_failures > train_parent.mean_active_failures + tolerance:
        return ScreeningGateResult(
            False, "rejected: Train failure exposure count increased"
        )
    if constraints is not None:
        if (
            train_child.min_active_candidates < constraints.min_active_candidates
            or dev_child.min_active_candidates < constraints.min_active_candidates
        ):
            return ScreeningGateResult(False, "rejected: active candidate pool is too small")
        if enforce_final_constraints and (
            train_child.max_active_candidates > constraints.max_active_candidates
            or dev_child.max_active_candidates > constraints.max_active_candidates
        ):
            return ScreeningGateResult(False, "rejected: active candidate pool is too large")
        if enforce_final_constraints and (
            train_child.unique_active_dictionaries
            < min(constraints.min_unique_active_dictionaries, train_child.task_count)
            or dev_child.unique_active_dictionaries
            < min(constraints.min_unique_active_dictionaries, dev_child.task_count)
            or (
                dev_child.task_count > 1
                and dev_child.mean_pairwise_jaccard
                > constraints.max_mean_pairwise_jaccard + tolerance
            )
        ):
            return ScreeningGateResult(
                False, "rejected: insufficient task-conditioned diversity"
            )
        if enforce_final_constraints and any(
            dev_child.conditioned_entries_by_family.get(family, 0) < 1
            for family in constraints.required_conditioned_families
        ):
            return ScreeningGateResult(
                False, "rejected: required families lack conditioned entries"
            )

    dimensions = {
        "active_success_rate": (
            train_child.active_success_rate > train_parent.active_success_rate + tolerance,
            dev_child.active_success_rate + 0.005 >= dev_parent.active_success_rate,
        ),
        "failure_exposure": (
            train_child.failure_exposure < train_parent.failure_exposure - tolerance,
            dev_child.failure_exposure <= dev_parent.failure_exposure + 0.005,
        ),
        "not_applicable_exposure": (
            train_child.not_applicable_exposure
            < train_parent.not_applicable_exposure - tolerance,
            dev_child.not_applicable_exposure
            <= dev_parent.not_applicable_exposure + 0.005,
        ),
        "compression": (
            train_child.compression < train_parent.compression - tolerance,
            dev_child.compression <= dev_parent.compression + 0.005,
        ),
    }
    improved = tuple(
        name for name, (train_better, dev_safe) in dimensions.items() if train_better and dev_safe
    )
    if not improved:
        return ScreeningGateResult(
            False,
            "rejected: no Train screening dimension improved safely on Dev",
        )
    return ScreeningGateResult(
        True,
        "accepted: screening improved on Train without a Dev regression",
        improved,
    )


def _needs_fallback(
    active: Sequence[ActiveCandidate], available_families: set[str]
) -> bool:
    families = {candidate.family for candidate in active}
    if len(active) < 3:
        return True
    if "statistical" in available_families and "statistical" not in families:
        return True
    return "tsfm" in available_families and "tsfm" not in families


def _finite_mase(row: Outcome) -> bool:
    return row.mase is not None and math.isfinite(float(row.mase))


def _median(values: Sequence[int]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _mean_pairwise_jaccard(signatures: Sequence[frozenset[str]]) -> float:
    if len(signatures) < 2:
        return 1.0
    similarities = []
    for left_index, left in enumerate(signatures[:-1]):
        for right in signatures[left_index + 1 :]:
            union = left | right
            similarities.append(len(left & right) / len(union) if union else 1.0)
    return sum(similarities) / len(similarities)


def _bucket(value: int, edges: Sequence[int]) -> str:
    return next((f"le_{edge}" for edge in edges if value <= edge), f"gt_{edges[-1]}")


def _policy_payload(policy: ScreeningPolicy) -> dict[str, object]:
    return {
        "entries": [
            {
                "name": entry.name,
                "family": entry.family,
                "status": entry.status,
                "reason": entry.reason,
                "any_of": [
                    {
                        "all_tags": list(clause.all_tags),
                        "feature_tests": [
                            {
                                "field": test.field,
                                "operator": test.operator,
                                "value": test.value,
                            }
                            for test in clause.feature_tests
                        ],
                    }
                    for clause in entry.applicability.any_of
                ],
            }
            for entry in policy.entries
        ],
        "fallback_names": list(policy.fallback_names),
    }


def _hash_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
