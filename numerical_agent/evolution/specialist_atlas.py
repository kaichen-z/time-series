"""Train-only specialist discovery and history-only routing features."""

from __future__ import annotations

import hashlib
import math
import statistics
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from common.metrics import drcik_point_metrics, joint_scaled_error
from common.payload import canonical_json_bytes

from .numerical_selector import CandidateDiagnostics
from .task_local_evolution import GroupFoldManifest, TaskLocalTaskRow


_ATLAS_BUILD_TASK_COUNT = 64
_NONFINITE_REGRET_SENTINEL = 1.0


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
        if self.maximum_predicted_regret > 0.25:
            raise ValueError("Atlas predicted regret cannot exceed one quarter")


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


@dataclass(frozen=True)
class AtlasMaterializedCandidate:
    """One history-only materialized forecast offered to the Atlas router."""

    candidate_name: str
    family: str
    feature: AtlasFeature | None
    forecast: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            type(self.candidate_name) is not str
            or not self.candidate_name.isidentifier()
        ):
            raise ValueError("Atlas materialized candidate must be an identifier")
        if type(self.family) is not str or self.family not in {
            "statistical",
            "tsfm",
            "combined",
        }:
            raise ValueError("Atlas materialized candidate family is unsupported")
        if self.feature is not None:
            if type(self.feature) is not AtlasFeature:
                raise ValueError("Atlas materialized feature must be exact or None")
            AtlasFeature.__post_init__(self.feature)
        if (
            type(self.forecast) is not tuple
            or not self.forecast
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in self.forecast
            )
        ):
            raise ValueError("Atlas materialized forecast must be a finite float tuple")


@dataclass(frozen=True)
class AtlasTaskCase:
    """Label-free Atlas routing input for one already-materialized task."""

    group_sha256: str
    anchor: AtlasMaterializedCandidate
    candidates: tuple[AtlasMaterializedCandidate, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.group_sha256, "Atlas task group")
        if type(self.anchor) is not AtlasMaterializedCandidate:
            raise ValueError("Atlas task case requires an exact anchor")
        AtlasMaterializedCandidate.__post_init__(self.anchor)
        if self.anchor.feature is not None:
            raise ValueError("Atlas anchor cannot carry a candidate routing feature")
        if type(self.candidates) is not tuple or any(
            type(candidate) is not AtlasMaterializedCandidate
            for candidate in self.candidates
        ):
            raise ValueError("Atlas task candidates must be an exact tuple")
        names: list[str] = []
        for candidate in self.candidates:
            AtlasMaterializedCandidate.__post_init__(candidate)
            if candidate.feature is None:
                raise ValueError("Atlas specialist candidates require routing features")
            if len(candidate.forecast) != len(self.anchor.forecast):
                raise ValueError("Atlas task forecasts must have one common horizon")
            names.append(candidate.candidate_name)
        if self.anchor.candidate_name in names or len(names) != len(set(names)):
            raise ValueError("Atlas task candidate identities must be unique")


@dataclass(frozen=True)
class AtlasCandidateEstimate:
    candidate_name: str
    independent_groups: int
    neighbor_count: int
    win_probability: float
    effect_smae: float
    effect_srmse: float
    lower_joint_effect: float
    predicted_regret: float

    def __post_init__(self) -> None:
        if (
            type(self.candidate_name) is not str
            or not self.candidate_name.isidentifier()
        ):
            raise ValueError("Atlas estimate candidate must be an identifier")
        if type(self.independent_groups) is not int or self.independent_groups < 0:
            raise ValueError("Atlas estimate group support must be nonnegative")
        if type(self.neighbor_count) is not int or self.neighbor_count < 0:
            raise ValueError("Atlas estimate neighbor count must be nonnegative")
        for name in (
            "win_probability",
            "effect_smae",
            "effect_srmse",
            "lower_joint_effect",
            "predicted_regret",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"Atlas estimate {name} must be a finite float")
        if not 0.0 <= self.win_probability <= 1.0 or self.predicted_regret < 0.0:
            raise ValueError("Atlas estimate probability or regret is outside bounds")


@dataclass(frozen=True)
class AtlasModel:
    """Frozen nearest-neighbor evidence fitted from one Build partition."""

    feature_scales: tuple[float, ...]
    training_task_sha256s: tuple[str, ...]
    training_group_sha256s: tuple[str, ...]
    selected_pool: tuple[str, ...]
    neighbor_count: int
    records: tuple[AtlasTrainingRecord, ...]

    def __post_init__(self) -> None:
        if (
            type(self.feature_scales) is not tuple
            or len(self.feature_scales) != 15
            or any(
                type(value) is not float or not math.isfinite(value) or value <= 0.0
                for value in self.feature_scales
            )
        ):
            raise ValueError("Atlas model scales must be 15 positive finite floats")
        for values, label in (
            (self.training_task_sha256s, "task"),
            (self.training_group_sha256s, "group"),
        ):
            if (
                type(values) is not tuple
                or values != tuple(sorted(values))
                or len(values) != len(set(values))
            ):
                raise ValueError(
                    f"Atlas model training {label} hashes must be unique sorted"
                )
            for value in values:
                _require_sha256(value, f"Atlas model training {label}")
        if (
            type(self.selected_pool) is not tuple
            or not self.selected_pool
            or self.selected_pool[0] != "toto_2_0"
            or len(self.selected_pool) != len(set(self.selected_pool))
            or any(
                type(name) is not str or not name.isidentifier()
                for name in self.selected_pool
            )
        ):
            raise ValueError("Atlas model pool must be unique and Champion-first")
        if type(self.neighbor_count) is not int or not 2 <= self.neighbor_count <= 31:
            raise ValueError("Atlas model neighbor count is outside its closed range")
        if type(self.records) is not tuple or any(
            type(record) is not AtlasTrainingRecord for record in self.records
        ):
            raise ValueError("Atlas model records must be an exact tuple")
        keys: list[tuple[str, str]] = []
        for record in self.records:
            AtlasTrainingRecord.__post_init__(record)
            if record.candidate_name not in self.selected_pool[1:]:
                raise ValueError("Atlas model record is outside its selected pool")
            if record.group_sha256 not in self.training_group_sha256s:
                raise ValueError("Atlas model record is outside its training groups")
            keys.append((record.candidate_name, record.group_sha256))
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError(
                "Atlas model records must be unique and canonically sorted"
            )

    @property
    def training_task_count(self) -> int:
        return len(self.training_task_sha256s)

    def to_payload(self) -> dict[str, object]:
        return {
            "feature_scales": list(self.feature_scales),
            "training_task_sha256s": list(self.training_task_sha256s),
            "training_group_sha256s": list(self.training_group_sha256s),
            "selected_pool": list(self.selected_pool),
            "neighbor_count": self.neighbor_count,
            "records": [
                {
                    "candidate_name": record.candidate_name,
                    "family": record.family,
                    "group_sha256": record.group_sha256,
                    "feature": {
                        "categorical": list(record.feature.categorical),
                        "numeric": list(record.feature.numeric),
                    },
                    "improvement_smae": record.improvement_smae,
                    "improvement_srmse": record.improvement_srmse,
                    "regret_smae_raw": record.regret_smae_raw,
                    "regret_srmse_raw": record.regret_srmse_raw,
                }
                for record in self.records
            ],
        }


@dataclass(frozen=True)
class AtlasRelease:
    schema_version: int
    policy: AtlasPolicy
    full_build_model: AtlasModel
    build_fold_models: tuple[tuple[int, AtlasModel], ...]
    build_fold_held_out_task_sha256s: tuple[tuple[int, tuple[str, ...]], ...]
    source_sha256: str
    policy_sha256: str
    fold_manifest_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Atlas release schema must be exactly one")
        if type(self.policy) is not AtlasPolicy:
            raise ValueError("Atlas release requires an exact policy")
        AtlasPolicy.__post_init__(self.policy)
        if type(self.full_build_model) is not AtlasModel:
            raise ValueError("Atlas release requires an exact full-Build model")
        AtlasModel.__post_init__(self.full_build_model)
        if self.full_build_model.training_task_count != _ATLAS_BUILD_TASK_COUNT:
            raise ValueError("Atlas release requires the exact 64-task Build universe")
        if type(self.build_fold_models) is not tuple or tuple(
            fold for fold, _model in self.build_fold_models
        ) != (0, 1, 2, 3, 4):
            raise ValueError("Atlas release requires Build models zero through four")
        for _fold, model in self.build_fold_models:
            if type(model) is not AtlasModel:
                raise ValueError("Atlas fold models must be exact")
            AtlasModel.__post_init__(model)
        if (
            type(self.build_fold_held_out_task_sha256s) is not tuple
            or tuple(
                fold for fold, _hashes in self.build_fold_held_out_task_sha256s
            )
            != (0, 1, 2, 3, 4)
        ):
            raise ValueError("Atlas release requires numbered held-out fold identities")
        for _fold, hashes in self.build_fold_held_out_task_sha256s:
            if (
                type(hashes) is not tuple
                or not hashes
                or hashes != tuple(sorted(hashes))
                or len(hashes) != len(set(hashes))
            ):
                raise ValueError("Atlas held-out task hashes must be unique and sorted")
            for value in hashes:
                _require_sha256(value, "Atlas held-out task")
        full_tasks = set(self.full_build_model.training_task_sha256s)
        full_groups = set(self.full_build_model.training_group_sha256s)
        omitted_tasks = [
            full_tasks - set(model.training_task_sha256s)
            for _fold, model in self.build_fold_models
        ]
        omitted_groups = [
            full_groups - set(model.training_group_sha256s)
            for _fold, model in self.build_fold_models
        ]
        expected_omissions = dict(self.build_fold_held_out_task_sha256s)
        if any(
            not omitted_task
            or not omitted_group
            or omitted_task != set(expected_omissions[fold])
            or not set(model.training_task_sha256s) < full_tasks
            or not set(model.training_group_sha256s) < full_groups
            for (fold, model), omitted_task, omitted_group in zip(
                self.build_fold_models, omitted_tasks, omitted_groups, strict=True
            )
        ):
            raise ValueError("Atlas fold model does not exclude held-out Build groups")
        if (
            set().union(*omitted_tasks) != full_tasks
            or sum(len(values) for values in omitted_tasks) != len(full_tasks)
            or set().union(*omitted_groups) != full_groups
            or sum(len(values) for values in omitted_groups) != len(full_groups)
        ):
            raise ValueError("Atlas held-out Build partitions are not disjoint and complete")
        for value, label in (
            (self.source_sha256, "source"),
            (self.policy_sha256, "policy"),
            (self.fold_manifest_sha256, "fold manifest"),
        ):
            _require_sha256(value, f"Atlas release {label}")
        if self.policy_sha256 != _sha256(_policy_payload(self.policy)):
            raise ValueError("Atlas release policy fingerprint mismatch")

    @property
    def fingerprint(self) -> str:
        return _sha256(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy": _policy_payload(self.policy),
            "full_build_model": self.full_build_model.to_payload(),
            "build_fold_models": [
                [fold, model.to_payload()] for fold, model in self.build_fold_models
            ],
            "build_fold_held_out_task_sha256s": [
                [fold, list(hashes)]
                for fold, hashes in self.build_fold_held_out_task_sha256s
            ],
            "source_sha256": self.source_sha256,
            "policy_sha256": self.policy_sha256,
            "fold_manifest_sha256": self.fold_manifest_sha256,
        }

    def validate_manifest(self, manifest: GroupFoldManifest) -> None:
        """Bind every numbered fold omission to one exact 64-task manifest."""
        if type(manifest) is not GroupFoldManifest or manifest.fold_count != 5:
            raise ValueError("Atlas release requires an exact five-fold manifest")
        if len(manifest.task_fold_map) != _ATLAS_BUILD_TASK_COUNT:
            raise ValueError("Atlas release manifest requires exactly 64 Build tasks")
        if self.fold_manifest_sha256 != _sha256(manifest.to_payload()):
            raise ValueError("Atlas release fold manifest fingerprint mismatch")
        expected_full = tuple(
            sorted(_task_sha256(task_id) for task_id in manifest.task_fold_map)
        )
        if self.full_build_model.training_task_sha256s != expected_full:
            raise ValueError("Atlas release tasks do not match the fold manifest")
        if self.build_fold_held_out_task_sha256s != _manifest_held_out_hashes(
            manifest
        ):
            raise ValueError("Atlas release numbered folds do not match the manifest")


@dataclass(frozen=True)
class AtlasTaskResult:
    group_sha256: str
    training_group_sha256s: tuple[str, ...]
    selected_supply: tuple[str, ...]
    estimates: tuple[AtlasCandidateEstimate, ...]
    forecast: tuple[float, ...]
    activated: bool
    fallback_reason: str | None

    def __post_init__(self) -> None:
        _require_sha256(self.group_sha256, "Atlas task result group")
        if type(self.training_group_sha256s) is not tuple:
            raise ValueError("Atlas task result training groups must be a tuple")
        for group in self.training_group_sha256s:
            _require_sha256(group, "Atlas task result training group")
        if type(self.selected_supply) is not tuple or not self.selected_supply:
            raise ValueError("Atlas task result requires a selected supply")
        if type(self.estimates) is not tuple or any(
            type(estimate) is not AtlasCandidateEstimate for estimate in self.estimates
        ):
            raise ValueError("Atlas task result estimates must be exact")
        if (
            type(self.forecast) is not tuple
            or not self.forecast
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in self.forecast
            )
        ):
            raise ValueError("Atlas task result forecast must be finite")
        if type(self.activated) is not bool:
            raise ValueError("Atlas task result activated must be an exact bool")
        if self.activated != (self.fallback_reason is None):
            raise ValueError(
                "Atlas task result activation and fallback reason disagree"
            )


@dataclass(frozen=True)
class AtlasOOFResult:
    release: AtlasRelease
    tasks: tuple[AtlasTaskResult, ...]
    fit_leakage_count: int

    def __post_init__(self) -> None:
        if type(self.release) is not AtlasRelease:
            raise ValueError("Atlas OOF result requires an exact release")
        if (
            type(self.tasks) is not tuple
            or not self.tasks
            or any(type(task) is not AtlasTaskResult for task in self.tasks)
        ):
            raise ValueError("Atlas OOF result requires exact task outcomes")
        expected = sum(
            task.group_sha256 in task.training_group_sha256s for task in self.tasks
        )
        if (
            type(self.fit_leakage_count) is not int
            or self.fit_leakage_count != expected
        ):
            raise ValueError("Atlas OOF leakage count is inconsistent")


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _policy_payload(policy: AtlasPolicy) -> dict[str, object]:
    return {
        "schema_version": policy.schema_version,
        "maximum_pool_size": policy.maximum_pool_size,
        "maximum_task_candidates": policy.maximum_task_candidates,
        "neighbor_grid": list(policy.neighbor_grid),
        "minimum_independent_groups": policy.minimum_independent_groups,
        "minimum_win_probability": policy.minimum_win_probability,
        "minimum_effect_margin": policy.minimum_effect_margin,
        "maximum_predicted_regret": policy.maximum_predicted_regret,
    }


def _group_map(manifest: GroupFoldManifest) -> dict[str, str]:
    return {
        task_id: group_sha
        for group_sha, task_ids, _fold in manifest.groups
        for task_id in task_ids
    }


def _task_sha256(task_id: str) -> str:
    return _sha256({"task_id": task_id})


def _manifest_held_out_hashes(
    manifest: GroupFoldManifest,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    task_folds = dict(manifest.task_fold_map)
    return tuple(
        (
            fold,
            tuple(
                sorted(
                    _task_sha256(task_id)
                    for task_id, assigned_fold in task_folds.items()
                    if assigned_fold == fold
                )
            ),
        )
        for fold in range(5)
    )


def _validated_atlas_rows(
    rows: Sequence[TaskLocalTaskRow], manifest: GroupFoldManifest
) -> tuple[tuple[TaskLocalTaskRow, ...], tuple[str, ...], dict[str, str]]:
    if type(rows) not in {tuple, list} or not rows:
        raise ValueError("Atlas fitting requires a nonempty exact row sequence")
    if type(manifest) is not GroupFoldManifest or manifest.fold_count != 5:
        raise ValueError("Atlas fitting requires an exact five-fold group manifest")
    snapshot = tuple(rows)
    if any(type(row) is not TaskLocalTaskRow for row in snapshot):
        raise ValueError("Atlas fitting requires exact task-local rows")
    keys = tuple((row.task_id, row.candidate_name) for row in snapshot)
    if len(keys) != len(set(keys)) or any(row.split != "train" for row in snapshot):
        raise ValueError("Atlas fitting rows must be unique Train rows")
    task_ids = tuple(sorted({row.task_id for row in snapshot}))
    group_ids = _group_map(manifest)
    if (
        len(task_ids) != _ATLAS_BUILD_TASK_COUNT
        or set(task_ids) != set(group_ids)
    ):
        raise ValueError(
            "Atlas fitting requires the exact registered 64-task Build universe"
        )
    by_task = _rows_by_task(snapshot, task_ids)
    expected_candidates = set(by_task[task_ids[0]])
    candidate_families: dict[str, str] = {}
    for task_id in task_ids:
        task_rows = by_task[task_id]
        if set(task_rows) != expected_candidates:
            raise ValueError("Atlas fitting requires complete per-task row identities")
        anchor = task_rows.get("toto_2_0")
        if anchor is None:
            raise ValueError("Atlas fitting requires a Toto anchor row on every task")
        for row in task_rows.values():
            if row.profile != anchor.profile:
                raise ValueError("Atlas task rows must share one exact profile")
            if row.history != anchor.history:
                raise ValueError("Atlas task rows must share one exact history")
            if row.truth != anchor.truth:
                raise ValueError("Atlas task rows must share one exact truth")
            prior_family = candidate_families.setdefault(
                row.candidate_name, row.family
            )
            if prior_family != row.family:
                raise ValueError(
                    "Atlas candidate identity must bind one reviewed family"
                )
    if any(
        "toto_2_0" not in by_task[task_id]
        or by_task[task_id]["toto_2_0"].forecast is None
        for task_id in task_ids
    ):
        raise ValueError(
            "Atlas fitting requires the successful Toto anchor on every task"
        )
    return snapshot, task_ids, group_ids


def _raw_regret(candidate: float, anchor: float) -> float:
    if not math.isfinite(candidate):
        return _NONFINITE_REGRET_SENTINEL
    if not math.isfinite(anchor):
        return 0.0
    return max(0.0, float(candidate - anchor))


def _training_record(
    candidate: TaskLocalTaskRow,
    anchor: TaskLocalTaskRow,
    group_sha256: str,
) -> AtlasTrainingRecord:
    if candidate.forecast is None or anchor.forecast is None:
        raise ValueError("Atlas record requires successful aligned forecasts")
    candidate_point = drcik_point_metrics(candidate.truth, candidate.forecast)
    anchor_point = drcik_point_metrics(anchor.truth, anchor.forecast)
    return AtlasTrainingRecord(
        candidate_name=candidate.candidate_name,
        family=candidate.family,
        group_sha256=group_sha256,
        feature=atlas_feature(candidate, anchor),
        improvement_smae=float(anchor_point["smae"] - candidate_point["smae"]),
        improvement_srmse=float(anchor_point["srmse"] - candidate_point["srmse"]),
        regret_smae_raw=_raw_regret(
            float(candidate_point["smae_raw"]),
            float(anchor_point["smae_raw"]),
        ),
        regret_srmse_raw=_raw_regret(
            float(candidate_point["srmse_raw"]),
            float(anchor_point["srmse_raw"]),
        ),
    )


def _mode(values: Sequence[str]) -> str:
    counts = {value: values.count(value) for value in set(values)}
    return min(counts, key=lambda value: (-counts[value], value))


def _aggregate_group_records(
    records: Sequence[AtlasTrainingRecord],
) -> tuple[AtlasTrainingRecord, ...]:
    grouped: dict[tuple[str, str], list[AtlasTrainingRecord]] = {}
    for record in records:
        grouped.setdefault((record.candidate_name, record.group_sha256), []).append(
            record
        )
    result: list[AtlasTrainingRecord] = []
    for key in sorted(grouped):
        members = grouped[key]
        feature = AtlasFeature(
            categorical=tuple(
                _mode([member.feature.categorical[index] for member in members])
                for index in range(9)
            ),
            numeric=tuple(
                float(
                    statistics.median(
                        member.feature.numeric[index] for member in members
                    )
                )
                for index in range(15)
            ),
        )
        result.append(
            AtlasTrainingRecord(
                candidate_name=key[0],
                family=_mode([member.family for member in members]),
                group_sha256=key[1],
                feature=feature,
                improvement_smae=float(
                    statistics.median(member.improvement_smae for member in members)
                ),
                improvement_srmse=float(
                    statistics.median(member.improvement_srmse for member in members)
                ),
                regret_smae_raw=max(member.regret_smae_raw for member in members),
                regret_srmse_raw=max(member.regret_srmse_raw for member in members),
            )
        )
    return tuple(result)


def _feature_scales(records: Sequence[AtlasTrainingRecord]) -> tuple[float, ...]:
    if not records:
        return (1.0,) * 15
    scales: list[float] = []
    for index in range(15):
        values = [record.feature.numeric[index] for record in records]
        center = float(statistics.median(values))
        deviation = float(statistics.median(abs(value - center) for value in values))
        scales.append(deviation if deviation > 1e-12 else 1.0)
    return tuple(scales)


def _distance(
    left: AtlasFeature, right: AtlasFeature, scales: tuple[float, ...]
) -> float:
    categorical = sum(
        a != b for a, b in zip(left.categorical, right.categorical, strict=True)
    )
    numeric = math.fsum(
        abs(a - b) / scale
        for a, b, scale in zip(left.numeric, right.numeric, scales, strict=True)
    )
    return float(categorical + numeric)


def _select_neighbor_count(
    records: tuple[AtlasTrainingRecord, ...],
    scales: tuple[float, ...],
    grid: tuple[int, ...],
) -> int:
    if len(records) < 2:
        return grid[0]
    scores: list[tuple[float, int]] = []
    for count in grid:
        errors: list[float] = []
        for held_out in records:
            peers = [
                record
                for record in records
                if record.candidate_name == held_out.candidate_name
                and record.group_sha256 != held_out.group_sha256
            ]
            if not peers:
                continue
            peers.sort(
                key=lambda record: (
                    _distance(held_out.feature, record.feature, scales),
                    record.group_sha256,
                )
            )
            selected = peers[:count]
            predicted = statistics.median(
                (record.improvement_smae + record.improvement_srmse) / 2.0
                for record in selected
            )
            actual = (held_out.improvement_smae + held_out.improvement_srmse) / 2.0
            errors.append(abs(float(predicted) - actual))
        scores.append((float(statistics.fmean(errors)) if errors else math.inf, count))
    return min(scores)[1]


def _fit_model(
    rows: tuple[TaskLocalTaskRow, ...],
    task_ids: tuple[str, ...],
    group_ids: Mapping[str, str],
    policy: AtlasPolicy,
) -> AtlasModel:
    pool = select_specialist_pool(
        rows,
        task_ids=task_ids,
        group_ids={task_id: group_ids[task_id] for task_id in task_ids},
        anchor_name="toto_2_0",
        policy=policy,
    )
    by_task = _rows_by_task(rows, task_ids)
    raw_records: list[AtlasTrainingRecord] = []
    for task_id in task_ids:
        anchor = by_task[task_id]["toto_2_0"]
        for candidate_name in pool[1:]:
            candidate = by_task[task_id].get(candidate_name)
            if candidate is None or candidate.forecast is None:
                continue
            raw_records.append(_training_record(candidate, anchor, group_ids[task_id]))
    records = _aggregate_group_records(raw_records)
    scales = _feature_scales(records)
    return AtlasModel(
        feature_scales=scales,
        training_task_sha256s=tuple(
            sorted(_sha256({"task_id": task_id}) for task_id in task_ids)
        ),
        training_group_sha256s=tuple(
            sorted({group_ids[task_id] for task_id in task_ids})
        ),
        selected_pool=pool,
        neighbor_count=_select_neighbor_count(records, scales, policy.neighbor_grid),
        records=records,
    )


def fit_atlas_release(
    rows: Sequence[TaskLocalTaskRow],
    fold_manifest: GroupFoldManifest,
    policy: AtlasPolicy,
) -> AtlasRelease:
    """Fit five OOF Atlas models and freeze a separate full-Build model."""
    if type(policy) is not AtlasPolicy:
        raise TypeError("Atlas release fitting requires an exact policy")
    AtlasPolicy.__post_init__(policy)
    snapshot, task_ids, group_ids = _validated_atlas_rows(rows, fold_manifest)
    task_folds = dict(fold_manifest.task_fold_map)
    full_model = _fit_model(snapshot, task_ids, group_ids, policy)
    fold_models: list[tuple[int, AtlasModel]] = []
    for fold in range(5):
        training_ids = tuple(
            task_id for task_id in task_ids if task_folds[task_id] != fold
        )
        held_out_groups = {
            group_sha
            for group_sha, _group_tasks, group_fold in fold_manifest.groups
            if group_fold == fold
        }
        training_groups = {group_ids[task_id] for task_id in training_ids}
        if not training_ids or training_groups & held_out_groups:
            raise ValueError("Atlas fold fitting overlaps held-out groups")
        fold_models.append(
            (fold, _fit_model(snapshot, training_ids, group_ids, policy))
        )
    return AtlasRelease(
        schema_version=1,
        policy=policy,
        full_build_model=full_model,
        build_fold_models=tuple(fold_models),
        build_fold_held_out_task_sha256s=_manifest_held_out_hashes(fold_manifest),
        source_sha256=_sha256(
            [
                {
                    "group_sha256": group_ids[row.task_id],
                    "row": asdict(row),
                }
                for row in sorted(
                    snapshot, key=lambda item: (item.task_id, item.candidate_name)
                )
            ]
        ),
        policy_sha256=_sha256(_policy_payload(policy)),
        fold_manifest_sha256=_sha256(fold_manifest.to_payload()),
    )


def _estimate(
    candidate: AtlasMaterializedCandidate,
    model: AtlasModel,
) -> AtlasCandidateEstimate | None:
    if candidate.feature is None:
        return None
    records = [
        record
        for record in model.records
        if record.candidate_name == candidate.candidate_name
    ]
    if not records:
        return None
    records.sort(
        key=lambda record: (
            _distance(candidate.feature, record.feature, model.feature_scales),
            record.group_sha256,
        )
    )
    nearest = records[: model.neighbor_count]
    wins = sum(
        record.improvement_smae > 0.0 and record.improvement_srmse > 0.0
        for record in nearest
    )
    probability = float((wins + 1) / (len(nearest) + 2))
    effect_smae = float(
        statistics.median(record.improvement_smae for record in nearest)
    )
    effect_srmse = float(
        statistics.median(record.improvement_srmse for record in nearest)
    )
    joint = [
        (record.improvement_smae + record.improvement_srmse) / 2.0 for record in nearest
    ]
    center = float(statistics.median(joint))
    dispersion = float(statistics.median(abs(value - center) for value in joint))
    predicted_regret = float(
        max(
            statistics.median(record.regret_smae_raw for record in nearest),
            statistics.median(record.regret_srmse_raw for record in nearest),
        )
    )
    return AtlasCandidateEstimate(
        candidate_name=candidate.candidate_name,
        independent_groups=len({record.group_sha256 for record in nearest}),
        neighbor_count=len(nearest),
        win_probability=probability,
        effect_smae=effect_smae,
        effect_srmse=effect_srmse,
        lower_joint_effect=center - dispersion,
        predicted_regret=predicted_regret,
    )


def _locally_consistent(feature: AtlasFeature, margin: float) -> bool:
    smae = feature.numeric[7:10]
    srmse = feature.numeric[10:13]
    return (
        sum(
            left >= margin and right >= margin
            for left, right in zip(smae, srmse, strict=True)
        )
        >= 3
    )


def route_atlas_task(
    task_case: AtlasTaskCase,
    atlas_release: AtlasRelease,
    *,
    fold: int | None,
) -> AtlasTaskResult:
    """Route one label-free task through an OOF or frozen full-Build Atlas model."""
    if type(task_case) is not AtlasTaskCase:
        raise TypeError("Atlas routing requires an exact task case")
    AtlasTaskCase.__post_init__(task_case)
    if type(atlas_release) is not AtlasRelease:
        raise TypeError("Atlas routing requires an exact release")
    AtlasRelease.__post_init__(atlas_release)
    if fold is None:
        model = atlas_release.full_build_model
    else:
        if type(fold) is not int or not 0 <= fold <= 4:
            raise ValueError("Atlas routing fold must be zero through four or None")
        model = dict(atlas_release.build_fold_models)[fold]
        if task_case.group_sha256 in model.training_group_sha256s:
            raise ValueError("Atlas OOF routing model contains the held-out group")

    offered = {
        candidate.candidate_name: candidate
        for candidate in task_case.candidates
        if candidate.candidate_name
        in model.selected_pool[1 : atlas_release.policy.maximum_task_candidates]
    }
    estimates = tuple(
        estimate
        for name in model.selected_pool[1:]
        if name in offered
        if (estimate := _estimate(offered[name], model)) is not None
    )
    qualified = [
        estimate
        for estimate in estimates
        if estimate.independent_groups
        >= atlas_release.policy.minimum_independent_groups
        and estimate.win_probability >= atlas_release.policy.minimum_win_probability
        and estimate.effect_smae >= atlas_release.policy.minimum_effect_margin
        and estimate.effect_srmse >= atlas_release.policy.minimum_effect_margin
        and estimate.lower_joint_effect > atlas_release.policy.minimum_effect_margin
        and estimate.predicted_regret <= atlas_release.policy.maximum_predicted_regret
        and _locally_consistent(
            offered[estimate.candidate_name].feature,  # type: ignore[arg-type]
            atlas_release.policy.minimum_effect_margin,
        )
    ]
    if not qualified:
        regret_blocked = any(
            estimate.predicted_regret > atlas_release.policy.maximum_predicted_regret
            and estimate.independent_groups
            >= atlas_release.policy.minimum_independent_groups
            and estimate.win_probability >= atlas_release.policy.minimum_win_probability
            and estimate.effect_smae >= atlas_release.policy.minimum_effect_margin
            and estimate.effect_srmse >= atlas_release.policy.minimum_effect_margin
            and estimate.lower_joint_effect > atlas_release.policy.minimum_effect_margin
            for estimate in estimates
        )
        return AtlasTaskResult(
            group_sha256=task_case.group_sha256,
            training_group_sha256s=model.training_group_sha256s,
            selected_supply=(task_case.anchor.candidate_name,),
            estimates=estimates,
            forecast=task_case.anchor.forecast,
            activated=False,
            fallback_reason=(
                "predicted_regret_exceeds_limit"
                if regret_blocked
                else "insufficient_atlas_evidence"
            ),
        )
    best = min(
        qualified,
        key=lambda estimate: (
            -estimate.lower_joint_effect,
            -estimate.win_probability,
            estimate.predicted_regret,
            _canonical_name(estimate.candidate_name),
        ),
    )
    specialist = offered[best.candidate_name]
    forecast = tuple(
        0.7 * anchor + 0.3 * candidate
        for anchor, candidate in zip(
            task_case.anchor.forecast, specialist.forecast, strict=True
        )
    )
    if any(not math.isfinite(value) for value in forecast):
        return AtlasTaskResult(
            group_sha256=task_case.group_sha256,
            training_group_sha256s=model.training_group_sha256s,
            selected_supply=(task_case.anchor.candidate_name,),
            estimates=estimates,
            forecast=task_case.anchor.forecast,
            activated=False,
            fallback_reason="nonfinite_atlas_forecast",
        )
    return AtlasTaskResult(
        group_sha256=task_case.group_sha256,
        training_group_sha256s=model.training_group_sha256s,
        selected_supply=(task_case.anchor.candidate_name, specialist.candidate_name),
        estimates=estimates,
        forecast=forecast,
        activated=True,
        fallback_reason=None,
    )


def _task_case(
    task_id: str,
    rows: Mapping[str, TaskLocalTaskRow],
    model: AtlasModel,
    group_sha256: str,
) -> AtlasTaskCase:
    anchor = rows["toto_2_0"]
    if anchor.forecast is None:  # pragma: no cover - fitting validates this
        raise AssertionError("validated Atlas anchor became unavailable")
    candidates: list[AtlasMaterializedCandidate] = []
    for name in model.selected_pool[1:]:
        row = rows.get(name)
        if row is None or row.forecast is None:
            continue
        try:
            feature = atlas_feature(row, anchor)
        except ValueError:
            continue
        candidates.append(
            AtlasMaterializedCandidate(name, row.family, feature, row.forecast)
        )
    del task_id
    return AtlasTaskCase(
        group_sha256=group_sha256,
        anchor=AtlasMaterializedCandidate(
            anchor.candidate_name,
            anchor.family,
            None,
            anchor.forecast,
        ),
        candidates=tuple(candidates),
    )


def fit_atlas_oof(
    rows: Sequence[TaskLocalTaskRow],
    fold_manifest: GroupFoldManifest,
    policy: AtlasPolicy,
) -> AtlasOOFResult:
    """Fit every Atlas partition and route each Build task out of fold."""
    snapshot, task_ids, group_ids = _validated_atlas_rows(rows, fold_manifest)
    release = fit_atlas_release(snapshot, fold_manifest, policy)
    by_task = _rows_by_task(snapshot, task_ids)
    fold_models = dict(release.build_fold_models)
    tasks = tuple(
        route_atlas_task(
            _task_case(
                task_id,
                by_task[task_id],
                fold_models[fold_manifest.task_fold_map[task_id]],
                group_ids[task_id],
            ),
            release,
            fold=fold_manifest.task_fold_map[task_id],
        )
        for task_id in task_ids
    )
    return AtlasOOFResult(
        release=release,
        tasks=tasks,
        fit_leakage_count=sum(
            task.group_sha256 in task.training_group_sha256s for task in tasks
        ),
    )


def _parse_feature(payload: object) -> AtlasFeature:
    if type(payload) is not dict or set(payload) != {"categorical", "numeric"}:
        raise ValueError("Atlas feature payload is malformed")
    if type(payload["categorical"]) is not list or type(payload["numeric"]) is not list:
        raise ValueError("Atlas feature arrays are malformed")
    return AtlasFeature(tuple(payload["categorical"]), tuple(payload["numeric"]))


def _parse_model(payload: object) -> AtlasModel:
    fields = {
        "feature_scales",
        "training_task_sha256s",
        "training_group_sha256s",
        "selected_pool",
        "neighbor_count",
        "records",
    }
    if type(payload) is not dict or set(payload) != fields:
        raise ValueError("Atlas model payload is malformed")
    if any(type(payload[name]) is not list for name in fields - {"neighbor_count"}):
        raise ValueError("Atlas model arrays are malformed")
    records: list[AtlasTrainingRecord] = []
    record_fields = {
        "candidate_name",
        "family",
        "group_sha256",
        "feature",
        "improvement_smae",
        "improvement_srmse",
        "regret_smae_raw",
        "regret_srmse_raw",
    }
    for record in payload["records"]:
        if type(record) is not dict or set(record) != record_fields:
            raise ValueError("Atlas training record payload is malformed")
        records.append(
            AtlasTrainingRecord(
                candidate_name=record["candidate_name"],
                family=record["family"],
                group_sha256=record["group_sha256"],
                feature=_parse_feature(record["feature"]),
                improvement_smae=record["improvement_smae"],
                improvement_srmse=record["improvement_srmse"],
                regret_smae_raw=record["regret_smae_raw"],
                regret_srmse_raw=record["regret_srmse_raw"],
            )
        )
    return AtlasModel(
        feature_scales=tuple(payload["feature_scales"]),
        training_task_sha256s=tuple(payload["training_task_sha256s"]),
        training_group_sha256s=tuple(payload["training_group_sha256s"]),
        selected_pool=tuple(payload["selected_pool"]),
        neighbor_count=payload["neighbor_count"],
        records=tuple(records),
    )


def parse_atlas_release(payload: object) -> AtlasRelease:
    """Parse an exact frozen Atlas release payload."""
    fields = {
        "schema_version",
        "policy",
        "full_build_model",
        "build_fold_models",
        "build_fold_held_out_task_sha256s",
        "source_sha256",
        "policy_sha256",
        "fold_manifest_sha256",
    }
    if type(payload) is not dict or set(payload) != fields:
        raise ValueError("Atlas release payload is malformed")
    policy_fields = set(_policy_payload(AtlasPolicy()))
    raw_policy = payload["policy"]
    if type(raw_policy) is not dict or set(raw_policy) != policy_fields:
        raise ValueError("Atlas policy payload is malformed")
    policy = AtlasPolicy(
        schema_version=raw_policy["schema_version"],
        maximum_pool_size=raw_policy["maximum_pool_size"],
        maximum_task_candidates=raw_policy["maximum_task_candidates"],
        neighbor_grid=tuple(raw_policy["neighbor_grid"]),
        minimum_independent_groups=raw_policy["minimum_independent_groups"],
        minimum_win_probability=raw_policy["minimum_win_probability"],
        minimum_effect_margin=raw_policy["minimum_effect_margin"],
        maximum_predicted_regret=raw_policy["maximum_predicted_regret"],
    )
    raw_folds = payload["build_fold_models"]
    if type(raw_folds) is not list:
        raise ValueError("Atlas fold model payload is malformed")
    fold_models = tuple(
        (pair[0], _parse_model(pair[1]))
        for pair in raw_folds
        if type(pair) is list and len(pair) == 2
    )
    if len(fold_models) != len(raw_folds):
        raise ValueError("Atlas fold model payload is malformed")
    raw_omissions = payload["build_fold_held_out_task_sha256s"]
    if type(raw_omissions) is not list:
        raise ValueError("Atlas held-out fold payload is malformed")
    held_out_task_sha256s = tuple(
        (pair[0], tuple(pair[1]))
        for pair in raw_omissions
        if type(pair) is list
        and len(pair) == 2
        and type(pair[1]) is list
    )
    if len(held_out_task_sha256s) != len(raw_omissions):
        raise ValueError("Atlas held-out fold payload is malformed")
    return AtlasRelease(
        schema_version=payload["schema_version"],
        policy=policy,
        full_build_model=_parse_model(payload["full_build_model"]),
        build_fold_models=fold_models,
        build_fold_held_out_task_sha256s=held_out_task_sha256s,
        source_sha256=payload["source_sha256"],
        policy_sha256=payload["policy_sha256"],
        fold_manifest_sha256=payload["fold_manifest_sha256"],
    )


__all__ = [
    "AtlasCandidateEstimate",
    "AtlasFeature",
    "AtlasMaterializedCandidate",
    "AtlasModel",
    "AtlasOOFResult",
    "AtlasPolicy",
    "AtlasRelease",
    "AtlasTaskCase",
    "AtlasTaskResult",
    "AtlasTrainingRecord",
    "atlas_feature",
    "fit_atlas_oof",
    "fit_atlas_release",
    "parse_atlas_release",
    "route_atlas_task",
    "select_specialist_pool",
]
