"""Deterministic history-only tournament around a frozen Numerical anchor."""

from __future__ import annotations

import hashlib
import itertools
import math
import statistics
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from common.metrics import drcik_point_metrics, joint_scaled_error, pareto_scaled_improvement
from common.payload import canonical_json_bytes

from .numerical_selector import CandidateDiagnostics
from .screening import TaskProfile


def _canonical_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def task_local_fingerprint(value: object) -> str:
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 value")
    return value


@dataclass(frozen=True)
class TaskLocalTournamentPolicy:
    """Frozen host-owned search and safety constants for one local tournament."""

    schema_version: int = 1
    anchor_name: str = "toto_2_0"
    maximum_candidates: int = 8
    maximum_specialists: int = 2
    minimum_successful_folds: int = 3
    minimum_anchor_weight: float = 0.5
    weight_step: float = 0.1
    minimum_joint_improvement: float = 0.02
    maximum_worst_joint_regret: float = 0.25
    maximum_raw_smae: float = 10.0
    maximum_raw_srmse: float = 10.0

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("task-local policy schema must be exactly one")
        if type(self.anchor_name) is not str or not self.anchor_name.isidentifier():
            raise ValueError("task-local anchor must be a Python identifier")
        for name, lower, upper in (
            ("maximum_candidates", 2, 8),
            ("maximum_specialists", 1, 2),
            ("minimum_successful_folds", 1, 100),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} is outside its closed range")
        if self.maximum_specialists >= self.maximum_candidates:
            raise ValueError("specialist bound must be smaller than candidate bound")
        for name in (
            "minimum_anchor_weight",
            "weight_step",
            "minimum_joint_improvement",
            "maximum_worst_joint_regret",
            "maximum_raw_smae",
            "maximum_raw_srmse",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if not 0.5 <= self.minimum_anchor_weight <= 1.0:
            raise ValueError("minimum anchor weight must be in [0.5, 1.0]")
        if not math.isclose(self.weight_step, 0.1, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("v1 uses the exact one-tenth weight grid")
        if self.maximum_raw_smae <= 0.0 or self.maximum_raw_srmse <= 0.0:
            raise ValueError("raw tail limits must be positive")


@dataclass(frozen=True)
class TaskLocalEnsembleResult:
    """One replayable local selection over already materialized forecasts."""

    forecast: tuple[float, ...]
    selected_names: tuple[str, ...]
    weights: tuple[float, ...]
    activated: bool
    fold_support: int
    maximum_fold_regret: float
    fallback_reason: str | None
    policy_fingerprint: str

    def __post_init__(self) -> None:
        if (
            type(self.forecast) is not tuple
            or not self.forecast
            or any(type(value) is not float or not math.isfinite(value) for value in self.forecast)
        ):
            raise ValueError("task-local result requires a finite float forecast")
        if (
            type(self.selected_names) is not tuple
            or not self.selected_names
            or len(self.selected_names) != len(set(self.selected_names))
            or any(type(name) is not str or not name for name in self.selected_names)
        ):
            raise ValueError("task-local result requires unique selected names")
        if (
            type(self.weights) is not tuple
            or len(self.weights) != len(self.selected_names)
            or any(type(weight) is not float or weight <= 0.0 for weight in self.weights)
            or not math.isclose(math.fsum(self.weights), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("task-local result weights must be positive and normalized")
        if type(self.activated) is not bool:
            raise ValueError("task-local activation must be an exact bool")
        if type(self.fold_support) is not int or self.fold_support < 0:
            raise ValueError("task-local fold support must be nonnegative")
        if (
            type(self.maximum_fold_regret) is not float
            or not math.isfinite(self.maximum_fold_regret)
            or self.maximum_fold_regret < 0.0
        ):
            raise ValueError("task-local maximum fold regret must be finite and nonnegative")
        if self.activated != (len(self.selected_names) > 1):
            raise ValueError("task-local activation must match selected specialists")
        if self.fallback_reason is not None and (
            type(self.fallback_reason) is not str or not self.fallback_reason
        ):
            raise ValueError("task-local fallback reason must be a nonempty string")
        if (self.activated and self.fallback_reason is not None) or (
            not self.activated and self.fallback_reason is None
        ):
            raise ValueError("task-local fallback reason must match activation")
        if (
            type(self.policy_fingerprint) is not str
            or len(self.policy_fingerprint) != 64
        ):
            raise ValueError("task-local policy fingerprint must be SHA-256")


@dataclass(frozen=True)
class GroupCandidateSupply:
    """Reviewed bounded candidate supply for one morphology bucket."""

    group_key: str
    candidate_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.group_key, "group candidate supply key")
        if (
            type(self.candidate_names) is not tuple
            or not self.candidate_names
            or len(self.candidate_names) > 8
            or any(type(name) is not str or not name.isidentifier() for name in self.candidate_names)
            or len({_canonical_name(name) for name in self.candidate_names})
            != len(self.candidate_names)
        ):
            raise ValueError("group candidate supply requires at most eight unique identifiers")

    def to_payload(self) -> dict[str, object]:
        return {"group_key": self.group_key, "candidate_names": list(self.candidate_names)}


@dataclass(frozen=True)
class TaskLocalEnsembleRelease:
    """Frozen candidate supplies plus the host-owned local tournament policy."""

    schema_version: int
    anchor_release_sha256: str
    anchor_name: str
    policy: TaskLocalTournamentPolicy
    default_candidate_names: tuple[str, ...]
    group_supplies: tuple[GroupCandidateSupply, ...]
    grouping_fingerprint: str
    oof_report_sha256: str
    source_hashes: tuple[tuple[str, str], ...]
    metric_policy_fingerprint: str
    lineage: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("task-local release schema must be exactly one")
        if type(self.policy) is not TaskLocalTournamentPolicy:
            raise ValueError("task-local release requires an exact tournament policy")
        TaskLocalTournamentPolicy.__post_init__(self.policy)
        if self.anchor_name != self.policy.anchor_name:
            raise ValueError("task-local release anchor must match its policy")
        for label, value in (
            ("anchor release", self.anchor_release_sha256),
            ("grouping", self.grouping_fingerprint),
            ("OOF report", self.oof_report_sha256),
            ("metric policy", self.metric_policy_fingerprint),
        ):
            _require_sha256(value, f"task-local {label} fingerprint")
        GroupCandidateSupply("0" * 64, self.default_candidate_names)
        if self.default_candidate_names[0] != self.anchor_name:
            raise ValueError("task-local default supply must begin with the anchor")
        if (
            type(self.group_supplies) is not tuple
            or any(type(item) is not GroupCandidateSupply for item in self.group_supplies)
            or tuple(item.group_key for item in self.group_supplies)
            != tuple(sorted(item.group_key for item in self.group_supplies))
            or len({item.group_key for item in self.group_supplies}) != len(self.group_supplies)
            or any(item.candidate_names[0] != self.anchor_name for item in self.group_supplies)
        ):
            raise ValueError("task-local group supplies must be unique sorted and anchored")
        if (
            type(self.source_hashes) is not tuple
            or not self.source_hashes
            or tuple(name for name, _digest in self.source_hashes)
            != tuple(sorted(name for name, _digest in self.source_hashes))
            or len({name for name, _digest in self.source_hashes}) != len(self.source_hashes)
            or any(
                type(name) is not str
                or not name.isidentifier()
                or _invalid_sha256(digest)
                for name, digest in self.source_hashes
            )
        ):
            raise ValueError("task-local source hashes must be unique sorted identifiers")
        if (
            type(self.lineage) is not tuple
            or not self.lineage
            or any(type(item) is not str or not item.isidentifier() for item in self.lineage)
            or len(self.lineage) != len(set(self.lineage))
        ):
            raise ValueError("task-local lineage must contain unique identifiers")

    def candidate_names(self, group_key: str) -> tuple[str, ...]:
        for item in self.group_supplies:
            if item.group_key == group_key:
                return item.candidate_names
        return self.default_candidate_names

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "anchor_release_sha256": self.anchor_release_sha256,
            "anchor_name": self.anchor_name,
            "policy": asdict(self.policy),
            "default_candidate_names": list(self.default_candidate_names),
            "group_supplies": [item.to_payload() for item in self.group_supplies],
            "grouping_fingerprint": self.grouping_fingerprint,
            "oof_report_sha256": self.oof_report_sha256,
            "source_hashes": dict(self.source_hashes),
            "metric_policy_fingerprint": self.metric_policy_fingerprint,
            "lineage": list(self.lineage),
        }


def _invalid_sha256(value: object) -> bool:
    try:
        _require_sha256(value, "value")
    except ValueError:
        return True
    return False


def parse_task_local_release(payload: object) -> TaskLocalEnsembleRelease:
    """Parse only the exact canonical v1 task-local release schema."""
    expected = {
        "schema_version",
        "anchor_release_sha256",
        "anchor_name",
        "policy",
        "default_candidate_names",
        "group_supplies",
        "grouping_fingerprint",
        "oof_report_sha256",
        "source_hashes",
        "metric_policy_fingerprint",
        "lineage",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("task-local release schema is malformed")
    policy_payload = payload["policy"]
    policy_fields = {
        "schema_version",
        "anchor_name",
        "maximum_candidates",
        "maximum_specialists",
        "minimum_successful_folds",
        "minimum_anchor_weight",
        "weight_step",
        "minimum_joint_improvement",
        "maximum_worst_joint_regret",
        "maximum_raw_smae",
        "maximum_raw_srmse",
    }
    if type(policy_payload) is not dict or set(policy_payload) != policy_fields:
        raise ValueError("task-local policy schema is malformed")
    raw_groups = payload["group_supplies"]
    if type(raw_groups) is not list:
        raise ValueError("task-local group supply schema is malformed")
    groups: list[GroupCandidateSupply] = []
    for raw_group in raw_groups:
        if type(raw_group) is not dict or set(raw_group) != {"group_key", "candidate_names"}:
            raise ValueError("task-local group supply schema is malformed")
        raw_names = raw_group["candidate_names"]
        if type(raw_names) is not list:
            raise ValueError("task-local group candidate names are malformed")
        groups.append(GroupCandidateSupply(raw_group["group_key"], tuple(raw_names)))
    source_payload = payload["source_hashes"]
    if type(source_payload) is not dict:
        raise ValueError("task-local source hash schema is malformed")
    raw_default = payload["default_candidate_names"]
    raw_lineage = payload["lineage"]
    if type(raw_default) is not list or type(raw_lineage) is not list:
        raise ValueError("task-local release arrays are malformed")
    try:
        return TaskLocalEnsembleRelease(
            schema_version=payload["schema_version"],
            anchor_release_sha256=payload["anchor_release_sha256"],
            anchor_name=payload["anchor_name"],
            policy=TaskLocalTournamentPolicy(**policy_payload),
            default_candidate_names=tuple(raw_default),
            group_supplies=tuple(groups),
            grouping_fingerprint=payload["grouping_fingerprint"],
            oof_report_sha256=payload["oof_report_sha256"],
            source_hashes=tuple(sorted(source_payload.items())),
            metric_policy_fingerprint=payload["metric_policy_fingerprint"],
            lineage=tuple(raw_lineage),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("task-local release schema is malformed") from error


def canonical_task_local_release_bytes(release: TaskLocalEnsembleRelease) -> bytes:
    if type(release) is not TaskLocalEnsembleRelease:
        raise TypeError("canonical release encoding requires an exact task-local release")
    return canonical_json_bytes(release.to_payload())


@dataclass(frozen=True)
class _FoldSummary:
    median_joint: float
    worst_joint: float
    median_smae: float
    median_srmse: float
    worst_smae_raw: float
    worst_srmse_raw: float
    joint_by_fold: tuple[float, ...]


def _finite_forecast(value: object, horizon: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != horizon:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _fold_data(
    diagnostic: CandidateDiagnostics,
    *,
    minimum_folds: int,
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...] | None:
    if (
        type(diagnostic) is not CandidateDiagnostics
        or not diagnostic.eligible
        or diagnostic.successful_folds < minimum_folds
        or len(diagnostic.fold_forecasts) != len(diagnostic.fold_truths)
        or len(diagnostic.fold_truths) < minimum_folds
    ):
        return None
    result: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for raw_forecast, raw_truth in zip(
        diagnostic.fold_forecasts, diagnostic.fold_truths, strict=True
    ):
        truth = _finite_forecast(raw_truth, len(raw_truth))
        forecast = _finite_forecast(raw_forecast, len(raw_truth))
        if truth is None or forecast is None or not truth:
            return None
        result.append((forecast, truth))
    return tuple(result)


def _summarize(
    folds: Sequence[tuple[tuple[float, ...], tuple[float, ...]]]
) -> _FoldSummary:
    smae: list[float] = []
    srmse: list[float] = []
    smae_raw: list[float] = []
    srmse_raw: list[float] = []
    joint: list[float] = []
    for forecast, truth in folds:
        point = drcik_point_metrics(truth, forecast)
        fold_smae = float(point["smae"])
        fold_srmse = float(point["srmse"])
        smae.append(fold_smae)
        srmse.append(fold_srmse)
        smae_raw.append(float(point["smae_raw"]))
        srmse_raw.append(float(point["srmse_raw"]))
        joint.append(joint_scaled_error(fold_smae, fold_srmse))
    return _FoldSummary(
        median_joint=float(statistics.median(joint)),
        worst_joint=max(joint),
        median_smae=float(statistics.median(smae)),
        median_srmse=float(statistics.median(srmse)),
        worst_smae_raw=max(smae_raw),
        worst_srmse_raw=max(srmse_raw),
        joint_by_fold=tuple(joint),
    )


def _blend(
    names: tuple[str, ...],
    weights: tuple[float, ...],
    forecasts: Mapping[str, tuple[float, ...]],
) -> tuple[float, ...] | None:
    horizon = len(forecasts[names[0]])
    equal = len(set(weights)) == 1
    values = tuple(
        statistics.fmean(forecasts[name][step] for name in names)
        if equal
        else sum(
            weight * forecasts[name][step]
            for name, weight in zip(names, weights, strict=True)
        )
        for step in range(horizon)
    )
    return values if all(math.isfinite(value) for value in values) else None


def _fallback(
    policy: TaskLocalTournamentPolicy,
    anchor: tuple[float, ...],
    reason: str,
    fold_support: int,
) -> TaskLocalEnsembleResult:
    return TaskLocalEnsembleResult(
        forecast=anchor,
        selected_names=(policy.anchor_name,),
        weights=(1.0,),
        activated=False,
        fold_support=fold_support,
        maximum_fold_regret=0.0,
        fallback_reason=reason,
        policy_fingerprint=task_local_fingerprint(policy),
    )


def _weight_candidates(
    anchor_name: str,
    specialists: tuple[str, ...],
) -> tuple[tuple[tuple[str, ...], tuple[float, ...]], ...]:
    result: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    for count in range(1, len(specialists) + 1):
        for chosen in itertools.combinations(specialists, count):
            for anchor_units in range(5, 10):
                remainder = 10 - anchor_units
                if count == 1:
                    allocations = ((remainder,),)
                else:
                    allocations = tuple(
                        (left, remainder - left) for left in range(1, remainder)
                    )
                for specialist_units in allocations:
                    if any(value <= 0 for value in specialist_units):
                        continue
                    names = (anchor_name, *chosen)
                    weights = tuple(float(value / 10.0) for value in (anchor_units, *specialist_units))
                    result.append((names, weights))
    return tuple(result)


def execute_task_local_ensemble(
    policy: TaskLocalTournamentPolicy,
    *,
    candidate_names: Sequence[str],
    forecasts: Mapping[str, Sequence[float]],
    diagnostics: Mapping[str, CandidateDiagnostics],
    horizon: int,
) -> TaskLocalEnsembleResult:
    """Choose one anchor-heavy full-horizon blend using paired history folds only."""
    if type(policy) is not TaskLocalTournamentPolicy:
        raise TypeError("task-local execution requires an exact policy")
    TaskLocalTournamentPolicy.__post_init__(policy)
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("task-local horizon must be a positive exact integer")
    supplied_names = tuple(candidate_names)
    if (
        not supplied_names
        or len(supplied_names) > policy.maximum_candidates
        or any(type(name) is not str or not name.isidentifier() for name in supplied_names)
    ):
        raise ValueError("task-local execution accepts at most eight candidate identifiers")
    if len({_canonical_name(name) for name in supplied_names}) != len(supplied_names):
        raise ValueError("task-local candidate namespace is not unique")
    if policy.anchor_name not in supplied_names:
        raise ValueError("task-local candidate supply must contain its anchor")
    anchor = _finite_forecast(forecasts.get(policy.anchor_name), horizon)
    if anchor is None:
        raise ValueError("task-local anchor forecast is missing or invalid")
    anchor_diagnostic = diagnostics.get(policy.anchor_name)
    if type(anchor_diagnostic) is not CandidateDiagnostics or anchor_diagnostic.name != policy.anchor_name:
        return _fallback(policy, anchor, "anchor_diagnostics_unavailable", 0)
    anchor_folds = _fold_data(
        anchor_diagnostic, minimum_folds=policy.minimum_successful_folds
    )
    if anchor_folds is None:
        return _fallback(policy, anchor, "anchor_diagnostics_unavailable", 0)
    anchor_summary = _summarize(anchor_folds)

    eligible: list[tuple[str, str, _FoldSummary, tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]]] = []
    for name in sorted(
        (item for item in supplied_names if item != policy.anchor_name),
        key=_canonical_name,
    ):
        full = _finite_forecast(forecasts.get(name), horizon)
        diagnostic = diagnostics.get(name)
        if full is None or type(diagnostic) is not CandidateDiagnostics or diagnostic.name != name:
            continue
        folds = _fold_data(diagnostic, minimum_folds=policy.minimum_successful_folds)
        if folds is None or len(folds) != len(anchor_folds):
            continue
        if any(
            truth != anchor_truth
            for (_forecast, truth), (_anchor_forecast, anchor_truth) in zip(
                folds, anchor_folds, strict=True
            )
        ):
            continue
        summary = _summarize(folds)
        dominated = (
            anchor_summary.median_smae <= summary.median_smae
            and anchor_summary.median_srmse <= summary.median_srmse
            and (
                anchor_summary.median_smae < summary.median_smae
                or anchor_summary.median_srmse < summary.median_srmse
            )
        )
        if (
            dominated
            or summary.worst_smae_raw > policy.maximum_raw_smae
            or summary.worst_srmse_raw > policy.maximum_raw_srmse
            or max(
                child - parent
                for child, parent in zip(
                    summary.joint_by_fold, anchor_summary.joint_by_fold, strict=True
                )
            ) > policy.maximum_worst_joint_regret
        ):
            continue
        eligible.append((name, diagnostic.family, summary, folds))
    if not eligible:
        return _fallback(
            policy, anchor, "no_eligible_specialist", len(anchor_folds)
        )

    eligible.sort(
        key=lambda item: (
            item[2].median_joint,
            item[2].worst_joint,
            item[2].median_smae,
            item[2].median_srmse,
            _canonical_name(item[0]),
        )
    )
    chosen = [eligible[0]]
    for item in eligible[1:]:
        if len(chosen) >= policy.maximum_specialists:
            break
        if item[1] != chosen[0][1]:
            chosen.append(item)
    for item in eligible[1:]:
        if len(chosen) >= policy.maximum_specialists:
            break
        if item not in chosen:
            chosen.append(item)
    specialist_names = tuple(item[0] for item in chosen)
    fold_maps = {policy.anchor_name: anchor_folds, **{item[0]: item[3] for item in chosen}}
    full_forecasts = {
        policy.anchor_name: anchor,
        **{name: _finite_forecast(forecasts[name], horizon) for name in specialist_names},
    }

    ranked: list[tuple[tuple[object, ...], tuple[str, ...], tuple[float, ...], _FoldSummary]] = []
    for names, weights in _weight_candidates(policy.anchor_name, specialist_names):
        blended_folds: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
        for fold_index in range(len(anchor_folds)):
            fold_forecasts = {
                name: fold_maps[name][fold_index][0] for name in names
            }
            blended = _blend(names, weights, fold_forecasts)
            if blended is None:
                break
            blended_folds.append((blended, anchor_folds[fold_index][1]))
        if len(blended_folds) != len(anchor_folds):
            continue
        summary = _summarize(blended_folds)
        if (
            not pareto_scaled_improvement(
                anchor_summary.median_smae,
                anchor_summary.median_srmse,
                summary.median_smae,
                summary.median_srmse,
            )
            or anchor_summary.median_joint - summary.median_joint
            < policy.minimum_joint_improvement
            or summary.worst_smae_raw > policy.maximum_raw_smae
            or summary.worst_srmse_raw > policy.maximum_raw_srmse
            or max(
                child - parent
                for child, parent in zip(
                    summary.joint_by_fold, anchor_summary.joint_by_fold, strict=True
                )
            ) > policy.maximum_worst_joint_regret
        ):
            continue
        key: tuple[object, ...] = (
            summary.median_joint,
            summary.worst_joint,
            summary.median_smae,
            summary.median_srmse,
            -weights[0],
            tuple(_canonical_name(name) for name in names),
        )
        ranked.append((key, names, weights, summary))
    if not ranked:
        return _fallback(
            policy, anchor, "no_pareto_safe_weight", len(anchor_folds)
        )
    _key, names, weights, _summary = min(ranked, key=lambda item: item[0])
    safe_full_forecasts = {
        name: value
        for name, value in full_forecasts.items()
        if value is not None
    }
    final = _blend(names, weights, safe_full_forecasts)
    if final is None:
        return _fallback(policy, anchor, "invalid_blend", len(anchor_folds))
    return TaskLocalEnsembleResult(
        forecast=final,
        selected_names=names,
        weights=weights,
        activated=True,
        fold_support=len(anchor_folds),
        maximum_fold_regret=max(
            0.0,
            max(
                child - parent
                for child, parent in zip(
                    _summary.joint_by_fold,
                    anchor_summary.joint_by_fold,
                    strict=True,
                )
            ),
        ),
        fallback_reason=None,
        policy_fingerprint=task_local_fingerprint(policy),
    )
