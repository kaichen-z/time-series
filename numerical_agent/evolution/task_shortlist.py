"""Canonical, history-only per-task candidate shortlists."""
from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass
from typing import Mapping, Sequence

from common.payload import canonical_json_bytes
from common.metrics import drcik_point_metrics, joint_scaled_error, linear_quantile

from .filtering import FAMILIES, FilterDictionary
from .screening import ScreeningPolicy, TaskProfile, materialize_active_dictionary


_SHORTLIST_MINIMUM_CANDIDATES = 6
_SHORTLIST_MAXIMUM_CANDIDATES = 10


def _name(value: object, field: str = "name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return unicodedata.normalize("NFKC", value).casefold()


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex string")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result

def _fields(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} fields mismatch")

def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value

def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return _finite(value, field)


def task_morphology_key(profile: TaskProfile) -> str:
    """Return the sole stable, history-only bucket used for priors and ranking."""
    if type(profile) is not TaskProfile:
        raise TypeError("morphology grouping requires an exact TaskProfile")
    history_bucket = (
        "short" if profile.history_length < 64 else
        "medium" if profile.history_length < 256 else
        "long"
    )
    ratio = profile.horizon / profile.history_length
    horizon_bucket = "short" if ratio <= 0.1 else "medium" if ratio <= 0.3 else "long"
    trend = (
        profile.trend_direction
        if profile.trend_strength >= 0.35 and profile.trend_direction != "flat"
        else "flat"
    )
    payload = {
        "frequency": unicodedata.normalize("NFKC", profile.frequency).casefold().strip(),
        "history": history_bucket,
        "horizon": horizon_bucket,
        "trend": trend,
        "periodic": bool(
            profile.periodicity_periods and profile.periodicity_confidence >= 0.5
        ),
        "intermittent": bool(
            profile.zero_fraction >= 0.5 or profile.intermittency_adi >= 1.32
        ),
        "recent_regime": bool(
            profile.recent_regime_start is not None
            and profile.recent_regime_confidence >= 0.5
        ),
        "signed": profile.signed,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_task_candidate_shortlist(
    candidate_names: Sequence[str],
    exclusion_reasons: Sequence[tuple[str, str]],
    shortlist_underfilled: bool,
) -> None:
    """Validate cardinality and membership invariants reusable by artifact readers."""
    if not candidate_names:
        raise ValueError("shortlist requires at least one selected candidate")
    if len(candidate_names) > _SHORTLIST_MAXIMUM_CANDIDATES:
        raise ValueError("shortlist exceeds the fixed maximum candidate count")
    if set(candidate_names).intersection(name for name, _reason in exclusion_reasons):
        raise ValueError("selected and excluded candidate names must be disjoint")
    if shortlist_underfilled != (len(candidate_names) < _SHORTLIST_MINIMUM_CANDIDATES):
        raise ValueError("shortlist underfilled flag disagrees with the fixed minimum")


def fit_candidate_priors(
    rows: Sequence[object], *, task_ids: Sequence[str],
    morphology_keys: Mapping[str, str], candidate_names: Sequence[str],
) -> tuple["CandidatePriorV1", ...]:
    """Fit deterministic candidate priors from labeled Train task rows only."""
    from .task_local_evolution import TaskLocalTaskRow

    raw_ids = tuple(task_ids)
    if not raw_ids or any(type(value) is not str or not value for value in raw_ids) or len(raw_ids) != len(set(raw_ids)):
        raise ValueError("prior fitting requires unique nonempty task IDs")
    ids = tuple(sorted(raw_ids))
    if set(morphology_keys) != set(ids):
        raise ValueError("prior fitting requires one morphology key per task")
    keys = {}
    for task_id in ids:
        value = morphology_keys[task_id]
        normalized = _name(value, "morphology key")
        if normalized != value:
            raise ValueError("morphology keys must be canonical lowercase NFKC")
        keys[task_id] = value
    names = tuple(_name(value, "candidate name") for value in candidate_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("candidate names must be unique canonical IDs")
    by_task: dict[str, dict[str, object]] = {task_id: {} for task_id in ids}
    families: dict[str, str] = {}
    for row in rows:
        if type(row) is not TaskLocalTaskRow:
            raise ValueError("prior fitting requires exact task-local rows")
        if row.split != "train":
            continue
        if row.task_id not in by_task:
            continue
        if row.candidate_name in by_task[row.task_id]:
            raise ValueError("prior fitting rows contain duplicate candidate/task keys")
        prior_family = families.setdefault(row.candidate_name, row.family)
        if prior_family != row.family:
            raise ValueError("candidate family is inconsistent")
        by_task[row.task_id][row.candidate_name] = row
    if any(name not in families for name in names):
        raise ValueError("prior fitting requires candidate family evidence")

    fitted: list[CandidatePriorV1] = []
    for name in sorted(names):
        task_scores: list[tuple[str, float]] = []
        successes = 0
        for task_id in ids:
            row = by_task[task_id].get(name)
            if row is None or row.forecast is None:
                score = 5.0
            else:
                point = drcik_point_metrics(row.truth, row.forecast)
                score = float(joint_scaled_error(float(point["smae"]), float(point["srmse"])))
                if not math.isfinite(score):
                    score = 5.0
                else:
                    successes += 1
            task_scores.append((task_id, score))
        values = [score for _task_id, score in task_scores]
        morphology_scores = tuple(
            (key, sum(score for task_id, score in task_scores if keys[task_id] == key) /
             sum(1 for task_id in ids if keys[task_id] == key))
            for key in sorted(set(keys.values()))
        )
        fitted.append(CandidatePriorV1(
            name, families[name], successes / len(ids), sum(values) / len(values),
            float(linear_quantile(values, 0.9)), morphology_scores,
        ))
    return tuple(fitted)


@dataclass(frozen=True)
class CandidatePriorV1:
    candidate_name: str
    family: str
    success_rate: float
    mean_joint: float
    p90_joint: float
    morphology_scores: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        normalized = _name(self.candidate_name, "candidate_name")
        if normalized != self.candidate_name:
            raise ValueError("candidate_name must be canonical lowercase NFKC")
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")
        for field in ("success_rate", "mean_joint", "p90_joint"):
            _finite(getattr(self, field), field)
        keys = tuple(_name(key, "morphology key") for key, _ in self.morphology_scores)
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate morphology score keys")
        if any(key != raw for key, (raw, _) in zip(keys, self.morphology_scores)):
            raise ValueError("morphology keys must be canonical lowercase NFKC")
        for _, score in self.morphology_scores:
            _finite(score, "morphology score")

    def to_payload(self) -> dict[str, object]:
        return {"candidate_name": self.candidate_name, "family": self.family,
                "success_rate": self.success_rate, "mean_joint": self.mean_joint,
                "p90_joint": self.p90_joint,
                "morphology_scores": [[k, v] for k, v in self.morphology_scores]}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CandidatePriorV1":
        if not isinstance(payload, Mapping):
            raise ValueError("candidate prior payload must be an object")
        _fields(payload, {"candidate_name", "family", "success_rate", "mean_joint", "p90_joint", "morphology_scores"}, "CandidatePriorV1")
        scores = payload.get("morphology_scores", ())
        if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
            raise ValueError("morphology_scores must be a list")
        if not isinstance(scores, list):
            raise ValueError("morphology_scores must be a JSON array")
        rows = []
        for item in scores:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("morphology score rows must be [name, score]")
            rows.append((_text(item[0], "morphology key"), _number(item[1], "morphology score")))
        return cls(_text(payload["candidate_name"], "candidate_name"), _text(payload["family"], "family"),
                   _number(payload["success_rate"], "success_rate"), _number(payload["mean_joint"], "mean_joint"), _number(payload["p90_joint"], "p90_joint"), tuple(rows))

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class TaskShortlistPolicyV1:
    schema_version: int = 1
    minimum_candidates: int = 6
    target_candidates: int = 8
    maximum_candidates: int = 10

    def __post_init__(self) -> None:
        if (self.schema_version, self.minimum_candidates, self.target_candidates, self.maximum_candidates) != (1, _SHORTLIST_MINIMUM_CANDIDATES, 8, _SHORTLIST_MAXIMUM_CANDIDATES):
            raise ValueError("TaskShortlistPolicyV1 bounds are exactly 1/6/8/10")

    def to_payload(self) -> dict[str, object]:
        return {"schema_version": 1, "minimum_candidates": _SHORTLIST_MINIMUM_CANDIDATES, "target_candidates": 8, "maximum_candidates": _SHORTLIST_MAXIMUM_CANDIDATES}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TaskShortlistPolicyV1":
        if not isinstance(payload, Mapping):
            raise ValueError("shortlist policy payload must be an object")
        _fields(payload, {"schema_version", "minimum_candidates", "target_candidates", "maximum_candidates"}, "TaskShortlistPolicyV1")
        values = [payload[key] for key in ("schema_version", "minimum_candidates", "target_candidates", "maximum_candidates")]
        if any(type(value) is not int for value in values):
            raise ValueError("shortlist policy numeric fields must be integers")
        return cls(*values)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class TaskCandidateShortlistV1:
    schema_version: int
    task_input_sha256: str
    dictionary_sha256: str
    policy_sha256: str
    candidate_names: tuple[str, ...]
    exclusion_reasons: tuple[tuple[str, str], ...]
    shortlist_underfilled: bool
    public_test_accessed: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported shortlist schema")
        _sha(self.task_input_sha256, "task_input_sha256"); _sha(self.dictionary_sha256, "dictionary_sha256"); _sha(self.policy_sha256, "policy_sha256")
        names = tuple(_name(v, "candidate name") for v in self.candidate_names)
        if names != self.candidate_names or len(names) != len(set(names)):
            raise ValueError("candidate names must be unique canonical IDs")
        exclusions = tuple((_name(k, "excluded candidate"), str(reason)) for k, reason in self.exclusion_reasons)
        if any(reason not in {"unsafe_status", "not_applicable", "unavailable", "ranked_out"} for _, reason in exclusions):
            raise ValueError("unknown exclusion reason")
        order = {reason: index for index, reason in enumerate(("unsafe_status", "not_applicable", "unavailable", "ranked_out"))}
        if tuple(sorted(exclusions, key=lambda item: (order[item[1]], item[0]))) != exclusions:
            raise ValueError("exclusions must use canonical deterministic order")
        if exclusions != self.exclusion_reasons or len(exclusions) != len(set(k for k, _ in exclusions)):
            raise ValueError("exclusions must be ordered and unique canonical IDs")
        if not isinstance(self.shortlist_underfilled, bool) or not isinstance(self.public_test_accessed, bool):
            raise ValueError("shortlist flags must be booleans")
        validate_task_candidate_shortlist(
            self.candidate_names, self.exclusion_reasons, self.shortlist_underfilled,
        )
        if self.public_test_accessed:
            raise ValueError("Public Test access is forbidden")

    def to_payload(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "task_input_sha256": self.task_input_sha256,
                "dictionary_sha256": self.dictionary_sha256, "policy_sha256": self.policy_sha256,
                "candidate_names": list(self.candidate_names), "exclusion_reasons": [list(x) for x in self.exclusion_reasons],
                "shortlist_underfilled": self.shortlist_underfilled, "public_test_accessed": self.public_test_accessed}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TaskCandidateShortlistV1":
        if not isinstance(payload, Mapping):
            raise ValueError("shortlist payload must be an object")
        _fields(payload, {"schema_version", "task_input_sha256", "dictionary_sha256", "policy_sha256", "candidate_names", "exclusion_reasons", "shortlist_underfilled", "public_test_accessed"}, "TaskCandidateShortlistV1")
        if type(payload["schema_version"]) is not int or type(payload["shortlist_underfilled"]) is not bool or type(payload["public_test_accessed"]) is not bool:
            raise ValueError("shortlist schema and flags have invalid JSON types")
        for key in ("task_input_sha256", "dictionary_sha256", "policy_sha256"):
            if not isinstance(payload[key], str):
                raise ValueError(f"{key} must be a string")
        names = payload["candidate_names"]; exclusions = payload["exclusion_reasons"]
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise ValueError("candidate_names must be a JSON array of strings")
        if not isinstance(exclusions, list):
            raise ValueError("exclusion_reasons must be a JSON array")
        rows = []
        for item in exclusions:
            if not isinstance(item, list) or len(item) != 2 or not all(isinstance(x, str) for x in item):
                raise ValueError("exclusion rows must be [name, reason] strings")
            rows.append(tuple(item))
        return cls(payload["schema_version"], payload["task_input_sha256"], payload["dictionary_sha256"], payload["policy_sha256"], tuple(names), tuple(rows), payload["shortlist_underfilled"], payload["public_test_accessed"])

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _dictionary_hash(dictionary: FilterDictionary) -> str:
    payload = {"entries": [{"name": e.name, "family": e.family, "status": e.status,
                            "applicability": list(e.applicability), "reason": e.reason} for e in dictionary.entries]}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_task_candidate_shortlist(*, dictionary: FilterDictionary, profile: TaskProfile,
    task_input_sha256: str, anchor_name: str, available_names: Sequence[str],
    priors: Sequence[CandidatePriorV1], policy: TaskShortlistPolicyV1,
    screening: ScreeningPolicy) -> TaskCandidateShortlistV1:
    if type(dictionary) is not FilterDictionary or type(profile) is not TaskProfile:
        raise TypeError("dictionary and profile must be exact contract types")
    _sha(task_input_sha256, "task_input_sha256")
    names = tuple(_name(v, "available candidate") for v in available_names)
    if len(names) != len(set(names)):
        raise ValueError("available_names contains duplicates after normalization")
    anchor = _name(anchor_name, "anchor_name")
    prior_map = {_name(p.candidate_name): p for p in priors}
    if len(prior_map) != len(priors):
        raise ValueError("priors contain duplicate candidate names")
    entries = {_name(e.name): e for e in dictionary.entries}
    if type(screening) is not ScreeningPolicy:
        raise TypeError("screening must be an exact ScreeningPolicy")
    active = materialize_active_dictionary(screening, profile)
    active_names = {_name(item.name) for item in active.active}
    eligible: list[tuple[str, CandidatePriorV1]] = []
    excluded: list[tuple[str, str]] = []
    for candidate in names:
        entry = entries.get(candidate)
        if entry is None or candidate not in prior_map:
            excluded.append((candidate, "unavailable")); continue
        screening_entry = screening.get(entry.name)
        if entry.status not in {"keep", "specialized"} or (screening_entry is not None and screening_entry.status not in {"keep", "specialized"}):
            excluded.append((candidate, "unsafe_status")); continue
        if candidate not in active_names:
            excluded.append((candidate, "not_applicable")); continue
        eligible.append((candidate, prior_map[candidate]))
    if not any(candidate == anchor for candidate, _ in eligible):
        raise ValueError("anchor must be an eligible available candidate")
    selected = [anchor]
    remaining = [(candidate, prior) for candidate, prior in eligible if candidate != anchor]
    family_counts = {prior.family: 0 for _, prior in eligible}; family_counts[prior_map[anchor].family] = 1
    def rank(item: tuple[str, CandidatePriorV1]) -> tuple[float, float, float, float, int, str]:
        candidate, prior = item
        scores = dict(prior.morphology_scores)
        morphology = scores.get(
            task_morphology_key(profile), scores.get("default", math.inf)
        )
        return (morphology, -prior.success_rate, prior.mean_joint, prior.p90_joint, family_counts.get(prior.family, 0), candidate)
    for _ in range(max(0, policy.target_candidates - 1)):
        if not remaining:
            break
        remaining.sort(key=rank)
        candidate, prior = remaining.pop(0)
        selected.append(candidate); family_counts[prior.family] = family_counts.get(prior.family, 0) + 1
    selected_set = set(selected)
    excluded.extend((candidate, "ranked_out") for candidate, _ in eligible if candidate not in selected_set)
    excluded.sort(key=lambda item: ({"unsafe_status": 0, "not_applicable": 1, "unavailable": 2, "ranked_out": 3}[item[1]], item[0]))
    return TaskCandidateShortlistV1(1, task_input_sha256, _dictionary_hash(dictionary), policy_sha(policy), tuple(selected), tuple(excluded), len(selected) < policy.minimum_candidates, False)


def policy_sha(policy: TaskShortlistPolicyV1) -> str:
    return hashlib.sha256(canonical_json_bytes(policy.to_payload())).hexdigest()
