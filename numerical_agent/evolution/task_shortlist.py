"""Canonical, history-only per-task candidate shortlists."""
from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass
from typing import Mapping, Sequence

from common.payload import canonical_json_bytes

from .filtering import FAMILIES, FilterDictionary
from .screening import ScreeningPolicy, TaskProfile, materialize_active_dictionary, profile_tags


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
        return cls(str(payload["candidate_name"]), str(payload["family"]),
                   _finite(payload["success_rate"], "success_rate"),
                   _finite(payload["mean_joint"], "mean_joint"),
                   _finite(payload["p90_joint"], "p90_joint"),
                   tuple((_name(item[0], "morphology key"), _finite(item[1], "morphology score")) for item in scores))

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
        if (self.schema_version, self.minimum_candidates, self.target_candidates, self.maximum_candidates) != (1, 6, 8, 10):
            raise ValueError("TaskShortlistPolicyV1 bounds are exactly 1/6/8/10")

    def to_payload(self) -> dict[str, object]:
        return {"schema_version": 1, "minimum_candidates": 6, "target_candidates": 8, "maximum_candidates": 10}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TaskShortlistPolicyV1":
        if not isinstance(payload, Mapping):
            raise ValueError("shortlist policy payload must be an object")
        _fields(payload, {"schema_version", "minimum_candidates", "target_candidates", "maximum_candidates"}, "TaskShortlistPolicyV1")
        return cls(int(payload["schema_version"]), int(payload["minimum_candidates"]), int(payload["target_candidates"]), int(payload["maximum_candidates"]))

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
        return cls(int(payload["schema_version"]), str(payload["task_input_sha256"]), str(payload["dictionary_sha256"]), str(payload["policy_sha256"]),
                   tuple(payload["candidate_names"]), tuple(tuple(x) for x in payload["exclusion_reasons"]), payload["shortlist_underfilled"], payload["public_test_accessed"])

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
    tags = profile_tags(profile)
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
        morphology = min((scores[tag] for tag in tags if tag in scores), default=scores.get("default", math.inf))
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
