"""Closed model-facing contract for task-level cross-coordinate evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal


class TaskFeedbackError(ValueError):
    """Task feedback is malformed, unsafe, or outside evolution authority."""


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID = re.compile(r"case_[0-9]{3}_[0-9a-f]{8}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}\Z")
_TASK_IDENTITY = re.compile(r"(?i)task[_-]?[0-9]")
_DOCUMENT_IDENTITY = re.compile(r"(?i)(?:document|doc)[_-]?[0-9]")
_FORBIDDEN_TEXT = (
    "future_values",
    "gt_evidence",
    "forecast array",
    "forecast vector",
    "forecast residual",
    "task score",
    "document id",
    "exact quote",
    "/private/",
    "/users/",
    "runs/",
)

FrequencyBucket = Literal[
    "subdaily", "daily", "weekly", "monthly", "quarterly", "yearly", "other"
]
LengthBucket = Literal["short", "medium", "long"]
TrendBucket = Literal[
    "strong_down", "weak_down", "flat", "weak_up", "strong_up"
]
StrengthBucket = Literal["none", "weak", "strong"]
IntermittencyBucket = Literal["dense", "intermittent", "sparse"]
RegimeBucket = Literal["stable", "recent_shift"]
Mechanism = Literal[
    "event_shock",
    "promotion",
    "policy_change",
    "capacity_change",
    "measurement_error",
    "regime_change",
    "calendar_effect",
    "macroeconomic",
    "unknown",
]

_FREQUENCIES = frozenset(
    {"subdaily", "daily", "weekly", "monthly", "quarterly", "yearly", "other"}
)
_LENGTHS = frozenset({"short", "medium", "long"})
_TRENDS = frozenset(
    {"strong_down", "weak_down", "flat", "weak_up", "strong_up"}
)
_STRENGTHS = frozenset({"none", "weak", "strong"})
_INTERMITTENCY = frozenset({"dense", "intermittent", "sparse"})
_REGIMES = frozenset({"stable", "recent_shift"})
_STANCES = frozenset({"supported", "falsified", "uncertain"})
_TARGET_MATCHES = frozenset({"matched", "unmatched"})
_WINDOW_RELATIONS = frozenset({"overlaps", "precedes", "after", "unknown"})
_MAGNITUDES = frozenset({"present", "missing", "conflicting", "not_applicable"})
_MECHANISMS = frozenset(
    {
        "event_shock",
        "promotion",
        "policy_change",
        "capacity_change",
        "measurement_error",
        "regime_change",
        "calendar_effect",
        "macroeconomic",
        "unknown",
    }
)
_ACTIONS = frozenset({"selected", "rejected", "unresolved"})


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_enum(value: object, allowed: frozenset[str], label: str) -> str:
    if type(value) is not str or value not in allowed:
        raise TaskFeedbackError(f"task feedback {label} is not in the closed enum")
    return value


def _require_safe_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > 512:
        raise TaskFeedbackError(f"task feedback {label} must be bounded text")
    normalized = value.casefold()
    if (
        _TASK_IDENTITY.search(value)
        or _DOCUMENT_IDENTITY.search(value)
        or any(marker in normalized for marker in _FORBIDDEN_TEXT)
    ):
        raise TaskFeedbackError(f"task feedback {label} contains forbidden identity or result data")
    return value


@dataclass(frozen=True)
class TaskMorphologyProjection:
    """Coarse, history-only morphology visible to the Numerical proposer."""

    frequency: FrequencyBucket
    history: LengthBucket
    horizon: LengthBucket
    trend: TrendBucket
    periodicity: StrengthBucket
    intermittency: IntermittencyBucket
    recent_regime: RegimeBucket

    def __post_init__(self) -> None:
        _require_enum(self.frequency, _FREQUENCIES, "frequency")
        _require_enum(self.history, _LENGTHS, "history")
        _require_enum(self.horizon, _LENGTHS, "horizon")
        _require_enum(self.trend, _TRENDS, "trend")
        _require_enum(self.periodicity, _STRENGTHS, "periodicity")
        _require_enum(self.intermittency, _INTERMITTENCY, "intermittency")
        _require_enum(self.recent_regime, _REGIMES, "recent_regime")

    def to_payload(self) -> dict[str, str]:
        return {
            "frequency": self.frequency,
            "history": self.history,
            "horizon": self.horizon,
            "trend": self.trend,
            "periodicity": self.periodicity,
            "intermittency": self.intermittency,
            "recent_regime": self.recent_regime,
        }


@dataclass(frozen=True)
class TaskEvidenceCase:
    """One sanitized task/assumption observation with no reusable task identity."""

    case_id: str
    morphology: TaskMorphologyProjection
    assumption_id: str
    claim: str
    failure_condition: str
    stance: Literal["supported", "falsified", "uncertain"]
    target_match: Literal["matched", "unmatched"]
    window_relation: Literal["overlaps", "precedes", "after", "unknown"]
    magnitude_status: Literal["present", "missing", "conflicting", "not_applicable"]
    mechanism: Mechanism
    decision_action: Literal["selected", "rejected", "unresolved"]
    evidence_chain_sha256: str

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or _CASE_ID.fullmatch(self.case_id) is None:
            raise TaskFeedbackError("task feedback case_id is not request-local")
        if type(self.morphology) is not TaskMorphologyProjection:
            raise TaskFeedbackError("task feedback morphology must use the exact projection")
        TaskMorphologyProjection.__post_init__(self.morphology)
        if (
            type(self.assumption_id) is not str
            or _IDENTIFIER.fullmatch(self.assumption_id) is None
        ):
            raise TaskFeedbackError("task feedback assumption_id is not a safe identifier")
        _require_safe_text(self.claim, "claim")
        _require_safe_text(self.failure_condition, "failure_condition")
        _require_enum(self.stance, _STANCES, "stance")
        _require_enum(self.target_match, _TARGET_MATCHES, "target_match")
        _require_enum(self.window_relation, _WINDOW_RELATIONS, "window_relation")
        _require_enum(self.magnitude_status, _MAGNITUDES, "magnitude_status")
        _require_enum(self.mechanism, _MECHANISMS, "mechanism")
        _require_enum(self.decision_action, _ACTIONS, "decision_action")
        if (
            type(self.evidence_chain_sha256) is not str
            or _SHA256.fullmatch(self.evidence_chain_sha256) is None
        ):
            raise TaskFeedbackError("task feedback evidence chain digest is invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "morphology": self.morphology.to_payload(),
            "assumption_id": self.assumption_id,
            "claim": self.claim,
            "failure_condition": self.failure_condition,
            "stance": self.stance,
            "target_match": self.target_match,
            "window_relation": self.window_relation,
            "magnitude_status": self.magnitude_status,
            "mechanism": self.mechanism,
            "decision_action": self.decision_action,
            "evidence_chain_sha256": self.evidence_chain_sha256,
        }


@dataclass(frozen=True)
class TaskEvidenceProjection:
    """A request-bound set of cases; only ``to_payload`` crosses into the model."""

    source_bundle_sha256: str
    request_namespace_sha256: str
    cases: tuple[TaskEvidenceCase, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.source_bundle_sha256, "source bundle"),
            (self.request_namespace_sha256, "request namespace"),
        ):
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise TaskFeedbackError(f"task feedback {label} digest is invalid")
        if type(self.cases) is not tuple or any(
            type(item) is not TaskEvidenceCase for item in self.cases
        ):
            raise TaskFeedbackError("task feedback cases must be an exact tuple")
        for item in self.cases:
            TaskEvidenceCase.__post_init__(item)
        ordered = tuple(sorted(self.cases, key=lambda item: item.case_id))
        identities = tuple(item.case_id for item in ordered)
        if len(identities) != len(set(identities)):
            raise TaskFeedbackError("task feedback contains a duplicate case identity")
        object.__setattr__(self, "cases", ordered)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "cases": [item.to_payload() for item in self.cases],
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "source_bundle_sha256": self.source_bundle_sha256,
                    "request_namespace_sha256": self.request_namespace_sha256,
                    "projection": self.to_payload(),
                }
            )
        ).hexdigest()


__all__ = [
    "TaskEvidenceCase",
    "TaskEvidenceProjection",
    "TaskFeedbackError",
    "TaskMorphologyProjection",
]
