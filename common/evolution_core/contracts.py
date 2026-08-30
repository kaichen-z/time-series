"""Typed contracts shared by self-evolution adapters."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, Mapping, Protocol, Sequence, TypeVar


ArtifactT = TypeVar("ArtifactT")
ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


METRIC_POLICY = {
    "schema_version": 2,
    "primary": ("smae", "srmse"),
    "cap": 5.0,
    "ordering": "mean_pair",
    "acceptance": "pareto_non_regression",
}
DIAGNOSTIC_ONLY_METRICS = ("mase", "mae", "smape", "rmsse")


def _metric_policy_fingerprint(policy: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(policy), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


METRIC_POLICY_FINGERPRINT = _metric_policy_fingerprint(METRIC_POLICY)


def metric_policy_metadata() -> dict[str, object]:
    """Return the canonical versioned binding for an active serialized artifact."""
    return {
        "metric_policy": dict(METRIC_POLICY),
        "metric_policy_fingerprint": METRIC_POLICY_FINGERPRINT,
    }


def metric_report_metadata() -> dict[str, object]:
    """Return metric provenance plus explicit primary/diagnostic report roles."""
    return {
        **metric_policy_metadata(),
        "primary_metrics": list(METRIC_POLICY["primary"]),
        "diagnostic_only": list(DIAGNOSTIC_ONLY_METRICS),
    }


def require_active_metric_policy(
    payload: Mapping[str, object], *, context: str = "active release"
) -> None:
    """Fail closed unless an active artifact carries the exact canonical policy."""
    raw = payload.get("metric_policy")
    if not isinstance(raw, Mapping):
        if _mentions_legacy_metric(payload):
            raise ValueError(f"{context} uses a legacy metric policy")
        raise ValueError(f"{context} is missing metric policy fields")
    if not _is_canonical_metric_policy(raw):
        if _mentions_legacy_metric(raw):
            raise ValueError(f"{context} uses a legacy metric policy")
        raise ValueError(f"{context} metric policy does not match the active contract")
    legacy_controls = (
        "ranking_order",
        "metric",
        "method_metric",
        "dictionary_metric",
        "objective",
    )
    if any(
        key in payload and _mentions_legacy_metric(payload[key])
        for key in legacy_controls
    ):
        raise ValueError(f"{context} uses a legacy metric policy")
    fingerprint = payload.get("metric_policy_fingerprint")
    if fingerprint != METRIC_POLICY_FINGERPRINT:
        if fingerprint is None:
            raise ValueError(f"{context} is missing metric policy fingerprint")
        raise ValueError(f"{context} metric policy fingerprint mismatch")


def load_active_release(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate and copy an active release; historical readers must be explicit."""
    require_active_metric_policy(payload)
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 2:
        raise ValueError("active release schema_version must be 2")
    return dict(payload)


def _normalized_metric_policy(policy: Mapping[str, object]) -> dict[str, object]:
    primary = policy.get("primary")
    if isinstance(primary, Sequence) and not isinstance(primary, (str, bytes)):
        primary = tuple(primary)
    return {**dict(policy), "primary": primary}


def _is_canonical_metric_policy(policy: Mapping[str, object]) -> bool:
    primary = policy.get("primary")
    return (
        set(policy) == set(METRIC_POLICY)
        and type(policy.get("schema_version")) is int
        and isinstance(primary, (list, tuple))
        and all(isinstance(name, str) for name in primary)
        and type(policy.get("cap")) is float
        and isinstance(policy.get("ordering"), str)
        and isinstance(policy.get("acceptance"), str)
        and _normalized_metric_policy(policy) == _normalized_metric_policy(METRIC_POLICY)
    )


def _mentions_legacy_metric(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            _mentions_legacy_metric(key) or _mentions_legacy_metric(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_mentions_legacy_metric(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        tokens = set(re.findall(r"[a-z0-9]+", lowered))
        return bool(tokens & {"mase", "mae", "smape", "rmsse"})
    return False


@dataclass(frozen=True)
class MetricSpec:
    """Primary metric and ordering used for screening and acceptance."""

    name: str
    objective: Literal["minimize", "maximize"] = "minimize"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("metric name must not be empty")
        if self.objective not in ("minimize", "maximize"):
            raise ValueError("objective must be 'minimize' or 'maximize'")

    def better(self, candidate: float, parent: float, margin: float = 0.0) -> bool:
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if not math.isfinite(candidate) or not math.isfinite(parent):
            raise ValueError("metric values must be finite")
        if self.objective == "minimize":
            return candidate < parent - margin
        return candidate > parent + margin


@dataclass(frozen=True)
class EvolutionConfig:
    """Domain-independent search budget and acceptance configuration."""

    generations: int = 1
    children_per_generation: int = 2
    seed: int = 20260816
    metric: MetricSpec = field(default_factory=lambda: MetricSpec("smae"))
    metric_policy: Mapping[str, object] = field(
        default_factory=lambda: dict(METRIC_POLICY)
    )
    metric_policy_fingerprint: str = METRIC_POLICY_FINGERPRINT
    acceptance_margin: float = 0.0
    resume: bool = True

    def __post_init__(self) -> None:
        positive = {
            "generations": self.generations,
            "children_per_generation": self.children_per_generation,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.acceptance_margin < 0:
            raise ValueError("acceptance_margin must be non-negative")
        if not _is_canonical_metric_policy(self.metric_policy):
            raise ValueError("metric_policy must match the active scaled metric contract")
        if self.metric_policy_fingerprint != METRIC_POLICY_FINGERPRINT:
            raise ValueError("metric_policy_fingerprint does not match metric_policy")


@dataclass(frozen=True)
class EvaluationReport:
    """Trusted aggregate for one frozen artifact on one split."""

    artifact_id: str
    split: str
    metrics: Mapping[str, float]
    item_count: int
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MutationContext:
    """Sanitized feedback available to a mutator."""

    generation: int
    parent_train_report: EvaluationReport
    failure_traces: tuple[Mapping[str, object], ...] = ()
    diversity_instruction: str = ""


class ArtifactAdapter(Protocol[ArtifactT]):
    def validate(self, artifact: ArtifactT) -> None: ...

    def artifact_id(self, artifact: ArtifactT) -> str: ...

    def to_payload(self, artifact: ArtifactT) -> dict[str, object]: ...

    def from_payload(self, payload: Mapping[str, object]) -> ArtifactT: ...

    def apply_train_report(
        self, artifact: ArtifactT, report: EvaluationReport
    ) -> ArtifactT: ...


class Mutator(Protocol[ArtifactT]):
    def propose(
        self, parent: ArtifactT, context: MutationContext, count: int
    ) -> Sequence[ArtifactT]: ...


class Executor(Protocol[ArtifactT, ItemT, ResultT]):
    def execute(
        self, artifact: ArtifactT, items: Sequence[ItemT], split: str
    ) -> Sequence[ResultT]: ...


class Evaluator(Protocol[ResultT]):
    def evaluate(
        self, artifact_id: str, results: Sequence[ResultT], split: str
    ) -> EvaluationReport: ...


class AcceptanceGate(Protocol):
    def accept(
        self, parent_report: EvaluationReport, child_report: EvaluationReport
    ) -> bool: ...


class ArtifactStore(Protocol):
    def save_artifact(self, name: str, payload: Mapping[str, object]) -> Path: ...

    def save_checkpoint(self, payload: Mapping[str, object]) -> Path: ...

    def load_checkpoint(self) -> dict[str, object] | None: ...

    def append_trace(self, payload: Mapping[str, object]) -> None: ...


@dataclass(frozen=True)
class EvolutionComponents(Generic[ArtifactT, ItemT, ResultT]):
    artifact_adapter: ArtifactAdapter[ArtifactT]
    mutator: Mutator[ArtifactT]
    executor: Executor[ArtifactT, ItemT, ResultT]
    evaluator: Evaluator[ResultT]
    acceptance_gate: AcceptanceGate
    store: ArtifactStore
