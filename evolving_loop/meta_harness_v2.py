"""Closed proposal and Train-memory contracts for Meta-Harness V2."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal, Mapping, Sequence, cast


MetaChildKind = Literal["coding", "retrieval", "decision", "joint"]
MetaScope = Literal["coding", "retrieval", "decision", "coordination"]
MetaTrainStatus = Literal["invalid", "train_improved", "train_rejected"]

_CHILD_KINDS = ("coding", "retrieval", "decision", "joint")
_SCOPES = ("coding", "retrieval", "decision", "coordination")
_TRAIN_STATUSES = ("invalid", "train_improved", "train_rejected")

_PROMPT_FIELDS = (
    "coding_generation_prompt",
    "coding_revision_prompt",
    "retrieval_prompt",
    "decision_prompt",
)
_INTEGER_RANGES = {
    "coding_initial_programs": (1, 12),
    "coding_mutations": (0, 6),
    "coding_mutation_children": (1, 6),
    "coding_validation_folds": (1, 8),
    "coding_validation_horizon": (1, 64),
    "max_evidence_adjustments": (0, 12),
}
_PROPOSAL_FIELDS = frozenset(
    {
        "mutation_scope",
        "interaction_hypothesis",
        *_PROMPT_FIELDS,
        *_INTEGER_RANGES,
        "workflow",
        "enable_evidence_adjustments",
        "decision_aggregation",
        "changelog",
    }
)
_MEMORY_FIELDS = frozenset(
    {
        "generation",
        "child_kind",
        "changed_scopes",
        "interaction_hypothesis",
        "status",
        "parent_train_smae",
        "parent_train_srmse",
        "child_train_smae",
        "child_train_srmse",
    }
)

_SCOPE_FIELDS: dict[MetaScope, frozenset[str]] = {
    "coding": frozenset(
        {
            "coding_generation_prompt",
            "coding_revision_prompt",
            "coding_initial_programs",
            "coding_mutations",
            "coding_mutation_children",
            "coding_validation_folds",
            "coding_validation_horizon",
        }
    ),
    "retrieval": frozenset({"retrieval_prompt"}),
    "decision": frozenset(
        {
            "decision_prompt",
            "enable_evidence_adjustments",
            "max_evidence_adjustments",
            "decision_aggregation",
        }
    ),
    "coordination": frozenset({"workflow"}),
}


META_HARNESS_V2_PROMPT = """You evolve one complete three-agent forecasting Harness Genome.
Return exactly one JSON object with every field shown below and no extra fields:
{
  "mutation_scope": ["coding|retrieval|decision|coordination", "..."],
  "interaction_hypothesis": "one testable explanation of the interaction",
  "coding_generation_prompt": "...",
  "coding_revision_prompt": "...",
  "retrieval_prompt": "...",
  "decision_prompt": "...",
  "coding_initial_programs": 3,
  "coding_mutations": 1,
  "coding_mutation_children": 1,
  "coding_validation_folds": 3,
  "coding_validation_horizon": 8,
  "workflow": ["retrieve", "decide"],
  "enable_evidence_adjustments": true,
  "max_evidence_adjustments": 3,
  "decision_aggregation": "last|majority",
  "changelog": "..."
}
The Host supplies requested_child_kind. A coding, retrieval, or decision Child must
change exactly that role. A joint Child must change at least two role scopes; it may
also change coordination. Do not change task splits, metrics, verification, sandbox,
runtime authority, Skill bodies, or accepted release artifacts. Evolution memory is
Train-only aggregate evidence; never infer or request task identities or future data.
"""


def _exact_dict(payload: object, fields: frozenset[str], label: str) -> dict[str, object]:
    if type(payload) is not dict:
        raise ValueError(f"{label} must be an exact JSON object")
    result = cast(dict[str, object], payload)
    if set(result) != fields or any(type(key) is not str for key in result):
        raise ValueError(f"{label} must use the exact schema")
    return result


def _nonempty_string(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _bounded_integer(value: object, field: str, lower: int, upper: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    if value < lower or value > upper:
        raise ValueError(f"{field} is outside the allowed range")
    return value


def _scopes(value: object, field: str, *, allow_empty: bool = False) -> tuple[MetaScope, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{field} must be a scope list")
    raw = cast(Sequence[object], value)
    if not raw and not allow_empty:
        raise ValueError(f"{field} must contain at least one scope")
    if any(type(item) is not str or item not in _SCOPES for item in raw):
        raise ValueError(f"{field} contains an unknown scope")
    if len(set(raw)) != len(raw):
        raise ValueError(f"{field} contains a duplicate scope")
    selected = set(cast(Sequence[str], raw))
    return cast(tuple[MetaScope, ...], tuple(scope for scope in _SCOPES if scope in selected))


def child_kind_for_slot(slot: int) -> MetaChildKind:
    if type(slot) is not int or slot < 0:
        raise ValueError("child slot must be a non-negative integer")
    if slot < 3:
        return cast(MetaChildKind, _CHILD_KINDS[slot])
    return "joint"


@dataclass(frozen=True)
class MetaHarnessProposal:
    mutation_scope: tuple[MetaScope, ...]
    interaction_hypothesis: str
    coding_generation_prompt: str
    coding_revision_prompt: str
    retrieval_prompt: str
    decision_prompt: str
    coding_initial_programs: int
    coding_mutations: int
    coding_mutation_children: int
    coding_validation_folds: int
    coding_validation_horizon: int
    workflow: tuple[str, ...]
    enable_evidence_adjustments: bool
    max_evidence_adjustments: int
    decision_aggregation: str
    changelog: str

    @classmethod
    def from_payload(cls, payload: object) -> "MetaHarnessProposal":
        raw = _exact_dict(payload, _PROPOSAL_FIELDS, "Meta-Harness proposal")
        prompts = {field: _nonempty_string(raw[field], field) for field in _PROMPT_FIELDS}
        integers = {
            field: _bounded_integer(raw[field], field, lower, upper)
            for field, (lower, upper) in _INTEGER_RANGES.items()
        }
        workflow_raw = raw["workflow"]
        if type(workflow_raw) not in (list, tuple):
            raise ValueError("workflow must be a JSON list")
        workflow = tuple(workflow_raw)
        if (
            not workflow
            or len(workflow) > 8
            or any(type(stage) is not str for stage in workflow)
            or set(workflow) - {"retrieve", "decide"}
            or "retrieve" not in workflow
            or "decide" not in workflow
        ):
            raise ValueError("workflow must contain retrieve and decide stages only")
        if type(raw["enable_evidence_adjustments"]) is not bool:
            raise ValueError("enable_evidence_adjustments must be a boolean")
        aggregation = raw["decision_aggregation"]
        if type(aggregation) is not str or aggregation not in {"last", "majority"}:
            raise ValueError("decision aggregation must be last or majority")
        return cls(
            mutation_scope=_scopes(raw["mutation_scope"], "mutation_scope"),
            interaction_hypothesis=_train_hypothesis(raw["interaction_hypothesis"]),
            workflow=cast(tuple[str, ...], workflow),
            enable_evidence_adjustments=cast(bool, raw["enable_evidence_adjustments"]),
            decision_aggregation=aggregation,
            changelog=_nonempty_string(raw["changelog"], "changelog")[:500],
            **prompts,
            **integers,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "mutation_scope": list(self.mutation_scope),
            "interaction_hypothesis": self.interaction_hypothesis,
            "coding_generation_prompt": self.coding_generation_prompt,
            "coding_revision_prompt": self.coding_revision_prompt,
            "retrieval_prompt": self.retrieval_prompt,
            "decision_prompt": self.decision_prompt,
            "coding_initial_programs": self.coding_initial_programs,
            "coding_mutations": self.coding_mutations,
            "coding_mutation_children": self.coding_mutation_children,
            "coding_validation_folds": self.coding_validation_folds,
            "coding_validation_horizon": self.coding_validation_horizon,
            "workflow": list(self.workflow),
            "enable_evidence_adjustments": self.enable_evidence_adjustments,
            "max_evidence_adjustments": self.max_evidence_adjustments,
            "decision_aggregation": self.decision_aggregation,
            "changelog": self.changelog,
        }


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def policy_change_scope(
    parent_payload: Mapping[str, object], child_payload: Mapping[str, object]
) -> tuple[MetaScope, ...]:
    changed: list[MetaScope] = []
    for scope in _SCOPES:
        if any(
            _plain(parent_payload.get(field)) != _plain(child_payload.get(field))
            for field in _SCOPE_FIELDS[cast(MetaScope, scope)]
        ):
            changed.append(cast(MetaScope, scope))
    return tuple(changed)


def validate_child_scope(
    child_kind: MetaChildKind | str,
    declared_scopes: Sequence[str],
    actual_scopes: Sequence[str],
) -> None:
    if child_kind not in _CHILD_KINDS:
        raise ValueError("unknown Meta-Harness child kind")
    declared = _scopes(tuple(declared_scopes), "declared mutation scope")
    actual = _scopes(tuple(actual_scopes), "actual mutation scope")
    if declared != actual:
        raise ValueError("declared mutation scope does not match the actual change")
    if child_kind != "joint":
        if actual != (child_kind,):
            raise ValueError("role-isolated Child changed fields outside its isolated scope")
        return
    role_count = len(set(actual) & {"coding", "retrieval", "decision"})
    if role_count < 2:
        raise ValueError("joint Child must change at least two role scopes")


def _metric(value: object, field: str, *, optional: bool) -> float | None:
    if value is None and optional:
        return None
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


_TASK_ID_PATTERN = re.compile(r"\btask[_-]?\d+\b", re.IGNORECASE)


def _train_hypothesis(value: object) -> str:
    hypothesis = _nonempty_string(value, "interaction_hypothesis")
    if _TASK_ID_PATTERN.search(hypothesis):
        raise ValueError("interaction_hypothesis cannot contain task-level identity")
    return hypothesis


@dataclass(frozen=True)
class MetaTrainMemoryRecord:
    generation: int
    child_kind: MetaChildKind
    changed_scopes: tuple[MetaScope, ...]
    interaction_hypothesis: str
    status: MetaTrainStatus
    parent_train_smae: float
    parent_train_srmse: float
    child_train_smae: float | None
    child_train_srmse: float | None

    def __post_init__(self) -> None:
        generation = _bounded_integer(self.generation, "generation", 0, 1_000_000)
        if self.child_kind not in _CHILD_KINDS:
            raise ValueError("unknown child kind")
        scopes = _scopes(self.changed_scopes, "changed_scopes", allow_empty=True)
        hypothesis = _train_hypothesis(self.interaction_hypothesis)
        if self.status not in _TRAIN_STATUSES:
            raise ValueError("unknown Train-memory status")
        parent_smae = _metric(self.parent_train_smae, "parent_train_smae", optional=False)
        parent_srmse = _metric(self.parent_train_srmse, "parent_train_srmse", optional=False)
        child_smae = _metric(self.child_train_smae, "child_train_smae", optional=True)
        child_srmse = _metric(self.child_train_srmse, "child_train_srmse", optional=True)
        if self.status == "invalid":
            if child_smae is not None or child_srmse is not None:
                raise ValueError("invalid memory records cannot carry Child metrics")
        elif child_smae is None or child_srmse is None:
            raise ValueError("evaluated memory records require both Child metrics")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "changed_scopes", scopes)
        object.__setattr__(self, "interaction_hypothesis", hypothesis)
        object.__setattr__(self, "parent_train_smae", parent_smae)
        object.__setattr__(self, "parent_train_srmse", parent_srmse)
        object.__setattr__(self, "child_train_smae", child_smae)
        object.__setattr__(self, "child_train_srmse", child_srmse)

    @classmethod
    def from_payload(cls, payload: object) -> "MetaTrainMemoryRecord":
        raw = _exact_dict(payload, _MEMORY_FIELDS, "Meta-Harness Train memory")
        changed = raw["changed_scopes"]
        if type(changed) not in (list, tuple):
            raise ValueError("changed_scopes must be a scope list")
        return cls(
            generation=cast(int, raw["generation"]),
            child_kind=cast(MetaChildKind, raw["child_kind"]),
            changed_scopes=cast(tuple[MetaScope, ...], tuple(changed)),
            interaction_hypothesis=cast(str, raw["interaction_hypothesis"]),
            status=cast(MetaTrainStatus, raw["status"]),
            parent_train_smae=cast(float, raw["parent_train_smae"]),
            parent_train_srmse=cast(float, raw["parent_train_srmse"]),
            child_train_smae=cast(float | None, raw["child_train_smae"]),
            child_train_srmse=cast(float | None, raw["child_train_srmse"]),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "child_kind": self.child_kind,
            "changed_scopes": list(self.changed_scopes),
            "interaction_hypothesis": self.interaction_hypothesis,
            "status": self.status,
            "parent_train_smae": self.parent_train_smae,
            "parent_train_srmse": self.parent_train_srmse,
            "child_train_smae": self.child_train_smae,
            "child_train_srmse": self.child_train_srmse,
        }


def project_train_memory(
    records: Sequence[MetaTrainMemoryRecord], *, window: int
) -> tuple[dict[str, object], ...]:
    if type(window) is not int or window <= 0:
        raise ValueError("memory window must be a positive integer")
    if not records:
        return ()
    generations = sorted({record.generation for record in records})
    selected = set(generations[-window:])
    return tuple(record.to_payload() for record in records if record.generation in selected)
