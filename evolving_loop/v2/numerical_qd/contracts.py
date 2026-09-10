"""Closed, deeply immutable Train-only Numerical QD artifact contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Literal

from ..budget import ResourceUse
from ..contracts import (
    _freeze_json_value,
    _require_choice,
    _require_exact_schema,
    _require_mapping,
    _strict_json_value,
    canonical_v2_bytes,
    fingerprint_payload,
    require_sha256,
)


MEMBER_FAMILIES = frozenset({"statistical", "tsfm", "combined", "program"})
MUTATION_OPERATORS = frozenset({
    "add", "repair", "fork", "combine", "route", "specialize", "crossover",
    "remove", "quarantine", "policy_tune",
})
CONSTRAINT_NAMES = frozenset({
    "coverage", "nonfinite_forecast", "wrong_horizon", "new_invalid_task",
    "new_catastrophic_task", "joint_regret", "protocol_mismatch",
    "runtime_mismatch", "cache_mismatch", "ownership", "scope",
    "future_leakage", "dev_leakage", "document_role_leakage", "public_leakage",
})
TRAIN_DIAGNOSTIC_CATEGORIES = CONSTRAINT_NAMES | frozenset({
    "timeout", "execution_failure", "invalid_response",
})
_BRACKET_RUNG_COUNTS = {"explore": 3, "confirm": 2, "replay": 1}
_MAX_TEMPLATE_BYTES = 65_536
_MAX_RESPONSE_BYTES = 1_048_576


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _schema_version(value: object) -> None:
    if type(value) is not int or value != 1:
        raise ValueError("schema_version must be exactly 1")


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sequence(value: object, field: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list or tuple")
    return tuple(value)


def _sorted_strings(value: object, field: str, *, sha: bool = False,
                    choices: frozenset[str] | None = None,
                    nonempty: bool = False) -> tuple[str, ...]:
    result = _sequence(value, field)
    for item in result:
        _text(item, field)
        if sha:
            require_sha256(item, field)
        if choices is not None:
            _require_choice(item, field, choices)
    if nonempty and not result:
        raise ValueError(f"{field} must not be empty")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{field} must be sorted and unique")
    return result


def _runtime(value: object) -> object:
    payload = _require_mapping(value, "runtime_fingerprints")
    if not payload:
        raise ValueError("runtime_fingerprints must not be empty")
    for name, digest in payload.items():
        _text(name, "runtime_fingerprints key")
        require_sha256(digest, f"runtime_fingerprints.{name}")
    return _freeze_json_value(payload)


def _nested(value, cls):
    if type(value) is cls:
        return value
    return cls.from_payload(value)


def _plain(value):
    if isinstance(value, _CanonicalContract):
        return value.to_payload()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


class _CanonicalContract:
    __slots__ = ()

    @classmethod
    def from_payload(cls, payload):
        values = _require_exact_schema(
            payload, tuple(field.name for field in fields(cls)), field=cls.__name__
        )
        return cls(**_strict_json_value(values))

    def to_payload(self) -> dict[str, object]:
        return _strict_json_value({
            field.name: _plain(getattr(self, field.name)) for field in fields(self)
        })

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


@dataclass(frozen=True, slots=True)
class MorphologyCellV2(_CanonicalContract):
    trend: Literal["low", "medium", "high"]
    seasonality: Literal["none", "short", "long"]
    intermittency: Literal["low", "high"]
    regime: Literal["stable", "shift"]
    horizon: Literal["short", "medium", "long"]
    family: Literal["statistical", "tsfm", "combined", "program"]

    def __post_init__(self):
        for name, choices in (
            ("trend", frozenset({"low", "medium", "high"})),
            ("seasonality", frozenset({"none", "short", "long"})),
            ("intermittency", frozenset({"low", "high"})),
            ("regime", frozenset({"stable", "shift"})),
            ("horizon", frozenset({"short", "medium", "long"})),
            ("family", MEMBER_FAMILIES),
        ):
            _require_choice(getattr(self, name), name, choices)


@dataclass(frozen=True, slots=True)
class NumericalObjectiveVectorV2(_CanonicalContract):
    mean_capped_smae: float
    mean_capped_srmse: float
    p95_capped_srmse: float
    mean_raw_joint_error: float
    normalized_execution_cost: float

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite float")


@dataclass(frozen=True, slots=True)
class ConstraintReportV2(_CanonicalContract):
    feasible: bool
    violations: tuple[str, ...]

    def __post_init__(self):
        if type(self.feasible) is not bool:
            raise ValueError("feasible must be a boolean")
        violations = _sorted_strings(self.violations, "violations", choices=CONSTRAINT_NAMES)
        if self.feasible != (not violations):
            raise ValueError("feasible must agree with the absence of violations")
        object.__setattr__(self, "violations", violations)


@dataclass(frozen=True, slots=True)
class NumericalMemberV2(_CanonicalContract):
    member_id: str
    family: Literal["statistical", "tsfm", "combined", "program"]
    source_sha256: str
    policy_sha256: str
    parent_ids: tuple[str, ...]
    applicability_cells: tuple[str, ...]
    status: Literal["active", "specialized", "quarantined"]

    def __post_init__(self):
        _text(self.member_id, "member_id")
        _require_choice(self.family, "family", MEMBER_FAMILIES)
        require_sha256(self.source_sha256, "source_sha256")
        require_sha256(self.policy_sha256, "policy_sha256")
        parents = _sorted_strings(self.parent_ids, "parent_ids")
        if self.member_id in parents:
            raise ValueError("parent_ids must not include member_id")
        object.__setattr__(self, "parent_ids", parents)
        object.__setattr__(self, "applicability_cells", _sorted_strings(
            self.applicability_cells, "applicability_cells", sha=True
        ))
        _require_choice(self.status, "status", frozenset({"active", "specialized", "quarantined"}))


@dataclass(frozen=True, slots=True)
class NumericalInventoryV2(_CanonicalContract):
    schema_version: int
    members: tuple[NumericalMemberV2, ...]

    def __post_init__(self):
        _schema_version(self.schema_version)
        members = tuple(_nested(item, NumericalMemberV2)
                        for item in _sequence(self.members, "members"))
        identities = [member.member_id for member in members]
        if len(identities) != len(set(identities)):
            raise ValueError("members must have unique member IDs")
        if not any(member.status == "active" for member in members):
            raise ValueError("members must contain at least one active member")
        object.__setattr__(self, "members", members)


@dataclass(frozen=True, slots=True)
class NumericalGenomeV2(_CanonicalContract):
    schema_version: int
    generation: int
    parent_genome_sha256s: tuple[str, ...]
    mutation_operator: str
    inventory_sha256: str
    screening_policy_sha256: str
    combined_policy_sha256: str
    mutation_policy_sha256: str
    proposer_prompt_sha256: str
    runtime_fingerprints: Mapping[str, str]
    protocol_fingerprint: str

    def __post_init__(self):
        _schema_version(self.schema_version)
        _nonnegative_int(self.generation, "generation")
        object.__setattr__(self, "parent_genome_sha256s", _sorted_strings(
            self.parent_genome_sha256s, "parent_genome_sha256s", sha=True
        ))
        _require_choice(self.mutation_operator, "mutation_operator", MUTATION_OPERATORS)
        for name in ("inventory_sha256", "screening_policy_sha256", "combined_policy_sha256",
                     "mutation_policy_sha256", "proposer_prompt_sha256", "protocol_fingerprint"):
            require_sha256(getattr(self, name), name)
        object.__setattr__(self, "runtime_fingerprints", _runtime(self.runtime_fingerprints))


@dataclass(frozen=True, slots=True)
class MutationOperatorStatsV2(_CanonicalContract):
    attempts: int
    feasible: int
    promotions: int
    insertions: int
    credit: int

    def __post_init__(self):
        for field in fields(self):
            _nonnegative_int(getattr(self, field.name), field.name)


@dataclass(frozen=True, slots=True)
class NumericalMutationPolicyV2(_CanonicalContract):
    schema_version: int
    operators: Mapping[str, MutationOperatorStatsV2]

    def __post_init__(self):
        _schema_version(self.schema_version)
        if not isinstance(self.operators, Mapping) or not self.operators:
            raise ValueError("operators must be a non-empty mapping")
        operators = {}
        for name, stats in self.operators.items():
            _require_choice(name, "operators key", MUTATION_OPERATORS)
            operators[name] = _nested(stats, MutationOperatorStatsV2)
        object.__setattr__(self, "operators", MappingProxyType(operators))


@dataclass(frozen=True, slots=True)
class NumericalProposerPromptV2(_CanonicalContract):
    schema_version: int
    template: str
    response_schema: Literal["numerical_mutation_batch_v1"]
    max_response_bytes: int
    allowed_mutation_operators: tuple[str, ...]
    parent_prompt_sha256: str | None

    def __post_init__(self):
        _schema_version(self.schema_version)
        _text(self.template, "template")
        if len(self.template.encode("utf-8")) > _MAX_TEMPLATE_BYTES:
            raise ValueError("template exceeds 65536 UTF-8 bytes")
        _require_choice(self.response_schema, "response_schema",
                        frozenset({"numerical_mutation_batch_v1"}))
        _nonnegative_int(self.max_response_bytes, "max_response_bytes")
        if not 1 <= self.max_response_bytes <= _MAX_RESPONSE_BYTES:
            raise ValueError("max_response_bytes must be in [1, 1048576]")
        object.__setattr__(self, "allowed_mutation_operators", _sorted_strings(
            self.allowed_mutation_operators, "allowed_mutation_operators",
            choices=MUTATION_OPERATORS, nonempty=True
        ))
        if self.parent_prompt_sha256 is not None:
            require_sha256(self.parent_prompt_sha256, "parent_prompt_sha256")


@dataclass(frozen=True, slots=True)
class NumericalQDEntryV2(_CanonicalContract):
    """One candidate's objectives on one cell's committed Train tasks."""

    schema_version: int
    genome_sha256: str
    evaluation_sha256: str
    cell: MorphologyCellV2
    task_ids: tuple[str, ...]
    objectives: NumericalObjectiveVectorV2
    constraints: ConstraintReportV2
    train_diagnostic_categories: tuple[str, ...]

    def __post_init__(self):
        _schema_version(self.schema_version)
        require_sha256(self.genome_sha256, "genome_sha256")
        require_sha256(self.evaluation_sha256, "evaluation_sha256")
        object.__setattr__(self, "cell", _nested(self.cell, MorphologyCellV2))
        object.__setattr__(self, "task_ids", _sorted_strings(
            self.task_ids, "task_ids", nonempty=True
        ))
        object.__setattr__(self, "objectives", _nested(self.objectives, NumericalObjectiveVectorV2))
        object.__setattr__(self, "constraints", _nested(self.constraints, ConstraintReportV2))
        object.__setattr__(self, "train_diagnostic_categories", _sorted_strings(
            self.train_diagnostic_categories, "train_diagnostic_categories",
            choices=TRAIN_DIAGNOSTIC_CATEGORIES
        ))


@dataclass(frozen=True, slots=True)
class NumericalEvaluationV2(_CanonicalContract):
    """Train-only evaluation; Dev comparison belongs to sealed Kernel evidence."""

    schema_version: int
    genome_sha256: str
    supply_sha256: str
    registry_sha256: str
    task_subset_sha256: str
    split_sha256: str
    metric_policy_sha256: str
    descriptor_policy_sha256: str
    execution_adapter_sha256: str
    runtime_fingerprints: Mapping[str, str]
    protocol_fingerprint: str
    split: Literal["train"]
    bracket: Literal["explore", "confirm", "replay"]
    rung: int
    task_ids: tuple[str, ...]
    task_statuses: Mapping[str, Literal["passed", "failed", "invalid"]]
    objectives: NumericalObjectiveVectorV2
    constraints: ConstraintReportV2
    cells: tuple[MorphologyCellV2, ...]
    train_diagnostic_categories: tuple[str, ...]
    cache_sha256s: tuple[str, ...]
    resource_use: Mapping[str, int | float]

    def __post_init__(self):
        _schema_version(self.schema_version)
        for name in ("genome_sha256", "supply_sha256", "registry_sha256",
                     "task_subset_sha256", "split_sha256", "metric_policy_sha256",
                     "descriptor_policy_sha256", "execution_adapter_sha256", "protocol_fingerprint"):
            require_sha256(getattr(self, name), name)
        object.__setattr__(self, "runtime_fingerprints", _runtime(self.runtime_fingerprints))
        _require_choice(self.split, "split", frozenset({"train"}))
        _require_choice(self.bracket, "bracket", frozenset(_BRACKET_RUNG_COUNTS))
        _nonnegative_int(self.rung, "rung")
        if self.rung >= _BRACKET_RUNG_COUNTS[self.bracket]:
            raise ValueError("rung is outside the committed bracket")
        tasks = _sorted_strings(self.task_ids, "task_ids", nonempty=True)
        object.__setattr__(self, "task_ids", tasks)
        statuses = _require_exact_schema(self.task_statuses, tasks, field="task_statuses")
        for task, status in statuses.items():
            _require_choice(status, f"task_statuses.{task}", frozenset({"passed", "failed", "invalid"}))
        object.__setattr__(self, "task_statuses", _freeze_json_value(statuses))
        object.__setattr__(self, "objectives", _nested(self.objectives, NumericalObjectiveVectorV2))
        object.__setattr__(self, "constraints", _nested(self.constraints, ConstraintReportV2))
        cells = tuple(_nested(cell, MorphologyCellV2) for cell in _sequence(self.cells, "cells"))
        _sorted_strings(tuple(cell.fingerprint() for cell in cells), "cells", sha=True, nonempty=True)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "train_diagnostic_categories", _sorted_strings(
            self.train_diagnostic_categories, "train_diagnostic_categories",
            choices=TRAIN_DIAGNOSTIC_CATEGORIES
        ))
        object.__setattr__(self, "cache_sha256s", _sorted_strings(
            self.cache_sha256s, "cache_sha256s", sha=True
        ))
        use = ResourceUse.from_payload(self.resource_use)
        object.__setattr__(self, "resource_use", _freeze_json_value(use.to_payload()))
