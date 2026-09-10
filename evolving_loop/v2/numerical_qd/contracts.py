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
class MutationStateV2(_CanonicalContract):
    """Closed transition aggregate; artifact authority remains with its components."""

    inventory: NumericalInventoryV2
    mutation_policy: NumericalMutationPolicyV2
    proposer_prompt: NumericalProposerPromptV2
    declared_cells: tuple[str, ...]
    max_parents_per_child: int
    max_inventory_size: int

    def __post_init__(self):
        for name, cls in (("inventory", NumericalInventoryV2),
                          ("mutation_policy", NumericalMutationPolicyV2),
                          ("proposer_prompt", NumericalProposerPromptV2)):
            object.__setattr__(self, name, _nested(getattr(self, name), cls))
        object.__setattr__(self, "declared_cells", _sorted_strings(
            self.declared_cells, "declared_cells", sha=True, nonempty=True
        ))
        for name in ("max_parents_per_child", "max_inventory_size"):
            if _nonnegative_int(getattr(self, name), name) == 0:
                raise ValueError(f"{name} must be positive")
        if len(self.inventory.members) > self.max_inventory_size:
            raise ValueError("inventory exceeds capacity")
        for member in self.inventory.members:
            if len(member.parent_ids) > self.max_parents_per_child:
                raise ValueError("member exceeds parent limit")
            if not set(member.applicability_cells) <= set(self.declared_cells):
                raise ValueError("member uses undeclared cell")


@dataclass(frozen=True, slots=True)
class TrainMutationFeedbackV2(_CanonicalContract):
    """Only Host-observed Train outcomes can feed proposal credit."""

    split: Literal["train"]
    operator: str
    feasible: bool
    promoted: bool
    inserted: bool
    diagnostic_categories: tuple[str, ...]

    def __post_init__(self):
        _require_choice(self.split, "split", frozenset({"train"}))
        _require_choice(self.operator, "operator", MUTATION_OPERATORS)
        for name in ("feasible", "promoted", "inserted"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if (self.promoted or self.inserted) and not self.feasible:
            raise ValueError("promotion/insertion requires Train feasibility")
        object.__setattr__(self, "diagnostic_categories", _sorted_strings(
            self.diagnostic_categories, "diagnostic_categories", choices=TRAIN_DIAGNOSTIC_CATEGORIES
        ))


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


_HYPERBAND_BRACKETS = {"explore": (8, 32, 80), "confirm": (32, 80), "replay": (80,)}


@dataclass(frozen=True, slots=True)
class HyperbandBracketV2(_CanonicalContract):
    name: str
    resources: tuple[int, ...]

    def __post_init__(self):
        _require_choice(self.name, "bracket name", frozenset(_HYPERBAND_BRACKETS))
        resources = _sequence(self.resources, "resources")
        if any(type(item) is not int for item in resources) or resources != _HYPERBAND_BRACKETS[self.name]:
            raise ValueError("resources must exactly match the registered bracket")
        object.__setattr__(self, "resources", resources)

    @classmethod
    def registered(cls, name):
        _require_choice(name, "bracket name", frozenset(_HYPERBAND_BRACKETS))
        return cls(name, _HYPERBAND_BRACKETS[name])


@dataclass(frozen=True, slots=True)
class TrainTaskV2(_CanonicalContract):
    """Committed metadata only; task bytes remain behind the Host callback."""

    task_id: str
    entity_id: str
    task_sha256: str
    split: Literal["train"]
    split_sha256: str
    protocol_sha256: str

    def __post_init__(self):
        _text(self.task_id, "task_id")
        _text(self.entity_id, "entity_id")
        _require_choice(self.split, "split", frozenset({"train"}))
        for name in ("task_sha256", "split_sha256", "protocol_sha256"):
            require_sha256(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class RungManifestV2(_CanonicalContract):
    """A fixed prefix of whole entities in a committed Train universe.

    The complete universe is retained so deserialization can verify grouping,
    nesting, and task-byte identity without opening Train data. An exact boundary
    inside an entity is rejected, never rounded or silently underfilled.
    """

    resource: int
    split_sha256: str
    protocol_sha256: str
    task_groups: Mapping[str, tuple[TrainTaskV2, ...]]

    def __post_init__(self):
        if type(self.resource) is not int or self.resource not in (8, 32, 80):
            raise ValueError("resource must be a registered Train resource")
        require_sha256(self.split_sha256, "split_sha256")
        require_sha256(self.protocol_sha256, "protocol_sha256")
        groups = self.task_groups
        if not isinstance(groups, Mapping) or any(type(key) is not str for key in groups):
            raise ValueError("task_groups must be an entity mapping")
        normalized = {}
        identities = set()
        boundaries = set()
        for entity, values in sorted(groups.items()):
            _text(entity, "entity_id")
            tasks = tuple(_nested(value, TrainTaskV2) for value in _sequence(values, "entity tasks"))
            if not tasks:
                raise ValueError("entity groups must not be empty")
            for task in tasks:
                if task.entity_id != entity:
                    raise ValueError("task entity does not match its group")
                if (task.split_sha256, task.protocol_sha256) != (self.split_sha256, self.protocol_sha256):
                    raise ValueError("task split/protocol does not match manifest")
                if task.task_id in identities:
                    raise ValueError("task IDs must be unique across the Train universe")
                identities.add(task.task_id)
            normalized[entity] = tuple(sorted(tasks, key=lambda item: item.task_id))
            boundaries.add(len(identities))
        if self.resource > len(identities):
            raise ValueError("resource exceeds the committed Train universe")
        if self.resource not in boundaries:
            raise ValueError("resource would split a partial entity group")
        object.__setattr__(self, "task_groups", MappingProxyType(normalized))

    @property
    def tasks(self):
        return tuple(task for tasks in self.task_groups.values() for task in tasks)[:self.resource]

    @property
    def task_ids(self):
        return tuple(sorted(task.task_id for task in self.tasks))


@dataclass(frozen=True, slots=True)
class HyperbandBudgetOutcomeV2(_CanonicalContract):
    """Closed work accounting plus the caller-owned ledger checkpoint identity."""

    status: Literal["completed", "blocked", "failed"]
    reason: str | None
    reservation_sha256: str | None
    resource_use: Mapping[str, int | float]
    ledger_checkpoint_sha256: str

    def __post_init__(self):
        _require_choice(self.status, "budget status", frozenset({"completed", "blocked", "failed"}))
        if self.status == "completed":
            if self.reason is not None:
                raise ValueError("completed budget outcome must not have a failure reason")
        else:
            _text(self.reason, "budget reason")
        if self.status != "blocked" or self.reservation_sha256 is not None:
            require_sha256(self.reservation_sha256, "reservation_sha256")
        require_sha256(self.ledger_checkpoint_sha256, "ledger_checkpoint_sha256")
        use = ResourceUse.from_payload(self.resource_use)
        if self.status == "blocked" and use != ResourceUse():
            raise ValueError("blocked work cannot carry a charge")
        object.__setattr__(self, "resource_use", _freeze_json_value(use.to_payload()))


def _hyperband_survivor_count(bracket, initial_count, index, count, reduction):
    if index == len(bracket.resources) - 1:
        return 1
    if initial_count == 3 and bracket.name == "explore":
        return (2, 1)[index]
    return max(1, count // reduction)


@dataclass(frozen=True, slots=True)
class HyperbandRungV2(_CanonicalContract):
    index: int
    manifest: RungManifestV2
    evaluations: tuple[NumericalEvaluationV2, ...]
    survivor_sha256s: tuple[str, ...]
    budget_outcome: HyperbandBudgetOutcomeV2

    def __post_init__(self):
        _nonnegative_int(self.index, "rung index")
        object.__setattr__(self, "manifest", _nested(self.manifest, RungManifestV2))
        evaluations = tuple(_nested(value, NumericalEvaluationV2)
                            for value in _sequence(self.evaluations, "evaluations"))
        if not evaluations or len({value.genome_sha256 for value in evaluations}) != len(evaluations):
            raise ValueError("completed rung requires unique candidate evaluations")
        for value in evaluations:
            if (value.rung != self.index or value.task_ids != self.manifest.task_ids
                    or value.task_subset_sha256 != self.manifest.fingerprint()
                    or value.split_sha256 != self.manifest.split_sha256
                    or value.protocol_fingerprint != self.manifest.protocol_sha256):
                raise ValueError("evaluation does not match the completed rung manifest")
        object.__setattr__(self, "evaluations", tuple(sorted(evaluations, key=lambda value: value.genome_sha256)))
        object.__setattr__(self, "survivor_sha256s", _sorted_strings(
            self.survivor_sha256s, "survivor_sha256s", sha=True, nonempty=True
        ))
        outcome = _nested(self.budget_outcome, HyperbandBudgetOutcomeV2)
        if outcome.status != "completed":
            raise ValueError("only completed closed rungs may be checkpointed")
        object.__setattr__(self, "budget_outcome", outcome)


@dataclass(frozen=True, slots=True)
class HyperbandStateV2(_CanonicalContract):
    bracket: HyperbandBracketV2
    candidate_sha256s: tuple[str, ...]
    reduction_factor: int
    split_sha256: str
    protocol_sha256: str
    rungs: tuple[HyperbandRungV2, ...]

    def __post_init__(self):
        object.__setattr__(self, "bracket", _nested(self.bracket, HyperbandBracketV2))
        candidates = _sorted_strings(self.candidate_sha256s, "candidate_sha256s", sha=True, nonempty=True)
        object.__setattr__(self, "candidate_sha256s", candidates)
        if _nonnegative_int(self.reduction_factor, "reduction_factor") == 0:
            raise ValueError("reduction_factor must be positive")
        require_sha256(self.split_sha256, "split_sha256")
        require_sha256(self.protocol_sha256, "protocol_sha256")
        rungs = tuple(_nested(value, HyperbandRungV2) for value in _sequence(self.rungs, "rungs"))
        if len(rungs) > len(self.bracket.resources):
            raise ValueError("too many rungs for bracket")
        for index, rung in enumerate(rungs):
            if rung.index != index or rung.manifest.resource != self.bracket.resources[index]:
                raise ValueError("rungs must form a completed contiguous bracket prefix")
            if (rung.manifest.split_sha256, rung.manifest.protocol_sha256) != (self.split_sha256, self.protocol_sha256):
                raise ValueError("rung split/protocol does not match state")
            if index and rung.manifest.task_groups != rungs[0].manifest.task_groups:
                raise ValueError("rungs must share the same committed Train universe")
            if tuple(value.genome_sha256 for value in rung.evaluations) != candidates:
                raise ValueError("partial rung or mismatched candidate evaluations")
            if any(value.bracket != self.bracket.name for value in rung.evaluations):
                raise ValueError("evaluation bracket does not match state")
            expected = _hyperband_survivor_count(self.bracket, len(self.candidate_sha256s), index,
                                                len(candidates), self.reduction_factor)
            if len(rung.survivor_sha256s) != expected or not set(rung.survivor_sha256s) <= set(candidates):
                raise ValueError("rung survivors do not match the promotion schedule")
            candidates = rung.survivor_sha256s
        object.__setattr__(self, "rungs", rungs)

    @property
    def active_candidates(self):
        return self.rungs[-1].survivor_sha256s if self.rungs else self.candidate_sha256s

    @property
    def complete(self):
        return len(self.rungs) == len(self.bracket.resources)


@dataclass(frozen=True, slots=True)
class HyperbandAdvanceV2(_CanonicalContract):
    state: HyperbandStateV2
    evaluations: tuple[NumericalEvaluationV2, ...]

    def __post_init__(self):
        object.__setattr__(self, "state", _nested(self.state, HyperbandStateV2))
        object.__setattr__(self, "evaluations", tuple(_nested(value, NumericalEvaluationV2)
                           for value in _sequence(self.evaluations, "evaluations")))


def _cache_identity(candidate, task, split, metric, descriptor, runtime, protocol, adapter):
    values = dict(candidate=candidate, task=task, split=split, metric=metric,
                  descriptor=descriptor, protocol=protocol, adapter=adapter)
    for name, value in values.items():
        require_sha256(value, name)
    values["runtime"] = (require_sha256(runtime, "runtime") if type(runtime) is str
                         else fingerprint_payload(dict(_runtime(runtime))))
    return fingerprint_payload(values)


@dataclass(frozen=True, slots=True)
class TaskCacheRowV2(_CanonicalContract):
    """An exact successful single-task evaluation, reusable across rungs."""

    cache_key: str
    task_sha256: str
    evaluation: NumericalEvaluationV2

    def __post_init__(self):
        require_sha256(self.cache_key, "cache_key")
        require_sha256(self.task_sha256, "task_sha256")
        value = _nested(self.evaluation, NumericalEvaluationV2)
        if len(value.task_ids) != 1 or tuple(value.task_statuses.values()) != ("passed",) or not value.constraints.feasible:
            raise ValueError("cache rows require one successful feasible task")
        key = _cache_identity(value.genome_sha256, self.task_sha256, value.split_sha256,
                              value.metric_policy_sha256, value.descriptor_policy_sha256,
                              value.runtime_fingerprints, value.protocol_fingerprint,
                              value.execution_adapter_sha256)
        if key != self.cache_key:
            raise ValueError("cache identity mismatch")
        object.__setattr__(self, "evaluation", value)


@dataclass(frozen=True, slots=True)
class HyperbandTaskResultV2(_CanonicalContract):
    candidate_sha256: str
    task_id: str
    cache_key: str
    status: Literal["passed", "failed", "invalid"]
    evaluation: NumericalEvaluationV2 | None
    cache_hit: bool
    failure_category: str | None

    def __post_init__(self):
        require_sha256(self.candidate_sha256, "candidate_sha256")
        require_sha256(self.cache_key, "cache_key")
        _text(self.task_id, "task_id")
        _require_choice(self.status, "task status", frozenset({"passed", "failed", "invalid"}))
        if type(self.cache_hit) is not bool:
            raise ValueError("cache_hit must be boolean")
        if self.evaluation is None:
            if self.status != "invalid" or self.cache_hit:
                raise ValueError("missing evaluation must be invalid and uncached")
            _require_choice(self.failure_category, "failure_category", TRAIN_DIAGNOSTIC_CATEGORIES)
        else:
            value = _nested(self.evaluation, NumericalEvaluationV2)
            if (value.genome_sha256 != self.candidate_sha256 or value.task_ids != (self.task_id,)
                    or value.task_statuses[self.task_id] != self.status):
                raise ValueError("task result does not match its evaluation")
            if self.failure_category is not None or (self.cache_hit and (self.status != "passed" or not value.constraints.feasible)):
                raise ValueError("invalid cache hit or task failure category")
            object.__setattr__(self, "evaluation", value)


@dataclass(frozen=True, slots=True)
class HyperbandExecutionV2(_CanonicalContract):
    task_results: tuple[HyperbandTaskResultV2, ...]
    cache_rows: tuple[TaskCacheRowV2, ...]
    budget_outcome: HyperbandBudgetOutcomeV2

    def __post_init__(self):
        for name, cls in (("task_results", HyperbandTaskResultV2), ("cache_rows", TaskCacheRowV2)):
            object.__setattr__(self, name, tuple(_nested(value, cls)
                               for value in _sequence(getattr(self, name), name)))
        object.__setattr__(self, "budget_outcome", _nested(self.budget_outcome, HyperbandBudgetOutcomeV2))
