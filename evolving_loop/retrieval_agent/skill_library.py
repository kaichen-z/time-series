"""Versioned, declarative retrieval skills and their auditable lifecycle."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Mapping


SKILL_LIBRARY_SCHEMA_VERSION = 1
SkillStage = Literal["round1", "round2", "both"]
SkillStatus = Literal["candidate", "accepted", "specialized", "quarantined"]
SkillOperationKind = Literal["add", "repair", "specialize", "merge", "quarantine"]


class RetrievalSkillError(ValueError):
    """Raised when a skill row or lifecycle transition is unsafe or malformed."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not (cleaned := value.strip()):
        raise RetrievalSkillError(f"{field} must be a non-empty string")
    return cleaned


def _text_tuple(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise RetrievalSkillError(f"{field} must be a list of strings")
    result = tuple(_text(item, field) for item in value)
    if len(set(result)) != len(result):
        raise RetrievalSkillError(f"{field} must not contain duplicates")
    return result


@dataclass(frozen=True)
class RetrievalApplicability:
    """Typed constraints used to project a skill into a stage prompt."""

    assumption_kinds: tuple[str, ...] = ()
    gap_types: tuple[str, ...] = ()
    temporal_relations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("assumption_kinds", "gap_types", "temporal_relations"):
            object.__setattr__(self, field, _text_tuple(getattr(self, field), field))

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "RetrievalApplicability":
        expected = {"assumption_kinds", "gap_types", "temporal_relations"}
        if set(raw) != expected:
            raise RetrievalSkillError("applicability must contain exactly its typed selectors")
        return cls(**{field: _text_tuple(raw[field], field) for field in expected})

    def to_payload(self) -> dict[str, list[str]]:
        return {
            "assumption_kinds": list(self.assumption_kinds),
            "gap_types": list(self.gap_types),
            "temporal_relations": list(self.temporal_relations),
        }

    def matches(
        self,
        *,
        assumption_kinds: Iterable[str] = (),
        gap_types: Iterable[str] = (),
        temporal_relations: Iterable[str] = (),
    ) -> bool:
        for required, available in (
            (self.assumption_kinds, tuple(assumption_kinds)),
            (self.gap_types, tuple(gap_types)),
            (self.temporal_relations, tuple(temporal_relations)),
        ):
            if required and (
                not available or not set(required).intersection(available)
            ):
                return False
        return True

    def is_narrower_than(self, parent: "RetrievalApplicability") -> bool:
        """Return whether every added constraint is no broader than ``parent``."""
        strict = False
        for child, previous in (
            (self.assumption_kinds, parent.assumption_kinds),
            (self.gap_types, parent.gap_types),
            (self.temporal_relations, parent.temporal_relations),
        ):
            if previous:
                if not child or not set(child).issubset(previous):
                    return False
                strict = strict or set(child) != set(previous)
            elif child:
                strict = True
        return strict


@dataclass(frozen=True, init=False)
class RetrievalSkill:
    """A versioned, host-interpreted retrieval strategy with no executable content.

    The custom initializer accepts the historical flat constructor as a migration
    boundary.  New callers must use the typed keyword fields documented below.
    """

    skill_id: str
    version: int
    parent_version: int | None
    stage: SkillStage
    status: SkillStatus
    name: str
    description: str
    applicability: RetrievalApplicability
    query_steps: tuple[str, ...]
    required_chain_fields: tuple[str, ...]
    counterevidence_rule: str
    failure_conditions: tuple[str, ...]
    validated_task_ids: tuple[str, ...]
    validated_entities: tuple[str, ...]
    validation_smae_gain: float | None
    validation_srmse_gain: float | None
    merged_from_skill_ids: tuple[str, ...]
    quarantine_reason: str | None

    _LEGACY_FIELDS = (
        "name",
        "description",
        "applicability",
        "query_strategy",
        "verification_rule",
        "created_from_task",
        "validation_score",
    )

    def __init__(self, skill_id: str, *args: object, **values: object) -> None:
        legacy_mode = bool(args and not isinstance(args[0], int)) or any(
            field in values
            for field in (
                "query_strategy",
                "verification_rule",
                "created_from_task",
                "validation_smae",
                "validation_srmse",
                "validation_score",
                "uses",
                "avg_smae",
                "avg_srmse",
                "avg_score",
            )
        ) or isinstance(values.get("applicability"), str)
        if legacy_mode:
            values = self._legacy_values(skill_id, args, values)
        elif args:
            field_names = (
                "version", "parent_version", "stage", "status", "name", "description",
                "applicability", "query_steps", "required_chain_fields",
                "counterevidence_rule", "failure_conditions", "validated_task_ids",
                "validated_entities", "validation_smae_gain", "validation_srmse_gain",
            )
            if len(args) > len(field_names):
                raise TypeError("too many positional RetrievalSkill arguments")
            values = {**dict(zip(field_names, args)), **values}
        self._set_typed(skill_id, values)

    @classmethod
    def _legacy_values(
        cls, skill_id: str, args: tuple[object, ...], supplied: Mapping[str, object]
    ) -> dict[str, object]:
        if len(args) > len(cls._LEGACY_FIELDS):
            raise TypeError("too many legacy RetrievalSkill arguments")
        raw = {**dict(zip(cls._LEGACY_FIELDS, args)), **dict(supplied)}
        applicability = raw.get("applicability", "")
        query_strategy = raw.get("query_strategy", "")
        verification_rule = raw.get("verification_rule", "")
        created_from_task = raw.get("created_from_task", "")
        smae = raw.get("validation_smae", raw.get("validation_smae_gain"))
        srmse = raw.get("validation_srmse", raw.get("validation_srmse_gain"))
        return {
            "version": raw.get("version", 1),
            "parent_version": raw.get("parent_version"),
            "stage": raw.get("stage", "both"),
            "status": raw.get(
                "status", "accepted" if smae is not None and srmse is not None else "candidate"
            ),
            "name": raw.get("name", skill_id),
            "description": raw.get("description", "legacy retrieval skill"),
            "applicability": RetrievalApplicability(),
            "query_steps": (query_strategy,) if isinstance(query_strategy, str) and query_strategy.strip() else (),
            "required_chain_fields": (),
            "counterevidence_rule": verification_rule if isinstance(verification_rule, str) and verification_rule.strip() else "Search for counterevidence.",
            "failure_conditions": (),
            "validated_task_ids": (created_from_task,) if isinstance(created_from_task, str) and created_from_task.strip() else (),
            "validated_entities": (),
            "validation_smae_gain": smae,
            "validation_srmse_gain": srmse,
            "merged_from_skill_ids": raw.get("merged_from_skill_ids", ()),
            "quarantine_reason": raw.get("quarantine_reason"),
        }

    def _set_typed(self, skill_id: object, values: Mapping[str, object]) -> None:
        allowed = {
            "version", "parent_version", "stage", "status", "name", "description",
            "applicability", "query_steps", "required_chain_fields", "counterevidence_rule",
            "failure_conditions", "validated_task_ids", "validated_entities",
            "validation_smae_gain", "validation_srmse_gain", "merged_from_skill_ids",
            "quarantine_reason",
        }
        unknown = set(values).difference(allowed)
        missing = {
            "version", "parent_version", "stage", "status", "name", "description",
            "applicability", "query_steps", "required_chain_fields", "counterevidence_rule",
            "failure_conditions",
        }.difference(values)
        if unknown or missing:
            raise RetrievalSkillError(
                f"invalid typed RetrievalSkill fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        typed_id = _text(skill_id, "skill_id")
        if not all(character.isalnum() or character in "_-" for character in typed_id):
            raise RetrievalSkillError("skill_id must be a stable identifier")
        version = values["version"]
        parent_version = values["parent_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise RetrievalSkillError("version must be a positive integer")
        if parent_version is not None and (
            isinstance(parent_version, bool)
            or not isinstance(parent_version, int)
            or not 1 <= parent_version < version
        ):
            raise RetrievalSkillError("parent_version must name an earlier version")
        stage = values["stage"]
        status = values["status"]
        if stage not in {"round1", "round2", "both"}:
            raise RetrievalSkillError("invalid skill stage")
        if status not in {"candidate", "accepted", "specialized", "quarantined"}:
            raise RetrievalSkillError("invalid skill status")
        applicability = values["applicability"]
        if isinstance(applicability, Mapping):
            applicability = RetrievalApplicability.from_payload(applicability)
        if not isinstance(applicability, RetrievalApplicability):
            raise RetrievalSkillError("applicability must be a RetrievalApplicability")
        smae = self._metric(values.get("validation_smae_gain"), "validation_smae_gain")
        srmse = self._metric(values.get("validation_srmse_gain"), "validation_srmse_gain")
        if (smae is None) != (srmse is None):
            raise RetrievalSkillError("validation gains must be supplied together")
        if status in {"accepted", "specialized"} and (smae is None or srmse is None):
            raise RetrievalSkillError("active skills require both validation gains")
        reason = values.get("quarantine_reason")
        if reason is not None:
            reason = _text(reason, "quarantine_reason")
        if status == "quarantined" and not reason:
            raise RetrievalSkillError("quarantined skills require a reason")
        if status != "quarantined" and reason is not None:
            raise RetrievalSkillError("only quarantined skills may carry a reason")
        typed = {
            "skill_id": typed_id,
            "version": version,
            "parent_version": parent_version,
            "stage": stage,
            "status": status,
            "name": _text(values["name"], "name"),
            "description": _text(values["description"], "description"),
            "applicability": applicability,
            "query_steps": _text_tuple(values["query_steps"], "query_steps"),
            "required_chain_fields": _text_tuple(values["required_chain_fields"], "required_chain_fields"),
            "counterevidence_rule": _text(values["counterevidence_rule"], "counterevidence_rule"),
            "failure_conditions": _text_tuple(values["failure_conditions"], "failure_conditions"),
            "validated_task_ids": _text_tuple(values.get("validated_task_ids", ()), "validated_task_ids"),
            "validated_entities": _text_tuple(values.get("validated_entities", ()), "validated_entities"),
            "validation_smae_gain": smae,
            "validation_srmse_gain": srmse,
            "merged_from_skill_ids": _text_tuple(values.get("merged_from_skill_ids", ()), "merged_from_skill_ids"),
            "quarantine_reason": reason,
        }
        for field, value in typed.items():
            object.__setattr__(self, field, value)

    @staticmethod
    def _metric(value: object, field: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RetrievalSkillError(f"{field} must be a finite number or null")
        return float(value)

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "RetrievalSkill":
        required = {
            "skill_id", "version", "parent_version", "stage", "status", "name", "description",
            "applicability", "query_steps", "required_chain_fields", "counterevidence_rule",
            "failure_conditions",
        }
        optional = {
            "validated_task_ids", "validated_entities", "validation_smae_gain",
            "validation_srmse_gain", "merged_from_skill_ids", "quarantine_reason",
        }
        if not required.issubset(raw) or set(raw).difference(required | optional):
            raise RetrievalSkillError("typed RetrievalSkill rows must contain only schema fields")
        values = {
            "validated_task_ids": (),
            "validated_entities": (),
            "validation_smae_gain": None,
            "validation_srmse_gain": None,
            "merged_from_skill_ids": (),
            "quarantine_reason": None,
            **dict(raw),
        }
        skill_id = values.pop("skill_id")
        return cls(skill_id, **values)

    def to_payload(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "parent_version": self.parent_version,
            "stage": self.stage,
            "status": self.status,
            "name": self.name,
            "description": self.description,
            "applicability": self.applicability.to_payload(),
            "query_steps": list(self.query_steps),
            "required_chain_fields": list(self.required_chain_fields),
            "counterevidence_rule": self.counterevidence_rule,
            "failure_conditions": list(self.failure_conditions),
            "validated_task_ids": list(self.validated_task_ids),
            "validated_entities": list(self.validated_entities),
            "validation_smae_gain": self.validation_smae_gain,
            "validation_srmse_gain": self.validation_srmse_gain,
            "merged_from_skill_ids": list(self.merged_from_skill_ids),
            "quarantine_reason": self.quarantine_reason,
        }

    @property
    def is_active(self) -> bool:
        return self.status in {"accepted", "specialized"}


@dataclass(frozen=True)
class RetrievalSkillOperation:
    kind: SkillOperationKind
    skill_id: str | None = None
    skill: RetrievalSkill | None = None
    reason: str | None = None
    source_skill_ids: tuple[str, ...] = ()

    @classmethod
    def add(cls, skill: RetrievalSkill) -> "RetrievalSkillOperation":
        return cls("add", skill=skill)

    @classmethod
    def repair(cls, skill_id: str, skill: RetrievalSkill) -> "RetrievalSkillOperation":
        return cls("repair", skill_id=skill_id, skill=skill)

    @classmethod
    def specialize(cls, skill_id: str, skill: RetrievalSkill) -> "RetrievalSkillOperation":
        return cls("specialize", skill_id=skill_id, skill=skill)

    @classmethod
    def merge(
        cls, source_skill_ids: Iterable[str], skill: RetrievalSkill
    ) -> "RetrievalSkillOperation":
        return cls("merge", skill=skill, source_skill_ids=tuple(source_skill_ids))

    @classmethod
    def quarantine(cls, skill_id: str, reason: str) -> "RetrievalSkillOperation":
        return cls("quarantine", skill_id=skill_id, reason=reason)


class RetrievalSkillLibrary:
    """An append-only skill history whose mutations commit atomically."""

    def __init__(
        self,
        path: str | Path,
        skills: Iterable[RetrievalSkill] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        self.path = Path(path)
        self.persist = persist
        self._skills = self._validated_index(skills or ())

    @classmethod
    def load(cls, path: str | Path, *, persist: bool = True) -> "RetrievalSkillLibrary":
        source = Path(path)
        if not source.exists():
            return cls(source, persist=persist)
        raw = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            skills = [cls._load_legacy_or_typed(record) for record in raw]
        elif isinstance(raw, dict) and set(raw) == {"schema_version", "skills"}:
            if raw["schema_version"] != SKILL_LIBRARY_SCHEMA_VERSION or not isinstance(raw["skills"], list):
                raise RetrievalSkillError("unsupported retrieval skill library schema")
            skills = [RetrievalSkill.from_payload(record) for record in raw["skills"] if isinstance(record, dict)]
            if len(skills) != len(raw["skills"]):
                raise RetrievalSkillError("skill rows must be objects")
        else:
            raise RetrievalSkillError("invalid retrieval skill library payload")
        return cls(source, skills, persist=persist)

    @staticmethod
    def _load_legacy_or_typed(record: object) -> RetrievalSkill:
        if not isinstance(record, dict):
            raise RetrievalSkillError("skill rows must be objects")
        if "version" in record and "query_steps" in record:
            return RetrievalSkill.from_payload(record)
        return RetrievalSkill(**record)

    @staticmethod
    def _validated_index(skills: Iterable[RetrievalSkill]) -> dict[str, tuple[RetrievalSkill, ...]]:
        grouped: dict[str, list[RetrievalSkill]] = {}
        for skill in skills:
            if not isinstance(skill, RetrievalSkill):
                raise RetrievalSkillError("library entries must be RetrievalSkill records")
            grouped.setdefault(skill.skill_id, []).append(skill)
        indexed: dict[str, tuple[RetrievalSkill, ...]] = {}
        for skill_id, history in grouped.items():
            ordered = tuple(sorted(history, key=lambda item: item.version))
            versions = tuple(item.version for item in ordered)
            if versions != tuple(range(1, len(ordered) + 1)):
                raise RetrievalSkillError(f"{skill_id} versions must be contiguous from one")
            for index, skill in enumerate(ordered):
                if index == 0:
                    if skill.parent_version is not None:
                        raise RetrievalSkillError(f"{skill_id} version one cannot have a parent")
                elif skill.parent_version != ordered[index - 1].version:
                    raise RetrievalSkillError(f"{skill_id} lineage must reference the previous version")
                if skill.name != ordered[0].name:
                    raise RetrievalSkillError(f"{skill_id} name must remain stable across versions")
            indexed[skill_id] = ordered
        return indexed

    def save(self) -> None:
        if self.persist:
            self._write(self._skills)

    def _write(self, proposed: Mapping[str, tuple[RetrievalSkill, ...]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "schema_version": SKILL_LIBRARY_SCHEMA_VERSION,
            "skills": [skill.to_payload() for skill in self._all_from(proposed)],
        }
        try:
            temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _all_from(
        skills: Mapping[str, tuple[RetrievalSkill, ...]]
    ) -> tuple[RetrievalSkill, ...]:
        return tuple(
            item
            for skill_id in sorted(skills)
            for item in skills[skill_id]
        )

    def apply_operations(self, operations: Iterable[RetrievalSkillOperation]) -> None:
        proposed = {skill_id: tuple(history) for skill_id, history in self._skills.items()}
        for operation in operations:
            if not isinstance(operation, RetrievalSkillOperation):
                raise RetrievalSkillError("invalid retrieval skill operation")
            proposed = self._apply_one(proposed, operation)
            proposed = self._validated_index(self._all_from(proposed))
        if self.persist:
            self._write(proposed)
        self._skills = proposed

    def _apply_one(
        self,
        proposed: dict[str, tuple[RetrievalSkill, ...]],
        operation: RetrievalSkillOperation,
    ) -> dict[str, tuple[RetrievalSkill, ...]]:
        if operation.kind == "add":
            skill = self._operation_skill(operation)
            if skill.skill_id in proposed:
                raise RetrievalSkillError(f"skill already exists: {skill.skill_id}")
            if skill.version != 1 or skill.parent_version is not None or skill.status != "candidate":
                raise RetrievalSkillError("add may create only a version-one candidate skill")
            return {**proposed, skill.skill_id: (skill,)}
        if operation.kind == "repair":
            current = self._current(proposed, operation.skill_id)
            replacement = self._next_version(current, self._operation_skill(operation))
            return {**proposed, current.skill_id: (*proposed[current.skill_id], replacement)}
        if operation.kind == "specialize":
            current = self._current(proposed, operation.skill_id)
            replacement = self._next_version(current, self._operation_skill(operation))
            if replacement.status != "specialized":
                raise RetrievalSkillError("specialize must produce a specialized skill")
            stage_narrowed = current.stage == "both" and replacement.stage != "both"
            if not stage_narrowed and not replacement.applicability.is_narrower_than(current.applicability):
                raise RetrievalSkillError("specialize must narrow applicability or stage")
            return {**proposed, current.skill_id: (*proposed[current.skill_id], replacement)}
        if operation.kind == "quarantine":
            current = self._current(proposed, operation.skill_id)
            if current.status == "quarantined":
                raise RetrievalSkillError(f"skill is already quarantined: {current.skill_id}")
            reason = _text(operation.reason, "quarantine reason")
            quarantined = replace(
                current,
                version=current.version + 1,
                parent_version=current.version,
                status="quarantined",
                quarantine_reason=reason,
            )
            return {**proposed, current.skill_id: (*proposed[current.skill_id], quarantined)}
        if operation.kind == "merge":
            successor = self._operation_skill(operation)
            sources = tuple(sorted({_text(value, "merge source skill id") for value in operation.source_skill_ids}))
            if len(sources) < 2:
                raise RetrievalSkillError("merge requires at least two source skills")
            if successor.skill_id in proposed:
                raise RetrievalSkillError("merge successor must use a new stable identity")
            source_records = tuple(self._current(proposed, source) for source in sources)
            if any(skill.status == "quarantined" for skill in source_records):
                raise RetrievalSkillError("merge sources must be active or candidates")
            if successor.version != 1 or successor.parent_version is not None or successor.status != "candidate":
                raise RetrievalSkillError("merge must create a version-one candidate successor")
            successor = replace(successor, merged_from_skill_ids=sources)
            result = {**proposed, successor.skill_id: (successor,)}
            for source in sources:
                current = result[source][-1]
                quarantined = replace(
                    current,
                    version=current.version + 1,
                    parent_version=current.version,
                    status="quarantined",
                    quarantine_reason=f"merged into {successor.skill_id}",
                )
                result[source] = (*result[source], quarantined)
            return result
        raise RetrievalSkillError(f"unknown retrieval skill operation: {operation.kind}")

    @staticmethod
    def _operation_skill(operation: RetrievalSkillOperation) -> RetrievalSkill:
        if operation.skill is None:
            raise RetrievalSkillError(f"{operation.kind} requires a skill")
        return operation.skill

    @staticmethod
    def _current(
        proposed: Mapping[str, tuple[RetrievalSkill, ...]], skill_id: str | None
    ) -> RetrievalSkill:
        key = _text(skill_id, "skill_id")
        if key not in proposed:
            raise RetrievalSkillError(f"unknown retrieval skill: {key}")
        return proposed[key][-1]

    @staticmethod
    def _next_version(current: RetrievalSkill, replacement: RetrievalSkill) -> RetrievalSkill:
        if replacement.skill_id != current.skill_id:
            raise RetrievalSkillError("repair and specialize must preserve skill_id")
        if replacement.name != current.name:
            raise RetrievalSkillError("repair and specialize must preserve the compatibility name")
        if replacement.quarantine_reason is not None and replacement.status != "quarantined":
            raise RetrievalSkillError("replacement may not carry a stale quarantine reason")
        return replace(
            replacement,
            version=current.version + 1,
            parent_version=current.version,
            merged_from_skill_ids=current.merged_from_skill_ids,
            quarantine_reason=(
                replacement.quarantine_reason if replacement.status == "quarantined" else None
            ),
        )

    def get_by_id(self, skill_id: str) -> RetrievalSkill | None:
        history = self._skills.get(skill_id)
        return history[-1] if history else None

    def history(self, skill_id: str) -> tuple[RetrievalSkill, ...]:
        return self._skills.get(skill_id, ())

    def get(self, name: str) -> RetrievalSkill | None:
        matches = [history[-1] for history in self._skills.values() if history[-1].name == name]
        return max(matches, key=lambda skill: (skill.version, skill.skill_id), default=None)

    def all(self) -> tuple[RetrievalSkill, ...]:
        return self._all_from(self._skills)

    def for_stage(
        self,
        stage: Literal["round1", "round2"],
        *,
        assumption_kinds: Iterable[str] = (),
        gap_types: Iterable[str] = (),
        temporal_relations: Iterable[str] = (),
    ) -> tuple[RetrievalSkill, ...]:
        if stage not in {"round1", "round2"}:
            raise RetrievalSkillError("prompt projection requires round1 or round2")
        return tuple(
            skill
            for skill in sorted((history[-1] for history in self._skills.values()), key=lambda item: item.skill_id)
            if skill.is_active
            and skill.stage in {stage, "both"}
            and skill.applicability.matches(
                assumption_kinds=assumption_kinds,
                gap_types=gap_types,
                temporal_relations=temporal_relations,
            )
        )

    def list_for_prompt(
        self,
        stage: Literal["round1", "round2"] = "round1",
        **applicability: Iterable[str],
    ) -> str:
        skills = self.for_stage(stage, **applicability)
        if not skills:
            return "(no active retrieval skills)"
        return "\n".join(
            "- {name} [{skill_id}@{version}]: {description}; query steps: {steps}; "
            "required chain fields: {fields}; counterevidence: {counterevidence}; failures: {failures}".format(
                name=skill.name,
                skill_id=skill.skill_id,
                version=skill.version,
                description=skill.description,
                steps=" | ".join(skill.query_steps) or "none",
                fields=", ".join(skill.required_chain_fields) or "none",
                counterevidence=skill.counterevidence_rule,
                failures=" | ".join(skill.failure_conditions) or "none",
            )
            for skill in skills
        )

    def add(self, skill: RetrievalSkill) -> None:
        """Compatibility wrapper: direct additions remain candidate-only."""
        if skill.skill_id in self._skills:
            self.apply_operations((RetrievalSkillOperation.repair(skill.skill_id, skill),))
        else:
            self.apply_operations((RetrievalSkillOperation.add(skill),))

    def record_use(self, name: str, smae: float, srmse: float) -> None:
        """Keep the legacy call site harmless; evaluator-owned credit is not skill state."""
        del name, smae, srmse

    def clone(self, *, persist: bool = False) -> "RetrievalSkillLibrary":
        return RetrievalSkillLibrary(self.path, self.all(), persist=persist)

    def __len__(self) -> int:
        return len(self._skills)
