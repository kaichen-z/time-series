"""Persistent, outcome-validated forecast selection rules."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class DecisionSkill:
    """A generalized rule for selecting among executed numerical hypotheses."""

    skill_id: str
    name: str
    description: str
    applicability: str
    decision_rule: str
    failure_condition: str
    created_from_task: str
    validation_smae: float | None = None
    validation_srmse: float | None = None
    validation_score: float | None = None
    uses: int = 0
    avg_smae: float | None = None
    avg_srmse: float | None = None
    avg_score: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "skill_id",
            "name",
            "description",
            "applicability",
            "decision_rule",
            "failure_condition",
            "created_from_task",
        ):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(f"Decision Skill {field_name} must be primitive text")
            if not value.strip():
                raise ValueError(f"Decision Skill {field_name} must be non-empty")
        if type(self.uses) is not int:
            raise TypeError("Decision Skill uses must be an integer")
        if self.uses < 0:
            raise ValueError("Decision Skill uses must be non-negative")
        for field_name in (
            "validation_smae",
            "validation_srmse",
            "validation_score",
            "avg_smae",
            "avg_srmse",
            "avg_score",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if type(value) not in {int, float}:
                raise TypeError(
                    f"Decision Skill {field_name} must be a primitive finite number"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Decision Skill {field_name} must be finite")
            object.__setattr__(self, field_name, numeric)

    def to_payload(self) -> dict[str, object]:
        """Return a primitive-only payload after revalidating the current frozen row."""
        self.__post_init__()
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "applicability": self.applicability,
            "decision_rule": self.decision_rule,
            "failure_condition": self.failure_condition,
            "created_from_task": self.created_from_task,
            "validation_smae": self.validation_smae,
            "validation_srmse": self.validation_srmse,
            "validation_score": self.validation_score,
            "uses": self.uses,
            "avg_smae": self.avg_smae,
            "avg_srmse": self.avg_srmse,
            "avg_score": self.avg_score,
        }


class DecisionSkillLibrary:
    def __init__(
        self,
        path: str | Path,
        skills: list[DecisionSkill] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        self.path = Path(path)
        self.persist = persist
        self._skills = {skill.name: skill for skill in (skills or [])}
        self._read_only = False

    @classmethod
    def load(cls, path: str | Path) -> "DecisionSkillLibrary":
        source = Path(path)
        if not source.exists():
            return cls(source)
        return cls(source, [DecisionSkill(**record) for record in json.loads(source.read_text())])

    def save(self) -> None:
        self._require_writable()
        if not self.persist:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([skill.to_payload() for skill in self._skills.values()], indent=2)
        )
        os.replace(temporary, self.path)

    def add(self, skill: DecisionSkill) -> None:
        self._require_writable()
        existing = self._skills.get(skill.name)
        if (
            existing is not None
            and existing.validation_srmse is None
            and skill.validation_srmse is None
        ):
            existing_signal = (
                existing.validation_score
                if existing.validation_score is not None
                else existing.validation_smae
            )
            incoming_signal = (
                skill.validation_score
                if skill.validation_score is not None
                else skill.validation_smae
            )
            if (
                existing_signal is not None
                and incoming_signal is not None
                and existing_signal > incoming_signal
            ):
                return
        if (
            existing is not None
            and existing.validation_smae is not None
            and existing.validation_srmse is not None
            and skill.validation_smae is not None
            and skill.validation_srmse is not None
            and not (
                skill.validation_smae >= existing.validation_smae
                and skill.validation_srmse >= existing.validation_srmse
                and (
                    skill.validation_smae > existing.validation_smae
                    or skill.validation_srmse > existing.validation_srmse
                )
            )
        ):
            return
        self._skills[skill.name] = skill
        self.save()

    def get(self, name: str) -> DecisionSkill | None:
        return self._skills.get(name)

    def all(self) -> tuple[DecisionSkill, ...]:
        return tuple(self._skills.values())

    def list_for_prompt(self) -> str:
        if not self._skills:
            return "(no validated decision skills yet)"
        return "\n".join(
            f"- {skill.name}: {skill.description}; applies when: {skill.applicability}; "
            f"rule: {skill.decision_rule}; fails when: {skill.failure_condition}; "
            f"sMAE signal={skill.validation_smae:.3f}; "
            f"sRMSE signal={skill.validation_srmse:.3f}"
            for skill in self._skills.values()
            if skill.validation_smae is not None and skill.validation_srmse is not None
        )

    def record_use(self, name: str, smae: float, srmse: float) -> None:
        self._require_writable()
        skill = self._skills.get(name)
        if skill is None:
            return
        uses = skill.uses + 1
        avg_smae = smae if skill.avg_smae is None else (skill.avg_smae * skill.uses + smae) / uses
        avg_srmse = srmse if skill.avg_srmse is None else (skill.avg_srmse * skill.uses + srmse) / uses
        self._skills[name] = replace(
            skill, uses=uses, avg_smae=avg_smae, avg_srmse=avg_srmse
        )
        self.save()

    def clone(self, *, persist: bool = False) -> "DecisionSkillLibrary":
        return DecisionSkillLibrary(self.path, list(self._skills.values()), persist=persist)

    @classmethod
    def frozen_execution_snapshot(
        cls,
        source: "DecisionSkillLibrary",
    ) -> "DecisionSkillLibrary":
        """Return a deep, non-persistent snapshot whose mutators fail closed."""
        if type(source) is not cls:
            raise TypeError("Decision Skill library must be canonical")
        source_rows = source.all()
        if any(type(item) is not DecisionSkill for item in source_rows):
            raise TypeError("Decision Skill rows must be canonical")
        rows = tuple(DecisionSkill(**item.to_payload()) for item in source_rows)
        snapshot = cls(source.path, list(rows), persist=False)
        snapshot._skills = MappingProxyType(dict(snapshot._skills))
        snapshot._read_only = True
        return snapshot

    def _require_writable(self) -> None:
        if self._read_only:
            raise RuntimeError("frozen Decision Skill snapshot is read-only")

    @property
    def read_only(self) -> bool:
        return self._read_only

    def __len__(self) -> int:
        return len(self._skills)
