"""Persistent, outcome-validated retrieval strategies."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class RetrievalSkill:
    """A generalized strategy for finding and validating forecast-useful evidence."""

    skill_id: str
    name: str
    description: str
    applicability: str
    query_strategy: str
    verification_rule: str
    created_from_task: str
    validation_smae: float | None = None
    validation_srmse: float | None = None
    validation_score: float | None = None
    uses: int = 0
    avg_smae: float | None = None
    avg_srmse: float | None = None
    avg_score: float | None = None


class RetrievalSkillLibrary:
    def __init__(
        self,
        path: str | Path,
        skills: list[RetrievalSkill] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        self.path = Path(path)
        self.persist = persist
        self._skills = {skill.name: skill for skill in (skills or [])}

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalSkillLibrary":
        source = Path(path)
        if not source.exists():
            return cls(source)
        return cls(source, [RetrievalSkill(**record) for record in json.loads(source.read_text())])

    def save(self) -> None:
        if not self.persist:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([asdict(skill) for skill in self._skills.values()], indent=2)
        )
        os.replace(temporary, self.path)

    def add(self, skill: RetrievalSkill) -> None:
        existing = self._skills.get(skill.name)
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

    def get(self, name: str) -> RetrievalSkill | None:
        return self._skills.get(name)

    def all(self) -> tuple[RetrievalSkill, ...]:
        return tuple(self._skills.values())

    def list_for_prompt(self) -> str:
        if not self._skills:
            return "(no validated retrieval skills yet)"
        return "\n".join(
            f"- {skill.name}: {skill.description}; applies when: {skill.applicability}; "
            f"query: {skill.query_strategy}; verify: {skill.verification_rule}; "
            f"sMAE signal={skill.validation_smae:.3f}; "
            f"sRMSE signal={skill.validation_srmse:.3f}"
            for skill in self._skills.values()
            if skill.validation_smae is not None and skill.validation_srmse is not None
        )

    def record_use(self, name: str, smae: float, srmse: float) -> None:
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

    def clone(self, *, persist: bool = False) -> "RetrievalSkillLibrary":
        return RetrievalSkillLibrary(self.path, list(self._skills.values()), persist=persist)

    def __len__(self) -> int:
        return len(self._skills)
