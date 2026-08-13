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
    validation_score: float
    uses: int = 0
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
        if existing is not None and existing.validation_score > skill.validation_score:
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
            f"validation={skill.validation_score:.3f}"
            for skill in self._skills.values()
        )

    def record_use(self, name: str, score: float) -> None:
        skill = self._skills.get(name)
        if skill is None:
            return
        uses = skill.uses + 1
        average = score if skill.avg_score is None else (skill.avg_score * skill.uses + score) / uses
        self._skills[name] = replace(skill, uses=uses, avg_score=average)
        self.save()

    def clone(self, *, persist: bool = False) -> "RetrievalSkillLibrary":
        return RetrievalSkillLibrary(self.path, list(self._skills.values()), persist=persist)

    def __len__(self) -> int:
        return len(self._skills)
