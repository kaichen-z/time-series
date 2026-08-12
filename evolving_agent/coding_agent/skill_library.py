"""A persistent, JSON-backed library of named, tested forecasting skills (Voyager-style)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    """A reusable, named capability: a description to retrieve it by, plus tested code."""

    skill_id: str
    name: str
    description: str
    code: str
    created_from_task: str
    uses: int = 0   
    avg_score: float | None = None


class SkillLibrary:
    """In-memory skill set backed by a single JSON file, written atomically on every change."""

    def __init__(self, path: str | Path, skills: list[Skill] | None = None) -> None:
        self.path = Path(path)
        self._skills: dict[str, Skill] = {s.name: s for s in (skills or [])}

    @classmethod
    def load(cls, path: str | Path) -> "SkillLibrary":
        """Load an existing library file, or start empty if none exists yet."""
        path = Path(path)
        if not path.exists():
            return cls(path)
        records = json.loads(path.read_text())
        return cls(path, skills=[Skill(**record) for record in records])

    def save(self) -> None:
        """Write the whole library out atomically (tmp file + rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps([asdict(s) for s in self._skills.values()], indent=2))
        os.replace(tmp_path, self.path)

    def add(self, skill: Skill) -> None:
        """Add or overwrite a skill by name, and persist immediately."""
        self._skills[skill.name] = skill
        self.save()

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_for_prompt(self) -> str:
        """Name + one-line description only (never full code) so the prompt stays small as the library grows."""
        if not self._skills:
            return "(no skills saved yet)"
        return "\n".join(f"- {s.name}: {s.description}" for s in self._skills.values())

    def record_use(self, name: str, score: float) -> None:
        """Update a skill's running average score after it was used on a task."""
        skill = self._skills[name]
        new_uses = skill.uses + 1
        new_avg = score if skill.avg_score is None else (skill.avg_score * skill.uses + score) / new_uses
        self._skills[name] = replace(skill, uses=new_uses, avg_score=new_avg)
        self.save()

    def __len__(self) -> int:
        return len(self._skills)
