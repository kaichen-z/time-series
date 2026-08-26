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
    assumption: str = ""
    failure_condition: str = ""
    validation_smae: float | None = None
    validation_srmse: float | None = None
    # Retained only so older artifacts remain loadable; new runs never use it.
    validation_score: float | None = None
    uses: int = 0
    failures: int = 0
    avg_smae: float | None = None
    avg_srmse: float | None = None
    avg_score: float | None = None


class SkillLibrary:
    """In-memory skill set backed by a single JSON file, written atomically on every change."""

    def __init__(
        self,
        path: str | Path,
        skills: list[Skill] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        self.path = Path(path)
        self._skills: dict[str, Skill] = {s.name: s for s in (skills or [])}
        self.persist = persist

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
        if not self.persist:
            return
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
        """Name + description (never full code), plus usage stats once a skill has actually been tried."""
        if not self._skills:
            return "(no skills saved yet)"
        return "\n".join(self._line_for_prompt(skill) for skill in self._skills.values())

    @staticmethod
    def _line_for_prompt(skill: Skill) -> str:
        line = f"- {skill.name}: {skill.description}"
        stats = []
        if skill.validation_smae is not None:
            stats.append(f"hindcast_sMAE={skill.validation_smae:.4f}")
        if skill.validation_srmse is not None:
            stats.append(f"hindcast_sRMSE={skill.validation_srmse:.4f}")
        if skill.uses > 0:
            ok_rate = (skill.uses - skill.failures) / skill.uses
            stats.append(f"uses={skill.uses}")
            stats.append(f"ok_rate={ok_rate:.2f}")
            if skill.avg_smae is not None:
                stats.append(f"mean_sMAE={skill.avg_smae:.4f}")
            if skill.avg_srmse is not None:
                stats.append(f"mean_sRMSE={skill.avg_srmse:.4f}")
            if skill.avg_score is not None:
                stats.append(f"mean_smape={skill.avg_score:.4f}")
        if stats:
            line += " [" + ", ".join(stats) + "]"
        return line

    def all(self) -> tuple[Skill, ...]:
        """Return an immutable snapshot for validated candidate execution."""
        return tuple(self._skills.values())

    def clone(self, *, persist: bool = False) -> "SkillLibrary":
        """Create an evaluation-local snapshot so competing policies cannot contaminate each other."""
        return SkillLibrary(self.path, list(self._skills.values()), persist=persist)

    def record_use(
        self,
        name: str,
        ok: bool,
        smae: float | None = None,
        srmse: float | None = None,
        score: float | None = None,
    ) -> None:
        """Record one use attempt.

        ``score`` is the legacy sMAPE input retained for old saved runners;
        current callers supply the Dr-CiK ``smae``/``srmse`` pair.
        """
        skill = self._skills[name]
        new_uses = skill.uses + 1
        if not ok:
            self._skills[name] = replace(skill, uses=new_uses, failures=skill.failures + 1)
            self.save()
            return
        successes_before = skill.uses - skill.failures
        avg_smae = (
            smae
            if skill.avg_smae is None
            else (skill.avg_smae * successes_before + smae) / (successes_before + 1)
        ) if smae is not None else skill.avg_smae
        avg_srmse = (
            srmse
            if skill.avg_srmse is None
            else (skill.avg_srmse * successes_before + srmse) / (successes_before + 1)
        ) if srmse is not None else skill.avg_srmse
        avg_score = (
            score
            if skill.avg_score is None
            else (skill.avg_score * successes_before + score) / (successes_before + 1)
        ) if score is not None else skill.avg_score
        self._skills[name] = replace(
            skill,
            uses=new_uses,
            avg_smae=avg_smae,
            avg_srmse=avg_srmse,
            avg_score=avg_score,
        )
        self.save()

    def revise(self, name: str, new_code: str) -> None:
        """Replace a skill's code; reset usage stats since the new code is functionally different."""
        skill = self._skills[name]
        self._skills[name] = replace(
            skill,
            code=new_code,
            uses=0,
            failures=0,
            avg_smae=None,
            avg_srmse=None,
            avg_score=None,
        )
        self.save()

    def __len__(self) -> int:
        return len(self._skills)
