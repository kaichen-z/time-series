from __future__ import annotations

import tempfile
from pathlib import Path

from evolving_agent.decision_agent.skill_library import DecisionSkill, DecisionSkillLibrary
from evolving_agent.retrieval_agent.skill_library import RetrievalSkill, RetrievalSkillLibrary


def test_retrieval_library_clone_is_isolated() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "retrieval.json"
        library = RetrievalSkillLibrary.load(path)
        library.add(
            RetrievalSkill(
                "r1", "window", "d", "a", "q", "v", "task", 0.8
            )
        )
        clone = library.clone(persist=False)
        clone.add(RetrievalSkill("r2", "other", "d", "a", "q", "v", "task", 0.9))
        assert len(clone) == 2
        assert len(library) == 1
        assert len(RetrievalSkillLibrary.load(path)) == 1


def test_decision_library_keeps_better_duplicate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "decision.json"
        library = DecisionSkillLibrary.load(path)
        strong = DecisionSkill("d1", "rule", "d", "a", "r", "f", "task", 0.9)
        weak = DecisionSkill("d2", "rule", "worse", "a", "r", "f", "task", 0.5)
        library.add(strong)
        library.add(weak)
        assert library.get("rule").description == "d"
