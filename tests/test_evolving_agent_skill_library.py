from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evolving_loop.coding_agent.skill_library import Skill, SkillLibrary


def _skill(name="detect_trend", description="fits a linear trend") -> Skill:
    return Skill(
        skill_id="s1",
        name=name,
        description=description,
        code="def forecast(history, horizon, frequency): return [0.0] * horizon",
        created_from_task="task_1",
    )


class SkillLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "library.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_loading_a_missing_file_starts_empty(self):
        library = SkillLibrary.load(self.path)
        self.assertEqual(len(library), 0)

    def test_add_then_get(self):
        library = SkillLibrary.load(self.path)
        library.add(_skill())
        self.assertEqual(library.get("detect_trend").description, "fits a linear trend")
        self.assertIsNone(library.get("does_not_exist"))

    def test_add_persists_to_disk_and_reloads(self):
        library = SkillLibrary.load(self.path)
        library.add(_skill())
        reloaded = SkillLibrary.load(self.path)
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded.get("detect_trend").code, _skill().code)

    def test_list_for_prompt_is_name_and_description_only(self):
        library = SkillLibrary.load(self.path)
        library.add(_skill())
        text = library.list_for_prompt()
        self.assertIn("detect_trend", text)
        self.assertIn("fits a linear trend", text)
        self.assertNotIn("def forecast", text)

    def test_list_for_prompt_when_empty(self):
        library = SkillLibrary.load(self.path)
        self.assertEqual(library.list_for_prompt(), "(no skills saved yet)")

    def test_record_use_updates_running_average(self):
        library = SkillLibrary.load(self.path)
        library.add(_skill())
        library.record_use("detect_trend", ok=True, score=10.0)
        library.record_use("detect_trend", ok=True, score=20.0)
        skill = library.get("detect_trend")
        self.assertEqual(skill.uses, 2)
        self.assertEqual(skill.failures, 0)
        self.assertEqual(skill.avg_score, 15.0)

    def test_record_use_failure_updates_failures_not_avg_score(self):
        library = SkillLibrary.load(self.path)
        library.add(_skill())
        library.record_use("detect_trend", ok=True, score=10.0)
        library.record_use("detect_trend", ok=False)
        skill = library.get("detect_trend")
        self.assertEqual(skill.uses, 2)
        self.assertEqual(skill.failures, 1)
        self.assertEqual(skill.avg_score, 10.0)  # unaffected by the failed attempt

    def test_list_for_prompt_shows_stats_once_used(self):
        library = SkillLibrary.load(self.path)
        library.add(_skill())
        library.record_use("detect_trend", ok=True, score=12.5)
        library.record_use("detect_trend", ok=False)
        text = library.list_for_prompt()
        self.assertIn("uses=2", text)
        self.assertIn("ok_rate=0.50", text)
        self.assertIn("mean_smape=12.5000", text)

    def test_list_for_prompt_omits_stats_for_a_never_used_skill(self):
        library = SkillLibrary.load(self.path)
        library.add(_skill())
        text = library.list_for_prompt()
        self.assertNotIn("uses=", text)
        self.assertNotIn("ok_rate", text)

    def test_revise_replaces_code_and_resets_stats(self):
        library = SkillLibrary.load(self.path)
        library.add(_skill())
        library.record_use("detect_trend", ok=True, score=10.0)
        new_code = "def forecast(history, horizon, frequency): return [1.0] * horizon"
        library.revise("detect_trend", new_code)
        skill = library.get("detect_trend")
        self.assertEqual(skill.code, new_code)
        self.assertEqual(skill.uses, 0)
        self.assertEqual(skill.failures, 0)
        self.assertIsNone(skill.avg_score)
        self.assertEqual(skill.description, "fits a linear trend")  # identity preserved

    def test_revise_persists_to_disk(self):
        library = SkillLibrary.load(self.path)
        library.add(_skill())
        new_code = "def forecast(history, horizon, frequency): return [2.0] * horizon"
        library.revise("detect_trend", new_code)
        reloaded = SkillLibrary.load(self.path)
        self.assertEqual(reloaded.get("detect_trend").code, new_code)

    def test_add_overwrites_existing_skill_with_the_same_name(self):
        library = SkillLibrary.load(self.path)
        library.add(_skill())
        library.add(_skill(description="a different description"))
        self.assertEqual(len(library), 1)
        self.assertEqual(library.get("detect_trend").description, "a different description")

    def test_nonpersistent_clone_does_not_contaminate_parent_or_disk(self):
        library = SkillLibrary.load(self.path)
        library.add(_skill())
        clone = library.clone(persist=False)
        clone.add(_skill(name="new_skill"))

        self.assertEqual(len(clone), 2)
        self.assertEqual(len(library), 1)
        self.assertEqual(len(SkillLibrary.load(self.path)), 1)


if __name__ == "__main__":
    unittest.main()
