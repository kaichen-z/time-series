from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolving_agent.coding_agent.baseline import run_baseline, select_tasks
from evolving_agent.coding_agent.skill_library import SkillLibrary
from evolving_agent.data import Task
from evolving_agent.tracing import configure
from common.llm import FakeLLMClient

VALID_CODE = 'def forecast(history, horizon, frequency):\n    return [history[-1]] * horizon\n'


def _write_skill_response(name="detect_naive") -> str:
    return json.dumps(
        {"action": "write_skill", "new_skill": {"name": name, "description": "repeats last value", "code": VALID_CODE}}
    )


def _use_skill_response(name="detect_naive") -> str:
    return json.dumps({"action": "use_skill", "skill_name": name})


def _task(task_id: str, entity: str) -> Task:
    return Task(
        task_id=task_id,
        history_values=(1.0, 2.0, 3.0),
        future_values=(3.0, 3.0),
        prediction_length=2,
        frequency="1 day",
        seasonal_period="D",
        entity_name=entity,
    )


class SelectTasksTests(unittest.TestCase):
    def test_deterministic_for_a_fixed_seed(self):
        tasks = [_task(f"t{i}", f"e{i}") for i in range(10)]
        a = select_tasks(tasks, seed=7, limit=None)
        b = select_tasks(tasks, seed=7, limit=None)
        self.assertEqual([t.task_id for t in a], [t.task_id for t in b])

    def test_limit_truncates(self):
        tasks = [_task(f"t{i}", f"e{i}") for i in range(10)]
        self.assertEqual(len(select_tasks(tasks, seed=7, limit=3)), 3)

    def test_does_not_mutate_the_input_list(self):
        tasks = [_task(f"t{i}", f"e{i}") for i in range(5)]
        original_order = [t.task_id for t in tasks]
        select_tasks(tasks, seed=7, limit=None)
        self.assertEqual([t.task_id for t in tasks], original_order)


class RunBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_path = Path(self.tmpdir.name) / "results.jsonl"
        self.library_path = Path(self.tmpdir.name) / "library.json"
        configure(Path(self.tmpdir.name) / "run.log")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_library_mode_writes_one_result_row_per_task_and_grows_the_library(self):
        tasks = [_task("t1", "e1"), _task("t2", "e2")]
        llm = FakeLLMClient([_write_skill_response("s1"), _use_skill_response("s1")])
        library = SkillLibrary.load(self.library_path)

        summary = run_baseline(tasks, "library", llm, library, self.results_path)

        rows = [json.loads(line) for line in self.results_path.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["action"], "write_skill")
        self.assertEqual(rows[1]["action"], "use_skill")
        self.assertEqual(summary["n_tasks"], 2)
        self.assertEqual(summary["skills_saved"], 1)
        self.assertEqual(library.get("s1").uses, 2)  # both the creating task and the reuse count

    def test_fresh_mode_reports_zero_skills_saved(self):
        tasks = [_task("t1", "e1")]
        llm = FakeLLMClient([_write_skill_response("s1")])

        summary = run_baseline(tasks, "fresh", llm, None, self.results_path)

        self.assertEqual(summary["skills_saved"], 0)

    def test_summary_has_first_and_second_half_means(self):
        tasks = [_task(f"t{i}", f"e{i}") for i in range(4)]
        llm = FakeLLMClient([_write_skill_response(f"s{i}") for i in range(4)])

        summary = run_baseline(tasks, "fresh", llm, None, self.results_path)

        self.assertIsNotNone(summary["mean_smape_first_half"])
        self.assertIsNotNone(summary["mean_smape_second_half"])

    def test_empty_task_list_produces_a_summary_without_crashing(self):
        summary = run_baseline([], "fresh", FakeLLMClient([]), None, self.results_path)
        self.assertEqual(summary["n_tasks"], 0)
        self.assertIsNone(summary["mean_smape"])


if __name__ == "__main__":
    unittest.main()
