"""Tests for the coding-skill agent: hypothesis generation, prompts, the skill library, baseline scoring, and tracing."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from evolving_loop.coding_agent.agent import CodingSkillAgent, FALLBACK_SKILL_NAME
from evolving_loop.coding_agent.skill_library import Skill, SkillLibrary
from evolving_loop.data import Task
from common.llm import FakeLLMClient
from evolving_loop.coding_agent.baseline import run_baseline, select_tasks
from evolving_loop.tracing import TraceEvent, configure, emit
from evolving_loop.coding_agent.prompts import (
    InvalidAgentResponseError,
    build_system_prompt,
    build_user_message,
    validate_response,
)
from evolving_loop.decision_agent.skill_library import DecisionSkill, DecisionSkillLibrary
from evolving_loop.retrieval_agent.skill_library import RetrievalSkill, RetrievalSkillLibrary


VALID_CODE = 'def forecast(history, horizon, frequency):\n    return [history[-1]] * horizon\n'


BROKEN_CODE = 'def forecast(history, horizon, frequency):\n    return undefined_name\n'


def _task() -> Task:
    return Task(
        task_id="task_1",
        history_values=(1.0, 2.0, 3.0),
        future_values=(3.0, 3.0),
        prediction_length=2,
        frequency="1 day",
        seasonal_period="D",
        entity_name="entity_a",
    )


def _write_skill_response(name="detect_naive", description="repeats the last value", code=VALID_CODE) -> str:
    return json.dumps(
        {"action": "write_skill", "new_skill": {"name": name, "description": description, "code": code}}
    )


def _use_skill_response(name="detect_naive") -> str:
    return json.dumps({"action": "use_skill", "skill_name": name})


class CodingSkillAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.library_path = Path(self.tmpdir.name) / "library.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_write_skill_path_saves_to_library(self):
        llm = FakeLLMClient([_write_skill_response()])
        library = SkillLibrary.load(self.library_path)
        agent = CodingSkillAgent(llm, library, mode="library")

        result = agent.run_task(_task())

        self.assertEqual(result.action, "write_skill")
        self.assertEqual(result.forecast, (3.0, 3.0))
        self.assertEqual(len(library), 1)
        self.assertEqual(library.get("detect_naive").code, VALID_CODE)

    def test_use_skill_path_reuses_existing_code_without_a_new_llm_call_for_code(self):
        library = SkillLibrary.load(self.library_path)
        library.add(Skill(skill_id="s1", name="detect_naive", description="d", code=VALID_CODE, created_from_task="t0"))
        llm = FakeLLMClient([_use_skill_response()])
        agent = CodingSkillAgent(llm, library, mode="library")

        result = agent.run_task(_task())

        self.assertEqual(result.action, "use_skill")
        self.assertEqual(result.skill_name, "detect_naive")
        self.assertEqual(result.forecast, (3.0, 3.0))
        self.assertEqual(len(library), 1)  # unchanged, not duplicated

    def test_fresh_mode_never_reads_or_writes_the_library(self):
        library = SkillLibrary.load(self.library_path)
        llm = FakeLLMClient([_write_skill_response()])
        agent = CodingSkillAgent(llm, library, mode="fresh")

        agent.run_task(_task())

        self.assertEqual(len(library), 0)
        system_prompt_used = llm.calls[0]["system"]
        self.assertNotIn("use_skill", system_prompt_used)

    def test_sandbox_failure_triggers_one_retry_then_succeeds(self):
        llm = FakeLLMClient([_write_skill_response(code=BROKEN_CODE), _write_skill_response(name="fixed")])
        library = SkillLibrary.load(self.library_path)
        agent = CodingSkillAgent(llm, library, mode="library")

        result = agent.run_task(_task())

        self.assertEqual(result.action, "write_skill")
        self.assertEqual(result.skill_name, "fixed")
        self.assertEqual(len(llm.calls), 2)
        second_call_message = llm.calls[1]["messages"][0]["content"]
        self.assertIn("previous code failed", second_call_message)

    def test_two_failures_fall_back_to_deterministic_forecast(self):
        llm = FakeLLMClient([_write_skill_response(code=BROKEN_CODE), _write_skill_response(code=BROKEN_CODE)])
        library = SkillLibrary.load(self.library_path)
        agent = CodingSkillAgent(llm, library, mode="library")

        result = agent.run_task(_task())

        self.assertEqual(result.action, "fallback")
        self.assertEqual(result.skill_name, FALLBACK_SKILL_NAME)
        self.assertEqual(result.forecast, (3.0, 3.0))  # repeats history[-1]
        self.assertEqual(len(library), 0)  # broken code is never saved
        self.assertIsNotNone(result.error)
        self.assertIn("undefined_name", result.error)

    def test_referencing_an_unknown_skill_triggers_retry(self):
        llm = FakeLLMClient([_use_skill_response(name="does_not_exist"), _write_skill_response()])
        library = SkillLibrary.load(self.library_path)
        agent = CodingSkillAgent(llm, library, mode="library")

        result = agent.run_task(_task())

        self.assertEqual(result.action, "write_skill")

    def test_malformed_json_triggers_retry(self):
        llm = FakeLLMClient(["not json at all", _write_skill_response()])
        library = SkillLibrary.load(self.library_path)
        agent = CodingSkillAgent(llm, library, mode="library")

        result = agent.run_task(_task())

        self.assertEqual(result.action, "write_skill")


if __name__ == "__main__":
    unittest.main()

VALID_CODE = 'def forecast(history, horizon, frequency):\n    return [history[-1]] * horizon\n'


def _baseline_write_skill_response(name="detect_naive") -> str:
    return json.dumps(
        {"action": "write_skill", "new_skill": {"name": name, "description": "repeats last value", "code": VALID_CODE}}
    )


def _baseline_use_skill_response(name="detect_naive") -> str:
    return json.dumps({"action": "use_skill", "skill_name": name})


def _baseline_task(task_id: str, entity: str) -> Task:
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
        tasks = [_baseline_task(f"t{i}", f"e{i}") for i in range(10)]
        a = select_tasks(tasks, seed=7, limit=None)
        b = select_tasks(tasks, seed=7, limit=None)
        self.assertEqual([t.task_id for t in a], [t.task_id for t in b])

    def test_limit_truncates(self):
        tasks = [_baseline_task(f"t{i}", f"e{i}") for i in range(10)]
        self.assertEqual(len(select_tasks(tasks, seed=7, limit=3)), 3)

    def test_does_not_mutate_the_input_list(self):
        tasks = [_baseline_task(f"t{i}", f"e{i}") for i in range(5)]
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
        tasks = [_baseline_task("t1", "e1"), _baseline_task("t2", "e2")]
        llm = FakeLLMClient([_baseline_write_skill_response("s1"), _baseline_use_skill_response("s1")])
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
        tasks = [_baseline_task("t1", "e1")]
        llm = FakeLLMClient([_baseline_write_skill_response("s1")])

        summary = run_baseline(tasks, "fresh", llm, None, self.results_path)

        self.assertEqual(summary["skills_saved"], 0)

    def test_summary_has_first_and_second_half_means(self):
        tasks = [_baseline_task(f"t{i}", f"e{i}") for i in range(4)]
        llm = FakeLLMClient([_baseline_write_skill_response(f"s{i}") for i in range(4)])

        summary = run_baseline(tasks, "fresh", llm, None, self.results_path)

        self.assertIsNotNone(summary["mean_smae_first_half"])
        self.assertIsNotNone(summary["mean_smae_second_half"])

    def test_empty_task_list_produces_a_summary_without_crashing(self):
        summary = run_baseline([], "fresh", FakeLLMClient([]), None, self.results_path)
        self.assertEqual(summary["n_tasks"], 0)
        self.assertIsNone(summary["mean_smae"])


if __name__ == "__main__":
    unittest.main()

def _prompts_task() -> Task:
    return Task(
        task_id="task_1",
        history_values=(1.0, 2.0, 3.0),
        future_values=(4.0, 5.0),
        prediction_length=2,
        frequency="1 day",
        seasonal_period="D",
        entity_name="entity_a",
    )


class BuildSystemPromptTests(unittest.TestCase):
    def test_library_prompt_mentions_reuse(self):
        self.assertIn("reuse", build_system_prompt(has_library=True).lower())

    def test_no_library_prompt_never_mentions_reuse(self):
        prompt = build_system_prompt(has_library=False).lower()
        self.assertNotIn("reuse", prompt)
        self.assertNotIn("use_skill", prompt)
        self.assertNotIn("revise_skill", prompt)

    def test_library_prompt_mentions_anti_duplication(self):
        prompt = build_system_prompt(has_library=True).lower()
        self.assertIn("never create a new skill that is a variant", prompt)

    def test_library_prompt_allows_helper_functions(self):
        prompt = build_system_prompt(has_library=True).lower()
        self.assertIn("helper function", prompt)


class BuildUserMessageTests(unittest.TestCase):
    def test_includes_history_and_horizon(self):
        message = build_user_message(_prompts_task(), library_text=None)
        self.assertIn("1.0, 2.0, 3.0", message)
        self.assertIn("horizon: 2", message)

    def test_includes_library_text_when_given(self):
        message = build_user_message(_prompts_task(), library_text="- detect_trend: fits a trend")
        self.assertIn("detect_trend", message)

    def test_omits_library_section_when_none(self):
        message = build_user_message(_prompts_task(), library_text=None)
        self.assertNotIn("Available skills", message)

    def test_includes_retry_error_when_given(self):
        message = build_user_message(_prompts_task(), library_text=None, retry_error="NameError: x is not defined")
        self.assertIn("NameError", message)
        self.assertIn("previous code failed", message)


class ValidateResponseTests(unittest.TestCase):
    def test_valid_use_skill(self):
        validate_response({"action": "use_skill", "skill_name": "detect_trend"})

    def test_valid_write_skill(self):
        validate_response(
            {
                "action": "write_skill",
                "new_skill": {"name": "n", "description": "d", "code": "def forecast(): pass"},
            }
        )

    def test_unknown_action_rejected(self):
        with self.assertRaises(InvalidAgentResponseError):
            validate_response({"action": "delete_everything"})

    def test_use_skill_without_name_rejected(self):
        with self.assertRaises(InvalidAgentResponseError):
            validate_response({"action": "use_skill"})

    def test_write_skill_missing_field_rejected(self):
        with self.assertRaises(InvalidAgentResponseError):
            validate_response({"action": "write_skill", "new_skill": {"name": "n", "description": "d"}})

    def test_write_skill_with_non_dict_new_skill_rejected(self):
        with self.assertRaises(InvalidAgentResponseError):
            validate_response({"action": "write_skill", "new_skill": "not a dict"})

    def test_write_skill_code_without_forecast_function_rejected(self):
        with self.assertRaises(InvalidAgentResponseError):
            validate_response(
                {
                    "action": "write_skill",
                    "new_skill": {"name": "n", "description": "d", "code": "x = 1\n# no forecast here"},
                }
            )

    def test_write_skill_code_with_syntax_error_rejected(self):
        with self.assertRaises(InvalidAgentResponseError):
            validate_response(
                {
                    "action": "write_skill",
                    "new_skill": {"name": "n", "description": "d", "code": "def forecast(:"},
                }
            )

    def test_valid_revise_skill(self):
        validate_response(
            {"action": "revise_skill", "skill_name": "detect_trend", "new_code": "def forecast(): pass"}
        )

    def test_revise_skill_without_name_rejected(self):
        with self.assertRaises(InvalidAgentResponseError):
            validate_response({"action": "revise_skill", "new_code": "def forecast(): pass"})

    def test_revise_skill_without_new_code_rejected(self):
        with self.assertRaises(InvalidAgentResponseError):
            validate_response({"action": "revise_skill", "skill_name": "detect_trend"})

    def test_revise_skill_new_code_without_forecast_function_rejected(self):
        with self.assertRaises(InvalidAgentResponseError):
            validate_response(
                {"action": "revise_skill", "skill_name": "detect_trend", "new_code": "x = 1"}
            )

    def test_write_skill_code_with_nested_forecast_rejected(self):
        # A forecast() defined inside another scope is not callable as a top-level function.
        with self.assertRaises(InvalidAgentResponseError):
            validate_response(
                {
                    "action": "write_skill",
                    "new_skill": {
                        "name": "n",
                        "description": "d",
                        "code": "def outer():\n    def forecast(history, horizon, frequency):\n        return []\n",
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()

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
        self.assertIn("mean_smae=12.5000", text)

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

class TracingTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.tmpdir.name) / "run.log"
        configure(self.log_file, console_level="INFO")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_every_event_reaches_the_file(self):
        emit(TraceEvent(task_id="task_1", mode="library", event_type="task_start"))
        emit(TraceEvent(task_id="task_1", mode="library", event_type="llm_call", detail={"prompt": "..."}))
        emit(TraceEvent(task_id="task_1", mode="library", event_type="task_end", detail={"score": 12.5}))
        lines = self.log_file.read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)
        event_types = [json.loads(line)["event_type"] for line in lines]
        self.assertEqual(event_types, ["task_start", "llm_call", "task_end"])

    def test_file_lines_are_valid_json_with_expected_fields(self):
        emit(TraceEvent(task_id="task_1", mode="fresh", event_type="tool_call", detail={"tool": "sandbox"}))
        record = json.loads(self.log_file.read_text().strip())
        self.assertEqual(record["task_id"], "task_1")
        self.assertEqual(record["mode"], "fresh")
        self.assertEqual(record["detail"], {"tool": "sandbox"})
        self.assertIn("timestamp", record)

    def test_reconfiguring_does_not_duplicate_handlers(self):
        configure(self.log_file, console_level="INFO")
        configure(self.log_file, console_level="INFO")
        emit(TraceEvent(task_id="task_1", mode="library", event_type="task_start"))
        lines = self.log_file.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()

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
