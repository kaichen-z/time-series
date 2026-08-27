from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolving_loop.coding_agent.agent import FALLBACK_SKILL_NAME, CodingSkillAgent
from evolving_loop.coding_agent.skill_library import Skill, SkillLibrary
from evolving_loop.data import ContextTask, Document, Task
from evolving_loop.retrieval_agent.agent import RetrievalAgent
from common.llm import FakeLLMClient

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


def test_legacy_retrieval_agent_projects_only_host_verified_numeric_impacts():
    task = ContextTask(
        numeric=Task(
            task_id="task_context",
            history_values=(1.0, 2.0),
            future_values=(),
            prediction_length=2,
            frequency="D",
            seasonal_period=None,
            entity_name="Entity A",
        ),
        target_name="sales",
        target_description="Daily sales",
        history_timestamps=("2026-01-01", "2026-01-02"),
        future_timestamps=("2026-01-03", "2026-01-04"),
        documents=(
            Document(
                "doc_1",
                "Entity A sales will increase from 2026-01-03 through 2026-01-04.",
            ),
        ),
        labels_public=False,
    )
    response = json.dumps({
        "query": "future sales",
        "selected_document_ids": ["doc_1"],
        "evidence": [{
            "document_id": "doc_1",
            "claim": "A future event changes sales.",
            "exact_quote": "Entity A sales will increase from 2026-01-03 through 2026-01-04.",
        }],
        "impacts": [{
            "source_document_ids": ["doc_1"],
            "mechanism_layer": "future_driver",
            "temporal_relation": "overlaps_future",
            "direction": "up",
            "permanence": "temporary",
            "adjustment_kind": "add",
            "adjustment_value": 7,
            "start_timestamp": "2026-01-03",
            "end_timestamp": "2026-01-04",
            "rationale": "An ungrounded magnitude must not reach Decision.",
        }],
        "sufficient": True,
        "missing_information": [],
    })

    result = RetrievalAgent(FakeLLMClient([response])).run(task, ())

    assert result.evidence
    assert result.impacts == ()
