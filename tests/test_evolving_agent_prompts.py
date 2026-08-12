from __future__ import annotations

import unittest

from evolving_agent.coding_agent.prompts import (
    InvalidAgentResponseError,
    build_system_prompt,
    build_user_message,
    validate_response,
)
from evolving_agent.data import Task


def _task() -> Task:
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


class BuildUserMessageTests(unittest.TestCase):
    def test_includes_history_and_horizon(self):
        message = build_user_message(_task(), library_text=None)
        self.assertIn("1.0, 2.0, 3.0", message)
        self.assertIn("horizon: 2", message)

    def test_includes_library_text_when_given(self):
        message = build_user_message(_task(), library_text="- detect_trend: fits a trend")
        self.assertIn("detect_trend", message)

    def test_omits_library_section_when_none(self):
        message = build_user_message(_task(), library_text=None)
        self.assertNotIn("Available skills", message)

    def test_includes_retry_error_when_given(self):
        message = build_user_message(_task(), library_text=None, retry_error="NameError: x is not defined")
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


if __name__ == "__main__":
    unittest.main()
