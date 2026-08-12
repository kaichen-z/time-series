from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolving_agent.tracing import TraceEvent, configure, emit


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
