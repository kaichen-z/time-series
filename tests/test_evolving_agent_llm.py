from __future__ import annotations

import unittest

from evolving_agent.llm import FakeLLMClient, JsonExtractionError, parse_json_object, pick_free_gpu


class ParseJsonObjectTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_json_object('{"a": 1}'), {"a": 1})

    def test_strips_think_block(self):
        text = "<think>let me reason about this</think>\n{\"a\": 1}"
        self.assertEqual(parse_json_object(text), {"a": 1})

    def test_strips_json_fence(self):
        text = '```json\n{"a": 1}\n```'
        self.assertEqual(parse_json_object(text), {"a": 1})

    def test_strips_think_and_fence_together(self):
        text = '<think>reasoning...</think>\nHere is my answer:\n```json\n{"action": "write_skill"}\n```'
        self.assertEqual(parse_json_object(text), {"action": "write_skill"})

    def test_invalid_json_raises(self):
        with self.assertRaises(JsonExtractionError):
            parse_json_object("not json at all")


class FakeLLMClientTests(unittest.TestCase):
    def test_returns_responses_in_order(self):
        client = FakeLLMClient(["first", "second"])
        self.assertEqual(client.complete(system="sys", messages=[]).text, "first")
        self.assertEqual(client.complete(system="sys", messages=[]).text, "second")

    def test_records_calls(self):
        client = FakeLLMClient(["ok"])
        client.complete(system="sys", messages=[{"role": "user", "content": "hi"}], temperature=0.7)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["temperature"], 0.7)

    def test_raises_when_exhausted(self):
        client = FakeLLMClient([])
        with self.assertRaises(RuntimeError):
            client.complete(system="sys", messages=[])


class PickFreeGpuTests(unittest.TestCase):
    def test_returns_a_cuda_device_or_cpu(self):
        result = pick_free_gpu()
        self.assertTrue(result == "cpu" or result.startswith("cuda:"))


if __name__ == "__main__":
    unittest.main()
