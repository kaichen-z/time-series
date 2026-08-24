"""Tests for common/llm: JSON extraction, the fake client, GPU placement, and the Codex/Claude CLI clients."""
from __future__ import annotations

import unittest
import json
import subprocess
import tempfile
from common.llm import (
    ClaudeCLIClient,
    ClaudeCLIConfig,
    CodexCLIClient,
    CodexCLIConfig,
    FakeLLMClient,
    JsonExtractionError,
    QwenClient,
    TransientLLMError,
    _parse_gpu_free,
    parse_json_object,
    pick_free_gpu,
    shard_max_memory,
)
from unittest.mock import patch
from pathlib import Path


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

NVIDIA_SMI = "0, 54696, 81559\n1, 35809, 81559\n7, 25055, 81559\n"


def test_parse_reads_free_memory_per_device() -> None:
    free = _parse_gpu_free(NVIDIA_SMI)

    assert free == {0: 81559 - 54696, 1: 81559 - 35809, 7: 81559 - 25055}


def test_parse_skips_malformed_rows() -> None:
    assert _parse_gpu_free("0, 10, 100\ngarbage\n1, x, 100\n") == {0: 90}


def test_pick_free_gpu_still_takes_the_freest_by_default() -> None:
    with patch("common.llm._gpu_free_mib", return_value=_parse_gpu_free(NVIDIA_SMI)):
        assert pick_free_gpu() == "cuda:7"


def test_pick_free_gpu_refuses_a_device_without_the_required_headroom() -> None:
    free = _parse_gpu_free(NVIDIA_SMI)  # freest card has ~55 GB

    with patch("common.llm._gpu_free_mib", return_value=free):
        assert pick_free_gpu(required_gb=50) == "cuda:7"
        # The OOM case: the freest card fits the weights but not the run.
        assert pick_free_gpu(required_gb=70) == "cpu"


def test_pick_free_gpu_falls_back_when_nvidia_smi_is_unavailable() -> None:
    with patch("common.llm._gpu_free_mib", return_value={}):
        assert pick_free_gpu() == "cpu"


def test_shard_budget_reserves_headroom_for_other_users() -> None:
    with patch("common.llm._gpu_free_mib", return_value=_parse_gpu_free(NVIDIA_SMI)):
        budget = shard_max_memory(reserve_gb=10.0, min_free_gb=40.0)

    # Only 7 (~55 GB free) clears the bar; 0 (~26) and 1 (~45... below in this fixture) do not.
    assert 7 in budget
    assert budget[7].endswith("GiB")


def test_shard_budget_excludes_contended_cards() -> None:
    # The real failure: a card with 26 GB free is not full, but a neighbour growing into it
    # evicts our shard mid-generation. It must not be offered at all.
    free = {0: 26 * 1024, 3: 54 * 1024, 4: 55 * 1024}

    with patch("common.llm._gpu_free_mib", return_value=free):
        budget = shard_max_memory(reserve_gb=10.0, min_free_gb=40.0)

    assert set(budget) == {3, 4}


def test_shard_budget_falls_back_to_the_roomiest_card_when_none_qualify() -> None:
    with patch("common.llm._gpu_free_mib", return_value={0: 20 * 1024, 1: 30 * 1024}):
        budget = shard_max_memory(reserve_gb=10.0, min_free_gb=40.0)

    # Degrade rather than hand accelerate an empty budget.
    assert set(budget) == {1}


def test_shard_budget_is_empty_without_gpus() -> None:
    with patch("common.llm._gpu_free_mib", return_value={}):
        assert shard_max_memory() == {}


def test_shard_budget_is_keyed_by_int_index_for_from_pretrained() -> None:
    with patch("common.llm._gpu_free_mib", return_value={3: 60 * 1024}):
        budget = shard_max_memory(reserve_gb=10.0)

    assert all(isinstance(key, int) for key in budget)


def test_an_explicit_device_pins_the_model_as_before() -> None:
    client = QwenClient(device="cuda:3")

    # The pinned path must keep using .to(device) and never pass device_map.
    assert client.device == "cuda:3"
    assert client._input_device() == "cuda:3"


def test_an_unset_device_defers_placement_to_sharding() -> None:
    client = QwenClient()

    # No eager pick_free_gpu(): placement is accelerate's job at load time.
    assert client.device is None
    assert client.reserve_gb == 10.0

def _envelope(result_text: str, is_error: bool = False) -> str:
    return json.dumps({"is_error": is_error, "result": result_text, "type": "result"})


def test_claude_cli_client_returns_and_caches_json() -> None:
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "cache"
        client = ClaudeCLIClient(
            ClaudeCLIConfig(model="test-model", cache_dir=cache, timeout_seconds=5)
        )

        def fake_run(command, **kwargs):
            assert "--permission-mode" not in command  # never force plan mode
            assert command[command.index("--allowedTools") + 1] == ""
            assert command[command.index("--system-prompt") + 1] == "Return JSON."
            assert not any(key.startswith("CLAUDE") for key in kwargs["env"])
            return subprocess.CompletedProcess(command, 0, _envelope('{"answer": 7}'), "")

        with patch("common.llm.subprocess.run", side_effect=fake_run) as mocked:
            first = client.complete(system="Return JSON.", messages=[{"role": "user", "content": "x"}])
            second = client.complete(system="Return JSON.", messages=[{"role": "user", "content": "x"}])

    assert parse_json_object(first.text) == {"answer": 7}
    assert second.text == first.text
    assert mocked.call_count == 1
    assert client.calls == 1
    assert client.cache_hits == 1


def test_claude_cli_client_raises_on_error_result() -> None:
    client = ClaudeCLIClient(ClaudeCLIConfig(cache_dir=None, timeout_seconds=5))

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, _envelope("refused", is_error=True), "")

    with patch("common.llm.subprocess.run", side_effect=fake_run):
        try:
            client.complete(system="s", messages=[{"role": "user", "content": "x"}])
            assert False, "expected a RuntimeError"
        except RuntimeError as exc:
            assert "error result" in str(exc)


def test_claude_cli_client_missing_binary_raises_clear_error() -> None:
    client = ClaudeCLIClient(ClaudeCLIConfig(binary="does-not-exist", cache_dir=None, timeout_seconds=5))

    def fake_run(command, **kwargs):
        raise FileNotFoundError(command[0])

    with patch("common.llm.subprocess.run", side_effect=fake_run):
        try:
            client.complete(system="s", messages=[{"role": "user", "content": "x"}])
            assert False, "expected a RuntimeError"
        except RuntimeError as exc:
            assert "was not found" in str(exc)

def test_codex_cli_client_returns_and_caches_json() -> None:
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "cache"
        client = CodexCLIClient(
            CodexCLIConfig(model="test-model", cache_dir=cache, timeout_seconds=5)
        )

        def fake_run(command, **kwargs):
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text('{"answer": 7}')
            assert "Do not use tools" in kwargs["input"]
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("common.llm.subprocess.run", side_effect=fake_run) as mocked:
            first = client.complete(system="Return JSON.", messages=[{"role": "user", "content": "x"}])
            second = client.complete(system="Return JSON.", messages=[{"role": "user", "content": "x"}])

    assert parse_json_object(first.text) == {"answer": 7}
    assert second.text == first.text
    assert mocked.call_count == 1
    assert client.calls == 1
    assert client.cache_hits == 1


def test_codex_cli_client_repairs_malformed_json_once() -> None:
    with tempfile.TemporaryDirectory() as directory:
        client = CodexCLIClient(
            CodexCLIConfig(cache_dir=Path(directory) / "cache", json_repair_attempts=2)
        )
        outputs = iter(('{"answer": 7', '{"answer": 7}'))

        def fake_run(command, **_kwargs):
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text(next(outputs))
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("common.llm.subprocess.run", side_effect=fake_run) as mocked:
            response = client.complete(
                system="Return JSON.", messages=[{"role": "user", "content": "x"}]
            )

    assert parse_json_object(response.text) == {"answer": 7}
    assert mocked.call_count == 2
    assert client.calls == 2


def test_codex_cli_client_ignores_malformed_cache() -> None:
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "cache"
        client = CodexCLIClient(CodexCLIConfig(cache_dir=cache))
        prompt = client._prompt("Return JSON.", [{"role": "user", "content": "x"}])
        cache_path = client._cache_path(prompt)
        assert cache_path is not None
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text('{"poisoned":')

        def fake_run(command, **_kwargs):
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text('{"answer": 9}')
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("common.llm.subprocess.run", side_effect=fake_run):
            response = client.complete(
                system="Return JSON.", messages=[{"role": "user", "content": "x"}]
            )

    assert parse_json_object(response.text) == {"answer": 9}
    assert client.cache_hits == 0


def test_codex_cli_client_retries_a_transient_network_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        client = CodexCLIClient(
            CodexCLIConfig(
                cache_dir=Path(directory) / "cache",
                transport_retries=1,
                transport_retry_delay_seconds=0.0,
            )
        )
        attempts = 0

        def fake_run(command, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return subprocess.CompletedProcess(
                    command,
                    1,
                    "",
                    "stream disconnected: Connection reset by peer",
                )
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text('{"answer": 11}')
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("common.llm.subprocess.run", side_effect=fake_run):
            response = client.complete(
                system="Return JSON.", messages=[{"role": "user", "content": "x"}]
            )

    assert parse_json_object(response.text) == {"answer": 11}
    assert attempts == 2
    assert client.calls == 2


def test_codex_cli_client_raises_transient_error_after_network_retries() -> None:
    client = CodexCLIClient(
        CodexCLIConfig(
            cache_dir=None,
            transport_retries=1,
            transport_retry_delay_seconds=0.0,
        )
    )

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            "failed to connect to WebSocket: Connection refused",
        )

    with patch("common.llm.subprocess.run", side_effect=fake_run) as mocked:
        try:
            client.complete(system="Return JSON.", messages=[{"role": "user", "content": "x"}])
            assert False, "expected a TransientLLMError"
        except TransientLLMError as exc:
            assert "Connection refused" in str(exc)

    assert mocked.call_count == 2
    assert client.calls == 2
