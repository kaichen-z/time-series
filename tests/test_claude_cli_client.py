from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from common.llm import ClaudeCLIClient, ClaudeCLIConfig, parse_json_object


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
