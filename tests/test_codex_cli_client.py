from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from evolving_agent.llm import CodexCLIClient, CodexCLIConfig, parse_json_object


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

        with patch("evolving_agent.llm.subprocess.run", side_effect=fake_run) as mocked:
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

        with patch("evolving_agent.llm.subprocess.run", side_effect=fake_run) as mocked:
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

        with patch("evolving_agent.llm.subprocess.run", side_effect=fake_run):
            response = client.complete(
                system="Return JSON.", messages=[{"role": "user", "content": "x"}]
            )

    assert parse_json_object(response.text) == {"answer": 9}
    assert client.cache_hits == 0
