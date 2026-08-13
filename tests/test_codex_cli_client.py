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
