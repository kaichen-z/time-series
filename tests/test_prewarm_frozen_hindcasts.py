from __future__ import annotations

import pytest

import numerical_agent.prewarm_frozen_hindcasts as prewarm
from numerical_agent.prewarm_frozen_hindcasts import build_parser, shard


def test_prewarm_cli_exposes_only_frozen_history_cache_inputs():
    options = {action.dest for action in build_parser()._actions}
    assert {"repo", "screening_dir", "selector_dir", "start", "end"} <= options
    assert not {"codex_model", "generations", "llm_backend"} & options


def test_prewarm_store_keeps_the_legacy_empty_runtime_identity(tmp_path, monkeypatch):
    captured = {}

    class CapturingStore:
        def __init__(self, *args, **kwargs):
            del args
            captured.update(kwargs)
            self.hits = 0
            self.misses = 0

        def close(self):
            pass

    monkeypatch.setattr(prewarm, "ForecastStore", CapturingStore)
    monkeypatch.setattr(prewarm, "verify_frozen_policies", lambda *args: ("screen", None))
    monkeypatch.setattr(prewarm, "read_module", lambda path: object())
    monkeypatch.setattr(prewarm, "read_policy_file", lambda path: object())
    monkeypatch.setattr(prewarm, "build_filter_dictionary", lambda *args: object())
    monkeypatch.setattr(prewarm, "migrate_filter_dictionary", lambda *args, **kwargs: object())
    monkeypatch.setattr(prewarm, "_public_test_tasks", lambda *args: ())
    monkeypatch.setattr(prewarm, "shard", lambda *args: ())

    assert prewarm.main([
        "--repo", str(tmp_path),
        "--screening-dir", str(tmp_path / "screening"),
        "--selector-dir", str(tmp_path / "selector"),
        "--output-dir", str(tmp_path / "output"),
        "--split-file", str(tmp_path / "split.json"),
        "--tasks-file", str(tmp_path / "tasks.json"),
        "--hindcast-cache-dir", str(tmp_path / "cache"),
        "--start", "0",
        "--end", "1",
    ]) == 0
    assert captured["runtime_identity"] == {}


def test_shard_returns_exact_half_open_slice_and_rejects_bad_bounds():
    assert shard(("a", "b", "c", "d"), 1, 3) == ("b", "c")
    with pytest.raises(ValueError, match="bounds"):
        shard(("a", "b"), 2, 1)
    with pytest.raises(ValueError, match="bounds"):
        shard(("a", "b"), 0, 3)
