from __future__ import annotations

import pytest

from numerical_agent.prewarm_frozen_hindcasts import build_parser, shard


def test_prewarm_cli_exposes_only_frozen_history_cache_inputs():
    options = {action.dest for action in build_parser()._actions}
    assert {"repo", "screening_dir", "selector_dir", "start", "end"} <= options
    assert not {"codex_model", "generations", "llm_backend"} & options


def test_shard_returns_exact_half_open_slice_and_rejects_bad_bounds():
    assert shard(("a", "b", "c", "d"), 1, 3) == ("b", "c")
    with pytest.raises(ValueError, match="bounds"):
        shard(("a", "b"), 2, 1)
    with pytest.raises(ValueError, match="bounds"):
        shard(("a", "b"), 0, 3)
