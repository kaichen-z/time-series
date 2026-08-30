from __future__ import annotations

from pathlib import Path
import json
import shutil

import pytest

from numerical_agent.evolution.cache import CacheError, CacheMissError, OutcomeCache
from numerical_agent.evolution.execution import SUCCESS, Task
from numerical_agent.evolution.module import parse_method


def method_source(value: float = 0.0) -> str:
    return f'''def cached_method(history, horizon, frequency):
    """Use when cache behavior is under test."""
    return [float(history[-1]) + {value}] * horizon
'''


def tasks() -> tuple[Task, ...]:
    return (
        Task("t1", (1.0, 2.0, 3.0), 2, "1 day", (3.0, 3.0)),
        Task("t2", (4.0, 5.0, 6.0), 2, "1 day", (6.0, 6.0)),
    )


def valid_cached_success_payload() -> dict[str, object]:
    return {
        "method": "cached_method",
        "task_id": "t1",
        "status": SUCCESS,
        "smae": 0.0,
        "srmse": 0.0,
        "smae_raw": 0.0,
        "srmse_raw": 0.0,
        "smae_clipped": False,
        "srmse_clipped": False,
        "smape": 0.0,
        "mae": 0.0,
        "mase": 0.0,
        "detail": "",
        "forecast": [3.0, 3.0],
    }


def test_cache_rejects_success_without_both_scaled_metrics() -> None:
    payload = valid_cached_success_payload()
    del payload["srmse"]

    with pytest.raises(CacheError, match="scaled metrics"):
        OutcomeCache.from_payload(payload)


def test_unchanged_method_and_tasks_hit_the_outcome_cache(tmp_path: Path) -> None:
    cache = OutcomeCache(tmp_path / "cache")
    method = parse_method(method_source())

    first = cache.evaluate_method(method, tasks(), isolated=False)
    second = cache.evaluate_method(method, tasks(), isolated=False)

    assert first == second
    assert all(outcome.status == SUCCESS for outcome in first)
    assert (cache.stats.hits, cache.stats.misses) == (2, 2)


def test_source_task_and_isolation_changes_use_distinct_cache_keys(tmp_path: Path) -> None:
    cache = OutcomeCache(tmp_path / "cache")
    original = parse_method(method_source())
    changed = parse_method(method_source(1.0))
    task = tasks()[0]

    cache.evaluate_method(original, (task,), isolated=False)
    cache.evaluate_method(changed, (task,), isolated=False)
    cache.evaluate_method(original, (Task("t1", task.history, 1, task.frequency, (3.0,)),), isolated=False)
    cache.evaluate_method(original, (task,), isolated=True)

    assert (cache.stats.hits, cache.stats.misses) == (0, 4)


def test_corrupt_existing_cache_entry_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = OutcomeCache(root)
    method = parse_method(method_source())
    cache.evaluate_method(method, (tasks()[0],), isolated=False)
    entry = next(root.glob("*.json"))
    entry.write_text("not-json", encoding="utf-8")

    with pytest.raises(CacheError, match="malformed active outcome cache row"):
        cache.evaluate_method(method, (tasks()[0],), isolated=False)
    assert entry.read_text(encoding="utf-8") == "not-json"


def test_cache_record_copied_under_another_key_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = OutcomeCache(root)
    method = parse_method(method_source())
    first, second = tasks()
    cache.evaluate_method(method, (first,), isolated=False)
    first_entry = next(root.glob("*.json"))
    second_key = cache.cache_key(method, second, isolated=False)
    shutil.copyfile(first_entry, root / f"{second_key}.json")

    with pytest.raises(CacheError, match="key mismatch"):
        cache.evaluate_method(method, (second,), isolated=False)


def test_success_cache_record_without_complete_metrics_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = OutcomeCache(root)
    method = parse_method(method_source())
    task = tasks()[0]
    cache.evaluate_method(method, (task,), isolated=False)
    entry = next(root.glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["outcome"]["mase"] = None
    entry.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CacheError, match="complete diagnostic metrics"):
        cache.evaluate_method(method, (task,), isolated=False)


def test_forecast_required_for_diagnosis_rejects_a_legacy_cache_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    cache = OutcomeCache(root)
    method = parse_method(method_source())
    task = tasks()[0]
    cache.evaluate_method(method, (task,), isolated=False)
    entry = next(root.glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["outcome"].pop("forecast", None)
    entry.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CacheError, match="forecast horizon mismatch"):
        cache.evaluate_method(method, (task,), isolated=False, require_forecasts=True)


def test_cache_round_trips_raw_infinite_scaled_tail_risk(tmp_path: Path) -> None:
    cache = OutcomeCache(tmp_path / "cache")
    method = parse_method(method_source())
    task = Task("zero-scale", (1.0, 2.0, 3.0), 2, "1 day", (0.0, 0.0))

    first = cache.evaluate_method(method, (task,), isolated=False)[0]
    second = cache.evaluate_method(method, (task,), isolated=False)[0]

    assert first.smae_clipped and first.srmse_clipped
    assert first.smae_raw == float("inf") and first.srmse_raw == float("inf")
    assert second == first


def test_cache_only_lookup_never_executes_a_missing_method(tmp_path: Path) -> None:
    cache = OutcomeCache(tmp_path / "cache")
    method = parse_method(method_source())

    with pytest.raises(CacheMissError, match="cached_method.*t1"):
        cache.require_cached_method(
            method, (tasks()[0],), isolated=False, require_forecasts=True
        )

    assert not tuple((tmp_path / "cache").glob("*.json"))
    assert (cache.stats.hits, cache.stats.misses) == (0, 1)
