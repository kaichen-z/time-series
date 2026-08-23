from __future__ import annotations

from pathlib import Path
import json
import shutil

import pytest

from numerical_agent.evolution.cache import CacheMissError, OutcomeCache
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


def test_corrupt_cache_entry_is_a_miss_and_is_replaced(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = OutcomeCache(root)
    method = parse_method(method_source())
    expected = cache.evaluate_method(method, (tasks()[0],), isolated=False)
    entry = next(root.glob("*.json"))
    entry.write_text("not-json", encoding="utf-8")

    actual = cache.evaluate_method(method, (tasks()[0],), isolated=False)

    assert actual == expected
    assert (cache.stats.hits, cache.stats.misses) == (0, 2)
    assert entry.read_text(encoding="utf-8").startswith("{")


def test_cache_record_copied_under_another_key_is_a_miss(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = OutcomeCache(root)
    method = parse_method(method_source())
    first, second = tasks()
    cache.evaluate_method(method, (first,), isolated=False)
    first_entry = next(root.glob("*.json"))
    second_key = cache.cache_key(method, second, isolated=False)
    shutil.copyfile(first_entry, root / f"{second_key}.json")

    outcome = cache.evaluate_method(method, (second,), isolated=False)

    assert outcome[0].task_id == second.task_id
    assert (cache.stats.hits, cache.stats.misses) == (0, 2)


def test_success_cache_record_without_complete_metrics_is_a_miss(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = OutcomeCache(root)
    method = parse_method(method_source())
    task = tasks()[0]
    cache.evaluate_method(method, (task,), isolated=False)
    entry = next(root.glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["outcome"]["mase"] = None
    entry.write_text(json.dumps(payload), encoding="utf-8")

    actual = cache.evaluate_method(method, (task,), isolated=False)

    assert actual[0].mase == 0.0
    assert (cache.stats.hits, cache.stats.misses) == (0, 2)


def test_forecast_required_for_diagnosis_refreshes_a_legacy_cache_record(
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

    actual = cache.evaluate_method(
        method, (task,), isolated=False, require_forecasts=True
    )

    assert actual[0].forecast == (3.0, 3.0)
    assert (cache.stats.hits, cache.stats.misses) == (0, 2)


def test_cache_only_lookup_never_executes_a_missing_method(tmp_path: Path) -> None:
    cache = OutcomeCache(tmp_path / "cache")
    method = parse_method(method_source())

    with pytest.raises(CacheMissError, match="cached_method.*t1"):
        cache.require_cached_method(
            method, (tasks()[0],), isolated=False, require_forecasts=True
        )

    assert not tuple((tmp_path / "cache").glob("*.json"))
    assert (cache.stats.hits, cache.stats.misses) == (0, 1)
