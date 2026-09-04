from __future__ import annotations

import inspect
from pathlib import Path

import numerical_agent.evolution.forecast_store as forecast_store
import pytest
from numerical_agent.evolution.forecast_store import CacheMissError, ForecastStore
from numerical_agent.evolution.module import MODULE_HEADER
from numerical_agent.evolution.portfolio import PolicyPortfolio
from numerical_agent.providers import RuntimeRegistry


HISTORY = (1.0, 2.0, 3.0, 4.0)


def test_shared_store_uses_the_planned_keyword_only_constructor(tmp_path, monkeypatch):
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def naive_last(history, horizon, frequency):
    """Return the final observed value."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )

    class NoopStatisticalRuntime:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def close(self):
            pass

    monkeypatch.setattr(
        forecast_store, "_IsolatedStatisticalRuntime", NoopStatisticalRuntime
    )
    store = ForecastStore(
        tmp_path / "cache",
        methods,
        None,
        PolicyPortfolio.flagship5(),
        RuntimeRegistry(),
        screening_hash="screen-hash",
        runtime_identity={},
    )
    store.close()
    signature = inspect.signature(ForecastStore)
    assert tuple(signature.parameters) == (
        "root",
        "module_path",
        "skills_path",
        "portfolio",
        "runtimes",
        "screening_hash",
        "runtime_identity",
        "statistical_time_budget_s",
        "statistical_failure_limit",
        "cache_only",
    )
    assert signature.parameters["screening_hash"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["runtime_identity"].kind is inspect.Parameter.KEYWORD_ONLY


def make_fixture_store(
    tmp_path: Path,
    monkeypatch,
    calls: list[str],
    *,
    cache_only: bool = False,
) -> ForecastStore:
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def naive_last(history, horizon, frequency):
    """Return the final observed value."""
    return [float(history[-1])] * horizon


def seasonal_naive(history, horizon, frequency):
    """Return a fixed seasonal leaf forecast."""
    return [100.0] * horizon
''',
        encoding="utf-8",
    )

    class CountingStatisticalRuntime:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def forecast(self, name, history, horizon, frequency):
            del frequency
            calls.append(name)
            if name == "naive_last":
                return (float(history[-1]),) * horizon
            if name == "seasonal_naive":
                return (100.0,) * horizon
            raise KeyError(name)

        def close(self):
            pass

    class CountingTimesFMRuntime:
        def supports(self, candidate):
            return candidate.method_id == "method_tsfm_0031"

        def forecast(self, candidate, history, horizon, frequency):
            del history, frequency
            calls.append("timesfm_2_5")
            assert candidate.method_id == "method_tsfm_0031"
            return (10.0,) * horizon

    monkeypatch.setattr(
        forecast_store, "_IsolatedStatisticalRuntime", CountingStatisticalRuntime
    )
    runtime = CountingTimesFMRuntime()
    return ForecastStore(
        tmp_path / "cache",
        methods,
        None,
        PolicyPortfolio.flagship5(),
        RuntimeRegistry({"timesfm": runtime, "tsfm_worker": runtime}),
        screening_hash="screen-hash",
        runtime_identity={},
        cache_only=cache_only,
    )


def test_shared_store_materializes_one_leaf_once(tmp_path, monkeypatch):
    calls = []
    store = make_fixture_store(tmp_path, monkeypatch, calls)
    try:
        first = store.forecast("naive_last", (1.0, 2.0), 2, "D")
        second = store.forecast("naive_last", (1.0, 2.0), 2, "D")
    finally:
        store.close()
    assert first == second == (2.0, 2.0)
    assert calls == ["naive_last"]
    assert (store.misses, store.hits) == (1, 1)


def test_shared_store_combined_reuses_materialized_leaves(tmp_path, monkeypatch):
    calls = []
    store = make_fixture_store(tmp_path, monkeypatch, calls)
    try:
        store.forecast("combined_timesfm_seasonal", HISTORY, 4, "D")
    finally:
        store.close()
    assert calls.count("timesfm_2_5") == 1
    assert calls.count("seasonal_naive") == 1


def test_shared_store_cache_only_never_executes_or_writes(tmp_path, monkeypatch):
    calls: list[str] = []
    store = make_fixture_store(tmp_path, monkeypatch, calls, cache_only=True)
    try:
        with pytest.raises(CacheMissError, match="cache-only"):
            store.forecast("naive_last", (1.0, 2.0), 2, "D")
    finally:
        store.close()

    assert calls == []
    assert store.hits == 0
    assert store.misses == 1
    assert list((tmp_path / "cache").iterdir()) == []
