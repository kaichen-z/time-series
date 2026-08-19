from __future__ import annotations

import math

from evolving_loop.data import Task
from evolving_loop.knowledge_base import TimeSeriesKnowledgeBase, diagnose


def _task(values, *, horizon=12, period=None) -> Task:
    return Task(
        task_id="knowledge-test",
        history_values=tuple(float(value) for value in values),
        future_values=(),
        prediction_length=horizon,
        frequency="daily",
        seasonal_period=str(period) if period else None,
        entity_name="synthetic",
    )


def test_extended_library_is_source_complete_and_substantially_larger() -> None:
    library = TimeSeriesKnowledgeBase()

    assert library.version == "setting2-tskb-2026-08-15"
    assert len(library.entries) == 90
    assert len(library.sources) == 48
    assert all(entry.source_ids for entry in library.entries)


def test_diagnostics_expose_new_count_and_multiscale_tags() -> None:
    intermittent = _task(([0] * 5 + [2]) * 30, horizon=24)
    profile = diagnose(intermittent)

    assert "intermittent" in profile.tags
    assert "count_like" in profile.tags
    assert "long_history" in profile.tags

    multiscale = _task(
        [
            20 + 3 * math.sin(2 * math.pi * index / 7) + 2 * math.sin(2 * math.pi * index / 30)
            for index in range(420)
        ],
        horizon=30,
        period=7,
    )
    assert "multiple_cycles" in diagnose(multiscale).tags


def test_retrieval_selects_specific_rules_and_keeps_tsfm_opt_in() -> None:
    library = TimeSeriesKnowledgeBase()
    task = _task(([0] * 7 + [3]) * 24, horizon=24)

    without_tsfm = library.retrieve(task, limit=20)
    with_tsfm = library.retrieve(task, limit=20, include_tsfm=True)

    assert "INTERMITTENT_TSB_OBSOLESCENCE" in without_tsfm.entry_ids
    assert not any(entry.category in {"tsfm", "neural_prior"} for entry in without_tsfm.entries)
    assert any(entry.category in {"tsfm", "neural_prior"} for entry in with_tsfm.entries)


def test_diagnostics_interpolate_nonfinite_public_history_gaps() -> None:
    task = _task([1.0, float("nan"), float("inf"), 4.0, 5.0], horizon=2)

    profile = diagnose(task)
    selection = TimeSeriesKnowledgeBase().retrieve(task)

    assert math.isfinite(profile.lag1_autocorrelation)
    assert math.isfinite(profile.trend_effect_over_horizon)
    assert selection.entries
