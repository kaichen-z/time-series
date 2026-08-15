from __future__ import annotations

import pytest

from evolving_agent.evolution_core.contracts import EvolutionConfig, MetricSpec


def test_evolution_config_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError, match="generations"):
        EvolutionConfig(generations=0)
    with pytest.raises(ValueError, match="children_per_generation"):
        EvolutionConfig(children_per_generation=0)
    with pytest.raises(ValueError, match="screen_train_items"):
        EvolutionConfig(screen_train_items=0)
    with pytest.raises(ValueError, match="acceptance_margin"):
        EvolutionConfig(acceptance_margin=-0.1)


def test_metric_spec_orders_minimized_scores() -> None:
    metric = MetricSpec(name="smape", objective="minimize")

    assert metric.better(10.0, 12.0)
    assert not metric.better(12.0, 10.0)
    assert not metric.better(10.0, 10.0)


def test_metric_spec_orders_maximized_scores_with_margin() -> None:
    metric = MetricSpec(name="reward", objective="maximize")

    assert metric.better(0.7, 0.5, margin=0.1)
    assert not metric.better(0.6, 0.5, margin=0.1)


def test_metric_spec_rejects_invalid_objective() -> None:
    with pytest.raises(ValueError, match="objective"):
        MetricSpec(name="smape", objective="sideways")  # type: ignore[arg-type]
