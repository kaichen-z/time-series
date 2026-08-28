"""Tests for the five foundation-model seeds and the adapters behind them.

Nothing here loads a checkpoint: the point is the wiring around the models -- preconditions,
the process-wide cache, and config validation -- which is what breaks silently.
"""
from __future__ import annotations

import pytest

from common.tsfm import (
    BackboneUnavailableError,
    Chronos2Config,
    MoiraiConfig,
    TotoConfig,
    resolve_device,
    shared_forecaster,
)
from numerical_agent.evolution import foundation_methods
from numerical_agent.evolution.module import read_module
from numerical_agent.evolution.seed import FOUNDATION_SEEDS

FOUNDATION_KINDS = ("chronos_bolt", "chronos_2", "timesfm_2_5", "toto", "moirai")


def test_the_seeded_names_are_exactly_the_written_methods() -> None:
    """A name in the catalog mapping with no method behind it seeds a hole in the module."""
    module = read_module(foundation_methods.__file__)

    assert set(module.names()) == set(FOUNDATION_SEEDS.values())


def test_every_foundation_method_declines_a_history_too_short_to_condition_on() -> None:
    module = read_module(foundation_methods.__file__)

    for name in module.names():
        method = getattr(foundation_methods, name)
        with pytest.raises(foundation_methods.NotApplicable, match="context"):
            method([1.0, 2.0, 3.0], 4, "1 hour")


def test_timesfm_declines_a_horizon_past_its_quantile_head() -> None:
    with pytest.raises(foundation_methods.NotApplicable, match="1024"):
        foundation_methods.timesfm_2_5_zero_shot([1.0] * 64, 2048, "1 hour")


def test_a_model_is_loaded_once_per_process() -> None:
    """run_module calls each method once per task; reloading a checkpoint each time would dominate."""
    first = shared_forecaster("toto")

    assert shared_forecaster("toto") is first


def test_an_unknown_forecaster_is_refused_by_name() -> None:
    with pytest.raises(BackboneUnavailableError, match="unknown forecaster"):
        shared_forecaster("nonexistent_model")


def test_every_seeded_kind_has_a_forecaster() -> None:
    for kind in FOUNDATION_KINDS:
        assert shared_forecaster(kind) is not None


def test_an_explicit_device_is_passed_through_untouched() -> None:
    assert resolve_device("cuda:3") == "cuda:3"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")


def test_a_nonsense_limit_is_refused_at_construction() -> None:
    """A zero context window fails at the first task otherwise, deep inside a library call."""
    with pytest.raises(ValueError, match="context"):
        Chronos2Config(max_context=0)
    with pytest.raises(ValueError, match="context"):
        MoiraiConfig(max_context=0)
    with pytest.raises(ValueError, match="sample counts"):
        TotoConfig(num_samples=0)


def test_the_bootstrap_hands_the_foundation_models_in_rather_than_writing_them() -> None:
    """A model cannot guess each package's calling convention; a wrong guess reads as a crash."""
    import argparse

    from numerical_agent.run_bootstrap import DEFAULT_CATALOG, _definitions

    written, preset = _definitions(
        argparse.Namespace(definitions=None, catalog=DEFAULT_CATALOG)
    )

    assert len(written) == 93
    assert {method.name for method in preset} == set(FOUNDATION_SEEDS.values())
    assert not {str(d["name"]) for d in written} & set(FOUNDATION_SEEDS.values())


def test_a_preset_method_survives_a_bootstrap_that_writes_nothing(tmp_path) -> None:
    """The seed must still hold the foundation models when every definition fails to parse."""
    from numerical_agent.evolution import bootstrap
    from numerical_agent.evolution.module import read_module

    class RefusingLLM:
        def complete(self, *, system, messages, temperature=0.0):
            raise AssertionError("no definition should have been sent")

    preset = read_module(foundation_methods.__file__).methods
    module = bootstrap(tmp_path / "seeded", [], RefusingLLM(), preset=preset)

    assert set(module.names()) == set(FOUNDATION_SEEDS.values())
