from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest

from numerical_agent.evolution.cache import OutcomeCache
from numerical_agent.evolution.execution import SUCCESS, Task, run_module
from numerical_agent.evolution.module import MODULE_HEADER, parse_method
from numerical_agent.evolution.prompts import BOOTSTRAP_SYSTEM, TARGETWISE_MUTATE_SYSTEM


def _skill_module():
    from numerical_agent.evolution import analysis_skills

    return analysis_skills


def test_skill_api_is_history_only() -> None:
    skills = _skill_module()

    for name in skills.ANALYSIS_SKILL_NAMES:
        parameters = tuple(inspect.signature(getattr(skills, name)).parameters)
        assert "future" not in parameters
        assert "labels" not in parameters
        assert parameters in {("history",), ("history", "frequency")}


def test_periodicity_reports_a_weekly_candidate() -> None:
    skills = _skill_module()
    history = [10.0 + math.sin(2.0 * math.pi * index / 7.0) for index in range(84)]

    result = skills.detect_periodicity(history, "1 day")

    assert result["candidate_periods"][0] == 7
    assert result["strength"] > 0.8
    assert 0.0 <= result["confidence"] <= 1.0


def test_outlier_skill_reports_but_does_not_modify_history() -> None:
    skills = _skill_module()
    history = [10.0] * 20
    history[11] = 100.0
    original = tuple(history)

    result = skills.detect_outliers(history)

    assert result["indices"] == [11]
    assert result["scores"][0] > 3.5
    assert tuple(history) == original


def test_trend_change_point_and_recent_regime_are_detected() -> None:
    skills = _skill_module()
    trend = [float(index) for index in range(60)]
    shifted = [2.0] * 40 + [20.0] * 24

    trend_result = skills.detect_trend(trend)
    changes = skills.detect_change_points(shifted)
    regime = skills.detect_recent_regime(shifted)

    assert trend_result["direction"] == "up"
    assert trend_result["strength"] > 0.8
    assert abs(changes["indices"][0] - 40) <= 2
    assert regime["regime_start"] is not None
    assert abs(regime["regime_start"] - 40) <= 2


def test_intermittency_noise_and_stationarity_profiles_are_bounded() -> None:
    skills = _skill_module()
    intermittent = [0.0, 0.0, 3.0, 0.0, 0.0, 2.0] * 10
    stationary = [1.0, 2.0, 1.5, 2.5] * 20

    intermittency = skills.detect_intermittency(intermittent)
    noise = skills.estimate_noise_scale(stationary)
    stationarity = skills.assess_stationarity(stationary)
    profile = skills.analyze_series(stationary, "1 day")

    assert intermittency["is_intermittent"] is True
    assert intermittency["zero_fraction"] > 0.6
    assert noise["robust_scale"] > 0.0
    assert stationarity["likely_stationary"] is True
    assert set(profile) == {
        "periodicity",
        "outliers",
        "trend",
        "change_points",
        "intermittency",
        "noise",
        "stationarity",
        "recent_regime",
    }


def test_evolved_method_can_call_injected_history_only_skill(tmp_path: Path) -> None:
    method_path = tmp_path / "methods.py"
    method_path.write_text(
        MODULE_HEADER
        + '''

def periodic_skill_method(history, horizon, frequency):
    """Use when a detected historical period should persist."""
    result = detect_periodicity(history, frequency)
    period = result["candidate_periods"][0]
    return [float(history[-period + (step % period)]) for step in range(horizon)]
''',
        encoding="utf-8",
    )
    history = tuple(float(index % 7) for index in range(70))
    task = Task("weekly", history, 7, "1 day", tuple(float(index) for index in range(7)))

    outcomes, reports = run_module(method_path, (task,), isolated=False)

    assert outcomes[0].status == SUCCESS
    assert outcomes[0].forecast == tuple(float(index) for index in range(7))
    assert reports[0].method == "periodic_skill_method"


def test_outcome_cache_identity_includes_skill_source(tmp_path: Path) -> None:
    method = parse_method(
        '''def alpha(history, horizon, frequency):
    """Use when checking the analysis-skill cache dependency."""
    return [float(history[-1])] * horizon
'''
    )
    task = Task("t", (1.0, 2.0), 1, "1 day", (2.0,))
    first_skills = tmp_path / "skills_a.py"
    second_skills = tmp_path / "skills_b.py"
    first_skills.write_text("SKILL_API_VERSION = 1\n", encoding="utf-8")
    second_skills.write_text("SKILL_API_VERSION = 2\n", encoding="utf-8")

    first = OutcomeCache(tmp_path / "cache", skills_path=first_skills)
    second = OutcomeCache(tmp_path / "cache", skills_path=second_skills)

    assert first.cache_key(method, task, isolated=False) != second.cache_key(
        method, task, isolated=False
    )


def test_skill_source_must_not_define_a_forecast_or_read_external_state(tmp_path: Path) -> None:
    from numerical_agent.evolution.analysis_skills import validate_skill_source

    with pytest.raises(ValueError, match="forbidden skill name"):
        validate_skill_source("def forecast(history, frequency):\n    return []\n")
    with pytest.raises(ValueError, match="forbidden name"):
        validate_skill_source(
            "def detect_periodicity(history, frequency):\n"
            "    return open('/tmp/leak').read()\n"
        )


def test_forecasting_prompts_expose_the_history_only_skill_api() -> None:
    for prompt in (BOOTSTRAP_SYSTEM, TARGETWISE_MUTATE_SYSTEM):
        assert "detect_periodicity" in prompt
        assert "detect_outliers" in prompt
        assert "history-only" in prompt
        assert "do not reimplement" in prompt.lower()
