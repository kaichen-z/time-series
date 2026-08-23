from __future__ import annotations

import json

import pytest

from common.llm import FakeLLMClient
from numerical_agent.evolution.execution import CRASHED, SUCCESS, Outcome, Task
from numerical_agent.evolution.filtering import FilterDictionary, FilterEntry
from numerical_agent.evolution.screening import (
    ApplicabilityClause,
    ApplicabilityPolicy,
    FeatureTest,
    ScreeningEntry,
    ScreeningPolicy,
)
from numerical_agent.evolution.screening_evolution import (
    ScreeningEvolutionError,
    apply_screening_response,
    evolve_screening_once,
    migrate_filter_dictionary,
    parse_screening_source,
    render_screening_source,
)


def _legacy() -> FilterDictionary:
    return FilterDictionary(
        (
            FilterEntry("stable", "statistical", "keep", (), "stable"),
            FilterEntry("special", "statistical", "specialized", ("intermittent",), "special"),
            FilterEntry("broken", "combined", "keep", (), "broken"),
            FilterEntry("timesfm", "tsfm", "keep", (), "tsfm"),
        )
    )


def test_migration_preserves_identity_status_and_legacy_and_rules() -> None:
    policy = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )

    assert [entry.name for entry in policy.entries] == ["stable", "special", "broken", "timesfm"]
    assert policy.get("special").status == "specialized"  # type: ignore[union-attr]
    assert policy.get("special").applicability == ApplicabilityPolicy(  # type: ignore[union-attr]
        (ApplicabilityClause(("intermittent",)),)
    )
    assert parse_screening_source(render_screening_source(policy)) == policy


def test_response_can_only_change_known_status_and_typed_applicability() -> None:
    parent = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )
    response = json.dumps(
        {
            "summary": "specialize the stable method to a periodic or intermittent regime",
            "actions": [
                {
                    "name": "stable",
                    "status": "specialized",
                    "any_of": [
                        {
                            "all_tags": ["intermittent"],
                            "feature_tests": [],
                        },
                        {
                            "all_tags": [],
                            "feature_tests": [
                                {
                                    "field": "periodicity_strength",
                                    "operator": ">=",
                                    "value": 0.7,
                                }
                            ],
                        },
                    ],
                    "reason": "two supported regimes",
                }
            ],
        }
    )

    child = apply_screening_response(parent, response, required_names=frozenset({"stable"}))

    assert child.get("stable") == ScreeningEntry(
        "stable",
        "statistical",
        "specialized",
        ApplicabilityPolicy(
            (
                ApplicabilityClause(("intermittent",)),
                ApplicabilityClause(
                    feature_tests=(FeatureTest("periodicity_strength", ">=", 0.7),)
                ),
            )
        ),
        "two supported regimes",
    )


def test_response_rejects_unknown_feature_and_missing_required_target() -> None:
    parent = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )
    response = json.dumps(
        {
            "summary": "invalid",
            "actions": [
                {
                    "name": "stable",
                    "status": "specialized",
                    "any_of": [{"all_tags": [], "feature_tests": [
                        {"field": "future", "operator": ">", "value": 0}
                    ]}],
                    "reason": "future leak",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="unsupported profile field"):
        apply_screening_response(parent, response, required_names=frozenset({"stable"}))
    with pytest.raises(ScreeningEvolutionError, match="must address required targets"):
        apply_screening_response(
            parent,
            json.dumps({"summary": "empty", "actions": []}),
            required_names=frozenset({"broken"}),
        )


def _tasks(prefix: str) -> tuple[Task, ...]:
    return tuple(
        Task(f"{prefix}-{index}", (1.0,) * 12, 2, "1 day", (1.0, 1.0))
        for index in range(2)
    )


def _outcomes(tasks: tuple[Task, ...]) -> tuple[Outcome, ...]:
    return tuple(
        row
        for task in tasks
        for row in (
            Outcome("stable", task.task_id, SUCCESS, mase=1.0, mae=1.0, smape=1.0),
            Outcome("special", task.task_id, SUCCESS, mase=0.8, mae=0.8, smape=0.8),
            Outcome("broken", task.task_id, CRASHED),
            Outcome("timesfm", task.task_id, SUCCESS, mase=1.2, mae=1.2, smape=1.2),
        )
    )


def test_evolution_prompt_is_train_only_and_accepts_reliability_improvement(tmp_path) -> None:
    parent = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )
    train, dev = _tasks("train"), _tasks("dev")
    agent = FakeLLMClient(
        [json.dumps({
            "summary": "remove a crashing candidate from selection",
            "actions": [{
                "name": "broken", "status": "repair", "any_of": [],
                "reason": "crashes on every Train task",
            }],
        })]
    )

    result = evolve_screening_once(
        parent,
        train,
        dev,
        _outcomes(train + dev),
        agent,
        generation=1,
        required_targets=("broken",),
        transcript_dir=tmp_path,
    )

    request = agent.calls[0]["messages"][0]["content"]
    assert "train-0" not in request
    assert "dev-0" not in request
    assert "future" not in request.lower()
    assert result.accepted
    assert result.child.get("broken").status == "repair"  # type: ignore[union-attr]
    assert result.gate.improved_dimensions == ("active_success_rate", "failure_exposure")
