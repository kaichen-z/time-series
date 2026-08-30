from __future__ import annotations

import json
import math
from inspect import signature

import pytest

from common.llm import FakeLLMClient
from numerical_agent.evolution.morphology import (
    AssumptionGrounding,
    MorphologyCard,
    MorphologyError,
    MorphologyInputError,
    MorphologyObservation,
    MorphologyReasoner,
    MorphologyToolCall,
)
from numerical_agent.evolution.morphology_credit import assign_tool_call_credit


def _history() -> list[float]:
    return [10.0 + math.sin(2.0 * math.pi * index / 7.0) for index in range(84)]


def _tool(call_id: str, *, start: int, end: int, tool: str = "detect_periodicity") -> str:
    return json.dumps(
        {
            "action": "tool",
            "call_id": call_id,
            "tool": tool,
            "window": {"start": start, "end": end},
        }
    )


def _final(
    *,
    call_ids: list[str] | None = None,
    candidates: list[str] | None = None,
    confidence: float = 0.8,
) -> str:
    return json.dumps(
        {
            "action": "final",
            "short_term": "The recent half retains the supported weekly rhythm.",
            "long_term": "The full history supports a stable weekly cycle.",
            "assumptions": [
                {
                    "assumption_id": "weekly_cycle",
                    "kind": "seasonality",
                    "claim": "The supported weekly cycle will persist over the horizon.",
                    "supporting_call_ids": call_ids or ["broad_period", "recent_period"],
                    "failure_condition": "The phase changes after the cutoff.",
                    "candidate_names": candidates or ["seasonal_naive", "toto_2_0"],
                    "prior_confidence": confidence,
                }
            ],
        },
        allow_nan=True,
    )


def _reason(responses: list[str], **kwargs: object):
    return MorphologyReasoner(FakeLLMClient(responses), **kwargs).reason(
        history=_history(),
        frequency="D",
        horizon=3,
        active_names=("seasonal_naive", "toto_2_0"),
        families={"seasonal_naive": "statistical", "toto_2_0": "tsfm"},
    )


def test_reasoner_returns_immutable_grounded_card_with_canonical_fingerprint() -> None:
    card = _reason(
        [
            _tool("broad_period", start=0, end=84),
            _tool("recent_period", start=42, end=84),
            _final(),
        ],
        max_turns=3,
        max_tool_calls=4,
    )

    assert card.assumption_call_ids("weekly_cycle") == ("broad_period", "recent_period")
    assert tuple(item.call_id for item in card.observations) == ("broad_period", "recent_period")
    assert card.fingerprint == _reason(
        [
            _tool("broad_period", start=0, end=84),
            _tool("recent_period", start=42, end=84),
            _final(),
        ],
        max_turns=3,
        max_tool_calls=4,
    ).fingerprint
    with pytest.raises(AttributeError):
        card.short_term = "mutated"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("responses", "match"),
    [
        ([_tool("bad_tool", start=0, end=84, tool="forecast")], "unknown reviewed tool"),
        ([_tool("bad_window", start=3, end=3)], "invalid tool window"),
        ([_tool("bad_window", start=0, end=85)], "invalid tool window"),
        ([_tool("broad_period", start=0, end=84, tool="detect_periodicity",)[:-1] + ', "extra": true}'], "schema"),
    ],
)
def test_reasoner_rejects_unknown_tools_invalid_windows_and_schema_drift(
    responses: list[str], match: str
) -> None:
    with pytest.raises(MorphologyError, match=match):
        _reason(responses)


def test_reasoner_rejects_invented_or_duplicate_call_ids() -> None:
    with pytest.raises(MorphologyError, match="unknown call id"):
        _reason([_tool("broad_period", start=0, end=84), _tool("recent_period", start=42, end=84), _final(call_ids=["broad_period", "invented"])])
    with pytest.raises(MorphologyError, match="duplicate tool call id"):
        _reason([_tool("broad_period", start=0, end=84), _tool("broad_period", start=42, end=84)])
    with pytest.raises(MorphologyError, match="duplicate supporting call id"):
        _reason([_tool("broad_period", start=0, end=84), _tool("recent_period", start=42, end=84), _final(call_ids=["broad_period", "broad_period"])])


def test_reasoner_rejects_inactive_candidates_and_non_finite_final_values() -> None:
    with pytest.raises(MorphologyError, match="inactive candidate"):
        _reason([_tool("broad_period", start=0, end=84), _tool("recent_period", start=42, end=84), _final(candidates=["invented_candidate"])])
    with pytest.raises(MorphologyError, match="finite"):
        _reason([_tool("broad_period", start=0, end=84), _tool("recent_period", start=42, end=84), _final(confidence=float("nan"))])


def test_reasoner_requires_distinct_broad_and_recent_inspections_before_finalization() -> None:
    with pytest.raises(MorphologyError, match="broad and recent"):
        _reason([_tool("broad_period", start=0, end=84), _final()])
    with pytest.raises(MorphologyError, match="broad and recent"):
        _reason([_tool("recent_period", start=42, end=84), _final()])


def test_reasoner_enforces_turn_and_tool_call_budgets() -> None:
    with pytest.raises(MorphologyError, match="tool-call budget"):
        _reason(
            [
                _tool("broad_period", start=0, end=84),
                _tool("recent_period", start=42, end=84),
            ],
            max_turns=3,
            max_tool_calls=1,
        )
    with pytest.raises(MorphologyError, match="turn budget"):
        _reason(
            [
                _tool("broad_period", start=0, end=84),
                _tool("recent_period", start=42, end=84),
            ],
            max_turns=1,
            max_tool_calls=4,
        )


def test_directly_constructed_artifacts_freeze_nested_collections_and_validate_types() -> None:
    broad_call = MorphologyToolCall("broad", "detect_periodicity", 0, 8)
    recent_call = MorphologyToolCall("recent", "detect_periodicity", 4, 8)
    supporting_ids = ["broad", "recent"]
    candidate_names = ["seasonal_naive"]
    assumption = AssumptionGrounding(
        "weekly_cycle",
        "seasonality",
        "Weekly pattern persists.",
        "The phase changes.",
        supporting_ids,
        candidate_names,
        0.8,
    )
    card = MorphologyCard(
        "Recent weekly pattern.",
        "Full-history weekly pattern.",
        (broad_call, recent_call),
        (
            MorphologyObservation(broad_call, {"strength": 0.8}),
            MorphologyObservation(recent_call, {"strength": 0.7}),
        ),
        [assumption],
    )
    fingerprint = card.fingerprint

    supporting_ids.append("invented")
    candidate_names.append("invented_candidate")

    assert isinstance(card.assumptions, tuple)
    assert card.assumptions[0].supporting_call_ids == ("broad", "recent")
    assert card.assumptions[0].candidate_names == ("seasonal_naive",)
    assert card.fingerprint == fingerprint
    with pytest.raises(MorphologyError, match="AssumptionGrounding"):
        MorphologyCard("short", "long", (), (), [object()])


def test_model_prompt_spells_out_the_complete_final_contract() -> None:
    client = FakeLLMClient([_tool("bad_tool", start=0, end=84, tool="forecast")])

    with pytest.raises(MorphologyError, match="unknown reviewed tool"):
        MorphologyReasoner(client).reason(
            history=_history(),
            frequency="D",
            horizon=3,
            active_names=("seasonal_naive", "toto_2_0"),
            families={"seasonal_naive": "statistical", "toto_2_0": "tsfm"},
        )

    system = client.calls[0]["system"]
    assert "assumption_id, kind, claim, failure_condition, supporting_call_ids, candidate_names, prior_confidence" in system
    assert "seasonality, trend, intermittency, regime, noise, level" in system
    assert "finite" in system and "[0, 1]" in system
    assert "active candidate" in system
    assert "executed call IDs" in system
    assert "full-history" in system and "recent" in system


@pytest.mark.parametrize(
    ("response", "expected_start", "expected_end"),
    (
        (
            '{"action":"tool","call_id":"real_luna","tool":"detect_trend",'
            '"window":{"start_inclusive":0,"end_exclusive":84}}',
            0,
            84,
        ),
        (
            '{"action":"tool","call_id":"real_sol","tool":"detect_trend",'
            '"window":{"start_inclusive":42,"end_exclusive":84}}',
            42,
            84,
        ),
        (
            '{"action":"tool","call_id":"real_terra","tool":"detect_trend",'
            '"window":[0,84]}',
            0,
            84,
        ),
    ),
)
def test_window_prompt_gives_the_canonical_contract_without_loosening_real_response_parsing(
    response: str, expected_start: int, expected_end: int
) -> None:
    client = FakeLLMClient([response])

    with pytest.raises(MorphologyError, match="tool window schema drift"):
        MorphologyReasoner(client).reason(
            history=_history(),
            frequency="D",
            horizon=3,
            active_names=("seasonal_naive", "toto_2_0"),
            families={"seasonal_naive": "statistical", "toto_2_0": "tsfm"},
        )

    system = client.calls[0]["system"]
    initial = client.calls[0]["messages"][0]["content"]
    prompt = system + "\n" + initial

    assert (
        '{"action":"tool","call_id":"full_window","tool":"detect_trend",'
        '"window":{"start":0,"end":N}}'
    ) in prompt
    assert (
        '{"action":"tool","call_id":"recent_window","tool":"detect_trend",'
        '"window":{"start":K,"end":N}}'
    ) in prompt
    assert "start is inclusive" in prompt
    assert "end is exclusive" in prompt
    assert "never an array" in prompt
    assert "only keys are start and end" in prompt
    assert "start_inclusive" not in prompt
    assert "end_exclusive" not in prompt
    assert expected_start < expected_end


def test_reasoner_rejects_duplicate_json_keys_at_action_and_assumption_levels() -> None:
    duplicate_action = (
        '{"action":"tool","action":"tool","call_id":"broad_period",'
        '"tool":"detect_periodicity","window":{"start":0,"end":84}}'
    )
    with pytest.raises(MorphologyError, match="duplicate JSON key"):
        _reason([duplicate_action])

    duplicate_assumption = _final().replace(
        '"kind": "seasonality",', '"kind": "seasonality", "kind": "noise",'
    )
    with pytest.raises(MorphologyError, match="duplicate JSON key"):
        _reason(
            [
                _tool("broad_period", start=0, end=84),
                _tool("recent_period", start=42, end=84),
                duplicate_assumption,
            ]
        )


def test_whitespace_normalized_candidate_and_family_names_fail_with_typed_error() -> None:
    with pytest.raises(MorphologyInputError, match="families"):
        MorphologyReasoner(FakeLLMClient([])).reason(
            history=_history(),
            frequency="D",
            horizon=3,
            active_names=(" seasonal_naive ", "toto_2_0"),
            families={"seasonal_naive": "statistical", "toto_2_0": "tsfm"},
        )


def test_train_credit_is_marginal_after_each_grounded_call_and_cannot_mutate_card() -> None:
    card = _reason(
        [
            _tool("broad_period", start=0, end=84),
            _tool("recent_period", start=42, end=84),
            _final(),
        ],
        max_turns=3,
        max_tool_calls=4,
    )
    fingerprint = card.fingerprint

    trace = assign_tool_call_credit(
        card,
        split="train",
        future_truth=(1.0, 1.0),
        forecasts_by_call_ids={
            frozenset(): (2.0, 2.0),
            frozenset({"broad_period"}): (1.0, 2.0),
            frozenset({"broad_period", "recent_period"}): (1.0, 1.0),
        },
    )

    assert tuple(item.call_id for item in trace.credits) == ("broad_period", "recent_period")
    assert trace.credits[0].smae_improvement == pytest.approx(0.5)
    assert trace.credits[0].srmse_improvement == pytest.approx(1.0 - math.sqrt(0.5))
    assert trace.credits[1].smae_improvement == pytest.approx(0.5)
    assert trace.credits[1].srmse_improvement == pytest.approx(math.sqrt(0.5))
    assert card.fingerprint == fingerprint
    with pytest.raises(AttributeError):
        card.assumptions = ()  # type: ignore[misc]


def test_credit_refuses_to_learn_outside_train_before_reading_future_truth() -> None:
    card = _reason(
        [
            _tool("broad_period", start=0, end=84),
            _tool("recent_period", start=42, end=84),
            _final(),
        ],
        max_turns=3,
        max_tool_calls=4,
    )

    trace = assign_tool_call_credit(
        card,
        split="frozen",
        future_truth=object(),
        forecasts_by_call_ids={},
    )

    assert not trace.learning_enabled
    assert trace.reason == "non_train_split"
    assert trace.credits == ()
    assert "future_truth" not in signature(MorphologyReasoner.reason).parameters
