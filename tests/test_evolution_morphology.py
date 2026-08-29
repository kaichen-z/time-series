from __future__ import annotations

import json
import math

import pytest

from common.llm import FakeLLMClient
from numerical_agent.evolution.morphology import MorphologyError, MorphologyReasoner


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
