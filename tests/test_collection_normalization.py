from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from numerical_agent.collection.normalization import (
    find_duplicate_candidates,
    normalize_name,
)
from numerical_agent.collection.registry import load_method_cards


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "method_collection"
    / "duplicate_methods.jsonl"
)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Auto-ARIMA", "auto arima"),
        ("Damped Holt's trend", "damped holts trend"),
        ("  Seasonal   Naïve  ", "seasonal naïve"),
    ],
)
def test_normalize_name_matches_punctuation_case_and_spacing(
    left: str, right: str
) -> None:
    assert normalize_name(left) == normalize_name(right)


def test_alias_collision_is_reported_without_automatic_merge() -> None:
    methods = load_method_cards(FIXTURE)

    candidates = find_duplicate_candidates(methods)

    damped = next(
        candidate
        for candidate in candidates
        if {candidate.left_method_uid, candidate.right_method_uid}
        == {"method_damped_alias", "method_damped_canonical"}
    )
    assert "alias_collision" in damped.reasons
    assert damped.requires_manual_review is True
    assert {method.method_uid for method in methods} >= {
        "method_damped_alias",
        "method_damped_canonical",
    }


def test_wrapper_and_underlying_method_are_only_flagged_for_manual_review() -> None:
    methods = load_method_cards(FIXTURE)

    candidates = find_duplicate_candidates(methods)

    wrapper_pair = next(
        candidate
        for candidate in candidates
        if {candidate.left_method_uid, candidate.right_method_uid}
        == {"method_auto_arima", "method_arima"}
    )
    assert wrapper_pair.reasons == ("shared_source_claim",)
    assert wrapper_pair.requires_manual_review is True


def test_shared_textbook_does_not_flag_distinct_methods_with_one_common_token() -> None:
    methods = load_method_cards(FIXTURE)
    first = replace(
        methods[0],
        method_uid="method_naive_last",
        canonical_name="naive last",
        aliases=(),
    )
    second = replace(
        methods[1],
        method_uid="method_naive_mean",
        canonical_name="naive mean",
        aliases=(),
    )

    assert find_duplicate_candidates((first, second)) == ()


def test_forecast_token_inside_name_distinguishes_reconciliation_methods() -> None:
    methods = load_method_cards(FIXTURE)
    historical = replace(
        methods[0],
        method_uid="method_historical_proportions",
        canonical_name="top down historical proportions",
        aliases=(),
        definition_source_ids=("source_shared",),
    )
    forecast = replace(
        methods[1],
        method_uid="method_forecast_proportions",
        canonical_name="top down forecast proportions",
        aliases=(),
        definition_source_ids=("source_shared",),
    )

    assert find_duplicate_candidates((historical, forecast)) == ()


def test_duplicate_report_order_is_deterministic() -> None:
    methods = load_method_cards(FIXTURE)

    forward = find_duplicate_candidates(methods)
    reverse = find_duplicate_candidates(tuple(reversed(methods)))

    assert forward == reverse
