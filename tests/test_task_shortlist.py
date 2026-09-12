import hashlib

import pytest

from common.payload import canonical_json_bytes
from numerical_agent.evolution.filtering import FilterDictionary, FilterEntry
from numerical_agent.evolution.screening import (
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
    TaskProfile,
)
from numerical_agent.evolution.task_shortlist import (
    CandidatePriorV1,
    TaskCandidateShortlistV1,
    TaskShortlistPolicyV1,
    build_task_candidate_shortlist,
)


def test_shortlist_contract_is_canonical_and_anchor_first():
    value = TaskCandidateShortlistV1(
        1, "1" * 64, "2" * 64, "3" * 64,
        ("anchor", "a", "b", "c", "d", "e"), (), False, False,
    )
    assert value.candidate_names[0] == "anchor"
    assert TaskCandidateShortlistV1.from_payload(value.to_payload()) == value
    assert value.canonical_bytes() == canonical_json_bytes(value.to_payload())


def test_policy_has_exact_bounds_and_prior_rejects_nonfinite():
    assert TaskShortlistPolicyV1() == TaskShortlistPolicyV1(1, 6, 8, 10)
    with pytest.raises(ValueError):
        TaskShortlistPolicyV1(1, 5, 8, 10)
    with pytest.raises(ValueError):
        CandidatePriorV1("a", "statistical", float("nan"), 1.0, 1.0, ())


def _profile(**changes):
    values = dict(
        task_id="task", frequency="daily", history_length=100, horizon=8,
        zero_fraction=0.0, signed=False, integer_valued=False,
        trend_direction="flat", trend_strength=0.0, periodicity_periods=(),
        periodicity_strength=0.0, periodicity_confidence=0.0, outlier_fraction=0.0,
        noise_relative_scale=0.1, likely_stationary=True, stationarity_score=1.0,
        recent_regime_start=None, recent_regime_confidence=0.0,
        intermittency_adi=1.0, intermittency_cv2=0.0,
    )
    values.update(changes)
    return TaskProfile(**values)


def _dictionary(names):
    return FilterDictionary(tuple(FilterEntry(n, "statistical", "keep", (), "safe") for n in names))


def _screening(names):
    return ScreeningPolicy(
        tuple(ScreeningEntry(n, "statistical", "keep", ApplicabilityPolicy(), "safe") for n in names),
        (names[0],),
    )


def test_selection_is_history_conditioned_and_underfill_excludes_unsafe():
    names = ("anchor", "a", "b", "c", "d", "unsafe")
    dictionary = FilterDictionary(tuple(
        [FilterEntry(n, "statistical", "quarantine", (), "unsafe") if n == "unsafe"
         else FilterEntry(n, "statistical", "keep", (), "safe") for n in names]
    ))
    screening = _screening(names)
    priors = tuple(CandidatePriorV1(n, "statistical", 0.9, 1.0, 1.0, ()) for n in names)
    result = build_task_candidate_shortlist(
        dictionary=dictionary, profile=_profile(), screening=screening,
        task_input_sha256="1" * 64, anchor_name="anchor", available_names=names,
        priors=priors, policy=TaskShortlistPolicyV1(),
    )
    assert result.candidate_names == ("anchor", "a", "b", "c", "d")
    assert result.shortlist_underfilled
    assert ("unsafe", "unsafe_status") in result.exclusion_reasons
