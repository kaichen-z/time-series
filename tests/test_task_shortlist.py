import pytest

from common.payload import canonical_json_bytes
from numerical_agent.evolution.filtering import FilterDictionary, FilterEntry
from numerical_agent.evolution.screening import (
    ApplicabilityClause,
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
    with pytest.raises(ValueError):
        TaskCandidateShortlistV1.from_payload({**value.to_payload(), "extra": 1})
    with pytest.raises(ValueError):
        TaskCandidateShortlistV1.from_payload({**value.to_payload(), "public_test_accessed": True})


def test_policy_has_exact_bounds_and_prior_rejects_nonfinite():
    assert TaskShortlistPolicyV1() == TaskShortlistPolicyV1(1, 6, 8, 10)
    with pytest.raises(ValueError):
        TaskShortlistPolicyV1(1, 5, 8, 10)
    with pytest.raises(ValueError):
        CandidatePriorV1("a", "statistical", float("nan"), 1.0, 1.0, ())
    assert TaskShortlistPolicyV1().canonical_bytes()
    assert len(TaskShortlistPolicyV1().fingerprint()) == 64
    assert len(CandidatePriorV1("a", "statistical", .5, 1, 1, ()).fingerprint()) == 64


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


def test_profile_applicability_changes_shortlist_and_all_reasons_are_canonical():
    names = ("anchor", "unsafe", "special", "missing", "r1", "r2", "r3", "r4", "r5", "r6")
    dictionary = FilterDictionary(tuple(
        FilterEntry(n, "statistical", "quarantine" if n == "unsafe" else "keep", (), "safe") for n in names if n != "missing"
    ))
    screening = ScreeningPolicy(tuple(
        ScreeningEntry(n, "statistical", "specialized" if n == "special" else "keep", ApplicabilityPolicy((ApplicabilityClause(("signed",)),)) if n == "special" else ApplicabilityPolicy(), "safe") for n in names if n not in {"unsafe", "missing"}
    ), ("anchor",))
    priors = tuple(CandidatePriorV1(n, "statistical", .9, 1, 1, ()) for n in names if n != "missing")
    result = build_task_candidate_shortlist(dictionary=dictionary, profile=_profile(), screening=screening,
        task_input_sha256="1" * 64, anchor_name="anchor", available_names=names, priors=priors, policy=TaskShortlistPolicyV1())
    assert result.candidate_names[:2] == ("anchor", "r1")
    assert result.exclusion_reasons[:3] == (("unsafe", "unsafe_status"), ("special", "not_applicable"), ("missing", "unavailable"))


def test_builder_requires_screening_and_selects_exact_target_with_dynamic_family_ties():
    names = ("anchor",) + tuple(f"c{i}" for i in range(9))
    families = ("statistical", "tsfm", "combined", "statistical", "tsfm", "combined", "statistical", "tsfm", "combined", "statistical")
    dictionary = FilterDictionary(tuple(FilterEntry(n, f, "keep", (), "safe") for n, f in zip(names, families)))
    screening = ScreeningPolicy(tuple(ScreeningEntry(n, f, "keep", ApplicabilityPolicy(), "safe") for n, f in zip(names, families)), ("anchor",))
    priors = tuple(CandidatePriorV1(n, f, .9, 1, 1, ()) for n, f in zip(names, families))
    result = build_task_candidate_shortlist(dictionary=dictionary, profile=_profile(), screening=screening,
        task_input_sha256="1" * 64, anchor_name="anchor", available_names=names, priors=priors, policy=TaskShortlistPolicyV1())
    assert len(result.candidate_names) == 8
    assert len(set(next(e.family for e in dictionary.entries if e.name == n) for n in result.candidate_names[:4])) == 3


def test_ranked_out_is_emitted_with_other_exclusion_reasons():
    names = ("anchor", "unsafe", "special", "missing") + tuple(f"r{i}" for i in range(10))
    dictionary = FilterDictionary(tuple(FilterEntry(n, "statistical", "quarantine" if n == "unsafe" else "keep", (), "safe") for n in names if n != "missing"))
    screening = ScreeningPolicy(tuple(ScreeningEntry(n, "statistical", "specialized" if n == "special" else "keep", ApplicabilityPolicy((ApplicabilityClause(("signed",)),)) if n == "special" else ApplicabilityPolicy(), "safe") for n in names if n not in {"unsafe", "missing"}), ("anchor",))
    priors = tuple(CandidatePriorV1(n, "statistical", .9, 1, 1, ()) for n in names if n != "missing")
    result = build_task_candidate_shortlist(dictionary=dictionary, profile=_profile(), screening=screening, task_input_sha256="1" * 64, anchor_name="anchor", available_names=names, priors=priors, policy=TaskShortlistPolicyV1())
    assert {reason for _, reason in result.exclusion_reasons} == {"unsafe_status", "not_applicable", "unavailable", "ranked_out"}


def test_from_payload_rejects_json_type_coercion_and_malformed_rows():
    prior = CandidatePriorV1("a", "statistical", .5, 1, 1, ()).to_payload()
    with pytest.raises(ValueError): CandidatePriorV1.from_payload({**prior, "success_rate": "0.5"})
    with pytest.raises(ValueError): CandidatePriorV1.from_payload({**prior, "candidate_name": 3})
    policy = TaskShortlistPolicyV1().to_payload()
    with pytest.raises(ValueError): TaskShortlistPolicyV1.from_payload({**policy, "schema_version": True})
    value = TaskCandidateShortlistV1(1, "1" * 64, "2" * 64, "3" * 64, ("anchor",), (), False, False).to_payload()
    with pytest.raises(ValueError): TaskCandidateShortlistV1.from_payload({**value, "candidate_names": ("anchor",)})
    with pytest.raises(ValueError): TaskCandidateShortlistV1.from_payload({**value, "task_input_sha256": 3})
    with pytest.raises(ValueError): TaskCandidateShortlistV1.from_payload({**value, "exclusion_reasons": [["x"]]})
