from dataclasses import FrozenInstanceError, replace

import pytest

from numerical_agent.evolution.champion import (
    ChampionContractError,
    ChampionRelease,
    FittedChampionPolicy,
    champion_fingerprint,
    parse_champion_recipe,
    parse_champion_release,
    preserve_parent,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
METRIC_POLICY_FINGERPRINT = "c" * 64


def assumption_payload(*, operator="weighted", assumption_id="seasonality"):
    return {
        "assumption_id": assumption_id,
        "candidate_name": "timesfm_2_5",
        "feature": "periodicity_strength",
        "direction": "above",
        "horizon_region": "full",
        "operator": operator,
        "rationale": "The history has a stable seasonal signal.",
        "failure_condition": "The seasonal signal disappears after the cutoff.",
    }


def valid_recipe_payload(**overrides):
    payload = {
        "name": "timesfm_seasonal_challenger",
        "kind": "weighted",
        "parents": ["timesfm_2_5", "seasonal_naive"],
        "fallback_parent": "timesfm_2_5",
        "assumptions": [assumption_payload()],
    }
    return {**payload, **overrides}


def valid_policy_payload(**overrides):
    payload = {
        "recipe": valid_recipe_payload(),
        "thresholds": [["seasonality", 0.75]],
        "weights": [0.6, 0.4],
        "overlay_alpha": 0.0,
        "correction_cap": 0.0,
        "horizon_split": 0.5,
    }
    return {**payload, **overrides}


def valid_release_payload(**overrides):
    payload = {
        "policy": valid_policy_payload(),
        "source_hashes": {"dictionary": HASH_A, "methods": HASH_B},
        "metric_policy_fingerprint": METRIC_POLICY_FINGERPRINT,
        "lineage": ["baseline_release", "weighted_candidate"],
    }
    return {**payload, **overrides}


def fixture_release(name):
    payload = valid_release_payload(lineage=[f"{name}_release"])
    return parse_champion_release(payload)


def test_recipe_supports_non_toto_combined_and_horizon_route():
    recipe = parse_champion_recipe({
        "name": "timesfm_seasonal_challenger",
        "kind": "horizon_route",
        "parents": ["timesfm_2_5", "seasonal_naive"],
        "fallback_parent": "timesfm_2_5",
        "assumptions": [assumption_payload(operator="horizon_route")],
    })
    assert recipe.parents == ("timesfm_2_5", "seasonal_naive")
    assert "toto_2_0" not in recipe.parents


@pytest.mark.parametrize("mutation", [
    lambda p: {**p, "unknown": 1},
    lambda p: {**p, "parents": ["same", "same"]},
    lambda p: {**p, "fallback_parent": "not_a_parent"},
])
def test_recipe_rejects_schema_or_namespace_drift(mutation):
    with pytest.raises(ChampionContractError):
        parse_champion_recipe(mutation(valid_recipe_payload()))


@pytest.mark.parametrize("mutation", [
    lambda p: {**p, "name": "not-a-python-identifier"},
    lambda p: {**p, "parents": ("timesfm_2_5", "seasonal_naive")},
    lambda p: {**p, "assumptions": [{**assumption_payload(), "operator": "route"}]},
    lambda p: {**p, "assumptions": [assumption_payload(), assumption_payload()]},
    lambda p: {**p, "kind": "select"},
    lambda p: {**p, "kind": "bounded_overlay"},
])
def test_recipe_rejects_noncanonical_types_assumption_drift_and_wrong_arity(mutation):
    with pytest.raises(ChampionContractError):
        parse_champion_recipe(mutation(valid_recipe_payload()))


def test_recipe_requires_exact_json_scalar_types():
    payload = valid_recipe_payload()
    payload["assumptions"] = [{**assumption_payload(), "direction": True}]

    with pytest.raises(ChampionContractError):
        parse_champion_recipe(payload)


@pytest.mark.parametrize("kind, parents", [
    ("select", ["timesfm_2_5"]),
    ("route", ["timesfm_2_5", "seasonal_naive"]),
    ("horizon_route", ["timesfm_2_5", "seasonal_naive"]),
    ("weighted", ["timesfm_2_5", "seasonal_naive"]),
    ("median", ["timesfm_2_5", "seasonal_naive"]),
    ("bounded_overlay", ["timesfm_2_5", "seasonal_naive"]),
])
def test_recipe_accepts_the_documented_operator_arities(kind, parents):
    recipe = parse_champion_recipe(valid_recipe_payload(
        kind=kind,
        parents=parents,
        fallback_parent=parents[0],
        assumptions=[assumption_payload(operator=kind)],
    ))

    assert recipe.kind == kind


def test_fitted_policy_requires_finite_normalized_weights_and_inactive_defaults():
    recipe = parse_champion_recipe(valid_recipe_payload())
    policy = FittedChampionPolicy(
        recipe=recipe,
        thresholds=(("seasonality", 0.75),),
        weights=(0.6, 0.4),
    )

    assert policy.weights == (0.6, 0.4)
    with pytest.raises(ChampionContractError):
        FittedChampionPolicy(recipe, (("seasonality", float("nan")),), (0.6, 0.4))
    with pytest.raises(ChampionContractError):
        FittedChampionPolicy(recipe, (("seasonality", 0.75),), (0.5, 0.4))
    with pytest.raises(ChampionContractError):
        FittedChampionPolicy(recipe, (("seasonality", 0.75),), (0.6, 0.4), 0.1)


def test_fitted_policy_canonicalizes_operator_specific_numeric_fields():
    route = parse_champion_recipe(valid_recipe_payload(
        kind="horizon_route",
        assumptions=[assumption_payload(operator="horizon_route")],
    ))
    policy = FittedChampionPolicy(route, (("seasonality", 0.75),), horizon_split=0.25)
    assert policy.horizon_split == 0.25

    with pytest.raises(ChampionContractError):
        FittedChampionPolicy(route, (("seasonality", 0.75),), horizon_split=0.5, weights=(0.5, 0.5))


def test_failed_child_can_return_exact_parent_object():
    parent = fixture_release("parent")
    child = fixture_release("child")
    assert preserve_parent(parent, child, accepted=False) is parent
    assert preserve_parent(parent, child, accepted=True) is child


def test_release_is_frozen_and_rejects_schema_or_hash_drift():
    release = parse_champion_release(valid_release_payload())
    assert isinstance(release, ChampionRelease)
    with pytest.raises(FrozenInstanceError):
        release.lineage = ()
    with pytest.raises(ChampionContractError):
        parse_champion_release({**valid_release_payload(), "unknown": True})
    with pytest.raises(ChampionContractError):
        parse_champion_release(valid_release_payload(metric_policy_fingerprint="not-a-hash"))
    with pytest.raises(ChampionContractError):
        parse_champion_release(valid_release_payload(source_hashes={"dictionary": "bad"}))


def test_payload_round_trip_and_fingerprint_are_canonical_but_semantic():
    release = parse_champion_release(valid_release_payload())
    reordered = {
        "lineage": ["baseline_release", "weighted_candidate"],
        "metric_policy_fingerprint": METRIC_POLICY_FINGERPRINT,
        "source_hashes": {"methods": HASH_B, "dictionary": HASH_A},
        "policy": {
            "horizon_split": 0.5,
            "correction_cap": 0.0,
            "overlay_alpha": 0.0,
            "weights": [0.6, 0.4],
            "thresholds": [["seasonality", 0.75]],
            "recipe": valid_recipe_payload(),
        },
    }
    reparsed = parse_champion_release(reordered)

    assert reparsed == release
    assert parse_champion_release(release.to_payload()) == release
    assert champion_fingerprint(reordered) == champion_fingerprint(release)
    assert champion_fingerprint(release) != champion_fingerprint(replace(
        release,
        policy=replace(release.policy, recipe=replace(
            release.policy.recipe, parents=("seasonal_naive", "timesfm_2_5")
        )),
    ))
    assert champion_fingerprint(release) != champion_fingerprint(replace(
        release,
        policy=replace(release.policy, weights=(0.5, 0.5)),
    ))
    assert champion_fingerprint(release) != champion_fingerprint(replace(
        release, lineage=("another_release",)
    ))
    assert champion_fingerprint({"metric_policy": {"cap": 5.0}}) != champion_fingerprint(
        {"metric_policy": {"cap": 4.0}}
    )
