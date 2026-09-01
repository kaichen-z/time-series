"""Strict structural proposal and trusted host-expansion regressions."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pytest

from common.llm import FakeLLMClient
from numerical_agent.dictionary import MethodDefinition, ToolDictionary
from numerical_agent.evolution import champion_proposal as proposal_module
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    EvolutionAssumption,
    champion_fingerprint,
)
from numerical_agent.evolution.champion_evidence import ChampionTaskRow, ProposerEvidence
from numerical_agent.evolution.champion_proposal import (
    CORRECTION_CAP_GRID,
    HORIZON_SPLIT_GRID,
    OVERLAY_ALPHA_GRID,
    WEIGHT_GRID,
    ChampionProposalError,
    expand_recipe,
    parse_champion_response,
    propose_champion_recipes,
)
from numerical_agent.evolution.screening import TaskProfile


def _inventory() -> ToolDictionary:
    return ToolDictionary(
        "champion_inventory",
        None,
        0,
        (
            MethodDefinition(
                "timesfm_2_5",
                "foundation",
                "reviewed foundation forecast",
                status="accepted",
            ),
            MethodDefinition(
                "seasonal_naive",
                "statistical",
                "reviewed seasonal forecast",
                status="accepted",
            ),
            MethodDefinition(
                "toto_2_0",
                "foundation",
                "reviewed foundation anchor",
                status="accepted",
            ),
        ),
    )


INVENTORY = _inventory()
PARENT = ChampionRecipe(
    name="current_champion",
    kind="select",
    parents=("timesfm_2_5",),
    fallback_parent="timesfm_2_5",
    assumptions=(
        EvolutionAssumption(
            assumption_id="stable_history",
            candidate_name="timesfm_2_5",
            feature="history_length",
            direction="above",
            horizon_region="full",
            operator="select",
            rationale="Long histories support the current forecast.",
            failure_condition="The history becomes too short.",
        ),
    ),
)
EVIDENCE = ProposerEvidence(
    label="adaptive_train_build_diagnostic",
    independent_generalization_claim=False,
    morphology=(),
    comparisons=(),
)


def _assumption(
    index: int,
    *,
    kind: str = "weighted",
    feature: str = "periodicity_strength",
    candidate_name: str = "timesfm_2_5",
) -> dict[str, object]:
    return {
        "assumption_id": f"assumption_{index}",
        "candidate_name": candidate_name,
        "feature": feature,
        "direction": "above",
        "horizon_region": "full",
        "operator": kind,
        "rationale": "A reviewed history feature may identify a stable region.",
        "failure_condition": "The history feature no longer separates the region.",
    }


def _recipe(
    index: int,
    *,
    kind: str = "weighted",
    parents: list[str] | None = None,
    feature: str = "periodicity_strength",
) -> dict[str, object]:
    selected_parents = parents or ["timesfm_2_5", "seasonal_naive"]
    return {
        "name": f"challenger_{index}",
        "kind": kind,
        "parents": selected_parents,
        "fallback_parent": selected_parents[0],
        "assumptions": [
            _assumption(index, kind=kind, feature=feature, candidate_name=selected_parents[0])
        ],
    }


def legacy_response() -> dict[str, object]:
    return {"recipes": [_recipe(index) for index in range(5)]}


def _closed_assumption(
    *,
    kind: str = "weighted",
    feature: str = "periodicity_strength",
    candidate_name: str = "timesfm_2_5",
    direction: str = "above",
    horizon_region: str = "full",
) -> dict[str, object]:
    return {
        "candidate_name": candidate_name,
        "feature": feature,
        "direction": direction,
        "horizon_region": horizon_region,
        "operator": kind,
    }


def _closed_recipe(
    *,
    kind: str = "weighted",
    feature: str = "periodicity_strength",
    direction: str = "above",
    horizon_region: str = "full",
) -> dict[str, object]:
    parents = ["timesfm_2_5"] if kind == "select" else [
        "timesfm_2_5",
        "seasonal_naive",
    ]
    return {
        "kind": kind,
        "parents": parents,
        "fallback_parent": parents[0],
        "assumptions": [
            _closed_assumption(
                kind=kind,
                feature=feature,
                candidate_name=parents[0],
                direction=direction,
                horizon_region=horizon_region,
            )
        ],
    }


def closed_response(**recipe_fields: str) -> dict[str, object]:
    primary_feature = recipe_fields.get("feature", "periodicity_strength")
    features = (
        primary_feature,
        *(
            feature
            for feature in (
                "history_length",
                "horizon",
                "horizon_ratio",
                "zero_fraction",
                "trend_strength",
                "periodicity_strength",
                "periodicity_confidence",
                "outlier_fraction",
                "noise_relative_scale",
                "stationarity_score",
                "recent_regime_confidence",
                "intermittency_adi",
                "intermittency_cv2",
            )
            if feature != primary_feature
        ),
    )
    return {
        "recipes": [
            _closed_recipe(**{**recipe_fields, "feature": feature})
            for feature in features[:5]
        ],
    }


def distinct_closed_response() -> dict[str, object]:
    return {
        "recipes": [
            _closed_recipe(feature=feature)
            for feature in (
                "history_length",
                "horizon",
                "horizon_ratio",
                "zero_fraction",
                "trend_strength",
            )
        ],
    }


def valid_response() -> dict[str, object]:
    return closed_response()


def timesfm_seasonal_response() -> dict[str, object]:
    return valid_response()


def test_parser_reconstructs_task2_text_and_ids_from_closed_structure() -> None:
    response = closed_response()
    recipes = parse_champion_response(response, INVENTORY)

    assert tuple(recipe.name for recipe in recipes) == tuple(
        f"proposed_recipe_{index}" for index in range(5)
    )
    assert tuple(
        recipe.assumptions[0].assumption_id for recipe in recipes
    ) == tuple(f"proposed_assumption_{index}_0" for index in range(5))
    assert recipes[0].assumptions[0].rationale == (
        "The periodicity_strength feature supports timesfm_2_5 for above full "
        "weighted structure."
    )
    assert recipes[0].assumptions[0].failure_condition == (
        "The periodicity_strength feature may not support timesfm_2_5 for above "
        "full weighted structure."
    )
    repeated = parse_champion_response(response, INVENTORY)
    assert repeated == recipes
    assert tuple(map(champion_fingerprint, repeated)) == tuple(
        map(champion_fingerprint, recipes)
    )


def test_parser_rejects_exact_duplicate_closed_structures_before_count() -> None:
    response = closed_response()
    response["recipes"][1] = response["recipes"][0]  # type: ignore[index]

    with pytest.raises(ChampionProposalError, match="duplicate"):
        parse_champion_response(response, INVENTORY)


def test_parser_rejects_reordered_object_keys_for_the_same_structure() -> None:
    response = distinct_closed_response()
    first = response["recipes"][0]  # type: ignore[index]
    response["recipes"][1] = dict(reversed(tuple(first.items())))  # type: ignore[index]

    with pytest.raises(ChampionProposalError, match="duplicate"):
        parse_champion_response(response, INVENTORY)


def test_parser_accepts_five_genuinely_distinct_closed_structures() -> None:
    recipes = parse_champion_response(distinct_closed_response(), INVENTORY)

    assert len(recipes) == 5
    assert len({champion_fingerprint(recipe) for recipe in recipes}) == 5


def _identifier_collision_inventory() -> ToolDictionary:
    return ToolDictionary(
        "identifier_collision_inventory",
        None,
        0,
        (
            *INVENTORY.methods,
            MethodDefinition(
                "ｐｒｏｐｏｓｅｄ＿ｒｅｃｉｐｅ＿０",
                "statistical",
                "normalized recipe identifier collision",
                status="accepted",
            ),
            MethodDefinition(
                "proposed_assumption_0_0",
                "statistical",
                "assumption identifier collision",
                status="accepted",
            ),
        ),
    )


def test_host_ids_skip_normalized_inventory_collisions_deterministically() -> None:
    inventory = _identifier_collision_inventory()
    response = distinct_closed_response()

    first = parse_champion_response(response, inventory)
    second = parse_champion_response(response, inventory)

    assert first[0].name == "proposed_recipe_0_1"
    assert first[0].assumptions[0].assumption_id == "proposed_assumption_0_0_1"
    assert tuple(map(champion_fingerprint, first)) == tuple(
        map(champion_fingerprint, second)
    )


def test_host_identifier_allocation_cap_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proposal_module, "_MAX_ID_ALLOCATION_ATTEMPTS", 1)

    with pytest.raises(ChampionProposalError, match="allocate"):
        parse_champion_response(
            distinct_closed_response(),
            _identifier_collision_inventory(),
        )


@pytest.mark.parametrize(
    "kind",
    ("select", "route", "horizon_route", "weighted", "median", "bounded_overlay"),
)
@pytest.mark.parametrize(
    "feature",
    (
        "history_length",
        "horizon",
        "horizon_ratio",
        "zero_fraction",
        "trend_strength",
        "periodicity_strength",
        "periodicity_confidence",
        "outlier_fraction",
        "noise_relative_scale",
        "stationarity_score",
        "recent_regime_confidence",
        "intermittency_adi",
        "intermittency_cv2",
    ),
)
@pytest.mark.parametrize("direction", ("above", "below"))
@pytest.mark.parametrize("horizon_region", ("early", "late", "full"))
def test_parser_accepts_every_closed_structural_combination(
    kind: str,
    feature: str,
    direction: str,
    horizon_region: str,
) -> None:
    recipes = parse_champion_response(
        closed_response(
            kind=kind,
            feature=feature,
            direction=direction,
            horizon_region=horizon_region,
        ),
        INVENTORY,
    )

    assumption = recipes[0].assumptions[0]
    assert (
        assumption.operator,
        assumption.feature,
        assumption.direction,
        assumption.horizon_region,
    ) == (kind, feature, direction, horizon_region)


@pytest.mark.parametrize(
    "prose",
    (
        "Choose as winner when hidden outcome quality improves.",
        "Promote this recipe when holdout outcomes improve.",
        "yield specialist",
        "Elect this contender after unseen results improve.",
        "Crown this option when concealed results improve.",
        "await specialist",
        "raise specialist",
    ),
)
def test_parser_rejects_legacy_llm_owned_text_even_when_schema_is_otherwise_valid(
    prose: str,
) -> None:
    response = legacy_response()
    response["recipes"][0]["assumptions"][0]["rationale"] = prose  # type: ignore[index]

    with pytest.raises(ChampionProposalError):
        parse_champion_response(response, INVENTORY)


def test_prompt_has_no_labels_tasks_or_numeric_authority() -> None:
    llm = FakeLLMClient([json.dumps(valid_response())])

    propose_champion_recipes(llm, PARENT, INVENTORY, EVIDENCE)

    payload = json.loads(llm.calls[0]["messages"][0]["content"])
    serialized = json.dumps(payload).lower()
    assert set(payload) == {"evidence", "inventory", "output_schema", "parent"}
    assert "task_" not in serialized
    assert "future" not in serialized
    assert "public" not in serialized
    assert "dev" not in serialized
    assert "accepted" not in serialized
    assert "weights" not in payload["output_schema"]["assumption_fields"]
    assert "threshold" not in payload["output_schema"]["assumption_fields"]
    assert payload["output_schema"]["recipe_fields"] == [
        "kind",
        "parents",
        "fallback_parent",
        "assumptions",
    ]
    assert payload["output_schema"]["assumption_fields"] == [
        "candidate_name",
        "feature",
        "direction",
        "horizon_region",
        "operator",
    ]
    assert payload["parent"] == {
        "kind": "select",
        "parents": ["timesfm_2_5"],
        "fallback_parent": "timesfm_2_5",
        "assumptions": [
            {
                "candidate_name": "timesfm_2_5",
                "feature": "history_length",
                "direction": "above",
                "horizon_region": "full",
                "operator": "select",
            }
        ],
    }
    assert payload["evidence"] == EVIDENCE.to_payload()
    assert payload["inventory"] == [
        {"family": "statistical", "name": "seasonal_naive"},
        {"family": "foundation", "name": "timesfm_2_5"},
        {"family": "foundation", "name": "toto_2_0"},
    ]


def test_prompt_projects_parent_without_open_names_ids_or_text() -> None:
    assumption = replace(
        PARENT.assumptions[0],
        assumption_id="holdout_outcome",
        rationale="Choose as winner when hidden outcome quality improves.",
        failure_condition="yield specialist",
    )
    parent = replace(PARENT, name="task_42", assumptions=(assumption,))
    llm = FakeLLMClient([json.dumps(valid_response())])

    propose_champion_recipes(llm, parent, INVENTORY, EVIDENCE)

    prompt = llm.calls[0]["messages"][0]["content"]
    assert "task_42" not in prompt
    assert "holdout_outcome" not in prompt
    assert "Choose as winner" not in prompt
    assert "yield specialist" not in prompt


def test_parser_rejects_llm_numeric_thresholds_before_expansion() -> None:
    response = valid_response()
    response["recipes"][0]["threshold"] = 0.8  # type: ignore[index]

    with pytest.raises(ChampionProposalError):
        parse_champion_response(response, INVENTORY)


def test_proposer_allows_tsfm_statistical_recipe_without_toto() -> None:
    recipes = parse_champion_response(timesfm_seasonal_response(), INVENTORY)

    assert recipes[0].parents == ("timesfm_2_5", "seasonal_naive")
    assert "toto_2_0" not in recipes[0].parents


def test_parser_rejects_noncanonical_inventory_definition_records() -> None:
    @dataclass(frozen=True)
    class DefinitionSubclass(MethodDefinition):
        pass

    inventory = ToolDictionary(
        "subclass_inventory",
        None,
        0,
        tuple(
            DefinitionSubclass(
                record.definition.method_id,
                record.definition.family,
                record.definition.description,
            )
            for record in INVENTORY.methods
        ),
    )

    with pytest.raises(ChampionProposalError, match="exact canonical"):
        parse_champion_response(valid_response(), inventory)


@pytest.mark.parametrize(
    "identifier", ("_hidden", "_gate", "class", "for", "match", "case")
)
@pytest.mark.parametrize("field", ("recipe", "assumption"))
def test_parser_rejects_private_and_keyword_recipe_namespaces(
    field: str,
    identifier: str,
) -> None:
    recipe = _typed_recipe("route")
    if field == "recipe":
        recipe = replace(recipe, name=identifier)
    else:
        assumption = replace(recipe.assumptions[0], assumption_id=identifier)
        recipe = replace(recipe, assumptions=(assumption,))

    with pytest.raises(ChampionProposalError, match="public non-keyword"):
        expand_recipe(recipe, _rows())


def test_parser_rejects_unicode_normalized_recipe_and_assumption_duplicates() -> None:
    source = _typed_recipe("weighted")
    duplicate_parents = replace(
        source,
        parents=("Model_K", "Model_K"),
        fallback_parent="Model_K",
        assumptions=(replace(source.assumptions[0], candidate_name="Model_K"),),
    )
    with pytest.raises(ChampionProposalError, match="normalized"):
        expand_recipe(duplicate_parents, _rows())

    duplicate_assumptions = replace(
        source,
        assumptions=(
            replace(source.assumptions[0], assumption_id="Signal"),
            replace(source.assumptions[0], assumption_id="Ｓｉｇｎａｌ"),
        ),
    )
    with pytest.raises(ChampionProposalError, match="normalized"):
        expand_recipe(duplicate_assumptions, _rows())


def test_inventory_candidate_names_are_public_nonkeyword_and_normalized_unique() -> None:
    keyword_inventory = ToolDictionary(
        "keyword_inventory",
        None,
        0,
        (
            *INVENTORY.methods,
            MethodDefinition(
                "class",
                "statistical",
                "syntactic keyword candidate",
                status="accepted",
            ),
        ),
    )
    with pytest.raises(ChampionProposalError, match="public non-keyword"):
        parse_champion_response(valid_response(), keyword_inventory)

    normalized_duplicate = ToolDictionary(
        "normalized_duplicate_inventory",
        None,
        0,
        (
            MethodDefinition(
                "Model_K", "foundation", "first candidate", status="accepted"
            ),
            MethodDefinition(
                "Model_K", "statistical", "second candidate", status="accepted"
            ),
        ),
    )
    with pytest.raises(ChampionProposalError, match="normalized"):
        parse_champion_response(valid_response(), normalized_duplicate)

def test_inactive_dictionary_records_neither_enter_prompt_nor_authorize_parents() -> None:
    inventory = ToolDictionary(
        "mixed_inventory",
        None,
        0,
        (
            *INVENTORY.methods,
            MethodDefinition(
                "quarantined_leaf",
                "statistical",
                "not active proposal supply",
                status="quarantined",
            ),
        ),
    )
    llm = FakeLLMClient([json.dumps(valid_response())])

    propose_champion_recipes(llm, PARENT, inventory, EVIDENCE)

    prompt = json.loads(llm.calls[0]["messages"][0]["content"])
    assert "quarantined_leaf" not in {
        card["name"] for card in prompt["inventory"]
    }
    response = valid_response()
    response["recipes"][0]["parents"] = [  # type: ignore[index]
        "timesfm_2_5",
        "quarantined_leaf",
    ]
    with pytest.raises(ChampionProposalError, match="unknown parent"):
        parse_champion_response(response, inventory)


@pytest.mark.parametrize("location", ("record", "definition"))
def test_inventory_rejects_hostile_status_before_hash_or_equality(
    location: str,
) -> None:
    class SpoofStatus:
        calls = 0

        def __hash__(self) -> int:
            type(self).calls += 1
            return hash("accepted")

        def __eq__(self, other: object) -> bool:
            type(self).calls += 1
            return other == "accepted"

    source = INVENTORY.methods[1]
    definition = replace(source.definition)
    record = replace(source, definition=definition)
    if location == "record":
        object.__setattr__(record, "status", SpoofStatus())
    else:
        object.__setattr__(definition, "status", SpoofStatus())
    inventory = ToolDictionary(
        "hostile_status_inventory",
        None,
        0,
        (INVENTORY.methods[0], record, INVENTORY.methods[2]),
    )

    with pytest.raises(ChampionProposalError, match="exact canonical"):
        parse_champion_response(valid_response(), inventory)

    assert SpoofStatus.calls == 0


@pytest.mark.parametrize(
    "mutation",
    (
        lambda response: {**response, "unknown": []},
        lambda response: {"recipes": tuple(response["recipes"])},
        lambda response: {"recipes": [*response["recipes"][:4]]},
        lambda response: {
            "recipes": [
                *response["recipes"],
                _closed_recipe(),
                _closed_recipe(),
                _closed_recipe(),
                _closed_recipe(),
                _closed_recipe(),
                _closed_recipe(),
            ]
        },
        lambda response: {
            "recipes": [
                {
                    **response["recipes"][0],
                    "parents": ["unknown_parent", "seasonal_naive"],
                },
                *response["recipes"][1:],
            ]
        },
        lambda response: {
            "recipes": [
                {
                    **response["recipes"][0],
                    "assumptions": [
                        _closed_assumption(feature="unknown_feature")
                    ],
                },
                *response["recipes"][1:],
            ]
        },
        lambda response: {
            "recipes": [
                {**response["recipes"][0], "name": "not-public"},
                *response["recipes"][1:],
            ]
        },
        lambda response: {
            "recipes": [
                {
                    **response["recipes"][0],
                    "assumptions": [
                        {
                            **response["recipes"][0]["assumptions"][0],
                            "assumption_id": "model_owned_id",
                        }
                    ],
                },
                response["recipes"][1],
                *response["recipes"][2:],
            ]
        },
    ),
)
def test_parser_rejects_schema_count_identifier_and_namespace_drift(mutation) -> None:
    with pytest.raises(ChampionProposalError):
        parse_champion_response(mutation(valid_response()), INVENTORY)


@pytest.mark.parametrize(
    "response",
    (
        '{"recipes": [], "recipes": []}',
        '```json\n{"recipes": []}\n```',
        '{"recipes": []} trailing',
        '{"recipes": NaN}',
    ),
)
def test_raw_parser_rejects_duplicate_keys_code_blocks_and_noncanonical_json(
    response: str,
) -> None:
    with pytest.raises(ChampionProposalError):
        parse_champion_response(response, INVENTORY)


@pytest.mark.parametrize(
    "prose",
    (
        "Use task_42 future labels at threshold 0.8, then call accept().",
        "Use ｔａｓｋ＿４２ and Ｐｕｂｌｉｃ labels for this route.",
        "Blend alpha=0.25 before routing.",
        "Run helper(value) when seasonality changes.",
        "def choose: return specialist",
        "if seasonality: choose route",
        "class Router",
        "call helper",
        "Approve and gate this route.",
        "Entity truth metrics score this split.",
    ),
)
def test_parser_rejects_authority_numeric_and_code_like_structural_prose(
    prose: str,
) -> None:
    response = legacy_response()
    response["recipes"][0]["assumptions"][0]["rationale"] = prose  # type: ignore[index]

    with pytest.raises(ChampionProposalError):
        parse_champion_response(response, INVENTORY)


def test_parser_rejects_even_benign_llm_owned_prose() -> None:
    response = legacy_response()
    assumption = response["recipes"][0]["assumptions"][0]  # type: ignore[index]
    assumption["rationale"] = (
        "Historical capacity may indicate stable seasonal behavior."
    )
    assumption["failure_condition"] = (
        "The captioned seasonal pattern may weaken gradually."
    )

    with pytest.raises(ChampionProposalError):
        parse_champion_response(response, INVENTORY)


def test_proposer_retries_unsafe_prose_once_and_retains_no_partial_batch() -> None:
    response = legacy_response()
    response["recipes"][0]["assumptions"][0]["rationale"] = (  # type: ignore[index]
        "Use task_42 labels at threshold 0.8."
    )
    llm = FakeLLMClient([json.dumps(response), json.dumps(response)])

    with pytest.raises(ChampionProposalError, match="after one schema retry") as caught:
        propose_champion_recipes(llm, PARENT, INVENTORY, EVIDENCE)

    assert len(llm.calls) == 2
    assert "task_42" not in str(caught.value)


def test_proposer_retries_reviewer_text_and_returns_only_closed_second_batch() -> None:
    first = legacy_response()
    first["recipes"][0]["assumptions"][0]["rationale"] = "yield specialist"  # type: ignore[index]
    second = closed_response(kind="route")
    llm = FakeLLMClient([json.dumps(first), json.dumps(second)])

    recipes = propose_champion_recipes(llm, PARENT, INVENTORY, EVIDENCE)

    assert len(llm.calls) == 2
    assert all(recipe.kind == "route" for recipe in recipes)
    assert "yield specialist" not in json.dumps(
        [recipe.to_payload() for recipe in recipes]
    )


def test_proposer_retries_schema_once_without_returning_partial_output() -> None:
    recovered = FakeLLMClient(["not json", json.dumps(valid_response())])

    assert len(propose_champion_recipes(recovered, PARENT, INVENTORY, EVIDENCE)) == 5
    assert len(recovered.calls) == 2
    assert recovered.calls[0]["system"] == recovered.calls[1]["system"]
    assert recovered.calls[0]["messages"] == recovered.calls[1]["messages"]

    first = valid_response()
    first["recipes"] = first["recipes"][:4]  # type: ignore[index]
    rejected = FakeLLMClient([json.dumps(first), "not json"])
    with pytest.raises(ChampionProposalError, match="after one schema retry"):
        propose_champion_recipes(rejected, PARENT, INVENTORY, EVIDENCE)
    assert len(rejected.calls) == 2


def _profile(task_id: str, periodicity_strength: float) -> TaskProfile:
    return TaskProfile(
        task_id=task_id,
        frequency="D",
        history_length=200,
        horizon=4,
        zero_fraction=0.0,
        signed=False,
        integer_valued=False,
        trend_direction="up",
        trend_strength=0.8,
        periodicity_periods=(7,),
        periodicity_strength=periodicity_strength,
        periodicity_confidence=0.9,
        outlier_fraction=0.0,
        noise_relative_scale=0.2,
        likely_stationary=False,
        stationarity_score=0.2,
        recent_regime_start=None,
        recent_regime_confidence=0.1,
        intermittency_adi=1.0,
        intermittency_cv2=0.1,
    )


def _rows() -> tuple[ChampionTaskRow, ...]:
    rows = []
    for index, value in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        task_id = f"build_{index}"
        rows.append(
            ChampionTaskRow(
                task_id=task_id,
                candidate_name="timesfm_2_5",
                profile=_profile(task_id, value),
                truth=(1.0,) * 4,
                forecast=(1.0,) * 4,
            )
        )
    return tuple(rows)


def _typed_recipe(kind: str, *, feature: str = "periodicity_strength") -> ChampionRecipe:
    parents = ("timesfm_2_5",) if kind == "select" else (
        "timesfm_2_5",
        "seasonal_naive",
    )
    return ChampionRecipe(
        name=f"{kind}_challenger",
        kind=kind,  # type: ignore[arg-type]
        parents=parents,
        fallback_parent=parents[0],
        assumptions=(
            EvolutionAssumption(
                assumption_id="build_feature",
                candidate_name=parents[0],
                feature=feature,
                direction="above",
                horizon_region="full",
                operator=kind,  # type: ignore[arg-type]
                rationale="The history feature identifies a region.",
                failure_condition="The history feature stops identifying that region.",
            ),
        ),
    )


def test_weighted_expansion_uses_only_deduplicated_build_quantiles_and_fixed_weights() -> None:
    policies = expand_recipe(_typed_recipe("weighted"), _rows())

    assert len(policies) == 15
    assert tuple(dict.fromkeys(policy.thresholds[0][1] for policy in policies)) == (
        0.2,
        0.4,
        0.5,
        0.6,
        0.8,
    )
    assert tuple(policy.weights for policy in policies[:3]) == WEIGHT_GRID
    assert all(policy.overlay_alpha == 0.0 for policy in policies)
    assert all(policy.correction_cap == 0.0 for policy in policies)
    assert all(policy.horizon_split == 0.5 for policy in policies)


def test_host_grids_are_operator_specific_finite_and_deterministic() -> None:
    rows = _rows()
    overlay = expand_recipe(_typed_recipe("bounded_overlay"), rows)
    horizon = expand_recipe(_typed_recipe("horizon_route"), rows)
    route = expand_recipe(_typed_recipe("route"), rows)

    assert len(overlay) == 5 * len(OVERLAY_ALPHA_GRID) * len(CORRECTION_CAP_GRID)
    assert tuple(
        (policy.overlay_alpha, policy.correction_cap)
        for policy in overlay[: len(OVERLAY_ALPHA_GRID) * len(CORRECTION_CAP_GRID)]
    ) == tuple(
        (alpha, cap) for alpha in OVERLAY_ALPHA_GRID for cap in CORRECTION_CAP_GRID
    )
    assert len(horizon) == 5 * len(HORIZON_SPLIT_GRID)
    assert tuple(policy.horizon_split for policy in horizon[:3]) == HORIZON_SPLIT_GRID
    assert len(route) == 5
    assert expand_recipe(_typed_recipe("bounded_overlay"), rows) == overlay


def test_expansion_deduplicates_profiles_and_quantiles_without_namespace_changes() -> None:
    rows = _rows()
    duplicate_candidates = rows + tuple(
        replace(row, candidate_name="seasonal_naive") for row in rows
    )

    policies = expand_recipe(_typed_recipe("weighted"), duplicate_candidates)

    assert len(policies) == 15
    assert all(policy.recipe is policies[0].recipe for policy in policies)
    assert all(policy.recipe.parents == ("timesfm_2_5", "seasonal_naive") for policy in policies)


@pytest.mark.parametrize(
    "recipe, rows",
    (
        (_typed_recipe("route", feature="unsupported_feature"), _rows()),
        (_typed_recipe("route"), ()),
        (_typed_recipe("route"), (replace(_rows()[0], split="calibration"),)),
    ),
)
def test_expansion_rejects_missing_features_empty_build_and_noncanonical_rows(
    recipe: ChampionRecipe, rows: object
) -> None:
    with pytest.raises(ChampionProposalError):
        expand_recipe(recipe, rows)  # type: ignore[arg-type]


def test_expansion_accepts_exact_builtin_row_containers_but_not_subclasses() -> None:
    class RowList(list):
        pass

    recipe = _typed_recipe("route")
    assert expand_recipe(recipe, list(_rows())) == expand_recipe(recipe, _rows())
    with pytest.raises(ChampionProposalError):
        expand_recipe(recipe, RowList(_rows()))


def test_expansion_rejects_unbounded_integer_features_without_raw_overflow() -> None:
    row = _rows()[0]
    object.__setattr__(row.profile, "history_length", 10**400)

    with pytest.raises(ChampionProposalError, match="bounded") as caught:
        expand_recipe(_typed_recipe("route", feature="history_length"), (row,))

    assert "10" not in str(caught.value)
