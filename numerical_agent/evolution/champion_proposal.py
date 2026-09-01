"""Strict Champion structure proposal and deterministic host-owned expansion."""
from __future__ import annotations

import itertools
import json
import math
from typing import cast

from common.llm import LLMClient
from common.metrics import linear_quantile
from common.payload import strict_json_loads

from numerical_agent.dictionary import MethodDefinition, MethodRecord, ToolDictionary

from .champion import (
    ChampionContractError,
    ChampionRecipe,
    EvolutionAssumption,
    FittedChampionPolicy,
    parse_champion_recipe,
)
from .champion_evidence import ChampionEvidenceError, ChampionTaskRow, ProposerEvidence
from .screening import TaskProfile


WEIGHT_GRID = ((0.75, 0.25), (0.5, 0.5), (0.25, 0.75))
OVERLAY_ALPHA_GRID = (0.25, 0.5, 0.75)
CORRECTION_CAP_GRID = (0.1, 0.25, 0.5, 1.0)
HORIZON_SPLIT_GRID = (0.25, 0.5, 0.75)

_QUANTILES = (0.2, 0.4, 0.5, 0.6, 0.8)
_OPERATORS = (
    "select",
    "route",
    "horizon_route",
    "weighted",
    "median",
    "bounded_overlay",
)
_FEATURES = (
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
_PROFILE_FLOAT_FEATURES = frozenset(_FEATURES) - {
    "history_length",
    "horizon",
    "horizon_ratio",
}
_RECIPE_FIELDS = (
    "name",
    "kind",
    "parents",
    "fallback_parent",
    "assumptions",
)
_ASSUMPTION_FIELDS = (
    "assumption_id",
    "candidate_name",
    "feature",
    "direction",
    "horizon_region",
    "operator",
    "rationale",
    "failure_condition",
)
_MAX_ASSUMPTIONS_PER_RECIPE = 3
_MAX_BUILD_ROWS = 1_000_000
_ACTIVE_INVENTORY_STATUSES = frozenset({"accepted", "specialized"})

CHAMPION_PROPOSAL_SYSTEM = """You propose structural Numerical Champion recipes only.
Return exactly one standards-JSON object matching the exact schema supplied by
the user. Use exact canonical JSON types, no wrappers, no duplicate keys, and
no unknown fields. Every recipe must use only supplied candidate names,
features, operators, directions, and horizon regions. Return no thresholds,
weights, strengths, correction caps, split positions, source code, data labels,
task identities, scores, gates, or acceptance decisions. Numeric expansion and
all evaluation belong exclusively to trusted host code. A malformed response
causes at most one identical schema retry and no partial recipe is retained."""


class ChampionProposalError(ValueError):
    """An untrusted Champion proposal or host expansion violates its boundary."""


def _fail(message: str) -> None:
    raise ChampionProposalError(message)


def _validate_limits(minimum: object, maximum: object) -> tuple[int, int]:
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or not 5 <= minimum <= maximum <= 10
    ):
        _fail("recipe count bounds must be exact integers within five through ten")
    return cast(int, minimum), cast(int, maximum)


def _inventory_records(inventory: object) -> tuple[MethodRecord, ...]:
    if type(inventory) is not ToolDictionary:
        _fail("inventory must be an exact canonical ToolDictionary")
    canonical = cast(ToolDictionary, inventory)
    if type(canonical.methods) is not tuple:
        _fail("inventory must be an exact canonical ToolDictionary")
    records = cast(tuple[MethodRecord, ...], canonical.methods)
    if not records or any(type(record) is not MethodRecord for record in records):
        _fail("inventory must contain exact canonical MethodRecord values")
    names: list[str] = []
    for record in records:
        try:
            MethodRecord.__post_init__(record)
            raw_definition = record.definition
            if type(raw_definition) is not MethodDefinition:
                _fail("inventory definitions must use exact canonical records")
            definition = cast(MethodDefinition, raw_definition)
            definition.__post_init__()
        except ChampionProposalError:
            raise
        except Exception as error:
            raise ChampionProposalError("inventory contains an invalid method record") from error
        name = definition.method_id
        if type(name) is not str or not name or not name.isidentifier() or name.startswith("_"):
            _fail("inventory method names must be public Python identifiers")
        names.append(name)
    if len(names) != len(set(names)):
        _fail("inventory method names must be unique")
    return records


def _inventory_names(inventory: object) -> tuple[str, ...]:
    return tuple(
        record.definition.method_id for record in _active_inventory_records(inventory)
    )


def _active_inventory_records(inventory: object) -> tuple[MethodRecord, ...]:
    active = tuple(
        record
        for record in _inventory_records(inventory)
        if record.status in _ACTIVE_INVENTORY_STATUSES
    )
    if not active:
        _fail("inventory has no active reviewed candidate supply")
    return active


def _inventory_payload(inventory: object) -> list[dict[str, str]]:
    records = _active_inventory_records(inventory)
    return [
        {
            "name": record.definition.method_id,
            "family": record.definition.family,
        }
        for record in sorted(records, key=lambda item: item.definition.method_id)
    ]


def _validate_recipe(recipe: object, *, inventory_names: tuple[str, ...]) -> ChampionRecipe:
    if type(recipe) is not ChampionRecipe:
        _fail("recipe must be an exact ChampionRecipe")
    canonical = cast(ChampionRecipe, recipe)
    try:
        for assumption in canonical.assumptions:
            if type(assumption) is not EvolutionAssumption:
                _fail("recipe assumptions must be exact EvolutionAssumption values")
            EvolutionAssumption.__post_init__(assumption)
        ChampionRecipe.__post_init__(canonical)
    except ChampionProposalError:
        raise
    except Exception as error:
        raise ChampionProposalError("recipe violates the Champion contract") from error
    if any(parent not in inventory_names for parent in canonical.parents):
        _fail("recipe contains an unknown parent")
    if any(assumption.feature not in _FEATURES for assumption in canonical.assumptions):
        _fail("recipe contains an unknown feature")
    return canonical


def _contains_code_block(value: object) -> bool:
    if type(value) is str:
        return "```" in value
    if type(value) is list:
        return any(_contains_code_block(item) for item in value)
    if type(value) is dict:
        return any(
            _contains_code_block(key) or _contains_code_block(item)
            for key, item in value.items()
        )
    return False


def _response_payload(response: object) -> dict[str, object]:
    if type(response) is str:
        if "```" in response:
            _fail("Champion responses cannot contain code blocks")
        try:
            parsed = strict_json_loads(response, context="Champion response")
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as error:
            raise ChampionProposalError("Champion response must be strict JSON") from error
    elif type(response) is dict:
        parsed = response
    else:
        _fail("Champion response must be an exact JSON object or response text")
    if type(parsed) is not dict:
        _fail("Champion response must be an exact JSON object")
    if _contains_code_block(parsed):
        _fail("Champion responses cannot contain code blocks")
    return parsed


def parse_champion_response(
    response: object,
    inventory: ToolDictionary,
    *,
    minimum: int = 5,
    maximum: int = 10,
) -> tuple[ChampionRecipe, ...]:
    """Parse an exact recipe batch with no model-owned numeric parameters."""
    lower, upper = _validate_limits(minimum, maximum)
    inventory_names = _inventory_names(inventory)
    payload = _response_payload(response)
    if set(payload) != {"recipes"} or type(payload["recipes"]) is not list:
        _fail("Champion response fields must be exactly one recipes array")
    raw_recipes = cast(list[object], payload["recipes"])
    if not lower <= len(raw_recipes) <= upper:
        _fail("Champion response recipe count is outside the configured bounds")

    parsed: list[ChampionRecipe] = []
    for raw_recipe in raw_recipes:
        if type(raw_recipe) is not dict:
            _fail("each Champion recipe must be an exact JSON object")
        recipe_payload = cast(dict[str, object], raw_recipe)
        assumptions = recipe_payload.get("assumptions")
        if (
            type(assumptions) is not list
            or not 1 <= len(assumptions) <= _MAX_ASSUMPTIONS_PER_RECIPE
        ):
            _fail("each Champion recipe requires one through three assumptions")
        try:
            recipe = parse_champion_recipe(recipe_payload)
        except (ChampionContractError, KeyError, TypeError, ValueError) as error:
            raise ChampionProposalError("Champion recipe violates its exact schema") from error
        _validate_recipe(recipe, inventory_names=inventory_names)
        if recipe.name in inventory_names:
            _fail("Champion recipe names cannot collide with the candidate inventory")
        parsed.append(recipe)
    names = tuple(recipe.name for recipe in parsed)
    if len(names) != len(set(names)):
        _fail("Champion recipe names must be unique")
    return tuple(parsed)


def _validated_parent(parent: object, inventory_names: tuple[str, ...]) -> ChampionRecipe:
    return _validate_recipe(parent, inventory_names=inventory_names)


def _validated_evidence(evidence: object) -> dict[str, object]:
    if type(evidence) is not ProposerEvidence:
        _fail("evidence must be exact ProposerEvidence")
    canonical = cast(ProposerEvidence, evidence)
    try:
        ProposerEvidence.__post_init__(canonical)
        return canonical.to_payload()
    except (AttributeError, ChampionEvidenceError, TypeError, ValueError) as error:
        raise ChampionProposalError("proposer evidence is invalid") from error


def _proposal_payload(
    parent: object,
    inventory: object,
    evidence: object,
    *,
    minimum: int,
    maximum: int,
) -> dict[str, object]:
    lower, upper = _validate_limits(minimum, maximum)
    inventory_names = _inventory_names(inventory)
    validated_parent = _validated_parent(parent, inventory_names)
    return {
        "parent": validated_parent.to_payload(),
        "inventory": _inventory_payload(inventory),
        "evidence": _validated_evidence(evidence),
        "output_schema": {
            "top_level_fields": ["recipes"],
            "recipe_fields": list(_RECIPE_FIELDS),
            "assumption_fields": list(_ASSUMPTION_FIELDS),
            "recipe_count": {"minimum": lower, "maximum": upper},
            "assumptions_per_recipe": {
                "minimum": 1,
                "maximum": _MAX_ASSUMPTIONS_PER_RECIPE,
            },
            "allowed_operators": list(_OPERATORS),
            "allowed_features": list(_FEATURES),
            "allowed_directions": ["above", "below"],
            "allowed_horizon_regions": ["early", "late", "full"],
        },
    }


def propose_champion_recipes(
    llm: LLMClient,
    parent: ChampionRecipe,
    inventory: ToolDictionary,
    evidence: ProposerEvidence,
    *,
    minimum: int = 5,
    maximum: int = 10,
) -> tuple[ChampionRecipe, ...]:
    """Request structures only, retrying one malformed schema exactly once."""
    if not hasattr(llm, "complete") or not callable(llm.complete):
        _fail("llm must implement the LLMClient completion boundary")
    payload = _proposal_payload(
        parent,
        inventory,
        evidence,
        minimum=minimum,
        maximum=maximum,
    )
    try:
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ChampionProposalError("proposal prompt is not standards JSON") from error
    messages = [{"role": "user", "content": content}]

    for attempt in range(2):
        response = llm.complete(
            system=CHAMPION_PROPOSAL_SYSTEM,
            messages=messages,
            temperature=0.0,
        )
        try:
            return parse_champion_response(
                response.text,
                inventory,
                minimum=minimum,
                maximum=maximum,
            )
        except ChampionProposalError as error:
            if attempt == 1:
                raise ChampionProposalError(
                    "Champion generation rejected after one schema retry"
                ) from error
    raise AssertionError("unreachable")


def _validated_build_profiles(
    build_rows: object,
) -> tuple[TaskProfile, ...]:
    if type(build_rows) not in {tuple, list} or not build_rows:
        _fail("Build rows must be a nonempty exact tuple or list")
    rows = cast(tuple[object, ...] | list[object], build_rows)
    if len(rows) > _MAX_BUILD_ROWS:
        _fail("Build rows exceed the finite host bound")
    seen_keys: set[tuple[str, str]] = set()
    profiles: dict[str, TaskProfile] = {}
    for raw_row in rows:
        if type(raw_row) is not ChampionTaskRow:
            _fail("Build rows must contain exact ChampionTaskRow values")
        row = cast(ChampionTaskRow, raw_row)
        try:
            ChampionTaskRow.__post_init__(row)
        except Exception as error:
            raise ChampionProposalError("Build row violates its trusted contract") from error
        if row.split != "build":
            _fail("host expansion accepts exact Build rows only")
        key = (row.candidate_name, row.task_id)
        if key in seen_keys:
            _fail("Build rows contain a duplicate candidate/task key")
        seen_keys.add(key)
        previous = profiles.get(row.task_id)
        if previous is not None and previous != row.profile:
            _fail("Build rows disagree on the task profile")
        profiles[row.task_id] = row.profile
    return tuple(profiles.values())


def _feature_value(profile: TaskProfile, feature: str) -> float | None:
    if feature == "horizon_ratio":
        value: object = profile.horizon / profile.history_length
    elif feature in {"history_length", "horizon"}:
        value = getattr(profile, feature)
    elif feature in _PROFILE_FLOAT_FEATURES:
        value = getattr(profile, feature)
    else:
        return None
    if type(value) not in {int, float}:
        return None
    number = float(cast(int | float, value))
    return number if math.isfinite(number) else None


def _threshold_grid(
    assumption: EvolutionAssumption,
    profiles: tuple[TaskProfile, ...],
) -> tuple[float, ...]:
    values = tuple(
        value
        for profile in profiles
        if (value := _feature_value(profile, assumption.feature)) is not None
    )
    if not values:
        _fail("Build has no finite value for an assumption feature")
    quantiles = tuple(float(linear_quantile(list(values), level)) for level in _QUANTILES)
    return tuple(dict.fromkeys(quantiles))


def _operator_parameters(recipe: ChampionRecipe) -> tuple[dict[str, object], ...]:
    if recipe.kind == "weighted":
        weights = tuple(grid for grid in WEIGHT_GRID if len(grid) == len(recipe.parents))
        if not weights:
            _fail("weighted grid has no entry for the recipe arity")
        return tuple({"weights": grid} for grid in weights)
    if recipe.kind == "bounded_overlay":
        return tuple(
            {"overlay_alpha": alpha, "correction_cap": cap}
            for alpha in OVERLAY_ALPHA_GRID
            for cap in CORRECTION_CAP_GRID
        )
    if recipe.kind == "horizon_route":
        return tuple({"horizon_split": split} for split in HORIZON_SPLIT_GRID)
    return ({},)


def expand_recipe(
    recipe: ChampionRecipe,
    build_rows: tuple[ChampionTaskRow, ...] | list[ChampionTaskRow],
) -> tuple[FittedChampionPolicy, ...]:
    """Expand one structure across finite Build quantiles and fixed host grids."""
    inventory_names = tuple(recipe.parents) if type(recipe) is ChampionRecipe else ()
    validated_recipe = _validate_recipe(recipe, inventory_names=inventory_names)
    if len(validated_recipe.assumptions) > _MAX_ASSUMPTIONS_PER_RECIPE:
        _fail("host expansion bounds recipes to three assumptions")
    profiles = _validated_build_profiles(build_rows)
    threshold_grids = tuple(
        _threshold_grid(assumption, profiles)
        for assumption in validated_recipe.assumptions
    )
    parameters = _operator_parameters(validated_recipe)
    policies: list[FittedChampionPolicy] = []
    try:
        for threshold_values in itertools.product(*threshold_grids):
            thresholds = tuple(
                (assumption.assumption_id, value)
                for assumption, value in zip(
                    validated_recipe.assumptions, threshold_values, strict=True
                )
            )
            for parameter in parameters:
                policies.append(
                    FittedChampionPolicy(
                        recipe=validated_recipe,
                        thresholds=thresholds,
                        **parameter,
                    )
                )
    except (ChampionContractError, TypeError, ValueError) as error:
        raise ChampionProposalError("host expansion produced an invalid fitted policy") from error
    if not policies:
        _fail("host expansion produced no finite fitted policies")
    return tuple(policies)


__all__ = [
    "CHAMPION_PROPOSAL_SYSTEM",
    "CORRECTION_CAP_GRID",
    "HORIZON_SPLIT_GRID",
    "OVERLAY_ALPHA_GRID",
    "WEIGHT_GRID",
    "ChampionProposalError",
    "expand_recipe",
    "parse_champion_response",
    "propose_champion_recipes",
]
