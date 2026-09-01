"""Strict Champion structure proposal and deterministic host-owned expansion."""
from __future__ import annotations

import itertools
import json
import keyword
import math
import unicodedata
from typing import cast

from common.llm import LLMClient
from common.metrics import linear_quantile
from common.payload import strict_json_loads

from numerical_agent.config import ALLOWED_FAMILIES, METHOD_STATUSES
from numerical_agent.dictionary import (
    MethodCandidate,
    MethodDefinition,
    MethodRecord,
    ToolDictionary,
)

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
    "kind",
    "parents",
    "fallback_parent",
    "assumptions",
)
_ASSUMPTION_FIELDS = (
    "candidate_name",
    "feature",
    "direction",
    "horizon_region",
    "operator",
)
_DIRECTIONS = ("above", "below")
_HORIZON_REGIONS = ("early", "late", "full")
_MAX_ASSUMPTIONS_PER_RECIPE = 3
_MAX_BUILD_ROWS = 1_000_000
_MAX_PROFILE_LENGTH = 1_000_000
_MAX_ID_ALLOCATION_ATTEMPTS = 10_000
_ACTIVE_INVENTORY_STATUSES = frozenset({"accepted", "specialized"})

CHAMPION_PROPOSAL_SYSTEM = """You propose structural Numerical Champion recipes only.
Return exactly one standards-JSON object matching the exact schema supplied by
the user. Use exact canonical JSON types, no wrappers, no duplicate keys, and
no unknown fields. Every recipe must use only supplied candidate names,
features, operators, directions, and horizon regions. Return no thresholds,
weights, strengths, correction caps, split positions, source code, data labels,
task identities, scores, gates, acceptance decisions, recipe names, assumption
IDs, rationales, or failure text. Trusted host code assigns all identifiers and
assembles canonical text after validating the closed structural fields. Numeric
expansion and all evaluation belong exclusively to trusted host code. A
malformed response causes at most one identical schema retry and no partial
recipe is retained. In every recipe, fallback_parent must be one of that same
recipe's parents; every assumption candidate_name must be one of those same
parents; and every assumption operator must exactly equal that recipe's kind."""


class ChampionProposalError(ValueError):
    """An untrusted Champion proposal or host expansion violates its boundary."""


def _fail(message: str) -> None:
    raise ChampionProposalError(message)


def _canonical_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _public_identifier(value: object, field_name: str) -> str:
    if type(value) is not str:
        _fail(f"{field_name} must be an exact public non-keyword identifier")
    identifier = cast(str, value)
    normalized = unicodedata.normalize("NFKC", identifier)
    if (
        not identifier
        or identifier.startswith("_")
        or normalized.startswith("_")
        or not identifier.isidentifier()
        or not normalized.isidentifier()
        or keyword.iskeyword(normalized)
        or keyword.issoftkeyword(normalized)
    ):
        _fail(f"{field_name} must be an exact public non-keyword identifier")
    return identifier


def _require_normalized_unique(values: tuple[str, ...], field_name: str) -> None:
    normalized = tuple(_canonical_text(value) for value in values)
    if len(normalized) != len(set(normalized)):
        _fail(f"{field_name} must be normalized-unique")


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
    if (
        type(canonical.dictionary_id) is not str
        or not canonical.dictionary_id
        or (
            canonical.parent_dictionary_id is not None
            and type(canonical.parent_dictionary_id) is not str
        )
        or type(canonical.generation) is not int
        or canonical.generation < 0
    ):
        _fail("inventory metadata must use exact canonical types")
    if type(canonical.methods) is not tuple:
        _fail("inventory must be an exact canonical ToolDictionary")
    records = cast(tuple[MethodRecord, ...], canonical.methods)
    if not records or any(type(record) is not MethodRecord for record in records):
        _fail("inventory must contain exact canonical MethodRecord values")
    names: list[str] = []
    for record in records:
        if (
            type(record.status) is not str
            or record.status not in METHOD_STATUSES
            or type(record.revision_count) is not int
            or record.revision_count < 0
            or type(record.implementation_attempts) is not int
            or record.implementation_attempts < 0
            or type(record.train_summary) is not dict
            or any(type(key) is not str for key in record.train_summary)
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in record.train_summary.values()
            )
        ):
            _fail("inventory records must use exact canonical authorization fields")
        raw_definition = record.definition
        if type(raw_definition) is not MethodDefinition:
            _fail("inventory definitions must use exact canonical records")
        definition = cast(MethodDefinition, raw_definition)
        if (
            type(definition.method_id) is not str
            or type(definition.family) is not str
            or definition.family not in ALLOWED_FAMILIES
            or type(definition.description) is not str
            or not definition.description.strip()
            or type(definition.status) is not str
            or definition.status not in METHOD_STATUSES
            or type(definition.assumptions) is not tuple
            or any(type(item) is not str for item in definition.assumptions)
            or type(definition.failure_conditions) is not tuple
            or any(type(item) is not str for item in definition.failure_conditions)
            or type(definition.dependencies) is not tuple
            or any(type(item) is not str for item in definition.dependencies)
            or type(definition.implementation_spec) is not dict
            or any(type(key) is not str for key in definition.implementation_spec)
        ):
            _fail("inventory definitions must use exact canonical authorization fields")
        candidate = record.candidate
        if candidate is not None:
            if type(candidate) is not MethodCandidate:
                _fail("inventory candidates must use exact canonical records")
            if (
                type(candidate.method_id) is not str
                or type(candidate.provider) is not str
                or not candidate.provider
                or type(candidate.implementation_kind) is not str
                or not candidate.implementation_kind
                or type(candidate.implementation) is not dict
                or any(type(key) is not str for key in candidate.implementation)
                or type(candidate.version) is not int
                or candidate.version <= 0
                or (
                    candidate.parent_version is not None
                    and (
                        type(candidate.parent_version) is not int
                        or candidate.parent_version <= 0
                    )
                )
            ):
                _fail("inventory candidates must use exact canonical authorization fields")
        try:
            definition.__post_init__()
            if candidate is not None:
                candidate.__post_init__()
            MethodRecord.__post_init__(record)
        except ChampionProposalError:
            raise
        except Exception as error:
            raise ChampionProposalError("inventory contains an invalid method record") from error
        name = _public_identifier(definition.method_id, "inventory method name")
        for dependency in definition.dependencies:
            _public_identifier(dependency, "inventory dependency name")
        if candidate is not None:
            _public_identifier(candidate.method_id, "inventory candidate name")
        names.append(name)
    _require_normalized_unique(tuple(names), "inventory method names")
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
    _public_identifier(canonical.name, "recipe name")
    if type(canonical.kind) is not str or canonical.kind not in _OPERATORS:
        _fail("recipe kind must be an exact supported operator")
    if type(canonical.parents) is not tuple or not canonical.parents:
        _fail("recipe parents must be an exact nonempty tuple")
    for parent in canonical.parents:
        _public_identifier(parent, "recipe parent")
    _require_normalized_unique(canonical.parents, "recipe parents")
    _public_identifier(canonical.fallback_parent, "recipe fallback parent")
    if type(canonical.assumptions) is not tuple or not canonical.assumptions:
        _fail("recipe assumptions must be an exact nonempty tuple")
    assumption_ids: list[str] = []
    try:
        for assumption in canonical.assumptions:
            if type(assumption) is not EvolutionAssumption:
                _fail("recipe assumptions must be exact EvolutionAssumption values")
            _public_identifier(assumption.assumption_id, "assumption ID")
            _public_identifier(assumption.candidate_name, "assumption candidate name")
            _public_identifier(assumption.feature, "assumption feature")
            if (
                type(assumption.direction) is not str
                or type(assumption.horizon_region) is not str
                or type(assumption.operator) is not str
            ):
                _fail("assumption enums must use exact strings")
            EvolutionAssumption.__post_init__(assumption)
            assumption_ids.append(assumption.assumption_id)
        _require_normalized_unique(tuple(assumption_ids), "assumption IDs")
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


def _closed_assumption_payload(
    assumption: object,
    *,
    recipe_kind: str,
    parents: tuple[str, ...],
) -> dict[str, object]:
    if type(assumption) is not dict:
        _fail("each proposal assumption must be an exact JSON object")
    raw = cast(dict[object, object], assumption)
    if any(type(key) is not str for key in raw) or set(raw) != set(_ASSUMPTION_FIELDS):
        _fail("proposal assumption fields must match the closed structural schema")
    if any(type(raw[field]) is not str for field in _ASSUMPTION_FIELDS):
        _fail("proposal assumption values must be exact structural strings")

    candidate_name = _public_identifier(raw["candidate_name"], "candidate name")
    feature = _public_identifier(raw["feature"], "assumption feature")
    direction = cast(str, raw["direction"])
    horizon_region = cast(str, raw["horizon_region"])
    operator = cast(str, raw["operator"])
    if candidate_name not in parents:
        _fail("proposal assumption candidate must be one of the recipe parents")
    if feature not in _FEATURES:
        _fail("proposal assumption feature must be from the closed feature set")
    if direction not in _DIRECTIONS:
        _fail("proposal assumption direction must be from the closed direction set")
    if horizon_region not in _HORIZON_REGIONS:
        _fail("proposal horizon region must be from the closed region set")
    if operator not in _OPERATORS or operator != recipe_kind:
        _fail("proposal assumption operator must match the closed recipe kind")

    return {
        "candidate_name": candidate_name,
        "feature": feature,
        "direction": direction,
        "horizon_region": horizon_region,
        "operator": operator,
    }


def _closed_recipe_payload(
    recipe: object,
    *,
    inventory_names: tuple[str, ...],
) -> dict[str, object]:
    if type(recipe) is not dict:
        _fail("each Champion recipe must be an exact JSON object")
    raw = cast(dict[object, object], recipe)
    if any(type(key) is not str for key in raw) or set(raw) != set(_RECIPE_FIELDS):
        _fail("proposal recipe fields must match the closed structural schema")
    if type(raw["kind"]) is not str or raw["kind"] not in _OPERATORS:
        _fail("proposal recipe kind must be from the closed operator set")
    if type(raw["parents"]) is not list or not raw["parents"]:
        _fail("proposal recipe parents must be an exact nonempty array")
    parent_values = cast(list[object], raw["parents"])
    if any(type(parent) is not str for parent in parent_values):
        _fail("proposal recipe parents must be exact candidate strings")
    parents = tuple(
        _public_identifier(parent, "recipe parent") for parent in parent_values
    )
    _require_normalized_unique(parents, "recipe parents")
    if any(parent not in inventory_names for parent in parents):
        _fail("recipe contains an unknown parent")
    fallback_parent = _public_identifier(
        raw["fallback_parent"], "recipe fallback parent"
    )
    if fallback_parent not in parents:
        _fail("proposal fallback parent must be one of the recipe parents")
    assumptions = raw["assumptions"]
    if (
        type(assumptions) is not list
        or not 1 <= len(assumptions) <= _MAX_ASSUMPTIONS_PER_RECIPE
    ):
        _fail("each Champion recipe requires one through three assumptions")
    recipe_kind = cast(str, raw["kind"])
    return {
        "kind": recipe_kind,
        "parents": list(parents),
        "fallback_parent": fallback_parent,
        "assumptions": [
            _closed_assumption_payload(
                assumption,
                recipe_kind=recipe_kind,
                parents=parents,
            )
            for assumption in cast(list[object], assumptions)
        ],
    }


def _canonical_closed_recipe(recipe: dict[str, object]) -> str:
    try:
        return json.dumps(
            recipe,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ChampionProposalError(
            "closed Champion structure must be canonical standards JSON"
        ) from error


def _allocate_host_identifier(base: str, reserved: set[str]) -> str:
    for suffix in range(_MAX_ID_ALLOCATION_ATTEMPTS):
        candidate = base if suffix == 0 else f"{base}_{suffix}"
        normalized = _canonical_text(candidate)
        if normalized not in reserved:
            reserved.add(normalized)
            return candidate
    _fail("unable to allocate a bounded collision-free Champion identifier")
    raise AssertionError("unreachable")


def _host_assumption_payload(
    assumption: dict[str, object],
    *,
    assumption_id: str,
) -> dict[str, object]:
    candidate_name = cast(str, assumption["candidate_name"])
    feature = cast(str, assumption["feature"])
    direction = cast(str, assumption["direction"])
    horizon_region = cast(str, assumption["horizon_region"])
    operator = cast(str, assumption["operator"])
    return {
        "assumption_id": assumption_id,
        **assumption,
        "rationale": (
            f"The {feature} feature supports {candidate_name} for {direction} "
            f"{horizon_region} {operator} structure."
        ),
        "failure_condition": (
            f"The {feature} feature may not support {candidate_name} for "
            f"{direction} {horizon_region} {operator} structure."
        ),
    }


def _host_recipe_payload(
    recipe: dict[str, object],
    *,
    recipe_index: int,
    reserved: set[str],
) -> dict[str, object]:
    recipe_name = _allocate_host_identifier(
        f"proposed_recipe_{recipe_index}", reserved
    )
    assumptions = cast(list[dict[str, object]], recipe["assumptions"])
    return {
        "name": recipe_name,
        "kind": recipe["kind"],
        "parents": recipe["parents"],
        "fallback_parent": recipe["fallback_parent"],
        "assumptions": [
            _host_assumption_payload(
                assumption,
                assumption_id=_allocate_host_identifier(
                    f"proposed_assumption_{recipe_index}_{assumption_index}",
                    reserved,
                ),
            )
            for assumption_index, assumption in enumerate(assumptions)
        ],
    }


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
    if len(raw_recipes) > upper:
        _fail("Champion response recipe count is outside the configured bounds")

    closed_recipes = tuple(
        _closed_recipe_payload(raw_recipe, inventory_names=inventory_names)
        for raw_recipe in raw_recipes
    )
    canonical_structures = tuple(
        _canonical_closed_recipe(recipe) for recipe in closed_recipes
    )
    if len(canonical_structures) != len(set(canonical_structures)):
        _fail("Champion response contains a duplicate closed recipe structure")
    if len(closed_recipes) < lower:
        _fail("Champion response recipe count is outside the configured bounds")

    reserved = {_canonical_text(name) for name in inventory_names}
    parsed: list[ChampionRecipe] = []
    for recipe_index, closed_recipe in enumerate(closed_recipes):
        recipe_payload = _host_recipe_payload(
            closed_recipe,
            recipe_index=recipe_index,
            reserved=reserved,
        )
        try:
            recipe = parse_champion_recipe(recipe_payload)
        except (ChampionContractError, KeyError, TypeError, ValueError) as error:
            raise ChampionProposalError("Champion recipe violates its exact schema") from error
        _validate_recipe(recipe, inventory_names=inventory_names)
        if _canonical_text(recipe.name) in {
            _canonical_text(name) for name in inventory_names
        }:
            _fail("Champion recipe names cannot collide with the candidate inventory")
        parsed.append(recipe)
    names = tuple(recipe.name for recipe in parsed)
    _require_normalized_unique(names, "Champion recipe names")
    return tuple(parsed)


def _validated_parent(parent: object, inventory_names: tuple[str, ...]) -> ChampionRecipe:
    return _validate_recipe(parent, inventory_names=inventory_names)


def _structural_recipe_payload(recipe: ChampionRecipe) -> dict[str, object]:
    return {
        "kind": recipe.kind,
        "parents": list(recipe.parents),
        "fallback_parent": recipe.fallback_parent,
        "assumptions": [
            {
                "candidate_name": assumption.candidate_name,
                "feature": assumption.feature,
                "direction": assumption.direction,
                "horizon_region": assumption.horizon_region,
                "operator": assumption.operator,
            }
            for assumption in recipe.assumptions
        ],
    }


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
        "parent": _structural_recipe_payload(validated_parent),
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
            "allowed_directions": list(_DIRECTIONS),
            "allowed_horizon_regions": list(_HORIZON_REGIONS),
            "relational_constraints": [
                "fallback_parent must be one of the same recipe's parents",
                "each assumption candidate_name must be one of the same recipe's parents",
                "each assumption operator must equal the same recipe's kind",
            ],
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
        if type(row.profile) is not TaskProfile or any(
            type(getattr(row.profile, field)) is not int
            or not 1 <= getattr(row.profile, field) <= _MAX_PROFILE_LENGTH
            for field in ("history_length", "horizon")
        ):
            _fail("Build profile integer features must be exact and bounded")
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
    try:
        if feature == "horizon_ratio":
            value: object = profile.horizon / profile.history_length
        elif feature in {"history_length", "horizon"}:
            value = getattr(profile, feature)
        elif feature in _PROFILE_FLOAT_FEATURES:
            value = getattr(profile, feature)
        else:
            return None
    except (ArithmeticError, TypeError, ValueError) as error:
        raise ChampionProposalError(
            "Build feature arithmetic must be finite and bounded"
        ) from error
    if type(value) not in {int, float}:
        return None
    try:
        number = float(cast(int | float, value))
    except (OverflowError, TypeError, ValueError) as error:
        raise ChampionProposalError(
            "Build feature values must be finite and bounded"
        ) from error
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
    try:
        quantiles = tuple(
            float(linear_quantile(list(values), level)) for level in _QUANTILES
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        raise ChampionProposalError(
            "Build feature quantiles must be finite and bounded"
        ) from error
    if any(not math.isfinite(value) for value in quantiles):
        _fail("Build feature quantiles must be finite and bounded")
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
