"""Immutable, fail-closed contracts for frozen Numerical Champions."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Literal, Mapping


AssumptionOperator = Literal[
    "select", "route", "horizon_route", "weighted", "median", "bounded_overlay"
]

_OPERATORS = frozenset({
    "select", "route", "horizon_route", "weighted", "median", "bounded_overlay",
})
_DIRECTIONS = frozenset({"above", "below"})
_HORIZON_REGIONS = frozenset({"early", "late", "full"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ChampionContractError(ValueError):
    """Raised when an untrusted Champion payload violates its closed schema."""


def _fail(message: str) -> None:
    raise ChampionContractError(message)


def _require_identifier(value: object, field_name: str) -> str:
    if type(value) is not str or not value or not value.isidentifier():
        _fail(f"{field_name} must be a non-empty Python identifier")
    return value


def _require_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        _fail(f"{field_name} must be a non-empty string")
    return value


def _require_float(value: object, field_name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _fail(f"{field_name} must be a finite float")
    return value


def _require_hash(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{field_name} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _require_payload(payload: object, fields: set[str], context: str) -> dict[str, object]:
    if type(payload) is not dict:
        _fail(f"{context} must be a JSON object")
    if set(payload) != fields:
        _fail(f"{context} fields must be exactly {sorted(fields)!r}")
    if any(type(key) is not str for key in payload):
        _fail(f"{context} keys must be strings")
    return payload


def _require_list(value: object, field_name: str) -> list[object]:
    if type(value) is not list:
        _fail(f"{field_name} must be a JSON array")
    return value


@dataclass(frozen=True)
class EvolutionAssumption:
    assumption_id: str
    candidate_name: str
    feature: str
    direction: Literal["above", "below"]
    horizon_region: Literal["early", "late", "full"]
    operator: AssumptionOperator
    rationale: str
    failure_condition: str

    def __post_init__(self) -> None:
        _require_identifier(self.assumption_id, "assumption_id")
        _require_identifier(self.candidate_name, "candidate_name")
        _require_identifier(self.feature, "feature")
        if type(self.direction) is not str or self.direction not in _DIRECTIONS:
            _fail("direction must be 'above' or 'below'")
        if type(self.horizon_region) is not str or self.horizon_region not in _HORIZON_REGIONS:
            _fail("horizon_region must be 'early', 'late', or 'full'")
        if type(self.operator) is not str or self.operator not in _OPERATORS:
            _fail("operator is unsupported")
        _require_text(self.rationale, "rationale")
        _require_text(self.failure_condition, "failure_condition")

    def to_payload(self) -> dict[str, object]:
        return {
            "assumption_id": self.assumption_id,
            "candidate_name": self.candidate_name,
            "feature": self.feature,
            "direction": self.direction,
            "horizon_region": self.horizon_region,
            "operator": self.operator,
            "rationale": self.rationale,
            "failure_condition": self.failure_condition,
        }


@dataclass(frozen=True)
class ChampionRecipe:
    name: str
    kind: AssumptionOperator
    parents: tuple[str, ...]
    fallback_parent: str
    assumptions: tuple[EvolutionAssumption, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.name, "name")
        if type(self.kind) is not str or self.kind not in _OPERATORS:
            _fail("kind is unsupported")
        if type(self.parents) is not tuple or not self.parents:
            _fail("parents must be a non-empty tuple")
        if any(type(parent) is not str for parent in self.parents):
            _fail("parents must contain strings")
        for parent in self.parents:
            _require_identifier(parent, "parent")
        if len(self.parents) != len(set(self.parents)):
            _fail("parents must be unique")
        expected_arity = 1 if self.kind == "select" else 2
        if len(self.parents) != expected_arity:
            _fail(f"{self.kind} requires exactly {expected_arity} parent(s)")
        _require_identifier(self.fallback_parent, "fallback_parent")
        if self.fallback_parent not in self.parents:
            _fail("fallback_parent must be one of parents")
        if type(self.assumptions) is not tuple or not self.assumptions:
            _fail("assumptions must be a non-empty tuple")
        if any(type(item) is not EvolutionAssumption for item in self.assumptions):
            _fail("assumptions must contain EvolutionAssumption values")
        assumption_ids = tuple(item.assumption_id for item in self.assumptions)
        if len(assumption_ids) != len(set(assumption_ids)):
            _fail("assumptions must have unique assumption_id values")
        if any(item.operator != self.kind for item in self.assumptions):
            _fail("assumption operator must match recipe kind")
        if any(item.candidate_name not in self.parents for item in self.assumptions):
            _fail("assumption candidate_name must be one of parents")

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "parents": list(self.parents),
            "fallback_parent": self.fallback_parent,
            "assumptions": [item.to_payload() for item in self.assumptions],
        }


@dataclass(frozen=True)
class FittedChampionPolicy:
    recipe: ChampionRecipe
    thresholds: tuple[tuple[str, float], ...]
    weights: tuple[float, ...] = ()
    overlay_alpha: float = 0.0
    correction_cap: float = 0.0
    horizon_split: float = 0.5

    def __post_init__(self) -> None:
        if type(self.recipe) is not ChampionRecipe:
            _fail("recipe must be a ChampionRecipe")
        if type(self.thresholds) is not tuple:
            _fail("thresholds must be a tuple")
        expected_thresholds = tuple(item.assumption_id for item in self.recipe.assumptions)
        threshold_ids: list[str] = []
        for pair in self.thresholds:
            if type(pair) is not tuple or len(pair) != 2:
                _fail("thresholds must contain (identifier, float) tuples")
            threshold_id, value = pair
            threshold_ids.append(_require_identifier(threshold_id, "threshold identifier"))
            _require_float(value, "threshold value")
        if tuple(threshold_ids) != expected_thresholds:
            _fail("thresholds must exactly follow the recipe assumption IDs")
        if type(self.weights) is not tuple:
            _fail("weights must be a tuple")
        for weight in self.weights:
            _require_float(weight, "weight")
        _require_float(self.overlay_alpha, "overlay_alpha")
        _require_float(self.correction_cap, "correction_cap")
        _require_float(self.horizon_split, "horizon_split")

        if self.recipe.kind == "weighted":
            if len(self.weights) != len(self.recipe.parents):
                _fail("weighted recipes require one weight per parent")
            if any(weight < 0.0 for weight in self.weights) or not math.isclose(
                sum(self.weights), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                _fail("weights must be non-negative and normalized to one")
        elif self.weights:
            _fail("weights must be empty outside weighted recipes")

        if self.recipe.kind == "bounded_overlay":
            if not 0.0 < self.overlay_alpha <= 1.0:
                _fail("bounded_overlay requires overlay_alpha in (0, 1]")
            if self.correction_cap <= 0.0:
                _fail("bounded_overlay requires a positive correction_cap")
        elif self.overlay_alpha != 0.0 or self.correction_cap != 0.0:
            _fail("inactive overlay fields must be canonical zeros")

        if self.recipe.kind == "horizon_route":
            if not 0.0 < self.horizon_split < 1.0:
                _fail("horizon_route requires horizon_split in (0, 1)")
        elif self.horizon_split != 0.5:
            _fail("inactive horizon_split must be 0.5")

    def to_payload(self) -> dict[str, object]:
        return {
            "recipe": self.recipe.to_payload(),
            "thresholds": [[name, value] for name, value in self.thresholds],
            "weights": list(self.weights),
            "overlay_alpha": self.overlay_alpha,
            "correction_cap": self.correction_cap,
            "horizon_split": self.horizon_split,
        }


@dataclass(frozen=True)
class ChampionRelease:
    policy: FittedChampionPolicy
    source_hashes: tuple[tuple[str, str], ...]
    metric_policy_fingerprint: str
    lineage: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.policy) is not FittedChampionPolicy:
            _fail("policy must be a FittedChampionPolicy")
        if type(self.source_hashes) is not tuple or not self.source_hashes:
            _fail("source_hashes must be a non-empty tuple")
        names: list[str] = []
        for pair in self.source_hashes:
            if type(pair) is not tuple or len(pair) != 2:
                _fail("source_hashes must contain (identifier, hash) tuples")
            name, source_hash = pair
            names.append(_require_identifier(name, "source hash name"))
            _require_hash(source_hash, "source hash")
        if tuple(names) != tuple(sorted(names)) or len(names) != len(set(names)):
            _fail("source_hashes must be uniquely sorted by identifier")
        _require_hash(self.metric_policy_fingerprint, "metric_policy_fingerprint")
        if type(self.lineage) is not tuple or not self.lineage:
            _fail("lineage must be a non-empty tuple")
        for item in self.lineage:
            _require_identifier(item, "lineage item")
        if len(self.lineage) != len(set(self.lineage)):
            _fail("lineage must be unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "policy": self.policy.to_payload(),
            "source_hashes": dict(self.source_hashes),
            "metric_policy_fingerprint": self.metric_policy_fingerprint,
            "lineage": list(self.lineage),
        }


def _parse_assumption(payload: object) -> EvolutionAssumption:
    values = _require_payload(payload, {
        "assumption_id", "candidate_name", "feature", "direction", "horizon_region",
        "operator", "rationale", "failure_condition",
    }, "assumption")
    return EvolutionAssumption(
        assumption_id=values["assumption_id"],
        candidate_name=values["candidate_name"],
        feature=values["feature"],
        direction=values["direction"],
        horizon_region=values["horizon_region"],
        operator=values["operator"],
        rationale=values["rationale"],
        failure_condition=values["failure_condition"],
    )


def parse_champion_recipe(payload: object) -> ChampionRecipe:
    """Parse a structural recipe, accepting only its exact JSON schema."""
    values = _require_payload(payload, {
        "name", "kind", "parents", "fallback_parent", "assumptions",
    }, "champion recipe")
    parents = tuple(_require_identifier(item, "parent") for item in _require_list(
        values["parents"], "parents"
    ))
    assumptions = tuple(_parse_assumption(item) for item in _require_list(
        values["assumptions"], "assumptions"
    ))
    return ChampionRecipe(
        name=values["name"],
        kind=values["kind"],
        parents=parents,
        fallback_parent=values["fallback_parent"],
        assumptions=assumptions,
    )


def _parse_fitted_policy(payload: object) -> FittedChampionPolicy:
    values = _require_payload(payload, {
        "recipe", "thresholds", "weights", "overlay_alpha", "correction_cap", "horizon_split",
    }, "fitted champion policy")
    thresholds = tuple(
        (_require_identifier(pair[0], "threshold identifier"), _require_float(pair[1], "threshold value"))
        if type(pair) is list and len(pair) == 2
        else _invalid_threshold_pair()
        for pair in _require_list(values["thresholds"], "thresholds")
    )
    weights = tuple(_require_float(weight, "weight") for weight in _require_list(
        values["weights"], "weights"
    ))
    return FittedChampionPolicy(
        recipe=parse_champion_recipe(values["recipe"]),
        thresholds=thresholds,
        weights=weights,
        overlay_alpha=_require_float(values["overlay_alpha"], "overlay_alpha"),
        correction_cap=_require_float(values["correction_cap"], "correction_cap"),
        horizon_split=_require_float(values["horizon_split"], "horizon_split"),
    )


def _invalid_threshold_pair() -> tuple[str, float]:
    _fail("thresholds must contain two-item JSON arrays")
    raise AssertionError("unreachable")


def parse_champion_release(payload: object) -> ChampionRelease:
    """Parse a frozen release and reject every schema or provenance variation."""
    values = _require_payload(payload, {
        "policy", "source_hashes", "metric_policy_fingerprint", "lineage",
    }, "champion release")
    source_hashes = values["source_hashes"]
    if type(source_hashes) is not dict:
        _fail("source_hashes must be a JSON object")
    parsed_hashes = tuple(sorted(
        (_require_identifier(name, "source hash name"), _require_hash(value, "source hash"))
        for name, value in source_hashes.items()
    ))
    lineage = tuple(_require_identifier(item, "lineage item") for item in _require_list(
        values["lineage"], "lineage"
    ))
    return ChampionRelease(
        policy=_parse_fitted_policy(values["policy"]),
        source_hashes=parsed_hashes,
        metric_policy_fingerprint=_require_hash(
            values["metric_policy_fingerprint"], "metric_policy_fingerprint"
        ),
        lineage=lineage,
    )


def _canonical_value(value: object) -> object:
    if type(value) in {
        EvolutionAssumption, ChampionRecipe, FittedChampionPolicy, ChampionRelease,
    }:
        return _canonical_value(value.to_payload())
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail("fingerprints require finite floats")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                _fail("fingerprint mappings require string keys")
            normalized[key] = _canonical_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    _fail(f"fingerprint value has unsupported type {type(value).__name__}")
    raise AssertionError("unreachable")


def champion_fingerprint(value: object) -> str:
    """Return a SHA-256 digest over strict canonical JSON for a Champion value."""
    encoded = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def preserve_parent(parent, child, *, accepted: bool):
    """Keep the exact parent object unless a child was explicitly accepted."""
    if type(accepted) is not bool:
        _fail("accepted must be a bool")
    return child if accepted else parent
