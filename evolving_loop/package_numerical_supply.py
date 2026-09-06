"""Immutable Numerical supply releases and bounded package projections."""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from common.payload import canonical_json_bytes
from evolving_loop.data import ContextTask
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from numerical_agent.evolution.champion import (
    ChampionContractError,
    ChampionRelease,
    FittedChampionPolicy,
    _parse_fitted_policy,
    champion_fingerprint,
    parse_champion_recipe,
    parse_champion_release,
)
from numerical_agent.evolution.champion_runtime import execute_champion
from numerical_agent.evolution.analysis_skills_template import analyze_series
from numerical_agent.evolution.morphology import (
    AssumptionGrounding,
    MorphologyCard,
    MorphologyObservation,
    MorphologyToolCall,
)
from numerical_agent.evolution.morphology_consistency import (
    check_morphology_assumptions,
)
from numerical_agent.evolution.numerical_handoff import safe_retrieval_projection
from numerical_agent.evolution.numerical_package import (
    NumericalForecastPackage,
    RankedNumericalForecast,
    valid_forecast,
)
from numerical_agent.evolution.numerical_selector import DecisionPolicy, SelectionDecision


NumericalSupplyFamily = Literal[
    "statistical", "tsfm", "combined", "atlas_overlay"
]
NumericalMaterializerKind = Literal[
    "dictionary", "champion", "atlas", "bounded_overlay"
]

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"n[0-9]{3}\Z")
_FAMILIES = ("statistical", "tsfm", "combined", "atlas_overlay")
_MATERIALIZER_KINDS = frozenset(
    {"dictionary", "champion", "atlas", "bounded_overlay"}
)
_RELEASE_KEYS = frozenset(
    {
        "schema_version",
        "version",
        "parent_sha256",
        "anchor_release_payload",
        "alternatives",
        "atlas_release_sha256",
        "source_fingerprints",
        "runtime_fingerprints",
    }
)
_ALTERNATIVE_KEYS = frozenset(
    {
        "candidate_id",
        "family",
        "materializer_kind",
        "recipe_payload",
        "full_build_policy_payload",
        "build_fold_policy_payloads",
        "assumption_ids",
        "failure_conditions",
    }
)


class NumericalSupplyError(ValueError):
    """Raised when a Numerical supply release violates its closed contract."""


def numerical_runtime_implementation(
    root: Path,
    file_sha256: Callable[[Path], str],
) -> Mapping[str, str]:
    """Hash the closed implementation surface that can alter Numerical packages."""
    agent = root.parent / "numerical_agent" / "evolution"
    paths = {
        "analysis_skills": agent / "analysis_skills_template.py",
        "assumptions": agent / "assumptions.py",
        "champion_contract": agent / "champion.py",
        "champion_runtime": agent / "champion_runtime.py",
        "handoff": agent / "numerical_handoff.py",
        "loop": agent / "numerical_loop.py",
        "morphology": agent / "morphology.py",
        "morphology_consistency": agent / "morphology_consistency.py",
        "numerical_package": agent / "numerical_package.py",
        "portfolio": agent / "portfolio.py",
        "screening": agent / "screening.py",
        "selector": agent / "numerical_selector.py",
        "specialist_atlas": agent / "specialist_atlas.py",
        "task_local_evolution": agent / "task_local_evolution.py",
        "metrics": root.parent / "common" / "metrics.py",
        "runner": root / "run_package_coevolution.py",
        "materializer": root / "package_numerical_evolution.py",
        "package_metrics": root / "package_metrics.py",
        "stage_runner": root / "package_stage_runner.py",
        "task_feedback": root / "package_task_feedback.py",
        "supply": root / "package_numerical_supply.py",
    }
    return MappingProxyType(
        {name: file_sha256(path) for name, path in sorted(paths.items())}
    )


def _fail(message: str) -> None:
    raise NumericalSupplyError(message)


def _require_hash(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{field_name} must be a non-empty string")
    return value


def _freeze_payload(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            _fail("supply payload mappings require string keys")
        return MappingProxyType(
            {str(key): _freeze_payload(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze_fingerprints(value: object, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        _fail(f"{field_name} must be a mapping")
    fingerprints = {
        _require_text(key, f"{field_name} key"): _require_hash(
            item, f"{field_name}[{key!r}]"
        )
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(fingerprints.items())))


def _parse_policy(payload: object, field_name: str) -> FittedChampionPolicy:
    try:
        return _parse_fitted_policy(payload)
    except ChampionContractError as error:
        raise NumericalSupplyError(f"{field_name} must be a fitted Champion policy") from error


@dataclass(frozen=True)
class NumericalAlternativeSpec:
    candidate_id: str
    family: NumericalSupplyFamily
    materializer_kind: NumericalMaterializerKind
    recipe_payload: Mapping[str, object]
    full_build_policy_payload: Mapping[str, object]
    build_fold_policy_payloads: tuple[tuple[int, Mapping[str, object]], ...]
    assumption_ids: tuple[str, ...]
    failure_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        candidate_id = _require_text(self.candidate_id, "candidate_id")
        if self.family not in _FAMILIES:
            _fail("alternative family is unsupported")
        if self.materializer_kind not in _MATERIALIZER_KINDS:
            _fail("alternative materializer_kind is unsupported")
        if not isinstance(self.recipe_payload, Mapping):
            _fail("recipe_payload must be a mapping")
        try:
            recipe = parse_champion_recipe(dict(self.recipe_payload))
        except ChampionContractError as error:
            raise NumericalSupplyError("recipe_payload must be a Champion recipe") from error
        if not isinstance(self.full_build_policy_payload, Mapping):
            _fail("full_build_policy_payload must be a mapping")
        full_policy = _parse_policy(
            dict(self.full_build_policy_payload), "full_build_policy_payload"
        )
        if full_policy.recipe != recipe:
            _fail("full-Build policy recipe must match recipe_payload")

        supplied_folds = self.build_fold_policy_payloads
        if type(supplied_folds) is not tuple:
            _fail("build_fold_policy_payloads must be a tuple")
        folds: list[tuple[int, FittedChampionPolicy]] = []
        for pair in supplied_folds:
            if type(pair) is not tuple or len(pair) != 2:
                _fail("build_fold_policy_payloads must contain (fold, policy) tuples")
            fold, policy_payload = pair
            if type(fold) is not int or not 0 <= fold <= 4:
                _fail("Build folds must be integers from 0 through 4")
            if not isinstance(policy_payload, Mapping):
                _fail("Build fold policies must be mappings")
            policy = _parse_policy(policy_payload, f"Build fold {fold} policy")
            if policy.recipe != recipe:
                _fail("Build fold policy recipe must match recipe_payload")
            folds.append((fold, policy))
        if tuple(fold for fold, _policy in folds) != (0, 1, 2, 3, 4):
            _fail("Build folds must contain exactly one fitted policy for folds 0 through 4")

        assumption_ids = tuple(
            _require_text(item, "assumption_ids item") for item in self.assumption_ids
        )
        failure_conditions = tuple(
            _require_text(item, "failure_conditions item")
            for item in self.failure_conditions
        )
        if not assumption_ids or len(assumption_ids) != len(failure_conditions):
            _fail("assumption_ids and failure_conditions must be aligned and non-empty")
        if assumption_ids != tuple(
            assumption.assumption_id for assumption in recipe.assumptions
        ) or failure_conditions != tuple(
            assumption.failure_condition for assumption in recipe.assumptions
        ):
            _fail("alternative assumptions must match its fitted policy recipe")

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "recipe_payload", _freeze_payload(recipe.to_payload()))
        object.__setattr__(
            self,
            "full_build_policy_payload",
            _freeze_payload(full_policy.to_payload()),
        )
        object.__setattr__(
            self,
            "build_fold_policy_payloads",
            tuple(
                (fold, _freeze_payload(policy.to_payload()))
                for fold, policy in folds
            ),
        )
        object.__setattr__(self, "assumption_ids", assumption_ids)
        object.__setattr__(self, "failure_conditions", failure_conditions)

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "materializer_kind": self.materializer_kind,
            "recipe_payload": _plain(self.recipe_payload),
            "full_build_policy_payload": _plain(self.full_build_policy_payload),
            "build_fold_policy_payloads": [
                [fold, _plain(policy)]
                for fold, policy in self.build_fold_policy_payloads
            ],
            "assumption_ids": list(self.assumption_ids),
            "failure_conditions": list(self.failure_conditions),
        }


@dataclass(frozen=True)
class NumericalSupplyRelease:
    schema_version: int
    version: str
    parent_sha256: str | None
    anchor_release_payload: Mapping[str, object]
    alternatives: tuple[NumericalAlternativeSpec, ...]
    atlas_release_sha256: str | None
    source_fingerprints: Mapping[str, str]
    runtime_fingerprints: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            _fail("Numerical supply schema_version must be 1")
        if type(self.version) is not str or _VERSION.fullmatch(self.version) is None:
            _fail("Numerical supply version must use nNNN format")
        if self.version == "n000":
            if self.parent_sha256 is not None:
                _fail("the n000 seed release must not have a parent")
        else:
            _require_hash(self.parent_sha256, "parent_sha256")
        if not isinstance(self.anchor_release_payload, Mapping):
            _fail("anchor_release_payload must be a mapping")
        try:
            anchor = parse_champion_release(dict(self.anchor_release_payload))
        except ChampionContractError as error:
            raise NumericalSupplyError("anchor_release_payload must be a ChampionRelease") from error

        alternatives = tuple(self.alternatives)
        if any(not isinstance(item, NumericalAlternativeSpec) for item in alternatives):
            _fail("alternatives must contain NumericalAlternativeSpec values")
        if len(alternatives) > 4:
            _fail("Numerical supply permits at most four additional alternatives")
        candidate_ids = tuple(item.candidate_id for item in alternatives)
        if len(candidate_ids) != len(set(candidate_ids)):
            _fail("Numerical supply candidate IDs must be unique")
        families = tuple(item.family for item in alternatives)
        if len(families) != len(set(families)):
            _fail("Numerical supply permits one alternative per family")
        if self.version != "n000":
            for item in alternatives:
                if all(
                    policy == item.full_build_policy_payload
                    for _fold, policy in item.build_fold_policy_payloads
                ):
                    _fail("non-seed alternatives must bind cross-fitted Build policies")

        if self.atlas_release_sha256 is not None:
            _require_hash(self.atlas_release_sha256, "atlas_release_sha256")
        source = _freeze_fingerprints(self.source_fingerprints, "source_fingerprints")
        runtime = _freeze_fingerprints(self.runtime_fingerprints, "runtime_fingerprints")

        object.__setattr__(self, "anchor_release_payload", _freeze_payload(anchor.to_payload()))
        object.__setattr__(self, "alternatives", alternatives)
        object.__setattr__(self, "source_fingerprints", source)
        object.__setattr__(self, "runtime_fingerprints", runtime)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_payload())).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "parent_sha256": self.parent_sha256,
            "anchor_release_payload": _plain(self.anchor_release_payload),
            "alternatives": [item.to_payload() for item in self.alternatives],
            "atlas_release_sha256": self.atlas_release_sha256,
            "source_fingerprints": dict(self.source_fingerprints),
            "runtime_fingerprints": dict(self.runtime_fingerprints),
        }


def parse_numerical_supply_release(payload: object) -> NumericalSupplyRelease:
    """Parse the exact canonical Numerical supply release schema."""
    if type(payload) is not dict or set(payload) != _RELEASE_KEYS:
        _fail("Numerical supply release fields must exactly match its schema")
    alternatives = payload["alternatives"]
    if type(alternatives) is not list:
        _fail("alternatives must be a JSON array")
    parsed_alternatives: list[NumericalAlternativeSpec] = []
    for item in alternatives:
        if type(item) is not dict or set(item) != _ALTERNATIVE_KEYS:
            _fail("Numerical supply alternative fields must exactly match its schema")
        folds = item["build_fold_policy_payloads"]
        if type(folds) is not list:
            _fail("build_fold_policy_payloads must be a JSON array")
        parsed_folds: list[tuple[int, Mapping[str, object]]] = []
        for pair in folds:
            if type(pair) is not list or len(pair) != 2 or not isinstance(pair[1], Mapping):
                _fail("Build fold policies must be two-item JSON arrays")
            parsed_folds.append((pair[0], pair[1]))
        if type(item["assumption_ids"]) is not list or type(item["failure_conditions"]) is not list:
            _fail("alternative assumptions must be JSON arrays")
        parsed_alternatives.append(
            NumericalAlternativeSpec(
                candidate_id=item["candidate_id"],
                family=item["family"],
                materializer_kind=item["materializer_kind"],
                recipe_payload=item["recipe_payload"],
                full_build_policy_payload=item["full_build_policy_payload"],
                build_fold_policy_payloads=tuple(parsed_folds),
                assumption_ids=tuple(item["assumption_ids"]),
                failure_conditions=tuple(item["failure_conditions"]),
            )
        )
    return NumericalSupplyRelease(
        schema_version=payload["schema_version"],
        version=payload["version"],
        parent_sha256=payload["parent_sha256"],
        anchor_release_payload=payload["anchor_release_payload"],
        alternatives=tuple(parsed_alternatives),
        atlas_release_sha256=payload["atlas_release_sha256"],
        source_fingerprints=payload["source_fingerprints"],
        runtime_fingerprints=payload["runtime_fingerprints"],
    )


def _forecast_sha256(forecast: tuple[float, ...]) -> str:
    return hashlib.sha256(
        canonical_json_bytes({"forecast": list(forecast)})
    ).hexdigest()


def _materialized_item(
    materialized: Mapping[str, RankedNumericalForecast],
    *,
    candidate_id: str,
    family: str | None,
    horizon: int,
) -> RankedNumericalForecast | None:
    item = materialized.get(candidate_id)
    if not isinstance(item, RankedNumericalForecast) or item.name != candidate_id:
        return None
    if family is not None and item.family != family:
        return None
    if not valid_forecast(item.forecast, horizon):
        return None
    return item


def _supply_parent_fingerprint(release: NumericalSupplyRelease) -> str:
    if release.parent_sha256 is not None:
        return release.parent_sha256
    return hashlib.sha256(
        canonical_json_bytes({"parent_sha256": None})
    ).hexdigest()


def _validate_champion_provenance(
    package: NumericalForecastPackage,
    release: NumericalSupplyRelease,
) -> None:
    try:
        anchor = parse_champion_release(_plain(release.anchor_release_payload))
    except ChampionContractError as error:  # pragma: no cover - release validates it
        raise AssertionError("validated supply anchor became unparsable") from error
    expected = {
        "champion_release": champion_fingerprint(anchor),
        "champion_recipe": champion_fingerprint(anchor.policy.recipe),
        "champion_assumptions": champion_fingerprint(anchor.policy.recipe.assumptions),
    }
    actual = package.component_fingerprints
    if any(actual.get(key) != value for key, value in expected.items()):
        _fail("package Champion provenance does not match the supply anchor")


def _morphology_kind(source: NumericalForecastPackage) -> tuple[str, float] | None:
    profile = source.task_profile
    choices: list[tuple[str, float]] = []
    if profile.periodicity_periods and profile.periodicity_confidence >= 0.25:
        choices.append(("seasonality", profile.periodicity_confidence))
    if profile.trend_direction != "flat" and profile.trend_strength >= 0.25:
        choices.append(("trend", profile.trend_strength))
    if profile.zero_fraction > 0.3 or profile.intermittency_adi > 1.32:
        choices.append(
            (
                "intermittency",
                max(
                    profile.zero_fraction,
                    min(1.0, max(0.0, profile.intermittency_adi - 1.0) / 2.0),
                ),
            )
        )
    if (
        profile.recent_regime_start is not None
        and profile.recent_regime_confidence >= 0.25
    ):
        choices.append(("regime", profile.recent_regime_confidence))
    if profile.noise_relative_scale >= 0.25 or profile.outlier_fraction >= 0.1:
        choices.append(
            (
                "noise",
                min(1.0, max(profile.noise_relative_scale, profile.outlier_fraction)),
            )
        )
    if profile.likely_stationary or (
        profile.trend_direction == "flat" and profile.trend_strength < 0.25
    ):
        choices.append(("level", max(0.5, profile.stationarity_score)))
    if not choices:
        return None
    return max(choices, key=lambda item: (item[1], item[0]))


def _verified_assumption_projection(
    source: NumericalForecastPackage,
    release: NumericalSupplyRelease,
    retained: Sequence[RankedNumericalForecast],
    materialized: Mapping[str, RankedNumericalForecast],
    *,
    history: Sequence[float] | None,
    task_fold: int | None,
    decision_policy: DecisionPolicy | None,
    min_successful_folds: int | None,
) -> tuple[
    MorphologyCard | None,
    tuple[AssumptionGrounding, ...],
    Mapping[str, str],
    tuple[Mapping[str, str], ...],
]:
    if history is None:
        return (
            source.morphology_card,
            source.accepted_assumptions,
            source.rejected_assumptions,
            source.retrieval_handoff,
        )
    if task_fold is not None and (
        type(task_fold) is not int or not 0 <= task_fold <= 4
    ):
        return None, (), {}, ()
    try:
        values = tuple(float(value) for value in history)
    except (TypeError, ValueError):
        return None, (), {}, ()
    if (
        len(values) < 2
        or len(values) != source.task_profile.history_length
        or not all(math.isfinite(value) for value in values)
    ):
        return None, (), {}, ()
    selected_kind = _morphology_kind(source)
    if selected_kind is None:
        return None, (), {}, ()

    full_call = MorphologyToolCall("host_full_history", "analyze_series", 0, len(values))
    recent_start = max(1, len(values) // 2)
    recent_call = MorphologyToolCall(
        "host_recent_history", "analyze_series", recent_start, len(values)
    )
    calls = (full_call, recent_call)
    observations = (
        MorphologyObservation(
            full_call,
            analyze_series(values, source.task_profile.frequency),
        ),
        MorphologyObservation(
            recent_call,
            analyze_series(values[recent_start:], source.task_profile.frequency),
        ),
    )
    forecasts = {name: item.forecast for name, item in materialized.items()}
    diagnostics = {name: item.diagnostics for name, item in materialized.items()}
    retained_by_name = {item.name: item for item in retained}
    kind, confidence = selected_kind
    proposals: list[AssumptionGrounding] = []
    for specification in release.alternatives:
        candidate_id = specification.candidate_id
        try:
            payload = (
                specification.full_build_policy_payload
                if task_fold is None
                else dict(specification.build_fold_policy_payloads)[task_fold]
            )
            policy = _parse_policy(
                _plain(payload),
                f"release policy for {candidate_id!r}",
            )
        except Exception:
            continue
        candidate = retained_by_name.get(candidate_id)
        if candidate is None or candidate.name == retained[0].name:
            continue
        try:
            execution = execute_champion(
                policy,
                forecasts,
                diagnostics,
                source.task_profile,
                values,
                source.task_profile.horizon,
            )
        except Exception:
            continue
        if (
            execution.fallback_reason is not None
            or not execution.activated_assumptions
            or tuple(execution.forecast) != candidate.forecast
        ):
            continue
        proposals.append(
            AssumptionGrounding(
                assumption_id=f"{candidate_id}__host_verified",
                kind=kind,
                claim=(
                    f"The frozen {candidate.family} policy is active under the "
                    "observed historical morphology."
                ),
                failure_condition=(
                    "The observed morphology or the frozen policy activation "
                    "condition no longer holds."
                ),
                supporting_call_ids=tuple(call.call_id for call in calls),
                candidate_names=(candidate_id,),
                prior_confidence=max(0.25, min(1.0, confidence)),
            )
        )
    if not proposals:
        return None, (), {}, ()

    card = MorphologyCard(
        short_term="Host-generated history-only morphology for frozen alternatives.",
        long_term="Only host-validated candidate assumptions may reach Retrieval.",
        tool_calls=calls,
        observations=observations,
        assumptions=tuple(proposals),
    )
    retained_diagnostics = {item.name: item.diagnostics for item in retained}
    retained_forecasts = {item.name: item.forecast for item in retained}
    consistency = check_morphology_assumptions(
        card,
        profile=source.task_profile,
        active_names=tuple(item.name for item in retained),
        diagnostics=retained_diagnostics,
        forecasts=retained_forecasts,
        policy=decision_policy,
        min_successful_folds=min_successful_folds,
        protected_anchor_name=retained[0].name,
    )
    return (card, *safe_retrieval_projection(consistency.accepted, consistency.rejected))


def bound_numerical_package(
    source: NumericalForecastPackage,
    release: NumericalSupplyRelease,
    materialized: Mapping[str, RankedNumericalForecast],
    *,
    history: Sequence[float] | None = None,
    task_fold: int | None = None,
    decision_policy: DecisionPolicy | None = None,
    min_successful_folds: int | None = None,
) -> NumericalForecastPackage:
    """Project a source package onto a bounded, diverse supply release."""
    if not isinstance(source, NumericalForecastPackage):
        _fail("source must be a NumericalForecastPackage")
    if not isinstance(release, NumericalSupplyRelease):
        _fail("release must be a NumericalSupplyRelease")
    if not isinstance(materialized, Mapping):
        _fail("materialized forecasts must be a mapping")
    _validate_champion_provenance(source, release)
    if source.protected_baseline.name == "atlas_70_30":
        _fail("fixed Atlas blend cannot be the protected source anchor")
    anchor = _materialized_item(
        materialized,
        candidate_id=source.protected_baseline.name,
        family=None,
        horizon=source.task_profile.horizon,
    )
    if anchor is None:
        _fail("materialized forecasts must contain the source protected anchor")
    if anchor != source.protected_baseline:
        _fail("materialized anchor must equal the exact protected anchor")

    retained = [anchor]
    retained_names = {anchor.name}
    retained_vectors = {_forecast_sha256(anchor.forecast)}
    by_family = {item.family: item for item in release.alternatives}
    for family in _FAMILIES:
        specification = by_family.get(family)
        if specification is None or specification.candidate_id in retained_names:
            continue
        candidate = _materialized_item(
            materialized,
            candidate_id=specification.candidate_id,
            family=family,
            horizon=source.task_profile.horizon,
        )
        if candidate is None:
            continue
        vector_sha256 = _forecast_sha256(candidate.forecast)
        if vector_sha256 in retained_vectors:
            continue
        retained.append(candidate)
        retained_names.add(candidate.name)
        retained_vectors.add(vector_sha256)

    ranked = tuple(
        RankedNumericalForecast(
            rank=index,
            name=item.name,
            family=item.family,
            forecast=item.forecast,
            diagnostics=item.diagnostics,
        )
        for index, item in enumerate(retained, start=1)
    )
    anchor = ranked[0]
    selection = SelectionDecision(
        mode="single",
        selected=(anchor.name,),
        weights=(1.0,),
        forecast=anchor.forecast,
        confidence=0.0,
        reason_codes=("package_safe_anchor",),
        rejected={},
        baseline_name=anchor.name,
        considered_candidates=tuple(item.name for item in ranked),
    )
    component_fingerprints = {
        **dict(source.component_fingerprints),
        "numerical_supply_release": release.fingerprint,
        "numerical_supply_parent": _supply_parent_fingerprint(release),
        "numerical_supply_runtime": hashlib.sha256(
            canonical_json_bytes(dict(release.runtime_fingerprints))
        ).hexdigest(),
    }
    morphology_card, accepted, rejected, handoff = _verified_assumption_projection(
        source,
        release,
        ranked,
        materialized,
        history=history,
        task_fold=task_fold,
        decision_policy=decision_policy,
        min_successful_folds=min_successful_folds,
    )
    component_fingerprints["morphology_card"] = (
        morphology_card.fingerprint
        if morphology_card is not None
        else hashlib.sha256(b'{"enabled":false}').hexdigest()
    )
    return NumericalForecastPackage(
        task_profile=source.task_profile,
        active_candidate_names=tuple(item.name for item in ranked),
        candidate_diagnostics={item.name: item.diagnostics for item in ranked},
        morphology_card=morphology_card,
        accepted_assumptions=accepted,
        rejected_assumptions=rejected,
        selection_decision=selection,
        final_forecast=anchor.forecast,
        protected_baseline=anchor,
        ranked_alternatives=ranked,
        retrieval_handoff=handoff,
        component_fingerprints=component_fingerprints,
        fallback_reason=source.fallback_reason,
    )


def build_package_registry(
    tasks: Sequence[ContextTask],
    release: NumericalSupplyRelease,
    package_builder: Callable[
        [ContextTask, NumericalSupplyRelease], NumericalForecastPackage
    ],
) -> FrozenNumericalPackageRegistry:
    """Build a registry whose exact task universe is bound to one supply release."""
    if not isinstance(release, NumericalSupplyRelease):
        _fail("registry release must be a NumericalSupplyRelease")
    if not callable(package_builder):
        _fail("package_builder must be callable")
    supplied_tasks = tuple(tasks)
    if any(not isinstance(task, ContextTask) for task in supplied_tasks):
        _fail("registry tasks must be ContextTask records")
    entries = tuple((task, package_builder(task, release)) for task in supplied_tasks)
    for _task, package in entries:
        _validate_champion_provenance(package, release)
    return FrozenNumericalPackageRegistry(
        entries,
        release_sha256=release.fingerprint,
        expected_task_ids=tuple(sorted(task.numeric.task_id for task in supplied_tasks)),
    )


__all__ = [
    "NumericalAlternativeSpec",
    "NumericalMaterializerKind",
    "NumericalSupplyError",
    "NumericalSupplyFamily",
    "NumericalSupplyRelease",
    "bound_numerical_package",
    "build_package_registry",
    "numerical_runtime_implementation",
    "parse_numerical_supply_release",
]
