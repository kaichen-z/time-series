"""Build-only successive halving for immutable Numerical Champions.

This module has no Calibration or Dev authority.  It ranks fitted structural
children on fixed Build memberships and always returns the exact active Parent.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Callable, Literal, cast

from .champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
    champion_fingerprint,
    parse_champion_recipe,
)
from .champion_evidence import (
    ChampionComparison,
    ChampionEvidenceError,
    ChampionGateConfig,
    ChampionHistoryDiagnostic,
    ChampionTaskRow,
    ProposerEvidence,
    compare_champion,
    sanitize_build_evidence,
    score_policy,
)
from .champion_proposal import ChampionProposalError, expand_recipe
from .champion_runtime import execute_champion
from .numerical_selector import CandidateDiagnostics


_FORMAL_SIZES = (64, 16, (8, 32, 64))
_SMOKE_SIZES = (8, 2, (4, 8))
_PROFILE_FEATURES = frozenset({
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
})


class ChampionControllerError(ValueError):
    """Build evolution input or evidence violates its fixed authority."""


def _fail(message: str) -> None:
    raise ChampionControllerError(message)


def _canonical_identity(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


@dataclass(frozen=True)
class ChampionEvolutionConfig:
    """Pre-registered Build schedule and candidate gates."""

    build_size: int = 64
    calibration_size: int = 16
    screen_sizes: tuple[int, ...] = (8, 32, 64)
    generations: int = 1
    minimum_recipes: int = 5
    maximum_recipes: int = 10
    maximum_full_build_children: int = 3
    maximum_finalists: int = 3
    candidate_minimum_gain: float = 0.005
    research_target_gain: float = 0.05
    gate_config: ChampionGateConfig = field(
        default_factory=lambda: ChampionGateConfig(minimum_improved_folds=4)
    )
    build_task_ids: tuple[str, ...] = ()
    screen_task_ids: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if type(self.build_size) is not int or type(self.calibration_size) is not int:
            _fail("Build and Calibration sizes must be exact integers")
        if type(self.screen_sizes) is not tuple or any(
            type(size) is not int for size in self.screen_sizes
        ):
            _fail("screen sizes must be an exact integer tuple")
        sizing = (self.build_size, self.calibration_size, self.screen_sizes)
        if sizing not in {_FORMAL_SIZES, _SMOKE_SIZES}:
            _fail("only the formal 64/16 or deterministic 8/2 smoke schedule is allowed")
        for name in (
            "generations",
            "minimum_recipes",
            "maximum_recipes",
            "maximum_full_build_children",
            "maximum_finalists",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                _fail(f"{name} must be an exact positive integer")
        if not 5 <= self.minimum_recipes <= self.maximum_recipes <= 10:
            _fail("each generation must propose five through ten recipes")
        if self.maximum_full_build_children > 3 or self.maximum_finalists > 3:
            _fail("full Build and finalist bounds cannot exceed three")
        for name in ("candidate_minimum_gain", "research_target_gain"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                _fail(f"{name} must be a finite nonnegative float")
        if self.research_target_gain < self.candidate_minimum_gain:
            _fail("research target cannot be below the candidate threshold")
        if type(self.gate_config) is not ChampionGateConfig:
            _fail("gate_config must be an exact ChampionGateConfig")
        ChampionGateConfig.__post_init__(self.gate_config)
        if type(self.build_task_ids) is not tuple or any(
            type(task_id) is not str or not task_id for task_id in self.build_task_ids
        ):
            _fail("Build task membership must be an exact string tuple")
        if self.build_task_ids and (
            len(self.build_task_ids) != self.build_size
            or len(self.build_task_ids) != len(set(self.build_task_ids))
        ):
            _fail("Build task membership must contain the fixed unique universe")
        if type(self.screen_task_ids) is not tuple or any(
            type(stage) is not tuple for stage in self.screen_task_ids
        ):
            _fail("screen membership must be an exact tuple of exact tuples")
        if self.screen_task_ids:
            if len(self.screen_task_ids) != len(self.screen_sizes):
                _fail("screen membership must cover every fixed stage")
            for size, stage in zip(self.screen_sizes, self.screen_task_ids, strict=True):
                if (
                    len(stage) != size
                    or any(type(task_id) is not str or not task_id for task_id in stage)
                    or len(stage) != len(set(stage))
                ):
                    _fail("each screen stage must have its exact unique membership")
            for earlier, later in zip(
                self.screen_task_ids, self.screen_task_ids[1:], strict=False
            ):
                if not set(earlier).issubset(later):
                    _fail("screen membership must be successively nested")

    @property
    def fingerprint(self) -> str:
        """Fingerprint schedule, gates, thresholds, and fixed task membership."""
        return champion_fingerprint(self)

    @property
    def screen_membership_sha256(self) -> str:
        return champion_fingerprint({"screen_task_ids": self.screen_task_ids})


@dataclass(frozen=True)
class BuildAttempt:
    """The latest trusted Build evidence for one expanded fitted Child."""

    fitted_id: str
    policy: FittedChampionPolicy
    stage_task_counts: tuple[int, ...]
    comparison: ChampionComparison | None
    status: Literal[
        "invalid", "pruned", "finalist", "below_candidate_threshold"
    ]
    invalid_reason: Literal["unscorable_child"] | None = None
    score_label: Literal["adaptive_train_build_diagnostic"] = (
        "adaptive_train_build_diagnostic"
    )
    independent_generalization_claim: Literal[False] = False


@dataclass(frozen=True)
class BuildGeneration:
    """One structural proposal generation evaluated against the same Parent."""

    number: int
    mutation_parent_sha256: str
    config_fingerprint: str
    stage_counts: tuple[int, ...]
    full_build_children: int
    attempts: tuple[BuildAttempt, ...]
    finalists: tuple[FittedChampionPolicy, ...]
    feedback: ProposerEvidence


@dataclass(frozen=True)
class BuildEvolutionResult:
    """Build diagnostics and shortlist; active_parent is never a Build Child."""

    active_parent: object
    generations: tuple[BuildGeneration, ...]
    shortlist: tuple[FittedChampionPolicy, ...]
    config: ChampionEvolutionConfig
    config_fingerprint: str
    score_label: Literal["adaptive_train_build_diagnostic"] = (
        "adaptive_train_build_diagnostic"
    )
    independent_generalization_claim: Literal[False] = False


@dataclass
class _AttemptState:
    fitted_id: str
    score_name: str
    policy: FittedChampionPolicy
    stage_task_counts: tuple[int, ...] = ()
    comparison: ChampionComparison | None = None
    evidence_rows: tuple[ChampionTaskRow, ...] = ()
    invalid_reason: Literal["unscorable_child"] | None = None


def _capture_object_graph(
    value: object,
) -> tuple[tuple[object, tuple[tuple[str, object], ...]], ...]:
    """Capture exact dataclass fields so hostile callbacks can be rolled back."""
    captured: list[tuple[object, tuple[tuple[str, object], ...]]] = []
    seen: set[int] = set()

    def visit(item: object) -> None:
        identity = id(item)
        if identity in seen:
            return
        if is_dataclass(item) and not isinstance(item, type):
            seen.add(identity)
            values = tuple((entry.name, getattr(item, entry.name)) for entry in fields(item))
            captured.append((item, values))
            for _, nested in values:
                visit(nested)
            return
        if type(item) in {tuple, list}:
            seen.add(identity)
            for nested in cast(tuple[object, ...] | list[object], item):
                visit(nested)
        elif type(item) is dict:
            seen.add(identity)
            for key, nested in cast(dict[object, object], item).items():
                visit(key)
                visit(nested)

    visit(value)
    return tuple(captured)


def _restore_object_graph(
    captured: tuple[tuple[object, tuple[tuple[str, object], ...]], ...],
) -> None:
    for item, values in captured:
        field_names = {name for name, _ in values}
        try:
            extra_names = tuple(set(vars(item)) - field_names)
        except TypeError:
            extra_names = ()
        for name in extra_names:
            object.__delattr__(item, name)
        for name, value in values:
            object.__setattr__(item, name, value)


def _fingerprint_changed(value: object, expected: str) -> bool:
    try:
        return champion_fingerprint(value) != expected
    except Exception:
        return True


def _validated_parent(
    parent: object,
) -> tuple[ChampionRecipe, FittedChampionPolicy | None]:
    def validate_recipe(recipe: object) -> ChampionRecipe:
        if type(recipe) is not ChampionRecipe:
            _fail("active Parent contains an invalid recipe")
        canonical = cast(ChampionRecipe, recipe)
        for assumption in canonical.assumptions:
            if type(assumption) is not EvolutionAssumption:
                _fail("active Parent contains an invalid assumption")
            EvolutionAssumption.__post_init__(assumption)
        ChampionRecipe.__post_init__(canonical)
        return canonical

    try:
        if type(parent) is ChampionRelease:
            ChampionRelease.__post_init__(parent)
            FittedChampionPolicy.__post_init__(parent.policy)
            return validate_recipe(parent.policy.recipe), parent.policy
        if type(parent) is FittedChampionPolicy:
            FittedChampionPolicy.__post_init__(parent)
            return validate_recipe(parent.recipe), parent
        if type(parent) is ChampionRecipe:
            return validate_recipe(parent), None
    except ChampionControllerError:
        raise
    except Exception as error:
        raise ChampionControllerError("active Parent violates its frozen contract") from error
    _fail("active Parent must be an exact ChampionRelease, policy, or recipe")
    raise AssertionError("unreachable")


def _validated_rows(
    rows: object,
    *,
    build_size: int,
) -> tuple[tuple[ChampionTaskRow, ...], tuple[str, ...], tuple[str, ...]]:
    if type(rows) not in {tuple, list} or not rows:
        _fail("Build rows must be a nonempty exact tuple or list")
    snapshot = tuple(cast(tuple[object, ...] | list[object], rows))
    seen_keys: set[tuple[str, str]] = set()
    task_ids: list[str] = []
    identities: dict[str, tuple[object, object, int, str, object]] = {}
    candidate_tasks: dict[str, set[str]] = {}
    normalized_candidates: dict[str, str] = {}
    for raw_row in snapshot:
        if type(raw_row) is not ChampionTaskRow:
            _fail("Build rows must contain exact ChampionTaskRow values")
        row = cast(ChampionTaskRow, raw_row)
        try:
            ChampionTaskRow.__post_init__(row)
        except Exception as error:
            raise ChampionControllerError("Build row violates its trusted contract") from error
        if row.split != "build":
            _fail("Build evolution accepts Build rows only")
        normalized_name = _canonical_identity(row.candidate_name)
        prior_name = normalized_candidates.setdefault(
            normalized_name, row.candidate_name
        )
        if prior_name != row.candidate_name:
            _fail("normalized row inventory contains a candidate collision")
        key = (row.candidate_name, row.task_id)
        if key in seen_keys:
            _fail("Build rows contain a duplicate candidate/task key")
        seen_keys.add(key)
        if row.task_id not in identities:
            task_ids.append(row.task_id)
        identity = (row.profile, row.truth, row.fold, row.split, row.history)
        previous = identities.setdefault(row.task_id, identity)
        if identity != previous:
            _fail(
                "Build row universe drifted across profile, truth, fold, split, "
                "or history"
            )
        candidate_tasks.setdefault(row.candidate_name, set()).add(row.task_id)
    if len(task_ids) != build_size:
        _fail("Build rows do not contain the configured task universe")
    universe = set(task_ids)
    if any(task_set != universe for task_set in candidate_tasks.values()):
        _fail("every materialized candidate must cover the fixed Build universe")
    return (
        cast(tuple[ChampionTaskRow, ...], snapshot),
        tuple(task_ids),
        tuple(candidate_tasks),
    )


def _bound_config(
    config: ChampionEvolutionConfig,
    task_ids: tuple[str, ...],
) -> ChampionEvolutionConfig:
    if config.build_task_ids and config.build_task_ids != task_ids:
        _fail("Build task membership drifted from the pre-registered universe")
    screens = config.screen_task_ids or tuple(
        task_ids[:size] for size in config.screen_sizes
    )
    bound = _clone_config(
        config,
        build_task_ids=task_ids,
        screen_task_ids=screens,
    )
    if set(bound.screen_task_ids[-1]) != set(task_ids):
        _fail("the final screen must equal the complete fixed Build universe")
    if any(not set(stage).issubset(task_ids) for stage in bound.screen_task_ids):
        _fail("screen membership contains a task outside fixed Build")
    return bound


def _clone_gate(config: ChampionGateConfig) -> ChampionGateConfig:
    return ChampionGateConfig(
        **{entry.name: getattr(config, entry.name) for entry in fields(config)}
    )


def _clone_config(
    config: ChampionEvolutionConfig,
    *,
    build_task_ids: tuple[str, ...] | None = None,
    screen_task_ids: tuple[tuple[str, ...], ...] | None = None,
) -> ChampionEvolutionConfig:
    """Detach every nested registered value from caller-owned aliases."""
    return ChampionEvolutionConfig(
        build_size=config.build_size,
        calibration_size=config.calibration_size,
        screen_sizes=tuple(config.screen_sizes),
        generations=config.generations,
        minimum_recipes=config.minimum_recipes,
        maximum_recipes=config.maximum_recipes,
        maximum_full_build_children=config.maximum_full_build_children,
        maximum_finalists=config.maximum_finalists,
        candidate_minimum_gain=config.candidate_minimum_gain,
        research_target_gain=config.research_target_gain,
        gate_config=_clone_gate(config.gate_config),
        build_task_ids=(
            tuple(config.build_task_ids)
            if build_task_ids is None
            else tuple(build_task_ids)
        ),
        screen_task_ids=(
            tuple(tuple(stage) for stage in config.screen_task_ids)
            if screen_task_ids is None
            else tuple(tuple(stage) for stage in screen_task_ids)
        ),
    )


def _require_config_fingerprint(
    config: ChampionEvolutionConfig,
    expected: str,
) -> None:
    if _fingerprint_changed(config, expected):
        _fail("registered Champion evolution config mutated")


def _feature_value(policy: FittedChampionPolicy, task_row: ChampionTaskRow, index: int) -> float:
    assumption = policy.recipe.assumptions[index]
    feature = assumption.feature
    profile = task_row.profile
    try:
        if feature == "horizon_ratio":
            value: object = profile.horizon / profile.history_length
        elif feature in _PROFILE_FEATURES:
            value = getattr(profile, feature)
        else:
            _fail("fitted Child references an unsupported Build feature")
    except (ArithmeticError, AttributeError, TypeError, ValueError) as error:
        raise ChampionControllerError("fitted Child feature is invalid") from error
    if type(value) not in {int, float}:
        _fail("fitted Child feature must be exactly numeric")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        _fail("fitted Child feature must be finite")
    return number


def _policy_forecast(
    policy: FittedChampionPolicy,
    task_rows: dict[str, ChampionTaskRow],
) -> tuple[tuple[float, ...] | None, str | None]:
    fallback = task_rows.get(policy.recipe.fallback_parent)
    if fallback is None or fallback.forecast is None:
        return None, "fallback_unavailable"
    parent_rows = tuple(task_rows.get(name) for name in policy.recipe.parents)
    enriched = fallback.history is not None or fallback.diagnostic is not None
    if enriched or any(
        row is not None and (row.history is not None or row.diagnostic is not None)
        for row in parent_rows
    ):
        if fallback.history is None:
            return None, "history_unavailable"
        forecasts: dict[str, tuple[float, ...]] = {}
        diagnostics: dict[str, CandidateDiagnostics] = {}
        for name, row in zip(policy.recipe.parents, parent_rows, strict=True):
            if row is None or row.forecast is None:
                return None, "parent_forecast_unavailable"
            if row.history != fallback.history:
                return None, "history_mismatch"
            if row.diagnostic is None or not _valid_history_diagnostic(row.diagnostic):
                return None, "history_diagnostic_unavailable"
            forecasts[name] = row.forecast
            diagnostics[name] = _runtime_diagnostic(row.diagnostic)
        execution = execute_champion(
            policy,
            forecasts,
            diagnostics,
            fallback.profile,
            fallback.history,
            fallback.profile.horizon,
        )
        if execution.fallback_reason not in {None, "assumption_not_satisfied"}:
            return None, f"runtime_{execution.fallback_reason}"
        return execution.forecast, None

    parents: dict[str, tuple[float, ...]] = {}
    for name in policy.recipe.parents:
        row = task_rows.get(name)
        if row is None or row.forecast is None:
            return fallback.forecast, None
        parents[name] = row.forecast
    for index, (assumption, (_, threshold)) in enumerate(
        zip(policy.recipe.assumptions, policy.thresholds, strict=True)
    ):
        value = _feature_value(policy, fallback, index)
        matches = value >= threshold if assumption.direction == "above" else value < threshold
        if not matches:
            return fallback.forecast, None

    kind = policy.recipe.kind
    if kind == "select":
        return parents[policy.recipe.parents[0]], None
    if kind == "route":
        routed = policy.recipe.assumptions[0].candidate_name
        if any(item.candidate_name != routed for item in policy.recipe.assumptions):
            return None, "invalid_route"
        return parents[routed], None
    left_name, right_name = policy.recipe.parents
    left, right = parents[left_name], parents[right_name]
    if kind == "horizon_route":
        split = int(len(left) * policy.horizon_split)
        return left[:split] + right[split:], None
    if kind == "weighted":
        return tuple(
            left[index] * policy.weights[0] + right[index] * policy.weights[1]
            for index in range(len(left))
        ), None
    if kind == "median":
        return tuple(
            left[index] / 2.0 + right[index] / 2.0 for index in range(len(left))
        ), None
    if kind == "bounded_overlay":
        return None, "history_scale_unavailable"
    return None, "unsupported_operator"


def _valid_history_diagnostic(diagnostic: object) -> bool:
    if type(diagnostic) is not ChampionHistoryDiagnostic:
        return False
    try:
        ChampionHistoryDiagnostic.__post_init__(diagnostic)
    except (ChampionEvidenceError, TypeError, ValueError):
        return False
    return True


def _runtime_diagnostic(
    diagnostic: ChampionHistoryDiagnostic,
) -> CandidateDiagnostics:
    """Construct Task 3 input with no fold objects, forecasts, truths, or cache."""
    return CandidateDiagnostics(
        name=diagnostic.name,
        family=diagnostic.family,
        folds=(),
        successful_folds=diagnostic.successful_folds,
        eligible=diagnostic.eligible,
        reason_code=diagnostic.reason_code,
        median_mase=diagnostic.median_mase,
        recent_mase=diagnostic.recent_mase,
        worst_mase=diagnostic.worst_mase,
        mase_mad=diagnostic.mase_mad,
        median_mae=diagnostic.median_mae,
        median_smape=diagnostic.median_smape,
        median_rmsse=diagnostic.median_rmsse,
        normalized_bias=diagnostic.normalized_bias,
        slope_error=diagnostic.slope_error,
        phase_error=diagnostic.phase_error,
        amplitude_ratio=diagnostic.amplitude_ratio,
        explosion=diagnostic.explosion,
        fold_forecasts=(),
        fold_truths=(),
        cache_key="",
        long_horizon_fold=None,
        long_horizon_coverage=diagnostic.long_horizon_coverage,
        median_joint_scaled_error=diagnostic.median_joint_scaled_error,
        recent_joint_scaled_error=diagnostic.recent_joint_scaled_error,
        worst_joint_scaled_error=diagnostic.worst_joint_scaled_error,
        median_smae=diagnostic.median_smae,
        recent_smae=diagnostic.recent_smae,
        worst_smae=diagnostic.worst_smae,
        smae_mad=diagnostic.smae_mad,
        median_srmse=diagnostic.median_srmse,
        recent_srmse=diagnostic.recent_srmse,
        worst_srmse=diagnostic.worst_srmse,
        srmse_mad=diagnostic.srmse_mad,
        worst_smae_raw=diagnostic.worst_smae_raw,
        worst_srmse_raw=diagnostic.worst_srmse_raw,
    )


def _materialize_policy(
    policy: FittedChampionPolicy,
    score_name: str,
    rows: tuple[ChampionTaskRow, ...],
    task_ids: tuple[str, ...],
) -> tuple[ChampionTaskRow, ...]:
    by_task: dict[str, dict[str, ChampionTaskRow]] = {task_id: {} for task_id in task_ids}
    identity: dict[str, ChampionTaskRow] = {}
    selected = set(task_ids)
    for row in rows:
        if row.task_id not in selected:
            continue
        by_task[row.task_id][row.candidate_name] = row
        identity.setdefault(row.task_id, row)
    materialized: list[ChampionTaskRow] = []
    for task_id in task_ids:
        base = identity[task_id]
        try:
            forecast, failure = _policy_forecast(policy, by_task[task_id])
        except (ArithmeticError, ChampionControllerError, OverflowError, ValueError):
            forecast, failure = None, "invalid_policy_materialization"
        if forecast is not None and any(not math.isfinite(value) for value in forecast):
            forecast, failure = None, "invalid_policy_materialization"
        materialized.append(
            ChampionTaskRow(
                task_id=task_id,
                candidate_name=score_name,
                profile=base.profile,
                truth=base.truth,
                forecast=forecast,
                failure_reason=failure if forecast is None else None,
                fold=base.fold,
                split="build",
            )
        )
    return tuple(materialized)


def _parent_rows(
    parent_recipe: ChampionRecipe,
    parent_policy: FittedChampionPolicy | None,
    rows: tuple[ChampionTaskRow, ...],
    task_ids: tuple[str, ...],
) -> tuple[str, tuple[ChampionTaskRow, ...]]:
    supplied = tuple(
        row
        for row in rows
        if row.task_id in set(task_ids) and row.candidate_name == parent_recipe.name
    )
    if supplied:
        if len(supplied) != len(task_ids):
            _fail("supplied Parent rows do not cover fixed screen membership")
        return parent_recipe.name, supplied
    if parent_policy is None:
        _fail("an unfitted Parent recipe requires trusted materialized Parent rows")
    return parent_recipe.name, _materialize_policy(
        parent_policy, parent_recipe.name, rows, task_ids
    )


def _stage_gate(config: ChampionEvolutionConfig, *, final: bool) -> ChampionGateConfig:
    del final
    return config.gate_config


def _evaluate_stage(
    state: _AttemptState,
    *,
    parent_recipe: ChampionRecipe,
    parent_policy: FittedChampionPolicy | None,
    rows: tuple[ChampionTaskRow, ...],
    task_ids: tuple[str, ...],
    gate: ChampionGateConfig,
) -> _AttemptState:
    parent_name, parent_rows = _parent_rows(
        parent_recipe, parent_policy, rows, task_ids
    )
    child_rows = _materialize_policy(state.policy, state.score_name, rows, task_ids)
    evidence_rows = parent_rows + child_rows
    try:
        parent_score = score_policy(evidence_rows, parent_name)
    except (ChampionEvidenceError, TypeError, ValueError) as error:
        raise ChampionControllerError("trusted Build scoring rejected the Parent") from error
    try:
        child_score = score_policy(evidence_rows, state.score_name)
        comparison = compare_champion(parent_score, child_score, gate)
    except (ChampionEvidenceError, TypeError, ValueError):
        return replace(
            state,
            stage_task_counts=state.stage_task_counts + (len(task_ids),),
            comparison=None,
            evidence_rows=evidence_rows,
            invalid_reason="unscorable_child",
        )
    return replace(
        state,
        stage_task_counts=state.stage_task_counts + (len(task_ids),),
        comparison=comparison,
        evidence_rows=evidence_rows,
        invalid_reason=None,
    )


def _rank(state: _AttemptState) -> tuple[float, float, float, str]:
    if state.comparison is None:
        _fail("attempt has no trusted comparison")
    comparison = cast(ChampionComparison, state.comparison)
    return (
        -comparison.joint_improvement,
        comparison.mean_delta_smae,
        comparison.mean_delta_srmse,
        state.fitted_id,
    )


def _diverse(
    states: tuple[_AttemptState, ...],
    *,
    limit: int,
) -> tuple[_AttemptState, ...]:
    ranked = tuple(sorted(states, key=_rank))
    selected: list[_AttemptState] = []
    seen_kinds: set[str] = set()
    seen_structures: set[tuple[str, tuple[str, ...]]] = set()
    for state in ranked:
        kind = state.policy.recipe.kind
        if kind in seen_kinds:
            continue
        selected.append(state)
        seen_kinds.add(kind)
        seen_structures.add((kind, state.policy.recipe.parents))
        if len(selected) == limit:
            return tuple(selected)
    for state in ranked:
        structure = (state.policy.recipe.kind, state.policy.recipe.parents)
        if state in selected or structure in seen_structures:
            continue
        selected.append(state)
        seen_structures.add(structure)
        if len(selected) == limit:
            break
    return tuple(selected)


def _feedback(states: tuple[_AttemptState, ...]) -> ProposerEvidence:
    scored_states = tuple(
        state
        for state in states
        if state.comparison is not None and state.evidence_rows
    )
    if not scored_states:
        return ProposerEvidence(
            label="adaptive_train_build_diagnostic",
            independent_generalization_claim=False,
            morphology=(),
            comparisons=(),
        )
    grouped: dict[tuple[str, ...], list[_AttemptState]] = {}
    for state in scored_states:
        comparison = cast(ChampionComparison, state.comparison)
        task_ids = tuple(
            row.task_id
            for row in state.evidence_rows
            if row.candidate_name == comparison.parent_name
        )
        grouped.setdefault(task_ids, []).append(state)
    evidence_parts: list[ProposerEvidence] = []
    for group in grouped.values():
        parent_name = cast(ChampionComparison, group[0].comparison).parent_name
        parent_rows = tuple(
            row for row in group[0].evidence_rows if row.candidate_name == parent_name
        )
        child_rows = tuple(
            row
            for state in group
            for row in state.evidence_rows
            if row.candidate_name == state.score_name
        )
        comparisons = tuple(cast(ChampionComparison, state.comparison) for state in group)
        try:
            evidence_parts.append(
                sanitize_build_evidence(parent_rows + child_rows, comparisons)
            )
        except (ChampionEvidenceError, TypeError, ValueError) as error:
            raise ChampionControllerError("Build feedback could not be sanitized") from error
    return ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=tuple(
            sorted(
                (item for part in evidence_parts for item in part.morphology),
                key=lambda item: (item.group_id, item.candidate_name),
            )
        ),
        comparisons=tuple(
            sorted(
                (item for part in evidence_parts for item in part.comparisons),
                key=lambda item: item.candidate_name,
            )
        ),
    )


def _call_proposer(
    proposer: Callable[[object, ProposerEvidence], object],
    parent: object,
    feedback: ProposerEvidence,
    *,
    minimum: int,
    maximum: int,
) -> tuple[ChampionRecipe, ...]:
    if not callable(proposer):
        _fail("proposer must be callable")
    try:
        proposed = proposer(parent, feedback)
    except Exception as error:
        raise ChampionControllerError("structural proposer callback failed") from error
    if type(proposed) not in {tuple, list}:
        _fail("proposer must return an exact tuple or list")
    recipes = tuple(cast(tuple[object, ...] | list[object], proposed))
    if not minimum <= len(recipes) <= maximum:
        _fail("proposer must return the configured five through ten recipes")
    validated: list[ChampionRecipe] = []
    for raw_recipe in recipes:
        if type(raw_recipe) is not ChampionRecipe:
            _fail("proposer output must contain exact ChampionRecipe values")
        recipe = cast(ChampionRecipe, raw_recipe)
        try:
            ChampionRecipe.__post_init__(recipe)
            detached = parse_champion_recipe(recipe.to_payload())
        except Exception as error:
            raise ChampionControllerError(
                "proposer returned an invalid Champion recipe"
            ) from error
        validated.append(detached)
    recipe_ids = tuple(champion_fingerprint(recipe) for recipe in validated)
    recipe_names = tuple(recipe.name for recipe in validated)
    canonical_names = tuple(_canonical_identity(name) for name in recipe_names)
    if len(recipe_ids) != len(set(recipe_ids)) or len(canonical_names) != len(
        set(canonical_names)
    ):
        _fail("proposer returned duplicate policy identities")
    for recipe in validated:
        canonical_parents = tuple(
            _canonical_identity(name) for name in recipe.parents
        )
        canonical_assumptions = tuple(
            _canonical_identity(item.assumption_id) for item in recipe.assumptions
        )
        if len(canonical_parents) != len(set(canonical_parents)) or len(
            canonical_assumptions
        ) != len(set(canonical_assumptions)):
            _fail("proposer returned duplicate policy identities")
    return tuple(validated)


def _stored_policies(
    generations: list[BuildGeneration],
    archive: list[_AttemptState],
) -> tuple[FittedChampionPolicy, ...]:
    values = [
        attempt.policy
        for generation in generations
        for attempt in generation.attempts
    ]
    values.extend(state.policy for state in archive)
    unique: list[FittedChampionPolicy] = []
    seen: set[int] = set()
    for policy in values:
        if id(policy) in seen:
            continue
        seen.add(id(policy))
        unique.append(policy)
    return tuple(unique)


def _stored_policy_fingerprints(
    generations: list[BuildGeneration],
    archive: list[_AttemptState],
) -> tuple[tuple[FittedChampionPolicy, str], ...]:
    return tuple(
        (policy, champion_fingerprint(policy))
        for policy in _stored_policies(generations, archive)
    )


def _stored_policy_changed(
    fingerprints: tuple[tuple[FittedChampionPolicy, str], ...],
) -> bool:
    return any(
        _fingerprint_changed(policy, expected)
        for policy, expected in fingerprints
    )


def _validate_stored_policy_ids(generations: list[BuildGeneration]) -> None:
    for generation in generations:
        for attempt in generation.attempts:
            if champion_fingerprint(attempt.policy) != attempt.fitted_id:
                _fail("stored fitted policy fingerprint drifted")


def run_build_evolution(
    parent: object,
    rows: tuple[ChampionTaskRow, ...] | list[ChampionTaskRow],
    proposer: Callable[[object, ProposerEvidence], object],
    config: ChampionEvolutionConfig,
) -> BuildEvolutionResult:
    """Run Build-only structural evolution without replacing ``parent``."""
    if type(config) is not ChampionEvolutionConfig:
        _fail("config must be an exact ChampionEvolutionConfig")
    ChampionEvolutionConfig.__post_init__(config)
    input_config_sha256 = config.fingerprint
    parent_recipe, parent_policy = _validated_parent(parent)
    snapshot, task_ids, row_candidate_names = _validated_rows(
        rows, build_size=config.build_size
    )
    bound_config = _bound_config(config, task_ids)
    bound_config_sha256 = bound_config.fingerprint
    parent_sha256 = champion_fingerprint(parent)
    rows_sha256 = champion_fingerprint(snapshot)
    feedback = ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=(),
        comparisons=(),
    )
    seen_recipe_ids: set[str] = {champion_fingerprint(parent_recipe)}
    seen_policy_names: set[str] = {_canonical_identity(parent_recipe.name)}
    reserved_row_names = {
        _canonical_identity(candidate_name)
        for candidate_name in row_candidate_names
    }
    seen_fitted_ids: set[str] = set()
    generations: list[BuildGeneration] = []
    archive: list[_AttemptState] = []

    for generation_number in range(1, bound_config.generations + 1):
        _require_config_fingerprint(config, input_config_sha256)
        _require_config_fingerprint(bound_config, bound_config_sha256)
        _validate_stored_policy_ids(generations)
        parent_state = _capture_object_graph(parent)
        row_state = _capture_object_graph(snapshot)
        feedback_state = _capture_object_graph(feedback)
        config_state = _capture_object_graph(config)
        bound_config_state = _capture_object_graph(bound_config)
        stored_fingerprints = _stored_policy_fingerprints(generations, archive)
        stored_state = _capture_object_graph(
            tuple(policy for policy, _ in stored_fingerprints)
        )
        feedback_sha256 = champion_fingerprint(feedback)
        parent_changed = row_changed = feedback_changed = False
        config_changed = bound_config_changed = stored_changed = False
        callback_error: Exception | None = None
        recipes: tuple[ChampionRecipe, ...] | None = None
        try:
            try:
                recipes = _call_proposer(
                    proposer,
                    parent,
                    feedback,
                    minimum=bound_config.minimum_recipes,
                    maximum=bound_config.maximum_recipes,
                )
            except Exception as error:
                callback_error = error
            finally:
                parent_changed = _fingerprint_changed(parent, parent_sha256)
                row_changed = _fingerprint_changed(snapshot, rows_sha256)
                feedback_changed = _fingerprint_changed(feedback, feedback_sha256)
                config_changed = _fingerprint_changed(config, input_config_sha256)
                bound_config_changed = _fingerprint_changed(
                    bound_config, bound_config_sha256
                )
                stored_changed = _stored_policy_changed(stored_fingerprints)
        finally:
            _restore_object_graph(parent_state)
            _restore_object_graph(row_state)
            _restore_object_graph(feedback_state)
            _restore_object_graph(config_state)
            _restore_object_graph(bound_config_state)
            _restore_object_graph(stored_state)
        if parent_changed:
            _fail("proposer attempted to mutate the active Parent")
        if row_changed:
            _fail("proposer attempted to mutate the fixed Build rows")
        if feedback_changed:
            _fail("proposer attempted to mutate sanitized Build feedback")
        if config_changed or bound_config_changed:
            _fail("proposer attempted to mutate the registered config")
        if stored_changed:
            _fail("proposer attempted to mutate an archived fitted policy")
        _require_config_fingerprint(config, input_config_sha256)
        _require_config_fingerprint(bound_config, bound_config_sha256)
        _validate_stored_policy_ids(generations)
        if callback_error is not None:
            raise callback_error
        if recipes is None:
            _fail("structural proposer returned no recipes")
        validated_recipes = cast(tuple[ChampionRecipe, ...], recipes)

        states: list[_AttemptState] = []
        for recipe in validated_recipes:
            recipe_id = champion_fingerprint(recipe)
            policy_name = _canonical_identity(recipe.name)
            if policy_name in reserved_row_names:
                _fail("proposal policy ID collides with materialized row inventory")
            if recipe_id in seen_recipe_ids or policy_name in seen_policy_names:
                _fail("duplicate policy ID across Build generations")
            seen_recipe_ids.add(recipe_id)
            seen_policy_names.add(policy_name)
            try:
                policies = expand_recipe(recipe, snapshot)
            except (ChampionProposalError, TypeError, ValueError) as error:
                raise ChampionControllerError(
                    "host numeric expansion rejected a recipe"
                ) from error
            for policy in policies:
                fitted_id = champion_fingerprint(policy)
                if fitted_id in seen_fitted_ids:
                    _fail("duplicate fitted policy ID across Build generations")
                seen_fitted_ids.add(fitted_id)
                states.append(
                    _AttemptState(
                        fitted_id=fitted_id,
                        score_name=f"build_child_{fitted_id}",
                        policy=policy,
                    )
                )
        if not states:
            _fail("host expansion produced an empty Build stage")

        active = tuple(states)
        for stage_index, stage_task_ids in enumerate(bound_config.screen_task_ids):
            if not active:
                break
            _require_config_fingerprint(config, input_config_sha256)
            _require_config_fingerprint(bound_config, bound_config_sha256)
            _validate_stored_policy_ids(generations)
            evaluated = tuple(
                _evaluate_stage(
                    state,
                    parent_recipe=parent_recipe,
                    parent_policy=parent_policy,
                    rows=snapshot,
                    task_ids=stage_task_ids,
                    gate=_stage_gate(
                        bound_config,
                        final=stage_index == len(bound_config.screen_task_ids) - 1,
                    ),
                )
                for state in active
            )
            _require_config_fingerprint(config, input_config_sha256)
            _require_config_fingerprint(bound_config, bound_config_sha256)
            updates = {state.fitted_id: state for state in evaluated}
            states = [updates.get(state.fitted_id, state) for state in states]
            safe = tuple(
                state
                for state in evaluated
                if state.comparison is not None and state.comparison.accepted
            )
            if stage_index == len(bound_config.screen_task_ids) - 1:
                active = safe
            elif stage_index == len(bound_config.screen_task_ids) - 2:
                active = _diverse(
                    safe, limit=bound_config.maximum_full_build_children
                )
            else:
                survivor_count = max(
                    bound_config.maximum_full_build_children,
                    (len(safe) + 1) // 2,
                )
                active = _diverse(safe, limit=survivor_count)

        full_build_children = sum(
            bool(state.stage_task_counts)
            and state.stage_task_counts[-1] == bound_config.build_size
            for state in states
        )
        eligible = tuple(
            state
            for state in active
            if state.comparison is not None
            and state.comparison.joint_improvement
            >= bound_config.candidate_minimum_gain
        )
        finalists = _diverse(eligible, limit=bound_config.maximum_finalists)
        finalist_ids = {state.fitted_id for state in finalists}
        attempts = tuple(
            BuildAttempt(
                fitted_id=state.fitted_id,
                policy=state.policy,
                stage_task_counts=state.stage_task_counts,
                comparison=state.comparison,
                status=(
                    "invalid"
                    if state.invalid_reason is not None
                    else "finalist"
                    if state.fitted_id in finalist_ids
                    else "below_candidate_threshold"
                    if state in active
                    else "pruned"
                ),
                invalid_reason=state.invalid_reason,
            )
            for state in states
        )
        generation_feedback = _feedback(tuple(states))
        feedback = ProposerEvidence(
            label="adaptive_train_build_diagnostic",
            independent_generalization_claim=False,
            morphology=feedback.morphology + generation_feedback.morphology,
            comparisons=feedback.comparisons + generation_feedback.comparisons,
        )
        generations.append(
            BuildGeneration(
                number=generation_number,
                mutation_parent_sha256=parent_sha256,
                config_fingerprint=bound_config.fingerprint,
                stage_counts=bound_config.screen_sizes,
                full_build_children=full_build_children,
                attempts=attempts,
                finalists=tuple(state.policy for state in finalists),
                feedback=generation_feedback,
            )
        )
        archive.extend(finalists)

    shortlist = _diverse(tuple(archive), limit=bound_config.maximum_finalists)
    _require_config_fingerprint(config, input_config_sha256)
    _require_config_fingerprint(bound_config, bound_config_sha256)
    _validate_stored_policy_ids(generations)
    if champion_fingerprint(parent) != parent_sha256:
        _fail("Build evolution attempted to mutate the active Parent")
    if champion_fingerprint(snapshot) != rows_sha256:
        _fail("Build evolution attempted to mutate the fixed Build rows")
    return BuildEvolutionResult(
        active_parent=parent,
        generations=tuple(generations),
        shortlist=tuple(state.policy for state in shortlist),
        config=bound_config,
        config_fingerprint=bound_config.fingerprint,
    )


__all__ = [
    "BuildAttempt",
    "BuildEvolutionResult",
    "BuildGeneration",
    "ChampionControllerError",
    "ChampionEvolutionConfig",
    "run_build_evolution",
]
