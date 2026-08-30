"""Immutable canonical artifacts for the morphology-guided Numerical loop."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from .morphology import AssumptionGrounding, MorphologyCard
from .numerical_selector import (
    CandidateDiagnostics,
    HindcastFold,
    SelectionArithmetic,
    SelectionDecision,
    replay_selection_forecast,
)
from .screening import TaskProfile


_RETRIEVAL_FIELDS = frozenset(
    {"assumption_id", "kind", "claim", "failure_condition"}
)
@dataclass(frozen=True)
class RankedNumericalForecast:
    """One materialized candidate retained in deterministic diagnostic order."""

    rank: int
    name: str
    family: str
    forecast: tuple[float, ...]
    diagnostics: CandidateDiagnostics

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("forecast rank must be positive")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ranked forecast requires a candidate name")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("ranked forecast requires a candidate family")
        if not isinstance(self.diagnostics, CandidateDiagnostics):
            raise ValueError("ranked forecast requires CandidateDiagnostics")
        if self.diagnostics.name != self.name or self.diagnostics.family != self.family:
            raise ValueError("ranked forecast identity must match its diagnostics")
        object.__setattr__(self, "forecast", forecast_tuple(self.forecast))


@dataclass(frozen=True)
class NumericalForecastPackage:
    """Immutable Numerical result and its sanitized Morphology/Retrieval audit trail."""

    task_profile: TaskProfile
    active_candidate_names: tuple[str, ...]
    candidate_diagnostics: Mapping[str, CandidateDiagnostics]
    morphology_card: MorphologyCard | None
    accepted_assumptions: tuple[AssumptionGrounding, ...]
    rejected_assumptions: Mapping[str, str]
    selection_decision: SelectionDecision
    final_forecast: tuple[float, ...]
    protected_baseline: RankedNumericalForecast
    ranked_alternatives: tuple[RankedNumericalForecast, ...]
    retrieval_handoff: tuple[Mapping[str, str], ...]
    component_fingerprints: Mapping[str, str]
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_profile, TaskProfile):
            raise ValueError("Numerical package requires a TaskProfile")
        active = _string_tuple(self.active_candidate_names, "active candidate names")
        if len(active) != len(set(active)):
            raise ValueError("active candidate names must be unique")

        diagnostics = dict(self.candidate_diagnostics)
        if any(
            not isinstance(name, str)
            or not isinstance(value, CandidateDiagnostics)
            or value.name != name
            for name, value in diagnostics.items()
        ):
            raise ValueError("candidate diagnostics contain an invalid identity")
        if not set(diagnostics) <= set(active):
            raise ValueError("candidate diagnostics must belong to the active dictionary")
        diagnostics = {
            name: _freeze_diagnostic(value) for name, value in diagnostics.items()
        }

        if self.morphology_card is not None and not isinstance(
            self.morphology_card, MorphologyCard
        ):
            raise ValueError("morphology_card must be a MorphologyCard or None")
        accepted = tuple(self.accepted_assumptions)
        if any(not isinstance(item, AssumptionGrounding) for item in accepted):
            raise ValueError("accepted assumptions must be grounded artifacts")
        if len({item.assumption_id for item in accepted}) != len(accepted):
            raise ValueError("accepted assumptions must have unique ids")
        rejected = dict(self.rejected_assumptions)
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in rejected.items()
        ):
            raise ValueError("rejected assumptions require string ids and reasons")
        if {item.assumption_id for item in accepted} & set(rejected):
            raise ValueError("an assumption cannot be accepted and rejected")

        selection = _freeze_selection(
            self.selection_decision, horizon=self.task_profile.horizon
        )
        final = forecast_tuple(self.final_forecast, horizon=self.task_profile.horizon)
        if final != selection.forecast:
            raise ValueError("final forecast must equal the protected selector output")
        if not set(selection.selected) <= set(active):
            raise ValueError("selected candidates must belong to the active dictionary")

        supplied_alternatives = tuple(self.ranked_alternatives)
        if any(
            not isinstance(item, RankedNumericalForecast)
            for item in supplied_alternatives
        ):
            raise ValueError("ranked alternatives must be numerical forecast artifacts")
        if any(item.name not in diagnostics for item in supplied_alternatives):
            raise ValueError("ranked alternatives require canonical diagnostics")
        alternatives = tuple(
            RankedNumericalForecast(
                rank=item.rank,
                name=item.name,
                family=item.family,
                forecast=forecast_tuple(
                    item.forecast, horizon=self.task_profile.horizon
                ),
                diagnostics=diagnostics[item.name],
            )
            for item in supplied_alternatives
        )
        if tuple(item.rank for item in alternatives) != tuple(
            range(1, len(alternatives) + 1)
        ):
            raise ValueError("ranked alternative ranks must be contiguous")
        if len({item.name for item in alternatives}) != len(alternatives):
            raise ValueError("ranked alternatives must have unique names")
        alternative_names = {item.name for item in alternatives}
        if not alternative_names <= set(active):
            raise ValueError("ranked alternatives must belong to the active dictionary")
        if not set(selection.selected) <= alternative_names:
            raise ValueError("selected names must have materialized ranked alternatives")
        if not set(selection.considered_candidates) <= set(active):
            raise ValueError("considered candidates must belong to the active dictionary")
        if (
            selection.baseline_name is not None
            and selection.baseline_name not in alternative_names
        ):
            raise ValueError("selection baseline must be a materialized ranked forecast")
        selected_artifacts = tuple(
            next(item for item in alternatives if item.name == name)
            for name in selection.selected
        )
        if self.morphology_card is not None and len(selected_artifacts) != 1:
            raise ValueError("Morphology-guided selection must be single")
        replayed = replay_selection_forecast(
            selection, {item.name: item.forecast for item in alternatives}
        )
        if selection.forecast != replayed:
            if selection.mode == "single":
                raise ValueError(
                    "single selection arithmetic replay must equal its materialized forecast"
                )
            raise ValueError(
                "selection arithmetic replay does not match the weighted combination "
                "or supported selector output"
            )
        if not isinstance(self.protected_baseline, RankedNumericalForecast):
            raise ValueError("Numerical package requires a protected baseline")
        if self.protected_baseline.name not in alternative_names:
            raise ValueError("protected baseline must be a materialized ranked forecast")
        protected = next(
            item for item in alternatives if item.name == self.protected_baseline.name
        )

        handoff = tuple(_freeze_handoff(item) for item in self.retrieval_handoff)
        if tuple(item["assumption_id"] for item in handoff) != host_assumption_ids(
            len(accepted)
        ):
            raise ValueError("Retrieval handoff must correspond exactly to accepted assumptions")
        fingerprints = freeze_string_mapping(
            self.component_fingerprints, "component fingerprints"
        )
        fallback = self.fallback_reason
        if fallback is not None and (not isinstance(fallback, str) or not fallback):
            raise ValueError("fallback reason must be a nonempty string")

        object.__setattr__(self, "active_candidate_names", active)
        object.__setattr__(
            self, "candidate_diagnostics", MappingProxyType(dict(sorted(diagnostics.items())))
        )
        object.__setattr__(self, "accepted_assumptions", accepted)
        object.__setattr__(
            self, "rejected_assumptions", MappingProxyType(dict(sorted(rejected.items())))
        )
        object.__setattr__(self, "selection_decision", selection)
        object.__setattr__(self, "final_forecast", final)
        object.__setattr__(self, "protected_baseline", protected)
        object.__setattr__(self, "ranked_alternatives", alternatives)
        object.__setattr__(self, "retrieval_handoff", handoff)
        object.__setattr__(self, "component_fingerprints", fingerprints)


def snapshot_diagnostics(
    diagnostics: Mapping[str, CandidateDiagnostics],
    active: Sequence[tuple[str, str]],
) -> dict[str, CandidateDiagnostics]:
    """Read one stable snapshot and detach mutable diagnostic containers."""
    if not isinstance(diagnostics, Mapping):
        raise TypeError("diagnostics must be a mapping")
    snapshot: dict[str, CandidateDiagnostics] = {}
    try:
        for name, family in active:
            value = diagnostics.get(name)
            if diagnostics.get(name) != value:
                raise ValueError("diagnostics changed while being read")
            if (
                not isinstance(value, CandidateDiagnostics)
                or value.name != name
                or value.family != family
            ):
                raise ValueError(f"invalid diagnostics for active candidate {name!r}")
            snapshot[name] = _freeze_diagnostic(value)
    except (AttributeError, TypeError, RuntimeError) as error:
        raise ValueError("diagnostics could not be read safely") from error
    return snapshot


def ranked_forecasts(
    active_names: Sequence[str],
    families: Mapping[str, str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
) -> tuple[RankedNumericalForecast, ...]:
    """Return every materialized forecast in deterministic diagnostic order."""
    available = [
        diagnostics[name]
        for name in active_names
        if name in diagnostics and name in forecasts
    ]
    available.sort(
        key=lambda item: (
            not item.eligible,
            -item.successful_folds,
            _rank_number(item.median_joint_scaled_error),
            _rank_number(item.recent_joint_scaled_error),
            _rank_number(item.worst_joint_scaled_error),
            _rank_number(item.median_smae),
            _rank_number(item.median_srmse),
            item.name,
        )
    )
    return tuple(
        RankedNumericalForecast(
            rank=index,
            name=item.name,
            family=families[item.name],
            forecast=tuple(float(value) for value in forecasts[item.name]),
            diagnostics=item,
        )
        for index, item in enumerate(available, start=1)
    )


def freeze_string_mapping(
    value: Mapping[str, str], context: str
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    result = dict(value)
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, str)
        or not item
        for key, item in result.items()
    ):
        raise ValueError(f"{context} must contain nonempty strings")
    return MappingProxyType(dict(sorted(result.items())))


def host_assumption_ids(count: int) -> tuple[str, ...]:
    """Mint opaque deterministic IDs from host-owned accepted-assumption order."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("accepted assumption count must be a nonnegative integer")
    return tuple(f"assumption_{index:03d}" for index in range(1, count + 1))


def forecast_tuple(
    values: Sequence[float], *, horizon: int | None = None
) -> tuple[float, ...]:
    result = _finite_tuple(values, allow_empty=False)
    if horizon is not None and len(result) != horizon:
        raise ValueError("forecast has the wrong horizon")
    return result


def valid_forecast(values: Sequence[float], horizon: int) -> bool:
    try:
        forecast_tuple(values, horizon=horizon)
    except (TypeError, ValueError):
        return False
    return True


def _freeze_selection(
    decision: SelectionDecision, *, horizon: int
) -> SelectionDecision:
    if not isinstance(decision, SelectionDecision):
        raise ValueError("Numerical package requires a SelectionDecision")
    if decision.mode not in {"single", "ensemble", "combined"}:
        raise ValueError("selection mode is unsupported")
    selected = _string_tuple(decision.selected, "selected candidate names")
    if len(selected) != len(set(selected)):
        raise ValueError("selected candidate names must be unique")
    if (decision.mode == "single") != (len(selected) == 1):
        raise ValueError("selection mode must align with selected candidate count")
    try:
        raw_weights = tuple(decision.weights)
    except (TypeError, ValueError) as error:
        raise ValueError("selection weights must be numerical") from error
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw_weights
    ):
        raise ValueError("selection weights must be numerical")
    weights = tuple(float(value) for value in raw_weights)
    if len(weights) != len(selected) or any(not math.isfinite(value) for value in weights):
        raise ValueError("selection weights must be finite and align with selected names")
    if any(value < 0.0 for value in weights) or math.fsum(weights) != 1.0:
        raise ValueError("selection weights must be nonnegative and normalized")
    confidence = decision.confidence
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("selection confidence must be finite and within [0, 1]")
    if decision.combination_type is not None and (
        not isinstance(decision.combination_type, str)
        or not decision.combination_type
    ):
        raise ValueError("selection combination type must be a nonempty string or None")
    if decision.baseline_name is not None and (
        not isinstance(decision.baseline_name, str) or not decision.baseline_name
    ):
        raise ValueError("selection baseline name must be a nonempty string or None")
    if decision.arithmetic is not None and not isinstance(
        decision.arithmetic, SelectionArithmetic
    ):
        raise ValueError("selection arithmetic must be immutable selector provenance")
    assumption_ids = _string_tuple(
        decision.assumption_ids, "selection assumption ids", allow_empty=True
    )
    assumption_kinds = _string_tuple(
        decision.assumption_kinds, "selection assumption kinds", allow_empty=True
    )
    if len(assumption_ids) != len(assumption_kinds):
        raise ValueError("selection assumption ids and kinds must align")
    return replace(
        decision,
        selected=selected,
        weights=weights,
        forecast=forecast_tuple(decision.forecast, horizon=horizon),
        confidence=float(confidence),
        reason_codes=_string_tuple(decision.reason_codes, "selection reason codes"),
        rejected=freeze_string_mapping(
            decision.rejected, "selection rejection trace"
        ),
        assumption_ids=assumption_ids,
        assumption_kinds=assumption_kinds,
        considered_candidates=_string_tuple(
            decision.considered_candidates,
            "considered candidate names",
            allow_empty=True,
        ),
    )


def _freeze_diagnostic(value: CandidateDiagnostics) -> CandidateDiagnostics:
    folds = tuple(
        _freeze_fold(item) if isinstance(item, HindcastFold) else item
        for item in value.folds
    )
    long_horizon_fold = value.long_horizon_fold
    if isinstance(long_horizon_fold, HindcastFold):
        long_horizon_fold = _freeze_fold(long_horizon_fold)
    return replace(
        value,
        folds=folds,
        fold_forecasts=tuple(
            _finite_tuple(fold, allow_empty=False) for fold in value.fold_forecasts
        ),
        fold_truths=tuple(
            _finite_tuple(fold, allow_empty=False) for fold in value.fold_truths
        ),
        long_horizon_fold=long_horizon_fold,
    )


def _freeze_fold(value: HindcastFold) -> HindcastFold:
    return replace(
        value,
        forecast=_finite_tuple(value.forecast, allow_empty=True),
        truth=_finite_tuple(value.truth, allow_empty=True),
    )


def _freeze_handoff(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != _RETRIEVAL_FIELDS:
        raise ValueError("Retrieval handoff must contain exactly four safe fields")
    result = dict(value)
    if any(
        not isinstance(key, str) or not isinstance(item, str) or not item
        for key, item in result.items()
    ):
        raise ValueError("Retrieval handoff fields must be nonempty strings")
    return MappingProxyType(result)


def _string_tuple(
    values: Sequence[str], context: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{context} must be a sequence of strings")
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{context} must be a sequence of strings") from error
    if (not result and not allow_empty) or any(
        not isinstance(value, str) or not value for value in result
    ):
        raise ValueError(f"{context} must contain nonempty strings")
    return result


def _finite_tuple(
    values: Sequence[float], *, allow_empty: bool
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("forecast must be a numerical sequence")
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ValueError("forecast must be a numerical sequence") from error
    if not raw and not allow_empty:
        raise ValueError("forecast must not be empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in raw
    ):
        raise ValueError("forecast contains a non-finite or non-numeric value")
    return tuple(float(value) for value in raw)


def _rank_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.inf
    number = float(value)
    return number if math.isfinite(number) else math.inf
