"""History-only orchestration for the morphology-guided Numerical loop."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

from .execution import CRASHED, INVALID, SUCCESS, Outcome, Task
from .morphology import AssumptionGrounding, MorphologyCard
from .morphology_consistency import check_morphology_assumptions
from .numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    HindcastConfig,
    HindcastFold,
    SelectionDecision,
    diagnose_active_candidates,
    select_grounded_morphology_forecast,
    select_numerical_forecast,
    select_protected_safe_anchor,
)
from .portfolio import CombinedPolicy, combine_materialized_outcome
from .screening import (
    ActiveDictionary,
    ScreeningPolicy,
    TaskProfile,
    materialize_active_dictionary,
    profile_task,
)


CandidateRunner = Callable[[str, tuple[float, ...], int, str], Sequence[float]]
_RETRIEVAL_FIELDS = frozenset(
    {"assumption_id", "kind", "claim", "failure_condition"}
)
_RETRIEVAL_KIND = {
    "seasonality": "seasonality",
    "trend": "trend_persistence",
    "intermittency": "other",
    "regime": "regime_persistence",
    "noise": "anomaly_reversion",
    "level": "level_persistence",
}
_FORBIDDEN_HANDOFF_TERMS = (
    "candidate_id",
    "candidate_name",
    "hindcast_smae",
    "hindcast_srmse",
    "future_values",
    "gt_evidence",
    "forecast_array",
    "source_code",
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
        forecast = _forecast_tuple(self.forecast)
        object.__setattr__(self, "forecast", forecast)


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
        active = tuple(self.active_candidate_names)
        if (
            not active
            or len(active) != len(set(active))
            or any(not isinstance(name, str) or not name for name in active)
        ):
            raise ValueError("active candidate names must be unique nonempty strings")

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

        if not isinstance(self.selection_decision, SelectionDecision):
            raise ValueError("Numerical package requires a SelectionDecision")
        selection = replace(
            self.selection_decision,
            rejected=MappingProxyType(dict(self.selection_decision.rejected)),
        )
        final = _forecast_tuple(self.final_forecast, horizon=self.task_profile.horizon)
        if final != tuple(selection.forecast):
            raise ValueError("final forecast must equal the protected selector output")
        if not set(selection.selected) <= set(active):
            raise ValueError("selected candidates must belong to the active dictionary")

        alternatives = tuple(self.ranked_alternatives)
        if any(not isinstance(item, RankedNumericalForecast) for item in alternatives):
            raise ValueError("ranked alternatives must be numerical forecast artifacts")
        if tuple(item.rank for item in alternatives) != tuple(
            range(1, len(alternatives) + 1)
        ):
            raise ValueError("ranked alternative ranks must be contiguous")
        if len({item.name for item in alternatives}) != len(alternatives):
            raise ValueError("ranked alternatives must have unique names")
        if not isinstance(self.protected_baseline, RankedNumericalForecast):
            raise ValueError("Numerical package requires a protected baseline")
        if self.protected_baseline.name not in {item.name for item in alternatives}:
            raise ValueError("protected baseline must be a materialized ranked forecast")

        handoff = tuple(_freeze_handoff(item) for item in self.retrieval_handoff)
        if tuple(item["assumption_id"] for item in handoff) != tuple(
            item.assumption_id for item in accepted
        ):
            raise ValueError("Retrieval handoff must correspond exactly to accepted assumptions")
        fingerprints = _freeze_string_mapping(
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
        object.__setattr__(self, "ranked_alternatives", alternatives)
        object.__setattr__(self, "retrieval_handoff", handoff)
        object.__setattr__(self, "component_fingerprints", fingerprints)


@dataclass(frozen=True)
class _CachedFailure:
    detail: str


class _LeafMemo:
    """Execute one leaf at most once for each exact history-only invocation."""

    def __init__(self, runner: CandidateRunner) -> None:
        if not callable(runner):
            raise TypeError("candidate_runner must be callable")
        self._runner = runner
        self._cache: dict[
            tuple[str, tuple[float, ...], int, str], tuple[float, ...] | _CachedFailure
        ] = {}

    def forecast(
        self, name: str, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        key = (name, tuple(history), horizon, frequency)
        cached = self._cache.get(key)
        if isinstance(cached, _CachedFailure):
            raise ValueError(cached.detail)
        if cached is not None:
            return cached
        try:
            raw = self._runner(name, key[1], horizon, frequency)
            forecast = _forecast_tuple(raw, horizon=horizon)
        except BaseException as error:
            detail = f"{type(error).__name__}: {error}"[:200]
            self._cache[key] = _CachedFailure(detail)
            raise ValueError(detail) from None
        self._cache[key] = forecast
        return forecast


def run_numerical_loop(
    task: Task,
    *,
    screening_policy: ScreeningPolicy,
    candidate_runner: CandidateRunner,
    combined_policies: Sequence[CombinedPolicy] = (),
    decision_policy: DecisionPolicy = DecisionPolicy(),
    hindcast_config: HindcastConfig = HindcastConfig(),
    morphology_reasoner: object | None = None,
    diagnostics: Mapping[str, CandidateDiagnostics] | None = None,
    component_fingerprints: Mapping[str, str] | None = None,
) -> NumericalForecastPackage:
    """Run screening, materialization, hindcasting, Morphology, and protected selection.

    ``task.future`` is never read.  Every downstream operation receives a new Task whose future
    is empty, so even a later helper cannot accidentally use an inference label.
    """
    safe_task = _history_only_task(task)
    if not isinstance(screening_policy, ScreeningPolicy):
        raise TypeError("screening_policy must be a ScreeningPolicy")
    if not isinstance(decision_policy, DecisionPolicy):
        raise TypeError("decision_policy must be a DecisionPolicy")
    if not isinstance(hindcast_config, HindcastConfig):
        raise TypeError("hindcast_config must be a HindcastConfig")
    policies = _combined_policy_map(combined_policies)

    # Deterministic morphology and screening always precede candidate execution.
    profile = profile_task(safe_task)
    active_dictionary = materialize_active_dictionary(screening_policy, profile)
    active_names = tuple(item.name for item in active_dictionary.active)
    families = {item.name: item.family for item in active_dictionary.active}
    _validate_active_namespace(active_names, families, policies)
    active_leaf_names = tuple(
        name for name in active_names if families[name] in {"statistical", "tsfm"}
    )
    active_combined_names = tuple(
        name for name in active_names if families[name] == "combined"
    )

    memo = _LeafMemo(candidate_runner)
    leaf_outcomes = {
        name: _materialize_leaf(
            memo,
            name=name,
            history=safe_task.history,
            horizon=safe_task.horizon,
            frequency=safe_task.frequency,
            task_id=safe_task.task_id,
        )
        for name in active_leaf_names
    }
    combined_outcomes = {
        name: combine_materialized_outcome(
            policies[name],
            {
                parent: leaf_outcomes[parent]
                for parent in policies[name].parents
                if parent in leaf_outcomes
            },
            task_id=safe_task.task_id,
            history=safe_task.history,
            horizon=safe_task.horizon,
            frequency=safe_task.frequency,
        )
        for name in active_combined_names
    }
    outcomes = {**leaf_outcomes, **combined_outcomes}
    forecasts = MappingProxyType(
        {
            name: outcome.forecast
            for name, outcome in outcomes.items()
            if outcome.status == SUCCESS
            and _valid_forecast(outcome.forecast, safe_task.horizon)
        }
    )

    def cached_candidate_runner(
        name: str, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        if name in active_leaf_names:
            return memo.forecast(name, tuple(history), horizon, frequency)
        policy = policies.get(name)
        if policy is None or name not in active_combined_names:
            raise ValueError(f"inactive or unknown candidate {name!r}")
        parents = {
            parent: _materialize_leaf(
                memo,
                name=parent,
                history=tuple(history),
                horizon=horizon,
                frequency=frequency,
                task_id=safe_task.task_id,
            )
            for parent in policy.parents
            if parent in active_leaf_names
        }
        outcome = combine_materialized_outcome(
            policy,
            parents,
            task_id=safe_task.task_id,
            history=history,
            horizon=horizon,
            frequency=frequency,
        )
        if outcome.status != SUCCESS or not _valid_forecast(outcome.forecast, horizon):
            raise ValueError(f"Combined candidate {name} failed: {outcome.status}")
        return outcome.forecast

    active = tuple((name, families[name]) for name in active_names)
    if diagnostics is None:
        stable_diagnostics = diagnose_active_candidates(
            safe_task,
            active,
            cached_candidate_runner,
            hindcast_config,
            screening_policy_hash=active_dictionary.screening_policy_hash,
            runtime_settings=dict(component_fingerprints or {}),
        )
    else:
        stable_diagnostics = _snapshot_diagnostics(diagnostics, active)
    stable_diagnostics = MappingProxyType(dict(stable_diagnostics))
    conditioned_names = tuple(
        item.name for item in active_dictionary.active if item.matched_clause >= 0
    )

    protected = select_protected_safe_anchor(
        decision_policy,
        active_names=active_names,
        diagnostics=stable_diagnostics,
        forecasts=forecasts,
        horizon=safe_task.horizon,
        fallback_reason="protected_baseline",
    )
    card: MorphologyCard | None = None
    accepted: tuple[AssumptionGrounding, ...] = ()
    rejected: dict[str, str] = {}
    handoff: tuple[Mapping[str, str], ...] = ()
    fallback_reason: str | None = None

    if morphology_reasoner is None:
        decision = select_numerical_forecast(
            decision_policy,
            profile=profile,
            active_names=active_names,
            diagnostics=stable_diagnostics,
            forecasts=forecasts,
            history=safe_task.history,
            conditioned_names=conditioned_names,
        )
    else:
        try:
            proposed = morphology_reasoner.reason(  # type: ignore[attr-defined]
                history=safe_task.history,
                frequency=safe_task.frequency,
                horizon=safe_task.horizon,
                active_names=active_names,
                families=dict(families),
            )
            if not isinstance(proposed, MorphologyCard):
                raise TypeError("reasoner returned a non-MorphologyCard result")
            card = proposed
            consistency = check_morphology_assumptions(
                card,
                profile=profile,
                active_names=active_names,
                diagnostics=stable_diagnostics,
                forecasts=forecasts,
                policy=decision_policy,
                min_successful_folds=hindcast_config.min_successful_folds,
            )
            accepted, rejected, handoff = _safe_retrieval_projection(
                consistency.accepted,
                consistency.rejected,
            )
        except Exception as error:
            fallback_reason = f"morphology_reasoner_failed:{type(error).__name__}"
            decision = _fallback_decision(
                protected, fallback_reason=fallback_reason
            )
        else:
            if not accepted:
                fallback_reason = "morphology_consistency_rejected"
                decision = _fallback_decision(
                    protected, fallback_reason=fallback_reason
                )
            else:
                try:
                    decision = select_grounded_morphology_forecast(
                        decision_policy,
                        assumptions=accepted,
                        profile=profile,
                        active_names=active_names,
                        diagnostics=stable_diagnostics,
                        forecasts=forecasts,
                        history=safe_task.history,
                        conditioned_names=conditioned_names,
                    )
                    if not _materialized_single_decision(
                        decision, forecasts, safe_task.horizon
                    ):
                        raise ValueError("selector returned an unmaterialized forecast")
                except Exception:
                    fallback_reason = "protected_selector_rejected"
                    for item in accepted:
                        rejected.setdefault(item.assumption_id, fallback_reason)
                    accepted = ()
                    handoff = ()
                    decision = _fallback_decision(
                        protected, fallback_reason=fallback_reason
                    )

    if not _valid_forecast(decision.forecast, safe_task.horizon):
        raise ValueError("protected selector returned a non-finite or wrong-horizon forecast")

    alternatives = _ranked_forecasts(
        active_names, families, stable_diagnostics, forecasts
    )
    protected_baseline = next(
        (item for item in alternatives if item.name == protected.selected[0]),
        None,
    )
    if protected_baseline is None:
        raise ValueError("protected Safe-Anchor was not materialized")
    fingerprints = _component_fingerprints(
        profile=profile,
        active_dictionary=active_dictionary,
        screening_policy=screening_policy,
        combined_policies=tuple(policies.values()),
        decision_policy=decision_policy,
        hindcast_config=hindcast_config,
        morphology_card=card,
        provided=component_fingerprints,
    )
    return NumericalForecastPackage(
        task_profile=profile,
        active_candidate_names=active_names,
        candidate_diagnostics=stable_diagnostics,
        morphology_card=card,
        accepted_assumptions=accepted,
        rejected_assumptions=rejected,
        selection_decision=decision,
        final_forecast=tuple(decision.forecast),
        protected_baseline=protected_baseline,
        ranked_alternatives=alternatives,
        retrieval_handoff=handoff,
        component_fingerprints=fingerprints,
        fallback_reason=fallback_reason,
    )


def _history_only_task(task: Task) -> Task:
    if not isinstance(task, Task):
        raise TypeError("task must be a Task")
    if not isinstance(task.task_id, str) or not task.task_id:
        raise ValueError("task_id must be a nonempty string")
    if isinstance(task.horizon, bool) or not isinstance(task.horizon, int) or task.horizon < 1:
        raise ValueError("task horizon must be positive")
    if not isinstance(task.frequency, str) or not task.frequency:
        raise ValueError("task frequency must be a nonempty string")
    history = _forecast_tuple(task.history)
    return Task(task.task_id, history, task.horizon, task.frequency, ())


def _combined_policy_map(
    policies: Sequence[CombinedPolicy],
) -> dict[str, CombinedPolicy]:
    if isinstance(policies, (str, bytes)):
        raise TypeError("combined_policies must be a sequence")
    result: dict[str, CombinedPolicy] = {}
    for policy in tuple(policies):
        if not isinstance(policy, CombinedPolicy):
            raise TypeError("combined_policies must contain CombinedPolicy artifacts")
        if policy.name in result:
            raise ValueError(f"duplicate Combined policy {policy.name!r}")
        result[policy.name] = policy
    combined_names = set(result)
    if any(parent in combined_names for policy in result.values() for parent in policy.parents):
        raise ValueError("Combined policies cannot consume Combined parents")
    return result


def _validate_active_namespace(
    active_names: Sequence[str],
    families: Mapping[str, str],
    policies: Mapping[str, CombinedPolicy],
) -> None:
    if not active_names:
        raise ValueError("screening produced no active candidates")
    active_combined = {name for name in active_names if families.get(name) == "combined"}
    if missing := active_combined - set(policies):
        raise ValueError(f"active Combined candidates have no policy: {sorted(missing)!r}")
    if collisions := {
        name for name in active_names if families.get(name) != "combined" and name in policies
    }:
        raise ValueError(f"Combined/leaf namespace collision: {sorted(collisions)!r}")
    if unsupported := {
        family for family in families.values() if family not in {"statistical", "tsfm", "combined"}
    }:
        raise ValueError(f"unsupported active candidate families: {sorted(unsupported)!r}")


def _materialize_leaf(
    memo: _LeafMemo,
    *,
    name: str,
    history: tuple[float, ...],
    horizon: int,
    frequency: str,
    task_id: str,
) -> Outcome:
    try:
        forecast = memo.forecast(name, history, horizon, frequency)
    except Exception as error:
        return Outcome(name, task_id, CRASHED, detail=str(error)[:200])
    if not _valid_forecast(forecast, horizon):
        return Outcome(name, task_id, INVALID, detail="leaf forecast is invalid")
    return Outcome(name, task_id, SUCCESS, forecast=forecast)


def _snapshot_diagnostics(
    diagnostics: Mapping[str, CandidateDiagnostics],
    active: Sequence[tuple[str, str]],
) -> dict[str, CandidateDiagnostics]:
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


def _freeze_diagnostic(value: CandidateDiagnostics) -> CandidateDiagnostics:
    """Detach the package from mutable containers on externally supplied diagnostics."""
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


def _safe_retrieval_projection(
    accepted: Sequence[AssumptionGrounding],
    rejected: Mapping[str, str],
) -> tuple[
    tuple[AssumptionGrounding, ...],
    dict[str, str],
    tuple[Mapping[str, str], ...],
]:
    safe: list[AssumptionGrounding] = []
    trace = dict(rejected)
    payloads: list[Mapping[str, str]] = []
    for item in accepted:
        text = f"{item.claim} {item.failure_condition}".lower()
        if any(term in text for term in _FORBIDDEN_HANDOFF_TERMS):
            trace[item.assumption_id] = "unsafe_retrieval_handoff"
            continue
        kind = _RETRIEVAL_KIND.get(item.kind)
        if kind is None:
            trace[item.assumption_id] = "unsupported_retrieval_kind"
            continue
        payload = MappingProxyType(
            {
                "assumption_id": item.assumption_id,
                "kind": kind,
                "claim": item.claim,
                "failure_condition": item.failure_condition,
            }
        )
        safe.append(item)
        payloads.append(payload)
    return tuple(safe), trace, tuple(payloads)


def _fallback_decision(
    protected: SelectionDecision, *, fallback_reason: str
) -> SelectionDecision:
    return replace(
        protected,
        reason_codes=("protected_safe_anchor", fallback_reason),
        rejected={},
    )


def _materialized_single_decision(
    decision: SelectionDecision,
    forecasts: Mapping[str, Sequence[float]],
    horizon: int,
) -> bool:
    if decision.mode != "single" or len(decision.selected) != 1:
        return False
    name = decision.selected[0]
    materialized = forecasts.get(name)
    return (
        materialized is not None
        and _valid_forecast(materialized, horizon)
        and tuple(float(value) for value in materialized) == tuple(decision.forecast)
    )


def _ranked_forecasts(
    active_names: Sequence[str],
    families: Mapping[str, str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
) -> tuple[RankedNumericalForecast, ...]:
    available = [
        diagnostics[name]
        for name in active_names
        if name in diagnostics and name in forecasts
    ]
    available.sort(
        key=lambda item: (
            not item.eligible,
            -item.successful_folds,
            _rank_number(item.median_mase),
            _rank_number(item.recent_mase),
            _rank_number(item.worst_mase),
            _rank_number(item.mase_mad),
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


def _component_fingerprints(
    *,
    profile: TaskProfile,
    active_dictionary: ActiveDictionary,
    screening_policy: ScreeningPolicy,
    combined_policies: Sequence[CombinedPolicy],
    decision_policy: DecisionPolicy,
    hindcast_config: HindcastConfig,
    morphology_card: MorphologyCard | None,
    provided: Mapping[str, str] | None,
) -> Mapping[str, str]:
    result = {
        "task_profile": active_dictionary.task_profile_hash,
        "screening_policy": screening_policy.fingerprint(),
        "active_dictionary": _fingerprint(
            {
                "profile": profile.to_public_payload(),
                "active": [item.name for item in active_dictionary.active],
                "excluded": [item.name for item in active_dictionary.excluded],
                "fallback_applied": active_dictionary.fallback_applied,
            }
        ),
        "combined_policies": _fingerprint(
            [policy.to_payload() for policy in combined_policies]
        ),
        "decision_policy": _fingerprint(asdict(decision_policy)),
        "hindcast_config": _fingerprint(asdict(hindcast_config)),
        "morphology_card": (
            morphology_card.fingerprint
            if morphology_card is not None
            else _fingerprint({"enabled": False})
        ),
    }
    if provided is not None:
        external = dict(_freeze_string_mapping(provided, "component fingerprints"))
        conflicts = {
            key for key in external if key in result and external[key] != result[key]
        }
        if conflicts:
            raise ValueError(
                f"provided component fingerprints conflict with host values: {sorted(conflicts)!r}"
            )
        result.update(external)
    return MappingProxyType(dict(sorted(result.items())))


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _freeze_string_mapping(
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


def _forecast_tuple(
    values: Sequence[float], *, horizon: int | None = None
) -> tuple[float, ...]:
    result = _finite_tuple(values, allow_empty=False)
    if horizon is not None and len(result) != horizon:
        raise ValueError("forecast has the wrong horizon")
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


def _valid_forecast(values: Sequence[float], horizon: int) -> bool:
    try:
        _forecast_tuple(values, horizon=horizon)
    except (TypeError, ValueError):
        return False
    return True


def _rank_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.inf
    number = float(value)
    return number if math.isfinite(number) else math.inf
