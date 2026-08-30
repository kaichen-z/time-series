"""History-only orchestration for the morphology-guided Numerical loop."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from .execution import CRASHED, INVALID, SUCCESS, Outcome, Task
from .morphology import AssumptionGrounding, MorphologyCard
from .morphology_consistency import check_morphology_assumptions
from .numerical_handoff import (
    component_fingerprints as build_component_fingerprints,
    safe_retrieval_projection,
)
from .numerical_package import (
    NumericalForecastPackage,
    RankedNumericalForecast,
    forecast_tuple,
    ranked_forecasts,
    snapshot_diagnostics,
    valid_forecast,
)
from .numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    HindcastConfig,
    SelectionDecision,
    diagnose_active_candidates,
    select_assumption_guided_forecast,
    select_grounded_morphology_forecast,
    select_protected_safe_anchor,
)
from .portfolio import CombinedPolicy, combine_materialized_outcome
from .screening import (
    ScreeningPolicy,
    materialize_active_dictionary,
    profile_task,
)


CandidateRunner = Callable[[str, tuple[float, ...], int, str], Sequence[float]]


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
            forecast = forecast_tuple(raw, horizon=horizon)
        except Exception as error:
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
            and valid_forecast(outcome.forecast, safe_task.horizon)
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
        if outcome.status != SUCCESS or not valid_forecast(outcome.forecast, horizon):
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
        stable_diagnostics = snapshot_diagnostics(diagnostics, active)
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
        decision = select_assumption_guided_forecast(
            decision_policy,
            profile=profile,
            active_names=active_names,
            diagnostics=stable_diagnostics,
            forecasts=forecasts,
            families=families,
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
                protected_anchor_name=protected.selected[0],
            )
            accepted, rejected, handoff = safe_retrieval_projection(
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
                        protected_anchor=protected,
                        horizon=safe_task.horizon,
                        profile=profile,
                        active_names=active_names,
                        diagnostics=stable_diagnostics,
                        forecasts=forecasts,
                        history=safe_task.history,
                        conditioned_names=conditioned_names,
                    )
                except Exception:
                    fallback_reason = "protected_selector_rejected"
                    for item in accepted:
                        rejected.setdefault(item.assumption_id, fallback_reason)
                    accepted = ()
                    handoff = ()
                    decision = _fallback_decision(
                        protected, fallback_reason=fallback_reason
                    )

    if not valid_forecast(decision.forecast, safe_task.horizon):
        raise ValueError("protected selector returned a non-finite or wrong-horizon forecast")

    alternatives = ranked_forecasts(
        active_names, families, stable_diagnostics, forecasts
    )
    protected_baseline = next(
        (item for item in alternatives if item.name == protected.selected[0]),
        None,
    )
    if protected_baseline is None:
        raise ValueError("protected Safe-Anchor was not materialized")
    fingerprints = build_component_fingerprints(
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
    history = forecast_tuple(task.history)
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
    if not valid_forecast(forecast, horizon):
        return Outcome(name, task_id, INVALID, detail="leaf forecast is invalid")
    return Outcome(name, task_id, SUCCESS, forecast=forecast)


def _fallback_decision(
    protected: SelectionDecision, *, fallback_reason: str
) -> SelectionDecision:
    return replace(
        protected,
        reason_codes=("protected_safe_anchor", fallback_reason),
        rejected={},
    )
