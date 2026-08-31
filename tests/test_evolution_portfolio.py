from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from numerical_agent.dictionary import MethodCandidate
import numerical_agent.evolution.portfolio as portfolio_module
from numerical_agent.evolution.cache import CacheMissError, OutcomeCache
from numerical_agent.evolution.execution import CRASHED, INVALID, NOT_APPLICABLE, SUCCESS, Outcome, Task
from numerical_agent.evolution.module import MODULE_HEADER, parse_module
from numerical_agent.evolution.portfolio import (
    FLAGSHIP_METHOD_IDS,
    CombinedPolicy,
    PolicyError,
    PolicyPortfolio,
    TSFMPolicy,
    PolicyOutcomeCache,
    _run_combined,
    combine_materialized_outcome,
    _run_tsfm,
    evaluate_portfolio,
    parse_policy_source,
    render_policy_source,
)
from numerical_agent.providers import RuntimeRegistry


def _canonical_combined(**overrides: object) -> CombinedPolicy:
    fields: dict[str, object] = {
        "name": "combined_tsfm_median",
        "parents": ("toto_2_0", "timesfm_2_5", "chronos_bolt"),
        "operator": "median",
        "weights": (),
        "signal": "periodicity_strength",
        "threshold": 0.0,
        "above_parent": "",
        "below_parent": "",
        "fallback_parent": "toto_2_0",
    }
    fields.update(overrides)
    return CombinedPolicy(**fields)


def _legacy_source() -> str:
    """Independent frozen v1 source; do not derive this fixture from flagship5()."""
    return """TSFM_POLICIES = (
    {'name': 'timesfm_2_5', 'method_id': 'method_tsfm_0031', 'applicability': 'all', 'context_window': 1024, 'preprocess': 'none', 'shrinkage_to_last': 0.0},
    {'name': 'moirai_2_0', 'method_id': 'method_tsfm_0017', 'applicability': 'all', 'context_window': 512, 'preprocess': 'none', 'shrinkage_to_last': 0.0},
    {'name': 'toto_2_0', 'method_id': 'method_tsfm_0014', 'applicability': 'all', 'context_window': 512, 'preprocess': 'none', 'shrinkage_to_last': 0.0},
    {'name': 'chronos_bolt', 'method_id': 'method_tsfm_0018', 'applicability': 'all', 'context_window': 512, 'preprocess': 'none', 'shrinkage_to_last': 0.0},
    {'name': 'granite_ttm_r2', 'method_id': 'method_tsfm_0006', 'applicability': 'all', 'context_window': 512, 'preprocess': 'none', 'shrinkage_to_last': 0.0},
)
COMBINED_POLICIES = (
    {'name': 'combined_timesfm_seasonal', 'tsfm_parent': 'timesfm_2_5', 'statistical_parent': 'seasonal_naive', 'mode': 'blend', 'weight': 0.65, 'signal': 'periodicity_strength', 'threshold': 0.45, 'tsfm_when': 'above'},
    {'name': 'combined_chronos_damped_trend', 'tsfm_parent': 'chronos_bolt', 'statistical_parent': 'holt_damped_trend', 'mode': 'blend', 'weight': 0.65, 'signal': 'trend_strength', 'threshold': 0.45, 'tsfm_when': 'above'},
    {'name': 'combined_moirai_croston_router', 'tsfm_parent': 'moirai_2_0', 'statistical_parent': 'croston_sba', 'mode': 'route', 'weight': 0.65, 'signal': 'zero_fraction', 'threshold': 0.30, 'tsfm_when': 'below'},
    {'name': 'combined_toto_robust_router', 'tsfm_parent': 'toto_2_0', 'statistical_parent': 'robust_loess_trend', 'mode': 'route', 'weight': 0.65, 'signal': 'outlier_fraction', 'threshold': 0.05, 'tsfm_when': 'below'},
    {'name': 'combined_granite_regime_profile', 'tsfm_parent': 'granite_ttm_r2', 'statistical_parent': 'median_seasonal_profile_forecast', 'mode': 'blend', 'weight': 0.60, 'signal': 'recent_regime_confidence', 'threshold': 0.50, 'tsfm_when': 'above'},
)
"""


def test_canonical_combined_policy_round_trips() -> None:
    policy = _canonical_combined()

    assert CombinedPolicy(**policy.to_payload()) == policy


def test_lead_time_route_round_trips_through_literal_python_policy_source() -> None:
    base = PolicyPortfolio.flagship5()
    policy = CombinedPolicy(
        "combined_lead_time_route",
        ("chronos_bolt", "timesfm_2_5", "seasonal_naive"),
        "lead_time_route",
        (0.25, 0.50, 0.25),
        fallback_parent="timesfm_2_5",
    )
    portfolio = PolicyPortfolio(base.tsfm, (policy,))

    assert parse_policy_source(render_policy_source(portfolio)) == portfolio


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parents": ("toto_2_0", "toto_2_0")}, "unique"),
        ({"parents": ("toto_2_0",)}, "between 2 and 5"),
        ({"parents": ("a", "b", "c", "d", "e", "f")}, "between 2 and 5"),
        ({"operator": "blend"}, "unsupported"),
        ({"weights": (0.5, 0.5, 0.0)}, "weights"),
        (
            {"operator": "route", "parents": ("toto_2_0", "timesfm_2_5", "chronos_bolt")},
            "exactly two",
        ),
        ({"above_parent": "missing"}, "route branch"),
        ({"fallback_parent": "missing"}, "fallback"),
        ({"operator": "weighted_mean", "weights": (-0.1, 1.1, 0.0)}, "weights"),
        ({"operator": "weighted_mean", "weights": (float("nan"), 0.5, 0.5)}, "weights"),
        ({"operator": "weighted_mean", "weights": (0.2, 0.2, 0.2)}, "sum"),
    ],
)
def test_invalid_canonical_combined_policy_is_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(PolicyError, match=message):
        _canonical_combined(**overrides)


def test_route_canonical_policy_requires_branches_in_parents() -> None:
    with pytest.raises(PolicyError, match="route branch"):
        CombinedPolicy(
            name="combined_route",
            parents=("toto_2_0", "timesfm_2_5"),
            operator="route",
            above_parent="missing",
            below_parent="timesfm_2_5",
            fallback_parent="toto_2_0",
        )


def test_legacy_combined_payload_migrates_to_canonical_schema() -> None:
    source = _legacy_source()

    policy = parse_policy_source(source).combined[0]

    assert policy.parents == ("timesfm_2_5", "seasonal_naive")
    assert policy.operator == "weighted_mean"
    assert policy.weights == (0.65, 0.35)
    assert policy.fallback_parent == "timesfm_2_5"
    rendered = render_policy_source(PolicyPortfolio.flagship5())
    assert "tsfm_parent" not in rendered
    assert "statistical_parent" not in rendered
    assert "mode" not in rendered
    assert "'weight':" not in rendered


def test_legacy_route_payload_migrates_to_explicit_branches() -> None:
    source = _legacy_source()

    policy = parse_policy_source(source).combined[2]

    assert policy.operator == "route"
    assert policy.weights == ()
    assert policy.above_parent == "croston_sba"
    assert policy.below_parent == "moirai_2_0"
    assert policy.fallback_parent == "moirai_2_0"


def test_legacy_flagship_five_preserves_names_and_forecasts_after_migration(
    tmp_path: Path,
) -> None:
    """Legacy policy files retain every identity and forecast after canonical migration."""
    canonical = PolicyPortfolio.flagship5()
    migrated = parse_policy_source(_legacy_source())

    expected_tsfm = (
        TSFMPolicy("timesfm_2_5", "method_tsfm_0031", context_window=1024),
        TSFMPolicy("moirai_2_0", "method_tsfm_0017", context_window=512),
        TSFMPolicy("toto_2_0", "method_tsfm_0014", context_window=512),
        TSFMPolicy("chronos_bolt", "method_tsfm_0018", context_window=512),
        TSFMPolicy("granite_ttm_r2", "method_tsfm_0006", context_window=512),
    )
    expected_combined = (
        CombinedPolicy("combined_timesfm_seasonal", ("timesfm_2_5", "seasonal_naive"), "weighted_mean", (0.65, 0.35), "periodicity_strength", 0.45, fallback_parent="timesfm_2_5"),
        CombinedPolicy("combined_chronos_damped_trend", ("chronos_bolt", "holt_damped_trend"), "weighted_mean", (0.65, 0.35), "trend_strength", 0.45, fallback_parent="chronos_bolt"),
        CombinedPolicy("combined_moirai_croston_router", ("moirai_2_0", "croston_sba"), "route", (), "zero_fraction", 0.30, above_parent="croston_sba", below_parent="moirai_2_0", fallback_parent="moirai_2_0"),
        CombinedPolicy("combined_toto_robust_router", ("toto_2_0", "robust_loess_trend"), "route", (), "outlier_fraction", 0.05, above_parent="robust_loess_trend", below_parent="toto_2_0", fallback_parent="toto_2_0"),
        CombinedPolicy("combined_granite_regime_profile", ("granite_ttm_r2", "median_seasonal_profile_forecast"), "weighted_mean", (0.60, 0.40), "recent_regime_confidence", 0.50, fallback_parent="granite_ttm_r2"),
    )

    assert tuple(policy.name for policy in migrated.combined) == (
        "combined_timesfm_seasonal",
        "combined_chronos_damped_trend",
        "combined_moirai_croston_router",
        "combined_toto_robust_router",
        "combined_granite_regime_profile",
    )
    assert migrated.tsfm == expected_tsfm
    assert migrated.combined == expected_combined
    assert canonical.tsfm == expected_tsfm
    assert canonical.combined == expected_combined

    runtime = FakeTSFMRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})
    canonical_outcomes = evaluate_portfolio(
        _module(),
        canonical,
        _tasks(),
        outcome_cache=OutcomeCache(tmp_path / "canonical-cache"),
        runtimes=_registry(runtime),
        isolated_methods=False,
    )
    migrated_outcomes = evaluate_portfolio(
        _module(),
        migrated,
        _tasks(),
        outcome_cache=OutcomeCache(tmp_path / "migrated-cache"),
        runtimes=_registry(runtime),
        isolated_methods=False,
    )

    daily = {
        "seasonal_naive": (27.0, 27.0),
        "holt_damped_trend": (27.0, 27.0),
        "croston_sba": (27.0, 27.0),
        "robust_loess_trend": (27.0, 27.0),
        "median_seasonal_profile_forecast": (27.0, 27.0),
        "timesfm_2_5": (29.0, 29.0),
        "moirai_2_0": (29.0, 29.0),
        "toto_2_0": (29.0, 29.0),
        "chronos_bolt": (29.0, 29.0),
        "granite_ttm_r2": (29.0, 29.0),
        "combined_timesfm_seasonal": (28.3, 28.3),
        "combined_chronos_damped_trend": (28.3, 28.3),
        "combined_moirai_croston_router": (29.0, 29.0),
        "combined_toto_robust_router": (29.0, 29.0),
        "combined_granite_regime_profile": (28.2, 28.2),
    }
    periodic = {
        "seasonal_naive": (6.0, 6.0),
        "holt_damped_trend": (6.0, 6.0),
        "croston_sba": (6.0, 6.0),
        "robust_loess_trend": (6.0, 6.0),
        "median_seasonal_profile_forecast": (6.0, 6.0),
        "timesfm_2_5": (8.0, 8.0),
        "moirai_2_0": (8.0, 8.0),
        "toto_2_0": (8.0, 8.0),
        "chronos_bolt": (8.0, 8.0),
        "granite_ttm_r2": (8.0, 8.0),
        "combined_timesfm_seasonal": (7.3, 7.3),
        "combined_chronos_damped_trend": (7.3, 7.3),
        "combined_moirai_croston_router": (8.0, 8.0),
        "combined_toto_robust_router": (8.0, 8.0),
        "combined_granite_regime_profile": (7.2, 7.2),
    }
    golden = {
        (method, task_id): (SUCCESS, forecast)
        for task_id, forecasts in (("daily", daily), ("periodic", periodic))
        for method, forecast in forecasts.items()
    }
    for outcomes in (canonical_outcomes, migrated_outcomes):
        actual = {
            (outcome.method, outcome.task_id): (outcome.status, outcome.forecast)
            for outcome in outcomes
        }
        assert actual == golden


class FakeTSFMRuntime:
    def __init__(self, offsets: dict[str, float]) -> None:
        self.offsets = offsets
        self.calls: list[tuple[str, tuple[float, ...], int, str]] = []

    def supports(self, candidate: MethodCandidate) -> bool:
        return candidate.method_id in self.offsets

    def forecast(
        self,
        candidate: MethodCandidate,
        history: list[float] | tuple[float, ...],
        horizon: int,
        frequency: str,
    ) -> tuple[float, ...]:
        values = tuple(float(value) for value in history)
        self.calls.append((candidate.method_id, values, horizon, frequency))
        return tuple(values[-1] + self.offsets[candidate.method_id] for _ in range(horizon))


def _portfolio() -> PolicyPortfolio:
    return parse_policy_source(render_policy_source(PolicyPortfolio.flagship5()))


def _module():
    functions = []
    for name in (
        "seasonal_naive",
        "holt_damped_trend",
        "croston_sba",
        "robust_loess_trend",
        "median_seasonal_profile_forecast",
    ):
        functions.append(
            f'''def {name}(history, horizon, frequency):
    """Use as a deterministic statistical parent in portfolio tests."""
    return [float(history[-1]) - 1.0] * horizon
'''
        )
    return parse_module(MODULE_HEADER + "\n\n" + "\n\n".join(functions))


def _tasks() -> tuple[Task, ...]:
    return (
        Task("daily", tuple(float(value) for value in range(1, 29)), 2, "1 day", (29.0, 30.0)),
        Task(
            "periodic",
            tuple(float((index % 7) + 1) for index in range(56)),
            2,
            "1 day",
            (1.0, 2.0),
        ),
    )


def _registry(runtime: FakeTSFMRuntime) -> RuntimeRegistry:
    # The production policies retain their reviewed provider names.  One fake
    # runtime is registered under each name so construction never loads a model.
    return RuntimeRegistry(
        {
            "timesfm": runtime,
            "chronos": runtime,
            "tsfm_worker": runtime,
        }
    )


def test_flagship_portfolio_contains_five_tsfm_and_five_combined() -> None:
    portfolio = PolicyPortfolio.flagship5()

    assert tuple(policy.method_id for policy in portfolio.tsfm) == FLAGSHIP_METHOD_IDS
    assert len(portfolio.tsfm) == 5
    assert len(portfolio.combined) == 5
    assert len({policy.name for policy in portfolio.all_policies}) == 10
    assert all(any(parent in portfolio.names for parent in policy.parents) for policy in portfolio.combined)


def test_policy_source_is_python_literal_and_round_trips() -> None:
    portfolio = PolicyPortfolio.flagship5()
    source = render_policy_source(portfolio)

    assert "TSFM_POLICIES =" in source
    assert "COMBINED_POLICIES =" in source
    assert parse_policy_source(source) == portfolio


def test_policy_parser_rejects_tsfm_identity_substitution() -> None:
    source = render_policy_source(_portfolio()).replace(
        "'method_tsfm_0031'", "'attacker/checkpoint'", 1
    )

    with pytest.raises(PolicyError, match="flagship TSFM identities"):
        parse_policy_source(source)


def test_policy_parser_rejects_empty_combined_portfolio() -> None:
    source = render_policy_source(_portfolio())
    source = source[: source.index("COMBINED_POLICIES =")] + "COMBINED_POLICIES = ()\n"

    with pytest.raises(PolicyError, match="between 1 and 32"):
        parse_policy_source(source)


def test_policy_parser_rejects_combined_to_combined_parent_dependency() -> None:
    source = render_policy_source(_portfolio()).replace(
        "'seasonal_naive'", "'combined_chronos_damped_trend'", 1
    )

    with pytest.raises(PolicyError, match="Combined.*parent"):
        parse_policy_source(source)


def test_combined_repair_can_change_parent_identity() -> None:
    portfolio = _portfolio()
    parent = portfolio.combined[0]
    replacement = CombinedPolicy(
        name=parent.name,
        parents=(parent.parents[0], "holt_damped_trend"),
        operator=parent.operator,
        weights=parent.weights,
        signal=parent.signal,
        threshold=parent.threshold,
        fallback_parent=parent.fallback_parent,
    )

    child = portfolio.replace(parent.name, replacement)

    assert child.get(parent.name) == replacement


def test_tsfm_repair_can_change_policy_but_not_method_id() -> None:
    portfolio = _portfolio()
    parent = portfolio.tsfm[0]
    replacement = TSFMPolicy(
        name=parent.name,
        method_id=parent.method_id,
        applicability="periodic",
        context_window=512,
        preprocess="robust_scale",
        shrinkage_to_last=0.2,
    )

    child = portfolio.replace(parent.name, replacement)

    assert child.get(parent.name) == replacement
    assert child.get(parent.name).method_id == parent.method_id  # type: ignore[union-attr]


def test_evaluate_portfolio_adds_ten_real_candidate_outcomes(tmp_path: Path) -> None:
    module = _module()
    portfolio = _portfolio()
    runtime = FakeTSFMRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})

    outcomes = evaluate_portfolio(
        module,
        portfolio,
        _tasks(),
        outcome_cache=OutcomeCache(tmp_path / "cache"),
        runtimes=_registry(runtime),
        isolated_methods=False,
    )

    names = set(module.names()) | set(portfolio.names)
    assert {outcome.method for outcome in outcomes} == names
    assert len(outcomes) == len(names) * len(_tasks())
    assert all(outcome.status == SUCCESS for outcome in outcomes)
    assert len(runtime.calls) == len(portfolio.tsfm) * len(_tasks())


def test_policy_cache_only_lookup_never_calls_runtime_on_a_miss(tmp_path: Path) -> None:
    cache = PolicyOutcomeCache(tmp_path / "policy-cache")
    runtime = FakeTSFMRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})
    policy = _portfolio().tsfm[0]

    with pytest.raises(CacheMissError, match=f"{policy.name}.*daily"):
        cache.require_cached(policy, _tasks()[0])

    assert runtime.calls == []


def test_policy_cache_existing_wrong_policy_binding_fails_closed(tmp_path: Path) -> None:
    cache = PolicyOutcomeCache(tmp_path / "policy-cache")
    runtime = FakeTSFMRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})
    policy = _portfolio().tsfm[0]
    task = _tasks()[0]
    cache.evaluate(policy, task, _registry(runtime))
    entry = next((tmp_path / "policy-cache").glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["metric_policy_fingerprint"] = "0" * 64
    entry.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        cache.evaluate(policy, task, _registry(runtime))


@pytest.mark.parametrize("schema", (True, 3.0))
def test_policy_cache_rejects_noninteger_schema_aliases(tmp_path: Path, schema) -> None:
    cache = PolicyOutcomeCache(tmp_path / "policy-cache")
    runtime = FakeTSFMRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})
    policy = _portfolio().tsfm[0]
    task = _tasks()[0]
    cache.evaluate(policy, task, _registry(runtime))
    entry = next((tmp_path / "policy-cache").glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["cache_schema"] = schema
    entry.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PolicyError, match="schema mismatch"):
        cache.evaluate(policy, task, _registry(runtime))


def test_combined_forecast_is_computed_from_both_parent_forecasts(tmp_path: Path) -> None:
    module = _module()
    portfolio = _portfolio()
    runtime = FakeTSFMRuntime({method_id: 2.0 for method_id in FLAGSHIP_METHOD_IDS})

    outcomes = evaluate_portfolio(
        module,
        portfolio,
        _tasks()[:1],
        outcome_cache=OutcomeCache(tmp_path / "cache"),
        runtimes=_registry(runtime),
        isolated_methods=False,
    )
    combined = next(
        outcome for outcome in outcomes
        if outcome.method == "combined_timesfm_seasonal"
    )

    # TSFM gives 30; statistical parent gives 27.  With the initial TSFM
    # weight 0.65, the executable blend is 28.95 rather than either parent.
    assert combined.forecast == pytest.approx((28.95, 28.95))
    assert combined.status == SUCCESS


def test_93_methods_plus_portfolio_reports_parent_and_child_counts() -> None:
    names = tuple(f"method_{index:03d}" for index in range(93))
    parent = _portfolio()
    child = parent.add_combined(_variable_combined("combined_count_child"))

    assert len(names) == 93
    assert len(parent.names) == 10
    assert len(child.names) == 11
    assert len(names) + len(parent.names) == 103
    assert len(names) + len(child.names) == 104


def _variable_combined(name: str = "combined_variable") -> CombinedPolicy:
    return CombinedPolicy(
        name=name,
        parents=("toto_2_0", "seasonal_naive"),
        operator="weighted_mean",
        weights=(0.5, 0.5),
        fallback_parent="toto_2_0",
    )


def test_portfolio_shape_accepts_one_to_32_variable_combined_policies() -> None:
    base = PolicyPortfolio.flagship5()
    one = PolicyPortfolio(base.tsfm, (_variable_combined(),))
    thirty_two = PolicyPortfolio(
        base.tsfm,
        tuple(_variable_combined(f"combined_{index}") for index in range(32)),
    )

    assert len(one.combined) == 1
    assert len(thirty_two.combined) == 32


@pytest.mark.parametrize("count", [0, 33])
def test_portfolio_shape_rejects_combined_count_outside_one_to_32(count: int) -> None:
    base = PolicyPortfolio.flagship5()
    combined = tuple(_variable_combined(f"combined_{index}") for index in range(count))

    with pytest.raises(PolicyError, match="between 1 and 32"):
        PolicyPortfolio(base.tsfm, combined)


def test_portfolio_preserves_fixed_tsfm_identity_and_order() -> None:
    base = PolicyPortfolio.flagship5()
    reordered = tuple(reversed(base.tsfm))

    with pytest.raises(PolicyError, match="identities and order"):
        PolicyPortfolio(reordered, (_variable_combined(),))


def test_validate_parents_accepts_cross_family_leaf_parents() -> None:
    base = PolicyPortfolio.flagship5()
    policy = CombinedPolicy(
        name="combined_cross_family",
        parents=("toto_2_0", "timesfm_2_5", "seasonal_naive"),
        operator="median",
        fallback_parent="toto_2_0",
    )
    portfolio = PolicyPortfolio(base.tsfm, (policy,))

    portfolio.validate_parents(_module().names())


def test_validate_parents_accepts_two_tsfm_parents() -> None:
    base = PolicyPortfolio.flagship5()
    policy = CombinedPolicy(
        name="combined_tsfm_pair",
        parents=("toto_2_0", "timesfm_2_5"),
        operator="median",
        fallback_parent="toto_2_0",
    )

    PolicyPortfolio(base.tsfm, (policy,)).validate_parents(_module().names())


def test_portfolio_rejects_duplicate_combined_names() -> None:
    base = PolicyPortfolio.flagship5()
    duplicate = _variable_combined("same_name")

    with pytest.raises(PolicyError, match="unique"):
        PolicyPortfolio(base.tsfm, (duplicate, duplicate))


@pytest.mark.parametrize(
    "parents",
    [
        ("seasonal_naive", "holt_damped_trend"),
        ("toto_2_0", "unknown_leaf"),
        ("toto_2_0", "combined_chronos_damped_trend"),
    ],
)
def test_validate_parents_rejects_non_leaf_or_all_statistical_parents(
    parents: tuple[str, str],
) -> None:
    base = PolicyPortfolio.flagship5()
    policy = CombinedPolicy(
        name="combined_parent_validation",
        parents=parents,
        operator="weighted_mean",
        weights=(0.5, 0.5),
        fallback_parent=parents[0],
    )

    if parents[1] == "combined_chronos_damped_trend":
        other = _variable_combined(parents[1])
        with pytest.raises(PolicyError, match="Combined.*parent"):
            PolicyPortfolio(base.tsfm, (policy, other))
    elif parents[0] == "seasonal_naive":
        with pytest.raises(PolicyError, match="TSFM parent"):
            PolicyPortfolio(base.tsfm, (policy,))
    else:
        portfolio = PolicyPortfolio(base.tsfm, (policy,))
        with pytest.raises(PolicyError, match="unknown parent"):
            portfolio.validate_parents(_module().names())


def test_atomic_combined_mutations_are_immutable_and_allow_parent_repair() -> None:
    parent = PolicyPortfolio.flagship5()
    new_policy = _variable_combined()
    changed = CombinedPolicy(
        name=new_policy.name,
        parents=("timesfm_2_5", "seasonal_naive"),
        operator="weighted_mean",
        weights=(0.25, 0.75),
        fallback_parent="timesfm_2_5",
    )
    fork = _variable_combined("combined_fork")

    child = parent.add_combined(new_policy)
    repaired = child.replace(new_policy.name, changed)
    forked = repaired.fork_combined(new_policy.name, fork)
    removed = forked.remove_combined(fork.name)

    assert parent == PolicyPortfolio.flagship5()
    assert repaired.get(new_policy.name) == changed
    assert forked.get(fork.name) == fork
    assert removed.get(fork.name) is None


def test_atomic_invalid_combined_mutations_leave_parent_byte_identical() -> None:
    parent = PolicyPortfolio.flagship5()
    source = render_policy_source(parent)

    with pytest.raises(PolicyError):
        parent.add_combined(parent.combined[0])
    with pytest.raises(PolicyError):
        parent.replace("missing_policy", _variable_combined("missing_policy"))
    with pytest.raises(PolicyError):
        parent.fork_combined("missing_policy", _variable_combined("forked"))
    with pytest.raises(PolicyError):
        parent.fork_combined(
            parent.combined[0].name,
            _variable_combined(parent.combined[1].name),
        )
    with pytest.raises(PolicyError):
        parent.remove_combined("timesfm_2_5")

    assert render_policy_source(parent) == source


def test_atomic_remove_combined_refuses_to_remove_final_combined() -> None:
    base = PolicyPortfolio.flagship5()
    parent = PolicyPortfolio(base.tsfm, (_variable_combined(),))

    with pytest.raises(PolicyError, match="final Combined"):
        parent.remove_combined(parent.combined[0].name)


def test_validate_parents_rejects_combined_name_colliding_with_statistical_leaf() -> None:
    base = PolicyPortfolio.flagship5()
    policy = CombinedPolicy(
        name="seasonal_naive",
        parents=("toto_2_0", "holt_damped_trend"),
        operator="weighted_mean",
        weights=(0.5, 0.5),
        fallback_parent="toto_2_0",
    )
    portfolio = PolicyPortfolio(base.tsfm, (policy,))
    source = render_policy_source(portfolio)

    with pytest.raises(PolicyError, match="collid"):
        portfolio.validate_parents(_module().names())

    assert render_policy_source(portfolio) == source


def _operator_task() -> Task:
    return Task(
        "operator",
        (0.0, 0.0, 0.0, 10.0),
        2,
        "1 day",
        (20.0, 20.0),
    )


def _successful_parent(name: str, forecast: tuple[float, ...]) -> Outcome:
    return Outcome(name, "operator", SUCCESS, forecast=forecast)


def _operator_outcomes(*outcomes: Outcome) -> dict[tuple[str, str], Outcome]:
    return {(outcome.method, outcome.task_id): outcome for outcome in outcomes}


def test_weighted_mean_combines_two_tsfm_parent_forecasts_pointwise() -> None:
    task = _operator_task()
    policy = CombinedPolicy(
        "combined_two_tsfm_weighted",
        ("toto_2_0", "timesfm_2_5"),
        "weighted_mean",
        (0.25, 0.75),
        fallback_parent="toto_2_0",
    )

    combined = _run_combined(
        policy,
        task,
        _operator_outcomes(
            _successful_parent("toto_2_0", (10.0, 10.0)),
            _successful_parent("timesfm_2_5", (20.0, 20.0)),
        ),
    )

    assert combined.status == SUCCESS
    assert combined.forecast == (17.5, 17.5)


def test_lead_time_route_uses_ordered_parents_for_contiguous_horizon_segments() -> None:
    policy = CombinedPolicy(
        "combined_lead_time_route",
        ("chronos_bolt", "timesfm_2_5", "seasonal_naive"),
        "lead_time_route",
        (0.25, 0.50, 0.25),
        fallback_parent="timesfm_2_5",
    )
    horizon = 24

    combined = combine_materialized_outcome(
        policy,
        {
            "chronos_bolt": _successful_parent(
                "chronos_bolt", tuple(100.0 + step for step in range(horizon))
            ),
            "timesfm_2_5": _successful_parent(
                "timesfm_2_5", tuple(200.0 + step for step in range(horizon))
            ),
            "seasonal_naive": _successful_parent(
                "seasonal_naive", tuple(300.0 + step for step in range(horizon))
            ),
        },
        task_id="operator",
        history=(1.0, 2.0),
        horizon=horizon,
        frequency="D",
    )

    assert combined.status == SUCCESS
    assert combined.forecast == (
        tuple(100.0 + step for step in range(6))
        + tuple(200.0 + step for step in range(6, 18))
        + tuple(300.0 + step for step in range(18, 24))
    )


def test_lead_time_route_duration_shares_apply_to_the_complete_horizon() -> None:
    policy = CombinedPolicy(
        "combined_short_tail_route",
        ("timesfm_2_5", "seasonal_naive"),
        "lead_time_route",
        (0.90, 0.10),
        fallback_parent="timesfm_2_5",
    )

    combined = combine_materialized_outcome(
        policy,
        {
            "timesfm_2_5": _successful_parent("timesfm_2_5", (10.0,) * 10),
            "seasonal_naive": _successful_parent("seasonal_naive", (20.0,) * 10),
        },
        task_id="operator",
        history=(1.0, 2.0),
        horizon=10,
        frequency="D",
    )

    assert combined.forecast == (10.0,) * 9 + (20.0,)


@pytest.mark.parametrize(
    ("weights", "message"),
    (
        ((0.5, 0.5), "match parents"),
        ((0.25, 0.75, 0.0), "strictly positive"),
        ((0.25, 0.50, 0.30), "sum to one"),
    ),
)
def test_lead_time_route_rejects_invalid_segment_proportions(
    weights: tuple[float, ...], message: str
) -> None:
    with pytest.raises(PolicyError, match=message):
        CombinedPolicy(
            "combined_lead_time_route",
            ("chronos_bolt", "timesfm_2_5", "seasonal_naive"),
            "lead_time_route",
            weights,
            fallback_parent="timesfm_2_5",
        )


def test_lead_time_route_falls_back_when_horizon_cannot_give_each_parent_a_segment() -> None:
    policy = CombinedPolicy(
        "combined_lead_time_route",
        ("chronos_bolt", "timesfm_2_5", "seasonal_naive"),
        "lead_time_route",
        (0.25, 0.50, 0.25),
        fallback_parent="timesfm_2_5",
    )

    combined = combine_materialized_outcome(
        policy,
        {
            "chronos_bolt": _successful_parent("chronos_bolt", (10.0, 11.0)),
            "timesfm_2_5": _successful_parent("timesfm_2_5", (20.0, 21.0)),
            "seasonal_naive": _successful_parent("seasonal_naive", (30.0, 31.0)),
        },
        task_id="operator",
        history=(1.0, 2.0),
        horizon=2,
        frequency="D",
    )

    assert combined.status == SUCCESS
    assert combined.forecast == (20.0, 21.0)
    assert combined.detail.startswith("fallback=timesfm_2_5")


def test_median_combines_three_tsfm_and_statistical_parent_forecasts_pointwise() -> None:
    task = _operator_task()
    policy = CombinedPolicy(
        "combined_three_parent_median",
        ("toto_2_0", "timesfm_2_5", "seasonal_naive"),
        "median",
        fallback_parent="toto_2_0",
    )

    combined = _run_combined(
        policy,
        task,
        _operator_outcomes(
            _successful_parent("toto_2_0", (10.0, 10.0)),
            _successful_parent("timesfm_2_5", (20.0, 20.0)),
            _successful_parent("seasonal_naive", (100.0, 100.0)),
        ),
    )

    assert combined.forecast == (20.0, 20.0)


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        ((sys.float_info.max, sys.float_info.max), sys.float_info.max),
        ((-sys.float_info.max, sys.float_info.max), 0.0),
        (
            (
                sys.float_info.max,
                sys.float_info.max,
                sys.float_info.max,
                sys.float_info.max,
            ),
            sys.float_info.max,
        ),
        (
            (
                -sys.float_info.max,
                -sys.float_info.max,
                sys.float_info.max,
                sys.float_info.max,
            ),
            0.0,
        ),
    ),
)
def test_even_median_is_finite_for_extreme_values(
    values: tuple[float, ...], expected: float
) -> None:
    parent_names = (
        "toto_2_0",
        "timesfm_2_5",
        "chronos_bolt",
        "granite_ttm_r2",
    )[: len(values)]
    policy = CombinedPolicy(
        "combined_even_extreme_median",
        parent_names,
        "median",
        fallback_parent="toto_2_0",
    )

    combined = combine_materialized_outcome(
        policy,
        {
            name: _successful_parent(name, (value,))
            for name, value in zip(parent_names, values, strict=True)
        },
        task_id="operator",
        history=(1.0, 2.0),
        horizon=1,
        frequency="D",
    )

    assert combined.status == SUCCESS
    assert combined.forecast == (expected,)
    assert combined.detail == ""


def test_trimmed_mean_removes_one_low_and_one_high_parent_pointwise() -> None:
    task = _operator_task()
    policy = CombinedPolicy(
        "combined_trimmed",
        ("toto_2_0", "timesfm_2_5", "chronos_bolt", "granite_ttm_r2", "seasonal_naive"),
        "trimmed_mean",
        fallback_parent="toto_2_0",
    )

    combined = _run_combined(
        policy,
        task,
        _operator_outcomes(
            _successful_parent("toto_2_0", (0.0, 0.0)),
            _successful_parent("timesfm_2_5", (10.0, 10.0)),
            _successful_parent("chronos_bolt", (20.0, 20.0)),
            _successful_parent("granite_ttm_r2", (30.0, 30.0)),
            _successful_parent("seasonal_naive", (100.0, 100.0)),
        ),
    )

    assert combined.forecast == (20.0, 20.0)


def test_route_uses_a_history_only_signal_to_select_explicit_parent() -> None:
    task = _operator_task()
    policy = CombinedPolicy(
        "combined_explicit_route",
        ("toto_2_0", "seasonal_naive"),
        "route",
        signal="zero_fraction",
        threshold=0.5,
        above_parent="seasonal_naive",
        below_parent="toto_2_0",
        fallback_parent="toto_2_0",
    )

    combined = _run_combined(
        policy,
        task,
        _operator_outcomes(
            _successful_parent("toto_2_0", (10.0, 10.0)),
            _successful_parent("seasonal_naive", (20.0, 20.0)),
        ),
    )

    assert combined.forecast == (20.0, 20.0)


@pytest.mark.parametrize(
    ("signal", "threshold"),
    (
        ("noise_relative_scale", 2.0),
        ("intermittency_adi", 2.5),
        ("history_length", 7.0),
        ("horizon", 1.0),
        ("horizon_ratio", 0.2),
    ),
)
def test_route_supports_extended_reviewed_python_signals(
    signal: str, threshold: float
) -> None:
    """Missing host-side signal computation would reject or misroute valid policies."""
    policy = CombinedPolicy(
        "combined_morphology_route",
        ("toto_2_0", "seasonal_naive"),
        "route",
        signal=signal,
        threshold=threshold,
        above_parent="seasonal_naive",
        below_parent="toto_2_0",
        fallback_parent="toto_2_0",
    )

    combined = combine_materialized_outcome(
        policy,
        {
            "toto_2_0": Outcome("toto_2_0", "operator", SUCCESS, forecast=(10.0, 10.0)),
            "seasonal_naive": Outcome(
                "seasonal_naive", "operator", SUCCESS, forecast=(20.0, 20.0)
            ),
        },
        task_id="operator",
        history=(0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0),
        horizon=2,
        frequency="1 day",
    )

    assert combined.status == SUCCESS
    assert combined.forecast == (20.0, 20.0)


def test_failed_nonfallback_parent_returns_successful_explicit_fallback() -> None:
    task = _operator_task()
    policy = CombinedPolicy(
        "combined_explicit_fallback",
        ("toto_2_0", "timesfm_2_5"),
        "weighted_mean",
        (0.5, 0.5),
        fallback_parent="toto_2_0",
    )

    combined = _run_combined(
        policy,
        task,
        _operator_outcomes(
            _successful_parent("toto_2_0", (10.0, 10.0)),
            Outcome("timesfm_2_5", "operator", INVALID, detail="wrong horizon"),
        ),
    )

    assert combined.status == SUCCESS
    assert combined.forecast == (10.0, 10.0)
    assert "fallback=toto_2_0" in combined.detail


def test_failed_fallback_returns_strongest_failure_without_a_forecast() -> None:
    task = _operator_task()
    policy = CombinedPolicy(
        "combined_failed_fallback",
        ("toto_2_0", "timesfm_2_5", "seasonal_naive"),
        "median",
        fallback_parent="toto_2_0",
    )

    combined = _run_combined(
        policy,
        task,
        _operator_outcomes(
            Outcome("toto_2_0", "operator", CRASHED, detail="runtime unavailable"),
            Outcome("timesfm_2_5", "operator", INVALID, detail="wrong horizon"),
            Outcome("seasonal_naive", "operator", NOT_APPLICABLE, detail="not seasonal"),
        ),
    )

    assert combined.status == CRASHED
    assert combined.forecast == ()


@pytest.mark.parametrize(
    ("policy", "outcomes", "expect_fallback"),
    [
        (
            CombinedPolicy(
                "combined_extreme_weighted",
                ("toto_2_0", "timesfm_2_5"),
                "weighted_mean",
                (0.5, 0.5),
                fallback_parent="toto_2_0",
            ),
            (
                _successful_parent("toto_2_0", (sys.float_info.max,)),
                _successful_parent("timesfm_2_5", (sys.float_info.max,)),
            ),
            False,
        ),
        (
            CombinedPolicy(
                "combined_extreme_median",
                ("toto_2_0", "timesfm_2_5", "seasonal_naive"),
                "median",
                fallback_parent="toto_2_0",
            ),
            (
                _successful_parent("toto_2_0", (sys.float_info.max,)),
                _successful_parent("timesfm_2_5", (sys.float_info.max,)),
                _successful_parent("seasonal_naive", (sys.float_info.max,)),
            ),
            False,
        ),
        (
            CombinedPolicy(
                "combined_extreme_trimmed",
                (
                    "toto_2_0",
                    "timesfm_2_5",
                    "chronos_bolt",
                    "granite_ttm_r2",
                    "seasonal_naive",
                ),
                "trimmed_mean",
                fallback_parent="toto_2_0",
            ),
            (
                _successful_parent("toto_2_0", (0.0,)),
                _successful_parent("timesfm_2_5", (sys.float_info.max,)),
                _successful_parent("chronos_bolt", (sys.float_info.max,)),
                _successful_parent("granite_ttm_r2", (sys.float_info.max,)),
                _successful_parent("seasonal_naive", (sys.float_info.max,)),
            ),
            False,
        ),
        (
            CombinedPolicy(
                "combined_extreme_route",
                ("toto_2_0", "timesfm_2_5"),
                "route",
                signal="zero_fraction",
                threshold=0.5,
                above_parent="timesfm_2_5",
                below_parent="toto_2_0",
                fallback_parent="toto_2_0",
            ),
            (
                _successful_parent("toto_2_0", (sys.float_info.max,)),
                _successful_parent("timesfm_2_5", (sys.float_info.max,)),
            ),
            True,
        ),
    ],
)
def test_extreme_combined_operators_never_escape_arithmetic(
    policy: CombinedPolicy,
    outcomes: tuple[Outcome, ...],
    expect_fallback: bool,
) -> None:
    task = Task(
        "operator",
        (sys.float_info.max,) * 4,
        1,
        "1 day",
        (sys.float_info.max,),
    )
    materialized = {outcome.method: outcome for outcome in outcomes}

    composed = combine_materialized_outcome(
        policy,
        materialized,
        task_id=task.task_id,
        history=task.history,
        horizon=task.horizon,
        frequency=task.frequency,
    )
    combined = _run_combined(policy, task, _operator_outcomes(*outcomes))

    assert composed.status == SUCCESS
    assert combined.status == SUCCESS
    assert composed.forecast == (sys.float_info.max,)
    assert combined.forecast == (sys.float_info.max,)
    assert ("fallback=" in composed.detail) is expect_fallback
    assert ("fallback=" in combined.detail) is expect_fallback


def test_arithmetic_fallback_for_nonfinite_weighted_composition() -> None:
    policy = CombinedPolicy(
        "combined_nonfinite_weighted",
        ("toto_2_0", "timesfm_2_5"),
        "weighted_mean",
        (0.50000000025, 0.50000000025),
        fallback_parent="toto_2_0",
    )
    task = Task(
        "operator",
        (sys.float_info.max,) * 4,
        1,
        "1 day",
        (sys.float_info.max,),
    )
    outcomes = (
        _successful_parent("toto_2_0", (sys.float_info.max,)),
        _successful_parent("timesfm_2_5", (sys.float_info.max,)),
    )

    composed = combine_materialized_outcome(
        policy,
        {outcome.method: outcome for outcome in outcomes},
        task_id=task.task_id,
        history=task.history,
        horizon=task.horizon,
        frequency=task.frequency,
    )
    combined = _run_combined(policy, task, _operator_outcomes(*outcomes))

    assert composed.status == SUCCESS
    assert combined.status == SUCCESS
    assert composed.forecast == (sys.float_info.max,)
    assert combined.forecast == (sys.float_info.max,)
    assert "fallback=toto_2_0" in composed.detail
    assert "fallback=toto_2_0" in combined.detail


@pytest.mark.parametrize(
    "weights",
    [
        (0.25, 0.25, 0.5),
        (0.250000000125, 0.250000000125, 0.50000000025),
    ],
)
def test_weighted_cancellation_preserves_subnormal_residual(
    weights: tuple[float, float, float],
) -> None:
    policy = CombinedPolicy(
        "combined_weighted_cancellation",
        ("toto_2_0", "timesfm_2_5", "seasonal_naive"),
        "weighted_mean",
        weights,
        fallback_parent="toto_2_0",
    )
    task = Task("operator", (1.0, 1.0), 1, "1 day", (5e-324,))
    outcomes = (
        _successful_parent("toto_2_0", (sys.float_info.max,)),
        _successful_parent("timesfm_2_5", (-sys.float_info.max,)),
        _successful_parent("seasonal_naive", (1e-323,)),
    )

    composed = combine_materialized_outcome(
        policy,
        {outcome.method: outcome for outcome in outcomes},
        task_id=task.task_id,
        history=task.history,
        horizon=task.horizon,
        frequency=task.frequency,
    )
    combined = _run_combined(policy, task, _operator_outcomes(*outcomes))

    assert composed.status == SUCCESS
    assert combined.status == SUCCESS
    assert composed.forecast == (5e-324,)
    assert combined.forecast == (5e-324,)


def test_trimmed_cancellation_preserves_subnormal_residual() -> None:
    policy = CombinedPolicy(
        "combined_trimmed_cancellation",
        (
            "toto_2_0",
            "timesfm_2_5",
            "chronos_bolt",
            "granite_ttm_r2",
            "seasonal_naive",
        ),
        "trimmed_mean",
        fallback_parent="toto_2_0",
    )
    task = Task("operator", (1.0, 1.0), 1, "1 day", (5e-324,))
    outcomes = (
        _successful_parent("toto_2_0", (-sys.float_info.max,)),
        _successful_parent("timesfm_2_5", (-sys.float_info.max,)),
        _successful_parent("chronos_bolt", (1e-323,)),
        _successful_parent("granite_ttm_r2", (sys.float_info.max,)),
        _successful_parent("seasonal_naive", (sys.float_info.max,)),
    )

    composed = combine_materialized_outcome(
        policy,
        {outcome.method: outcome for outcome in outcomes},
        task_id=task.task_id,
        history=task.history,
        horizon=task.horizon,
        frequency=task.frequency,
    )
    combined = _run_combined(policy, task, _operator_outcomes(*outcomes))

    assert composed.status == SUCCESS
    assert combined.status == SUCCESS
    assert composed.forecast == (5e-324,)
    assert combined.forecast == (5e-324,)


def test_invalid_fallback_returns_sanitized_invalid_outcome() -> None:
    policy = CombinedPolicy(
        "combined_invalid_fallback",
        ("toto_2_0", "timesfm_2_5"),
        "weighted_mean",
        (0.5, 0.5),
        fallback_parent="toto_2_0",
    )

    combined = combine_materialized_outcome(
        policy,
        {
            "toto_2_0": Outcome(
                "toto_2_0", "operator", INVALID, detail="sensitive runtime detail"
            ),
            "timesfm_2_5": _successful_parent("timesfm_2_5", (20.0, 20.0)),
        },
        task_id="operator",
        history=(10.0, 10.0),
        horizon=2,
        frequency="1 day",
    )

    assert combined.status == INVALID
    assert combined.forecast == ()
    assert combined.detail == "toto_2_0=invalid"


def test_tsfm_runtime_executes_once_when_multiple_combined_policies_consume_it(
    tmp_path: Path,
) -> None:
    base = PolicyPortfolio.flagship5()
    portfolio = PolicyPortfolio(
        base.tsfm,
        (
            CombinedPolicy(
                "combined_tsfm_pair_mean",
                ("toto_2_0", "timesfm_2_5"),
                "weighted_mean",
                (0.5, 0.5),
                fallback_parent="toto_2_0",
            ),
            CombinedPolicy(
                "combined_tsfm_stat_median",
                ("toto_2_0", "seasonal_naive"),
                "median",
                fallback_parent="toto_2_0",
            ),
        ),
    )
    runtime = FakeTSFMRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})

    outcomes = evaluate_portfolio(
        _module(),
        portfolio,
        _tasks()[:1],
        outcome_cache=OutcomeCache(tmp_path / "cache"),
        runtimes=_registry(runtime),
        isolated_methods=False,
    )

    assert all(outcome.status == SUCCESS for outcome in outcomes)
    assert len(runtime.calls) == len(portfolio.tsfm)


def test_successful_tsfm_and_combined_outcomes_record_scaled_metrics(
    tmp_path: Path,
) -> None:
    policy_cache = PolicyOutcomeCache(tmp_path / "policy-cache")
    outcomes = evaluate_portfolio(
        _module(),
        _portfolio(),
        _tasks()[:1],
        outcome_cache=OutcomeCache(tmp_path / "cache"),
        runtimes=_registry(FakeTSFMRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})),
        isolated_methods=False,
        policy_cache=policy_cache,
    )
    cached = evaluate_portfolio(
        _module(),
        _portfolio(),
        _tasks()[:1],
        outcome_cache=OutcomeCache(tmp_path / "cache"),
        runtimes=_registry(FakeTSFMRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})),
        isolated_methods=False,
        policy_cache=policy_cache,
    )

    for outcome in outcomes + cached:
        if outcome.status == SUCCESS:
            assert outcome.smae is not None and outcome.srmse is not None
            assert outcome.smae_raw is not None and outcome.srmse_raw is not None
            assert outcome.smae_clipped is not None and outcome.srmse_clipped is not None


def test_forecast_tsfm_is_label_free_and_rejects_invalid_runtime_output() -> None:
    class InvalidRuntime(FakeTSFMRuntime):
        def forecast(self, candidate, history, horizon, frequency):
            del candidate, history, horizon, frequency
            return ()

    assert hasattr(portfolio_module, "forecast_tsfm")
    assert hasattr(portfolio_module, "InvalidTSFMForecastError")
    with pytest.raises(portfolio_module.InvalidTSFMForecastError, match="invalid forecast"):
        portfolio_module.forecast_tsfm(
            _portfolio().tsfm[0],
            history=_tasks()[0].history,
            horizon=_tasks()[0].horizon,
            frequency=_tasks()[0].frequency,
            runtimes=_registry(
                InvalidRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})
            ),
        )


def test_scored_tsfm_preserves_invalid_status_for_wrong_length_output() -> None:
    class InvalidRuntime(FakeTSFMRuntime):
        def forecast(self, candidate, history, horizon, frequency):
            del candidate, history, horizon, frequency
            return ()

    outcome = _run_tsfm(
        _portfolio().tsfm[0],
        _tasks()[0],
        _registry(InvalidRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})),
    )

    assert outcome.status == INVALID
    assert "invalid forecast" in outcome.detail


def test_evaluate_portfolio_rejects_duplicate_task_ids_before_execution(tmp_path: Path) -> None:
    task = _tasks()[0]

    with pytest.raises(ValueError, match="duplicate task IDs"):
        evaluate_portfolio(
            _module(),
            _portfolio(),
            (task, task),
            outcome_cache=OutcomeCache(tmp_path / "cache"),
            runtimes=_registry(
                FakeTSFMRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})
            ),
            isolated_methods=False,
        )


def test_evaluate_portfolio_rejects_statistical_tsfm_name_collision_before_execution(
    tmp_path: Path,
) -> None:
    module = parse_module(
        MODULE_HEADER
        + '''

def seasonal_naive(history, horizon, frequency):
    """Satisfy the reviewed Combined parent contract."""
    return [0.0] * horizon

def holt_damped_trend(history, horizon, frequency):
    """Satisfy the reviewed Combined parent contract."""
    return [0.0] * horizon

def croston_sba(history, horizon, frequency):
    """Satisfy the reviewed Combined parent contract."""
    return [0.0] * horizon

def robust_loess_trend(history, horizon, frequency):
    """Satisfy the reviewed Combined parent contract."""
    return [0.0] * horizon

def median_seasonal_profile_forecast(history, horizon, frequency):
    """Satisfy the reviewed Combined parent contract."""
    return [0.0] * horizon

def timesfm_2_5(history, horizon, frequency):
    """An invalid statistical name collision fixture."""
    return [0.0] * horizon
'''
    )
    runtime = FakeTSFMRuntime({method_id: 1.0 for method_id in FLAGSHIP_METHOD_IDS})
    cache = OutcomeCache(tmp_path / "cache")

    with pytest.raises(PolicyError, match="namespace.*timesfm_2_5"):
        evaluate_portfolio(
            module,
            _portfolio(),
            _tasks()[:1],
            outcome_cache=cache,
            runtimes=_registry(runtime),
            isolated_methods=False,
        )

    assert cache.stats.hits == 0
    assert cache.stats.misses == 0
    assert runtime.calls == []
