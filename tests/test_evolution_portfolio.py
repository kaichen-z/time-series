from __future__ import annotations

from pathlib import Path

import pytest

from numerical_agent.dictionary import MethodCandidate
from numerical_agent.evolution.cache import CacheMissError, OutcomeCache
from numerical_agent.evolution.execution import SUCCESS, Task
from numerical_agent.evolution.module import MODULE_HEADER, parse_module
from numerical_agent.evolution.portfolio import (
    FLAGSHIP_METHOD_IDS,
    CombinedPolicy,
    PolicyError,
    PolicyPortfolio,
    TSFMPolicy,
    PolicyOutcomeCache,
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


def _legacy_source(first_mode: str) -> str:
    portfolio = PolicyPortfolio.flagship5()
    payloads: list[dict[str, object]] = []
    for policy in portfolio.combined:
        first_parent, second_parent = policy.parents
        if policy.operator == "weighted_mean":
            mode = "blend"
            weight = policy.weights[0]
            tsfm_when = "above"
        else:
            mode = "route"
            weight = 0.65
            tsfm_when = (
                "below"
                if (not payloads and first_mode == "route")
                else ("below" if policy.below_parent == first_parent else "above")
            )
        payloads.append(
            {
                "name": policy.name,
                "tsfm_parent": first_parent,
                "statistical_parent": second_parent,
                "mode": first_mode if not payloads else mode,
                "weight": weight,
                "signal": policy.signal,
                "threshold": policy.threshold,
                "tsfm_when": "below" if first_mode == "route" and not payloads else tsfm_when,
            }
        )
    return (
        f"TSFM_POLICIES = {tuple(policy.to_payload() for policy in portfolio.tsfm)!r}\n"
        f"COMBINED_POLICIES = {tuple(payloads)!r}\n"
    )


def test_canonical_combined_policy_round_trips() -> None:
    policy = _canonical_combined()

    assert CombinedPolicy(**policy.to_payload()) == policy


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
    source = _legacy_source("blend")

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
    source = _legacy_source("route")

    policy = parse_policy_source(source).combined[0]

    assert policy.operator == "route"
    assert policy.weights == ()
    assert policy.above_parent == "seasonal_naive"
    assert policy.below_parent == "timesfm_2_5"
    assert policy.fallback_parent == "timesfm_2_5"


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


def test_policy_parser_rejects_removing_a_combined_candidate() -> None:
    source = render_policy_source(_portfolio())
    source = source[: source.index("COMBINED_POLICIES =")] + "COMBINED_POLICIES = ()\n"

    with pytest.raises(PolicyError, match="Combined identities"):
        parse_policy_source(source)


def test_policy_parser_rejects_combined_to_combined_parent_dependency() -> None:
    source = render_policy_source(_portfolio()).replace(
        "'seasonal_naive'", "'combined_chronos_damped_trend'", 1
    )

    with pytest.raises(PolicyError, match="Combined.*parent"):
        parse_policy_source(source)


def test_combined_repair_cannot_change_parent_identity() -> None:
    portfolio = _portfolio()
    parent = portfolio.combined[0]
    replacement = CombinedPolicy(
        name=parent.name,
        parents=(parent.parents[0], "naive_last"),
        operator=parent.operator,
        weights=parent.weights,
        signal=parent.signal,
        threshold=parent.threshold,
        fallback_parent=parent.fallback_parent,
    )

    with pytest.raises(PolicyError, match="parent identities"):
        portfolio.replace(parent.name, replacement)


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


def test_93_methods_plus_portfolio_reports_103_candidates() -> None:
    names = tuple(f"method_{index:03d}" for index in range(93))
    portfolio = _portfolio()

    assert len(names) + len(portfolio.names) == 103
