"""Tests for the bounded typed Combined-policy proposal boundary."""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import pytest

from common.llm import FakeLLMClient
from numerical_agent.evolution.portfolio import CombinedPolicy, PolicyPortfolio, TSFMPolicy


def test_combined_evolution_module_exposes_typed_boundary() -> None:
    """Removing the adapter module makes the Agent proposal boundary unavailable."""
    from numerical_agent.evolution.combined_evolution import (  # noqa: PLC0415
        COMBINED_EVOLUTION_SYSTEM,
        apply_combined_operations,
        parse_combined_operations,
        propose_combined_child,
    )

    assert COMBINED_EVOLUTION_SYSTEM
    assert callable(parse_combined_operations)
    assert callable(apply_combined_operations)
    assert callable(propose_combined_child)


def _policy(name: str, *, parents: list[str] | None = None) -> dict[str, object]:
    return {
        "name": name,
        "parents": parents or ["toto_2_0", "timesfm_2_5", "chronos_bolt"],
        "operator": "median",
        "weights": [],
        "signal": "periodicity_strength",
        "threshold": 0.0,
        "above_parent": "",
        "below_parent": "",
        "fallback_parent": "toto_2_0",
    }


def _response(*operations: dict[str, object]) -> str:
    return json.dumps({"operations": list(operations)})


def _diagnostics():
    from numerical_agent.evolution.combined_evolution import CombinedProposalDiagnostics

    return CombinedProposalDiagnostics(
        history_length=128,
        forecast_disagreement=0.25,
        successful_leaf_count=4,
        unavailable_leaf_count=1,
    )


class _SentinelPortfolio(PolicyPortfolio):
    """A valid subclass whose serializer must never be trusted at this boundary."""

    def to_payload(self) -> dict[str, object]:
        return {"sentinel": "PORTFOLIO_SERIALIZER_MUST_NOT_RUN"}


class _SentinelCombinedPolicy(CombinedPolicy):
    """A valid subclass whose serializer must never be trusted at this boundary."""

    def to_payload(self) -> dict[str, object]:
        return {"sentinel": "COMBINED_SERIALIZER_MUST_NOT_RUN"}


class _SentinelTSFMPolicy(TSFMPolicy):
    """A valid subclass whose serializer must never be trusted at this boundary."""

    def to_payload(self) -> dict[str, object]:
        return {"sentinel": "TSFM_SERIALIZER_MUST_NOT_RUN"}


@pytest.mark.parametrize(
    "values",
    (
        (True, 0.25, 4, 1),
        (0, 0.25, 4, 1),
        (1_000_001, 0.25, 4, 1),
        (128, True, 4, 1),
        (128, 1, 4, 1),
        (128, float("nan"), 4, 1),
        (128, float("inf"), 4, 1),
        (128, -0.01, 4, 1),
        (128, 1_000_000.01, 4, 1),
        (128, 0.25, True, 1),
        (128, 0.25, -1, 1),
        (128, 0.25, 1_000_001, 1),
        (128, 0.25, 4, True),
        (128, 0.25, 4, -1),
        (128, 0.25, 4, 1_000_001),
    ),
)
def test_diagnostic_contract_rejects_noncanonical_or_unbounded_values(
    values: tuple[object, object, object, object],
) -> None:
    """Weak or deferred diagnostics validation could reach the untrusted prompt."""
    from numerical_agent.evolution.combined_evolution import (  # noqa: PLC0415
        CombinedEvolutionError,
        CombinedProposalDiagnostics,
    )

    with pytest.raises(CombinedEvolutionError):
        CombinedProposalDiagnostics(*values)  # type: ignore[arg-type]


def test_diagnostic_contract_accepts_only_exact_builtins_at_documented_caps() -> None:
    """Accepting numeric subclasses would reintroduce polymorphic prompt values."""
    from numerical_agent.evolution.combined_evolution import (  # noqa: PLC0415
        CombinedEvolutionError,
        CombinedProposalDiagnostics,
    )

    class IntSubclass(int):
        pass

    class FloatSubclass(float):
        pass

    valid = CombinedProposalDiagnostics(1_000_000, 1_000_000.0, 1_000_000, 0)
    assert valid.to_payload() == {
        "forecast_disagreement": 1_000_000.0,
        "history_length": 1_000_000,
        "successful_leaf_count": 1_000_000,
        "unavailable_leaf_count": 0,
    }

    for values in (
        (IntSubclass(128), 0.25, 4, 1),
        (128, FloatSubclass(0.25), 4, 1),
        (128, 0.25, IntSubclass(4), 1),
        (128, 0.25, 4, IntSubclass(1)),
    ):
        with pytest.raises(CombinedEvolutionError):
            CombinedProposalDiagnostics(*values)  # type: ignore[arg-type]


def test_parse_combined_operations_accepts_each_exact_operation_schema() -> None:
    """Dropping an operation type would reject a valid typed portfolio edit."""
    from numerical_agent.evolution.combined_evolution import parse_combined_operations

    operations = parse_combined_operations(
        _response(
            {"op": "add", "reason": "robust centre", "policy": _policy("combined_added")},
            {
                "op": "repair",
                "target": "combined_timesfm_seasonal",
                "reason": "change leaves",
                "policy": _policy("combined_timesfm_seasonal", parents=["toto_2_0", "seasonal_naive"]),
            },
            {
                "op": "fork",
                "source": "combined_moirai_croston_router",
                "reason": "preserve source and explore median",
                "policy": _policy("combined_fork"),
            },
            {"op": "remove", "target": "combined_chronos_damped_trend", "reason": "retire duplicate"},
        )
    )

    assert tuple(operation.op for operation in operations) == ("add", "repair", "fork", "remove")
    assert operations[0].policy is not None
    assert operations[0].policy.to_payload() == {
        **_policy("combined_added"),
        "parents": ("toto_2_0", "timesfm_2_5", "chronos_bolt"),
        "weights": (),
    }


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        json.dumps({"operations": [] , "extra": True}),
        _response({"op": "unknown", "reason": "x", "policy": _policy("combined_x")}),
        _response({"op": "add", "reason": "x", "target": "combined_x", "policy": _policy("combined_x")}),
        _response({"op": "remove", "target": "combined_x", "reason": "x", "policy": _policy("combined_x")}),
        _response({"op": "add", "reason": "x", "policy": {**_policy("combined_x"), "checkpoint": "other"}}),
        _response({"op": "add", "reason": "x", "policy": {**_policy("combined_x"), "scorer": "other"}}),
        _response({"op": "add", "reason": "x", "policy": {**_policy("combined_x"), "parents": ["toto_2_0", "toto_2_0"]}}),
        _response({"op": "repair", "target": "combined_x", "reason": "x", "policy": _policy("combined_y")}),
        _response(
            {"op": "add", "reason": "x", "policy": _policy("combined_x")},
            {"op": "add", "reason": "x", "policy": _policy("combined_x")},
        ),
        _response(*({"op": "add", "reason": "x", "policy": _policy(f"combined_{index}")} for index in range(9))),
    ],
)
def test_parse_combined_operations_rejects_noncanonical_or_unsafe_batches(response: str) -> None:
    """Relaxing exact schemas could admit model-binding or scoring mutations."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        parse_combined_operations,
    )

    with pytest.raises(CombinedEvolutionError):
        parse_combined_operations(response)


@pytest.mark.parametrize(
    "response",
    [
        '{"operations": [], "operations": []}',
        '{"operations": [{"op": "remove", "op": "add", "reason": "x", '
        '"policy": {"name": "combined_x", "parents": ["toto_2_0", "timesfm_2_5"], '
        '"operator": "median", "weights": [], "signal": "periodicity_strength", '
        '"threshold": 0.0, "above_parent": "", "below_parent": "", '
        '"fallback_parent": "toto_2_0"}}]}',
        '{"operations": [{"op": "add", "reason": "x", "policy": '
        '{"name": "combined_x", "name": "combined_x", '
        '"parents": ["toto_2_0", "timesfm_2_5"], "operator": "median", '
        '"weights": [], "signal": "periodicity_strength", "threshold": 0.0, '
        '"above_parent": "", "below_parent": "", "fallback_parent": "toto_2_0"}}]}',
    ],
)
def test_parse_combined_operations_rejects_duplicate_json_keys_at_every_level(response: str) -> None:
    """Last-write-wins decoding would let a duplicate field bypass exact schemas."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        parse_combined_operations,
    )

    with pytest.raises(CombinedEvolutionError):
        parse_combined_operations(response)


@pytest.mark.parametrize(
    "response",
    (
        '{"operations": []}{"operations": []}',
        '{"operations": []} trailing',
        '```json\n{"operations": []}\n```\n```json\n{"operations": []}\n```',
    ),
)
def test_parse_combined_operations_rejects_ambiguous_or_trailing_objects(response: str) -> None:
    """Scanning for a later object would make a response boundary ambiguous."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        parse_combined_operations,
    )

    with pytest.raises(CombinedEvolutionError):
        parse_combined_operations(response)


def test_parse_combined_operations_accepts_wrapper_contract() -> None:
    """The permitted wrappers still carry exactly one complete JSON object."""
    from numerical_agent.evolution.combined_evolution import parse_combined_operations

    assert parse_combined_operations("{\"operations\": []}") == ()
    assert parse_combined_operations("<think>internal</think>{\"operations\": []}") == ()
    assert parse_combined_operations("```json\n{\"operations\": []}\n```") == ()


@pytest.mark.parametrize(
    "response",
    (
        "<think>internal</think>```json\n{\"operations\": []}\n```",
        "<think>one</think><think>two</think>{\"operations\": []}",
        "<think>unterminated{\"operations\": []}",
        "{\"operations\": []}</think>",
        "```json\n{\"operations\": []}\n```, trailing",
        "```json\n{\"operations\": []}\n",
        "{\"operations\": []} ```json",
    ),
)
def test_parse_combined_operations_rejects_wrapper_contract_ambiguity(response: str) -> None:
    """Stacked, unmatched, or trailing wrappers make the response boundary ambiguous."""
    from numerical_agent.evolution.combined_evolution import (  # noqa: PLC0415
        CombinedEvolutionError,
        parse_combined_operations,
    )

    with pytest.raises(CombinedEvolutionError):
        parse_combined_operations(response)


@pytest.mark.parametrize("literal", ("NaN", "Infinity", "-Infinity"))
def test_parse_combined_operations_rejects_nonfinite_json_constants(literal: str) -> None:
    """Accepting JSON constants would defer invalid numeric input past parsing."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        parse_combined_operations,
    )

    response = _response(
        {"op": "add", "reason": "x", "policy": _policy("combined_x")}
    ).replace("0.0", literal, 1)

    with pytest.raises(CombinedEvolutionError):
        parse_combined_operations(response)


@pytest.mark.parametrize(
    "policy",
    [
        {**_policy("combined_huge_threshold"), "threshold": 10**400},
        {
            **_policy("combined_huge_weight"),
            "operator": "weighted_mean",
            "weights": [10**400, 0.0, 0.0],
        },
        {**_policy("combined_bool_threshold"), "threshold": True},
        {
            **_policy("combined_bool_weight"),
            "operator": "weighted_mean",
            "weights": [True, 0.0, 0.0],
        },
    ],
)
def test_parse_combined_operations_sanitizes_invalid_numeric_literals(policy: dict[str, object]) -> None:
    """Numeric conversion failures must not escape the typed proposal boundary."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        parse_combined_operations,
    )

    with pytest.raises(CombinedEvolutionError):
        parse_combined_operations(_response({"op": "add", "reason": "x", "policy": policy}))


def test_apply_combined_operations_is_atomic_and_validates_final_namespace() -> None:
    """Applying a bad batch in place would corrupt the immutable Parent portfolio."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        apply_combined_operations,
        parse_combined_operations,
    )

    parent = PolicyPortfolio.flagship5()
    operations = parse_combined_operations(
        _response(
            {"op": "add", "reason": "add leaf", "policy": _policy("combined_added")},
            {
                "op": "repair",
                "target": "combined_timesfm_seasonal",
                "reason": "use reviewed statistical leaf",
                "policy": _policy("combined_timesfm_seasonal", parents=["toto_2_0", "seasonal_naive"]),
            },
            {
                "op": "fork",
                "source": "combined_moirai_croston_router",
                "reason": "source remains and child gets a new name",
                "policy": _policy("combined_fork"),
            },
            {"op": "remove", "target": "combined_chronos_damped_trend", "reason": "discard experiment"},
        )
    )

    child = apply_combined_operations(
        parent, operations, statistical_names=("seasonal_naive", "holt_damped_trend", "croston_sba", "robust_loess_trend", "median_seasonal_profile_forecast"),
    )

    assert child is not parent
    assert "combined_added" in child.names
    assert "combined_fork" in child.names
    assert "combined_chronos_damped_trend" not in child.names
    assert parent.names == PolicyPortfolio.flagship5().names

    invalid = parse_combined_operations(
        _response({"op": "add", "reason": "bad parent", "policy": _policy("combined_bad", parents=["toto_2_0", "unknown_leaf"])})
    )
    with pytest.raises(CombinedEvolutionError):
        apply_combined_operations(parent, invalid, statistical_names=("seasonal_naive",))
    assert parent is parent
    assert parent.names == PolicyPortfolio.flagship5().names


def test_apply_combined_operations_rejects_malformed_direct_operations_atomically() -> None:
    """An asserting direct operation must not escape the trusted mutation boundary."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        CombinedOperation,
        apply_combined_operations,
    )

    parent = PolicyPortfolio.flagship5()
    malformed = CombinedOperation("add", "missing required policy")

    with pytest.raises(CombinedEvolutionError):
        apply_combined_operations(
            parent,
            (malformed,),
            statistical_names=("seasonal_naive",),
        )

    assert parent is parent
    assert parent.names == PolicyPortfolio.flagship5().names


def test_propose_combined_child_sends_only_bounded_allowed_inputs() -> None:
    """Forwarding task labels or runtime details would violate the Agent boundary."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    parent = PolicyPortfolio.flagship5()
    response = _response(
        {"op": "add", "reason": "robust centre", "policy": _policy("combined_proposed")}
    )
    agent = FakeLLMClient([response])

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive", "holt_damped_trend", "croston_sba", "robust_loess_trend", "median_seasonal_profile_forecast"),
        diagnostics=_diagnostics(),
        agent=agent,
    )

    assert result.parent is parent
    assert result.changed is True
    assert "combined_proposed" in result.child.names
    assert result.rejection_reason == ""
    prompt = json.loads(agent.calls[0]["messages"][0]["content"])
    assert set(prompt) == {"allowed_operations", "current_policies", "diagnostics", "statistical_names", "tsfm_names"}
    assert prompt["current_policies"] == json.loads(
        json.dumps([policy.to_payload() for policy in parent.combined])
    )
    assert prompt["statistical_names"] == ["croston_sba", "holt_damped_trend", "median_seasonal_profile_forecast", "robust_loess_trend", "seasonal_naive"]
    assert prompt["diagnostics"] == _diagnostics().to_payload()
    assert prompt["tsfm_names"] == [policy.name for policy in parent.tsfm]


def test_propose_combined_child_rejects_nonpolymorphic_portfolio_records_before_llm() -> None:
    """Subclass serializers must not smuggle a payload across the trusted boundary."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    base = PolicyPortfolio.flagship5()
    sentinel_combined = _SentinelCombinedPolicy(**base.combined[0].to_payload())
    sentinel_tsfm = _SentinelTSFMPolicy(**base.tsfm[0].to_payload())
    parents = (
        _SentinelPortfolio(base.tsfm, base.combined),
        PolicyPortfolio(base.tsfm, (sentinel_combined, *base.combined[1:])),
        PolicyPortfolio((sentinel_tsfm, *base.tsfm[1:]), base.combined),
    )

    for parent in parents:
        agent = FakeLLMClient([_response()])
        result = propose_combined_child(
            parent,
            statistical_names=("seasonal_naive",),
            diagnostics=_diagnostics(),
            agent=agent,
        )

        assert result.parent is parent
        assert result.child is parent
        assert result.changed is False
        assert agent.calls == []


def test_propose_combined_child_rejects_diagnostic_dataclass_subclass_before_llm() -> None:
    """Only the trusted diagnostics dataclass may establish label-free aggregates."""
    from numerical_agent.evolution.combined_evolution import (  # noqa: PLC0415
        CombinedProposalDiagnostics,
        propose_combined_child,
    )

    @dataclass(frozen=True)
    class DiagnosticSubclass(CombinedProposalDiagnostics):
        pass

    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient([_response()])
    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics=DiagnosticSubclass(128, 0.25, 4, 1),
        agent=agent,
    )

    assert result.parent is parent
    assert result.child is parent
    assert agent.calls == []


def test_propose_combined_child_returns_exact_parent_with_sanitized_rejection() -> None:
    """Returning a partial child or echoing model text would leak and break atomicity."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient(["{not valid json: runtime_secret=do-not-echo}"])

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics=_diagnostics(),
        agent=agent,
    )

    assert result.parent is parent
    assert result.child is parent
    assert result.changed is False
    assert result.operations == ()
    assert result.rejection_reason
    assert "runtime_secret" not in result.rejection_reason
    assert "do-not-echo" not in result.rejection_reason


def test_propose_combined_child_bounds_deep_diagnostics_without_resetting_depth() -> None:
    """Nested mappings used to reset depth, allowing oversized prompt diagnostics."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    nested: dict[str, object] = {"future_values": [99], "leaf": "safe"}
    for index in range(20):
        nested = {f"layer_{index}": nested, f"side_{index}": "x" * 200}
    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient([_response()])

    result = propose_combined_child(
        parent,
        statistical_names=(
            "seasonal_naive",
            "holt_damped_trend",
            "croston_sba",
            "robust_loess_trend",
            "median_seasonal_profile_forecast",
        ),
        diagnostics={"aggregate": nested},
        agent=agent,
    )

    assert result.child is parent
    assert agent.calls == []


def test_propose_combined_child_drops_secret_strings_under_allowed_keys() -> None:
    """An allowed diagnostic key must not smuggle arbitrary secret-bearing text."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    secret = "SENTINEL_SECRET_DO_NOT_FORWARD"
    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient([_response()])

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics={"aggregate_note": secret, "mean_smape": 0.25},
        agent=agent,
    )

    assert result.child is parent
    assert agent.calls == []


def test_propose_combined_child_drops_oversized_diagnostics_before_traversal() -> None:
    """Oversized containers and strings must fail closed before sorting or scanning."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient([_response()])
    large_map = _LargeMapping()
    huge_string = "SENTINEL_" + "x" * 10_000

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics={"large": large_map, "note": huge_string},
        agent=agent,
    )

    assert result.child is parent
    assert agent.calls == []


def test_propose_combined_child_keeps_cyclic_diagnostics_bounded() -> None:
    """A self-referential mapping must not prevent a bounded proposal call."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient([_response()])

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics={"aggregate": cyclic},
        agent=agent,
    )

    assert result.child is parent
    assert agent.calls == []


def test_propose_combined_child_rejects_lying_length_mapping_before_iteration() -> None:
    """A custom map can lie about its size and must not reach sorting or the LLM."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient([_response()])

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics={"aggregate": _LyingLengthMapping()},
        agent=agent,
    )

    assert result.child is parent
    assert result.changed is False
    assert result.rejection_reason
    assert agent.calls == []


def test_propose_combined_child_rejects_infinite_yield_mapping_before_iteration() -> None:
    """An unbounded key iterator must be rejected without asking it for one key."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient([_response()])
    infinite = _InfiniteYieldMapping()

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics={"aggregate": infinite},
        agent=agent,
    )

    assert result.child is parent
    assert result.changed is False
    assert result.rejection_reason
    assert infinite.iterated is False
    assert agent.calls == []


def test_propose_combined_child_avoids_collision_key_lookups_after_filtering() -> None:
    """A rejected collision key must never be compared after the map is armed."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    collision = _ArmedCollisionKey()
    diagnostics: dict[object, object] = {collision: 0.75, "mean_smape": 0.25}
    collision.armed = True
    collision.comparisons = 0
    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient([_response()])

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics=diagnostics,  # type: ignore[arg-type]
        agent=agent,
    )

    assert result.child is parent
    assert collision.comparisons == 0
    assert agent.calls == []


@pytest.mark.parametrize(
    "diagnostics",
    (
        {"history_length": 128, "forecast_summary": "future-list"},
        {"future": [99, 100]},
        {"mean_smape": 0.25},
        object(),
    ),
)
def test_propose_combined_child_rejects_noncontract_diagnostics_without_llm(diagnostics: object) -> None:
    """Generic caller data cannot establish that it is label-free evidence."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient([_response()])

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics=diagnostics,  # type: ignore[arg-type]
        agent=agent,
    )

    assert result.child is parent
    assert result.changed is False
    assert result.rejection_reason
    assert agent.calls == []


def test_propose_combined_child_sends_only_typed_label_free_diagnostics_and_complete_dsl() -> None:
    """The proposal prompt must expose only fixed names and the complete typed DSL."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient([_response()])
    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics=_diagnostics(),
        agent=agent,
    )

    assert result.child is parent
    prompt = json.loads(agent.calls[0]["messages"][0]["content"])
    assert prompt["diagnostics"] == {
        "forecast_disagreement": 0.25,
        "history_length": 128,
        "successful_leaf_count": 4,
        "unavailable_leaf_count": 1,
    }
    assert prompt["tsfm_names"] == [
        "timesfm_2_5",
        "moirai_2_0",
        "toto_2_0",
        "chronos_bolt",
        "granite_ttm_r2",
    ]
    dsl = prompt["allowed_operations"]
    assert dsl == {
        "maximum_operations": 8,
        "mutation_targets_unique": True,
        "operations": {
            "add": {
                "exact_fields": ["op", "reason", "policy"],
                "name_rule": "policy.name is a new public Python identifier",
            },
            "fork": {
                "exact_fields": ["op", "source", "reason", "policy"],
                "name_rule": "source is an existing Combined name; policy.name is a new public Python identifier",
            },
            "remove": {
                "exact_fields": ["op", "target", "reason"],
                "name_rule": "target is an existing Combined name and cannot remove the final Combined policy",
            },
            "repair": {
                "exact_fields": ["op", "target", "reason", "policy"],
                "name_rule": "target is an existing Combined name and equals policy.name",
            },
        },
        "policy": {
            "exact_fields": [
                "name",
                "parents",
                "operator",
                "weights",
                "signal",
                "threshold",
                "above_parent",
                "below_parent",
                "fallback_parent",
            ],
            "json_types": {
                "above_parent": "string",
                "below_parent": "string",
                "fallback_parent": "string",
                "name": "string",
                "operator": "string",
                "parents": "array of strings",
                "signal": "string",
                "threshold": "finite number",
                "weights": "array of finite numbers",
            },
            "identifiers": "name and every parent are public Python identifiers",
            "parents": {
                "at_least_one_fixed_tsfm": True,
                "combined_parents_allowed": False,
                "leaf_parents_only": True,
                "maximum": 5,
                "minimum": 2,
                "unique": True,
            },
            "operators": {
                "median": "two to five parents; weights must be empty",
                "route": "exactly two parents; weights must be empty; above_parent and below_parent are distinct parents",
                "trimmed_mean": "three to five parents; weights must be empty",
                "weighted_mean": "two to five parents; weights match parents and are finite, non-negative, and sum to one",
            },
            "non_route_branches": "above_parent and below_parent must be empty",
            "fallback": "fallback_parent must occur in parents",
            "signals": [
                "outlier_fraction",
                "periodicity_strength",
                "recent_regime_confidence",
                "trend_strength",
                "zero_fraction",
            ],
        },
        "portfolio": {"combined_policy_count": {"minimum": 1, "maximum": 32}},
        "reason": {
            "json_type": "non-empty string",
            "maximum_length": 500,
            "control_characters": "only newline and tab are allowed",
        },
    }
    system = agent.calls[0]["system"]
    for phrase in (
        "exact canonical JSON types",
        "fixed TSFM identities",
        "no Combined parent",
        "no future values, documents, ground truth, task-role labels, Public, hidden",
        "public Python identifiers",
        "unique mutation target",
        "one through 32",
    ):
        assert phrase in system
    serialized_prompt = json.dumps(prompt, sort_keys=True)
    for forbidden in ("method_id", "checkpoint", "adapter", "runtime", "license", "scorer", "secret"):
        assert forbidden not in serialized_prompt


def test_propose_combined_child_rejects_invalid_typed_diagnostics_without_llm() -> None:
    """The host-owned diagnostic type validates its own finite label-free fields."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        CombinedProposalDiagnostics,
    )

    with pytest.raises(CombinedEvolutionError):
        CombinedProposalDiagnostics(True, 0.25, 4, 1)


class _LargeMapping(Mapping[str, object]):
    """A map that exposes whether the sanitizer sorts before checking its hard cap."""

    def __len__(self) -> int:
        return 17

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("oversized map must not be iterated")

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)


class _LyingLengthMapping(Mapping[str, object]):
    """Claims it is empty, then exposes a normal finite key stream."""

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[str]:
        return iter(("mean_smape",))

    def __getitem__(self, key: str) -> object:
        if key == "mean_smape":
            return 0.25
        raise KeyError(key)


class _InfiniteYieldMapping(Mapping[str, object]):
    """Raises if the sanitizer ever asks it to begin its unbounded key stream."""

    def __init__(self) -> None:
        self.iterated = False

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[str]:
        self.iterated = True
        while True:
            yield "mean_smape"

    def __getitem__(self, key: str) -> object:
        return 0.25


class _ArmedCollisionKey:
    """Collides with a trusted key and faults on any post-arm comparison/render."""

    def __init__(self) -> None:
        self.armed = False
        self.comparisons = 0

    def __hash__(self) -> int:
        return hash("mean_smape")

    def __eq__(self, other: object) -> bool:
        self.comparisons += 1
        if self.armed:
            raise AssertionError("collision key compared after arming")
        return False

    def __str__(self) -> str:
        raise AssertionError("collision key stringified")

    def __repr__(self) -> str:
        raise AssertionError("collision key repr'd")


def _max_mapping_depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_max_mapping_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return max((_max_mapping_depth(item) for item in value), default=0)
    return 0
