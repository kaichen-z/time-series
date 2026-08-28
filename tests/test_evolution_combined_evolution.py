"""Tests for the bounded typed Combined-policy proposal boundary."""
from __future__ import annotations

import json

import pytest

from common.llm import FakeLLMClient
from numerical_agent.evolution.portfolio import PolicyPortfolio


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
        diagnostics={
            "mean_smape": 0.25,
            "family_summary": {"tsfm": 0.20},
            "future_values": [99, 100],
            "documents": ["secret document"],
            "ground_truth": "forbidden",
            "runtime_secret": "never expose",
            "dev_task_id": "never expose",
            "public_task_id": "never expose",
            "hidden_task_id": "never expose",
        },
        agent=agent,
    )

    assert result.parent is parent
    assert result.changed is True
    assert "combined_proposed" in result.child.names
    assert result.rejection_reason == ""
    prompt = json.loads(agent.calls[0]["messages"][0]["content"])
    assert set(prompt) == {"allowed_operations", "current_policies", "diagnostics", "statistical_names"}
    assert prompt["current_policies"] == json.loads(
        json.dumps([policy.to_payload() for policy in parent.combined])
    )
    assert prompt["statistical_names"] == ["croston_sba", "holt_damped_trend", "median_seasonal_profile_forecast", "robust_loess_trend", "seasonal_naive"]
    rendered = json.dumps(prompt, sort_keys=True)
    for forbidden in ("99", "secret document", "forbidden", "never expose", "dev_task_id", "public_task_id", "hidden_task_id"):
        assert forbidden not in rendered


def test_propose_combined_child_returns_exact_parent_with_sanitized_rejection() -> None:
    """Returning a partial child or echoing model text would leak and break atomicity."""
    from numerical_agent.evolution.combined_evolution import propose_combined_child

    parent = PolicyPortfolio.flagship5()
    agent = FakeLLMClient(["{not valid json: runtime_secret=do-not-echo}"])

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics={"mean_smape": 0.25},
        agent=agent,
    )

    assert result.parent is parent
    assert result.child is parent
    assert result.changed is False
    assert result.operations == ()
    assert result.rejection_reason
    assert "runtime_secret" not in result.rejection_reason
    assert "do-not-echo" not in result.rejection_reason
