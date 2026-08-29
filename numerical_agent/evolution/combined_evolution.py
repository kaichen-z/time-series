"""Strict, history-only LLM proposal boundary for Combined policies.

This module deliberately proposes typed portfolio edits only.  It neither runs
forecasts nor decides whether a proposed child is accepted.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from common.llm import LLMClient

from .portfolio import CombinedPolicy, PolicyError, PolicyPortfolio, TSFMPolicy


COMBINED_EVOLUTION_SYSTEM = """You propose bounded, typed history-only Combined-policy edits.
Return exactly one JSON object with exactly an operations array. Do not use a
think wrapper or JSON fence. Every value must use the exact canonical JSON types
described in the user schema; do not add fields or use duplicate keys. Propose
at most eight operations, and each operation must have a unique mutation target.

The only permitted operations are add, repair, fork, and remove. Each uses its
exact schema: add creates a new public Python identifier; repair targets an
existing Combined name and its policy name must equal target; fork names an
existing Combined source and creates a new public Python identifier; remove
targets an existing Combined name and cannot remove the final Combined policy.
Reasons are non-empty strings of at most 500 characters, with only tab and
newline control characters allowed.

A policy has exactly name, parents, operator, weights, signal, threshold,
above_parent, below_parent, and fallback_parent. Name and every parent are
public Python identifiers. Parents are two to five unique materialized leaf
names, include at least one supplied fixed TSFM identity, and have no Combined parent.
Never change the fixed TSFM identities or order. weighted_mean has one
finite non-negative weight per parent summing to one; median has empty weights;
trimmed_mean has three to five parents and empty weights; route has exactly two
parents, empty weights, and distinct above_parent and below_parent values from
parents. Non-route policies have empty branches. fallback_parent occurs in
parents. signal is one supplied history-only signal and threshold is finite.

Do not score, select, or accept a child. Use only history and materialized leaf
information. Use no future values, documents, ground truth, task-role labels, Public, hidden data, secrets,
model bindings, checkpoints, adapters, runtime options, licenses, caches, or evaluation labels.
The child portfolio must retain one through 32 Combined policies."""

_POLICY_FIELDS = frozenset(
    {
        "name",
        "parents",
        "operator",
        "weights",
        "signal",
        "threshold",
        "above_parent",
        "below_parent",
        "fallback_parent",
    }
)
_OPERATION_FIELDS: dict[str, frozenset[str]] = {
    "add": frozenset({"op", "reason", "policy"}),
    "repair": frozenset({"op", "target", "reason", "policy"}),
    "fork": frozenset({"op", "source", "reason", "policy"}),
    "remove": frozenset({"op", "target", "reason"}),
}
_THINK_PREFIX_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_SIGNALS = (
    "outlier_fraction",
    "periodicity_strength",
    "recent_regime_confidence",
    "trend_strength",
    "zero_fraction",
)


class CombinedEvolutionError(ValueError):
    """A structured Combined proposal violates the trusted mutation boundary."""


class _StrictJsonError(ValueError):
    """A JSON response violates strict object or numeric parsing rules."""


@dataclass(frozen=True)
class CombinedProposalDiagnostics:
    """Trusted label-free aggregate inputs for a single Combined proposal call."""

    history_length: int
    forecast_disagreement: float
    successful_leaf_count: int
    unavailable_leaf_count: int

    def __post_init__(self) -> None:
        _validate_combined_diagnostics(
            self.history_length,
            self.forecast_disagreement,
            self.successful_leaf_count,
            self.unavailable_leaf_count,
        )

    def to_payload(self) -> dict[str, int | float]:
        _validate_combined_diagnostics(
            self.history_length,
            self.forecast_disagreement,
            self.successful_leaf_count,
            self.unavailable_leaf_count,
        )
        return {
            "forecast_disagreement": self.forecast_disagreement,
            "history_length": self.history_length,
            "successful_leaf_count": self.successful_leaf_count,
            "unavailable_leaf_count": self.unavailable_leaf_count,
        }


@dataclass(frozen=True)
class CombinedOperation:
    """One parsed operation with a canonical policy payload where applicable."""

    op: Literal["add", "repair", "fork", "remove"]
    reason: str
    policy: CombinedPolicy | None = None
    target: str = ""
    source: str = ""

    @property
    def mutation_target(self) -> str:
        """The name this operation writes; a fork's source remains untouched."""
        if self.op in {"add", "fork"}:
            if not isinstance(self.policy, CombinedPolicy):
                raise CombinedEvolutionError("operation requires a Combined policy")
            return self.policy.name
        return self.target


@dataclass(frozen=True)
class CombinedProposalResult:
    """An unscored proposal result; the caller owns all acceptance decisions."""

    parent: PolicyPortfolio
    child: PolicyPortfolio
    operations: tuple[CombinedOperation, ...]
    changed: bool
    rejection_reason: str = ""


def parse_combined_operations(response: str) -> tuple[CombinedOperation, ...]:
    """Parse one exact JSON batch into literal-only typed Combined operations."""
    try:
        payload = _parse_strict_json_object(response)
    except Exception as error:
        raise CombinedEvolutionError("response must contain one JSON object") from error
    if set(payload) != {"operations"} or not isinstance(payload["operations"], list):
        raise CombinedEvolutionError("response must contain exactly operations")
    raw_operations = payload["operations"]
    if len(raw_operations) > 8:
        raise CombinedEvolutionError("operations must contain at most eight entries")
    operations = tuple(_parse_operation(raw) for raw in raw_operations)
    targets = tuple(operation.mutation_target for operation in operations)
    if len(targets) != len(set(targets)):
        raise CombinedEvolutionError("operation targets must be unique")
    return operations


def _parse_strict_json_object(text: str) -> dict[str, object]:
    """Accept exactly one object after at most one permitted wrapper."""
    if not isinstance(text, str):
        raise _StrictJsonError("response must be text")
    stripped = text.strip()
    has_think_prefix = stripped.startswith("<think>")
    if has_think_prefix:
        match = _THINK_PREFIX_RE.match(stripped)
        if match is None:
            raise _StrictJsonError("invalid think wrapper")
        stripped = stripped[match.end():].strip()
        if stripped.startswith("```"):
            raise _StrictJsonError("response cannot stack think and JSON fence wrappers")
    fences = tuple(_FENCE_RE.finditer(stripped))
    if fences:
        if len(fences) != 1 or fences[0].span() != (0, len(stripped)):
            raise _StrictJsonError("response has ambiguous JSON fences")
        stripped = fences[0].group(1).strip()
    elif "```" in stripped:
        raise _StrictJsonError("invalid JSON fence")
    decoder = json.JSONDecoder(
        object_pairs_hook=_strict_object,
        parse_constant=_reject_nonfinite_constant,
    )
    try:
        parsed, end = decoder.raw_decode(stripped)
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise _StrictJsonError("response must contain one JSON object") from error
    if stripped[end:].strip() or not isinstance(parsed, dict):
        raise _StrictJsonError("response must contain exactly one JSON object")
    return parsed


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise _StrictJsonError(f"non-finite JSON constant {value!r}")


def apply_combined_operations(
    parent: PolicyPortfolio,
    operations: Sequence[CombinedOperation] | str,
    *,
    statistical_names: Sequence[str],
) -> PolicyPortfolio:
    """Atomically apply a parsed batch and validate its final leaf namespace."""
    if not isinstance(parent, PolicyPortfolio):
        raise CombinedEvolutionError("parent must be a PolicyPortfolio")
    try:
        parsed = parse_combined_operations(operations) if isinstance(operations, str) else tuple(operations)
        _validate_operation_batch(parsed)
        names = _reviewed_statistical_names(statistical_names)
        candidate = parent
        for operation in parsed:
            if operation.op == "add":
                candidate = candidate.add_combined(_required_policy(operation))
            elif operation.op == "repair":
                candidate = candidate.replace(operation.target, _required_policy(operation))
            elif operation.op == "fork":
                candidate = candidate.fork_combined(operation.source, _required_policy(operation))
            else:
                candidate = candidate.remove_combined(operation.target)
        # Validation is intentionally deferred until the complete candidate exists.
        candidate.validate_namespace(names)
        return candidate
    except CombinedEvolutionError:
        raise
    except (AssertionError, AttributeError, KeyError, OverflowError, PolicyError, TypeError, ValueError) as error:
        raise CombinedEvolutionError("combined operations are invalid") from error


def propose_combined_child(
    parent: PolicyPortfolio,
    *,
    statistical_names: Sequence[str],
    diagnostics: CombinedProposalDiagnostics,
    agent: LLMClient,
) -> CombinedProposalResult:
    """Request one bounded proposal and return Parent exactly on every failure."""
    try:
        _validate_proposal_parent(parent)
        reviewed_names = _reviewed_statistical_names(statistical_names)
        diagnostic_payload = _proposal_diagnostics_payload(diagnostics)
        prompt = {
            "current_policies": [_canonical_combined_payload(policy) for policy in parent.combined],
            "statistical_names": list(reviewed_names),
            "tsfm_names": [policy.name for policy in parent.tsfm],
            "diagnostics": diagnostic_payload,
            "allowed_operations": _allowed_operations_payload(),
        }
        response = agent.complete(
            system=COMBINED_EVOLUTION_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(prompt, sort_keys=True, allow_nan=False)}],
            temperature=0.0,
        )
        operations = parse_combined_operations(response.text)
        child = apply_combined_operations(
            parent, operations, statistical_names=reviewed_names
        )
    except Exception:
        return CombinedProposalResult(
            parent=parent,
            child=parent,
            operations=(),
            changed=False,
            rejection_reason="proposal rejected by the Combined policy boundary",
        )
    return CombinedProposalResult(
        parent=parent,
        child=child,
        operations=operations,
        changed=child != parent,
    )


def _parse_operation(raw: object) -> CombinedOperation:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("op"), str):
        raise CombinedEvolutionError("each operation must be an object with an op")
    op = raw["op"]
    if op not in _OPERATION_FIELDS or set(raw) != _OPERATION_FIELDS[op]:
        raise CombinedEvolutionError("operation fields are not exact")
    reason = _reason(raw["reason"])
    if op == "add":
        return CombinedOperation("add", reason, policy=_policy(raw["policy"]))
    if op == "repair":
        target = _name(raw["target"], "repair target")
        policy = _policy(raw["policy"])
        if policy.name != target:
            raise CombinedEvolutionError("repair target must equal policy name")
        return CombinedOperation("repair", reason, policy=policy, target=target)
    if op == "fork":
        return CombinedOperation(
            "fork",
            reason,
            policy=_policy(raw["policy"]),
            source=_name(raw["source"], "fork source"),
        )
    return CombinedOperation("remove", reason, target=_name(raw["target"], "remove target"))


def _policy(raw: object) -> CombinedPolicy:
    if not isinstance(raw, Mapping) or set(raw) != _POLICY_FIELDS:
        raise CombinedEvolutionError("policy fields are not exact")
    if not isinstance(raw["parents"], list) or not isinstance(raw["weights"], list):
        raise CombinedEvolutionError("policy parents and weights must be JSON arrays")
    if not all(isinstance(value, str) for value in raw["parents"]):
        raise CombinedEvolutionError("policy parents must be strings")
    if not all(_finite_json_number(value) for value in raw["weights"]):
        raise CombinedEvolutionError("policy weights must be numbers")
    for field in ("name", "operator", "signal", "above_parent", "below_parent", "fallback_parent"):
        if not isinstance(raw[field], str):
            raise CombinedEvolutionError(f"policy {field} must be a string")
    threshold = raw["threshold"]
    if not _finite_json_number(threshold):
        raise CombinedEvolutionError("policy threshold must be a number")
    try:
        return CombinedPolicy(
            name=raw["name"],
            parents=tuple(raw["parents"]),
            operator=raw["operator"],
            weights=tuple(raw["weights"]),
            signal=raw["signal"],
            threshold=threshold,
            above_parent=raw["above_parent"],
            below_parent=raw["below_parent"],
            fallback_parent=raw["fallback_parent"],
        )
    except (OverflowError, PolicyError, ValueError) as error:
        raise CombinedEvolutionError("policy violates the canonical Combined contract") from error


def _finite_json_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _validate_combined_diagnostics(
    history_length: object,
    forecast_disagreement: object,
    successful_leaf_count: object,
    unavailable_leaf_count: object,
) -> None:
    """Reject every non-canonical diagnostic before it can enter a prompt."""
    if type(history_length) is not int or not 1 <= history_length <= 1_000_000:
        raise CombinedEvolutionError("history_length must be a bounded exact integer")
    if (
        type(forecast_disagreement) is not float
        or not math.isfinite(forecast_disagreement)
        or not 0.0 <= forecast_disagreement <= 1_000_000.0
    ):
        raise CombinedEvolutionError(
            "forecast_disagreement must be a bounded exact finite float"
        )
    for name, value in (
        ("successful_leaf_count", successful_leaf_count),
        ("unavailable_leaf_count", unavailable_leaf_count),
    ):
        if type(value) is not int or not 0 <= value <= 1_000_000:
            raise CombinedEvolutionError(f"{name} must be a bounded exact integer")


def _validate_proposal_parent(parent: object) -> None:
    """Reject polymorphic portfolio records before reading or serializing them."""
    if type(parent) is not PolicyPortfolio:
        raise CombinedEvolutionError("parent must be an exact PolicyPortfolio")
    if type(parent.tsfm) is not tuple or type(parent.combined) is not tuple:
        raise CombinedEvolutionError("portfolio policy collections must be exact tuples")
    if not all(type(policy) is TSFMPolicy for policy in parent.tsfm):
        raise CombinedEvolutionError("portfolio TSFM members must be exact TSFMPolicy records")
    if not all(type(policy) is CombinedPolicy for policy in parent.combined):
        raise CombinedEvolutionError(
            "portfolio Combined members must be exact CombinedPolicy records"
        )
    PolicyPortfolio.__post_init__(parent)


def _canonical_combined_payload(policy: CombinedPolicy) -> dict[str, object]:
    """Serialize only canonical base-record fields without polymorphic dispatch."""
    return {
        "name": policy.name,
        "parents": policy.parents,
        "operator": policy.operator,
        "weights": policy.weights,
        "signal": policy.signal,
        "threshold": policy.threshold,
        "above_parent": policy.above_parent,
        "below_parent": policy.below_parent,
        "fallback_parent": policy.fallback_parent,
    }


def _validate_operation_batch(operations: tuple[CombinedOperation, ...]) -> None:
    if len(operations) > 8:
        raise CombinedEvolutionError("operations must contain at most eight entries")
    if not all(isinstance(operation, CombinedOperation) for operation in operations):
        raise CombinedEvolutionError("operations must be parsed Combined operations")
    targets = tuple(_validate_direct_operation(operation) for operation in operations)
    if len(targets) != len(set(targets)):
        raise CombinedEvolutionError("operation targets must be unique")


def _validate_direct_operation(operation: CombinedOperation) -> str:
    _reason(operation.reason)
    if operation.op == "add":
        if not isinstance(operation.policy, CombinedPolicy) or operation.target or operation.source:
            raise CombinedEvolutionError("add operation fields are invalid")
        return operation.policy.name
    if operation.op == "repair":
        if (
            not isinstance(operation.policy, CombinedPolicy)
            or operation.source
            or _name(operation.target, "repair target") != operation.policy.name
        ):
            raise CombinedEvolutionError("repair operation fields are invalid")
        return operation.target
    if operation.op == "fork":
        if (
            not isinstance(operation.policy, CombinedPolicy)
            or operation.target
            or not _name(operation.source, "fork source")
        ):
            raise CombinedEvolutionError("fork operation fields are invalid")
        return operation.policy.name
    if operation.op == "remove":
        if operation.policy is not None or operation.source:
            raise CombinedEvolutionError("remove operation fields are invalid")
        return _name(operation.target, "remove target")
    raise CombinedEvolutionError("unsupported operation")


def _required_policy(operation: CombinedOperation) -> CombinedPolicy:
    if not isinstance(operation.policy, CombinedPolicy):
        raise CombinedEvolutionError("operation requires a Combined policy")
    return operation.policy


def _reviewed_statistical_names(names: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(names, (str, bytes))
        or len(names) > 256
        or not all(isinstance(name, str) and 1 <= len(name) <= 64 for name in names)
    ):
        raise CombinedEvolutionError("statistical names must be a sequence of strings")
    if len(set(names)) != len(names):
        raise CombinedEvolutionError("statistical names must be unique")
    return tuple(sorted(names))


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CombinedEvolutionError(f"{label} must be a non-empty string")
    return value


def _reason(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 500:
        raise CombinedEvolutionError("operation reason must be a bounded non-empty string")
    if any(ord(character) < 32 and character not in "\n\t" for character in value):
        raise CombinedEvolutionError("operation reason contains control characters")
    return value


def _allowed_operations_payload() -> dict[str, object]:
    return {
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
                "name", "parents", "operator", "weights", "signal", "threshold",
                "above_parent", "below_parent", "fallback_parent",
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
                "minimum": 2,
                "maximum": 5,
                "unique": True,
                "leaf_parents_only": True,
                "at_least_one_fixed_tsfm": True,
                "combined_parents_allowed": False,
            },
            "operators": {
                "median": "two to five parents; weights must be empty",
                "route": "exactly two parents; weights must be empty; above_parent and below_parent are distinct parents",
                "trimmed_mean": "three to five parents; weights must be empty",
                "weighted_mean": "two to five parents; weights match parents and are finite, non-negative, and sum to one",
            },
            "non_route_branches": "above_parent and below_parent must be empty",
            "fallback": "fallback_parent must occur in parents",
            "signals": list(_SIGNALS),
        },
        "portfolio": {"combined_policy_count": {"minimum": 1, "maximum": 32}},
        "reason": {
            "json_type": "non-empty string",
            "maximum_length": 500,
            "control_characters": "only newline and tab are allowed",
        },
    }


def _proposal_diagnostics_payload(value: object) -> dict[str, int | float]:
    if type(value) is not CombinedProposalDiagnostics:
        raise CombinedEvolutionError("diagnostics must use CombinedProposalDiagnostics")
    return value.to_payload()
