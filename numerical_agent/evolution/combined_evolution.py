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

from .portfolio import CombinedPolicy, PolicyError, PolicyPortfolio


COMBINED_EVOLUTION_SYSTEM = """You propose bounded, typed history-only Combined-policy edits.
Return exactly one JSON object with an operations array.  Each operation must be
one of add, repair, fork, or remove and must use its exact allowed fields.  A
policy uses only name, parents, operator, weights, signal, threshold,
above_parent, below_parent, and fallback_parent.  Do not score, select, or
accept a child.  Propose at most eight operations."""

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
_FORBIDDEN_DIAGNOSTIC_KEY = re.compile(
    r"(?:future|document|ground.?truth|\bgt\b|role|subtype|secret|token|"
    r"password|credential|runtime|checkpoint|dev|public|hidden|label)",
    re.IGNORECASE,
)
_SAFE_DIAGNOSTIC_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_DIAGNOSTIC_MAX_DEPTH = 3
_DIAGNOSTIC_MAX_NODES = 64
_DIAGNOSTIC_MAX_ITEMS = 64
_DIAGNOSTIC_MAX_BYTES = 3072
_DIAGNOSTIC_MAX_CHILDREN = 16
_DIAGNOSTIC_MAX_RAW_STRING_LENGTH = 256
_DROP = object()


class CombinedEvolutionError(ValueError):
    """A structured Combined proposal violates the trusted mutation boundary."""


class _StrictJsonError(ValueError):
    """A JSON response violates strict object or numeric parsing rules."""


@dataclass
class _DiagnosticsBudget:
    nodes: int = 0
    items: int = 0
    bytes: int = 0

    def consume(self, value: object) -> bool:
        try:
            encoded = json.dumps(value, ensure_ascii=True, allow_nan=False)
        except (OverflowError, TypeError, ValueError):
            return False
        size = len(encoded.encode("utf-8"))
        if (
            self.nodes >= _DIAGNOSTIC_MAX_NODES
            or self.items >= _DIAGNOSTIC_MAX_ITEMS
            or self.bytes + size > _DIAGNOSTIC_MAX_BYTES
        ):
            return False
        self.nodes += 1
        self.items += 1
        self.bytes += size
        return True


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
    """Mirror the shared parser's wrappers while rejecting ambiguous JSON objects."""
    if not isinstance(text, str):
        raise _StrictJsonError("response must be text")
    stripped = _THINK_RE.sub("", text).strip()
    candidates = [match.group(1).strip() for match in _FENCE_RE.finditer(stripped)]
    candidates.append(stripped)
    decoder = json.JSONDecoder(
        object_pairs_hook=_strict_object,
        parse_constant=_reject_nonfinite_constant,
    )
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(
                candidate,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite_constant,
            )
            if isinstance(parsed, dict):
                return parsed
        except _StrictJsonError:
            raise
        except (json.JSONDecodeError, RecursionError, ValueError) as error:
            last_error = error
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                parsed, _end = decoder.raw_decode(candidate[index:])
            except _StrictJsonError:
                raise
            except (json.JSONDecodeError, RecursionError, ValueError) as error:
                last_error = error
                continue
            if isinstance(parsed, dict):
                return parsed
    detail = str(last_error) if last_error is not None else "response contained no JSON object"
    raise _StrictJsonError(f"no valid JSON object in response: {detail}")


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
    diagnostics: Mapping[str, object],
    agent: LLMClient,
) -> CombinedProposalResult:
    """Request one bounded proposal and return Parent exactly on every failure."""
    try:
        reviewed_names = _reviewed_statistical_names(statistical_names)
        prompt = {
            "current_policies": [policy.to_payload() for policy in parent.combined],
            "statistical_names": list(reviewed_names),
            "diagnostics": _sanitize_diagnostics(diagnostics),
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
    if isinstance(names, (str, bytes)) or not all(isinstance(name, str) for name in names):
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


def _allowed_operations_payload() -> dict[str, list[str]]:
    return {operation: sorted(fields) for operation, fields in sorted(_OPERATION_FIELDS.items())}


def _sanitize_diagnostics(value: Mapping[str, object]) -> dict[str, object]:
    """Keep small aggregate, JSON-safe diagnostics and omit sensitive-looking keys."""
    if not isinstance(value, Mapping):
        return {}
    budget = _DiagnosticsBudget()
    sanitized = _sanitize_diagnostic_mapping(value, depth=0, budget=budget)
    return sanitized if isinstance(sanitized, dict) else {}


def _sanitize_diagnostic_mapping(
    value: Mapping[object, object], *, depth: int, budget: _DiagnosticsBudget
) -> dict[str, object] | object:
    if (
        depth > _DIAGNOSTIC_MAX_DEPTH
        or not _container_within_limit(value)
        or not budget.consume({})
    ):
        return _DROP
    result: dict[str, object] = {}
    for key in sorted(value, key=str):
        if len(result) >= _DIAGNOSTIC_MAX_CHILDREN or not _safe_diagnostic_key(key):
            continue
        cleaned = _sanitize_diagnostic_value(value[key], depth=depth + 1, budget=budget)
        if cleaned is _DROP or not budget.consume(key):
            continue
        result[key] = cleaned
    return result


def _sanitize_diagnostic_value(
    value: object, *, depth: int, budget: _DiagnosticsBudget
) -> object:
    if depth > _DIAGNOSTIC_MAX_DEPTH:
        return _DROP
    if value is None or isinstance(value, bool):
        return value if budget.consume(value) else _DROP
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if _finite_json_number(value) and budget.consume(value) else _DROP
    if isinstance(value, str):
        # Diagnostics are aggregate-only.  Never pass arbitrary text through
        # this boundary, even beneath an otherwise allowed key.
        if len(value) > _DIAGNOSTIC_MAX_RAW_STRING_LENGTH:
            return _DROP
        return _DROP
    if isinstance(value, Mapping):
        return _sanitize_diagnostic_mapping(value, depth=depth, budget=budget)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        if not _container_within_limit(value) or not budget.consume([]):
            return _DROP
        result: list[object] = []
        for index, item in enumerate(value):
            if index == _DIAGNOSTIC_MAX_CHILDREN:
                break
            cleaned = _sanitize_diagnostic_value(item, depth=depth + 1, budget=budget)
            if cleaned is not _DROP:
                result.append(cleaned)
        return result
    return _DROP


def _safe_diagnostic_key(key: object) -> bool:
    return (
        isinstance(key, str)
        and len(key) <= 64
        and _SAFE_DIAGNOSTIC_KEY.fullmatch(key) is not None
        and _FORBIDDEN_DIAGNOSTIC_KEY.search(key) is None
    )


def _container_within_limit(value: Mapping[object, object] | Sequence[object]) -> bool:
    try:
        return len(value) <= _DIAGNOSTIC_MAX_CHILDREN
    except (OverflowError, TypeError, ValueError):
        return False
