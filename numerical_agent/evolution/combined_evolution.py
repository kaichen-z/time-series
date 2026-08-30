"""Strict, history-only LLM proposal boundary for Combined policies.

This module deliberately proposes typed portfolio edits only.  It neither runs
forecasts nor decides whether a proposed child is accepted.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from common.llm import LLMClient
from common.metrics import drcik_point_metrics

from .cache import SCALED_METRIC_CAP
from .portfolio import CombinedPolicy, PolicyError, PolicyPortfolio, TSFMPolicy
from .screening import TaskProfile


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
_LEGACY_SIGNALS = (
    "outlier_fraction",
    "periodicity_strength",
    "recent_regime_confidence",
    "trend_strength",
    "zero_fraction",
)
_SIGNALS = (
    "history_length",
    "horizon",
    "horizon_ratio",
    "intermittency_adi",
    "noise_relative_scale",
    *_LEGACY_SIGNALS,
)
_MORPHOLOGY_GROUP_PREDICATES: dict[str, tuple[str, str, float]] = {
    "periodic_high_confidence": ("periodicity_strength", "at_least", 0.6),
    "high_zero_fraction": ("zero_fraction", "at_least", 0.3),
    "high_outlier_fraction": ("outlier_fraction", "at_least", 0.05),
    "strong_trend": ("trend_strength", "at_least", 0.6),
    "recent_regime_shift": ("recent_regime_confidence", "at_least", 0.5),
    "high_noise": ("noise_relative_scale", "at_least", 1.0),
    "intermittent": ("intermittency_adi", "at_least", 1.32),
    "long_history": ("history_length", "at_least", 168.0),
    "long_horizon": ("horizon", "at_least", 24.0),
    "long_horizon_ratio": ("horizon_ratio", "at_least", 0.25),
}
_PROFILE_PREDICATE_FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "periodicity_strength": (0.0, 1.0),
    "zero_fraction": (0.0, 1.0),
    "outlier_fraction": (0.0, 1.0),
    "trend_strength": (0.0, 1.0),
    "recent_regime_confidence": (0.0, 1.0),
    "noise_relative_scale": (0.0, 1_000_000.0),
    "intermittency_adi": (0.0, 1_000_000.0),
}


class CombinedEvolutionError(ValueError):
    """A structured Combined proposal violates the trusted mutation boundary."""


class _StrictJsonError(ValueError):
    """A JSON response violates strict object or numeric parsing rules."""


_CANONICAL_EVIDENCE_TOKEN = object()


@dataclass(frozen=True)
class CanonicalScaledDelta:
    """Opaque per-task evidence produced only from one aligned forecast pair."""

    winsorized_smae_delta: float
    winsorized_srmse_delta: float
    candidate_smae: float
    candidate_srmse: float
    candidate_smae_raw: float
    candidate_srmse_raw: float
    candidate_smae_clipped: bool
    candidate_srmse_clipped: bool
    baseline_smae: float
    baseline_srmse: float
    baseline_smae_raw: float
    baseline_srmse_raw: float
    baseline_smae_clipped: bool
    baseline_srmse_clipped: bool
    _canonical_token: object = field(default=None, repr=False, compare=False)
    _canonical_values: tuple[object, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._canonical_token is not _CANONICAL_EVIDENCE_TOKEN
            or self._canonical_values != _canonical_scaled_delta_values(self)
        ):
            raise CombinedEvolutionError(
                "scaled delta records must come from the canonical scorer"
            )
        _validate_canonical_scaled_delta(self)


def winsorized_scaled_metric_delta(
    *,
    truth: Sequence[float] | object,
    candidate_forecast: Sequence[float] | object,
    baseline_forecast: Sequence[float] | object,
) -> CanonicalScaledDelta:
    """Score one complete candidate/baseline pair under the canonical capped contract."""
    if any(
        isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
        for value in (truth, candidate_forecast, baseline_forecast)
    ):
        raise CombinedEvolutionError("scaled metric evidence requires complete sequences")
    try:
        if (
            not truth
            or len(candidate_forecast) != len(truth)  # type: ignore[arg-type]
            or len(baseline_forecast) != len(truth)  # type: ignore[arg-type]
        ):
            raise CombinedEvolutionError(
                "scaled metric evidence requires a complete forecast pair"
            )
        candidate = drcik_point_metrics(
            truth, candidate_forecast, cap=SCALED_METRIC_CAP  # type: ignore[arg-type]
        )
        baseline = drcik_point_metrics(
            truth, baseline_forecast, cap=SCALED_METRIC_CAP  # type: ignore[arg-type]
        )
    except CombinedEvolutionError:
        raise
    except (OverflowError, TypeError, ValueError) as error:
        raise CombinedEvolutionError(
            "scaled metric evidence requires a complete finite forecast pair"
        ) from error
    values = {
        "winsorized_smae_delta": (
            float(candidate["smae"]) - float(baseline["smae"])
        ),
        "winsorized_srmse_delta": (
            float(candidate["srmse"]) - float(baseline["srmse"])
        ),
        "candidate_smae": float(candidate["smae"]),
        "candidate_srmse": float(candidate["srmse"]),
        "candidate_smae_raw": float(candidate["smae_raw"]),
        "candidate_srmse_raw": float(candidate["srmse_raw"]),
        "candidate_smae_clipped": bool(candidate["smae_clipped"]),
        "candidate_srmse_clipped": bool(candidate["srmse_clipped"]),
        "baseline_smae": float(baseline["smae"]),
        "baseline_srmse": float(baseline["srmse"]),
        "baseline_smae_raw": float(baseline["smae_raw"]),
        "baseline_srmse_raw": float(baseline["srmse_raw"]),
        "baseline_smae_clipped": bool(baseline["smae_clipped"]),
        "baseline_srmse_clipped": bool(baseline["srmse_clipped"]),
    }
    provisional = CanonicalScaledDelta(
        **values,
        _canonical_token=_CANONICAL_EVIDENCE_TOKEN,
        _canonical_values=tuple(values.values()),
    )
    return provisional


@dataclass(frozen=True)
class MorphologyGroupEvidence:
    """One sanitized fixed-predicate summary derived from Train tasks only."""

    group_id: str
    feature: str
    operator: str
    threshold: float
    task_count: int
    entity_count: int
    eligible_leaves: tuple[str, ...]
    baseline: str
    winsorized_smae_delta: float
    winsorized_srmse_delta: float
    coverage: float
    failure_rate: float
    forecast_disagreement: float = 0.0
    candidate_worst_smae_raw: float = 0.0
    candidate_worst_srmse_raw: float = 0.0
    baseline_worst_smae_raw: float = 0.0
    baseline_worst_srmse_raw: float = 0.0
    candidate_smae_clipped_count: int = 0
    candidate_srmse_clipped_count: int = 0
    baseline_smae_clipped_count: int = 0
    baseline_srmse_clipped_count: int = 0
    candidate_smae_clipped_rate: float = 0.0
    candidate_srmse_clipped_rate: float = 0.0
    baseline_smae_clipped_rate: float = 0.0
    baseline_srmse_clipped_rate: float = 0.0
    _canonical_token: object = field(default=None, repr=False, compare=False)
    _canonical_values: tuple[object, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._canonical_token is not _CANONICAL_EVIDENCE_TOKEN
            or self._canonical_values != _morphology_group_evidence_values(self)
        ):
            raise CombinedEvolutionError(
                "morphology group evidence must come from the canonical scorer"
            )
        _validate_morphology_group_evidence(self)
        object.__setattr__(self, "eligible_leaves", tuple(sorted(self.eligible_leaves)))

    def to_payload(self) -> dict[str, object]:
        _validate_morphology_group_evidence(self)
        return _morphology_group_payload(self)


@dataclass(frozen=True)
class CombinedProposalDiagnostics:
    """Trusted label-free aggregate inputs for a single Combined proposal call."""

    history_length: int
    forecast_disagreement: float
    successful_leaf_count: int
    unavailable_leaf_count: int
    morphology_groups: tuple[MorphologyGroupEvidence, ...] = ()

    def __post_init__(self) -> None:
        _validate_combined_diagnostics(
            self.history_length,
            self.forecast_disagreement,
            self.successful_leaf_count,
            self.unavailable_leaf_count,
            self.morphology_groups,
        )
        object.__setattr__(
            self,
            "morphology_groups",
            tuple(sorted(self.morphology_groups, key=lambda evidence: evidence.group_id)),
        )

    def to_payload(self) -> dict[str, object]:
        _validate_combined_diagnostics(
            self.history_length,
            self.forecast_disagreement,
            self.successful_leaf_count,
            self.unavailable_leaf_count,
            self.morphology_groups,
        )
        payload: dict[str, object] = {
            "forecast_disagreement": self.forecast_disagreement,
            "history_length": self.history_length,
            "successful_leaf_count": self.successful_leaf_count,
            "unavailable_leaf_count": self.unavailable_leaf_count,
        }
        if self.morphology_groups:
            payload["morphology_groups"] = [
                _morphology_group_payload(evidence)
                for evidence in sorted(
                    self.morphology_groups, key=lambda evidence: evidence.group_id
                )
            ]
        return payload


def summarize_morphology_group_evidence(
    profiles: tuple[TaskProfile, ...],
    *,
    entity_ids: tuple[str, ...],
    split: str,
    group_id: str,
    eligible_leaves: tuple[str, ...],
    baseline: str,
    reviewed_leaf_names: tuple[str, ...],
    scaled_deltas: tuple[CanonicalScaledDelta | None, ...],
    forecast_disagreements: tuple[float | None, ...],
) -> MorphologyGroupEvidence:
    """Aggregate aligned Train-only task metrics under one fixed profile predicate."""
    if type(split) is not str or split != "train":
        raise CombinedEvolutionError("morphology evidence is restricted to Train")
    aligned = (
        profiles,
        entity_ids,
        scaled_deltas,
        forecast_disagreements,
    )
    if any(type(values) is not tuple for values in aligned):
        raise CombinedEvolutionError("morphology aggregate inputs must be exact tuples")
    if not profiles or len(profiles) > 1_000_000:
        raise CombinedEvolutionError("morphology profiles must be nonempty and bounded")
    if any(len(values) != len(profiles) for values in aligned[1:]):
        raise CombinedEvolutionError("morphology aggregate inputs must align by task")
    if not all(type(profile) is TaskProfile for profile in profiles):
        raise CombinedEvolutionError("profiles must contain exact TaskProfile records")
    for profile in profiles:
        try:
            TaskProfile.__post_init__(profile)
        except (TypeError, ValueError) as error:
            raise CombinedEvolutionError("profile values are invalid") from error
        _validate_profile_predicate_inputs(profile)
    task_ids = tuple(profile.task_id for profile in profiles)
    if (
        any(type(task_id) is not str or not task_id for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise CombinedEvolutionError("profiles must have unique internal task identities")
    if any(type(entity_id) is not str or not entity_id for entity_id in entity_ids):
        raise CombinedEvolutionError("entity support identities must be bounded strings")
    reviewed = _reviewed_leaf_names(reviewed_leaf_names)
    _validate_reviewed_evidence_names(eligible_leaves, baseline, reviewed)
    feature, operator, threshold = _fixed_group_predicate(group_id)
    complete: list[tuple[CanonicalScaledDelta, float]] = []
    matched_indices: list[int] = []
    for index, profile in enumerate(profiles):
        delta = scaled_deltas[index]
        disagreement = forecast_disagreements[index]
        if delta is None and disagreement is None:
            pass
        elif delta is None or disagreement is None:
            raise CombinedEvolutionError("successful morphology metrics must be complete")
        else:
            if type(delta) is not CanonicalScaledDelta:
                raise CombinedEvolutionError(
                    "scaled deltas must be exact canonical records"
                )
            CanonicalScaledDelta.__post_init__(delta)
            if not _bounded_exact_float(disagreement, lower=0.0, upper=1_000_000.0):
                raise CombinedEvolutionError("forecast disagreement must be finite")
        if _profile_matches(profile, feature, operator, threshold):
            matched_indices.append(index)
            if delta is not None:
                assert disagreement is not None
                complete.append((delta, disagreement))
    if not matched_indices:
        raise CombinedEvolutionError("fixed morphology group has no task support")
    entity_count = len({entity_ids[index] for index in matched_indices})
    if entity_count < 3:
        raise CombinedEvolutionError("morphology group requires at least three entities")
    if not complete:
        raise CombinedEvolutionError("morphology group has no successful metrics")
    task_count = len(matched_indices)
    successful_count = len(complete)
    candidate_smae_clipped_count = sum(row[0].candidate_smae_clipped for row in complete)
    candidate_srmse_clipped_count = sum(row[0].candidate_srmse_clipped for row in complete)
    baseline_smae_clipped_count = sum(row[0].baseline_smae_clipped for row in complete)
    baseline_srmse_clipped_count = sum(row[0].baseline_srmse_clipped for row in complete)
    values: dict[str, object] = dict(
        group_id=group_id,
        feature=feature,
        operator=operator,
        threshold=threshold,
        task_count=task_count,
        entity_count=entity_count,
        eligible_leaves=tuple(sorted(eligible_leaves)),
        baseline=baseline,
        winsorized_smae_delta=float(
            statistics.fmean(row[0].winsorized_smae_delta for row in complete)
        ),
        winsorized_srmse_delta=float(
            statistics.fmean(row[0].winsorized_srmse_delta for row in complete)
        ),
        coverage=float(successful_count / task_count),
        failure_rate=float((task_count - successful_count) / task_count),
        forecast_disagreement=float(statistics.fmean(row[1] for row in complete)),
        candidate_worst_smae_raw=max(row[0].candidate_smae_raw for row in complete),
        candidate_worst_srmse_raw=max(row[0].candidate_srmse_raw for row in complete),
        baseline_worst_smae_raw=max(row[0].baseline_smae_raw for row in complete),
        baseline_worst_srmse_raw=max(row[0].baseline_srmse_raw for row in complete),
        candidate_smae_clipped_count=candidate_smae_clipped_count,
        candidate_srmse_clipped_count=candidate_srmse_clipped_count,
        baseline_smae_clipped_count=baseline_smae_clipped_count,
        baseline_srmse_clipped_count=baseline_srmse_clipped_count,
        candidate_smae_clipped_rate=float(candidate_smae_clipped_count / successful_count),
        candidate_srmse_clipped_rate=float(candidate_srmse_clipped_count / successful_count),
        baseline_smae_clipped_rate=float(baseline_smae_clipped_count / successful_count),
        baseline_srmse_clipped_rate=float(baseline_srmse_clipped_count / successful_count),
    )
    return MorphologyGroupEvidence(
        **values,
        _canonical_token=_CANONICAL_EVIDENCE_TOKEN,
        _canonical_values=tuple(values.values()),
    )


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
        tsfm_names = tuple(policy.name for policy in parent.tsfm)
        diagnostic_payload = _proposal_diagnostics_payload(
            diagnostics,
            known_leaves=(*reviewed_names, *tsfm_names),
        )
        prompt = {
            "current_policies": [_canonical_combined_payload(policy) for policy in parent.combined],
            "statistical_names": list(reviewed_names),
            "tsfm_names": list(tsfm_names),
            "diagnostics": diagnostic_payload,
            "allowed_operations": _allowed_operations_payload(
                include_morphology_signals=bool(diagnostics.morphology_groups)
            ),
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
    morphology_groups: object = (),
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
    if type(morphology_groups) is not tuple or len(morphology_groups) > 32:
        raise CombinedEvolutionError("morphology_groups must be a bounded exact tuple")
    if not all(type(value) is MorphologyGroupEvidence for value in morphology_groups):
        raise CombinedEvolutionError(
            "morphology_groups must contain exact MorphologyGroupEvidence records"
        )
    group_ids = tuple(value.group_id for value in morphology_groups)
    if len(group_ids) != len(set(group_ids)):
        raise CombinedEvolutionError("morphology group identifiers must be unique")
    for value in morphology_groups:
        _validate_morphology_group_evidence(value)


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


def _allowed_operations_payload(
    *, include_morphology_signals: bool = False
) -> dict[str, object]:
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
            "signals": list(_SIGNALS if include_morphology_signals else _LEGACY_SIGNALS),
        },
        "portfolio": {"combined_policy_count": {"minimum": 1, "maximum": 32}},
        "reason": {
            "json_type": "non-empty string",
            "maximum_length": 500,
            "control_characters": "only newline and tab are allowed",
        },
    }


def _proposal_diagnostics_payload(
    value: object,
    *,
    known_leaves: tuple[str, ...],
) -> dict[str, object]:
    if type(value) is not CombinedProposalDiagnostics:
        raise CombinedEvolutionError("diagnostics must use CombinedProposalDiagnostics")
    _validate_combined_diagnostics(
        value.history_length,
        value.forecast_disagreement,
        value.successful_leaf_count,
        value.unavailable_leaf_count,
        value.morphology_groups,
    )
    reviewed = frozenset(known_leaves)
    for evidence in value.morphology_groups:
        leaves = frozenset(evidence.eligible_leaves)
        if evidence.baseline not in reviewed or not leaves <= reviewed:
            raise CombinedEvolutionError("morphology evidence contains an unknown leaf")
    payload: dict[str, object] = {
        "forecast_disagreement": value.forecast_disagreement,
        "history_length": value.history_length,
        "successful_leaf_count": value.successful_leaf_count,
        "unavailable_leaf_count": value.unavailable_leaf_count,
    }
    if value.morphology_groups:
        payload["morphology_groups"] = [
            _morphology_group_payload(evidence)
            for evidence in sorted(
                value.morphology_groups, key=lambda evidence: evidence.group_id
            )
        ]
    return payload


def _fixed_group_predicate(group_id: object) -> tuple[str, str, float]:
    if type(group_id) is not str or group_id not in _MORPHOLOGY_GROUP_PREDICATES:
        raise CombinedEvolutionError("unsupported fixed morphology group")
    return _MORPHOLOGY_GROUP_PREDICATES[group_id]


def _validate_morphology_group_evidence(value: object) -> None:
    if not isinstance(value, MorphologyGroupEvidence):
        raise CombinedEvolutionError("invalid morphology group evidence")
    if (
        value._canonical_token is not _CANONICAL_EVIDENCE_TOKEN
        or value._canonical_values != _morphology_group_evidence_values(value)
    ):
        raise CombinedEvolutionError(
            "morphology group evidence must come from the canonical scorer"
        )
    expected = _fixed_group_predicate(value.group_id)
    if (
        type(value.feature) is not str
        or type(value.operator) is not str
        or type(value.threshold) is not float
        or (value.feature, value.operator, value.threshold) != expected
    ):
        raise CombinedEvolutionError("morphology group predicate is not fixed")
    for name, count in (
        ("task_count", value.task_count),
        ("entity_count", value.entity_count),
    ):
        if type(count) is not int or not 0 <= count <= 1_000_000:
            raise CombinedEvolutionError(f"{name} must be a bounded exact integer")
    if value.entity_count < 3 or value.task_count < value.entity_count:
        raise CombinedEvolutionError("morphology evidence has insufficient entity support")
    _validate_leaf_tuple(value.eligible_leaves, "eligible_leaves")
    if not _public_identifier(value.baseline):
        raise CombinedEvolutionError("baseline must be a public Python identifier")
    if not _bounded_exact_float(
        value.winsorized_smae_delta, lower=-5.0, upper=5.0
    ):
        raise CombinedEvolutionError("winsorized_smae_delta must be finite")
    if not _bounded_exact_float(
        value.winsorized_srmse_delta, lower=-5.0, upper=5.0
    ):
        raise CombinedEvolutionError("winsorized_srmse_delta must be finite")
    for name, rate in (("coverage", value.coverage), ("failure_rate", value.failure_rate)):
        if not _bounded_exact_float(rate, lower=0.0, upper=1.0):
            raise CombinedEvolutionError(f"{name} must be a finite rate")
    if value.coverage + value.failure_rate > 1.0 + 1e-12:
        raise CombinedEvolutionError("coverage and failure_rate cannot exceed total support")
    if not _bounded_exact_float(
        value.forecast_disagreement, lower=0.0, upper=1_000_000.0
    ):
        raise CombinedEvolutionError("forecast_disagreement must be finite")
    for name in (
        "candidate_worst_smae_raw",
        "candidate_worst_srmse_raw",
        "baseline_worst_smae_raw",
        "baseline_worst_srmse_raw",
    ):
        if not _bounded_exact_float(
            getattr(value, name), lower=0.0, upper=1_000_000_000_000.0
        ):
            raise CombinedEvolutionError(f"{name} must be a bounded finite raw tail")
    for name in (
        "candidate_smae_clipped_count",
        "candidate_srmse_clipped_count",
        "baseline_smae_clipped_count",
        "baseline_srmse_clipped_count",
    ):
        count = getattr(value, name)
        if type(count) is not int or not 0 <= count <= value.task_count:
            raise CombinedEvolutionError(f"{name} must be a bounded exact count")
    successful_count = value.task_count - round(value.failure_rate * value.task_count)
    for prefix in ("candidate_smae", "candidate_srmse", "baseline_smae", "baseline_srmse"):
        count = getattr(value, f"{prefix}_clipped_count")
        rate = getattr(value, f"{prefix}_clipped_rate")
        if not _bounded_exact_float(rate, lower=0.0, upper=1.0):
            raise CombinedEvolutionError(f"{prefix}_clipped_rate must be a finite rate")
        if successful_count < 1 or not math.isclose(
            rate,
            count / successful_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CombinedEvolutionError(
                f"{prefix} clipping count and rate must describe successful tasks"
            )


def _morphology_group_payload(value: MorphologyGroupEvidence) -> dict[str, object]:
    _validate_morphology_group_evidence(value)
    return {
        "baseline": value.baseline,
        "coverage": value.coverage,
        "eligible_leaves": sorted(value.eligible_leaves),
        "entity_count": value.entity_count,
        "failure_rate": value.failure_rate,
        "feature": value.feature,
        "forecast_disagreement": value.forecast_disagreement,
        "group_id": value.group_id,
        "operator": value.operator,
        "task_count": value.task_count,
        "threshold": value.threshold,
        "winsorized_smae_delta": value.winsorized_smae_delta,
        "winsorized_srmse_delta": value.winsorized_srmse_delta,
        "candidate_worst_smae_raw": value.candidate_worst_smae_raw,
        "candidate_worst_srmse_raw": value.candidate_worst_srmse_raw,
        "baseline_worst_smae_raw": value.baseline_worst_smae_raw,
        "baseline_worst_srmse_raw": value.baseline_worst_srmse_raw,
        "candidate_smae_clipped_count": value.candidate_smae_clipped_count,
        "candidate_srmse_clipped_count": value.candidate_srmse_clipped_count,
        "baseline_smae_clipped_count": value.baseline_smae_clipped_count,
        "baseline_srmse_clipped_count": value.baseline_srmse_clipped_count,
        "candidate_smae_clipped_rate": value.candidate_smae_clipped_rate,
        "candidate_srmse_clipped_rate": value.candidate_srmse_clipped_rate,
        "baseline_smae_clipped_rate": value.baseline_smae_clipped_rate,
        "baseline_srmse_clipped_rate": value.baseline_srmse_clipped_rate,
    }


def _canonical_scaled_delta_values(value: CanonicalScaledDelta) -> tuple[object, ...]:
    return tuple(
        getattr(value, name)
        for name in (
            "winsorized_smae_delta",
            "winsorized_srmse_delta",
            "candidate_smae",
            "candidate_srmse",
            "candidate_smae_raw",
            "candidate_srmse_raw",
            "candidate_smae_clipped",
            "candidate_srmse_clipped",
            "baseline_smae",
            "baseline_srmse",
            "baseline_smae_raw",
            "baseline_srmse_raw",
            "baseline_smae_clipped",
            "baseline_srmse_clipped",
        )
    )


def _validate_canonical_scaled_delta(value: CanonicalScaledDelta) -> None:
    for prefix in ("candidate", "baseline"):
        for metric in ("smae", "srmse"):
            capped = getattr(value, f"{prefix}_{metric}")
            raw = getattr(value, f"{prefix}_{metric}_raw")
            clipped = getattr(value, f"{prefix}_{metric}_clipped")
            if (
                type(capped) is not float
                or not 0.0 <= capped <= SCALED_METRIC_CAP
                or type(raw) is not float
                or math.isnan(raw)
                or raw < 0.0
                or type(clipped) is not bool
                or capped != min(SCALED_METRIC_CAP, raw)
                or clipped != (raw > SCALED_METRIC_CAP)
            ):
                raise CombinedEvolutionError(
                    "canonical scaled delta contains inconsistent capped/raw evidence"
                )
    for metric in ("smae", "srmse"):
        delta = getattr(value, f"winsorized_{metric}_delta")
        expected = getattr(value, f"candidate_{metric}") - getattr(
            value, f"baseline_{metric}"
        )
        if type(delta) is not float or not math.isfinite(delta) or delta != expected:
            raise CombinedEvolutionError(
                "canonical scaled delta contains an inconsistent metric delta"
            )


def _morphology_group_evidence_values(
    value: MorphologyGroupEvidence,
) -> tuple[object, ...]:
    return tuple(
        getattr(value, name)
        for name in (
            "group_id",
            "feature",
            "operator",
            "threshold",
            "task_count",
            "entity_count",
            "eligible_leaves",
            "baseline",
            "winsorized_smae_delta",
            "winsorized_srmse_delta",
            "coverage",
            "failure_rate",
            "forecast_disagreement",
            "candidate_worst_smae_raw",
            "candidate_worst_srmse_raw",
            "baseline_worst_smae_raw",
            "baseline_worst_srmse_raw",
            "candidate_smae_clipped_count",
            "candidate_srmse_clipped_count",
            "baseline_smae_clipped_count",
            "baseline_srmse_clipped_count",
            "candidate_smae_clipped_rate",
            "candidate_srmse_clipped_rate",
            "baseline_smae_clipped_rate",
            "baseline_srmse_clipped_rate",
        )
    )


def _validate_leaf_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or not 2 <= len(value) <= 5
        or not all(_public_identifier(name) for name in value)
        or len(value) != len(set(value))
    ):
        raise CombinedEvolutionError(f"{label} must contain two to five unique leaf names")
    return value


def _reviewed_leaf_names(value: object) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or not 1 <= len(value) <= 256
        or not all(_public_identifier(name) for name in value)
        or len(value) != len(set(value))
    ):
        raise CombinedEvolutionError("reviewed leaf names must be a bounded exact tuple")
    return value


def _validate_reviewed_evidence_names(
    eligible_leaves: object,
    baseline: object,
    reviewed_leaf_names: tuple[str, ...],
) -> None:
    leaves = _validate_leaf_tuple(eligible_leaves, "eligible_leaves")
    if not _public_identifier(baseline):
        raise CombinedEvolutionError("baseline must be a public Python identifier")
    reviewed = frozenset(reviewed_leaf_names)
    if baseline not in reviewed or not frozenset(leaves) <= reviewed:
        raise CombinedEvolutionError("morphology aggregate contains an unknown leaf")


def _public_identifier(value: object) -> bool:
    return type(value) is str and bool(value) and value.isidentifier() and not value.startswith("_")


def _bounded_exact_float(value: object, *, lower: float, upper: float) -> bool:
    return (
        type(value) is float
        and math.isfinite(value)
        and lower <= value <= upper
    )


def _profile_matches(
    profile: TaskProfile, feature: str, operator: str, threshold: float
) -> bool:
    _validate_profile_predicate_inputs(profile)
    if feature == "horizon_ratio":
        measurement = float(profile.horizon / profile.history_length)
    else:
        measurement = float(getattr(profile, feature))
    if operator == "at_least":
        return measurement >= threshold
    raise CombinedEvolutionError("unsupported fixed morphology operator")


def _validate_profile_predicate_inputs(profile: TaskProfile) -> None:
    for name, value in (
        ("history_length", profile.history_length),
        ("horizon", profile.horizon),
    ):
        if type(value) is not int or not 1 <= value <= 1_000_000:
            raise CombinedEvolutionError(f"{name} must be a bounded exact integer")
    for name, (lower, upper) in _PROFILE_PREDICATE_FLOAT_BOUNDS.items():
        if not _bounded_exact_float(getattr(profile, name), lower=lower, upper=upper):
            raise CombinedEvolutionError(
                "profile predicate measurements must be bounded exact floats"
            )
