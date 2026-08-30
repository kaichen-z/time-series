"""Safe Morphology handoff projection and deterministic component fingerprints."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from types import MappingProxyType

from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT

from .morphology import AssumptionGrounding, MorphologyCard
from .numerical_package import freeze_string_mapping, host_assumption_ids
from .numerical_selector import DecisionPolicy, HindcastConfig
from .portfolio import CombinedPolicy
from .screening import ActiveDictionary, ScreeningPolicy, TaskProfile


_HANDOFF_TEMPLATES = {
    "seasonality": (
        "seasonality",
        "A history-supported seasonal pattern persists through the requested horizon.",
        "The seasonal pattern weakens, shifts phase, or disappears.",
    ),
    "trend": (
        "trend_persistence",
        "A history-supported directional trend persists through the requested horizon.",
        "The directional trend flattens or reverses.",
    ),
    "intermittency": (
        "other",
        "The history-supported intermittent arrival pattern persists.",
        "The arrival pattern becomes materially denser or sparser.",
    ),
    "regime": (
        "regime_persistence",
        "The history-supported recent regime persists through the requested horizon.",
        "The recent regime ends or changes materially.",
    ),
    "noise": (
        "anomaly_reversion",
        "History-supported irregular deviations revert toward the established level.",
        "Irregular deviations persist or establish a new regime.",
    ),
    "level": (
        "level_persistence",
        "The history-supported level persists through the requested horizon.",
        "The established level shifts materially.",
    ),
}


def safe_retrieval_projection(
    accepted: Sequence[AssumptionGrounding],
    rejected: Mapping[str, str],
) -> tuple[
    tuple[AssumptionGrounding, ...],
    dict[str, str],
    tuple[Mapping[str, str], ...],
]:
    """Project grounded assumptions through host-owned four-field templates."""
    safe: list[AssumptionGrounding] = []
    trace = dict(rejected)
    payloads: list[Mapping[str, str]] = []
    opaque_ids = host_assumption_ids(len(accepted))
    for opaque_id, item in zip(opaque_ids, accepted, strict=True):
        template = _HANDOFF_TEMPLATES.get(item.kind)
        if template is None:
            trace[item.assumption_id] = "unsupported_retrieval_kind"
            continue
        kind, claim, failure_condition = template
        payloads.append(
            MappingProxyType(
                {
                    "assumption_id": opaque_id,
                    "kind": kind,
                    "claim": claim,
                    "failure_condition": failure_condition,
                }
            )
        )
        safe.append(item)
    return tuple(safe), trace, tuple(payloads)


def component_fingerprints(
    *,
    input_fingerprint: str,
    profile: TaskProfile,
    active_dictionary: ActiveDictionary,
    screening_policy: ScreeningPolicy,
    combined_policies: Sequence[CombinedPolicy],
    decision_policy: DecisionPolicy,
    hindcast_config: HindcastConfig,
    morphology_card: MorphologyCard | None,
    provided: Mapping[str, str] | None,
) -> Mapping[str, str]:
    """Hash reviewed component payloads with caller-order-independent policies."""
    if not isinstance(input_fingerprint, str) or len(input_fingerprint) != 64:
        raise ValueError("task input fingerprint must be a canonical SHA-256 string")
    result = {
        "metric_policy_fingerprint": METRIC_POLICY_FINGERPRINT,
        "task_input": input_fingerprint,
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
            [
                policy.to_payload()
                for policy in sorted(combined_policies, key=lambda item: item.name)
            ]
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
        external = dict(freeze_string_mapping(provided, "component fingerprints"))
        conflicts = {
            key for key in external if key in result and external[key] != result[key]
        }
        if conflicts:
            raise ValueError(
                "provided component fingerprints conflict with host values: "
                f"{sorted(conflicts)!r}"
            )
        result.update(external)
    return MappingProxyType(dict(sorted(result.items())))


def task_input_fingerprint(
    *,
    task_id: str,
    history: Sequence[float],
    frequency: str,
    horizon: int,
) -> str:
    """Bind a Numerical package to one exact history-only forecasting input."""
    return _fingerprint(
        {
            "schema_version": 1,
            "task_id": task_id,
            "history": [float(value) for value in history],
            "frequency": frequency,
            "horizon": horizon,
        }
    )


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
