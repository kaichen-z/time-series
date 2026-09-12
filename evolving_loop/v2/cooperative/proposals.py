"""Deterministic single-coordinate and joint cooperative proposals."""
from __future__ import annotations

import re
from collections.abc import Mapping

from ..bundle import EvolutionBundleV2
from ..contracts import SanitizedEvolutionFeedback, canonical_v2_bytes
from .adapters import (
    CooperativeArtifactCatalog,
    DecisionCoordinateAdapter,
    NumericalCoordinateAdapter,
    RetrievalCoordinateAdapter,
)
from .contracts import BundleCandidateV2


_SINGLE_ARMS = ("numerical", "retrieval", "decision")
_SENSITIVE_KEY_FRAGMENTS = (
    "dev",
    "public",
    "future",
    "holdout",
    "evaluator",
    "label",
    "taskid",
    "forecast",
    "document",
    "pertask",
    "residual",
)


def _reject_sensitive_feedback_keys(value: object, *, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            compact = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if any(fragment in compact for fragment in _SENSITIVE_KEY_FRAGMENTS):
                raise ValueError(f"{field} contains sensitive evaluator key {key}")
            _reject_sensitive_feedback_keys(item, field=f"{field}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_sensitive_feedback_keys(item, field=f"{field}[{index}]")


def _validated_feedback(
    parent: EvolutionBundleV2,
    feedback: SanitizedEvolutionFeedback,
) -> None:
    if type(feedback) is not SanitizedEvolutionFeedback:
        raise TypeError("proposal feedback must be SanitizedEvolutionFeedback")
    # Reparse at the proposal boundary so even a tampered frozen instance cannot
    # smuggle evaluator-only data into a coordinate provider.
    payload = feedback.to_payload()
    _reject_sensitive_feedback_keys(payload, field="feedback")
    SanitizedEvolutionFeedback.from_payload(payload)
    canonical_v2_bytes(payload)
    if feedback.parent_sha256 != parent.fingerprint():
        raise ValueError("proposal feedback Parent mismatch")


def _one_change(
    *,
    parent: EvolutionBundleV2,
    arm: str,
    catalog: CooperativeArtifactCatalog,
    adapters: Mapping[str, object],
    step: int,
) -> tuple[str, str | tuple[str, str]] | None:
    try:
        adapter = adapters[arm]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing {arm} coordinate adapter") from error

    if arm == "numerical":
        if type(adapter) is not NumericalCoordinateAdapter:
            raise TypeError("numerical adapter must be NumericalCoordinateAdapter")
        current = catalog.resolve_numerical(
            parent.numerical_release_sha256,
            parent.numerical_registry_sha256,
        )
        proposed = adapter.propose(current)
        if proposed is None:
            return None
        identity = catalog.add_numerical(proposed)
        if identity == (
            parent.numerical_release_sha256,
            parent.numerical_registry_sha256,
        ):
            return None
        return arm, identity

    if arm == "retrieval":
        if type(adapter) is not RetrievalCoordinateAdapter:
            raise TypeError("retrieval adapter must be RetrievalCoordinateAdapter")
        current = catalog.resolve_retrieval(parent.retrieval_release_sha256)
        proposed = adapter.propose(current, step)
        identity = catalog.add_retrieval(proposed)
        return None if identity == parent.retrieval_release_sha256 else (arm, identity)

    if arm == "decision":
        if type(adapter) is not DecisionCoordinateAdapter:
            raise TypeError("decision adapter must be DecisionCoordinateAdapter")
        current = catalog.resolve_decision(parent.decision_policy_sha256)
        proposed = adapter.propose(current, step)
        if proposed is None:
            return None
        if (
            proposed.prompt == current.prompt
            or proposed.skills != current.skills
            or proposed.enable_evidence_adjustments
            != current.enable_evidence_adjustments
            or proposed.max_evidence_adjustments
            != current.max_evidence_adjustments
            or proposed.aggregation != current.aggregation
        ):
            raise ValueError(
                "cooperative Decision proposal must be prompt-only"
            )
        identity = catalog.add_decision(proposed)
        return None if identity == parent.decision_policy_sha256 else (arm, identity)

    raise ValueError("proposal arm must be numerical, retrieval, decision, or joint")


def propose_bundle_candidate(
    parent: EvolutionBundleV2,
    arm: str,
    catalog: CooperativeArtifactCatalog,
    adapters: Mapping[str, object],
    feedback: SanitizedEvolutionFeedback,
    step: int,
) -> BundleCandidateV2 | None:
    """Build one typed Child using only sanitized Train feedback.

    The Project 3 prototype admits Decision prompt mutations only. Decision
    Skills and execution settings remain immutable under their full module SHA.
    """
    if type(parent) is not EvolutionBundleV2:
        raise TypeError("proposal Parent must be EvolutionBundleV2")
    if type(catalog) is not CooperativeArtifactCatalog:
        raise TypeError("proposal catalog must be CooperativeArtifactCatalog")
    if not isinstance(adapters, Mapping):
        raise TypeError("proposal adapters must be a mapping")
    if type(step) is not int or step < 0:
        raise ValueError("proposal step must be a non-negative integer")
    _validated_feedback(parent, feedback)

    arms = _SINGLE_ARMS if arm == "joint" else (arm,)
    changes: dict[str, str | tuple[str, str]] = {}
    for scope in arms:
        proposed = _one_change(
            parent=parent,
            arm=scope,
            catalog=catalog,
            adapters=adapters,
            step=step,
        )
        if proposed is not None:
            name, identity = proposed
            changes[name] = identity
        if arm == "joint" and len(changes) == 2:
            break

    if not changes or (arm == "joint" and len(changes) < 2):
        return None
    numerical = changes.get("numerical")
    return BundleCandidateV2(
        schema_version=1,
        target=arm,  # type: ignore[arg-type]
        parent_bundle_sha256=parent.fingerprint(),
        operator=("paired_typed_mutation" if arm == "joint" else "typed_mutation"),
        numerical_release_sha256=(
            numerical[0] if isinstance(numerical, tuple) else None
        ),
        numerical_registry_sha256=(
            numerical[1] if isinstance(numerical, tuple) else None
        ),
        retrieval_release_sha256=(
            changes.get("retrieval")
            if isinstance(changes.get("retrieval"), str)
            else None
        ),
        decision_policy_sha256=(
            changes.get("decision")
            if isinstance(changes.get("decision"), str)
            else None
        ),
    )


__all__ = ["propose_bundle_candidate"]
