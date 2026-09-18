"""Document-conditioned, bounded, auditable forecast adjustment.

The two-stage pipeline (numerical -> retrieval -> decision) can only *select* a
pre-materialized numerical candidate; it can never *modify* the forecast, so
document evidence adds ~nothing over a strong base model (e.g. Toto). This package
introduces an evolvable ``post_adjust`` stage that applies a bounded, gated,
evidence-grounded delta on top of a base forecast, turning the retrieval evidence
(direction / magnitude / window / grounding already produced per chain) into an
actual change to the numbers.
"""
from .post_adjust import (
    EvidenceEffect,
    apply_bounded_delta,
    horizon_window_mask,
    identity_adjust,
    project_evidence,
    reference_future_event_adjust,
)
from .dsl import (
    CALIBRATED_SEEDS,
    EVENT_SCALE,
    EVENT_SCALE_RELAXED_ENTITY,
    GROUNDED_EVENT,
    GROUNDED_EVENT_CAL_CASCADE,
    GROUNDED_EVENT_CAL_HOUR,
    GROUNDED_EVENT_CAL_LOWQ,
    GROUNDED_EVENT_CAL_WEEKEND,
    IDENTITY,
    Action,
    Policy,
    Predicate,
    Rule,
    apply_policy,
    available_regimes,
    policy_to_text,
)

__all__ = [
    "EvidenceEffect",
    "apply_bounded_delta",
    "horizon_window_mask",
    "identity_adjust",
    "project_evidence",
    "reference_future_event_adjust",
    # Level-0 interaction DSL (rules-as-data policy)
    "Predicate",
    "Action",
    "Rule",
    "Policy",
    "apply_policy",
    "policy_to_text",
    "available_regimes",
    "IDENTITY",
    "EVENT_SCALE",
    "EVENT_SCALE_RELAXED_ENTITY",
    "GROUNDED_EVENT",
    "GROUNDED_EVENT_CAL_WEEKEND",
    "GROUNDED_EVENT_CAL_LOWQ",
    "GROUNDED_EVENT_CAL_HOUR",
    "GROUNDED_EVENT_CAL_CASCADE",
    "CALIBRATED_SEEDS",
]
