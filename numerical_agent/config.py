"""Task-specific parameters for dictionary curation."""
from __future__ import annotations

from dataclasses import dataclass


ALLOWED_ACTIONS = ("keep", "revise", "quarantine", "discard")
ALLOWED_FAMILIES = ("statistical", "foundation", "combined")
METHOD_STATUSES = (
    "unimplemented",
    "accepted",
    "specialized",
    "quarantined",
    "unavailable",
    "discarded",
)


@dataclass(frozen=True)
class DictionaryCurationConfig:
    """Mutable task policy kept outside the generic controller."""

    allowed_actions: tuple[str, ...] = ALLOWED_ACTIONS
    # Phase 1 materializes classical methods in the Python sandbox.  Foundation
    # and combined cards require dedicated runtimes and must be opted in explicitly.
    allowed_families: tuple[str, ...] = ("statistical",)
    max_revisions_per_method: int = 1
    max_implementation_attempts: int = 3
    method_statuses: tuple[str, ...] = METHOD_STATUSES
    method_metric: str = "smae"
    dictionary_metric: str = "smae"
    discard_requires_dominance_evidence: bool = True
    allow_dev_learning: bool = False
    accepted_max_error: float = 1.0
    specialized_max_error: float = 2.0
    min_success_rate: float = 0.8
    selection_folds: int = 3
    selection_horizon: int = 8

    def __post_init__(self) -> None:
        if self.max_revisions_per_method < 0:
            raise ValueError("max_revisions_per_method must be non-negative")
        if self.max_implementation_attempts <= 0:
            raise ValueError("max_implementation_attempts must be positive")
        if not self.allowed_actions or not set(self.allowed_actions).issubset(ALLOWED_ACTIONS):
            raise ValueError("allowed_actions contains an unsupported action")
        if not self.allowed_families or not set(self.allowed_families).issubset(ALLOWED_FAMILIES):
            raise ValueError("allowed_families contains an unsupported family")
        if not self.method_statuses or not set(self.method_statuses).issubset(METHOD_STATUSES):
            raise ValueError("method_statuses contains an unsupported status")
        if not self.method_metric or not self.dictionary_metric:
            raise ValueError("metric names must not be empty")
        if self.accepted_max_error < 0 or self.specialized_max_error < 0:
            raise ValueError("status thresholds must be non-negative")
        if self.specialized_max_error < self.accepted_max_error:
            raise ValueError("specialized_max_error must not be below accepted_max_error")
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be between 0 and 1")
        if self.selection_folds <= 0 or self.selection_horizon <= 0:
            raise ValueError("selection_folds and selection_horizon must be positive")
        if self.allow_dev_learning:
            raise ValueError("allow_dev_learning must remain false")
