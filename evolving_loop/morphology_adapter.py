"""Retrieval-side bridge for sanitized Numerical morphology assumptions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from evolving_loop.data import ContextTask
from evolving_loop.retrieval_agent.schemas import RetrievalAssumption, RetrievalContractError


@runtime_checkable
class MorphologyProvider(Protocol):
    """Small host boundary required by the two-stage Retrieval topology."""

    def assumptions(self, task: ContextTask) -> tuple[RetrievalAssumption, ...]: ...


@dataclass(frozen=True)
class MorphologyAdapter:
    """Adapt a Numerical reasoner's Morphology Card without importing its package."""

    numerical: object

    def __post_init__(self) -> None:
        if not callable(getattr(self.numerical, "run", None)):
            raise RetrievalContractError("Numerical morphology reasoner requires run(task)")

    def assumptions(self, task: ContextTask) -> tuple[RetrievalAssumption, ...]:
        run = getattr(self.numerical, "run", None)
        assert callable(run)
        card = run(task.numeric_view())
        raw_assumptions = (
            card.get("assumptions")
            if isinstance(card, Mapping)
            else getattr(card, "assumptions", None)
        )
        if isinstance(raw_assumptions, (str, bytes)) or not isinstance(
            raw_assumptions, Sequence
        ):
            raise RetrievalContractError("Morphology Card requires assumptions")
        parsed = tuple(self._assumption(item) for item in raw_assumptions)
        identities = [item.assumption_id for item in parsed]
        if len(identities) != len(set(identities)):
            raise RetrievalContractError("duplicate Morphology assumption_id")
        return parsed

    @staticmethod
    def _assumption(raw: object) -> RetrievalAssumption:
        if isinstance(raw, Mapping):
            get = raw.get
        else:
            get = lambda field: getattr(raw, field, None)
        return RetrievalAssumption.from_payload(
            {
                "assumption_id": get("assumption_id"),
                "kind": get("kind"),
                "claim": get("claim"),
                "failure_condition": get("failure_condition"),
            }
        )


__all__ = ["MorphologyAdapter", "MorphologyProvider"]
