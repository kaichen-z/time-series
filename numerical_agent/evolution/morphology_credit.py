"""Trusted Train-only post-forecast credit for immutable morphology cards."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from common.metrics import drcik_point_metrics, joint_scaled_error

from .morphology import MorphologyCard


@dataclass(frozen=True)
class ToolCallCredit:
    """The signed metric change attributable to one newly available grounded call."""

    call_id: str
    available_call_ids: frozenset[str]
    smae_improvement: float
    srmse_improvement: float
    joint_improvement: float

    @property
    def reward(self) -> float:
        """Compatibility-only alias for the read-only joint diagnostic."""
        return self.joint_improvement


@dataclass(frozen=True)
class MorphologyCreditTrace:
    """An auditable post-forecast Train trace; it never edits the source card."""

    split: str
    learning_enabled: bool
    reason: str
    card_fingerprint: str
    baseline_smae: float | None
    baseline_srmse: float | None
    credits: tuple[ToolCallCredit, ...]


def assign_tool_call_credit(
    card: MorphologyCard,
    *,
    split: str,
    future_truth: Sequence[float] | object,
    forecasts_by_call_ids: Mapping[Iterable[str], Sequence[float]],
) -> MorphologyCreditTrace:
    """Assign Train-only marginal score changes to already-grounded tool calls.

    The host supplies all forecast trajectories; this evaluator only scores them and never
    creates a trajectory, changes a card, or admits a non-Train split into learning.
    """
    if not isinstance(card, MorphologyCard):
        raise ValueError("card must be a MorphologyCard")
    if not isinstance(split, str):
        raise ValueError("split must be a string")
    if split != "train":
        # Do this before inspecting labels or forecast containers: frozen/public/dev/hidden
        # evaluation paths must remain write-free even if their values are hostile.
        return MorphologyCreditTrace(
            split=split,
            learning_enabled=False,
            reason="non_train_split",
            card_fingerprint=card.fingerprint,
            baseline_smae=None,
            baseline_srmse=None,
            credits=(),
        )
    if not isinstance(forecasts_by_call_ids, Mapping):
        raise ValueError("forecasts_by_call_ids must be a mapping")
    call_ids = _grounded_call_ids(card)
    paths = _normalize_forecasts(forecasts_by_call_ids, allowed_call_ids=frozenset(call_ids))
    truth = _finite_vector(future_truth, "future_truth")
    required = [frozenset(call_ids[:index]) for index in range(len(call_ids) + 1)]
    if any(path not in paths for path in required):
        raise ValueError("missing forecast for grounded call prefix")
    if any(len(paths[path]) != len(truth) for path in required):
        raise ValueError("forecast horizon must match future_truth")

    prior = _metrics(truth, paths[frozenset()])
    credits: list[ToolCallCredit] = []
    for index, call_id in enumerate(call_ids, start=1):
        available = frozenset(call_ids[:index])
        current = _metrics(truth, paths[available])
        smae_improvement = prior["smae"] - current["smae"]
        srmse_improvement = prior["srmse"] - current["srmse"]
        credits.append(
            ToolCallCredit(
                call_id=call_id,
                available_call_ids=available,
                smae_improvement=smae_improvement,
                srmse_improvement=srmse_improvement,
                joint_improvement=(
                    joint_scaled_error(prior["smae"], prior["srmse"])
                    - joint_scaled_error(current["smae"], current["srmse"])
                ),
            )
        )
        prior = current
    baseline = _metrics(truth, paths[frozenset()])
    return MorphologyCreditTrace(
        split="train",
        learning_enabled=True,
        reason="train_post_forecast_credit",
        card_fingerprint=card.fingerprint,
        baseline_smae=baseline["smae"],
        baseline_srmse=baseline["srmse"],
        credits=tuple(credits),
    )


def _grounded_call_ids(card: MorphologyCard) -> tuple[str, ...]:
    cited = {
        call_id
        for assumption in card.assumptions
        for call_id in assumption.supporting_call_ids
    }
    return tuple(call.call_id for call in card.tool_calls if call.call_id in cited)


def _normalize_forecasts(
    raw: Mapping[Iterable[str], Sequence[float]], *, allowed_call_ids: frozenset[str]
) -> dict[frozenset[str], tuple[float, ...]]:
    normalized: dict[frozenset[str], tuple[float, ...]] = {}
    try:
        items = raw.items()
        for key, forecast in items:
            if isinstance(key, (str, bytes)):
                raise ValueError("forecast call-id key must be a collection")
            call_ids = frozenset(key)
            if any(not isinstance(call_id, str) or not call_id for call_id in call_ids):
                raise ValueError("forecast call-id key contains an invalid call id")
            if not call_ids <= allowed_call_ids:
                raise ValueError("forecast call-id key contains an ungrounded call id")
            if call_ids in normalized:
                raise ValueError("duplicate normalized forecast call-id key")
            normalized[call_ids] = _finite_vector(forecast, "forecast")
    except (AttributeError, TypeError) as exc:
        raise ValueError("forecasts_by_call_ids contains an invalid container") from exc
    return normalized


def _finite_vector(value: object, field: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{field} must be a nonempty finite sequence")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a nonempty finite sequence") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must be a nonempty finite sequence")
    return result


def _metrics(truth: tuple[float, ...], forecast: tuple[float, ...]) -> dict[str, float]:
    raw = drcik_point_metrics(truth, forecast)
    return {"smae": float(raw["smae"]), "srmse": float(raw["srmse"])}
