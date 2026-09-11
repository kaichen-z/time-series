"""Pure discounted-UCB and deterministic-Thompson scheduler transitions."""
from __future__ import annotations

import hashlib
import math
import random

from .contracts import CooperativeSchedulerStateV2, SchedulerArmStateV2


def select_arm(state: CooperativeSchedulerStateV2) -> str:
    """Select one enabled arm without mutating or advancing scheduler state."""
    if not isinstance(state, CooperativeSchedulerStateV2):
        raise TypeError("select_arm requires a CooperativeSchedulerStateV2")

    for name, arm in state.arms.items():
        if arm.attempts == 0:
            return name

    total_attempts = sum(arm.attempts for arm in state.arms.values())
    best_name: str | None = None
    best_score = -math.inf
    for name, arm in state.arms.items():
        if state.mode == "ucb":
            score = (
                arm.discounted_reward_sum / arm.attempts
                - arm.discounted_cost_sum / arm.attempts
                + math.sqrt(
                    2.0 * math.log(total_attempts + 1.0) / arm.attempts
                )
            )
        else:
            draw_seed = int.from_bytes(
                hashlib.sha256(
                    f"{state.seed}:{state.draw_counter}:{name}".encode()
                ).digest()[:8],
                "big",
            )
            sample = random.Random(draw_seed).betavariate(
                1 + arm.acceptances,
                1 + arm.attempts - arm.acceptances,
            )
            score = sample - arm.discounted_cost_sum / arm.attempts
        if score > best_score:
            best_name = name
            best_score = score

    assert best_name is not None
    return best_name


def record_outcome(
    state: CooperativeSchedulerStateV2,
    arm: str,
    *,
    train_reward: float,
    normalized_cost: float,
    accepted: bool,
) -> CooperativeSchedulerStateV2:
    """Return the next discounted scheduler state for one Train-only outcome."""
    if not isinstance(state, CooperativeSchedulerStateV2):
        raise TypeError("record_outcome requires a CooperativeSchedulerStateV2")
    if type(arm) is not str or arm not in state.arms:
        raise ValueError("outcome arm must be enabled in scheduler state")
    if type(train_reward) is not float or not math.isfinite(train_reward):
        raise ValueError("train_reward must be a finite float")
    if (
        type(normalized_cost) is not float
        or not math.isfinite(normalized_cost)
        or normalized_cost < 0.0
    ):
        raise ValueError("normalized_cost must be a non-negative finite float")
    if type(accepted) is not bool:
        raise ValueError("accepted must be a boolean")

    updated: dict[str, SchedulerArmStateV2] = {}
    for name, previous in state.arms.items():
        reward_sum = previous.discounted_reward_sum * state.discount
        cost_sum = previous.discounted_cost_sum * state.discount
        if name == arm:
            updated[name] = SchedulerArmStateV2(
                attempts=previous.attempts + 1,
                acceptances=previous.acceptances + int(accepted),
                discounted_reward_sum=reward_sum + train_reward,
                discounted_cost_sum=cost_sum + normalized_cost,
            )
        else:
            updated[name] = SchedulerArmStateV2(
                attempts=previous.attempts,
                acceptances=previous.acceptances,
                discounted_reward_sum=reward_sum,
                discounted_cost_sum=cost_sum,
            )

    return CooperativeSchedulerStateV2(
        schema_version=state.schema_version,
        mode=state.mode,
        seed=state.seed,
        draw_counter=state.draw_counter + (state.mode == "thompson"),
        completed_step=state.completed_step + 1,
        discount=state.discount,
        arms=updated,
    )


__all__ = ["record_outcome", "select_arm"]
