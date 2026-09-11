from __future__ import annotations

from evolving_loop.v2.cooperative.contracts import (
    CooperativeSchedulerStateV2,
    SchedulerArmStateV2,
)
from evolving_loop.v2.cooperative.schedulers import record_outcome, select_arm


def state_for(
    mode: str,
    *,
    arms: tuple[str, ...] = ("numerical", "retrieval"),
    seed: int = 7,
) -> CooperativeSchedulerStateV2:
    return CooperativeSchedulerStateV2(
        schema_version=1,
        mode=mode,
        seed=seed,
        draw_counter=0,
        completed_step=0,
        discount=0.9,
        arms={name: SchedulerArmStateV2(0, 0, 0.0, 0.0) for name in arms},
    )


def exercised_state(mode: str, *, seed: int) -> CooperativeSchedulerStateV2:
    state = state_for(mode, arms=("numerical", "retrieval", "decision"), seed=seed)
    for arm, reward, cost, accepted in (
        ("numerical", 0.5, 0.2, True),
        ("retrieval", -0.1, 0.1, False),
        ("decision", 0.2, 0.4, False),
    ):
        state = record_outcome(
            state,
            arm,
            train_reward=reward,
            normalized_cost=cost,
            accepted=accepted,
        )
    return state


def test_ucb_visits_untried_arms_in_canonical_order():
    state = state_for("ucb", arms=("numerical", "retrieval", "decision", "joint"))
    selected = []
    for _ in range(4):
        arm = select_arm(state)
        selected.append(arm)
        state = record_outcome(
            state, arm, train_reward=0.0, normalized_cost=0.25, accepted=False
        )
    assert selected == ["numerical", "retrieval", "decision", "joint"]


def test_thompson_also_visits_untried_arms_before_sampling():
    state = state_for("thompson", arms=("retrieval", "joint"))
    assert select_arm(state) == "retrieval"
    state = record_outcome(
        state, "retrieval", train_reward=0.0, normalized_cost=0.0, accepted=False
    )
    assert select_arm(state) == "joint"


def test_thompson_resume_repeats_the_same_next_draw():
    state = exercised_state("thompson", seed=17)
    restored = CooperativeSchedulerStateV2.from_payload(state.to_payload())
    assert select_arm(restored) == select_arm(state)


def test_ucb_uses_the_specified_total_plus_one_exploration_formula():
    state = CooperativeSchedulerStateV2(
        1,
        "ucb",
        3,
        0,
        5,
        0.9,
        {
            "numerical": SchedulerArmStateV2(1, 0, 0.0, 0.0),
            "retrieval": SchedulerArmStateV2(4, 0, 4.68, 1.0),
        },
    )
    assert select_arm(state) == "numerical"


def test_ucb_breaks_an_exact_post_warmup_tie_by_canonical_arm_order():
    state = CooperativeSchedulerStateV2(
        1,
        "ucb",
        3,
        0,
        2,
        0.9,
        {
            "retrieval": SchedulerArmStateV2(1, 0, 0.25, 0.1),
            "decision": SchedulerArmStateV2(1, 0, 0.25, 0.1),
        },
    )
    assert select_arm(state) == "retrieval"


def test_record_outcome_discounts_every_arm_and_updates_only_selected_counts():
    state = CooperativeSchedulerStateV2(
        1,
        "ucb",
        3,
        0,
        2,
        0.5,
        {
            "numerical": SchedulerArmStateV2(1, 1, 2.0, 1.0),
            "decision": SchedulerArmStateV2(1, 0, 4.0, 2.0),
        },
    )
    updated = record_outcome(
        state,
        "decision",
        train_reward=0.25,
        normalized_cost=0.75,
        accepted=True,
    )
    assert updated.arms["numerical"] == SchedulerArmStateV2(1, 1, 1.0, 0.5)
    assert updated.arms["decision"] == SchedulerArmStateV2(2, 1, 2.25, 1.75)
    assert updated.completed_step == 3
    assert updated.draw_counter == 0
    assert state.arms["decision"] == SchedulerArmStateV2(1, 0, 4.0, 2.0)


def test_thompson_record_advances_one_draw_counter():
    state = state_for("thompson")
    updated = record_outcome(
        state,
        "numerical",
        train_reward=0.0,
        normalized_cost=0.0,
        accepted=False,
    )
    assert updated.draw_counter == state.draw_counter + 1


def test_selection_does_not_mutate_state():
    state = exercised_state("ucb", seed=11)
    before = state.canonical_bytes()
    assert select_arm(state) in state.arms
    assert state.canonical_bytes() == before
