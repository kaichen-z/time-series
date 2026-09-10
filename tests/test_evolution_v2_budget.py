from __future__ import annotations

import copy
import hashlib
import math

import pytest

from evolving_loop.v2 import (
    BudgetContractError,
    BudgetLedger,
    BudgetPlan,
    EvolutionV2Config,
    ResourceUse,
)


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def sha256_for(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def config(*, profile: str = "pilot") -> EvolutionV2Config:
    hard_limit = 14_400 if profile == "formal" else 100
    reserve = 0.2 if profile == "formal" else 0.1
    return EvolutionV2Config.from_payload(
        {
            "schema_version": 1,
            "profile": profile,
            "seed": 7,
            "scheduler": "ucb",
            "enabled_mutation_scopes": ["numerical"],
            "archive_capacities": {"numerical": 2},
            "hyperband": {"resource_levels": [1]},
            "runtime_fingerprints": {"python": sha256_for("python")},
            "kernel_protocol": {
                field: sha256_for(field)
                for field in (
                    "task_materializer",
                    "split_manifest",
                    "metric_policy",
                    "label_firewall",
                    "artifact_validator",
                    "sandbox_policy",
                    "promotion_policy",
                )
            },
            "hard_limit_seconds": hard_limit,
            "finalization_reserve_fraction": reserve,
            "runner": "deterministic_fake",
        }
    )


def generous_use(**overrides: int | float) -> ResourceUse:
    values: dict[str, int | float] = {
        "wall_seconds": 100.0,
        "task_executions": 100,
        "llm_calls": 100,
        "input_tokens": 100,
        "output_tokens": 100,
        "gpu_seconds": 100.0,
        "subprocesses": 100,
        "artifact_bytes": 100,
    }
    values.update(overrides)
    return ResourceUse(**values)


def plan(**ceiling_overrides: int | float) -> BudgetPlan:
    return BudgetPlan.from_config(config(), ceilings=generous_use(**ceiling_overrides))


def test_formal_budget_reserves_last_twenty_percent():
    formal = BudgetPlan.from_config(config(profile="formal"), ceilings=generous_use())
    clock = FakeClock(0.0)
    ledger = BudgetLedger(formal, monotonic=clock)

    assert ledger.plan.hard_limit_seconds == 14_400
    assert ledger.plan.search_deadline_seconds == 11_520
    clock.advance(11_519)
    assert ledger.can_open_stage(ResourceUse(wall_seconds=1.0)).allowed is True
    clock.advance(1)
    denial = ledger.can_open_stage(ResourceUse(wall_seconds=1.0))
    assert denial.allowed is False
    assert denial.reason == "finalization_reserve"


@pytest.mark.parametrize(
    "field, ceiling, charged, estimate",
    [
        ("wall_seconds", 8.0, 7.0, 2.0),
        ("task_executions", 8, 7, 2),
        ("llm_calls", 8, 7, 2),
        ("input_tokens", 8, 7, 2),
        ("output_tokens", 8, 7, 2),
        ("gpu_seconds", 8.0, 7.0, 2.0),
        ("subprocesses", 8, 7, 2),
        ("artifact_bytes", 8, 7, 2),
    ],
)
def test_stage_must_fit_every_resource_ceiling(field, ceiling, charged, estimate):
    ledger = BudgetLedger(plan(**{field: ceiling}), monotonic=FakeClock())
    ledger.charge(ResourceUse(**{field: charged}))

    denial = ledger.can_open_stage(ResourceUse(**{field: estimate}))

    assert denial.allowed is False
    assert denial.reason == f"{field}_exhausted"


def test_resource_use_rejects_boolean_negative_and_nonfinite_values():
    for field in (
        "task_executions",
        "llm_calls",
        "input_tokens",
        "output_tokens",
        "subprocesses",
        "artifact_bytes",
    ):
        with pytest.raises(ValueError, match=field):
            ResourceUse(**{field: True})
        with pytest.raises(ValueError, match=field):
            ResourceUse(**{field: -1})

    for field in ("wall_seconds", "gpu_seconds"):
        with pytest.raises(ValueError, match=field):
            ResourceUse(**{field: 1})
        for invalid in (-1.0, math.inf, -math.inf, math.nan):
            with pytest.raises(ValueError, match=field):
                ResourceUse(**{field: invalid})


def test_budget_plan_is_immutable_and_fingerprint_binds_every_ceiling():
    baseline = plan()
    with pytest.raises(AttributeError):
        baseline.hard_limit_seconds = 1

    for field in ResourceUse.field_names():
        value = 99.0 if field in {"wall_seconds", "gpu_seconds"} else 99
        assert plan(**{field: value}).fingerprint() != baseline.fingerprint()


def test_pending_reservations_count_against_every_ceiling():
    for field in ResourceUse.field_names():
        ceiling = 3.0 if field in {"wall_seconds", "gpu_seconds"} else 3
        estimate = 2.0 if field in {"wall_seconds", "gpu_seconds"} else 2
        ledger = BudgetLedger(plan(**{field: ceiling}), monotonic=FakeClock())
        first = ledger.reserve_stage("first", ResourceUse(**{field: estimate}))

        denial = ledger.reserve_stage("second", ResourceUse(**{field: estimate}))

        assert first.allowed is True
        assert first.reservation_sha256 is not None
        assert denial.allowed is False
        assert denial.reason == f"{field}_exhausted"


def test_charge_counts_pending_reservations_and_checkpoints_exhaustion():
    ledger = BudgetLedger(plan(task_executions=3), monotonic=FakeClock())
    assert ledger.reserve_stage("pending", ResourceUse(task_executions=3)).allowed

    result = ledger.charge(ResourceUse(task_executions=1))

    assert result.allowed is False
    assert result.reason == "task_executions_exhausted"
    resumed = BudgetLedger.resume(
        ledger.plan, ledger.checkpoint(), monotonic=FakeClock()
    )
    assert resumed.can_open_stage(ResourceUse()).reason == "task_executions_exhausted"


def test_stage_can_end_exactly_at_deadline_but_cannot_open_at_deadline():
    clock = FakeClock()
    ledger = BudgetLedger(plan(), monotonic=clock)
    clock.advance(89.0)
    reservation = ledger.reserve_stage("last", ResourceUse(wall_seconds=1.0))
    assert reservation.allowed is True

    clock.advance(1.0)
    closed = ledger.close_stage(reservation, ResourceUse(wall_seconds=1.0))
    assert closed.allowed is True
    assert ledger.can_open_stage(ResourceUse()).reason == "finalization_reserve"


def test_close_releases_unused_reservation_and_charges_actual_use():
    ledger = BudgetLedger(plan(task_executions=3), monotonic=FakeClock())
    reservation = ledger.reserve_stage("first", ResourceUse(task_executions=3))

    closed = ledger.close_stage(reservation, ResourceUse(task_executions=1))

    assert closed.allowed is True
    assert ledger.charged_use.task_executions == 1
    assert ledger.can_open_stage(ResourceUse(task_executions=2)).allowed is True


def test_overrun_closes_reservation_charges_actual_and_exhausts_ledger():
    ledger = BudgetLedger(plan(task_executions=4), monotonic=FakeClock())
    reservation = ledger.reserve_stage("overrun", ResourceUse(task_executions=2))

    result = ledger.close_stage(reservation, ResourceUse(task_executions=5))

    assert result.allowed is False
    assert result.reason == "budget_overrun"
    assert ledger.charged_use.task_executions == 5
    assert ledger.can_open_stage(ResourceUse()).reason == "budget_overrun"
    assert ledger.reserve_stage("later", ResourceUse()).reason == "budget_overrun"


def test_invalid_actual_does_not_close_a_valid_reservation():
    ledger = BudgetLedger(plan(), monotonic=FakeClock())
    reservation = ledger.reserve_stage("stage", ResourceUse(task_executions=2))

    for invalid in (
        {"task_executions": -1},
        {"task_executions": True},
        {"wall_seconds": math.inf},
    ):
        with pytest.raises(ValueError, match="actual"):
            ledger.close_stage(reservation, invalid)

    assert ledger.close_stage(reservation, ResourceUse(task_executions=1)).allowed


def test_double_close_is_a_typed_closed_stage_denial():
    ledger = BudgetLedger(plan(), monotonic=FakeClock())
    reservation = ledger.reserve_stage("once", ResourceUse(task_executions=1))
    assert ledger.close_stage(reservation, ResourceUse(task_executions=1)).allowed

    denial = ledger.close_stage(reservation, ResourceUse())

    assert denial.allowed is False
    assert denial.reason == "stage_closed"


def test_double_close_remains_typed_after_checkpoint_resume():
    ledger = BudgetLedger(plan(), monotonic=FakeClock())
    reservation = ledger.reserve_stage("once", ResourceUse(task_executions=1))
    assert ledger.close_stage(reservation, ResourceUse(task_executions=1)).allowed

    resumed = BudgetLedger.resume(
        ledger.plan, ledger.checkpoint(), monotonic=FakeClock()
    )
    denial = resumed.close_stage(reservation, ResourceUse())

    assert denial.allowed is False
    assert denial.reason == "stage_closed"


def test_denied_and_completed_stage_ids_can_never_reopen():
    ledger = BudgetLedger(plan(task_executions=1), monotonic=FakeClock())
    denied = ledger.reserve_stage("denied", ResourceUse(task_executions=2))
    assert denied.reason == "task_executions_exhausted"
    assert ledger.reserve_stage("denied", ResourceUse()).reason == "stage_closed"

    accepted = ledger.reserve_stage("completed", ResourceUse(task_executions=1))
    assert ledger.close_stage(accepted, ResourceUse(task_executions=1)).allowed
    assert ledger.reserve_stage("completed", ResourceUse()).reason == "stage_closed"


def test_duplicate_open_stage_id_is_rejected_without_destroying_reservation():
    ledger = BudgetLedger(plan(), monotonic=FakeClock())
    reservation = ledger.reserve_stage("open", ResourceUse(task_executions=1))

    denial = ledger.reserve_stage("open", ResourceUse())

    assert denial.reason == "stage_already_open"
    assert ledger.close_stage(reservation, ResourceUse(task_executions=1)).allowed


def test_begin_finalization_permanently_closes_search_but_allows_close():
    ledger = BudgetLedger(plan(), monotonic=FakeClock())
    reservation = ledger.reserve_stage("open", ResourceUse(task_executions=1))

    ledger.begin_finalization()

    assert ledger.reserve_stage("later", ResourceUse()).reason == "finalization_started"
    assert ledger.close_stage(reservation, ResourceUse(task_executions=1)).allowed


def test_checkpoint_round_trip_adds_prior_elapsed_to_fresh_monotonic_origin():
    clock = FakeClock(10.0)
    original = BudgetLedger(plan(), monotonic=clock)
    first = original.reserve_stage("open", ResourceUse(task_executions=2))
    assert first.allowed
    original.charge(ResourceUse(llm_calls=3))
    clock.advance(40.0)
    checkpoint = original.checkpoint()

    resumed_clock = FakeClock(1_000.0)
    resumed = BudgetLedger.resume(original.plan, checkpoint, monotonic=resumed_clock)

    assert resumed.elapsed_wall_seconds == 40.0
    assert resumed.charged_use.llm_calls == 3
    assert resumed.reserve_stage("open", ResourceUse()).reason == "stage_already_open"
    resumed_clock.advance(49.0)
    assert resumed.can_open_stage(ResourceUse(wall_seconds=1.0)).allowed
    resumed_clock.advance(1.0)
    assert resumed.can_open_stage(ResourceUse()).reason == "finalization_reserve"


def test_checkpoint_preserves_closed_ids_finalization_and_exhaustion():
    first = BudgetLedger(plan(), monotonic=FakeClock())
    assert (
        first.reserve_stage("denied", ResourceUse(task_executions=101)).allowed is False
    )
    completed = first.reserve_stage("completed", ResourceUse())
    assert first.close_stage(completed, ResourceUse()).allowed
    first.begin_finalization()
    resumed = BudgetLedger.resume(first.plan, first.checkpoint(), monotonic=FakeClock())
    assert resumed.reserve_stage("denied", ResourceUse()).reason == "stage_closed"
    assert resumed.reserve_stage("completed", ResourceUse()).reason == "stage_closed"
    assert resumed.reserve_stage("new", ResourceUse()).reason == "finalization_started"

    second = BudgetLedger(plan(), monotonic=FakeClock())
    overrun = second.reserve_stage("overrun", ResourceUse(task_executions=1))
    second.close_stage(overrun, ResourceUse(task_executions=2))
    exhausted = BudgetLedger.resume(
        second.plan, second.checkpoint(), monotonic=FakeClock()
    )
    assert exhausted.can_open_stage(ResourceUse()).reason == "budget_overrun"


def test_checkpoint_rejects_tampering_and_plan_mismatch():
    ledger = BudgetLedger(plan(), monotonic=FakeClock())
    reservation = ledger.reserve_stage("open", ResourceUse(task_executions=2))
    assert reservation.allowed
    checkpoint = ledger.checkpoint()

    tampered = copy.deepcopy(checkpoint)
    tampered["charged_use"]["llm_calls"] = 99
    with pytest.raises(BudgetContractError, match="checkpoint SHA"):
        BudgetLedger.resume(ledger.plan, tampered, monotonic=FakeClock())

    different_plan = plan(llm_calls=99)
    with pytest.raises(BudgetContractError, match="plan SHA"):
        BudgetLedger.resume(different_plan, checkpoint, monotonic=FakeClock())


def test_checkpoint_rejects_a_tampered_open_reservation_even_with_new_checksum():
    ledger = BudgetLedger(plan(), monotonic=FakeClock())
    assert ledger.reserve_stage("open", ResourceUse(task_executions=2)).allowed
    checkpoint = ledger.checkpoint()
    checkpoint["open_reservations"][0]["estimate"]["task_executions"] = 3
    checkpoint["checkpoint_sha256"] = BudgetLedger.checkpoint_sha256(checkpoint)

    with pytest.raises(BudgetContractError, match="reservation SHA"):
        BudgetLedger.resume(ledger.plan, checkpoint, monotonic=FakeClock())


def test_checkpoint_validates_closed_reservation_hashes():
    ledger = BudgetLedger(plan(), monotonic=FakeClock())
    reservation = ledger.reserve_stage("closed", ResourceUse())
    assert ledger.close_stage(reservation, ResourceUse()).allowed
    checkpoint = ledger.checkpoint()
    checkpoint["closed_reservation_sha256s"][0] = "not-a-sha"
    checkpoint["checkpoint_sha256"] = BudgetLedger.checkpoint_sha256(checkpoint)

    with pytest.raises(BudgetContractError, match="closed reservation SHA"):
        BudgetLedger.resume(ledger.plan, checkpoint, monotonic=FakeClock())
