"""Build-only Champion evolution controller regressions."""
from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

import numerical_agent.evolution.champion_controller as controller_module
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
    champion_fingerprint,
)
from numerical_agent.evolution.champion_controller import (
    ChampionControllerError,
    ChampionEvolutionConfig,
    run_build_evolution,
)
from numerical_agent.evolution.champion_evidence import (
    ChampionGateConfig,
    ChampionHistoryDiagnostic,
    ChampionTaskRow,
    ProposerEvidence,
)
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.screening import TaskProfile


def _profile(task_id: str, index: int) -> TaskProfile:
    return TaskProfile(
        task_id=task_id,
        frequency="D",
        history_length=100 + index,
        horizon=2,
        zero_fraction=0.0,
        signed=False,
        integer_valued=False,
        trend_direction="up",
        trend_strength=0.5,
        periodicity_periods=(7,),
        periodicity_strength=float(index % 8) / 8.0,
        periodicity_confidence=0.75,
        outlier_fraction=0.0,
        noise_relative_scale=0.2,
        likely_stationary=False,
        stationarity_score=0.25,
        recent_regime_start=None,
        recent_regime_confidence=0.1,
        intermittency_adi=1.0,
        intermittency_cv2=0.1,
    )


def _parent() -> ChampionRelease:
    recipe = ChampionRecipe(
        name="active_parent",
        kind="select",
        parents=("baseline_leaf",),
        fallback_parent="baseline_leaf",
        assumptions=(
            EvolutionAssumption(
                assumption_id="parent_history",
                candidate_name="baseline_leaf",
                feature="history_length",
                direction="above",
                horizon_region="full",
                operator="select",
                rationale="History length supports the baseline forecast.",
                failure_condition="History length no longer supports the baseline forecast.",
            ),
        ),
    )
    return ChampionRelease(
        policy=FittedChampionPolicy(
            recipe=recipe,
            thresholds=(("parent_history", 0.0),),
        ),
        source_hashes=(("baseline_leaf", "0" * 64),),
        metric_policy_fingerprint="1" * 64,
        lineage=("active_parent",),
    )


def _history_diagnostic(
    name: str, *, eligible: bool = True
) -> ChampionHistoryDiagnostic:
    return ChampionHistoryDiagnostic.from_candidate(
        CandidateDiagnostics.synthetic(
            name=name,
            family="statistical",
            median_mase=0.5,
            eligible=eligible,
        )
    )


def _rows(
    count: int = 64, *, enriched: bool = False
) -> tuple[ChampionTaskRow, ...]:
    errors = {
        "baseline_leaf": 2.0,
        "better_a": 0.8,
        "better_b": 0.9,
        "better_c": 1.0,
        "better_d": 1.1,
        "better_e": 1.2,
        "worse_a": 2.8,
        "worse_b": 2.9,
        "worse_c": 3.0,
        "worse_d": 3.1,
        "worse_e": 3.2,
    }
    rows: list[ChampionTaskRow] = []
    for index in range(count):
        task_id = f"build_case_{index:03d}"
        profile = _profile(task_id, index)
        truth = (10.0 + index, 12.0 + index)
        history = (10.0,) * profile.history_length if enriched else None
        for candidate_name, error in errors.items():
            rows.append(
                ChampionTaskRow(
                    task_id=task_id,
                    candidate_name=candidate_name,
                    profile=profile,
                    truth=truth,
                    forecast=tuple(value + error for value in truth),
                    fold=index % 5,
                    split="build",
                    history=history,
                    diagnostic=(
                        _history_diagnostic(candidate_name) if enriched else None
                    ),
                )
            )
    return tuple(rows)


def _config(*, generations: int = 1) -> ChampionEvolutionConfig:
    task_ids = tuple(f"build_case_{index:03d}" for index in range(64))
    return ChampionEvolutionConfig(
        generations=generations,
        screen_task_ids=(task_ids[:8], task_ids[:32], task_ids),
        gate_config=ChampionGateConfig(minimum_improved_folds=0),
    )


def _proposal_batch(prefix: str, generation: int = 0) -> tuple[ChampionRecipe, ...]:
    return tuple(
        ChampionRecipe(
            name=f"proposal_{generation}_{index}",
            kind="select",
            parents=(f"{prefix}_{letter}",),
            fallback_parent=f"{prefix}_{letter}",
            assumptions=(
                EvolutionAssumption(
                    assumption_id=f"history_{generation}_{index}",
                    candidate_name=f"{prefix}_{letter}",
                    feature="history_length",
                    direction="above",
                    horizon_region="full",
                    operator="select",
                    rationale="History length supports the candidate forecast.",
                    failure_condition="History length stops supporting the candidate forecast.",
                ),
            ),
        )
        for index, letter in enumerate("abcde")
    )


@dataclass
class RecordingProposer:
    prefix: str
    calls: int = 0

    def __call__(
        self,
        parent: ChampionRelease,
        evidence: ProposerEvidence,
    ) -> tuple[ChampionRecipe, ...]:
        assert parent is PARENT
        assert evidence.label == "adaptive_train_build_diagnostic"
        generation = self.calls
        self.calls += 1
        return _proposal_batch(self.prefix, generation)


@dataclass
class DiverseProposer:
    def __call__(
        self,
        parent: ChampionRelease,
        evidence: ProposerEvidence,
    ) -> tuple[ChampionRecipe, ...]:
        assert parent is PARENT

        def assumption(index: int, kind: str, candidate: str) -> EvolutionAssumption:
            return EvolutionAssumption(
                assumption_id=f"diverse_history_{index}",
                candidate_name=candidate,
                feature="history_length",
                direction="above",
                horizon_region="full",
                operator=kind,  # type: ignore[arg-type]
                rationale="History length supports the candidate forecast.",
                failure_condition="History length stops supporting the candidate forecast.",
            )

        recipes: list[ChampionRecipe] = []
        for index, letter in enumerate("abc"):
            candidate = f"better_{letter}"
            recipes.append(
                ChampionRecipe(
                    name=f"select_proposal_{index}",
                    kind="select",
                    parents=(candidate,),
                    fallback_parent=candidate,
                    assumptions=(assumption(index, "select", candidate),),
                )
            )
        for index, kind in enumerate(("weighted", "median"), start=3):
            recipes.append(
                ChampionRecipe(
                    name=f"{kind}_proposal",
                    kind=kind,  # type: ignore[arg-type]
                    parents=("better_d", "better_e"),
                    fallback_parent="better_d",
                    assumptions=(assumption(index, kind, "better_d"),),
                )
            )
        return tuple(recipes)


@dataclass
class AllOperatorProposer:
    def __call__(
        self,
        parent: ChampionRelease,
        evidence: ProposerEvidence,
    ) -> tuple[ChampionRecipe, ...]:
        kinds = (
            "select",
            "route",
            "horizon_route",
            "weighted",
            "median",
            "bounded_overlay",
        )
        recipes: list[ChampionRecipe] = []
        for index, kind in enumerate(kinds):
            parents = ("better_a",) if kind == "select" else (
                "baseline_leaf",
                "better_a",
            )
            recipes.append(
                ChampionRecipe(
                    name=f"all_operator_{kind}",
                    kind=kind,  # type: ignore[arg-type]
                    parents=parents,
                    fallback_parent=parents[0],
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"operator_history_{index}",
                            candidate_name=parents[-1],
                            feature="periodicity_confidence",
                            direction="above",
                            horizon_region="full",
                            operator=kind,  # type: ignore[arg-type]
                            rationale="History diagnostics support this operator.",
                            failure_condition="History diagnostics stop supporting this operator.",
                        ),
                    ),
                )
            )
        return tuple(recipes)


PARENT = _parent()
ROWS_64 = _rows()


def test_successive_halving_runs_fixed_8_32_64_schedule() -> None:
    result = run_build_evolution(PARENT, ROWS_64, RecordingProposer("better"), _config())

    assert result.generations[0].stage_counts == (8, 32, 64)
    assert result.generations[0].full_build_children <= 3


def test_rejected_child_feedback_does_not_change_the_mutation_parent() -> None:
    parent = _parent()
    proposer = RecordingProposer("worse")
    global PARENT
    previous = PARENT
    PARENT = parent
    try:
        result = run_build_evolution(parent, ROWS_64, proposer, _config(generations=2))
    finally:
        PARENT = previous

    assert all(
        generation.mutation_parent_sha256 == champion_fingerprint(parent)
        for generation in result.generations
    )
    assert proposer.calls == 2
    assert result.active_parent is parent


def test_build_feedback_is_labeled_non_independent() -> None:
    result = run_build_evolution(PARENT, ROWS_64, RecordingProposer("better"), _config())

    assert result.score_label == "adaptive_train_build_diagnostic"
    assert result.independent_generalization_claim is False
    assert all(
        attempt.score_label == "adaptive_train_build_diagnostic"
        and attempt.independent_generalization_claim is False
        for attempt in result.generations[0].attempts
    )
    assert len(result.generations[0].feedback.comparisons) == len(
        result.generations[0].attempts
    )


def test_threshold_defaults_and_membership_are_fingerprinted() -> None:
    default = ChampionEvolutionConfig()
    candidate_changed = replace(default, candidate_minimum_gain=0.006)
    target_changed = replace(default, research_target_gain=0.051)

    assert default.candidate_minimum_gain == 0.005
    assert default.research_target_gain == 0.05
    assert default.fingerprint != candidate_changed.fingerprint
    assert default.fingerprint != target_changed.fingerprint
    assert _config().fingerprint != _config().screen_membership_sha256


def test_candidate_threshold_gates_but_research_target_does_not() -> None:
    high_target = replace(_config(), research_target_gain=0.99)
    high_candidate = replace(
        _config(), candidate_minimum_gain=0.99, research_target_gain=1.0
    )

    target_result = run_build_evolution(
        PARENT, ROWS_64, RecordingProposer("better"), high_target
    )
    candidate_result = run_build_evolution(
        PARENT, ROWS_64, RecordingProposer("better"), high_candidate
    )

    assert target_result.shortlist
    assert candidate_result.shortlist == ()


def test_halving_preserves_distinct_recipe_kinds_before_a_second_kind() -> None:
    result = run_build_evolution(PARENT, ROWS_64, DiverseProposer(), _config())

    kinds = tuple(policy.recipe.kind for policy in result.shortlist)
    assert len(kinds) == 3
    assert set(kinds) == {"select", "weighted", "median"}


def test_controller_rejects_normalized_duplicate_policy_ids() -> None:
    class DuplicatePolicyProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            names = ("Model_K", "Model_K", "unique_two", "unique_three", "unique_four")
            return tuple(
                ChampionRecipe(
                    name=name,
                    kind="select",
                    parents=("better_a",),
                    fallback_parent="better_a",
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"unique_assumption_{index}",
                            candidate_name="better_a",
                            feature="history_length",
                            direction="above",
                            horizon_region="full",
                            operator="select",
                            rationale="History length supports the candidate forecast.",
                            failure_condition=(
                                "History length stops supporting the candidate forecast."
                            ),
                        ),
                    ),
                )
                for index, name in enumerate(names)
            )

    with pytest.raises(ChampionControllerError, match="duplicate policy"):
        run_build_evolution(PARENT, ROWS_64, DuplicatePolicyProposer(), _config())


def test_hostile_callback_cannot_leave_the_exact_parent_mutated() -> None:
    parent = _parent()
    original_sha256 = champion_fingerprint(parent)

    class MutatingProposer:
        def __call__(self, received, evidence) -> tuple[ChampionRecipe, ...]:
            assert received is parent
            object.__setattr__(received.policy.recipe, "name", "tampered_parent")
            return _proposal_batch("better")

    with pytest.raises(ChampionControllerError, match="mutate the active Parent"):
        run_build_evolution(parent, ROWS_64, MutatingProposer(), _config())

    assert parent.policy.recipe.name == "active_parent"
    assert champion_fingerprint(parent) == original_sha256


def test_win_tie_loss_is_diagnostic_and_never_a_standalone_gate() -> None:
    mixed_rows = list(ROWS_64)
    for row in ROWS_64:
        if row.candidate_name != "baseline_leaf":
            continue
        assert row.truth is not None
        index = int(row.task_id.rsplit("_", 1)[1])
        error = 0.0 if index < 4 else 2.05
        mixed_rows.append(
            replace(
                row,
                candidate_name="mixed_leaf",
                forecast=tuple(value + error for value in row.truth),
            )
        )

    class MixedProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            candidates = ("mixed_leaf", "worse_b", "worse_c", "worse_d", "worse_e")
            return tuple(
                ChampionRecipe(
                    name=f"mixed_batch_{index}",
                    kind="select",
                    parents=(candidate,),
                    fallback_parent=candidate,
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"mixed_history_{index}",
                            candidate_name=candidate,
                            feature="history_length",
                            direction="above",
                            horizon_region="full",
                            operator="select",
                            rationale="History length supports the candidate forecast.",
                            failure_condition=(
                                "History length stops supporting the candidate forecast."
                            ),
                        ),
                    ),
                )
                for index, candidate in enumerate(candidates)
            )

    gate = ChampionGateConfig(
        tail_regression_tolerance=1.0,
        maximum_task_regret_smae=1.0,
        maximum_task_regret_srmse=1.0,
        minimum_improved_folds=0,
    )
    result = run_build_evolution(
        PARENT,
        tuple(mixed_rows),
        MixedProposer(),
        replace(_config(), gate_config=gate),
    )

    full = next(
        attempt
        for attempt in result.generations[0].attempts
        if attempt.policy.recipe.parents == ("mixed_leaf",)
        and attempt.stage_task_counts[-1] == 64
    )
    assert full.comparison is not None
    assert full.comparison.wtl.losses > full.comparison.wtl.wins
    assert result.shortlist


def test_malformed_proposer_output_fails_closed() -> None:
    class MalformedProposer:
        def __call__(self, parent, evidence) -> object:
            return {"recipes": []}

    with pytest.raises(ChampionControllerError, match="exact tuple or list"):
        run_build_evolution(PARENT, ROWS_64, MalformedProposer(), _config())


def test_row_universe_and_split_drift_fail_before_proposer_execution() -> None:
    rows = list(ROWS_64)
    rows[0] = replace(rows[0], split="calibration")
    proposer = RecordingProposer("better")

    with pytest.raises(ChampionControllerError, match="Build rows only"):
        run_build_evolution(PARENT, tuple(rows), proposer, _config())

    assert proposer.calls == 0


def test_hostile_callback_cannot_mutate_sanitized_feedback() -> None:
    class FeedbackMutator:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            object.__setattr__(evidence, "label", "tampered")
            return _proposal_batch("better")

    with pytest.raises(ChampionControllerError, match="sanitized Build feedback"):
        run_build_evolution(PARENT, ROWS_64, FeedbackMutator(), _config())


def test_unscorable_child_is_typed_pruned_without_aborting_valid_siblings() -> None:
    class MixedValidityProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            valid = _proposal_batch("better")[:1]
            invalid = tuple(
                ChampionRecipe(
                    name=f"overlay_{index}",
                    kind="bounded_overlay",
                    parents=("better_a", "better_b"),
                    fallback_parent="better_a",
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"overlay_history_{index}",
                            candidate_name="better_b",
                            feature="periodicity_confidence",
                            direction="above",
                            horizon_region="full",
                            operator="bounded_overlay",
                            rationale="History length supports the candidate forecast.",
                            failure_condition=(
                                "History length stops supporting the candidate forecast."
                            ),
                        ),
                    ),
                )
                for index in range(4)
            )
            return valid + invalid

    result = run_build_evolution(
        PARENT, ROWS_64, MixedValidityProposer(), _config()
    )

    invalid = tuple(
        attempt
        for attempt in result.generations[0].attempts
        if attempt.status == "invalid"
    )
    assert invalid
    assert all(attempt.comparison is None for attempt in invalid)
    assert all(attempt.invalid_reason == "unscorable_child" for attempt in invalid)
    assert result.shortlist
    assert result.active_parent is PARENT


def test_deterministic_smoke_uses_only_the_4_8_schedule() -> None:
    task_ids = tuple(f"build_case_{index:03d}" for index in range(8))
    config = ChampionEvolutionConfig(
        build_size=8,
        calibration_size=2,
        screen_sizes=(4, 8),
        screen_task_ids=(task_ids[:4], task_ids),
        gate_config=ChampionGateConfig(minimum_improved_folds=0),
    )

    result = run_build_evolution(PARENT, _rows(8), RecordingProposer("better"), config)

    assert result.generations[0].stage_counts == (4, 8)
    assert result.config.build_task_ids == task_ids
    assert result.config_fingerprint == result.config.fingerprint


def test_history_only_materializer_scores_all_six_task3_operators() -> None:
    result = run_build_evolution(
        PARENT,
        _rows(enriched=True),
        AllOperatorProposer(),
        _config(),
    )

    valid_kinds = {
        attempt.policy.recipe.kind
        for attempt in result.generations[0].attempts
        if attempt.comparison is not None
    }
    assert valid_kinds == {
        "select",
        "route",
        "horizon_route",
        "weighted",
        "median",
        "bounded_overlay",
    }


def test_overlay_forecast_materialization_never_reads_future_truth(monkeypatch) -> None:
    rows = [row for row in _rows(1, enriched=True) if row.candidate_name in {
        "baseline_leaf",
        "better_a",
    }]

    class PoisonTruth:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("future truth reached history-only materialization")

    for row in rows:
        object.__setattr__(row, "truth", PoisonTruth())
    assumption = EvolutionAssumption(
        assumption_id="poison_free_history",
        candidate_name="better_a",
        feature="periodicity_confidence",
        direction="above",
        horizon_region="full",
        operator="bounded_overlay",
        rationale="History diagnostics support the overlay.",
        failure_condition="History diagnostics stop supporting the overlay.",
    )
    policy = FittedChampionPolicy(
        recipe=ChampionRecipe(
            name="poison_free_overlay",
            kind="bounded_overlay",
            parents=("baseline_leaf", "better_a"),
            fallback_parent="baseline_leaf",
            assumptions=(assumption,),
        ),
        thresholds=(("poison_free_history", 0.75),),
        overlay_alpha=0.5,
        correction_cap=0.1,
    )
    runtime_calls = []
    execute = controller_module.execute_champion

    def history_only_execute(policy, forecasts, diagnostics, profile, history, horizon):
        runtime_calls.append((forecasts, diagnostics, profile, history, horizon))
        assert all(diagnostic.folds == () for diagnostic in diagnostics.values())
        assert all(diagnostic.fold_forecasts == () for diagnostic in diagnostics.values())
        assert all(diagnostic.fold_truths == () for diagnostic in diagnostics.values())
        assert all(diagnostic.long_horizon_fold is None for diagnostic in diagnostics.values())
        return execute(policy, forecasts, diagnostics, profile, history, horizon)

    monkeypatch.setattr(controller_module, "execute_champion", history_only_execute)

    forecast, failure = controller_module._policy_forecast(
        policy, {row.candidate_name: row for row in rows}
    )

    assert len(runtime_calls) == 1
    assert failure is None
    assert forecast == pytest.approx((11.4, 13.4))


def test_valid_unsatisfied_history_condition_materializes_the_declared_fallback() -> None:
    rows = [
        row
        for row in _rows(1, enriched=True)
        if row.candidate_name in {"baseline_leaf", "better_a"}
    ]
    specialist_index = next(
        index for index, row in enumerate(rows) if row.candidate_name == "better_a"
    )
    specialist = rows[specialist_index]
    rows[specialist_index] = replace(
        specialist,
        diagnostic=_history_diagnostic("better_a", eligible=False),
    )
    assumption = EvolutionAssumption(
        assumption_id="unsatisfied_history",
        candidate_name="better_a",
        feature="periodicity_confidence",
        direction="above",
        horizon_region="full",
        operator="weighted",
        rationale="History diagnostics support the specialist.",
        failure_condition="History diagnostics do not support the specialist.",
    )
    policy = FittedChampionPolicy(
        recipe=ChampionRecipe(
            name="fallback_on_unsatisfied_history",
            kind="weighted",
            parents=("baseline_leaf", "better_a"),
            fallback_parent="baseline_leaf",
            assumptions=(assumption,),
        ),
        thresholds=(("unsatisfied_history", 0.0),),
        weights=(0.5, 0.5),
    )

    forecast, failure = controller_module._policy_forecast(
        policy, {row.candidate_name: row for row in rows}
    )

    assert failure is None
    assert forecast == rows[0].forecast


def test_proposal_policy_id_cannot_collide_with_the_active_parent() -> None:
    class ParentCollisionProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            return (parent.policy.recipe, *_proposal_batch("better")[:4])

    with pytest.raises(ChampionControllerError, match="duplicate policy ID"):
        run_build_evolution(PARENT, ROWS_64, ParentCollisionProposer(), _config())


def test_invalid_nested_parent_is_rejected_before_the_proposer() -> None:
    parent = _parent()
    object.__setattr__(parent.policy.recipe, "name", "")

    class CountingProposer:
        calls = 0

        def __call__(self, received, evidence) -> tuple[ChampionRecipe, ...]:
            type(self).calls += 1
            return _proposal_batch("better")

    with pytest.raises(ChampionControllerError, match="active Parent"):
        run_build_evolution(parent, ROWS_64, CountingProposer(), _config())

    assert CountingProposer.calls == 0


def test_callback_recipe_alias_cannot_corrupt_prior_generation_fitted_ids() -> None:
    class AliasingProposer:
        calls = 0
        first_batch: tuple[ChampionRecipe, ...] | None = None

        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            if self.first_batch is not None:
                object.__setattr__(
                    self.first_batch[0], "name", "mutated_external_alias"
                )
            batch = _proposal_batch("better", self.calls)
            if self.first_batch is None:
                self.first_batch = batch
            self.calls += 1
            return batch

    result = run_build_evolution(
        PARENT, ROWS_64, AliasingProposer(), _config(generations=2)
    )

    assert all(
        attempt.fitted_id == champion_fingerprint(attempt.policy)
        for generation in result.generations
        for attempt in generation.attempts
    )
    assert all(
        attempt.policy.recipe.name != "mutated_external_alias"
        for generation in result.generations
        for attempt in generation.attempts
    )


def test_callback_cannot_loosen_or_leave_the_registered_gate_mutated() -> None:
    gate = ChampionGateConfig(minimum_improved_folds=0)
    config = replace(_config(), gate_config=gate)

    class GateMutator:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            object.__setattr__(gate, "minimum_coverage", 0.0)
            return _proposal_batch("better")

    original_fingerprint = config.fingerprint
    with pytest.raises(ChampionControllerError, match="config"):
        run_build_evolution(PARENT, ROWS_64, GateMutator(), config)

    assert gate.minimum_coverage == 1.0
    assert config.fingerprint == original_fingerprint


def test_registered_fold_gate_remains_authoritative_on_the_first_screen() -> None:
    all_ids = tuple(f"build_case_{index:03d}" for index in range(64))
    first_stage = tuple(f"build_case_{index:03d}" for index in range(0, 40, 5))
    second_stage = first_stage + tuple(
        task_id for task_id in all_ids if task_id not in first_stage
    )[:24]
    config = replace(
        _config(),
        screen_task_ids=(first_stage, second_stage, all_ids),
        gate_config=ChampionGateConfig(minimum_improved_folds=2),
    )

    result = run_build_evolution(
        PARENT, ROWS_64, RecordingProposer("better"), config
    )

    assert all(attempt.stage_task_counts == (8,) for attempt in result.generations[0].attempts)
    assert result.generations[0].full_build_children == 0
    assert result.shortlist == ()


def test_normalized_row_inventory_collision_fails_before_proposer() -> None:
    rows = list(ROWS_64)
    for index in range(64):
        task_id = f"build_case_{index:03d}"
        source = next(
            row
            for row in ROWS_64
            if row.task_id == task_id and row.candidate_name == "baseline_leaf"
        )
        rows.extend(
            (
                replace(source, candidate_name="Model_K"),
                replace(source, candidate_name="Model_K"),
            )
        )
    proposer = RecordingProposer("better")

    with pytest.raises(ChampionControllerError, match="normalized row inventory"):
        run_build_evolution(PARENT, tuple(rows), proposer, _config())

    assert proposer.calls == 0


def test_proposal_name_cannot_collide_with_materialized_row_inventory() -> None:
    class InventoryCollisionProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            first, *remaining = _proposal_batch("better")
            return (replace(first, name="better_a"), *remaining)

    with pytest.raises(ChampionControllerError, match="materialized row inventory"):
        run_build_evolution(PARENT, ROWS_64, InventoryCollisionProposer(), _config())
