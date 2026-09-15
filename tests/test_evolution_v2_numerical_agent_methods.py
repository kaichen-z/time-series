from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from evolving_loop.v2.numerical_qd.agent_methods import (
    CurriculumTargetV2,
    MutationPromptLineageV2,
    MutationPromptPopulationV2,
    VerifiedReusableProgramV2,
    apply_prompt_train_credit,
    derive_curriculum_targets,
    insert_prompt_child,
    select_prompt_lineage,
    select_reusable_programs,
)
from evolving_loop.v2.numerical_qd.contracts import (
    ConstraintReportV2,
    MorphologyCellV2,
    MutationOperatorStatsV2,
    NumericalObjectiveVectorV2,
    NumericalProposerPromptV2,
    NumericalQDEntryV2,
    TrainMutationFeedbackV2,
)
from evolving_loop.v2.numerical_qd.map_elites import NumericalQDArchive


def _entry(marker, cell, *, diagnostics=(), violations=()):
    return NumericalQDEntryV2(
        1,
        marker * 64,
        marker * 64,
        cell,
        ("train-task",),
        NumericalObjectiveVectorV2(*(1.0,) * 5),
        ConstraintReportV2(not violations, violations),
        diagnostics,
    )


def test_curriculum_prioritizes_unoccupied_then_least_visited_then_failure_matched():
    unoccupied = MorphologyCellV2("low", "none", "low", "stable", "short", "program")
    least = replace(unoccupied, trend="medium")
    failed = replace(unoccupied, trend="high")
    archive = NumericalQDArchive().insert((
        _entry("a", least),
        _entry("b", failed, diagnostics=("timeout",)),
        _entry("c", failed),
    ))
    feedback = TrainMutationFeedbackV2(
        "train", "repair", False, False, False, ("timeout",),
    )

    targets = derive_curriculum_targets(
        tuple(reversed((unoccupied, least, failed))), archive, feedback, maximum_targets=3,
    )

    assert [(target.cell, target.reason, target.visit_count) for target in targets] == [
        (unoccupied, "unoccupied", 0),
        (least, "least_visited", 1),
        (failed, "failure_matched", 2),
    ]
    assert targets[2].train_diagnostic_categories == ("timeout",)
    assert all(isinstance(target, CurriculumTargetV2) for target in targets)
    with pytest.raises(FrozenInstanceError):
        targets[0].visit_count = 1


def test_curriculum_contract_is_exact_canonical_and_rejects_non_train_inputs():
    cell = MorphologyCellV2("low", "none", "low", "stable", "short", "program")
    payload = {
        "schema_version": 1,
        "cell": cell.to_payload(),
        "reason": "failure_matched",
        "visit_count": 2,
        "train_diagnostic_categories": ["timeout"],
    }
    target = CurriculumTargetV2.from_payload(payload)
    assert json.loads(target.canonical_bytes()) == payload
    with pytest.raises(ValueError, match="exact schema"):
        CurriculumTargetV2.from_payload(payload | {"dev_metrics": {}})
    with pytest.raises((TypeError, ValueError)):
        CurriculumTargetV2.from_payload(payload | {"train_diagnostic_categories": ["public"]})
    with pytest.raises(TypeError, match="sanitized"):
        derive_curriculum_targets((cell,), NumericalQDArchive(), {"split": "train"}, maximum_targets=1)
    with pytest.raises(ValueError, match="must not be empty"):
        derive_curriculum_targets((), NumericalQDArchive(),
                                  TrainMutationFeedbackV2("train", "repair", True, False, False, ()),
                                  maximum_targets=1)
    with pytest.raises(TypeError, match="MorphologyCellV2"):
        derive_curriculum_targets((cell.to_payload(),), NumericalQDArchive(),
                                  TrainMutationFeedbackV2("train", "repair", True, False, False, ()),
                                  maximum_targets=1)
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        derive_curriculum_targets((cell,), NumericalQDArchive(),
                                  TrainMutationFeedbackV2("train", "repair", True, False, False, ()),
                                  maximum_targets=33)


def test_curriculum_failure_matching_uses_retained_history_not_only_cell_survivors():
    base = MorphologyCellV2("low", "none", "low", "stable", "short", "program")
    least = replace(base, trend="medium")
    failed = replace(base, trend="high")
    diagnostic = _entry("a", failed, diagnostics=("timeout",), violations=("coverage",))
    survivor = _entry("b", failed)
    archive = NumericalQDArchive(capacity=1).insert((
        _entry("c", least), diagnostic, survivor,
    ))
    failed_record = next(record for record in archive.cells if record.cell == failed)
    assert diagnostic.fingerprint() not in failed_record.entry_sha256s
    assert diagnostic.fingerprint() in archive.entries

    targets = derive_curriculum_targets(
        (least, failed),
        archive,
        TrainMutationFeedbackV2("train", "repair", False, False, False, ("timeout",)),
        maximum_targets=2,
    )

    assert [(target.cell, target.reason) for target in targets] == [
        (least, "least_visited"),
        (failed, "failure_matched"),
    ]


def _program(member_id, genome_marker, source, *cells):
    return VerifiedReusableProgramV2(
        1,
        member_id,
        genome_marker * 64,
        tuple(sorted(cell.fingerprint() for cell in cells)),
        hashlib.sha256(source.encode()).hexdigest(),
        source,
    )


def test_reusable_program_selection_filters_eligibility_and_targets_then_uses_sha_order():
    first = MorphologyCellV2("low", "none", "low", "stable", "short", "program")
    second = replace(first, trend="high")
    exact = _program("exact", "a", "def exact():\n    return 1\n", first, second)
    one_a = _program("one-a", "b", "def a():\n    return 1\n", first)
    one_b = _program("one-b", "c", "def b():\n    return 1\n", first)
    ineligible = _program("ineligible", "d", "def x():\n    return 1\n", first, second)
    wrong_cell = _program("wrong", "e", "def y():\n    return 1\n", replace(first, horizon="long"))
    records = (one_b, wrong_cell, ineligible, exact, one_a)

    selected = select_reusable_programs(
        records,
        eligible_genome_sha256s=tuple(sorted(("a" * 64, "b" * 64, "c" * 64, "e" * 64))),
        target_cell_sha256s=tuple(sorted((first.fingerprint(), second.fingerprint()))),
        maximum_records=2,
    )

    assert selected == (exact, min((one_a, one_b), key=lambda item: item.source_sha256))
    assert records == (one_b, wrong_cell, ineligible, exact, one_a)


def test_verified_reusable_program_binds_exact_source_and_has_closed_schema():
    cell = MorphologyCellV2("low", "none", "low", "stable", "short", "program")
    source = "def forecast():\n    return 0\n"
    record = _program("safe", "a", source, cell)
    assert VerifiedReusableProgramV2.from_payload(record.to_payload()) == record
    with pytest.raises(ValueError, match="source SHA"):
        replace(record, source_text=source + "# changed")
    with pytest.raises(ValueError, match="exact schema"):
        VerifiedReusableProgramV2.from_payload(record.to_payload() | {"raw_forecasts": [1.0]})
    with pytest.raises(TypeError, match="immutable tuple"):
        select_reusable_programs([record], eligible_genome_sha256s=("a" * 64,),
                                 target_cell_sha256s=(cell.fingerprint(),), maximum_records=1)
    with pytest.raises(ValueError):
        select_reusable_programs((record,), eligible_genome_sha256s=("a" * 64,),
                                 target_cell_sha256s=(cell.fingerprint(),), maximum_records=True)


def _prompt(text, parent=None, *, maximum=4096, operators=("policy_tune",)):
    return NumericalProposerPromptV2(
        1, text, "numerical_mutation_batch_v1", maximum, operators, parent,
    )


def _lineage(prompt, stats=(0, 0, 0, 0, 0)):
    return MutationPromptLineageV2(1, prompt, MutationOperatorStatsV2(*stats))


def test_prompt_population_selects_credit_then_sha_without_mutation():
    first = _lineage(_prompt("mutation prompt one"), (2, 2, 1, 1, 1))
    second = _lineage(_prompt("mutation prompt two"), (2, 2, 1, 1, 1))
    population = MutationPromptPopulationV2(1, 2, (second, first))
    before = population.canonical_bytes()

    selected = select_prompt_lineage(population)

    assert selected == min((first, second), key=lambda item: item.mutation_prompt.fingerprint())
    assert population.canonical_bytes() == before
    assert population.lineages == tuple(sorted(
        (first, second), key=lambda item: item.mutation_prompt.fingerprint(),
    ))


def test_prompt_child_insertion_preserves_child_and_evicts_weakest_incumbent():
    strong = _lineage(_prompt("strong mutation prompt"), (3, 3, 2, 2, 2))
    weak = _lineage(_prompt("weak mutation prompt"), (3, 0, 0, 0, 0))
    population = MutationPromptPopulationV2(1, 2, (strong, weak))
    parent_sha = strong.mutation_prompt.fingerprint()
    child = _prompt("child mutation prompt", parent_sha)

    updated = insert_prompt_child(population, parent_sha, child)

    by_sha = {item.mutation_prompt.fingerprint(): item for item in updated.lineages}
    assert set(by_sha) == {parent_sha, child.fingerprint()}
    assert by_sha[child.fingerprint()].stats == MutationOperatorStatsV2(0, 0, 0, 0, 0)
    assert len(updated.lineages) == updated.capacity == 2
    assert population.lineages == tuple(sorted(
        (strong, weak), key=lambda item: item.mutation_prompt.fingerprint(),
    ))


def test_prompt_credit_updates_only_named_lineage_from_typed_train_feedback():
    first = _lineage(_prompt("first mutation prompt"))
    second = _lineage(_prompt("second mutation prompt"))
    population = MutationPromptPopulationV2(1, 2, (first, second))
    second_sha = second.mutation_prompt.fingerprint()
    feedback = TrainMutationFeedbackV2(
        "train", "policy_tune", True, True, True, (),
    )

    credited = apply_prompt_train_credit(population, second_sha, feedback)

    by_sha = {item.mutation_prompt.fingerprint(): item for item in credited.lineages}
    assert by_sha[first.mutation_prompt.fingerprint()].stats == first.stats
    assert by_sha[second_sha].stats == MutationOperatorStatsV2(1, 1, 1, 1, 1)
    assert population.lineages[0].stats == MutationOperatorStatsV2(0, 0, 0, 0, 0)
    with pytest.raises(TypeError, match="sanitized"):
        apply_prompt_train_credit(population, second_sha, feedback.to_payload())
    with pytest.raises(ValueError, match="unknown"):
        apply_prompt_train_credit(population, "f" * 64, feedback)


def test_prompt_credit_rejects_feedback_for_another_operator():
    lineage = _lineage(_prompt("mutation prompt"))
    population = MutationPromptPopulationV2(1, 1, (lineage,))
    feedback = TrainMutationFeedbackV2(
        "train", "repair", True, True, True, (),
    )

    with pytest.raises(ValueError, match="policy_tune"):
        apply_prompt_train_credit(
            population, lineage.mutation_prompt.fingerprint(), feedback,
        )


def test_prompt_lineage_contract_closes_authority_and_population_schema():
    prompt = _prompt("safe mutation prompt")
    lineage = _lineage(prompt)
    population = MutationPromptPopulationV2(1, 1, (lineage,))
    assert MutationPromptPopulationV2.from_payload(population.to_payload()) == population
    with pytest.raises(ValueError, match="policy_tune"):
        _lineage(_prompt("expanded authority", operators=("add", "policy_tune")))
    with pytest.raises(ValueError, match="exact schema"):
        MutationPromptPopulationV2.from_payload(population.to_payload() | {"public_score": 1.0})
    child = _prompt("oversized child", prompt.fingerprint(), maximum=8192)
    with pytest.raises(ValueError, match="expand"):
        insert_prompt_child(population, prompt.fingerprint(), child)


def test_agent_method_contracts_and_pure_functions_are_public_exports():
    from evolving_loop.v2 import numerical_qd

    expected = {
        "CurriculumTargetV2": CurriculumTargetV2,
        "VerifiedReusableProgramV2": VerifiedReusableProgramV2,
        "MutationPromptLineageV2": MutationPromptLineageV2,
        "MutationPromptPopulationV2": MutationPromptPopulationV2,
        "derive_curriculum_targets": derive_curriculum_targets,
        "select_reusable_programs": select_reusable_programs,
        "select_prompt_lineage": select_prompt_lineage,
        "insert_prompt_child": insert_prompt_child,
        "apply_prompt_train_credit": apply_prompt_train_credit,
    }
    assert {name: getattr(numerical_qd, name) for name in expected} == expected
    assert set(expected) <= set(numerical_qd.__all__)
