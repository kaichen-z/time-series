"""Constrained-diff enforcement: a mutation may change one thing, and never bookkeeping."""

from __future__ import annotations

import json

import pytest
from dr_cik.llm import FakeLLMClient

from evolving_agents.bundles import load_seed
from evolving_agents.evolve.evaluate import TaskResult
from evolving_agents.evolve.mutate import (
    MutationError,
    apply_change,
    build_mutation_prompt,
    mutate,
    mutate_triple,
)
from evolving_agents.models import BundleTriple

SEED = load_seed("coding")
WORST = [TaskResult(task_id="task_4", score=-2.5, trace={"best_assumption": "flat", "hindcast_error": 2.5})]


def test_system_prompt_change_leaves_templates_alone() -> None:
    child = apply_change(SEED, {"change_type": "system_prompt", "system_prompt": "NEW", "changelog": "rewrote"}, "v001")
    assert child.system_prompt == "NEW"
    assert child.code_templates == SEED.code_templates


def test_a_response_claiming_two_changes_only_applies_one() -> None:
    sneaky = {
        "change_type": "system_prompt",
        "system_prompt": "NEW",
        "code_template": "def forecast(h, z, f): return [0] * z",
        "target_template_name": "seasonal_naive",
        "changelog": "tried to do both",
    }
    child = apply_change(SEED, sneaky, "v001")
    assert child.system_prompt == "NEW"
    assert child.code_templates == SEED.code_templates  # the smuggled template edit was ignored


def test_template_change_leaves_the_prompt_alone() -> None:
    child = apply_change(
        SEED,
        {"change_type": "add_code_template", "target_template_name": "holiday", "code_template": "def forecast(h, z, f): return [1] * z", "changelog": "add"},
        "v001",
    )
    assert set(child.code_templates) - set(SEED.code_templates) == {"holiday"}
    assert child.system_prompt == SEED.system_prompt


def test_bookkeeping_is_always_carried_from_the_parent() -> None:
    child = apply_change(
        SEED,
        {"change_type": "system_prompt", "system_prompt": "NEW", "changelog": "x", "hyperparameters": {"k_hypotheses": 999}, "agent": "hijacked"},
        "v007",
    )
    assert child.agent == "coding"
    assert child.version == "v007"
    assert child.parent == "v000"
    assert child.bundle_id == "coding/v007"
    assert child.hyperparameters == SEED.hyperparameters


def test_changelog_is_truncated() -> None:
    child = apply_change(SEED, {"change_type": "system_prompt", "system_prompt": "NEW", "changelog": "x" * 500}, "v001")
    assert len(child.notes_from_evolver) == 200


@pytest.mark.parametrize(
    "parsed",
    [
        {"change_type": "nonsense"},
        {"change_type": "system_prompt", "system_prompt": ""},
        {"change_type": "system_prompt"},
        {"change_type": "edit_code_template", "target_template_name": "absent", "code_template": "x"},
        {"change_type": "remove_code_template", "target_template_name": "absent"},
        {"change_type": "add_code_template", "target_template_name": "seasonal_naive", "code_template": "x"},
        {"change_type": "add_code_template", "code_template": "x"},
    ],
)
def test_illegal_changes_are_rejected(parsed: dict) -> None:
    with pytest.raises(MutationError):
        apply_change(SEED, parsed, "v001")


def test_remove_drops_exactly_one_template() -> None:
    child = apply_change(SEED, {"change_type": "remove_code_template", "target_template_name": "linear_trend", "changelog": "drop"}, "v001")
    assert "linear_trend" not in child.code_templates
    assert len(child.code_templates) == len(SEED.code_templates) - 1


def test_mutate_writes_the_child_and_returns_it(tmp_path) -> None:
    llm = FakeLLMClient(responses=[json.dumps({"change_type": "system_prompt", "system_prompt": "EVOLVED", "changelog": "clearer"})])
    child = mutate(SEED, WORST, llm, tmp_path)
    assert child.system_prompt == "EVOLVED"
    assert child.version == "v000"  # first child written into an empty directory
    assert (tmp_path / "coding" / "v000.json").is_file()


def test_mutate_returns_an_unchanged_child_when_the_response_is_unparseable(tmp_path) -> None:
    child = mutate(SEED, WORST, FakeLLMClient(responses=["not json at all"]), tmp_path)
    assert child.system_prompt == SEED.system_prompt
    assert child.code_templates == SEED.code_templates
    assert "mutation failed" in child.notes_from_evolver


def test_mutate_returns_an_unchanged_child_when_the_change_is_illegal(tmp_path) -> None:
    llm = FakeLLMClient(responses=[json.dumps({"change_type": "edit_code_template", "target_template_name": "absent", "code_template": "x"})])
    child = mutate(SEED, WORST, llm, tmp_path)
    assert "mutation failed" in child.notes_from_evolver


def test_mutate_strips_a_reasoning_block_before_parsing(tmp_path) -> None:
    payload = json.dumps({"change_type": "system_prompt", "system_prompt": "EVOLVED", "changelog": "c"})
    llm = FakeLLMClient(responses=[f"<think>The failures all involve seasonality.</think>{payload}"])
    assert mutate(SEED, WORST, llm, tmp_path).system_prompt == "EVOLVED"


def test_mutation_prompt_carries_the_failures_and_the_current_definition() -> None:
    prompt = build_mutation_prompt(SEED, WORST)
    assert "task_4" in prompt
    assert "best_assumption" in prompt
    assert SEED.system_prompt[:60] in prompt
    assert "seasonal_naive" in prompt


def test_mutation_prompt_truncates_a_huge_trace() -> None:
    huge = [TaskResult(task_id="task_9", score=-1.0, trace={"blob": "x" * 50000})]
    assert "(truncated)" in build_mutation_prompt(SEED, huge)


def test_mutate_triple_targets_the_decision_bundle_most_often(tmp_path) -> None:
    import random

    triple = BundleTriple(coding=SEED, retrieval=load_seed("retrieval"), decision=load_seed("decision"))
    payload = json.dumps({"change_type": "system_prompt", "system_prompt": "EVOLVED", "changelog": "c"})
    changed = []
    for seed in range(20):
        llm = FakeLLMClient(responses=[payload])
        child = mutate_triple(triple, WORST, llm, tmp_path, rng=random.Random(seed))
        changed.append(next(slot for slot in ("coding", "retrieval", "decision") if getattr(child, slot).system_prompt == "EVOLVED"))
    assert changed.count("decision") > changed.count("coding") + changed.count("retrieval")
