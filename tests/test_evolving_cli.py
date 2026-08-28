from __future__ import annotations

from types import SimpleNamespace

import pytest

from evolving_loop.cli import (
    BASELINE_CHOICES,
    EVOLUTION_CHOICES,
    _baseline_argv,
    _factory,
    _three_way_entity_split,
    build_parser,
)
from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.coding_agent.skill_library import SkillLibrary
from evolving_loop.data import ContextTask, Task
from evolving_loop.decision_agent.skill_library import DecisionSkillLibrary
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from common.llm import FakeLLMClient


def test_evolve_cli_exposes_three_evolution_modes() -> None:
    parser = build_parser()
    for mode in ("prompt", "genome", "source"):
        args = parser.parse_args(["evolve", "--evolution-mode", mode])
        assert args.evolution_mode == mode


def test_genome_remains_the_default_evolution_mode() -> None:
    assert build_parser().parse_args(["evolve"]).evolution_mode == "genome"


def test_evolve_cli_exposes_targeted_agent_evolution() -> None:
    parser = build_parser()
    for target in ("auto", "coding", "retrieval", "decision"):
        args = parser.parse_args(["evolve", "--evolve-target", target])
        assert args.evolve_target == target
    assert parser.parse_args(["evolve"]).evolve_target == "auto"


def test_setting2_knowledge_is_explicitly_opt_in_for_both_interfaces() -> None:
    parser = build_parser()

    assert parser.parse_args(["evolve"]).setting2_knowledge is False
    assert parser.parse_args(["evolve", "--setting2-knowledge"]).setting2_knowledge is True
    unified = parser.parse_args(
        ["--evolution", "genome", "--tasks-file", "tasks.jsonl", "--setting2-knowledge"]
    )
    assert unified.setting2_knowledge is True


def test_evolve_cli_exposes_successive_halving_controls() -> None:
    parser = build_parser()
    values = [
        "--successive-halving",
        "--screen-train-tasks",
        "6",
        "--screen-val-tasks",
        "2",
        "--screen-promote",
        "1",
        "--screen-tolerance",
        "0.01",
    ]
    for prefix in (["evolve"], ["--evolution", "genome", "--tasks-file", "tasks"]):
        args = parser.parse_args([*prefix, *values])
        assert args.successive_halving is True
        assert args.screen_train_tasks == 6
        assert args.screen_val_tasks == 2
        assert args.screen_promote == 1
        assert args.screen_tolerance == pytest.approx(0.01)


def test_unified_cli_exposes_every_named_method() -> None:
    parser = build_parser()
    for name in BASELINE_CHOICES:
        args = parser.parse_args(["--baseline", name, "--sample-dir", "sample"])
        assert args.baseline == name
    for name in EVOLUTION_CHOICES:
        args = parser.parse_args(["--evolution", name, "--tasks-file", "tasks.jsonl"])
        assert args.evolution == name
        frozen = parser.parse_args(["--inference", name, "--hidden-test"])
        assert frozen.inference == name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("chronos", ("--system", "backbone-only", "--backbone", "chronos")),
        ("timesfm", ("--system", "backbone-only", "--backbone", "timesfm")),
        ("statistical", ("--system", "backbone-only", "--backbone", "statistical")),
        ("codex-contract", ("--system", "codex-contract")),
        ("codex-direct", ("--system", "codex-direct")),
    ],
)
def test_baseline_names_route_to_existing_drcik_systems(name, expected) -> None:
    args = build_parser().parse_args(["--baseline", name, "--sample-dir", "sample"])
    translated = _baseline_argv(args)
    for index in range(len(expected) - 1):
        if expected[index].startswith("--"):
            position = translated.index(expected[index])
            assert translated[position + 1] == expected[index + 1]


def test_codex_and_rules_triad_have_distinct_reasoning_backends() -> None:
    parser = build_parser()
    codex = _baseline_argv(
        parser.parse_args(["--baseline", "codex-triad", "--sample-dir", "sample"])
    )
    rules = _baseline_argv(
        parser.parse_args(["--baseline", "rules-triad", "--sample-dir", "sample"])
    )
    assert codex[codex.index("--reasoning-agent") + 1] == "codex"
    assert rules[rules.index("--reasoning-agent") + 1] == "rules"


def test_baseline_requires_a_data_source() -> None:
    args = build_parser().parse_args(["--baseline", "chronos"])
    with pytest.raises(SystemExit, match="requires one data source"):
        _baseline_argv(args)


def test_three_way_split_is_entity_disjoint_and_reproducible() -> None:
    tasks = [
        ContextTask(
            numeric=Task(
                task_id=f"task_{index}",
                history_values=(1.0, 2.0),
                future_values=(3.0,),
                prediction_length=1,
                frequency="1 day",
                seasonal_period=None,
                entity_name=f"entity_{index // 2}",
            ),
            target_name="target",
            target_description="",
            history_timestamps=("1", "2"),
            future_timestamps=("3",),
            documents=(),
        )
        for index in range(12)
    ]
    first = _three_way_entity_split(tasks, 7, 0.25, 0.20)
    second = _three_way_entity_split(tasks, 7, 0.25, 0.20)
    assert [[item.numeric.task_id for item in split] for split in first] == [
        [item.numeric.task_id for item in split] for split in second
    ]
    entity_sets = [
        {item.numeric.entity_name for item in split}
        for split in first
    ]
    assert all(first)
    assert entity_sets[0].isdisjoint(entity_sets[1])
    assert entity_sets[0].isdisjoint(entity_sets[2])
    assert entity_sets[1].isdisjoint(entity_sets[2])


def test_harness_factory_hydrates_policy_embedded_skills(tmp_path) -> None:
    policy = HarnessPolicy(
        coding_skills=(
            {
                "skill_id": "coding-1",
                "name": "embedded_numeric",
                "description": "Embedded numeric strategy.",
                "code": "def forecast(history, horizon, frequency): return [history[-1]] * horizon",
                "created_from_task": "task_train",
                "assumption": "The level persists.",
                "failure_condition": "The regime changes.",
                "validation_score": 1.0,
                "uses": 0,
                "avg_score": None,
            },
        ),
        retrieval_skills=(
            {
                "skill_id": "retrieval-1",
                "name": "embedded_retrieval",
                "description": "Embedded retrieval strategy.",
                "applicability": "Scheduled events.",
                "query_strategy": "Search exact dates.",
                "verification_rule": "Require an exact quote.",
                "created_from_task": "task_train",
                "validation_score": 0.8,
                "uses": 0,
                "avg_score": None,
            },
        ),
        decision_skills=(
            {
                "skill_id": "decision-1",
                "name": "embedded_decision",
                "description": "Embedded selection strategy.",
                "applicability": "Multiple candidates.",
                "decision_rule": "Preserve the validated leader.",
                "failure_condition": "Evidence falsifies it.",
                "created_from_task": "task_train",
                "validation_score": 0.9,
                "uses": 0,
                "avg_score": None,
            },
        ),
    )
    harness = _factory(
        SimpleNamespace(setting="llm_only"),
        FakeLLMClient([]),
        SkillLibrary(tmp_path / "coding.json"),
        RetrievalSkillLibrary(tmp_path / "retrieval.json"),
        DecisionSkillLibrary(tmp_path / "decision.json"),
        None,
        isolate_library=True,
    )(policy)

    assert harness.coding.library.get("embedded_numeric") is not None
    assert harness.retrieval.library.get("embedded_retrieval") is not None
    assert harness.decision.library.get("embedded_decision") is not None
