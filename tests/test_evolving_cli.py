from __future__ import annotations

import pytest

from evolving_agent.cli import (
    BASELINE_CHOICES,
    EVOLUTION_CHOICES,
    _baseline_argv,
    _three_way_entity_split,
    build_parser,
)
from evolving_agent.data import ContextTask, Task


def test_evolve_cli_exposes_three_evolution_modes() -> None:
    parser = build_parser()
    for mode in ("prompt", "genome", "source"):
        args = parser.parse_args(["evolve", "--evolution-mode", mode])
        assert args.evolution_mode == mode


def test_genome_remains_the_default_evolution_mode() -> None:
    assert build_parser().parse_args(["evolve"]).evolution_mode == "genome"


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
