from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evolving_loop.v2.budget import ResourceUse
from evolving_loop.v2.cli import build_parser
from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.cooperative.contracts import DecisionModuleV2, RetrievalModuleV2
from evolving_loop.v2.fakes import smoke_config
from numerical_agent.evolution.champion import parse_champion_release


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "configs" / "evolution_v2" / "cooperative"
BUILDER = ROOT / "tests" / "build_evolution_v2_cooperative_fixture.py"
FIXTURES = ROOT / "tests" / "fixtures" / "evolution_v2_cooperative"


def _payload() -> dict[str, object]:
    control = replace(
        smoke_config(),
        runner="production",
        enabled_mutation_scopes=("numerical", "retrieval", "decision", "joint"),
    )
    return {
        "schema_version": 1,
        "control": control.to_payload(),
        "max_steps": 4,
        "children_per_step": 1,
        "discount": 0.9,
        "task_cost_weight": 0.05,
        "metric_cap": 5.0,
        "acceptance_tolerance": 1e-12,
        "resource_ceilings": ResourceUse(
            wall_seconds=600.0,
            task_executions=100,
            artifact_bytes=1_000_000,
        ).to_payload(),
    }


def test_cooperative_config_is_a_strict_prototype_profile():
    from evolving_loop.v2.cooperative.config import CooperativeConfigV2

    config = CooperativeConfigV2.from_payload(_payload())

    assert config.max_steps == 4
    assert config.children_per_step == 1
    assert config.control.runner == "production"
    assert config.resource_ceilings.wall_seconds == 600.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("children_per_step", 2),
        ("discount", 0.0),
        ("task_cost_weight", float("inf")),
        ("metric_cap", 0.0),
        ("acceptance_tolerance", 1e-9),
    ],
)
def test_cooperative_config_rejects_nonprototype_values(field, value):
    from evolving_loop.v2.cooperative.config import CooperativeConfigV2

    with pytest.raises(ValueError):
        CooperativeConfigV2.from_payload(_payload() | {field: value})


def test_cooperative_config_requires_exact_fields_and_resource_ceilings():
    from evolving_loop.v2.cooperative.config import CooperativeConfigV2

    with pytest.raises(ValueError, match="exact schema"):
        CooperativeConfigV2.from_payload(_payload() | {"extra": True})
    ceilings = dict(_payload()["resource_ceilings"])
    ceilings.pop("llm_calls")
    with pytest.raises(ValueError, match="exact schema"):
        CooperativeConfigV2.from_payload(_payload() | {"resource_ceilings": ceilings})


def test_evolve_parser_accepts_cooperative_seed_inputs():
    args = build_parser().parse_args(
        [
            "evolve",
            "--config",
            "c.json",
            "--seed-supply",
            "s.json",
            "--task-manifest",
            "t.json",
            "--retrieval-release",
            "r.json",
            "--decision-policy",
            "d.json",
            "--output-dir",
            "out",
        ]
    )

    assert args.command == "evolve"
    assert args.retrieval_release.name == "r.json"


def test_shipped_smoke_profiles_are_canonical_and_differ_only_by_scheduler():
    from evolving_loop.v2.cooperative.config import load_cooperative_config

    loaded = {
        scheduler: load_cooperative_config(PROFILES / f"smoke-{scheduler}.json")
        for scheduler in ("ucb", "thompson")
    }

    for scheduler, config in loaded.items():
        path = PROFILES / f"smoke-{scheduler}.json"
        assert path.read_bytes() == config.canonical_bytes()
        assert config.control.scheduler == scheduler
        assert config.control.profile == "smoke"
        assert config.control.hard_limit_seconds == 600
    ucb = loaded["ucb"].to_payload()
    thompson = loaded["thompson"].to_payload()
    assert ucb["control"].pop("scheduler") == "ucb"
    assert thompson["control"].pop("scheduler") == "thompson"
    assert ucb == thompson


def _snapshot_bytes(root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.iterdir()
        if path.is_file()
    }


def test_fixture_builder_writes_exact_canonical_4_train_1_dev_inputs(tmp_path):
    output = tmp_path / "fixture"
    command = [sys.executable, str(BUILDER), "--output-dir", str(output)]

    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert set(_snapshot_bytes(output)) == {
        "seed_supply.json",
        "tasks_4_1.json",
        "retrieval_release.json",
        "decision_policy.json",
    }
    for path in output.iterdir():
        payload = json.loads(path.read_bytes())
        assert path.read_bytes() == canonical_v2_bytes(payload)

    seed_payload = json.loads((output / "seed_supply.json").read_bytes())
    release = parse_numerical_supply_release(seed_payload)
    anchor = parse_champion_release(seed_payload["anchor_release_payload"])
    assert anchor.policy.recipe.parents == ("safe_anchor",)
    assert tuple(item.candidate_id for item in release.alternatives) == (
        "seasonal_naive",
    )
    RetrievalModuleV2.from_payload(
        json.loads((output / "retrieval_release.json").read_bytes())
    )
    DecisionModuleV2.from_payload(
        json.loads((output / "decision_policy.json").read_bytes())
    )
    tasks = json.loads((output / "tasks_4_1.json").read_bytes())
    assert set(tasks) == {"schema_version", "train", "dev"}
    assert len(tasks["train"]) == 4
    assert len(tasks["dev"]) == 1
    assert b"public" not in (output / "tasks_4_1.json").read_bytes().lower()

    before = _snapshot_bytes(output)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert _snapshot_bytes(output) == before


def _cooperative_args(scheduler: str, output: Path) -> list[str]:
    return [
        "evolve",
        "--config",
        str(PROFILES / f"smoke-{scheduler}.json"),
        "--seed-supply",
        str(FIXTURES / "seed_supply.json"),
        "--task-manifest",
        str(FIXTURES / "tasks_4_1.json"),
        "--retrieval-release",
        str(FIXTURES / "retrieval_release.json"),
        "--decision-policy",
        str(FIXTURES / "decision_policy.json"),
        "--output-dir",
        str(output),
    ]


def _run_cli(capsys, scheduler: str, output: Path) -> dict[str, object]:
    from evolving_loop.v2 import cli

    assert cli.main(_cooperative_args(scheduler, output)) == 0
    return json.loads(capsys.readouterr().out)


def test_cooperative_manifest_parser_is_4_1_and_independent_of_numerical_80_20():
    from evolving_loop.v2.cli import (
        _parse_cooperative_task_manifest,
        _parse_task_manifest,
    )

    payload = json.loads((FIXTURES / "tasks_4_1.json").read_bytes())
    parsed = _parse_cooperative_task_manifest(payload)

    assert len(parsed["train"]) == 4
    assert len(parsed["dev"]) == 1
    with pytest.raises(ValueError, match="exact schema|Train80"):
        _parse_task_manifest(payload)


@pytest.mark.parametrize("scheduler", ["ucb", "thompson"])
def test_unified_evolve_completes_cooperative_smoke(tmp_path, capsys, scheduler):
    completed = _run_cli(capsys, scheduler, tmp_path / "run")

    assert completed["status"] == "cooperative_complete"
    assert completed["attempted_arms"] == [
        "numerical",
        "retrieval",
        "decision",
        "joint",
    ]
    assert completed["public_test_accessed"] is False


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("scheduler", ["ucb", "thompson"])
def test_complete_cli_resume_is_byte_identical_and_read_only(
    tmp_path, capsys, scheduler
):
    output = tmp_path / "run"
    first = _run_cli(capsys, scheduler, output)
    before = _snapshot(output)

    second = _run_cli(capsys, scheduler, output)

    assert second == first
    assert _snapshot(output) == before


def test_fake_evolve_rejects_cooperative_only_flags(tmp_path, capsys):
    config = tmp_path / "fake.json"
    config.write_bytes(canonical_v2_bytes(smoke_config().to_payload()))

    assert (
        __import__("evolving_loop.v2.cli", fromlist=["main"]).main(
            [
                "evolve",
                "--config",
                str(config),
                "--seed-supply",
                str(FIXTURES / "seed_supply.json"),
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )
    assert "cooperative" in capsys.readouterr().err.lower()
    assert not (tmp_path / "run").exists()


def test_legacy_shipped_fake_config_remains_accepted(tmp_path, capsys):
    from evolving_loop.v2.cli import main

    assert (
        main(
            [
                "evolve",
                "--config",
                str(ROOT / "configs" / "evolution_v2" / "smoke.json"),
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["status"] == "deterministic_fake_complete"
    )
