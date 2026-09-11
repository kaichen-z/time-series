from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.kernel import SeedBootstrapAuthority, _material_bytes
from evolving_loop.v2.numerical_qd.config import load_numerical_qd_config
from evolving_loop.v2.store import V2RunStore


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "configs/evolution_v2/numerical_qd"
BUILDER = ROOT / "tests/build_evolution_v2_numerical_fixture.py"


def cli():
    from evolving_loop.v2 import cli as module

    return module


def snapshot(root: Path) -> dict[str, tuple[str, int]]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def build_fixture(root: Path) -> tuple[Path, Path]:
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--output-dir", str(root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return root / "seed_supply.json", root / "task_manifest.json"


def numerical(config: Path, seed: Path, tasks: Path, output: Path) -> int:
    return cli().main(
        [
            "numerical-evolve",
            "--config",
            str(config),
            "--seed-supply",
            str(seed),
            "--task-manifest",
            str(tasks),
            "--output-dir",
            str(output),
        ]
    )


def test_parser_exposes_numerical_evolve_with_all_required_flags():
    parser = cli().build_parser()
    parsed = parser.parse_args(
        [
            "numerical-evolve",
            "--config",
            "config.json",
            "--seed-supply",
            "seed.json",
            "--task-manifest",
            "tasks.json",
            "--output-dir",
            "run",
        ]
    )
    assert parsed.command == "numerical-evolve"
    for args in (
        ["numerical-evolve"],
        ["numerical-evolve", "--config", "config.json"],
        [
            "numerical-evolve",
            "--config",
            "config.json",
            "--seed-supply",
            "seed.json",
        ],
        [
            "numerical-evolve",
            "--config",
            "config.json",
            "--seed-supply",
            "seed.json",
            "--task-manifest",
            "tasks.json",
        ],
    ):
        with pytest.raises(SystemExit) as error:
            parser.parse_args(args)
        assert error.value.code == 2


def test_fixture_builder_writes_exact_grouped_train80_dev20_and_refuses_nonempty(tmp_path):
    output = tmp_path / "fixtures"
    seed_path, manifest_path = build_fixture(output)
    assert set(path.name for path in output.iterdir()) == {
        "seed_supply.json",
        "task_manifest.json",
    }
    for path in (seed_path, manifest_path):
        payload = json.loads(path.read_bytes())
        assert path.read_bytes() == canonical_v2_bytes(payload)
    parse_numerical_supply_release(json.loads(seed_path.read_bytes()))
    manifest = json.loads(manifest_path.read_bytes())
    assert set(manifest) == {
        "schema_version",
        "fold_manifest",
        "train",
        "dev",
    }
    assert len(manifest["train"]) == 80
    assert len(manifest["dev"]) == 20
    assert len({row["task_id"] for row in manifest["train"]}) == 80
    assert len({row["task_id"] for row in manifest["dev"]}) == 20
    assert not ({row["task_id"] for row in manifest["train"]} & {
        row["task_id"] for row in manifest["dev"]
    })
    assert b"public" not in manifest_path.read_bytes().lower()
    before = snapshot(output)
    second = subprocess.run(
        [sys.executable, str(BUILDER), "--output-dir", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert second.returncode == 2
    assert "empty" in second.stderr
    assert snapshot(output) == before


@pytest.mark.parametrize(
    "name,profile,provider,seconds,reserve",
    [
        ("smoke", "smoke", "deterministic", 600, 0.2),
        ("pilot", "pilot", "hybrid", 7200, 0.2),
        ("formal", "formal", "hybrid", 14400, 0.2),
    ],
)
def test_shipped_numerical_profiles_are_canonical_and_truthful(
    name, profile, provider, seconds, reserve
):
    path = PROFILES / f"{name}.json"
    config = load_numerical_qd_config(path)
    assert path.read_bytes() == config.canonical_bytes()
    assert config.profile == profile
    assert config.proposer["provider"] == provider
    assert config.budget.hard_limit_seconds == seconds
    assert config.budget.finalization_reserve_fraction == reserve
    assert config.hyperband["brackets"] == {
        "explore": (8, 32, 80),
        "confirm": (32, 80),
        "replay": (80,),
    }


@pytest.mark.parametrize("damage", ["duplicate", "nan", "noncanonical", "array"])
def test_invalid_canonical_inputs_fail_before_output(tmp_path, damage):
    seed, tasks = build_fixture(tmp_path / "fixtures")
    target = {"duplicate": seed, "nan": tasks, "noncanonical": seed, "array": tasks}[damage]
    if damage == "duplicate":
        target.write_bytes(b'{"schema_version":1,' + target.read_bytes()[1:])
    elif damage == "nan":
        target.write_bytes(target.read_bytes().replace(b"0.0", b"NaN", 1))
    elif damage == "noncanonical":
        target.write_text(json.dumps(json.loads(target.read_bytes()), indent=2))
    else:
        target.write_bytes(b"[]\n")
    output = tmp_path / "run"
    assert numerical(PROFILES / "smoke.json", seed, tasks, output) == 2
    assert not output.exists()


@pytest.mark.parametrize("kind", ["input_symlink", "output_child", "output_parent", "output_symlink"])
def test_filesystem_identity_overlap_and_links_fail_before_adapter_access(
    tmp_path, monkeypatch, kind
):
    fixture_root = tmp_path / "fixtures"
    seed, tasks = build_fixture(fixture_root)
    output = tmp_path / "run"
    if kind == "input_symlink":
        alias = tmp_path / "seed-link.json"
        alias.symlink_to(seed)
        seed = alias
    elif kind == "output_child":
        output = fixture_root / "run"
    elif kind == "output_parent":
        output = tmp_path
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        output.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        cli(),
        "_build_numerical_adapter",
        lambda *args, **kwargs: pytest.fail("adapter constructed before preflight"),
        raising=False,
    )
    before = snapshot(tmp_path)
    assert numerical(PROFILES / "smoke.json", seed, tasks, output) == 2
    assert snapshot(tmp_path) == before


def test_preflight_accepts_physical_system_tmp_alias_without_creating_output(tmp_path):
    seed, tasks = build_fixture(tmp_path / "fixtures")
    with tempfile.TemporaryDirectory(dir="/tmp") as root:
        output = Path(root) / "run"
        assert cli()._preflight_numerical_paths(
            (PROFILES.joinpath("smoke.json").resolve(), seed.resolve(), tasks.resolve()),
            output,
        ) is False
        assert not output.exists()


def test_physical_system_tmp_alias_reaches_v2_bootstrap_authority():
    with tempfile.TemporaryDirectory(dir="/tmp") as root:
        output = Path(root) / "run"
        store = V2RunStore.create(output)
        SeedBootstrapAuthority.verify_preseed_paths(store)


def test_material_receipt_traverses_physical_system_tmp_alias():
    raw = b"def forecast():\n    return [1.0]\n"
    digest = hashlib.sha256(raw).hexdigest()
    with tempfile.TemporaryDirectory(dir="/tmp") as root:
        path = Path(root) / "numerical_qd" / "sources" / f"{digest}.py"
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        receipt = {
            "kind": "source",
            "relative_path": f"sources/{digest}.py",
            "content_sha256": digest,
            "size_bytes": len(raw),
        }
        assert _material_bytes(Path(root), receipt) == raw


def test_smoke_executes_under_physical_system_tmp_alias():
    with tempfile.TemporaryDirectory(dir="/tmp") as root:
        fixture_root = Path(root) / "fixtures"
        seed, tasks = build_fixture(fixture_root)
        output = Path(root) / "run"
        assert numerical(PROFILES / "smoke.json", seed, tasks, output) == 0
        assert json.loads((output / "evaluation_complete.json").read_bytes())["status"] == (
            "numerical_qd_complete"
        )


@pytest.fixture(scope="module")
def completed_smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp("numerical-cli")
    seed, tasks = build_fixture(root / "fixtures")
    inputs_before = snapshot(root / "fixtures")
    output = root / "run"
    assert numerical(PROFILES / "smoke.json", seed, tasks, output) == 0
    return root, seed, tasks, output, inputs_before


def test_smoke_executes_real_runner_and_writes_truthful_completion(completed_smoke):
    root, _seed, _tasks, output, inputs_before = completed_smoke
    assert snapshot(root / "fixtures") == inputs_before
    completion = json.loads((output / "evaluation_complete.json").read_bytes())
    assert completion["status"] == "numerical_qd_complete"
    assert set(completion["summary"]) == {
        "supply_sha256",
        "registry_sha256",
        "bundle_sha256",
        "qd_snapshot_sha256",
        "mutation_policy_sha256",
        "proposer_prompt_sha256",
        "occupied_cells",
        "accepted_count",
        "rejected_count",
        "provider_attempts",
        "llm_attempts",
        "llm_calls",
        "llm_provider_used",
        "budget",
        "dev_accessed",
        "public_test_accessed",
    }
    summary = completion["summary"]
    for key in (
        "supply_sha256",
        "registry_sha256",
        "bundle_sha256",
        "qd_snapshot_sha256",
        "mutation_policy_sha256",
        "proposer_prompt_sha256",
    ):
        assert len(summary[key]) == 64
    assert summary["occupied_cells"] >= 1
    assert summary["provider_attempts"] >= 1
    assert summary["llm_provider_used"] is False
    assert type(summary["dev_accessed"]) is bool
    assert summary["public_test_accessed"] is False
    forbidden = {
        "retrieval_score",
        "decision_score",
        "joint_score",
        "public_score",
        "real_model_used",
    }
    assert not forbidden.intersection(completion) and not forbidden.intersection(summary)


def test_completed_resume_is_verified_mtime_noop(completed_smoke, capsys):
    _root, seed, tasks, output, _inputs_before = completed_smoke
    before = snapshot(output)
    assert numerical(PROFILES / "smoke.json", seed, tasks, output) == 0
    assert snapshot(output) == before
    emitted = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert emitted == json.loads((output / "evaluation_complete.json").read_bytes())


def test_semantically_equal_but_byte_distinct_task_input_rejects_resume_without_writes(
    completed_smoke, tmp_path
):
    _root, seed, tasks, output, _inputs_before = completed_smoke
    payload = json.loads(tasks.read_bytes())
    original = payload["train"][0]["history_values"][0]
    assert type(original) is float
    payload["train"][0]["history_values"][0] = int(original)
    changed = tmp_path / "task_manifest.json"
    changed.write_bytes(canonical_v2_bytes(payload))
    assert changed.read_bytes() != tasks.read_bytes()
    before = snapshot(output)
    assert numerical(PROFILES / "smoke.json", seed, changed, output) == 2
    assert snapshot(output) == before


def test_project_one_cli_commands_keep_their_existing_behavior(tmp_path, capsys):
    from evolving_loop.v2.fakes import smoke_config

    config = tmp_path / "config.json"
    config.write_bytes(canonical_v2_bytes(smoke_config().to_payload()))
    run = tmp_path / "fake"
    assert cli().main(["evolve", "--config", str(config), "--output-dir", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "deterministic_fake_complete"
    public = tmp_path / "public"
    assert cli().main([
        "public-evaluate",
        "--bundle",
        str(run / "accepted_bundle.json"),
        "--output-dir",
        str(public),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated_only_no_public_evaluator"


def test_module_entrypoint_runs_numerical_evolve(completed_smoke, tmp_path):
    _root, seed, tasks, _output, _inputs_before = completed_smoke
    output = tmp_path / "entrypoint-run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolving_loop.v2",
            "numerical-evolve",
            "--config",
            str(PROFILES / "smoke.json"),
            "--seed-supply",
            str(seed),
            "--task-manifest",
            str(tasks),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "numerical_qd_complete"
