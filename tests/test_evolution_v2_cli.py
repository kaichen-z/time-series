import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evolving_loop.v2.contracts import canonical_v2_bytes, load_v2_config
from evolving_loop.v2.fakes import run_fake_kernel, smoke_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "evolution_v2"


def api():
    return importlib.import_module("evolving_loop.v2.cli")


def snapshot(root):
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def config_file(tmp_path, **changes):
    path = tmp_path / "config.json"
    path.write_bytes(canonical_v2_bytes(smoke_config().to_payload() | changes))
    return str(path)


def evolve(config, output):
    return api().main(["evolve", "--config", str(config), "--output-dir", str(output)])


def public(bundle, output):
    return api().main(
        ["public-evaluate", "--bundle", str(bundle), "--output-dir", str(output)]
    )


def test_v2_parser_exposes_only_parallel_commands():
    parser = api().build_parser()
    assert (
        parser.parse_args(
            ["evolve", "--config", "c.json", "--output-dir", "out"]
        ).command
        == "evolve"
    )
    assert (
        parser.parse_args(
            ["public-evaluate", "--bundle", "b.json", "--output-dir", "pub"]
        ).command
        == "public-evaluate"
    )


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["evolve"],
        ["evolve", "--config", "c"],
        ["public-evaluate"],
        ["public-evaluate", "--bundle", "b"],
        ["legacy"],
    ],
)
def test_missing_flags_and_legacy_commands_rejected(args):
    parser = api().build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(args)
    assert error.value.code == 2


def test_v2_cli_never_dispatches_to_legacy_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "evolving_loop.cli.main", lambda *args: pytest.fail("legacy CLI called")
    )
    assert evolve(config_file(tmp_path), tmp_path / "run") == 0
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert output.encode() == canonical_v2_bytes(summary)
    assert summary["status"] == "deterministic_fake_complete"
    assert summary["accepted_steps"] == summary["rejected_steps"] == 1
    assert summary["public_test_accessed"] is False


@pytest.mark.parametrize(
    "profile,runner,message",
    [
        ("pilot", "production", "production evolution requires Project 2+ adapters"),
        ("formal", "production", "production evolution requires Project 2+ adapters"),
        ("pilot", "deterministic_fake", "smoke"),
        ("public", "deterministic_fake", "public-evaluate"),
    ],
)
def test_runner_profile_rejection_precedes_output(
    tmp_path, capsys, profile, runner, message
):
    config = config_file(
        tmp_path,
        profile=profile,
        runner=runner,
        hard_limit_seconds=14400 if profile == "formal" else 600,
    )
    assert evolve(config, tmp_path / "run") == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "scopes", [["numerical"], ["numerical", "retrieval", "decision"]]
)
def test_fake_refuses_scopes_it_cannot_truthfully_execute(tmp_path, scopes):
    assert (
        evolve(config_file(tmp_path, enabled_mutation_scopes=scopes), tmp_path / "run")
        == 2
    )
    assert not (tmp_path / "run").exists()


def test_strict_config_errors_leave_no_output(tmp_path):
    cli = api()
    path = tmp_path / "config.json"
    for raw in ('{"seed":1,"seed":2}', '{"seed":NaN}', "[]", "{}"):
        path.write_text(raw)
        assert (
            cli.main(
                ["evolve", "--config", str(path), "--output-dir", str(tmp_path / "run")]
            )
            == 2
        )
        assert not (tmp_path / "run").exists()


def test_existing_legacy_run_rejected_without_writes(tmp_path):
    config = config_file(tmp_path)
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "run_manifest.json").write_text('{"system":"legacy"}')
    before = snapshot(root)
    assert evolve(config, root) == 2
    assert snapshot(root) == before


def test_empty_directory_and_exact_resume(tmp_path):
    config = config_file(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    assert evolve(config, root) == 0
    before = snapshot(root)
    assert evolve(config, root) == 0
    assert snapshot(root) == before
    (root / "evaluation_complete.json").write_text("{}")
    corrupt = snapshot(root)
    assert evolve(config, root) == 2
    assert snapshot(root) == corrupt


def test_partial_resume_and_config_mismatch(tmp_path):
    config = config_file(tmp_path)
    root = tmp_path / "run"
    run_fake_kernel(root, smoke_config(), stop_after_stage="fake-numerical")
    before = snapshot(root)
    bad = config_file(tmp_path, seed=8)
    assert evolve(bad, root) == 2
    assert snapshot(root) == before
    config = config_file(tmp_path)
    assert evolve(config, root) == 0
    assert (
        json.loads((root / "evaluation_complete.json").read_bytes())["rejected_steps"]
        == 1
    )


def test_public_only_validates_and_never_mutates_source(tmp_path, capsys):
    root, output = tmp_path / "run", tmp_path / "public"
    result = run_fake_kernel(root, smoke_config())
    before = snapshot(root)
    index = (root / "archive/index.jsonl").read_bytes()
    assert public(root / "accepted_bundle.json", output) == 0
    assert snapshot(root) == before
    assert (root / "archive/index.jsonl").read_bytes() == index
    identity = result.accepted_bundle.fingerprint()
    assert set(snapshot(output)) == {
        "run_manifest.json",
        f"objects/{identity}.json",
        "evaluation_complete.json",
    }
    assert (
        output / f"objects/{identity}.json"
    ).read_bytes() == result.accepted_bundle.canonical_bytes()
    completion = json.loads((output / "evaluation_complete.json").read_bytes())
    assert completion["status"] == "validated_only_no_public_evaluator"
    assert completion["public_test_accessed"] is False
    assert completion["bundle_sha256"] == identity
    assert capsys.readouterr().out.encode() == canonical_v2_bytes(completion)
    public_before = snapshot(output)
    assert public(root / "accepted_bundle.json", output) == 2
    assert snapshot(output) == public_before


@pytest.mark.parametrize(
    "overlap", ["same", "child", "ancestor", "symlink_child", "symlink_source"]
)
def test_public_rejects_source_output_containment(tmp_path, overlap):
    root = tmp_path / "run"
    run_fake_kernel(root, smoke_config())
    bundle = root / "accepted_bundle.json"
    if overlap == "same":
        output = root
    elif overlap == "child":
        output = root / "public"
    elif overlap == "ancestor":
        output = tmp_path
    elif overlap == "symlink_child":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        output = alias / "public"
    else:
        bundle = tmp_path / "accepted_bundle.json"
        bundle.symlink_to(root / "accepted_bundle.json")
        output = root / "public"
    before = snapshot(root)
    assert public(bundle, output) == 2
    assert snapshot(root) == before
    assert not (root / "public").exists()


@pytest.mark.parametrize(
    "overlap", ["child", "deep_child", "existing_child", "same", "ancestor"]
)
def test_public_rejects_case_alias_by_filesystem_identity(tmp_path, capsys, overlap):
    root = tmp_path / "case-sensitive-run"
    run_fake_kernel(root, smoke_config())
    alias = root.with_name(root.name.upper())
    if not alias.exists() or not alias.samefile(root):
        pytest.skip("requires a filesystem that aliases case variants")
    if overlap == "child":
        output = alias / "public"
    elif overlap == "deep_child":
        output = alias / "missing-parent" / "public"
    elif overlap == "existing_child":
        (root / "existing").mkdir()
        output = alias / "existing" / "public"
    elif overlap == "same":
        output = alias
    else:
        output = tmp_path.with_name(tmp_path.name.upper())
    before = snapshot(root)
    directories = {str(path.relative_to(root)) for path in root.rglob("*")}
    assert public(root / "accepted_bundle.json", output) == 2
    assert "must not overlap" in capsys.readouterr().err
    assert snapshot(root) == before
    assert {str(path.relative_to(root)) for path in root.rglob("*")} == directories


@pytest.mark.parametrize(
    "damage",
    [
        "unsealed",
        "bad_evidence",
        "unknown",
        "duplicate",
        "nan",
        "noncanonical",
        "fingerprint",
        "archive_corrupt",
        "manifest",
        "missing",
    ],
)
def test_public_rejects_invalid_or_unbound_bundle_before_output(tmp_path, damage):
    root, output = tmp_path / "run", tmp_path / "public"
    result = run_fake_kernel(root, smoke_config())
    path = root / "accepted_bundle.json"
    payload = json.loads(path.read_bytes())
    if damage == "unsealed":
        payload["acceptance_evidence_sha256"] = None
    elif damage == "bad_evidence":
        payload["acceptance_evidence_sha256"] = "../evidence"
    elif damage == "unknown":
        payload["public_score"] = 1
    elif damage == "fingerprint":
        payload["decision_policy_sha256"] = "f" * 64
    path.write_bytes(canonical_v2_bytes(payload))
    if damage == "duplicate":
        path.write_bytes(b'{"generation":1,' + path.read_bytes()[1:])
    elif damage == "nan":
        path.write_bytes(
            path.read_bytes().replace(b'"generation":1', b'"generation":NaN')
        )
    elif damage == "noncanonical":
        path.write_text(json.dumps(payload, indent=2))
    elif damage == "archive_corrupt":
        (
            root / f"archive/objects/{result.accepted_bundle.fingerprint()}.json"
        ).write_text("{}")
    elif damage == "manifest":
        (root / "run_manifest.json").write_text("{}")
    elif damage == "missing":
        path.unlink()
    before = snapshot(root)
    assert public(path, output) == 2
    assert not output.exists()
    assert snapshot(root) == before


@pytest.mark.parametrize(
    "name,seconds,runner",
    [
        ("smoke", 600, "deterministic_fake"),
        ("pilot", 7200, "production"),
        ("formal", 14400, "production"),
    ],
)
def test_shipped_configs_have_truthful_profiles_and_reproducible_component_digests(
    name, seconds, runner
):
    assert (CONFIGS / f"{name}.json").exists(), "profile config is absent"
    config = load_v2_config(CONFIGS / f"{name}.json")
    payload = config.to_payload()
    assert config.hard_limit_seconds == seconds
    assert config.runner == runner
    assert config.finalization_reserve_fraction == 0.2
    assert config.hyperband["dev_tasks"] == (20 if name == "formal" else 2)
    assert config.hyperband["resource_levels"] == (
        (8, 32, 80) if name == "formal" else (8,)
    )
    assert config.hyperband["children_per_enabled_arm"] == (
        1 if name == "smoke" else None
    )
    assert config.hyperband["arms_per_epoch"] == (1 if name == "formal" else None)
    assert set(config.enabled_mutation_scopes) == (
        {"numerical", "retrieval"}
        if name == "smoke"
        else {"numerical", "retrieval", "decision", "joint"}
    )
    base = {
        key: value
        for key, value in payload.items()
        if key not in {"kernel_protocol", "runtime_fingerprints"}
    }
    for component, digest in {
        **payload["kernel_protocol"],
        **payload["runtime_fingerprints"],
    }.items():
        preimage = {
            "binding_kind": "config_intent",
            "component": component,
            "configuration": base,
        }
        raw = (
            json.dumps(
                preimage,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        assert digest == hashlib.sha256(raw).hexdigest()
        assert digest != "0" * 64


def test_module_entrypoint_executes_both_commands(tmp_path):
    config = config_file(tmp_path)
    root = tmp_path / "run"
    for command in (
        ["evolve", "--config", config, "--output-dir", str(root)],
        [
            "public-evaluate",
            "--bundle",
            str(root / "accepted_bundle.json"),
            "--output-dir",
            str(tmp_path / "public"),
        ],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "evolving_loop.v2", *command],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["public_test_accessed"] is False
