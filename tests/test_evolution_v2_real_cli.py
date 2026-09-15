from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2, RealRunResultV2
from evolving_loop.v2.real.runner import RealStageContextV2


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _manifest(path: Path, *, input_path: str | None = None) -> Path:
    payload = {
        "schema_version": 1,
        "profile": "real-30m",
        "model": {"schema_version": 1, "name": "gpt-5.6-luna", "reasoning_effort": "medium"},
        "files": [] if input_path is None else [{"role": "split", "relative_path": input_path, "sha256": _digest("split")}],
        "runtime_locations": [],
        "l0_fingerprints": {},
    }
    path.write_bytes(canonical_v2_bytes(payload))
    return path


def _result(manifest: RealEvolutionManifestV2) -> RealRunResultV2:
    return RealRunResultV2("incomplete", manifest.fingerprint(), manifest.model.fingerprint(), (), None, False)


def test_real_evolve_dispatches_canonical_manifest_and_closes_host(tmp_path: Path, monkeypatch, capsys):
    from evolving_loop.v2 import cli

    manifest_path = _manifest(tmp_path / "real.json")
    data_root = tmp_path / "authority"
    data_root.mkdir()
    output = tmp_path / "output"
    calls = []

    class Host:
        def close(self):
            calls.append("close")

    def build(manifest, *, repo_root, output_dir, **_kwargs):
        calls.append(("build", manifest, repo_root, output_dir))
        return Host()

    def run(output_dir, manifest, ports):
        calls.append(("run", output_dir, manifest, ports))
        return _result(manifest)

    monkeypatch.setattr(cli, "build_real_host", build)
    monkeypatch.setattr(cli, "build_real_stage_ports", lambda host, **kwargs: object())
    monkeypatch.setattr(cli, "run_real_evolution", run)

    assert cli.main(["real-evolve", "--manifest", str(manifest_path), "--authority-root", str(data_root), "--output-dir", str(output)]) == 0

    stdout = capsys.readouterr().out.encode()
    assert stdout == canonical_v2_bytes(_result(RealEvolutionManifestV2.from_payload({
        "schema_version": 1, "profile": "real-30m",
        "model": {"schema_version": 1, "name": "gpt-5.6-luna", "reasoning_effort": "medium"},
        "files": [], "runtime_locations": [], "l0_fingerprints": {},
    })).to_payload())
    assert calls[-1] == "close"
    assert calls[0][0] == "build"
    assert calls[0][2] == data_root.resolve()


def test_real_evolve_forwards_explicit_p2_generation_target(
    tmp_path: Path, monkeypatch, capsys
):
    from evolving_loop.v2 import cli

    manifest_path = _manifest(tmp_path / "real.json")
    data_root = tmp_path / "authority"
    data_root.mkdir()
    observed = {}

    class Host:
        def close(self):
            pass

    monkeypatch.setattr(cli, "build_real_host", lambda *_args, **_kwargs: Host())

    def build_ports(_host, **kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "build_real_stage_ports", build_ports)
    def run(_output, manifest, _ports, **kwargs):
        observed.update({f"run_{key}": value for key, value in kwargs.items()})
        return _result(manifest)

    monkeypatch.setattr(cli, "run_real_evolution", run)

    assert cli.main([
        "real-evolve",
        "--manifest", str(manifest_path),
        "--authority-root", str(data_root),
        "--output-dir", str(tmp_path / "output"),
        "--p2-generations", "10",
    ]) == 0

    capsys.readouterr()
    assert observed["p2_generations"] == 10
    assert observed["run_p2_generations"] == 10


def test_real_evolve_rejects_output_inside_data_root_before_host_creation(tmp_path: Path, monkeypatch, capsys):
    from evolving_loop.v2 import cli

    data_root = tmp_path / "authority"
    data_root.mkdir()
    (data_root / "input").write_text("split", encoding="utf-8")
    manifest_path = _manifest(tmp_path / "real.json", input_path="input")
    monkeypatch.setattr(cli, "build_real_host", lambda *_args, **_kwargs: pytest.fail("Host must not be built"))

    assert cli.main(["real-evolve", "--manifest", str(manifest_path), "--authority-root", str(data_root), "--output-dir", str(data_root / "input" / "run")]) == 2
    assert "must not overlap" in capsys.readouterr().err


def test_real_evolve_closes_host_when_a_child_port_fails(tmp_path: Path, monkeypatch, capsys):
    from evolving_loop.v2 import cli

    data_root = tmp_path / "authority"
    data_root.mkdir()
    closed = []

    class Host:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(cli, "build_real_host", lambda *_args, **_kwargs: Host())
    monkeypatch.setattr(cli, "build_real_stage_ports", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "run_real_evolution", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("child failed")))

    assert cli.main(["real-evolve", "--manifest", str(_manifest(tmp_path / "real.json")), "--authority-root", str(data_root), "--output-dir", str(tmp_path / "output")]) == 2
    assert closed == [True]
    assert "child failed" in capsys.readouterr().err


def test_real_cli_production_modules_do_not_import_test_or_fake_runtime():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "evolving_loop/v2/cli.py", root / "evolving_loop/v2/real/runner.py"):
        source = path.read_text(encoding="utf-8")
        assert "from tests" not in source
        if path.name == "runner.py":
            assert "FakeLLM" not in source


def test_real_manifests_bind_historical_forecast_runtime():
    root = Path(__file__).resolve().parents[1]
    for profile in ("real-30m", "real-1h"):
        manifest = RealEvolutionManifestV2.from_payload(
            json.loads(
                (root / f"configs/evolution_v2/real/{profile}-toto-balanced-v3.json")
                .read_text(encoding="utf-8")
            )
        )
        locations = {row.role: row.relative_path for row in manifest.runtime_locations}
        assert locations["runtime"] == "tmp/toto2_worker_smoke.json"
        assert locations["model_cache"] == "outputs/model-cache/hub"


def test_real_p2_initial_priors_fail_closed_on_corrupt_cache():
    from evolving_loop.v2.real.runner import _initial_prior_rows

    numeric = SimpleNamespace(
        task_id="task_1",
        entity_name="entity",
        history_values=(1.0, 2.0, 3.0),
        future_values=(4.0,),
        prediction_length=1,
        frequency="1 day",
        seasonal_period=1,
    )

    class CorruptStore:
        not_applicable = type("NotApplicable", (Exception,), {})

        def forecast(self, *_args):
            raise ValueError("corrupt cache row")

    host = SimpleNamespace(
        train_tasks=(SimpleNamespace(numeric=numeric),),
        forecast_store=CorruptStore(),
    )
    with pytest.raises(ValueError, match="corrupt cache row"):
        _initial_prior_rows(host, (("naive_last", "statistical"),))


def test_real_p2_initial_priors_check_deadline_before_each_cache_read():
    from evolving_loop.v2.real.runner import _initial_prior_rows

    numeric = SimpleNamespace(
        task_id="task_1",
        entity_name="entity",
        history_values=(1.0, 2.0, 3.0),
        future_values=(4.0,),
        prediction_length=1,
        frequency="1 day",
        seasonal_period=1,
    )
    calls = []

    class Store:
        not_applicable = type("NotApplicable", (Exception,), {})

        def forecast(self, name, *_args):
            calls.append(name)
            return (4.0,)

    checks = [None, RuntimeError("deadline")]

    def check_deadline():
        result = checks.pop(0)
        if result is not None:
            raise result

    host = SimpleNamespace(
        train_tasks=(SimpleNamespace(numeric=numeric),), forecast_store=Store()
    )
    with pytest.raises(RuntimeError, match="deadline"):
        _initial_prior_rows(
            host,
            (("naive_last", "statistical"), ("naive_mean", "statistical")),
            check_deadline=check_deadline,
        )

    assert calls == ["naive_last"]


def test_real_source_seed_binds_derived_p3_protocol():
    from evolving_loop.v2.cooperative import CooperativeConfigV2
    from evolving_loop.v2.real.runner import _derived_cooperative_config
    from evolving_loop.v2.source import SourceVariantV2

    root = Path(__file__).resolve().parents[1]
    seed = SourceVariantV2.from_payload(
        json.loads(
            (root / "configs/evolution_v2/real/source-seed.json").read_text(
                encoding="utf-8"
            )
        )
    )
    context = SimpleNamespace(grant_seconds=360)
    host = SimpleNamespace(resource_reporter_sha256=seed.runtime_fingerprint)
    config = CooperativeConfigV2.from_payload(
        _derived_cooperative_config(context, host)
    )
    assert seed.protocol_fingerprint == config.control.kernel_protocol.fingerprint()
    ceilings = config.resource_ceilings
    assert ceilings.llm_calls > 0
    assert ceilings.input_tokens > 0
    assert ceilings.output_tokens > 0
    assert ceilings.subprocesses > 0
    source = (root / "evolving_loop/v2/real/runner.py").read_text(encoding="utf-8")
    assert "cooperative/smoke-ucb.json" not in source


def test_real_p2_reads_sha_bound_noncanonical_champion(tmp_path: Path):
    from evolving_loop.v2.real.runner import _read_sha_bound_json

    path = tmp_path / "champion.json"
    raw = b'{\n  "lineage": []\n}\n'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert _read_sha_bound_json(path, digest) == {"lineage": []}
    with pytest.raises(ValueError, match="identity mismatch"):
        _read_sha_bound_json(path, "0" * 64)


def test_production_ports_prepare_and_run_real_p2(tmp_path: Path, monkeypatch):
    """Catches replacing the production P2 adapter with an unavailable stub."""
    from evolving_loop.v2.real import bridges, runner

    manifest = RealEvolutionManifestV2.from_payload(
        {
            "schema_version": 1,
            "profile": "real-30m",
            "model": {
                "schema_version": 1,
                "name": "gpt-5.6-luna",
                "reasoning_effort": "medium",
            },
            "files": [],
            "runtime_locations": [],
            "l0_fingerprints": {},
        }
    )
    now = [100.0]
    context = RealStageContextV2(
        "p2",
        tmp_path / "root/p2",
        840,
        manifest,
        manifest.fingerprint(),
        manifest.model.fingerprint(),
        {},
        deadline_monotonic=940.0,
        monotonic=lambda: now[0],
    )
    host = SimpleNamespace(llm_client=object())
    prepared = SimpleNamespace(
        config_payload={"config": True},
        seed_payload={"seed": True},
        task_manifest_payload={"tasks": True},
        input_sha256s={
            "config": "1" * 64,
            "seed_supply": "2" * 64,
            "task_manifest": "3" * 64,
            "champion_release": "4" * 64,
            "dictionary": "5" * 64,
            "task_local_evidence": "6" * 64,
        },
        evidence_path=tmp_path / "root/prepared/p2",
        dictionary=object(),
    )
    observed = {}
    def prepare(*_args, **kwargs):
        assert kwargs["remaining_seconds"]() == 840
        now[0] += 275
        assert kwargs["remaining_seconds"]() == 565
        return prepared

    monkeypatch.setattr(runner, "prepare_real_p2_inputs", prepare, raising=False)
    def run_real_numerical(context_value, host_value, **kwargs):
        observed.update(context=context_value, host=host_value, **kwargs)
        return {"status": "numerical_qd_complete"}

    monkeypatch.setattr(bridges, "run_real_numerical", run_real_numerical)

    ports = runner.build_real_stage_ports(
        host, manifest=manifest, repo_root=tmp_path
    )

    assert ports.run_p2(context) == {"status": "numerical_qd_complete"}
    assert observed == {
        "context": context,
        "host": host,
        "config_payload": prepared.config_payload,
        "seed_payload": prepared.seed_payload,
        "task_manifest_payload": prepared.task_manifest_payload,
        "input_sha256s": {
            "config": "1" * 64,
            "seed_supply": "2" * 64,
            "task_manifest": "3" * 64,
        },
        "task_local_evidence_path": prepared.evidence_path,
        "task_local_dictionary": prepared.dictionary,
    }


def test_production_p2_finishes_incomplete_when_preparation_consumes_grant(
    tmp_path: Path, monkeypatch
):
    from evolving_loop.v2.real import bridges, runner

    manifest = RealEvolutionManifestV2.from_payload(
        {
            "schema_version": 1,
            "profile": "real-30m",
            "model": {
                "schema_version": 1,
                "name": "gpt-5.6-luna",
                "reasoning_effort": "medium",
            },
            "files": [],
            "runtime_locations": [],
            "l0_fingerprints": {},
        }
    )
    now = [100.0]
    context = RealStageContextV2(
        "p2",
        tmp_path / "root/p2",
        10,
        manifest,
        manifest.fingerprint(),
        manifest.model.fingerprint(),
        {},
        deadline_monotonic=110.0,
        monotonic=lambda: now[0],
    )

    def exhausted(*_args, **kwargs):
        now[0] = 111.0
        assert kwargs["remaining_seconds"]() == 0
        raise runner._RealStageBudgetExhausted("consumed")

    monkeypatch.setattr(runner, "prepare_real_p2_inputs", exhausted)
    monkeypatch.setattr(
        bridges,
        "run_real_numerical",
        lambda *_args, **_kwargs: pytest.fail("numerical child must not start"),
    )
    ports = runner.build_real_stage_ports(
        SimpleNamespace(llm_client=object()), manifest=manifest, repo_root=tmp_path
    )

    result = ports.run_p2(context)
    sealed = ports.seal_p2(context, result)

    assert sealed.summary["status"] == "incomplete"
    assert sealed.handoff_payload["reason"] == "p2_preparation_budget_exhausted"


def test_production_ports_complete_root_and_resume_byte_identically(
    tmp_path: Path, monkeypatch
):
    """Exercises the real port assembly while replacing only expensive children."""
    from evolving_loop.v2 import cooperative, protocol
    from evolving_loop.v2.real import bridges, runner
    from evolving_loop.v2.source import archive as source_archive
    from evolving_loop.v2.source import authority as source_authority
    from evolving_loop.v2 import source
    from evolving_loop.v2.numerical_qd import persistence
    from evolving_loop.v2.source import SourceRunResultV2

    manifest = RealEvolutionManifestV2.from_payload(
        {
            "schema_version": 1,
            "profile": "real-30m",
            "model": {
                "schema_version": 1,
                "name": "gpt-5.6-luna",
                "reasoning_effort": "medium",
            },
            "files": [
                {
                    "role": "source_seed",
                    "relative_path": "configs/evolution_v2/real/source-seed.json",
                    "sha256": "a" * 64,
                }
            ],
            "runtime_locations": [],
            "l0_fingerprints": {},
        }
    )
    prepared = SimpleNamespace(
        config_payload={}, seed_payload={}, task_manifest_payload={},
        input_sha256s={
            "config": "1" * 64,
            "seed_supply": "2" * 64,
            "task_manifest": "3" * 64,
            "champion_release": "4" * 64,
            "dictionary": "5" * 64,
            "task_local_evidence": "6" * 64,
        },
        evidence_path=tmp_path / "prepared/p2",
        dictionary=object(),
    )
    monkeypatch.setattr(runner, "prepare_real_p2_inputs", lambda *_a, **_k: prepared)
    monkeypatch.setattr(runner, "_load_prepared_real_p2", lambda *_a, **_k: prepared)

    host = SimpleNamespace(
        tasks=(), train_tasks=(), dev_tasks=(), llm_client=object(),
        resource_reporter_sha256="36fdec41981b25e148d6bad23cbf2cef4926f9a49d98035830d24a32f55c9bcc",
    )
    release = SimpleNamespace(fingerprint="b" * 64)
    registry = SimpleNamespace(fingerprint="c" * 64)
    envelope = SimpleNamespace(fingerprint=lambda: "d" * 64)
    pair = SimpleNamespace(
        release=release, registry=registry, envelope=envelope,
        selected_genome_sha256s=(),
    )
    alternative = SimpleNamespace(
        release=SimpleNamespace(fingerprint="8" * 64),
        registry=SimpleNamespace(fingerprint="9" * 64),
        envelope=SimpleNamespace(fingerprint=lambda: "a" * 64),
        selected_genome_sha256s=("f" * 64,),
    )

    class PairStore:
        def __init__(self, _root):
            pass

        def load_active_frozen_pair(self, *, tasks):
            assert tasks == host.tasks
            return pair, "e" * 64

        def load_frozen_pairs(self, *, tasks):
            assert tasks == host.tasks
            return ((pair, "e" * 64), (alternative, "f" * 64))

    monkeypatch.setattr(persistence, "NumericalQDRunStore", PairStore)
    dictionary = SimpleNamespace(fingerprint=lambda: "0" * 64)
    monkeypatch.setattr(
        cooperative,
        "build_p3_numerical_dictionary",
        lambda pairs, tasks: dictionary,
    )

    def numerical(context, _host, **_kwargs):
        payload = {
            "schema_version": 1,
            "status": "numerical_qd_complete",
            "summary": {"public_test_accessed": False},
        }
        (context.output_dir / "evaluation_complete.json").write_bytes(
            canonical_v2_bytes(payload)
        )
        return payload

    monkeypatch.setattr(bridges, "run_real_numerical", numerical)

    bundle = SimpleNamespace(
        protocol_fingerprint="9691cd3607c9f7ceb6928086b3f0b91ae70a1d155926fe54aa5b14d74fc18d14",
        fingerprint=lambda: "7" * 64,
    )
    closure = SimpleNamespace(
        active_bundle=bundle,
        completion_sha256="8" * 64,
        checkpoint_sha256="1" * 64,
        proposal_space_sha256="2" * 64,
        p5_handoff_available=True,
        p5_handoff_reason=None,
    )

    def cooperative(*, output_dir, **_kwargs):
        payload = {
            "schema_version": 1,
            "status": "cooperative_complete",
            "public_test_accessed": False,
        }
        (output_dir / "evaluation_complete.json").write_bytes(
            canonical_v2_bytes(payload)
        )
        return payload

    monkeypatch.setattr(bridges, "run_real_cooperative", cooperative)
    monkeypatch.setattr(bridges, "load_sealed_bundle_closure", lambda *_a, **_k: closure)
    monkeypatch.setattr(runner, "_derived_cooperative_config", lambda *_a, **_k: {})

    monkeypatch.setattr(source, "build_source_case_from_p3", lambda *_a, **_k: object())
    source_result = SourceRunResultV2(
        1, "source_evolution_complete", "3" * 64, "4" * 64,
        0, 0, 0, 0, (), False,
    )

    def source_run(output, *_args, **_kwargs):
        output.mkdir(parents=True, exist_ok=True)
        (output / "authority/sealed").mkdir(parents=True)
        (output / "authority/active_source.json").write_bytes(
            canonical_v2_bytes({"active": "3" * 64})
        )
        (output / "source_archive/objects").mkdir(parents=True)
        (output / "source_archive/events.jsonl").write_text("", encoding="utf-8")
        (output / "evaluation_complete.json").write_bytes(source_result.canonical_bytes())
        return source_result

    monkeypatch.setattr(source, "run_source_evolution", source_run)
    monkeypatch.setattr(
        source_archive,
        "SourceArchiveV2",
        lambda _root: SimpleNamespace(snapshot_sha256=lambda: "4" * 64),
    )
    monkeypatch.setattr(
        source_authority,
        "SourceAuthorityV2",
        lambda *_a: SimpleNamespace(
            active_source=lambda: SimpleNamespace(fingerprint=lambda: "3" * 64)
        ),
    )

    class ProtocolCase:
        def run(self, output):
            handoff = {"schema_version": 1, "public_test_accessed": False}
            handoff_sha = hashlib.sha256(canonical_v2_bytes(handoff)).hexdigest()
            payload = {
                "schema_version": 1,
                "status": "protocol_evolution_complete",
                "active_protocol_sha256": "5" * 64,
                "active_release_sha256": "6" * 64,
                "frozen_handoff_sha256": handoff_sha,
                "public_test_accessed": False,
            }
            output.mkdir(parents=True, exist_ok=True)
            (output / "frozen_protocol_handoff.json").write_bytes(
                canonical_v2_bytes(handoff)
            )
            (output / "completion.json").write_bytes(canonical_v2_bytes(payload))
            return payload

    monkeypatch.setattr(
        protocol, "build_protocol_case_from_p3", lambda *_a, **_k: ProtocolCase()
    )

    output = tmp_path / "root"
    ports = runner.build_real_stage_ports(host, manifest=manifest, repo_root=tmp_path)
    first = runner.run_real_evolution(output, manifest, ports)
    first_bytes = (output / "evaluation_complete.json").read_bytes()
    second = runner.run_real_evolution(output, manifest, ports)

    assert first.status == second.status == "complete"
    assert first_bytes == (output / "evaluation_complete.json").read_bytes()
    assert host.p3_dictionary is dictionary
    p2_handoff = json.loads((output / "handoffs/p2.json").read_bytes())
    p2_bindings = p2_handoff["handoff_payload"]
    assert p2_bindings["p3_dictionary_sha256"] == dictionary.fingerprint()
    assert not {"pair_sha256", "release_sha256", "registry_sha256"} & set(
        p2_bindings
    )
    assert [record.stage for record in first.stage_records] == ["p2", "p3", "p4", "p5"]
    assert all((output / stage / "root_stage_completion.json").is_file() for stage in ("p2", "p3", "p4", "p5"))

    wrapper = output / "p3/root_stage_completion.json"
    wrapper_bytes = wrapper.read_bytes()
    wrapper.unlink()
    with pytest.raises(ValueError, match="root stage wrapper"):
        runner.run_real_evolution(output, manifest, ports)
    assert not wrapper.exists()
    wrapper.write_bytes(wrapper_bytes)

    active_source = output / "p4/authority/active_source.json"
    active_source_bytes = active_source.read_bytes()
    active_source.unlink()
    with pytest.raises(ValueError):
        runner.run_real_evolution(output, manifest, ports)
    assert not active_source.exists()
    active_source.write_bytes(active_source_bytes)

    native_bytes = (output / "p5/completion.json").read_bytes()
    (output / "p5/completion.json").unlink()
    with pytest.raises(ValueError):
        runner.run_real_evolution(output, manifest, ports)
    assert not (output / "p5/completion.json").exists()
    (output / "p5/completion.json").write_bytes(native_bytes)

    native = output / "p5/completion.json"
    tampered = json.loads(native.read_text(encoding="utf-8"))
    tampered["active_protocol_sha256"] = "9" * 64
    native.write_bytes(canonical_v2_bytes(tampered))
    with pytest.raises(ValueError):
        runner.run_real_evolution(output, manifest, ports)


def test_production_p5_port_seals_missing_second_bundle_as_incomplete(tmp_path: Path):
    from evolving_loop.v2.real import runner

    manifest = RealEvolutionManifestV2.from_payload(
        {
            "schema_version": 1,
            "profile": "real-30m",
            "model": {
                "schema_version": 1,
                "name": "gpt-5.6-luna",
                "reasoning_effort": "medium",
            },
            "files": [],
            "runtime_locations": [],
            "l0_fingerprints": {},
        }
    )
    context = RealStageContextV2(
        "p5", tmp_path / "p5", 120, manifest, manifest.fingerprint(),
        manifest.model.fingerprint(), {"p3": "3" * 64, "p4": "4" * 64},
    )
    ports = runner.build_real_stage_ports(
        SimpleNamespace(), manifest=manifest, repo_root=tmp_path
    )

    sealed = ports.seal_p5(
        context, {"schema_version": 1, "status": "p5_handoff_unavailable"}
    )

    assert sealed.summary["status"] == "incomplete"
    assert sealed.handoff_payload["reason"] == "p5_handoff_unavailable"
    assert sealed.public_test_accessed is False
    assert (context.output_dir / "root_stage_completion.json").is_file()
    assert not (context.output_dir / "completion.json").exists()
