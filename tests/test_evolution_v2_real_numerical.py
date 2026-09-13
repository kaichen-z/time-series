from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolving_loop.v2.bundle import EvolutionBundleV2
from evolving_loop.v2.cli import _parse_task_manifest
from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
from evolving_loop.v2.numerical_qd.persistence import (
    NumericalQDRunStore,
    NumericalQDStoreError,
)
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tests/build_evolution_v2_numerical_fixture.py"
SMOKE_CONFIG = ROOT / "configs/evolution_v2/numerical_qd/smoke.json"
REAL_NUMERICAL_SOURCE_SEED = (
    ROOT / "configs/evolution_v2/real/numerical-source-seed-v1.py"
)
AUTHORITY_ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
).resolve().parent


@pytest.fixture(scope="module")
def completed_p2(tmp_path_factory):
    from common.llm import FakeLLMClient
    from evolving_loop.package_numerical_supply import parse_numerical_supply_release
    from evolving_loop.v2.cli import numerical_evolve_payload

    root = tmp_path_factory.mktemp("real-numerical")
    fixtures = root / "fixtures"
    built = subprocess.run(
        [sys.executable, str(BUILDER), "--output-dir", str(fixtures)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert built.returncode == 0, built.stderr
    seed = fixtures / "seed_supply.json"
    task_manifest = fixtures / "task_manifest.json"
    output = root / "p2"
    config_payload = json.loads(SMOKE_CONFIG.read_bytes())
    # Admit bootstrap and finalization but no promotable generation. This is a
    # real deterministic seed-only P2 completion, not a rewritten fixture.
    config_payload["budget"]["ceilings"]["task_executions"] = 800
    seed_payload = json.loads(seed.read_bytes())
    task_payload = json.loads(task_manifest.read_bytes())
    completion = numerical_evolve_payload(
        config_payload,
        seed_payload,
        task_payload,
        output,
        input_sha256s={
            "config": fingerprint_payload(config_payload),
            "seed_supply": fingerprint_payload(seed_payload),
            "task_manifest": fingerprint_payload(task_payload),
        },
        host_runtime=object(),
        llm_client=FakeLLMClient([]),
    )
    tasks, _folds = _parse_task_manifest(task_payload)
    return output, tasks, completion, parse_numerical_supply_release(seed_payload)


def test_completed_store_loads_exact_active_frozen_pair(completed_p2):
    output, tasks, completion, seed = completed_p2
    frozen, pair_sha256 = NumericalQDRunStore(output).load_active_frozen_pair(
        tasks=tasks
    )
    active = EvolutionBundleV2.from_payload(
        json.loads((output / "accepted_bundle.json").read_bytes())
    )

    assert active.fingerprint() == completion["active_bundle_sha256"]
    assert completion["summary"]["accepted_count"] == 0
    assert frozen.release.fingerprint == seed.fingerprint
    assert frozen.release.fingerprint == active.numerical_release_sha256
    assert frozen.registry.fingerprint == active.numerical_registry_sha256
    assert frozen.envelope.release_sha256 == active.numerical_release_sha256
    assert frozen.envelope.registry_sha256 == active.numerical_registry_sha256
    assert tuple(frozen.envelope.entries) == tuple(
        task.numeric.task_id for task in tasks
    )
    assert pair_sha256 == fingerprint_payload(
        {
            "supply": frozen.release.to_payload(),
            "registry": frozen.envelope.to_payload(),
        }
    )


def test_active_pair_loader_rejects_changed_or_duplicate_pair(
    completed_p2, tmp_path
):
    output, tasks, _completion, _seed = completed_p2
    changed = tmp_path / "changed"
    shutil.copytree(output, changed)
    store = NumericalQDRunStore(changed)
    checkpoint = json.loads((changed / "numerical_qd/checkpoint.json").read_bytes())
    catalog = checkpoint["completed_operation_sha256s"]
    pair_name = next(
        name
        for name in catalog
        if name.startswith("objects/")
        and set(json.loads((changed / "numerical_qd" / name).read_bytes()))
        == {"supply", "registry"}
        and json.loads((changed / "numerical_qd" / name).read_bytes())["registry"][
            "registry_sha256"
        ]
        == json.loads((changed / "accepted_bundle.json").read_bytes())[
            "numerical_registry_sha256"
        ]
    )
    pair_path = changed / "numerical_qd" / pair_name
    pair_path.write_bytes(pair_path.read_bytes().replace(b'"supply"', b'"Supply"', 1))
    with pytest.raises(NumericalQDStoreError):
        store.load_active_frozen_pair(tasks=tasks)

    duplicate = tmp_path / "duplicate"
    shutil.copytree(output, duplicate)
    pair = json.loads((output / "numerical_qd" / pair_name).read_bytes())
    alias = pair | {"duplicate_marker": True}
    alias_sha = fingerprint_payload(alias)
    alias_path = duplicate / "numerical_qd/objects" / f"{alias_sha}.json"
    alias_path.write_bytes(canonical_v2_bytes(alias))
    duplicate_checkpoint = json.loads(
        (duplicate / "numerical_qd/checkpoint.json").read_bytes()
    )
    duplicate_checkpoint["completed_operation_sha256s"][
        f"objects/{alias_sha}.json"
    ] = fingerprint_payload(alias)
    # The runner checkpoint cannot be legitimately extended after completion;
    # the loader must reject this catalog ambiguity rather than choose a pair.
    with pytest.raises(NumericalQDStoreError):
        NumericalQDRunStore(duplicate).load_active_frozen_pair(tasks=tasks)


def test_payload_entrypoint_runs_from_root_derived_payload_without_path_overlap(
    tmp_path, monkeypatch
):
    from common.llm import FakeLLMClient
    from evolving_loop.v2 import cli

    fixtures = tmp_path / "root/inputs"
    built = subprocess.run(
        [sys.executable, str(BUILDER), "--output-dir", str(fixtures)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert built.returncode == 0, built.stderr
    config_payload = json.loads(SMOKE_CONFIG.read_bytes())
    seed_payload = json.loads((fixtures / "seed_supply.json").read_bytes())
    task_payload = json.loads((fixtures / "task_manifest.json").read_bytes())
    input_sha256s = {
        "config": "1" * 64,
        "seed_supply": "2" * 64,
        "task_manifest": "3" * 64,
    }
    observed = {}

    def fake_runner(output, config, seed, folds, adapter, **kwargs):
        observed.update(
            config=config,
            seed=seed,
            folds=folds,
            adapter=adapter,
            llm_client=kwargs["llm_client"],
            resume=kwargs["resume"],
        )
        output.mkdir(parents=True)
        (output / "evaluation_complete.json").write_bytes(
            canonical_v2_bytes({"status": "payload-complete"})
        )

    monkeypatch.setattr(cli, "run_numerical_qd", fake_runner)
    llm = FakeLLMClient([])
    output = tmp_path / "root/p2"
    result = cli.numerical_evolve_payload(
        config_payload,
        seed_payload,
        task_payload,
        output,
        input_sha256s=input_sha256s,
        host_runtime=SimpleNamespace(),
        llm_client=llm,
    )

    assert result == {"status": "payload-complete"}
    assert observed["config"].to_payload() == config_payload
    assert observed["adapter"].operator_input_sha256s == input_sha256s
    assert observed["llm_client"] is llm
    assert observed["resume"] is False


def test_real_numerical_bridge_cannot_override_host_llm(tmp_path, monkeypatch):
    from evolving_loop.v2.real import bridges

    observed = {}

    def numerical_payload(*_args, **kwargs):
        observed.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(bridges, "numerical_evolve_payload", numerical_payload)
    host_llm = object()
    arguments = {
        "context": SimpleNamespace(output_dir=tmp_path / "p2"),
        "host": SimpleNamespace(llm_client=host_llm),
        "config_payload": {},
        "seed_payload": {},
        "task_manifest_payload": {},
        "input_sha256s": {},
    }
    assert bridges.run_real_numerical(**arguments) == {"status": "complete"}
    assert observed["llm_client"] is host_llm
    with pytest.raises(TypeError):
        bridges.run_real_numerical(**arguments, llm_client=object())


def test_real_numerical_bridge_projects_prepared_provenance_at_adapter_boundary(
    completed_p2, tmp_path, monkeypatch
):
    from common.llm import FakeLLMClient
    from evolving_loop.v2 import cli
    from evolving_loop.v2.real import runner

    completed_output, _tasks, _completion, _seed = completed_p2
    fixtures = completed_output.parent / "fixtures"
    config = json.loads(SMOKE_CONFIG.read_bytes())
    seed = json.loads((fixtures / "seed_supply.json").read_bytes())
    task_manifest = json.loads((fixtures / "task_manifest.json").read_bytes())
    operator_inputs = {
        "config": "1" * 64,
        "seed_supply": "2" * 64,
        "task_manifest": "3" * 64,
    }
    prepared_provenance = operator_inputs | {
        "champion_release": "4" * 64,
        "dictionary": "5" * 64,
        "task_local_evidence": "6" * 64,
    }
    observed = {}

    def fake_runner(output, _config, _seed, _folds, adapter, **_kwargs):
        observed["operator_inputs"] = dict(adapter.operator_input_sha256s)
        output.mkdir(parents=True)
        (output / "evaluation_complete.json").write_bytes(
            canonical_v2_bytes({"status": "parsed"})
        )

    monkeypatch.setattr(cli, "run_numerical_qd", fake_runner)
    llm = FakeLLMClient([])
    manifest = RealEvolutionManifestV2.from_payload(_real_manifest_payload())
    context = runner.RealStageContextV2(
        "p2",
        tmp_path / "p2",
        840,
        manifest,
        manifest.fingerprint(),
        manifest.model.fingerprint(),
        {},
    )
    prepared = SimpleNamespace(
        config_payload=config,
        seed_payload=seed,
        task_manifest_payload=task_manifest,
        input_sha256s=prepared_provenance,
        evidence_path=None,
        dictionary=None,
    )
    monkeypatch.setattr(
        runner, "prepare_real_p2_inputs", lambda *_args, **_kwargs: prepared
    )
    ports = runner.build_real_stage_ports(
        SimpleNamespace(llm_client=llm), manifest=manifest, repo_root=tmp_path
    )
    result = ports.run_p2(context)

    assert result == {"status": "parsed"}
    assert observed["operator_inputs"] == operator_inputs


def test_loaded_prepared_payloads_cross_strict_real_numerical_parser(
    tmp_path, monkeypatch
):
    from evolving_loop.package_numerical_supply import parse_numerical_supply_release
    from evolving_loop.v2.real import bridges, runner
    from numerical_agent.evolution import task_shortlist
    from numerical_agent import run_task_local_ensemble_evolution as task_local
    from tests.test_package_numerical_supply import _supply_release

    manifest = RealEvolutionManifestV2.from_payload(_real_manifest_payload())
    champion_sha = next(
        row.sha256 for row in manifest.files if row.role == "numerical_seed"
    )
    config = {"schema_version": 1, "kind": "test_config"}
    seed_release = _supply_release()
    seed = seed_release.to_payload()
    tasks = {
        "train": [{"task_id": f"train-{index}"} for index in range(80)],
        "dev": [{"task_id": f"dev-{index}"} for index in range(20)],
    }
    dictionary = object()
    dictionary_sha = "d" * 64
    evidence_index = {"schema_version": 1, "evidence": "closed"}
    evidence_sha = fingerprint_payload(evidence_index)
    seal = {
        "schema_version": 1,
        "kind": "real_p2_prepared_inputs",
        "manifest_sha256": manifest.fingerprint(),
        "champion_sha256": champion_sha,
        "dictionary_sha256": dictionary_sha,
        "config_sha256": fingerprint_payload(config),
        "seed_supply_sha256": fingerprint_payload(seed),
        "task_manifest_sha256": fingerprint_payload(tasks),
        "task_local_evidence_sha256": evidence_sha,
    }
    payloads = {
        "prepared_inputs.json": seal,
        "numerical_config.json": config,
        "seed_supply.json": seed,
        "task_manifest.json": tasks,
        "dictionary.json": {"schema_version": 1, "entries": []},
    }
    monkeypatch.setattr(
        runner, "_read_canonical", lambda path: payloads[Path(path).name]
    )
    monkeypatch.setattr(runner, "_dictionary_from_payload", lambda _value: dictionary)
    monkeypatch.setattr(task_shortlist, "_dictionary_hash", lambda _value: dictionary_sha)
    monkeypatch.setattr(
        task_local,
        "load_task_local_evidence_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(
            index=evidence_index,
            by_task={row["task_id"]: object() for split in ("train", "dev") for row in tasks[split]},
        ),
    )
    prepared = runner._load_prepared_real_p2(tmp_path, manifest=manifest)
    assert prepared.seed_payload == seed
    assert len(prepared.seed_payload["alternatives"]) == len(seed["alternatives"])
    assert sum(len(prepared.task_manifest_payload[split]) for split in ("train", "dev")) == 100

    def strict_payload_seam(config_payload, seed_payload, task_payload, *_args, **_kwargs):
        assert type(config_payload) is dict
        assert type(seed_payload) is dict
        assert type(task_payload) is dict
        assert (
            parse_numerical_supply_release(seed_payload).fingerprint
            == seed_release.fingerprint
        )
        return {"status": "parsed"}

    monkeypatch.setattr(bridges, "numerical_evolve_payload", strict_payload_seam)
    result = bridges.run_real_numerical(
        SimpleNamespace(output_dir=tmp_path / "p2"),
        SimpleNamespace(llm_client=object()),
        config_payload=prepared.config_payload,
        seed_payload=prepared.seed_payload,
        task_manifest_payload=prepared.task_manifest_payload,
        input_sha256s=prepared.input_sha256s,
        task_local_evidence_path=prepared.evidence_path,
        task_local_dictionary=prepared.dictionary,
    )

    assert result == {"status": "parsed"}


def test_real_numerical_source_seed_is_strict_and_matches_historical_naive_last(
    tmp_path,
):
    """Catches exposing the unreviewed historical catalog as mutation code."""
    from evolving_loop.v2.numerical_qd.adapters import LegacyNumericalAdapter
    from numerical_agent.evolution.execution import IsolatedForecastRuntime
    from numerical_agent.evolution.module import MODULE_HEADER, read_module

    seed_source = REAL_NUMERICAL_SOURCE_SEED.read_text(encoding="utf-8")
    seed_module = LegacyNumericalAdapter.validate_source(seed_source)
    historical = read_module(
        AUTHORITY_ROOT / "runs/method_evolution/v001/methods.py"
    ).get("naive_last")
    assert historical is not None
    assert seed_module.names() == ("naive_last",)

    paths = []
    for name, source in (("seed", seed_source), ("historical", historical.source)):
        path = tmp_path / f"{name}.py"
        path.write_text(MODULE_HEADER + "\n" + source + "\n", encoding="utf-8")
        paths.append(path)
    runtimes = tuple(IsolatedForecastRuntime(path) for path in paths)
    try:
        for history, horizon, frequency in (
            ((1.0,), 1, "D"),
            ((-3.0, 2.5, 9.0), 4, "1 day"),
            ((0.0, -0.0, 1.25), 3, "H"),
        ):
            assert runtimes[0].forecast(
                "naive_last", history, horizon, frequency
            ) == runtimes[1].forecast("naive_last", history, horizon, frequency)
    finally:
        for runtime in runtimes:
            runtime.close()

    seed_sha = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()
    origin_sha = hashlib.sha256(historical.source.encode("utf-8")).hexdigest()
    for profile in ("real-30m", "real-1h"):
        manifest = RealEvolutionManifestV2.from_payload(
            json.loads(
                (
                    ROOT
                    / f"configs/evolution_v2/real/{profile}-toto-balanced-v3.json"
                ).read_bytes()
            )
        )
        source_row = next(
            row for row in manifest.files if row.role == "numerical_source_seed"
        )
        assert source_row.relative_path == (
            "configs/evolution_v2/real/numerical-source-seed-v1.py"
        )
        assert source_row.sha256 == seed_sha
        assert manifest.l0_fingerprints["numerical_source_seed"] == seed_sha
        assert manifest.l0_fingerprints["numerical_source_origin"] == origin_sha


def test_real_screening_keeps_task_114_timesfm_from_nonfirst_clause():
    from evolving_loop.data import load_context_tasks_by_ids
    from numerical_agent.evolution.execution import Task
    from numerical_agent.evolution.screening import (
        materialize_active_dictionary,
        profile_task,
    )
    from numerical_agent.run_champion_evolution import _load_screening_policy

    screening = _load_screening_policy(
        AUTHORITY_ROOT / "runs/method_evolution/v001/dictionary.py"
    )
    context = load_context_tasks_by_ids(
        AUTHORITY_ROOT / "external/Dr-CiK/full-download/Dr-CiK_public/tasks",
        ("task_114",),
    )[0]
    numeric = context.numeric
    profile = profile_task(
        Task(
            numeric.task_id,
            numeric.history_values,
            numeric.prediction_length,
            numeric.frequency,
            (),
        )
    )

    entry = screening.get("timesfm_2_5")
    active = materialize_active_dictionary(screening, profile)

    assert entry is not None
    assert entry.status == "specialized"
    assert "timesfm_2_5" in screening.fallback_names
    assert entry.applicability.match(profile) == 1
    assert "timesfm_2_5" in {candidate.name for candidate in active.active}


def test_real_train_fold_groups_require_exact_nested_rung_packing():
    from collections import Counter

    from evolving_loop.data import load_context_tasks_by_ids
    from evolving_loop.v2.numerical_qd.hyperband import pack_fold_groups
    from numerical_agent.evolution.task_local_evolution import (
        build_group_fold_manifest,
    )

    split = json.loads(
        (AUTHORITY_ROOT / "splits/drcik_public_80_20_99_v3.json").read_bytes()
    )
    train_ids = tuple(split["partitions"]["train"]["task_ids"])
    tasks = load_context_tasks_by_ids(
        AUTHORITY_ROOT / "external/Dr-CiK/full-download/Dr-CiK_public/tasks",
        train_ids,
    )
    folds = build_group_fold_manifest(
        tuple(task.numeric for task in tasks), seed=20260903
    )
    sizes = tuple(len(task_ids) for _sha, task_ids, _fold in folds.groups)
    packed = pack_fold_groups(folds.groups)

    assert len(sizes) == 28
    assert Counter(sizes) == Counter(
        {1: 14, 2: 8, 3: 1, 4: 1, 5: 1, 9: 1, 14: 1, 15: 1}
    )
    assert {8, 32}.isdisjoint(__import__("itertools").accumulate(sizes))
    assert tuple(
        sum(len(task_ids) for _entity, task_ids in bin_) for bin_ in packed
    ) == (8, 24, 48)


def _real_manifest_payload():
    return {
        "schema_version": 1,
        "profile": "real-30m",
        "model": {
            "schema_version": 1,
            "name": "gpt-5.6-luna",
            "reasoning_effort": "medium",
        },
        "files": [
            {"role": role, "relative_path": path, "sha256": str(index) * 64}
            for index, (role, path) in enumerate(
                (
                    ("split", "split.json"),
                    ("tasks", "tasks"),
                    ("numerical_seed", "champion.json"),
                    (
                        "numerical_source_seed",
                        "configs/evolution_v2/real/numerical-source-seed-v1.py",
                    ),
                    ("forecast_cache", "forecast-cache"),
                    ("retrieval_seed", "retrieval-seed"),
                    ("source_seed", "source-seed.json"),
                ),
                start=1,
            )
        ],
        "runtime_locations": [
            {
                "role": role,
                "relative_path": path,
                "identity_sha256": chr(ord("a") + index) * 64,
            }
            for index, (role, path) in enumerate(
                (
                    ("python", "python"),
                    ("runtime", "tmp/toto2_worker_smoke.json"),
                    ("task_loader", "task-loader.py"),
                    ("forecast_store", "forecast-store.py"),
                    ("model_cache", "model-cache/hub"),
                    ("codex_cli", "codex"),
                )
            )
        ],
        "l0_fingerprints": {"metric": "f" * 64},
    }


def _materialize_real_manifest_paths(root, payload):
    directories = {
        ("files", "tasks"),
        ("files", "forecast_cache"),
        ("files", "retrieval_seed"),
        ("runtime_locations", "model_cache"),
    }
    for collection in ("files", "runtime_locations"):
        for row in payload[collection]:
            path = root / row["relative_path"]
            if (collection, row["role"]) in directories:
                path.mkdir(parents=True, exist_ok=True)
                (path / "identity.txt").write_text(row["role"], encoding="utf-8")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                if row["role"] == "numerical_source_seed":
                    path.write_text(
                        REAL_NUMERICAL_SOURCE_SEED.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                else:
                    path.write_text(row["role"], encoding="utf-8")


def _bind_real_manifest_identities(root, payload):
    from evolving_loop.v2.real.host import (
        resolve_real_input_identity,
        resolve_real_runtime_identity,
    )

    for row in payload["files"]:
        row["sha256"] = resolve_real_input_identity(
            row["role"], root / row["relative_path"]
        )
    for row in payload["runtime_locations"]:
        row["identity_sha256"] = resolve_real_runtime_identity(
            row["role"], root / row["relative_path"]
        )
    return RealEvolutionManifestV2.from_payload(payload)


@pytest.mark.parametrize("digest_kind", ("git-sha1", "lfs-sha256"))
def test_model_cache_accepts_verified_hugging_face_cas_blobs(
    tmp_path, digest_kind
):
    from evolving_loop.v2.real.host import resolve_real_runtime_identity

    content = f"verified-{digest_kind}".encode("utf-8")
    if digest_kind == "git-sha1":
        framed = f"blob {len(content)}\0".encode("ascii") + content
        digest = hashlib.sha1(framed).hexdigest()
    else:
        digest = hashlib.sha256(content).hexdigest()
    blob = tmp_path / "models--owner--model/blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)

    identity = resolve_real_runtime_identity("model_cache", tmp_path)

    assert len(identity) == 64


def test_model_cache_rejects_same_size_cas_replacement_or_malformed_name(tmp_path):
    from evolving_loop.v2.real.host import resolve_real_runtime_identity

    original = b"original"
    digest = hashlib.sha256(original).hexdigest()
    blob = tmp_path / "models--owner--model/blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(original)
    assert len(resolve_real_runtime_identity("model_cache", tmp_path)) == 64

    blob.write_bytes(b"mutated!")
    with pytest.raises(ValueError, match="CAS blob content digest mismatch"):
        resolve_real_runtime_identity("model_cache", tmp_path)

    blob.unlink()
    (blob.parent / "not-a-cas-digest").write_bytes(b"metadata")
    with pytest.raises(ValueError, match="malformed CAS blob name"):
        resolve_real_runtime_identity("model_cache", tmp_path)


@pytest.mark.parametrize(
    ("collection", "role"),
    (
        ("files", "split"),
        ("files", "numerical_source_seed"),
        ("runtime_locations", "runtime"),
    ),
)
def test_real_host_rejects_file_and_runtime_identity_drift(
    tmp_path, monkeypatch, collection, role
):
    from evolving_loop.v2.real import host as module

    payload = _real_manifest_payload()
    _materialize_real_manifest_paths(tmp_path, payload)
    manifest = _bind_real_manifest_identities(tmp_path, payload)
    row = next(item for item in payload[collection] if item["role"] == role)
    (tmp_path / row["relative_path"]).write_text("drift", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_validated_split",
        lambda *_args, **_kwargs: pytest.fail("drift admitted before loading"),
    )

    with pytest.raises(ValueError, match=role):
        module.build_real_host(
            manifest, repo_root=tmp_path, output_dir=tmp_path / "output"
        )


def test_real_host_owns_exact_tasks_cache_agents_and_resource_cleanup(
    tmp_path, monkeypatch
):
    from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
    from evolving_loop.v2.real import host as module
    from numerical_agent.evolution.portfolio import PolicyPortfolio
    from numerical_agent.evolution.screening import (
        ApplicabilityPolicy,
        ScreeningEntry,
        ScreeningPolicy,
    )
    from numerical_agent.providers import RuntimeRegistry

    manifest_payload = _real_manifest_payload()
    _materialize_real_manifest_paths(tmp_path, manifest_payload)
    source = tmp_path / "runs/method_evolution/v001"
    source.mkdir(parents=True)
    historical_source = (
        "def naive_last(history, horizon, frequency):\n"
        '    """Repeat the last observed value."""\n'
        "    if len(history) < 1:\n"
        '        raise NotApplicable(f"needs 1 point, got {len(history)}")\n'
        "    return [float(history[-1])] * horizon\n"
    )
    (source / "methods.py").write_text(historical_source, encoding="utf-8")
    (source / "skills.py").write_text("", encoding="utf-8")
    worker = tmp_path / "tmp/toto2_worker_smoke.json"
    worker.write_text("{}", encoding="utf-8")
    numerical_source_seed = tmp_path / (
        "configs/evolution_v2/real/numerical-source-seed-v1.py"
    )
    numerical_source_seed_sha256 = hashlib.sha256(
        numerical_source_seed.read_bytes()
    ).hexdigest()
    numerical_source_origin_sha256 = hashlib.sha256(
        historical_source.strip().encode("utf-8")
    ).hexdigest()
    manifest_payload["l0_fingerprints"].update(
        {
            "numerical_source_seed": numerical_source_seed_sha256,
            "numerical_source_origin": numerical_source_origin_sha256,
        }
    )
    manifest = _bind_real_manifest_identities(tmp_path, manifest_payload)

    requested = {}
    fixture_payload = json.loads(
        subprocess.run(
            [sys.executable, str(BUILDER), "--output-dir", str(tmp_path / "fixture")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        or (tmp_path / "fixture/task_manifest.json").read_text()
    )
    tasks, _folds = _parse_task_manifest(fixture_payload)
    train_ids = tuple(task.numeric.task_id for task in tasks[:80])
    dev_ids = tuple(task.numeric.task_id for task in tasks[80:])

    monkeypatch.setattr(
        module,
        "_validated_split",
        lambda *_args, **_kwargs: ({}, train_ids, dev_ids),
    )

    def load_tasks(path, task_ids):
        requested["path"] = Path(path)
        requested["ids"] = tuple(task_ids)
        return tasks

    monkeypatch.setattr(module, "load_context_tasks_by_ids", load_tasks)
    monkeypatch.setattr(
        module, "read_policy_file", lambda _path: PolicyPortfolio.flagship5()
    )
    screening = ScreeningPolicy(
        (
            ScreeningEntry(
                "naive_last",
                "statistical",
                "keep",
                ApplicabilityPolicy(),
                "reviewed",
            ),
        ),
        ("naive_last",),
    )
    monkeypatch.setattr(module, "_load_screening_policy", lambda _path: screening)
    runtimes = RuntimeRegistry()
    closed = {"runtime": 0}
    original_close = runtimes.close

    def close_runtimes():
        closed["runtime"] += 1
        original_close()

    monkeypatch.setattr(runtimes, "close", close_runtimes)
    def runtime_registry(args):
        requested["runtime_args"] = args
        return runtimes

    monkeypatch.setattr(module, "_runtime_registry", runtime_registry)
    class FakeForecastStore:
        def __init__(self, root, *_args, cache_only, **_kwargs):
            self.root = Path(root)
            self.cache_only = cache_only
            self.identity_hash = module.EXPECTED_REAL_FORECAST_STORE_IDENTITY
            self._statistical = SimpleNamespace(connection="unopened")

        def close(self):
            self._statistical.connection = None

    monkeypatch.setattr(module, "ForecastStore", FakeForecastStore)
    library = RetrievalSkillLibrary(tmp_path / "skills.json", persist=False).clone(
        read_only=True
    )
    monkeypatch.setattr(
        module.RetrievalSkillLibrary,
        "from_release",
        classmethod(lambda _cls, _path: library),
    )

    host = module.build_real_host(
        manifest, repo_root=tmp_path, output_dir=tmp_path / "output"
    )
    assert requested["path"] == tmp_path / "tasks"
    assert requested["ids"] == (*train_ids, *dev_ids)
    assert requested["runtime_args"].tsfm_runtimes == "chronos,timesfm"
    assert requested["runtime_args"].model_cache_dir == tmp_path / "model-cache"
    assert requested["runtime_args"].tsfm_workers_config == worker
    assert host.tasks == tasks
    assert host.train_tasks == tasks[:80]
    assert host.dev_tasks == tasks[80:]
    assert host.forecast_store.root == tmp_path / "forecast-cache"
    assert host.forecast_store.cache_only is True
    assert (
        host.forecast_store.identity_hash
        == module.EXPECTED_REAL_FORECAST_STORE_IDENTITY
    )
    assert host.llm_client.config.model == "gpt-5.6-luna"
    assert host.llm_client.config.reasoning_effort == "medium"
    assert host.llm_client.config.binary == str(tmp_path / "codex")
    assert getattr(host.retrieval_skill_library, "_read_only", False) is True
    assert callable(host.retrieval_factory)
    assert callable(host.decision_factory)
    assert host.screening_policy is screening
    assert dict(host.sources) == {
        numerical_source_seed_sha256: numerical_source_seed.read_text(
            encoding="utf-8"
        )
    }
    assert numerical_source_origin_sha256 not in host.sources
    assert host.resource_reporter_sha256 == fingerprint_payload(
        {
            "schema_version": 1,
            "kind": "cache_only_real_host_resource_reporter",
            "forecast_store_identity": module.EXPECTED_REAL_FORECAST_STORE_IDENTITY,
            "numerical_source_seed_sha256": numerical_source_seed_sha256,
            "numerical_source_origin_sha256": numerical_source_origin_sha256,
            "runtime_identity": {
                row.role: row.identity_sha256 for row in manifest.runtime_locations
            },
            "resource_kinds": [
                "input_tokens",
                "llm_calls",
                "output_tokens",
                "subprocesses",
            ],
            "llm_accounting": "codex_utf8_bytes_v1",
        }
    )
    assert host.resource_reporter().to_payload() == {
        "wall_seconds": 0.0,
        "task_executions": 0,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "gpu_seconds": 0.0,
        "subprocesses": 0,
        "artifact_bytes": 0,
    }
    forecast_worker = host.forecast_store._statistical
    host.close()
    assert forecast_worker.connection is None
    assert closed["runtime"] == 1
    host.close()
    assert closed["runtime"] == 1


def test_real_host_rejects_numerical_source_origin_drift_before_task_loading(
    tmp_path, monkeypatch
):
    from evolving_loop.v2.real import host as module

    payload = _real_manifest_payload()
    _materialize_real_manifest_paths(tmp_path, payload)
    source_repo = tmp_path / "runs/method_evolution/v001"
    source_repo.mkdir(parents=True)
    (source_repo / "methods.py").write_text(
        "def naive_last(history, horizon, frequency):\n"
        '    """Repeat the last observation."""\n'
        "    return [float(history[-1])] * horizon\n",
        encoding="utf-8",
    )
    seed_path = tmp_path / "configs/evolution_v2/real/numerical-source-seed-v1.py"
    payload["l0_fingerprints"].update(
        {
            "numerical_source_seed": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
            "numerical_source_origin": "0" * 64,
        }
    )
    manifest = _bind_real_manifest_identities(tmp_path, payload)
    monkeypatch.setattr(
        module,
        "_validated_split",
        lambda *_args, **_kwargs: pytest.fail(
            "source provenance drift admitted before task loading"
        ),
    )

    with pytest.raises(ValueError, match="source origin L0 identity mismatch"):
        module.build_real_host(
            manifest, repo_root=tmp_path, output_dir=tmp_path / "output"
        )
