from __future__ import annotations

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
                    ("model_cache", "model-cache"),
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


@pytest.mark.parametrize(
    ("collection", "role"),
    (("files", "split"), ("runtime_locations", "runtime")),
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
    from numerical_agent.providers import RuntimeRegistry

    manifest_payload = _real_manifest_payload()
    _materialize_real_manifest_paths(tmp_path, manifest_payload)
    source = tmp_path / "runs/method_evolution/v001"
    source.mkdir(parents=True)
    (source / "methods.py").write_text(
        'def safe_anchor(history, horizon, frequency):\n'
        '    """Return the last observed value."""\n'
        '    return [float(history[-1])] * horizon\n',
        encoding="utf-8",
    )
    (source / "skills.py").write_text("", encoding="utf-8")
    worker = tmp_path / "tmp/toto2_worker_smoke.json"
    worker.write_text("{}", encoding="utf-8")
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
    monkeypatch.setattr(
        module,
        "_load_screening_policy",
        lambda _path: SimpleNamespace(fingerprint=lambda: "e" * 64),
    )
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
