"""Fresh-interpreter Project 2 isolation and legacy byte/mtime compatibility."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run_isolated(script: str, tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_real_numerical_cli_preserves_persisted_legacy_artifacts_and_defaults(
    tmp_path,
):
    """A fresh-process Numerical run cannot rewrite any legacy authority."""

    _run_isolated(
        r'''
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

root = Path.cwd()
sys.path.insert(0, str(root / "tests"))
work = Path(sys.argv[1])

import evolving_loop.cli as legacy_cli
from evolving_loop.package_numerical_supply import (
    canonical_json_bytes,
    parse_numerical_supply_release,
)
from evolving_loop.v2.contracts import canonical_v2_bytes
from tests.test_package_coordinate_evolution import _bundle

inputs = work / "inputs"
subprocess.run(
    [
        sys.executable,
        str(root / "tests/build_evolution_v2_numerical_fixture.py"),
        "--output-dir",
        str(inputs),
    ],
    cwd=root,
    check=True,
    capture_output=True,
    text=True,
)

persisted = work / "persisted-legacy"
persisted.mkdir()
shutil.copy2(inputs / "seed_supply.json", persisted / "supply.json")
_task, state = _bundle(work / "legacy-source")

def plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: plain(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(member) for member in value]
    return value

(persisted / "registry.json").write_bytes(
    canonical_json_bytes(plain(state.registry.manifest))
)
for source, name in (
    (root / "numerical_agent/dictionaries/statistical_base_methods_v000.json", "method-catalog.json"),
    (root / "numerical_agent/tsfm/runtime_manifests.json", "runtime-manifest.json"),
):
    (persisted / name).write_bytes(
        canonical_json_bytes(json.loads(source.read_bytes()))
    )
method_source = persisted / "method-source.py"
method_source.write_bytes(
    b'def legacy_seasonal_naive(history, horizon, frequency):\n'
    b'    """Forecast the last finite observation for a bounded horizon."""\n'
    b'    del frequency\n'
    b'    return [float(history[-1])] * horizon\n'
)
release = parse_numerical_supply_release(
    plain(state.bundle.numerical_release_payload)
)
shutil.copytree(
    root / "evolving_loop/retrieval_agent/releases/v000",
    persisted / "retrieval-v000",
    copy_function=shutil.copy2,
)
assert (persisted / "supply.json").read_bytes() == canonical_v2_bytes(
    json.loads((persisted / "supply.json").read_bytes())
)
for artifact in persisted.rglob("*.json"):
    if artifact.name == "supply.json":
        continue
    assert artifact.read_bytes() == canonical_json_bytes(
        json.loads(artifact.read_bytes())
    )

source_authorities = (
    root / "evolving_loop/package_numerical_supply.py",
    root / "evolving_loop/package_registry.py",
    root / "numerical_agent/dictionaries/statistical_base_methods_v000.json",
    root / "numerical_agent/tsfm/runtime_manifests.json",
)
retrieval = root / "evolving_loop/retrieval_agent/releases/v000"

def files():
    paths = (
        *source_authorities,
        *(path for path in retrieval.rglob("*") if path.is_file()),
        *(path for path in persisted.rglob("*") if path.is_file()),
    )
    return {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in paths
    }

supply_before = canonical_json_bytes(state.bundle.numerical_release_payload)
registry_before = (state.registry.fingerprint, state.registry.release_sha256, state.registry.task_ids)
forms = (["evolve"], ["--evolution", "genome"])
defaults_before = [vars(legacy_cli.build_parser().parse_args(args)) for args in forms]
files_before = files()
assert "evolving_loop.v2.numerical_qd" not in sys.modules

from evolving_loop.v2.numerical_qd.adapters import import_numerical_seed

imported = import_numerical_seed(
    release,
    state.registry,
    tasks=(_task,),
    source_paths=(method_source,),
)
method_sha256 = hashlib.sha256(method_source.read_bytes()).hexdigest()
assert imported.sources == {method_sha256: method_source.read_text()}
namespace = {}
exec(compile(imported.sources[method_sha256], str(method_source), "exec"), namespace)
assert namespace["legacy_seasonal_naive"]((1.0, 2.0), 2, "D") == [2.0, 2.0]

run = work / "run"
command = [
    sys.executable,
    "-m",
    "evolving_loop.v2",
    "numerical-evolve",
    "--config",
    str(root / "configs/evolution_v2/numerical_qd/smoke.json"),
    "--seed-supply",
    str(persisted / "supply.json"),
    "--task-manifest",
    str(inputs / "task_manifest.json"),
    "--output-dir",
    str(run),
]
completed = subprocess.run(
    command, cwd=root, capture_output=True, text=True, timeout=120
)
assert completed.returncode == 0, completed.stdout + completed.stderr
completion = json.loads((run / "evaluation_complete.json").read_bytes())
assert completion["status"] == "numerical_qd_complete"

run_before_resume = {
    str(path.relative_to(run)): (path.read_bytes(), path.stat().st_mtime_ns)
    for path in run.rglob("*")
    if path.is_file()
}
resumed = subprocess.run(
    command, cwd=root, capture_output=True, text=True, timeout=60
)
assert resumed.returncode == 0, resumed.stdout + resumed.stderr
assert {
    str(path.relative_to(run)): (path.read_bytes(), path.stat().st_mtime_ns)
    for path in run.rglob("*")
    if path.is_file()
} == run_before_resume

import evolving_loop.v2.numerical_qd

assert canonical_json_bytes(state.bundle.numerical_release_payload) == supply_before
assert (state.registry.fingerprint, state.registry.release_sha256, state.registry.task_ids) == registry_before
assert [vars(legacy_cli.build_parser().parse_args(args)) for args in forms] == defaults_before
assert files() == files_before
''',
        tmp_path,
    )


def test_pre_project2_project1_run_resumes_after_numerical_qd_import_without_writes(
    tmp_path,
):
    """Project 2 imports must not change a completed Project 1 resume."""

    _run_isolated(
        r'''
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "project1"
from evolving_loop.v2.fakes import run_fake_kernel, smoke_config

assert "evolving_loop.v2.numerical_qd" not in sys.modules
first = run_fake_kernel(root, smoke_config())
assert first.accepted_steps == first.rejected_steps == 1

def snapshot():
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }

before = snapshot()
import evolving_loop.v2.numerical_qd
resumed = run_fake_kernel(root, smoke_config(), resume=True)
assert resumed.accepted_bundle.canonical_bytes() == first.accepted_bundle.canonical_bytes()
assert snapshot() == before
''',
        tmp_path,
    )
