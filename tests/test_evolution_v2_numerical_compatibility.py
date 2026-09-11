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
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_numerical_qd_import_and_proposal_preserve_legacy_artifacts_and_defaults(
    tmp_path,
):
    """Import side effects or writes to legacy authorities fail byte-for-byte."""

    _run_isolated(
        r'''
import json
from pathlib import Path
import sys

root = Path.cwd()
sys.path.insert(0, str(root / "tests"))
work = Path(sys.argv[1])

import evolving_loop.cli as legacy_cli
from evolving_loop.package_numerical_supply import canonical_json_bytes
from tests.test_package_coordinate_evolution import _bundle

legacy_paths = (
    root / "evolving_loop/package_numerical_supply.py",
    root / "evolving_loop/package_registry.py",
    root / "numerical_agent/dictionaries/statistical_base_methods_v000.json",
    root / "numerical_agent/tsfm/runtime_manifests.json",
)
retrieval = root / "evolving_loop/retrieval_agent/releases/v000"

def files():
    paths = (*legacy_paths, *(path for path in retrieval.rglob("*") if path.is_file()))
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in paths
    }

task, state = _bundle(work / "legacy")
supply_before = canonical_json_bytes(state.bundle.numerical_release_payload)
registry_before = (state.registry.fingerprint, state.registry.release_sha256, state.registry.task_ids)
forms = (["evolve"], ["--evolution", "genome"])
defaults_before = [vars(legacy_cli.build_parser().parse_args(args)) for args in forms]
files_before = files()
assert "evolving_loop.v2.numerical_qd" not in sys.modules

from evolving_loop.v2.numerical_qd.proposers import DeterministicProposalProvider
from tests.test_evolution_v2_numerical_proposers import request

batch = DeterministicProposalProvider(monotonic=lambda: 0.0).propose(request())
assert batch.proposals and batch.resource_use.llm_calls == 0
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
