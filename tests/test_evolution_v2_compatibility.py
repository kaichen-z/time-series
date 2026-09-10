"""Fresh-interpreter legacy import gates and frozen artifact non-rewrite checks."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_isolated(script, tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_v2_import_and_smoke_preserve_legacy_cli_defaults(tmp_path):
    run_isolated(
        """
import sys
from pathlib import Path
from evolving_loop.cli import build_parser

forms = (["evolve"], ["--evolution", "genome"])
before = [vars(build_parser().parse_args(args)) for args in forms]
assert before[0]["evolution_mode"] == "genome"
assert before[0]["meta_harness_v2"] is False
assert before[0]["evolve_target"] == "auto"
assert "evolving_loop.v2" not in sys.modules
import evolving_loop.v2
from evolving_loop.v2.fakes import run_fake_kernel, smoke_config
assert [vars(build_parser().parse_args(args)) for args in forms] == before
result = run_fake_kernel(Path(sys.argv[1]) / "v2", smoke_config())
assert result.accepted_steps == result.rejected_steps == 1
assert [vars(build_parser().parse_args(args)) for args in forms] == before
""",
        tmp_path,
    )


def test_legacy_frozen_package_parse_identity_survives_v2_import_and_run(tmp_path):
    run_isolated(
        """
import json
import sys
from pathlib import Path
from evolving_loop.evaluate_frozen_package_bundle import _bundle_from_payload
from tests.test_package_coordinate_evolution import _bundle

root = Path(sys.argv[1])
_, state = _bundle(root / "legacy")
payload = json.loads(state.bundle.canonical_bytes())
before = _bundle_from_payload(payload).canonical_bytes()
assert before == state.bundle.canonical_bytes()
release_before = state.registry.release_sha256
assert "evolving_loop.v2" not in sys.modules
import evolving_loop.v2
from evolving_loop.v2.fakes import run_fake_kernel, smoke_config
assert _bundle_from_payload(payload).canonical_bytes() == before
run_fake_kernel(root / "v2", smoke_config())
assert _bundle_from_payload(payload).canonical_bytes() == before
assert state.registry.release_sha256 == release_before
""",
        tmp_path,
    )


def test_v2_smoke_and_resume_do_not_rewrite_frozen_legacy_release(tmp_path):
    from evolving_loop.v2.fakes import run_fake_kernel, smoke_config

    release = ROOT / "evolving_loop" / "retrieval_agent" / "releases" / "v000"

    def frozen_files():
        return {
            path.relative_to(release): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in release.rglob("*")
            if path.is_file()
        }

    before = frozen_files()
    assert Path("manifest.json") in before
    assert Path("genome.json") in before
    result = run_fake_kernel(tmp_path / "v2", smoke_config())
    assert result.accepted_steps == result.rejected_steps == 1
    assert frozen_files() == before
    resumed = run_fake_kernel(tmp_path / "v2", smoke_config(), resume=True)
    assert (
        resumed.accepted_bundle.canonical_bytes()
        == result.accepted_bundle.canonical_bytes()
    )
    assert frozen_files() == before
