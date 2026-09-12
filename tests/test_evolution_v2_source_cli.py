from __future__ import annotations

import hashlib
import json
import subprocess
import sys

from evolving_loop.v2.contracts import canonical_v2_bytes
from tests.build_evolution_v2_source_fixture import export_source_manifest, load_source_manifest


def test_cli_smoke_and_completed_resume(tmp_path):
    base = [sys.executable, "-m", "evolving_loop.v2.source"]
    subprocess.run(base + ["make-smoke-inputs", "--output-dir", str(tmp_path / "in")], check=True, capture_output=True)
    command = base + ["evolve", "--config", "configs/evolution_v2/source/smoke.json",
                      "--input-manifest", str(tmp_path / "in/manifest.json"), "--output-dir", str(tmp_path / "run")]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    result_file = tmp_path / "run" / "evaluation_complete.json"
    before = result_file.read_bytes()
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    assert result_file.read_bytes() == before
    result = json.loads(before)
    assert result["public_test_accessed"] is False
    assert result["status"] == "source_evolution_complete"
    assert result["activated"] >= 1


def test_manifest_task_bytes_are_deserialized_into_case(tmp_path):
    manifest_path = export_source_manifest(tmp_path / "in")
    tasks_path = tmp_path / "in" / "tasks.json"
    tasks = json.loads(tasks_path.read_bytes())
    tasks["train"][0]["history_values"][0] = 101.0
    tasks_bytes = canonical_v2_bytes(tasks)
    tasks_path.write_bytes(tasks_bytes)
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["tasks.json"]["sha256"] = hashlib.sha256(tasks_bytes).hexdigest()
    manifest_path.write_bytes(canonical_v2_bytes(manifest))

    case = load_source_manifest(manifest_path)

    assert case.train_tasks[0].numeric.history_values[0] == 101.0
