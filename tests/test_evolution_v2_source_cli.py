from __future__ import annotations

import json
import subprocess
import sys


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
