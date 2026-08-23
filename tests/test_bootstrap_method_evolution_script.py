from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/bootstrap_method_evolution.sh"


def run(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "ME_DRY_RUN": "1",
        "ME_REPO": str(tmp_path / "v001"),
        "PYTHON": "python3",
        **overrides,
    }
    return subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_bootstrap_runner_defaults_to_all_statistical_methods_with_codex(tmp_path: Path) -> None:
    completed = run(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "-m numerical_agent.bootstrap_evolution" in completed.stdout
    assert "--family statistical" in completed.stdout
    assert "--llm-backend codex" in completed.stdout
    assert "--codex-model gpt-5.6-sol" in completed.stdout
    assert "--codex-reasoning-effort high" in completed.stdout
    assert "--attempts-per-method 2" in completed.stdout
    assert "methods selected: 93" in completed.stdout


def test_bootstrap_runner_refuses_to_overwrite_an_existing_method_module(tmp_path: Path) -> None:
    repo = tmp_path / "v001"
    repo.mkdir(parents=True)
    (repo / "methods.py").write_text("existing", encoding="utf-8")

    completed = run(tmp_path)

    assert completed.returncode != 0
    assert "already contains methods.py" in completed.stderr
