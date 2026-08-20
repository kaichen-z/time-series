from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from numerical_agent.tsfm.broker import WorkerCommand
from numerical_agent.tsfm.manifests import ManifestRegistry
from numerical_agent.tsfm.protocol import WorkerResponse


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts/setup_tsfm_environment.sh"
TARGET_CONTRACT = ROOT / "scripts/tsfm_target_contract.py"
LOCK_VALIDATOR = ROOT / "scripts/validate_tsfm_locks.py"
ENVIRONMENT_DIR = ROOT / "configs/tsfm-environments"

EXPECTED_ENVIRONMENTS = {
    "timesfm_v1",
    "uni2ts",
    "lag_llama",
    "granite_tsfm",
    "timer_legacy",
    "transformers_recent",
    "tempo_legacy",
    "toto2",
    "kairos",
    "tirex",
    "tabpfn_ts",
}

EXPECTED_PYTHON_TARGETS = {
    "timesfm_v1": "3.11",
    "uni2ts": "3.11",
    "lag_llama": "3.10",
    "granite_tsfm": "3.11",
    "timer_legacy": "3.10",
    "transformers_recent": "3.11",
    "tempo_legacy": "3.10",
    "toto2": "3.12",
    "kairos": "3.11",
    "tirex": "3.11",
    "tabpfn_ts": "3.11",
}


def _run_lock_validator(
    *,
    directory: Path = ENVIRONMENT_DIR,
    environment: str | None = None,
    emit_build_lock: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(LOCK_VALIDATOR),
        "--directory",
        str(directory),
    ]
    if environment is not None:
        command.extend(("--environment", environment))
    if emit_build_lock:
        command.append("--emit-build-lock")
    return subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )


def _copy_environment_specs(tmp_path: Path) -> Path:
    destination = tmp_path / "tsfm-environments"
    shutil.copytree(ENVIRONMENT_DIR, destination)
    return destination


def _run_setup(
    environment: str,
    target: Path,
    *,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SETUP), "--dry-run", environment, str(target)],
        cwd=ROOT,
        env={**os.environ, **(extra_environment or {})},
        capture_output=True,
        text=True,
        check=False,
    )


def _fake_linux_setup_environment(
    tmp_path: Path,
    *,
    final_target: Path | None = None,
    replace_target_after_stage: int | None = None,
    fail_stage: int | None = None,
) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    pip_marker = tmp_path / "pip-stage-ran"
    stage_trace = tmp_path / "setup-stage-trace"
    target = shlex.quote(str(final_target)) if final_target is not None else "''"
    fail_venv = "  exit 9\n" if fail_stage == 0 else ""
    fake_python = fake_bin / "python3.11"
    fake_python.write_text(
        f"""#!/usr/bin/env bash
set -eu
if [[ "${{1:-}}" == "-I" && "${{2:-}}" == "-c" ]]; then
  echo "cpython:3.11"
  exit 0
fi
if [[ "${{1:-}}" == "-m" && "${{2:-}}" == "venv" ]]; then
  printf 'venv\\t%s\\t%s\\n' "$PWD" "$3" >> {shlex.quote(str(stage_trace))}
  mkdir -p -- "$3/bin"
  cp "$0" "$3/bin/python"
  chmod +x "$3/bin/python"
  printf 'home = %s\\n' "$PWD" > "$3/pyvenv.cfg"
{fail_venv}
  exit 0
fi
if [[ "${{1:-}}" == "-m" && "${{2:-}}" == "pip" ]]; then
  printf 'ran\\n' >> {shlex.quote(str(pip_marker))}
  printf 'pip\\t%s\\n' "$PWD" >> {shlex.quote(str(stage_trace))}
  printf '#!%s/bin/python\\n' "$PWD" > "$PWD/bin/generated-tool"
  chmod +x "$PWD/bin/generated-tool"
  stage=$(wc -l < {shlex.quote(str(pip_marker))} | tr -d ' ')
  if [[ "$stage" -eq {fail_stage if fail_stage is not None else -1} ]]; then
    exit 9
  fi
  exit 0
fi
exit 99
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    replacement_attack = ""
    if replace_target_after_stage is not None:
        replacement_attack = f"""
pip_stage_count=0
replace_final_target() {{
  command mv -- {target} {target}.retained-original
  command mkdir -m 700 -- {target}
  printf replacement > {target}/keep.txt
}}
env() {{
  if [[ " $* " == *" PIP_DISABLE_PIP_VERSION_CHECK=1 "* ]]; then
    pip_stage_count=$((pip_stage_count + 1))
    command env "$@"
    result=$?
    if [[ "$pip_stage_count" -eq {replace_target_after_stage} ]]; then
      replace_final_target
    fi
    return "$result"
  fi
  command env "$@"
}}
"""
    bash_environment = tmp_path / "bash-environment"
    bash_environment.write_text(
        f"""function /usr/bin/uname() {{
  case "${{1:-}}" in
    -s) printf 'Linux\\n' ;;
    -m) printf 'x86_64\\n' ;;
    *) return 2 ;;
  esac
}}
{replacement_attack}
""",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "BASH_ENV": str(bash_environment),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    return environment, pip_marker


def test_environment_specs_cover_exactly_the_reviewed_worker_environments() -> None:
    assert {path.stem for path in ENVIRONMENT_DIR.glob("*.txt")} == EXPECTED_ENVIRONMENTS
    assert {path.stem for path in ENVIRONMENT_DIR.glob("*.in")} == EXPECTED_ENVIRONMENTS
    assert {path.stem for path in ENVIRONMENT_DIR.glob("*.vcs")} == {
        "kairos",
        "lag_llama",
    }


def _load_target_contract_module():
    spec = importlib.util.spec_from_file_location(
        "target_contract_test", TARGET_CONTRACT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_contract_rejects_an_unsafe_user_owned_ancestor(
    tmp_path: Path,
) -> None:
    contract = _load_target_contract_module()
    ancestor = tmp_path / "ancestor"
    parent = ancestor / "parent"
    ancestor.mkdir(mode=0o700)
    parent.mkdir(mode=0o700)
    ancestor.chmod(0o770)

    with pytest.raises(ValueError, match="trusted parent"):
        contract.prepare_target(parent / "environment")

    assert not (parent / "environment").exists()


def test_target_contract_rejects_an_unowned_parent_before_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _load_target_contract_module()
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o700)
    other_uid = os.geteuid() + 1
    monkeypatch.setattr(contract.os, "geteuid", lambda: other_uid)

    with pytest.raises(ValueError, match="trusted parent"):
        contract.prepare_target(parent / "environment")

    assert not (parent / "environment").exists()


def test_target_contract_rejects_an_unrelated_uid_system_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_target_contract_module()
    effective_uid = 501
    parent = Path("/system-boundary/user-tree/parent")
    metadata = {
        parent: SimpleNamespace(st_uid=effective_uid, st_mode=stat.S_IFDIR | 0o700),
        parent.parent: SimpleNamespace(
            st_uid=effective_uid,
            st_mode=stat.S_IFDIR | 0o700,
        ),
        parent.parent.parent: SimpleNamespace(
            st_uid=502,
            st_mode=stat.S_IFDIR | 0o755,
        ),
    }
    monkeypatch.setattr(contract.os, "geteuid", lambda: effective_uid)
    monkeypatch.setattr(
        contract,
        "_directory_metadata",
        lambda path, _error_type: metadata[path],
    )

    with pytest.raises(ValueError, match="trusted parent"):
        contract._validate_parent_chain(parent)


def test_target_contract_accepts_a_root_owned_system_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_target_contract_module()
    effective_uid = 501
    parent = Path("/system-boundary/user-tree/parent")
    metadata = {
        parent: SimpleNamespace(st_uid=effective_uid, st_mode=stat.S_IFDIR | 0o700),
        parent.parent: SimpleNamespace(
            st_uid=effective_uid,
            st_mode=stat.S_IFDIR | 0o700,
        ),
        parent.parent.parent: SimpleNamespace(
            st_uid=0,
            st_mode=stat.S_IFDIR | 0o755,
        ),
    }
    monkeypatch.setattr(contract.os, "geteuid", lambda: effective_uid)
    monkeypatch.setattr(
        contract,
        "_directory_metadata",
        lambda path, _error_type: metadata[path],
    )

    contract._validate_parent_chain(parent)


def test_offline_validator_accepts_all_reviewed_locks() -> None:
    completed = _run_lock_validator()

    assert completed.returncode == 0, completed.stderr
    assert "validated 11 TSFM environment locks" in completed.stdout


def test_offline_validator_rejects_a_requirement_without_a_hash(tmp_path: Path) -> None:
    directory = _copy_environment_specs(tmp_path)
    lock = directory / "timesfm_v1.txt"
    lock.write_text(
        re.sub(
            (
                r"(?m)^(?P<requirement>[A-Za-z0-9_.-]+(?:\[[^]]+\])?==[^\s]+)"
                r"(?: \\\n    --hash=sha256:[0-9a-f]{64})+"
            ),
            r"\g<requirement>",
            lock.read_text(encoding="utf-8"),
            count=1,
        ),
        encoding="utf-8",
    )

    completed = _run_lock_validator(directory=directory, environment="timesfm_v1")

    assert completed.returncode != 0
    assert "missing SHA-256 hash" in completed.stderr


def test_offline_validator_rejects_floating_transitive_dependency(tmp_path: Path) -> None:
    directory = _copy_environment_specs(tmp_path)
    lock = directory / "uni2ts.txt"
    floating_requirement = (
        "\nurllib3>=2.0 \\\n    --hash=sha256:" + "a" * 64 + "\n"
    )
    lock.write_text(
        lock.read_text(encoding="utf-8") + floating_requirement,
        encoding="utf-8",
    )

    completed = _run_lock_validator(directory=directory, environment="uni2ts")

    assert completed.returncode != 0
    assert "must use an exact == pin" in completed.stderr


def test_offline_validator_rejects_floating_direct_input(tmp_path: Path) -> None:
    directory = _copy_environment_specs(tmp_path)
    direct_input = directory / "granite_tsfm.in"
    direct_input.write_text(
        direct_input.read_text(encoding="utf-8").replace(
            "transformers==4.57.6", "transformers>=4.57"
        ),
        encoding="utf-8",
    )

    completed = _run_lock_validator(directory=directory, environment="granite_tsfm")

    assert completed.returncode != 0
    assert "direct input must use an exact == pin" in completed.stderr


def test_offline_validator_requires_vcs_build_backend_in_hashed_closure(
    tmp_path: Path,
) -> None:
    directory = _copy_environment_specs(tmp_path)
    direct_input = directory / "kairos.in"
    direct_input.write_text(
        direct_input.read_text(encoding="utf-8").replace("hatchling==1.27.0\n", ""),
        encoding="utf-8",
    )

    completed = _run_lock_validator(directory=directory, environment="kairos")

    assert completed.returncode != 0
    assert "required VCS build backend hatchling==1.27.0" in completed.stderr


def test_offline_validator_emits_only_hashed_build_tools() -> None:
    completed = _run_lock_validator(
        environment="uni2ts",
        emit_build_lock=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "setuptools==80.9.0" in completed.stdout
    assert "wheel==0.45.1" in completed.stdout
    assert "--hash=sha256:" in completed.stdout
    assert "torch==" not in completed.stdout
    assert "validated" not in completed.stdout


@pytest.mark.parametrize(
    ("metadata", "replacement", "diagnostic"),
    [
        ("# python-version: 3.11", "# python-version: 3.12", "Python metadata"),
        ("# platform: linux", "# platform: darwin", "platform metadata"),
        ("# architecture: x86_64", "# architecture: arm64", "architecture metadata"),
    ],
)
def test_offline_validator_rejects_wrong_target_metadata(
    tmp_path: Path,
    metadata: str,
    replacement: str,
    diagnostic: str,
) -> None:
    directory = _copy_environment_specs(tmp_path)
    lock = directory / "timesfm_v1.txt"
    lock.write_text(
        lock.read_text(encoding="utf-8").replace(metadata, replacement, 1),
        encoding="utf-8",
    )

    completed = _run_lock_validator(directory=directory, environment="timesfm_v1")

    assert completed.returncode != 0
    assert diagnostic in completed.stderr


def test_tempo_lock_requires_numpy_1_x_in_input_and_hashed_closure() -> None:
    direct_input = (ENVIRONMENT_DIR / "tempo_legacy.in").read_text(encoding="utf-8")
    lock = (ENVIRONMENT_DIR / "tempo_legacy.txt").read_text(encoding="utf-8")

    assert re.search(r"(?m)^numpy==1\.26\.4$", direct_input)
    assert re.search(
        r"(?m)^numpy==1\.26\.4 \\\n    --hash=sha256:[0-9a-f]{64}",
        lock,
    )


def test_environment_specs_preserve_audited_incompatible_stacks() -> None:
    specs = {
        path.stem: path.read_text(encoding="utf-8")
        for path in ENVIRONMENT_DIR.glob("*.in")
    }

    assert "timesfm[torch]==1.3.0" in specs["timesfm_v1"]
    assert "torch==2.4.1" in specs["uni2ts"]
    assert "gluonts==0.14.4" in specs["lag_llama"]
    assert "torch==2.10.0" in specs["granite_tsfm"]
    assert "torch==2.0.1" in specs["timer_legacy"]
    assert "transformers==4.57.6" in specs["transformers_recent"]
    assert "torch==1.13.0" in specs["tempo_legacy"]
    assert "toto-2==2.0.0" in specs["toto2"]
    kairos_vcs = (ENVIRONMENT_DIR / "kairos.vcs").read_text(encoding="utf-8")
    assert "kairos @ git+https://github.com/foundation-model-research/Kairos.git@" in kairos_vcs
    assert "tirex-ts==1.4.2" in specs["tirex"]
    assert "tabpfn-time-series==1.2.0" in specs["tabpfn_ts"]


def test_setup_rejects_unknown_environment_before_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "must-not-exist"

    completed = _run_setup("not_reviewed", target)

    assert completed.returncode != 0
    assert "allowlisted" in completed.stderr
    assert not target.exists()


def test_setup_refuses_nonempty_and_symlink_targets_without_mutation(tmp_path: Path) -> None:
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    marker = nonempty / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    linked_target = tmp_path / "linked"
    linked_target.symlink_to(nonempty, target_is_directory=True)

    nonempty_result = _run_setup("timesfm_v1", nonempty)
    symlink_result = _run_setup("timesfm_v1", linked_target)

    assert nonempty_result.returncode != 0
    assert nonempty_result.stderr == (
        "error: target must be an empty trusted directory\n"
    )
    assert symlink_result.returncode != 0
    assert "symbolic link" in symlink_result.stderr
    assert marker.read_text(encoding="utf-8") == "keep"


def test_setup_accepts_a_preexisting_empty_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "preexisting-empty"
    target.mkdir()
    environment, pip_marker = _fake_linux_setup_environment(
        tmp_path,
        final_target=target,
    )

    completed = subprocess.run(
        ["bash", str(SETUP), "timesfm_v1", str(target)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert pip_marker.read_text(encoding="utf-8").splitlines() == ["ran"] * 3
    assert (target / "bin" / "python").is_file()
    assert str(target) not in completed.stdout + completed.stderr


def test_setup_ignores_forged_guard_continuation_environment(
    tmp_path: Path,
) -> None:
    target = tmp_path / "preexisting-empty"
    target.mkdir()

    completed = _run_setup(
        "timesfm_v1",
        target,
        extra_environment={"NA_TSFM_GUARDED_CONTINUATION": "1"},
    )

    assert completed.returncode == 0
    assert list(target.iterdir()) == []


def test_setup_refuses_a_missing_parent_without_recursive_creation(
    tmp_path: Path,
) -> None:
    environment, _pip_marker = _fake_linux_setup_environment(
        tmp_path,
    )
    parent = tmp_path / "missing-parent"
    target = parent / "environment"

    completed = subprocess.run(
        ["bash", str(SETUP), "timesfm_v1", str(target)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stderr == "error: target parent is not trusted\n"
    assert not parent.exists()
    assert str(target) not in completed.stdout + completed.stderr


def test_setup_refuses_a_symbolic_link_parent_without_mutation(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    target = linked_parent / "environment"

    completed = _run_setup("timesfm_v1", target)

    assert completed.returncode != 0
    assert completed.stderr == (
        "error: target must not be or traverse a symbolic link\n"
    )
    assert list(real_parent.iterdir()) == []
    assert str(target) not in completed.stdout + completed.stderr


@pytest.mark.parametrize("mode", [0o720, 0o702])
def test_setup_refuses_a_group_or_world_writable_parent_before_writes(
    tmp_path: Path,
    mode: int,
) -> None:
    parent = tmp_path / "unsafe-parent"
    parent.mkdir(mode=0o700)
    parent.chmod(mode)
    target = parent / "environment"

    completed = _run_setup("timesfm_v1", target)

    assert completed.returncode != 0
    assert completed.stderr == "error: target parent is not trusted\n"
    assert not target.exists()
    assert str(target) not in completed.stdout + completed.stderr


@pytest.mark.parametrize("dangerous", [Path("/"), ROOT, Path.home()])
def test_setup_refuses_dangerous_targets(dangerous: Path) -> None:
    completed = _run_setup("timesfm_v1", dangerous)

    assert completed.returncode != 0
    assert "unsafe target" in completed.stderr


def test_setup_dry_run_is_sanitized_side_effect_free_and_orders_installs(
    tmp_path: Path,
) -> None:
    target = tmp_path / "private-host-path" / "environment"
    target.parent.mkdir(mode=0o700)
    token = "setup-must-never-print-this-token"

    completed = _run_setup(
        "uni2ts",
        target,
        extra_environment={
            "HF_TOKEN": token,
            "TABPFN_TOKEN": token,
            "NA_ACCEPT_MODEL_LICENSES": "CC-BY-NC-4.0",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert not target.exists()
    assert str(target) not in completed.stdout
    assert str(ROOT) not in completed.stdout
    assert token not in completed.stdout + completed.stderr
    assert "<TARGET>" in completed.stdout
    assert "<REPOSITORY>" in completed.stdout
    assert "STAGING" not in completed.stdout
    dependency_install = completed.stdout.index("--requirement")
    repository_install = completed.stdout.index(
        "--no-deps --no-build-isolation <REPOSITORY>"
    )
    assert dependency_install < repository_install
    assert "--require-hashes" in completed.stdout
    assert "--only-binary=:all:" in completed.stdout
    assert "--no-build-isolation --requirement <REPOSITORY>" in completed.stdout
    assert completed.stdout.index("<BUILD-LOCK>") < completed.stdout.index(
        "--no-build-isolation --requirement <REPOSITORY>"
    )
    assert "accept" not in completed.stdout.lower()


def test_setup_dry_run_installs_immutable_vcs_root_without_dependency_resolution(
    tmp_path: Path,
) -> None:
    completed = _run_setup("kairos", tmp_path / "environment")

    assert completed.returncode == 0, completed.stderr
    lock_install = completed.stdout.index(
        "--require-hashes --no-build-isolation --requirement "
        "<REPOSITORY>/configs/tsfm-environments/kairos.txt"
    )
    vcs_install = completed.stdout.index(
        "--no-deps --no-build-isolation --requirement "
        "<REPOSITORY>/configs/tsfm-environments/kairos.vcs"
    )
    repository_install = completed.stdout.rindex(
        "--no-deps --no-build-isolation <REPOSITORY>"
    )
    assert lock_install < vcs_install < repository_install


def test_setup_creates_a_missing_leaf_and_builds_directly_at_final_path(
    tmp_path: Path,
) -> None:
    target = tmp_path / "environment"
    environment, pip_marker = _fake_linux_setup_environment(
        tmp_path,
        final_target=target,
    )

    completed = subprocess.run(
        ["bash", str(SETUP), "kairos", str(target)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    trace = (tmp_path / "setup-stage-trace").read_text(encoding="utf-8").splitlines()
    assert len(trace) == 5
    assert trace[0].split("\t") == ["venv", str(ROOT), str(target)]
    stage_directories = {line.split("\t")[1] for line in trace}
    assert stage_directories == {str(ROOT), str(target)}
    assert pip_marker.read_text(encoding="utf-8").splitlines() == ["ran"] * 4
    assert target.is_dir()
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert (target / "pyvenv.cfg").read_text(encoding="utf-8") == (
        f"home = {ROOT}\n"
    )
    assert (target / "bin" / "generated-tool").read_text(encoding="utf-8") == (
        f"#!{target}/bin/python\n"
    )
    assert not list(tmp_path.glob(".na-tsfm-*"))


@pytest.mark.parametrize(
    ("pip_stage", "stage_name"),
    [
        (1, "build requirements"),
        (2, "hashed dependencies"),
        (3, "immutable VCS root"),
        (4, "repository"),
    ],
)
def test_setup_stops_when_target_identity_changes_between_install_stages(
    tmp_path: Path,
    pip_stage: int,
    stage_name: str,
) -> None:
    target = tmp_path / "environment"
    environment, pip_marker = _fake_linux_setup_environment(
        tmp_path,
        final_target=target,
        replace_target_after_stage=pip_stage,
    )

    completed = subprocess.run(
        ["bash", str(SETUP), "kairos", str(target)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0, stage_name
    assert completed.stderr == "error: environment target identity changed\n"
    assert {path.name for path in target.iterdir()} == {"keep.txt"}
    assert (target / "keep.txt").read_text(encoding="utf-8") == "replacement"
    assert pip_marker.read_text(encoding="utf-8").splitlines() == ["ran"] * pip_stage
    assert (tmp_path / "environment.retained-original").is_dir()
    assert str(target) not in completed.stdout + completed.stderr


def test_setup_failure_leaves_partial_environment_for_inspection(tmp_path: Path) -> None:
    target = tmp_path / "environment"
    environment, _pip_marker = _fake_linux_setup_environment(
        tmp_path,
        final_target=target,
        fail_stage=2,
    )

    completed = subprocess.run(
        ["bash", str(SETUP), "timesfm_v1", str(target)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stderr == (
        "error: dependency installation failed; subprocess output withheld\n"
    )
    assert target.is_dir()
    assert (target / "bin" / "python").is_file()
    assert str(target) not in completed.stdout + completed.stderr


def test_setup_failure_leaves_an_ordinary_real_stdlib_venv_usable(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "real-venv-bin"
    fake_bin.mkdir()
    launcher = fake_bin / "python3.11"
    launcher.write_text(
        f"""#!/usr/bin/env bash
set -eu
if [[ "${{1:-}}" == "-I" && "${{2:-}}" == "-c" ]]; then
  echo "cpython:3.11"
  exit 0
fi
if [[ "${{1:-}}" == "-m" && "${{2:-}}" == "venv" ]]; then
  {shlex.quote(sys.executable)} -m venv "$3"
  exit 9
fi
exit 99
""",
        encoding="utf-8",
    )
    launcher.chmod(0o700)
    bash_environment = tmp_path / "real-venv-bash-environment"
    bash_environment.write_text(
        """function /usr/bin/uname() {
  case "${1:-}" in
    -s) printf 'Linux\n' ;;
    -m) printf 'x86_64\n' ;;
    *) return 2 ;;
  esac
}
""",
        encoding="utf-8",
    )
    target = tmp_path / "real-environment"

    completed = subprocess.run(
        ["bash", str(SETUP), "timesfm_v1", str(target)],
        cwd=ROOT,
        env={
            **os.environ,
            "BASH_ENV": str(bash_environment),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stderr == (
        "error: environment creation failed; subprocess output withheld\n"
    )
    python_probe = subprocess.run(
        [str(target / "bin" / "python"), "-c", "print('python-ok')"],
        capture_output=True,
        text=True,
        check=False,
    )
    pip_probe = subprocess.run(
        [str(target / "bin" / "python"), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert python_probe.returncode == 0, python_probe.stderr
    assert python_probe.stdout == "python-ok\n"
    assert pip_probe.returncode == 0, pip_probe.stderr
    assert str(target) not in completed.stdout + completed.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="real setup is Linux-only")
def test_setup_withholds_subprocess_paths_and_unrecognized_credentials(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3.11"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == "-I" && "${2:-}" == "-c" ]]; then
  echo "cpython:3.11"
  exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "venv" ]]; then
  mkdir -p "$3/bin"
  cp "$0" "$3/bin/python"
  chmod +x "$3/bin/python"
  exit 0
fi
echo "leaked=${GITHUB_TOKEN:-missing} args=$*" >&2
exit 9
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    target = tmp_path / "private-target"
    token = "unrecognized-setup-credential"

    completed = subprocess.run(
        ["bash", str(SETUP), "timesfm_v1", str(target)],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "GITHUB_TOKEN": token,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "build requirement installation failed" in completed.stderr
    assert token not in completed.stdout + completed.stderr
    assert str(target) not in completed.stdout + completed.stderr
    assert str(ROOT) not in completed.stdout + completed.stderr


def test_setup_withholds_target_path_when_parent_is_not_a_directory(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3.11"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == "-I" && "${2:-}" == "-c" ]]; then
  echo "cpython:3.11"
  exit 0
fi
exit 99
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    target = Path("/dev/null/private-host-path")

    completed = subprocess.run(
        ["bash", str(SETUP), "timesfm_v1", str(target)],
        cwd=ROOT,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stderr == "error: target parent is not trusted\n"
    assert str(target) not in completed.stdout + completed.stderr


@pytest.mark.skipif(sys.platform == "linux", reason="requires an unsupported host")
def test_setup_real_run_fails_closed_outside_linux_x86_64(tmp_path: Path) -> None:
    target = tmp_path / "must-not-exist"

    completed = subprocess.run(
        ["bash", str(SETUP), "timesfm_v1", str(target)],
        cwd=ROOT,
        env=os.environ,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "support only Linux x86_64" in completed.stderr
    assert not target.exists()


def test_checkpoint_smoke_records_complete_success_metadata_atomically(
    tmp_path: Path,
) -> None:
    from numerical_agent.tsfm.smoke import run_checkpoint_smoke

    manifest_id = "method_tsfm_0003"
    command = WorkerCommand((sys.executable, "-m", "fixture"))
    output = tmp_path / "smoke.json"
    deployment_config = tmp_path / "workers.json"
    deployment_config.write_text("not read by injected loader", encoding="utf-8")
    calls: dict[str, object] = {}
    order: list[str] = []

    def deployment_loader(path, *, manifests, acknowledged_licenses):
        calls["deployment"] = (path, manifests, tuple(acknowledged_licenses))
        return SimpleNamespace(
            commands={"uni2ts": command},
            enabled_manifest_ids=frozenset({manifest_id}),
        )

    class Broker:
        def __init__(self, commands, *, timeout_seconds, parent_environment):
            order.append("broker_init")
            calls["broker_init"] = (commands, timeout_seconds, parent_environment)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls["broker_closed"] = True

        def request(self, worker_key, request):
            calls["request"] = (worker_key, request)
            return WorkerResponse.success(
                request.request_id,
                [10.0, 11.0, 12.0, 13.0],
                {
                    "checkpoint_revision": request.checkpoint_revision,
                    "device": "cpu",
                    "peak_memory_bytes": 4096,
                },
            )

    def revision_reader(_command, _manifest, _environment):
        order.append("revision_resolved")
        return "a" * 40

    ticks = iter((20.0, 20.25))
    report = run_checkpoint_smoke(
        manifest_id=manifest_id,
        deployment_config=deployment_config,
        device="cpu",
        output_path=output,
        acknowledged_licenses=("CC-BY-NC-4.0",),
        _deployment_loader=deployment_loader,
        _broker_factory=Broker,
        _version_reader=lambda _command, _environment: {
            "torch": "2.4.1",
            "uni2ts": "2.0.0",
        },
        _revision_reader=revision_reader,
        _clock=lambda: next(ticks),
    )

    assert report == {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "worker_environment": "uni2ts",
        "adapter": "uni2ts",
        "status": "success",
        "reason_code": "",
        "message": "",
        "checkpoint": "Salesforce/moirai-1.1-R-base",
        "requested_revision": "main",
        "checkpoint_revision": "a" * 40,
        "package_versions": {"torch": "2.4.1", "uni2ts": "2.0.0"},
        "requested_device": "cpu",
        "device": "cpu",
        "latency_seconds": 0.25,
        "peak_memory_bytes": 4096,
        "horizon": 4,
        "history_length": 96,
        "finite_output": True,
        "output_status": "finite",
        "output_length": 4,
    }
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    worker_key, request = calls["request"]  # type: ignore[misc]
    assert worker_key == "uni2ts"
    assert len(request.history) == 96
    assert request.history[:4] == (0.0, 0.5, 1.25, 0.75)
    assert request.history[-4:] == (23.0, 23.5, 24.25, 23.75)
    assert request.horizon == 4
    assert request.frequency == "H"
    assert request.checkpoint_revision == "a" * 40
    assert order == ["revision_resolved", "broker_init"]
    assert calls["broker_closed"] is True


def test_checkpoint_smoke_preserves_typed_license_unavailability_without_starting_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from numerical_agent.tsfm.smoke import run_checkpoint_smoke

    token = "smoke-output-must-redact-token"
    monkeypatch.setenv("HF_TOKEN", token)
    output = tmp_path / "unavailable.json"

    def deployment_loader(_path, *, manifests, acknowledged_licenses):
        assert not acknowledged_licenses
        return SimpleNamespace(commands={}, enabled_manifest_ids=frozenset())

    def forbidden_broker(*_args, **_kwargs):
        raise AssertionError("license-gated smoke must not start a worker")

    report = run_checkpoint_smoke(
        manifest_id="method_tsfm_0003",
        deployment_config=tmp_path / "workers.json",
        device="auto",
        output_path=output,
        _deployment_loader=deployment_loader,
        _broker_factory=forbidden_broker,
        _version_reader=lambda *_args: {"bad": token},
        _revision_reader=lambda *_args: token,
    )

    serialized = output.read_text(encoding="utf-8")
    assert report["status"] == "unavailable"
    assert report["reason_code"] == "license_not_acknowledged"
    assert report["finite_output"] is False
    assert report["output_status"] == "not_produced"
    assert token not in serialized
    assert "Dr-CiK" not in serialized
    assert str(tmp_path) not in serialized


def test_checkpoint_smoke_preserves_worker_failure_type_and_redacts_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from numerical_agent.tsfm.smoke import run_checkpoint_smoke

    token = "worker-smoke-secret"
    monkeypatch.setenv("TABPFN_TOKEN", token)
    output = tmp_path / "failure.json"
    manifest_id = "method_tsfm_0001"
    command = WorkerCommand((sys.executable, "-m", "fixture"))

    def deployment_loader(_path, *, manifests, acknowledged_licenses):
        return SimpleNamespace(
            commands={"timesfm_v1": command},
            enabled_manifest_ids=frozenset({manifest_id}),
        )

    class Broker:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def request(self, _worker_key, request):
            return WorkerResponse.failure(
                request.request_id,
                "unavailable",
                "checkpoint_unavailable",
                f"upstream exposed {token} at /home/private/cache/Dr-CiK-labels.json",
            )

    report = run_checkpoint_smoke(
        manifest_id=manifest_id,
        deployment_config=tmp_path / "workers.json",
        device="auto",
        output_path=output,
        _deployment_loader=deployment_loader,
        _broker_factory=Broker,
        _version_reader=lambda *_args: {"timesfm": "1.3.0"},
        _revision_reader=lambda *_args: "b" * 40,
    )

    assert report["status"] == "unavailable"
    assert report["reason_code"] == "checkpoint_unavailable"
    assert report["message"] == "worker reported unavailable: checkpoint_unavailable"
    serialized = output.read_text(encoding="utf-8")
    assert token not in serialized
    assert "/home/private" not in serialized
    assert "Dr-CiK" not in serialized


@pytest.mark.parametrize("device", ["", "gpu", "cuda:-1", "cuda:0;touch /tmp/x"])
def test_checkpoint_smoke_rejects_unreviewed_device_values(
    device: str, tmp_path: Path
) -> None:
    from numerical_agent.tsfm.smoke import run_checkpoint_smoke

    with pytest.raises(ValueError, match="device"):
        run_checkpoint_smoke(
            manifest_id="method_tsfm_0001",
            deployment_config=tmp_path / "workers.json",
            device=device,
            output_path=tmp_path / "output.json",
        )


def test_checkpoint_smoke_launcher_only_forwards_to_no_download_module() -> None:
    script = ROOT / "scripts/run_tsfm_checkpoint_smoke.sh"
    completed = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--manifest-id" in completed.stdout
    assert "--workers-config" in completed.stdout
    assert "--device" in completed.stdout
    assert "--output" in completed.stdout


def test_production_worker_reports_runtime_measurements_for_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.protocol import WorkerRequest

    class Adapter:
        def forecast(self, _request):
            return (7.0, 8.0)

    monkeypatch.setattr(worker_main, "_load_adapter", lambda _name: Adapter())
    monkeypatch.setattr(
        worker_main,
        "_runtime_measurements",
        lambda _adapter, _request: {
            "device": "cuda:0",
            "peak_memory_bytes": 8192,
        },
    )
    request = WorkerRequest(
        request_id="runtime-measurements",
        provider="legacy",
        checkpoint="official/checkpoint",
        history=(1.0, 2.0),
        horizon=2,
        frequency="H",
    )
    output = io.StringIO()

    worker_main.serve("legacy", io.StringIO(request.to_json() + "\n"), output)

    response = WorkerResponse.from_json(output.getvalue())
    assert response.metadata == {
        "checkpoint": "official/checkpoint",
        "device": "cuda:0",
        "peak_memory_bytes": 8192,
    }


def test_production_worker_returns_post_load_checkpoint_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.protocol import WorkerRequest

    revision = "a" * 40

    class Adapter:
        def forecast(self, _request):
            return (7.0, 8.0)

        def loaded_checkpoint_revision(self, _request):
            return revision

    monkeypatch.setattr(worker_main, "_load_adapter", lambda _name: Adapter())
    monkeypatch.setattr(worker_main, "_runtime_measurements", lambda *_args: {})
    request = WorkerRequest(
        request_id="attested-worker",
        provider="legacy",
        checkpoint="official/checkpoint",
        checkpoint_revision=revision,
        history=(1.0, 2.0),
        horizon=2,
        frequency="H",
    )
    output = io.StringIO()

    worker_main.serve("legacy", io.StringIO(request.to_json() + "\n"), output)

    response = WorkerResponse.from_json(output.getvalue())
    assert response.status == "success"
    assert response.metadata["checkpoint_revision"] == revision


@pytest.mark.parametrize("observed", [None, "main"])
def test_production_worker_rejects_missing_or_malformed_load_attestation(
    monkeypatch: pytest.MonkeyPatch, observed: str | None
) -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.protocol import WorkerRequest

    class Adapter:
        def forecast(self, _request):
            return (7.0, 8.0)

        if observed is not None:
            def loaded_checkpoint_revision(self, _request):
                return observed

    monkeypatch.setattr(worker_main, "_load_adapter", lambda _name: Adapter())
    request = WorkerRequest(
        request_id="unattested-worker",
        provider="legacy",
        checkpoint="official/checkpoint",
        checkpoint_revision="a" * 40,
        history=(1.0, 2.0),
        horizon=2,
        frequency="H",
    )
    output = io.StringIO()

    worker_main.serve("legacy", io.StringIO(request.to_json() + "\n"), output)

    response = WorkerResponse.from_json(output.getvalue())
    assert response.status == "unavailable"
    assert response.reason_code == "checkpoint_attestation_unavailable"


def test_version_inventory_never_uses_the_forbidden_project_label() -> None:
    from numerical_agent.tsfm.smoke import _read_package_versions

    versions = _read_package_versions(
        WorkerCommand((sys.executable,)),
        os.environ,
    )

    assert all("drcik" not in name.lower() for name in versions)


def test_ttm_smoke_records_the_exact_reviewed_model_revision() -> None:
    from numerical_agent.tsfm.smoke import _repository_revision, _requested_revision

    manifest = ManifestRegistry.load_default()["method_tsfm_0006"]

    assert _requested_revision(manifest) == "512-96-ft-r2.1"
    assert _repository_revision(manifest) == "512-96-ft-r2.1"


def test_worker_does_not_infer_cpu_from_cuda_unavailability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.protocol import WorkerRequest

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    request = WorkerRequest(
        request_id="mps-device",
        provider="dedicated",
        checkpoint="Prior-Labs/TabPFN-v2-reg",
        history=(1.0, 2.0),
        horizon=2,
        frequency="H",
    )

    assert worker_main._adapter_device(object(), request) is None


def test_smoke_device_request_survives_the_controlled_worker_boundary() -> None:
    from numerical_agent.tsfm.security import controlled_worker_environment
    from numerical_agent.tsfm.smoke import _environment_for_device

    environment = _environment_for_device("cpu")

    assert controlled_worker_environment(environment)["NA_TSFM_DEVICE"] == "cpu"


def test_tabpfn_local_device_selection_honors_explicit_smoke_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm.workers.dedicated import _tabpfn_execution_device

    fake_torch = SimpleNamespace(
        device=lambda value: value,
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setenv("NA_TSFM_DEVICE", "cpu")

    assert _tabpfn_execution_device(fake_torch) == "cpu"


def test_worker_records_process_peak_memory_for_tabpfn_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm import worker_main
    from numerical_agent.tsfm.protocol import WorkerRequest

    fake_resource = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _scope: SimpleNamespace(ru_maxrss=2048),
    )
    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    adapter = SimpleNamespace(
        _models={"Prior-Labs/tabpfn_3": object()},
        _backends={"method_tsfm_0029": SimpleNamespace(device="mps")},
    )
    request = WorkerRequest(
        request_id="mps-measurements",
        provider="dedicated",
        checkpoint="Prior-Labs/tabpfn_3",
        history=(1.0, 2.0),
        horizon=2,
        frequency="H",
    )

    measurements = worker_main._runtime_measurements(adapter, request)

    assert measurements["device"] == "mps"
    assert isinstance(measurements["peak_memory_bytes"], int)
    assert measurements["peak_memory_bytes"] > 0


@pytest.mark.parametrize(
    ("versions", "peak_memory", "reason_code"),
    [
        ({}, 1024, "package_versions_unavailable"),
        ({"torch": "2.4.1"}, None, "peak_memory_unavailable"),
    ],
)
def test_successful_forecast_requires_mandatory_smoke_evidence(
    tmp_path: Path,
    versions: dict[str, str],
    peak_memory: int | None,
    reason_code: str,
) -> None:
    from numerical_agent.tsfm.smoke import run_checkpoint_smoke

    manifest_id = "method_tsfm_0001"
    command = WorkerCommand((sys.executable, "-m", "fixture"))

    def deployment_loader(_path, *, manifests, acknowledged_licenses):
        return SimpleNamespace(
            commands={"timesfm_v1": command},
            enabled_manifest_ids=frozenset({manifest_id}),
        )

    class Broker:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def request(self, _worker_key, request):
            metadata: dict[str, object] = {
                "checkpoint_revision": request.checkpoint_revision,
                "device": "cpu",
            }
            if peak_memory is not None:
                metadata["peak_memory_bytes"] = peak_memory
            return WorkerResponse.success(
                request.request_id,
                [1.0, 2.0, 3.0, 4.0],
                metadata,
            )

    report = run_checkpoint_smoke(
        manifest_id=manifest_id,
        deployment_config=tmp_path / "workers.json",
        device="cpu",
        output_path=tmp_path / "report.json",
        _deployment_loader=deployment_loader,
        _broker_factory=Broker,
        _version_reader=lambda *_args: versions,
        _revision_reader=lambda *_args: "c" * 40,
    )

    assert report["status"] == "unavailable"
    assert report["reason_code"] == reason_code
    assert report["finite_output"] is True


def test_smoke_does_not_start_broker_without_pre_resolved_revision(
    tmp_path: Path,
) -> None:
    from numerical_agent.tsfm.smoke import run_checkpoint_smoke

    manifest_id = "method_tsfm_0001"
    command = WorkerCommand((sys.executable, "-m", "fixture"))

    def deployment_loader(_path, *, manifests, acknowledged_licenses):
        return SimpleNamespace(
            commands={"timesfm_v1": command},
            enabled_manifest_ids=frozenset({manifest_id}),
        )

    def forbidden_broker(*_args, **_kwargs):
        raise AssertionError("unresolved revision must stop before broker start")

    report = run_checkpoint_smoke(
        manifest_id=manifest_id,
        deployment_config=tmp_path / "workers.json",
        device="cpu",
        output_path=tmp_path / "report.json",
        _deployment_loader=deployment_loader,
        _broker_factory=forbidden_broker,
        _version_reader=lambda *_args: {"torch": "2.4.1"},
        _revision_reader=lambda *_args: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    assert report["status"] == "unavailable"
    assert report["reason_code"] == "checkpoint_revision_unavailable"
    assert report["finite_output"] is False


@pytest.mark.parametrize(
    ("observed", "reason_code"),
    [
        (None, "checkpoint_revision_unavailable"),
        ("main", "checkpoint_revision_unavailable"),
        ("b" * 40, "checkpoint_revision_mismatch"),
    ],
)
def test_smoke_rejects_missing_malformed_or_mismatched_worker_attestation(
    tmp_path: Path, observed: str | None, reason_code: str
) -> None:
    from numerical_agent.tsfm.smoke import run_checkpoint_smoke

    manifest_id = "method_tsfm_0001"
    command = WorkerCommand((sys.executable, "-m", "fixture"))

    def deployment_loader(_path, *, manifests, acknowledged_licenses):
        return SimpleNamespace(
            commands={"timesfm_v1": command},
            enabled_manifest_ids=frozenset({manifest_id}),
        )

    class Broker:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def request(self, _worker_key, request):
            metadata: dict[str, object] = {
                "device": "cpu",
                "peak_memory_bytes": 1024,
            }
            if observed is not None:
                metadata["checkpoint_revision"] = observed
            return WorkerResponse.success(
                request.request_id,
                [1.0, 2.0, 3.0, 4.0],
                metadata,
            )

    report = run_checkpoint_smoke(
        manifest_id=manifest_id,
        deployment_config=tmp_path / "workers.json",
        device="cpu",
        output_path=tmp_path / "report.json",
        _deployment_loader=deployment_loader,
        _broker_factory=Broker,
        _version_reader=lambda *_args: {"torch": "2.4.1"},
        _revision_reader=lambda *_args: "a" * 40,
    )

    assert report["status"] == "unavailable"
    assert report["reason_code"] == reason_code


def test_explicit_cuda_smoke_rejects_cpu_execution(tmp_path: Path) -> None:
    from numerical_agent.tsfm.smoke import run_checkpoint_smoke

    manifest_id = "method_tsfm_0001"
    command = WorkerCommand((sys.executable, "-m", "fixture"))

    def deployment_loader(_path, *, manifests, acknowledged_licenses):
        return SimpleNamespace(
            commands={"timesfm_v1": command},
            enabled_manifest_ids=frozenset({manifest_id}),
        )

    class Broker:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def request(self, _worker_key, request):
            return WorkerResponse.success(
                request.request_id,
                [1.0, 2.0, 3.0, 4.0],
                {
                    "checkpoint_revision": request.checkpoint_revision,
                    "device": "cpu",
                    "peak_memory_bytes": 1024,
                },
            )

    report = run_checkpoint_smoke(
        manifest_id=manifest_id,
        deployment_config=tmp_path / "workers.json",
        device="cuda:0",
        output_path=tmp_path / "report.json",
        _deployment_loader=deployment_loader,
        _broker_factory=Broker,
        _version_reader=lambda *_args: {"torch": "2.4.1"},
        _revision_reader=lambda *_args: "d" * 40,
    )

    assert report["status"] == "unavailable"
    assert report["reason_code"] == "device_mismatch"
    assert report["device"] == "cpu"
