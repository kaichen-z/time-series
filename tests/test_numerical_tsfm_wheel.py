from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]


def _requirement_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0]


def _copy_distribution(name: str, site_packages: Path) -> None:
    """Provision one declared local build tool without contacting an index."""

    distribution = metadata.distribution(name)
    source_root = Path(distribution.locate_file("")).resolve()
    for entry in distribution.files or ():
        source = Path(distribution.locate_file(entry)).resolve()
        try:
            relative = source.relative_to(source_root)
        except ValueError:
            continue
        if not source.is_file():
            continue
        destination = site_packages / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    metadata_path = Path(getattr(distribution, "_path", "")).resolve()
    try:
        metadata_relative = metadata_path.relative_to(source_root)
    except ValueError as error:
        raise AssertionError(
            f"{name} metadata is outside its installed distribution root"
        ) from error
    metadata_destination = site_packages / metadata_relative
    if metadata_path.is_dir():
        shutil.copytree(metadata_path, metadata_destination, dirs_exist_ok=True)
    elif metadata_path.is_file():
        metadata_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata_path, metadata_destination)
    else:
        raise AssertionError(f"{name} installed distribution metadata is missing")


def _provision_declared_build_tools(clean_python: Path, source: Path) -> None:
    """Copy only build requirements also promised by the development extra."""

    pyproject = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    build_names = {
        _requirement_name(requirement)
        for requirement in pyproject["build-system"]["requires"]
    }
    dev_names = {
        _requirement_name(requirement)
        for requirement in pyproject["project"]["optional-dependencies"]["dev"]
    }
    site_packages = Path(
        subprocess.check_output(
            [
                str(clean_python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            text=True,
        ).strip()
    )
    for name in sorted(build_names & dev_names):
        _copy_distribution(name, site_packages)


def test_copy_distribution_includes_metadata_omitted_from_declared_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-site"
    package = source / "fixture_build_tool"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    egg_info = source / "fixture_build_tool-1.0-py3.12.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: fixture-build-tool\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (egg_info / "entry_points.txt").write_text(
        "[distutils.commands]\ndist_info = fixture_build_tool:VALUE\n",
        encoding="utf-8",
    )

    class EggInfoDistribution:
        files = (Path("fixture_build_tool/__init__.py"),)
        _path = egg_info

        @staticmethod
        def locate_file(entry: str | Path) -> Path:
            return source / entry

    monkeypatch.setattr(
        metadata,
        "distribution",
        lambda _name: EggInfoDistribution(),
    )
    destination = tmp_path / "destination-site"

    _copy_distribution("fixture-build-tool", destination)

    assert (destination / "fixture_build_tool" / "__init__.py").is_file()
    assert (
        destination
        / "fixture_build_tool-1.0-py3.12.egg-info"
        / "entry_points.txt"
    ).is_file()


def test_installed_wheel_constructs_default_registry_without_repository_data(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PIP_NO_INDEX"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    source = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            ".venv",
            "__pycache__",
            "*.egg-info",
            "build",
            "external",
            "outputs",
            "runs",
        ),
    )
    clean_venv = tmp_path / "clean-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(clean_venv)],
        check=True,
        text=True,
        capture_output=True,
    )
    clean_python = (
        clean_venv / "Scripts" / "python.exe"
        if os.name == "nt"
        else clean_venv / "bin" / "python"
    )
    _provision_declared_build_tools(clean_python, source)

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    build = subprocess.run(
        [
            str(clean_python),
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(source),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    wheel = next(wheel_dir.glob("drcik_minimal_agent-*.whl"))

    install = subprocess.run(
        [
            str(clean_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            str(wheel),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr

    smoke = subprocess.run(
        [
            str(clean_python),
            "-I",
            "-c",
            (
                "from argparse import Namespace; "
                "from pathlib import Path; "
                "import numerical_agent; "
                "from numerical_agent.main import _runtime_registry; "
                "from numerical_agent.tsfm.manifests import ManifestRegistry; "
                "assert Path(numerical_agent.__file__).resolve().is_relative_to("
                "Path(__import__('sys').prefix).resolve()); "
                "registry = ManifestRegistry.load_default(); "
                "assert len(registry) == 31; "
                "runtimes = _runtime_registry(Namespace(tsfm_runtimes='', "
                "model_cache_dir=None, tsfm_workers_config=None, "
                "acknowledged_model_licenses='')); "
                "runtimes.close(); "
                "print(len(registry))"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert smoke.returncode == 0, smoke.stderr
    assert smoke.stdout.strip() == "31"
