#!/usr/bin/env python3
"""Offline structural validation for reviewed TSFM environment locks."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


PYTHON_TARGETS = {
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
VCS_ENVIRONMENTS = frozenset({"kairos", "lag_llama"})
VCS_BUILD_BACKENDS = {
    "kairos": ("hatchling", "1.27.0"),
    "lag_llama": ("setuptools", "80.9.0"),
}
BUILD_TOOLS = {
    "setuptools": "80.9.0",
    "wheel": "0.45.1",
}

_EXACT_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^]]+\])?==(?P<version>[^\s;\\]+)$"
)
_LOCK_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^]]+\])?==(?P<version>[^\s;\\]+)\s*(?:\\)?$"
)
_HASH = re.compile(r"^--hash=sha256:[0-9a-f]{64}(?:\s+\\)?$")
_VCS = re.compile(
    r"^[A-Za-z0-9_.-]+\s+@\s+git\+https://github\.com/[^@\s]+"
    r"@[0-9a-f]{40}(?:#[^\s]+)?$"
)


class LockValidationError(ValueError):
    """A checked-in environment specification violates the lock contract."""


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(path: Path, *, lock: bool) -> dict[str, str]:
    parsed: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue

        matcher = _LOCK_REQUIREMENT if lock else _EXACT_REQUIREMENT
        match = matcher.fullmatch(stripped)
        if match is None:
            kind = "lock requirement" if lock else "direct input"
            raise LockValidationError(
                f"{path.name}: {kind} must use an exact == pin"
            )

        name = _canonical_name(match.group("name"))
        if name in parsed:
            raise LockValidationError(f"{path.name}: duplicate requirement {name}")
        parsed[name] = match.group("version")

        if not lock:
            continue

        hashes = 0
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate.startswith("--hash="):
                break
            if _HASH.fullmatch(candidate) is None:
                raise LockValidationError(
                    f"{path.name}: malformed SHA-256 hash for {name}"
                )
            hashes += 1
            index += 1
        if hashes == 0:
            raise LockValidationError(
                f"{path.name}: requirement {name} is missing SHA-256 hash"
            )

    if not parsed:
        raise LockValidationError(f"{path.name}: contains no requirements")
    return parsed


def _metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"# ([a-z-]+): ([^\s]+)", raw_line)
        if match is not None:
            metadata[match.group(1)] = match.group(2)
    return metadata


def _hashed_requirement_blocks(path: Path) -> dict[str, str]:
    blocks: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        match = _LOCK_REQUIREMENT.fullmatch(lines[index].strip())
        if match is None:
            index += 1
            continue
        name = _canonical_name(match.group("name"))
        block = [lines[index]]
        index += 1
        while index < len(lines) and lines[index].strip().startswith("--hash="):
            block.append(lines[index])
            index += 1
        blocks[name] = "\n".join(block)
    return blocks


def _validate_metadata(path: Path, environment: str) -> None:
    metadata = _metadata(path)
    expected = {
        "tsfm-lock-version": "1",
        "environment": environment,
        "implementation": "CPython",
        "python-version": PYTHON_TARGETS[environment],
        "platform": "linux",
        "architecture": "x86_64",
    }
    labels = {
        "python-version": "Python metadata",
        "platform": "platform metadata",
        "architecture": "architecture metadata",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            label = labels.get(key, f"{key} metadata")
            raise LockValidationError(
                f"{path.name}: {label} must be {value}"
            )


def _validate_vcs(path: Path) -> None:
    requirements = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(requirements) != 1 or _VCS.fullmatch(requirements[0]) is None:
        raise LockValidationError(
            f"{path.name}: VCS root must be a single immutable 40-hex GitHub commit"
        )


def validate_environment(directory: Path, environment: str) -> None:
    if environment not in PYTHON_TARGETS:
        raise LockValidationError("environment key is not allowlisted")

    direct_path = directory / f"{environment}.in"
    lock_path = directory / f"{environment}.txt"
    for path in (direct_path, lock_path):
        if not path.is_file():
            raise LockValidationError(f"{path.name}: required specification is missing")

    _validate_metadata(lock_path, environment)
    direct = _requirements(direct_path, lock=False)
    locked = _requirements(lock_path, lock=True)
    for name, version in direct.items():
        if locked.get(name) != version:
            raise LockValidationError(
                f"{lock_path.name}: direct requirement {name}=={version} "
                "is missing from the hashed closure"
            )
    for build_tool, version in BUILD_TOOLS.items():
        if direct.get(build_tool) != version or locked.get(build_tool) != version:
            raise LockValidationError(
                f"{environment}: required build tool {build_tool}=={version} "
                "must be in the hashed closure"
            )

    vcs_path = directory / f"{environment}.vcs"
    if environment in VCS_ENVIRONMENTS:
        if not vcs_path.is_file():
            raise LockValidationError(f"{vcs_path.name}: immutable VCS root is missing")
        _validate_vcs(vcs_path)
        backend, version = VCS_BUILD_BACKENDS[environment]
        if direct.get(backend) != version or locked.get(backend) != version:
            raise LockValidationError(
                f"{environment}: required VCS build backend {backend}=={version} "
                "must be in the hashed closure"
            )
    elif vcs_path.exists():
        raise LockValidationError(f"{vcs_path.name}: unexpected VCS root")

    if environment == "tempo_legacy" and (
        direct.get("numpy") != "1.26.4" or locked.get("numpy") != "1.26.4"
    ):
        raise LockValidationError(
            "tempo_legacy: NumPy must be pinned to the tested 1.x release 1.26.4"
        )


def validate_directory(directory: Path) -> None:
    expected = set(PYTHON_TARGETS)
    for suffix in (".in", ".txt"):
        actual = {path.stem for path in directory.glob(f"*{suffix}")}
        if actual != expected:
            raise LockValidationError(
                f"{suffix} specifications must cover exactly the 11 allowlisted environments"
            )
    actual_vcs = {path.stem for path in directory.glob("*.vcs")}
    if actual_vcs != set(VCS_ENVIRONMENTS):
        raise LockValidationError(
            ".vcs specifications must cover exactly kairos and lag_llama"
        )
    for environment in PYTHON_TARGETS:
        validate_environment(directory, environment)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/tsfm-environments",
    )
    parser.add_argument("--environment", choices=tuple(PYTHON_TARGETS))
    parser.add_argument("--emit-build-lock", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.emit_build_lock and arguments.environment is None:
        _parser().error("--emit-build-lock requires --environment")
    try:
        if arguments.environment is None:
            validate_directory(arguments.directory)
            count = len(PYTHON_TARGETS)
        else:
            validate_environment(arguments.directory, arguments.environment)
            count = 1
    except (LockValidationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if arguments.emit_build_lock:
        blocks = _hashed_requirement_blocks(
            arguments.directory / f"{arguments.environment}.txt"
        )
        print("\n\n".join(blocks[name] for name in BUILD_TOOLS))
        return 0
    print(f"validated {count} TSFM environment locks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
