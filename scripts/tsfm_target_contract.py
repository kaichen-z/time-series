#!/usr/bin/env python3
"""Validate the local trusted-parent contract for a TSFM virtual environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat


_TRUSTED_SYSTEM_UIDS = frozenset({0})


class ParentContractError(ValueError):
    pass


class TargetContractError(ValueError):
    pass


def _directory_metadata(path: Path, error_type: type[ValueError]) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise error_type("trusted parent contract failed") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise error_type("trusted parent contract failed")
    return metadata


def _validate_parent_chain(parent: Path) -> None:
    """Validate the contiguous euid-owned chain up to its system boundary."""
    effective_uid = os.geteuid()
    current = parent
    metadata = _directory_metadata(current, ParentContractError)
    if metadata.st_uid != effective_uid:
        raise ParentContractError("trusted parent contract failed")

    while True:
        if metadata.st_uid != effective_uid or metadata.st_mode & 0o022:
            raise ParentContractError("trusted parent contract failed")
        ancestor = current.parent
        if ancestor == current:
            return
        ancestor_metadata = _directory_metadata(ancestor, ParentContractError)
        if ancestor_metadata.st_uid != effective_uid:
            if ancestor_metadata.st_uid not in _TRUSTED_SYSTEM_UIDS:
                raise ParentContractError("trusted parent contract failed")
            return
        current = ancestor
        metadata = ancestor_metadata


def _validate_target(path: Path, expected_identity: str | None = None) -> str:
    metadata = _directory_metadata(path, TargetContractError)
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise TargetContractError("target contract failed")
    identity = f"{metadata.st_dev}:{metadata.st_ino}"
    if expected_identity is not None and identity != expected_identity:
        raise TargetContractError("target identity changed")
    return identity


def check_target(path: Path) -> None:
    _validate_parent_chain(path.parent)
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    _validate_target(path)
    try:
        if any(path.iterdir()):
            raise TargetContractError("target is not empty")
    except OSError as exc:
        raise TargetContractError("target contract failed") from exc


def prepare_target(path: Path) -> str:
    check_target(path)
    try:
        os.lstat(path)
    except FileNotFoundError:
        try:
            os.mkdir(path, 0o700)
        except OSError as exc:
            raise TargetContractError("target could not be created") from exc
        metadata = _directory_metadata(path, TargetContractError)
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise TargetContractError("created target mode is not 0700")
    return _validate_target(path)


def verify_target(path: Path, expected_identity: str) -> None:
    _validate_parent_chain(path.parent)
    _validate_target(path, expected_identity)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("action", choices=("check", "prepare", "verify"))
    parser.add_argument("target", type=Path)
    parser.add_argument("identity", nargs="?")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        if arguments.action == "check":
            check_target(arguments.target)
        elif arguments.action == "prepare":
            print(prepare_target(arguments.target))
        else:
            if arguments.identity is None:
                raise TargetContractError("missing target identity")
            verify_target(arguments.target, arguments.identity)
    except ParentContractError:
        return 2
    except (OSError, TargetContractError, ValueError):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
