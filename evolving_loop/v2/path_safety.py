"""Small, shared filesystem exceptions for trusted V2 run boundaries."""
from __future__ import annotations

import stat
from pathlib import Path


_SYSTEM_TMP_ALIAS = Path("/tmp")
_SYSTEM_TMP_TARGET = Path("/private/tmp")


def _system_tmp_alias(path: Path, mode: int | None = None) -> bool:
    """Return true only for macOS's fixed ``/tmp -> /private/tmp`` link.

    Other platforms ordinarily expose ``/tmp`` as a real directory, which is
    already accepted by the normal directory checks.  A caller-created link is
    never an equivalent substitute, even if it points at a directory.
    """
    try:
        path = Path(path)
        mode = path.lstat().st_mode if mode is None else mode
        return (
            path == _SYSTEM_TMP_ALIAS
            and stat.S_ISLNK(mode)
            and path.resolve(strict=True) == _SYSTEM_TMP_TARGET
            and stat.S_ISDIR(_SYSTEM_TMP_TARGET.lstat().st_mode)
        )
    except OSError:
        return False


def physical_system_tmp_path(path: Path) -> Path:
    """Map the one accepted system alias to its physical directory."""
    path = Path(path)
    if path.is_relative_to(_SYSTEM_TMP_ALIAS) and _system_tmp_alias(_SYSTEM_TMP_ALIAS):
        return _SYSTEM_TMP_TARGET / path.relative_to(_SYSTEM_TMP_ALIAS)
    return path
