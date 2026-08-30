"""Filesystem persistence for generic self-evolution runs."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping

from .contracts import active_envelope, load_active_release
from common.payload import (
    canonical_json_bytes,
    canonical_json_line_bytes,
    is_simple_filename,
    read_json_object,
)


class JsonArtifactStore:
    """Persist JSON artifacts atomically and traces append-only."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_artifact(self, name: str, payload: Mapping[str, object]) -> Path:
        if not is_simple_filename(name):
            raise ValueError("artifact name must be a simple non-empty filename stem")
        destination = self.root / f"{name}.json"
        self._write_json(
            destination,
            active_envelope(payload, context=f"active artifact {name!r}"),
        )
        return destination

    def save_checkpoint(self, payload: Mapping[str, object]) -> Path:
        destination = self.root / "checkpoint.json"
        self._write_json(destination, active_envelope(payload, context="active checkpoint"))
        return destination

    def load_checkpoint(self) -> dict[str, object] | None:
        source = self.root / "checkpoint.json"
        if not source.exists():
            return None
        payload = read_json_object(source)
        return load_active_release(payload)

    def append_trace(self, payload: Mapping[str, object]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("trace payload must be a mapping")
        trace_path = self.root / "evolution_trace.jsonl"
        bound = active_envelope(payload, context="active evolution trace")
        with trace_path.open("ab") as handle:
            handle.write(canonical_json_line_bytes(bound))

    @staticmethod
    def _write_json(destination: Path, payload: Mapping[str, object]) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                # Write through a temporary file so a crash can never leave a partial artifact.
                handle.write(canonical_json_bytes(dict(payload)))
                temporary_path = Path(handle.name)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
