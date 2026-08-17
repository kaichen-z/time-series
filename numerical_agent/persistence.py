"""Artifact storage that also writes each method's generated code as a readable file."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from common.evolution_core.persistence import JsonArtifactStore


class MethodSourceArtifactStore(JsonArtifactStore):
    """Persist the JSON artifact, then mirror each implemented method into a .py file."""

    def save_artifact(self, name: str, payload: Mapping[str, object]) -> Path:
        destination = super().save_artifact(name, payload)
        self._write_method_sources(name, payload)
        return destination

    def _write_method_sources(self, name: str, payload: Mapping[str, object]) -> None:
        methods = payload.get("methods")
        if not isinstance(methods, list):
            return
        directory = self.root / name / "methods"
        for method in methods:
            if not isinstance(method, Mapping):
                continue
            method_id, code = self._method_source(method)
            if method_id is None or code is None:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{method_id}.py").write_text(code, encoding="utf-8")

    @staticmethod
    def _method_source(
        method: Mapping[str, object],
    ) -> tuple[str | None, str | None]:
        """Return the method's filename stem and code, or (None, None) when absent."""
        definition = method.get("definition")
        candidate = method.get("candidate")
        if not isinstance(definition, Mapping) or not isinstance(candidate, Mapping):
            return None, None
        implementation = candidate.get("implementation")
        if not isinstance(implementation, Mapping):
            return None, None
        code = implementation.get("code")
        if not isinstance(code, str) or not code.strip():
            return None, None
        method_id = str(definition.get("method_id", ""))
        # Reject any id that is not a plain filename so it cannot escape the directory.
        if not method_id or Path(method_id).name != method_id:
            return None, None
        return method_id, code
