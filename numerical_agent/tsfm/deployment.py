"""Deployment-local interpreter bindings and explicit TSFM license gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import MappingProxyType

from .broker import WorkerCommand
from .manifests import ManifestRegistry
from .security import SecretRedactor


_DEPLOYMENT_FIELDS = frozenset({"schema_version", "environments"})
_ENVIRONMENT_FIELDS = frozenset({"interpreter"})
_WORKER_MODULE = "numerical_agent.tsfm.worker_main"


def _strict_json_object(payload: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r} is not allowed")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        decoded: dict[str, object] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError(f"deployment JSON has duplicate field {key!r}")
            decoded[key] = value
        return decoded

    try:
        decoded = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("TSFM deployment must be valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("TSFM deployment must be a JSON object")
    return decoded


def _exact_fields(
    payload: Mapping[str, object], expected: frozenset[str], *, kind: str
) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    details = []
    if missing:
        details.append(f"missing fields: {missing!r}")
    if unknown:
        details.append(f"unknown fields: {unknown!r}")
    if details:
        raise ValueError(f"{kind} " + "; ".join(details))


def _required_license_ids(manifests: ManifestRegistry) -> frozenset[str]:
    return frozenset(
        manifest.license_id
        for manifest in manifests.values()
        if manifest.status == "experimental_unverified"
        and manifest.license_acknowledgement_required
    )


def _validate_acknowledgements(
    values: Sequence[str], manifests: ManifestRegistry
) -> frozenset[str]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(
                "model license acknowledgements must be exact non-empty identifiers"
            )
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError("model license acknowledgements contain duplicate values")
    unknown = set(normalized) - _required_license_ids(manifests)
    if unknown:
        raise ValueError(
            "model license acknowledgements contain unknown license identifiers"
        )
    return frozenset(normalized)


def parse_acknowledged_licenses(
    value: str, manifests: ManifestRegistry
) -> frozenset[str]:
    """Parse exact comma-separated deployment-local license acknowledgements."""

    if not isinstance(value, str):
        raise ValueError("model license acknowledgements must be a string")
    if not value.strip():
        return frozenset()
    raw = value.split(",")
    if any(not item.strip() for item in raw):
        raise ValueError("model license acknowledgements contain an empty value")
    return _validate_acknowledgements(
        tuple(item.strip() for item in raw), manifests
    )


def redact_environment_secrets(message: object) -> str:
    """Compatibility wrapper for one-shot local error sanitization."""

    return SecretRedactor.from_environment().redact_text(message)


@dataclass(frozen=True)
class TSFMDeployment:
    """Validated commands and manifest IDs enabled by one local deployment."""

    commands: Mapping[str, WorkerCommand]
    enabled_manifest_ids: frozenset[str]
    acknowledged_licenses: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "commands", MappingProxyType(dict(self.commands)))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        manifests: ManifestRegistry | None = None,
        acknowledged_licenses: Sequence[str] = (),
    ) -> "TSFMDeployment":
        registry = (
            manifests if manifests is not None else ManifestRegistry.load_default()
        )
        acknowledgements = _validate_acknowledgements(
            acknowledged_licenses, registry
        )
        try:
            payload = _strict_json_object(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as error:
            raise ValueError(
                f"cannot load TSFM deployment ({type(error).__name__})"
            ) from None
        _exact_fields(payload, _DEPLOYMENT_FIELDS, kind="TSFM deployment")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("TSFM deployment schema_version must be 1")
        environments = payload["environments"]
        if not isinstance(environments, Mapping) or not environments:
            raise ValueError("TSFM deployment environments must be a non-empty object")
        if not all(isinstance(key, str) and key for key in environments):
            raise ValueError("TSFM deployment environment names must be non-empty strings")

        reviewed_adapters: dict[str, set[str]] = {}
        for manifest in registry.values():
            if manifest.status == "experimental_unverified":
                reviewed_adapters.setdefault(manifest.worker_environment, set()).add(
                    manifest.adapter
                )
        unknown = set(environments) - set(reviewed_adapters)
        if unknown:
            raise ValueError(f"unknown worker environments: {sorted(unknown)!r}")

        commands: dict[str, WorkerCommand] = {}
        for environment, raw_entry in environments.items():
            if not isinstance(raw_entry, Mapping):
                raise ValueError(
                    f"worker environment {environment!r} must be an object"
                )
            _exact_fields(
                raw_entry,
                _ENVIRONMENT_FIELDS,
                kind=f"worker environment {environment!r}",
            )
            interpreter = raw_entry["interpreter"]
            if not isinstance(interpreter, str) or not interpreter:
                raise ValueError(
                    f"worker environment {environment!r} interpreter must be a path"
                )
            interpreter_path = Path(interpreter).expanduser()
            if not interpreter_path.is_absolute():
                raise ValueError(
                    f"worker environment {environment!r} interpreter must be absolute"
                )
            if not interpreter_path.exists() or not interpreter_path.is_file():
                raise ValueError(
                    f"worker environment {environment!r} interpreter does not exist"
                )
            if not os.access(interpreter_path, os.X_OK):
                raise ValueError(
                    f"worker environment {environment!r} interpreter is not executable"
                )
            adapters = reviewed_adapters[environment]
            if len(adapters) != 1:
                raise ValueError(
                    f"worker environment {environment!r} has inconsistent reviewed adapters"
                )
            adapter = next(iter(adapters))
            commands[environment] = WorkerCommand(
                (
                    os.path.normpath(str(interpreter_path)),
                    "-m",
                    _WORKER_MODULE,
                    "--adapter",
                    adapter,
                )
            )

        enabled = frozenset(
            manifest.method_id
            for manifest in registry.values()
            if manifest.status == "experimental_unverified"
            and manifest.worker_environment in commands
            and (
                not manifest.license_acknowledgement_required
                or manifest.license_id in acknowledgements
            )
        )
        return cls(
            commands=commands,
            enabled_manifest_ids=enabled,
            acknowledged_licenses=acknowledgements,
        )
