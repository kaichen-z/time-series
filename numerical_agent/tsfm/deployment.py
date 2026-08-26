"""Deployment-local interpreter bindings and explicit TSFM license gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from types import MappingProxyType

from .broker import WorkerCommand
from .manifests import ManifestRegistry
from .security import SecretRedactor, controlled_worker_environment


_DEPLOYMENT_FIELDS = frozenset({"schema_version", "environments"})
_ENVIRONMENT_FIELDS = frozenset({"interpreter"})
_WORKER_MODULE = "numerical_agent.tsfm.worker_main"
_RUNTIME_IMPORTS = MappingProxyType(
    {
        "timesfm_v1": (
            "torch",
            "timesfm",
            "timesfm.timesfm_torch",
            "huggingface_hub",
            "numerical_agent.tsfm.workers.legacy",
        ),
        "uni2ts": (
            "torch",
            "pandas",
            "gluonts.dataset.pandas",
            "uni2ts.model.moirai",
            "uni2ts.model.moirai2",
            "uni2ts.model.moirai_moe",
            "numerical_agent.tsfm.workers.uni2ts",
        ),
        "lag_llama": (
            "torch",
            "pandas",
            "gluonts.dataset.pandas",
            "huggingface_hub",
            "lag_llama.gluon.estimator",
            "numerical_agent.tsfm.workers.legacy",
        ),
        "granite_tsfm": (
            "torch",
            "tsfm_public",
            "tsfm_public.toolkit.get_model",
            "numerical_agent.tsfm.workers.granite",
        ),
        "timer_legacy": (
            "torch",
            "transformers",
            "numerical_agent.tsfm.workers.transformer_generation",
        ),
        "transformers_recent": (
            "torch",
            "transformers",
            "numerical_agent.tsfm.workers.transformer_generation",
        ),
        "tempo_legacy": (
            "torch",
            "huggingface_hub",
            "omegaconf",
            "tempo.models.TEMPO",
            "numerical_agent.tsfm.workers.legacy",
        ),
        "toto2": (
            "torch",
            "toto2",
            "numerical_agent.tsfm.workers.dedicated",
        ),
        "kairos": (
            "torch",
            "tsfm.model.kairos",
            "numerical_agent.tsfm.workers.transformer_generation",
        ),
        "tirex": (
            "torch",
            "tirex",
            "numerical_agent.tsfm.workers.dedicated",
        ),
        "tabpfn_ts": (
            "torch",
            "pandas",
            "huggingface_hub",
            "tabpfn.errors",
            "tabpfn.model_loading",
            "tabpfn_time_series",
            "numerical_agent.tsfm.workers.dedicated",
        ),
    }
)
_RUNTIME_SYMBOLS = MappingProxyType(
    {
        "timesfm_v1": {
            "huggingface_hub": ("hf_hub_download",),
            "timesfm": ("TimesFmCheckpoint", "TimesFmHparams", "freq_map"),
            "timesfm.timesfm_torch": ("TimesFmTorch",),
        },
        "uni2ts": {
            "gluonts.dataset.pandas": ("PandasDataset",),
            "uni2ts.model.moirai": ("MoiraiForecast", "MoiraiModule"),
            "uni2ts.model.moirai2": ("Moirai2Forecast", "Moirai2Module"),
            "uni2ts.model.moirai_moe": (
                "MoiraiMoEForecast",
                "MoiraiMoEModule",
            ),
        },
        "lag_llama": {
            "gluonts.dataset.pandas": ("PandasDataset",),
            "huggingface_hub": ("hf_hub_download",),
            "lag_llama.gluon.estimator": ("LagLlamaEstimator",),
        },
        "granite_tsfm": {
            "tsfm_public": ("FlowStateForPrediction", "PatchTSTFMForPrediction"),
            "tsfm_public.toolkit.get_model": ("get_model",),
        },
        "timer_legacy": {"transformers": ("AutoModelForCausalLM",)},
        "transformers_recent": {"transformers": ("AutoModelForCausalLM",)},
        "tempo_legacy": {
            "huggingface_hub": ("hf_hub_download",),
            "omegaconf": ("OmegaConf",),
            "tempo.models.TEMPO": ("TEMPO",),
        },
        "toto2": {"toto2": ("Toto2Model",)},
        "kairos": {"tsfm.model.kairos": ("AutoModel",)},
        "tirex": {"tirex": ("load_model",)},
        "tabpfn_ts": {
            "huggingface_hub": ("hf_hub_download",),
            "tabpfn.errors": (
                "TabPFNHuggingFaceGatedRepoError",
                "TabPFNLicenseError",
            ),
            "tabpfn.model_loading": ("prepend_cache_path",),
            "tabpfn_time_series": ("TabPFNMode", "TabPFNTSPipeline"),
        },
    }
)
_RUNTIME_PROBE = """\
import importlib
import json
import sys

module_names = json.loads(sys.argv[1])
required_symbols = json.loads(sys.argv[2])
missing = []
modules = {}
for module_name in module_names:
    try:
        modules[module_name] = importlib.import_module(module_name)
    except BaseException:
        missing.append(module_name)
for module_name, symbol_names in required_symbols.items():
    module = modules.get(module_name)
    if module is None:
        continue
    for symbol_name in symbol_names:
        if not hasattr(module, symbol_name):
            missing.append(f"{module_name}:{symbol_name}")
print(json.dumps({
    "base_prefix": sys.base_prefix,
    "executable": sys.executable,
    "missing": missing,
    "prefix": sys.prefix,
}, sort_keys=True))
"""


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


def _isolates_system_site_packages(marker: Path) -> bool:
    try:
        lines = marker.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return False
    settings: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            settings[key.strip().lower()] = value.strip().lower()
    return settings.get("include-system-site-packages") == "false"


@dataclass(frozen=True)
class TSFMDeployment:
    """Validated commands and manifest IDs enabled by one local deployment."""

    commands: Mapping[str, WorkerCommand]
    enabled_manifest_ids: frozenset[str]
    acknowledged_licenses: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "commands", MappingProxyType(dict(self.commands)))

    def validate_runtime(
        self,
        environment_keys: Sequence[str] | None = None,
        *,
        parent_environment: Mapping[str, str] | None = None,
    ) -> None:
        """Fail fast when a configured worker interpreter is incomplete."""

        selected = tuple(self.commands) if environment_keys is None else tuple(
            environment_keys
        )
        unknown = set(selected) - set(self.commands)
        if unknown:
            raise ValueError(f"unknown configured worker environments: {sorted(unknown)!r}")
        source_environment = os.environ if parent_environment is None else parent_environment
        probe_environment = dict(controlled_worker_environment(source_environment))
        for environment in selected:
            command = self.commands[environment]
            interpreter = Path(command.argv[0])
            logical_root = interpreter.parent.parent
            marker = logical_root / "pyvenv.cfg"
            if not marker.is_file():
                raise ValueError(
                    f"worker environment {environment!r} requires an explicit virtual environment"
                )
            if not _isolates_system_site_packages(marker):
                raise ValueError(
                    f"worker environment {environment!r} must isolate system site-packages"
                )
            try:
                completed = subprocess.run(
                    [
                        str(interpreter),
                        "-I",
                        "-c",
                        _RUNTIME_PROBE,
                        json.dumps((*_RUNTIME_IMPORTS[environment], _WORKER_MODULE)),
                        json.dumps(_RUNTIME_SYMBOLS[environment]),
                    ],
                    shell=False,
                    env=probe_environment,
                    capture_output=True,
                    text=True,
                    timeout=60.0,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                raise ValueError(
                    f"worker environment {environment!r} runtime probe failed"
                ) from None
            try:
                result = json.loads(completed.stdout)
            except (json.JSONDecodeError, TypeError):
                result = None
            if (
                completed.returncode != 0
                or not isinstance(result, Mapping)
                or not isinstance(result.get("prefix"), str)
                or not isinstance(result.get("missing"), list)
                or not all(isinstance(name, str) for name in result["missing"])
            ):
                raise ValueError(
                    f"worker environment {environment!r} runtime probe failed"
                )
            reported_prefix = Path(result["prefix"])
            if logical_root.resolve() != reported_prefix.resolve():
                raise ValueError(
                    f"worker environment {environment!r} virtual environment identity changed"
                )
            missing = tuple(result["missing"])
            if missing:
                raise ValueError(
                    f"worker environment {environment!r} is missing reviewed dependencies: "
                    f"{', '.join(missing)}"
                )

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
