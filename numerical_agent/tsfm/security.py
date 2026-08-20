"""Controlled worker environments and immutable secret redaction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import re
from types import MappingProxyType

from .protocol import WorkerResponse


# These credentials are consumed by the supported local checkpoint loaders.
WORKER_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "TABPFN_TOKEN",
    }
)

# Runtime/cache/device settings required by Python and local model libraries. No
# general application environment or arbitrary credentials cross this boundary.
SAFE_WORKER_ENV_NAMES = frozenset(
    {
        "ALL_PROXY",
        "CUDA_VISIBLE_DEVICES",
        "CURL_CA_BUNDLE",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_OFFLINE",
        "HIP_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "MPLCONFIGDIR",
        "NA_TSFM_DEVICE",
        "NO_PROXY",
        "PATH",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "REQUESTS_CA_BUNDLE",
        "ROCR_VISIBLE_DEVICES",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TORCH_HOME",
        "TRANSFORMERS_CACHE",
        "TRANSFORMERS_OFFLINE",
        "TZ",
        "XDG_CACHE_HOME",
    }
)

# Explicitly recognized parent credentials are snapshotted for redaction even
# when they are intentionally excluded from the child process.
KNOWN_SECRET_ENV_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ALL_PROXY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_OPENAI_API_KEY",
        "DATABASE_URL",
        "GITHUB_PAT",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GOOGLE_API_KEY",
        "HF_API_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NPM_TOKEN",
        "OPENAI_API_KEY",
        "PYPI_TOKEN",
        "SSH_PRIVATE_KEY",
        "TWINE_PASSWORD",
        *WORKER_CREDENTIAL_ENV_NAMES,
    }
)


def controlled_worker_environment(
    source: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
    """Snapshot only reviewed runtime settings and supported credentials."""

    environment = os.environ if source is None else source
    allowed = SAFE_WORKER_ENV_NAMES | WORKER_CREDENTIAL_ENV_NAMES
    return MappingProxyType(
        {
            name: value
            for name, value in environment.items()
            if name in allowed and isinstance(value, str)
        }
    )


@dataclass(frozen=True)
class SecretRedactor:
    """Redact a fixed snapshot of explicit credential values."""

    values: tuple[str, ...]

    @classmethod
    def from_environment(
        cls, source: Mapping[str, str] | None = None
    ) -> "SecretRedactor":
        environment = os.environ if source is None else source
        values = {
            value
            for name, value in environment.items()
            if name in KNOWN_SECRET_ENV_NAMES and isinstance(value, str) and value
        }
        return cls(tuple(sorted(values, key=lambda value: (-len(value), value))))

    def redact_text(self, value: object) -> str:
        text = str(value)
        if not self.values:
            return text
        pattern = re.compile("|".join(re.escape(secret) for secret in self.values))
        return pattern.sub("[REDACTED]", text)

    def sanitize_json(self, value: object) -> object:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, Mapping):
            return {
                self.redact_text(key): self.sanitize_json(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.sanitize_json(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.sanitize_json(item) for item in value)
        return value

    def sanitize_response(self, response: WorkerResponse) -> WorkerResponse:
        request_id = self.redact_text(response.request_id)
        if response.status == "success":
            metadata = self.sanitize_json(response.metadata)
            assert isinstance(metadata, Mapping)
            return WorkerResponse.success(request_id, response.values, metadata)
        return WorkerResponse.failure(
            request_id,
            response.status,
            self.redact_text(response.reason_code),
            self.redact_text(response.message),
        )
