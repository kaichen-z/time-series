"""Deterministic bindings from foundation catalog cards to TSFM runtimes."""

from __future__ import annotations

from typing import Mapping

from .dictionary import MethodCandidate, MethodDefinition
from .tsfm.manifests import ManifestRegistry


TSFM_IMPLEMENTATION_KIND = "tsfm_checkpoint"
FOUNDATION_UNAVAILABLE_PROVIDER = "foundation_unavailable"


class FoundationCandidateFactory:
    """Create runtime candidates directly from preserved catalog metadata."""

    def __init__(self, manifests: ManifestRegistry | None = None) -> None:
        self._manifests = (
            manifests if manifests is not None else ManifestRegistry.load_default()
        )

    def create(self, method: MethodDefinition) -> MethodCandidate:
        if method.family != "foundation":
            raise ValueError("foundation candidate factory requires a foundation method")

        raw_metadata = method.implementation_spec.get("foundation_metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
        manifest = self._manifests.require(method.method_id)
        model_id = manifest.checkpoint
        release_version = metadata.get("release_version", "")
        implementation: dict[str, object] = {
            **manifest.candidate_binding(),
            "release": release_version,
            "context_limit": metadata.get("context_length"),
            "prediction_limit": metadata.get("prediction_length"),
        }
        if manifest.status == "experimental_unverified":
            provider = "tsfm_worker"
        elif manifest.status == "direct":
            provider = manifest.adapter
        else:
            provider = FOUNDATION_UNAVAILABLE_PROVIDER
            implementation["unavailable_reason"] = manifest.reason_code
        return MethodCandidate(
            method_id=method.method_id,
            provider=provider,
            implementation_kind=TSFM_IMPLEMENTATION_KIND,
            implementation=implementation,
        )
