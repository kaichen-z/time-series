"""Typed coordinate adapters for cooperative Evolution V2."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType

from evolving_loop.package_numerical_supply import NumericalSupplyRelease
from evolving_loop.package_registry import FrozenNumericalPackageRegistry, _digest

from ..contracts import fingerprint_payload, require_sha256
from ..numerical_qd.adapters import FrozenNumericalArtifactsV2
from ..numerical_qd.contracts import FrozenNumericalRegistryEnvelopeV2
from .contracts import DecisionModuleV2, RetrievalModuleV2


ROUND1_NEXT = MappingProxyType({
    "timeline_first": "entity_first",
    "entity_first": "contrastive",
    "contrastive": "timeline_first",
})
ROUND2_NEXT = MappingProxyType({
    "counterevidence_first": "gap_first",
    "gap_first": "causal_chain_first",
    "causal_chain_first": "counterevidence_first",
})
TRIGGER_NEXT = MappingProxyType({
    "on_named_gap": "on_incomplete_chain",
    "on_incomplete_chain": "always",
    "always": "never",
    "never": "on_named_gap",
})


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


class CooperativeArtifactCatalog:
    """Keep exact runtime objects beside their canonical persisted payloads."""

    def __init__(self, write_object: Callable[[str, object], object]) -> None:
        if not callable(write_object):
            raise ValueError("write_object must be callable")
        self._write_object = write_object
        self._numerical: dict[tuple[str, str], FrozenNumericalArtifactsV2] = {}
        self._retrieval: dict[str, RetrievalModuleV2] = {}
        self._decision: dict[str, DecisionModuleV2] = {}

    def add_numerical(
        self, artifacts: FrozenNumericalArtifactsV2
    ) -> tuple[str, str]:
        if type(artifacts) is not FrozenNumericalArtifactsV2:
            raise ValueError("Numerical artifact must be a frozen pair")
        if (
            type(artifacts.release) is not NumericalSupplyRelease
            or type(artifacts.registry) is not FrozenNumericalPackageRegistry
            or type(artifacts.envelope) is not FrozenNumericalRegistryEnvelopeV2
        ):
            raise ValueError("exact Numerical release, registry, and envelope required")
        selected = artifacts.selected_genome_sha256s
        if type(selected) is not tuple or selected != tuple(sorted(set(selected))):
            raise ValueError("exact Numerical selected Genome SHA tuple required")
        for identity in selected:
            require_sha256(identity, "selected Numerical Genome SHA")
        release_sha = artifacts.release.fingerprint
        registry_sha = artifacts.registry.fingerprint
        if release_sha != artifacts.registry.release_sha256:
            raise ValueError("Numerical release/registry mismatch")
        if artifacts.envelope.release_sha256 != release_sha:
            raise ValueError("Numerical release/envelope mismatch")
        if registry_sha != artifacts.envelope.registry_sha256:
            raise ValueError("Numerical registry/envelope mismatch")
        registry_payload = _plain_json(artifacts.registry.manifest)
        if _digest(registry_payload) != registry_sha:
            raise ValueError("Numerical registry content mismatch")
        key = (release_sha, registry_sha)
        existing = self._numerical.get(key)
        if existing is not None and (
            existing.envelope.fingerprint() != artifacts.envelope.fingerprint()
            or existing.selected_genome_sha256s != selected
        ):
            raise ValueError("conflicting Numerical artifact pair")
        release_payload = artifacts.release.to_payload()
        self._write_object(fingerprint_payload(release_payload), release_payload)
        self._write_object(fingerprint_payload(registry_payload), registry_payload)
        self._write_object(
            artifacts.envelope.fingerprint(), artifacts.envelope.to_payload()
        )
        if existing is None:
            self._numerical[key] = artifacts
        return release_sha, registry_sha

    def add_retrieval(self, artifact: RetrievalModuleV2) -> str:
        if type(artifact) is not RetrievalModuleV2:
            raise ValueError("Retrieval artifact must be a RetrievalModuleV2")
        identity = artifact.fingerprint()
        self._write_object(identity, artifact.to_payload())
        self._retrieval[identity] = artifact
        return identity

    def add_decision(self, artifact: DecisionModuleV2) -> str:
        if type(artifact) is not DecisionModuleV2:
            raise ValueError("Decision artifact must be a DecisionModuleV2")
        identity = artifact.fingerprint()
        self._write_object(identity, artifact.to_payload())
        self._decision[identity] = artifact
        return identity

    def resolve_numerical(
        self, release_sha256: str, registry_sha256: str
    ) -> FrozenNumericalArtifactsV2:
        require_sha256(release_sha256, "Numerical release SHA")
        require_sha256(registry_sha256, "Numerical registry SHA")
        try:
            artifact = self._numerical[(release_sha256, registry_sha256)]
        except KeyError as error:
            raise ValueError("Numerical artifact pair is missing") from error
        if (
            artifact.release.fingerprint != release_sha256
            or artifact.registry.fingerprint != registry_sha256
            or artifact.registry.release_sha256 != release_sha256
            or artifact.envelope.registry_sha256 != registry_sha256
        ):
            raise ValueError("Numerical artifact pair is mismatched")
        return artifact

    def resolve_retrieval(self, identity: str) -> RetrievalModuleV2:
        require_sha256(identity, "Retrieval artifact SHA")
        try:
            artifact = self._retrieval[identity]
        except KeyError as error:
            raise ValueError("Retrieval artifact is missing") from error
        if artifact.fingerprint() != identity:
            raise ValueError("Retrieval artifact is mismatched")
        return artifact

    def resolve_decision(self, identity: str) -> DecisionModuleV2:
        require_sha256(identity, "Decision artifact SHA")
        try:
            artifact = self._decision[identity]
        except KeyError as error:
            raise ValueError("Decision artifact is missing") from error
        if artifact.fingerprint() != identity:
            raise ValueError("Decision artifact is mismatched")
        return artifact


class NumericalCoordinateAdapter:
    """Select the first canonical frozen pair that differs from the Parent."""

    def __init__(self, alternatives: Sequence[FrozenNumericalArtifactsV2]) -> None:
        values = tuple(alternatives)
        if any(type(value) is not FrozenNumericalArtifactsV2 for value in values):
            raise ValueError("Numerical alternatives must be frozen pairs")
        self._alternatives = tuple(
            sorted(
                values,
                key=lambda value: (
                    value.release.fingerprint,
                    value.registry.fingerprint,
                ),
            )
        )

    def propose(
        self, parent: FrozenNumericalArtifactsV2
    ) -> FrozenNumericalArtifactsV2 | None:
        if type(parent) is not FrozenNumericalArtifactsV2:
            raise ValueError("Numerical Parent must be a frozen pair")
        parent_identity = (parent.release.fingerprint, parent.registry.fingerprint)
        return next(
            (
                value
                for value in self._alternatives
                if (value.release.fingerprint, value.registry.fingerprint)
                != parent_identity
            ),
            None,
        )


class RetrievalCoordinateAdapter:
    """Apply one deterministic, bounded mutation to a Retrieval Genome."""

    def propose(self, parent: RetrievalModuleV2, step: int) -> RetrievalModuleV2:
        if type(parent) is not RetrievalModuleV2:
            raise ValueError("Retrieval Parent must be a RetrievalModuleV2")
        if type(step) is not int or step < 0:
            raise ValueError("Retrieval step must be a non-negative integer")
        genome = parent.genome
        changes: dict[str, object] = {
            "version": f"v{int(genome.version[1:]) + 1:03d}",
            "parent": genome.version,
        }
        coordinate = step % 4
        if coordinate == 0:
            changes["round1_strategy"] = ROUND1_NEXT[genome.round1_strategy]
        elif coordinate == 1:
            changes["round2_strategy"] = ROUND2_NEXT[genome.round2_strategy]
        elif coordinate == 2:
            changes["second_round_trigger"] = TRIGGER_NEXT[genome.second_round_trigger]
        else:
            changes["max_selected_documents"] = (
                genome.max_selected_documents % 20
            ) + 1
        child = replace(genome, **changes)
        return RetrievalModuleV2(
            schema_version=1,
            source_release_sha256=parent.source_release_sha256,
            genome_payload=child.to_payload(),
            skills_payload=parent.skills_payload,
        )


class DecisionCoordinateAdapter:
    """Change one Decision-owned prompt or bounded setting."""

    def __init__(
        self,
        prompts: Sequence[str],
        *,
        settings_cycle: bool = False,
    ) -> None:
        values = tuple(prompts)
        if any(type(value) is not str or not value.strip() for value in values):
            raise ValueError("Decision prompts must be non-empty strings")
        if type(settings_cycle) is not bool:
            raise ValueError("settings_cycle must be a boolean")
        if not values and not settings_cycle:
            raise ValueError("Decision adapter requires prompts or settings")
        self._prompts = values
        self._settings_cycle = settings_cycle

    def propose(
        self, parent: DecisionModuleV2, step: int
    ) -> DecisionModuleV2 | None:
        if type(parent) is not DecisionModuleV2:
            raise ValueError("Decision Parent must be a DecisionModuleV2")
        if type(step) is not int or step < 0:
            raise ValueError("Decision step must be a non-negative integer")
        if self._settings_cycle:
            if step % 2 == 0:
                return replace(
                    parent,
                    max_evidence_adjustments=(parent.max_evidence_adjustments + 1) % 4,
                )
            return replace(
                parent,
                aggregation="mean" if parent.aggregation == "last" else "last",
            )
        index = step % len(self._prompts)
        prompt = self._prompts[index]
        if prompt == parent.prompt:
            prompt = self._prompts[(index + 1) % len(self._prompts)]
        return None if prompt == parent.prompt else replace(parent, prompt=prompt)


__all__ = [
    "CooperativeArtifactCatalog",
    "DecisionCoordinateAdapter",
    "NumericalCoordinateAdapter",
    "RetrievalCoordinateAdapter",
    "ROUND1_NEXT",
    "ROUND2_NEXT",
]
