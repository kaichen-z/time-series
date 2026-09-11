from __future__ import annotations

from dataclasses import replace

import pytest

from evolving_loop.package_numerical_supply import NumericalSupplyRelease
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.v2.contracts import fingerprint_payload
from evolving_loop.v2.cooperative.adapters import (
    CooperativeArtifactCatalog,
    DecisionCoordinateAdapter,
    NumericalCoordinateAdapter,
    RetrievalCoordinateAdapter,
)
from evolving_loop.v2.cooperative.contracts import DecisionModuleV2, RetrievalModuleV2
from evolving_loop.v2.numerical_qd.adapters import FrozenNumericalArtifactsV2
from evolving_loop.v2.numerical_qd.contracts import FrozenNumericalRegistryEnvelopeV2
from tests.build_evolution_v2_numerical_fixture import _seed_supply


def _frozen_pair(release: NumericalSupplyRelease) -> FrozenNumericalArtifactsV2:
    manifest = {
        "schema_version": 1,
        "release_sha256": release.fingerprint,
        "task_ids": ["fixture"],
        "entries": [],
    }
    registry = object.__new__(FrozenNumericalPackageRegistry)
    registry._release_sha256 = release.fingerprint
    registry._packages = {}
    registry._manifest = manifest
    registry.fingerprint = fingerprint_payload(manifest)

    envelope = object.__new__(FrozenNumericalRegistryEnvelopeV2)
    object.__setattr__(envelope, "schema_version", 1)
    object.__setattr__(envelope, "release_sha256", release.fingerprint)
    object.__setattr__(envelope, "registry_sha256", registry.fingerprint)
    object.__setattr__(envelope, "entries", {})
    object.__setattr__(envelope, "package_sha256s", ())
    object.__setattr__(envelope, "packages", {})
    return FrozenNumericalArtifactsV2(release, registry, envelope, ())


@pytest.fixture
def seed_pair() -> FrozenNumericalArtifactsV2:
    return _frozen_pair(_seed_supply())


@pytest.fixture
def alternate_pair(seed_pair) -> FrozenNumericalArtifactsV2:
    payload = seed_pair.release.to_payload()
    release = NumericalSupplyRelease(
        schema_version=1,
        version="n001",
        parent_sha256=seed_pair.release.fingerprint,
        anchor_release_payload=payload["anchor_release_payload"],
        alternatives=(),
        atlas_release_sha256=None,
        source_fingerprints=payload["source_fingerprints"],
        runtime_fingerprints=payload["runtime_fingerprints"],
    )
    return _frozen_pair(release)


@pytest.fixture
def retrieval_module() -> RetrievalModuleV2:
    return RetrievalModuleV2(1, "a" * 64, RetrievalGenome.seed().to_payload(), ())


@pytest.fixture
def decision_module() -> DecisionModuleV2:
    return DecisionModuleV2(1, "seed prompt", (), True, 2, "last")


def test_numerical_adapter_returns_a_different_frozen_pair(seed_pair, alternate_pair):
    adapter = NumericalCoordinateAdapter((alternate_pair,))
    assert (
        adapter.propose(seed_pair).release.fingerprint
        == alternate_pair.release.fingerprint
    )


def test_numerical_adapter_is_canonical_and_exhausts(seed_pair, alternate_pair):
    payload = alternate_pair.release.to_payload()
    later = _frozen_pair(
        NumericalSupplyRelease(
            schema_version=1,
            version="n002",
            parent_sha256=alternate_pair.release.fingerprint,
            anchor_release_payload=payload["anchor_release_payload"],
            alternatives=(),
            atlas_release_sha256=None,
            source_fingerprints=payload["source_fingerprints"],
            runtime_fingerprints=payload["runtime_fingerprints"],
        )
    )
    adapter = NumericalCoordinateAdapter((later, alternate_pair, seed_pair))
    expected = min(
        (alternate_pair, later),
        key=lambda pair: (pair.release.fingerprint, pair.registry.fingerprint),
    )
    assert adapter.propose(seed_pair) == expected
    assert NumericalCoordinateAdapter((seed_pair,)).propose(seed_pair) is None


def test_retrieval_adapter_changes_one_typed_field(retrieval_module):
    child = RetrievalCoordinateAdapter().propose(retrieval_module, step=0)
    parent_genome = retrieval_module.genome
    child_genome = child.genome
    changed = [
        name
        for name in child_genome.to_payload()
        if child_genome.to_payload()[name] != parent_genome.to_payload()[name]
    ]
    assert changed == ["version", "parent", "round1_strategy"]
    assert child_genome.require_counterevidence_search is True
    assert child_genome.require_target_match is True
    assert child_genome.require_temporal_overlap is True
    assert child.skills_payload == retrieval_module.skills_payload
    assert child.source_release_sha256 == retrieval_module.source_release_sha256


@pytest.mark.parametrize(
    ("step", "field"),
    (
        (1, "round2_strategy"),
        (2, "second_round_trigger"),
        (3, "max_selected_documents"),
    ),
)
def test_retrieval_adapter_cycles_each_owned_field(retrieval_module, step, field):
    child = RetrievalCoordinateAdapter().propose(retrieval_module, step=step)
    ignored = {"version", "parent", field}
    parent = retrieval_module.genome.to_payload()
    changed = {
        name for name, value in child.genome.to_payload().items() if value != parent[name]
    }
    assert changed == ignored


def test_retrieval_budget_wraps_inside_the_existing_bound(retrieval_module):
    genome = replace(retrieval_module.genome, max_selected_documents=20)
    bounded = replace(retrieval_module, genome_payload=genome.to_payload())
    child = RetrievalCoordinateAdapter().propose(bounded, step=3)
    assert child.genome.max_selected_documents == 1


def test_decision_adapter_changes_only_the_prompt(decision_module):
    child = DecisionCoordinateAdapter(("prompt variant",)).propose(
        decision_module, step=0
    )
    assert child.prompt == "prompt variant"
    assert child.skills == decision_module.skills
    assert (
        child.enable_evidence_adjustments
        == decision_module.enable_evidence_adjustments
    )
    assert child.max_evidence_adjustments == decision_module.max_evidence_adjustments
    assert child.aggregation == decision_module.aggregation


def test_decision_adapter_skips_the_parent_prompt(decision_module):
    adapter = DecisionCoordinateAdapter((decision_module.prompt, "other prompt"))
    assert adapter.propose(decision_module, step=0).prompt == "other prompt"


@pytest.mark.parametrize(
    ("step", "field"),
    ((0, "max_evidence_adjustments"), (1, "aggregation")),
)
def test_optional_decision_settings_cycle_changes_one_owned_field(
    decision_module, step, field
):
    adapter = DecisionCoordinateAdapter((), settings_cycle=True)
    child = adapter.propose(decision_module, step=step)
    changed = {
        name
        for name in decision_module.to_payload()
        if child.to_payload()[name] != decision_module.to_payload()[name]
    }
    assert changed == {field}


def test_decision_adjustment_setting_wraps_inside_bound():
    parent = DecisionModuleV2(1, "seed prompt", (), True, 3, "last")
    child = DecisionCoordinateAdapter((), settings_cycle=True).propose(parent, step=0)
    assert child.max_evidence_adjustments == 0


def test_catalog_round_trips_exact_typed_artifacts(
    seed_pair, retrieval_module, decision_module
):
    writes: list[tuple[str, object]] = []
    catalog = CooperativeArtifactCatalog(
        lambda identity, payload: writes.append((identity, payload))
    )

    assert catalog.add_numerical(seed_pair) == (
        seed_pair.release.fingerprint,
        seed_pair.registry.fingerprint,
    )
    retrieval_sha = catalog.add_retrieval(retrieval_module)
    decision_sha = catalog.add_decision(decision_module)

    assert catalog.resolve_numerical(*catalog.add_numerical(seed_pair)) is seed_pair
    assert catalog.resolve_retrieval(retrieval_sha) is retrieval_module
    assert catalog.resolve_decision(decision_sha) is decision_module
    assert writes
    assert all(
        not isinstance(
            payload,
            (FrozenNumericalArtifactsV2, RetrievalModuleV2, DecisionModuleV2),
        )
        for _, payload in writes
    )
    assert all(fingerprint_payload(payload) == identity for identity, payload in writes)


def test_catalog_rejects_missing_or_mismatched_artifacts(
    seed_pair, retrieval_module
):
    catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
    with pytest.raises(ValueError, match="Numerical"):
        catalog.resolve_numerical("1" * 64, "2" * 64)
    with pytest.raises(ValueError, match="Retrieval"):
        catalog.resolve_retrieval("3" * 64)
    with pytest.raises(ValueError, match="Decision"):
        catalog.resolve_decision("4" * 64)

    bad_registry = object.__new__(FrozenNumericalPackageRegistry)
    bad_registry._release_sha256 = "5" * 64
    bad_registry.fingerprint = seed_pair.registry.fingerprint
    bad_registry._manifest = seed_pair.registry.manifest
    bad_registry._packages = {}
    bad_pair = object.__new__(FrozenNumericalArtifactsV2)
    object.__setattr__(bad_pair, "release", seed_pair.release)
    object.__setattr__(bad_pair, "registry", bad_registry)
    object.__setattr__(bad_pair, "envelope", seed_pair.envelope)
    object.__setattr__(bad_pair, "selected_genome_sha256s", ())
    with pytest.raises(ValueError, match="release"):
        catalog.add_numerical(bad_pair)

    payload = retrieval_module.to_payload()
    payload["source_release_sha256"] = "6" * 64
    impostor = RetrievalModuleV2.from_payload(payload)
    catalog.add_retrieval(retrieval_module)
    with pytest.raises(ValueError, match="Retrieval"):
        catalog.resolve_retrieval(impostor.fingerprint())
