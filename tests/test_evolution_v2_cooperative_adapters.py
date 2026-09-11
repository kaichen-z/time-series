from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from evolving_loop.package_numerical_supply import (
    NumericalSupplyRelease,
    build_package_registry,
)
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.v2.contracts import fingerprint_payload
from evolving_loop.v2.cooperative.adapters import (
    ROUND1_NEXT,
    ROUND2_NEXT,
    TRIGGER_NEXT,
    CooperativeArtifactCatalog,
    DecisionCoordinateAdapter,
    NumericalCoordinateAdapter,
    RetrievalCoordinateAdapter,
)
from evolving_loop.v2.cooperative.contracts import DecisionModuleV2, RetrievalModuleV2
from evolving_loop.v2.numerical_qd.adapters import (
    FrozenNumericalArtifactsV2,
    import_numerical_seed,
)
from evolving_loop.v2.numerical_qd.contracts import FrozenNumericalRegistryEnvelopeV2
from tests.test_package_numerical_supply import (
    _alternative,
    _package_for_task,
    _registry_tasks,
    _supply_release,
)


def _frozen_pair(
    release: NumericalSupplyRelease,
) -> FrozenNumericalArtifactsV2:
    tasks = _registry_tasks()
    registry = build_package_registry(
        tasks,
        release,
        lambda task, supplied: _package_for_task(task, supplied),
    )
    envelope = import_numerical_seed(release, registry, tasks=tasks).envelope
    return FrozenNumericalArtifactsV2(release, registry, envelope, ())


@pytest.fixture(scope="module")
def seed_pair() -> FrozenNumericalArtifactsV2:
    return _frozen_pair(_supply_release(alternatives=()))


@pytest.fixture(scope="module")
def alternate_pair() -> FrozenNumericalArtifactsV2:
    return _frozen_pair(_supply_release())


@pytest.fixture
def retrieval_module() -> RetrievalModuleV2:
    return RetrievalModuleV2(1, "a" * 64, RetrievalGenome.seed().to_payload(), ())


@pytest.fixture
def decision_module() -> DecisionModuleV2:
    return DecisionModuleV2(1, "seed prompt", (), True, 2, "last")


def _strict_writer(writes: list[tuple[str, object]]):
    def write(identity: str, payload: object) -> None:
        if fingerprint_payload(payload) != identity:
            raise ValueError("strict object identity mismatch")
        writes.append((identity, payload))

    return write


def test_numerical_adapter_returns_a_different_frozen_pair(seed_pair, alternate_pair):
    adapter = NumericalCoordinateAdapter((alternate_pair,))
    assert (
        adapter.propose(seed_pair).release.fingerprint
        == alternate_pair.release.fingerprint
    )


def test_numerical_adapter_is_canonical_and_exhausts(seed_pair, alternate_pair):
    later = _frozen_pair(
        _supply_release(
            alternatives=(
                _alternative("seasonal_naive", "statistical"),
                _alternative("weighted_pair", "combined"),
            )
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
    parent = retrieval_module.genome.to_payload()
    changed = {
        name for name, value in child.genome.to_payload().items() if value != parent[name]
    }
    assert changed == {"version", "parent", field}


def test_retrieval_budget_wraps_inside_the_existing_bound(retrieval_module):
    genome = replace(retrieval_module.genome, max_selected_documents=20)
    bounded = replace(retrieval_module, genome_payload=genome.to_payload())
    child = RetrievalCoordinateAdapter().propose(bounded, step=3)
    assert child.genome.max_selected_documents == 1


def test_retrieval_cycles_are_read_only():
    for cycle in (ROUND1_NEXT, ROUND2_NEXT, TRIGGER_NEXT):
        with pytest.raises(TypeError):
            cycle[next(iter(cycle))] = next(iter(cycle))


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


def test_catalog_separates_legacy_authority_from_v2_object_identity(seed_pair):
    writes: list[tuple[str, object]] = []
    catalog = CooperativeArtifactCatalog(_strict_writer(writes))

    legacy_pair = catalog.add_numerical(seed_pair)

    assert legacy_pair == (
        seed_pair.release.fingerprint,
        seed_pair.registry.fingerprint,
    )
    assert fingerprint_payload(seed_pair.release.to_payload()) != legacy_pair[0]
    assert fingerprint_payload(seed_pair.registry.manifest) != legacy_pair[1]
    assert catalog.resolve_numerical(*legacy_pair) is seed_pair
    assert len(writes) == 3


def test_catalog_round_trips_exact_typed_artifacts(
    seed_pair, retrieval_module, decision_module
):
    writes: list[tuple[str, object]] = []
    catalog = CooperativeArtifactCatalog(_strict_writer(writes))

    numerical_pair = catalog.add_numerical(seed_pair)
    retrieval_sha = catalog.add_retrieval(retrieval_module)
    decision_sha = catalog.add_decision(decision_module)

    assert catalog.resolve_numerical(*numerical_pair) is seed_pair
    assert catalog.resolve_retrieval(retrieval_sha) is retrieval_module
    assert catalog.resolve_decision(decision_sha) is decision_module
    assert all(
        not isinstance(
            payload,
            (FrozenNumericalArtifactsV2, RetrievalModuleV2, DecisionModuleV2),
        )
        for _, payload in writes
    )


def test_catalog_rejects_conflicting_duplicate_pair(seed_pair):
    catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
    catalog.add_numerical(seed_pair)
    conflict = replace(seed_pair, selected_genome_sha256s=("a" * 64,))

    with pytest.raises(ValueError, match="conflicting Numerical"):
        catalog.add_numerical(conflict)

    assert catalog.resolve_numerical(
        seed_pair.release.fingerprint, seed_pair.registry.fingerprint
    ) is seed_pair


def test_catalog_rejects_same_pair_with_a_different_envelope(seed_pair):
    catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
    catalog.add_numerical(seed_pair)
    conflict = copy.copy(seed_pair)
    envelope = copy.copy(seed_pair.envelope)
    object.__setattr__(
        envelope,
        "package_sha256s",
        tuple(reversed(envelope.package_sha256s)),
    )
    object.__setattr__(conflict, "envelope", envelope)

    with pytest.raises(ValueError, match="conflicting Numerical"):
        catalog.add_numerical(conflict)


def test_catalog_rejects_envelope_release_mismatch(seed_pair):
    tampered = copy.copy(seed_pair)
    envelope = copy.copy(seed_pair.envelope)
    object.__setattr__(envelope, "release_sha256", "f" * 64)
    object.__setattr__(tampered, "envelope", envelope)

    with pytest.raises(ValueError, match="release/envelope"):
        CooperativeArtifactCatalog(lambda _identity, _payload: None).add_numerical(
            tampered
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("release", object()),
        ("registry", object()),
        ("envelope", object()),
    ),
)
def test_catalog_requires_exact_internal_numerical_types(
    seed_pair, field, replacement
):
    tampered = copy.copy(seed_pair)
    object.__setattr__(tampered, field, replacement)
    with pytest.raises(ValueError, match="exact Numerical"):
        CooperativeArtifactCatalog(lambda _identity, _payload: None).add_numerical(
            tampered
        )


def test_catalog_rejects_missing_artifacts():
    catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
    with pytest.raises(ValueError, match="Numerical"):
        catalog.resolve_numerical("1" * 64, "2" * 64)
    with pytest.raises(ValueError, match="Retrieval"):
        catalog.resolve_retrieval("3" * 64)
    with pytest.raises(ValueError, match="Decision"):
        catalog.resolve_decision("4" * 64)
