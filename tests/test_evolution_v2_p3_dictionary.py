from __future__ import annotations

from dataclasses import replace

import pytest

from evolving_loop.v2.cooperative.numerical_dictionary import (
    DictionarySelectorGenomeV2,
    build_p3_numerical_dictionary,
    materialize_selector_pair,
    mutate_selector_genome,
    seed_selector_genome,
)
from evolving_loop.v2.cooperative.adapters import (
    CooperativeArtifactCatalog,
    NumericalCoordinateAdapter,
)
from evolving_loop.package_numerical_supply import (
    build_package_registry,
    parse_numerical_supply_release,
)
from evolving_loop.v2.numerical_qd.adapters import (
    FrozenNumericalArtifactsV2,
    frozen_local_evidence_references,
    import_numerical_seed,
    validate_frozen_local_evidence,
)
from evolving_loop.v2.numerical_qd.artifacts import artifact_bytes
from tests.test_package_numerical_supply import (
    _alternative,
    _package_for_task,
    _ranked,
    _registry_tasks,
    _supply_release,
)


def _frozen_pair(candidate_id: str | None = None) -> FrozenNumericalArtifactsV2:
    families = {
        "seasonal_naive": "statistical",
        "toto_2_0": "tsfm",
        "weighted_pair": "combined",
        "atlas_70_30": "atlas_overlay",
    }
    alternatives = (
        ()
        if candidate_id is None
        else (_alternative(candidate_id, families[candidate_id]),)
    )
    payload = _supply_release(alternatives=alternatives).to_payload()
    payload["schema_version"] = 2
    release = parse_numerical_supply_release(payload)
    tasks = _registry_tasks()
    registry = build_package_registry(
        tasks,
        release,
        lambda task, supplied: _package_for_task(task, supplied),
    )
    envelope = import_numerical_seed(release, registry, tasks=tasks).envelope
    return FrozenNumericalArtifactsV2(release, registry, envelope, ())


def _wide_dictionary_pair() -> FrozenNumericalArtifactsV2:
    candidates = (
        ("a_stat", "statistical"),
        ("b_stat", "statistical"),
        ("c_stat", "statistical"),
        ("d_tsfm", "tsfm"),
        ("e_tsfm", "tsfm"),
        ("f_combined", "combined"),
        ("g_combined", "combined"),
        ("h_stat", "statistical"),
        ("i_tsfm", "tsfm"),
    )
    payload = _supply_release(alternatives=()).to_payload()
    payload["schema_version"] = 2
    payload["alternatives"] = [
        _alternative(name, family).to_payload() for name, family in candidates
    ]
    release = parse_numerical_supply_release(payload)
    tasks = _registry_tasks()

    def package(task, supplied):
        base = _package_for_task(task, supplied)
        ranked = (base.protected_baseline,) + tuple(
            replace(
                _ranked(name, family, (float(index + 1),) * 2),
                rank=index + 2,
            )
            for index, (name, family) in enumerate(candidates)
        )
        names = tuple(item.name for item in ranked)
        return replace(
            base,
            active_candidate_names=names,
            candidate_diagnostics={item.name: item.diagnostics for item in ranked},
            selection_decision=replace(
                base.selection_decision,
                considered_candidates=names,
            ),
            ranked_alternatives=ranked,
        )

    registry = build_package_registry(tasks, release, package)
    envelope = import_numerical_seed(release, registry, tasks=tasks).envelope
    return FrozenNumericalArtifactsV2(release, registry, envelope, ())


def test_selector_genome_round_trips_as_canonical_value():
    seed = seed_selector_genome()

    assert DictionarySelectorGenomeV2.from_payload(seed.to_payload()) == seed
    assert seed.target_candidates == 8
    assert seed.parent_selector_sha256 is None
    assert len(seed.fingerprint()) == 64


def test_selector_genome_rejects_unknown_fields():
    payload = seed_selector_genome().to_payload()
    payload["future_score"] = 0.1

    with pytest.raises(ValueError, match="exact schema"):
        DictionarySelectorGenomeV2.from_payload(payload)


def test_selector_mutation_changes_one_owned_policy_coordinate():
    parent = seed_selector_genome()
    child = mutate_selector_genome(parent, 0)

    changed = {
        key
        for key in parent.to_payload()
        if parent.to_payload()[key] != child.to_payload()[key]
    }
    assert changed == {
        "generation",
        "parent_selector_sha256",
        "morphology_weight",
    }
    assert child.parent_selector_sha256 == parent.fingerprint()


@pytest.mark.parametrize(
    "change",
    (
        {"target_candidates": 5},
        {"target_candidates": 9},
        {"mean_error_weight": -0.1},
        {"p90_error_weight": float("inf")},
    ),
)
def test_selector_genome_rejects_out_of_bounds_policy(change):
    with pytest.raises(ValueError):
        replace(seed_selector_genome(), **change)


def test_dictionary_closure_unions_members_instead_of_preserving_p2_shortlists():
    tasks = _registry_tasks()
    closure = build_p3_numerical_dictionary(
        (_frozen_pair("seasonal_naive"), _frozen_pair("weighted_pair")),
        tasks,
    )

    assert closure.candidate_names == (
        "safe_anchor",
        "seasonal_naive",
        "weighted_pair",
    )
    assert set(closure.package_inputs_for(tasks[0])) == {
        "safe_anchor",
        "seasonal_naive",
        "weighted_pair",
    }
    assert closure.public_test_accessed is False


def test_dictionary_closure_rejects_anchor_forecast_drift():
    tasks = _registry_tasks()
    original = _frozen_pair("seasonal_naive")

    def changed_anchor_package(task, supplied):
        package = _package_for_task(task, supplied)
        anchor = replace(package.protected_baseline, forecast=(123.0, 123.0))
        ranked = (anchor, *package.ranked_alternatives[1:])
        selection = replace(
            package.selection_decision,
            forecast=anchor.forecast,
        )
        return replace(
            package,
            selection_decision=selection,
            final_forecast=anchor.forecast,
            protected_baseline=anchor,
            ranked_alternatives=ranked,
        )

    changed_registry = build_package_registry(
        tasks,
        original.release,
        changed_anchor_package,
    )
    changed = FrozenNumericalArtifactsV2(
        original.release,
        changed_registry,
        import_numerical_seed(
            original.release, changed_registry, tasks=tasks
        ).envelope,
        (),
    )

    with pytest.raises(ValueError, match="Anchor"):
        build_p3_numerical_dictionary((original, changed), tasks)


def test_different_selector_genomes_change_shortlists_from_one_dictionary():
    tasks = _registry_tasks()
    closure = build_p3_numerical_dictionary((_wide_dictionary_pair(),), tasks)
    no_diversity = replace(
        seed_selector_genome(),
        target_candidates=6,
        family_diversity_weight=0.0,
    )
    diverse = replace(no_diversity, family_diversity_weight=10.0)

    first = materialize_selector_pair(closure, no_diversity, tasks)
    second = materialize_selector_pair(closure, diverse, tasks)

    assert first.release.source_fingerprints["p3_dictionary"] == closure.fingerprint()
    assert (
        first.release.source_fingerprints["p3_selector"]
        != second.release.source_fingerprints["p3_selector"]
    )
    assert (
        first.registry.package_for(tasks[0]).active_candidate_names
        != second.registry.package_for(tasks[0]).active_candidate_names
    )


def test_materialized_selector_pair_obeys_anchor_and_cardinality_bounds():
    tasks = _registry_tasks()
    closure = build_p3_numerical_dictionary((_wide_dictionary_pair(),), tasks)

    pair = materialize_selector_pair(closure, seed_selector_genome(), tasks)
    package = pair.registry.package_for(tasks[0])

    assert len(package.active_candidate_names) <= 8
    anchor = package.protected_baseline.name
    assert package.active_candidate_names[0] == anchor
    assert anchor in package.selection_decision.selected
    assert len(package.selection_decision.selected) <= 3
    assert package.selection_decision.weights[
        package.selection_decision.selected.index(anchor)
    ] >= 0.5
    assert pair.selected_genome_sha256s == (seed_selector_genome().fingerprint(),)


def test_selector_pair_persists_schema2_dictionary_genome_and_local_evidence():
    tasks = _registry_tasks()
    closure = build_p3_numerical_dictionary((_wide_dictionary_pair(),), tasks)
    selector = seed_selector_genome()

    pair = materialize_selector_pair(closure, selector, tasks)

    assert pair.envelope.schema_version == 2
    assert closure.fingerprint() in pair.support_objects
    assert selector.fingerprint() in pair.support_objects
    evidence_refs = frozen_local_evidence_references(pair.envelope)
    raw = {
        sha: artifact_bytes(kind, pair.support_objects[sha])
        for sha, kind in evidence_refs.items()
    }
    validate_frozen_local_evidence(
        pair.release,
        pair.envelope,
        artifact_bytes_by_sha=raw,
        tasks=tasks,
    )

    written = {}
    catalog = CooperativeArtifactCatalog(written.setdefault)
    catalog.add_numerical(pair)
    assert set(pair.support_objects) <= set(written)


def test_fresh_cooperative_catalog_load_accepts_only_bound_legacy_evidence(tmp_path):
    from evolving_loop.retrieval_agent.policy import RetrievalGenome
    from evolving_loop.v2.cooperative import DecisionModuleV2, RetrievalModuleV2
    from evolving_loop.v2.cooperative.runner import (
        _load_catalog_and_cache,
        _write_legacy_object,
        _write_object,
    )

    tasks = _registry_tasks()
    dictionary = build_p3_numerical_dictionary((_wide_dictionary_pair(),), tasks)
    pair = materialize_selector_pair(dictionary, seed_selector_genome(), tasks)
    retrieval = RetrievalModuleV2(
        1, "a" * 64, RetrievalGenome.seed().to_payload(), ()
    )
    decision = DecisionModuleV2(1, "seed prompt", (), True, 2, "last")
    root = tmp_path / "fresh-p3"
    catalog = CooperativeArtifactCatalog(
        lambda identity, payload: _write_object(root, identity, payload),
        lambda identity, payload: _write_legacy_object(root, identity, payload),
    )

    cache = _load_catalog_and_cache(
        root,
        catalog,
        pair,
        retrieval,
        decision,
        {"numerical": NumericalCoordinateAdapter(())},
    )

    assert cache == {}
    assert catalog.resolve_numerical(
        pair.release.fingerprint, pair.registry.fingerprint
    ) is pair


def test_sealed_numerical_pair_load_restores_authenticated_support_closure(tmp_path):
    from evolving_loop.v2.cooperative.runner import (
        _write_legacy_object,
        _write_object,
    )
    from evolving_loop.v2.real.bridges import _load_numerical_pair, _pair_row

    tasks = _registry_tasks()
    dictionary = build_p3_numerical_dictionary((_wide_dictionary_pair(),), tasks)
    pair = materialize_selector_pair(dictionary, seed_selector_genome(), tasks)
    root = tmp_path / "sealed-p3"
    catalog = CooperativeArtifactCatalog(
        lambda identity, payload: _write_object(root, identity, payload),
        lambda identity, payload: _write_legacy_object(root, identity, payload),
    )
    catalog.add_numerical(pair)

    loaded = _load_numerical_pair(root, _pair_row(pair), tasks)

    assert set(loaded.support_objects) == set(pair.support_objects)
    CooperativeArtifactCatalog(lambda _identity, _payload: None).add_numerical(
        loaded
    )


def test_numerical_coordinate_evolves_selector_instead_of_p2_pair():
    tasks = _registry_tasks()
    closure = build_p3_numerical_dictionary((_wide_dictionary_pair(),), tasks)
    seed_genome = seed_selector_genome()
    seed_pair = materialize_selector_pair(closure, seed_genome, tasks)
    adapter = NumericalCoordinateAdapter.for_dictionary(
        closure,
        tasks,
        seed_genome,
    )

    child = adapter.propose(seed_pair, step=0)

    assert adapter.mode == "p3_dictionary"
    assert child is not None
    assert child.release.source_fingerprints["p3_dictionary"] == closure.fingerprint()
    assert (
        child.release.source_fingerprints["p3_selector"]
        == mutate_selector_genome(seed_genome, 0).fingerprint()
    )
    assert child not in (_wide_dictionary_pair(),)


def test_dictionary_coordinate_precomputes_resume_safe_selector_space():
    tasks = _registry_tasks()
    closure = build_p3_numerical_dictionary((_wide_dictionary_pair(),), tasks)
    seed_genome = seed_selector_genome()
    adapter = NumericalCoordinateAdapter.for_dictionary(
        closure,
        tasks,
        seed_genome,
        max_steps=2,
    )

    assert len(adapter.materialized_pairs) >= 2
    assert all(
        pair.release.source_fingerprints["p3_dictionary"] == closure.fingerprint()
        for pair in adapter.materialized_pairs
    )
    assert len(adapter.proposal_pairs) == len(adapter.materialized_pairs) - 1
    assert all(
        pair.release.source_fingerprints["p3_selector"] != seed_genome.fingerprint()
        for pair in adapter.proposal_pairs
    )


def test_real_bridge_seeds_p3_from_dictionary_selector(tmp_path, monkeypatch):
    from evolving_loop.v2.real import bridges
    from tests.test_evolution_v2_real_cooperative import (
        _host,
        _p2_pair,
        _real_config_payload,
        _tasks_100,
    )

    tasks = _tasks_100()
    dictionary = build_p3_numerical_dictionary((_p2_pair(tasks),), tasks)
    host = _host(tasks, object())

    class Observed(RuntimeError):
        pass

    def observe(_output, _config, seed, _projected, adapters, **_kwargs):
        numerical = seed["numerical"]
        assert numerical.release.source_fingerprints["p3_dictionary"] == dictionary.fingerprint()
        assert numerical.release.source_fingerprints["p3_selector"] == seed_selector_genome().fingerprint()
        assert adapters["numerical"].mode == "p3_dictionary"
        raise Observed

    monkeypatch.setattr(bridges, "run_cooperative_evolution", observe)

    with pytest.raises(Observed):
        bridges.run_real_cooperative(
            dictionary=dictionary,
            host=host,
            config_payload=_real_config_payload(),
            output_dir=tmp_path / "p3",
        )
