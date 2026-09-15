"""Frozen evidence is verified from content bytes without numerical execution."""
import json
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
from common.payload import canonical_json_bytes
from evolving_loop.v2.numerical_qd.adapters import import_numerical_seed
from evolving_loop.v2.numerical_qd.contracts import FrozenNumericalRegistryEnvelopeV2
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.task_shortlist import TaskCandidateShortlistV1, TaskShortlistPolicyV1
from numerical_agent.run_task_local_ensemble_evolution import TaskLocalEvidenceBundleV1, _task_input_sha
from tests.test_evolution_v2_kernel import kernel


def evidence_fixture():
    from tests.test_evolution_v2_numerical_runner import fixture
    from tests.test_package_numerical_supply import _policy
    from evolving_loop.package_numerical_supply import NumericalAlternativeSpec
    config, release, folds, adapter = fixture(raw_seed=True)
    policy = _policy("seasonal_naive")
    policy = replace(policy, recipe=replace(policy.recipe, name="seasonal_naive"))
    spec = NumericalAlternativeSpec("seasonal_naive", "statistical", "dictionary",
        policy.recipe.to_payload(), policy.to_payload(),
        tuple((fold, replace(policy, thresholds=(("seasonal_naive_ready", float(fold + 1)),)).to_payload()) for fold in range(5)),
        ("seasonal_naive_ready",), ("The candidate no longer matches the history.",))
    dictionary = adapter.materializer.source_fingerprints["dictionary"]
    release = replace(release, schema_version=2, alternatives=(spec,),
        anchor_release_payload=release.to_payload()["anchor_release_payload"],
        source_fingerprints=dict(release.source_fingerprints) | {"dictionary": dictionary})
    policy = TaskShortlistPolicyV1()
    entries, by_task = [], {}
    for task in adapter.tasks:
        numeric = task.numeric
        input_sha = _task_input_sha(Task(numeric.task_id, numeric.history_values,
            numeric.prediction_length, numeric.frequency, ()))
        shortlist = TaskCandidateShortlistV1(1, input_sha, dictionary, policy.fingerprint(),
            ("toto_2_0", "seasonal_naive"), (), True, False)
        diagnostics = {"schema_version": 1, "task_id": numeric.task_id,
            "task_input_sha256": input_sha, "rows": [], "public_test_accessed": False}
        diagnostic_sha = hashlib.sha256(canonical_json_bytes(diagnostics)).hexdigest()
        entries.append({"task_id": numeric.task_id, "task_input_sha256": input_sha,
            "shortlist_sha256": shortlist.fingerprint(), "diagnostics_sha256": diagnostic_sha})
        by_task[numeric.task_id] = shortlist, diagnostics, shortlist.fingerprint(), diagnostic_sha
    evidence = TaskLocalEvidenceBundleV1(policy, dictionary,
        {"schema_version": 1, "policy_sha256": policy.fingerprint(), "entries": entries,
         "public_test_accessed": False}, by_task)
    adapter.task_local_evidence = evidence
    return config, release, folds, adapter


def frozen_fixture():
    from evolving_loop.v2.numerical_qd.runner import _seed_registry
    config, release, folds, adapter = evidence_fixture()
    registry = _seed_registry(release, adapter)
    imported = import_numerical_seed(release, registry, tasks=adapter.tasks,
        evidence=adapter.task_local_evidence)
    evidence = adapter.task_local_evidence
    payloads = [evidence.policy.to_payload()]
    for shortlist, diagnostics, _, _ in evidence.by_task.values():
        payloads.extend((shortlist.to_payload(), dict(diagnostics)))
    raw = {hashlib.sha256(canonical_json_bytes(payload)).hexdigest(): canonical_json_bytes(payload) for payload in payloads}
    raw[fingerprint_payload(dict(evidence.index))] = canonical_v2_bytes(dict(evidence.index))
    return imported, adapter, raw


def test_seed_execution_never_visits_nonshortlisted_catalog_candidates(monkeypatch):
    from evolving_loop.v2.numerical_qd.runner import _seed_registry
    config, release, folds, adapter = evidence_fixture()
    calls = []
    original = adapter.materializer.forecast_store.forecast
    def forecast(name, *args):
        calls.append(name)
        assert name in {"toto_2_0", "seasonal_naive"}
        return original(name, *args)
    adapter.materializer.forecast_store.forecast = forecast
    from numerical_agent.evolution.screening import ScreeningEntry, ApplicabilityPolicy
    screen = adapter.materializer.screening_policy
    adapter.materializer.screening_policy = replace(screen, entries=screen.entries + tuple(
        ScreeningEntry(f"extra_{i}", "statistical", "keep", ApplicabilityPolicy(), "reviewed") for i in range(12)))
    registry = _seed_registry(release, adapter)
    assert calls == [name for _ in adapter.tasks for name in ("toto_2_0", "seasonal_naive")]
    assert registry.package_for(adapter.tasks[0]).fallback_reason == "anchor_diagnostics_unavailable"


def test_evolved_source_is_scored_and_frozen_outside_the_immutable_seed_shortlist(monkeypatch):
    from evolving_loop.package_registry import task_registry_fingerprint
    from evolving_loop.package_numerical_supply import parse_numerical_supply_release
    from evolving_loop.v2.numerical_qd import adapters
    from evolving_loop.v2.numerical_qd.adapters import evaluate_numerical_child, freeze_qd_supply
    from evolving_loop.v2.numerical_qd.contracts import NumericalInventoryV2, NumericalMemberV2, NumericalQDEntryV2
    from evolving_loop.v2.numerical_qd.map_elites import NumericalQDArchive
    from evolving_loop.v2.numerical_qd.runner import _bootstrap, _packed_train_task_groups, _seed_registry
    from evolving_loop.v2.numerical_qd.hyperband import fixed_rung_manifest
    from tests.test_evolution_v2_numerical_adapters import _recipe, descriptor_policy
    from tests.test_package_numerical_evolution import _build_rows

    config, release, _folds, adapter = evidence_fixture()
    release_payload = json.loads(canonical_v2_bytes(release.to_payload()))
    seed_spec = release_payload["alternatives"][0]
    seed_spec["build_fold_policy_payloads"] = [
        [fold, seed_spec["full_build_policy_payload"]] for fold in range(5)
    ]
    release = parse_numerical_supply_release(release_payload)
    parent_registry = _seed_registry(release, adapter)
    state, seed_genome, policies, _screen, _combined = _bootstrap(config, release, adapter)
    source = '''def evolved_forecast(history, horizon, frequency):
    """Use a bounded recent level for finite time-series histories."""
    width = min(8, len(history))
    level = sum(float(value) for value in history[-width:]) / width
    return [level] * horizon
'''
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    recipe = _recipe("evolved_forecast")
    member = NumericalMemberV2("evolved_recent_mean", "statistical", source_sha,
        fingerprint_payload(recipe.to_payload()), (), state.declared_cells, "active")
    state = replace(state, inventory=NumericalInventoryV2(1, (member, *state.inventory.members)))
    genome = replace(seed_genome, generation=1, mutation_operator="repair",
        inventory_sha256=state.inventory.fingerprint())
    adapter.sources = dict(adapter.sources) | {source_sha: source}
    rows = _build_rows(tuple(task.numeric for task in adapter.tasks[:80]))

    child = adapter.materialize_child(release, genome, state, member_id=member.member_id,
        policies=policies | {member.policy_sha256: recipe}, build_rows=rows,
        descriptor_policy=descriptor_policy(), version="n001")
    assert any(
        item.name == recipe.name
        and item.diagnostics.reason_code != "frozen_package_recipe"
        and item.diagnostics.successful_folds > 0
        for task in adapter.tasks
        for item in child.candidate.registry.package_for(task).ranked_alternatives
    )
    local_calls = []
    def select_local(policy, *, candidate_names, forecasts, diagnostics, horizon):
        local_calls.append(candidate_names)
        specialist = recipe.name
        assert specialist in candidate_names
        selected = (policy.anchor_name, specialist)
        weights = (0.5, 0.5)
        forecast = tuple(
            weights[0] * forecasts[selected[0]][index]
            + weights[1] * forecasts[selected[1]][index]
            for index in range(horizon)
        )
        return SimpleNamespace(activated=True, selected_names=selected,
            weights=weights, forecast=forecast, fallback_reason=None)
    monkeypatch.setattr(adapters, "execute_task_local_ensemble", select_local)
    commitments = {task.numeric.task_id: task_registry_fingerprint(task) for task in adapter.tasks}
    groups = _packed_train_task_groups(adapter, commitments,
        config.kernel_protocol.split_manifest, config.kernel_protocol.fingerprint())
    manifest = fixed_rung_manifest(groups, 8,
        config.kernel_protocol.split_manifest, config.kernel_protocol.fingerprint())
    evaluation = evaluate_numerical_child(adapter, child, manifest,
        descriptor_policy=descriptor_policy(),
        metric_policy_sha256=config.kernel_protocol.metric_policy,
        bracket="explore", rung=0)
    entry = NumericalQDEntryV2(1, evaluation.genome_sha256, evaluation.fingerprint(),
        evaluation.cells[0], evaluation.task_ids, evaluation.objectives,
        evaluation.constraints, evaluation.train_diagnostic_categories)
    archive = NumericalQDArchive().insert((entry,))
    frozen = freeze_qd_supply(adapter, release, parent_registry, archive, (child,),
        descriptor_policy=descriptor_policy(), version="n002",
        required_genome_sha256=genome.fingerprint())

    assert any(item.name == recipe.name for item in child.candidate.registry.package_for(adapter.tasks[0]).ranked_alternatives)
    assert any(item.candidate_id == recipe.name for item in frozen.release.alternatives)
    assert any(item.name == recipe.name for item in frozen.registry.package_for(adapter.tasks[0]).ranked_alternatives)
    assert local_calls
    assert any(recipe.name in names for names in local_calls)
    assert any(recipe.name in frozen.registry.package_for(task).selection_decision.selected
        for task in adapter.tasks)
    assert frozen.envelope.schema_version == 1


def test_resealed_package_diagnostics_cannot_disagree_with_evidence():
    from evolving_loop.v2.numerical_qd.adapters import _envelope, validate_frozen_local_evidence
    from evolving_loop.package_numerical_supply import build_package_registry
    imported, adapter, raw = frozen_fixture()
    restored = imported.envelope.restore(adapter.tasks)
    def corrupt(task, release):
        package = restored.package_for(task)
        diagnostics = dict(package.candidate_diagnostics)
        anchor = package.protected_baseline.name
        diagnostics[anchor] = replace(diagnostics[anchor], cache_key="resealed-different-content")
        return replace(package, candidate_diagnostics=diagnostics,
            protected_baseline=replace(package.protected_baseline, diagnostics=diagnostics[anchor]))
    registry = build_package_registry(adapter.tasks, imported.release, corrupt)
    envelope = _envelope(registry, adapter.tasks, evidence=adapter.task_local_evidence)
    with pytest.raises(ValueError, match="diagnostic content"):
        validate_frozen_local_evidence(imported.release, envelope, artifact_bytes_by_sha=raw)


def test_v2_roundtrip_and_cache_only_closure(monkeypatch):
    from evolving_loop.v2.numerical_qd import adapters
    imported, adapter, raw = frozen_fixture()
    envelope = imported.envelope
    assert envelope.schema_version == 2
    assert set(envelope.to_payload()) == {"schema_version", "release_sha256", "registry_sha256",
        "entries", "packages", "package_sha256s", "shortlist_policy_sha256",
        "dictionary_sha256", "shortlist_index_sha256"}
    assert FrozenNumericalRegistryEnvelopeV2.from_payload(envelope.to_payload()).canonical_bytes() == envelope.canonical_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("frozen restore executed numerical work")
    monkeypatch.setattr(adapters, "bound_numerical_package", forbidden)
    monkeypatch.setattr(adapter, "forecast_trusted", forbidden)
    adapters.validate_frozen_local_evidence(imported.release, envelope,
        artifact_bytes_by_sha=raw, tasks=adapter.tasks)
    adapters.validate_frozen_local_evidence(imported.release, envelope,
        artifact_bytes_by_sha=raw)
    assert envelope.restore(adapter.tasks).fingerprint == envelope.registry_sha256


@pytest.mark.parametrize("field", ["shortlist_policy_sha256", "dictionary_sha256", "shortlist_index_sha256"])
def test_v2_exact_top_schema_rejects_missing_fields(field):
    imported, _, _ = frozen_fixture()
    payload = imported.envelope.to_payload()
    del payload[field]
    with pytest.raises(ValueError):
        FrozenNumericalRegistryEnvelopeV2.from_payload(payload)


@pytest.fixture(scope="module")
def frozen_closure():
    return frozen_fixture()


@pytest.mark.parametrize("mutation", ["index_task", "input", "dictionary", "shortlist", "diagnostics", "component", "weight", "fallback"])
def test_frozen_closure_rejects_resealed_reference_and_weight_mutations(frozen_closure, mutation):
    from evolving_loop.v2.numerical_qd.adapters import validate_frozen_local_evidence, _envelope, _encode
    from evolving_loop.package_registry import FrozenNumericalPackageRegistry
    from numerical_agent.evolution.numerical_selector import replay_selection_forecast
    imported, adapter, originals = frozen_closure
    payload, raw = imported.envelope.to_payload(), dict(originals)
    task = adapter.tasks[0]
    task_id = task.numeric.task_id
    entry = payload["entries"][task_id]
    if mutation == "index_task":
        index = json.loads(raw[payload["shortlist_index_sha256"]])
        index["entries"][0]["shortlist_sha256"] = index["entries"][1]["shortlist_sha256"]
        identity = fingerprint_payload(index)
        raw[identity] = canonical_v2_bytes(index)
        payload["shortlist_index_sha256"] = identity
    elif mutation in {"input", "dictionary", "shortlist", "diagnostics", "component"}:
        if mutation == "input":
            entry["task_input_sha256"] = "f" * 64
        elif mutation == "dictionary":
            payload["dictionary_sha256"] = "f" * 64
        elif mutation == "component":
            payload["shortlist_policy_sha256"] = "f" * 64
        else:
            key = entry["task_shortlist_sha256" if mutation == "shortlist" else "hindcast_diagnostics_sha256"]
            raw[key] = raw[key] + b" "
    else:
        registry = imported.envelope.restore(adapter.tasks)
        package = registry.package_for(task)
        selection = replace(package.selection_decision, mode="ensemble", selected=package.active_candidate_names,
            weights=(0.4, 0.6) if mutation == "weight" else (0.6, 0.4), arithmetic=None)
        forecast = replay_selection_forecast(selection, {item.name: item.forecast for item in package.ranked_alternatives})
        selection = replace(selection, forecast=forecast)
        package = replace(package, morphology_card=None, selection_decision=selection, final_forecast=forecast,
            fallback_reason=None if mutation == "weight" else "insufficient_evidence")
        changed = FrozenNumericalPackageRegistry(
            [(value, package if value == task else registry.package_for(value)) for value in adapter.tasks],
            release_sha256=imported.release.fingerprint, expected_task_ids=registry.task_ids)
        payload = _envelope(changed, adapter.tasks, evidence=adapter.task_local_evidence).to_payload()
    with pytest.raises(ValueError):
        envelope = FrozenNumericalRegistryEnvelopeV2.from_payload(payload)
        validate_frozen_local_evidence(imported.release, envelope, artifact_bytes_by_sha=raw, tasks=adapter.tasks)


def test_legacy_envelope_keeps_exact_six_fields_and_bytes():
    from tests.test_evolution_v2_numerical_runner import fixture
    _, imported, _, adapter = fixture()
    payload = imported.envelope.to_payload()
    assert set(payload) == {"schema_version", "release_sha256", "registry_sha256", "entries", "package_sha256s", "packages"}
    assert all(set(row) == {"task_sha256", "package_sha256"} for row in payload["entries"].values())
    assert FrozenNumericalRegistryEnvelopeV2.from_payload(payload).canonical_bytes() == canonical_v2_bytes(payload)
    assert imported.envelope.restore(adapter.tasks).fingerprint == imported.envelope.registry_sha256
    with pytest.raises(ValueError):
        FrozenNumericalRegistryEnvelopeV2.from_payload(payload | {"shortlist_index_sha256": None})


@pytest.mark.parametrize("corruption", ["evidence", "non_winner_package"])
def test_kernel_accepts_v2_closure_from_bytes_and_rejects_changed_evidence(frozen_closure, tmp_path, monkeypatch, corruption):
    from types import SimpleNamespace
    from evolving_loop.v2.kernel import EvolutionKernel, KernelAuthorityError
    from evolving_loop.v2.numerical_qd import adapters
    from evolving_loop.v2.numerical_qd.contracts import NumericalEvaluationV2, NumericalGenomeV2
    from evolving_loop.package_registry import FrozenNumericalPackageRegistry
    from tests.test_evolution_v2_numerical_contracts import payloads
    imported, adapter, raw = frozen_closure
    genome = NumericalGenomeV2.from_payload(payloads()[NumericalGenomeV2])
    release = replace(imported.release, anchor_release_payload=imported.release.to_payload()["anchor_release_payload"],
        source_fingerprints=dict(imported.release.source_fingerprints) | {"genome": genome.fingerprint(), "train_winner": genome.fingerprint()})
    old = imported.envelope.restore(adapter.tasks)
    registry = FrozenNumericalPackageRegistry([(task, replace(old.package_for(task),
        component_fingerprints=dict(old.package_for(task).component_fingerprints) | {"numerical_supply_release": release.fingerprint}))
        for task in adapter.tasks], release_sha256=release.fingerprint, expected_task_ids=old.task_ids)
    envelope = adapters._envelope(registry, adapter.tasks, evidence=adapter.task_local_evidence)
    evaluation = payloads()[NumericalEvaluationV2]
    ids = sorted(adapter.fold_manifest.task_fold_map)
    evaluation.update(genome_sha256=genome.fingerprint(), supply_sha256=release.fingerprint,
        registry_sha256=registry.fingerprint, task_ids=ids, task_statuses={task_id: "passed" for task_id in ids})
    evaluation = NumericalEvaluationV2.from_payload(evaluation)
    executable = {"materialized_numerical_child": {"genome": genome.to_payload(), "supply": release.to_payload(),
        "registry": envelope.to_payload(), "fit": {"recipe": release.alternatives[0].to_payload()["recipe_payload"]}}}
    pair = {"supply": release.to_payload(), "registry": envelope.to_payload()}
    objects = tmp_path / "numerical_qd/objects"
    objects.mkdir(parents=True)
    for sha, data in raw.items():
        (objects / f"{sha}.json").write_bytes(data)
    for value in (evaluation.to_payload(), executable, pair):
        (objects / f"{fingerprint_payload(value)}.json").write_bytes(canonical_v2_bytes(value))
    bundle = SimpleNamespace(numerical_release_sha256=release.fingerprint, numerical_registry_sha256=registry.fingerprint,
        runtime_fingerprints=evaluation.runtime_fingerprints, protocol_fingerprint=evaluation.protocol_fingerprint)
    train = {"train_objectives": evaluation.objectives.to_payload(), "train_behavior_descriptors": {
        "numerical_artifacts_sha256": fingerprint_payload(pair), "numerical_winner_genome_sha256": genome.fingerprint(),
        "numerical_train_evaluation_sha256": evaluation.fingerprint(), "numerical_winner_materialized_sha256": fingerprint_payload(executable)}}
    host = SimpleNamespace(store=SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(adapters, "bound_numerical_package", lambda *args, **kwargs: pytest.fail("kernel ran numerical work"))
    assert EvolutionKernel._numerical_release_references(host, bundle, train) == tuple(sorted((release.fingerprint, registry.fingerprint)))
    if corruption == "evidence":
        path = objects / f"{envelope.shortlist_index_sha256}.json"
        path.write_bytes(path.read_bytes() + b" ")
        expected_error = "local evidence"
    else:
        from evolving_loop.numerical_two_stage import numerical_package_fingerprint
        from evolving_loop.v2.numerical_qd.contracts import FrozenNumericalPackageEnvelopeV2
        # Alter only the Anchor row; the winner row and all evidence joins stay
        # valid. Reseal the package and pair while retaining the old registry SHA.
        task = adapter.tasks[-1]
        package = registry.package_for(task)
        anchor = replace(package.protected_baseline,
            forecast=tuple(value + 1.0 for value in package.protected_baseline.forecast))
        changed = replace(package, protected_baseline=anchor, final_forecast=anchor.forecast,
            selection_decision=replace(package.selection_decision, forecast=anchor.forecast),
            ranked_alternatives=tuple(anchor if row.name == anchor.name else row for row in package.ranked_alternatives))
        artifact = FrozenNumericalPackageEnvelopeV2(1, numerical_package_fingerprint(changed), adapters._encode(changed))
        entry = pair["registry"]["entries"][task.numeric.task_id]
        del pair["registry"]["packages"][entry["package_sha256"]]
        entry["package_sha256"] = artifact.fingerprint()
        pair["registry"]["packages"][artifact.fingerprint()] = artifact.to_payload()
        pair["registry"]["package_sha256s"] = sorted(pair["registry"]["packages"])
        (objects / f"{fingerprint_payload(pair)}.json").write_bytes(canonical_v2_bytes(pair))
        train["train_behavior_descriptors"]["numerical_artifacts_sha256"] = fingerprint_payload(pair)
        expected_error = "local evidence|registry identity"
    with pytest.raises(KernelAuthorityError, match=expected_error):
        EvolutionKernel._numerical_release_references(host, bundle, train)


def test_evidence_seed_and_final_pair_persist_original_bytes(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd.runner import run_numerical_qd, _persist, _persist_task_local_evidence
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
    from evolving_loop.v2.numerical_qd.adapters import frozen_local_evidence_references, freeze_qd_supply, validate_frozen_local_evidence
    from evolving_loop.v2.numerical_qd.map_elites import NumericalQDArchive
    config, release, folds, adapter = evidence_fixture()
    config = replace(config, budget=replace(config.budget, ceilings=replace(config.budget.ceilings, task_executions=800)))
    root = tmp_path / "run"
    result = run_numerical_qd(root, config, release, folds, adapter)
    pairs = [json.loads(path.read_bytes()) for path in (root / "numerical_qd/objects").glob("*.json")
        if set(json.loads(path.read_bytes())) == {"supply", "registry"}]
    assert pairs
    for pair in pairs:
        envelope = FrozenNumericalRegistryEnvelopeV2.from_payload(pair["registry"])
        assert envelope.schema_version == 2
        for sha in frozen_local_evidence_references(envelope):
            assert hashlib.sha256((root / "numerical_qd/objects" / f"{sha}.json").read_bytes()).hexdigest() == sha
    def forbidden(*args, **kwargs):
        pytest.fail("frozen load executed numerical work")
    monkeypatch.setattr(adapter, "forecast_trusted", forbidden)
    frozen, _ = NumericalQDRunStore(root).load_active_frozen_pair(tasks=adapter.tasks)
    assert frozen.envelope.schema_version == 2
    # Freeze a catalog-preserving final projection without the unrelated legacy
    # duplicate-forecast conflict exercised by the adapter projection tests.
    projected = freeze_qd_supply(adapter, frozen.release, frozen.registry, NumericalQDArchive(), (),
        descriptor_policy=config.descriptor_policy, version="n001")
    store = NumericalQDRunStore(root)
    _persist_task_local_evidence(store, adapter.task_local_evidence)
    pair_sha = _persist(store, {"supply": projected.release.to_payload(), "registry": projected.envelope.to_payload()}, kind="frozen_pair")
    assert store._object(pair_sha)["registry"]["schema_version"] == 2
    raw = {sha: (store.directory / "objects" / f"{sha}.json").read_bytes()
        for sha in frozen_local_evidence_references(projected.envelope)}
    validate_frozen_local_evidence(projected.release, projected.envelope, artifact_bytes_by_sha=raw, tasks=adapter.tasks)


@pytest.mark.parametrize("kind,mutation", [
    ("shortlist_index", "sha"), ("shortlist_index", "duplicate"),
    ("shortlist_index", "type"), ("hindcast_diagnostics", "sha"),
    ("hindcast_diagnostics", "duplicate"), ("hindcast_diagnostics", "type"),
    ("hindcast_diagnostics", "diagnostic"),
])
def test_typed_evidence_rejects_malformed_payload_before_write(kernel, kind, mutation):
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
    from evolving_loop.v2.numerical_qd.runner import _persist
    _, _, _, adapter = evidence_fixture()
    evidence = adapter.task_local_evidence
    payload = json.loads(canonical_v2_bytes(dict(evidence.index) if kind == "shortlist_index"
        else dict(next(iter(evidence.by_task.values()))[1])))
    if mutation == "sha":
        payload["policy_sha256" if kind == "shortlist_index" else "task_input_sha256"] = "INVALID"
    elif mutation == "type":
        payload["schema_version"] = True
    elif kind == "shortlist_index":
        payload["entries"].append(payload["entries"][0])
    else:
        row = {"candidate_name": "toto_2_0", "failure_reason": None, "diagnostic": None}
        payload["rows"] = [row, row] if mutation == "duplicate" else [row | {"diagnostic": {"bogus": 1}}]
    store = NumericalQDRunStore.create(kernel.store.root)
    with pytest.raises((TypeError, ValueError)):
        _persist(store, payload, kind=kind)
    assert not (store.directory / "objects" / (fingerprint_payload(payload) + ".json")).exists()
