"""Real legacy materialization, strict resume, and label/source boundaries."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from evolving_loop.package_numerical_evolution import NumericalPackageMaterializer
from evolving_loop.package_numerical_evolution import NumericalCoordinateCandidate
from evolving_loop.package_numerical_supply import NumericalAlternativeSpec, NumericalSupplyRelease, build_package_registry, bound_numerical_package
from evolving_loop.package_registry import task_registry_fingerprint
from evolving_loop.v2.contracts import fingerprint_payload
from evolving_loop.v2.numerical_qd.adapters import (
    LegacyNumericalAdapter, import_numerical_seed, evaluate_numerical_child,
    freeze_qd_supply,
)
from evolving_loop.v2.numerical_qd.contracts import (
    ConstraintReportV2, FrozenNumericalRegistryEnvelopeV2, MorphologyCellV2,
    NumericalGenomeV2, NumericalInventoryV2, NumericalMemberV2,
    NumericalObjectiveVectorV2, NumericalQDEntryV2, RungManifestV2, TrainTaskV2,
)
from evolving_loop.v2.numerical_qd.descriptors import describe_history
from evolving_loop.v2.numerical_qd.map_elites import NumericalQDArchive
from evolving_loop.v2.numerical_qd.mutation import MutationProposalV2, apply_mutation
from evolving_loop.v2.numerical_qd.proposers import _structural_candidate
from numerical_agent.evolution.numerical_loop import run_numerical_loop
from numerical_agent.evolution.execution import Task as RuntimeTask
from numerical_agent.evolution.screening import ScreeningPolicy, ScreeningEntry, ApplicabilityPolicy
from numerical_agent.evolution.task_local_evolution import build_group_fold_manifest
from tests.test_evolution_v2_numerical_descriptors import policy as descriptor_policy
from tests.test_evolution_v2_numerical_mutation import parent_state
from tests.test_package_numerical_evolution import (
    _evolution_tasks, _build_rows, _recipe, _supply_parent,
)
from tests.test_package_numerical_supply import _policy, _ranked


ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''def seasonal_naive(history, horizon, frequency):
    """Use for finite daily histories with stable local levels."""
    return [float(history[-1]) + 1.0] * horizon

def lagged(history, horizon, frequency):
    """Use for daily histories requiring a conservative lag."""
    return [float(history[-1]) + 2.0] * horizon
'''
SOURCE_SHA = hashlib.sha256(SOURCE.encode()).hexdigest()


def snapshots(paths):
    return {str(path): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in paths}


@pytest.fixture(scope="module", autouse=True)
def unchanged_legacy():
    paths = [path for folder in ("common", "numerical_agent", "evolving_loop")
             for path in (ROOT / folder).rglob("*")
             if path.is_file() and "__pycache__" not in path.parts
             and "v2" not in path.relative_to(ROOT).parts]
    paths += list((ROOT / "runs/frozen_two_stage").rglob("*.json"))
    before = snapshots(paths)
    yield
    assert snapshots(paths) == before


class CheapStore:
    def forecast(self, name, history, horizon, frequency):
        assert not hasattr(history, "future_values")
        return (float(history[-1]) + {"seasonal_naive": 1.0, "lagged": 2.0}.get(name, 0.0),) * horizon


def screening_policy():
    return ScreeningPolicy(tuple(ScreeningEntry(name, family, "keep", ApplicabilityPolicy(), "reviewed")
        for name, family in (("toto_2_0", "tsfm"), ("seasonal_naive", "statistical"), ("lagged", "statistical"))), ("toto_2_0",))


@pytest.fixture(scope="module")
def world():
    tasks = tuple(replace(task, numeric=replace(task.numeric, entity_name=f"entity-{i // 2:03d}"))
                  for i, task in enumerate(_evolution_tasks()))
    folds = build_group_fold_manifest(tuple(t.numeric for t in tasks[:80]), seed=20260903)
    screening = screening_policy()
    materializer = NumericalPackageMaterializer(forecast_store=CheapStore(), screening_policy=screening,
        fold_manifest=folds, original_tasks=tasks, source_fingerprints={"dictionary": SOURCE_SHA},
        runtime_fingerprints={"materializer": "3" * 64})
    adapter = LegacyNumericalAdapter(materializer=materializer, tasks=tasks, fold_manifest=folds,
                                     sources={SOURCE_SHA: SOURCE})
    release = _supply_parent()
    from numerical_agent.evolution.champion import parse_champion_release
    anchor = parse_champion_release(release.to_payload()["anchor_release_payload"])
    def builder(task, supplied):
        source = run_numerical_loop(RuntimeTask(task.numeric.task_id, task.numeric.history_values,
            task.numeric.prediction_length, task.numeric.frequency, ()), screening_policy=screening,
            candidate_runner=CheapStore().forecast, champion_release=anchor)
        return bound_numerical_package(source, supplied, {v.name: v for v in source.ranked_alternatives})
    registry = build_package_registry(tasks, release, builder)
    cells = tuple(sorted({describe_history(t.numeric.history_values, 2, "D", "statistical", descriptor_policy()).fingerprint()
                          for t in tasks}))
    member = NumericalMemberV2("seasonal", "statistical", SOURCE_SHA,
        fingerprint_payload(_recipe().to_payload()), (), cells, "active")
    state = parent_state(inventory=NumericalInventoryV2(1, (member, replace(member, member_id="backup",
                         policy_sha256=fingerprint_payload(_recipe("lagged").to_payload())))),
                         declared_cells=cells, max_inventory_size=12)
    rows = _build_rows(tuple(t.numeric for t in tasks[:80]))
    rows += tuple(replace(row, candidate_name="lagged", forecast=tuple(v + 0.5 for v in row.forecast))
                  for row in rows if row.candidate_name == "seasonal_naive")
    return adapter, release, registry, state, rows


def genome(state, generation=1, operator="repair"):
    from numerical_agent.evolution.screening import _policy_payload
    return NumericalGenomeV2(1, generation, (), operator, state.inventory.fingerprint(),
        fingerprint_payload(_policy_payload(screening_policy())), fingerprint_payload({"policies": []}), state.mutation_policy.fingerprint(), state.proposer_prompt.fingerprint(),
        {"materializer": "3" * 64}, "4" * 64)


def materialize(world, state=None, **kwargs):
    adapter, release, registry, original, rows = world
    state = state or original
    return adapter.materialize_child(release, genome(state), state, member_id=kwargs.pop("member_id", "seasonal"),
        policies=kwargs.pop("policies", {fingerprint_payload(recipe.to_payload()): recipe for recipe in (_recipe(), _recipe("lagged"))}), build_rows=rows,
        descriptor_policy=descriptor_policy(), version="n001", **kwargs)


def test_qd_materialization_refits_and_preserves_v2_seed_catalog(world):
    adapter, release, _registry, state, rows = world
    seed_policy = _policy("seasonal_naive")
    seed = NumericalAlternativeSpec(
        candidate_id="seasonal_naive",
        family="statistical",
        materializer_kind="dictionary",
        recipe_payload=seed_policy.recipe.to_payload(),
        full_build_policy_payload=seed_policy.to_payload(),
        build_fold_policy_payloads=tuple(
            (fold, seed_policy.to_payload()) for fold in range(5)
        ),
        assumption_ids=("seasonal_naive_ready",),
        failure_conditions=("The candidate no longer matches the history.",),
    )
    v2_seed = NumericalSupplyRelease(
        schema_version=2,
        version=release.version,
        parent_sha256=release.parent_sha256,
        anchor_release_payload=release.to_payload()["anchor_release_payload"],
        alternatives=(seed,),
        atlas_release_sha256=release.atlas_release_sha256,
        source_fingerprints=release.source_fingerprints,
        runtime_fingerprints=release.runtime_fingerprints,
    )

    child = adapter.materialize_child(
        v2_seed,
        genome(state),
        state,
        member_id="seasonal",
        policies={fingerprint_payload(_recipe().to_payload()): _recipe()},
        build_rows=rows,
        descriptor_policy=descriptor_policy(),
        version="n001",
    )

    assert child.candidate.release.schema_version == 2
    assert tuple(item.candidate_id for item in child.candidate.release.alternatives)[0] == "seasonal_naive"


def test_v2_projection_keeps_parent_anchor_when_child_diagnostic_cache_differs(world):
    adapter, parent, registry, state, rows = world
    parent = replace(parent, schema_version=2, anchor_release_payload=parent.to_payload()["anchor_release_payload"])
    parent_registry = build_package_registry(adapter.tasks, parent,
        lambda task, supplied: bound_numerical_package(registry.package_for(task), supplied,
            {item.name: item for item in registry.package_for(task).ranked_alternatives}))
    child = adapter.materialize_child(parent, genome(state), state, member_id="seasonal",
        policies={fingerprint_payload(_recipe().to_payload()): _recipe()}, build_rows=rows,
        descriptor_policy=descriptor_policy(), version="n001")
    task = adapter.tasks[0]
    assert child.candidate.registry.package_for(task).protected_baseline.diagnostics.cache_key != parent_registry.package_for(task).protected_baseline.diagnostics.cache_key
    entry = NumericalQDEntryV2(1, child.genome.fingerprint(), "7" * 64,
        describe_history(task.numeric.history_values, 2, "D", "statistical", descriptor_policy()),
        (task.numeric.task_id,), NumericalObjectiveVectorV2(1., 1., 1., 1., 1.), ConstraintReportV2(True, ()), ())
    frozen = freeze_qd_supply(adapter, parent, parent_registry, NumericalQDArchive().insert((entry,)), (child,),
        descriptor_policy=descriptor_policy(), version="n002")
    assert frozen.registry.package_for(task).protected_baseline == parent_registry.package_for(task).protected_baseline


def test_verified_store_accounts_uncached_dispatches_and_real_worker_starts(tmp_path):
    from evolving_loop.v2.budget import ResourceUse
    from evolving_loop.v2.numerical_qd.adapters import _VerifiedForecastStore
    from numerical_agent.evolution.execution import MethodForecastError
    charges = []
    store = _VerifiedForecastStore(tmp_path, {SOURCE_SHA: SOURCE}, {"toto_2_0"}, CheapStore(),
                                  account_work=charges.append)
    try:
        store.forecast("seasonal_naive", (1.0, 2.0), 2, "D")
        store.forecast("seasonal_naive", (1.0, 2.0), 2, "D")
        store.forecast("lagged", (1.0, 2.0), 2, "D")
        store.forecast("toto_2_0", (1.0, 2.0), 2, "D")
        with pytest.raises(MethodForecastError):
            store.forecast("unknown", (1.0, 2.0), 2, "D")
    finally:
        store.close()
    assert sum(charges, ResourceUse()) == ResourceUse(task_executions=3, subprocesses=1)


@pytest.mark.parametrize("failure_at", [1, 3])
def test_materialization_failure_keeps_exact_started_forecasts(world, failure_at):
    import copy
    from evolving_loop.v2.budget import ResourceUse
    charges = []
    adapter = copy.copy(world[0])
    materializer = copy.copy(adapter.materializer)
    class FailingAnchor(CheapStore):
        calls = 0
        def forecast(self, *args):
            self.calls += 1
            if self.calls == failure_at:
                raise ValueError("anchor failed")
            return super().forecast(*args)
    materializer.forecast_store = FailingAnchor()
    adapter.materializer = materializer
    with pytest.raises(ValueError, match="anchor failed"):
        materialize((adapter, *world[1:]), account_work=charges.append)
    assert materializer.forecast_store.calls == failure_at
    assert sum(charges, ResourceUse()) == ResourceUse(task_executions=2 * failure_at, subprocesses=1)


@pytest.mark.parametrize("resource", ["gpu_seconds", "subprocesses", "output_tokens"])
def test_declared_external_resources_require_an_identified_host_reporter(world, resource):
    import copy
    adapter = world[0]
    materializer = copy.copy(adapter.materializer)
    class ExternalStore(CheapStore):
        resource_kinds = (resource,)
        def forecast(self, *args):
            raise AssertionError("uninstrumented external work must not start")
    materializer.forecast_store = ExternalStore()
    with pytest.raises(ValueError, match="resource reporter"):
        LegacyNumericalAdapter(materializer=materializer, tasks=adapter.tasks,
            fold_manifest=adapter.fold_manifest, sources=adapter.sources)


def test_host_reporter_identity_and_failed_work_are_accounted(world):
    import copy
    from evolving_loop.v2.budget import ResourceUse
    adapter = world[0]
    materializer = copy.copy(adapter.materializer)
    class ExternalStore(CheapStore):
        resource_kinds = ("gpu_seconds", "subprocesses", "output_tokens")
        used = ResourceUse()
        def forecast(self, *args):
            self.used += ResourceUse(gpu_seconds=0.25, subprocesses=1, output_tokens=3)
            raise ValueError("reported external failure")
    external = ExternalStore()
    materializer.forecast_store = external
    instrumented = LegacyNumericalAdapter(materializer=materializer, tasks=adapter.tasks,
        fold_manifest=adapter.fold_manifest, sources=adapter.sources,
        resource_reporter=lambda: external.used, resource_reporter_sha256="a" * 64)
    other = copy.copy(instrumented)
    other.resource_reporter_sha256 = "b" * 64
    assert instrumented.fingerprint != other.fingerprint
    charges = []
    def account(use, *, begun=False):
        charges.append(use)
    with pytest.raises(ValueError, match="reported external failure"):
        materialize((instrumented, *world[1:]), account_work=account)
    assert sum(charges, ResourceUse()) == ResourceUse(task_executions=2,
        gpu_seconds=0.25, subprocesses=2, output_tokens=3)


def test_seed_import_and_strict_registry_roundtrip_do_not_rewrite_inputs(world, tmp_path):
    adapter, release, registry, _, _ = world
    artifacts = {"supply.json": release.to_payload(), "registry.json": dict(registry.manifest)}
    # Mapping proxies need a primitive projection; the frozen envelope does not use pickle.
    from common.payload import canonical_json_bytes
    def plain(value):
        from collections.abc import Mapping
        if isinstance(value, Mapping): return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)): return [plain(v) for v in value]
        return value
    for name, payload in artifacts.items():
        (tmp_path / name).write_bytes(canonical_json_bytes(plain(payload)))
    source = tmp_path / "methods.py"
    source.write_text(SOURCE)
    paths = tuple(tmp_path.iterdir())
    before = snapshots(paths)
    imported = import_numerical_seed(release, registry, tasks=adapter.tasks, source_paths=(source,))
    encoded = imported.envelope.canonical_bytes()
    resumed = FrozenNumericalRegistryEnvelopeV2.from_payload(json.loads(encoded)).restore(adapter.tasks)
    assert resumed is not registry and resumed.fingerprint == registry.fingerprint
    assert resumed.release_sha256 == release.fingerprint
    assert imported.sources[SOURCE_SHA] == SOURCE
    assert imported.envelope.package_sha256s == tuple(sorted(imported.envelope.packages))
    for task in adapter.tasks:
        assert resumed.package_for(task) == registry.package_for(task)
    assert snapshots(paths) == before


def test_v2_persisted_screening_identity_drives_real_materialization(world, tmp_path):
    from numerical_agent.evolution.screening import _policy_payload
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
    from evolving_loop.v2.budget import BudgetLedger, BudgetPlan, ResourceUse
    from evolving_loop.v2.kernel import EvolutionKernel
    from evolving_loop.v2.store import V2RunStore
    from tests.test_evolution_v2_kernel import seed, protocol
    adapter, release, _, state, rows = world
    EvolutionKernel(V2RunStore.create(tmp_path / "run"), protocol(),
        BudgetLedger(BudgetPlan(100, 0.2, ResourceUse()), monotonic=lambda: 0.0), seed=seed())
    store = NumericalQDRunStore.create(tmp_path / "run")
    payload = _policy_payload(adapter.materializer.screening_policy)
    identity = fingerprint_payload(payload)
    store.write_object(identity, payload)
    candidate = replace(genome(state), screening_policy_sha256=identity)
    child = adapter.materialize_child(release, candidate, state, member_id="seasonal",
        policies={fingerprint_payload(recipe.to_payload()): recipe for recipe in (_recipe(), _recipe("lagged"))},
        build_rows=rows, descriptor_policy=descriptor_policy(), version="n001")
    assert child.genome.screening_policy_sha256 == identity


def test_registry_envelope_rejects_extra_fields_tampering_missing_and_changed_tasks(world):
    adapter, release, registry, _, _ = world
    envelope = import_numerical_seed(release, registry, tasks=adapter.tasks).envelope
    payload = envelope.to_payload()
    for altered in (payload | {"pickle": "payload"}, payload | {"package_sha256s": []},
                    payload | {"packages": {}}, payload | {"registry_sha256": "f" * 64}):
        with pytest.raises(ValueError):
            FrozenNumericalRegistryEnvelopeV2.from_payload(altered).restore(adapter.tasks)
    sha = next(iter(payload["packages"]))
    payload["packages"][sha]["payload"]["fields"]["final_forecast"] = [42.0, 42.0]
    with pytest.raises(ValueError): FrozenNumericalRegistryEnvelopeV2.from_payload(payload)
    with pytest.raises(ValueError): envelope.restore(adapter.tasks[:-1])
    with pytest.raises(ValueError): envelope.restore((*adapter.tasks[:-1], adapter.tasks[0]))
    changed = replace(adapter.tasks[0], numeric=replace(adapter.tasks[0].numeric, future_values=(9999.0, 9999.0)))
    with pytest.raises(ValueError): envelope.restore((changed, *adapter.tasks[1:]))


@pytest.mark.parametrize("source", [
    "import evolving_loop.v2.kernel\n" + SOURCE,
    SOURCE.replace("return [float(history[-1]) + 1.0] * horizon", "open('/tmp/qd-escape', 'w')\n    return [1.0] * horizon"),
    SOURCE.replace("float(history[-1]) + 1.0", "__import__('os')"),
    SOURCE.replace("float(history[-1]) + 1.0", "eval('1.0')"),
    SOURCE.replace("float(history[-1]) + 1.0", "history.__class__"),
    SOURCE.replace("history, horizon, frequency", "history, horizon"),
    SOURCE.replace('    """Use for finite daily histories with stable local levels."""\n', ""),
    SOURCE.replace("    return", "    try:\n        raise ValueError('bad')\n    except Exception:\n        return"),
    "def native_crash(history, horizon, frequency):\n    \"\"\"Crash fixture.\"\"\"\n    import os\n    os._exit(23)\n",
])
def test_hostile_sources_fail_before_execution(world, source):
    with pytest.raises(ValueError): world[0].validate_source(source)


@pytest.mark.parametrize("expression, status", [("float(history[-1]) + 1.0", "passed"), ("float('nan')", "invalid")])
def test_real_isolated_forecast_is_finite_exact_horizon_or_typed_invalid(world, expression, status):
    source = SOURCE.replace("float(history[-1]) + 1.0", expression)
    outcome = world[0].forecast_source(source, "seasonal_naive", world[0].tasks[0])
    assert outcome.status == status
    assert outcome.forecast == ((8.0, 8.0) if status == "passed" else ())
    assert "future" not in outcome.to_payload()


def test_wrong_horizon_is_typed_invalid(world):
    outcome = world[0].forecast_source(SOURCE.replace("* horizon", "* (horizon + 1)"), "seasonal_naive", world[0].tasks[0])
    assert outcome.status == "invalid" and outcome.forecast == ()


def test_real_materialization_has_exact_grouped_80_train_20_dev_and_future_free_evaluation(world):
    adapter, release, registry, state, _ = world
    child = materialize(world)
    assert len(adapter.tasks) == 100 and len(adapter.fold_manifest.task_fold_map) == 80
    assert all(len(task_ids) == 2 for _, task_ids, _ in adapter.fold_manifest.groups)
    assert child.candidate.release.fingerprint == child.candidate.registry.release_sha256
    assert child.candidate.registry.task_ids == registry.task_ids
    assert len(child.fit.full_build_task_ids) == 80
    assert all(not t.numeric.future_values and not t.gt_evidence and not t.labels_public
               and all(d.role is None and d.subtype is None for d in t.documents) for t in adapter.candidate_tasks)
    for task in adapter.tasks:
        package = child.candidate.registry.package_for(task)
        assert package.protected_baseline.name == "toto_2_0"
        assert len(package.final_forecast) == task.numeric.prediction_length
        assert all(__import__('math').isfinite(v) for item in package.ranked_alternatives for v in item.forecast)
    manifest = train_manifest(adapter, resource=80)
    evaluation = evaluate_numerical_child(adapter, child, manifest, descriptor_policy=descriptor_policy(),
        metric_policy_sha256="6" * 64, bracket="replay", rung=0, normalized_execution_cost=0.1)
    assert len(evaluation.task_ids) == 80 and set(evaluation.task_statuses.values()) == {"passed"}
    assert evaluation.constraints.feasible
    assert evaluation.supply_sha256 == child.candidate.release.fingerprint
    assert "future_values" not in json.dumps(evaluation.to_payload())
    assert "truth" not in json.dumps(evaluation.to_payload())


def test_unknown_policy_digest_fails_closed(world):
    state = world[3]
    member = replace(state.inventory.members[0], policy_sha256="f" * 64)
    altered = replace(state, inventory=NumericalInventoryV2(1, (member, state.inventory.members[1])))
    with pytest.raises(ValueError, match="policy"):
        materialize(world, altered)


def test_specialization_changes_eligible_packages_without_source_rewrite(world):
    state = world[3]
    assert len(state.declared_cells) > 1
    proposal = MutationProposalV2.from_payload(dict(operator="specialize", reason="Train specialization",
        member_id="seasonal", applicability_cells=[state.declared_cells[0]]))
    narrowed = apply_mutation(state, proposal).state
    original = materialize(world)
    child = materialize(world, narrowed, parent_state=state, proposal=proposal)
    before = [t.numeric.task_id for t in world[0].tasks
              if "select_seasonal_naive" in original.candidate.registry.package_for(t).active_candidate_names]
    after = [t.numeric.task_id for t in world[0].tasks
             if "select_seasonal_naive" in child.candidate.registry.package_for(t).active_candidate_names]
    assert set(after) < set(before) and after
    assert narrowed.inventory.members[0].source_sha256 == state.inventory.members[0].source_sha256


@pytest.mark.parametrize("operator", ["repair", "fork", "combine", "route"])
def test_deterministic_policy_is_reconstructed_and_translated_to_legacy_recipe(world, operator):
    state = world[3]
    parents = sorted(state.inventory.members, key=lambda m: m.member_id)
    if operator in {"repair", "fork"}: parents = parents[:1]
    proposal = MutationProposalV2.from_payload(_structural_candidate(operator, parents, state.declared_cells[:1]))
    evolved = apply_mutation(state, proposal).state
    target = proposal.payload.get("child", proposal.payload.get("replacement"))["member_id"]
    child = materialize(world, evolved, member_id=target, parent_state=state, proposal=proposal)
    assert child.fit.recipe.kind == {"repair": "select", "fork": "select", "combine": "weighted", "route": "route"}[operator]
    assert child.fit.recipe.name == target
    assert child.candidate.release.source_fingerprints["member_policy"] == next(m.policy_sha256 for m in evolved.inventory.members if m.member_id == target)
    assert child.member.source_sha256 == SOURCE_SHA
    bad_member = replace(child.member, policy_sha256="e" * 64)
    bad_state = replace(evolved, inventory=NumericalInventoryV2(1, tuple(bad_member if m.member_id == target else m for m in evolved.inventory.members)))
    with pytest.raises(ValueError):
        materialize(world, bad_state, member_id=target, parent_state=state, proposal=proposal)


def test_qd_projection_prefers_new_cell_coverage_and_binds_all_sources(world):
    child = materialize(world)
    # Multiple cells for one genome must be counted once each, independent of insertion order.
    entries = tuple(NumericalQDEntryV2(1, child.genome.fingerprint(), fingerprint_payload({"i": i}),
        MorphologyCellV2(trend, seasonality, "low", "stable", "short", "statistical"),
        (world[0].tasks[0].numeric.task_id,), NumericalObjectiveVectorV2(*(float(i + 1),) * 5),
        ConstraintReportV2(True, ()), ()) for i, (trend, seasonality) in enumerate(
            (("low", "none"), ("medium", "none"), ("high", "none"), ("low", "short"), ("low", "long"))))
    archive = NumericalQDArchive().insert(entries)
    frozen = freeze_qd_supply(world[0], world[1], world[2], archive, (child,),
                             descriptor_policy=descriptor_policy(), version="n002")
    other = freeze_qd_supply(world[0], world[1], world[2], NumericalQDArchive().insert(reversed(entries)),
                            (child,), descriptor_policy=descriptor_policy(), version="n002")
    assert frozen.release.fingerprint == other.release.fingerprint
    assert len(frozen.release.alternatives) == 1
    assert frozen.envelope.restore(world[0].tasks).fingerprint == frozen.registry.fingerprint
    sources = frozen.release.source_fingerprints
    for key in ("qd_snapshot", "genomes", "inventories", "screening_policies", "combined_policies", "mutation_policies",
                "proposer_prompts", "descriptor_policy", "selected_sources", "selected_member_policies"):
        assert key in sources
    assert sources["qd_snapshot"] == archive.fingerprint()
    for task in world[0].tasks:
        assert frozen.registry.package_for(task).protected_baseline.forecast == world[2].package_for(task).protected_baseline.forecast


def projection_child(world, base, index, family="statistical"):
    """Exact legacy packages with distinct alternatives for projection competition."""
    recipe = replace(base.fit.recipe, name=f"choice_{index}")
    fit = replace(base.fit, recipe=recipe, full_build_policy=replace(base.fit.full_build_policy, recipe=recipe),
        build_fold_policies=tuple((fold, replace(policy, recipe=recipe)) for fold, policy in base.fit.build_fold_policies))
    spec = NumericalAlternativeSpec(recipe.name, family, "champion", recipe.to_payload(), fit.full_build_policy.to_payload(),
        tuple((fold, policy.to_payload()) for fold, policy in fit.build_fold_policies),
        tuple(a.assumption_id for a in recipe.assumptions), tuple(a.failure_condition for a in recipe.assumptions))
    release = replace(base.candidate.release, alternatives=(spec,),
        anchor_release_payload=base.candidate.release.to_payload()["anchor_release_payload"])
    def builder(task, supplied):
        source = world[2].package_for(task)
        ranked = _ranked(recipe.name, family, (task.numeric.history_values[-1] + index + 1.0,) * 2)
        return bound_numerical_package(source, supplied, {source.protected_baseline.name: source.protected_baseline,
                                                        recipe.name: ranked})
    registry = build_package_registry(world[0].tasks, release, builder)
    return replace(base, genome=replace(base.genome, generation=index + 2), fit=fit,
                   candidate=NumericalCoordinateCandidate(release, registry, fingerprint_payload({"index": index})))


def projection_entry(child, cell, scores):
    return NumericalQDEntryV2(1, child.genome.fingerprint(), fingerprint_payload({"name": child.fit.recipe.name, "cell": cell.to_payload()}),
        cell, ("build_case_000",), NumericalObjectiveVectorV2(*map(float, scores)), ConstraintReportV2(True, ()), ())


def test_projection_coverage_precedes_rank_and_keeps_at_most_one_of_each_legacy_family(world):
    base = materialize(world)
    children = tuple(projection_child(world, base, i, family) for i, family in enumerate(
        ("statistical", "statistical", "tsfm", "tsfm", "combined", "atlas_overlay")))
    cells = tuple(MorphologyCellV2(trend, season, "low", "stable", "short", "statistical")
                  for trend, season in (("low", "none"), ("medium", "none"), ("high", "none"), ("low", "short"), ("low", "long")))
    rows = [projection_entry(children[0], cell, (9,) * 5) for cell in cells[:4]]
    rows += [projection_entry(children[1], cells[0], (0.1,) * 5)]
    rows += [projection_entry(child, cells[4], (1,) * 5) for child in children[2:]]
    archive = NumericalQDArchive().insert(rows)
    frozen = freeze_qd_supply(world[0], world[1], world[2], archive, children,
        descriptor_policy=descriptor_policy(), version="n002")
    assert frozen.selected_genome_sha256s[0] == children[0].genome.fingerprint()
    assert {spec.family for spec in frozen.release.alternatives} == {"statistical", "tsfm", "combined", "atlas_overlay"}
    assert len(frozen.release.alternatives) == 4
    assert children[1].genome.fingerprint() not in frozen.selected_genome_sha256s
    again = freeze_qd_supply(world[0], world[1], world[2], archive, reversed(children),
        descriptor_policy=descriptor_policy(), version="n002")
    assert frozen.envelope.canonical_bytes() == again.envelope.canonical_bytes()


def test_projection_must_include_exact_train_winner_even_when_coverage_prefers_peer(world):
    base = materialize(world)
    broad, winner = (projection_child(world, base, index) for index in range(2))
    cells = tuple(MorphologyCellV2(trend, "none", "low", "stable", "short", "statistical")
                  for trend in ("low", "medium", "high"))
    archive = NumericalQDArchive().insert([
        *(projection_entry(broad, cell, (9,) * 5) for cell in cells),
        projection_entry(winner, cells[0], (0.1,) * 5)])
    frozen = freeze_qd_supply(world[0], world[1], world[2], archive, (broad, winner),
        descriptor_policy=descriptor_policy(), version="n002", required_genome_sha256=winner.genome.fingerprint())
    assert winner.genome.fingerprint() in frozen.selected_genome_sha256s
    assert all(any(row.name == winner.fit.recipe.name for row in frozen.registry.package_for(task).ranked_alternatives)
               for task in world[0].tasks)


def test_projection_includes_evaluated_train_winner_evicted_by_historical_occupant(world):
    base = materialize(world)
    historical, winner = (projection_child(world, base, index) for index in range(2))
    cell = MorphologyCellV2("low", "none", "low", "stable", "short", "statistical")
    archive = NumericalQDArchive(capacity=1).insert((projection_entry(historical, cell, (0.1,) * 5),))
    archive = archive.insert((projection_entry(winner, cell, (1,) * 5),))
    assert {archive.entries[sha].genome_sha256 for state in archive.cells for sha in state.entry_sha256s} == {historical.genome.fingerprint()}
    frozen = freeze_qd_supply(world[0], world[1], world[2], archive, (historical, winner),
        descriptor_policy=descriptor_policy(), version="n003", required_genome_sha256=winner.genome.fingerprint())
    assert winner.genome.fingerprint() in frozen.selected_genome_sha256s
    assert all(any(row.name == winner.fit.recipe.name for row in frozen.registry.package_for(task).ranked_alternatives)
               for task in world[0].tasks)


def test_add_mutation_selects_new_executable_from_persisted_inventory_order(world):
    from evolving_loop.v2.numerical_qd.adapters import _canonical_member
    state = world[3]
    member = replace(state.inventory.members[1], member_id="new_added")
    proposal = MutationProposalV2.from_payload({"operator": "add", "member": member.to_payload(), "reason": "new executable"})
    evolved = apply_mutation(state, proposal).state
    assert _canonical_member(evolved) == member
    child = materialize(world, evolved, member_id=member.member_id, parent_state=state, proposal=proposal)
    assert child.fit.recipe.parents == ("lagged",)


def test_materialized_executable_envelope_restores_exact_registry_and_fit(world):
    from evolving_loop.v2.numerical_qd.adapters import MaterializedNumericalChildV2
    child = materialize(world)
    payload = child.to_payload(world[0].tasks)
    restored = MaterializedNumericalChildV2.from_payload(payload, world[0].tasks)
    assert restored.genome == child.genome
    assert restored.fit == child.fit
    assert restored.candidate.registry.fingerprint == child.candidate.registry.fingerprint
    assert restored.to_payload(world[0].tasks) == payload


@pytest.mark.parametrize("scores, winners", [
    (((3, 3, 3, 3, 3), (1, 1, 1, 1, 1), (2, 2, 2, 2, 2)), (1,)),
    (((0, 2, 1, 1, 1), (1, 1, 1, 1, 1), (2, 0, 1, 1, 1)), (0, 2)),
    (((0, 2, 0, 2, 3), (1, 1, 1, 1, 1), (2, 0, 2, 0, 2)), (1,)),
    (((1, 1, 1, 1, 1),) * 3, (0, 1, 2)),
])
def test_projection_rank_crowding_cost_and_sha_ties(world, scores, winners):
    base = materialize(world)
    children = tuple(projection_child(world, base, i) for i in range(3))
    cell = MorphologyCellV2("low", "none", "low", "stable", "short", "statistical")
    archive = NumericalQDArchive().insert(projection_entry(child, cell, score) for child, score in zip(children, scores))
    frozen = freeze_qd_supply(world[0], world[1], world[2], archive, children,
        descriptor_policy=descriptor_policy(), version="n002")
    assert frozen.selected_genome_sha256s == (min(children[i].genome.fingerprint() for i in winners),)


def test_member_cannot_claim_source_for_a_different_executable(world):
    state = world[3]
    foreign = _recipe("unverified_method")
    member = replace(state.inventory.members[0], policy_sha256=fingerprint_payload(foreign.to_payload()))
    with pytest.raises(ValueError, match="source"):
        world[0]._recipe(member, {member.policy_sha256: foreign}, None, None)


def test_runtime_and_screening_fingerprints_cannot_be_mislabeled(world):
    adapter, release, _, state, rows = world
    for altered in (replace(genome(state), runtime_fingerprints={"materializer": "9" * 64}),
                    replace(genome(state), screening_policy_sha256="9" * 64)):
        with pytest.raises(ValueError, match="runtime|screening"):
            adapter.materialize_child(release, altered, state, member_id="seasonal",
                policies={fingerprint_payload(_recipe().to_payload()): _recipe()}, build_rows=rows,
                descriptor_policy=descriptor_policy(), version="n001")


@pytest.mark.parametrize("primitive_recipes", [False, True])
def test_structural_policy_can_resume_and_evolve_for_a_second_generation(world, primitive_recipes):
    state = world[3]
    parent = state.inventory.members[0]
    first = MutationProposalV2.from_payload(_structural_candidate("fork", (parent,), state.declared_cells[:1]))
    evolved = apply_mutation(state, first).state
    first_member = next(member for member in evolved.inventory.members if member.member_id == first.payload["child"]["member_id"])
    policies = {fingerprint_payload(recipe.to_payload()): recipe for recipe in (_recipe(), _recipe("lagged"))}
    if primitive_recipes:
        policies = {sha: recipe.to_payload() for sha, recipe in policies.items()}
    policies[first_member.policy_sha256] = dict(schema_version=1, operator="fork", parents=[parent.to_payload()],
                                              applicability_cells=list(first_member.applicability_cells))
    child = materialize(world, evolved, member_id=first_member.member_id, policies=policies)
    second = MutationProposalV2.from_payload(_structural_candidate("repair", (first_member,), first_member.applicability_cells))
    second_state = apply_mutation(evolved, second).state
    member_id = second.payload["replacement"]["member_id"]
    successor = materialize(world, second_state, member_id=member_id, policies=policies,
                            parent_state=evolved, proposal=second)
    assert successor.fit.recipe.parents == child.fit.recipe.parents == ("seasonal_naive",)
    assert successor.fit.recipe.name != child.fit.recipe.name
    assert successor.member.source_sha256 == child.member.source_sha256 == SOURCE_SHA
    corrupted = dict(policies[first_member.policy_sha256]) | {"operator": "repair"}
    with pytest.raises(ValueError, match="policy"):
        materialize(world, evolved, member_id=first_member.member_id,
                    policies=policies | {first_member.policy_sha256: corrupted})


def train_manifest(adapter, resource=8):
    from evolving_loop.v2.numerical_qd.runner import _packed_train_task_groups

    groups = _packed_train_task_groups(
        adapter,
        {
            task.numeric.task_id: task_registry_fingerprint(task)
            for task in adapter.tasks
        },
        "5" * 64,
        "4" * 64,
    )
    return RungManifestV2(resource, "5" * 64, "4" * 64, groups)


def test_existing_package_evaluation_adapter_receives_trusted_train_only_and_emits_closed_aggregate(world):
    from evolving_loop.package_metrics import PackageEvaluation
    from tests.test_package_metrics import _score
    from evolving_loop.numerical_two_stage import numerical_package_fingerprint
    child = materialize(world)
    def host_evaluate(candidate, tasks):
        assert candidate is child.candidate
        assert len(tasks) == 8 and all(task.numeric.future_values for task in tasks)
        rows = tuple(replace(_score(task.numeric.task_id, smae=0.25, srmse=0.5),
            numerical_package_sha256=numerical_package_fingerprint(candidate.registry.package_for(task)),
            entity_name=task.numeric.entity_name) for task in tasks)
        return PackageEvaluation.from_rows(candidate.proposal_sha256, rows, [task.numeric.task_id for task in tasks])
    original = world[0]
    adapter = LegacyNumericalAdapter(materializer=original.materializer, tasks=original.tasks,
        fold_manifest=original.fold_manifest, sources=original.sources, host_evaluator=host_evaluate,
        host_evaluator_sha256="7" * 64)
    result = evaluate_numerical_child(adapter, child, train_manifest(adapter), descriptor_policy=descriptor_policy(),
        metric_policy_sha256="6" * 64, bracket="explore", rung=0)
    assert result.objectives.mean_capped_smae == 0.25 and result.objectives.mean_capped_srmse == 0.5
    assert "future_values" not in json.dumps(result.to_payload()) and "final_forecast" not in json.dumps(result.to_payload())
    adapter.host_evaluator = lambda candidate, tasks: {"future_values": [1.0]}
    with pytest.raises(ValueError, match="PackageEvaluation"):
        evaluate_numerical_child(adapter, child, train_manifest(adapter), descriptor_policy=descriptor_policy(),
            metric_policy_sha256="6" * 64, bracket="explore", rung=0)


def test_exact_legacy_proposer_wraps_three_real_materialized_children(world):
    from evolving_loop.package_numerical_evolution import NumericalPackageProposer
    from numerical_agent.evolution.champion_controller import ChampionProposerAdapter
    from numerical_agent.evolution.champion_evidence import ProposerEvidence
    from tests.test_package_numerical_evolution import _proposal_batch
    original, release, registry, _, rows = world
    proposer = NumericalPackageProposer(proposer=ChampionProposerAdapter.scripted(identity="qd_legacy_test",
        proposal_batches=(_proposal_batch(),), config={"fixture": True}), materializer=original.materializer,
        build_rows=rows, fold_manifest=original.fold_manifest, tasks=original.tasks)
    adapter = LegacyNumericalAdapter(materializer=original.materializer, tasks=original.tasks,
        fold_manifest=original.fold_manifest, sources=original.sources, proposer=proposer)
    candidates = adapter.propose_legacy(release, registry,
        ProposerEvidence("adaptive_train_build_diagnostic", False, (), ()), generation=0)
    assert len(candidates) == 3 and all(candidate.invalid_reason is None for candidate in candidates)
    assert all(candidate.registry.task_ids == registry.task_ids for candidate in candidates)
    assert all(candidate.registry.release_sha256 == candidate.release.fingerprint for candidate in candidates)


def test_combined_export_binds_both_verified_parent_source_identities(world):
    original, release, registry, state, rows = world
    from numerical_agent.evolution.module import parse_module
    methods = parse_module(SOURCE).methods
    sources = {hashlib.sha256(method.source.encode()).hexdigest(): method.source for method in methods}
    by_name = {method.name: hashlib.sha256(method.source.encode()).hexdigest() for method in methods}
    members = tuple(replace(member, source_sha256=by_name["seasonal_naive" if member.member_id == "seasonal" else "lagged"])
                    for member in state.inventory.members)
    state = replace(state, inventory=NumericalInventoryV2(1, members))
    adapter = LegacyNumericalAdapter(materializer=original.materializer, tasks=original.tasks,
        fold_manifest=original.fold_manifest, sources=sources)
    proposal = MutationProposalV2.from_payload(_structural_candidate("combine", sorted(members, key=lambda m: m.member_id), state.declared_cells))
    evolved = apply_mutation(state, proposal).state
    child = materialize((adapter, release, registry, state, rows), evolved,
        member_id=proposal.payload["child"]["member_id"], parent_state=state, proposal=proposal)
    assert child.source_sha256s == tuple(sorted(sources))
    entry = projection_entry(child, MorphologyCellV2("low", "none", "low", "stable", "short", "combined"), (1,) * 5)
    frozen = freeze_qd_supply(adapter, release, registry, NumericalQDArchive().insert((entry,)), (child,),
        descriptor_policy=descriptor_policy(), version="n002")
    assert frozen.release.source_fingerprints["selected_sources"] == fingerprint_payload({"sha256s": sorted(sources)})


def test_host_evaluator_identity_changes_execution_cache_identity(world):
    original = world[0]
    kwargs = dict(materializer=original.materializer, tasks=original.tasks,
                  fold_manifest=original.fold_manifest, sources=original.sources, host_evaluator=lambda *_: None)
    with pytest.raises(ValueError, match="evaluator"):
        LegacyNumericalAdapter(**kwargs)
    left = LegacyNumericalAdapter(**kwargs, host_evaluator_sha256="1" * 64)
    right = LegacyNumericalAdapter(**kwargs, host_evaluator_sha256="2" * 64)
    assert left.fingerprint != right.fingerprint


def test_operator_input_hashes_are_exact_immutable_and_bound_to_adapter(world):
    original = world[0]
    kwargs = dict(
        materializer=original.materializer,
        tasks=original.tasks,
        fold_manifest=original.fold_manifest,
        sources=original.sources,
    )
    committed = {
        "config": "1" * 64,
        "seed_supply": "2" * 64,
        "task_manifest": "3" * 64,
    }
    adapter = LegacyNumericalAdapter(
        **kwargs, operator_input_sha256s=committed
    )
    assert dict(adapter.operator_input_sha256s) == committed
    with pytest.raises(TypeError):
        adapter.operator_input_sha256s["config"] = "4" * 64
    with pytest.raises(AttributeError):
        adapter.operator_input_sha256s = committed
    changed = LegacyNumericalAdapter(
        **kwargs,
        operator_input_sha256s=committed | {"task_manifest": "4" * 64},
    )
    assert changed.fingerprint != adapter.fingerprint
    for malformed in (
        {},
        committed | {"extra": "4" * 64},
        {"config": "1" * 64, "seed_supply": "2" * 64},
        committed | {"config": "not-a-sha"},
    ):
        with pytest.raises(ValueError, match="operator input"):
            LegacyNumericalAdapter(
                **kwargs, operator_input_sha256s=malformed
            )


def test_single_task_evaluation_satisfies_existing_hyperband_cache_boundary(world):
    from evolving_loop.v2.numerical_qd.hyperband import _task_evaluation
    child = materialize(world)
    adapter = world[0]
    task = train_manifest(adapter).tasks[0]
    result = evaluate_numerical_child(adapter, child, task, descriptor_policy=descriptor_policy(),
        metric_policy_sha256="6" * 64, bracket="explore", rung=0)
    assert result.task_ids == (task.task_id,) and result.task_subset_sha256 == task.fingerprint()
    assert _task_evaluation(result, child.genome.fingerprint(), task, "6" * 64,
        descriptor_policy().fingerprint(), child.genome.runtime_fingerprints, adapter.fingerprint) == result


def test_single_task_evaluation_authenticates_opaque_fold_group_identity(world):
    from evolving_loop.v2.numerical_qd.runner import _packed_train_task_groups

    adapter = world[0]
    child = materialize(world)
    commitments = {
        task.numeric.task_id: task_registry_fingerprint(task)
        for task in adapter.tasks
    }
    groups = _packed_train_task_groups(adapter, commitments, "5" * 64, "4" * 64)
    task = RungManifestV2(8, "5" * 64, "4" * 64, groups).tasks[0]

    result = evaluate_numerical_child(
        adapter,
        child,
        task,
        descriptor_policy=descriptor_policy(),
        metric_policy_sha256="6" * 64,
        bracket="explore",
        rung=0,
    )
    assert result.task_ids == (task.task_id,)
    with pytest.raises(ValueError, match="Train task content or group mismatch"):
        evaluate_numerical_child(
            adapter,
            child,
            replace(task, entity_id="fold-group-999-" + "f" * 64),
            descriptor_policy=descriptor_policy(),
            metric_policy_sha256="6" * 64,
            bracket="explore",
            rung=0,
        )


def test_one_genome_cannot_accept_two_executable_members_or_cache_forecasts(world):
    from evolving_loop.v2.numerical_qd.hyperband import _task_evaluation
    child = materialize(world)
    task = train_manifest(world[0]).tasks[0]
    result = evaluate_numerical_child(world[0], child, task, descriptor_policy=descriptor_policy(),
        metric_policy_sha256="6" * 64, bracket="explore", rung=0)
    assert _task_evaluation(result, child.genome.fingerprint(), task, "6" * 64,
        descriptor_policy().fingerprint(), child.genome.runtime_fingerprints, world[0].fingerprint) == result
    with pytest.raises(ValueError, match="canonical member"):
        materialize(world, member_id="backup")
    with pytest.raises(ValueError, match="canonical member"):
        replace(child, member=world[3].inventory.members[1])
    repeated = materialize(world)
    assert repeated.candidate.registry.fingerprint == child.candidate.registry.fingerprint


def test_verified_source_bytes_override_independent_store_in_fit_and_materialization(world):
    original, release, registry, state, rows = world
    source = SOURCE.replace("+ 1.0", "+ 9000.0")
    sha = hashlib.sha256(source.encode()).hexdigest()
    altered = replace(state, inventory=NumericalInventoryV2(1, tuple(
        replace(member, source_sha256=sha) for member in state.inventory.members)))
    adapter = LegacyNumericalAdapter(materializer=original.materializer, tasks=original.tasks,
        fold_manifest=original.fold_manifest, sources={sha: source})
    child = materialize((adapter, release, registry, altered, rows))
    task = adapter.tasks[0]
    actual = next(item.forecast for item in child.candidate.registry.package_for(task).ranked_alternatives
                  if item.name == child.fit.recipe.name)
    assert actual == (task.numeric.history_values[-1] + 9000.0,) * task.numeric.prediction_length
    assert child.member.source_sha256 == sha
    assert child.genome.fingerprint() != genome(state).fingerprint()
    # Caller-held cached fitting forecasts must not alter a committed recipe's fit.
    poisoned = tuple(replace(row, forecast=(900000.0,) * row.profile.horizon) for row in rows)
    again = materialize((adapter, release, registry, altered, poisoned))
    assert again.fit == child.fit
    assert again.candidate.registry.fingerprint == child.candidate.registry.fingerprint


@pytest.mark.parametrize("statement", [
    "import numpy as np\n    np.savetxt(PATH, [1.0])",
    "import numpy as np\n    np.save(PATH, [1.0])",
    "import numpy as np\n    np.savez(PATH, data=[1.0])",
    "import numpy as np\n    np.savez_compressed(PATH, data=[1.0])",
    "import numpy as np\n    np.array([1.0]).tofile(PATH)",
    "import numpy as np\n    np.memmap(PATH, mode='w+', shape=(1,))",
    "import numpy as np\n    np.load(PATH)",
    "import numpy as np\n    np.loadtxt(PATH)",
    "import numpy as np\n    np.genfromtxt(PATH)",
    "import numpy as np\n    np.fromfile(PATH)",
    "from numpy import savetxt as writer\n    writer(PATH, [1.0])",
    "import numpy as np\n    writer = np.savetxt\n    writer(PATH, [1.0])",
    "import numpy as np\n    getattr(np, 'save' + 'txt')(PATH, [1.0])",
    "from numpy.lib.format import open_memmap\n    open_memmap(PATH, mode='w+', shape=(1,))",
    "import pandas as pd\n    pd.DataFrame([1.0]).to_csv(PATH)",
    "import numpy as np\n    np.ctypeslib.load_library('evil', PATH)",
])
def test_library_io_surfaces_rejected_before_execution(world, tmp_path, statement):
    target = tmp_path / "must-not-exist"
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        statement.replace("PATH", repr(str(target))) + "\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)
    assert not target.exists()


def test_library_gate_rejects_builtin_namespace_escape(world):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        "__builtins__['open']('/tmp/qd-escape', 'w')\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)


def test_closed_library_gate_preserves_finite_numpy_forecasting(world):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        "import numpy as np\n    return np.full(horizon, np.mean(np.asarray(history))).tolist()")
    outcome = world[0].forecast_source(source, "seasonal_naive", world[0].tasks[0])
    assert outcome.status == "passed" and outcome.forecast == (3.5, 3.5)


def test_conflicting_source_names_fail_before_train_row_access(world):
    original, release, registry, state, rows = world
    other = SOURCE.replace("+ 1.0", "+ 9000.0")
    sha = hashlib.sha256(other.encode()).hexdigest()
    state = replace(state, inventory=NumericalInventoryV2(1, (state.inventory.members[0],
        replace(state.inventory.members[1], source_sha256=sha))))
    proposal = MutationProposalV2.from_payload(_structural_candidate("combine",
        sorted(state.inventory.members, key=lambda member: member.member_id), state.declared_cells))
    evolved = apply_mutation(state, proposal).state
    adapter = LegacyNumericalAdapter(materializer=original.materializer, tasks=original.tasks,
        fold_manifest=original.fold_manifest, sources={SOURCE_SHA: SOURCE, sha: other})
    class UntouchedRows:
        def __iter__(self):
            raise AssertionError("ambiguous source reached Train data")
    with pytest.raises(ValueError, match="ambiguous executable source"):
        materialize((adapter, release, registry, state, UntouchedRows()), evolved,
            member_id=proposal.payload["child"]["member_id"], parent_state=state, proposal=proposal)


@pytest.mark.parametrize("name", ["hash", "id", "license", "help", "credits", "copyright",
    "breakpoint", "input", "repr", "str", "format", "print", "set", "frozenset"])
@pytest.mark.parametrize("use", ["{name}(frequency)", "callback = {name}", "map({name}, [frequency])"])
def test_unbound_builtins_rejected_as_calls_references_and_callbacks(world, name, use):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        use.format(name=name) + "\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)


@pytest.mark.parametrize("statement", [
    "import numpy as np\n    values = np.empty(horizon)",
    "import numpy as np\n    values = np.empty_like(history)",
    "values = {'short', 'long'}",
    "values = {frequency for value in history}",
    "import numpy as np\n    np.pi += 1.0",
    "global license\n    license()",
])
def test_process_dependent_or_shared_runtime_surfaces_reject(world, statement):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        statement + "\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)


@pytest.mark.parametrize("binding", [
    "def helper():\n        license = sum\n    license()",
    "values = [license for license in (sum,)]\n    license()",
])
def test_inner_scope_binding_cannot_authorize_outer_builtin(world, binding):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        binding + "\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)


def test_other_function_binding_cannot_authorize_builtin(world):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        "license()\n    return [1.0] * horizon").replace(
        "return [float(history[-1]) + 2.0] * horizon", "license = sum\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)


def test_pure_locals_import_aliases_and_callbacks_survive_closed_names(world):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon", '''import math as math_ops
    from numpy import asarray as array, mean as average
    license = sum
    values = list(map(abs, history))
    def center(items):
        return [value - min(items) for value in items]
    callback = lambda item: math_ops.fabs(item)
    value = average(array(center(values))) + license([callback(0.0)])
    return [float(value)] * horizon''')
    outcome = world[0].forecast_source(source, "seasonal_naive", world[0].tasks[0])
    assert outcome.status == "passed" and outcome.forecast == (3.5, 3.5)


def test_accepted_forecasts_match_across_fresh_hash_randomized_interpreters():
    import os
    import subprocess
    import sys
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon", '''import math as math_ops
    import numpy as np
    values = [math_ops.fabs(value) for index, value in enumerate(history) if index >= 0]
    total = sum(map(float, values))
    return np.full(horizon, total / max(1, len(values))).tolist()''')
    command = '''import json, sys
from evolving_loop.v2.numerical_qd.adapters import LegacyNumericalAdapter
from tests.test_package_numerical_evolution import _evolution_tasks
adapter = object.__new__(LegacyNumericalAdapter)
outcome = adapter.forecast_source(sys.stdin.read(), "seasonal_naive", _evolution_tasks()[0])
print(json.dumps(outcome.to_payload(), sort_keys=True))
'''
    outcomes = []
    for seed in ("1", "2", "314159"):
        result = subprocess.run([sys.executable, "-c", command], input=source, text=True,
            capture_output=True, check=True, timeout=30, cwd=ROOT,
            env=dict(os.environ, PYTHONHASHSEED=seed))
        outcomes.append(json.loads(result.stdout))
    assert all(result["status"] == "passed" and result["forecast"] == [3.5, 3.5] for result in outcomes)
    assert outcomes[0] == outcomes[1] == outcomes[2]


def test_hash_forecast_and_unknown_site_injected_name_fail_closed(world):
    for expression in ("float(hash(frequency))", "future_site_hook(frequency)"):
        source = SOURCE.replace("float(history[-1]) + 1.0", expression)
        with pytest.raises(ValueError, match="closed deterministic capabilities"):
            world[0].validate_source(source)


@pytest.mark.parametrize("initialization", [
    "license()\nfrom math import fsum as license\n",
    "counter = 0\n",
])
def test_module_initialization_cannot_fall_back_to_builtin_or_keep_state(world, initialization):
    with pytest.raises(ValueError, match="module initialization"):
        world[0].validate_source(initialization + SOURCE)


@pytest.mark.parametrize("signature", [
    "history=license(), horizon=2, frequency='D'",
    "history: license(), horizon, frequency",
])
def test_definition_time_expression_cannot_use_a_later_safe_alias(world, signature):
    source = SOURCE.replace("history, horizon, frequency", signature, 1)
    source += "\nfrom math import fsum as license\n"
    with pytest.raises(ValueError, match="at import"):
        world[0].validate_source(source)


def test_existing_method_header_and_module_import_alias_remain_accepted(world):
    from numerical_agent.evolution.module import MODULE_HEADER
    source = MODULE_HEADER + "\nfrom math import fsum as total\n" + SOURCE.replace(
        "float(history[-1]) + 1.0", "total(history) / len(history)")
    outcome = world[0].forecast_source(source, "seasonal_naive", world[0].tasks[0])
    assert outcome.status == "passed" and outcome.forecast == (3.5, 3.5)


@pytest.mark.parametrize("statement", ["nonlocal missing_binding", "value = 1\n    global value"])
def test_invalid_lexical_binding_is_a_source_validation_error(world, statement):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        statement + "\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)


@pytest.mark.parametrize("statement", [
    'label = f"{(lambda: 0)}"',
    'label = f"{(lambda: 0)!r}"',
    'label = f"{history[-1]}"',
    'label = "%s" % (lambda: 0)',
    'template = "%r"\n    label = template % (lambda: 0)',
    'label = b"%r" % (lambda: 0)',
    'label = "{}".format(lambda: 0)',
    'formatter = "{}".format\n    label = formatter(lambda: 0)',
    'import numpy as np\n    label = np.mod("%s", lambda: 0)',
    'from numpy import mod as formatter\n    label = formatter("%s", lambda: 0)',
])
def test_implicit_object_formatting_cannot_derive_function_addresses(world, statement):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon", statement + '''
    address = int(label.split("0x")[1].split(">")[0], 16)
    return [float(address)] * horizon''')
    with pytest.raises(ValueError):
        world[0].validate_source(source)


def test_numeric_remainders_remain_available_without_formatting(world):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon", '''import math
    _, remainder = divmod(int(history[-1]), 3)
    value = math.fmod(float(remainder), 2.0)
    return [value] * horizon''')
    outcome = world[0].forecast_source(source, "seasonal_naive", world[0].tasks[0])
    assert outcome.status == "passed" and outcome.forecast == (1.0, 1.0)


def test_fresh_process_materialization_preserves_cache_identity_and_evaluation():
    import os
    import subprocess
    import sys
    command = '''import json
from evolving_loop.v2.numerical_qd.adapters import evaluate_numerical_child
from evolving_loop.v2.numerical_qd.hyperband import evaluation_cache_key, _task_evaluation
from tests.test_evolution_v2_numerical_adapters import world, materialize, train_manifest, descriptor_policy
fixture = world.__wrapped__()
adapter = fixture[0]
child = materialize(fixture)
task = train_manifest(adapter).tasks[0]
evaluation = evaluate_numerical_child(adapter, child, task, descriptor_policy=descriptor_policy(),
    metric_policy_sha256="6" * 64, bracket="explore", rung=0)
_task_evaluation(evaluation, child.genome.fingerprint(), task, "6" * 64,
    descriptor_policy().fingerprint(), child.genome.runtime_fingerprints, adapter.fingerprint)
key = evaluation_cache_key(child.genome.fingerprint(), task.task_sha256, task.split_sha256,
    "6" * 64, descriptor_policy().fingerprint(), child.genome.runtime_fingerprints,
    task.protocol_sha256, adapter.fingerprint)
print(json.dumps(dict(cache_key=key, evaluation=evaluation.to_payload()), sort_keys=True))
'''
    results = []
    for seed in ("7", "777"):
        completed = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True,
            check=True, timeout=30, cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED=seed))
        results.append(json.loads(completed.stdout))
    assert results[0] == results[1]
    assert len(results[0]["cache_key"]) == 64
    assert len(results[0]["evaluation"]["task_statuses"]) == 1
    assert set(results[0]["evaluation"]["task_statuses"].values()) == {"passed"}


@pytest.mark.parametrize("statement", [
    'label = np.asarray(lambda: 0, dtype="U100").tolist()',
    'label = np.asarray(lambda: 0, "U100").tolist()',
    'from numpy import asarray as convert\n    label = convert(lambda: 0, "U100").tolist()',
    'convert = np.asarray\n    label = convert(lambda: 0, "U100").tolist()',
    'label = np.asarray(lambda: 0, dtype="U" + "100").tolist()',
    'label = np.asarray(lambda: 0, **{"dtype": "U100"}).tolist()',
    'label = np.asarray(*((lambda: 0), "U100")).tolist()',
    'label = np.asarray(lambda: 0).astype("U100").tolist()',
    'label = np.full_like(np.asarray(" " * 100), lambda: 0).tolist()',
    'values = np.asarray([" " * 100])\n    values[0] = lambda: 0\n    label = values[0]',
    'values = np.asarray([" " * 100])\n    values.fill(lambda: 0)\n    label = values[0]',
    'float = "U100"\n    label = np.asarray(lambda: 0, dtype=float).tolist()',
    'label = np.pad(np.asarray([" " * 100]), (1, 0), constant_values=lambda: 0)[0]',
    'label = np.take(np.asarray([lambda: 0]), [0], None, np.asarray([" " * 100]))[0]',
    'label = np.concatenate((np.asarray([lambda: 0]),), out=np.asarray([" " * 100]), casting="unsafe")[0]',
    'label = np.concatenate((np.asarray([lambda: 0]),), 0, np.asarray([" " * 100]), **{"casting": "unsafe"})[0]',
])
def test_numpy_object_string_coercion_cannot_format_process_addresses(world, statement):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon", "import numpy as np\n    " + statement + '''
    address = int(label.split("0x")[1].split(">")[0], 16)
    return [address * 1.0] * horizon''')
    with pytest.raises(ValueError):
        world[0].validate_source(source)


@pytest.mark.parametrize("conversion", ["np.asarray(history, dtype=float)", "np.asarray(history, 'float64')",
    "np.array(history, np.float64)", "np.asarray(history, dtype=None)"])
def test_explicit_numeric_array_dtypes_remain_executable(world, conversion):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon", "import numpy as np\n    values = "
        + conversion + "\n    return np.full(horizon, np.mean(values)).tolist()")
    outcome = world[0].forecast_source(source, "seasonal_naive", world[0].tasks[0])
    assert outcome.status == "passed" and outcome.forecast == (3.5, 3.5)


# Independent public-behavior cases for the intentionally small NumPy surface.
# The last argument list reaches the permitted positional boundary, so adding
# one more argument must reject instead of reaching an output/casting slot.
_NUMPY_CALL_CASES = [
    ("array", "[1.0, 3.0], None", 2.0),
    ("asarray", "[1.0, 3.0], None", 2.0),
    ("full", "2, 2.0, None", 2.0),
    ("zeros", "2, None", 0.0),
    ("ones", "2, None", 1.0),
    ("mean", "[1.0, 3.0], None, None", 2.0),
    ("min", "[1.0, 3.0], None", 1.0),
    ("max", "[1.0, 3.0], None", 3.0),
    ("sum", "[1.0, 3.0], None, None", 4.0),
    ("std", "[1.0, 3.0], None, None", 1.0),
    ("var", "[1.0, 3.0], None, None", 1.0),
    ("median", "[1.0, 3.0], None", 2.0),
    ("bool_", "1", 1.0),
    ("int32", "2", 2.0),
    ("int64", "2", 2.0),
    ("float32", "2.0", 2.0),
    ("float64", "2.0", 2.0),
]


def test_numpy_max_positional_output_address_reproducer_rejects(world):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon", '''import numpy as np
    values = np.asarray([lambda: 0])
    output = np.asarray(" " * 100)
    label = np.max(values, None, output).tolist()
    address = int(label.split("0x")[1].split(">")[0], 16)
    return [address * 1.0] * horizon''')
    with pytest.raises(ValueError):
        world[0].validate_source(source)


@pytest.mark.parametrize("name, arguments, expected", _NUMPY_CALL_CASES)
@pytest.mark.parametrize("form", ["direct", "import_alias", "assignment_alias", "star", "keywords"])
def test_numpy_signature_rejects_positional_overflow_and_hidden_arguments(world, name, arguments, expected, form):
    setup = "import numpy as np"
    if form == "import_alias":
        setup += f"\n    from numpy import {name} as operation"
        call = f"operation({arguments}, None)"
    elif form == "assignment_alias":
        setup += f"\n    operation = np.{name}"
        call = f"operation({arguments}, None)"
    elif form == "star":
        call = f"np.{name}(*({arguments}, None))"
    elif form == "keywords":
        call = f"np.{name}({arguments}, **{{'out': None}})"
    else:
        call = f"np.{name}({arguments}, None)"
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        setup + "\n    value = " + call + "\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)


@pytest.mark.parametrize("name, arguments, expected", _NUMPY_CALL_CASES)
@pytest.mark.parametrize("keyword", ["out", "casting", "dtype", "like", "order", "subok",
    "signature", "where", "device", "overwrite_input", "unknown_future_option"])
def test_numpy_signature_rejects_unreviewed_semantic_keywords(world, name, arguments, expected, keyword):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        f"import numpy as np\n    value = np.{name}({arguments}, {keyword}='U100')\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)


@pytest.mark.parametrize("name, arguments, expected", _NUMPY_CALL_CASES)
@pytest.mark.parametrize("form", ["direct", "import_alias"])
def test_numpy_signature_numeric_controls_execute(world, name, arguments, expected, form):
    setup = "import numpy as np"
    operation = "np." + name
    if form == "import_alias":
        setup += f"\n    from numpy import {name} as operation"
        operation = "operation"
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon", setup
        + f"\n    value = {operation}({arguments})\n    return [float(np.mean(value))] * horizon")
    outcome = world[0].forecast_source(source, "seasonal_naive", world[0].tasks[0])
    assert outcome.status == "passed" and outcome.forecast == (expected, expected)


@pytest.mark.parametrize("statement", [
    "module = np\n    value = module.max(history, None, None)",
    "operation = np.max\n    value = operation(history, None, None)",
    "operations = [np.max]\n    value = operations[0](history, None, None)",
    "value = list(map(np.max, [history]))",
    "def invoke(operation):\n        return operation(history, None, None)\n    value = invoke(np.max)",
    "value = (lambda module: module.max(history, None, None))(np)",
    "value = np.asarray(history).max(None, None)",
    "operation = np.asarray(history).max\n    value = operation(None, None)",
    "value = np.max(*(history,))",
    "value = np.max(history, **{})",
    "from numpy import max as operation\n    value = operation(*(history, None, None))",
    "from numpy import max as operation\n    value = operation(history, **{'out': None})",
    "value = np.concatenate((np.asarray(history),), 0, None)",
    "value = np.add(history, history, None)",
    "value = np.add.reduce(history, None, None, None)",
    "value = np.linalg.norm(history)",
    "from numpy.linalg import norm\n    value = norm(history)",
    "from numpy import float64 as numeric_type\n    numeric_type = 'U100'\n    value = np.asarray(lambda: 0, dtype=numeric_type)",
    "np = {'dtype': 'U100'}\n    value = np.get('dtype')",
    "def helper(np):\n        return np.max(history, None, None)\n    value = helper(history)",
])
def test_numpy_signature_unresolved_aliases_and_unreviewed_operations_reject(world, statement):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        "import numpy as np\n    " + statement + "\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)


@pytest.mark.parametrize("expression, expected", [
    ("np.mean(a=history, axis=None, dtype=float, keepdims=True)", 3.5),
    ("np.max(a=history, axis=None, keepdims=True)", 7.0),
    ("np.asarray(history, dtype=numeric_type)", 3.5),
    ("np.full(shape=horizon, fill_value=2.0, dtype=float)", 2.0),
])
def test_numpy_signature_reviewed_keywords_and_imported_dtype_execute(world, expression, expected):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        "import numpy as np\n    from numpy import float64 as numeric_type\n    value = "
        + expression + "\n    return [float(np.mean(value))] * horizon")
    outcome = world[0].forecast_source(source, "seasonal_naive", world[0].tasks[0])
    assert outcome.status == "passed" and outcome.forecast == (expected, expected)


@pytest.mark.parametrize("name, arguments", [
    ("array", "[lambda: 0], 'U100'"),
    ("asarray", "[lambda: 0], 'U100'"),
    ("full", "1, lambda: 0, 'U100'"),
    ("zeros", "1, 'U100'"),
    ("ones", "1, 'U100'"),
    ("mean", "[lambda: 0], None, 'U100'"),
    ("sum", "[lambda: 0], None, 'U100'"),
    ("std", "[lambda: 0], None, 'U100'"),
    ("var", "[lambda: 0], None, 'U100'"),
    ("mean", "[lambda: 0], None, None, np.asarray(' ' * 100)"),
    ("min", "[lambda: 0], None, np.asarray(' ' * 100)"),
    ("max", "[lambda: 0], None, np.asarray(' ' * 100)"),
    ("sum", "[lambda: 0], None, None, np.asarray(' ' * 100)"),
    ("std", "[lambda: 0], None, None, np.asarray(' ' * 100)"),
    ("var", "[lambda: 0], None, None, np.asarray(' ' * 100)"),
    ("median", "[lambda: 0], None, np.asarray(' ' * 100)"),
])
@pytest.mark.parametrize("form", ["direct", "import_alias"])
def test_numpy_signature_all_positional_dtype_and_output_slots_reject(world, name, arguments, form):
    setup = "import numpy as np"
    operation = "np." + name
    if form == "import_alias":
        setup += f"\n    from numpy import {name} as operation"
        operation = "operation"
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        setup + f"\n    value = {operation}({arguments})\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)


@pytest.mark.parametrize("statement", [
    "values = np.asarray(history)\n    value = values.tolist(None)",
    "values = np.asarray(history)\n    value = values.tolist(*())",
    "values = np.asarray(history)\n    value = values.tolist(**{})",
    "operation = np.asarray(history).tolist\n    value = operation()",
    "try:\n        raise ValueError('bad')\n    except ValueError as np:\n        value = 1.0",
    "value = [np for np in history]",
    "match history:\n        case [np, *rest]:\n            value = 1.0",
    "import math as np\n    value = np.mean(history)",
    "def helper():\n        from math import fsum as operation\n        return operation(history)\n    from numpy import max as operation\n    value = operation(history)",
    "value = np.float64(dtype='U100')",
    "value = np.asarray(history, None, dtype=float)",
])
def test_numpy_signature_methods_and_all_import_binding_forms_reject(world, statement):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        "import numpy as np\n    " + statement + "\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)


@pytest.mark.parametrize("statement", ["from . import numpy as np", "from .numpy import max as operation"])
def test_numpy_signature_relative_imports_reject_as_validation_errors(world, statement):
    source = SOURCE.replace("return [float(history[-1]) + 1.0] * horizon",
        statement + "\n    return [1.0] * horizon")
    with pytest.raises(ValueError):
        world[0].validate_source(source)
