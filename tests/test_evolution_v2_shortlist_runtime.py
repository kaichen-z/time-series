"""Production P2 constructor, seed and child execution over a complete catalog."""
import hashlib
from dataclasses import asdict, replace
from types import SimpleNamespace

from common.payload import canonical_json_bytes
from evolving_loop.v2 import cli
from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
from evolving_loop.v2.numerical_qd.adapters import freeze_qd_supply, validate_frozen_local_evidence
from evolving_loop.v2.numerical_qd.contracts import ConstraintReportV2, NumericalQDEntryV2, NumericalObjectiveVectorV2
from evolving_loop.v2.numerical_qd.descriptors import describe_history
from evolving_loop.v2.numerical_qd.map_elites import NumericalQDArchive
from evolving_loop.v2.numerical_qd.runner import _bootstrap, _seed_registry, _train_rows
from numerical_agent.evolution.filtering import FilterDictionary, FilterEntry
from numerical_agent.evolution.task_shortlist import TaskCandidateShortlistV1, TaskShortlistPolicyV1, _dictionary_hash
from numerical_agent.run_task_local_ensemble_evolution import task_input_sha256
from numerical_agent.evolution.execution import Task
from tests.test_evolution_v2_numerical_runner import fixture
from tests.test_package_numerical_supply import _alternative, _policy
from tests.test_task_local_ensemble import _diagnostic


class CatalogStore:
    identity_hash = "8" * 64
    def __init__(self):
        self.calls = []
    def forecast(self, name, history, horizon, frequency):
        self.calls.append((name, tuple(history)))
        if name == "seasonal_naive":
            return (float(history[-1]) + 1.0,) * horizon
        return (14.0 if name == "toto_2_0" else 10.0,) * horizon


def production_world(tmp_path):
    config, seed, folds, original = fixture(raw_seed=True)
    names = ("seasonal_naive", "select_seasonal_naive") + tuple(f"method_{i:02d}" for i in range(12))
    dictionary = FilterDictionary((FilterEntry("toto_2_0", "tsfm", "keep", (), "reviewed"),) + tuple(
        FilterEntry(name, "statistical", "keep", (), "reviewed") for name in names))
    dictionary_sha = _dictionary_hash(dictionary)
    alternatives = tuple(_alternative(name, "statistical") for name in names)
    alias = _policy("seasonal_naive")
    alternatives = tuple(replace(spec, materializer_kind="champion", recipe_payload=alias.recipe.to_payload(),
        full_build_policy_payload=alias.to_payload(), build_fold_policy_payloads=tuple(
            (fold, _policy("seasonal_naive", float(fold + 1)).to_payload()) for fold in range(5)),
        assumption_ids=("seasonal_naive_ready",)) if spec.candidate_id == "select_seasonal_naive" else spec for spec in alternatives)
    seed = replace(seed, schema_version=2, alternatives=alternatives,
        source_fingerprints=dict(seed.source_fingerprints) | {"dictionary": dictionary_sha},
        anchor_release_payload=seed.to_payload()["anchor_release_payload"])
    policy = TaskShortlistPolicyV1()
    entries, raw, expected = [], {}, {}
    def write(relative, payload):
        data = canonical_json_bytes(payload)
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        raw[sha] = data
        return sha
    write("task_shortlist_policy.json", policy.to_payload())
    for task in original.tasks:
        numeric = task.numeric
        offset = 0 if len(set(numeric.history_values)) == 1 else 6
        chosen = ("toto_2_0", "seasonal_naive", "select_seasonal_naive") + tuple(f"method_{i:02d}" for i in range(offset, offset + 5))
        expected[numeric.task_id] = chosen
        input_sha = task_input_sha256(Task(numeric.task_id, numeric.history_values, numeric.prediction_length, numeric.frequency, ()))
        shortlist = TaskCandidateShortlistV1(1, input_sha, dictionary_sha, policy.fingerprint(), chosen,
            tuple((name, "ranked_out") for name in names if name not in chosen), False, False)
        diagnostics = {"schema_version": 1, "task_id": numeric.task_id, "task_input_sha256": input_sha,
            "rows": [{"candidate_name": name, "failure_reason": None,
                "diagnostic": asdict(_diagnostic(name, ((14.0, 14.0) if name == "toto_2_0" else (10.0, 10.0),) * 3,
                    family="tsfm" if name == "toto_2_0" else "statistical"))} for name in sorted(chosen)],
            "public_test_accessed": False}
        entries.append({"task_id": numeric.task_id, "task_input_sha256": input_sha,
            "shortlist_sha256": write(f"task_shortlists/{input_sha}.json", shortlist.to_payload()),
            "diagnostics_sha256": write(f"task_diagnostics/{input_sha}.json", diagnostics)})
    write("task_shortlist_index.json", {"schema_version": 1, "policy_sha256": policy.fingerprint(),
        "entries": entries, "public_test_accessed": False})
    store = CatalogStore()
    config = replace(config, profile="pilot")
    host = SimpleNamespace(forecast_store=store, sources=original.sources)
    adapter = cli._build_numerical_adapter(config, original.tasks, folds,
        {"config": "0" * 64, "seed_supply": "1" * 64, "task_manifest": "2" * 64}, host_runtime=host,
        seed_release=seed, task_local_evidence_path=tmp_path, task_local_dictionary=dictionary)
    index = dict(adapter.task_local_evidence.index)
    raw[fingerprint_payload(index)] = canonical_v2_bytes(index)
    return config, seed, folds, adapter, store, expected, raw


def test_production_seed_child_and_nonempty_freeze_keep_shortlist_result(tmp_path, monkeypatch):
    config, seed, folds, adapter, store, expected, raw = production_world(tmp_path)
    registry = _seed_registry(seed, adapter)
    assert len(seed.alternatives) == 14
    assert len(set(expected.values())) == 2
    assert len(store.calls) == 700
    for task, offset in zip(adapter.tasks, range(0, 700, 7)):
        assert tuple(name for name, _ in store.calls[offset:offset + 7]) == tuple(
            name for name in expected[task.numeric.task_id] if name != "select_seasonal_naive")
    state, genome, policies, _, _ = _bootstrap(config, seed, adapter)
    store.calls.clear()
    package_calls = []
    original_materialize = adapter.materialize_local_package
    def observe(task, release, forecast):
        def traced(name, *args):
            package_calls.append(name)
            return forecast(name, *args)
        return original_materialize(task, release, traced)
    monkeypatch.setattr(adapter, "materialize_local_package", observe)
    child = adapter.materialize_child(seed, genome, state, member_id=state.inventory.members[0].member_id,
        policies=policies, build_rows=_train_rows(adapter), descriptor_policy=config.descriptor_policy, version="n001")
    # The separate Train fitting stage forecasts its two recipe inputs. Local package
    # execution itself contains exactly the eight authorized names per task.
    for task, offset in zip(adapter.tasks, range(0, 700, 7)):
        assert tuple(package_calls[offset:offset + 7]) == tuple(
            name for name in expected[task.numeric.task_id] if name != "select_seasonal_naive")
    cell = describe_history(adapter.tasks[0].numeric.history_values, 2, "D", "statistical", config.descriptor_policy)
    entry = NumericalQDEntryV2(1, genome.fingerprint(), "7" * 64, cell,
        (adapter.tasks[0].numeric.task_id,), NumericalObjectiveVectorV2(1., 1., 1., 1., 1.), ConstraintReportV2(True, ()), ())
    archive = NumericalQDArchive().insert((entry,))
    frozen = freeze_qd_supply(adapter, seed, registry, archive, (child,), descriptor_policy=config.descriptor_policy,
        version="n002", required_genome_sha256=genome.fingerprint())
    assert frozen.selected_genome_sha256s == (genome.fingerprint(),)
    assert frozen.envelope.schema_version == 2
    from evolving_loop.v2.numerical_qd.adapters import import_numerical_seed
    from evolving_loop.v2.numerical_qd.runner import run_numerical_qd
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
    imported = import_numerical_seed(frozen.release, frozen.registry, tasks=adapter.tasks,
        evidence=adapter.task_local_evidence)
    bounded = replace(config, budget=replace(config.budget,
        ceilings=replace(config.budget.ceilings, task_executions=1)))
    run_root = tmp_path / "p2-frozen-run"
    run_numerical_qd(run_root, bounded, imported, folds, adapter)
    def forbidden(*args, **kwargs):
        raise AssertionError("P3 restore executed Numerical work")
    monkeypatch.setattr(store, "forecast", forbidden)
    validate_frozen_local_evidence(frozen.release, frozen.envelope, artifact_bytes_by_sha=raw)
    loaded, _ = NumericalQDRunStore(run_root).load_active_frozen_pair(tasks=adapter.tasks)
    restored = loaded.registry
    assert restored.fingerprint == frozen.registry.fingerprint
    for task in adapter.tasks:
        package = restored.package_for(task)
        assert package.selection_decision.weights[0] >= 0.5
        assert 2 <= len(package.selection_decision.selected) <= 3
        assert package.final_forecast == (12.0, 12.0)
        assert package == child.candidate.registry.package_for(task) or package.selection_decision == child.candidate.registry.package_for(task).selection_decision
