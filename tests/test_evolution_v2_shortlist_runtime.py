"""Production P2 constructor, seed and child execution over a complete catalog."""
import hashlib
from dataclasses import asdict, replace
from types import SimpleNamespace
import pytest

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
from numerical_agent.evolution.screening import (
    ApplicabilityClause,
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
)
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


def production_world(tmp_path, *, dictionary=None, host_screening=None):
    config, seed, folds, original = fixture(raw_seed=True)
    names = ("seasonal_naive", "select_seasonal_naive") + tuple(f"method_{i:02d}" for i in range(12))
    dictionary = dictionary or FilterDictionary((FilterEntry("toto_2_0", "tsfm", "keep", (), "reviewed"),) + tuple(
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
    # Evidence authorizes the exact fitted policy which unchanged child replay
    # will execute, including all five task-specific cross-fit policies.
    from evolving_loop.package_numerical_evolution import fit_numerical_recipe
    from evolving_loop.v2.numerical_qd.adapters import _verified_build_rows
    from numerical_agent.evolution.champion import parse_champion_recipe, parse_champion_release
    state, _, policies, _, _ = _bootstrap(config, seed, original)
    recipe = parse_champion_recipe(policies[state.inventory.members[0].policy_sha256])
    anchor = parse_champion_release(seed.to_payload()["anchor_release_payload"])
    fitted = fit_numerical_recipe(recipe, _verified_build_rows(_train_rows(original), original.tasks,
        folds, recipe, anchor, CatalogStore()), folds, anchor)
    fitted_spec = next(spec for spec in original.materializer._release(seed, fitted, version="n001", anchor=anchor).alternatives
                       if spec.candidate_id == recipe.name)
    seed = replace(seed, alternatives=tuple(fitted_spec if spec.candidate_id == recipe.name else spec for spec in seed.alternatives),
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
    if host_screening is not None:
        host.screening_policy = host_screening
    adapter = cli._build_numerical_adapter(config, original.tasks, folds,
        {"config": "0" * 64, "seed_supply": "1" * 64, "task_manifest": "2" * 64}, host_runtime=host,
        seed_release=seed, task_local_evidence_path=tmp_path, task_local_dictionary=dictionary)
    index = dict(adapter.task_local_evidence.index)
    raw[fingerprint_payload(index)] = canonical_v2_bytes(index)
    return config, seed, folds, adapter, store, expected, raw


def test_schema_two_adapter_uses_exact_host_screening_and_rejects_mismatch(
    tmp_path
):
    """Catches flattening rich Host any-of applicability into one clause."""
    names = ("seasonal_naive", "select_seasonal_naive") + tuple(
        f"method_{index:02d}" for index in range(12)
    )
    dictionary = FilterDictionary(
        (FilterEntry("toto_2_0", "tsfm", "keep", (), "reviewed"),)
        + tuple(
            FilterEntry(
                name,
                "statistical",
                "specialized" if name == "seasonal_naive" else "keep",
                ("frequency:1 second",) if name == "seasonal_naive" else (),
                "reviewed",
            )
            for name in names
        )
    )
    screening = ScreeningPolicy(
        (
            ScreeningEntry(
                "toto_2_0", "tsfm", "keep", ApplicabilityPolicy(), "reviewed"
            ),
        )
        + tuple(
            ScreeningEntry(
                name,
                "statistical",
                "specialized" if name == "seasonal_naive" else "keep",
                ApplicabilityPolicy(
                    (
                        ApplicabilityClause(("frequency:1 second",)),
                        ApplicabilityClause(("frequency:D",)),
                    )
                )
                if name == "seasonal_naive"
                else ApplicabilityPolicy(),
                "reviewed",
            )
            for name in names
        ),
        ("toto_2_0",),
    )
    config, seed, folds, adapter, store, _expected, _raw = production_world(
        tmp_path, dictionary=dictionary, host_screening=screening
    )

    assert adapter.materializer.screening_policy is screening
    registry = _seed_registry(seed, adapter)
    assert len(registry.task_ids) == 100

    mismatched = ScreeningPolicy(
        tuple(
            replace(entry, reason="changed provenance")
            if entry.name == "seasonal_naive"
            else entry
            for entry in screening.entries
        ),
        screening.fallback_names,
    )
    with pytest.raises(ValueError, match="Host screening policy differs"):
        cli._build_numerical_adapter(
            config,
            adapter.tasks,
            folds,
            {
                "config": "0" * 64,
                "seed_supply": "1" * 64,
                "task_manifest": "2" * 64,
            },
            host_runtime=SimpleNamespace(
                forecast_store=store,
                sources=adapter.sources,
                screening_policy=mismatched,
            ),
            seed_release=seed,
            task_local_evidence_path=tmp_path,
            task_local_dictionary=dictionary,
        )


def test_same_id_policy_change_requires_new_evidence_before_child_execution(tmp_path, monkeypatch):
    config, seed, folds, adapter, store, expected, raw = production_world(tmp_path)
    state, genome, policies, _, _ = _bootstrap(config, seed, adapter)
    # Replacing the specialist's executable parent with the Anchor changes its
    # forecast, while retaining its candidate ID and old successful folds.
    spec = next(item for item in seed.alternatives if item.candidate_id == "select_seasonal_naive")
    policy = adapter.materializer._stored_policy_for_task(spec, adapter.tasks[-1].numeric.task_id)
    changed_recipe = replace(policy.recipe, parents=("toto_2_0",), fallback_parent="toto_2_0",
        assumptions=tuple(replace(item, candidate_name="toto_2_0") for item in policy.recipe.assumptions))
    changed_policy = replace(policy, recipe=changed_recipe)
    changed = replace(spec, recipe_payload=changed_recipe.to_payload(),
        full_build_policy_payload=changed_policy.to_payload(),
        build_fold_policy_payloads=tuple((fold, replace(changed_policy,
            thresholds=tuple((key, float(fold + 1)) for key, _ in changed_policy.thresholds)).to_payload()) for fold in range(5)))
    original_release = type(adapter.materializer)._release
    def changed_release(self, *args, **kwargs):
        release = original_release(self, *args, **kwargs)
        return replace(release, alternatives=tuple(changed if item.candidate_id == changed.candidate_id else item for item in release.alternatives),
            anchor_release_payload=release.to_payload()["anchor_release_payload"])
    monkeypatch.setattr(type(adapter.materializer), "_release", changed_release)
    monkeypatch.setattr(adapter, "materialize_local_package", lambda *_args, **_kwargs: pytest.fail("changed policy reached old diagnostics"))
    with pytest.raises(ValueError, match="refreshed Task 4 evidence"):
        adapter.materialize_child(seed, genome, state, member_id=state.inventory.members[0].member_id,
            policies=policies, build_rows=_train_rows(adapter), descriptor_policy=config.descriptor_policy, version="n001")


def test_failed_shortlisted_specialist_keeps_exact_anchor_and_reloads(tmp_path):
    config, seed, folds, adapter, store, expected, raw = production_world(tmp_path)
    original = store.forecast
    def fail_one(name, *args):
        if name == "seasonal_naive":
            raise RuntimeError("specialist unavailable")
        return original(name, *args)
    store.forecast = fail_one
    registry = _seed_registry(seed, adapter)
    from evolving_loop.v2.numerical_qd.adapters import import_numerical_seed
    imported = import_numerical_seed(seed, registry, tasks=adapter.tasks, evidence=adapter.task_local_evidence)
    validate_frozen_local_evidence(seed, imported.envelope, artifact_bytes_by_sha=raw)
    restored = imported.envelope.restore(adapter.tasks)
    for task in adapter.tasks:
        package = restored.package_for(task)
        assert package.final_forecast == package.protected_baseline.forecast == (14.0, 14.0)
        assert package.selection_decision.selected == ("toto_2_0",)
        assert package.selection_decision.rejected["seasonal_naive"] == "shortlisted_runtime_failure"
        assert "seasonal_naive" not in {item.name for item in package.ranked_alternatives}


def test_local_materialization_uses_fallback_aware_active_dictionary(tmp_path, monkeypatch):
    """Catches rechecking a producer-admitted fallback with raw applicability."""
    _config, seed, _folds, adapter, store, _expected, _raw = production_world(
        tmp_path
    )
    task = adapter.tasks[0]
    original_shortlist, original_payload, _shortlist_sha, _diagnostic_sha = (
        adapter.local_evidence_for(task)
    )
    selected = ("toto_2_0", "seasonal_naive", "method_00")
    shortlist = replace(
        original_shortlist,
        candidate_names=selected,
        exclusion_reasons=(),
        shortlist_underfilled=True,
    )
    payload = dict(original_payload)
    payload["rows"] = [
        row for row in original_payload["rows"] if row["candidate_name"] in selected
    ]
    impossible = ApplicabilityPolicy(
        (ApplicabilityClause(("integer_valued", "continuous_valued")),)
    )
    adapter.materializer.screening_policy = ScreeningPolicy(
        (
            ScreeningEntry(
                "toto_2_0", "tsfm", "keep", ApplicabilityPolicy(), "anchor"
            ),
            ScreeningEntry(
                "seasonal_naive",
                "statistical",
                "specialized",
                impossible,
                "reviewed fallback",
            ),
            ScreeningEntry(
                "method_00",
                "statistical",
                "keep",
                ApplicabilityPolicy(),
                "active",
            ),
            ScreeningEntry(
                "method_01",
                "statistical",
                "quarantine",
                ApplicabilityPolicy(),
                "inactive",
            ),
        ),
        ("toto_2_0", "seasonal_naive"),
    )
    monkeypatch.setattr(
        adapter,
        "local_evidence_for",
        lambda _task: (
            shortlist,
            payload,
            "1" * 64,
            hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        ),
    )

    package = adapter.materialize_local_package(task, seed, store.forecast)

    assert "seasonal_naive" in {
        candidate.name for candidate in package.ranked_alternatives
    }

    inactive_names = ("toto_2_0", "method_00", "method_01")
    inactive_shortlist = replace(shortlist, candidate_names=inactive_names)
    inactive_payload = dict(original_payload)
    inactive_payload["rows"] = [
        row
        for row in original_payload["rows"]
        if row["candidate_name"] in inactive_names
    ]
    monkeypatch.setattr(
        adapter,
        "local_evidence_for",
        lambda _task: (
            inactive_shortlist,
            inactive_payload,
            "3" * 64,
            "4" * 64,
        ),
    )
    with pytest.raises(ValueError, match="not eligible in the Host Dictionary"):
        adapter.materialize_local_package(task, seed, store.forecast)


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
