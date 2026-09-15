"""End-to-end Kernel authority with real mutation, QD, ranking and legacy fitting."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from common.llm import LLMResponse
from evolving_loop.package_numerical_evolution import NumericalPackageMaterializer
from evolving_loop.package_registry import task_registry_fingerprint
from evolving_loop.v2.budget import ResourceUse
from evolving_loop.v2.fakes import FakeClock
from evolving_loop.v2.kernel import EvolutionKernel, KernelAuthorityError
from evolving_loop.v2.numerical_qd.adapters import LegacyNumericalAdapter
from evolving_loop.v2.numerical_qd.config import NumericalQDConfigV2
from evolving_loop.v2.numerical_qd.runner import run_numerical_qd
from evolving_loop.v2.store import V2RunStore
from numerical_agent.evolution.screening import ScreeningPolicy
from numerical_agent.evolution.task_local_evolution import build_group_fold_manifest
from tests.test_evolution_v2_numerical_adapters import CheapStore, SOURCE, screening_policy
from tests.test_evolution_v2_numerical_config import valid_config_payload
from tests.test_package_numerical_evolution import _evolution_tasks, _supply_parent
from tests.test_evolution_v2_kernel import kernel, child
from tests.test_evolution_v2_numerical_artifacts import MATERIAL_KINDS, CONTROL_KINDS


@pytest.fixture
def material_catalog(monkeypatch):
    from evolving_loop.v2.numerical_qd.runner import _MaterialAccounting
    original, catalog = _MaterialAccounting.write, {}
    def write(accounting, relative, data, dispatch, *, kind):
        assert kind.value in MATERIAL_KINDS | CONTROL_KINDS
        path = accounting.store.directory / relative
        try:
            return original(accounting, relative, data, dispatch, kind=kind)
        finally:
            if path.is_file():
                if path in catalog:
                    assert catalog[path] == kind.value
                catalog[path] = kind.value
    monkeypatch.setattr(_MaterialAccounting, "write", write)
    return catalog


def material_sizes(root, catalog):
    return [path.stat().st_size for path, kind in catalog.items()
            if path.is_relative_to(root) and kind in MATERIAL_KINDS]


def fixture(seed=7, *, task_budget=1840, provider="deterministic", clock=None,
            raw_seed=False, reverse_entities=False, operator_input_sha256s=None):
    tasks = []
    for i, task in enumerate(_evolution_tasks()):
        history = (float(i),) * 20 if i < 40 else task.numeric.history_values
        tasks.append(replace(task, numeric=replace(task.numeric,
            history_values=history, future_values=(history[-1] + 1.0,) * 2,
            entity_name=f"entity-{(99 - i) // 2 if reverse_entities else i // 2:03d}")))
    tasks = tuple(tasks)
    folds = build_group_fold_manifest(tuple(t.numeric for t in tasks[:80]), seed=20260903)
    screen = screening_policy()
    screen = ScreeningPolicy(tuple(entry for entry in screen.entries if entry.name != "lagged"), screen.fallback_names)
    source = SOURCE.split("\ndef lagged")[0]
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    materializer = NumericalPackageMaterializer(forecast_store=CheapStore(), screening_policy=screen,
        fold_manifest=folds, original_tasks=tasks, source_fingerprints={"dictionary": source_sha},
        runtime_fingerprints={"materializer": "3" * 64})
    adapter = LegacyNumericalAdapter(materializer=materializer, tasks=tasks,
        fold_manifest=folds, sources={source_sha: source},
        operator_input_sha256s=operator_input_sha256s)
    adapter.monotonic = clock or FakeClock()
    payload = valid_config_payload()
    payload.update(profile="smoke", seed=seed, runtime_fingerprints={"materializer": "3" * 64})
    payload["mutation"]["operators"] = ["repair"]
    payload["proposer"].update(provider=provider, max_proposals_per_generation=3)
    payload["budget"]["ceilings"]["task_executions"] = task_budget
    payload["adapter"]["task_timeout_seconds"] = 0.01
    supply = _supply_parent()
    if not raw_seed:
        from evolving_loop.v2.numerical_qd.runner import _seed_registry
        from evolving_loop.v2.numerical_qd.adapters import import_numerical_seed
        supply = import_numerical_seed(supply, _seed_registry(supply, adapter), tasks=adapter.tasks)
    return NumericalQDConfigV2.from_payload(payload), supply, folds, adapter


def files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def run_fixture(root, **kwargs):
    options = {key: kwargs.pop(key) for key in tuple(kwargs) if key in {
        "seed", "task_budget", "provider", "clock", "operator_input_sha256s"
    }}
    return run_numerical_qd(root, *fixture(**options), **kwargs)


def force_dev_results(monkeypatch, *outcomes):
    """Keep non-promotion tests focused on the authority behavior they cover."""
    from evolving_loop.v2.numerical_qd import runner

    remaining = iter(outcomes)
    def compare(_parent, _child, _adapter, _kernel, account_work):
        for _ in range(40):
            account_work()
        passed = next(remaining, False)
        return {
            "passed": passed,
            "parent_metrics": {"mean_smae": 1.0, "mean_srmse": 1.0},
            "candidate_metrics": {
                "mean_smae": 0.5 if passed else 1.0,
                "mean_srmse": 0.5 if passed else 1.0,
            },
        }
    monkeypatch.setattr(runner, "_dev_compare", compare)


def test_llm_proposal_budget_is_independent_of_numerical_task_timeout():
    from evolving_loop.v2.numerical_qd.runner import _proposal_budget

    available = ResourceUse(wall_seconds=500.0, llm_calls=2,
        input_tokens=1000, output_tokens=1000)

    assert _proposal_budget(available).wall_seconds == 60.0


def test_runner_reusable_context_requires_feasible_store_verified_programs():
    from evolving_loop.v2.numerical_qd.agent_methods import CurriculumTargetV2
    from evolving_loop.v2.numerical_qd.contracts import (
        ConstraintReportV2, MorphologyCellV2, NumericalObjectiveVectorV2,
        NumericalQDEntryV2, TrainMutationFeedbackV2,
    )
    from evolving_loop.v2.numerical_qd.map_elites import NumericalQDArchive
    from evolving_loop.v2.numerical_qd.runner import _trusted_reusable_program_context

    cell = MorphologyCellV2("low", "none", "low", "stable", "short", "program")
    entry = NumericalQDEntryV2(
        1, "a" * 64, "b" * 64, cell, ("task-a",),
        NumericalObjectiveVectorV2(*(1.0,) * 5), ConstraintReportV2(True, ()), (),
    )
    archive = NumericalQDArchive().insert((entry,))
    target = CurriculumTargetV2(1, cell, "least_visited", 1, ())

    class ForgedStore:
        def verify_candidate(self, _genome_sha):
            raise ValueError("candidate is not Host-verifiable")

    assert _trusted_reusable_program_context(
        ForgedStore(), archive, (target,), maximum_records=4,
    ) == ((), ())


def test_runner_host_derives_and_inserts_policy_tune_prompt_child():
    from evolving_loop.v2.numerical_qd.runner import (
        _seed_prompt_population, _host_mutation_prompt_child,
    )
    from evolving_loop.v2.numerical_qd.contracts import NumericalProposerPromptV2
    parent = _seed_prompt_population(
        NumericalProposerPromptV2(1, "task", "numerical_mutation_batch_v1", 4096,
                                  ("policy_tune",), None)
    )
    selected = parent.lineages[0].mutation_prompt
    child = _host_mutation_prompt_child(selected, "child task prompt")
    assert child.parent_prompt_sha256 == selected.fingerprint()
    assert child.allowed_mutation_operators == ("policy_tune",)


def test_authenticated_fold_groups_pack_into_nested_label_free_rungs():
    from evolving_loop.v2.numerical_qd import hyperband
    from evolving_loop.v2.numerical_qd import runner

    config, _supply, folds, adapter = fixture()
    commitments = {
        task.numeric.task_id: task_registry_fingerprint(task)
        for task in adapter.tasks
    }
    groups = runner._packed_train_task_groups(
        adapter,
        commitments,
        config.kernel_protocol.split_manifest,
        config.kernel_protocol.fingerprint(),
    )
    manifests = tuple(
        hyperband.fixed_rung_manifest(
            groups,
            resource,
            config.kernel_protocol.split_manifest,
            config.kernel_protocol.fingerprint(),
        )
        for resource in (8, 32, 80)
    )

    expected_groups = {
        frozenset(task_ids) for _group_sha, task_ids, _fold in folds.groups
    }
    actual_groups = {
        frozenset(task.task_id for task in tasks) for tasks in groups.values()
    }
    assert actual_groups == expected_groups
    assert all(
        entity_id.startswith("fold-group-")
        and not any(task.numeric.entity_name in entity_id for task in adapter.tasks)
        and all(task.entity_id == entity_id for task in tasks)
        for entity_id, tasks in groups.items()
    )
    assert tuple(len(manifest.tasks) for manifest in manifests) == (8, 32, 80)
    assert set(manifests[0].task_ids) < set(manifests[1].task_ids) < set(manifests[2].task_ids)
    for manifest in manifests:
        selected = set(manifest.task_ids)
        assert all(not (selected & group) or group <= selected for group in expected_groups)


def test_group_packing_fails_when_a_registered_rung_cannot_keep_groups_whole():
    from evolving_loop.v2.numerical_qd import hyperband

    groups = (
        ("1" * 64, tuple(f"task-{index:02d}" for index in range(9)), 0),
        ("2" * 64, tuple(f"task-{index:02d}" for index in range(9, 80)), 1),
    )
    with pytest.raises(ValueError, match="cannot pack authenticated fold groups"):
        hyperband.pack_fold_groups(groups)


def test_schema_two_host_run_requires_evidence_or_explicit_bootstrap(tmp_path):
    config, release, folds, adapter = fixture(raw_seed=True, task_budget=1)
    release = replace(release, schema_version=2, anchor_release_payload=release.to_payload()["anchor_release_payload"])
    with pytest.raises(ValueError, match="Task 4 evidence"):
        run_numerical_qd(tmp_path / "run", config, release, folds, adapter)
    assert not (tmp_path / "run").exists()


def test_schema_two_bootstrap_excludes_only_the_sealed_protected_anchor():
    """Catches dropping a declared safe mutation seed merely because it is a fallback."""
    from evolving_loop.v2.numerical_qd.runner import _bootstrap

    config, release, _folds, adapter = fixture(raw_seed=True)
    screening = adapter.materializer.screening_policy
    adapter.materializer.screening_policy = ScreeningPolicy(
        screening.entries, ("toto_2_0", "seasonal_naive")
    )
    adapter.task_local_evidence = SimpleNamespace(
        by_task={
            "sealed-task": (
                SimpleNamespace(candidate_names=("toto_2_0",)),
                {},
                "1" * 64,
                "2" * 64,
            )
        }
    )

    state, _genome, _policies, _screen, _combined = _bootstrap(
        config, release, adapter
    )

    assert tuple(member.member_id for member in state.inventory.members) == (
        "seasonal_naive",
    )


def test_rung_binds_adapter_local_evidence_before_task_evaluation(monkeypatch):
    from tests.test_evolution_v2_numerical_hyperband import (
        ADAPTER, DESCRIPTOR, METRIC, PROTOCOL, RUNTIME, evaluation, ledger, manifest,
        sha, state,
    )
    from evolving_loop.v2.numerical_qd import runner
    candidate, committed = state(count=1).active_candidates[0], manifest()
    clock = FakeClock()
    budget = ledger(clock)
    evidence_sha256 = sha("runner shortlist evidence")
    seen = set()

    adapter = SimpleNamespace(
        fingerprint=ADAPTER,
        local_evidence_sha256_for=lambda candidate_sha256, task: seen.add(
            (candidate_sha256, task.task_sha256)
        ) or evidence_sha256,
        monotonic=clock,
        resource_snapshot=lambda: None,
        resource_delta=lambda snapshot: ResourceUse(),
    )
    config = SimpleNamespace(
        kernel_protocol=SimpleNamespace(metric_policy=METRIC),
        descriptor_policy=SimpleNamespace(fingerprint=lambda: DESCRIPTOR),
        runtime_fingerprints=RUNTIME,
        adapter={"task_timeout_seconds": 1.0},
    )
    child = SimpleNamespace(genome=SimpleNamespace(fingerprint=lambda: candidate))
    work = SimpleNamespace(
        can_open_stage=budget.can_open_stage,
        reserve_stage=budget.reserve_stage,
        close_stage=lambda permit, actual, **kwargs: budget.close_stage(permit, actual),
    )
    kernel = SimpleNamespace(budget=budget, checkpoint_path="checkpoint.json")

    def evaluate_after_evidence(_adapter, _child, task, **kwargs):
        assert (candidate, task.task_sha256) in seen
        return evaluation(candidate, (task.task_id,), subset=task.fingerprint())

    monkeypatch.setattr(runner, "evaluate_numerical_child", evaluate_after_evidence)
    monkeypatch.setattr(runner, "_read", lambda _path: {"budget": budget.checkpoint()})
    advanced, results, reason = runner._rung(
        kernel, work, state(count=1), committed, {candidate: child}, adapter, config, {}
    )
    assert reason is None and advanced is not None
    assert all(result.local_evidence_sha256 == evidence_sha256
               and result.cache_identity_version == 2 for _, result in results)


def test_operator_input_raw_identity_is_checkpointed_and_rejects_resume(tmp_path):
    committed = {
        "config": "1" * 64,
        "seed_supply": "2" * 64,
        "task_manifest": "3" * 64,
    }
    config, supply, manifest, adapter = fixture(
        task_budget=0, operator_input_sha256s=committed
    )
    root = tmp_path / "run"
    run_numerical_qd(root, config, supply, manifest, adapter)
    before = files(root)
    checkpoint = json.loads((root / "numerical_qd/checkpoint.json").read_bytes())
    assert {
        key.removeprefix("operator_"): value
        for key, value in checkpoint["input_sha256s"].items()
        if key.startswith("operator_")
    } == committed

    _config, _supply, _manifest, changed = fixture(
        task_budget=0,
        operator_input_sha256s=committed | {"task_manifest": "4" * 64},
    )
    with pytest.raises(ValueError, match="input_sha256s mismatch"):
        run_numerical_qd(root, config, supply, manifest, changed, resume=True)
    assert files(root) == before


@pytest.mark.parametrize("ancestor", [False, True])
def test_fresh_run_rejects_symlink_without_external_writes(tmp_path, ancestor):
    config, supply, manifest, adapter = fixture(raw_seed=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    output = link / "child" if ancestor else link
    before = files(outside)
    with pytest.raises(ValueError):
        run_numerical_qd(output, config, supply, manifest, adapter)
    assert files(outside) == before
    assert not tuple(outside.iterdir())


def test_fresh_source_overlap_is_rejected_before_output_creation(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import persistence
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(persistence, "__file__", str(repository / "evolving_loop/v2/numerical_qd/persistence.py"))
    config, supply, manifest, adapter = fixture(raw_seed=True)
    output = repository / "numerical_agent/new-run"
    with pytest.raises(ValueError):
        run_numerical_qd(output, config, supply, manifest, adapter)
    assert not output.exists()


@pytest.mark.parametrize("location", ["docs/new-run", "configs/new-run", "tests/new-run", "runs/other/new-run", "runs/evolution_v2", "other-run"])
def test_repo_output_allowlist_rejects_before_creation(tmp_path, monkeypatch, location):
    from evolving_loop.v2.numerical_qd import persistence
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(persistence, "__file__", str(repository / "evolving_loop/v2/numerical_qd/persistence.py"))
    output = repository / location
    with pytest.raises(ValueError):
        run_numerical_qd(output, *fixture(task_budget=0, raw_seed=True))
    assert list(repository.iterdir()) == []


def test_repo_output_allowlist_permits_only_explicit_v2_runs_and_external_outputs(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import persistence
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(persistence, "__file__", str(repository / "evolving_loop/v2/numerical_qd/persistence.py"))
    for output in (repository / "runs/evolution_v2/epoch-1", tmp_path / "external-run"):
        assert persistence.NumericalQDRunStore.preflight_fresh(output) == output
        assert not output.exists()
    for output in (tmp_path, repository, repository / "runs"):
        with pytest.raises(ValueError):
            persistence.NumericalQDRunStore.preflight_fresh(output)


@pytest.mark.parametrize("boundary", ["tasks", "deadline", "reporter", "failure"])
def test_raw_bootstrap_stops_before_next_dispatch_and_closes_receipt(tmp_path, boundary):
    from evolving_loop.v2.numerical_qd.adapters import NumericalWorkStopped
    config, supply, manifest, adapter = fixture(task_budget=1 if boundary == "tasks" else 5000, raw_seed=True)
    calls = 0
    class LimitedStore(CheapStore):
        def forecast(self, *args):
            nonlocal calls
            calls += 1
            if boundary == "deadline":
                adapter.monotonic.advance(config.budget.search_deadline_seconds + 1.0)
            if boundary == "failure":
                raise NumericalWorkStopped("trusted execution stopped")
            return super().forecast(*args)
    adapter.materializer.forecast_store = LimitedStore()
    if boundary == "reporter":
        adapter.resource_kinds = ("gpu_seconds",)
        adapter.resource_reporter_sha256 = "9" * 64
        adapter.resource_reporter = lambda: ResourceUse(gpu_seconds=float(calls) * (config.budget.ceilings.gpu_seconds + 1.0))
    with pytest.raises(ValueError, match="bootstrap"):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    assert calls == 1
    receipt = json.loads((tmp_path / "run/seed_bootstrap_receipt.json").read_bytes())
    assert receipt["resource_use"]["task_executions"] == 1
    assert receipt["budget_after"]["open_reservations"] == []
    assert len(receipt["budget_after"]["closed_reservation_sha256s"]) == 1
    assert receipt["status"] in {"stopped", "failed"}
    assert not (tmp_path / "run/evaluation_complete.json").exists()
    assert not (tmp_path / "run/numerical_qd").exists()


def test_dev_rejects_a_registry_with_a_different_task_universe():
    from evolving_loop.v2.numerical_qd.runner import _dev_compare

    config, supply, _, adapter = fixture()
    registry = supply.envelope.restore(adapter.tasks)
    kernel = SimpleNamespace(budget=SimpleNamespace(elapsed_wall_seconds=0.0, plan=config.budget))
    incomplete = SimpleNamespace(task_ids=registry.task_ids[:-1], _packages=registry._packages)
    with pytest.raises(ValueError, match="task universe"):
        _dev_compare(registry, incomplete, adapter, kernel, lambda: None)


def test_dev_reads_the_already_verified_frozen_package_map(monkeypatch):
    from evolving_loop.v2.numerical_qd.runner import _dev_compare

    config, supply, _, adapter = fixture()
    registry = supply.envelope.restore(adapter.tasks)
    kernel = SimpleNamespace(budget=SimpleNamespace(elapsed_wall_seconds=0.0, plan=config.budget))
    monkeypatch.setattr(type(registry), "package_for",
        lambda *_args: pytest.fail("Dev rehashed an already frozen package"))

    comparison = _dev_compare(registry, registry, adapter, kernel, lambda: None)

    assert comparison["passed"] is False
    assert comparison["parent_metrics"] == comparison["candidate_metrics"]


def test_dev_promotion_scores_the_frozen_task_local_bundle_not_a_standalone_member():
    from evolving_loop.v2.numerical_qd.runner import _dev_compare

    config, _supply, _, adapter = fixture()
    ordered = tuple(sorted(adapter.tasks, key=lambda task: task.numeric.task_id))
    task_ids = tuple(task.numeric.task_id for task in ordered)
    parent_packages = {}
    child_packages = {}
    for task in ordered:
        truth = task.numeric.future_values
        parent_forecast = tuple(value + 1.0 for value in truth)
        standalone_child = tuple(value + 2.0 for value in truth)
        parent_packages[task.numeric.task_id] = SimpleNamespace(
            final_forecast=parent_forecast,
            ranked_alternatives=(
                SimpleNamespace(name="parent_member", forecast=parent_forecast),
            ),
        )
        child_packages[task.numeric.task_id] = SimpleNamespace(
            # The frozen local selector used Anchor + specialist successfully.
            final_forecast=truth,
            ranked_alternatives=(
                SimpleNamespace(name="child_member", forecast=standalone_child),
            ),
        )
    parent_registry = SimpleNamespace(task_ids=task_ids, _packages=parent_packages)
    child_registry = SimpleNamespace(task_ids=task_ids, _packages=child_packages)
    kernel = SimpleNamespace(
        budget=SimpleNamespace(elapsed_wall_seconds=0.0, plan=config.budget)
    )

    comparison = _dev_compare(
        parent_registry,
        child_registry,
        adapter,
        kernel,
        lambda: None,
    )

    assert comparison["passed"] is True
    assert comparison["candidate_metrics"] == {"mean_smae": 0.0, "mean_srmse": 0.0}


@pytest.mark.parametrize("reverse_entities", [False, True])
def test_cell_entry_binds_its_exact_persisted_subset_evaluation(tmp_path, reverse_entities):
    run_numerical_qd(tmp_path / "run", *fixture(task_budget=920, reverse_entities=reverse_entities), stop_after=1)
    root = tmp_path / "run/numerical_qd"
    entries = [json.loads(line)["entry"] for line in (root / "archive/entries.jsonl").read_bytes().splitlines()]
    assert len(entries) >= 2
    for entry in entries:
        evaluation = json.loads((root / f"objects/{entry['evaluation_sha256']}.json").read_bytes())
        assert evaluation["task_ids"] == entry["task_ids"]
        assert evaluation["objectives"] == entry["objectives"]
        assert evaluation["cells"] == [entry["cell"]]
        subset = json.loads((root / f"objects/{evaluation['task_subset_sha256']}.json").read_bytes())
        assert subset["task_ids"] == entry["task_ids"]


def test_freeze_rehydrates_all_occupied_archive_genomes_across_generations(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    original = runner.freeze_qd_supply
    generations = []
    def freeze(adapter, release, registry, archive, children, **kwargs):
        children = tuple(children)
        required = {archive.entries[sha].genome_sha256 for cell in archive.cells for sha in cell.entry_sha256s}
        assert required <= {child.genome.fingerprint() for child in children}
        generations.append({child.genome.generation for child in children})
        return original(adapter, release, registry, archive, children, **kwargs)
    monkeypatch.setattr(runner, "freeze_qd_supply", freeze)
    run_fixture(tmp_path / "run", stop_after=1)
    run_fixture(tmp_path / "run", resume=True)
    assert any(1 in generations_ and 2 in generations_ for generations_ in generations)


def test_finalize_after_closes_a_bounded_generation_instead_of_pausing(tmp_path):
    result = run_fixture(tmp_path / "run", task_budget=920, finalize_after=1)

    assert result.status == "numerical_qd_complete"
    assert (tmp_path / "run/evaluation_complete.json").is_file()
    assert len(list((tmp_path / "run/numerical_qd/proposals").glob("*.json"))) == 1


def test_finalize_after_runs_every_requested_generation(tmp_path):
    result = run_fixture(
        tmp_path / "run",
        task_budget=2760,
        finalize_after=3,
    )

    assert result.status == "numerical_qd_complete"
    records = list((tmp_path / "run/numerical_qd/proposals").glob("*.json"))
    assert len(records) == 3


def test_train_credit_uses_completed_early_rungs_and_actual_survivors(tmp_path):
    config, supply, manifest, adapter = fixture(task_budget=2472, provider="hybrid")
    payload = config.to_payload()
    payload["mutation"]["operators"] = ["add"]
    # This tests credit at the FakeClock/task-count boundary, not real worker
    # scheduling latency. A 10 ms IPC timeout can discard a valid child under
    # broad-suite load before it ever reaches the completed-rung credit check.
    payload["adapter"]["task_timeout_seconds"] = 1.0
    config = NumericalQDConfigV2.from_payload(payload)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, ThreeChildren("legal"), stop_after=1)
    policy = json.loads((tmp_path / f"run/numerical_qd/objects/{result.mutation_policy_sha256}.json").read_bytes())
    assert policy["operators"]["add"]["attempts"] == 3
    assert policy["operators"]["add"]["feasible"] == 3
    assert policy["operators"]["add"]["promotions"] == 2
    assert policy["operators"]["add"]["insertions"] == 0


@pytest.mark.parametrize("bracket,promotions", [("explore", 1), ("confirm", 1), ("replay", 0)])
def test_train_credit_excludes_terminal_survivors_in_every_bracket(tmp_path, monkeypatch, bracket, promotions):
    from evolving_loop.v2.numerical_qd import runner
    from evolving_loop.v2.numerical_qd.contracts import HyperbandBracketV2
    monkeypatch.setattr(runner, "choose_bracket", lambda *args: HyperbandBracketV2.registered(bracket))
    result = run_fixture(tmp_path / "run", task_budget=920, stop_after=1)
    policy = json.loads((tmp_path / f"run/numerical_qd/objects/{result.mutation_policy_sha256}.json").read_bytes())
    assert policy["operators"]["repair"]["feasible"] == 1
    assert policy["operators"]["repair"]["promotions"] == promotions


def test_timed_out_rung_keeps_completed_rows_under_closed_partial_authority(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=920)
    original = runner.evaluate_numerical_child
    calls = 0
    def evaluate(*args, **kwargs):
        nonlocal calls
        value = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            adapter.monotonic.advance(config.adapter["task_timeout_seconds"])
        return value
    monkeypatch.setattr(runner, "evaluate_numerical_child", evaluate)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    root = tmp_path / "run/numerical_qd"
    partial = [json.loads(path.read_bytes())["closed_partial_rung"] for path in (root / "objects").glob("*.json")
               if "closed_partial_rung" in json.loads(path.read_bytes())]
    assert len(partial) == 1
    assert len(partial[0]["task_results"]) == 2
    assert partial[0]["reason"] == "timeout"
    assert not list((root / "rungs").glob("*.json"))
    assert result.accepted_steps == result.occupied_cells == 0
    before = files(tmp_path / "run")
    run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, resume=True, stop_after=1)
    assert files(tmp_path / "run") == before


@pytest.mark.parametrize("stage,error", [("proposal", OSError), ("rung", RuntimeError)])
def test_external_material_after_write_failure_keeps_failed_closure(tmp_path, monkeypatch, stage, error):
    from evolving_loop.v2.numerical_qd import runner
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
    config, supply, manifest, adapter = fixture(task_budget=920)
    original_evaluate, dispatches, captured = runner.evaluate_numerical_child, [], {}
    def evaluate(*args, **kwargs):
        dispatches.append(args[2].task_id)
        value = original_evaluate(*args, **kwargs)
        adapter.monotonic.advance(0.001)
        return value
    monkeypatch.setattr(runner, "evaluate_numerical_child", evaluate)
    method = "write_proposal_attempt" if stage == "proposal" else "write_task_result"
    original_write = getattr(NumericalQDRunStore, method)
    def write(store, *args, **kwargs):
        path = original_write(store, *args, **kwargs)
        raw = path.read_bytes()
        receipt = {"relative_path": path.relative_to(store.directory).as_posix(),
            "kind": "proposal_attempt" if stage == "proposal" else "task_result",
            "content_sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        # Exercise the real writer, exact readback, and Kernel registration before
        # the injected Host exception; no evaluation/closure decision is mocked.
        store.accounting.kernel.verify_existing_material(**receipt)
        captured.update(receipt=receipt, reservation=store.accounting.external_permit.reservation_sha256,
            before=store.accounting.kernel.budget.checkpoint(), raw=raw)
        raise error("after registered external material")
    monkeypatch.setattr(NumericalQDRunStore, method, write)
    root = tmp_path / "run"
    with pytest.raises(error, match="after registered external material"):
        run_numerical_qd(root, config, supply, manifest, adapter, stop_after=1)
    checkpoint = json.loads((root / "checkpoint.json").read_bytes())
    ref = checkpoint["budget_closures"][captured["reservation"]]
    folder = root / "evaluations" / ref["candidate_bundle_sha256"]
    train = json.loads((folder / "train.json").read_bytes())
    closed = json.loads((folder / "closed.json").read_bytes())
    closure = json.loads((folder / "budget_closure.json").read_bytes())
    assert train["status"] == closed["status"] == "failed"
    expected = ResourceUse(task_executions=int(stage == "rung"),
        wall_seconds=0.001 if stage == "rung" else 0.0,
        artifact_bytes=len(captured["raw"])).to_payload()
    assert train["resource_use"] == closed["resource_use"] == closure["resource_use"] == expected
    assert train["train_objectives"]["material_receipts"] == [captured["receipt"]]
    assert (root / "numerical_qd" / captured["receipt"]["relative_path"]).read_bytes() == captured["raw"]
    assert len(dispatches) == int(stage == "rung")
    assert not checkpoint["budget"]["open_reservations"]
    assert set(checkpoint["budget_closures"]) == set(captured["before"]["closed_reservation_sha256s"]) | {captured["reservation"]}
    assert not list((root / "numerical_qd/rungs").glob("*.json"))
    entries = root / "numerical_qd/archive/entries.jsonl"
    assert not entries.exists() or entries.read_bytes() == b""
    assert not (root / "evaluation_complete.json").exists()
    resumed = EvolutionKernel.resume(V2RunStore(root), config.budget, monotonic=lambda: 0.0)
    assert resumed.active_bundle().generation == 0
    assert resumed.active_bundle().numerical_release_sha256 == supply.release.fingerprint
    assert resumed.budget.checkpoint()["charged_use"] == checkpoint["budget"]["charged_use"]


def test_rung_manifest_is_paid_before_dispatch_and_partial_references_it(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    from evolving_loop.v2.numerical_qd.artifacts import ArtifactKindV2
    from evolving_loop.v2.contracts import fingerprint_payload
    config, supply, manifest, adapter = fixture(task_budget=920)
    rung, evaluate, current = runner._rung, runner.evaluate_numerical_child, {}
    def run_rung(*args, **kwargs):
        current["manifest"] = args[3]
        return rung(*args, **kwargs)
    def evaluate_task(*args, **kwargs):
        fixed = current["manifest"]
        path = tmp_path / f"run/numerical_qd/objects/{fixed.fingerprint()}.json"
        assert path.is_file(), "dispatched before persisting fixed manifest"
        checkpoint = json.loads((tmp_path / "run/checkpoint.json").read_bytes())
        charges = []
        for ref in checkpoint["budget_closures"].values():
            folder = tmp_path / f"run/evaluations/{ref['candidate_bundle_sha256']}"
            train = json.loads((folder / "train.json").read_bytes())
            expected = {"kind": "rung_manifest", "relative_path": f"objects/{fixed.fingerprint()}.json",
                "content_sha256": fixed.fingerprint(), "size_bytes": path.stat().st_size}
            if expected in train["train_objectives"].get("material_receipts", []):
                charges.append(json.loads((folder / "budget_closure.json").read_bytes()))
        assert len(charges) == 1 and charges[0]["allowed"]
        assert charges[0]["resource_use"] == ResourceUse(artifact_bytes=path.stat().st_size).to_payload()
        value = evaluate(*args, **kwargs)
        adapter.monotonic.advance(config.adapter["task_timeout_seconds"])
        return value
    monkeypatch.setattr(runner, "_rung", run_rung)
    monkeypatch.setattr(runner, "evaluate_numerical_child", evaluate_task)
    run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    store = runner.NumericalQDRunStore(tmp_path / "run")
    record = next(json.loads(path.read_bytes()) for path in (store.directory / "objects").glob("*.json")
                  if "closed_partial_rung" in json.loads(path.read_bytes()))
    partial = record["closed_partial_rung"]
    assert "manifest" not in partial and partial["manifest_sha256"] == current["manifest"].fingerprint()
    paid = store.directory / f"objects/{partial['manifest_sha256']}.json"
    paid.unlink()
    before = files(store.root)
    with pytest.raises(ValueError, match="manifest|missing"):
        store.write_object(fingerprint_payload(record), record, kind=ArtifactKindV2.PARTIAL_RUNG)
    assert files(store.root) == before
    with pytest.raises(ValueError):
        run_numerical_qd(store.root, config, supply, manifest, adapter, resume=True, stop_after=1)
    assert files(store.root) == before


def test_manifest_admission_denial_closes_zero_task_partial_without_free_manifest(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    original, target = runner._rung, {}
    reserve = runner._KernelWork.reserve_stage
    def run_rung(*args, **kwargs):
        target["sha"] = args[3].fingerprint()
        return original(*args, **kwargs)
    def reserve_stage(work, stage, estimate):
        if stage == f"material-objects/{target.get('sha')}.json":
            estimate = replace(estimate, artifact_bytes=work.kernel.budget.plan.ceilings.artifact_bytes + 1)
        return reserve(work, stage, estimate)
    monkeypatch.setattr(runner, "_rung", run_rung)
    monkeypatch.setattr(runner._KernelWork, "reserve_stage", reserve_stage)
    monkeypatch.setattr(runner, "evaluate_numerical_child", lambda *args, **kwargs: pytest.fail("task dispatched before manifest admission"))
    result = run_fixture(tmp_path / "run", task_budget=920, stop_after=1)
    objects = tmp_path / "run/numerical_qd/objects"
    assert not (objects / f"{target['sha']}.json").exists()
    partial = next(json.loads(path.read_bytes())["closed_partial_rung"] for path in objects.glob("*.json")
                   if "closed_partial_rung" in json.loads(path.read_bytes()))
    assert "manifest" not in partial and partial["manifest_sha256"] is None
    assert partial["reason"] == "artifact_bytes_exhausted"
    assert partial["task_results"] == partial["unpersisted_task_results"] == []
    assert partial["budget_outcome"]["resource_use"] == ResourceUse().to_payload()
    assert not result.budget["open_reservations"]
    before = files(tmp_path / "run")
    run_fixture(tmp_path / "run", task_budget=920, resume=True, stop_after=1)
    assert files(tmp_path / "run") == before


def test_paid_manifest_deadline_boundary_publishes_only_blocked_control(tmp_path, monkeypatch, material_catalog):
    from evolving_loop.v2.numerical_qd import runner
    from evolving_loop.v2.numerical_qd.artifacts import ArtifactKindV2
    config, supply, manifest, adapter = fixture(task_budget=920)
    original, paid = runner._MaterialAccounting.write, {}
    def write(accounting, relative, data, dispatch, *, kind):
        result = original(accounting, relative, data, dispatch, kind=kind)
        if kind is ArtifactKindV2.RUNG_MANIFEST:
            assert adapter.monotonic() < config.budget.search_deadline_seconds
            paid.update(json.loads((tmp_path / "run/checkpoint.json").read_bytes()))
            assert paid["budget"]["open_reservations"] == []
            adapter.monotonic.advance(config.budget.search_deadline_seconds)
        return result
    monkeypatch.setattr(runner._MaterialAccounting, "write", write)
    monkeypatch.setattr(runner, "evaluate_numerical_child", lambda *args, **kwargs: pytest.fail("deadline task dispatch"))
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    root = tmp_path / "run"
    final = json.loads((root / "checkpoint.json").read_bytes())
    assert final["budget_closures"] == paid["budget_closures"]
    assert result.budget["closed_reservation_sha256s"] == paid["budget"]["closed_reservation_sha256s"]
    assert result.budget["closed_stage_ids"] == paid["budget"]["closed_stage_ids"]
    partial = next(json.loads(path.read_bytes())["closed_partial_rung"] for path in (root / "numerical_qd/objects").glob("*.json")
                   if "closed_partial_rung" in json.loads(path.read_bytes()))
    assert partial["manifest_sha256"] is not None and "manifest" not in partial
    assert partial["reason"] == "finalization_reserve" and partial["budget_outcome"]["status"] == "blocked"
    assert partial["task_results"] == partial["unpersisted_task_results"] == []
    assert partial["budget_outcome"]["reservation_sha256"] is None
    assert partial["budget_outcome"]["resource_use"] == ResourceUse().to_payload()
    assert result.budget["charged_use"]["artifact_bytes"] == sum(material_sizes(root, material_catalog))
    assert "partial_rung" in material_catalog.values()
    before = files(root)
    run_numerical_qd(root, config, supply, manifest, adapter, resume=True, stop_after=1)
    assert files(root) == before


def test_deadline_after_proposal_persistence_finalizes_without_new_material(tmp_path, monkeypatch):
    """Catches opening recipe/genome writes after the search deadline begins."""
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore

    config, supply, manifest, adapter = fixture(task_budget=920)
    original = NumericalQDRunStore.write_proposal_attempt

    def write_attempt(store, *args, **kwargs):
        result = original(store, *args, **kwargs)
        adapter.monotonic.advance(config.budget.search_deadline_seconds)
        return result

    monkeypatch.setattr(NumericalQDRunStore, "write_proposal_attempt", write_attempt)
    result = run_numerical_qd(
        tmp_path / "run", config, supply, manifest, adapter, stop_after=1
    )
    assert result.status == "numerical_qd_complete"
    assert (tmp_path / "run/evaluation_complete.json").is_file()
    steps = [
        json.loads(path.read_bytes())["numerical_qd_step"]
        for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
        if "numerical_qd_step" in json.loads(path.read_bytes())
    ]
    assert len(steps) == 1
    assert steps[0]["status"] == "finalization_reserve"
    assert steps[0]["materialization_failures"] == []


def test_deadline_after_rng_checkpoint_stops_before_proposer_request(
    tmp_path, monkeypatch
):
    """A slow durable RNG checkpoint must not open later proposal material."""
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore

    config, supply, manifest, adapter = fixture(task_budget=920)
    original = NumericalQDRunStore.write_state
    writes = 0

    def checkpoint(store, *args, **kwargs):
        nonlocal writes
        result = original(store, *args, **kwargs)
        writes += 1
        if writes == 2:
            adapter.monotonic.advance(config.budget.search_deadline_seconds)
        return result

    monkeypatch.setattr(NumericalQDRunStore, "write_state", checkpoint)
    result = run_numerical_qd(
        tmp_path / "run", config, supply, manifest, adapter, stop_after=1
    )

    assert result.status == "numerical_qd_complete"
    assert not list((tmp_path / "run/numerical_qd/proposals").glob("*.json"))
    steps = [
        json.loads(path.read_bytes())["numerical_qd_step"]
        for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
        if "numerical_qd_step" in json.loads(path.read_bytes())
    ]
    assert len(steps) == 1
    assert steps[0]["status"] == "finalization_reserve"
    assert steps[0]["proposal_attempt_sha256"] is None
    assert result.budget["open_reservations"] == []


def test_deadline_during_candidate_processing_closes_generation_without_new_material(tmp_path, monkeypatch):
    """Catches a batch continuing material writes after candidate work uses its time."""
    from evolving_loop.v2.numerical_qd import runner

    config, supply, manifest, adapter = fixture(task_budget=2472, provider="hybrid")
    payload = config.to_payload()
    payload["mutation"]["operators"] = ["add"]
    config = NumericalQDConfigV2.from_payload(payload)
    original = runner.apply_mutation

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        adapter.monotonic.advance(config.budget.search_deadline_seconds)
        return result

    monkeypatch.setattr(runner, "apply_mutation", mutate)
    result = run_numerical_qd(
        tmp_path / "run", config, supply, manifest, adapter,
        ThreeChildren("legal"), stop_after=1,
    )
    assert result.status == "numerical_qd_complete"
    step = next(
        json.loads(path.read_bytes())["numerical_qd_step"]
        for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
        if "numerical_qd_step" in json.loads(path.read_bytes())
    )
    assert step["status"] == "finalization_reserve"
    assert step["winner_genome_sha256"] is None


def test_rung_artifact_denial_closes_partial_without_persisting_denied_payload(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
    from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
    original, denied, written, calls = NumericalQDRunStore.write_task_result, {}, [], 0
    evaluate, evaluated = runner.evaluate_numerical_child, 0
    def marked_evaluation(*args, **kwargs):
        nonlocal evaluated
        value = evaluate(*args, **kwargs)
        evaluated += 1
        if evaluated == 2:
            # Payload sentinel is never ranked: this row's persistence is denied.
            value = replace(value, objectives=replace(value.objectives, mean_raw_joint_error=987654321.125))
        return value
    monkeypatch.setattr(runner, "evaluate_numerical_child", marked_evaluation)
    def write(store, task_sha, result):
        nonlocal calls
        calls += 1
        if calls == 2:
            denied.update(task_sha256=task_sha, result=result)
            # Exercise the live rung admission hook itself, not a mocked close.
            store.accounting.external(10 ** 18, False)
            pytest.fail("oversized material was admitted")
        written.append((task_sha, result))
        return original(store, task_sha, result)
    monkeypatch.setattr(NumericalQDRunStore, "write_task_result", write)
    result = run_fixture(tmp_path / "run", task_budget=920, stop_after=1)
    root = tmp_path / "run/numerical_qd"
    partial = [json.loads(path.read_bytes())["closed_partial_rung"] for path in (root / "objects").glob("*.json")
               if "closed_partial_rung" in json.loads(path.read_bytes())]
    assert calls == 2 and len(partial) == 1
    record = partial[0]
    assert record["reason"] == "artifact_bytes_exhausted"
    assert len(record["task_results"]) == 1
    assert record["unpersisted_task_results"] == [{"task_sha256": denied["task_sha256"],
        "task_id": denied["result"].task_id, "candidate_sha256": denied["result"].candidate_sha256,
        "cache_key": denied["result"].cache_key, "result_sha256": fingerprint_payload(denied["result"].to_payload()),
        "status": "persistence_denied", "resource_use": dict(denied["result"].evaluation.resource_use)}]
    assert record["budget_outcome"]["resource_use"]["task_executions"] == 2
    assert record["budget_outcome"]["resource_use"]["artifact_bytes"] == len(canonical_v2_bytes(written[0][1].to_payload()))
    checkpoint = json.loads((tmp_path / "run/checkpoint.json").read_bytes())
    reference = checkpoint["budget_closures"][record["budget_outcome"]["reservation_sha256"]]
    train = json.loads((tmp_path / f"run/evaluations/{reference['candidate_bundle_sha256']}/train.json").read_bytes())
    assert train["train_objectives"]["material_receipts"] == [{"kind": "task_result",
        "relative_path": f"results/{written[0][1].candidate_sha256}/{written[0][0]}.json",
        "content_sha256": fingerprint_payload(written[0][1].to_payload()),
        "size_bytes": len(canonical_v2_bytes(written[0][1].to_payload()))}]
    assert record["budget_outcome"]["status"] == "failed"
    assert not list((root / "rungs").glob("*.json"))
    assert not (root / f"results/{denied['result'].candidate_sha256}/{denied['task_sha256']}.json").exists()
    assert all(canonical_v2_bytes(denied["result"].to_payload()) not in data for data in files(tmp_path / "run").values())
    assert all(b"987654321.125" not in data for data in files(tmp_path / "run").values())
    assert result.accepted_steps == result.occupied_cells == 0
    assert not result.budget["open_reservations"]
    before = files(tmp_path / "run")
    run_fixture(tmp_path / "run", task_budget=920, resume=True, stop_after=1)
    assert files(tmp_path / "run") == before


def test_all_evolvable_material_bytes_are_charged_once_but_authority_is_excluded(tmp_path, material_catalog):
    result = run_fixture(tmp_path / "run", task_budget=920, stop_after=1)
    root = tmp_path / "run/numerical_qd"
    assert result.budget["charged_use"]["artifact_bytes"] == sum(material_sizes(root, material_catalog))
    assert {"source", "config", "genome", "inventory", "screening_policy", "combined_policy", "recipe_policy",
        "mutation_policy", "prompt", "proposer_request", "proposal_attempt", "executable_child", "evaluation",
        "rung_manifest", "task_result", "qd_entry", "cell_subset", "frozen_pair"} <= set(material_catalog.values())
    from tests.test_evolution_v2_numerical_artifacts import MATERIAL_KINDS
    checkpoint = json.loads((tmp_path / "run/checkpoint.json").read_bytes())
    receipts = []
    for ref in checkpoint["budget_closures"].values():
        folder = tmp_path / f"run/evaluations/{ref['candidate_bundle_sha256']}"
        receipts.extend(json.loads((folder / "train.json").read_bytes())["train_objectives"].get("material_receipts", []))
    expected = [{"kind": kind, "relative_path": path.relative_to(root).as_posix(),
        "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size} for path, kind in material_catalog.items() if kind in MATERIAL_KINDS]
    assert sorted(receipts, key=lambda row: row["relative_path"]) == sorted(expected, key=lambda row: row["relative_path"])
    before = files(tmp_path / "run")
    resumed = run_fixture(tmp_path / "run", task_budget=920, resume=True, stop_after=1)
    assert resumed.budget == result.budget
    assert files(tmp_path / "run") == before


def test_interrupted_bootstrap_resumes_closed_forecast_cache_without_recharging(tmp_path, monkeypatch):
    from evolving_loop.v2.kernel import SeedBootstrapAuthority
    config, supply, manifest, adapter = fixture(task_budget=5000, raw_seed=True)
    original = SeedBootstrapAuthority.forecast
    calls = 0
    class CountedStore(CheapStore):
        def forecast(self, *args):
            nonlocal calls
            calls += 1
            return super().forecast(*args)
    adapter.materializer.forecast_store = CountedStore()
    def interrupted(authority, *args):
        result = original(authority, *args)
        if calls == 3:
            raise KeyboardInterrupt("process stopped between closed forecasts")
        return result
    monkeypatch.setattr(SeedBootstrapAuthority, "forecast", interrupted)
    with pytest.raises((ValueError, KeyboardInterrupt)):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    assert calls == 3
    assert not (tmp_path / "run/seed_bootstrap_receipt.json").exists()
    monkeypatch.setattr(SeedBootstrapAuthority, "forecast", original)
    force_dev_results(monkeypatch, True)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, resume=True, stop_after=1)
    assert calls == 1200  # 800 bootstrap + 400 protected-anchor child dispatches; no replayed 3.
    assert result.budget["charged_use"]["task_executions"] == 1720
    assert result.accepted_steps == 1
    assert not result.budget["open_reservations"]


@pytest.mark.parametrize("consumed", [0, 1, 3])
def test_bootstrap_passed_receipt_requires_hash_linked_complete_replay(tmp_path, consumed):
    from evolving_loop.v2.kernel import SeedBootstrapAuthority
    from evolving_loop.v2.contracts import fingerprint_payload
    from evolving_loop.v2.store import write_once_json
    from types import SimpleNamespace
    config, seed, _, adapter = fixture(raw_seed=True)
    store = V2RunStore.create(tmp_path / "run")
    preflight = {"schema_version": 1, "stage": "seed_bootstrap", "seed_supply_sha256": seed.fingerprint,
        "input_sha256s": {"test": "1" * 64},
        "protocol_sha256": config.kernel_protocol.fingerprint(), "budget_plan_sha256": config.budget.fingerprint(),
        "estimate": replace(config.budget.ceilings, wall_seconds=1.0).to_payload()}
    authority = SeedBootstrapAuthority(store, config.kernel_protocol, config.budget, preflight, monotonic=adapter.monotonic)
    host = SimpleNamespace(forecast_trusted=lambda *args, **kwargs: [float(args[0])])
    for index in range(3):
        authority.forecast(host, index)
    authority.close("interrupted", "process_interrupted")
    material_bytes = sum(path.stat().st_size for path in (store.root / "seed_bootstrap_cache").glob("*.json"))
    assert authority.receipt["resource_use"]["artifact_bytes"] == material_bytes
    resumed = SeedBootstrapAuthority.resume(store, config.kernel_protocol, config.budget,
        preflight["input_sha256s"], monotonic=adapter.monotonic)
    forbidden = SimpleNamespace(forecast_trusted=lambda *args, **kwargs: pytest.fail("replayed actual dispatch"))
    for index in range(consumed):
        assert resumed.forecast(forbidden, index) == [float(index)]
    receipt = resumed.close("passed")
    assert receipt["status"] == ("passed" if consumed == 3 else "failed")
    assert receipt["replay_total"] == 3 and receipt["replay_consumed"] == consumed
    assert receipt["resource_use"] == ResourceUse().to_payload()
    assert not receipt["budget_after"]["open_reservations"]
    replay_events = [json.loads((store.root / f"seed_bootstrap/{sha}.json").read_bytes()) for sha in receipt["events"]]
    assert [event["kind"] for event in replay_events] == ["replayed_forecast"] * consumed
    assert [event["replay_index"] for event in replay_events] == list(range(consumed))
    for field in ("replay_total", "replay_consumed"):
        forged = {**receipt, field: receipt[field] + 1}
        sha = fingerprint_payload(forged)
        write_once_json(store.root / f"seed_bootstrap_segments/{sha}.json", forged)
        with pytest.raises(KernelAuthorityError, match="replay"):
            SeedBootstrapAuthority.verify(store, config.budget, authority.identity, sha, terminal=False)
    assert not (store.root / "active_bundle.json").exists()


def test_bootstrap_resume_rejects_cache_symlink_before_reading_it(tmp_path, monkeypatch):
    from pathlib import Path
    config, supply, manifest, adapter = fixture(task_budget=1, raw_seed=True)
    with pytest.raises(ValueError, match="bootstrap"):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    cached = next((tmp_path / "run/seed_bootstrap_cache").glob("*.json"))
    outside = tmp_path / "outside.json"
    cached.replace(outside)
    cached.symlink_to(outside)
    original = Path.read_bytes
    opened = []
    def read(path):
        if path == cached:
            opened.append(path)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, resume=True)
    assert opened == []


@pytest.mark.parametrize("orphan_kind", ["regular", "directory", "nested", "link"])
def test_bootstrap_resume_rejects_root_orphans_before_any_new_dispatch(tmp_path, monkeypatch, orphan_kind):
    from evolving_loop.v2.kernel import SeedBootstrapAuthority
    config, supply, manifest, adapter = fixture(task_budget=5000, raw_seed=True)
    original = SeedBootstrapAuthority.forecast
    calls = 0
    class CountedStore(CheapStore):
        def forecast(self, *args):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise KeyboardInterrupt("unexpected post-orphan dispatch")
            return super().forecast(*args)
    adapter.materializer.forecast_store = CountedStore()
    def interrupt_after_closed(authority, *args):
        original(authority, *args)
        raise KeyboardInterrupt("closed prefix")
    monkeypatch.setattr(SeedBootstrapAuthority, "forecast", interrupt_after_closed)
    root = tmp_path / "run"
    with pytest.raises((ValueError, KeyboardInterrupt)):
        run_numerical_qd(root, config, supply, manifest, adapter)
    monkeypatch.setattr(SeedBootstrapAuthority, "forecast", original)
    if orphan_kind == "regular":
        (root / "orphan.json").write_bytes(b"{}\n")
    elif orphan_kind == "directory":
        (root / "orphan").mkdir()
    elif orphan_kind == "nested":
        (root / "archive/objects/orphan.json").write_bytes(b"{}\n")
    else:
        (root / "orphan").symlink_to(tmp_path / "outside")
    before = files(root)
    with pytest.raises((ValueError, KeyboardInterrupt)):
        run_numerical_qd(root, config, supply, manifest, adapter, resume=True)
    assert calls == 1
    assert files(root) == before


@pytest.mark.parametrize("mismatch", [False, True])
def test_kernel_typed_promotion_binds_full_train_evaluation_and_exact_winner(tmp_path, mismatch, monkeypatch):
    force_dev_results(monkeypatch, True)
    result = run_fixture(tmp_path / "run", task_budget=920, stop_after=1)
    assert result.accepted_steps == 1
    root = tmp_path / "run"
    records = [json.loads(path.read_bytes()) for path in (root / "evaluations").glob("*/train.json")]
    train = next(row for row in records if "numerical_artifacts_sha256" in row["train_behavior_descriptors"])
    bindings = train["train_behavior_descriptors"]
    if mismatch:
        kernel = EvolutionKernel.resume(V2RunStore(root), fixture(task_budget=920)[0].budget, monotonic=lambda: 0.0)
        changed = train | {"train_behavior_descriptors": bindings | {"numerical_winner_genome_sha256": "f" * 64}}
        with pytest.raises(KernelAuthorityError, match="winner"):
            kernel._numerical_release_references(result.active_bundle, changed)
        return
    pair = json.loads((root / f"numerical_qd/objects/{bindings['numerical_artifacts_sha256']}.json").read_bytes())
    winner = bindings.get("numerical_winner_genome_sha256")
    assert winner is not None
    assert pair["supply"]["source_fingerprints"]["train_winner"] == winner
    evaluation = json.loads((root / f"numerical_qd/objects/{bindings['numerical_train_evaluation_sha256']}.json").read_bytes())
    assert evaluation["genome_sha256"] == winner
    assert len(evaluation["task_ids"]) == 80
    assert evaluation["objectives"] == train["train_objectives"]


def test_kernel_rejects_declared_winner_with_forged_final_supply_or_task_member(tmp_path, monkeypatch):
    from common.payload import canonical_json_bytes
    from evolving_loop.package_registry import _digest
    from evolving_loop.package_numerical_supply import parse_numerical_supply_release, build_package_registry, bound_numerical_package
    from evolving_loop.v2.contracts import fingerprint_payload
    from evolving_loop.v2.numerical_qd.adapters import _envelope
    from evolving_loop.v2.numerical_qd.contracts import FrozenNumericalRegistryEnvelopeV2
    from evolving_loop.v2.store import write_once_json
    config, supply, manifest, adapter = fixture(task_budget=920)
    root = tmp_path / "run"
    force_dev_results(monkeypatch, True)
    result = run_numerical_qd(root, config, supply, manifest, adapter, stop_after=1)
    kernel = EvolutionKernel.resume(V2RunStore(root), config.budget, monotonic=adapter.monotonic)
    train = next(json.loads(path.read_bytes()) for path in (root / "evaluations").glob("*/train.json")
                 if "numerical_artifacts_sha256" in json.loads(path.read_bytes())["train_behavior_descriptors"])
    bindings = train["train_behavior_descriptors"]
    objects = root / "numerical_qd/objects"
    pair = json.loads((objects / f"{bindings['numerical_artifacts_sha256']}.json").read_bytes())
    release = parse_numerical_supply_release(pair["supply"])
    registry = FrozenNumericalRegistryEnvelopeV2.from_payload(pair["registry"]).restore(adapter.tasks)
    executable = json.loads((objects / f"{bindings['numerical_winner_materialized_sha256']}.json").read_bytes())
    winner_name = executable["materialized_numerical_child"]["fit"]["recipe"]["name"]
    assert kernel._numerical_release_references(result.active_bundle, train) == tuple(sorted((release.fingerprint, registry.fingerprint)))
    rejected = []
    attacks = ("missing_spec", "changed_spec", "missing_member", "changed_forecast", "changed_diagnostics", "registry_identity", "task_identity")
    for attack in attacks:
        release_payload = release.to_payload()
        if attack == "missing_spec":
            release_payload["alternatives"] = []
        elif attack == "changed_spec":
            for spec in release_payload["alternatives"]:
                if spec["candidate_id"] == winner_name:
                    spec["materializer_kind"] = "dictionary"
        altered_release = parse_numerical_supply_release(release_payload)
        def builder(task, supplied):
            source = registry.package_for(task)
            available = {row.name: row for row in source.ranked_alternatives}
            if task.numeric.task_id == adapter.tasks[-1].numeric.task_id:
                if attack == "missing_member":
                    available.pop(winner_name)
                elif attack == "changed_forecast":
                    item = available[winner_name]
                    available[winner_name] = replace(item, forecast=tuple(value + 0.5 for value in item.forecast))
                elif attack == "changed_diagnostics":
                    item = available[winner_name]
                    available[winner_name] = replace(item, diagnostics=replace(item.diagnostics, cache_key="forged-cache-key"))
            return bound_numerical_package(source, supplied, available)
        altered_registry = build_package_registry(adapter.tasks, altered_release, builder)
        envelope = _envelope(altered_registry, adapter.tasks)
        if attack == "registry_identity":
            envelope = replace(envelope, registry_sha256="f" * 64)
        elif attack == "task_identity":
            entries = {key: dict(value) for key, value in envelope.entries.items()}
            task_id = adapter.tasks[-1].numeric.task_id
            entries[task_id]["task_sha256"] = "f" * 64
            registry_manifest = json.loads(canonical_json_bytes(altered_registry.manifest))
            next(row for row in registry_manifest["entries"] if row["task_id"] == task_id)["task_sha256"] = "f" * 64
            envelope = replace(envelope, entries=entries,
                registry_sha256=_digest(registry_manifest))
        altered_pair = {"supply": altered_release.to_payload(), "registry": envelope.to_payload()}
        identity = fingerprint_payload(altered_pair)
        write_once_json(objects / f"{identity}.json", altered_pair)
        bundle = replace(result.active_bundle, numerical_release_sha256=altered_release.fingerprint,
                         numerical_registry_sha256=envelope.registry_sha256)
        changed_train = train | {"train_behavior_descriptors": bindings | {"numerical_artifacts_sha256": identity}}
        try:
            kernel._numerical_release_references(bundle, changed_train)
        except KernelAuthorityError:
            rejected.append(attack)
    assert rejected == list(attacks)


def test_bootstrap_resume_rejects_unclosed_dispatch_after_last_closed_segment(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    from evolving_loop.v2.contracts import fingerprint_payload
    from evolving_loop.v2.store import write_once_json
    config, supply, manifest, adapter = fixture(task_budget=5000, raw_seed=True)
    original = runner.SeedBootstrapAuthority.forecast
    def interrupted(authority, *args):
        value = original(authority, *args)
        raise KeyboardInterrupt("closed forecast then process pause")
    monkeypatch.setattr(runner.SeedBootstrapAuthority, "forecast", interrupted)
    with pytest.raises((ValueError, KeyboardInterrupt)):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    segment = next((tmp_path / "run/seed_bootstrap_segments").glob("*.json"))
    orphan = {"kind": "admission", "sequence": 0, "previous_sha256": segment.stem,
              "arguments": ["seasonal_naive", [1.0], 2, "D"], "charged_use": ResourceUse(task_executions=1).to_payload()}
    write_once_json(tmp_path / f"run/seed_bootstrap/{fingerprint_payload(orphan)}.json", orphan)
    before = files(tmp_path / "run")
    preflight = json.loads((tmp_path / "run/seed_bootstrap_preflight.json").read_bytes())
    with pytest.raises(ValueError, match="bootstrap.*(open|unclosed|unreferenced)"):
        runner.SeedBootstrapAuthority.resume(V2RunStore(tmp_path / "run"), config.kernel_protocol, config.budget,
            preflight["input_sha256s"], monotonic=adapter.monotonic)
    assert files(tmp_path / "run") == before


def test_qd_runner_evolves_supply_policy_and_prompt(tmp_path, monkeypatch):
    force_dev_results(monkeypatch, True, False)
    result = run_fixture(tmp_path / "run")
    assert result.status == "numerical_qd_complete"
    assert result.occupied_cells >= 2
    assert result.accepted_steps == 1
    assert result.rejected_steps == 1
    assert result.mutation_policy_sha256 != result.seed_mutation_policy_sha256
    assert result.proposer_prompt_sha256 != result.seed_proposer_prompt_sha256
    assert result.public_test_accessed is False
    assert result.budget["charged_use"]["task_executions"] == 1840
    assert result.budget["open_reservations"] == []


def test_acceptance_promotes_pair_and_rejection_keeps_exact_parent(tmp_path, monkeypatch):
    config, supply, manifest, adapter = fixture()
    force_dev_results(monkeypatch, True, False)
    first = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    pointer = (tmp_path / "run/accepted_bundle.json").read_bytes()
    assert first.active_bundle.numerical_release_sha256 != supply.release.fingerprint
    seed_registry = json.loads((tmp_path / "run/run_manifest.json").read_bytes())["seed_bundle_sha256"]
    seed_bundle = json.loads((tmp_path / f"run/archive/objects/{seed_registry}.json").read_bytes())
    assert first.active_bundle.numerical_registry_sha256 != seed_bundle["numerical_registry_sha256"]
    resumed = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, resume=True)
    assert resumed.active_bundle.canonical_bytes() == first.active_bundle.canonical_bytes()
    assert (tmp_path / "run/accepted_bundle.json").read_bytes() == pointer
    records = [json.loads(line)["record"] for line in (tmp_path / "run/archive/index.jsonl").read_bytes().splitlines()]
    accepted = [row for row in records if row["artifact_kind"] == "acceptance_seal"]
    assert len(accepted) == 1
    assert accepted[0]["accepted_release_sha256s"] == sorted([
        first.active_bundle.numerical_release_sha256, first.active_bundle.numerical_registry_sha256])
    assert resumed.rejected_steps == 1
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
    pairs = NumericalQDRunStore(tmp_path / "run").load_frozen_pairs(
        tasks=adapter.tasks
    )
    candidates = [pair for pair, _sha in pairs if pair.selected_genome_sha256s]
    assert len(candidates) == 2
    assert len({pair.release.fingerprint for pair in candidates}) == 2


def test_stop_resume_is_byte_identical_and_completed_run_is_mtime_noop(tmp_path):
    run_fixture(tmp_path / "full")
    run_fixture(tmp_path / "resumed", stop_after=1)
    result = run_fixture(tmp_path / "resumed", resume=True)
    assert files(tmp_path / "full") == files(tmp_path / "resumed")
    before = {name: path.stat().st_mtime_ns for name in files(tmp_path / "resumed")
              for path in [tmp_path / "resumed" / name]}
    repeated = run_fixture(tmp_path / "resumed", resume=True)
    assert repeated.active_bundle == result.active_bundle
    assert before == {name: (tmp_path / "resumed" / name).stat().st_mtime_ns for name in before}


class Client:
    def __init__(self, response):
        self.response = response

    def complete(self, **kwargs):
        request = json.loads(kwargs["messages"][0]["content"])["request"]
        assert "future_values" not in json.dumps(request)
        if isinstance(self.response, Exception):
            raise self.response
        if self.response == "legal":
            state = request["parent_state"]
            member = state["inventory"]["members"][0]
            replacement = member | {"member_id": f'fresh_{request["counter_draw"]}',
                "source_sha256": "code", "parent_ids": [member["member_id"]]}
            replacement.pop("policy_sha256")
            response = {"source_candidates": [{"local_id": "code", "code": SOURCE.split("\ndef lagged")[0]}],
                "proposals": [{"operator": "repair", "reason": "Train repair", "member_id": member["member_id"], "replacement": replacement}]}
            return LLMResponse(json.dumps(response))
        return LLMResponse(self.response)


@pytest.mark.parametrize("response,reason,calls", [("{", "malformed", 1), (TimeoutError(), "timeout", 1),
                                                  (None, "unavailable", 0), ("legal", None, 1)])
def test_provider_attempts_are_kernel_charged_and_fallback_is_deterministic(tmp_path, response, reason, calls, monkeypatch):
    force_dev_results(monkeypatch, True)
    client = None if response is None else Client(response)
    result = run_fixture(tmp_path / "run", task_budget=920, provider="hybrid", llm_client=client)
    attempts = [json.loads(path.read_bytes())["batch"] for path in (tmp_path / "run/numerical_qd/proposals").glob("*.json")]
    assert attempts[0]["attempts"][0]["failure_reason"] == reason
    assert result.budget["charged_use"]["llm_calls"] == calls
    assert result.accepted_steps == 1
    if reason:
        assert attempts[0]["attempts"][-1]["provider"] == "deterministic"
    checkpoint = json.loads((tmp_path / "run/checkpoint.json").read_bytes())
    assert set(checkpoint["budget_closures"]) == set(checkpoint["budget"]["closed_reservation_sha256s"])
    assert not checkpoint["budget"]["open_reservations"]


def test_account_only_closure_is_kernel_owned_and_cannot_be_promoted(kernel):
    parent = kernel.active_bundle()
    candidate = child(parent)
    permit = kernel.reserve_evaluation(candidate, ResourceUse(task_executions=2))
    result = kernel.close_evaluation(parent, candidate, permit=permit, status="passed",
        train_objectives={}, train_behavior_descriptors={},
        dev_comparison={"passed": False, "parent_metrics": {}, "candidate_metrics": {}},
        resource_use=ResourceUse(task_executions=1), account_only=True)
    assert kernel.budget.charged_use.task_executions == 1
    with pytest.raises(KernelAuthorityError):
        kernel.evaluate_transition(parent, candidate, target="retrieval", evaluation=result, permit=permit)
    kernel.finalize()
    restored = EvolutionKernel.resume(V2RunStore(kernel.store.root), kernel.budget.plan, monotonic=lambda: 0.0)
    assert restored.active_bundle() == parent


def test_account_only_cannot_carry_dev(kernel):
    parent = kernel.active_bundle()
    candidate = child(parent)
    permit = kernel.reserve_evaluation(candidate, ResourceUse(task_executions=1))
    with pytest.raises(KernelAuthorityError, match="Dev"):
        kernel.close_evaluation(parent, candidate, permit=permit, status="passed",
            train_objectives={}, train_behavior_descriptors={},
            dev_comparison={"passed": True, "parent_metrics": {"loss": 1.0}, "candidate_metrics": {"loss": 0.0}},
            resource_use=ResourceUse(), account_only=True)


def test_no_feasible_child_closes_work_preserves_parent_and_never_scores_dev(tmp_path):
    config, supply, manifest, adapter = fixture(task_budget=1, provider="hybrid")
    class InvalidSource(Client):
        def complete(self, **kwargs):
            response = super().complete(**kwargs)
            payload = json.loads(response.text)
            payload["source_candidates"][0]["code"] = SOURCE.split("\ndef lagged")[0].replace(
                "float(history[-1]) + 1.0", "float('nan')")
            return LLMResponse(json.dumps(payload))
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, InvalidSource("legal"))
    assert result.active_bundle.numerical_release_sha256 == supply.release.fingerprint
    assert result.accepted_steps == result.rejected_steps == 0
    assert result.occupied_cells == 0
    assert json.loads((tmp_path / "run/evaluation_complete.json").read_bytes())["summary"]["dev_accessed"] is False
    steps = [json.loads(path.read_bytes())["numerical_qd_step"]
        for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
        if "numerical_qd_step" in json.loads(path.read_bytes())]
    assert steps[0]["status"] == "no_feasible_child"
    assert not result.budget["open_reservations"]


def test_materialization_failure_reason_is_persisted_in_generation_status(tmp_path, monkeypatch):
    """Catches collapsing an actionable execution error into no_feasible_child."""
    config, supply, manifest, adapter = fixture(task_budget=920)

    def fail_materialization(*args, **kwargs):
        raise ValueError("recipe executable is absent from the verified member source")

    monkeypatch.setattr(adapter, "materialize_child", fail_materialization)
    run_numerical_qd(
        tmp_path / "run", config, supply, manifest, adapter, stop_after=1
    )
    steps = [
        json.loads(path.read_bytes())["numerical_qd_step"]
        for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
        if "numerical_qd_step" in json.loads(path.read_bytes())
    ]
    assert len(steps) == 1
    assert steps[0]["materialization_failures"] == [{
        "member_id": steps[0]["materialization_failures"][0]["member_id"],
        "error_type": "ValueError",
        "message": "recipe executable is absent from the verified member source",
    }]


def test_candidate_verification_failure_is_persisted_in_generation_status(tmp_path, monkeypatch):
    """A rejected candidate must not disappear behind no_feasible_child."""
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore

    config, supply, manifest, adapter = fixture(task_budget=920)
    original = NumericalQDRunStore.verify_candidate

    def reject_child(store, genome_sha256):
        payload = store._object(genome_sha256)
        if payload.get("generation", 0) > 0:
            raise ValueError("candidate source failed the executable Host gate")
        return original(store, genome_sha256)

    monkeypatch.setattr(NumericalQDRunStore, "verify_candidate", reject_child)
    run_numerical_qd(
        tmp_path / "run", config, supply, manifest, adapter, stop_after=1
    )
    step = next(
        json.loads(path.read_bytes())["numerical_qd_step"]
        for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
        if "numerical_qd_step" in json.loads(path.read_bytes())
    )

    assert step["materialization_failures"] == [{
        "member_id": step["materialization_failures"][0]["member_id"],
        "error_type": "ValueError",
        "message": "candidate source failed the executable Host gate",
    }]


def test_policy_only_materialization_failure_is_attributed_to_candidate_member(tmp_path, monkeypatch):
    """Catches diagnostics reading a missing/stale source-mutation member."""
    config, supply, manifest, adapter = fixture(task_budget=920)
    payload = config.to_payload()
    payload["mutation"]["operators"] = ["policy_tune"]
    config = NumericalQDConfigV2.from_payload(payload)

    def fail_materialization(*args, **kwargs):
        raise ValueError("policy-only candidate failed")

    monkeypatch.setattr(adapter, "materialize_child", fail_materialization)
    run_numerical_qd(
        tmp_path / "run", config, supply, manifest, adapter, stop_after=1
    )
    step = next(
        json.loads(path.read_bytes())["numerical_qd_step"]
        for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
        if "numerical_qd_step" in json.loads(path.read_bytes())
    )
    assert step["materialization_failures"] == [{
        "member_id": "seasonal_naive",
        "error_type": "ValueError",
        "message": "policy-only candidate failed",
    }]


def test_completed_but_infeasible_train_rung_records_closed_no_improvement(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    from evolving_loop.v2.numerical_qd.contracts import ConstraintReportV2
    config, supply, manifest, adapter = fixture(task_budget=920)
    original = runner.evaluate_numerical_child
    def constrained(*args, **kwargs):
        value = original(*args, **kwargs)
        return replace(value, constraints=ConstraintReportV2(False, ("joint_regret",)))
    monkeypatch.setattr(runner, "evaluate_numerical_child", constrained)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    steps = [json.loads(path.read_bytes())["numerical_qd_step"]
             for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
             if "numerical_qd_step" in json.loads(path.read_bytes())]
    assert steps[0]["status"] == "no_feasible_child"
    assert result.budget["charged_use"]["task_executions"] == 880
    assert result.occupied_cells == result.accepted_steps == 0


class ThreeChildren(Client):
    def complete(self, **kwargs):
        request = json.loads(kwargs["messages"][0]["content"])["request"]
        member = request["parent_state"]["inventory"]["members"][0]
        proposed = {key: value for key, value in member.items() if key != "policy_sha256"}
        return LLMResponse(json.dumps({"source_candidates": [
            {"local_id": f"source_{i}", "code": SOURCE.split("\ndef lagged")[0]} for i in range(3)],
            "proposals": [{"operator": "add", "reason": "Train alternative", "member": proposed | {
                "member_id": f"new_{i}", "source_sha256": f"source_{i}", "parent_ids": []}} for i in range(3)]}))


@pytest.mark.parametrize("boundary,expected_tasks", [(0, 2400), (1, 2424), (2, 2472), (3, 2520)])
def test_each_rung_and_dev_deadline_blocks_new_dispatch(tmp_path, monkeypatch, boundary, expected_tasks):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=2560, provider="hybrid")
    payload = config.to_payload()
    payload["mutation"]["operators"] = ["add"]
    config = NumericalQDConfigV2.from_payload(payload)
    original_rung, original_freeze = runner._rung, runner.freeze_qd_supply
    calls = []
    def rung(*args, **kwargs):
        if len(calls) == boundary:
            adapter.monotonic.advance(config.budget.search_deadline_seconds)
        calls.append(len(calls))
        return original_rung(*args, **kwargs)
    def freeze(*args, **kwargs):
        value = original_freeze(*args, **kwargs)
        if boundary == 3:
            adapter.monotonic.advance(config.budget.search_deadline_seconds)
        return value
    monkeypatch.setattr(runner, "_rung", rung)
    monkeypatch.setattr(runner, "freeze_qd_supply", freeze)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, ThreeChildren("legal"))
    assert result.budget["charged_use"]["task_executions"] == expected_tasks
    assert result.accepted_steps == result.rejected_steps == 0
    assert not result.budget["open_reservations"]
    assert result.budget["finalization_started"]
    complete = json.loads((tmp_path / "run/evaluation_complete.json").read_bytes())
    assert complete["summary"]["dev_accessed"] is False
    assert len(list((tmp_path / "run/numerical_qd/rungs").glob("*.json"))) == min(boundary, 3)


def test_provider_receives_the_counter_already_persisted(tmp_path):
    class CounterClient(Client):
        def complete(self, **kwargs):
            checkpoint = json.loads((tmp_path / "run/numerical_qd/checkpoint.json").read_bytes())
            assert checkpoint["counter"]["counter"] > 0
            return super().complete(**kwargs)
    run_fixture(tmp_path / "run", task_budget=920, provider="hybrid", llm_client=CounterClient("legal"))


def test_kernel_resume_requires_the_exact_accepted_typed_pair(tmp_path):
    config, supply, manifest, adapter = fixture(task_budget=920)
    run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    checkpoints = json.loads((tmp_path / "run/checkpoint.json").read_bytes())
    accepted_candidate = next(candidate for candidate, value in checkpoints["completed_transitions"].items()
                              if value["decision"] == "accept")
    train = json.loads((tmp_path / f"run/evaluations/{accepted_candidate}/train.json").read_bytes())
    identity = train["train_behavior_descriptors"]["numerical_artifacts_sha256"]
    (tmp_path / f"run/numerical_qd/objects/{identity}.json").unlink()
    with pytest.raises(KernelAuthorityError):
        EvolutionKernel.resume(V2RunStore(tmp_path / "run"), config.budget, monotonic=adapter.monotonic)


def test_complete_explore_has_exact_cumulative_charges_and_a_frozen_winner(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=2560, provider="hybrid")
    payload = config.to_payload()
    payload["mutation"]["operators"] = ["add"]
    config = NumericalQDConfigV2.from_payload(payload)
    selected = []
    original = runner.freeze_qd_supply
    def freeze(*args, **kwargs):
        result = original(*args, **kwargs)
        selected.extend(result.selected_genome_sha256s)
        return result
    monkeypatch.setattr(runner, "freeze_qd_supply", freeze)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, ThreeChildren("legal"))
    rungs = sorted((json.loads(path.read_bytes()) for path in (tmp_path / "run/numerical_qd/rungs").glob("*.json")),
                   key=lambda row: row["index"])
    assert [len(row["evaluations"]) for row in rungs] == [3, 2, 1]
    assert [row["budget_outcome"]["resource_use"]["task_executions"] for row in rungs] == [24, 48, 48]
    assert result.budget["charged_use"]["task_executions"] == 2560
    assert result.accepted_steps == 1
    assert len(list((tmp_path / "run/numerical_qd/results").rglob("*.json"))) == 120
    steps = [json.loads(path.read_bytes())["numerical_qd_step"]
             for path in (tmp_path / "run/numerical_qd/objects").glob("*.json")
             if "numerical_qd_step" in json.loads(path.read_bytes())]
    assert steps[0]["winner_genome_sha256"] in selected


def test_cached_partial_rows_reuse_exact_original_paid_results(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=2560, provider="hybrid")
    payload = config.to_payload()
    payload["mutation"]["operators"] = ["add"]
    config = NumericalQDConfigV2.from_payload(payload)
    original, calls = runner.evaluate_numerical_child, 0
    def evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = original(*args, **kwargs)
        if calls == 25:
            adapter.monotonic.advance(config.adapter["task_timeout_seconds"])
        return value
    monkeypatch.setattr(runner, "evaluate_numerical_child", evaluate)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, ThreeChildren("legal"), stop_after=1)
    root = tmp_path / "run"
    assert calls == 25 and result.budget["charged_use"]["task_executions"] == 2425
    partial = next(json.loads(path.read_bytes())["closed_partial_rung"] for path in (root / "numerical_qd/objects").glob("*.json")
                   if "closed_partial_rung" in json.loads(path.read_bytes()))
    assert partial["reason"] == "timeout" and len(partial["task_results"]) == 9
    checkpoint = json.loads((root / "checkpoint.json").read_bytes())
    receipts = []
    for ref in checkpoint["budget_closures"].values():
        folder = root / f"evaluations/{ref['candidate_bundle_sha256']}"
        receipts.extend(json.loads((folder / "train.json").read_bytes())["train_objectives"].get("material_receipts", []))
    for row in partial["task_results"]:
        name = f"results/{row['result']['candidate_sha256']}/{row['task_sha256']}.json"
        raw = (root / "numerical_qd" / name).read_bytes()
        assert json.loads(raw) == row["result"]
        matching = [receipt for receipt in receipts if receipt["relative_path"] == name]
        assert matching == [{"kind": "task_result", "relative_path": name,
            "content_sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}]
    before = files(root)
    run_numerical_qd(root, config, supply, manifest, adapter, ThreeChildren("legal"), resume=True, stop_after=1)
    assert files(root) == before


def test_task_timeout_closes_actual_partial_work_without_a_rung(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=920)
    original = runner.evaluate_numerical_child
    def evaluate(*args, **kwargs):
        value = original(*args, **kwargs)
        adapter.monotonic.advance(0.02)
        return value
    monkeypatch.setattr(runner, "evaluate_numerical_child", evaluate)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    assert result.budget["charged_use"]["task_executions"] == 801
    assert result.budget["charged_use"]["wall_seconds"] == 0.02
    assert not list((tmp_path / "run/numerical_qd/rungs").glob("*.json"))
    assert len(list((tmp_path / "run/numerical_qd/results").rglob("*.json"))) == 1
    partial = [json.loads(path.read_bytes())["closed_partial_rung"] for path in
               (tmp_path / "run/numerical_qd/objects").glob("*.json")
               if "closed_partial_rung" in json.loads(path.read_bytes())]
    assert len(partial) == 1 and len(partial[0]["task_results"]) == 1
    assert result.occupied_cells == result.accepted_steps == 0
    assert not result.budget["open_reservations"]


def test_finalization_reserve_before_proposal_never_opens_work(tmp_path, monkeypatch, material_catalog):
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
    config, supply, manifest, adapter = fixture(task_budget=920)
    original = NumericalQDRunStore.write_state
    count = 0
    initial, sizes = {}, []
    def checkpoint(*args, **kwargs):
        nonlocal count
        value = original(*args, **kwargs)
        count += 1
        if count == 1:
            initial.update(json.loads((tmp_path / "run/checkpoint.json").read_bytes()))
            root = tmp_path / "run/numerical_qd"
            sizes.extend(material_sizes(root, material_catalog))
            adapter.monotonic.advance(config.budget.search_deadline_seconds)
        return value
    monkeypatch.setattr(NumericalQDRunStore, "write_state", checkpoint)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    assert sizes and result.budget["charged_use"] == ResourceUse(artifact_bytes=sum(sizes)).to_payload()
    final = json.loads((tmp_path / "run/checkpoint.json").read_bytes())
    assert final["budget_closures"] == initial["budget_closures"]
    assert result.budget["closed_reservation_sha256s"] == initial["budget"]["closed_reservation_sha256s"]
    assert set(final["budget_closures"]) == set(result.budget["closed_reservation_sha256s"])
    assert not result.budget["open_reservations"]
    actual_sizes = []
    for reservation, reference in final["budget_closures"].items():
        closure = json.loads((tmp_path / f"run/evaluations/{reference['candidate_bundle_sha256']}/budget_closure.json").read_bytes())
        assert closure["reservation_sha256"] == reservation and closure["allowed"]
        size = closure["resource_use"]["artifact_bytes"]
        assert closure["resource_use"] == ResourceUse(artifact_bytes=size).to_payload()
        actual_sizes.append(size)
    assert sorted(actual_sizes) == sorted(sizes)
    assert not list((tmp_path / "run/numerical_qd/proposals").glob("*.json"))
    assert not list((tmp_path / "run/numerical_qd/rungs").glob("*.json"))
    assert not list((tmp_path / "run/numerical_qd/results").rglob("*.json"))
    assert not final["completed_transitions"]


def test_sources_policy_and_proposal_are_persisted_before_atomic_mutation(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    original = runner.apply_mutation
    def mutate(state, proposal):
        payload = proposal.to_payload()
        member = payload["replacement"]
        assert (tmp_path / f'run/numerical_qd/objects/{member["policy_sha256"]}.json').is_file()
        assert (tmp_path / f'run/numerical_qd/sources/{member["source_sha256"]}.py').is_file()
        assert any(payload in json.loads(path.read_bytes())["batch"]["proposals"]
                   for path in (tmp_path / "run/numerical_qd/proposals").glob("*.json"))
        return original(state, proposal)
    monkeypatch.setattr(runner, "apply_mutation", mutate)
    run_fixture(tmp_path / "run", task_budget=920)


def test_llm_source_recipe_is_persisted_before_atomic_mutation(tmp_path, monkeypatch):
    """Catches dropping the Host-derived recipe between proposal and execution."""
    from evolving_loop.v2.numerical_qd import runner

    class NewSource(Client):
        def complete(self, **kwargs):
            response = json.loads(super().complete(**kwargs).text)
            response["source_candidates"][0]["code"] = response["source_candidates"][0]["code"].replace(
                "seasonal_naive", "evolved_forecast"
            )
            return LLMResponse(json.dumps(response))

    original = runner.apply_mutation
    crossed = []

    def mutate(state, proposal):
        member = proposal.to_payload()["replacement"]
        policy_path = tmp_path / f'run/numerical_qd/objects/{member["policy_sha256"]}.json'
        policy = json.loads(policy_path.read_bytes())
        assert policy["name"] == "select_evolved_forecast"
        assert policy["parents"] == ["evolved_forecast"]
        crossed.append(member["member_id"])
        return original(state, proposal)

    monkeypatch.setattr(runner, "apply_mutation", mutate)
    run_fixture(
        tmp_path / "run", task_budget=1, provider="hybrid",
        llm_client=NewSource("legal"), stop_after=1,
    )
    assert len(crossed) == 1


def test_model_authored_source_policy_is_rejected_before_safe_fallback(tmp_path):
    class UnknownPolicy(Client):
        def complete(self, **kwargs):
            response = json.loads(super().complete(**kwargs).text)
            response["proposals"][0]["replacement"]["policy_sha256"] = "f" * 64
            return LLMResponse(json.dumps(response))
    result = run_fixture(tmp_path / "run", task_budget=1, provider="hybrid",
                         llm_client=UnknownPolicy("legal"), stop_after=1)
    attempts = [
        json.loads(path.read_bytes())["batch"]
        for path in (tmp_path / "run/numerical_qd/proposals").glob("*.json")
    ]
    assert attempts[0]["attempts"][0]["failure_reason"] == "malformed"
    assert attempts[0]["attempts"][-1]["provider"] == "deterministic"
    assert not (tmp_path / f"run/numerical_qd/objects/{'f' * 64}.json").exists()
    assert result.budget["charged_use"]["task_executions"] == 1
    assert result.accepted_steps == 0


def test_no_feasible_generation_continues_with_distinct_context_and_train_attempt_credit(tmp_path):
    config, supply, manifest, adapter = fixture(task_budget=921, provider="hybrid")
    class OnceInvalid(Client):
        def __init__(self):
            super().__init__("legal")
            self.count = 0
        def complete(self, **kwargs):
            response = super().complete(**kwargs)
            self.count += 1
            if self.count == 1:
                payload = json.loads(response.text)
                payload["source_candidates"][0]["code"] = SOURCE.split("\ndef lagged")[0].replace(
                    "float(history[-1]) + 1.0", "float('nan')")
                return LLMResponse(json.dumps(payload))
            return response
    result = run_numerical_qd(
        tmp_path / "run", config, supply, manifest, adapter, OnceInvalid(),
        finalize_after=1,
    )
    assert result.status == "numerical_qd_complete"
    assert result.accepted_steps == 1
    assert result.budget["charged_use"]["task_executions"] == 921
    records = [json.loads(path.read_bytes()) for path in (tmp_path / "run/numerical_qd/proposals").glob("*.json")]
    assert sorted(record["context"]["generation"] for record in records) == [1, 2]
    policy = json.loads((tmp_path / f"run/numerical_qd/objects/{result.mutation_policy_sha256}.json").read_bytes())
    assert policy["operators"]["repair"]["attempts"] == 2
    assert policy["operators"]["repair"]["feasible"] == 1
    assert json.loads((tmp_path / "run/evaluation_complete.json").read_bytes())["summary"]["provider_attempts"] == 2


def test_ticking_clock_keeps_proposal_estimate_bounded_and_uses_exact_kernel_budget(tmp_path):
    class TickClock:
        def __init__(self):
            self.seconds = 0.0
        def __call__(self):
            self.seconds += 0.0000001
            return self.seconds
    result = run_fixture(tmp_path / "run", task_budget=920, clock=TickClock())
    assert result.accepted_steps == 1
    checkpoint = json.loads((tmp_path / "run/checkpoint.json").read_bytes())
    completed = json.loads((tmp_path / "run/evaluation_complete.json").read_bytes())
    assert completed["summary"]["budget"] == checkpoint["budget"] == result.budget


def test_zero_worker_capacity_stops_without_repeated_proposal_dispatch(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=920)
    payload = config.to_payload()
    payload["budget"]["ceilings"]["subprocesses"] = 0
    config = NumericalQDConfigV2.from_payload(payload)
    original = runner.DeterministicProposalProvider.propose
    count = 0
    def propose(*args, **kwargs):
        nonlocal count
        count += 1
        assert count == 1, "dispatched another proposal after materialization budget denial"
        return original(*args, **kwargs)
    monkeypatch.setattr(runner.DeterministicProposalProvider, "propose", propose)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    assert result.budget["charged_use"]["task_executions"] == 1
    assert result.accepted_steps == 0


def test_mid_generation_checkpoint_cannot_silently_resample_or_replay(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore, NumericalQDStoreError
    config, supply, manifest, adapter = fixture()
    original = NumericalQDRunStore.write_state
    def checkpoint(store, **kwargs):
        result = original(store, **kwargs)
        if store._object(kwargs["hyperband_state_sha256"])["rungs"]:
            raise RuntimeError("stopped after a closed rung")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(NumericalQDRunStore, "write_state", checkpoint)
        with pytest.raises(RuntimeError, match="closed rung"):
            run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    before = files(tmp_path / "run")
    with pytest.raises(NumericalQDStoreError, match="unfinished generation"):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, resume=True, stop_after=1)
    assert files(tmp_path / "run") == before


@pytest.mark.parametrize("failure_at", [1, 3])
def test_materialization_charges_only_dispatches_begun_even_on_failure(tmp_path, failure_at):
    config, supply, manifest, adapter = fixture()
    class FailingAnchor(CheapStore):
        calls = 0
        def forecast(self, *args):
            self.calls += 1
            if self.calls == failure_at:
                raise ValueError("anchor failed")
            return super().forecast(*args)
    adapter.materializer.forecast_store = FailingAnchor()
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    assert adapter.materializer.forecast_store.calls == failure_at
    assert result.budget["charged_use"]["task_executions"] == 2 * failure_at
    assert result.budget["charged_use"]["subprocesses"] == 1
    assert not result.budget["open_reservations"]
    assert not list((tmp_path / "run/numerical_qd/rungs").glob("*.json"))


def test_successful_materialization_exact_dispatch_charge(tmp_path):
    result = run_fixture(tmp_path / "run", task_budget=5000, stop_after=1)
    assert result.accepted_steps == 1
    # 100 distinct histories x 2 methods x (full horizon + 3 hindcasts),
    # followed by Train80 and the Dev20 Parent/Child pair.
    assert result.budget["charged_use"]["task_executions"] == 920
    assert result.budget["charged_use"]["subprocesses"] == 1


def test_freeze_material_output_is_exactly_billed_once(tmp_path, material_catalog):
    config, supply, manifest, adapter = fixture(task_budget=920)
    root = tmp_path / "run"
    result = run_numerical_qd(root, config, supply, manifest, adapter)
    assert result.accepted_steps == 1
    assert result.budget["charged_use"]["artifact_bytes"] == sum(material_sizes(root, material_catalog))
    before = files(root)
    resumed = run_numerical_qd(root, config, supply, manifest, adapter, resume=True)
    assert resumed.budget == result.budget
    assert files(root) == before


def test_freeze_reserves_the_full_declared_task_timeout_envelope(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner

    config, supply, manifest, adapter = fixture(task_budget=920)
    payload = config.to_payload()
    payload["budget"]["hard_limit_seconds"] = 2
    payload["budget"]["ceilings"]["wall_seconds"] = 2.0
    config = NumericalQDConfigV2.from_payload(payload)
    original = runner._KernelWork.reserve_stage
    observed = []

    def reserve_stage(work, stage, estimate):
        if stage.startswith("freeze-"):
            observed.append(estimate.wall_seconds)
        return original(work, stage, estimate)

    monkeypatch.setattr(runner._KernelWork, "reserve_stage", reserve_stage)

    run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)

    assert observed == [len(adapter.tasks) * config.adapter["task_timeout_seconds"]]


@pytest.mark.parametrize("after_output", [False, True])
def test_freeze_failure_bills_only_material_already_produced(tmp_path, monkeypatch, after_output, material_catalog):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=920)
    root = tmp_path / "run"
    if after_output:
        original = runner._persist
        def persist(store, value, **kwargs):
            result = original(store, value, **kwargs)
            if type(value) is dict and set(value) == {"supply", "registry"} and (
                    value["registry"]["release_sha256"] != supply.release.fingerprint):
                raise ValueError("freeze failed after material output")
            return result
        monkeypatch.setattr(runner, "_persist", persist)
    else:
        def freeze(*args, **kwargs):
            raise ValueError("freeze failed before material output")
        monkeypatch.setattr(runner, "freeze_qd_supply", freeze)
    result = run_numerical_qd(root, config, supply, manifest, adapter, stop_after=1)
    assert result.budget["charged_use"]["task_executions"] == 880
    assert result.budget["charged_use"]["artifact_bytes"] == sum(material_sizes(root, material_catalog))
    assert not result.budget["open_reservations"]
    assert result.active_bundle.numerical_release_sha256 == supply.release.fingerprint
    before = files(root)
    resumed = run_numerical_qd(root, config, supply, manifest, adapter, resume=True, stop_after=1)
    assert resumed.budget == result.budget
    assert files(root) == before


@pytest.mark.parametrize("failure_at", [1, 7])
def test_partial_dev_failure_charges_only_comparisons_started(tmp_path, monkeypatch, failure_at):
    from evolving_loop.v2.numerical_qd import runner
    original = runner.drcik_point_metrics
    calls = 0
    def metric(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise ValueError("Dev metric failed")
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "drcik_point_metrics", metric)
    result = run_fixture(tmp_path / "run", stop_after=1)
    assert calls == failure_at
    assert result.accepted_steps == 0
    assert result.budget["charged_use"]["task_executions"] == 880 + failure_at
    assert not result.budget["open_reservations"]


@pytest.mark.parametrize("mode", ["success", "task_overrun", "deadline"])
def test_raw_seed_bootstrap_is_precommitted_charged_and_never_replayed(tmp_path, monkeypatch, mode):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture(task_budget=5000 if mode != "task_overrun" else 1, raw_seed=True)
    calls = 0
    class BootstrapStore(CheapStore):
        def forecast(self, *args):
            nonlocal calls
            preflight = json.loads((tmp_path / "run/seed_bootstrap_preflight.json").read_bytes())
            assert preflight["stage"] == "seed_bootstrap"
            assert not (tmp_path / "run/run_manifest.json").exists()
            calls += 1
            if mode == "deadline" and calls == 1:
                adapter.monotonic.advance(config.budget.search_deadline_seconds + 1.0)
            return super().forecast(*args)
    bootstrap_store = BootstrapStore()
    adapter.materializer.forecast_store = bootstrap_store
    original = runner._seed_registry
    def seed_registry(*args, **kwargs):
        assert (tmp_path / "run/seed_bootstrap_preflight.json").is_file()
        result = original(*args, **kwargs)
        adapter.materializer.forecast_store = CheapStore()
        return result
    monkeypatch.setattr(runner, "_seed_registry", seed_registry)
    if mode != "success":
        with pytest.raises(ValueError, match="bootstrap"):
            run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
        assert calls == 1
        receipt = json.loads((tmp_path / "run/seed_bootstrap_receipt.json").read_bytes())
        assert receipt["resource_use"]["task_executions"] == 1
        assert receipt["resource_use"]["wall_seconds"] == (config.budget.search_deadline_seconds + 1.0 if mode == "deadline" else 0.0)
        assert not receipt["budget_after"]["open_reservations"]
        assert not (tmp_path / "run/accepted_bundle.json").exists()
        assert not (tmp_path / "run/run_manifest.json").exists()
        assert not (tmp_path / "run/evaluation_complete.json").exists()
        before = files(tmp_path / "run")
        with pytest.raises(ValueError, match="bootstrap"):
            run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, resume=True)
        assert calls == 1
        assert files(tmp_path / "run") == before
        return
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    assert calls > 0
    assert result.budget["charged_use"]["task_executions"] == calls + (920 if mode == "success" else 0)
    assert result.accepted_steps == int(mode == "success")
    assert result.budget["charged_use"]["wall_seconds"] == (
        config.budget.search_deadline_seconds + 1.0 if mode == "deadline" else 0.0)
    assert not result.budget["open_reservations"]
    if mode != "success":
        assert result.status == "numerical_qd_complete"
        assert not list((tmp_path / "run/numerical_qd/proposals").glob("*.json"))
    before = files(tmp_path / "run")
    # Resume binds the original operator input identity. Restoring this store
    # also makes any accidental replay fail its pre-manifest assertion above.
    adapter.materializer.forecast_store = bootstrap_store
    resumed = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, resume=True, stop_after=1)
    assert resumed.budget == result.budget
    assert files(tmp_path / "run") == before


def test_bootstrap_bridge_exposes_real_time_to_every_later_stage(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    from evolving_loop.v2.numerical_qd.persistence import NumericalQDRunStore
    config, supply, manifest, adapter = fixture(task_budget=5000, raw_seed=True)
    original = NumericalQDRunStore.write_state
    def checkpoint(*args, **kwargs):
        result = original(*args, **kwargs)
        adapter.monotonic.advance(config.budget.search_deadline_seconds)
        return result
    monkeypatch.setattr(NumericalQDRunStore, "write_state", checkpoint)
    result = runner.run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    assert result.budget["charged_use"]["task_executions"] > 0
    assert result.accepted_steps == 0
    assert not list((tmp_path / "run/numerical_qd/proposals").glob("*.json"))


def test_raw_seed_preflight_zero_budget_never_opens_a_forecast(tmp_path):
    config, supply, manifest, adapter = fixture(task_budget=0, raw_seed=True)
    class ForbiddenStore:
        def forecast(self, *args):
            raise AssertionError("bootstrap forecast opened without admission")
    adapter.materializer.forecast_store = ForbiddenStore()
    with pytest.raises(ValueError, match="preflight budget denied"):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)


def test_bootstrap_authority_cannot_be_removed_from_the_kernel_manifest(tmp_path):
    from evolving_loop.v2.contracts import canonical_v2_bytes
    config, supply, manifest, adapter = fixture(task_budget=800, raw_seed=True)
    run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    path = tmp_path / "run/run_manifest.json"
    payload = json.loads(path.read_bytes())
    payload.pop("seed_bootstrap_preflight_sha256")
    path.write_bytes(canonical_v2_bytes(payload))
    with pytest.raises(KernelAuthorityError, match="bootstrap"):
        EvolutionKernel.resume(V2RunStore(tmp_path / "run"), config.budget, monotonic=adapter.monotonic)


def test_external_resource_declaration_is_rechecked_before_run_task_access(tmp_path):
    config, supply, manifest, adapter = fixture()
    class GPUStore(CheapStore):
        resource_kinds = ("gpu_seconds",)
        def forecast(self, *args):
            raise AssertionError("GPU access before reporter preflight")
    adapter.materializer.forecast_store = GPUStore()
    with pytest.raises(ValueError, match="resource reporter"):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    assert not (tmp_path / "run").exists()


def test_forecast_size_limit_is_enforced_before_any_task_access(tmp_path):
    config, supply, manifest, adapter = fixture(raw_seed=True)
    payload = config.to_payload()
    payload["adapter"]["max_forecast_values"] = 1
    config = NumericalQDConfigV2.from_payload(payload)
    with pytest.raises(ValueError, match="forecast.*limit"):
        run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter)
    assert not (tmp_path / "run").exists()


def test_reported_external_failure_closes_with_exact_gpu_native_token_use(tmp_path):
    config, supply, manifest, adapter = fixture()
    class GPUStore(CheapStore):
        resource_kinds = ("gpu_seconds", "subprocesses", "output_tokens")
        used = ResourceUse()
        def forecast(self, *args):
            self.used += ResourceUse(gpu_seconds=0.5, subprocesses=1, output_tokens=7)
            raise ValueError("GPU operation failed after beginning")
    external = GPUStore()
    adapter.materializer.forecast_store = external
    adapter.resource_reporter = lambda: external.used
    adapter.resource_reporter_sha256 = "a" * 64
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    charged = ResourceUse.from_payload(result.budget["charged_use"])
    assert charged.gpu_seconds == 0.5
    assert charged.output_tokens == 7
    assert charged.subprocesses == 2  # Verified source worker + reported native work.
    assert charged.task_executions == 2
    assert not result.budget["open_reservations"]


def test_rung_host_reporter_work_is_charged_even_when_evaluation_fails(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import runner
    config, supply, manifest, adapter = fixture()
    used = ResourceUse()
    adapter.resource_kinds = ("gpu_seconds", "output_tokens")
    adapter.resource_reporter = lambda: used
    adapter.resource_reporter_sha256 = "a" * 64
    def evaluate(*args, **kwargs):
        nonlocal used
        used += ResourceUse(gpu_seconds=0.25, output_tokens=3)
        raise ValueError("Host evaluation failed after GPU work")
    monkeypatch.setattr(runner, "evaluate_numerical_child", evaluate)
    result = run_numerical_qd(tmp_path / "run", config, supply, manifest, adapter, stop_after=1)
    assert result.budget["charged_use"]["gpu_seconds"] == 0.25
    assert result.budget["charged_use"]["output_tokens"] == 3
    assert result.budget["charged_use"]["task_executions"] == 801
