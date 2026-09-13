from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.package_numerical_supply import build_package_registry
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.cooperative import CooperativeCheckpointV2, CooperativeRunResultV2
from evolving_loop.v2.numerical_qd.adapters import (
    FrozenNumericalArtifactsV2,
    import_numerical_seed,
)
from evolving_loop.v2.real.host import RealHostRuntimeV2
from numerical_agent.evolution.screening import (
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
)
from tests.test_evolution_v2_cooperative_cli import _payload as cooperative_payload
from tests.test_package_numerical_supply import (
    _package_for_task,
    _registry_tasks,
    _supply_release,
)
from tests.test_package_stage_runner import _evaluation


ROOT = Path(__file__).resolve().parents[1]


def _tasks_100(*, entity_conflicts=False):
    base = _registry_tasks()[0]
    tasks = tuple(
        replace(
            base,
            numeric=replace(
                base.numeric,
                task_id=f"real-task-{index:03d}",
                entity_name=f"Real Entity {index:03d}",
            ),
        )
        for index in range(100)
    )
    if entity_conflicts:
        entities = ("A", "A", "B", "C", "D") + ("A",) * 75 + ("A", "E") + ("F",) * 18
        tasks = tuple(
            replace(task, numeric=replace(task.numeric, entity_name=entity))
            for task, entity in zip(tasks, entities, strict=True)
        )
    return tasks


@pytest.mark.parametrize(
    "train_entities,dev_entities,train_indices,dev_index",
    [
        (("A", "A", "B", "C", "D", "E"), ("A", "Z"), (2, 3, 4, 5), 0),
        (("A", "A", "B", "C", "D"), ("A", "Z"), (0, 2, 3, 4), 1),
        (("D", "C", "B", "A", "A"), ("Z", "A"), (0, 1, 2, 3), 0),
    ],
)
def test_real_projection_uses_first_feasible_dev_and_host_order(
    train_entities, dev_entities, train_indices, dev_index
):
    from evolving_loop.v2.real import host as module

    # Deliberately expose only entity identity: selecting must not read labels/metrics.
    train = tuple(SimpleNamespace(numeric=SimpleNamespace(entity_name=name)) for name in train_entities)
    dev = tuple(SimpleNamespace(numeric=SimpleNamespace(entity_name=name)) for name in dev_entities)
    select = getattr(module, "select_real_task_projection", None)
    assert callable(select), "the real stages need a shared entity-disjoint selector"
    expected = (tuple(train[index] for index in train_indices), (dev[dev_index],))
    assert select(train, dev) == expected
    assert select(train, dev) == expected
    assert all(actual is original for actual, original in zip(select(train, dev)[0], expected[0]))


@pytest.mark.parametrize("train_entities,dev_entities", [
    (("A", "A", "B", "C"), ("Z",)),
    (("A", "B", "C", "D"), ("A", "B")),
    (("A", "B", "C", "D"), ()),
])
def test_real_projection_fails_closed_when_no_disjoint_4_1_exists(train_entities, dev_entities):
    from evolving_loop.v2.real import host as module

    train = tuple(SimpleNamespace(numeric=SimpleNamespace(entity_name=name)) for name in train_entities)
    dev = tuple(SimpleNamespace(numeric=SimpleNamespace(entity_name=name)) for name in dev_entities)
    select = getattr(module, "select_real_task_projection", None)
    assert callable(select), "the real stages need a shared entity-disjoint selector"
    with pytest.raises(ValueError, match="entity-disjoint Train4/Dev1"):
        select(train, dev)


def _p2_pair(tasks) -> FrozenNumericalArtifactsV2:
    release = _supply_release(alternatives=())
    registry = build_package_registry(
        tasks,
        release,
        lambda task, supplied: _package_for_task(task, supplied),
    )
    envelope = import_numerical_seed(release, registry, tasks=tasks).envelope
    return FrozenNumericalArtifactsV2(release, registry, envelope, ())


def _host(
    tasks,
    shared_llm,
    numerical_alternatives: tuple[FrozenNumericalArtifactsV2, ...] = (),
) -> RealHostRuntimeV2:
    return RealHostRuntimeV2(
        manifest=object(),
        tasks=tasks,
        train_tasks=tasks[:80],
        dev_tasks=tasks[80:],
        forecast_store=object(),
        runtime_registry=object(),
        llm_client=shared_llm,
        retrieval_skill_library=RetrievalSkillLibrary(
            ROOT / "unused-real-skills.json", persist=False
        ).clone(persist=False, read_only=True),
        source_repo=ROOT,
        screening_policy=ScreeningPolicy(
            (
                ScreeningEntry(
                    "safe_anchor",
                    "statistical",
                    "keep",
                    ApplicabilityPolicy(),
                    "reviewed",
                ),
            ),
            ("safe_anchor",),
        ),
        resource_reporter_sha256="9" * 64,
        numerical_alternatives=numerical_alternatives,
    )


def _real_config_payload():
    payload = cooperative_payload()
    control = dict(payload["control"])
    control["profile"] = "pilot"
    control["runtime_fingerprints"] = {"real_host": "9" * 64}
    payload["control"] = control
    return payload


def test_real_bridge_projects_4_1_but_preserves_p2_100_task_registry(
    tmp_path, monkeypatch
):
    from evolving_loop.v2.real import bridges

    tasks = _tasks_100()
    p2 = _p2_pair(tasks)
    shared_llm = object()
    host = _host(tasks, shared_llm)

    class Observed(RuntimeError):
        pass

    def observe(_output, _config, seed, projected, adapters, **_kwargs):
        assert seed["numerical"] is p2
        assert tuple(p2.envelope.entries) == tuple(
            sorted(task.numeric.task_id for task in tasks)
        )
        assert projected == {"train": tasks[:4], "dev": tasks[80:81]}
        retrieval = adapters["pipeline"].retrieval_factory(
            seed["retrieval"].genome, host.retrieval_skill_library
        )
        decision = adapters["pipeline"].decision_factory(seed["decision"])
        assert retrieval.llm is shared_llm
        assert decision.llm is shared_llm
        raise Observed

    monkeypatch.setattr(bridges, "run_cooperative_evolution", observe)

    with pytest.raises(Observed):
        bridges.run_real_cooperative(
            p2=p2,
            host=host,
            config_payload=_real_config_payload(),
            output_dir=tmp_path / "p3",
        )


@pytest.fixture(params=[False, True], ids=["unique-entities", "conflicting-entities"])
def sealed_p3(tmp_path, monkeypatch, request):
    from evolving_loop.v2.cooperative.adapters import CooperativePipelineAdapter
    from evolving_loop.v2.real.bridges import run_real_cooperative

    tasks = _tasks_100(entity_conflicts=request.param)
    p2 = _p2_pair(tasks)
    host = _host(tasks, object())

    def deterministic_evaluate(self, bundle, projected, stage):
        del self, stage
        error = 2.0 if bundle.generation == 0 else 3.0
        return _evaluation(
            bundle.fingerprint(),
            tuple(task.numeric.task_id for task in projected),
            error,
        )

    monkeypatch.setattr(
        CooperativePipelineAdapter, "evaluate", deterministic_evaluate
    )
    output = tmp_path / "p3"
    result = run_real_cooperative(
        p2=p2,
        host=host,
        config_payload=_real_config_payload(),
        output_dir=output,
    )
    return output, tasks, host, p2, result


def test_real_bridge_persists_canonical_proposal_space_and_loads_exact_closure(
    sealed_p3,
):
    from evolving_loop.v2.real.bridges import load_sealed_bundle_closure

    output, tasks, host, p2, result = sealed_p3
    closure = load_sealed_bundle_closure(output, tasks=tasks, host=host)
    raw_manifest = (output / "proposal_space_manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)

    assert result.status == "cooperative_complete"
    assert raw_manifest == canonical_v2_bytes(manifest)
    assert closure.active_bundle.fingerprint() == result.active_bundle_sha256
    assert closure.numerical.release.fingerprint == p2.release.fingerprint
    assert closure.numerical.registry.fingerprint == p2.registry.fingerprint
    assert closure.catalog.resolve_retrieval(
        closure.active_bundle.retrieval_release_sha256
    ) == closure.retrieval
    assert closure.catalog.resolve_decision(
        closure.active_bundle.decision_policy_sha256
    ) == closure.decision
    assert closure.runtime_identity == "9" * 64
    if tasks[0].numeric.entity_name == "A":
        assert closure.train_tasks == (tasks[0], tasks[2], tasks[3], tasks[4])
        assert closure.dev_tasks == (tasks[81],)
    else:
        assert closure.train_tasks == tasks[:4]
        assert closure.dev_tasks == tasks[80:81]
    assert closure.metric_cap == 5.0
    assert closure.config_sha256 == manifest["config_sha256"]
    assert closure.numerical_alternatives == ()
    assert closure.decision_prompts == (
        "Prefer the lowest finite complete-pipeline error.",
    )
    assert closure.decision_settings_cycle is False
    assert closure.p5_handoff_available is True
    assert closure.p5_handoff_reason is None

    progress = [
        json.loads(line)
        for line in (output / "cooperative_progress.jsonl").read_text().splitlines()
    ]
    persisted = {
        path.stem for path in (output / "archive/objects").glob("*.json")
    }
    expected_candidates = tuple(
        row["candidate_sha256"]
        for row in progress
        if row["candidate_sha256"] in persisted
    )
    assert tuple(
        bundle.fingerprint() for bundle in closure.closed_candidate_bundles
    ) == expected_candidates
    assert len(closure.acceptance_evidence) == len(expected_candidates)


def test_p5_handoff_threshold_counts_seed_plus_one_closed_candidate(sealed_p3):
    from evolving_loop.v2.real.bridges import _p5_handoff_available

    output, tasks, host, _p2, _result = sealed_p3
    from evolving_loop.v2.real.bridges import load_sealed_bundle_closure

    closure = load_sealed_bundle_closure(output, tasks=tasks, host=host)
    assert closure.active_bundle.generation == 0
    assert closure.closed_candidate_bundles
    assert _p5_handoff_available(
        closure.active_bundle, (closure.closed_candidate_bundles[0],)
    )
    assert not _p5_handoff_available(closure.active_bundle, ())


def test_closure_rejects_proposal_space_or_archive_drift(sealed_p3):
    from evolving_loop.v2.kernel import KernelAuthorityError
    from evolving_loop.v2.real.bridges import load_sealed_bundle_closure

    output, tasks, host, _p2, _result = sealed_p3
    manifest_path = output / "proposal_space_manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    manifest["runtime_identity"] = "8" * 64
    manifest_path.write_bytes(canonical_v2_bytes(manifest))
    with pytest.raises(KernelAuthorityError, match="proposal-space|runtime"):
        load_sealed_bundle_closure(output, tasks=tasks, host=host)

    manifest_path.write_bytes(original)
    progress = [
        json.loads(line)
        for line in (output / "cooperative_progress.jsonl").read_text().splitlines()
    ]
    candidate = next(
        row["candidate_sha256"]
        for row in progress
        if (output / "archive/objects" / f"{row['candidate_sha256']}.json").exists()
    )
    bundle_path = output / "archive/objects" / f"{candidate}.json"
    bundle_path.write_bytes(bundle_path.read_bytes().replace(b'"generation":1', b'"generation":2'))
    with pytest.raises(KernelAuthorityError):
        load_sealed_bundle_closure(output, tasks=tasks, host=host)


def test_real_bridge_passes_typed_numerical_alternatives_to_p3(
    tmp_path, monkeypatch
):
    from evolving_loop.v2.real import bridges

    tasks = _tasks_100()
    p2 = _p2_pair(tasks)
    alternative = _p2_pair(tasks)
    host = _host(tasks, object(), (alternative,))

    class Observed(RuntimeError):
        pass

    def observe(_output, _config, _seed, _projected, adapters, **_kwargs):
        assert adapters["numerical"]._alternatives == (alternative,)
        raise Observed

    monkeypatch.setattr(bridges, "run_cooperative_evolution", observe)

    with pytest.raises(Observed):
        bridges.run_real_cooperative(
            p2=p2,
            host=host,
            config_payload=_real_config_payload(),
            output_dir=tmp_path / "p3",
        )


def test_closure_rejects_metric_cap_unbound_from_committed_config(sealed_p3):
    from evolving_loop.v2.kernel import KernelAuthorityError
    from evolving_loop.v2.real.bridges import load_sealed_bundle_closure

    output, tasks, host, _p2, _result = sealed_p3
    path = output / "proposal_space_manifest.json"
    manifest = json.loads(path.read_bytes())
    assert manifest["config"]["metric_cap"] == manifest["metric_cap"]
    manifest["metric_cap"] = 4.0
    path.write_bytes(canonical_v2_bytes(manifest))

    with pytest.raises(KernelAuthorityError, match="config|metric_cap"):
        load_sealed_bundle_closure(output, tasks=tasks, host=host)


@pytest.mark.parametrize(
    ("artifact", "tamper"),
    (
        (
            "evaluation_complete.json",
            lambda payload: payload | {"accepted_steps": 1, "rejected_steps": 3},
        ),
        (
            "evaluation_complete.json",
            lambda payload: payload | {"scheduler_state_sha256": "0" * 64},
        ),
        (
            "cooperative_checkpoint.json",
            lambda payload: CooperativeCheckpointV2.seal(
                **(
                    {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
                    | {"kernel_checkpoint_sha256": "0" * 64}
                )
            ).to_payload(),
        ),
    ),
)
def test_closure_authenticates_completion_and_kernel_linkage(
    sealed_p3, artifact, tamper
):
    from evolving_loop.v2.kernel import KernelAuthorityError
    from evolving_loop.v2.real.bridges import load_sealed_bundle_closure

    output, tasks, host, _p2, _result = sealed_p3
    path = output / artifact
    payload = json.loads(path.read_bytes())
    path.write_bytes(canonical_v2_bytes(tamper(payload)))

    with pytest.raises(KernelAuthorityError, match="completion|checkpoint|kernel"):
        load_sealed_bundle_closure(output, tasks=tasks, host=host)


def test_closure_authenticates_progress_continuity(sealed_p3):
    from evolving_loop.v2.kernel import KernelAuthorityError
    from evolving_loop.v2.real.bridges import load_sealed_bundle_closure

    output, tasks, host, _p2, _result = sealed_p3
    path = output / "cooperative_progress.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) > 1
    rows[1]["active_before"] = "0" * 64
    path.write_bytes(b"".join(canonical_v2_bytes(row) for row in rows))

    with pytest.raises(KernelAuthorityError, match="progress|checkpoint"):
        load_sealed_bundle_closure(output, tasks=tasks, host=host)
