"""Frozen package Public-99 verification and reporting contracts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from evolving_loop.evaluate_frozen_package_bundle import (
    _claim_output,
    _inert_context_is_fixed,
    _PublicStateEvaluator,
    _public_supply_screening,
    FrozenPackageEvaluationError,
    build_attribution_states,
    main,
    score_frozen_states,
    verify_frozen_package_run,
)
from evolving_loop.package_numerical_supply import NumericalSupplyRelease
from numerical_agent.evolution.screening import (
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
)
from evolving_loop.package_artifacts import PackageArtifactStore, PackageCheckpoint
from evolving_loop.package_coordinate_evolution import package_principal_fingerprints
from evolving_loop.run_package_coevolution import _digest as _run_digest
from tests.test_package_coordinate_evolution import _bundle
from tests.test_package_stage_runner import _evaluation, _schedule


def _split_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sealed_run(tmp_path):
    evolution = tmp_path / "evolution"
    store = PackageArtifactStore(evolution)
    _task, state = _bundle(tmp_path / "bundle")
    schedule = _schedule()
    runtime = dict(state.bundle.runtime_fingerprints)
    core = {
        "schema_version": 1,
        "formal_run": True,
        "schedule_sha256": schedule.fingerprint,
        "split_manifest_sha256": "a" * 64,
        "runtime_fingerprints": runtime,
        "input_fingerprints": {"retrieval_seed_manifest": "b" * 64},
        "public_test_accessed": False,
    }
    store.write_run_manifest({**core, "run_sha256": _run_digest(core)})
    store.write_schedule(schedule.to_payload())
    parent_sha = state.bundle.fingerprint()
    principals = dict(package_principal_fingerprints(state.bundle))
    step = {
        "generation": 0,
        "target": "numerical",
        "accepted": False,
        "reason": "no finalist",
        "parent_fingerprints": principals,
        "child_fingerprints": principals,
        "accepted_fingerprints": principals,
        "changed_modules": [],
        "parent_bytes_sha256": parent_sha,
        "child_bytes_sha256": parent_sha,
        "accepted_bytes_sha256": parent_sha,
        "parent_registry_sha256": state.registry.fingerprint,
        "child_registry_sha256": state.registry.fingerprint,
        "accepted_registry_sha256": state.registry.fingerprint,
        "public_test_accessed": False,
        "accepted_bundle_payload": state.bundle.to_payload(),
    }
    store.append_coordinate_step(step)
    store.write_checkpoint(
        PackageCheckpoint(
            schema_version=1,
            run_sha256=_run_digest(core),
            schedule_sha256=schedule.fingerprint,
            initial_bundle_payload=state.bundle.to_payload(),
            current_bundle_payload=state.bundle.to_payload(),
            completed_steps=(step,),
            candidate_fingerprints=(),
            cache_fingerprints=(),
            consumed_stages=(),
        )
    )
    store.complete(
        state.bundle.to_payload(),
        accepted_steps=0,
        rejected_steps=1,
        formal_run=True,
    )
    return evolution, evolution / "final_bundle.json", runtime, state, schedule


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("missing_complete", "complete evolution"),
        ("trace_gap", "canonical trace"),
        ("detached_final", "final bundle"),
        ("public_marker", "Public access"),
        ("runtime_drift", "runtime"),
    ),
)
def test_public_evaluator_rejects_unsealed_or_drifted_run(
    tmp_path, mutation, match
):
    evolution, final_bundle, runtime, state, _schedule_value = _sealed_run(tmp_path)
    if mutation == "missing_complete":
        (evolution / "evaluation_complete.json").unlink()
    elif mutation == "trace_gap":
        row = json.loads((evolution / "coordinate_trace.jsonl").read_text())
        row["generation"] = 1
        (evolution / "coordinate_trace.jsonl").write_text(json.dumps(row) + "\n")
    elif mutation == "detached_final":
        detached = replace(
            state.bundle,
            policy=replace(state.bundle.policy, decision_prompt="detached"),
        )
        final_bundle.write_text(
            json.dumps(detached.to_payload(), sort_keys=True, separators=(",", ":"))
        )
        complete = json.loads((evolution / "evaluation_complete.json").read_text())
        complete["final_bundle_sha256"] = hashlib.sha256(final_bundle.read_bytes()).hexdigest()
        (evolution / "evaluation_complete.json").write_text(
            json.dumps(complete, sort_keys=True, separators=(",", ":"))
        )
    elif mutation == "public_marker":
        row = json.loads((evolution / "coordinate_trace.jsonl").read_text())
        row["public_test_accessed"] = True
        (evolution / "coordinate_trace.jsonl").write_text(json.dumps(row) + "\n")
    else:
        runtime = {**runtime, "llm_runtime": "9" * 64}

    with pytest.raises(FrozenPackageEvaluationError, match=match):
        verify_frozen_package_run(evolution, final_bundle, runtime)


def test_attribution_states_are_nested_and_report_has_no_probabilistic_metric(tmp_path):
    evolution, final_bundle, runtime, _state, _schedule_value = _sealed_run(tmp_path)
    verified = verify_frozen_package_run(evolution, final_bundle, runtime)
    states = build_attribution_states(verified)

    assert tuple(item.name for item in states) == (
        "initial_toto",
        "final_numerical_seed_context",
        "final_numerical_retrieval_seed_decision",
        "final_bundle",
    )
    assert tuple(item.direct_parent_name for item in states) == (
        None,
        "initial_toto",
        "final_numerical_seed_context",
        "final_numerical_retrieval_seed_decision",
    )

    task_ids = ("public_a", "public_b")

    class FakeEvaluator:
        def evaluate_state(self, named, tasks):
            error = 1.0 - 0.1 * tuple(item.name for item in states).index(named.name)
            return _evaluation(
                named.state.bundle.fingerprint(),
                tuple(task.numeric.task_id for task in tasks),
                error,
            )

        def details_for(self, name):
            return {task_id: (f"doc_{name}",) for task_id in task_ids}

    from tests.test_run_package_coevolution import _context_task

    report = score_frozen_states(
        tuple(_context_task(task_id) for task_id in task_ids), states, FakeEvaluator()
    )
    assert report["metadata"] == {
        "benchmark_role": "historically_consumed_final_regression",
        "selection_used_public": False,
        "public_result_may_feed_evolution": False,
        "primary_metrics": ["smae", "srmse"],
        "probabilistic_metric_reported": False,
    }
    assert "scrps" not in json.dumps(report).lower()
    assert report["states"]["final_bundle"]["task_count"] == 2
    assert report["comparisons"]["final_bundle_vs_initial_toto"]["ties"] == 2
    assert report["per_task"]["final_bundle"][0]["supporting_document_ids"] == [
        "doc_initial_toto"
    ]


def test_identical_attribution_states_are_scored_once(tmp_path):
    evolution, final_bundle, runtime, _state, _schedule_value = _sealed_run(tmp_path)
    verified = verify_frozen_package_run(evolution, final_bundle, runtime)
    states = build_attribution_states(verified)
    task_ids = ("public_a", "public_b")

    class CountingEvaluator:
        def __init__(self):
            self.calls = []

        def evaluate_state(self, named, tasks):
            self.calls.append(named.name)
            return _evaluation(
                named.state.bundle.fingerprint(),
                tuple(task.numeric.task_id for task in tasks),
                0.5,
            )

        def details_for(self, name):
            return {task_id: (f"doc_{name}",) for task_id in task_ids}

    from tests.test_run_package_coevolution import _context_task

    evaluator = CountingEvaluator()
    report = score_frozen_states(
        tuple(_context_task(task_id) for task_id in task_ids), states, evaluator
    )

    assert evaluator.calls == ["initial_toto"]
    assert report["comparisons"]["final_bundle_vs_initial_toto"] == {
        "wins": 0,
        "ties": 2,
        "losses": 0,
        "mean_delta_smae": 0.0,
        "mean_delta_srmse": 0.0,
        "mean_delta_joint": 0.0,
    }
    assert report["per_task"]["final_bundle"][0]["supporting_document_ids"] == [
        "doc_initial_toto"
    ]


def test_public_claim_resumes_only_an_explicit_unpublished_start(tmp_path):
    evolution = tmp_path / "evolution"
    evolution.mkdir()
    output = tmp_path / "public"

    _claim_output(output, evolution)
    with pytest.raises(FrozenPackageEvaluationError, match="already claimed"):
        _claim_output(output, evolution)

    _claim_output(output, evolution, resume_unscored_start=True)
    (output / "llm-cache").mkdir()
    _claim_output(output, evolution, resume_unscored_start=True)
    (output / "public_report.json").write_text("{}")
    with pytest.raises(FrozenPackageEvaluationError, match="contains artifacts"):
        _claim_output(output, evolution, resume_unscored_start=True)


def test_empty_handoff_with_selected_anchor_is_a_fixed_context_path():
    from tests.test_package_retrieval_evolution import _package

    original = _package()
    anchor = original.protected_baseline
    package = replace(
        original,
        retrieval_handoff=(),
        accepted_assumptions=(),
        selection_decision=replace(
            original.selection_decision,
            selected=(anchor.name,),
            weights=(1.0,),
            forecast=anchor.forecast,
        ),
        final_forecast=anchor.forecast,
    )

    assert _inert_context_is_fixed(package)
    assert not _inert_context_is_fixed(
        replace(
            package,
            retrieval_handoff=original.retrieval_handoff,
            accepted_assumptions=original.accepted_assumptions,
        )
    )


def test_public_registry_uses_verified_source_fingerprints(monkeypatch, tmp_path):
    evolution, final_bundle, runtime, _state, _schedule_value = _sealed_run(tmp_path)
    verified = verify_frozen_package_run(evolution, final_bundle, runtime)
    trusted = {"methods": "1" * 64, "dictionary": "2" * 64}
    evaluator = object.__new__(_PublicStateEvaluator)
    screening = ScreeningPolicy(
        entries=(
            ScreeningEntry(
                "specialist",
                "tsfm",
                "keep",
                ApplicabilityPolicy(),
                "reviewed test candidate",
            ),
        ),
        fallback_names=("specialist",),
    )
    evaluator.resources = SimpleNamespace(
        store=object(),
        screening=screening,
        source_fingerprints=trusted,
        portfolio=SimpleNamespace(combined=()),
    )
    evaluator.verified = verified
    evaluator.atlas = None
    evaluator._registries = {}
    seen = {}
    registry = SimpleNamespace(fingerprint="registry")

    def capture(_tasks, _release, materializer, **kwargs):
        seen["source_fingerprints"] = materializer.source_fingerprints
        seen["fallback_screening_policy"] = kwargs["fallback_screening_policy"]
        return registry

    monkeypatch.setattr(
        "evolving_loop.evaluate_frozen_package_bundle._build_registry", capture
    )

    assert evaluator._registry(build_attribution_states(verified)[0], ()) is registry
    assert seen["source_fingerprints"] == trusted
    assert seen["fallback_screening_policy"] is None


def test_public_screening_limits_toto_release_to_frozen_supply():
    from tests.test_package_retrieval_evolution import _release

    champion = _release()
    assumption = replace(
        champion.policy.recipe.assumptions[0], candidate_name="toto_2_0"
    )
    recipe = replace(
        champion.policy.recipe,
        name="toto_anchor",
        parents=("toto_2_0",),
        fallback_parent="toto_2_0",
        assumptions=(assumption,),
    )
    champion = replace(
        champion,
        policy=replace(champion.policy, recipe=recipe),
    )
    release = NumericalSupplyRelease(
        schema_version=1,
        version="n000",
        parent_sha256=None,
        anchor_release_payload=champion.to_payload(),
        alternatives=(),
        atlas_release_sha256=None,
        source_fingerprints={"dictionary": "4" * 64},
        runtime_fingerprints={"materializer": "5" * 64},
    )
    policy = ScreeningPolicy(
        entries=(
            ScreeningEntry(
                "unrelated",
                "statistical",
                "keep",
                ApplicabilityPolicy(),
                "not in the frozen supply",
            ),
            ScreeningEntry(
                "toto_2_0",
                "tsfm",
                "keep",
                ApplicabilityPolicy(),
                "frozen anchor",
            ),
        ),
        fallback_names=("unrelated", "toto_2_0"),
    )

    restricted = _public_supply_screening(policy, (), release)

    assert tuple(item.name for item in restricted.entries) == ("toto_2_0",)
    assert restricted.fallback_names == ()


def _split_payload(public_ids: list[str]) -> dict[str, object]:
    train = [f"train_{index:03d}" for index in range(80)]
    dev = [f"dev_{index:03d}" for index in range(20)]
    payload: dict[str, object] = {
        "selection_uses_future_values": False,
        "selection_uses_gt_evidence": False,
        "selection_uses_document_labels": False,
        "target_sizes": {"train": 80, "dev": 20, "public_test": 99},
        "actual_sizes": {"train": 80, "dev": 20, "public_test": 99},
        "partitions": {
            "train": {"task_ids": train},
            "dev": {"task_ids": dev},
            "public_test": {"task_ids": public_ids},
        },
    }
    payload["manifest_sha256"] = _split_digest(payload)
    return payload


def test_public_evaluator_loads_exactly_99_public_tasks(monkeypatch, tmp_path):
    evolution, final_bundle, runtime, _state, _schedule_value = _sealed_run(tmp_path)
    public_ids = [f"public_{index:03d}" for index in range(99)]
    split = _split_payload(public_ids)
    # Bind the fixture run to this exact split authority.
    manifest = json.loads((evolution / "run_manifest.json").read_text())
    core = {key: value for key, value in manifest.items() if key != "run_sha256"}
    core["split_manifest_sha256"] = split["manifest_sha256"]
    run_sha = _run_digest(core)
    (evolution / "run_manifest.json").write_text(
        json.dumps(
            {**core, "run_sha256": run_sha},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    checkpoint = json.loads((evolution / "checkpoint.json").read_text())
    checkpoint["run_sha256"] = run_sha
    (evolution / "checkpoint.json").write_text(
        json.dumps(checkpoint, sort_keys=True, separators=(",", ":"))
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    seen: list[str] = []

    from tests.test_run_package_coevolution import _context_task

    monkeypatch.setattr(
        "evolving_loop.evaluate_frozen_package_bundle.load_context_tasks_by_ids",
        lambda _path, ids: seen.extend(ids)
        or tuple(_context_task(task_id) for task_id in ids),
    )
    monkeypatch.setattr(
        "evolving_loop.evaluate_frozen_package_bundle._runtime_authority",
        lambda _args, _manifest: (runtime, None),
    )
    monkeypatch.setattr(
        "evolving_loop.evaluate_frozen_package_bundle._public_evaluator",
        lambda *_args, **_kwargs: type(
            "Evaluator",
            (),
            {
                "evaluate_state": lambda self, named, tasks: _evaluation(
                    named.state.bundle.fingerprint(),
                    tuple(task.numeric.task_id for task in tasks),
                    0.5,
                ),
                "close": lambda self: None,
            },
        )(),
    )
    output = tmp_path / "public-output"
    args = [
        "--evolution-dir", str(evolution),
        "--final-bundle", str(final_bundle),
        "--split-file", str(split_path),
        "--tasks-file", str(tmp_path / "tasks"),
        "--repo", str(tmp_path / "repo"),
        "--forecast-store", str(tmp_path / "store"),
        "--output-dir", str(output),
    ]

    assert main(args) == 0
    assert seen == public_ids
    assert len(seen) == 99
    report = json.loads((output / "public_report.json").read_text())
    assert report["metadata"]["public_result_may_feed_evolution"] is False
    with pytest.raises(FrozenPackageEvaluationError, match="already"):
        main(args)
