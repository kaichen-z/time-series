from __future__ import annotations

import math
from dataclasses import replace

import pytest

from evolving_loop.v2.contracts import fingerprint_payload
from evolving_loop.package_registry import task_registry_fingerprint
from evolving_loop.v2.protocol import (
    InfrastructureProtocolV2,
    ProtocolComponentV2,
    ProtocolHostInputs,
    ProtocolRuntimeRegistry,
    migrate_envelope,
)
from tests.test_evolution_v2_cooperative_pipeline import pipeline_case


@pytest.fixture
def runtime_case(tmp_path):
    """The P3 fixture remains real: only protocol choices vary."""
    p3 = pipeline_case.__wrapped__(tmp_path)
    tasks = p3["tasks"]
    baseline_calls: list[dict] = []

    host = ProtocolHostInputs(
        raw_fixture_records={task.numeric.task_id: task for task in tasks},
        canonical_records={task.numeric.task_id: task for task in tasks},
        catalog=p3["catalog"],
        frozen_numerical=p3["catalog"].resolve_numerical(
            p3["bundle"].numerical_release_sha256,
            p3["bundle"].numerical_registry_sha256,
        ),
        retrieval_factory=p3["pipeline"].retrieval_factory,
        decision_factory=p3["pipeline"].decision_factory,
        committed_task_metadata={
            task.numeric.task_id: task_registry_fingerprint(task) for task in tasks
        },
        l0_commitment_sha256="a" * 64,
        primary_metric_cap=5.0,
        baseline_verifier=lambda evidence: baseline_calls.append(evidence) or evidence.get("baseline") is True,
        known_supports={"document-1": ("support-1",)},
    )

    def resolve(**choices):
        defaults = {
            "backbone": "last_value",
            "loader": "canonical_json",
            "verifier_strategy": "exact_support",
            "diagnostic_metric": "forecast_spread",
            "schema_migration": "identity_envelope",
        }
        defaults.update(choices)
        components = tuple(
            ProtocolComponentV2(kind, defaults[kind], 1, 1)
            for kind in ("backbone", "loader", "verifier_strategy", "diagnostic_metric", "schema_migration")
        )
        return ProtocolRuntimeRegistry().resolve(
            InfrastructureProtocolV2(1, 1, None, "a" * 64, components), host
        )

    return resolve, p3["bundle"], tasks, baseline_calls


def test_migration_preserves_embedded_identity_and_is_idempotent():
    """Changing the wrapper schema must never change an archived artifact."""
    content = {"bundle": "f" * 64}
    old = {
        "schema_version": 1,
        "artifact": content,
        "artifact_sha256": fingerprint_payload(content),
    }

    new = migrate_envelope(old, 2)

    assert new == {
        "schema_version": 2,
        "content": content,
        "content_sha256": old["artifact_sha256"],
    }
    assert migrate_envelope(new, 2) == new
    assert old == {
        "schema_version": 1,
        "artifact": content,
        "artifact_sha256": fingerprint_payload(content),
    }


@pytest.mark.parametrize(
    "payload,target",
    (
        ({"schema_version": 1, "artifact": {}, "artifact_sha256": "0" * 64}, 2),
        ({"schema_version": 2, "content": {}, "content_sha256": "0" * 64}, 1),
        ({"schema_version": 3, "content": {}, "content_sha256": "0" * 64}, 2),
        ({"schema_version": 1, "artifact": {}, "artifact_sha256": "0" * 64, "x": 1}, 2),
    ),
)
def test_migration_rejects_unverified_or_noncanonical_envelopes(payload, target):
    """A bad embedded identity must never be adapted into a new wrapper."""
    with pytest.raises(ValueError):
        migrate_envelope(payload, target)


def test_runtime_diagnostics_are_finite_and_do_not_accept_empty_series():
    """Diagnostic adapters reject values that cannot safely enter artifacts."""
    from evolving_loop.v2.protocol.runtime import _diagnostic

    assert _diagnostic("forecast_spread", (0.0, 2.0), (1.0, 3.0)) == {
        "forecast_spread": 2.0
    }
    assert _diagnostic("absolute_movement", (0.0, 2.0), (1.0, 3.0)) == {
        "absolute_movement": 1.0
    }
    for history, forecast in (((), (1.0,)), ((1.0,), ()), ((math.nan,), (1.0,))):
        with pytest.raises(ValueError):
            _diagnostic("forecast_spread", history, forecast)


def test_backbone_versions_change_real_pipeline_forecasts(runtime_case):
    """A protocol backbone changes the scorer's actual numerical inputs."""
    resolve, bundle, tasks, _calls = runtime_case

    old = resolve(backbone="last_value").evaluate(bundle, tasks, "train")
    new = resolve(backbone="history_mean").evaluate(bundle, tasks, "train")

    assert {row.final_forecast for row in old.task_rows} == {(3.0, 3.0)}
    assert {row.final_forecast for row in new.task_rows} == {(2.0, 2.0)}
    assert new.mean_smae < old.mean_smae


def test_runtime_never_scores_public_or_noncanonical_stages(runtime_case):
    """Public is a frozen handoff boundary, not a protocol scoring stage."""
    resolve, bundle, tasks, _calls = runtime_case

    with pytest.raises(ValueError, match="train or dev"):
        resolve().evaluate(bundle, tasks, "public")


def test_alternate_loader_preserves_identity_and_changed_history_rejects(runtime_case):
    """Aliases use the full canonical Host record; altered history never does."""
    resolve, _bundle, _tasks, _calls = runtime_case

    canonical = resolve(loader="canonical_json").load_tasks()
    alternate = resolve(loader="alternate_history_json").load_tasks()

    assert tuple(task_registry_fingerprint(task) for task in canonical) == tuple(
        task_registry_fingerprint(task) for task in alternate
    )
    with pytest.raises(ValueError, match="changed history"):
        resolve(loader="changed_history_json").load_tasks()


def test_verifier_runs_baseline_before_strategy_filtering(runtime_case):
    """A duplicate may fail exact filtering only after Host baseline verification."""
    resolve, _bundle, _tasks, calls = runtime_case
    duplicate = {"baseline": True, "document_id": "document-1", "support_ids": ("support-1", "support-1")}

    assert resolve(verifier_strategy="exact_support").verify(duplicate) is False
    assert resolve(verifier_strategy="deduplicate_support").verify(duplicate) is True
    assert resolve().verify({"baseline": True, "document_id": "fabricated", "support_id": "support-1"}) is False
    assert resolve().verify({"baseline": True, "evaluation": "not evidence"}) is False
    assert len(calls) == 4


def test_loader_rejects_a_raw_fixture_with_the_wrong_task_identity(runtime_case):
    """An alias can rename a history key, never a committed task identity."""
    resolve, _bundle, tasks, _calls = runtime_case
    runtime = resolve()
    task = tasks[0]
    bad_host = replace(
        runtime.host_inputs,
        raw_fixture_records={
            **runtime.host_inputs.raw_fixture_records,
            task.numeric.task_id: replace(task, numeric=replace(task.numeric, task_id="wrong-id")),
        },
    )
    bad_runtime = ProtocolRuntimeRegistry().resolve(runtime.protocol, bad_host)

    with pytest.raises(ValueError, match="raw fixture task ID mismatch"):
        bad_runtime.load_tasks()
