from __future__ import annotations

import json

import pytest

from common.evolution_core.acceptance import MetricAcceptanceGate
from common.evolution_core.contracts import (
    METRIC_POLICY_FINGERPRINT,
    EvaluationReport,
    MetricSpec,
    metric_policy_metadata,
)
from common.evolution_core.persistence import JsonArtifactStore


def report(score: float) -> EvaluationReport:
    return EvaluationReport(
        artifact_id="v",
        split="dev",
        metrics={"smape": score},
        item_count=2,
        diagnostics={},
    )


def test_acceptance_requires_strict_improvement() -> None:
    gate = MetricAcceptanceGate(MetricSpec("smape", "minimize"), margin=0.0)

    assert gate.accept(report(10.0), report(9.9))
    assert not gate.accept(report(10.0), report(10.0))
    assert not gate.accept(report(10.0), report(10.1))


def test_acceptance_rejects_missing_metric() -> None:
    gate = MetricAcceptanceGate(MetricSpec("smape", "minimize"))
    missing = EvaluationReport("v", "dev", {"mae": 1.0}, 2, {})

    with pytest.raises(ValueError, match="smape"):
        gate.accept(report(10.0), missing)


def test_json_store_round_trips_checkpoint_and_artifact(tmp_path) -> None:
    store = JsonArtifactStore(tmp_path)

    artifact_path = store.save_artifact("parent", {"id": "v000", "quality": 1})
    checkpoint_path = store.save_checkpoint(
        {"generation": 2, "accepted_artifact": {"id": "v002"}}
    )

    assert json.loads(artifact_path.read_text())["id"] == "v000"
    assert checkpoint_path.name == "checkpoint.json"
    assert store.load_checkpoint() == {
        "schema_version": 2,
        **metric_policy_metadata(),
        "generation": 2,
        "accepted_artifact": {"id": "v002"},
    }


@pytest.mark.parametrize(
    "payload",
    (
        {"generation": 1, "accepted_artifact": {"id": "v001"}},
        {
            "schema_version": 2,
            **metric_policy_metadata(),
            "metric_policy_fingerprint": "forged",
            "generation": 1,
            "accepted_artifact": {"id": "v001"},
        },
    ),
)
def test_json_store_rejects_missing_or_wrong_checkpoint_policy(tmp_path, payload) -> None:
    (tmp_path / "checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metric policy|fingerprint"):
        JsonArtifactStore(tmp_path).load_checkpoint()


def test_json_store_checkpoint_persists_exact_canonical_fingerprint(tmp_path) -> None:
    path = JsonArtifactStore(tmp_path).save_checkpoint(
        {"generation": 3, "accepted_artifact": {"id": "v003"}}
    )

    payload = json.loads(path.read_text())
    assert payload["metric_policy_fingerprint"] == METRIC_POLICY_FINGERPRINT


def test_json_store_appends_one_trace_object_per_line(tmp_path) -> None:
    store = JsonArtifactStore(tmp_path)

    store.append_trace({"generation": 0, "accepted": False})
    store.append_trace({"generation": 1, "accepted": True})

    trace_path = tmp_path / "evolution_trace.jsonl"
    assert [json.loads(line) for line in trace_path.read_text().splitlines()] == [
        {"generation": 0, "accepted": False},
        {"generation": 1, "accepted": True},
    ]
