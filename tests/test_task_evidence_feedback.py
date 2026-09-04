"""Closed contracts for Retrieval-to-Numerical task evidence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from common.evolution_core.task_feedback import (
    TaskEvidenceCase,
    TaskEvidenceProjection,
    TaskFeedbackError,
    TaskMorphologyProjection,
)


def _morphology() -> TaskMorphologyProjection:
    return TaskMorphologyProjection(
        frequency="daily",
        history="medium",
        horizon="short",
        trend="strong_up",
        periodicity="weak",
        intermittency="dense",
        recent_regime="stable",
    )


def _case(case_id: str = "case_000_deadbeef") -> TaskEvidenceCase:
    return TaskEvidenceCase(
        case_id=case_id,
        morphology=_morphology(),
        assumption_id="trend_ready",
        claim="The recent trend persists.",
        failure_condition="A verified event reverses the trend.",
        stance="falsified",
        target_match="matched",
        window_relation="overlaps",
        magnitude_status="present",
        mechanism="event_shock",
        decision_action="rejected",
        evidence_chain_sha256="c" * 64,
    )


def test_projection_is_canonical_and_contains_no_host_identity() -> None:
    projection = TaskEvidenceProjection(
        source_bundle_sha256="a" * 64,
        request_namespace_sha256="b" * 64,
        cases=(_case("case_002_deadbeef"), _case("case_001_deadbeef")),
    )

    payload = projection.to_payload()

    assert [item["case_id"] for item in payload["cases"]] == [
        "case_001_deadbeef",
        "case_002_deadbeef",
    ]
    assert "task_17" not in json.dumps(payload, sort_keys=True)
    assert len(projection.fingerprint) == 64


def test_model_facing_case_does_not_expose_train_or_dev_partition() -> None:
    assert "partition" not in _case().to_payload()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("claim", "task_17 proves the trend"),
        ("failure_condition", "Read document_9 before choosing"),
        ("claim", "Use the forecast residual"),
        ("failure_condition", "/private/repo/result.json"),
    ),
)
def test_case_rejects_sensitive_identity_or_result_text(field: str, value: str) -> None:
    with pytest.raises(TaskFeedbackError, match="forbidden"):
        replace(_case(), **{field: value})


def test_projection_rejects_duplicate_case_identity() -> None:
    with pytest.raises(TaskFeedbackError, match="duplicate"):
        TaskEvidenceProjection(
            source_bundle_sha256="a" * 64,
            request_namespace_sha256="b" * 64,
            cases=(_case(), _case()),
        )


def test_case_requires_closed_morphology_and_feedback_enums() -> None:
    with pytest.raises(TaskFeedbackError, match="trend"):
        replace(_morphology(), trend="possibly_up")
    with pytest.raises(TaskFeedbackError, match="stance"):
        replace(_case(), stance="probably_supported")
