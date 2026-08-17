from __future__ import annotations

import json
from pathlib import Path

from numerical_agent.collection.contracts import MethodCard
from numerical_agent.collection.coverage import audit_coverage, audit_saturation


QUERY_MANIFEST = (
    Path(__file__).parent.parent
    / "numerical_agent"
    / "datasets"
    / "collection_queries_v001.json"
)


def query_manifest() -> dict[str, object]:
    return json.loads(QUERY_MANIFEST.read_text(encoding="utf-8"))


def method(
    method_uid: str,
    family: str,
    category: str,
    *,
    verification_status: str = "verified",
    implementation_availability: str = "available",
) -> MethodCard:
    foundation_metadata = {}
    if family == "foundation":
        foundation_metadata = {
            "checkpoint_or_api": "org/model",
            "release_version": "1.0",
            "context_length": 512,
            "prediction_length": 64,
            "inference_mode": "zero_shot",
            "probabilistic_output": True,
            "covariate_support": False,
            "device_requirements": "cpu_or_gpu",
            "license": "Apache-2.0",
            "weights_available": True,
            "code_available": True,
        }
    parents = ["parent_a", "parent_b"] if family == "combined" else []
    return MethodCard.from_payload(
        {
            "method_uid": method_uid,
            "definition_version": 1,
            "canonical_name": method_uid,
            "aliases": [],
            "family": family,
            "category": category,
            "description": "A coverage-test forecasting method.",
            "assumptions": ["Its historical relationship persists."],
            "failure_conditions": ["A structural break invalidates it."],
            "applicability": {"minimum_history": 2},
            "hyperparameters": [],
            "definition_source_ids": ["source_primary"],
            "implementation_source_ids": [],
            "implementation_availability": implementation_availability,
            "verification_status": verification_status,
            "lineage": {"operation": "collected", "parent_method_uids": parents},
            "foundation_metadata": foundation_metadata,
        }
    )


def test_coverage_reports_empty_required_taxonomy_cells() -> None:
    report = audit_coverage(
        (method("naive", "statistical", "baseline"),), query_manifest()
    )

    assert report.all_required_cells_covered is False
    assert "foundation.zero_shot" in report.empty_cells
    assert "statistical.baseline" not in report.empty_cells


def test_coverage_counts_families_statuses_and_unknown_cells() -> None:
    methods = (
        method("naive", "statistical", "baseline"),
        method("model", "foundation", "zero_shot"),
        method(
            "experimental",
            "statistical",
            "new_category",
            verification_status="unverified",
            implementation_availability="unknown",
        ),
    )

    report = audit_coverage(methods, query_manifest())

    assert report.total_method_count == 3
    assert report.family_counts == {"foundation": 1, "statistical": 2}
    assert report.verification_counts == {"unverified": 1, "verified": 2}
    assert report.implementation_counts == {"available": 2, "unknown": 1}
    assert report.unknown_cells == ("statistical.new_category",)


def test_coverage_has_no_method_count_cap() -> None:
    methods = tuple(
        method(f"naive_{index:03d}", "statistical", "baseline")
        for index in range(350)
    )

    report = audit_coverage(methods, query_manifest())

    assert report.total_method_count == 350
    assert report.family_counts["statistical"] == 350


def test_saturation_requires_three_consecutive_batches_below_two_percent() -> None:
    assert audit_saturation((4, 2, 1), base_count=120).saturated is False
    saturated = audit_saturation((2, 1, 1), base_count=120)

    assert saturated.saturated is True
    assert saturated.threshold == 2.4
    assert saturated.consecutive_low_yield_batches == 3


def test_saturation_uses_strictly_below_threshold() -> None:
    report = audit_saturation((2, 2, 2), base_count=100)

    assert report.saturated is False
    assert report.consecutive_low_yield_batches == 0
