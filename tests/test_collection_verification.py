from __future__ import annotations

from dataclasses import replace

from numerical_agent.collection.contracts import MethodCard, SourceRecord
from numerical_agent.collection.verification import verify_registry


def source(
    source_id: str = "source_primary",
    *,
    source_type: str = "paper",
    primary: bool = True,
    review_status: str = "verified",
) -> SourceRecord:
    return SourceRecord.from_payload(
        {
            "source_id": source_id,
            "title": "Forecasting source",
            "authors": ["A. Author"],
            "year": 2024,
            "source_type": source_type,
            "url": f"https://example.org/{source_id}",
            "retrieved_at": "2026-08-17",
            "primary": primary,
            "review_status": review_status,
        }
    )


def method(
    method_uid: str = "method_base",
    *,
    family: str = "statistical",
    source_ids: tuple[str, ...] = ("source_primary",),
    verification_status: str = "verified",
    parent_method_uids: tuple[str, ...] = (),
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
    return MethodCard.from_payload(
        {
            "method_uid": method_uid,
            "definition_version": 1,
            "canonical_name": method_uid.replace("_", " "),
            "aliases": [],
            "family": family,
            "category": "ensemble" if family == "combined" else "baseline",
            "description": "A complete method definition.",
            "assumptions": ["Its fitted historical relationship persists."],
            "failure_conditions": ["A structural break invalidates the relationship."],
            "applicability": {
                "minimum_history": 2,
                "frequencies": ["any"],
                "supports_univariate": True,
                "supports_covariates": False,
                "supports_probabilistic_output": False,
            },
            "hyperparameters": [],
            "definition_source_ids": list(source_ids),
            "implementation_source_ids": [],
            "implementation_availability": "unknown",
            "verification_status": verification_status,
            "lineage": {
                "operation": "collected",
                "parent_method_uids": list(parent_method_uids),
            },
            "foundation_metadata": foundation_metadata,
        }
    )


def test_valid_registry_is_publishable() -> None:
    report = verify_registry((source(),), (method(),))

    assert report.is_publishable is True
    assert report.issue_codes == ()


def test_verified_method_requires_authoritative_primary_definition_source() -> None:
    survey = source(
        "source_survey", source_type="survey", primary=False, review_status="verified"
    )

    report = verify_registry(
        (survey,), (method(source_ids=(survey.source_id,)),)
    )

    assert report.is_publishable is False
    assert "missing_authoritative_definition" in report.issue_codes


def test_unknown_or_unverified_referenced_sources_block_publication() -> None:
    unknown = verify_registry((source(),), (method(source_ids=("missing",)),))
    candidate_source = source("source_candidate", review_status="candidate")
    candidate = verify_registry(
        (candidate_source,),
        (method(source_ids=(candidate_source.source_id,)),),
    )

    assert "unknown_definition_source" in unknown.issue_codes
    assert "unverified_definition_source" in candidate.issue_codes


def test_unverified_method_blocks_publication_even_with_valid_source() -> None:
    unverified = replace(
        method(), verification_status="unverified", definition_source_ids=()
    )

    report = verify_registry((source(),), (unverified,))

    assert "method_not_verified" in report.issue_codes
    assert report.is_publishable is False


def test_combined_method_requires_two_known_nonself_parents() -> None:
    base = method("method_base")
    other = method("method_other")
    too_few = method(
        "method_combined", family="combined", parent_method_uids=(base.method_uid,)
    )
    self_parent = method(
        "method_self",
        family="combined",
        parent_method_uids=("method_self", base.method_uid),
    )

    too_few_report = verify_registry((source(),), (base, other, too_few))
    self_report = verify_registry((source(),), (base, self_parent))

    assert "combined_requires_two_parents" in too_few_report.issue_codes
    assert "self_parent" in self_report.issue_codes


def test_unknown_and_cyclic_lineage_block_publication() -> None:
    base_a = method("method_a")
    base_b = method("method_b")
    unknown_parent = method(
        "method_unknown_combined",
        family="combined",
        parent_method_uids=(base_a.method_uid, "method_missing"),
    )
    left = method(
        "method_left",
        family="combined",
        parent_method_uids=("method_right", base_a.method_uid),
    )
    right = method(
        "method_right",
        family="combined",
        parent_method_uids=("method_left", base_b.method_uid),
    )

    unknown_report = verify_registry((source(),), (base_a, unknown_parent))
    cyclic_report = verify_registry((source(),), (base_a, base_b, left, right))

    assert "unknown_parent_method" in unknown_report.issue_codes
    assert "cyclic_lineage" in cyclic_report.issue_codes
