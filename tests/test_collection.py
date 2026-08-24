"""Tests for numerical_agent/collection: method-card contracts, coverage/saturation auditing, duplicate normalization, release registry, and provenance verification."""
from __future__ import annotations

import pytest
import json
import hashlib
from numerical_agent.collection.contracts import DatasetRelease, MethodCard, SourceRecord
from pathlib import Path
from numerical_agent.collection.coverage import audit_coverage, audit_saturation
from dataclasses import replace
from numerical_agent.collection.normalization import find_duplicate_candidates, normalize_name
from numerical_agent.collection.registry import (
    build_release,
    load_method_cards,
    load_source_records,
    write_release,
)
from numerical_agent.collection.verification import verify_registry


def source_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_id": "source_000001",
        "title": "A primary forecasting source",
        "authors": ["A. Author"],
        "year": 2024,
        "source_type": "paper",
        "url": "https://example.org/paper",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    }
    payload.update(overrides)
    return payload


def method_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "method_uid": "method_000001",
        "definition_version": 1,
        "canonical_name": "Damped trend",
        "aliases": ["Damped Holt trend"],
        "family": "statistical",
        "category": "exponential_smoothing",
        "description": "Extrapolate a trend whose contribution decays with horizon.",
        "assumptions": ["The history contains a trend."],
        "failure_conditions": ["An untreated structural break changes the level."],
        "applicability": {
            "minimum_history": 20,
            "frequencies": ["any"],
            "supports_univariate": True,
            "supports_covariates": False,
            "supports_probabilistic_output": False,
        },
        "hyperparameters": ["damping_factor"],
        "definition_source_ids": ["source_000001"],
        "implementation_source_ids": [],
        "implementation_availability": "unknown",
        "verification_status": "verified",
        "lineage": {"operation": "collected", "parent_method_uids": []},
        "foundation_metadata": {},
    }
    payload.update(overrides)
    return payload


def release_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": "forecast_method_dataset_v001",
        "release_date": "2026-08-17",
        "collection_cutoff": "2026-08-17",
        "sources": [source_payload()],
        "methods": [method_payload()],
        "taxonomy": {"statistical": ["exponential_smoothing"]},
        "collection_batches": [],
        "content_hash": "",
    }
    payload.update(overrides)
    return payload


def test_source_record_accepts_https_or_doi_as_authoritative_locator() -> None:
    https_source = SourceRecord.from_payload(source_payload())
    doi_source = SourceRecord.from_payload(
        source_payload(url="", doi="10.1000/example", source_id="source_000002")
    )

    assert https_source.url == "https://example.org/paper"
    assert doi_source.doi == "10.1000/example"


def test_source_record_rejects_missing_authoritative_locator() -> None:
    with pytest.raises(ValueError, match="authoritative locator"):
        SourceRecord.from_payload(source_payload(url="", doi="", isbn=""))


def test_source_record_rejects_invalid_date_and_future_year() -> None:
    with pytest.raises(ValueError, match="retrieved_at"):
        SourceRecord.from_payload(source_payload(retrieved_at="17 August 2026"))
    with pytest.raises(ValueError, match="year"):
        SourceRecord.from_payload(source_payload(year=2027))


def test_method_card_round_trip_preserves_complete_behavior_contract() -> None:
    original = MethodCard.from_payload(method_payload())

    restored = MethodCard.from_payload(original.to_payload())

    assert restored == original
    assert restored.assumptions == ("The history contains a trend.",)
    assert restored.failure_conditions == (
        "An untreated structural break changes the level.",
    )


def test_method_card_rejects_incomplete_verified_behavior() -> None:
    with pytest.raises(ValueError, match="assumptions"):
        MethodCard.from_payload(method_payload(assumptions=[]))
    with pytest.raises(ValueError, match="failure_conditions"):
        MethodCard.from_payload(method_payload(failure_conditions=[]))
    with pytest.raises(ValueError, match="definition_source_ids"):
        MethodCard.from_payload(method_payload(definition_source_ids=[]))


def test_foundation_method_requires_complete_foundation_metadata() -> None:
    with pytest.raises(ValueError, match="foundation_metadata"):
        MethodCard.from_payload(
            method_payload(
                family="foundation",
                category="zero_shot",
                foundation_metadata={},
            )
        )

    card = MethodCard.from_payload(
        method_payload(
            family="foundation",
            category="zero_shot",
            foundation_metadata={
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
            },
        )
    )
    assert card.foundation_metadata["checkpoint_or_api"] == "org/model"


def test_dataset_release_rejects_duplicate_source_and_method_ids() -> None:
    duplicate_source = source_payload()
    with pytest.raises(ValueError, match="duplicate source_id"):
        DatasetRelease.from_payload(
            release_payload(sources=[source_payload(), duplicate_source])
        )

    duplicate_method = method_payload()
    with pytest.raises(ValueError, match="duplicate method_uid"):
        DatasetRelease.from_payload(
            release_payload(methods=[method_payload(), duplicate_method])
        )


def test_dataset_release_round_trip_is_lossless() -> None:
    original = DatasetRelease.from_payload(release_payload())

    assert DatasetRelease.from_payload(original.to_payload()) == original

QUERY_MANIFEST = (
    Path(__file__).parent.parent
    / "numerical_agent"
    / "datasets"
    / "collection_queries_v001.json"
)


def query_manifest() -> dict[str, object]:
    return json.loads(QUERY_MANIFEST.read_text(encoding="utf-8"))


def coverage_method(
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
        (coverage_method("naive", "statistical", "baseline"),), query_manifest()
    )

    assert report.all_required_cells_covered is False
    assert "foundation.zero_shot" in report.empty_cells
    assert "statistical.baseline" not in report.empty_cells


def test_coverage_counts_families_statuses_and_unknown_cells() -> None:
    methods = (
        coverage_method("naive", "statistical", "baseline"),
        coverage_method("model", "foundation", "zero_shot"),
        coverage_method(
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
        coverage_method(f"naive_{index:03d}", "statistical", "baseline")
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

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "method_collection"
    / "duplicate_methods.jsonl"
)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Auto-ARIMA", "auto arima"),
        ("Damped Holt's trend", "damped holts trend"),
        ("  Seasonal   Naïve  ", "seasonal naïve"),
    ],
)
def test_normalize_name_matches_punctuation_case_and_spacing(
    left: str, right: str
) -> None:
    assert normalize_name(left) == normalize_name(right)


def test_alias_collision_is_reported_without_automatic_merge() -> None:
    methods = load_method_cards(FIXTURE)

    candidates = find_duplicate_candidates(methods)

    damped = next(
        candidate
        for candidate in candidates
        if {candidate.left_method_uid, candidate.right_method_uid}
        == {"method_damped_alias", "method_damped_canonical"}
    )
    assert "alias_collision" in damped.reasons
    assert damped.requires_manual_review is True
    assert {method.method_uid for method in methods} >= {
        "method_damped_alias",
        "method_damped_canonical",
    }


def test_wrapper_and_underlying_method_are_only_flagged_for_manual_review() -> None:
    methods = load_method_cards(FIXTURE)

    candidates = find_duplicate_candidates(methods)

    wrapper_pair = next(
        candidate
        for candidate in candidates
        if {candidate.left_method_uid, candidate.right_method_uid}
        == {"method_auto_arima", "method_arima"}
    )
    assert wrapper_pair.reasons == ("shared_source_claim",)
    assert wrapper_pair.requires_manual_review is True


def test_shared_textbook_does_not_flag_distinct_methods_with_one_common_token() -> None:
    methods = load_method_cards(FIXTURE)
    first = replace(
        methods[0],
        method_uid="method_naive_last",
        canonical_name="naive last",
        aliases=(),
    )
    second = replace(
        methods[1],
        method_uid="method_naive_mean",
        canonical_name="naive mean",
        aliases=(),
    )

    assert find_duplicate_candidates((first, second)) == ()


def test_forecast_token_inside_name_distinguishes_reconciliation_methods() -> None:
    methods = load_method_cards(FIXTURE)
    historical = replace(
        methods[0],
        method_uid="method_historical_proportions",
        canonical_name="top down historical proportions",
        aliases=(),
        definition_source_ids=("source_shared",),
    )
    forecast = replace(
        methods[1],
        method_uid="method_forecast_proportions",
        canonical_name="top down forecast proportions",
        aliases=(),
        definition_source_ids=("source_shared",),
    )

    assert find_duplicate_candidates((historical, forecast)) == ()


def test_duplicate_report_order_is_deterministic() -> None:
    methods = load_method_cards(FIXTURE)

    forward = find_duplicate_candidates(methods)
    reverse = find_duplicate_candidates(tuple(reversed(methods)))

    assert forward == reverse

FIXTURES = Path(__file__).parent / "fixtures" / "method_collection"


def release_from_fixtures():
    return build_release(
        load_source_records(FIXTURES / "valid_sources.jsonl"),
        load_method_cards(FIXTURES / "valid_methods.jsonl"),
        dataset_id="forecast_method_dataset_v001",
        release_date="2026-08-17",
        collection_cutoff="2026-08-17",
        taxonomy={"statistical": ("baseline", "seasonal")},
        collection_batches=(),
    )


def test_registry_reports_path_and_line_for_invalid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text("{}\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"sources\.jsonl:1"):
        load_source_records(path)

    path.write_text(
        json.dumps(
            {
                "source_id": "source_000001",
                "title": "Source",
                "authors": ["Author"],
                "year": 2024,
                "source_type": "paper",
                "url": "https://example.org/source",
                "retrieved_at": "2026-08-17",
            }
        )
        + "\nnot-json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"sources\.jsonl:2"):
        load_source_records(path)


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    source_line = (FIXTURES / "valid_sources.jsonl").read_text(encoding="utf-8").splitlines()[0]
    source_path = tmp_path / "sources.jsonl"
    source_path.write_text(f"{source_line}\n{source_line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source_id"):
        load_source_records(source_path)

    method_line = (FIXTURES / "valid_methods.jsonl").read_text(encoding="utf-8").splitlines()[0]
    method_path = tmp_path / "methods.jsonl"
    method_path.write_text(f"{method_line}\n{method_line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate method_uid"):
        load_method_cards(method_path)


def test_release_builder_sorts_sources_and_methods() -> None:
    release = release_from_fixtures()

    assert [source.source_id for source in release.sources] == [
        "source_000001",
        "source_000002",
    ]
    assert [method.method_uid for method in release.methods] == [
        "method_000001",
        "method_000002",
    ]


def test_release_writer_is_byte_deterministic_and_hashes_canonical_payload(
    tmp_path: Path,
) -> None:
    release = release_from_fixtures()
    first = tmp_path / "first" / "forecast_method_dataset_v001.json"
    second = tmp_path / "second" / "forecast_method_dataset_v001.json"

    write_release(release, first, first.with_suffix(".sha256"))
    write_release(release, second, second.with_suffix(".sha256"))

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["content_hash"].startswith("sha256:")
    unhashed = dict(payload)
    unhashed["content_hash"] = ""
    canonical = (
        json.dumps(unhashed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert payload["content_hash"] == f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    sidecar = first.with_suffix(".sha256").read_text(encoding="utf-8")
    assert sidecar == f"{hashlib.sha256(first.read_bytes()).hexdigest()}  {first.name}\n"

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


def verification_method(
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
    report = verify_registry((source(),), (verification_method(),))

    assert report.is_publishable is True
    assert report.issue_codes == ()


def test_verified_method_requires_authoritative_primary_definition_source() -> None:
    survey = source(
        "source_survey", source_type="survey", primary=False, review_status="verified"
    )

    report = verify_registry(
        (survey,), (verification_method(source_ids=(survey.source_id,)),)
    )

    assert report.is_publishable is False
    assert "missing_authoritative_definition" in report.issue_codes


def test_unknown_or_unverified_referenced_sources_block_publication() -> None:
    unknown = verify_registry((source(),), (verification_method(source_ids=("missing",)),))
    candidate_source = source("source_candidate", review_status="candidate")
    candidate = verify_registry(
        (candidate_source,),
        (verification_method(source_ids=(candidate_source.source_id,)),),
    )

    assert "unknown_definition_source" in unknown.issue_codes
    assert "unverified_definition_source" in candidate.issue_codes


def test_unverified_method_blocks_publication_even_with_valid_source() -> None:
    unverified = replace(
        verification_method(), verification_status="unverified", definition_source_ids=()
    )

    report = verify_registry((source(),), (unverified,))

    assert "method_not_verified" in report.issue_codes
    assert report.is_publishable is False


def test_combined_method_requires_two_known_nonself_parents() -> None:
    base = verification_method("method_base")
    other = verification_method("method_other")
    too_few = verification_method(
        "method_combined", family="combined", parent_method_uids=(base.method_uid,)
    )
    self_parent = verification_method(
        "method_self",
        family="combined",
        parent_method_uids=("method_self", base.method_uid),
    )

    too_few_report = verify_registry((source(),), (base, other, too_few))
    self_report = verify_registry((source(),), (base, self_parent))

    assert "combined_requires_two_parents" in too_few_report.issue_codes
    assert "self_parent" in self_report.issue_codes


def test_unknown_and_cyclic_lineage_block_publication() -> None:
    base_a = verification_method("method_a")
    base_b = verification_method("method_b")
    unknown_parent = verification_method(
        "method_unknown_combined",
        family="combined",
        parent_method_uids=(base_a.method_uid, "method_missing"),
    )
    left = verification_method(
        "method_left",
        family="combined",
        parent_method_uids=("method_right", base_a.method_uid),
    )
    right = verification_method(
        "method_right",
        family="combined",
        parent_method_uids=("method_left", base_b.method_uid),
    )

    unknown_report = verify_registry((source(),), (base_a, unknown_parent))
    cyclic_report = verify_registry((source(),), (base_a, base_b, left, right))

    assert "unknown_parent_method" in unknown_report.issue_codes
    assert "cyclic_lineage" in cyclic_report.issue_codes
