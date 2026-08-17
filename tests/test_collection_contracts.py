from __future__ import annotations

import pytest

from numerical_agent.collection.contracts import DatasetRelease, MethodCard, SourceRecord


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
