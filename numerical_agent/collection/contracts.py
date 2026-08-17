"""Strict JSON-compatible contracts for forecast-method collection releases."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Mapping, Sequence, cast

from numerical_agent.config import ALLOWED_FAMILIES


SourceType = Literal[
    "paper",
    "textbook",
    "official_docs",
    "model_card",
    "official_repo",
    "survey",
    "benchmark",
]
ReviewStatus = Literal["candidate", "verified", "rejected"]
VerificationStatus = Literal["unverified", "verified", "rejected"]
ImplementationAvailability = Literal["available", "partial", "unavailable", "unknown"]

SOURCE_TYPES = {
    "paper",
    "textbook",
    "official_docs",
    "model_card",
    "official_repo",
    "survey",
    "benchmark",
}
REVIEW_STATUSES = {"candidate", "verified", "rejected"}
VERIFICATION_STATUSES = {"unverified", "verified", "rejected"}
IMPLEMENTATION_AVAILABILITIES = {"available", "partial", "unavailable", "unknown"}
FOUNDATION_METADATA_FIELDS = {
    "checkpoint_or_api",
    "release_version",
    "context_length",
    "prediction_length",
    "inference_mode",
    "probabilistic_output",
    "covariate_support",
    "device_requirements",
    "license",
    "weights_available",
    "code_available",
}


def _non_empty(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _strings(value: object, field_name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of strings")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{field_name} must not contain empty values")
    if not allow_empty and not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _object(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def _iso_date(value: object, field_name: str) -> str:
    text = _non_empty(value, field_name)
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc
    return text


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    authors: tuple[str, ...]
    year: int
    source_type: SourceType
    url: str
    doi: str = ""
    isbn: str = ""
    retrieved_at: str = "2026-08-17"
    primary: bool = False
    review_status: ReviewStatus = "candidate"

    def __post_init__(self) -> None:
        _non_empty(self.source_id, "source_id")
        _non_empty(self.title, "title")
        if not self.authors:
            raise ValueError("authors must not be empty")
        if self.year < 1600 or self.year > 2026:
            raise ValueError("year must be between 1600 and 2026")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"unsupported source_type: {self.source_type!r}")
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError(f"unsupported review_status: {self.review_status!r}")
        _iso_date(self.retrieved_at, "retrieved_at")
        if self.url and not self.url.startswith("https://"):
            raise ValueError("url must use HTTPS")
        if not self.url and not self.doi.strip() and not self.isbn.strip():
            raise ValueError("source needs an authoritative locator: HTTPS URL, DOI, or ISBN")

    def to_payload(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "source_type": self.source_type,
            "url": self.url,
            "doi": self.doi,
            "isbn": self.isbn,
            "retrieved_at": self.retrieved_at,
            "primary": self.primary,
            "review_status": self.review_status,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SourceRecord":
        return cls(
            source_id=_non_empty(payload.get("source_id", ""), "source_id"),
            title=_non_empty(payload.get("title", ""), "title"),
            authors=_strings(payload.get("authors", []), "authors", allow_empty=False),
            year=int(payload.get("year", 0)),
            source_type=cast(SourceType, payload.get("source_type", "")),
            url=str(payload.get("url", "")).strip(),
            doi=str(payload.get("doi", "")).strip(),
            isbn=str(payload.get("isbn", "")).strip(),
            retrieved_at=_iso_date(payload.get("retrieved_at", ""), "retrieved_at"),
            primary=bool(payload.get("primary", False)),
            review_status=cast(ReviewStatus, payload.get("review_status", "candidate")),
        )


@dataclass(frozen=True)
class MethodCard:
    method_uid: str
    definition_version: int
    canonical_name: str
    aliases: tuple[str, ...]
    family: Literal["statistical", "foundation", "combined"]
    category: str
    description: str
    assumptions: tuple[str, ...]
    failure_conditions: tuple[str, ...]
    applicability: Mapping[str, object]
    hyperparameters: tuple[str, ...]
    definition_source_ids: tuple[str, ...]
    implementation_source_ids: tuple[str, ...]
    implementation_availability: ImplementationAvailability
    verification_status: VerificationStatus
    lineage: Mapping[str, object]
    foundation_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty(self.method_uid, "method_uid")
        if self.definition_version <= 0:
            raise ValueError("definition_version must be positive")
        _non_empty(self.canonical_name, "canonical_name")
        if self.family not in ALLOWED_FAMILIES:
            raise ValueError(f"unsupported method family: {self.family!r}")
        _non_empty(self.category, "category")
        _non_empty(self.description, "description")
        if not self.assumptions:
            raise ValueError("assumptions must not be empty")
        if not self.failure_conditions:
            raise ValueError("failure_conditions must not be empty")
        if not self.applicability:
            raise ValueError("applicability must not be empty")
        if self.implementation_availability not in IMPLEMENTATION_AVAILABILITIES:
            raise ValueError(
                f"unsupported implementation_availability: {self.implementation_availability!r}"
            )
        if self.verification_status not in VERIFICATION_STATUSES:
            raise ValueError(f"unsupported verification_status: {self.verification_status!r}")
        if self.verification_status == "verified" and not self.definition_source_ids:
            raise ValueError("verified methods require definition_source_ids")
        if self.family == "foundation":
            missing = FOUNDATION_METADATA_FIELDS - set(self.foundation_metadata)
            if missing:
                raise ValueError(
                    f"foundation_metadata is missing fields: {sorted(missing)!r}"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "method_uid": self.method_uid,
            "definition_version": self.definition_version,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "family": self.family,
            "category": self.category,
            "description": self.description,
            "assumptions": list(self.assumptions),
            "failure_conditions": list(self.failure_conditions),
            "applicability": dict(self.applicability),
            "hyperparameters": list(self.hyperparameters),
            "definition_source_ids": list(self.definition_source_ids),
            "implementation_source_ids": list(self.implementation_source_ids),
            "implementation_availability": self.implementation_availability,
            "verification_status": self.verification_status,
            "lineage": dict(self.lineage),
            "foundation_metadata": dict(self.foundation_metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MethodCard":
        return cls(
            method_uid=_non_empty(payload.get("method_uid", ""), "method_uid"),
            definition_version=int(payload.get("definition_version", 0)),
            canonical_name=_non_empty(payload.get("canonical_name", ""), "canonical_name"),
            aliases=_strings(payload.get("aliases", []), "aliases"),
            family=cast(
                Literal["statistical", "foundation", "combined"],
                payload.get("family", ""),
            ),
            category=_non_empty(payload.get("category", ""), "category"),
            description=_non_empty(payload.get("description", ""), "description"),
            assumptions=_strings(
                payload.get("assumptions", []), "assumptions", allow_empty=False
            ),
            failure_conditions=_strings(
                payload.get("failure_conditions", []),
                "failure_conditions",
                allow_empty=False,
            ),
            applicability=_object(payload.get("applicability", {}), "applicability"),
            hyperparameters=_strings(payload.get("hyperparameters", []), "hyperparameters"),
            definition_source_ids=_strings(
                payload.get("definition_source_ids", []), "definition_source_ids"
            ),
            implementation_source_ids=_strings(
                payload.get("implementation_source_ids", []), "implementation_source_ids"
            ),
            implementation_availability=cast(
                ImplementationAvailability,
                payload.get("implementation_availability", "unknown"),
            ),
            verification_status=cast(
                VerificationStatus, payload.get("verification_status", "unverified")
            ),
            lineage=_object(payload.get("lineage", {}), "lineage"),
            foundation_metadata=_object(
                payload.get("foundation_metadata", {}), "foundation_metadata"
            ),
        )


@dataclass(frozen=True)
class DatasetRelease:
    schema_version: int
    dataset_id: str
    release_date: str
    collection_cutoff: str
    sources: tuple[SourceRecord, ...]
    methods: tuple[MethodCard, ...]
    taxonomy: Mapping[str, tuple[str, ...]]
    collection_batches: tuple[Mapping[str, object], ...]
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        _non_empty(self.dataset_id, "dataset_id")
        _iso_date(self.release_date, "release_date")
        _iso_date(self.collection_cutoff, "collection_cutoff")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("dataset contains duplicate source_id values")
        method_ids = [method.method_uid for method in self.methods]
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("dataset contains duplicate method_uid values")
        if self.content_hash and not re.fullmatch(r"sha256:[0-9a-f]{64}", self.content_hash):
            raise ValueError("content_hash must be empty or sha256:<64 lowercase hex>")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "release_date": self.release_date,
            "collection_cutoff": self.collection_cutoff,
            "sources": [source.to_payload() for source in self.sources],
            "methods": [method.to_payload() for method in self.methods],
            "taxonomy": {
                key: list(values) for key, values in sorted(self.taxonomy.items())
            },
            "collection_batches": [dict(batch) for batch in self.collection_batches],
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DatasetRelease":
        raw_sources = payload.get("sources", [])
        raw_methods = payload.get("methods", [])
        raw_taxonomy = payload.get("taxonomy", {})
        raw_batches = payload.get("collection_batches", [])
        if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
            raise ValueError("sources must be a list")
        if not isinstance(raw_methods, Sequence) or isinstance(raw_methods, (str, bytes)):
            raise ValueError("methods must be a list")
        if not isinstance(raw_taxonomy, Mapping):
            raise ValueError("taxonomy must be an object")
        if not isinstance(raw_batches, Sequence) or isinstance(raw_batches, (str, bytes)):
            raise ValueError("collection_batches must be a list")
        sources = tuple(
            SourceRecord.from_payload(_object(item, "source")) for item in raw_sources
        )
        methods = tuple(MethodCard.from_payload(_object(item, "method")) for item in raw_methods)
        taxonomy = {
            str(key): _strings(value, f"taxonomy.{key}")
            for key, value in raw_taxonomy.items()
        }
        batches = tuple(_object(item, "collection_batch") for item in raw_batches)
        return cls(
            schema_version=int(payload.get("schema_version", 0)),
            dataset_id=_non_empty(payload.get("dataset_id", ""), "dataset_id"),
            release_date=_iso_date(payload.get("release_date", ""), "release_date"),
            collection_cutoff=_iso_date(
                payload.get("collection_cutoff", ""), "collection_cutoff"
            ),
            sources=sources,
            methods=methods,
            taxonomy=taxonomy,
            collection_batches=batches,
            content_hash=str(payload.get("content_hash", "")),
        )
