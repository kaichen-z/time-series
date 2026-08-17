"""Taxonomy coverage and uncapped collection-saturation auditing."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import MethodCard, SourceRecord


@dataclass(frozen=True)
class CoverageReport:
    total_method_count: int
    family_counts: Mapping[str, int]
    category_counts: Mapping[str, int]
    verification_counts: Mapping[str, int]
    implementation_counts: Mapping[str, int]
    source_type_counts: Mapping[str, int]
    empty_cells: tuple[str, ...]
    unknown_cells: tuple[str, ...]

    @property
    def all_required_cells_covered(self) -> bool:
        return not self.empty_cells

    def to_payload(self) -> dict[str, object]:
        return {
            "total_method_count": self.total_method_count,
            "family_counts": dict(self.family_counts),
            "category_counts": dict(self.category_counts),
            "verification_counts": dict(self.verification_counts),
            "implementation_counts": dict(self.implementation_counts),
            "source_type_counts": dict(self.source_type_counts),
            "empty_cells": list(self.empty_cells),
            "unknown_cells": list(self.unknown_cells),
            "all_required_cells_covered": self.all_required_cells_covered,
        }


@dataclass(frozen=True)
class SaturationReport:
    saturated: bool
    threshold: float
    batch_new_method_counts: tuple[int, ...]
    consecutive_low_yield_batches: int
    minimum_consecutive_batches: int = 3

    def to_payload(self) -> dict[str, object]:
        return {
            "saturated": self.saturated,
            "threshold": self.threshold,
            "batch_new_method_counts": list(self.batch_new_method_counts),
            "consecutive_low_yield_batches": self.consecutive_low_yield_batches,
            "minimum_consecutive_batches": self.minimum_consecutive_batches,
        }


def _sorted_counts(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _required_cells(query_manifest: Mapping[str, object]) -> set[str]:
    taxonomy = query_manifest.get("taxonomy")
    if not isinstance(taxonomy, Mapping):
        raise ValueError("query manifest taxonomy must be an object")
    cells: set[str] = set()
    for family, raw_categories in taxonomy.items():
        if not isinstance(raw_categories, Mapping):
            raise ValueError(f"query manifest taxonomy.{family} must be an object")
        for category, raw_terms in raw_categories.items():
            if not isinstance(raw_terms, Sequence) or isinstance(raw_terms, (str, bytes)):
                raise ValueError(
                    f"query manifest taxonomy.{family}.{category} must be a list"
                )
            if not raw_terms or any(not str(term).strip() for term in raw_terms):
                raise ValueError(
                    f"query manifest taxonomy.{family}.{category} needs search terms"
                )
            cells.add(f"{family}.{category}")
    return cells


def audit_coverage(
    methods: Sequence[MethodCard],
    query_manifest: Mapping[str, object],
    sources: Sequence[SourceRecord] = (),
) -> CoverageReport:
    required = _required_cells(query_manifest)
    all_cells = {f"{method.family}.{method.category}" for method in methods}
    verified_cells = {
        f"{method.family}.{method.category}"
        for method in methods
        if method.verification_status == "verified"
    }
    return CoverageReport(
        total_method_count=len(methods),
        family_counts=_sorted_counts([method.family for method in methods]),
        category_counts=_sorted_counts(
            [f"{method.family}.{method.category}" for method in methods]
        ),
        verification_counts=_sorted_counts(
            [method.verification_status for method in methods]
        ),
        implementation_counts=_sorted_counts(
            [method.implementation_availability for method in methods]
        ),
        source_type_counts=_sorted_counts([source.source_type for source in sources]),
        empty_cells=tuple(sorted(required - verified_cells)),
        unknown_cells=tuple(sorted(all_cells - required)),
    )


def audit_saturation(
    batch_new_method_counts: Sequence[int],
    *,
    base_count: int,
    minimum_consecutive_batches: int = 3,
    yield_fraction: float = 0.02,
) -> SaturationReport:
    if base_count <= 0:
        raise ValueError("base_count must be positive")
    if minimum_consecutive_batches <= 0:
        raise ValueError("minimum_consecutive_batches must be positive")
    if yield_fraction <= 0:
        raise ValueError("yield_fraction must be positive")
    counts = tuple(int(value) for value in batch_new_method_counts)
    if any(value < 0 for value in counts):
        raise ValueError("batch new-method counts must be non-negative")
    threshold = base_count * yield_fraction
    consecutive = 0
    for count in reversed(counts):
        if count >= threshold:
            break
        consecutive += 1
    return SaturationReport(
        saturated=consecutive >= minimum_consecutive_batches,
        threshold=threshold,
        batch_new_method_counts=counts,
        consecutive_low_yield_batches=consecutive,
        minimum_consecutive_batches=minimum_consecutive_batches,
    )
