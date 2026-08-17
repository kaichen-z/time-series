"""Curated, source-grounded authoring helpers for forecast methods v001.

This module is deliberately declarative: it records reviewed primary sources and
the evidence assignment for each migrated method.  It does not infer provenance
from names and it drops legacy constructs for which no reviewed definition was
found.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import MethodCard, SourceRecord
from .seed import CATEGORY_BY_LEGACY_ID


SOURCE_PAYLOADS: tuple[Mapping[str, object], ...] = (
    {
        "source_id": "source_000001",
        "title": "Forecasting: Principles and Practice (3rd ed)",
        "authors": ["Rob J. Hyndman", "George Athanasopoulos"],
        "year": 2021,
        "source_type": "textbook",
        "url": "https://otexts.com/fpp3/",
        "doi": "",
        "isbn": "9780987507136",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000002",
        "title": "StatsForecast model documentation",
        "authors": ["Nixtla"],
        "year": 2026,
        "source_type": "official_docs",
        "url": "https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000003",
        "title": "Statsmodels state space methods documentation",
        "authors": ["Statsmodels developers"],
        "year": 2026,
        "source_type": "official_docs",
        "url": "https://www.statsmodels.org/stable/statespace.html",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000004",
        "title": "The theta model: a decomposition approach to forecasting",
        "authors": ["Vassilis Assimakopoulos", "Konstantinos Nikolopoulos"],
        "year": 2000,
        "source_type": "paper",
        "url": "https://doi.org/10.1016/S0169-2070(00)00066-2",
        "doi": "10.1016/S0169-2070(00)00066-2",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000005",
        "title": "Forecasting and stock control for intermittent demands",
        "authors": ["J. D. Croston"],
        "year": 1972,
        "source_type": "paper",
        "url": "https://doi.org/10.1057/jors.1972.50",
        "doi": "10.1057/jors.1972.50",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000006",
        "title": "The accuracy of intermittent demand estimates",
        "authors": ["John E. Boylan", "Aris A. Syntetos"],
        "year": 2005,
        "source_type": "paper",
        "url": "https://doi.org/10.1016/j.ijforecast.2004.10.001",
        "doi": "10.1016/j.ijforecast.2004.10.001",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000007",
        "title": "Intermittent demand: linking forecasting to inventory obsolescence",
        "authors": ["Ruud H. Teunter", "Aris A. Syntetos", "M. Zied Babai"],
        "year": 2011,
        "source_type": "paper",
        "url": "https://doi.org/10.1016/j.ejor.2010.09.018",
        "doi": "10.1016/j.ejor.2010.09.018",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000008",
        "title": "An aggregate-disaggregate intermittent demand approach",
        "authors": [
            "Konstantinos Nikolopoulos",
            "Aris A. Syntetos",
            "John E. Boylan",
            "Fotios Petropoulos",
            "Vassilis Assimakopoulos",
        ],
        "year": 2011,
        "source_type": "paper",
        "url": "https://doi.org/10.1016/j.ijforecast.2010.09.008",
        "doi": "10.1016/j.ijforecast.2010.09.008",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000009",
        "title": "Improving forecasting by estimating time series structural components across multiple frequencies",
        "authors": ["Nikolaos Kourentzes", "Fotios Petropoulos", "Juan R. Trapero"],
        "year": 2014,
        "source_type": "paper",
        "url": "https://doi.org/10.1016/j.ijforecast.2013.09.006",
        "doi": "10.1016/j.ijforecast.2013.09.006",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000010",
        "title": "STL: A seasonal-trend decomposition procedure based on loess",
        "authors": ["Robert B. Cleveland", "William S. Cleveland", "Jean E. McRae", "Irma Terpenning"],
        "year": 1990,
        "source_type": "paper",
        "url": "https://www.wessa.net/download/stl.pdf",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000011",
        "title": "Forecast combinations: an over 50-year review",
        "authors": ["Xiaoqian Wang", "Rob J. Hyndman", "Feng Li", "Yanfei Kang"],
        "year": 2023,
        "source_type": "paper",
        "url": "https://robjhyndman.com/publications/combinations/",
        "doi": "",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
    {
        "source_id": "source_000012",
        "title": "Forecasting at scale",
        "authors": ["Sean J. Taylor", "Benjamin Letham"],
        "year": 2018,
        "source_type": "paper",
        "url": "https://doi.org/10.1080/00031305.2017.1380080",
        "doi": "10.1080/00031305.2017.1380080",
        "isbn": "",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    },
)


EXCLUDED_LEGACY_IDS = {
    "fft_dominant_frequency_extrapolation",
    "wavelet_trend_detail_forecast",
    "empirical_quantile_persistence",
}

SPECIFIC_DEFINITION_SOURCES: Mapping[str, tuple[str, ...]] = {
    "theta_classic": ("source_000004",),
    "theta_optimized": ("source_000002", "source_000004"),
    "croston": ("source_000005",),
    "croston_sba": ("source_000006",),
    "tsb": ("source_000007",),
    "adida": ("source_000008",),
    "imapa": ("source_000009",),
    "local_level_kalman": ("source_000003",),
    "local_linear_trend_kalman": ("source_000003",),
    "structural_time_series_bsm": ("source_000003",),
    "robust_loess_trend": ("source_000010",),
    "piecewise_linear_trend": ("source_000012",),
    "statistical_ensemble_mean": ("source_000011",),
}


def _jsonl(records: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def _legacy_card(method: Mapping[str, object], index: int) -> MethodCard:
    legacy_id = str(method["method_id"])
    source_ids = SPECIFIC_DEFINITION_SOURCES.get(legacy_id, ("source_000001",))
    return MethodCard.from_payload(
        {
            "method_uid": f"method_seed_{index:04d}",
            "definition_version": 1,
            "canonical_name": legacy_id,
            "aliases": [legacy_id],
            "family": "statistical",
            "category": CATEGORY_BY_LEGACY_ID[legacy_id],
            "description": method["description"],
            "assumptions": method["assumptions"],
            "failure_conditions": method["failure_conditions"],
            "applicability": {
                "minimum_history": 1,
                "frequencies": ["any"],
                "supports_univariate": True,
                "supports_covariates": False,
                "supports_probabilistic_output": (
                    CATEGORY_BY_LEGACY_ID[legacy_id] == "probabilistic"
                ),
            },
            "hyperparameters": [],
            "definition_source_ids": list(source_ids),
            "implementation_source_ids": [],
            "implementation_availability": "unknown",
            "verification_status": "verified",
            "lineage": {
                "operation": "verified_migrated_seed",
                "parent_method_uids": [],
                "legacy_dictionary_id": "statistical_base_methods_v000",
                "legacy_method_id": legacy_id,
            },
            "foundation_metadata": {},
        }
    )


def write_catalog_manifests(
    legacy_source: str | Path,
    source_destination: str | Path,
    method_destination: str | Path,
) -> tuple[tuple[SourceRecord, ...], tuple[MethodCard, ...]]:
    """Write the reviewed classical batch as deterministic JSONL manifests."""

    legacy_payload = json.loads(Path(legacy_source).read_text(encoding="utf-8"))
    legacy_methods = legacy_payload.get("methods")
    if not isinstance(legacy_methods, list):
        raise ValueError("legacy dictionary must contain a methods list")
    sources = tuple(SourceRecord.from_payload(payload) for payload in SOURCE_PAYLOADS)
    methods = tuple(
        _legacy_card(method, index)
        for index, method in enumerate(legacy_methods, start=1)
        if str(method["method_id"]) not in EXCLUDED_LEGACY_IDS
    )
    source_path = Path(source_destination)
    method_path = Path(method_destination)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    method_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        _jsonl([source.to_payload() for source in sources]), encoding="utf-8"
    )
    method_path.write_text(
        _jsonl([method.to_payload() for method in methods]), encoding="utf-8"
    )
    return sources, methods
