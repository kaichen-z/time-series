from __future__ import annotations

from pathlib import Path

from numerical_agent.collection.catalog_v001 import write_catalog_manifests
from numerical_agent.collection.coverage import audit_coverage
from numerical_agent.collection.registry import load_method_cards, load_source_records
from numerical_agent.collection.verification import verify_registry


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "numerical_agent/dictionaries/statistical_base_methods_v000.json"


def test_classical_catalog_batch_is_source_grounded_and_deterministic(
    tmp_path: Path,
) -> None:
    sources_path = tmp_path / "sources.jsonl"
    methods_path = tmp_path / "methods.jsonl"

    write_catalog_manifests(LEGACY, sources_path, methods_path)
    first = (sources_path.read_bytes(), methods_path.read_bytes())
    write_catalog_manifests(LEGACY, sources_path, methods_path)

    sources = load_source_records(sources_path)
    methods = load_method_cards(methods_path)
    report = verify_registry(sources, methods)

    assert (sources_path.read_bytes(), methods_path.read_bytes()) == first
    assert len(sources) >= 25
    assert len(methods) >= 65
    assert {method.verification_status for method in methods} == {"verified"}
    assert report.is_publishable, report.to_payload()
    assert all(method.definition_source_ids for method in methods)


def test_classical_catalog_covers_every_statistical_taxonomy_cell(
    tmp_path: Path,
) -> None:
    sources_path = tmp_path / "sources.jsonl"
    methods_path = tmp_path / "methods.jsonl"
    write_catalog_manifests(LEGACY, sources_path, methods_path)
    query_manifest = __import__("json").loads(
        (ROOT / "numerical_agent/datasets/collection_queries_v001.json").read_text(
            encoding="utf-8"
        )
    )

    report = audit_coverage(
        load_method_cards(methods_path),
        query_manifest,
        load_source_records(sources_path),
    )

    assert not [
        cell for cell in report.empty_cells if cell.startswith("statistical.")
    ]


def test_catalog_covers_foundation_modes_with_complete_release_metadata(
    tmp_path: Path,
) -> None:
    sources_path = tmp_path / "sources.jsonl"
    methods_path = tmp_path / "methods.jsonl"
    write_catalog_manifests(LEGACY, sources_path, methods_path)
    methods = load_method_cards(methods_path)
    foundation = [method for method in methods if method.family == "foundation"]

    assert len(foundation) >= 14
    assert {method.category for method in foundation} == {
        "zero_shot",
        "fine_tuned",
        "probabilistic_tsfm",
        "covariate_tsfm",
    }
    assert all(len(method.foundation_metadata) == 11 for method in foundation)


def test_catalog_contains_lineage_valid_combined_methods(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.jsonl"
    methods_path = tmp_path / "methods.jsonl"
    write_catalog_manifests(LEGACY, sources_path, methods_path)
    methods = load_method_cards(methods_path)
    combined = [method for method in methods if method.family == "combined"]

    assert len(combined) >= 12
    assert {method.category for method in combined} == {
        "ensemble",
        "selector",
        "residual_correction",
        "fallback",
    }
    assert all(len(set(method.lineage["parent_method_uids"])) >= 2 for method in combined)
    assert verify_registry(load_source_records(sources_path), methods).is_publishable


def test_catalog_excludes_unverified_constructed_seed_variants(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.jsonl"
    methods_path = tmp_path / "methods.jsonl"

    write_catalog_manifests(LEGACY, sources_path, methods_path)
    names = {method.canonical_name for method in load_method_cards(methods_path)}

    assert "fft_dominant_frequency_extrapolation" not in names
    assert "wavelet_trend_detail_forecast" not in names
    assert "empirical_quantile_persistence" not in names


def test_catalog_has_depth_in_previously_sparse_method_families(
    tmp_path: Path,
) -> None:
    sources_path = tmp_path / "sources.jsonl"
    methods_path = tmp_path / "methods.jsonl"
    write_catalog_manifests(LEGACY, sources_path, methods_path)
    methods = load_method_cards(methods_path)

    counts: dict[tuple[str, str], int] = {}
    for method in methods:
        key = (method.family, method.category)
        counts[key] = counts.get(key, 0) + 1

    assert counts[("statistical", "change_point")] >= 3
    assert counts[("statistical", "neural")] >= 16
    assert counts[("statistical", "probabilistic")] >= 3
    assert counts[("statistical", "reconciliation")] >= 8
    assert counts[("statistical", "robust")] >= 3
    assert counts[("statistical", "spectral")] >= 2
    assert sum(method.family == "foundation" for method in methods) >= 15
