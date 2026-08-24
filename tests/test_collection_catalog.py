"""Tests for the curated method catalog: build_release output, the legacy seed-method migration, and the catalog adapter."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from numerical_agent.collection.catalog_v001 import write_catalog_manifests
from numerical_agent.collection.coverage import audit_coverage
from numerical_agent.collection.registry import load_method_cards, load_source_records
from numerical_agent.collection.verification import verify_registry
from numerical_agent.collection.seed import migrate_legacy_statistical_seed
from numerical_agent.collection.catalog_adapter import tool_dictionary_from_payload


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

    assert len(combined) >= 23
    assert {method.category for method in combined} == {
        "ensemble",
        "selector",
        "residual_correction",
        "fallback",
    }
    assert all(len(set(method.lineage["parent_method_uids"])) >= 2 for method in combined)
    assert {
        "serial_dependence_corrected_combination",
        "fformpp_performance_selector",
        "zoocast_model_zoo_selector",
        "adapts_online_forecaster_weighter",
        "adapts_multivariate_adapter",
        "seqfusion_sequential_ptm_fusion",
        "llm_zero_cost_model_selector",
    } <= {method.canonical_name for method in combined}
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
    assert counts[("statistical", "calibration")] >= 3
    assert counts[("statistical", "neural")] >= 23
    assert counts[("statistical", "probabilistic")] >= 3
    assert counts[("statistical", "reconciliation")] >= 8
    assert counts[("statistical", "robust")] >= 3
    assert counts[("statistical", "spectral")] >= 2
    assert sum(method.family == "foundation" for method in methods) >= 31
    assert {
        "Chronos-2",
        "Chronos-Bolt",
        "Moirai 2.0",
        "Moirai-MoE",
        "FlowState",
        "Xihe",
        "Kairos",
        "TimeFound",
        "Reverso",
        "Falcon-X",
        "SEMPO",
        "TiRex",
        "TiRex-2",
        "TabPFN-TS",
        "PatchTST-FM",
        "TimesFM 2.5",
    } <= {
        method.canonical_name for method in methods
    }
    assert {
        "arch_volatility_forecast",
        "garch_volatility_forecast",
        "vector_autoregression",
        "vector_error_correction_model",
        "varmax_state_space",
        "dynamic_factor_forecast",
        "auto_ces",
        "auto_mfles",
        "dynamic_theta",
        "film_legendre_memory",
        "auto_tbats",
        "conformal_seasonal_pool",
        "crossformer",
        "micn_multiscale_convolution",
        "nonstationary_transformer",
        "pyraformer",
        "lightts_sampling_mlp",
    } <= {method.canonical_name for method in methods}

ROOT = Path(__file__).resolve().parents[1]


LEGACY = ROOT / "numerical_agent/dictionaries/statistical_base_methods_v000.json"


def test_legacy_seed_migration_is_deterministic_and_does_not_invent_sources(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "method_candidates_v001.jsonl"

    first = migrate_legacy_statistical_seed(LEGACY, destination)
    first_bytes = destination.read_bytes()
    second = migrate_legacy_statistical_seed(LEGACY, destination)

    assert first == second
    assert destination.read_bytes() == first_bytes
    assert len(first) == 41
    assert [card.method_uid for card in first] == [
        f"method_seed_{index:04d}" for index in range(1, 42)
    ]
    assert all(card.verification_status == "unverified" for card in first)
    assert all(not card.definition_source_ids for card in first)
    assert all(not card.implementation_source_ids for card in first)


def test_legacy_seed_migration_preserves_behavior_and_legacy_identity(
    tmp_path: Path,
) -> None:
    legacy_payload = json.loads(LEGACY.read_text(encoding="utf-8"))
    legacy_methods = legacy_payload["methods"]
    destination = tmp_path / "method_candidates_v001.jsonl"

    migrated = migrate_legacy_statistical_seed(LEGACY, destination)

    assert [card.canonical_name for card in migrated] == [
        method["method_id"] for method in legacy_methods
    ]
    assert [card.aliases for card in migrated] == [
        (method["method_id"],) for method in legacy_methods
    ]
    assert [list(card.assumptions) for card in migrated] == [
        method["assumptions"] for method in legacy_methods
    ]
    assert [list(card.failure_conditions) for card in migrated] == [
        method["failure_conditions"] for method in legacy_methods
    ]
    assert load_method_cards(destination) == migrated


def test_checked_in_catalog_manifests_are_parseable_after_curation() -> None:
    cards = load_method_cards(
        ROOT / "numerical_agent/datasets/method_candidates_v001.jsonl"
    )
    sources = load_source_records(
        ROOT / "numerical_agent/datasets/source_registry_v001.jsonl"
    )

    assert len(cards) >= 38
    assert len(sources) >= 10
    assert {card.verification_status for card in cards} == {"verified"}

ROOT = Path(__file__).resolve().parents[1]


def test_release_catalog_imports_as_an_executable_statistical_dictionary() -> None:
    payload = json.loads(
        (
            ROOT / "numerical_agent/datasets/forecast_method_dataset_v001.json"
        ).read_text()
    )

    dictionary = tool_dictionary_from_payload(
        payload, allowed_families=("statistical",)
    )

    assert dictionary.dictionary_id == "forecast_method_dataset_v001.statistical.v000"
    assert dictionary.methods
    assert all(
        record.definition.family == "statistical" for record in dictionary.methods
    )
    assert dictionary.methods[0].definition.method_id.startswith("method_")
    assert all(record.candidate is None for record in dictionary.methods)


def test_existing_tool_dictionary_payload_remains_supported() -> None:
    payload = {
        "dictionary_id": "existing",
        "parent_dictionary_id": None,
        "generation": 0,
        "methods": [
            {
                "method_id": "m1",
                "family": "statistical",
                "description": "existing method",
            }
        ],
    }

    dictionary = tool_dictionary_from_payload(
        payload, allowed_families=("statistical",)
    )

    assert dictionary.dictionary_id == "existing"
    assert dictionary.methods[0].definition.method_id == "m1"


def test_release_import_rejects_combined_methods_without_their_parent_families() -> (
    None
):
    payload = json.loads(
        (
            ROOT / "numerical_agent/datasets/forecast_method_dataset_v001.json"
        ).read_text()
    )

    with pytest.raises(ValueError, match="requires excluded parent methods"):
        tool_dictionary_from_payload(payload, allowed_families=("combined",))
