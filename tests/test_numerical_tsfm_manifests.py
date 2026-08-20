from __future__ import annotations

import json
from pathlib import Path

import pytest

from numerical_agent.collection.registry import load_method_cards, load_source_records
from numerical_agent.tsfm.manifests import ManifestRegistry, TSFMManifest


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_EXECUTABLE = {
    "method_tsfm_0001",
    "method_tsfm_0002",
    "method_tsfm_0003",
    "method_tsfm_0004",
    "method_tsfm_0006",
    "method_tsfm_0007",
    "method_tsfm_0008",
    "method_tsfm_0011",
    "method_tsfm_0013",
    "method_tsfm_0014",
    "method_tsfm_0015",
    "method_tsfm_0016",
    "method_tsfm_0017",
    "method_tsfm_0018",
    "method_tsfm_0019",
    "method_tsfm_0020",
    "method_tsfm_0022",
    "method_tsfm_0027",
    "method_tsfm_0029",
    "method_tsfm_0030",
    "method_tsfm_0031",
}

EXPECTED_DIRECT = {
    "method_tsfm_0002",
    "method_tsfm_0016",
    "method_tsfm_0018",
    "method_tsfm_0031",
}

EXPECTED_REASONS = {
    "method_tsfm_0005": "forecast_head_requires_training",
    "method_tsfm_0009": "official_checkpoint_missing",
    "method_tsfm_0010": "no_public_local_weights",
    "method_tsfm_0012": "no_generic_zero_shot_api",
    "method_tsfm_0021": "no_public_local_weights",
    "method_tsfm_0023": "no_public_local_weights",
    "method_tsfm_0024": "no_public_local_weights",
    "method_tsfm_0025": "no_public_local_weights",
    "method_tsfm_0026": "dataset_specific_cli_only",
    "method_tsfm_0028": "no_public_local_weights",
}


def manifest_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "method_id": "method_tsfm_test",
        "checkpoint": "official/model",
        "worker_environment": "fixture_env",
        "adapter": "fixture_adapter",
        "license_id": "Apache-2.0",
        "license_acknowledgement_required": False,
        "point_reduction": "median",
        "status": "experimental_unverified",
        "reason_code": "",
        "runtime_options": {"revision": "main", "limits": {"lengths": [1, 2]}},
        "official_source_ids": ["source_fixture"],
    }
    payload.update(updates)
    return payload


def test_default_registry_covers_exactly_31_audited_cards() -> None:
    registry = ManifestRegistry.load_default()
    catalog_checkpoints = {
        card.method_uid: card.foundation_metadata["checkpoint_or_api"]
        for card in load_method_cards(
            ROOT / "numerical_agent/datasets/method_candidates_v002.jsonl"
        )
        if card.family == "foundation"
    }

    assert set(registry) == {
        f"method_tsfm_{index:04d}" for index in range(1, 32)
    }
    assert {
        manifest.method_id
        for manifest in registry.values()
        if manifest.status != "unavailable"
    } == EXPECTED_EXECUTABLE
    assert {
        manifest.method_id
        for manifest in registry.values()
        if manifest.status == "direct"
    } == EXPECTED_DIRECT
    assert {
        manifest.method_id
        for manifest in registry.values()
        if manifest.status == "experimental_unverified"
    } == EXPECTED_EXECUTABLE - EXPECTED_DIRECT
    assert {
        method_id: registry[method_id].checkpoint for method_id in registry
    } == catalog_checkpoints


def test_default_registry_preserves_precise_unavailability_reasons() -> None:
    registry = ManifestRegistry.load_default()

    assert {
        manifest.method_id: manifest.reason_code
        for manifest in registry.values()
        if manifest.status == "unavailable"
    } == EXPECTED_REASONS


def test_ttm_manifest_binds_an_exact_frequency_tuned_revision() -> None:
    manifest = ManifestRegistry.load_default()["method_tsfm_0006"]

    assert dict(manifest.runtime_options) == {
        "context_length": 512,
        "prediction_length": 96,
        "model_revision": "512-96-ft-r2.1",
        "frequency_token_hourly": 7,
        "frequency_token_daily": 8,
        "frequency_token_weekly": 9,
    }


def test_manifest_options_are_immutable_and_round_trip_as_plain_json() -> None:
    manifest = TSFMManifest.from_payload(manifest_payload())

    with pytest.raises(TypeError):
        manifest.runtime_options["revision"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.runtime_options["limits"]["lengths"] += (3,)  # type: ignore[index,operator]
    assert manifest.to_payload() == manifest_payload()


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"reason_code": "not_really_available"}, "reason_code"),
        (
            {
                "status": "unavailable",
                "reason_code": "",
                "worker_environment": "",
                "adapter": "",
                "point_reduction": "none",
                "runtime_options": {},
            },
            "reason_code",
        ),
        (
            {
                "status": "unavailable",
                "reason_code": "no_public_local_weights",
            },
            "worker_environment",
        ),
        ({"status": "experimental_unverified", "worker_environment": ""}, "environment"),
        ({"point_reduction": "mode"}, "point_reduction"),
    ],
)
def test_manifest_rejects_inconsistent_entries(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TSFMManifest.from_payload(manifest_payload(**updates))


def test_registry_rejects_duplicate_method_bindings(tmp_path: Path) -> None:
    duplicate = manifest_payload()
    path = tmp_path / "manifests.json"
    path.write_text(json.dumps([duplicate, duplicate]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        ManifestRegistry.load(path)


@pytest.mark.parametrize(
    "malformed",
    [
        '[{"method_id":"first","method_id":"second"}]',
        '[{"runtime_options":{"limit":NaN}}]',
        '[{"runtime_options":{"nested":[Infinity]}}]',
    ],
)
def test_registry_rejects_duplicate_keys_and_non_finite_json(
    tmp_path: Path, malformed: str
) -> None:
    path = tmp_path / "manifests.json"
    path.write_text(malformed, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate|non-finite"):
        ManifestRegistry.load(path)


def test_default_manifest_provenance_matches_every_catalog_card_and_source() -> None:
    registry = ManifestRegistry.load_default()
    cards = {
        card.method_uid: card
        for card in load_method_cards(
            ROOT / "numerical_agent/datasets/method_candidates_v002.jsonl"
        )
        if card.family == "foundation"
    }
    source_ids = {
        source.source_id
        for source in load_source_records(
            ROOT / "numerical_agent/datasets/source_registry_v002.jsonl"
        )
    }

    for method_id, manifest in registry.items():
        card = cards[method_id]
        expected_sources = tuple(
            dict.fromkeys(
                card.definition_source_ids + card.implementation_source_ids
            )
        )
        assert manifest.checkpoint == card.foundation_metadata["checkpoint_or_api"]
        assert manifest.license_id == card.foundation_metadata["license"]
        assert manifest.official_source_ids == expected_sources
        assert set(manifest.official_source_ids) <= source_ids


def test_checked_in_collection_contains_audited_checkpoint_and_license_corrections() -> None:
    cards = {
        card.method_uid: card
        for card in load_method_cards(
            ROOT / "numerical_agent/datasets/method_candidates_v002.jsonl"
        )
        if card.family == "foundation"
    }

    expected = {
        "method_tsfm_0001": ("google/timesfm-1.0-200m-pytorch", "Apache-2.0"),
        "method_tsfm_0003": ("Salesforce/moirai-1.1-R-base", "CC-BY-NC-4.0"),
        "method_tsfm_0007": ("thuml/timer-base-84m", "Apache-2.0 weights; MIT code"),
        "method_tsfm_0011": ("Melady/TEMPO", "Apache-2.0 weights; MIT code"),
        "method_tsfm_0013": ("thuml/sundial-base-128m", "Apache-2.0 weights; MIT code"),
        "method_tsfm_0015": ("thuml/Timer-S1", "Apache-2.0"),
        "method_tsfm_0017": ("Salesforce/moirai-2.0-R-small", "CC-BY-NC-4.0"),
        "method_tsfm_0019": ("Salesforce/moirai-moe-1.0-R-small", "CC-BY-NC-4.0"),
        "method_tsfm_0020": (
            "ibm-research/flowstate",
            "research/non-commercial; official terms ambiguous",
        ),
        "method_tsfm_0022": ("mldi-lab/Kairos_50m", "Apache-2.0"),
        "method_tsfm_0026": ("mala-lab/SEMPO", "Apache-2.0"),
        "method_tsfm_0027": ("NX-AI/TiRex", "NXAI Community License"),
        "method_tsfm_0029": (
            "Prior-Labs/tabpfn_3",
            "TabPFN-3 Non-Commercial License; Apache-2.0 code",
        ),
        "method_tsfm_0030": (
            "ibm-research/patchtst-fm-r1",
            "CC-BY-NC-SA-4.0",
        ),
    }

    assert {
        method_id: (
            cards[method_id].foundation_metadata["checkpoint_or_api"],
            cards[method_id].foundation_metadata["license"],
        )
        for method_id in expected
    } == expected
