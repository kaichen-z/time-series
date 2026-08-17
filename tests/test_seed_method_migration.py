from __future__ import annotations

import json
from pathlib import Path

from numerical_agent.collection.registry import load_method_cards, load_source_records
from numerical_agent.collection.seed import migrate_legacy_statistical_seed


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


def test_checked_in_seed_candidates_and_empty_source_registry_are_parseable() -> None:
    cards = load_method_cards(
        ROOT / "numerical_agent/datasets/method_candidates_v001.jsonl"
    )
    sources = load_source_records(
        ROOT / "numerical_agent/datasets/source_registry_v001.jsonl"
    )

    assert len(cards) == 41
    assert sources == ()
