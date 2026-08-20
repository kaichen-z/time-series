from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.sandbox import ALLOWED_IMPORTS, UnsafeCodeError, check_code
from numerical_agent.evolution.module import EVOLUTION_IMPORTS
from numerical_agent.evolution.seed import EXCLUDED_CATEGORIES, batches, seed_definitions


CATALOG = "numerical_agent/datasets/forecast_method_dataset_v001.json"


def test_the_shared_allow_list_still_rejects_heavy_libraries() -> None:
    # evolving_loop's published-results code depends on this staying narrow.
    with pytest.raises(UnsafeCodeError, match="torch"):
        check_code("import torch\n")
    assert "torch" not in ALLOWED_IMPORTS


def test_the_evolution_allow_list_permits_them() -> None:
    check_code("import torch\nimport sklearn\nimport xgboost\n", EVOLUTION_IMPORTS)

    assert ALLOWED_IMPORTS < EVOLUTION_IMPORTS


def test_the_wider_list_still_blocks_dangerous_modules() -> None:
    with pytest.raises(UnsafeCodeError, match="os"):
        check_code("import os\n", EVOLUTION_IMPORTS)


def test_seed_selection_splits_the_catalog_as_expected() -> None:
    seeds, excluded = seed_definitions(CATALOG)

    assert len(seeds) == 93
    assert len(excluded) == 18
    assert len(seeds) + len(excluded) == 111


def test_excluded_methods_carry_a_recorded_reason() -> None:
    _, excluded = seed_definitions(CATALOG)

    assert {entry["category"] for entry in excluded} == set(EXCLUDED_CATEGORIES)
    assert all(entry["reason"] for entry in excluded)


def test_no_seed_belongs_to_an_excluded_category() -> None:
    seeds, _ = seed_definitions(CATALOG)

    assert not {s["category"] for s in seeds} & set(EXCLUDED_CATEGORIES)


def test_every_seed_name_is_a_usable_function_name() -> None:
    seeds, _ = seed_definitions(CATALOG)
    names = [str(s["name"]) for s in seeds]

    assert all(name.isidentifier() for name in names)
    assert len(set(names)) == len(names)


def test_every_seed_carries_the_text_the_prompt_needs() -> None:
    seeds, _ = seed_definitions(CATALOG)

    assert all(s["description"] for s in seeds)
    assert all(isinstance(s["assumptions"], list) for s in seeds)


def test_batches_cover_every_definition_exactly_once() -> None:
    seeds, _ = seed_definitions(CATALOG)

    grouped = batches(seeds, 10)

    assert sum(len(batch) for batch in grouped) == len(seeds)
    flattened = [str(item["name"]) for batch in grouped for item in batch]
    assert sorted(flattened) == sorted(str(s["name"]) for s in seeds)


def test_batches_keep_a_category_together() -> None:
    seeds, _ = seed_definitions(CATALOG)

    grouped = batches(seeds, 10)

    # Sorted by category, so a category spans at most ceil(n/size)+1 batches.
    baseline_batches = {
        index
        for index, batch in enumerate(grouped)
        for item in batch
        if item["category"] == "baseline"
    }
    assert len(baseline_batches) <= 2
