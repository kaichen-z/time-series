"""Select which catalog methods can be implemented under the single-series method contract."""
from __future__ import annotations

from typing import Mapping, Sequence

from common.payload import read_json_object


# Categories whose methods cannot satisfy forecast(history, horizon, frequency) at all: they
# need several series, a hierarchy, a wrapped base forecaster, or forecast variance not level.
EXCLUDED_CATEGORIES: Mapping[str, str] = {
    "reconciliation": "needs a hierarchy of series, not one univariate history",
    "multivariate": "needs several related series, not one univariate history",
    "calibration": "wraps a base forecaster and produces intervals, not a point forecast",
    "volatility": "forecasts conditional variance, not the level of the series",
}

# The foundation models we seed, and the function name each becomes. The catalog identifies them
# by display name ("TimesFM 2.5"), which is not a Python identifier, so the mapping is explicit
# rather than derived: a renamed catalog entry should fail loudly, not silently seed nothing.
FOUNDATION_SEEDS: Mapping[str, str] = {
    "TimesFM 2.5": "timesfm_2_5_zero_shot",
    "Chronos-Bolt": "chronos_bolt_zero_shot",
    "Toto 2.0": "toto_zero_shot",
    "Moirai 2.0": "moirai_zero_shot",
    "Chronos-2": "chronos_2_zero_shot",
}


def seed_definitions(
    catalog_path: str,
    family: str = "statistical",
    exclude_categories: Sequence[str] = (),
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Split one family of a catalog release into implementable seeds and recorded exclusions.

    exclude_categories drops further categories on top of EXCLUDED_CATEGORIES, for categories a
    caller does not want rather than ones the contract cannot express.
    """
    payload = read_json_object(catalog_path)
    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError(f"{catalog_path} has no methods list")

    caller_excluded = {
        category: "excluded by the caller" for category in exclude_categories
    }
    reasons = {**EXCLUDED_CATEGORIES, **caller_excluded}

    seeds: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    for card in methods:
        if not isinstance(card, dict) or card.get("family") != family:
            continue
        if card.get("verification_status") != "verified":
            continue
        category = str(card.get("category", ""))
        name = str(card.get("canonical_name", ""))
        if category in reasons:
            excluded.append(
                {"name": name, "category": category, "reason": reasons[category]}
            )
            continue
        seeds.append(
            {
                "name": name,
                "category": category,
                "description": str(card.get("description", "")),
                "assumptions": [str(item) for item in card.get("assumptions", ())],
                "failure_conditions": [str(item) for item in card.get("failure_conditions", ())],
            }
        )
    return seeds, excluded


def foundation_definitions(
    catalog_path: str,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Seed the five foundation models by display name, renamed to valid function names.

    Every other foundation card is recorded as excluded: they are real models we chose not to
    seed, not models the contract cannot express.
    """
    payload = read_json_object(catalog_path)
    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError(f"{catalog_path} has no methods list")

    seeds: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    for card in methods:
        if not isinstance(card, dict) or card.get("family") != "foundation":
            continue
        if card.get("verification_status") != "verified":
            continue
        display = str(card.get("canonical_name", ""))
        category = str(card.get("category", ""))
        if display not in FOUNDATION_SEEDS:
            excluded.append(
                {"name": display, "category": category, "reason": "not one of the five seeded"}
            )
            continue
        seeds.append(
            {
                "name": FOUNDATION_SEEDS[display],
                "category": category,
                "catalog_name": display,
                "description": str(card.get("description", "")),
                "assumptions": [str(item) for item in card.get("assumptions", ())],
                "failure_conditions": [str(item) for item in card.get("failure_conditions", ())],
            }
        )

    missing = set(FOUNDATION_SEEDS) - {str(seed["catalog_name"]) for seed in seeds}
    if missing:
        raise ValueError(f"{catalog_path} has no verified card for {sorted(missing)}")
    return seeds, excluded


def full_seed_definitions(
    catalog_path: str,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Every method the loop starts from: statistical (neural included) plus five foundation models."""
    statistical, statistical_excluded = seed_definitions(catalog_path)
    foundation, foundation_excluded = foundation_definitions(catalog_path)
    return statistical + foundation, statistical_excluded + foundation_excluded


def batches(
    definitions: Sequence[Mapping[str, object]], size: int
) -> list[list[Mapping[str, object]]]:
    """Split definitions into batches, keeping methods of one category together."""
    ordered = sorted(definitions, key=lambda item: (str(item["category"]), str(item["name"])))
    return [list(ordered[start : start + size]) for start in range(0, len(ordered), size)]
