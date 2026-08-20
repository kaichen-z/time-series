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


def seed_definitions(
    catalog_path: str, family: str = "statistical"
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Split one family of a catalog release into implementable seeds and recorded exclusions."""
    payload = read_json_object(catalog_path)
    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError(f"{catalog_path} has no methods list")

    seeds: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    for card in methods:
        if not isinstance(card, dict) or card.get("family") != family:
            continue
        if card.get("verification_status") != "verified":
            continue
        category = str(card.get("category", ""))
        name = str(card.get("canonical_name", ""))
        if category in EXCLUDED_CATEGORIES:
            excluded.append(
                {"name": name, "category": category, "reason": EXCLUDED_CATEGORIES[category]}
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


def batches(
    definitions: Sequence[Mapping[str, object]], size: int
) -> list[list[Mapping[str, object]]]:
    """Split definitions into batches, keeping methods of one category together."""
    ordered = sorted(definitions, key=lambda item: (str(item["category"]), str(item["name"])))
    return [list(ordered[start : start + size]) for start in range(0, len(ordered), size)]
