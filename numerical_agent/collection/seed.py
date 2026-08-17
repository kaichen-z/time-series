"""Deterministic migration of the pre-existing statistical method seed.

The legacy dictionary is useful as a discovery seed, but it contains no source
provenance.  Migration therefore preserves every behavioral claim while marking
every card unverified and leaving all source references empty.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .contracts import MethodCard


CATEGORY_BY_LEGACY_ID: Mapping[str, str] = {
    "naive_last": "baseline",
    "naive_mean": "baseline",
    "naive_drift": "baseline",
    "seasonal_naive": "baseline",
    "simple_moving_average": "baseline",
    "ses": "exponential_smoothing",
    "holt_linear_trend": "trend",
    "holt_damped_trend": "trend",
    "holt_winters_additive": "exponential_smoothing",
    "holt_winters_multiplicative": "exponential_smoothing",
    "holt_winters_damped": "exponential_smoothing",
    "ets_auto": "exponential_smoothing",
    "theta_classic": "trend",
    "theta_optimized": "trend",
    "ar": "autoregressive",
    "ma": "autoregressive",
    "arma": "autoregressive",
    "arima_auto": "arima",
    "sarima_auto": "arima",
    "stl_naive": "decomposition",
    "classical_decomposition": "decomposition",
    "stl_ets": "decomposition",
    "stl_arima": "decomposition",
    "croston": "intermittent_demand",
    "croston_sba": "intermittent_demand",
    "tsb": "intermittent_demand",
    "adida": "intermittent_demand",
    "imapa": "intermittent_demand",
    "local_level_kalman": "local_level",
    "local_linear_trend_kalman": "state_space",
    "structural_time_series_bsm": "state_space",
    "robust_loess_trend": "robust",
    "fourier_harmonic_regression": "spectral",
    "fft_dominant_frequency_extrapolation": "spectral",
    "wavelet_trend_detail_forecast": "spectral",
    "linear_trend_regression": "regression",
    "polynomial_trend_regression": "regression",
    "piecewise_linear_trend": "change_point",
    "seasonal_block_bootstrap": "probabilistic",
    "empirical_quantile_persistence": "probabilistic",
    "statistical_ensemble_mean": "reconciliation",
}


def _card_payload(method: Mapping[str, object], index: int) -> dict[str, object]:
    legacy_id = str(method["method_id"])
    try:
        category = CATEGORY_BY_LEGACY_ID[legacy_id]
    except KeyError as exc:
        raise ValueError(f"legacy method {legacy_id!r} has no category mapping") from exc
    return {
        "method_uid": f"method_seed_{index:04d}",
        "definition_version": 1,
        "canonical_name": legacy_id,
        "aliases": [legacy_id],
        "family": "statistical",
        "category": category,
        "description": method["description"],
        "assumptions": method["assumptions"],
        "failure_conditions": method["failure_conditions"],
        "applicability": {
            "minimum_history": 1,
            "frequencies": ["any"],
            "supports_univariate": True,
            "supports_covariates": False,
            "supports_probabilistic_output": category == "probabilistic",
        },
        "hyperparameters": [],
        "definition_source_ids": [],
        "implementation_source_ids": [],
        "implementation_availability": "unknown",
        "verification_status": "unverified",
        "lineage": {
            "operation": "migrated_seed",
            "parent_method_uids": [],
            "legacy_dictionary_id": "statistical_base_methods_v000",
            "legacy_method_id": legacy_id,
        },
        "foundation_metadata": {},
    }


def migrate_legacy_statistical_seed(
    source: str | Path,
    destination: str | Path,
) -> tuple[MethodCard, ...]:
    """Convert the legacy JSON dictionary into deterministic MethodCard JSONL."""

    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError("legacy dictionary must contain a methods list")
    cards = tuple(
        MethodCard.from_payload(_card_payload(method, index))
        for index, method in enumerate(methods, start=1)
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(
        json.dumps(card.to_payload(), ensure_ascii=False, sort_keys=True) + "\n"
        for card in cards
    )
    output.write_text(serialized, encoding="utf-8")
    return cards
