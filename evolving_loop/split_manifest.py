"""Deterministic, entity-disjoint Dr-CiK public Train/Dev/Test manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from common.metrics import linear_quantile

from .accuracy_profile import (
    BASELINE_PANEL,
    task_difficulty_features,
    task_toto_difficulty_features,
    validate_accuracy_profile,
    validate_toto_accuracy_profile,
)


PARTITION_NAMES = ("train", "dev", "public_test")
STRATIFICATION_FEATURES = ("frequency", "horizon_bin", "reasoning_hops", "origin")
RECOMMENDED_PUBLIC_SPLIT_SIZES = {
    "train": 80,
    "dev": 20,
    "public_test": 99,
}
DEFAULT_ACCURACY_TRIALS = 32768
MAX_RELATIVE_DIFFICULTY_GAP = 0.05


def horizon_bin(length: int) -> str:
    if length <= 30:
        return "le_30"
    if length <= 60:
        return "31_60"
    if length <= 100:
        return "61_100"
    return "gt_100"


def _stable_key(seed: int, trial: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{trial}:{value}".encode()).hexdigest()


def _entity(record: dict) -> str:
    showcase = record.get("showcase", {})
    entity = showcase.get("entity", {})
    value = record.get("entity_name") or entity.get("name")
    if not value:
        raise ValueError(f"task {record.get('benchmark_id', '<unknown>')} has no entity name")
    return str(value)


def _features(record: dict) -> dict[str, str]:
    metadata = record.get("task_metadata", record)
    length = int(metadata["prediction_length"])
    return {
        "frequency": str(metadata["frequency"]),
        "horizon_bin": horizon_bin(length),
        "reasoning_hops": str(record.get("reasoning_hops", "unknown")),
        "origin": str(record.get("origin", "unknown")),
    }


def _exact_subset(
    entity_sizes: dict[str, int], candidates: Sequence[str], target: int
) -> tuple[str, ...] | None:
    states: dict[int, tuple[str, ...]] = {0: ()}
    for entity in candidates:
        size = entity_sizes[entity]
        for total, selected in sorted(tuple(states.items()), reverse=True):
            updated = total + size
            if updated <= target and updated not in states:
                states[updated] = selected + (entity,)
    return states.get(target)


def _distribution(records: Iterable[dict]) -> dict[str, dict[str, int]]:
    counters = {feature: Counter() for feature in STRATIFICATION_FEATURES}
    for record in records:
        for feature, value in _features(record).items():
            counters[feature][value] += 1
    return {
        feature: dict(sorted(counter.items()))
        for feature, counter in counters.items()
    }


def _balance_score(partitions: dict[str, list[dict]], all_records: list[dict]) -> float:
    overall = _distribution(all_records)
    total = len(all_records)
    score = 0.0
    for name, rows in partitions.items():
        observed = _distribution(rows)
        ratio = len(rows) / total
        for feature in STRATIFICATION_FEATURES:
            for value, count in overall[feature].items():
                expected = count * ratio
                score += abs(observed[feature].get(value, 0) - expected) / max(1.0, expected)
    return score


def _partition_records(
    records: list[dict], dev_entities: set[str], test_entities: set[str]
) -> dict[str, list[dict]]:
    result = {name: [] for name in PARTITION_NAMES}
    for record in records:
        entity = _entity(record)
        if entity in dev_entities:
            result["dev"].append(record)
        elif entity in test_entities:
            result["public_test"].append(record)
        else:
            result["train"].append(record)
    return result


def _best_assignment(
    records: list[dict], *, seed: int, dev_size: int, public_test_size: int
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[_entity(record)].append(record)
    sizes = {entity: len(rows) for entity, rows in grouped.items()}
    entities = sorted(grouped)
    best: tuple[float, str, dict[str, list[dict]]] | None = None
    for trial in range(2048):
        ordered = sorted(entities, key=lambda item: _stable_key(seed, trial, item))
        dev = _exact_subset(sizes, ordered, dev_size)
        if dev is None:
            continue
        dev_set = set(dev)
        remaining = [entity for entity in ordered if entity not in dev_set]
        test = _exact_subset(sizes, remaining, public_test_size)
        if test is None:
            continue
        partitions = _partition_records(records, dev_set, set(test))
        signature = "|".join(
            ",".join(sorted(_entity(record) for record in partitions[name]))
            for name in PARTITION_NAMES
        )
        candidate = (_balance_score(partitions, records), signature, partitions)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError(
            "entity-disjoint groups cannot satisfy the requested exact Dev/Public-Test sizes"
        )
    return best[2]


def build_split_manifest(
    records: Sequence[dict],
    *,
    seed: int,
    train_size: int,
    dev_size: int,
    public_test_size: int,
) -> dict:
    rows = sorted(records, key=lambda item: str(item["benchmark_id"]))
    requested_total = train_size + dev_size + public_test_size
    if requested_total != len(rows):
        raise ValueError("requested split sizes must sum to the number of public tasks")
    if min(train_size, dev_size, public_test_size) <= 0:
        raise ValueError("every requested split size must be positive")
    ids = [str(record["benchmark_id"]) for record in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark_id values must be unique")
    if any(record.get("labels_public", True) is False for record in rows):
        raise ValueError("hidden/unlabeled tasks cannot enter a public evolution manifest")

    partitions = _best_assignment(
        rows, seed=seed, dev_size=dev_size, public_test_size=public_test_size
    )

    def summarize(items: list[dict]) -> dict:
        return {
            "task_ids": sorted(str(item["benchmark_id"]) for item in items),
            "entities": sorted({_entity(item) for item in items}),
            "distribution": _distribution(items),
        }

    payload = {
        "schema_version": 1,
        "dataset": "ServiceNow/Dr-CiK",
        "source_split": "public_dev",
        "seed": seed,
        "grouping": "entity_disjoint",
        "stratification_features": list(STRATIFICATION_FEATURES),
        "selection_uses_future_values": False,
        "selection_uses_gt_evidence": False,
        "selection_uses_document_labels": False,
        "target_sizes": {
            "train": train_size,
            "dev": dev_size,
            "public_test": public_test_size,
        },
        "actual_sizes": {name: len(partitions[name]) for name in PARTITION_NAMES},
        "partitions": {name: summarize(partitions[name]) for name in PARTITION_NAMES},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def _accuracy_stratification_features() -> tuple[str, ...]:
    return (
        "frequency",
        "horizon_bin",
        "difficulty_decile",
        *(f"{model}_difficulty_quintile" for model in BASELINE_PANEL),
    )


def _accuracy_values(record: dict, difficulty: dict[str, dict[str, object]]) -> dict[str, str]:
    task_id = str(record["benchmark_id"])
    task_difficulty = difficulty[task_id]
    metadata = _features(record)
    values = {
        "frequency": metadata["frequency"],
        "horizon_bin": metadata["horizon_bin"],
        "difficulty_decile": str(task_difficulty["difficulty_decile"]),
    }
    quintiles = task_difficulty["model_quintiles"]
    for model in BASELINE_PANEL:
        values[f"{model}_difficulty_quintile"] = str(quintiles[model])
    return values


def _accuracy_distribution(
    records: Iterable[dict], difficulty: dict[str, dict[str, object]]
) -> dict[str, dict[str, int]]:
    counters = {feature: Counter() for feature in _accuracy_stratification_features()}
    for record in records:
        for feature, value in _accuracy_values(record, difficulty).items():
            counters[feature][value] += 1
    return {
        feature: dict(sorted(counter.items()))
        for feature, counter in counters.items()
    }


def _accuracy_objective(
    partitions: dict[str, list[dict]],
    all_records: list[dict],
    difficulty: dict[str, dict[str, object]],
) -> dict[str, float]:
    overall = _accuracy_distribution(all_records, difficulty)
    total = len(all_records)
    deviations = []
    means = []
    for rows in partitions.values():
        observed = _accuracy_distribution(rows, difficulty)
        ratio = len(rows) / total
        for feature in _accuracy_stratification_features():
            for value, count in overall[feature].items():
                expected = count * ratio
                deviations.append(
                    abs(observed[feature].get(value, 0) - expected) / max(1.0, expected)
                )
        means.append(
            statistics.fmean(
                float(difficulty[str(record["benchmark_id"])]["difficulty_score"])
                for record in rows
            )
        )
    absolute_gap = max(means) - min(means)
    overall_mean = statistics.fmean(
        float(difficulty[str(record["benchmark_id"])]["difficulty_score"])
        for record in all_records
    )
    return {
        "max_normalized_bin_deviation": max(deviations, default=0.0),
        "total_normalized_distribution_deviation": sum(deviations),
        "max_mean_difficulty_gap": absolute_gap,
        "relative_mean_difficulty_gap": absolute_gap / max(overall_mean, 1e-12),
    }


def _accuracy_assignment_key(
    objective: dict[str, float], signature: str
) -> tuple[int, float, float, float, str]:
    return (
        int(objective["relative_mean_difficulty_gap"] > MAX_RELATIVE_DIFFICULTY_GAP),
        objective["relative_mean_difficulty_gap"],
        objective["max_normalized_bin_deviation"],
        objective["total_normalized_distribution_deviation"],
        signature,
    )


def _best_accuracy_assignment(
    records: list[dict],
    difficulty: dict[str, dict[str, object]],
    *,
    seed: int,
    dev_size: int,
    public_test_size: int,
    trials: int,
) -> tuple[dict[str, list[dict]], dict[str, float]]:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[_entity(record)].append(record)
    sizes = {entity: len(rows) for entity, rows in grouped.items()}
    entities = sorted(grouped)
    best: tuple[
        tuple[int, float, float, float, str],
        dict[str, list[dict]],
        dict[str, float],
    ] | None = None
    for trial in range(trials):
        ordered = sorted(entities, key=lambda item: _stable_key(seed, trial, item))
        dev = _exact_subset(sizes, ordered, dev_size)
        if dev is None:
            continue
        dev_set = set(dev)
        remaining = [entity for entity in ordered if entity not in dev_set]
        test = _exact_subset(sizes, remaining, public_test_size)
        if test is None:
            continue
        partitions = _partition_records(records, dev_set, set(test))
        if any(
            len({_entity(record) for record in partition_rows})
            < (len(partition_rows) + 1) // 2
            for partition_rows in partitions.values()
        ):
            continue
        signature = "|".join(
            ",".join(sorted(str(record["benchmark_id"]) for record in partitions[name]))
            for name in PARTITION_NAMES
        )
        objective = _accuracy_objective(partitions, records, difficulty)
        candidate = (_accuracy_assignment_key(objective, signature), partitions, objective)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError(
            "entity-disjoint groups cannot satisfy the requested exact Dev/Public-Test sizes"
        )
    best_gap = best[2]["relative_mean_difficulty_gap"]
    if best_gap > MAX_RELATIVE_DIFFICULTY_GAP:
        raise ValueError(
            "no candidate passed the 5% relative mean difficulty gate; "
            f"best achieved {best_gap:.6f}"
        )
    return best[1], best[2]


def build_accuracy_stratified_split_manifest(
    records: Sequence[dict],
    accuracy_profile: dict,
    *,
    seed: int,
    train_size: int,
    dev_size: int,
    public_test_size: int,
    trials: int = DEFAULT_ACCURACY_TRIALS,
) -> dict:
    """Build an accuracy-balanced v2 manifest without changing the v1 contract."""
    rows = sorted(records, key=lambda item: str(item["benchmark_id"]))
    requested_total = train_size + dev_size + public_test_size
    if requested_total != len(rows):
        raise ValueError("requested split sizes must sum to the number of public tasks")
    if min(train_size, dev_size, public_test_size) <= 0:
        raise ValueError("every requested split size must be positive")
    task_ids = [str(record["benchmark_id"]) for record in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("benchmark_id values must be unique")
    if any(record.get("labels_public", True) is False for record in rows):
        raise ValueError("hidden/unlabeled tasks cannot enter a public evolution manifest")
    profile = validate_accuracy_profile(accuracy_profile, task_ids)
    difficulty = task_difficulty_features(profile)
    partitions, objective = _best_accuracy_assignment(
        rows,
        difficulty,
        seed=seed,
        dev_size=dev_size,
        public_test_size=public_test_size,
        trials=trials,
    )

    def summarize(items: list[dict]) -> dict:
        scores = [
            float(difficulty[str(item["benchmark_id"])]["difficulty_score"])
            for item in items
        ]
        return {
            "task_ids": sorted(str(item["benchmark_id"]) for item in items),
            "entities": sorted({_entity(item) for item in items}),
            "distribution": _accuracy_distribution(items, difficulty),
            "difficulty": {
                "mean_score": statistics.fmean(scores),
                "median_score": statistics.median(scores),
                "p90_score": linear_quantile(scores, 0.90),
                "max_score": max(scores),
                "baseline_mean_smae": {
                    model: statistics.fmean(
                        float(profile["tasks"][str(item["benchmark_id"])]["smae"][model])
                        for item in items
                    )
                    for model in BASELINE_PANEL
                },
            },
        }

    payload = {
        "schema_version": 2,
        "dataset": "ServiceNow/Dr-CiK",
        "source_split": "public_dev",
        "seed": seed,
        "grouping": "entity_disjoint",
        "stratification_features": list(_accuracy_stratification_features()),
        "selection_uses_history_values": True,
        "selection_uses_future_values": True,
        "selection_uses_gt_evidence": False,
        "selection_uses_document_labels": False,
        "selection_uses_model_metrics": True,
        "difficulty_profile_schema": profile["profile_schema"],
        "difficulty_profile_sha256": profile["profile_sha256"],
        "difficulty_source_commit": profile["source_commit"],
        "difficulty_panel_models": list(BASELINE_PANEL),
        "assignment_trials": trials,
        "minimum_entity_fraction": 0.5,
        "target_sizes": {
            "train": train_size,
            "dev": dev_size,
            "public_test": public_test_size,
        },
        "actual_sizes": {name: len(partitions[name]) for name in PARTITION_NAMES},
        "objective": objective,
        "partitions": {name: summarize(partitions[name]) for name in PARTITION_NAMES},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def _toto_stratification_features() -> tuple[str, ...]:
    return (
        *_accuracy_stratification_features(),
        "toto_smae_decile",
        "toto_smae_quintile",
        "toto_srmse_decile",
        "toto_srmse_quintile",
    )


def _toto_balance_values(
    record: dict,
    baseline_difficulty: dict[str, dict[str, object]],
    toto_difficulty: dict[str, dict[str, object]],
) -> dict[str, str]:
    task_id = str(record["benchmark_id"])
    values = _accuracy_values(record, baseline_difficulty)
    values.update(
        {
            "toto_smae_decile": str(toto_difficulty[task_id]["smae_decile"]),
            "toto_smae_quintile": str(toto_difficulty[task_id]["smae_quintile"]),
            "toto_srmse_decile": str(toto_difficulty[task_id]["srmse_decile"]),
            "toto_srmse_quintile": str(toto_difficulty[task_id]["srmse_quintile"]),
        }
    )
    return values


def _toto_balance_distribution(
    records: Iterable[dict],
    baseline_difficulty: dict[str, dict[str, object]],
    toto_difficulty: dict[str, dict[str, object]],
) -> dict[str, dict[str, int]]:
    counters = {feature: Counter() for feature in _toto_stratification_features()}
    for record in records:
        for feature, value in _toto_balance_values(
            record, baseline_difficulty, toto_difficulty
        ).items():
            counters[feature][value] += 1
    return {
        feature: dict(sorted(counter.items()))
        for feature, counter in counters.items()
    }


def _relative_mean_gap(values_by_partition: list[float], overall_mean: float) -> float:
    return (max(values_by_partition) - min(values_by_partition)) / max(overall_mean, 1e-12)


def _toto_balance_objective(
    partitions: dict[str, list[dict]],
    all_records: list[dict],
    baseline_difficulty: dict[str, dict[str, object]],
    toto_difficulty: dict[str, dict[str, object]],
) -> dict[str, float]:
    overall = _toto_balance_distribution(
        all_records, baseline_difficulty, toto_difficulty
    )
    total = len(all_records)
    deviations = []
    baseline_means = []
    toto_smae_means = []
    toto_srmse_means = []
    for rows in partitions.values():
        observed = _toto_balance_distribution(
            rows, baseline_difficulty, toto_difficulty
        )
        ratio = len(rows) / total
        for feature in _toto_stratification_features():
            for value, count in overall[feature].items():
                expected = count * ratio
                deviations.append(
                    abs(observed[feature].get(value, 0) - expected) / max(1.0, expected)
                )
        task_ids = [str(record["benchmark_id"]) for record in rows]
        baseline_means.append(
            statistics.fmean(
                float(baseline_difficulty[task_id]["difficulty_score"])
                for task_id in task_ids
            )
        )
        toto_smae_means.append(
            statistics.fmean(float(toto_difficulty[task_id]["smae"]) for task_id in task_ids)
        )
        toto_srmse_means.append(
            statistics.fmean(float(toto_difficulty[task_id]["srmse"]) for task_id in task_ids)
        )
    baseline_overall = statistics.fmean(
        float(baseline_difficulty[str(record["benchmark_id"])]["difficulty_score"])
        for record in all_records
    )
    toto_smae_overall = statistics.fmean(
        float(toto_difficulty[str(record["benchmark_id"])]["smae"])
        for record in all_records
    )
    toto_srmse_overall = statistics.fmean(
        float(toto_difficulty[str(record["benchmark_id"])]["srmse"])
        for record in all_records
    )
    baseline_gap = _relative_mean_gap(baseline_means, baseline_overall)
    toto_smae_gap = _relative_mean_gap(toto_smae_means, toto_smae_overall)
    toto_srmse_gap = _relative_mean_gap(toto_srmse_means, toto_srmse_overall)
    return {
        "max_normalized_bin_deviation": max(deviations, default=0.0),
        "total_normalized_distribution_deviation": sum(deviations),
        "baseline_relative_mean_difficulty_gap": baseline_gap,
        "toto_smae_relative_mean_gap": toto_smae_gap,
        "toto_srmse_relative_mean_gap": toto_srmse_gap,
        "max_relative_mean_gap": max(baseline_gap, toto_smae_gap, toto_srmse_gap),
    }


def _best_toto_balance_assignment(
    records: list[dict],
    baseline_difficulty: dict[str, dict[str, object]],
    toto_difficulty: dict[str, dict[str, object]],
    *,
    seed: int,
    dev_size: int,
    public_test_size: int,
    trials: int,
) -> tuple[dict[str, list[dict]], dict[str, float]]:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[_entity(record)].append(record)
    sizes = {entity: len(rows) for entity, rows in grouped.items()}
    entities = sorted(grouped)
    best_passing = None
    best_failing = None
    for trial in range(trials):
        ordered = sorted(entities, key=lambda item: _stable_key(seed, trial, item))
        dev = _exact_subset(sizes, ordered, dev_size)
        if dev is None:
            continue
        dev_set = set(dev)
        remaining = [entity for entity in ordered if entity not in dev_set]
        test = _exact_subset(sizes, remaining, public_test_size)
        if test is None:
            continue
        partitions = _partition_records(records, dev_set, set(test))
        if any(
            len({_entity(record) for record in rows}) < (len(rows) + 1) // 2
            for rows in partitions.values()
        ):
            continue
        signature = "|".join(
            ",".join(sorted(str(record["benchmark_id"]) for record in partitions[name]))
            for name in PARTITION_NAMES
        )
        objective = _toto_balance_objective(
            partitions, records, baseline_difficulty, toto_difficulty
        )
        component_gaps = (
            objective["baseline_relative_mean_difficulty_gap"],
            objective["toto_smae_relative_mean_gap"],
            objective["toto_srmse_relative_mean_gap"],
        )
        gate_failures = sum(
            gap > MAX_RELATIVE_DIFFICULTY_GAP for gap in component_gaps
        )
        if gate_failures == 0:
            key = (
                objective["max_normalized_bin_deviation"],
                objective["total_normalized_distribution_deviation"],
                objective["max_relative_mean_gap"],
                *component_gaps,
                signature,
            )
            candidate = (key, partitions, objective)
            if best_passing is None or candidate[0] < best_passing[0]:
                best_passing = candidate
        else:
            key = (
                gate_failures,
                objective["max_relative_mean_gap"],
                *component_gaps,
                objective["max_normalized_bin_deviation"],
                objective["total_normalized_distribution_deviation"],
                signature,
            )
            candidate = (key, partitions, objective)
            if best_failing is None or candidate[0] < best_failing[0]:
                best_failing = candidate
    if best_passing is not None:
        return best_passing[1], best_passing[2]
    if best_failing is None:
        raise ValueError(
            "entity-disjoint groups cannot satisfy the requested exact Dev/Public-Test sizes"
        )
    raise ValueError(
        "no candidate passed all 5% baseline/Toto relative mean difficulty gates; "
        f"best achieved {best_failing[2]['max_relative_mean_gap']:.6f}"
    )


def build_toto_balanced_split_manifest(
    records: Sequence[dict],
    accuracy_profile: dict,
    toto_accuracy_profile: dict,
    *,
    seed: int,
    train_size: int,
    dev_size: int,
    public_test_size: int,
    trials: int = DEFAULT_ACCURACY_TRIALS,
) -> dict:
    """Build a v3 entity-disjoint split balanced on panel and Toto accuracy."""
    rows = sorted(records, key=lambda item: str(item["benchmark_id"]))
    if train_size + dev_size + public_test_size != len(rows):
        raise ValueError("requested split sizes must sum to the number of public tasks")
    if min(train_size, dev_size, public_test_size) <= 0:
        raise ValueError("every requested split size must be positive")
    task_ids = [str(record["benchmark_id"]) for record in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("benchmark_id values must be unique")
    if any(record.get("labels_public", True) is False for record in rows):
        raise ValueError("hidden/unlabeled tasks cannot enter a public evolution manifest")
    baseline_profile = validate_accuracy_profile(accuracy_profile, task_ids)
    toto_profile = validate_toto_accuracy_profile(toto_accuracy_profile, task_ids)
    baseline_difficulty = task_difficulty_features(baseline_profile)
    toto_difficulty = task_toto_difficulty_features(toto_profile)
    partitions, objective = _best_toto_balance_assignment(
        rows,
        baseline_difficulty,
        toto_difficulty,
        seed=seed,
        dev_size=dev_size,
        public_test_size=public_test_size,
        trials=trials,
    )

    def summarize(items: list[dict]) -> dict:
        task_ids = [str(item["benchmark_id"]) for item in items]
        baseline_scores = [
            float(baseline_difficulty[task_id]["difficulty_score"])
            for task_id in task_ids
        ]
        toto_summary = {}
        for metric in ("smae", "srmse"):
            values = [float(toto_difficulty[task_id][metric]) for task_id in task_ids]
            toto_summary[metric] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p90": linear_quantile(values, 0.90),
                "max": max(values),
            }
        return {
            "task_ids": sorted(task_ids),
            "entities": sorted({_entity(item) for item in items}),
            "distribution": _toto_balance_distribution(
                items, baseline_difficulty, toto_difficulty
            ),
            "difficulty": {
                "mean_score": statistics.fmean(baseline_scores),
                "median_score": statistics.median(baseline_scores),
                "p90_score": linear_quantile(baseline_scores, 0.90),
                "max_score": max(baseline_scores),
            },
            "toto_difficulty": toto_summary,
        }

    payload = {
        "schema_version": 3,
        "dataset": "ServiceNow/Dr-CiK",
        "source_split": "public_dev",
        "seed": seed,
        "grouping": "entity_disjoint",
        "stratification_features": list(_toto_stratification_features()),
        "selection_uses_history_values": True,
        "selection_uses_future_values": True,
        "selection_uses_gt_evidence": False,
        "selection_uses_document_labels": False,
        "selection_uses_model_metrics": True,
        "difficulty_profile_schema": baseline_profile["profile_schema"],
        "difficulty_profile_sha256": baseline_profile["profile_sha256"],
        "difficulty_source_commit": baseline_profile["source_commit"],
        "difficulty_panel_models": list(BASELINE_PANEL),
        "toto_difficulty_profile_schema": toto_profile["profile_schema"],
        "toto_difficulty_profile_sha256": toto_profile["profile_sha256"],
        "toto_difficulty_model": toto_profile["model_name"],
        "assignment_trials": trials,
        "maximum_relative_mean_gap": MAX_RELATIVE_DIFFICULTY_GAP,
        "minimum_entity_fraction": 0.5,
        "target_sizes": {
            "train": train_size,
            "dev": dev_size,
            "public_test": public_test_size,
        },
        "actual_sizes": {name: len(partitions[name]) for name in PARTITION_NAMES},
        "objective": objective,
        "partitions": {name: summarize(partitions[name]) for name in PARTITION_NAMES},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def write_split_manifest(manifest: dict, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_public_records(tasks_path: str | Path) -> list[dict]:
    path = Path(tasks_path)
    if path.is_dir():
        records = [
            json.loads(item.read_text(encoding="utf-8"))
            for item in sorted(path.glob("*.json"))
        ]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
    else:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def is_public(record: dict) -> bool:
        series = record.get("series", record)
        future = series.get("future_values")
        return (
            record.get("labels_public", True) is not False
            and bool(future)
            and future[0] is not None
        )

    return [record for record in records if is_public(record)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--accuracy-profile")
    parser.add_argument("--toto-accuracy-profile")
    parser.add_argument("--trials", type=int, default=DEFAULT_ACCURACY_TRIALS)
    parser.add_argument(
        "--train-size", type=int, default=RECOMMENDED_PUBLIC_SPLIT_SIZES["train"]
    )
    parser.add_argument(
        "--dev-size", type=int, default=RECOMMENDED_PUBLIC_SPLIT_SIZES["dev"]
    )
    parser.add_argument(
        "--public-test-size",
        type=int,
        default=RECOMMENDED_PUBLIC_SPLIT_SIZES["public_test"],
    )
    args = parser.parse_args(argv)
    records = load_public_records(args.tasks_path)
    if args.toto_accuracy_profile and not args.accuracy_profile:
        parser.error("--toto-accuracy-profile requires --accuracy-profile")
    if args.toto_accuracy_profile:
        profile = json.loads(Path(args.accuracy_profile).read_text(encoding="utf-8"))
        toto_profile = json.loads(
            Path(args.toto_accuracy_profile).read_text(encoding="utf-8")
        )
        manifest = build_toto_balanced_split_manifest(
            records,
            profile,
            toto_profile,
            seed=args.seed,
            train_size=args.train_size,
            dev_size=args.dev_size,
            public_test_size=args.public_test_size,
            trials=args.trials,
        )
    elif args.accuracy_profile:
        profile = json.loads(Path(args.accuracy_profile).read_text(encoding="utf-8"))
        manifest = build_accuracy_stratified_split_manifest(
            records,
            profile,
            seed=args.seed,
            train_size=args.train_size,
            dev_size=args.dev_size,
            public_test_size=args.public_test_size,
            trials=args.trials,
        )
    else:
        manifest = build_split_manifest(
            records,
            seed=args.seed,
            train_size=args.train_size,
            dev_size=args.dev_size,
            public_test_size=args.public_test_size,
        )
    path = write_split_manifest(manifest, args.output)
    print(path)
    print(json.dumps(manifest["actual_sizes"], sort_keys=True))


if __name__ == "__main__":
    main()
