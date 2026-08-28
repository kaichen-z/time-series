"""Deterministic, entity-disjoint Dr-CiK public Train/Dev/Test manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


PARTITION_NAMES = ("train", "val", "public_test")
STRATIFICATION_FEATURES = ("frequency", "horizon_bin", "reasoning_hops", "origin")
RECOMMENDED_PUBLIC_SPLIT_SIZES = {
    "train": 80,
    "val": 20,
    "public_test": 99,
}


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
    records: list[dict], val_entities: set[str], test_entities: set[str]
) -> dict[str, list[dict]]:
    result = {name: [] for name in PARTITION_NAMES}
    for record in records:
        entity = _entity(record)
        if entity in val_entities:
            result["val"].append(record)
        elif entity in test_entities:
            result["public_test"].append(record)
        else:
            result["train"].append(record)
    return result


def _best_assignment(
    records: list[dict], *, seed: int, val_size: int, public_test_size: int
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[_entity(record)].append(record)
    sizes = {entity: len(rows) for entity, rows in grouped.items()}
    entities = sorted(grouped)
    best: tuple[float, str, dict[str, list[dict]]] | None = None
    for trial in range(2048):
        ordered = sorted(entities, key=lambda item: _stable_key(seed, trial, item))
        dev = _exact_subset(sizes, ordered, val_size)
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
    val_size: int,
    public_test_size: int,
) -> dict:
    rows = sorted(records, key=lambda item: str(item["benchmark_id"]))
    requested_total = train_size + val_size + public_test_size
    if requested_total != len(rows):
        raise ValueError("requested split sizes must sum to the number of public tasks")
    if min(train_size, val_size, public_test_size) <= 0:
        raise ValueError("every requested split size must be positive")
    ids = [str(record["benchmark_id"]) for record in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark_id values must be unique")
    if any(record.get("labels_public", True) is False for record in rows):
        raise ValueError("hidden/unlabeled tasks cannot enter a public evolution manifest")

    partitions = _best_assignment(
        rows, seed=seed, val_size=val_size, public_test_size=public_test_size
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
            "val": val_size,
            "public_test": public_test_size,
        },
        "actual_sizes": {name: len(partitions[name]) for name in PARTITION_NAMES},
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
    parser.add_argument(
        "--train-size", type=int, default=RECOMMENDED_PUBLIC_SPLIT_SIZES["train"]
    )
    parser.add_argument(
        "--val-size", type=int, default=RECOMMENDED_PUBLIC_SPLIT_SIZES["val"]
    )
    parser.add_argument(
        "--public-test-size",
        type=int,
        default=RECOMMENDED_PUBLIC_SPLIT_SIZES["public_test"],
    )
    args = parser.parse_args(argv)
    manifest = build_split_manifest(
        load_public_records(args.tasks_path),
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        public_test_size=args.public_test_size,
    )
    path = write_split_manifest(manifest, args.output)
    print(path)
    print(json.dumps(manifest["actual_sizes"], sort_keys=True))


if __name__ == "__main__":
    main()
