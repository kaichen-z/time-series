from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download


def horizon_bin(length: int) -> str:
    if length <= 30:
        return "le_30"
    if length <= 60:
        return "31_60"
    if length <= 100:
        return "61_100"
    return "gt_100"


def stable_key(seed: int, benchmark_id: str) -> str:
    return hashlib.sha256(f"{seed}:{benchmark_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--exclude", action="append", default=[])
    arguments = parser.parse_args()

    task_path = Path(
        hf_hub_download(
            "ServiceNow/Dr-CiK",
            "data/tasks/train.jsonl",
            repo_type="dataset",
        )
    )
    rows = [
        json.loads(line)
        for line in task_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    excluded = set(arguments.exclude)
    eligible = [
        row
        for row in rows
        if bool(row["labels_public"]) and row["benchmark_id"] not in excluded
    ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in eligible:
        groups[(str(row["frequency"]), horizon_bin(int(row["prediction_length"])))].append(row)
    if arguments.size < len(groups) or arguments.size > len(eligible):
        raise SystemExit(
            f"size must be between number of strata ({len(groups)}) and tasks ({len(eligible)})"
        )
    for group in groups.values():
        group.sort(key=lambda row: stable_key(arguments.seed, row["benchmark_id"]))

    selected_by_group: dict[tuple[str, str], list[dict]] = {
        key: [group[0]] for key, group in groups.items()
    }
    total = len(eligible)
    while sum(len(items) for items in selected_by_group.values()) < arguments.size:
        candidates = [
            key
            for key, group in groups.items()
            if len(selected_by_group[key]) < len(group)
        ]
        # Preserve coverage, then move toward proportional allocation. Stable
        # stratum ordering resolves exact ties without looking at any labels.
        key = max(
            candidates,
            key=lambda item: (
                arguments.size * len(groups[item]) / total
                - len(selected_by_group[item]),
                str(item),
            ),
        )
        index = len(selected_by_group[key])
        selected_by_group[key].append(groups[key][index])

    selected = [
        row
        for key in sorted(selected_by_group)
        for row in selected_by_group[key]
    ]
    selected.sort(key=lambda row: stable_key(arguments.seed, row["benchmark_id"]))
    output = {
        "dataset": "ServiceNow/Dr-CiK",
        "split": "public_dev",
        "selection_uses_future_values": False,
        "selection_uses_gt_evidence": False,
        "selection_uses_document_labels": False,
        "policy": (
            "At least one task per frequency x horizon-bin stratum, then proportional "
            "allocation; stable SHA-256 ordering within strata."
        ),
        "seed": arguments.seed,
        "size": arguments.size,
        "excluded_development_tasks": sorted(excluded),
        "tasks": [
            {
                "benchmark_id": row["benchmark_id"],
                "frequency": row["frequency"],
                "prediction_length": row["prediction_length"],
                "horizon_bin": horizon_bin(int(row["prediction_length"])),
                "reasoning_hops": row.get("reasoning_hops"),
            }
            for row in selected
        ],
    }
    destination = Path(arguments.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination)
    for item in output["tasks"]:
        print(
            item["benchmark_id"],
            item["frequency"],
            item["prediction_length"],
            item["horizon_bin"],
        )


if __name__ == "__main__":
    main()
