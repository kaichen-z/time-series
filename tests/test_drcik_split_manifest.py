from __future__ import annotations

import json

import pytest

from evolving_loop.accuracy_profile import BASELINE_PANEL, build_accuracy_profile
from evolving_loop.split_manifest import (
    RECOMMENDED_PUBLIC_SPLIT_SIZES,
    build_accuracy_stratified_split_manifest,
    build_split_manifest,
    load_public_records,
    write_split_manifest,
)


def _record(task_id: str, entity: str, *, frequency: str, hops: int, origin: str) -> dict:
    return {
        "benchmark_id": task_id,
        "origin": origin,
        "reasoning_hops": hops,
        "showcase": {"entity": {"name": entity}},
        "task_metadata": {
            "frequency": frequency,
            "prediction_length": 24 if frequency == "1 hour" else 100,
        },
        # Evaluator-only fields deliberately appear in the fixture. The split
        # manifest must never copy them into its output.
        "series": {"future_values": [999.0]},
        "annotations": {"gt_evidence": [{"evidence": "secret label"}]},
    }


def _records() -> list[dict]:
    rows = []
    entity_sizes = (2, 2, 2, 2, 1, 1, 1, 1)
    for entity_index, size in enumerate(entity_sizes):
        for offset in range(size):
            rows.append(
                _record(
                    f"task_{entity_index}_{offset}",
                    f"entity_{entity_index}",
                    frequency="1 hour" if (entity_index + offset) % 2 == 0 else "1 day",
                    hops=2 if entity_index % 2 == 0 else 4,
                    origin="synthetic" if offset == 0 else "human",
                )
            )
    return rows


def _single_entity_records() -> list[dict]:
    return [
        _record(
            f"task_{index}",
            f"single_entity_{index}",
            frequency="1 hour" if index % 2 == 0 else "1 day",
            hops=4,
            origin="synthetic",
        )
        for index in range(12)
    ]


def _accuracy_profile(records: list[dict]) -> dict:
    task_ids = sorted(str(row["benchmark_id"]) for row in records)
    scores = {
        model: {
            task_id: float(index) / max(1, len(task_ids) - 1)
            for index, task_id in enumerate(task_ids)
        }
        for model in BASELINE_PANEL
    }
    sources = {
        model: {
            "path": f"runs/baselines/{model}_dev.log",
            "sha256": str(index + 1) * 64,
        }
        for index, model in enumerate(BASELINE_PANEL)
    }
    return build_accuracy_profile(
        scores,
        source_commit="1" * 40,
        source_files=sources,
    )


def test_build_split_manifest_is_exact_deterministic_and_entity_disjoint() -> None:
    """Catches task loss, unstable assignment, or one entity leaking across partitions."""
    first = build_split_manifest(
        _records(), seed=17, train_size=6, dev_size=3, public_test_size=3
    )
    second = build_split_manifest(
        list(reversed(_records())), seed=17, train_size=6, dev_size=3, public_test_size=3
    )

    assert first == second
    assert first["actual_sizes"] == {"train": 6, "dev": 3, "public_test": 3}

    partitions = first["partitions"]
    task_sets = [set(partitions[name]["task_ids"]) for name in ("train", "dev", "public_test")]
    entity_sets = [set(partitions[name]["entities"]) for name in ("train", "dev", "public_test")]
    assert set.union(*task_sets) == {
        "task_0_0", "task_0_1", "task_1_0", "task_1_1",
        "task_2_0", "task_2_1", "task_3_0", "task_3_1",
        "task_4_0", "task_5_0", "task_6_0", "task_7_0",
    }
    assert sum(len(items) for items in task_sets) == 12
    assert task_sets[0].isdisjoint(task_sets[1])
    assert task_sets[0].isdisjoint(task_sets[2])
    assert task_sets[1].isdisjoint(task_sets[2])
    assert entity_sets[0].isdisjoint(entity_sets[1])
    assert entity_sets[0].isdisjoint(entity_sets[2])
    assert entity_sets[1].isdisjoint(entity_sets[2])
    serialized = json.dumps(first)
    assert "999.0" not in serialized
    assert "secret label" not in serialized
    assert first["selection_uses_future_values"] is False
    assert first["selection_uses_gt_evidence"] is False
    assert first["manifest_sha256"]


def test_build_split_manifest_rejects_invalid_requested_total() -> None:
    """Catches silently dropping tasks when requested split sizes do not cover the dataset."""
    with pytest.raises(ValueError, match="must sum to the number of public tasks"):
        build_split_manifest(
            _records(), seed=17, train_size=5, dev_size=3, public_test_size=3
        )


def test_recommended_public_split_reserves_dev_and_large_test() -> None:
    """Catches the formal protocol drifting away from 80/20 development and 99 test."""
    assert RECOMMENDED_PUBLIC_SPLIT_SIZES == {
        "train": 80,
        "dev": 20,
        "public_test": 99,
    }


def test_accuracy_stratified_split_is_exact_deterministic_and_explicit() -> None:
    """Catches v2 losing entity isolation or hiding its outcome-derived selection inputs."""
    records = _single_entity_records()
    profile = _accuracy_profile(records)

    first = build_accuracy_stratified_split_manifest(
        records,
        profile,
        seed=29,
        train_size=6,
        dev_size=3,
        public_test_size=3,
        trials=512,
    )
    second = build_accuracy_stratified_split_manifest(
        list(reversed(records)),
        profile,
        seed=29,
        train_size=6,
        dev_size=3,
        public_test_size=3,
        trials=512,
    )

    assert first == second
    assert first["schema_version"] == 2
    assert first["actual_sizes"] == {"train": 6, "dev": 3, "public_test": 3}
    assert first["selection_uses_future_values"] is True
    assert first["selection_uses_model_metrics"] is True
    assert first["selection_uses_gt_evidence"] is False
    assert first["difficulty_profile_sha256"] == profile["profile_sha256"]
    assert first["assignment_trials"] == 512
    entity_sets = [
        set(first["partitions"][name]["entities"])
        for name in ("train", "dev", "public_test")
    ]
    assert entity_sets[0].isdisjoint(entity_sets[1])
    assert entity_sets[0].isdisjoint(entity_sets[2])
    assert entity_sets[1].isdisjoint(entity_sets[2])


def test_accuracy_stratified_split_balances_controlled_difficulty() -> None:
    """Catches an objective that records difficulty fields without balancing them."""
    records = _single_entity_records()
    manifest = build_accuracy_stratified_split_manifest(
        records,
        _accuracy_profile(records),
        seed=29,
        train_size=6,
        dev_size=3,
        public_test_size=3,
        trials=512,
    )

    means = [
        manifest["partitions"][name]["difficulty"]["mean_score"]
        for name in ("train", "dev", "public_test")
    ]
    assert max(means) - min(means) <= 0.20
    assert manifest["objective"]["max_mean_difficulty_gap"] <= 0.20


def test_accuracy_stratified_split_rejects_wrong_task_coverage() -> None:
    """Catches a partial accuracy ledger silently assigning unprofiled public tasks."""
    records = _single_entity_records()
    profile = _accuracy_profile(records[:-1])

    with pytest.raises(ValueError, match="task coverage"):
        build_accuracy_stratified_split_manifest(
            records,
            profile,
            seed=29,
            train_size=6,
            dev_size=3,
            public_test_size=3,
            trials=32,
        )


def test_write_split_manifest_round_trips_json(tmp_path) -> None:
    """Catches writing a different artifact than the manifest that was validated."""
    manifest = build_split_manifest(
        _records(), seed=17, train_size=6, dev_size=3, public_test_size=3
    )
    destination = tmp_path / "drcik_public_v1.json"

    write_split_manifest(manifest, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == manifest
    assert destination.read_text(encoding="utf-8").endswith("\n")


def test_load_public_records_excludes_hidden_rows_from_mixed_directory(tmp_path) -> None:
    """Catches the official mixed export accidentally placing Hidden 80 into Public 199."""
    public = _record("task_public", "entity_public", frequency="1 day", hops=2, origin="human")
    public["labels_public"] = True
    hidden = _record("task_hidden", "entity_hidden", frequency="1 day", hops=2, origin="human")
    hidden["labels_public"] = False
    hidden["series"]["future_values"] = [None]
    (tmp_path / "task_public.json").write_text(json.dumps(public), encoding="utf-8")
    (tmp_path / "task_hidden.json").write_text(json.dumps(hidden), encoding="utf-8")

    rows = load_public_records(tmp_path)

    assert [row["benchmark_id"] for row in rows] == ["task_public"]
