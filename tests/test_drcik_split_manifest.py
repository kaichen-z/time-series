from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import pytest

from common.metrics import linear_quantile
import evolving_loop.accuracy_profile as accuracy_profile_module
from evolving_loop.accuracy_profile import (
    BASELINE_PANEL,
    SOURCE_COMMIT,
    SOURCE_LOGS,
    SOURCE_SHA256,
    build_accuracy_profile,
    task_difficulty_features,
    validate_accuracy_profile,
)
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
        for index in range(13)
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
            "path": SOURCE_LOGS[model],
            "sha256": SOURCE_SHA256[model],
        }
        for model in BASELINE_PANEL
    }
    return build_accuracy_profile(
        scores,
        source_commit=SOURCE_COMMIT,
        source_files=sources,
    )


def _toto_source_artifacts() -> dict:
    return {
        "task_snapshot": {
            "kind": "labeled_task_snapshot",
            "split_manifest_sha256": "0" * 64,
            "selected_tasks_sha256": "f" * 64,
        },
        "train_dev": {
            "kind": "forecast_store",
            "manifest_sha256": "1" * 64,
            "identity_sha256": "2" * 64,
            "selected_entries_sha256": "3" * 64,
        },
        "public_test": {
            "kind": "frozen_toto_forecasts",
            "artifact_sha256": "4" * 64,
            "row_key": "E_toto_reference",
            "source_commit": "5" * 40,
            "evaluation_source_sha256": "6" * 64,
            "runtime_manifest_sha256": "7" * 64,
            "worker_adapter_sha256": "8" * 64,
            "model_checkpoint": "Datadog/Toto-2.0-22m",
        },
    }


def _toto_accuracy_profile(records: list[dict]) -> dict:
    build = getattr(accuracy_profile_module, "build_toto_accuracy_profile")
    task_ids = sorted(str(row["benchmark_id"]) for row in records)
    return build(
        {
            task_id: {
                "smae": 1.0 + index * 0.01,
                "srmse": 1.12 - index * 0.01,
            }
            for index, task_id in enumerate(task_ids)
        },
        source_artifacts=_toto_source_artifacts(),
    )


def test_toto_accuracy_profile_is_canonical_and_tamper_evident() -> None:
    """Catches Toto difficulty evidence accepting unbound or edited task metrics."""
    build = getattr(accuracy_profile_module, "build_toto_accuracy_profile", None)
    validate = getattr(accuracy_profile_module, "validate_toto_accuracy_profile", None)
    assert callable(build) and callable(validate)
    source_artifacts = _toto_source_artifacts()
    profile = build(
        {
            "task_a": {"smae": 0.25, "srmse": 0.5},
            "task_b": {"smae": 0.75, "srmse": 1.0},
        },
        source_artifacts=source_artifacts,
    )

    assert profile["profile_schema"] == "drcik-public-toto-accuracy-v1"
    assert profile["model_name"] == "toto_2_0"
    assert profile["metric_cap"] == 5.0
    assert validate(profile, {"task_a", "task_b"}) == profile

    edited = json.loads(json.dumps(profile))
    edited["tasks"]["task_a"]["smae"] = 0.125
    with pytest.raises(ValueError, match="digest mismatch"):
        validate(edited)


def test_toto_accuracy_profile_rejects_unbound_task_snapshot() -> None:
    """Catches metrics being regenerated from altered labels under unchanged forecast bindings."""
    with pytest.raises(ValueError, match="task snapshot"):
        accuracy_profile_module.build_toto_accuracy_profile(
            {"task_a": {"smae": 0.25, "srmse": 0.5}},
            source_artifacts={
                "train_dev": {
                    "kind": "forecast_store",
                    "manifest_sha256": "1" * 64,
                    "identity_sha256": "2" * 64,
                    "selected_entries_sha256": "3" * 64,
                },
                "public_test": {
                    "kind": "frozen_toto_forecasts",
                    "artifact_sha256": "4" * 64,
                    "row_key": "E_toto_reference",
                },
            },
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
        train_size=7,
        dev_size=3,
        public_test_size=3,
        trials=512,
    )
    second = build_accuracy_stratified_split_manifest(
        list(reversed(records)),
        profile,
        seed=29,
        train_size=7,
        dev_size=3,
        public_test_size=3,
        trials=512,
    )

    assert first == second
    assert first["schema_version"] == 2
    assert first["actual_sizes"] == {"train": 7, "dev": 3, "public_test": 3}
    assert first["selection_uses_history_values"] is True
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


def test_toto_balanced_split_is_exact_deterministic_and_gated() -> None:
    """Catches v3 recording Toto accuracy without actually balancing both metrics."""
    split_module = __import__("evolving_loop.split_manifest", fromlist=["x"])
    build = getattr(split_module, "build_toto_balanced_split_manifest", None)
    assert callable(build)
    records = _single_entity_records()
    baseline = _accuracy_profile(records)
    toto = _toto_accuracy_profile(records)

    first = build(
        records,
        baseline,
        toto,
        seed=29,
        train_size=7,
        dev_size=3,
        public_test_size=3,
        trials=512,
    )
    second = build(
        list(reversed(records)),
        baseline,
        toto,
        seed=29,
        train_size=7,
        dev_size=3,
        public_test_size=3,
        trials=512,
    )

    assert first == second
    assert first["schema_version"] == 3
    assert first["actual_sizes"] == {"train": 7, "dev": 3, "public_test": 3}
    assert first["toto_difficulty_profile_sha256"] == toto["profile_sha256"]
    assert first["selection_uses_history_values"] is True
    assert first["selection_uses_future_values"] is True
    assert first["selection_uses_model_metrics"] is True
    assert "toto_smae_decile" in first["stratification_features"]
    assert "toto_srmse_quintile" in first["stratification_features"]
    assert first["objective"]["baseline_relative_mean_difficulty_gap"] <= 0.05
    assert first["objective"]["toto_smae_relative_mean_gap"] <= 0.05
    assert first["objective"]["toto_srmse_relative_mean_gap"] <= 0.05
    entity_sets = [
        set(first["partitions"][name]["entities"])
        for name in ("train", "dev", "public_test")
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(entity_sets)
        for right in entity_sets[index + 1 :]
    )


def test_toto_balanced_split_rejects_when_toto_gate_is_impossible() -> None:
    """Catches Toto's 5% gate degrading into a non-binding ranking signal."""
    split_module = __import__("evolving_loop.split_manifest", fromlist=["x"])
    build = getattr(split_module, "build_toto_balanced_split_manifest")
    records = _single_entity_records()[:3]
    baseline = _accuracy_profile(records)
    toto = _toto_accuracy_profile(records)
    raw = {
        task_id: {"smae": value, "srmse": value}
        for task_id, value in zip(sorted(toto["tasks"]), (0.1, 0.1, 5.0))
    }
    toto = accuracy_profile_module.build_toto_accuracy_profile(
        raw, source_artifacts=toto["source_artifacts"]
    )

    with pytest.raises(ValueError, match="baseline/Toto"):
        build(
            records,
            baseline,
            toto,
            seed=29,
            train_size=1,
            dev_size=1,
            public_test_size=1,
            trials=32,
        )


def test_accuracy_stratified_split_balances_controlled_difficulty() -> None:
    """Catches an objective that records difficulty fields without balancing them."""
    records = _single_entity_records()
    manifest = build_accuracy_stratified_split_manifest(
        records,
        _accuracy_profile(records),
        seed=29,
        train_size=7,
        dev_size=3,
        public_test_size=3,
        trials=512,
    )

    means = [
        manifest["partitions"][name]["difficulty"]["mean_score"]
        for name in ("train", "dev", "public_test")
    ]
    assert max(means) - min(means) <= 0.025
    assert manifest["objective"]["relative_mean_difficulty_gap"] <= 0.05

    profile = _accuracy_profile(records)
    dev_scores = [
        profile["tasks"][task_id]["smae"][BASELINE_PANEL[0]]
        for task_id in manifest["partitions"]["dev"]["task_ids"]
    ]
    assert manifest["partitions"]["dev"]["difficulty"]["p90_score"] == pytest.approx(
        linear_quantile(dev_scores, 0.9)
    )


def test_accuracy_stratified_split_rejects_wrong_task_coverage() -> None:
    """Catches a partial accuracy ledger silently assigning unprofiled public tasks."""
    records = _single_entity_records()
    profile = _accuracy_profile(records[:-1])

    with pytest.raises(ValueError, match="task coverage"):
        build_accuracy_stratified_split_manifest(
            records,
            profile,
            seed=29,
            train_size=7,
            dev_size=3,
            public_test_size=3,
            trials=32,
        )


def test_accuracy_stratified_split_rejects_when_no_candidate_passes_gate() -> None:
    """Catches the documented 5% difficulty gate degrading into a ranking hint."""
    records = _single_entity_records()[:3]

    with pytest.raises(ValueError, match="best achieved"):
        build_accuracy_stratified_split_manifest(
            records,
            _accuracy_profile(records),
            seed=29,
            train_size=1,
            dev_size=1,
            public_test_size=1,
            trials=32,
        )


def test_committed_v2_retains_effective_entity_diversity() -> None:
    """Catches accuracy balancing collapsing Dev onto too few repeated entities."""
    path = Path(__file__).parents[1] / "splits" / "drcik_public_80_20_99_v2.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    for name in ("train", "dev", "public_test"):
        task_count = manifest["actual_sizes"][name]
        entity_count = len(manifest["partitions"][name]["entities"])
        assert entity_count >= (task_count + 1) // 2


def test_committed_v2_artifacts_are_bound_complete_and_improve_v1() -> None:
    """Catches stale, partial, or hand-edited committed split evidence."""
    root = Path(__file__).parents[1]
    profile = json.loads(
        (root / "splits" / "drcik_public_baseline_accuracy_v1.json").read_text(
            encoding="utf-8"
        )
    )
    v1_path = root / "splits" / "drcik_public_80_20_99_v1.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(
        (root / "splits" / "drcik_public_80_20_99_v2.json").read_text(
            encoding="utf-8"
        )
    )

    validated = validate_accuracy_profile(profile)
    assert v1["manifest_sha256"] == (
        "3cc81f45878c1aae93e5ba48dc367df6553698db6661dbe06fbe5efb06afca92"
    )
    unsigned_v1 = dict(v1)
    submitted_v1_digest = unsigned_v1.pop("manifest_sha256")
    canonical_v1 = json.dumps(unsigned_v1, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical_v1.encode()).hexdigest() == submitted_v1_digest
    unsigned = dict(v2)
    submitted_digest = unsigned.pop("manifest_sha256")
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == submitted_digest
    assert v2["difficulty_profile_sha256"] == validated["profile_sha256"]
    assert v2["difficulty_source_commit"] == SOURCE_COMMIT
    assert v2["difficulty_panel_models"] == list(BASELINE_PANEL)
    assert v2["selection_uses_history_values"] is True
    assert v2["selection_uses_future_values"] is True
    assert v2["actual_sizes"] == {"train": 80, "dev": 20, "public_test": 99}

    profile_ids = set(validated["tasks"])
    v1_ids = {
        task_id
        for name in ("train", "dev", "public_test")
        for task_id in v1["partitions"][name]["task_ids"]
    }
    v2_ids = {
        task_id
        for name in ("train", "dev", "public_test")
        for task_id in v2["partitions"][name]["task_ids"]
    }
    assert len(profile_ids) == 199
    assert v1_ids == v2_ids == profile_ids
    v2_task_sets = [
        set(v2["partitions"][name]["task_ids"])
        for name in ("train", "dev", "public_test")
    ]
    v2_entity_sets = [
        set(v2["partitions"][name]["entities"])
        for name in ("train", "dev", "public_test")
    ]
    assert [len(items) for items in v2_task_sets] == [80, 20, 99]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(v2_task_sets)
        for right in v2_task_sets[index + 1 :]
    )
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(v2_entity_sets)
        for right in v2_entity_sets[index + 1 :]
    )

    difficulty = task_difficulty_features(validated)

    def relative_gap(manifest: dict) -> float:
        means = []
        for name in ("train", "dev", "public_test"):
            task_ids = manifest["partitions"][name]["task_ids"]
            scores = [float(difficulty[task_id]["difficulty_score"]) for task_id in task_ids]
            means.append(statistics.fmean(scores))
            if manifest["schema_version"] == 2:
                summary = manifest["partitions"][name]["difficulty"]
                assert summary["mean_score"] == pytest.approx(statistics.fmean(scores))
                assert summary["p90_score"] == pytest.approx(linear_quantile(scores, 0.9))
        overall = statistics.fmean(
            float(item["difficulty_score"]) for item in difficulty.values()
        )
        return (max(means) - min(means)) / overall

    v1_gap = relative_gap(v1)
    v2_gap = relative_gap(v2)
    assert v2_gap == pytest.approx(v2["objective"]["relative_mean_difficulty_gap"])
    assert v2_gap <= 0.05
    assert v2_gap < v1_gap


def test_committed_v3_is_toto_bound_complete_and_within_all_gates() -> None:
    """Catches stale or hand-edited v3 evidence and any difficulty-gate regression."""
    root = Path(__file__).parents[1]
    validate_toto = getattr(accuracy_profile_module, "validate_toto_accuracy_profile")
    toto = validate_toto(
        json.loads(
            (root / "splits" / "drcik_public_toto_accuracy_v1.json").read_text(
                encoding="utf-8"
            )
        )
    )
    baseline = validate_accuracy_profile(
        json.loads(
            (root / "splits" / "drcik_public_baseline_accuracy_v1.json").read_text(
                encoding="utf-8"
            )
        )
    )
    v3 = json.loads(
        (root / "splits" / "drcik_public_80_20_99_v3.json").read_text(
            encoding="utf-8"
        )
    )
    unsigned = dict(v3)
    submitted_digest = unsigned.pop("manifest_sha256")
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))

    assert submitted_digest == (
        "5f6ddec2ae460b292629f78e76784db1d58da79df1fcbbe993b5e9aa1d93b835"
    )
    assert hashlib.sha256(canonical.encode()).hexdigest() == submitted_digest
    assert v3["schema_version"] == 3
    assert v3["target_sizes"] == {"train": 80, "dev": 20, "public_test": 99}
    assert v3["actual_sizes"] == {"train": 80, "dev": 20, "public_test": 99}
    assert v3["assignment_trials"] == 32768
    assert v3["difficulty_profile_sha256"] == baseline["profile_sha256"]
    assert v3["difficulty_source_commit"] == SOURCE_COMMIT
    assert v3["toto_difficulty_profile_sha256"] == toto["profile_sha256"]
    assert toto["source_artifacts"]["task_snapshot"] == {
        "kind": "labeled_task_snapshot",
        "split_manifest_sha256": (
            "d66afc4857613d7ec66d4692eefef34fe61212dddf0041e663f8434116a681a9"
        ),
        "selected_tasks_sha256": (
            "b47d1bbf19e9dcbecf527c6555ec2d99675a37a01f3d1bc845f0778a0b508d43"
        ),
    }
    assert toto["source_artifacts"]["public_test"]["source_commit"] == (
        "728ea88493d66c54cca413fe4285f34923eabffe"
    )
    task_sets = [
        set(v3["partitions"][name]["task_ids"])
        for name in ("train", "dev", "public_test")
    ]
    assert [len(task_ids) for task_ids in task_sets] == [80, 20, 99]
    assert set.union(*task_sets) == set(toto["tasks"]) == set(baseline["tasks"])
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(task_sets)
        for right in task_sets[index + 1 :]
    )
    assert all(
        v3["objective"][key] <= 0.05
        for key in (
            "baseline_relative_mean_difficulty_gap",
            "toto_smae_relative_mean_gap",
            "toto_srmse_relative_mean_gap",
        )
    )
    entity_sets = [
        set(v3["partitions"][name]["entities"])
        for name in ("train", "dev", "public_test")
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(entity_sets)
        for right in entity_sets[index + 1 :]
    )
    baseline_difficulty = task_difficulty_features(baseline)
    overall_means = {
        "baseline": statistics.fmean(
            float(row["difficulty_score"]) for row in baseline_difficulty.values()
        ),
        "smae": statistics.fmean(float(row["smae"]) for row in toto["tasks"].values()),
        "srmse": statistics.fmean(float(row["srmse"]) for row in toto["tasks"].values()),
    }
    partition_means = {metric: [] for metric in overall_means}
    for name in ("train", "dev", "public_test"):
        task_ids = v3["partitions"][name]["task_ids"]
        baseline_values = [
            float(baseline_difficulty[task_id]["difficulty_score"])
            for task_id in task_ids
        ]
        summary = v3["partitions"][name]["difficulty"]
        assert summary["mean_score"] == pytest.approx(statistics.fmean(baseline_values))
        assert summary["median_score"] == pytest.approx(statistics.median(baseline_values))
        assert summary["p90_score"] == pytest.approx(linear_quantile(baseline_values, 0.9))
        assert summary["max_score"] == pytest.approx(max(baseline_values))
        partition_means["baseline"].append(statistics.fmean(baseline_values))
        for metric in ("smae", "srmse"):
            values = [float(toto["tasks"][task_id][metric]) for task_id in task_ids]
            metric_summary = v3["partitions"][name]["toto_difficulty"][metric]
            assert metric_summary["mean"] == pytest.approx(statistics.fmean(values))
            assert metric_summary["median"] == pytest.approx(statistics.median(values))
            assert metric_summary["p90"] == pytest.approx(linear_quantile(values, 0.9))
            assert metric_summary["max"] == pytest.approx(max(values))
            partition_means[metric].append(statistics.fmean(values))
    expected_gaps = {
        metric: (max(means) - min(means)) / overall_means[metric]
        for metric, means in partition_means.items()
    }
    assert v3["objective"]["baseline_relative_mean_difficulty_gap"] == pytest.approx(
        expected_gaps["baseline"]
    )
    assert v3["objective"]["toto_smae_relative_mean_gap"] == pytest.approx(
        expected_gaps["smae"]
    )
    assert v3["objective"]["toto_srmse_relative_mean_gap"] == pytest.approx(
        expected_gaps["srmse"]
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
