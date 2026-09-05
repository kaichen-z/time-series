"""Build a frozen Toto difficulty profile from existing, label-backed forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Sequence

from common.data import load_tasks_by_id
from common.metrics import drcik_point_metrics
from numerical_agent.evolution.forecast_store import ForecastStore
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.providers import RuntimeRegistry
from numerical_agent.run_champion_evolution import _load_screening_policy

from .accuracy_profile import build_toto_accuracy_profile
from .split_manifest import write_split_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_blob(repo: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _partition_ids(manifest: dict, name: str) -> list[str]:
    return [str(task_id) for task_id in manifest["partitions"][name]["task_ids"]]


def build_from_frozen_evidence(
    *,
    tasks_path: Path,
    split_manifest_path: Path,
    forecast_store_path: Path,
    numerical_repo: Path,
    run_manifest_path: Path,
    workers_config_path: Path,
    public_results_path: Path,
    public_source_repo: Path,
    public_source_commit: str,
    expected_store_identity: str | None = None,
) -> dict:
    """Recompute Toto sMAE/sRMSE while binding every selected forecast artifact."""
    split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    train_dev_ids = _partition_ids(split, "train") + _partition_ids(split, "dev")
    public_ids = _partition_ids(split, "public_test")
    all_ids = train_dev_ids + public_ids
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("source split contains duplicate task IDs")
    tasks = {task.task_id: task for task in load_tasks_by_id(tasks_path, all_ids)}
    if set(tasks) != set(all_ids):
        raise ValueError("source tasks do not cover the frozen split")
    task_snapshot = {
        task_id: {
            "history_values": list(tasks[task_id].history_values),
            "future_values": list(tasks[task_id].future_values),
            "prediction_length": tasks[task_id].prediction_length,
            "frequency": tasks[task_id].frequency,
            "seasonal_period": tasks[task_id].seasonal_period,
            "entity_name": tasks[task_id].entity_name,
        }
        for task_id in sorted(tasks)
    }

    portfolio = read_policy_file(numerical_repo / "policies.py")
    screening = _load_screening_policy(numerical_repo / "dictionary.py")
    runtime_identity = {
        "tsfm_runtimes": "timesfm",
        "chronos_device_map": "cpu",
        "model_cache_dir": "outputs/model-cache",
        "deployment_sha256": _sha256(workers_config_path),
        "acknowledged_model_licenses": "CC-BY-NC-4.0",
    }
    runtimes = RuntimeRegistry()
    store = ForecastStore(
        forecast_store_path,
        numerical_repo / "methods.py",
        numerical_repo / "skills.py",
        portfolio,
        runtimes,
        screening_hash=screening.fingerprint(),
        runtime_identity=runtime_identity,
        cache_only=True,
    )
    metrics = {}
    selected_entries = {}
    try:
        if expected_store_identity and store.identity_hash != expected_store_identity:
            raise ValueError(
                "forecast-store identity mismatch: "
                f"expected {expected_store_identity}, got {store.identity_hash}"
            )
        for task_id in train_dev_ids:
            task = tasks[task_id]
            forecast = store.forecast(
                "toto_2_0",
                task.history_values,
                task.prediction_length,
                task.frequency,
            )
            result = drcik_point_metrics(task.future_values, forecast)
            metrics[task_id] = {
                "smae": float(result["smae"]),
                "srmse": float(result["srmse"]),
            }
            key = store._key(
                "toto_2_0",
                task.history_values,
                task.prediction_length,
                task.frequency,
            )
            selected_entries[task_id] = {
                "cache_key": key,
                "sha256": _sha256(forecast_store_path / f"{key}.json"),
            }
    finally:
        store.close()
        runtimes.close()

    seen_public = set()
    for line in public_results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = str(row["task_id"])
        if task_id not in set(public_ids):
            continue
        if task_id in seen_public:
            raise ValueError(f"duplicate public Toto forecast for {task_id}")
        toto = row["rows"]["E_toto_reference"]
        if toto["selected"] != ["toto_2_0"] or toto["task_id"] != task_id:
            raise ValueError(f"public Toto row is malformed for {task_id}")
        result = drcik_point_metrics(tasks[task_id].future_values, toto["forecast"])
        metrics[task_id] = {
            "smae": float(result["smae"]),
            "srmse": float(result["srmse"]),
        }
        seen_public.add(task_id)
    if seen_public != set(public_ids):
        raise ValueError("public Toto evidence does not cover the frozen Public Test")

    evaluation_source = _git_blob(
        public_source_repo,
        public_source_commit,
        "numerical_agent/evaluate_frozen_two_stage.py",
    )
    runtime_manifest = _git_blob(
        public_source_repo,
        public_source_commit,
        "numerical_agent/tsfm/runtime_manifests.json",
    )
    worker_adapter = _git_blob(
        public_source_repo,
        public_source_commit,
        "numerical_agent/tsfm/workers/dedicated.py",
    )
    manifest_rows = json.loads(runtime_manifest)
    toto_rows = [
        row
        for row in manifest_rows
        if row.get("checkpoint") == "Datadog/Toto-2.0-22m"
    ]
    if len(toto_rows) != 1 or toto_rows[0].get("point_reduction") != "median":
        raise ValueError("public source revision does not identify the expected Toto runtime")

    return build_toto_accuracy_profile(
        metrics,
        source_artifacts={
            "task_snapshot": {
                "kind": "labeled_task_snapshot",
                "split_manifest_sha256": _sha256(split_manifest_path),
                "selected_tasks_sha256": _canonical_digest(task_snapshot),
            },
            "train_dev": {
                "kind": "forecast_store",
                "manifest_sha256": _sha256(run_manifest_path),
                "identity_sha256": store.identity_hash,
                "selected_entries_sha256": _canonical_digest(selected_entries),
            },
            "public_test": {
                "kind": "frozen_toto_forecasts",
                "artifact_sha256": _sha256(public_results_path),
                "row_key": "E_toto_reference",
                "source_commit": public_source_commit,
                "evaluation_source_sha256": hashlib.sha256(
                    evaluation_source
                ).hexdigest(),
                "runtime_manifest_sha256": hashlib.sha256(runtime_manifest).hexdigest(),
                "worker_adapter_sha256": hashlib.sha256(worker_adapter).hexdigest(),
                "model_checkpoint": "Datadog/Toto-2.0-22m",
            },
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-path", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--forecast-store", required=True, type=Path)
    parser.add_argument("--numerical-repo", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--workers-config", required=True, type=Path)
    parser.add_argument("--public-results", required=True, type=Path)
    parser.add_argument("--public-source-repo", required=True, type=Path)
    parser.add_argument("--public-source-commit", required=True)
    parser.add_argument("--expected-store-identity")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    profile = build_from_frozen_evidence(
        tasks_path=args.tasks_path,
        split_manifest_path=args.split_manifest,
        forecast_store_path=args.forecast_store,
        numerical_repo=args.numerical_repo,
        run_manifest_path=args.run_manifest,
        workers_config_path=args.workers_config,
        public_results_path=args.public_results,
        public_source_repo=args.public_source_repo,
        public_source_commit=args.public_source_commit,
        expected_store_identity=args.expected_store_identity,
    )
    path = write_split_manifest(profile, args.output)
    print(path)
    print(json.dumps({"task_count": profile["task_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
