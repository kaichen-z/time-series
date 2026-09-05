"""Build and validate a frozen per-task Dr-CiK baseline-accuracy profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT


BASELINE_PANEL = (
    "arima",
    "ets",
    "ses",
    "chronos",
    "aurora",
    "moirai",
    "seasonal_naive",
)
PROFILE_SCHEMA = "drcik-public-baseline-accuracy-v1"
TOTO_PROFILE_SCHEMA = "drcik-public-toto-accuracy-v1"
SOURCE_COMMIT = "1d0d9690e6d81fd00d344700216cfd40f35638f5"
SOURCE_LOGS = {model: f"runs/baselines/{model}_dev.log" for model in BASELINE_PANEL}
SOURCE_SHA256 = {
    "arima": "6fbe90f0d35c03508b9a621d4d610ce20facd5317ca23eca3ae973d29b4af242",
    "ets": "4c0df11ee45209001537bf94399c93ba3c2089515b5f7c45601bc1c82d318b55",
    "ses": "b76c5ec656d9c525efb134be4d1c340da604412d6eaad967365c4e4353f3aea7",
    "chronos": "676677765977d0a839fc190fae880055c03b429838384033b006e4a5b66eceeb",
    "aurora": "d073890b63eceaa9ceb28166f8526b43750dc935cc371d3570857b3d8974774c",
    "moirai": "4336575b6638ff6439c91aec5805c66fe296c9cf55577a251985f5efa7c45993",
    "seasonal_naive": "54465c4494323f97963f3d289fb57c99fe0a9396f6d0a278993f36a789f3cb39",
}
_SCORE_PATTERN = re.compile(
    r"\b(task_\d+)\s+H=\d+\s+paths=\d+/\d+\s+sMAE=([^\s]+)"
)
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_baseline_log(text: str) -> dict[str, float]:
    """Extract exactly one finite capped sMAE for every task mentioned in a log."""
    scores: dict[str, float] = {}
    for match in _SCORE_PATTERN.finditer(text):
        task_id, raw_score = match.groups()
        if task_id in scores:
            raise ValueError(f"duplicate task score for {task_id}")
        try:
            score = float(raw_score)
        except ValueError as error:
            raise ValueError(f"sMAE for {task_id} must be a number") from error
        if not math.isfinite(score):
            raise ValueError(f"sMAE for {task_id} must be finite")
        if not 0.0 <= score <= 5.0:
            raise ValueError(f"sMAE for {task_id} must be in [0, 5]")
        scores[task_id] = score
    if not scores:
        raise ValueError("baseline log contains no task scores")
    return dict(sorted(scores.items()))


def _validated_scores(
    scores_by_model: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if set(scores_by_model) != set(BASELINE_PANEL):
        raise ValueError("accuracy profile must contain the exact baseline panel")
    expected_tasks: set[str] | None = None
    normalized: dict[str, dict[str, float]] = {}
    for model in BASELINE_PANEL:
        raw_scores = scores_by_model[model]
        if not isinstance(raw_scores, Mapping) or not raw_scores:
            raise ValueError(f"scores for {model} must be a non-empty mapping")
        task_ids = set(raw_scores)
        if expected_tasks is None:
            expected_tasks = task_ids
        elif task_ids != expected_tasks:
            raise ValueError(f"task coverage for {model} does not match the panel")
        model_scores: dict[str, float] = {}
        for task_id, raw_score in raw_scores.items():
            if type(task_id) is not str or not task_id:
                raise ValueError("task IDs must be non-empty strings")
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ValueError(f"sMAE for {task_id}/{model} must be a number")
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError(f"sMAE for {task_id}/{model} must be finite")
            if not 0.0 <= score <= 5.0:
                raise ValueError(f"sMAE for {task_id}/{model} must be in [0, 5]")
            model_scores[task_id] = score
        normalized[model] = dict(sorted(model_scores.items()))
    return normalized


def _validated_sources(
    source_files: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    if set(source_files) != set(BASELINE_PANEL):
        raise ValueError("source files must contain the exact baseline panel")
    normalized = {}
    for model in BASELINE_PANEL:
        source = source_files[model]
        if set(source) != {"path", "sha256"}:
            raise ValueError(f"source file for {model} needs path and sha256")
        path = source["path"]
        digest = source["sha256"]
        if type(path) is not str or not path:
            raise ValueError(f"source path for {model} must be a non-empty string")
        if type(digest) is not str or _HEX_64.fullmatch(digest) is None:
            raise ValueError(f"source sha256 for {model} must be 64 lowercase hex characters")
        normalized[model] = {"path": path, "sha256": digest}
    return normalized


def _pinned_sources() -> dict[str, dict[str, str]]:
    return {
        model: {"path": SOURCE_LOGS[model], "sha256": SOURCE_SHA256[model]}
        for model in BASELINE_PANEL
    }


def _validated_toto_source_artifacts(
    source_artifacts: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    if set(source_artifacts) != {"task_snapshot", "train_dev", "public_test"}:
        raise ValueError(
            "Toto source artifacts must bind the task snapshot, Train/Dev, and Public Test"
        )
    task_snapshot = source_artifacts["task_snapshot"]
    train_dev = source_artifacts["train_dev"]
    public_test = source_artifacts["public_test"]
    if not isinstance(task_snapshot, Mapping) or set(task_snapshot) != {
        "kind",
        "split_manifest_sha256",
        "selected_tasks_sha256",
    }:
        raise ValueError("Toto task snapshot source binding is malformed")
    if task_snapshot["kind"] != "labeled_task_snapshot":
        raise ValueError("Toto task snapshot source must contain labeled numeric tasks")
    if not isinstance(train_dev, Mapping) or set(train_dev) != {
        "kind",
        "manifest_sha256",
        "identity_sha256",
        "selected_entries_sha256",
    }:
        raise ValueError("Toto Train/Dev source binding is malformed")
    if train_dev["kind"] != "forecast_store":
        raise ValueError("Toto Train/Dev source must be a forecast store")
    if not isinstance(public_test, Mapping) or set(public_test) != {
        "kind",
        "artifact_sha256",
        "row_key",
        "source_commit",
        "evaluation_source_sha256",
        "runtime_manifest_sha256",
        "worker_adapter_sha256",
        "model_checkpoint",
    }:
        raise ValueError("Toto Public Test source binding is malformed")
    if public_test["kind"] != "frozen_toto_forecasts":
        raise ValueError("Toto Public Test source must contain frozen forecasts")
    if public_test["row_key"] != "E_toto_reference":
        raise ValueError("Toto Public Test source row is not the frozen reference")
    if public_test["model_checkpoint"] != "Datadog/Toto-2.0-22m":
        raise ValueError("Toto Public Test checkpoint identity mismatch")
    for field in ("split_manifest_sha256", "selected_tasks_sha256"):
        value = task_snapshot[field]
        if type(value) is not str or _HEX_64.fullmatch(value) is None:
            raise ValueError(f"Toto task snapshot {field} must be 64 lowercase hex characters")
    for field in ("manifest_sha256", "identity_sha256", "selected_entries_sha256"):
        value = train_dev[field]
        if type(value) is not str or _HEX_64.fullmatch(value) is None:
            raise ValueError(f"Toto Train/Dev {field} must be 64 lowercase hex characters")
    source_commit = public_test["source_commit"]
    if type(source_commit) is not str or _HEX_40.fullmatch(source_commit) is None:
        raise ValueError("Toto Public Test source commit must be 40 lowercase hex characters")
    for field in (
        "artifact_sha256",
        "evaluation_source_sha256",
        "runtime_manifest_sha256",
        "worker_adapter_sha256",
    ):
        value = public_test[field]
        if type(value) is not str or _HEX_64.fullmatch(value) is None:
            raise ValueError(f"Toto Public Test {field} must be 64 lowercase hex characters")
    return {
        "task_snapshot": {str(key): str(value) for key, value in task_snapshot.items()},
        "train_dev": {str(key): str(value) for key, value in train_dev.items()},
        "public_test": {str(key): str(value) for key, value in public_test.items()},
    }


def build_toto_accuracy_profile(
    task_metrics: Mapping[str, Mapping[str, float]],
    *,
    source_artifacts: Mapping[str, object],
) -> dict:
    """Create a canonical per-task Toto sMAE/sRMSE difficulty artifact."""
    if not isinstance(task_metrics, Mapping) or not task_metrics:
        raise ValueError("Toto accuracy profile requires task metrics")
    normalized = {}
    for task_id, raw_metrics in task_metrics.items():
        if type(task_id) is not str or not task_id:
            raise ValueError("Toto task IDs must be non-empty strings")
        if not isinstance(raw_metrics, Mapping) or set(raw_metrics) != {"smae", "srmse"}:
            raise ValueError(f"Toto task {task_id} must contain sMAE and sRMSE")
        metrics = {}
        for metric in ("smae", "srmse"):
            raw_value = raw_metrics[metric]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"Toto {metric} for {task_id} must be a number")
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 <= value <= 5.0:
                raise ValueError(f"Toto {metric} for {task_id} must be in [0, 5]")
            metrics[metric] = value
        normalized[task_id] = metrics
    payload = {
        "schema_version": 1,
        "profile_schema": TOTO_PROFILE_SCHEMA,
        "dataset": "ServiceNow/Dr-CiK",
        "source_split": "public_dev",
        "model_name": "toto_2_0",
        "metric_cap": 5.0,
        "metric_policy_fingerprint": METRIC_POLICY_FINGERPRINT,
        "source_artifacts": _validated_toto_source_artifacts(source_artifacts),
        "task_count": len(normalized),
        "tasks": dict(sorted(normalized.items())),
    }
    payload["profile_sha256"] = _canonical_digest(payload)
    return payload


def validate_toto_accuracy_profile(
    profile: Mapping[str, object],
    expected_task_ids: Collection[str] | None = None,
) -> dict:
    """Fail closed unless Toto task metrics are complete and provenance-bound."""
    expected_fields = {
        "schema_version",
        "profile_schema",
        "dataset",
        "source_split",
        "model_name",
        "metric_cap",
        "metric_policy_fingerprint",
        "source_artifacts",
        "task_count",
        "tasks",
        "profile_sha256",
    }
    if set(profile) != expected_fields:
        raise ValueError("Toto accuracy profile fields do not match schema")
    submitted_digest = profile["profile_sha256"]
    if type(submitted_digest) is not str or _HEX_64.fullmatch(submitted_digest) is None:
        raise ValueError("Toto accuracy profile digest must be 64 lowercase hex characters")
    unsigned = dict(profile)
    unsigned.pop("profile_sha256")
    if _canonical_digest(unsigned) != submitted_digest:
        raise ValueError("Toto accuracy profile digest mismatch")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        raise ValueError("unsupported Toto accuracy profile schema")
    if profile["profile_schema"] != TOTO_PROFILE_SCHEMA:
        raise ValueError("unsupported Toto accuracy profile schema")
    if profile["dataset"] != "ServiceNow/Dr-CiK" or profile["source_split"] != "public_dev":
        raise ValueError("Toto accuracy profile dataset identity mismatch")
    if profile["model_name"] != "toto_2_0" or profile["metric_cap"] != 5.0:
        raise ValueError("Toto accuracy profile model or metric cap mismatch")
    if profile["metric_policy_fingerprint"] != METRIC_POLICY_FINGERPRINT:
        raise ValueError("Toto accuracy profile metric policy mismatch")
    raw_tasks = profile["tasks"]
    if not isinstance(raw_tasks, Mapping) or not raw_tasks:
        raise ValueError("Toto accuracy profile tasks must be a non-empty mapping")
    source_artifacts = profile["source_artifacts"]
    if not isinstance(source_artifacts, Mapping):
        raise ValueError("Toto accuracy profile source artifacts must be a mapping")
    rebuilt = build_toto_accuracy_profile(
        raw_tasks,
        source_artifacts=source_artifacts,
    )
    if type(profile["task_count"]) is not int or profile["task_count"] != len(raw_tasks):
        raise ValueError("Toto accuracy profile task count mismatch")
    if expected_task_ids is not None and set(raw_tasks) != set(expected_task_ids):
        raise ValueError("Toto accuracy profile task coverage mismatch")
    if dict(profile) != rebuilt:
        raise ValueError("Toto accuracy profile is not canonical")
    return rebuilt


def build_accuracy_profile(
    scores_by_model: Mapping[str, Mapping[str, float]],
    *,
    source_commit: str,
    source_files: Mapping[str, Mapping[str, str]],
) -> dict:
    """Create a canonical offline evidence artifact from the fixed baseline panel."""
    if type(source_commit) is not str or _HEX_40.fullmatch(source_commit) is None:
        raise ValueError("source commit must be 40 lowercase hex characters")
    if source_commit != SOURCE_COMMIT:
        raise ValueError("accuracy profile must use the pinned source commit")
    scores = _validated_scores(scores_by_model)
    sources = _validated_sources(source_files)
    if sources != _pinned_sources():
        raise ValueError("accuracy profile source files do not match pinned paths and digests")
    task_ids = sorted(scores[BASELINE_PANEL[0]])
    payload = {
        "schema_version": 1,
        "profile_schema": PROFILE_SCHEMA,
        "dataset": "ServiceNow/Dr-CiK",
        "source_split": "public_dev",
        "source_commit": source_commit,
        "panel_models": list(BASELINE_PANEL),
        "source_files": sources,
        "task_count": len(task_ids),
        "tasks": {
            task_id: {
                "smae": {model: scores[model][task_id] for model in BASELINE_PANEL}
            }
            for task_id in task_ids
        },
    }
    payload["profile_sha256"] = _canonical_digest(payload)
    return payload


def validate_accuracy_profile(
    profile: Mapping[str, object],
    expected_task_ids: Collection[str] | None = None,
) -> dict:
    """Fail closed unless a profile is canonical, complete, and provenance-bound."""
    expected_fields = {
        "schema_version",
        "profile_schema",
        "dataset",
        "source_split",
        "source_commit",
        "panel_models",
        "source_files",
        "task_count",
        "tasks",
        "profile_sha256",
    }
    if set(profile) != expected_fields:
        raise ValueError("accuracy profile fields do not match schema")
    submitted_digest = profile["profile_sha256"]
    if type(submitted_digest) is not str or _HEX_64.fullmatch(submitted_digest) is None:
        raise ValueError("accuracy profile digest must be 64 lowercase hex characters")
    unsigned = dict(profile)
    unsigned.pop("profile_sha256")
    if _canonical_digest(unsigned) != submitted_digest:
        raise ValueError("accuracy profile digest mismatch")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        raise ValueError("unsupported accuracy profile schema")
    if type(profile["task_count"]) is not int or profile["task_count"] <= 0:
        raise ValueError("accuracy profile task_count must be a positive integer")
    if profile["profile_schema"] != PROFILE_SCHEMA:
        raise ValueError("unsupported accuracy profile schema")
    if profile["dataset"] != "ServiceNow/Dr-CiK" or profile["source_split"] != "public_dev":
        raise ValueError("accuracy profile dataset identity mismatch")
    if profile["panel_models"] != list(BASELINE_PANEL):
        raise ValueError("accuracy profile panel order mismatch")
    raw_tasks = profile["tasks"]
    if not isinstance(raw_tasks, Mapping) or not raw_tasks:
        raise ValueError("accuracy profile tasks must be a non-empty mapping")
    scores_by_model: dict[str, dict[str, float]] = {model: {} for model in BASELINE_PANEL}
    for task_id, raw_task in raw_tasks.items():
        if type(task_id) is not str or not task_id:
            raise ValueError("task IDs must be non-empty strings")
        if not isinstance(raw_task, Mapping) or set(raw_task) != {"smae"}:
            raise ValueError(f"task {task_id} must contain only smae")
        raw_smae = raw_task["smae"]
        if not isinstance(raw_smae, Mapping) or set(raw_smae) != set(BASELINE_PANEL):
            raise ValueError(f"task {task_id} must contain the exact baseline panel")
        for model in BASELINE_PANEL:
            scores_by_model[model][task_id] = raw_smae[model]
    normalized_scores = _validated_scores(scores_by_model)
    source_commit = profile["source_commit"]
    source_files = profile["source_files"]
    if not isinstance(source_files, Mapping):
        raise ValueError("source_files must be a mapping")
    rebuilt = build_accuracy_profile(
        normalized_scores,
        source_commit=source_commit,
        source_files=source_files,
    )
    if profile["task_count"] != len(raw_tasks):
        raise ValueError("accuracy profile task count mismatch")
    if expected_task_ids is not None and set(raw_tasks) != set(expected_task_ids):
        raise ValueError("accuracy profile task coverage mismatch")
    if profile["profile_sha256"] != rebuilt["profile_sha256"]:
        raise ValueError("accuracy profile digest mismatch")
    if dict(profile) != rebuilt:
        raise ValueError("accuracy profile is not canonical")
    return rebuilt


def task_difficulty_features(profile: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Return normalized aggregate and per-model difficulty strata for every task."""
    validated = validate_accuracy_profile(profile)
    task_ids = sorted(validated["tasks"])
    denominator = max(1, len(task_ids) - 1)
    percentiles: dict[str, dict[str, float]] = {task_id: {} for task_id in task_ids}
    for model in BASELINE_PANEL:
        ordered = sorted(
            task_ids,
            key=lambda task_id: (validated["tasks"][task_id]["smae"][model], task_id),
        )
        for rank, task_id in enumerate(ordered):
            percentiles[task_id][model] = rank / denominator
    result = {}
    for task_id in task_ids:
        aggregate_percentile = statistics.median(percentiles[task_id].values())
        aggregate_smae = statistics.median(
            float(validated["tasks"][task_id]["smae"][model])
            for model in BASELINE_PANEL
        )
        result[task_id] = {
            "difficulty_score": aggregate_smae,
            "difficulty_percentile_score": aggregate_percentile,
            "difficulty_decile": str(min(9, int(aggregate_percentile * 10.0))),
            "model_percentiles": dict(percentiles[task_id]),
            "model_quintiles": {
                model: str(min(4, int(percentiles[task_id][model] * 5.0)))
                for model in BASELINE_PANEL
            },
        }
    return result


def task_toto_difficulty_features(
    profile: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Return independent Toto sMAE and sRMSE strata for every public task."""
    validated = validate_toto_accuracy_profile(profile)
    task_ids = sorted(validated["tasks"])
    denominator = max(1, len(task_ids) - 1)
    percentiles: dict[str, dict[str, float]] = {task_id: {} for task_id in task_ids}
    for metric in ("smae", "srmse"):
        ordered = sorted(
            task_ids,
            key=lambda task_id: (validated["tasks"][task_id][metric], task_id),
        )
        percentile_by_value = {}
        for rank, task_id in enumerate(ordered):
            value = float(validated["tasks"][task_id][metric])
            percentile_by_value.setdefault(value, rank / denominator)
            percentiles[task_id][metric] = percentile_by_value[value]
    result = {}
    for task_id in task_ids:
        item = {}
        for metric in ("smae", "srmse"):
            percentile = percentiles[task_id][metric]
            item[metric] = float(validated["tasks"][task_id][metric])
            item[f"{metric}_percentile"] = percentile
            item[f"{metric}_decile"] = str(min(9, int(percentile * 10.0)))
            item[f"{metric}_quintile"] = str(min(4, int(percentile * 5.0)))
        result[task_id] = item
    return result


def _load_public_task_ids(tasks_path: str | Path) -> list[str]:
    from .split_manifest import load_public_records

    return sorted(str(row["benchmark_id"]) for row in load_public_records(tasks_path))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines-dir", required=True)
    parser.add_argument("--tasks-path", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    baseline_root = Path(args.baselines_dir)
    scores_by_model = {}
    source_files = {}
    for model, relative_path in SOURCE_LOGS.items():
        path = baseline_root / Path(relative_path).name
        data = path.read_bytes()
        scores_by_model[model] = parse_baseline_log(data.decode("utf-8", "replace"))
        source_files[model] = {
            "path": relative_path,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    task_ids = _load_public_task_ids(args.tasks_path)
    profile = build_accuracy_profile(
        scores_by_model,
        source_commit=args.source_commit,
        source_files=source_files,
    )
    validate_accuracy_profile(profile, task_ids)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps({"tasks": profile["task_count"], "models": len(BASELINE_PANEL)}))


if __name__ == "__main__":
    main()
