"""Label-free Dr-CiK Hidden Test export for frozen two-stage numerical policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

from common.payload import read_json_object

from .evaluate_frozen_two_stage import (
    ForecastResult,
    _build_case,
    _hindcast_config_for_policy,
    _selector_forecast,
    verify_frozen_policies,
)
from .evolution.execution import Task
from .evolution.filtering import build_filter_dictionary
from .evolution.module import read_module
from .evolution.numerical_selector import DecisionPolicy, HindcastConfig
from .evolution.portfolio import read_policy_file
from .evolution.screening import ScreeningPolicy
from .evolution.screening_evolution import migrate_filter_dictionary, parse_screening_source
from .evolution.selector_evolution import parse_decision_source
from .main import _add_tsfm_runtime_options, _runtime_registry
from .run_selector_evolution import ForecastStore
from .run_task_conditioned_screening import require_flagship_runtimes


def load_hidden_tasks(tasks_file: str | Path) -> tuple[Task, ...]:
    """Load only official hidden inputs and discard every embedded evaluator field."""
    source = Path(tasks_file)
    if source.is_dir():
        records = (
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(source.glob("task_*.json"))
        )
    else:
        records = (
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    tasks = []
    for record in records:
        if record.get("labels_public", True) is not False:
            continue
        series = record.get("series", record)
        metadata = record.get("task_metadata", record)
        history = tuple(float(value) for value in series["history_values"])
        horizon = int(metadata["prediction_length"])
        timestamps = tuple(series.get("future_timestamps") or ())
        if not history:
            raise ValueError(f"{record['benchmark_id']}: empty history")
        if horizon <= 0 or len(timestamps) != horizon:
            raise ValueError(f"{record['benchmark_id']}: invalid hidden prediction horizon")
        tasks.append(Task(
            str(record["benchmark_id"]),
            history,
            horizon,
            str(metadata["frequency"]),
            (),
        ))
    return tuple(tasks)


def write_hidden_submission(
    tasks: Sequence[Task],
    results: Sequence[ForecastResult],
    *,
    output_dir: str | Path,
    samples: int,
    screening_policy_sha256: str,
    decision_policy_sha256: str,
    deployment_id: str = "",
) -> dict[str, object]:
    """Write a complete point-valued submission without reading or scoring labels."""
    if samples < 100:
        raise ValueError("official Hidden Test export requires at least 100 samples")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValueError("Hidden Test submission export has already completed or is occupied")
    expected = {task.task_id: task for task in tasks}
    counts = Counter(result.task_id for result in results)
    if set(counts) != set(expected) or any(count != 1 for count in counts.values()):
        raise ValueError("Hidden Test submission coverage must match the selected tasks exactly")

    rows = []
    selections = []
    for result in results:
        task = expected[result.task_id]
        values = tuple(float(value) for value in result.forecast)
        if len(values) != task.horizon or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{task.task_id}: invalid forecast")
        rows.append({
            "benchmark_id": task.task_id,
            "samples": [list(values) for _ in range(samples)],
        })
        selections.append({
            "benchmark_id": task.task_id,
            "selected": list(result.selected),
            "families": list(result.families),
            "mode": result.mode,
            "assumption_ids": list(result.assumption_ids),
        })

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        forecasts_path = staging / "forecasts.jsonl"
        _write_text_fsync(
            forecasts_path,
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        )
        selections_path = staging / "selection_trace.jsonl"
        _write_text_fsync(
            selections_path,
            "".join(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                for row in selections
            ),
        )
        summary: dict[str, object] = {
            "schema_version": 1,
            "task_count": len(tasks),
            "labels_accessed": False,
            "samples_per_task": samples,
            "probabilistic": False,
            "probabilistic_note": "Each point forecast is repeated; uncertainty is not modeled.",
            "screening_policy_sha256": screening_policy_sha256,
            "decision_policy_sha256": decision_policy_sha256,
            "deployment_id": deployment_id,
            "forecasts_sha256": _sha256(forecasts_path),
            "selection_trace_sha256": _sha256(selections_path),
        }
        summary_path = staging / "submission_summary.json"
        _write_json_fsync(summary_path, summary)
        completion = {
            "task_count": len(tasks),
            "screening_policy_sha256": screening_policy_sha256,
            "decision_policy_sha256": decision_policy_sha256,
            "forecasts_sha256": summary["forecasts_sha256"],
            "selection_trace_sha256": summary["selection_trace_sha256"],
            "submission_summary_sha256": _sha256(summary_path),
        }
        _write_json_fsync(staging / "submission_complete.json", completion)
        _verify_staged_submission(staging, completion)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_hidden_deployment(
    repo: str | Path,
    screening_dir: str | Path,
    selector_dir: str | Path,
    deployment_manifest: str | Path,
) -> dict[str, object]:
    """Bind a hidden run to one explicitly reviewed policy and candidate implementation."""
    root = Path(repo)
    screen = Path(screening_dir)
    selector = Path(selector_dir)
    deployment = read_json_object(deployment_manifest)
    required = {
        "schema_version",
        "deployment_id",
        "candidate_count",
        "methods_sha256",
        "policies_sha256",
        "skills_sha256",
        "screening_policy_sha256",
        "decision_policy_sha256",
        "runtime_manifest_sha256",
    }
    if not required <= set(deployment) or deployment.get("schema_version") != 1:
        raise ValueError("invalid Hidden Test deployment manifest")
    if deployment.get("candidate_count") != 103:
        raise ValueError("Hidden Test deployment candidate_count must be exactly 103")
    actual = {
        "methods_sha256": _sha256(root / "methods.py"),
        "policies_sha256": _sha256(root / "policies.py"),
        "screening_policy_sha256": _sha256(screen / "frozen_screening_policy.py"),
        "decision_policy_sha256": _sha256(selector / "frozen_decision_policy.py"),
    }
    actual["skills_sha256"] = _sha256(root / "skills.py")
    actual["runtime_manifest_sha256"] = _sha256(
        Path(__file__).resolve().parent / "tsfm" / "runtime_manifests.json"
    )
    labels = {
        "methods_sha256": "methods.py",
        "policies_sha256": "policies.py",
        "screening_policy_sha256": "screening policy",
        "decision_policy_sha256": "decision policy",
        "skills_sha256": "skills.py",
        "runtime_manifest_sha256": "TSFM runtime manifest",
    }
    for field, value in actual.items():
        if deployment.get(field) != value:
            raise ValueError(f"Hidden Test deployment {labels[field]} hash mismatch")
    screen_manifest = read_json_object(screen / "screening_manifest.json")
    source_hashes = screen_manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("screening manifest is missing candidate source hashes")
    for name, field in (("methods.py", "methods_sha256"), ("policies.py", "policies_sha256")):
        if source_hashes.get(name) != actual[field]:
            raise ValueError(f"screening manifest {name} hash mismatch")
    return deployment


def _write_text_fsync(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_fsync(path: Path, payload: object) -> None:
    _write_text_fsync(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_staged_submission(staging: Path, completion: dict[str, object]) -> None:
    expected = {
        "forecasts_sha256": _sha256(staging / "forecasts.jsonl"),
        "selection_trace_sha256": _sha256(staging / "selection_trace.jsonl"),
        "submission_summary_sha256": _sha256(staging / "submission_summary.json"),
    }
    if any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError("staged Hidden Test submission failed hash verification")
    if read_json_object(staging / "submission_complete.json") != completion:
        raise ValueError("staged Hidden Test completion seal failed verification")


def run_hidden_two_stage(
    tasks: Sequence[Task],
    *,
    screening: ScreeningPolicy,
    screening_hash: str,
    decision_policy: DecisionPolicy,
    store: ForecastStore,
    hindcast_config: HindcastConfig,
) -> tuple[ForecastResult, ...]:
    """Execute immutable history-only screening and selection on hidden inputs."""
    results = []
    for task in tasks:
        if task.future:
            raise ValueError(f"{task.task_id}: Hidden Test runner received future labels")
        case = _build_case(
            task,
            screening,
            screening_hash,
            {},
            store,
            hindcast_config,
            forecast_from_store=True,
        )
        result = _selector_forecast(case, decision_policy)
        if len(result.forecast) != task.horizon:
            raise ValueError(f"{task.task_id}: frozen selector produced no valid forecast")
        results.append(result)
    return tuple(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--screening-dir", required=True)
    parser.add_argument("--selector-dir", required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--hindcast-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=100)
    _add_tsfm_runtime_options(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    deployment = verify_hidden_deployment(
        args.repo, args.screening_dir, args.selector_dir, args.deployment_manifest
    )
    screening_hash, decision_hash = verify_frozen_policies(
        args.screening_dir, args.selector_dir, output
    )
    tasks = load_hidden_tasks(args.tasks_file)
    if len(tasks) != 80:
        raise ValueError(f"official Hidden Test export requires exactly 80 tasks, got {len(tasks)}")

    repo = Path(args.repo).resolve()
    module_path = repo / "methods.py"
    skills_path = repo / "skills.py"
    module = read_module(module_path)
    portfolio = read_policy_file(repo / "policies.py")
    screening = parse_screening_source(
        (Path(args.screening_dir) / "frozen_screening_policy.py").read_text(encoding="utf-8")
    )
    # Ensure the policy still covers the exact executable 103-candidate universe.
    reference = migrate_filter_dictionary(
        build_filter_dictionary(module, portfolio),
        fallback_names=("naive_last", "timesfm_2_5", "toto_2_0"),
    )
    if {entry.name for entry in screening.entries} != {
        entry.name for entry in reference.entries
    }:
        raise ValueError("frozen screening policy does not match the executable candidate set")
    decision_policy = parse_decision_source(
        (Path(args.selector_dir) / "frozen_decision_policy.py").read_text(encoding="utf-8")
    )

    runtimes = _runtime_registry(args)
    try:
        require_flagship_runtimes(portfolio, runtimes)
        store = ForecastStore(
            args.hindcast_cache_dir,
            module_path,
            skills_path if skills_path.is_file() else None,
            module,
            portfolio,
            runtimes,
            screening_hash,
            deployment_fingerprint=hashlib.sha256(
                json.dumps(
                    deployment, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            ).hexdigest(),
        )
        try:
            results = run_hidden_two_stage(
                tasks,
                screening=screening,
                screening_hash=screening_hash,
                decision_policy=decision_policy,
                store=store,
                hindcast_config=_hindcast_config_for_policy(decision_policy),
            )
        finally:
            store.close()
    finally:
        runtimes.close()

    summary = write_hidden_submission(
        tasks,
        results,
        output_dir=output,
        samples=args.samples,
        screening_policy_sha256=screening_hash,
        decision_policy_sha256=decision_hash,
        deployment_id=str(deployment["deployment_id"]),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
