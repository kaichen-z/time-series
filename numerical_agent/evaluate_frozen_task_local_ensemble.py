"""Run one mandatory report-only Public-99 regression for a frozen local ensemble."""

from __future__ import annotations

import argparse
import hashlib
import statistics
from pathlib import Path

from common.data import load_tasks_by_id
from common.metrics import drcik_point_metrics, joint_scaled_error, linear_quantile
from common.payload import (
    canonical_json_bytes,
    canonical_json_line_bytes,
    read_json_object,
)

from .evolution.champion import champion_fingerprint
from .evolution.forecast_store import ForecastStore
from .evolution.module import read_module
from .evolution.portfolio import read_policy_file
from .evolution.task_local_ensemble import (
    TaskLocalEnsembleRelease,
    canonical_task_local_release_bytes,
    execute_task_local_ensemble,
    parse_task_local_release,
    task_local_fingerprint,
)
from .evolution.task_local_evolution import TaskLocalTaskRow, task_morphology_key
from .main import _add_tsfm_runtime_options, _runtime_registry
from .run_champion_evolution import (
    _load_parent,
    _load_screening_policy,
    _source_files,
)
from .run_selector_evolution import _forecast_runtime_identity
from .run_task_local_ensemble_evolution import (
    _materialize_rows,
    _reviewed_candidates,
    _smoke_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repo")
    parser.add_argument("--split-file")
    parser.add_argument("--tasks-file")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--anchor-release-dir")
    parser.add_argument("--forecast-store")
    parser.add_argument("--output-dir", required=True)
    _add_tsfm_runtime_options(parser)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_release(release_dir: Path) -> TaskLocalEnsembleRelease:
    complete = read_json_object(release_dir / "evaluation_complete.json")
    if complete.get("status") != "accepted":
        raise ValueError("Public regression requires an accepted task-local release")
    path = release_dir / "task_local_release.json"
    if not path.is_file():
        raise ValueError("accepted task-local release artifact is missing")
    release = parse_task_local_release(read_json_object(path))
    if path.read_bytes() != canonical_task_local_release_bytes(release):
        raise ValueError("task-local release is not canonical immutable JSON")
    if complete.get("release_fingerprint") != task_local_fingerprint(release):
        raise ValueError("task-local completion marker does not bind the release")
    dev = read_json_object(release_dir / "dev_report.json")
    if dev.get("accepted") is not True:
        raise ValueError("task-local Dev report did not accept the release")
    oof = read_json_object(release_dir / "oof_report.json")
    if (
        oof.get("accepted") is not True
        or oof.get("report_fingerprint") != release.oof_report_sha256
    ):
        raise ValueError("task-local OOF report does not bind the release")
    return release


def _rows_by_task(
    rows: tuple[TaskLocalTaskRow, ...], task_ids: tuple[str, ...]
) -> dict[str, dict[str, TaskLocalTaskRow]]:
    result = {task_id: {} for task_id in task_ids}
    for row in rows:
        if row.task_id in result:
            if row.candidate_name in result[row.task_id]:
                raise ValueError("Public rows contain duplicate candidate/task keys")
            result[row.task_id][row.candidate_name] = row
    return result


def _score_public_rows(
    release: TaskLocalEnsembleRelease,
    rows: tuple[TaskLocalTaskRow, ...],
    task_ids: tuple[str, ...],
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    by_task = _rows_by_task(rows, task_ids)
    records: list[dict[str, object]] = []
    parent_smae: list[float] = []
    parent_srmse: list[float] = []
    child_smae: list[float] = []
    child_srmse: list[float] = []
    parent_smae_raw: list[float] = []
    parent_srmse_raw: list[float] = []
    child_smae_raw: list[float] = []
    child_srmse_raw: list[float] = []
    wins = ties = losses = failures = 0
    for task_id in task_ids:
        task_rows = by_task[task_id]
        anchor = task_rows.get(release.anchor_name)
        if anchor is None or anchor.forecast is None:
            failures += 1
            records.append({"task_id": task_id, "status": "anchor_failed"})
            continue
        group_key = task_morphology_key(anchor.profile)
        names = release.candidate_names(group_key)
        forecasts = {
            name: row.forecast
            for name in names
            if (row := task_rows.get(name)) is not None and row.forecast is not None
        }
        diagnostics = {
            name: row.diagnostic
            for name in names
            if (row := task_rows.get(name)) is not None and row.diagnostic is not None
        }
        result = execute_task_local_ensemble(
            release.policy,
            candidate_names=names,
            forecasts=forecasts,
            diagnostics=diagnostics,
            horizon=anchor.profile.horizon,
        )
        parent = drcik_point_metrics(anchor.truth, anchor.forecast)
        child = drcik_point_metrics(anchor.truth, result.forecast)
        for destination, point, field in (
            (parent_smae, parent, "smae"),
            (parent_srmse, parent, "srmse"),
            (child_smae, child, "smae"),
            (child_srmse, child, "srmse"),
            (parent_smae_raw, parent, "smae_raw"),
            (parent_srmse_raw, parent, "srmse_raw"),
            (child_smae_raw, child, "smae_raw"),
            (child_srmse_raw, child, "srmse_raw"),
        ):
            destination.append(float(point[field]))
        parent_joint = joint_scaled_error(float(parent["smae"]), float(parent["srmse"]))
        child_joint = joint_scaled_error(float(child["smae"]), float(child["srmse"]))
        if child_joint < parent_joint - 1e-12:
            wins += 1
        elif child_joint > parent_joint + 1e-12:
            losses += 1
        else:
            ties += 1
        records.append(
            {
                "task_id": task_id,
                "status": "success",
                "anchor_forecast": list(anchor.forecast),
                "forecast": list(result.forecast),
                "selected_names": list(result.selected_names),
                "weights": list(result.weights),
                "fallback_reason": result.fallback_reason,
                "parent_smae": float(parent["smae"]),
                "parent_srmse": float(parent["srmse"]),
                "child_smae": float(child["smae"]),
                "child_srmse": float(child["srmse"]),
            }
        )
    if not child_smae:
        raise ValueError("Public regression produced no valid paired scores")

    def q(values: list[float], probability: float) -> float:
        return float(linear_quantile(values, probability))

    comparison: dict[str, object] = {
        "baseline": release.anchor_name,
        "coverage": len(child_smae) / len(task_ids),
        "failures": failures,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "parent_mean_smae": statistics.fmean(parent_smae),
        "child_mean_smae": statistics.fmean(child_smae),
        "parent_mean_srmse": statistics.fmean(parent_srmse),
        "child_mean_srmse": statistics.fmean(child_srmse),
        "parent_median_smae": float(statistics.median(parent_smae)),
        "child_median_smae": float(statistics.median(child_smae)),
        "parent_median_srmse": float(statistics.median(parent_srmse)),
        "child_median_srmse": float(statistics.median(child_srmse)),
        "parent_p90_smae_raw": q(parent_smae_raw, 0.9),
        "child_p90_smae_raw": q(child_smae_raw, 0.9),
        "parent_p95_smae_raw": q(parent_smae_raw, 0.95),
        "child_p95_smae_raw": q(child_smae_raw, 0.95),
        "parent_p90_srmse_raw": q(parent_srmse_raw, 0.9),
        "child_p90_srmse_raw": q(child_srmse_raw, 0.9),
        "parent_p95_srmse_raw": q(parent_srmse_raw, 0.95),
        "child_p95_srmse_raw": q(child_srmse_raw, 0.95),
        "parent_smae_clipped_count": sum(value > 5.0 for value in parent_smae_raw),
        "child_smae_clipped_count": sum(value > 5.0 for value in child_smae_raw),
        "parent_srmse_clipped_count": sum(value > 5.0 for value in parent_srmse_raw),
        "child_srmse_clipped_count": sum(value > 5.0 for value in child_srmse_raw),
    }
    return comparison, tuple(records)


def _publish(output: Path, report: dict[str, object], records: tuple[dict[str, object], ...]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    forecasts = output / "public_forecasts.jsonl"
    temporary_forecasts = output / ".public_forecasts.jsonl.tmp"
    temporary_report = output / ".public_regression_report.json.tmp"
    with temporary_forecasts.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line_bytes(record))
        handle.flush()
    with temporary_report.open("xb") as handle:
        handle.write(canonical_json_bytes(report))
        handle.flush()
    temporary_forecasts.replace(forecasts)
    temporary_report.replace(output / "public_regression_report.json")
    (output / "evaluation_complete.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "status": "public_regression_complete",
                "public_task_count": report["public_task_count"],
                "release_fingerprint": report["release_fingerprint"],
            }
        )
    )


def _smoke_main(release_dir: Path, output: Path) -> int:
    release = _load_release(release_dir)
    task_ids = tuple(f"public_{index}" for index in range(99))
    rows = _smoke_rows(task_ids, split="public")
    comparison, records = _score_public_rows(release, rows, task_ids)
    _publish(
        output,
        {
            "schema_version": 1,
            "public_task_count": 99,
            "release_fingerprint": task_local_fingerprint(release),
            "comparison": comparison,
        },
        records,
    )
    return 0


def _public_ids(split_file: Path) -> tuple[str, ...]:
    payload = read_json_object(split_file)
    try:
        values = payload["partitions"]["public_test"]["task_ids"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError("split manifest needs Public-99 task IDs") from error
    if type(values) is not list or len(values) != 99 or any(
        type(value) is not str or not value for value in values
    ):
        raise ValueError("Public regression requires exactly 99 task IDs")
    if len(values) != len(set(values)):
        raise ValueError("Public task IDs must be unique")
    return tuple(values)


def _formal_main(args: argparse.Namespace, release_dir: Path, output: Path) -> int:
    required = {
        "repo": args.repo,
        "split_file": args.split_file,
        "tasks_file": args.tasks_file,
        "anchor_release_dir": args.anchor_release_dir,
        "forecast_store": args.forecast_store,
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise ValueError(f"formal Public regression is missing {missing}")
    release = _load_release(release_dir)
    run_manifest = read_json_object(release_dir / "run_manifest.json")
    repo = Path(args.repo).resolve()
    split_file = Path(args.split_file).resolve()
    source_hashes = tuple((name, _sha256(path)) for name, path in _source_files(repo))
    anchor = _load_parent(Path(args.anchor_release_dir) / "champion_release.json")
    if (
        run_manifest.get("split_sha256") != _sha256(split_file)
        or run_manifest.get("source_hashes") != dict(source_hashes)
        or champion_fingerprint(anchor) != release.anchor_release_sha256
        or tuple(sorted(run_manifest.get("source_hashes", {}).items())) != release.source_hashes
    ):
        raise ValueError("Public authority does not match the frozen evolution run")
    module = read_module(repo / "methods.py")
    portfolio = read_policy_file(repo / "policies.py")
    portfolio.validate_namespace(module.names())
    screening = _load_screening_policy(repo / "dictionary.py")
    candidates = _reviewed_candidates(module, portfolio, screening)
    required_names = {
        name
        for supply in (release.default_candidate_names, *(item.candidate_names for item in release.group_supplies))
        for name in supply
    }
    candidates = tuple(item for item in candidates if item[0] in required_names)
    if {name for name, _family in candidates} != required_names:
        raise ValueError("Public runtime is missing a frozen release candidate")
    runtimes = _runtime_registry(args)
    store: ForecastStore | None = None
    try:
        store = ForecastStore(
            Path(args.forecast_store),
            repo / "methods.py",
            repo / "skills.py" if (repo / "skills.py").is_file() else None,
            portfolio,
            runtimes,
            screening_hash=screening.fingerprint(),
            runtime_identity=_forecast_runtime_identity(args),
        )
        if run_manifest.get("forecast_store_fingerprint") != store.identity_hash:
            raise ValueError("Public runtime identity differs from frozen evolution")
        task_ids = _public_ids(split_file)
        # Bodies are opened only after all release/source/runtime authority is verified.
        loaded = {task.task_id: task for task in load_tasks_by_id(args.tasks_file, task_ids)}
        if set(loaded) != set(task_ids):
            raise ValueError("Public regression is missing task bodies")
        tasks = tuple(loaded[task_id] for task_id in task_ids)
        rows = _materialize_rows(store, tasks, candidates, screening, split="public")
        comparison, records = _score_public_rows(release, rows, task_ids)
        _publish(
            output,
            {
                "schema_version": 1,
                "public_task_count": 99,
                "release_fingerprint": task_local_fingerprint(release),
                "split_sha256": _sha256(split_file),
                "comparison": comparison,
            },
            records,
        )
        return 0
    finally:
        if store is not None:
            store.close()
        runtimes.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    release_dir = Path(args.release_dir).resolve()
    output = Path(args.output_dir).resolve()
    if (output / "evaluation_complete.json").exists():
        raise ValueError("Public regression has already completed")
    if args.smoke:
        return _smoke_main(release_dir, output)
    return _formal_main(args, release_dir, output)


if __name__ == "__main__":
    raise SystemExit(main())
