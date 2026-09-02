"""Fit a task-local Numerical ensemble on Train OOF and gate it once on Dev."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

from common.data import Task as DataTask, load_tasks_by_id
from common.payload import canonical_json_bytes

from .evolution.champion import champion_fingerprint
from .evolution.execution import Task as RuntimeTask
from .evolution.forecast_store import ForecastStore
from .evolution.module import read_module
from .evolution.numerical_selector import HindcastConfig, diagnose_candidate
from .evolution.portfolio import read_policy_file
from .evolution.screening import (
    ScreeningPolicy,
    materialize_active_dictionary,
    profile_task,
)
from .evolution.task_local_ensemble import (
    TaskLocalTournamentPolicy,
    canonical_task_local_release_bytes,
    task_local_fingerprint,
)
from .evolution.task_local_confidence import ConfidencePolicy
from .evolution.task_local_evolution import (
    ConditionalUpliftReport,
    TaskLocalTaskRow,
    build_group_fold_manifest,
    evaluate_task_local_release,
    fit_oof_release,
)
from .main import _add_tsfm_runtime_options, _runtime_registry
from .run_champion_evolution import (
    _clean_git_source,
    _load_parent,
    _load_screening_policy,
    _partition_ids,
    _source_files,
)
from .run_selector_evolution import _forecast_runtime_identity


_CONFIDENCE_HINDCAST_CONFIG = HindcastConfig(folds=5, min_successful_folds=3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-dev-regression", action="store_true")
    parser.add_argument("--repo")
    parser.add_argument("--split-file")
    parser.add_argument("--tasks-file")
    parser.add_argument("--anchor-release-dir")
    parser.add_argument("--forecast-store")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partition-seed", type=int, default=20260902)
    _add_tsfm_runtime_options(parser)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, payload: dict[str, object] | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload if isinstance(payload, bytes) else canonical_json_bytes(payload)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()


def _report_payload(report: ConditionalUpliftReport) -> dict[str, object]:
    return asdict(report)


def _smoke_profile(task_id: str, *, periodic: bool):
    from .evolution.screening import TaskProfile

    return TaskProfile(
        task_id=task_id,
        frequency="D",
        history_length=4,
        horizon=2,
        zero_fraction=0.0,
        signed=False,
        integer_valued=True,
        trend_direction="flat",
        trend_strength=0.1,
        periodicity_periods=(2,) if periodic else (),
        periodicity_strength=0.8 if periodic else 0.1,
        periodicity_confidence=0.9 if periodic else 0.1,
        outlier_fraction=0.0,
        noise_relative_scale=0.1,
        likely_stationary=True,
        stationarity_score=0.9,
        recent_regime_start=None,
        recent_regime_confidence=0.1,
        intermittency_adi=1.0,
        intermittency_cv2=0.1,
    )


def _smoke_rows(
    task_ids: Iterable[str],
    *,
    split: str,
    specialist_regression: bool = False,
) -> tuple[TaskLocalTaskRow, ...]:
    from .evolution.numerical_selector import CandidateDiagnostics

    rows: list[TaskLocalTaskRow] = []
    for index, task_id in enumerate(task_ids):
        profile = _smoke_profile(task_id, periodic=bool(index % 2))
        truth = (10.0, 10.0)
        history = (7.0, 8.0, 9.0, 10.0)
        for name, family, full, folds in (
            ("toto_2_0", "tsfm", (8.0, 8.0), ((8.0, 8.0),) * 5),
            (
                "seasonal_naive",
                "statistical",
                (0.0, 0.0) if specialist_regression else (10.0, 10.0),
                ((10.0, 10.0),) * 5,
            ),
        ):
            diagnostic = CandidateDiagnostics.synthetic(
                name=name,
                family=family,
                median_mase=1.0,
                fold_forecasts=folds,
                fold_truths=(truth,) * 5,
                median_smae=1.0,
                median_srmse=1.0,
            )
            rows.append(
                TaskLocalTaskRow(
                    task_id=task_id,
                    candidate_name=name,
                    family=family,
                    profile=profile,
                    history=history,
                    truth=truth,
                    forecast=full,
                    diagnostic=diagnostic,
                    split=split,
                )
            )
    return tuple(rows)


def _run_smoke(output: Path, *, dev_regression: bool) -> int:
    train_ids = tuple(f"train_{index}" for index in range(8))
    dev_ids = ("dev_periodic", "dev_dense")
    tasks = tuple(
        DataTask(
            task_id=task_id,
            history_values=(float(index + 1), 2.0, 3.0, 4.0),
            future_values=(10.0, 10.0),
            prediction_length=2,
            frequency="D",
            seasonal_period="2",
            entity_name=f"Entity {index}",
        )
        for index, task_id in enumerate(train_ids)
    )
    policy = TaskLocalTournamentPolicy(
        minimum_activation_support=2,
        minimum_activation_groups=2,
    )
    manifest = build_group_fold_manifest(tasks, seed=17)
    release, oof = fit_oof_release(
        _smoke_rows(train_ids, split="train"),
        manifest,
        anchor_release_sha256="a" * 64,
        anchor_name="toto_2_0",
        source_hashes=(("dictionary", "b" * 64),),
        policy=policy,
        confidence_policy=ConfidencePolicy(
            exact_minimum_support=2,
            coarse_minimum_support=2,
            global_minimum_support=2,
        ),
    )
    _write_once(output / "group_folds.json", manifest.to_payload())
    _write_once(output / "oof_report.json", _report_payload(oof))
    if not oof.accepted:
        _write_once(
            output / "evaluation_complete.json",
            {
                "schema_version": 1,
                "status": "oof_rejected",
                "train_oof_tasks": 8,
                "dev_tasks": 0,
            },
        )
        return 0
    dev = evaluate_task_local_release(
        release,
        _smoke_rows(
            dev_ids,
            split="dev",
            specialist_regression=dev_regression,
        ),
        task_ids=dev_ids,
        split="dev",
    )
    _write_once(output / "dev_report.json", _report_payload(dev))
    status = "accepted" if dev.accepted else "dev_rejected"
    if dev.accepted:
        _write_once(output / "task_local_release.json", canonical_task_local_release_bytes(release))
    _write_once(
        output / "evaluation_complete.json",
        {
            "schema_version": 1,
            "status": status,
            "train_oof_tasks": 8,
            "dev_tasks": 2,
            "release_fingerprint": task_local_fingerprint(release),
        },
    )
    return 0


def _reviewed_candidates(
    module: object,
    portfolio: object,
    screening: ScreeningPolicy,
) -> tuple[tuple[str, str], ...]:
    method_names = tuple(item.name for item in module.methods)  # type: ignore[attr-defined]
    runtime = [(name, "statistical") for name in method_names]
    runtime.extend(
        (
            item.name,
            "tsfm" if item in portfolio.tsfm else "combined",  # type: ignore[attr-defined]
        )
        for item in portfolio.all_policies  # type: ignore[attr-defined]
    )
    entries = {entry.name: entry for entry in screening.entries}
    if set(entries) != {name for name, _family in runtime}:
        raise ValueError("task-local runtime and screening namespaces differ")
    return tuple(
        (name, family)
        for name, family in runtime
        if entries[name].status in {"keep", "specialized"}
    )


def _materialize_rows(
    store: ForecastStore,
    tasks: Iterable[DataTask],
    candidates: tuple[tuple[str, str], ...],
    screening: ScreeningPolicy,
    *,
    split: str,
    hindcast_config: HindcastConfig,
) -> tuple[TaskLocalTaskRow, ...]:
    rows: list[TaskLocalTaskRow] = []
    for source in tasks:
        task = RuntimeTask(
            source.task_id,
            tuple(source.history_values),
            source.prediction_length,
            source.frequency,
            tuple(source.future_values),
        )
        profile = profile_task(task)
        active = {
            item.name for item in materialize_active_dictionary(screening, profile).active
        }
        for name, family in candidates:
            forecast = None
            failure: str | None = None
            diagnostic = None
            if name not in active and name != "toto_2_0":
                failure = "NotApplicable: screening_policy"
            else:
                try:
                    forecast = store.forecast(name, task.history, task.horizon, task.frequency)
                except Exception as error:
                    failure = f"{type(error).__name__}: {error}"[:10000]
                try:
                    diagnostic = diagnose_candidate(
                        task,
                        name,
                        family,
                        store.forecast,
                        hindcast_config,
                        runtime_settings={"forecast_store": store.identity_hash},
                    )
                except Exception as error:
                    if failure is None:
                        forecast = None
                        failure = f"HistoryDiagnosticUnavailable: {type(error).__name__}"[:10000]
            rows.append(
                TaskLocalTaskRow(
                    task_id=task.task_id,
                    candidate_name=name,
                    family=family,
                    profile=profile,
                    history=tuple(task.history),
                    truth=tuple(task.future),
                    forecast=forecast,
                    diagnostic=diagnostic,
                    split=split,
                    failure_reason=failure,
                )
            )
    return tuple(rows)


def _formal_main(args: argparse.Namespace, output: Path) -> int:
    required = {
        "repo": args.repo,
        "split_file": args.split_file,
        "tasks_file": args.tasks_file,
        "anchor_release_dir": args.anchor_release_dir,
        "forecast_store": args.forecast_store,
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise ValueError(f"formal task-local evolution is missing {missing}")
    repo = Path(args.repo).resolve()
    _clean_git_source(repo)
    train_ids, dev_ids = _partition_ids(args.split_file, include_dev=True)
    if len(train_ids) != 80 or len(dev_ids) != 20:
        raise ValueError("formal task-local evolution requires exactly 80 Train and 20 Dev")
    train_by_id = {task.task_id: task for task in load_tasks_by_id(args.tasks_file, train_ids)}
    if set(train_by_id) != set(train_ids):
        raise ValueError("formal task-local evolution is missing Train tasks")
    train = tuple(train_by_id[task_id] for task_id in train_ids)
    source_files = _source_files(repo)
    source_hashes = tuple((name, _sha256(path)) for name, path in source_files)
    anchor_path = Path(args.anchor_release_dir) / "champion_release.json"
    anchor = _load_parent(anchor_path)
    module = read_module(repo / "methods.py")
    portfolio = read_policy_file(repo / "policies.py")
    portfolio.validate_namespace(module.names())
    screening = _load_screening_policy(repo / "dictionary.py")
    candidates = _reviewed_candidates(module, portfolio, screening)
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
        train_rows = _materialize_rows(
            store,
            train,
            candidates,
            screening,
            split="train",
            hindcast_config=_CONFIDENCE_HINDCAST_CONFIG,
        )
        manifest = build_group_fold_manifest(
            train, seed=int(args.partition_seed)
        )
        release, oof = fit_oof_release(
            train_rows,
            manifest,
            anchor_release_sha256=champion_fingerprint(anchor),
            anchor_name=anchor.policy.recipe.fallback_parent,
            source_hashes=source_hashes,
            policy=TaskLocalTournamentPolicy(
                anchor_name=anchor.policy.recipe.fallback_parent
            ),
            confidence_policy=ConfidencePolicy(),
        )
        run_manifest = {
            "schema_version": 2,
            "split_sha256": _sha256(Path(args.split_file)),
            "source_hashes": dict(source_hashes),
            "anchor_release_sha256": champion_fingerprint(anchor),
            "forecast_store_fingerprint": store.identity_hash,
            "grouping_fingerprint": manifest.grouping_fingerprint,
            "tournament_policy_fingerprint": task_local_fingerprint(release.policy),
            "confidence_policy_fingerprint": task_local_fingerprint(
                release.confidence_evidence.policy
                if release.confidence_evidence is not None
                else {}
            ),
            "hindcast_config_fingerprint": task_local_fingerprint(
                _CONFIDENCE_HINDCAST_CONFIG
            ),
            "train_task_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
            "dev_task_ids_sha256": hashlib.sha256("\n".join(dev_ids).encode()).hexdigest(),
        }
        _write_once(output / "run_manifest.json", run_manifest)
        _write_once(output / "group_folds.json", manifest.to_payload())
        _write_once(output / "oof_report.json", _report_payload(oof))
        if not oof.accepted:
            _write_once(
                output / "evaluation_complete.json",
                {
                    "schema_version": 1,
                    "status": "oof_rejected",
                    "train_oof_tasks": 80,
                    "dev_tasks": 0,
                },
            )
            return 0

        # Dev task bodies are deliberately loaded only after frozen OOF acceptance.
        dev_by_id = {task.task_id: task for task in load_tasks_by_id(args.tasks_file, dev_ids)}
        if set(dev_by_id) != set(dev_ids):
            raise ValueError("formal task-local evolution is missing Dev tasks")
        dev = tuple(dev_by_id[task_id] for task_id in dev_ids)
        dev_rows = _materialize_rows(
            store,
            dev,
            candidates,
            screening,
            split="dev",
            hindcast_config=_CONFIDENCE_HINDCAST_CONFIG,
        )
        dev_report = evaluate_task_local_release(
            release, dev_rows, task_ids=dev_ids, split="dev"
        )
        _write_once(output / "dev_report.json", _report_payload(dev_report))
        status = "accepted" if dev_report.accepted else "dev_rejected"
        if dev_report.accepted:
            _write_once(
                output / "task_local_release.json",
                canonical_task_local_release_bytes(release),
            )
        _write_once(
            output / "evaluation_complete.json",
            {
                "schema_version": 1,
                "status": status,
                "train_oof_tasks": 80,
                "dev_tasks": 20,
                "release_fingerprint": task_local_fingerprint(release),
            },
        )
        return 0
    finally:
        if store is not None:
            store.close()
        runtimes.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    if (output / "evaluation_complete.json").exists():
        raise ValueError("task-local evolution has already completed")
    if args.smoke:
        return _run_smoke(output, dev_regression=bool(args.smoke_dev_regression))
    if args.smoke_dev_regression:
        raise ValueError("--smoke-dev-regression requires --smoke")
    return _formal_main(args, output)


if __name__ == "__main__":
    raise SystemExit(main())
