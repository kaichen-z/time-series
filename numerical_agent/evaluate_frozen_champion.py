"""Score one immutable Champion release on the sealed Public partition once."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path

from common.data import load_tasks_by_id
from common.payload import canonical_json_bytes, read_json_object

from .evolution.champion import champion_fingerprint, parse_champion_release
from .evolution.champion_controller import (
    ChampionArtifactStore,
    ChampionRunManifest,
    canonical_release_bytes,
)
from .evolution.champion_evidence import (
    ChampionHistoryDiagnostic,
    ChampionTaskRow,
    score_policy,
)
from .evolution.champion_runtime import execute_champion
from .evolution.execution import Task
from .evolution.forecast_store import ForecastStore
from .evolution.module import read_module
from .evolution.numerical_selector import HindcastConfig, diagnose_candidate
from .evolution.portfolio import read_policy_file
from .evolution.screening import profile_task
from .main import _add_tsfm_runtime_options, _runtime_registry
from .run_selector_evolution import _forecast_runtime_identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--forecast-store", required=True)
    parser.add_argument("--output-dir", required=True)
    _add_tsfm_runtime_options(parser)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_ids(split_file: str | Path) -> tuple[str, ...]:
    payload = read_json_object(split_file)
    try:
        values = payload["partitions"]["public"]["task_ids"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError("split manifest needs Public task IDs") from error
    if type(values) is not list or any(
        type(value) is not str or not value for value in values
    ):
        raise ValueError("Public task IDs must be nonempty strings")
    ids = tuple(values)
    if len(ids) != len(set(ids)):
        raise ValueError("Public task IDs must be unique")
    return ids


def load_public_partition(
    split_file: str | Path, tasks_file: str | Path
) -> tuple[Task, ...]:
    """Load only the Public records named by the frozen split."""
    ids = _public_ids(split_file)
    loaded = {task.task_id: task for task in load_tasks_by_id(tasks_file, ids)}
    rows: list[Task] = []
    for task_id in ids:
        task = loaded.get(task_id)
        if task is None:
            raise ValueError(f"missing public task {task_id}")
        rows.append(
            Task(
                task.task_id,
                tuple(task.history_values),
                task.prediction_length,
                task.frequency,
                tuple(task.future_values),
            )
        )
    return tuple(rows)


def _release(release_dir: Path):
    path = release_dir / "champion_release.json"
    if not path.is_file():
        raise ValueError("frozen evaluation requires champion_release.json")
    try:
        release = parse_champion_release(read_json_object(path))
    except Exception as error:
        raise ValueError("champion release is malformed") from error
    if path.read_bytes() != canonical_release_bytes(release):
        raise ValueError("champion release is not immutable canonical JSON")
    return release


def _load_manifest(release_dir: Path) -> ChampionRunManifest:
    path = release_dir / "run_manifest.json"
    if not path.is_file():
        raise ValueError("frozen evaluation requires canonical run_manifest.json")
    payload = read_json_object(path)
    expected = {item.name for item in fields(ChampionRunManifest)} | {
        "input_fingerprint"
    }
    if set(payload) != expected:
        raise ValueError("run manifest schema is malformed")
    values = dict(payload)
    input_fingerprint = values.pop("input_fingerprint")
    for name in (
        "source_hashes",
        "train_tasks",
        "train_task_hashes",
        "dev_tasks",
        "dev_task_hashes",
        "build_tasks",
        "calibration_tasks",
        "dictionary_hashes",
    ):
        raw = values.get(name)
        if type(raw) is list:
            values[name] = tuple(tuple(item) for item in raw)
    try:
        manifest = ChampionRunManifest(**values)
    except Exception as error:
        raise ValueError("run manifest schema is malformed") from error
    if (
        input_fingerprint != manifest.input_fingerprint
        or path.read_bytes() != canonical_json_bytes(manifest.to_payload())
    ):
        raise ValueError("run manifest is not canonical immutable JSON")
    return manifest


def _verify_source_closure(repo: Path, manifest: ChampionRunManifest, release) -> None:
    source_map = {
        "dictionary": repo / "dictionary.py",
        "methods": repo / "methods.py",
        "policies": repo / "policies.py",
    }
    skills = repo / "skills.py"
    if skills.is_file():
        source_map["skills"] = skills
    actual = tuple(
        sorted(
            (name, _sha256(path)) for name, path in source_map.items() if path.is_file()
        )
    )
    if (
        actual != manifest.dictionary_hashes
        or release.source_hashes != manifest.dictionary_hashes
    ):
        raise ValueError("frozen source closure does not match release authority")


def _validate_frozen_authority(
    repo: Path,
    release_dir: Path,
    split_file: str | Path,
    runtime_identity: object,
):
    """Validate every immutable evolution authority before opening Public data."""
    release = _release(release_dir)
    manifest = _load_manifest(release_dir)
    if _sha256(Path(split_file)) != manifest.split_manifest_fingerprint:
        raise ValueError("split fingerprint does not match frozen evolution authority")
    if (
        champion_fingerprint(runtime_identity)
        != manifest.forecast_runtime_identity_fingerprint
    ):
        raise ValueError(
            "forecast runtime identity does not match frozen evolution authority"
        )
    _verify_source_closure(repo, manifest, release)
    artifacts = ChampionArtifactStore(release_dir)
    try:
        if not artifacts.has_accepted_release():
            raise ValueError("frozen evaluation requires an accepted release authority")
        artifacts.ensure_release(release)
    finally:
        artifacts.close()
    return release


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    report_path = output / "public_regression_report.json"
    if report_path.exists():
        raise ValueError(
            "Public regression has already completed and cannot be overwritten"
        )
    release_dir, repo = Path(args.release_dir).resolve(), Path(args.repo).resolve()
    runtime_identity = _forecast_runtime_identity(args)
    release = _validate_frozen_authority(
        repo, release_dir, args.split_file, runtime_identity
    )
    public = load_public_partition(args.split_file, args.tasks_file)
    module, portfolio = read_module(repo / "methods.py"), read_policy_file(
        repo / "policies.py"
    )
    portfolio.validate_namespace(module.names())
    names = tuple(module.names()) + portfolio.names
    required = set(release.policy.recipe.parents)
    if required - set(names):
        raise ValueError(
            "champion release references a candidate absent from the frozen runtime"
        )
    runtimes = _runtime_registry(args)
    store: ForecastStore | None = None
    try:
        store = ForecastStore(
            Path(args.forecast_store),
            repo / "methods.py",
            repo / "skills.py" if (repo / "skills.py").is_file() else None,
            portfolio,
            runtimes,
            screening_hash="formal_champion_all_candidates",
            runtime_identity=runtime_identity,
        )
        rows: list[ChampionTaskRow] = []
        output.mkdir(parents=True, exist_ok=True)
        forecast_path = output / "public_frozen_forecasts.jsonl"
        with forecast_path.open("x", encoding="utf-8") as handle:
            for task in public:
                profile = profile_task(task)
                forecasts: dict[str, tuple[float, ...]] = {}
                diagnostics = {}
                for name in release.policy.recipe.parents:
                    try:
                        forecasts[name] = store.forecast(
                            name, task.history, task.horizon, task.frequency
                        )
                    except Exception:
                        continue
                    family = "statistical" if name in module.names() else "tsfm"
                    diagnostics[name] = diagnose_candidate(
                        task,
                        name,
                        family,
                        store.forecast,
                        HindcastConfig(),
                        runtime_settings={"forecast_store": store.identity_hash},
                    )
                execution = execute_champion(
                    release.policy,
                    forecasts,
                    diagnostics,
                    profile,
                    task.history,
                    task.horizon,
                )
                failure = (
                    None
                    if len(execution.forecast) == task.horizon
                    else "frozen_champion_invalid_forecast"
                )
                row = ChampionTaskRow(
                    task.task_id,
                    release.policy.recipe.name,
                    profile,
                    tuple(task.future),
                    execution.forecast if failure is None else None,
                    failure,
                    0,
                    "public",
                    tuple(task.history),
                    None,
                )
                rows.append(row)
                handle.write(
                    json.dumps(
                        {
                            "task_id": task.task_id,
                            "forecast": list(execution.forecast),
                            "selected_names": list(execution.selected_names),
                            "fallback_reason": execution.fallback_reason,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
        score = score_policy(tuple(rows), release.policy.recipe.name)
        payload = {
            "schema_version": 1,
            "release_fingerprint": _sha256(release_dir / "champion_release.json"),
            "split_manifest_fingerprint": _sha256(Path(args.split_file)),
            "forecast_runtime_identity": store.identity_hash,
            "public_task_count": len(public),
            "score": asdict(score),
        }
        report_path.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"public": len(public), "coverage": score.coverage}, sort_keys=True
            )
        )
        return 0
    finally:
        if store is not None:
            store.close()
        runtimes.close()


if __name__ == "__main__":
    raise SystemExit(main())
