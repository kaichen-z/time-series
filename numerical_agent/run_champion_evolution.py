"""Run the sealed Train/Dev Champion--Challenger lifecycle.

This command deliberately has no Public input.  Public regression is the
separate ``evaluate_frozen_champion`` command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from common.data import Task as DataTask, load_tasks_by_id
from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT
from common.payload import read_json_object

from .dictionary import MethodDefinition, ToolDictionary
from .evolution.champion import ChampionRelease, parse_champion_release
from .evolution.champion_controller import (
    ChampionArtifactStore,
    ChampionAuthorityStore,
    ChampionEvolutionConfig,
    ChampionEvolutionController,
    ChampionProposerAdapter,
    ChampionRowProviderAdapter,
    ChampionRunAttestations,
    ChampionRunManifest,
    ChampionRuntimeBindings,
    partition_train_tasks,
    task_content_fingerprint,
)
from .evolution.champion_evidence import ChampionHistoryDiagnostic, ChampionTaskRow
from .evolution.execution import Task as RuntimeTask
from .evolution.forecast_store import ForecastStore
from .evolution.module import read_module
from .evolution.numerical_selector import HindcastConfig, diagnose_candidate
from .evolution.portfolio import read_policy_file
from .evolution.screening import profile_task
from .main import _add_tsfm_runtime_options, _runtime_registry
from .run_selector_evolution import _forecast_runtime_identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo")
    parser.add_argument("--split-file")
    parser.add_argument("--tasks-file")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--proposer-model", default="gpt-5.6-sol")
    parser.add_argument("--proposer-reasoning-effort", default="high")
    parser.add_argument("--proposer-timeout", type=int, default=900)
    parser.add_argument("--proposer-cache-dir", default=None)
    parser.add_argument("--candidate-minimum-gain", type=float, default=0.005)
    parser.add_argument("--research-target-gain", type=float, default=0.05)
    parser.add_argument("--output-dir")
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--authority-identity", required=True)
    parser.add_argument("--provision-authority", action="store_true")
    parser.add_argument("--parent-release", default=None)
    parser.add_argument("--forecast-store", default=None)
    parser.add_argument("--partition-seed", type=int, default=20260901)
    parser.add_argument(
        "--include-dev", action=argparse.BooleanOptionalAction, default=True
    )
    _add_tsfm_runtime_options(parser)
    return parser


def _require_normal_arguments(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("repo", "split_file", "tasks_file", "output_dir")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(
            "normal evolution requires "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_git_source(repo: Path) -> None:
    if not (repo / ".git").is_dir():
        raise ValueError("Champion source repository must be a Git worktree")
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode or result.stdout.strip():
        raise ValueError("Champion source repository must be clean")


def _partition_ids(
    split_file: str | Path, *, include_dev: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = read_json_object(split_file)
    try:
        partitions = payload["partitions"]
        train = partitions["train"]["task_ids"]  # type: ignore[index]
        dev = partitions["dev"]["task_ids"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError("split manifest needs Train and Dev task IDs") from error
    if type(train) is not list or type(dev) is not list:
        raise ValueError("split task IDs must be JSON arrays")
    train_ids, dev_ids = tuple(train), tuple(dev) if include_dev else ()
    if any(type(item) is not str or not item for item in (*train_ids, *dev_ids)):
        raise ValueError("split task IDs must be nonempty strings")
    if len(train_ids) != len(set(train_ids)) or len(dev_ids) != len(set(dev_ids)):
        raise ValueError("split task IDs must be unique")
    if set(train_ids) & set(dev_ids):
        raise ValueError("Train and Dev task IDs must be disjoint")
    return train_ids, dev_ids


def load_evolution_partitions(
    split_file: str | Path, tasks_file: str | Path, *, include_dev: bool = True
) -> tuple[tuple[DataTask, ...], tuple[DataTask, ...]]:
    """Load precisely the declared Train and optional Dev bodies, never Public."""
    train_ids, dev_ids = _partition_ids(split_file, include_dev=include_dev)
    loaded = {
        item.task_id: item
        for item in load_tasks_by_id(tasks_file, [*train_ids, *dev_ids])
    }

    def convert(ids: tuple[str, ...], label: str) -> tuple[DataTask, ...]:
        tasks: list[DataTask] = []
        for task_id in ids:
            source = loaded.get(task_id)
            if source is None:
                raise ValueError(f"missing {label} task {task_id}")
            tasks.append(source)
        return tuple(tasks)

    return convert(train_ids, "train"), convert(dev_ids, "dev")


def _source_files(repo: Path) -> tuple[tuple[str, Path], ...]:
    required = {
        "dictionary": repo / "dictionary.py",
        "methods": repo / "methods.py",
        "policies": repo / "policies.py",
    }
    if (repo / "skills.py").is_file():
        required["skills"] = repo / "skills.py"
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError("Champion source is missing " + ", ".join(missing))
    return tuple(sorted(required.items()))


def _inventory(module, portfolio) -> ToolDictionary:
    definitions = [
        MethodDefinition(
            method.name,
            "statistical",
            method.docstring or "Executable statistical candidate.",
        )
        for method in module.methods
    ]
    definitions.extend(
        MethodDefinition(
            policy.name,
            "foundation" if policy in portfolio.tsfm else "combined",
            "Executable reviewed candidate.",
        )
        for policy in portfolio.all_policies
    )
    return ToolDictionary("champion_runtime_inventory", None, 0, tuple(definitions))


def _load_parent(path: Path) -> ChampionRelease:
    if not path.is_file():
        raise ValueError("normal evolution requires an immutable --parent-release")
    try:
        return parse_champion_release(read_json_object(path))
    except Exception as error:
        raise ValueError("parent release is malformed") from error


def _materialize_rows(
    store: ForecastStore,
    tasks: Iterable[DataTask],
    candidates: tuple[tuple[str, str], ...],
    split: str,
) -> tuple[ChampionTaskRow, ...]:
    rows: list[ChampionTaskRow] = []
    for source in tasks:
        task = RuntimeTask(
            source.task_id,
            tuple(source.history_values),
            source.prediction_length,
            source.frequency,
            tuple(source.future_values),
        )
        profile = profile_task(task)
        for name, family in candidates:
            try:
                forecast = store.forecast(
                    name, task.history, task.horizon, task.frequency
                )
                failure = None
            except Exception as error:
                forecast, failure = None, f"{type(error).__name__}: {error}"[:10000]
            diagnostic = ChampionHistoryDiagnostic.from_candidate(
                diagnose_candidate(
                    task,
                    name,
                    family,
                    store.forecast,
                    HindcastConfig(),
                    runtime_settings={"forecast_store": store.identity_hash},
                )
            )
            rows.append(
                ChampionTaskRow(
                    task.task_id,
                    name,
                    profile,
                    tuple(task.future),
                    forecast,
                    failure,
                    0,
                    split,
                    tuple(task.history),
                    diagnostic,
                )
            )
    return tuple(rows)


def _manifest(
    train: tuple[DataTask, ...],
    dev: tuple[DataTask, ...],
    config: ChampionEvolutionConfig,
    attestations: ChampionRunAttestations,
    seed: int,
) -> ChampionRunManifest:
    parts = partition_train_tasks(train, build_size=64, calibration_size=16, seed=seed)
    return ChampionRunManifest(
        schema_version=1,
        partition_seed=seed,
        source_hashes=attestations.source_hashes,
        train_tasks=tuple((task.task_id, task.entity_name) for task in train),
        train_task_hashes=tuple(
            (task.task_id, task_content_fingerprint(task)) for task in train
        ),
        dev_tasks=tuple((task.task_id, task.entity_name) for task in dev),
        dev_task_hashes=tuple(
            (task.task_id, task_content_fingerprint(task)) for task in dev
        ),
        split_manifest_fingerprint=attestations.split_manifest_fingerprint,
        build_tasks=tuple((task.task_id, task.entity_name) for task in parts.build),
        calibration_tasks=tuple(
            (task.task_id, task.entity_name) for task in parts.calibration
        ),
        dictionary_hashes=attestations.dictionary_hashes,
        forecast_store_fingerprint=attestations.forecast_store_fingerprint,
        metric_policy_fingerprint=METRIC_POLICY_FINGERPRINT,
        proposal_model=attestations.proposal_model,
        proposal_config_fingerprint=attestations.proposal_config_fingerprint,
        proposal_implementation_fingerprint=attestations.proposal_implementation_fingerprint,
        row_provider_fingerprint=attestations.row_provider_fingerprint,
        schedule_fingerprint=config.fingerprint,
        numeric_grid_fingerprint=attestations.numeric_grid_fingerprint,
        candidate_minimum_gain=config.candidate_minimum_gain,
        research_target_gain=config.research_target_gain,
        runtime_fingerprint=attestations.runtime_fingerprint,
        runtime_implementation_fingerprint=attestations.runtime_implementation_fingerprint,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.provision_authority:
        provisioned_authority = ChampionAuthorityStore.provision(
            args.authority_root, authority_identity=args.authority_identity
        )
        provisioned_authority.close()
        print("authority provisioned")
        return 0
    _require_normal_arguments(args)
    # Opening this first makes missing/replaced authority fail before a task body,
    # model runtime, or mutable forecast cache can be touched.
    authority_preflight = ChampionAuthorityStore(
        args.authority_root, expected_authority_identity=args.authority_identity
    )
    authority_preflight.close()
    train, dev = load_evolution_partitions(
        args.split_file, args.tasks_file, include_dev=args.include_dev
    )
    if len(train) != 80 or (args.include_dev and len(dev) != 20):
        raise ValueError("formal evolution requires exactly 80 Train and 20 Dev tasks")
    if not args.include_dev:
        raise ValueError(
            "formal evolution cannot open without its registered Dev partition"
        )
    repo, output = Path(args.repo).resolve(), Path(args.output_dir).resolve()
    _clean_git_source(repo)
    sources = _source_files(repo)
    parent = _load_parent(
        Path(args.parent_release)
        if args.parent_release
        else output / "champion_release.json"
    )
    parts = partition_train_tasks(
        train, build_size=64, calibration_size=16, seed=args.partition_seed
    )
    config = ChampionEvolutionConfig(
        generations=args.generations,
        candidate_minimum_gain=float(args.candidate_minimum_gain),
        research_target_gain=float(args.research_target_gain),
        build_task_ids=tuple(task.task_id for task in parts.build),
        screen_task_ids=(
            tuple(task.task_id for task in parts.build[:8]),
            tuple(task.task_id for task in parts.build[:32]),
            tuple(task.task_id for task in parts.build),
        ),
    )
    module, portfolio = read_module(repo / "methods.py"), read_policy_file(
        repo / "policies.py"
    )
    portfolio.validate_namespace(module.names())
    inventory = _inventory(module, portfolio)
    proposer = ChampionProposerAdapter.codex_cli(
        identity=args.proposer_model,
        model=args.proposer_model,
        reasoning_effort=args.proposer_reasoning_effort,
        inventory=inventory,
        timeout_seconds=args.proposer_timeout,
        cache_dir=args.proposer_cache_dir,
    )
    runtimes = _runtime_registry(args)
    store: ForecastStore | None = None
    artifacts: ChampionArtifactStore | None = None
    authority: ChampionAuthorityStore | None = None
    try:
        cache_root = Path(args.forecast_store or output / "forecast-store")
        store = ForecastStore(
            cache_root,
            repo / "methods.py",
            repo / "skills.py" if (repo / "skills.py").is_file() else None,
            portfolio,
            runtimes,
            screening_hash="formal_champion_all_candidates",
            runtime_identity=_forecast_runtime_identity(args),
        )
        candidates = tuple(
            (method.name, "statistical") for method in module.methods
        ) + tuple(
            (policy.name, "tsfm" if policy in portfolio.tsfm else "combined")
            for policy in portfolio.all_policies
        )
        rows = _materialize_rows(
            store, (*parts.build, *parts.calibration), candidates, "build"
        )
        rows += _materialize_rows(store, parts.calibration, candidates, "calibration")
        rows += _materialize_rows(store, dev, candidates, "dev")
        provider = ChampionRowProviderAdapter.materialized(
            identity="materialized_forecast_store",
            rows=rows,
            config={"forecast_store": store.identity_hash},
        )
        runtime = ChampionRuntimeBindings.formal()
        attestations = ChampionRunAttestations(
            source_files=sources,
            split_manifest_file=Path(args.split_file),
            dictionary_files=sources,
            forecast_store=cache_root,
            proposal_model=args.proposer_model,
            proposal_config=proposer.config,
            proposer_binding=proposer,
            row_provider_binding=provider,
            runtime_bindings=runtime,
            numeric_grid={"formal": "champion_grid_v1"},
            runtime_files=(
                (
                    "champion_controller",
                    Path(__file__).parent / "evolution" / "champion_controller.py",
                ),
            ),
        )
        manifest = _manifest(train, dev, config, attestations, args.partition_seed)
        artifacts = ChampionArtifactStore(output)
        authority = ChampionAuthorityStore(
            args.authority_root, expected_authority_identity=args.authority_identity
        )
        controller = ChampionEvolutionController(
            manifest=manifest,
            config=config,
            attestations=attestations,
            proposer=proposer,
            row_provider=provider,
            runtime_bindings=runtime,
            artifact_store=artifacts,
            authority_store=authority,
        )
        outcome = controller.evolve(parent, train, dev)
        print(
            json.dumps(
                {
                    "train": len(train),
                    "calibration": len(parts.calibration),
                    "dev": len(dev),
                    "release": outcome.release.policy.recipe.name,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        if store is not None:
            store.close()
        if authority is not None:
            authority.close()
        if artifacts is not None:
            artifacts.close()
        runtimes.close()


if __name__ == "__main__":
    raise SystemExit(main())
