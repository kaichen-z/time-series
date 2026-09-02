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
from .evolution.champion import ChampionRelease, champion_fingerprint, parse_champion_release
from .evolution.champion_controller import (
    ChampionArtifactStore,
    ChampionAuthorityStore,
    ChampionEvolutionConfig,
    ChampionEvolutionController,
    ChampionEvolutionOutcome,
    ChampionProposerAdapter,
    ChampionRowProviderAdapter,
    ChampionRunAttestations,
    ChampionRunManifest,
    ChampionRuntimeBindings,
    partition_train_tasks,
    task_content_fingerprint,
)
from .evolution.champion_evidence import (
    ChampionEvidenceError,
    ChampionHistoryDiagnostic,
    ChampionTaskRow,
)
from .evolution.champion_evidence import ChampionGateConfig
from .evolution.execution import Task as RuntimeTask
from .evolution.forecast_store import ForecastStore
from .evolution.module import read_module
from .evolution.numerical_selector import HindcastConfig, diagnose_candidate
from .evolution.portfolio import read_policy_file
from .evolution.screening import (
    ScreeningPolicy,
    TaskProfile,
    materialize_active_dictionary,
    profile_task,
)
from .evolution.screening_evolution import parse_screening_source
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
    parser.add_argument("--maximum-task-regret-smae", type=float, default=0.25)
    parser.add_argument("--maximum-task-regret-srmse", type=float, default=0.25)
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


def _formal_gate_config(args: argparse.Namespace) -> ChampionGateConfig:
    return ChampionGateConfig(
        maximum_task_regret_smae=float(args.maximum_task_regret_smae),
        maximum_task_regret_srmse=float(args.maximum_task_regret_srmse),
        minimum_improved_folds=4,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_git_source(repo: Path) -> None:
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if inside.returncode or inside.stdout.strip() != "true":
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


def _load_screening_policy(path: str | Path) -> ScreeningPolicy:
    try:
        return parse_screening_source(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError("Champion dictionary.py is not a valid screening policy") from error


def _inventory(module, portfolio, screening: ScreeningPolicy) -> ToolDictionary:
    runtime = {
        method.name: (
            "statistical",
            method.docstring or "Executable statistical candidate.",
        )
        for method in module.methods
    }
    runtime.update(
        {
            policy.name: (
                "foundation" if policy in portfolio.tsfm else "combined",
                "Executable reviewed candidate.",
            )
            for policy in portfolio.all_policies
        }
    )
    entries = {entry.name: entry for entry in screening.entries}
    if set(entries) != set(runtime):
        missing = sorted(set(runtime) - set(entries))
        extra = sorted(set(entries) - set(runtime))
        raise ValueError(
            f"Champion dictionary namespace mismatch: missing={missing}, extra={extra}"
        )
    status_map = {
        "keep": "accepted",
        "specialized": "specialized",
        "repair": "quarantined",
        "quarantine": "quarantined",
        "discard": "discarded",
    }
    definitions = []
    for name, (family, description) in runtime.items():
        entry = entries[name]
        expected_family = "tsfm" if family == "foundation" else family
        if entry.family != expected_family:
            raise ValueError(
                f"Champion dictionary family mismatch for {name!r}: "
                f"{entry.family!r} != {expected_family!r}"
            )
        definitions.append(
            MethodDefinition(
                name,
                family,
                description,
                status=status_map[entry.status],
            )
        )
    return ToolDictionary("champion_runtime_inventory", None, 0, tuple(definitions))


def _screened_candidates(
    screening: ScreeningPolicy,
    profile: TaskProfile,
    candidates: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    active = materialize_active_dictionary(screening, profile)
    active_names = {item.name for item in active.active}
    return tuple(item for item in candidates if item[0] in active_names)


def _balanced_task_folds(tasks: Iterable[object], *, seed: int) -> dict[str, int]:
    identified: list[tuple[str, str]] = []
    for task in tasks:
        task_id = getattr(task, "task_id", None)
        entity = getattr(task, "entity_name", None)
        if type(task_id) is not str or not task_id or type(entity) is not str:
            raise ValueError("fold assignment requires task_id and entity_name strings")
        identified.append((task_id, entity))
    if len({task_id for task_id, _entity in identified}) != len(identified):
        raise ValueError("fold assignment requires unique task IDs")
    ranked = sorted(
        identified,
        key=lambda item: hashlib.sha256(
            f"{seed}\0{item[0]}\0{item[1]}".encode("utf-8")
        ).hexdigest(),
    )
    return {task_id: index % 5 for index, (task_id, _entity) in enumerate(ranked)}


def _fold_stratified_screen_task_ids(
    tasks: Iterable[object],
    folds: dict[str, int],
    *,
    sizes: tuple[int, ...],
) -> tuple[tuple[str, ...], ...]:
    task_ids = tuple(getattr(task, "task_id", None) for task in tasks)
    if any(type(task_id) is not str or not task_id for task_id in task_ids):
        raise ValueError("screen assignment requires nonempty task IDs")
    if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(folds):
        raise ValueError("screen assignment must cover the exact unique task universe")
    if (
        type(sizes) is not tuple
        or not sizes
        or any(type(size) is not int or size <= 0 for size in sizes)
        or tuple(sorted(sizes)) != sizes
        or sizes[-1] != len(task_ids)
    ):
        raise ValueError("screen sizes must be increasing and end at the task universe")
    buckets: dict[int, list[str]] = {fold: [] for fold in range(5)}
    for task_id in task_ids:
        fold = folds[task_id]
        if fold not in buckets:
            raise ValueError("formal screen assignment requires folds zero through four")
        buckets[fold].append(task_id)
    if any(not bucket for bucket in buckets.values()):
        raise ValueError("formal screen assignment requires all five folds")
    for bucket in buckets.values():
        bucket.sort()
    ordered: list[str] = []
    offset = 0
    while len(ordered) < len(task_ids):
        for fold in range(5):
            bucket = buckets[fold]
            if offset < len(bucket):
                ordered.append(bucket[offset])
        offset += 1
    screens = [tuple(ordered[:size]) for size in sizes]
    screens[-1] = task_ids
    return tuple(screens)


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
    screening: ScreeningPolicy,
    folds: dict[str, int],
    split: str,
) -> tuple[ChampionTaskRow, ...]:
    rows: list[ChampionTaskRow] = []
    entries = {entry.name: entry for entry in screening.entries}
    reviewed_candidates = tuple(
        (name, family)
        for name, family in candidates
        if entries[name].status in {"keep", "specialized"}
    )
    for source in tasks:
        task = RuntimeTask(
            source.task_id,
            tuple(source.history_values),
            source.prediction_length,
            source.frequency,
            tuple(source.future_values),
        )
        profile = profile_task(task)
        active_names = {
            name for name, _family in _screened_candidates(
                screening, profile, reviewed_candidates
            )
        }
        for name, family in reviewed_candidates:
            if name not in active_names:
                rows.append(
                    ChampionTaskRow(
                        task.task_id,
                        name,
                        profile,
                        tuple(task.future),
                        None,
                        "NotApplicable: screening_policy",
                        folds[task.task_id],
                        split,
                        tuple(task.history),
                        None,
                    )
                )
                continue
            try:
                forecast = store.forecast(
                    name, task.history, task.horizon, task.frequency
                )
                failure = None
            except Exception as error:
                forecast, failure = None, f"{type(error).__name__}: {error}"[:10000]
            candidate_diagnostic = diagnose_candidate(
                task,
                name,
                family,
                store.forecast,
                HindcastConfig(),
                runtime_settings={"forecast_store": store.identity_hash},
            )
            try:
                diagnostic = ChampionHistoryDiagnostic.from_candidate(
                    candidate_diagnostic
                )
            except ChampionEvidenceError:
                diagnostic = None
                if failure is None:
                    forecast = None
                    failure = (
                        "HistoryDiagnosticUnavailable: "
                        f"{candidate_diagnostic.reason_code}"
                    )[:10000]
            rows.append(
                ChampionTaskRow(
                    task.task_id,
                    name,
                    profile,
                    tuple(task.future),
                    forecast,
                    failure,
                    folds[task.task_id],
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
        forecast_runtime_identity_fingerprint=attestations.forecast_runtime_identity_fingerprint,
    )


def _smoke_parent(source_hashes: tuple[tuple[str, str], ...]) -> ChampionRelease:
    """Return the fixed non-Toto Parent used only by deterministic wiring smoke."""
    from .evolution.champion import ChampionRecipe, EvolutionAssumption, FittedChampionPolicy

    assumption = EvolutionAssumption(
        assumption_id="timesfm_parent_history",
        candidate_name="timesfm_2_5",
        feature="history_length",
        direction="above",
        horizon_region="full",
        operator="select",
        rationale="The smoke Parent uses a host-materialized TimesFM forecast.",
        failure_condition="The host-materialized TimesFM forecast is unavailable.",
    )
    return ChampionRelease(
        policy=FittedChampionPolicy(
            recipe=ChampionRecipe(
                name="timesfm_smoke_parent",
                kind="select",
                parents=("timesfm_2_5",),
                fallback_parent="timesfm_2_5",
                assumptions=(assumption,),
            ),
            thresholds=((assumption.assumption_id, 0.0),),
        ),
        source_hashes=source_hashes,
        metric_policy_fingerprint=METRIC_POLICY_FINGERPRINT,
        lineage=("timesfm_smoke_parent",),
    )


def _smoke_rows(
    tasks: Iterable[DataTask],
    *,
    split: str,
    timesfm_error: float,
    seasonal_error: float,
) -> tuple[ChampionTaskRow, ...]:
    """Materialize fixed fake leaves through the same typed row-provider contract."""
    rows: list[ChampionTaskRow] = []
    for index, source in enumerate(tasks):
        task = RuntimeTask(
            source.task_id,
            tuple(source.history_values),
            source.prediction_length,
            source.frequency,
            tuple(source.future_values),
        )
        profile = profile_task(task)
        for name, family, error in (
            ("timesfm_2_5", "tsfm", timesfm_error),
            ("seasonal_naive", "statistical", seasonal_error),
        ):
            diagnostic = ChampionHistoryDiagnostic.from_candidate(
                diagnose_candidate(
                    task,
                    name,
                    family,
                    lambda _name, _history, horizon, _frequency, offset=error: tuple(
                        value + offset for value in task.future[:horizon]
                    ),
                    HindcastConfig(),
                    runtime_settings={"smoke": "deterministic"},
                )
            )
            rows.append(
                ChampionTaskRow(
                    task_id=task.task_id,
                    candidate_name=name,
                    profile=profile,
                    truth=tuple(task.future),
                    forecast=tuple(value + error for value in task.future),
                    failure_reason=None,
                    fold=index % 5,
                    split=split,  # type: ignore[arg-type]
                    history=tuple(task.history),
                    diagnostic=diagnostic,
                )
            )
    return tuple(rows)


def run_fake_champion_evolution(
    *,
    build_tasks: tuple[DataTask, ...],
    calibration_tasks: tuple[DataTask, ...],
    dev_tasks: tuple[DataTask, ...],
    proposal: object,
    screen_sizes: tuple[int, ...],
    output_dir: str | Path,
    timesfm_error: float = 2.0,
    seasonal_error: float = 0.2,
    dev_seasonal_error: float | None = None,
) -> ChampionEvolutionOutcome:
    """Run the sealed 8/2 fake lifecycle without a model download or LLM call.

    This is deliberately a wiring-only smoke entry point.  It accepts only the
    registered 8/2 schedule and creates the same sealed adapters, manifest,
    checkpoint, split authority, and release artifacts as the formal runner.
    """
    if type(build_tasks) is not tuple or type(calibration_tasks) is not tuple:
        raise ValueError("fake smoke requires exact Build and Calibration task tuples")
    if type(dev_tasks) is not tuple or len(build_tasks) != 8 or len(calibration_tasks) != 2:
        raise ValueError("fake smoke requires exactly 8 Build and 2 Calibration tasks")
    if len(dev_tasks) != 2 or screen_sizes != (4, 8):
        raise ValueError("fake smoke requires exactly 2 Dev tasks and screens (4, 8)")
    if (
        type(timesfm_error) is not float
        or type(seasonal_error) is not float
        or (dev_seasonal_error is not None and type(dev_seasonal_error) is not float)
    ):
        raise ValueError("fake smoke forecast errors must be exact floats")
    output = Path(output_dir).resolve()
    inputs = output.parent / f"{output.name}.smoke-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    contents = {
        "split.json": b'{"mode":"deterministic_8_2_smoke"}\n',
        "seasonal_naive.py": b"# deterministic smoke seasonal leaf\n",
        "timesfm_2_5.py": b"# deterministic smoke TimesFM leaf\n",
        "forecast.store": b"deterministic fake forecast store\n",
        "runtime.py": b"deterministic fake runtime\n",
    }
    for name, content in contents.items():
        path = inputs / name
        if not path.exists():
            path.write_bytes(content)
    train = build_tasks + calibration_tasks
    parts = partition_train_tasks(train, build_size=8, calibration_size=2, seed=20260901)
    config = ChampionEvolutionConfig(
        build_size=8,
        calibration_size=2,
        screen_sizes=screen_sizes,
        build_task_ids=tuple(task.task_id for task in parts.build),
        screen_task_ids=(
            tuple(task.task_id for task in parts.build[:4]),
            tuple(task.task_id for task in parts.build),
        ),
        gate_config=ChampionGateConfig(minimum_improved_folds=0),
    )
    if type(proposal) is not tuple:
        raise ValueError("fake smoke proposal must be an exact recipe tuple")
    proposer = ChampionProposerAdapter.scripted(
        identity="deterministic-fake-llm",
        proposal_batches=(proposal,),
        config={"mode": "deterministic_8_2_smoke"},
    )
    rows = (
        _smoke_rows(parts.build, split="build", timesfm_error=timesfm_error, seasonal_error=seasonal_error)
        + _smoke_rows(parts.calibration, split="calibration", timesfm_error=timesfm_error, seasonal_error=seasonal_error)
        + _smoke_rows(
            dev_tasks,
            split="dev",
            timesfm_error=timesfm_error,
            seasonal_error=(
                seasonal_error if dev_seasonal_error is None else dev_seasonal_error
            ),
        )
    )
    provider = ChampionRowProviderAdapter.materialized(
        identity="deterministic-fake-forecast-store",
        rows=rows,
        config={"mode": "deterministic_8_2_smoke"},
    )
    runtime = ChampionRuntimeBindings.formal()
    dictionary_files = (
        ("seasonal_naive", inputs / "seasonal_naive.py"),
        ("timesfm_2_5", inputs / "timesfm_2_5.py"),
    )
    attestations = ChampionRunAttestations(
        source_files=dictionary_files,
        split_manifest_file=inputs / "split.json",
        dictionary_files=dictionary_files,
        forecast_store=inputs / "forecast.store",
        proposal_model=proposer.identity,
        proposal_config=proposer.config,
        proposer_binding=proposer,
        row_provider_binding=provider,
        runtime_bindings=runtime,
        numeric_grid={"mode": "deterministic_8_2_smoke"},
        runtime_files=(("deterministic_runtime", inputs / "runtime.py"),),
        forecast_runtime_identity={"mode": "deterministic_8_2_smoke"},
    )
    manifest = ChampionRunManifest(
        schema_version=1,
        partition_seed=20260901,
        source_hashes=attestations.source_hashes,
        train_tasks=tuple((task.task_id, task.entity_name) for task in train),
        train_task_hashes=tuple((task.task_id, task_content_fingerprint(task)) for task in train),
        dev_tasks=tuple((task.task_id, task.entity_name) for task in dev_tasks),
        dev_task_hashes=tuple((task.task_id, task_content_fingerprint(task)) for task in dev_tasks),
        split_manifest_fingerprint=attestations.split_manifest_fingerprint,
        build_tasks=tuple((task.task_id, task.entity_name) for task in parts.build),
        calibration_tasks=tuple((task.task_id, task.entity_name) for task in parts.calibration),
        dictionary_hashes=attestations.dictionary_hashes,
        forecast_store_fingerprint=attestations.forecast_store_fingerprint,
        metric_policy_fingerprint=METRIC_POLICY_FINGERPRINT,
        proposal_model=proposer.identity,
        proposal_config_fingerprint=attestations.proposal_config_fingerprint,
        proposal_implementation_fingerprint=attestations.proposal_implementation_fingerprint,
        row_provider_fingerprint=attestations.row_provider_fingerprint,
        schedule_fingerprint=config.fingerprint,
        numeric_grid_fingerprint=attestations.numeric_grid_fingerprint,
        candidate_minimum_gain=config.candidate_minimum_gain,
        research_target_gain=config.research_target_gain,
        runtime_fingerprint=attestations.runtime_fingerprint,
        runtime_implementation_fingerprint=attestations.runtime_implementation_fingerprint,
        forecast_runtime_identity_fingerprint=attestations.forecast_runtime_identity_fingerprint,
    )
    authority_root = output.parent / f"{output.name}.smoke-authority"
    authority_identity = champion_fingerprint({"smoke_authority": str(authority_root)})
    if authority_root.exists():
        authority = ChampionAuthorityStore(
            authority_root, expected_authority_identity=authority_identity
        )
    else:
        authority = ChampionAuthorityStore.provision(
            authority_root, authority_identity=authority_identity
        )
    artifacts = ChampionArtifactStore(output)
    try:
        return ChampionEvolutionController(
            manifest=manifest,
            config=config,
            attestations=attestations,
            proposer=proposer,
            row_provider=provider,
            runtime_bindings=runtime,
            artifact_store=artifacts,
            authority_store=authority,
        ).evolve(_smoke_parent(attestations.dictionary_hashes), train, dev_tasks)
    finally:
        authority.close()
        artifacts.close()


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
    build_folds = _balanced_task_folds(parts.build, seed=args.partition_seed)
    calibration_folds = _balanced_task_folds(
        parts.calibration, seed=args.partition_seed
    )
    dev_folds = _balanced_task_folds(dev, seed=args.partition_seed)
    config = ChampionEvolutionConfig(
        generations=args.generations,
        candidate_minimum_gain=float(args.candidate_minimum_gain),
        research_target_gain=float(args.research_target_gain),
        gate_config=_formal_gate_config(args),
        build_task_ids=tuple(task.task_id for task in parts.build),
        screen_task_ids=_fold_stratified_screen_task_ids(
            parts.build,
            build_folds,
            sizes=(8, 32, 64),
        ),
    )
    module, portfolio = read_module(repo / "methods.py"), read_policy_file(
        repo / "policies.py"
    )
    portfolio.validate_namespace(module.names())
    screening = _load_screening_policy(repo / "dictionary.py")
    inventory = _inventory(module, portfolio, screening)
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
            screening_hash=screening.fingerprint(),
            runtime_identity=_forecast_runtime_identity(args),
        )
        candidates = tuple(
            (method.name, "statistical") for method in module.methods
        ) + tuple(
            (policy.name, "tsfm" if policy in portfolio.tsfm else "combined")
            for policy in portfolio.all_policies
        )
        rows = _materialize_rows(
            store,
            parts.build,
            candidates,
            screening,
            build_folds,
            "build",
        )
        rows += _materialize_rows(
            store,
            parts.calibration,
            candidates,
            screening,
            calibration_folds,
            "calibration",
        )
        rows += _materialize_rows(
            store,
            dev,
            candidates,
            screening,
            dev_folds,
            "dev",
        )
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
            forecast_runtime_identity=_forecast_runtime_identity(args),
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
