"""CLI composition for parameterized numerical self-evolution experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

from common.evolution_core.contracts import (
    EvolutionConfig,
    MetricSpec,
    require_active_metric_policy,
)
from common.evolution_core.controller import SelfEvolutionEngine
from common.llm import (
    ClaudeCLIClient,
    ClaudeCLIConfig,
    CodexCLIClient,
    CodexCLIConfig,
    QwenClient,
)
from common.metrics import drcik_point_metrics, mae, smape
from common.payload import read_json_object, require_object, write_json
from common.tracing import configure

from .curation import (
    DictionaryArtifactAdapter,
    DictionaryCurationTask,
    DictionaryEvaluator,
    DictionaryExecutor,
    NumericalTaskItem,
)
from .curation.codegen import (
    SANDBOX_PROVIDER,
    FamilyRoutingImplementer,
    LLMMethodImplementer,
    SandboxMethodRuntime,
)
from .collection.catalog_adapter import tool_dictionary_from_payload
from .collection.coverage import audit_coverage, audit_saturation
from .collection.contracts import MethodCard, SourceRecord
from .collection.normalization import find_duplicate_candidates
from .collection.registry import (
    build_release,
    load_method_cards,
    load_source_records,
    write_release,
)
from .collection.verification import verify_registry
from .config import DictionaryCurationConfig
from .dictionary import MethodRecord
from .experiment import build_experiment, build_frozen_test
from .curation.persistence import MethodSourceArtifactStore
from .providers import RuntimeRegistry
from .curation.smoke import FixtureMethodImplementer, FixtureMethodRuntime
from .tsfm import ChronosRuntime, TimesFMRuntime
from .tsfm.broker import WorkerBroker, WorkerMethodRuntime
from .tsfm.deployment import TSFMDeployment, parse_acknowledged_licenses
from .tsfm.manifests import ManifestRegistry
from .tsfm.security import SecretRedactor


APPROVED_PROVIDERS = ("fake", "llm")
LLM_BACKENDS = ("codex", "qwen", "claude")
TSFM_RUNTIMES = ("chronos", "timesfm")


def _add_tsfm_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tsfm-runtimes",
        default="",
        help="comma-separated optional runtimes: chronos,timesfm",
    )
    parser.add_argument("--chronos-device-map", default="cpu")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument(
        "--tsfm-workers-config",
        default=os.environ.get("NA_TSFM_WORKERS_CONFIG") or None,
    )
    parser.add_argument(
        "--acknowledged-model-licenses",
        default=os.environ.get("NA_ACCEPT_MODEL_LICENSES", ""),
        help="comma-separated exact third-party model license identifiers",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    curate = subparsers.add_parser("curate", help="run dictionary-curation self-evolution")
    curate.add_argument("--experiment-config", required=True)
    curate.add_argument("--base-methods", required=True)
    curate.add_argument("--provider", required=True)
    curate.add_argument("--output-dir", required=True)
    curate.add_argument("--llm-backend", choices=LLM_BACKENDS, default=None)
    curate.add_argument("--codex-model", default=None)
    curate.add_argument(
        "--codex-reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default=None,
    )
    curate.add_argument("--codex-cache-dir", default=None)
    curate.add_argument("--codex-timeout", type=int, default=None)
    curate.add_argument("--claude-model", default=None)
    curate.add_argument("--claude-cache-dir", default=None)
    curate.add_argument("--claude-timeout", type=int, default=None)
    curate.add_argument("--model-id", default=None)
    curate.add_argument("--device", default=None)
    _add_tsfm_runtime_options(curate)

    build = subparsers.add_parser(
        "build-experiment", help="build a curation experiment config from a frozen split"
    )
    build.add_argument("--tasks-file", required=True)
    build.add_argument("--split-file", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--generations", type=int, default=1)
    build.add_argument("--children-per-generation", type=int, default=1)
    build.add_argument("--seed", type=int, default=20260816)
    build.add_argument("--max-revisions-per-method", type=int, default=1)
    build.add_argument("--max-implementation-attempts", type=int, default=3)
    build.add_argument("--accepted-max-error", type=float, default=50.0)
    build.add_argument("--specialized-max-error", type=float, default=100.0)
    build.add_argument("--min-success-rate", type=float, default=0.8)
    build.add_argument("--selection-folds", type=int, default=3)
    build.add_argument("--selection-horizon", type=int, default=8)
    build.add_argument("--train-limit", type=int, default=None)
    build.add_argument("--dev-limit", type=int, default=None)

    frozen = subparsers.add_parser(
        "evaluate-frozen",
        help="score a frozen working dictionary once on the sealed Public Test split",
    )
    frozen.add_argument("--tasks-file", required=True)
    frozen.add_argument("--split-file", required=True)
    frozen.add_argument("--experiment-config", required=True)
    frozen.add_argument("--dictionary", required=True)
    frozen.add_argument("--output-dir", required=True)
    _add_tsfm_runtime_options(frozen)

    collect_methods = subparsers.add_parser(
        "collect-methods", help="normalize collected source and method manifests"
    )
    collect_methods.add_argument("--sources", required=True)
    collect_methods.add_argument("--methods", required=True)
    collect_methods.add_argument("--output-dir", required=True)

    verify_methods = subparsers.add_parser(
        "verify-methods", help="audit method provenance and taxonomy coverage"
    )
    verify_methods.add_argument("--sources", required=True)
    verify_methods.add_argument("--methods", required=True)
    verify_methods.add_argument("--queries", required=True)
    verify_methods.add_argument("--output", required=True)

    build_dataset = subparsers.add_parser(
        "build-dataset", help="publish a verified deterministic method dataset"
    )
    build_dataset.add_argument("--sources", required=True)
    build_dataset.add_argument("--methods", required=True)
    build_dataset.add_argument("--queries", required=True)
    build_dataset.add_argument("--collection-journal", required=True)
    build_dataset.add_argument("--output", required=True)
    build_dataset.add_argument("--audit-output", required=True)
    build_dataset.add_argument("--sha256-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-experiment":
        return _build_experiment(args)
    if args.command == "evaluate-frozen":
        return _evaluate_frozen(args)
    if args.command == "collect-methods":
        return _collect_methods(args)
    if args.command == "verify-methods":
        return _verify_methods(args)
    if args.command == "build-dataset":
        return _build_dataset(args)
    if args.provider not in APPROVED_PROVIDERS:
        parser.error(
            f"provider must be an approved provider name: {', '.join(APPROVED_PROVIDERS)}"
        )
    if args.provider == "fake" and args.llm_backend is not None:
        parser.error("--llm-backend applies only to --provider llm")
    if args.command != "curate":
        parser.error(f"unsupported command: {args.command}")

    experiment = read_json_object(Path(args.experiment_config))
    curation = _curation_config(experiment)
    dictionary = tool_dictionary_from_payload(
        read_json_object(Path(args.base_methods)),
        allowed_families=curation.allowed_families,
    )
    evolution = _evolution_config(experiment, curation)
    train_items, dev_items = _task_items(experiment)
    labels = _labels(experiment)
    output_dir = Path(args.output_dir)
    configure(output_dir / "curation_trace.jsonl")
    store = MethodSourceArtifactStore(output_dir)

    implementer, runtimes = _providers(args.provider, args)
    try:
        task = DictionaryCurationTask(
            base_dictionary=dictionary,
            config=curation,
            implementer=implementer, # type: ignore
            runtimes=runtimes,
            labels=labels,
            metric=_metric(curation.method_metric),
            store=store,
        )
        engine = SelfEvolutionEngine(evolution, task.components())
        outcome = engine.evolve(dictionary, train_items, dev_items)
    finally:
        runtimes.close()
    best = outcome.accepted_artifact
    store.save_artifact("working_dictionary", best.to_payload())
    _write_method_evaluations(output_dir, outcome.steps)
    quarantined = [
        record.to_payload()
        for record in best.methods
        if isinstance(record, MethodRecord)
        and record.status in ("quarantined", "unavailable", "discarded")
    ]
    store.save_artifact("quarantine", {"methods": quarantined})

    summary = {
        "accepted_dictionary_id": best.dictionary_id,
        "generation": best.generation,
        "method_count": len(best.methods),
        "steps": len(outcome.steps),
        "resumed_from_generation": outcome.resumed_from_generation,
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def _build_experiment(args: argparse.Namespace) -> int:
    experiment = build_experiment(
        tasks_file=args.tasks_file,
        split_file=args.split_file,
        generations=args.generations,
        children_per_generation=args.children_per_generation,
        seed=args.seed,
        max_revisions_per_method=args.max_revisions_per_method,
        max_implementation_attempts=args.max_implementation_attempts,
        accepted_max_error=args.accepted_max_error,
        specialized_max_error=args.specialized_max_error,
        min_success_rate=args.min_success_rate,
        selection_folds=args.selection_folds,
        selection_horizon=args.selection_horizon,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output": str(destination),
        "train_tasks": len(experiment["tasks"]["train"]), # type: ignore
        "dev_tasks": len(experiment["tasks"]["dev"]), # type: ignore
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def _evaluate_frozen(args: argparse.Namespace) -> int:
    experiment = read_json_object(Path(args.experiment_config))
    curation = _curation_config(experiment)
    dictionary_path = Path(args.dictionary)
    dictionary_sha256 = hashlib.sha256(dictionary_path.read_bytes()).hexdigest()
    dictionary = tool_dictionary_from_payload(
        read_json_object(dictionary_path),
        allowed_families=curation.allowed_families,
    )
    DictionaryArtifactAdapter(curation).validate(dictionary)

    frozen = build_frozen_test(
        tasks_file=args.tasks_file,
        split_file=args.split_file,
    )
    items = _task_items_for_split(frozen, "public_test")
    labels = _labels_for_splits(frozen, ("public_test",))
    metric = _metric(curation.method_metric)
    runtimes = _runtime_registry(args)
    try:
        executor = DictionaryExecutor(
            runtimes,
            selection_metric=metric,
            selection_folds=curation.selection_folds,
            selection_horizon=curation.selection_horizon,
            min_selection_success_rate=curation.min_success_rate,
        )
        results = tuple(executor.execute(dictionary, items, "public_test"))
    finally:
        runtimes.close()
    evaluator = DictionaryEvaluator(curation, labels, metric)
    report = evaluator.evaluate(dictionary.dictionary_id, results, "public_test")

    output_dir = Path(args.output_dir)
    report_path = output_dir / "frozen_test_report.json"
    if report_path.exists():
        raise FileExistsError(
            f"frozen test report already exists and will not be overwritten: {report_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "artifact_id": report.artifact_id,
        "split": report.split,
        "item_count": report.item_count,
        "metrics": dict(report.metrics),
        "diagnostics": dict(report.diagnostics),
        "manifest_sha256": frozen["manifest_sha256"],
        "dictionary_sha256": dictionary_sha256,
    }
    with (output_dir / "frozen_test_forecasts.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for result in results:
            payload: dict[str, object] = {
                "item_id": result.item_id,
                "method_id": result.method_id,
                "status": result.status,
                "forecast": list(result.forecast),
                "selected": result.selected,
                "selection_score": result.selection_score,
            }
            if result.error:
                payload["error"] = result.error
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    # Write the report last: its presence is the immutable completion marker.
    write_json(report_path, report_payload)

    score = float(report.metrics[curation.dictionary_metric])
    summary = {
        "artifact_id": dictionary.dictionary_id,
        "metric": curation.dictionary_metric,
        "public_test_tasks": len(items),
        "score": score,
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def _collect_methods(args: argparse.Namespace) -> int:
    sources = load_source_records(args.sources)
    methods = load_method_cards(args.methods)
    duplicates = find_duplicate_candidates(methods)
    output_dir = Path(args.output_dir)
    write_json(
        output_dir / "raw_method_registry.json",
        {
            "schema_version": 1,
            "sources": [source.to_payload() for source in sources],
            "methods": [method.to_payload() for method in methods],
        },
    )
    write_json(
        output_dir / "duplicate_candidates.json",
        {"duplicate_candidates": [candidate.to_payload() for candidate in duplicates]},
    )
    summary = {
        "source_count": len(sources),
        "method_count": len(methods),
        "duplicate_candidate_count": len(duplicates),
        "output_dir": str(output_dir),
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def _query_manifest(path: str | Path) -> dict[str, object]:
    payload = read_json_object(Path(path))
    taxonomy = require_object(payload.get("taxonomy"), "query manifest taxonomy")
    return payload


def _base_audit(
    sources: Sequence[SourceRecord],
    methods: Sequence[MethodCard],
    queries: Mapping[str, object],
) -> tuple[dict[str, object], bool, bool, tuple[dict[str, object], ...]]:
    verification = verify_registry(sources, methods)
    coverage = audit_coverage(methods, queries, sources)
    duplicates = find_duplicate_candidates(methods)
    duplicate_payloads = tuple(candidate.to_payload() for candidate in duplicates)
    payload = {
        "verification": verification.to_payload(),
        "coverage": coverage.to_payload(),
        "duplicate_candidates": list(duplicate_payloads),
    }
    return (
        payload,
        verification.is_publishable,
        coverage.all_required_cells_covered,
        duplicate_payloads,
    )


def _verify_methods(args: argparse.Namespace) -> int:
    sources = load_source_records(args.sources)
    methods = load_method_cards(args.methods)
    queries = _query_manifest(args.queries)
    audit, verification_ok, coverage_ok, duplicates = _base_audit(
        sources, methods, queries
    )
    publishable = verification_ok and coverage_ok and not duplicates
    write_json(Path(args.output), audit)
    summary = {
        "publishable": publishable,
        "source_count": len(sources),
        "method_count": len(methods),
        "audit_output": str(args.output),
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if publishable else 2


def _journal(path: str | Path) -> dict[str, object]:
    payload = read_json_object(Path(path))
    base_count = payload.get("saturation_base_count")
    batches = payload.get("collection_batches")
    resolutions = payload.get("duplicate_resolutions")
    if not isinstance(base_count, int) or isinstance(base_count, bool) or base_count <= 0:
        raise ValueError("collection journal saturation_base_count must be positive")
    if not isinstance(batches, list):
        raise ValueError("collection journal collection_batches must be a list")
    if not isinstance(resolutions, list):
        raise ValueError("collection journal duplicate_resolutions must be a list")
    return payload


def _duplicate_resolution_status(
    duplicate_payloads: Sequence[Mapping[str, object]],
    journal: Mapping[str, object],
) -> tuple[bool, tuple[dict[str, object], ...]]:
    raw_resolutions = journal.get("duplicate_resolutions", [])
    if not isinstance(raw_resolutions, list):
        raise ValueError("collection journal duplicate_resolutions must be a list")
    allowed = {"distinct_wrapper", "distinct_checkpoint_variant", "not_duplicate"}
    resolved_pairs: set[tuple[str, str]] = set()
    for raw in raw_resolutions:
        if not isinstance(raw, Mapping):
            raise ValueError("duplicate resolution must be an object")
        left = str(raw.get("left_method_uid", "")).strip()
        right = str(raw.get("right_method_uid", "")).strip()
        decision = str(raw.get("decision", "")).strip()
        if not left or not right or left == right:
            raise ValueError("duplicate resolution requires two distinct method UIDs")
        if decision not in allowed:
            raise ValueError(
                "duplicate resolution must be distinct_wrapper, "
                "distinct_checkpoint_variant, or not_duplicate; same concepts must be "
                "merged in the method manifest"
            )
        resolved_pairs.add(tuple(sorted((left, right))))
    unresolved = []
    for candidate in duplicate_payloads:
        pair = tuple(
            sorted(
                (
                    str(candidate["left_method_uid"]),
                    str(candidate["right_method_uid"]),
                )
            )
        )
        if pair not in resolved_pairs:
            unresolved.append(dict(candidate))
    return not unresolved, tuple(unresolved)


def _taxonomy_for_release(queries: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    raw = require_object(queries["taxonomy"], "query manifest taxonomy")
    taxonomy = {}
    for family, categories in raw.items():
        if not isinstance(categories, Mapping):
            raise ValueError(f"query manifest taxonomy.{family} must be an object")
        taxonomy[str(family)] = tuple(sorted(str(category) for category in categories))
    return taxonomy


def _build_dataset(args: argparse.Namespace) -> int:
    sources = load_source_records(args.sources)
    methods = load_method_cards(args.methods)
    queries = _query_manifest(args.queries)
    journal = _journal(args.collection_journal)
    audit, verification_ok, coverage_ok, duplicate_payloads = _base_audit(
        sources, methods, queries
    )

    batches = cast(list[object], journal["collection_batches"])
    counts = []
    for index, raw_batch in enumerate(batches):
        if not isinstance(raw_batch, Mapping):
            raise ValueError(f"collection batch {index + 1} must be an object")
        count = raw_batch.get("new_canonical_methods")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(
                f"collection batch {index + 1} new_canonical_methods must be non-negative"
            )
        counts.append(count)
    saturation = audit_saturation(
        counts, base_count=int(journal["saturation_base_count"])
    )
    duplicates_resolved, unresolved = _duplicate_resolution_status(
        duplicate_payloads, journal
    )
    audit.update(
        {
            "saturation": saturation.to_payload(),
            "duplicate_resolutions": journal["duplicate_resolutions"],
            "unresolved_duplicate_candidates": list(unresolved),
        }
    )
    write_json(Path(args.audit_output), audit)

    publishable = (
        verification_ok
        and coverage_ok
        and duplicates_resolved
        and saturation.saturated
    )
    summary = {
        "publishable": publishable,
        "saturated": saturation.saturated,
        "source_count": len(sources),
        "method_count": len(methods),
        "unresolved_duplicate_count": len(unresolved),
        "audit_output": str(args.audit_output),
    }
    if not publishable:
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return 2

    release = build_release(
        sources,
        methods,
        dataset_id="forecast_method_dataset_v001",
        release_date="2026-08-17",
        collection_cutoff="2026-08-17",
        taxonomy=_taxonomy_for_release(queries),
        collection_batches=cast(list[Mapping[str, object]], journal["collection_batches"]),
    )
    write_release(release, args.output, args.sha256_output)
    summary["output"] = str(args.output)
    summary["sha256_output"] = str(args.sha256_output)
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def _curation_config(experiment: Mapping[str, object]) -> DictionaryCurationConfig:
    payload = require_object(experiment.get("curation", {}), "curation config")
    require_active_metric_policy(payload, context="active curation config")
    normalized = dict(payload)
    for field_name in ("allowed_actions", "allowed_families", "method_statuses"):
        if field_name in normalized:
            value = normalized[field_name]
            if not isinstance(value, list):
                raise ValueError(f"{field_name} must be a list")
            normalized[field_name] = tuple(str(item) for item in value)
    return DictionaryCurationConfig(**normalized)


def _evolution_config(
    experiment: Mapping[str, object], curation: DictionaryCurationConfig
) -> EvolutionConfig:
    payload = require_object(experiment.get("evolution", {}), "evolution config")
    allowed = {
        "generations",
        "children_per_generation",
        "seed",
        "acceptance_margin",
        "resume",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown evolution config fields: {sorted(unknown)!r}")
    return EvolutionConfig(
        **dict(payload),
        metric=MetricSpec(curation.dictionary_metric, "minimize"),
    )


def _task_items(
    experiment: Mapping[str, object],
) -> tuple[tuple[NumericalTaskItem, ...], tuple[NumericalTaskItem, ...]]:
    return (
        _task_items_for_split(experiment, "train"),
        _task_items_for_split(experiment, "dev"),
    )


def _task_items_for_split(
    experiment: Mapping[str, object], split: str
) -> tuple[NumericalTaskItem, ...]:
    tasks = require_object(experiment.get("tasks"), "tasks")
    values = tasks.get(split)
    if not isinstance(values, list):
        raise ValueError(f"tasks.{split} must be a list")
    parsed = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError(f"tasks.{split} entries must be objects")
        history = value.get("history")
        if not isinstance(history, list):
            raise ValueError("task history must be a list")
        characteristics = value.get("characteristics", [])
        if not isinstance(characteristics, list):
            raise ValueError("task characteristics must be a list")
        parsed.append(
            NumericalTaskItem(
                item_id=str(value["item_id"]),
                history=tuple(float(item) for item in history),
                horizon=int(value["horizon"]),
                frequency=str(value["frequency"]),
                characteristics=tuple(str(item) for item in characteristics),
            )
        )
    return tuple(parsed)


def _labels(
    experiment: Mapping[str, object],
) -> dict[str, dict[str, tuple[float, ...]]]:
    return _labels_for_splits(experiment, ("train", "dev"))


def _labels_for_splits(
    experiment: Mapping[str, object], splits: Sequence[str]
) -> dict[str, dict[str, tuple[float, ...]]]:
    labels = require_object(experiment.get("labels"), "labels")
    normalized: dict[str, dict[str, tuple[float, ...]]] = {}
    for split in splits:
        values = labels.get(split)
        if not isinstance(values, Mapping):
            raise ValueError(f"labels.{split} must be an object")
        normalized[split] = {}
        for item_id, truth in values.items():
            if not isinstance(truth, list):
                raise ValueError("label trajectories must be lists")
            normalized[split][str(item_id)] = tuple(float(value) for value in truth)
    return normalized


def _providers(provider: str, args: argparse.Namespace) -> tuple[object, RuntimeRegistry]:
    if provider == "fake":
        implementer = FamilyRoutingImplementer(FixtureMethodImplementer())
        runtimes = _runtime_registry(args, base_factories={"fake": FixtureMethodRuntime})
        return implementer, runtimes
    if provider == "llm":
        implementer = FamilyRoutingImplementer(
            LLMMethodImplementer(
                _llm_client(args), transcript_dir=Path(args.output_dir) / "transcripts"
            )
        )
        return implementer, _runtime_registry(args)
    raise ValueError(f"unsupported approved provider {provider!r}")


def _runtime_registry(
    args: argparse.Namespace,
    *,
    base_factories: Mapping[str, Callable[[], object]] | None = None,
) -> RuntimeRegistry:
    """Build the sandbox runtime plus any optional TSFM runtimes the caller enabled."""
    names = _comma_separated(
        getattr(args, "tsfm_runtimes", "") or "",
        allowed=TSFM_RUNTIMES,
        option="--tsfm-runtimes",
        allow_empty=True,
    )
    model_cache_dir = getattr(args, "model_cache_dir", None)
    if names and model_cache_dir:
        os.environ["HF_HOME"] = str(Path(model_cache_dir).expanduser())

    runtimes: dict[str, object] = {SANDBOX_PROVIDER: SandboxMethodRuntime()}
    for name, factory in (base_factories or {}).items():
        runtimes[name] = factory()
    if "chronos" in names:
        runtimes["chronos"] = ChronosRuntime(
            device_map=getattr(args, "chronos_device_map", "cpu")
        )
    if "timesfm" in names:
        runtimes["timesfm"] = TimesFMRuntime()

    manifests = ManifestRegistry.load_default()
    acknowledged = parse_acknowledged_licenses(
        getattr(args, "acknowledged_model_licenses", "") or "", manifests
    )
    deployment_path = getattr(args, "tsfm_workers_config", None)
    if acknowledged and not deployment_path:
        raise ValueError("--acknowledged-model-licenses requires --tsfm-workers-config")
    if deployment_path:
        parent_environment = dict(os.environ)
        deployment = TSFMDeployment.load(
            deployment_path,
            manifests=manifests,
            acknowledged_licenses=tuple(sorted(acknowledged)),
        )
        deployment.validate_runtime(parent_environment=parent_environment)
        broker = WorkerBroker(
            deployment.commands,
            timeout_seconds=300.0,
            parent_environment=parent_environment,
            redactor=SecretRedactor.from_environment(parent_environment),
        )
        runtimes["tsfm_worker"] = WorkerMethodRuntime(
            broker, manifests=manifests, enabled_manifest_ids=deployment.enabled_manifest_ids
        )
    return RuntimeRegistry(runtimes) # type: ignore[arg-type]


def _comma_separated(
    value: str,
    *,
    allowed: Sequence[str],
    option: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names and not allow_empty:
        raise ValueError(f"{option} must not be empty")
    unknown = set(names) - set(allowed)
    if unknown:
        raise ValueError(f"{option} contains unsupported values: {sorted(unknown)!r}")
    if len(names) != len(set(names)):
        raise ValueError(f"{option} contains duplicate values")
    return names


def _llm_client(args: argparse.Namespace):
    """Build the requested LLM client, keeping each config's own defaults."""
    backend = args.llm_backend or "codex"
    if backend == "codex":
        return CodexCLIClient(
            CodexCLIConfig(
                **_present(
                    model=args.codex_model,
                    reasoning_effort=args.codex_reasoning_effort,
                    timeout_seconds=args.codex_timeout,
                    cache_dir=args.codex_cache_dir,
                ) # type: ignore
            )
        )
    if backend == "claude":
        return ClaudeCLIClient(
            ClaudeCLIConfig(
                **_present(
                    model=args.claude_model,
                    timeout_seconds=args.claude_timeout,
                    cache_dir=args.claude_cache_dir,
                ) # type: ignore
            )
        )
    if backend == "qwen":
        return QwenClient(**_present(model_id=args.model_id, device=args.device)) # type: ignore
    raise ValueError(f"unsupported llm backend {backend!r}")


def _present(**values: object) -> dict[str, object]:
    """Drop unset options so each config keeps its declared default."""
    return {name: value for name, value in values.items() if value is not None}


def _metric(name: str):
    if name == "smae":
        return lambda prediction, truth: float(
            drcik_point_metrics(list(truth), list(prediction))["smae"]
        )
    if name == "smape":
        return lambda prediction, truth: smape(list(truth), list(prediction))
    if name == "mae":
        return lambda prediction, truth: mae(list(truth), list(prediction))
    raise ValueError(f"unsupported metric {name!r}")


def _write_method_evaluations(output_dir: Path, steps: Sequence[object]) -> None:
    destination = output_dir / "method_evaluations.jsonl"
    with destination.open("w", encoding="utf-8") as handle:
        for step in steps:
            reports = getattr(step, "child_train_reports", ())
            for report in reports:
                handle.write(
                    json.dumps(
                        {
                            "artifact_id": report.artifact_id,
                            "split": report.split,
                            "metrics": dict(report.metrics),
                            "diagnostics": dict(report.diagnostics),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                handle.write("\n")
