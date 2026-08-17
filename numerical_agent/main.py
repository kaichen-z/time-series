"""CLI composition for parameterized numerical self-evolution experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence, cast

from common.evolution_core.contracts import EvolutionConfig, MetricSpec
from common.evolution_core.controller import SelfEvolutionEngine
from common.llm import (
    ClaudeCLIClient,
    ClaudeCLIConfig,
    CodexCLIClient,
    CodexCLIConfig,
    QwenClient,
)
from common.metrics import mae, smape
from common.tracing import configure

from .adapters.dictionary_curation import DictionaryCurationTask, NumericalTaskItem
from .codegen import SANDBOX_PROVIDER, LLMMethodImplementer, SandboxMethodRuntime
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
from .dictionary import MethodRecord, ToolDictionary
from .experiment import build_experiment
from .persistence import MethodSourceArtifactStore
from .providers import RuntimeRegistry
from .smoke import FixtureMethodImplementer, FixtureMethodRuntime


APPROVED_PROVIDERS = ("fake", "llm")
LLM_BACKENDS = ("codex", "qwen", "claude")


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
    build.add_argument("--accepted-max-error", type=float, default=50.0)
    build.add_argument("--specialized-max-error", type=float, default=100.0)
    build.add_argument("--train-limit", type=int, default=None)
    build.add_argument("--dev-limit", type=int, default=None)

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

    experiment = _read_object(Path(args.experiment_config))
    dictionary = ToolDictionary.from_payload(_read_object(Path(args.base_methods)))
    curation = _curation_config(experiment)
    evolution = _evolution_config(experiment, curation)
    train_items, dev_items = _task_items(experiment)
    labels = _labels(experiment)
    output_dir = Path(args.output_dir)
    configure(output_dir / "curation_trace.jsonl")
    store = MethodSourceArtifactStore(output_dir)

    implementer, runtimes = _providers(args.provider, args)
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
        accepted_max_error=args.accepted_max_error,
        specialized_max_error=args.specialized_max_error,
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


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _collect_methods(args: argparse.Namespace) -> int:
    sources = load_source_records(args.sources)
    methods = load_method_cards(args.methods)
    duplicates = find_duplicate_candidates(methods)
    output_dir = Path(args.output_dir)
    _write_json(
        output_dir / "raw_method_registry.json",
        {
            "schema_version": 1,
            "sources": [source.to_payload() for source in sources],
            "methods": [method.to_payload() for method in methods],
        },
    )
    _write_json(
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
    payload = _read_object(Path(path))
    taxonomy = payload.get("taxonomy")
    if not isinstance(taxonomy, Mapping):
        raise ValueError("query manifest taxonomy must be an object")
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
    _write_json(Path(args.output), audit)
    summary = {
        "publishable": publishable,
        "source_count": len(sources),
        "method_count": len(methods),
        "audit_output": str(args.output),
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if publishable else 2


def _journal(path: str | Path) -> dict[str, object]:
    payload = _read_object(Path(path))
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
    raw = queries["taxonomy"]
    if not isinstance(raw, Mapping):
        raise ValueError("query manifest taxonomy must be an object")
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
    _write_json(Path(args.audit_output), audit)

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


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _curation_config(experiment: Mapping[str, object]) -> DictionaryCurationConfig:
    payload = experiment.get("curation", {})
    if not isinstance(payload, Mapping):
        raise ValueError("curation config must be an object")
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
    payload = experiment.get("evolution", {})
    if not isinstance(payload, Mapping):
        raise ValueError("evolution config must be an object")
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
    tasks = experiment.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError("tasks must be an object with Train and Dev lists")

    def parse(split: str) -> tuple[NumericalTaskItem, ...]:
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

    return parse("train"), parse("dev")


def _labels(
    experiment: Mapping[str, object],
) -> dict[str, dict[str, tuple[float, ...]]]:
    labels = experiment.get("labels")
    if not isinstance(labels, Mapping):
        raise ValueError("labels must be an object")
    normalized: dict[str, dict[str, tuple[float, ...]]] = {}
    for split in ("train", "dev"):
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
        return FixtureMethodImplementer(), RuntimeRegistry({"fake": FixtureMethodRuntime()})
    if provider == "llm":
        implementer = LLMMethodImplementer(
            _llm_client(args), transcript_dir=Path(args.output_dir) / "transcripts"
        )
        return implementer, RuntimeRegistry({SANDBOX_PROVIDER: SandboxMethodRuntime()})
    raise ValueError(f"unsupported approved provider {provider!r}")


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
