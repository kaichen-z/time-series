"""CLI composition for parameterized numerical self-evolution experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from common.evolution_core.contracts import EvolutionConfig, MetricSpec
from common.evolution_core.controller import SelfEvolutionEngine
from common.evolution_core.persistence import JsonArtifactStore
from common.metrics import mae, smape

from .adapters.dictionary_curation import DictionaryCurationTask, NumericalTaskItem
from .config import DictionaryCurationConfig
from .dictionary import MethodRecord, ToolDictionary
from .providers import RuntimeRegistry
from .smoke import FixtureMethodImplementer, FixtureMethodRuntime


APPROVED_PROVIDERS = ("fake",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    curate = subparsers.add_parser("curate", help="run dictionary-curation self-evolution")
    curate.add_argument("--experiment-config", required=True)
    curate.add_argument("--base-methods", required=True)
    curate.add_argument("--provider", required=True)
    curate.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.provider not in APPROVED_PROVIDERS:
        parser.error(
            f"provider must be an approved provider name: {', '.join(APPROVED_PROVIDERS)}"
        )
    if args.command != "curate":
        parser.error(f"unsupported command: {args.command}")

    experiment = _read_object(Path(args.experiment_config))
    dictionary = ToolDictionary.from_payload(_read_object(Path(args.base_methods)))
    curation = _curation_config(experiment)
    evolution = _evolution_config(experiment, curation)
    train_items, dev_items = _task_items(experiment)
    labels = _labels(experiment)
    output_dir = Path(args.output_dir)
    store = JsonArtifactStore(output_dir)

    implementer, runtimes = _providers(args.provider)
    task = DictionaryCurationTask(
        base_dictionary=dictionary,
        config=curation,
        implementer=implementer,
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
        "successive_halving",
        "screen_train_items",
        "screen_dev_items",
        "max_promoted_children",
        "screening_tolerance",
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


def _providers(provider: str) -> tuple[FixtureMethodImplementer, RuntimeRegistry]:
    if provider != "fake":
        raise ValueError(f"unsupported approved provider {provider!r}")
    return FixtureMethodImplementer(), RuntimeRegistry({"fake": FixtureMethodRuntime()})


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
