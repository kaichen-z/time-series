"""Deterministic Host fixture for the Source V2 real-pipeline meta evaluator."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from common.llm import FakeLLMClient
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from evolving_loop.v2.bundle import EvolutionBundleV2
from evolving_loop.v2.cli import (
    _cooperative_numerical_pair,
    _parse_cooperative_task_manifest,
)
from evolving_loop.v2.cooperative.adapters import (
    CooperativeArtifactCatalog,
    CooperativePipelineAdapter,
    DecisionCoordinateAdapter,
    NumericalCoordinateAdapter,
    RetrievalCoordinateAdapter,
)
from evolving_loop.v2.cooperative.contracts import DecisionModuleV2, RetrievalModuleV2
from evolving_loop.v2.source.contracts import SourceVariantV2
from evolving_loop.v2.contracts import canonical_v2_bytes
from tests.build_evolution_v2_cooperative_fixture import _seed_supply, _task


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _round1() -> str:
    return json.dumps({"evidence_chains": [], "counterevidence": [], "missing_information": [], "sufficient": True})


def _decision(candidate: str) -> str:
    return json.dumps({"selected_candidate_id": candidate, "supporting_document_ids": [], "rationale": "fixture selection", "request_more_retrieval": False, "gaps": [], "used_skill_names": []})


@dataclass
class SourceCase:
    seed_source: SourceVariantV2
    improving_source: SourceVariantV2
    neutral_source: SourceVariantV2
    seed_bundle: EvolutionBundleV2
    train_tasks: tuple
    dev_tasks: tuple
    pipeline_trace: list[dict[str, str]]
    policy_requests: list[dict[str, object]]
    catalog_factory: object
    adapters_factory: object
    pipeline_factory: object
    evaluator: object | None = None


def build_source_case(root: Path, *, failed_held_out: bool = False) -> SourceCase:
    """Build real P3 inputs; source policies choose real typed candidates only."""
    del failed_held_out
    manifest = _parse_cooperative_task_manifest(
        {"schema_version": 1, "train": [_task(index) for index in range(4)], "dev": [_task(4)]}
    )
    train_tasks, dev_tasks = manifest["train"], manifest["dev"]
    release = _seed_supply()
    numerical_seed = _cooperative_numerical_pair(release, (*train_tasks, *dev_tasks), anchor_offset=0.0)
    alternate_payload = release.to_payload()
    alternate_payload["version"] = "n002"
    alternate_payload["parent_sha256"] = release.fingerprint
    from evolving_loop.package_numerical_supply import parse_numerical_supply_release
    numerical_alternate = _cooperative_numerical_pair(
        parse_numerical_supply_release(alternate_payload), (*train_tasks, *dev_tasks), anchor_offset=1.0
    )
    retrieval = RetrievalModuleV2(1, _digest("source-meta-retrieval"), RetrievalGenome.seed().to_payload(), ())
    decision = DecisionModuleV2(1, "Use safe anchor.", (), True, 2, "last")
    trace: list[dict[str, str]] = []
    requests: list[dict[str, object]] = []

    def catalog_factory():
        catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
        numerical_ids = catalog.add_numerical(numerical_seed)
        retrieval_sha = catalog.add_retrieval(retrieval)
        decision_sha = catalog.add_decision(decision)
        bundle = EvolutionBundleV2(2, 0, None, *numerical_ids, retrieval_sha, decision_sha,
            _digest("harness"), _digest("archive"), _digest("scheduler"), _digest("protocol"), {"python": _digest("runtime")}, None)
        return catalog, bundle

    def adapters_factory(_catalog):
        return {
            "numerical": NumericalCoordinateAdapter((numerical_alternate,)),
            "retrieval": RetrievalCoordinateAdapter(),
            "decision": DecisionCoordinateAdapter(("Prefer seasonal_naive.",)),
        }

    def pipeline_factory(catalog):
        def retrieval_factory(genome, skills):
            trace.append({"agent": "numerical"})
            trace.append({"agent": "retrieval"})
            return TwoStageRetrievalAgent(FakeLLMClient([_round1()]), genome, skills)

        def decision_factory(module):
            trace.append({"agent": "decision"})
            selected = "seasonal_naive" if "seasonal_naive" in module.prompt else "safe_anchor"
            return DecisionAgent(FakeLLMClient([_decision(selected), _decision(selected)]), prompt=module.prompt)

        return CooperativePipelineAdapter(catalog, retrieval_factory, decision_factory,
            empty_skill_path=root / "source-meta-empty-skills.json")

    catalog, seed_bundle = catalog_factory()
    del catalog
    protocol, runtime = _digest("source-protocol"), _digest("source-runtime")
    seed_source = SourceVariantV2.seed("def choose_arm(request):\n    return 'retrieval'\n", protocol, runtime)
    improving_source = SourceVariantV2.child(seed_source, "def choose_arm(request):\n    return 'numerical'\n", "select_numerical")
    neutral_source = SourceVariantV2.child(seed_source, "def choose_arm(request):\n    return 'retrieval'\n", "select_retrieval")
    case = SourceCase(seed_source, improving_source, neutral_source, seed_bundle, train_tasks, dev_tasks,
        trace, requests, catalog_factory, adapters_factory, pipeline_factory)
    from evolving_loop.v2.source.meta import SourceMetaEvaluatorV2
    case.evaluator = SourceMetaEvaluatorV2(case)
    return case


def export_source_manifest(root: Path) -> Path:
    """Materialize the tiny frozen Project-3-compatible offline input set."""
    if root.exists() and any(root.iterdir()):
        raise ValueError("smoke input destination must be empty")
    root.mkdir(parents=True, exist_ok=True)
    case = build_source_case(root)
    frozen = {
        "seed_source.json": case.seed_source.to_payload(),
        "tasks.json": {
            "train": [task.numeric.task_id for task in case.train_tasks],
            "dev": [task.numeric.task_id for task in case.dev_tasks],
        },
    }
    files: dict[str, dict[str, str]] = {}
    for name, payload in frozen.items():
        content = canonical_v2_bytes(payload)
        (root / name).write_bytes(content)
        files[name] = {"path": name, "sha256": hashlib.sha256(content).hexdigest()}
    manifest = {
        "schema_version": 1, "files": files,
        "train_task_ids": [task.numeric.task_id for task in case.train_tasks],
        "dev_task_ids": [task.numeric.task_id for task in case.dev_tasks],
        "train_folds": [[0, 1], [2, 3]],
        "protocol_fingerprint": case.seed_source.protocol_fingerprint,
        "runtime_fingerprint": case.seed_source.runtime_fingerprint,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(canonical_v2_bytes(manifest))
    return manifest_path


def load_source_manifest(path: Path) -> SourceCase:
    """Verify frozen bytes before rebuilding the deterministic fixture seam."""
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if canonical_v2_bytes(manifest) != raw:
        raise ValueError("source manifest must be canonical")
    required = {"schema_version", "files", "train_task_ids", "dev_task_ids", "train_folds", "protocol_fingerprint", "runtime_fingerprint"}
    if set(manifest) != required or manifest["schema_version"] != 1 or not isinstance(manifest["files"], dict):
        raise ValueError("source manifest schema mismatch")
    for name, entry in manifest["files"].items():
        if not isinstance(name, str) or not isinstance(entry, dict) or set(entry) != {"path", "sha256"} or entry["path"] != name:
            raise ValueError("source manifest file entry is invalid")
        target = path.parent / name
        if target.parent != path.parent or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("source manifest file digest mismatch")
    case = build_source_case(path.parent)
    if (manifest["train_task_ids"] != [task.numeric.task_id for task in case.train_tasks]
            or manifest["dev_task_ids"] != [task.numeric.task_id for task in case.dev_tasks]
            or manifest["train_folds"] != [[0, 1], [2, 3]]
            or manifest["protocol_fingerprint"] != case.seed_source.protocol_fingerprint
            or manifest["runtime_fingerprint"] != case.seed_source.runtime_fingerprint):
        raise ValueError("source manifest commitments do not match fixture")
    return case
