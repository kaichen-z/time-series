"""Production-owned compact P3 fixture for the offline protocol smoke."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Mapping

from common.data import Task as NumericTask
from common.llm import FakeLLMClient
from evolving_loop.data import ContextTask
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.package_numerical_supply import NumericalSupplyRelease, build_package_registry, parse_numerical_supply_release
from evolving_loop.package_registry import task_registry_fingerprint
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from evolving_loop.v2.bundle import EvolutionBundleV2
from evolving_loop.v2.cooperative.adapters import CooperativeArtifactCatalog
from evolving_loop.v2.cooperative.contracts import DecisionModuleV2, RetrievalModuleV2
from evolving_loop.v2.numerical_qd.adapters import FrozenNumericalArtifactsV2, import_numerical_seed
from numerical_agent.evolution.champion import ChampionRecipe, ChampionRelease, EvolutionAssumption, FittedChampionPolicy
from numerical_agent.evolution.execution import Task as ExecutionTask
from numerical_agent.evolution.numerical_loop import run_numerical_loop
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics, DecisionPolicy
from numerical_agent.evolution.screening import ApplicabilityPolicy, ScreeningEntry, ScreeningPolicy, profile_task
from numerical_agent.evolution.numerical_handoff import task_input_fingerprint

from .compatibility import CompatibilityCorpusV2, CompatibilityHostInputsV2
from .contracts import InfrastructureProtocolV2, ProtocolComponentV2
from evolving_loop.v2.contracts import fingerprint_payload
from evolving_loop.v2.protocol.runtime import ProtocolHostInputs


_L0 = "a" * 64
_RUNTIME = "d" * 64


def _task_payloads() -> list[dict[str, object]]:
    return [
        {"task_id": f"protocol-{'train' if index < 4 else 'dev'}-{index}", "history_values": [1.0, 2.0, 3.0] * 12, "future_values": [2.0, 2.0]}
        for index in range(5)
    ]


def smoke_host_payloads() -> dict[str, dict[str, object]]:
    retrieval = RetrievalModuleV2(1, _L0, RetrievalGenome.seed().to_payload(), ())
    decision = DecisionModuleV2(1, "safe decision prompt", (), True, 2, "last")
    return {
        "host_tasks.json": {"schema_version": 1, "tasks": _task_payloads()},
        "host_catalog.json": {"schema_version": 1, "numerical_supply": _supply().to_payload(), "retrieval_module": retrieval.to_payload(), "decision_module": decision.to_payload(), "bundle_acceptance_evidence": {"schema_version": 1, "accepted": True}},
        "host_runtime.json": {"schema_version": 1, "runtime_fingerprint": _RUNTIME, "l0_commitment_sha256": _L0, "primary_metric_cap": 5.0, "known_supports": {"document-1": ["support-1"]}, "verifier_fixtures": [{"baseline": True, "document_id": "document-1", "support_id": "support-1"}, {"baseline": True, "document_id": "document-1", "support_id": "missing"}, {"baseline": True, "document_id": "document-1", "evaluation": "private"}, {"baseline": True, "document_id": "fabricated", "support_id": "support-1"}]},
    }


def _tasks(payload: Mapping[str, object]) -> tuple[ContextTask, ...]:
    rows = payload.get("tasks")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("smoke task input requires exactly five tasks")
    result = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"task_id", "history_values", "future_values"}:
            raise ValueError("smoke task record must use the exact schema")
        task_id, history, future = row["task_id"], row["history_values"], row["future_values"]
        if type(task_id) is not str or not all(type(value) in (int, float) for value in history) or not all(type(value) in (int, float) for value in future) or not history or not future:
            raise ValueError("smoke task values are malformed")
        result.append(ContextTask(numeric=NumericTask(task_id, tuple(history), tuple(future), 2, "D", None, f"Protocol {task_id}"), target_name="sales", target_description="daily sales", history_timestamps=tuple(f"2026-01-{day:02d}" for day in range(1, 37)), future_timestamps=("2026-02-06", "2026-02-07"), documents=()))
    return tuple(result)


def _champion() -> ChampionRelease:
    assumption = EvolutionAssumption("history_ready", "specialist", "history_length", "above", "full", "select", "fixture", "fixture")
    return ChampionRelease(FittedChampionPolicy(ChampionRecipe("select_specialist", "select", ("specialist",), "specialist", (assumption,)), (("history_ready", 0.0),)), (("dictionary", "4" * 64),), "1" * 64, ("protocol_smoke",))


def _supply() -> NumericalSupplyRelease:
    return NumericalSupplyRelease(1, "n000", None, _champion().to_payload(), (), None, {"dictionary": "4" * 64}, {"materializer": "5" * 64})


def _package(task: ContextTask, release: NumericalSupplyRelease):
    forecast = (2.0, 2.0)
    diagnostics = CandidateDiagnostics.synthetic(name="specialist", family="statistical", median_mase=0.1, fold_forecasts=(forecast,) * 3, fold_truths=((2.0, 2.0),) * 3, median_smae=0.1, recent_smae=0.1, worst_smae=0.1, median_srmse=0.1, recent_srmse=0.1, worst_srmse=0.1, worst_smae_raw=0.1, worst_srmse_raw=0.1)
    policy = ScreeningPolicy((ScreeningEntry("specialist", "statistical", "keep", ApplicabilityPolicy(), "fixture"),), ("specialist",))
    package = run_numerical_loop(ExecutionTask(task.numeric.task_id, task.numeric.history_values, 2, "D", ()), screening_policy=policy, candidate_runner=lambda _name, _history, _horizon, _frequency: forecast, diagnostics={"specialist": diagnostics}, decision_policy=DecisionPolicy(ensemble_enabled=False), champion_release=_champion())
    return replace(package, task_profile=profile_task(ExecutionTask(task.numeric.task_id, task.numeric.history_values, 2, "D", ())), component_fingerprints={**dict(package.component_fingerprints), "task_input": task_input_fingerprint(task_id=task.numeric.task_id, history=task.numeric.history_values, frequency="D", horizon=2), "numerical_supply_release": release.fingerprint})


def _protocol() -> InfrastructureProtocolV2:
    return InfrastructureProtocolV2(1, 1, None, _L0, (ProtocolComponentV2("backbone", "last_value", 1, 1), ProtocolComponentV2("loader", "canonical_json", 1, 1), ProtocolComponentV2("verifier_strategy", "exact_support", 1, 1), ProtocolComponentV2("diagnostic_metric", "forecast_spread", 1, 1), ProtocolComponentV2("schema_migration", "identity_envelope", 1, 1)))


@dataclass(frozen=True)
class SmokeCase:
    corpus: CompatibilityCorpusV2
    inputs: CompatibilityHostInputsV2
    bundles: tuple[EvolutionBundleV2, ...]
    seed_protocol: InfrastructureProtocolV2


def _build_smoke_case(files: Mapping[str, Mapping[str, object]], *, allow_bootstrap: bool) -> SmokeCase:
    tasks = _tasks(files["host_tasks.json"])
    catalog_input, runtime_input = files["host_catalog.json"], files["host_runtime.json"]
    if set(catalog_input) != {"schema_version", "numerical_supply", "retrieval_module", "decision_module", "bundle_acceptance_evidence"} or catalog_input["schema_version"] != 1:
        raise ValueError("smoke catalog input must use the exact schema")
    if set(runtime_input) != {"schema_version", "runtime_fingerprint", "l0_commitment_sha256", "primary_metric_cap", "known_supports", "verifier_fixtures"} or runtime_input["schema_version"] != 1 or type(runtime_input["runtime_fingerprint"]) is not str or type(runtime_input["l0_commitment_sha256"]) is not str or type(runtime_input["primary_metric_cap"]) not in (int, float):
        raise ValueError("smoke runtime input must use the exact schema")
    release = parse_numerical_supply_release(catalog_input["numerical_supply"])
    registry = build_package_registry(tasks, release, _package)
    numerical = FrozenNumericalArtifactsV2(release, registry, import_numerical_seed(release, registry, tasks=tasks).envelope, ())
    catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
    numerical_ids = catalog.add_numerical(numerical)
    retrieval = RetrievalModuleV2.from_payload(catalog_input["retrieval_module"])
    decision = DecisionModuleV2.from_payload(catalog_input["decision_module"])
    retrieval_sha, decision_sha = catalog.add_retrieval(retrieval), catalog.add_decision(decision)
    bundle_evidence = catalog_input["bundle_acceptance_evidence"]
    if not isinstance(bundle_evidence, Mapping):
        raise ValueError("smoke catalog Bundle evidence must be an object")
    bundle_evidence = dict(bundle_evidence)
    bundle_evidence_sha = fingerprint_payload(bundle_evidence)
    fixtures = runtime_input["verifier_fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) != 4 or not all(isinstance(item, Mapping) for item in fixtures) or not isinstance(runtime_input["known_supports"], Mapping) or not all(type(key) is str and isinstance(value, list) and all(type(item) is str for item in value) for key, value in runtime_input["known_supports"].items()):
        raise ValueError("smoke runtime verifier inputs are malformed")
    host = ProtocolHostInputs(raw_fixture_records={task.numeric.task_id: task for task in tasks}, canonical_records={task.numeric.task_id: task for task in tasks}, catalog=catalog, frozen_numerical=numerical, retrieval_factory=lambda genome, skills: TwoStageRetrievalAgent(FakeLLMClient([json.dumps({"evidence_chains": [], "counterevidence": [], "missing_information": [], "sufficient": True})]), genome, skills), decision_factory=lambda module: DecisionAgent(FakeLLMClient([json.dumps({"selected_candidate_id": "specialist", "supporting_document_ids": [], "rationale": "fixture", "request_more_retrieval": False, "gaps": [], "used_skill_names": []}), json.dumps({"selected_candidate_id": "specialist", "supporting_document_ids": [], "rationale": "fixture", "request_more_retrieval": False, "gaps": [], "used_skill_names": []})]), prompt=module.prompt), committed_task_metadata={task.numeric.task_id: task_registry_fingerprint(task) for task in tasks}, l0_commitment_sha256=str(runtime_input["l0_commitment_sha256"]), primary_metric_cap=float(runtime_input["primary_metric_cap"]), baseline_verifier=lambda evidence: evidence.get("baseline") is True, known_supports={key: tuple(value) for key, value in runtime_input["known_supports"].items()}, fixture_scope="smoke")
    train_shas, dev_shas = tuple(task_registry_fingerprint(task) for task in tasks[:4]), (task_registry_fingerprint(tasks[4]),)
    closure = files.get("host_closure.json")
    if closure is None:
        if not allow_bootstrap:
            raise ValueError("smoke case requires a verified Host authority closure")
        bundles = tuple(EvolutionBundleV2(2, 1, "b" * 64, *numerical_ids, retrieval_sha, decision_sha, "5" * 64, "6" * 64, str(index) * 64, str(runtime_input["l0_commitment_sha256"]), {"python": "9" * 64}, bundle_evidence_sha) for index in (1, 2))
        envelopes = {fingerprint_payload({"schema_version": 1, "artifact": bundle.to_payload(), "artifact_sha256": bundle.fingerprint()}): {"schema_version": 1, "artifact": bundle.to_payload(), "artifact_sha256": bundle.fingerprint()} for bundle in bundles}
        corpus = CompatibilityCorpusV2(1, str(runtime_input["l0_commitment_sha256"]), train_shas, dev_shas, tuple(bundle.fingerprint() for bundle in bundles), tuple(bundle.fingerprint() for bundle in bundles) + tuple(envelopes), fingerprint_payload({"fixtures": 1}))
    else:
        required_closure = {"schema_version", "corpus", "archive_bundles", "artifact_envelopes", "bundle_acceptance_evidence", "split_metadata", "frozen_bundle_sha256"}
        if set(closure) != required_closure or closure.get("schema_version") != 1 or closure.get("split_metadata") != {"train_task_sha256s": list(train_shas), "dev_task_sha256s": list(dev_shas), "public_task_sha256s": []}:
            raise ValueError("host closure does not match deserialized task/catalog/runtime authority")
        rows = closure["archive_bundles"]
        raw_envelopes = closure["artifact_envelopes"]
        raw_evidence = closure["bundle_acceptance_evidence"]
        if not isinstance(rows, list) or not isinstance(raw_envelopes, Mapping) or not isinstance(raw_evidence, Mapping):
            raise ValueError("host closure fields are malformed")
        bundles = tuple(EvolutionBundleV2.from_payload(row) for row in rows)
        envelopes = {str(identity): dict(payload) for identity, payload in raw_envelopes.items() if isinstance(payload, Mapping)}
        evidence_map = {str(key): dict(value) for key, value in raw_evidence.items() if isinstance(value, Mapping)}
        corpus = CompatibilityCorpusV2.from_payload(closure["corpus"])
        if len(envelopes) != len(raw_envelopes) or len(evidence_map) != len(raw_evidence) or len(bundles) != 2 or corpus.l0_commitment_sha256 != runtime_input["l0_commitment_sha256"] or corpus.train_task_sha256s != train_shas or corpus.dev_task_sha256s != dev_shas or corpus.archive_bundle_sha256s != tuple(bundle.fingerprint() for bundle in bundles) or not all(bundle.numerical_release_sha256 == numerical_ids[0] and bundle.numerical_registry_sha256 == numerical_ids[1] and bundle.retrieval_release_sha256 == retrieval_sha and bundle.decision_policy_sha256 == decision_sha and bundle.protocol_fingerprint == runtime_input["l0_commitment_sha256"] and bundle.acceptance_evidence_sha256 in evidence_map for bundle in bundles) or any(fingerprint_payload(payload) != identity or payload.get("artifact_sha256") not in {bundle.fingerprint() for bundle in bundles} for identity, payload in envelopes.items()) or any(fingerprint_payload(payload) != identity for identity, payload in evidence_map.items()) or closure.get("frozen_bundle_sha256") != bundles[0].fingerprint():
            raise ValueError("host closure identities are not the exact executable authority")
    inputs = CompatibilityHostInputsV2(host, {"train_task_sha256s": train_shas, "dev_task_sha256s": dev_shas, "public_task_sha256s": ()}, {bundle.fingerprint(): bundle for bundle in bundles}, envelopes, tuple(dict(item) for item in fixtures), str(runtime_input["runtime_fingerprint"]), {task_registry_fingerprint(task): {"task_id": task.numeric.task_id, "history_values": list(task.numeric.history_values), "prediction_length": 2, "frequency": "D"} for task in tasks}, bundle_acceptance_evidence=evidence_map if closure is not None else {bundle_evidence_sha: bundle_evidence})
    return SmokeCase(corpus, inputs, bundles, _protocol())


def build_smoke_case(files: Mapping[str, Mapping[str, object]]) -> SmokeCase:
    """Deserialize and bind the byte-verified Host closure to executable adapters."""
    return _build_smoke_case(files, allow_bootstrap=False)
