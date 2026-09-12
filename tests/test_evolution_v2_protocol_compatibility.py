from __future__ import annotations

from dataclasses import replace

import pytest

from evolving_loop.v2.bundle import EvolutionBundleV2
from evolving_loop.v2.contracts import fingerprint_payload
from evolving_loop.v2.protocol import (
    CompatibilityCorpusV2,
    CompatibilityHostInputsV2,
    InfrastructureProtocolV2,
    ProtocolComponentV2,
    ProtocolHostInputs,
    ProtocolRuntimeRegistry,
    check_compatibility,
    decide_protocol,
)
from evolving_loop.v2.cooperative.adapters import CooperativeArtifactCatalog
from evolving_loop.v2.cooperative.contracts import DecisionModuleV2, RetrievalModuleV2
from evolving_loop.v2.numerical_qd.adapters import FrozenNumericalArtifactsV2, import_numerical_seed
from evolving_loop.package_numerical_supply import build_package_registry
from evolving_loop.package_registry import task_registry_fingerprint
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from tests.test_evolution_v2_cooperative_pipeline import (
    _decision_response,
    _pipeline_package,
    _round1_response,
    _train_tasks,
)
from common.llm import FakeLLMClient
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent


def _protocol(*, loader="canonical_json", diagnostic="forecast_spread"):
    return InfrastructureProtocolV2(
        1,
        1,
        None,
        "a" * 64,
        (
            ProtocolComponentV2("backbone", "last_value", 1, 1),
            ProtocolComponentV2("loader", loader, 1, 1),
            ProtocolComponentV2("verifier_strategy", "exact_support", 1, 1),
            ProtocolComponentV2("diagnostic_metric", diagnostic, 1, 1),
            ProtocolComponentV2("schema_migration", "identity_envelope", 1, 1),
        ),
    )


@pytest.fixture
def compatibility_case(tmp_path):
    train = _train_tasks()
    dev = (replace(train[0], numeric=replace(train[0].numeric, task_id="protocol-dev")),)
    tasks = train + dev
    release = __import__("tests.test_package_retrieval_evolution", fromlist=["_seed_supply_release"])._seed_supply_release()
    registry = build_package_registry(tasks, release, _pipeline_package)
    numerical = FrozenNumericalArtifactsV2(
        release, registry, import_numerical_seed(release, registry, tasks=tasks).envelope, ()
    )
    catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
    numerical_ids = catalog.add_numerical(numerical)
    retrieval = RetrievalModuleV2(1, "a" * 64, RetrievalGenome.seed().to_payload(), ())
    decision = DecisionModuleV2(1, "safe decision prompt", (), True, 2, "last")
    retrieval_sha, decision_sha = catalog.add_retrieval(retrieval), catalog.add_decision(decision)
    bundles = tuple(
        EvolutionBundleV2(
            2, 1, "b" * 64, *numerical_ids, retrieval_sha, decision_sha,
            "5" * 64, "6" * 64, str(index) * 64, "a" * 64, {"python": "9" * 64}, "c" * 64,
        )
        for index in (1, 2)
    )

    def retrieval_factory(genome, skills):
        return TwoStageRetrievalAgent(FakeLLMClient([_round1_response()]), genome, skills)

    def decision_factory(module):
        return DecisionAgent(FakeLLMClient([_decision_response(), _decision_response()]), prompt=module.prompt)

    host = ProtocolHostInputs(
        raw_fixture_records={task.numeric.task_id: task for task in tasks},
        canonical_records={task.numeric.task_id: task for task in tasks},
        catalog=catalog,
        frozen_numerical=numerical,
        retrieval_factory=retrieval_factory,
        decision_factory=decision_factory,
        committed_task_metadata={task.numeric.task_id: task_registry_fingerprint(task) for task in tasks},
        l0_commitment_sha256="a" * 64,
        primary_metric_cap=5.0,
        baseline_verifier=lambda evidence: evidence.get("baseline") is True,
        known_supports={"document-1": ("support-1",)},
        fixture_scope="test",
        empty_skill_path=tmp_path / "skills.json",
    )
    train_shas = tuple(task_registry_fingerprint(task) for task in train)
    dev_shas = (task_registry_fingerprint(dev[0]),)
    envelopes = {
        fingerprint_payload({"schema_version": 1, "artifact": bundle.to_payload(), "artifact_sha256": bundle.fingerprint()}): {
            "schema_version": 1, "artifact": bundle.to_payload(), "artifact_sha256": bundle.fingerprint()
        }
        for bundle in bundles
    }
    corpus = CompatibilityCorpusV2(
        1, "a" * 64, train_shas, dev_shas, tuple(bundle.fingerprint() for bundle in bundles),
        tuple(bundle.fingerprint() for bundle in bundles) + tuple(envelopes), fingerprint_payload({"fixtures": 1}),
    )
    cache = {}
    inputs = CompatibilityHostInputsV2(
        host,
        {"train_task_sha256s": train_shas, "dev_task_sha256s": dev_shas, "public_task_sha256s": ()},
        {bundle.fingerprint(): bundle for bundle in bundles},
        envelopes,
        (
            {"baseline": True, "document_id": "document-1", "support_id": "support-1"},
            {"baseline": True, "document_id": "document-1", "support_id": "missing"},
            {"baseline": True, "document_id": "document-1", "evaluation": "private"},
            {"baseline": True, "document_id": "fabricated", "support_id": "support-1"},
        ),
        runtime_fingerprint="d" * 64,
        evaluation_cache=cache,
    )

    class Case:
        dev_calls = 0
        runtime_resolutions = 0
        old_score_cache_misses_for_new_protocol = 0

        def check(self, change):
            old = _protocol()
            if change == "changed_history_json":
                proposed = _protocol(loader=change)
            else:
                proposed = _protocol(diagnostic=change)
            outer = self
            class SpyRegistry(ProtocolRuntimeRegistry):
                def resolve(self, *args, **kwargs):
                    outer.runtime_resolutions += 1
                    return super().resolve(*args, **kwargs)
            result = check_compatibility(old, proposed, corpus, SpyRegistry(), host_inputs=inputs, sealed_store=tmp_path / "sealed")
            self.dev_calls = sum(key[-1] == "dev" for key in cache)
            self.old_score_cache_misses_for_new_protocol = sum(
                key[0] == proposed.fingerprint() and key[-1] == "train" for key in cache
            )
            return result

        def with_public_membership(self):
            return type(self)( )
    case = Case()
    case.inputs = inputs
    case.corpus = corpus
    case.bundles = bundles
    # The returned helper needs a public-injection variant while retaining spies.
    def public_case():
        return CompatibilityHostInputsV2(
            host, {"train_task_sha256s": train_shas, "dev_task_sha256s": dev_shas, "public_task_sha256s": ("f" * 64,)},
            {bundle.fingerprint(): bundle for bundle in bundles}, envelopes, inputs.verifier_fixtures, "d" * 64,
        )
    case.with_public_membership = lambda: type("PublicCase", (), {"check": lambda self, _change: check_compatibility(_protocol(), _protocol(), corpus, ProtocolRuntimeRegistry(), host_inputs=public_case(), sealed_store=tmp_path / "sealed")})()
    return case


def test_changed_task_hash_rejects_before_dev(compatibility_case):
    """A loader change that cannot reproduce commitments never reaches Dev."""
    evidence = compatibility_case.check("changed_history_json")
    decision = decide_protocol(evidence)
    assert decision.decision == "reject"
    assert "task_hash_mismatch" in decision.reason_codes
    assert compatibility_case.dev_calls == 0


def test_each_archive_bundle_is_scored_under_both_protocols(compatibility_case):
    """Proposed diagnostics must replay both frozen archives under their own key."""
    evidence = compatibility_case.check("absolute_movement")
    assert decide_protocol(evidence).decision == "accept"
    assert len(evidence.train_rows) == 4
    assert len({(r["protocol_sha256"], r["bundle_sha256"]) for r in evidence.train_rows}) == 4
    assert compatibility_case.old_score_cache_misses_for_new_protocol == 2


def test_public_membership_rejected_before_resolving_runtime(compatibility_case):
    """Public membership is rejected at the split boundary, before adapters run."""
    with pytest.raises(ValueError, match="Public"):
        compatibility_case.with_public_membership().check("history_mean")
