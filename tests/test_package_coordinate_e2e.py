"""Deterministic two-cycle N -> R -> D package co-evolution end to end."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from common.llm import FakeLLMClient
from evolving_loop.co_evolution import HarnessPolicy, embed_retrieval_release
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.package_candidate_proposal import (
    PackageCandidate,
    embed_retrieval_candidate,
)
from evolving_loop.package_coordinate_evolution import (
    PackageCoordinateBundle,
    PackageCoordinateController,
    PackageCoordinateState,
)
from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from evolving_loop.package_metrics import PackageGateConfig
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from evolving_loop.package_stage_runner import (
    InMemoryPackageArtifactSink,
    PackageCoordinatePhaseRunner,
)
from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    _write_accepted_retrieval_release,
)
from evolving_loop.retrieval_agent.schemas import RetrievalRoundResult
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from evolving_loop.retrieval_agent.verifier import merge_verified_rounds
from tests.test_coordinate_evolution import _audit
from tests.test_package_coordinate_evolution import _bundle
from tests.test_package_decision_evolution import (
    _decision_task,
    _request_specialist,
    _round2_response,
)
from tests.test_package_stage_runner import (
    DEV_20,
    TRAIN_80,
    _evaluation,
    _schedule,
    _task_map,
)
from tests.test_package_retrieval_evolution import (
    _decision_response,
    _frozen_registry,
    _package,
    _round_response,
)


# --------------------------------------------------------------------------
# deterministic evaluator: strictly improving in cycle one, regressing after
# --------------------------------------------------------------------------


class _CycleEvaluator:
    def __init__(self) -> None:
        self.stages_seen: list[str] = []
        self.task_ids_seen: list[tuple[str, ...]] = []
        self.cache_only_calls = 0

    @staticmethod
    def _error(generation: int) -> float:
        if generation == 0:
            return 1.0
        if generation <= 3:
            return round(1.0 - 0.05 * generation, 4)
        return 1.5

    def evaluate(self, bundle, registry, tasks, *, stage, cache_only=False):
        task_ids = tuple(task.numeric.task_id for task in tasks)
        if cache_only:
            self.cache_only_calls += 1
        else:
            self.stages_seen.append(stage)
            self.task_ids_seen.append(task_ids)
        return _evaluation(bundle.fingerprint(), task_ids, self._error(bundle.generation))


# --------------------------------------------------------------------------
# scripted proposers, one per coordinate
# --------------------------------------------------------------------------


def _numerical_variant(current: PackageCoordinateState, slot: int) -> PackageCoordinateState:
    parent_release = parse_numerical_supply_release(
        current.bundle.to_payload()["numerical_release_payload"]
    )
    generation = int(parent_release.version[1:]) + 1
    payload = parent_release.to_payload()
    payload["version"] = f"n{generation:03d}"
    payload["parent_sha256"] = parent_release.fingerprint
    payload["source_fingerprints"] = {
        "dictionary": hashlib.sha256(f"gen{generation}:slot{slot}".encode()).hexdigest()
    }
    release = parse_numerical_supply_release(payload)
    package = _package()
    registry = _frozen_registry(
        (
            (
                _decision_task(),
                replace(
                    package,
                    component_fingerprints={
                        **dict(package.component_fingerprints),
                        "numerical_supply_release": release.fingerprint,
                    },
                ),
            ),
        ),
        release=release,
    )
    return current.with_numerical(release, registry)


class _NumericalProposer:
    target = "numerical"

    def propose(self, parent, feedback, *, generation, child_count):
        return tuple(
            PackageCandidate(
                slot=slot,
                target="numerical",
                state=_numerical_variant(parent, slot),
                proposal_sha256=_numerical_variant(parent, slot).bundle.fingerprint(),
            )
            for slot in range(child_count)
        )


class _RetrievalProposer:
    target = "retrieval"

    def __init__(self, skills: RetrievalSkillLibrary) -> None:
        self.skills = skills

    def _candidate_genome(self, parent, slot: int) -> RetrievalGenome:
        return replace(
            RetrievalGenome.seed(),
            version=f"v{900 + slot:03d}",
            parent=parent.bundle.policy.version,
            max_selected_documents=6 + slot,
        )

    def propose(self, parent, feedback, *, generation, child_count):
        children = []
        for slot in range(child_count):
            genome = self._candidate_genome(parent, slot)
            policy = embed_retrieval_candidate(
                parent.bundle.policy,
                genome,
                self.skills,
                changelog=f"candidate {slot}",
            )
            state = parent.with_policy(policy, target="retrieval")
            children.append(
                PackageCandidate(
                    slot=slot,
                    target="retrieval",
                    state=state,
                    proposal_sha256=state.bundle.fingerprint(),
                )
            )
        return tuple(children)


class _DecisionProposer:
    target = "decision"

    def propose(self, parent, feedback, *, generation, child_count):
        children = []
        for slot in range(child_count):
            policy = replace(
                parent.bundle.policy,
                decision_prompt=f"{parent.bundle.policy.decision_prompt} :: child {slot}",
            )
            state = parent.with_policy(policy, target="decision")
            children.append(
                PackageCandidate(
                    slot=slot,
                    target="decision",
                    state=state,
                    proposal_sha256=state.bundle.fingerprint(),
                )
            )
        return tuple(children)


class _RetrievalPublisher:
    def __init__(self, tmp_path, skills: RetrievalSkillLibrary) -> None:
        self.tmp_path = tmp_path
        self.skills = skills
        self._n = 1

    def publish(self, finalist, phase_parent):
        self._n += 1
        version = f"v{self._n:03d}"
        accepted_genome = replace(
            finalist.state.bundle.policy.retrieval_genome,
            version=version,
            parent=phase_parent.bundle.policy.version,
        )
        release = _write_accepted_retrieval_release(
            self.tmp_path / f"accepted-{version}",
            accepted_genome,
            audit=_audit(version[-1]),
        )
        policy = replace(
            embed_retrieval_release(
                phase_parent.bundle.policy, release, changelog="accepted"
            ),
            version=version,
            parent=phase_parent.bundle.policy.version,
        )
        return phase_parent.with_policy(policy, target="retrieval")


# --------------------------------------------------------------------------
# the end-to-end test
# --------------------------------------------------------------------------


def _phase_runner(target, proposer, evaluator, sink, *, publisher=None):
    return PackageCoordinatePhaseRunner(
        target=target,
        proposer=proposer,
        evaluator=evaluator,
        schedule=_schedule(),
        task_map=_task_map(),
        gate_config=PackageGateConfig(),
        artifact_store=sink,
        retrieval_publisher=publisher,
        child_count=3,
    )


def test_package_coordinate_evolution_closes_deterministic_two_cycle_loop(
    tmp_path, monkeypatch
) -> None:
    forbidden_calls = {"public_loader": 0, "legacy_runner": 0}

    def forbid_public_loader(*_args, **_kwargs):
        forbidden_calls["public_loader"] += 1
        raise AssertionError("package evolution attempted to load Public tasks")

    def forbid_legacy_runner(*_args, **_kwargs):
        forbidden_calls["legacy_runner"] += 1
        raise AssertionError("package evolution attempted to run legacy co-evolution")

    monkeypatch.setattr(
        "evolving_loop.data.load_huggingface_context_tasks", forbid_public_loader
    )
    monkeypatch.setattr(
        "evolving_loop.co_evolution.CoEvolutionEngine.evolve", forbid_legacy_runner
    )
    _task, seed_state = _bundle(tmp_path)
    skills = RetrievalSkillLibrary(tmp_path / "skills.json", persist=False).clone(
        persist=False, read_only=True
    )
    evaluator = _CycleEvaluator()
    sink = InMemoryPackageArtifactSink()

    controller = PackageCoordinateController(
        _phase_runner("numerical", _NumericalProposer(), evaluator, sink),
        _phase_runner(
            "retrieval",
            _RetrievalProposer(skills),
            evaluator,
            sink,
            publisher=_RetrievalPublisher(tmp_path, skills),
        ),
        _phase_runner("decision", _DecisionProposer(), evaluator, sink),
    )

    selected, trace = controller.run(seed_state, seed_state)

    assert tuple(step.target for step in trace) == (
        "numerical",
        "retrieval",
        "decision",
        "numerical",
        "retrieval",
        "decision",
    )
    assert _schedule().counts == (8, 32, 64, 16, 20)
    assert [step.accepted for step in trace] == [True, True, True, False, False, False]
    assert all(not step.public_test_accessed for step in trace)
    assert forbidden_calls == {"public_loader": 0, "legacy_runner": 0}

    # every accepted step's parent hash chains to the preceding accepted bundle
    accepted_chain = [step for step in trace if step.accepted]
    for earlier, later in zip(accepted_chain, accepted_chain[1:]):
        assert later.parent_bytes_sha256 == earlier.accepted_bytes_sha256
    # every rejected step preserves the exact Parent
    for step in trace:
        if not step.accepted:
            assert step.accepted_bytes_sha256 == step.parent_bytes_sha256
            assert step.accepted_registry_sha256 == step.parent_registry_sha256

    assert selected.bundle.generation == 3
    assert selected.bundle.acceptance_evidence_sha256 is not None
    # no Public partition task ever entered the evaluator
    assert "public" not in " ".join(evaluator.stages_seen)
    assert set(evaluator.stages_seen) <= {
        "screen8",
        "screen32",
        "build64",
        "calibration16",
        "dev20",
    }
    assert {
        task_id for task_ids in evaluator.task_ids_seen for task_id in task_ids
    } == {task.task_id for task in (*TRAIN_80, *DEV_20)}
    # the Numerical coordinate replaced both the release and the registry
    assert selected.bundle.numerical_release_sha256 != seed_state.bundle.numerical_release_sha256
    assert selected.bundle.numerical_manifest_sha256 != seed_state.bundle.numerical_manifest_sha256

    # Durable bundle bytes reconstruct the exact state when bound to the same registry.
    replay_payload = selected.bundle.to_payload()
    assert replay_payload.pop("schema_version") == 2
    policy_payload = replay_payload.pop("policy")
    replayed = PackageCoordinateState(
        PackageCoordinateBundle(
            policy=HarnessPolicy(**policy_payload),
            **replay_payload,
        ),
        selected.registry,
    )
    assert replayed.registry.task_ids == selected.registry.task_ids
    assert replayed.registry.fingerprint == selected.registry.fingerprint
    assert replayed.bundle.canonical_bytes() == selected.bundle.canonical_bytes()

    # Exercise the typed second Retrieval round through the final package pipeline.
    retrieval_llm = FakeLLMClient([_round_response(), _round2_response(_task)])
    retrieval = TwoStageRetrievalAgent(
        retrieval_llm,
        selected.bundle.policy.retrieval_genome,
        skills,
    )
    decision = DecisionAgent(
        FakeLLMClient([_request_specialist(), _decision_response("specialist")]),
        prompt=selected.bundle.policy.decision_prompt,
    )
    typed_round1: list[RetrievalRoundResult] = []
    typed_round2: list[RetrievalRoundResult] = []
    original_round1 = TwoStageRetrievalAgent.run_round1
    original_round2 = TwoStageRetrievalAgent.run_round2

    def record_round1(self, *args, **kwargs):
        result = original_round1(self, *args, **kwargs)
        typed_round1.append(result)
        return result

    def make_round2_malformed(self, *args, **kwargs):
        result = original_round2(self, *args, **kwargs)
        typed_round2.append(result)
        return replace(
            result,
            rejected=(*result.rejected, "invalid_round2_response"),
        )

    monkeypatch.setattr(TwoStageRetrievalAgent, "run_round1", record_round1)
    monkeypatch.setattr(TwoStageRetrievalAgent, "run_round2", make_round2_malformed)
    evaluation = PackagePipelineEvaluator(
        lambda _policy: retrieval,
        lambda _policy: decision,
    ).evaluate(
        selected.bundle,
        selected.registry,
        (_task,),
        stage="screen8",
    )

    expected_round1_card = merge_verified_rounds(typed_round1[0], None)
    expected_round1_sha256 = hashlib.sha256(
        json.dumps(
            expected_round1_card.to_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    row = evaluation.task_rows[0]
    assert len(retrieval_llm.calls) == 2
    assert len(typed_round2) == 1
    assert all(
        type(result) is RetrievalRoundResult
        for result in (*typed_round1, *typed_round2)
    )
    assert row.final_retrieval_sha256 == expected_round1_sha256
    assert row.fallback_count == 0
