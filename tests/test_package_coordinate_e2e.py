"""Deterministic two-cycle N -> R -> D package co-evolution end to end."""
from __future__ import annotations

import hashlib
from dataclasses import replace

from evolving_loop.co_evolution import HarnessPolicy, embed_retrieval_release
from evolving_loop.package_candidate_proposal import (
    PackageCandidate,
    embed_retrieval_candidate,
)
from evolving_loop.package_coordinate_evolution import (
    PackageCoordinateController,
    PackageCoordinateState,
)
from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from evolving_loop.package_metrics import PackageGateConfig
from evolving_loop.package_stage_runner import (
    InMemoryPackageArtifactSink,
    PackageCoordinatePhaseRunner,
)
from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    _write_accepted_retrieval_release,
)
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from tests.test_coordinate_evolution import _audit
from tests.test_package_coordinate_evolution import _bundle
from tests.test_package_decision_evolution import _decision_task
from tests.test_package_stage_runner import (
    DEV_20,
    TRAIN_80,
    _evaluation,
    _schedule,
    _task_map,
)
from tests.test_package_retrieval_evolution import _package, _frozen_registry


# --------------------------------------------------------------------------
# deterministic evaluator: strictly improving in cycle one, regressing after
# --------------------------------------------------------------------------


class _CycleEvaluator:
    def __init__(self) -> None:
        self.stages_seen: list[str] = []
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


def test_package_coordinate_evolution_closes_deterministic_two_cycle_loop(tmp_path) -> None:
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
    assert [step.accepted for step in trace] == [True, True, True, False, False, False]
    assert all(not step.public_test_accessed for step in trace)

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
    # the Numerical coordinate replaced both the release and the registry
    assert selected.bundle.numerical_release_sha256 != seed_state.bundle.numerical_release_sha256
    assert selected.bundle.numerical_manifest_sha256 != seed_state.bundle.numerical_manifest_sha256
