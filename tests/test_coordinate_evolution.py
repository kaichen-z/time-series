from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.llm import FakeLLMClient
from evolving_loop.co_evolution import (
    CoEvolutionConfig,
    CoEvolutionEngine,
    HarnessPolicy,
    embed_retrieval_release,
)
from evolving_loop.coordinate_evolution import (
    CoordinateDiagnostics,
    CoordinateEvolutionConfig,
    CoordinateEvolutionController,
    CoordinatePhaseOutcome,
    DecisionEvolutionPhaseAdapter,
    RetrievalEvolutionPhaseAdapter,
)
from evolving_loop.retrieval_agent.evolution import (
    RetrievalEvolutionConfig,
    RetrievalEvolutionEngine,
    RetrievalEvolutionResult,
)
from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    _write_accepted_retrieval_release,
)


def _audit(marker: str) -> dict[str, object]:
    return {
        "state": "accepted",
        "train_dev_split_sha256": marker * 64,
        "verifier_sha256": "2" * 64,
        "evaluator_sha256": "3" * 64,
        "metric_sha256": "4" * 64,
        "metric_cap": 5.0,
        "train_summary": {"task_count": 80},
        "dev_summary": {"task_count": 20},
        "acceptance_reason": "all gates passed",
    }


def _accepted_release(
    root: Path,
    version: str,
    parent: str,
    *,
    strategy: str = "timeline_first",
):
    genome = replace(
        RetrievalGenome.seed(),
        version=version,
        parent=parent,
        round1_strategy=strategy,
    )
    return _write_accepted_retrieval_release(
        root,
        genome,
        audit=_audit(version[-1]),
    )


def _bound_policy(release) -> HarnessPolicy:
    return embed_retrieval_release(
        HarnessPolicy(), release, changelog=f"Accepted {release.genome.version}."
    )


def _result(parent: RetrievalGenome, child: RetrievalGenome, *, accepted: bool):
    return RetrievalEvolutionResult(
        original_parent=parent,
        train_winner=child,
        selected_genome=child if accepted else parent,
        accepted=accepted,
        acceptance_reasons=("all_dev_gates_passed",) if accepted else (),
        rejection_reasons=() if accepted else ("no_strict_gain",),
        parent_dev=None,
        child_dev=None,
        generations=(),
        trace=(),
        release_genome=child if accepted else None,
        release_published=accepted,
    )


class _Evaluator:
    def evaluate(self, *_args, **_kwargs):  # pragma: no cover - replaced per test
        raise AssertionError("the adapter test replaces engine.evolve")


def test_harness_policy_freezes_and_typed_validates_embedded_retrieval_release(
    tmp_path,
) -> None:
    """Catches a mutable or merely hash-checked accepted release snapshot."""
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    policy = _bound_policy(release)
    payload = policy.to_payload()

    with pytest.raises(TypeError):
        policy.retrieval_release_payload["genome"]["round1_strategy"] = "entity_first"

    with pytest.raises(ValueError, match="mirror|prompt|Retrieval release"):
        replace(policy, retrieval_prompt="forged legacy Retrieval prompt")

    tampered = json.loads(json.dumps(payload))
    tampered["retrieval_release_payload"]["genome"]["round1_prompt"] = "forged"
    embedded = tampered["retrieval_release_payload"]
    tampered["retrieval_release_sha256"] = HarnessPolicy.retrieval_payload_fingerprint(
        embedded
    )
    with pytest.raises(ValueError, match="prompt|Retrieval release"):
        HarnessPolicy(**tampered)


def test_retrieval_adapter_uses_engine_result_and_operator_loaded_release(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches accepting raw child JSON without a trusted release load."""
    parent_release = _accepted_release(tmp_path / "releases", "v001", "v000")
    child_release = _accepted_release(
        tmp_path / "releases", "v002", "v001", strategy="entity_first"
    )
    parent = _bound_policy(parent_release)
    engine = RetrievalEvolutionEngine(
        FakeLLMClient([]),
        _Evaluator(),
        RetrievalEvolutionConfig(generations=1),
    )
    monkeypatch.setattr(
        engine,
        "evolve",
        lambda genome, _train, _dev: _result(
            genome, child_release.genome, accepted=True
        ),
    )
    adapter = RetrievalEvolutionPhaseAdapter(
        engine,
        parent_release_path=parent_release.path,
        accepted_release_path=lambda _result: child_release.path,
    )

    outcome = adapter.run(parent, (object(),), (object(),))

    assert outcome.accepted is True
    assert outcome.improved is True
    assert outcome.bundle.retrieval_genome == child_release.genome
    assert outcome.bundle.retrieval_release_payload["manifest"]["state"] == "accepted"


def test_retrieval_adapter_rejects_release_that_does_not_match_engine_winner(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches swapping a different accepted filesystem release after Dev."""
    parent_release = _accepted_release(tmp_path / "releases", "v001", "v000")
    winner = replace(
        RetrievalGenome.seed(),
        version="v002",
        parent="v001",
        round1_strategy="entity_first",
    )
    parent = _bound_policy(parent_release)
    engine = RetrievalEvolutionEngine(
        FakeLLMClient([]), _Evaluator(), RetrievalEvolutionConfig(generations=1)
    )
    monkeypatch.setattr(
        engine,
        "evolve",
        lambda genome, _train, _dev: _result(genome, winner, accepted=True),
    )
    adapter = RetrievalEvolutionPhaseAdapter(
        engine,
        parent_release_path=parent_release.path,
        accepted_release_path=lambda _result: parent_release.path,
    )

    outcome = adapter.run(parent, (object(),), (object(),))

    assert outcome.accepted is False
    assert outcome.bundle is parent
    assert "winner" in outcome.reason


def test_retrieval_adapter_rejects_parent_not_matching_operator_loaded_release(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches treating a serialized Parent payload as release authority."""
    releases = tmp_path / "releases"
    trusted_parent = _accepted_release(releases, "v001", "v000")
    other_parent = _accepted_release(
        releases, "v002", "v001", strategy="entity_first"
    )
    parent = _bound_policy(trusted_parent)
    engine = RetrievalEvolutionEngine(
        FakeLLMClient([]), _Evaluator(), RetrievalEvolutionConfig(generations=1)
    )
    monkeypatch.setattr(
        engine,
        "evolve",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("untrusted Parent must fail before evolution")
        ),
    )

    outcome = RetrievalEvolutionPhaseAdapter(
        engine,
        parent_release_path=other_parent.path,
        accepted_release_path=lambda _result: other_parent.path,
    ).run(parent, (object(),), (object(),))

    assert outcome.accepted is False
    assert outcome.bundle is parent
    assert "Parent" in outcome.reason


def test_retrieval_adapter_accepts_trusted_contiguous_rebase_of_internal_winner(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches rejecting Task 8's required vNNN publication rebase."""
    parent_release = _accepted_release(tmp_path / "releases", "v001", "v000")
    internal_winner = replace(
        RetrievalGenome.seed(),
        version="v004",
        parent="v001",
        round1_strategy="entity_first",
    )
    published = _accepted_release(
        tmp_path / "releases", "v002", "v001", strategy="entity_first"
    )
    parent = _bound_policy(parent_release)
    engine = RetrievalEvolutionEngine(
        FakeLLMClient([]), _Evaluator(), RetrievalEvolutionConfig(generations=1)
    )
    monkeypatch.setattr(
        engine,
        "evolve",
        lambda genome, _train, _dev: _result(
            genome, internal_winner, accepted=True
        ),
    )

    outcome = RetrievalEvolutionPhaseAdapter(
        engine,
        parent_release_path=parent_release.path,
        accepted_release_path=lambda _result: published.path,
    ).run(parent, (object(),), (object(),))

    assert outcome.accepted is True
    assert outcome.bundle.retrieval_genome == published.genome


def test_retrieval_adapter_rejects_semantic_noop_before_release_publication(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a metadata-only winner invoking an accepted-release publisher."""
    parent_release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(parent_release)
    noop_winner = replace(
        parent_release.genome,
        version="v004",
        parent=parent_release.genome.version,
    )
    engine = RetrievalEvolutionEngine(
        FakeLLMClient([]), _Evaluator(), RetrievalEvolutionConfig(generations=1)
    )
    monkeypatch.setattr(
        engine,
        "evolve",
        lambda genome, _train, _dev: _result(genome, noop_winner, accepted=True),
    )

    outcome = RetrievalEvolutionPhaseAdapter(
        engine,
        parent_release_path=parent_release.path,
        accepted_release_path=lambda _result: (_ for _ in ()).throw(
            AssertionError("no-op Retrieval result must not reach publication")
        ),
    ).run(parent, (object(),), (object(),))

    assert outcome.accepted is False
    assert outcome.improved is False
    assert outcome.bundle is parent


def test_retrieval_adapter_preserves_decision_lineage_with_fresh_bundle_version(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches Retrieval release identity overwriting the global bundle lineage."""
    releases = tmp_path / "releases"
    _accepted_release(releases, "v001", "v000")
    parent_release = _accepted_release(releases, "v002", "v001")
    child_release = _accepted_release(
        releases, "v003", "v002", strategy="entity_first"
    )
    retrieval_parent = _bound_policy(parent_release)
    parent = replace(
        retrieval_parent,
        version="v003",
        parent=retrieval_parent.version,
        decision_prompt="Accepted Decision v003.",
    )
    internal_winner = replace(
        child_release.genome,
        version="v005",
        parent=parent_release.genome.version,
    )
    engine = RetrievalEvolutionEngine(
        FakeLLMClient([]), _Evaluator(), RetrievalEvolutionConfig(generations=1)
    )
    monkeypatch.setattr(
        engine,
        "evolve",
        lambda genome, _train, _dev: _result(
            genome, internal_winner, accepted=True
        ),
    )

    outcome = RetrievalEvolutionPhaseAdapter(
        engine,
        parent_release_path=parent_release.path,
        accepted_release_path=lambda _result: child_release.path,
    ).run(parent, (object(),), (object(),))

    assert outcome.accepted is True
    assert outcome.bundle.version == "v004"
    assert outcome.bundle.parent == parent.version
    assert outcome.bundle.decision_prompt == parent.decision_prompt
    assert outcome.bundle.retrieval_genome == child_release.genome


def test_decision_adapter_wires_the_existing_decision_targeted_engine(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches bypassing the existing Decision Train/Dev acceptance engine."""
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)
    child = replace(
        parent,
        version="v101",
        parent=parent.version,
        decision_prompt="Accepted Decision prompt.",
    )
    engine = CoEvolutionEngine(
        FakeLLMClient([]),
        lambda _policy: None,
        CoEvolutionConfig(generations=1, mode="genome", target="decision"),
    )
    monkeypatch.setattr(
        engine,
        "evolve",
        lambda *_args: (child, (SimpleNamespace(accepted_version=child.version),)),
    )

    outcome = DecisionEvolutionPhaseAdapter(
        engine, accepted_release_path=release.path
    ).run(
        parent, (object(),), (object(),)
    )

    assert outcome.accepted is True
    assert outcome.bundle is child
    assert outcome.bundle.retrieval_release_sha256 == parent.retrieval_release_sha256


@pytest.mark.parametrize("failure", ("noop", "cross_coordinate"))
def test_decision_adapter_rejects_nondecision_engine_outcomes_as_exact_parent(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Catches trusting an engine acceptance marker without coordinate ownership."""
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)
    changes = (
        {}
        if failure == "noop"
        else {"coding_generation_prompt": "Foreign Numerical mutation."}
    )
    child = replace(
        parent,
        version="v101",
        parent=parent.version,
        changelog="engine claimed acceptance",
        **changes,
    )
    engine = CoEvolutionEngine(
        FakeLLMClient([]),
        lambda _policy: None,
        CoEvolutionConfig(generations=1, mode="genome", target="decision"),
    )
    monkeypatch.setattr(
        engine,
        "evolve",
        lambda *_args: (
            child,
            (SimpleNamespace(accepted_version=child.version),),
        ),
    )

    outcome = DecisionEvolutionPhaseAdapter(
        engine,
        accepted_release_path=release.path,
    ).run(parent, (object(),), (object(),))

    assert outcome.accepted is False
    assert outcome.improved is False
    assert outcome.bundle is parent


def test_decision_adapter_rejects_accepted_child_with_wrong_bundle_parent(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a Decision acceptance marker detaching global bundle lineage."""
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)
    child = replace(
        parent,
        version="v101",
        parent="v099",
        decision_prompt="Accepted but detached Decision prompt.",
    )
    engine = CoEvolutionEngine(
        FakeLLMClient([]),
        lambda _policy: None,
        CoEvolutionConfig(generations=1, mode="genome", target="decision"),
    )
    monkeypatch.setattr(
        engine,
        "evolve",
        lambda *_args: (
            child,
            (SimpleNamespace(accepted_version=child.version),),
        ),
    )

    outcome = DecisionEvolutionPhaseAdapter(
        engine,
        accepted_release_path=release.path,
    ).run(parent, (object(),), (object(),))

    assert outcome.accepted is False
    assert outcome.bundle is parent


def test_decision_adapter_advances_shared_policy_version_past_parent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches Decision children colliding with an accepted Retrieval version."""
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)
    engine = CoEvolutionEngine(
        FakeLLMClient([]),
        lambda _policy: None,
        CoEvolutionConfig(generations=1, mode="genome", target="decision"),
    )
    observed: list[int] = []

    def evolve(*_args):
        observed.append(engine._version)
        return parent, ()

    monkeypatch.setattr(engine, "evolve", evolve)

    outcome = DecisionEvolutionPhaseAdapter(
        engine, accepted_release_path=release.path
    ).run(
        parent, (object(),), (object(),)
    )

    assert observed == [2]
    assert outcome.bundle is parent


class _QueuedPhase:
    def __init__(self, bundles: list[HarnessPolicy], target: str) -> None:
        self.bundles = list(bundles)
        self.target = target
        self.calls = 0

    def run(self, parent, _train, _dev):
        self.calls += 1
        candidate = self.bundles.pop(0)
        return CoordinatePhaseOutcome(
            target=self.target,
            bundle=candidate,
            accepted=True,
            improved=True,
            reason="accepted",
        )


def test_alternate_cycle_is_retrieval_first_decision_second_then_diagnostic(
    tmp_path,
) -> None:
    """Catches simultaneous or wrongly ordered multi-module mutation."""
    releases = tmp_path / "releases"
    first = _accepted_release(releases, "v001", "v000")
    second = _accepted_release(releases, "v002", "v001", strategy="entity_first")
    third = _accepted_release(releases, "v003", "v002", strategy="contrastive")
    seed = _bound_policy(first)
    retrieval_one = embed_retrieval_release(seed, second, changelog="r1")
    decision = replace(
        retrieval_one,
        version="v101",
        parent=retrieval_one.version,
        decision_prompt="Decision two.",
    )
    retrieval_two = embed_retrieval_release(decision, third, changelog="r2")
    retrieval_phase = _QueuedPhase([retrieval_one, retrieval_two], "retrieval")
    decision_phase = _QueuedPhase([decision], "decision")
    controller = CoordinateEvolutionController(
        retrieval_phase,
        decision_phase,
        CoordinateEvolutionConfig(phase="alternate", generations=3),
        diagnostics=lambda _bundle: CoordinateDiagnostics(
            retrieval_gain=-0.5,
            decision_regret=0.1,
        ),
    )

    accepted, trace = controller.run(seed, (object(),), (object(),))

    assert [step.target for step in trace] == ["retrieval", "decision", "retrieval"]
    assert [step.changed_modules for step in trace] == [
        ("retrieval",),
        ("decision",),
        ("retrieval",),
    ]
    assert accepted is retrieval_two
    assert accepted.public_test_accessed is False
    assert trace[0].parent_bytes_sha256 == hashlib.sha256(
        seed.canonical_bytes()
    ).hexdigest()


@pytest.mark.parametrize("failure", ("rejected", "noop", "cross_coordinate"))
def test_failed_coordinate_phase_returns_the_exact_parent_bundle(
    tmp_path, failure: str
) -> None:
    """Catches rejected phases leaking versions, changelogs, or foreign edits."""
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)
    if failure == "rejected":
        candidate = replace(parent, version="v999", changelog="rejected drift")
        phase_outcome = CoordinatePhaseOutcome(
            target="decision",
            bundle=candidate,
            accepted=False,
            improved=False,
            reason="Dev rejected",
        )
    elif failure == "noop":
        candidate = replace(parent, version="v999", changelog="no improvement")
        phase_outcome = CoordinatePhaseOutcome(
            target="decision",
            bundle=candidate,
            accepted=True,
            improved=False,
            reason="no strict gain",
        )
    else:
        candidate = replace(
            parent,
            version="v999",
            coding_generation_prompt="foreign Numerical mutation",
            decision_prompt="Decision mutation",
        )
        phase_outcome = CoordinatePhaseOutcome(
            target="decision",
            bundle=candidate,
            accepted=True,
            improved=True,
            reason="claimed accepted",
        )

    class Phase:
        def run(self, *_args):
            return phase_outcome

    controller = CoordinateEvolutionController(
        None,
        Phase(),
        CoordinateEvolutionConfig(phase="decision", generations=1),
    )
    before = parent.canonical_bytes()

    accepted, trace = controller.run(parent, (object(),), (object(),))

    assert accepted is parent
    assert accepted.canonical_bytes() == before
    assert trace[0].accepted is False
    assert trace[0].parent_bytes_sha256 == trace[0].accepted_bytes_sha256


def test_rejected_phase_cannot_mutate_parent_skill_snapshots_in_place(
    tmp_path,
) -> None:
    """Catches shallow frozen policies whose nested dicts can change by alias."""
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = replace(
        _bound_policy(release),
        coding_skills=({"name": "frozen Coding Skill"},),
        decision_skills=({"name": "frozen Decision Skill"},),
    )

    class Phase:
        def run(self, received, *_args):
            try:
                received.coding_skills[0]["name"] = "tampered after fingerprint"
                received.decision_skills[0]["name"] = "tampered after fingerprint"
            except TypeError:
                pass
            return CoordinatePhaseOutcome(
                target="decision",
                bundle=received,
                accepted=False,
                improved=False,
                reason="rejected",
            )

    controller = CoordinateEvolutionController(
        None,
        Phase(),
        CoordinateEvolutionConfig(phase="decision", generations=1),
    )
    before = parent.canonical_bytes()

    accepted, trace = controller.run(parent, (object(),), (object(),))

    assert accepted is parent
    assert accepted.canonical_bytes() == before
    assert trace[0].accepted_bytes_sha256 == trace[0].parent_bytes_sha256


def test_rejected_phase_cannot_mutate_parent_workflow_through_caller_alias(
    tmp_path,
) -> None:
    """Catches a caller-owned workflow list changing an accepted bundle in place."""
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    workflow_alias = ["retrieve", "decide"]
    parent = replace(_bound_policy(release), workflow=workflow_alias)

    class Phase:
        def run(self, received, *_args):
            workflow_alias.append("retrieve")
            return CoordinatePhaseOutcome(
                target="decision",
                bundle=received,
                accepted=False,
                improved=False,
                reason="rejected",
            )

    controller = CoordinateEvolutionController(
        None,
        Phase(),
        CoordinateEvolutionConfig(phase="decision", generations=1),
    )
    before = parent.canonical_bytes()

    accepted, trace = controller.run(parent, (object(),), (object(),))

    assert parent.workflow == ("retrieve", "decide")
    assert isinstance(parent.workflow, tuple)
    assert json.loads(parent.canonical_bytes())["workflow"] == [
        "retrieve",
        "decide",
    ]
    assert accepted is parent
    assert accepted.canonical_bytes() == before
    assert trace[0].accepted_bytes_sha256 == trace[0].parent_bytes_sha256


@pytest.mark.parametrize(
    "workflow",
    ("retrieve", ["retrieve", 1]),
)
def test_harness_policy_rejects_nonsequence_or_nonstr_workflow(workflow) -> None:
    """Catches malformed workflow values bypassing immutable normalization."""
    with pytest.raises(ValueError, match="workflow"):
        HarnessPolicy(workflow=workflow)


def test_public_access_claim_is_rejected_without_changing_parent(tmp_path) -> None:
    """Catches a phase normalizing Public access into an ordinary rejection."""
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)

    class Phase:
        def run(self, *_args):
            return CoordinatePhaseOutcome(
                target="decision",
                bundle=replace(parent, decision_prompt="changed"),
                accepted=True,
                improved=True,
                reason="accepted",
                public_test_accessed=True,
            )

    controller = CoordinateEvolutionController(
        None,
        Phase(),
        CoordinateEvolutionConfig(phase="decision", generations=1),
    )

    with pytest.raises(ValueError, match="Public"):
        controller.run(parent, (object(),), (object(),))
