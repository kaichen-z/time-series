from __future__ import annotations

from dataclasses import replace

from evolving_loop.co_evolution import HarnessPolicy, embed_retrieval_release
from evolving_loop.coordinate_evolution import CoordinatePhaseOutcome
from evolving_loop.package_coordinate_evolution import (
    PackageCoordinateBundle,
    PackageCoordinateController,
    package_principal_fingerprints,
)
from tests.test_coordinate_evolution import _accepted_release, _bound_policy
from tests.test_package_decision_evolution import _decision_task
from tests.test_package_retrieval_evolution import _frozen_registry, _package


def _runtime_fingerprints() -> dict[str, str]:
    return {
        "bridge_runtime": "1" * 64,
        "retrieval_runtime": "2" * 64,
        "decision_runtime": "3" * 64,
        "retrieval_verifier": "4" * 64,
        "metric_policy": "5" * 64,
    }


def _bundle(tmp_path, *, policy: HarnessPolicy | None = None):
    task = _decision_task()
    registry = _frozen_registry(((task, _package()),))
    if policy is None:
        release = _accepted_release(tmp_path / "releases", "v001", "v000")
        policy = _bound_policy(release)
    return task, PackageCoordinateBundle(
        generation=0,
        parent_sha256=None,
        numerical_manifest_sha256=registry.fingerprint,
        policy=policy,
        runtime_fingerprints=_runtime_fingerprints(),
    )


class _Phase:
    def __init__(self, outcome: CoordinatePhaseOutcome) -> None:
        self.outcome = outcome
        self.calls = 0

    def run(self, _parent, _train, _dev):
        self.calls += 1
        return self.outcome


def test_bundle_identity_binds_numerical_retrieval_and_decision(tmp_path) -> None:
    _task_value, bundle = _bundle(tmp_path)
    fingerprints = package_principal_fingerprints(bundle)

    changed_numerical = replace(
        bundle,
        numerical_manifest_sha256="9" * 64,
    )
    changed_decision = replace(
        bundle,
        policy=replace(bundle.policy, decision_prompt="changed Decision"),
    )
    other_release = _accepted_release(
        tmp_path / "other-releases",
        "v002",
        "v001",
        strategy="entity_first",
    )
    changed_retrieval = replace(
        bundle,
        policy=embed_retrieval_release(
            bundle.policy,
            other_release,
            changelog="changed Retrieval",
        ),
    )

    assert set(fingerprints) == {"numerical", "retrieval", "decision"}
    assert (
        package_principal_fingerprints(changed_numerical)["numerical"]
        != fingerprints["numerical"]
    )
    assert (
        package_principal_fingerprints(changed_retrieval)["retrieval"]
        != fingerprints["retrieval"]
    )
    assert (
        package_principal_fingerprints(changed_decision)["decision"]
        != fingerprints["decision"]
    )


def test_controller_accepts_retrieval_then_decision_only(tmp_path) -> None:
    task, parent = _bundle(tmp_path)
    retrieval_release = _accepted_release(
        tmp_path / "retrieval-child",
        "v002",
        "v001",
        strategy="entity_first",
    )
    retrieval_policy = embed_retrieval_release(
        parent.policy,
        retrieval_release,
        changelog="accepted Retrieval",
    )
    retrieval_policy = replace(
        retrieval_policy,
        version="v002",
        parent=parent.policy.version,
    )
    decision_policy = replace(
        retrieval_policy,
        version="v003",
        parent=retrieval_policy.version,
        decision_prompt="accepted Decision prompt",
    )
    retrieval = _Phase(
        CoordinatePhaseOutcome(
            "retrieval",
            retrieval_policy,
            True,
            True,
            "Retrieval accepted",
        )
    )
    decision = _Phase(
        CoordinatePhaseOutcome(
            "decision",
            decision_policy,
            True,
            True,
            "Decision accepted",
        )
    )

    selected, trace = PackageCoordinateController(retrieval, decision).run(
        parent,
        (task,),
        (task,),
    )

    assert tuple(step.target for step in trace) == ("retrieval", "decision")
    assert trace[0].changed_modules == ("retrieval",)
    assert trace[1].changed_modules == ("decision",)
    assert all(step.accepted for step in trace)
    assert selected.policy == decision_policy
    assert selected.numerical_manifest_sha256 == parent.numerical_manifest_sha256
    assert selected.parent_sha256 == trace[1].parent_bytes_sha256


def test_controller_rejection_preserves_exact_parent_bytes(tmp_path) -> None:
    task, parent = _bundle(tmp_path)
    release = _accepted_release(
        tmp_path / "crossed-release",
        "v002",
        "v001",
        strategy="entity_first",
    )
    crossed = embed_retrieval_release(
        parent.policy,
        release,
        changelog="crossed coordinate",
    )
    crossed = replace(
        crossed,
        version="v002",
        parent=parent.policy.version,
        decision_prompt="illegal simultaneous Decision change",
    )
    retrieval = _Phase(
        CoordinatePhaseOutcome(
            "retrieval",
            crossed,
            True,
            True,
            "claimed acceptance",
        )
    )

    selected, trace = PackageCoordinateController(retrieval, None).run(
        parent,
        (task,),
        (task,),
    )

    assert selected is parent
    assert selected.canonical_bytes() == parent.canonical_bytes()
    assert trace[0].changed_modules == ("retrieval", "decision")
    assert trace[0].accepted is False
    assert trace[0].accepted_bytes_sha256 == trace[0].parent_bytes_sha256


def test_controller_blocks_decision_before_non_v000_retrieval(tmp_path) -> None:
    task, parent = _bundle(tmp_path, policy=HarnessPolicy())
    decision = _Phase(
        CoordinatePhaseOutcome(
            "decision",
            replace(
                parent.policy,
                version="v001",
                parent="v000",
                decision_prompt="must not run",
            ),
            True,
            True,
            "claimed acceptance",
        )
    )

    selected, trace = PackageCoordinateController(None, decision).run(
        parent,
        (task,),
        (task,),
    )

    assert selected is parent
    assert decision.calls == 0
    assert trace[0].accepted is False
    assert "non-v000" in trace[0].reason


def test_controller_rejects_any_phase_that_reports_public_access(tmp_path) -> None:
    task, parent = _bundle(tmp_path)
    candidate = replace(
        parent.policy,
        version="v002",
        parent=parent.policy.version,
        decision_prompt="candidate",
    )
    decision = _Phase(
        CoordinatePhaseOutcome(
            "decision",
            candidate,
            True,
            True,
            "claimed acceptance",
            public_test_accessed=True,
        )
    )

    selected, trace = PackageCoordinateController(None, decision).run(
        parent,
        (task,),
        (task,),
    )

    assert selected is parent
    assert trace[0].accepted is False
    assert trace[0].public_test_accessed is True
    assert trace[0].accepted_bytes_sha256 == trace[0].parent_bytes_sha256
