from __future__ import annotations

from dataclasses import replace

import pytest

from evolving_loop.co_evolution import HarnessPolicy, embed_retrieval_release
from evolving_loop.coordinate_evolution import CoordinatePhaseOutcome
from evolving_loop.package_coordinate_evolution import (
    PackageCoordinateBundle,
    PackageCoordinateController,
    PackageCoordinateState,
    package_principal_fingerprints,
)
from evolving_loop.package_numerical_supply import (
    NumericalSupplyRelease,
    parse_numerical_supply_release,
)
from tests.test_coordinate_evolution import _accepted_release, _bound_policy
from tests.test_package_decision_evolution import _decision_task
from tests.test_package_retrieval_evolution import (
    _frozen_registry,
    _package,
    _seed_supply_release,
)


def _runtime_fingerprints() -> dict[str, str]:
    return {
        "bridge_runtime": "1" * 64,
        "numerical_runtime": "6" * 64,
        "retrieval_runtime": "2" * 64,
        "decision_runtime": "3" * 64,
        "retrieval_verifier": "4" * 64,
        "metric_policy": "5" * 64,
        "model_runtime": "7" * 64,
        "llm_runtime": "8" * 64,
    }


def _bundle(
    tmp_path,
    *,
    policy: HarnessPolicy | None = None,
    numerical_release: NumericalSupplyRelease | None = None,
    registry=None,
):
    task = _decision_task()
    numerical_release = numerical_release or _seed_supply_release()
    registry = registry or _frozen_registry(
        ((task, _package()),), release=numerical_release
    )
    if policy is None:
        release = _accepted_release(tmp_path / "releases", "v001", "v000")
        policy = _bound_policy(release)
    return task, PackageCoordinateState(
        PackageCoordinateBundle(
            generation=0,
            coordinate="seed",
            parent_sha256=None,
            numerical_release_payload=numerical_release.to_payload(),
            numerical_release_sha256=numerical_release.fingerprint,
            numerical_manifest_sha256=registry.fingerprint,
            policy=policy,
            runtime_fingerprints=_runtime_fingerprints(),
            acceptance_evidence_sha256=None,
        ),
        registry,
    )


def test_bundle_schema_binds_numerical_release_and_registry_manifest(tmp_path) -> None:
    _task, state = _bundle(tmp_path)
    bundle = state.bundle
    payload = bundle.to_payload()

    assert payload["schema_version"] == 2
    assert payload["coordinate"] == "seed"
    assert payload["numerical_release_sha256"] == state.registry.release_sha256
    assert payload["numerical_manifest_sha256"] == state.registry.fingerprint


def test_coordinate_state_rejects_release_or_registry_mismatch(tmp_path) -> None:
    task = _decision_task()
    release = _seed_supply_release()
    left = _frozen_registry(((task, _package()),), release=release)
    right = _frozen_registry(
        ((replace(task, target_description="different manifest"), _package()),),
        release=release,
    )
    right_payload = release.to_payload()
    right_payload["source_fingerprints"] = {"dictionary": "9" * 64}
    right_release = parse_numerical_supply_release(right_payload)
    _task, state = _bundle(tmp_path, numerical_release=release, registry=left)

    with pytest.raises(ValueError, match="registry manifest"):
        PackageCoordinateState(state.bundle, right)
    with pytest.raises(ValueError, match="Numerical release"):
        PackageCoordinateState(
            replace(
                state.bundle,
                numerical_release_payload=right_release.to_payload(),
                numerical_release_sha256=right_release.fingerprint,
            ),
            left,
        )


def test_seed_bundle_cannot_claim_a_parent_sha(tmp_path) -> None:
    _task, state = _bundle(tmp_path)

    with pytest.raises(ValueError, match="seed"):
        replace(state.bundle, parent_sha256="a" * 64)


def test_state_coordinate_mutators_preserve_unowned_bytes_and_seal_once(tmp_path) -> None:
    _task, parent = _bundle(tmp_path)
    retrieval_policy = replace(
        parent.bundle.policy,
        version="v002",
        parent=parent.bundle.policy.version,
        decision_prompt="changed Retrieval-owned policy payload",
    )
    child = parent.with_policy(retrieval_policy, target="retrieval")

    assert child.bundle.parent_sha256 == parent.bundle.fingerprint()
    assert child.bundle.coordinate == "retrieval"
    assert child.bundle.acceptance_evidence_sha256 is None
    assert child.registry is parent.registry
    assert child.bundle.numerical_release_payload == parent.bundle.numerical_release_payload
    assert child.bundle.numerical_manifest_sha256 == parent.bundle.numerical_manifest_sha256
    sealed = child.bundle.seal_acceptance("a" * 64)
    assert sealed.acceptance_evidence_sha256 == "a" * 64
    with pytest.raises(ValueError, match="once"):
        sealed.seal_acceptance("b" * 64)


class _Phase:
    def __init__(self, outcome: CoordinatePhaseOutcome) -> None:
        self.outcome = outcome
        self.calls = 0

    def run(self, _parent, _train, _dev):
        self.calls += 1
        return self.outcome


def test_bundle_identity_binds_numerical_retrieval_and_decision(tmp_path) -> None:
    _task_value, state = _bundle(tmp_path)
    bundle = state.bundle
    fingerprints = package_principal_fingerprints(bundle)

    changed_numerical_payload = bundle.to_payload()["numerical_release_payload"]
    assert isinstance(changed_numerical_payload, dict)
    changed_numerical_payload["source_fingerprints"] = {"dictionary": "9" * 64}
    changed_numerical_release = parse_numerical_supply_release(
        changed_numerical_payload
    )
    changed_numerical = replace(
        bundle,
        numerical_release_payload=changed_numerical_release.to_payload(),
        numerical_release_sha256=changed_numerical_release.fingerprint,
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
        parent.bundle.policy,
        retrieval_release,
        changelog="accepted Retrieval",
    )
    retrieval_policy = replace(
        retrieval_policy,
        version="v002",
        parent=parent.bundle.policy.version,
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
    assert selected.bundle.policy == decision_policy
    assert selected.bundle.numerical_manifest_sha256 == parent.bundle.numerical_manifest_sha256
    assert selected.bundle.parent_sha256 == trace[1].parent_bytes_sha256
    assert all(step.accepted_registry_sha256 == parent.registry.fingerprint for step in trace)


def test_controller_rejection_preserves_exact_parent_bytes(tmp_path) -> None:
    task, parent = _bundle(tmp_path)
    release = _accepted_release(
        tmp_path / "crossed-release",
        "v002",
        "v001",
        strategy="entity_first",
    )
    crossed = embed_retrieval_release(
        parent.bundle.policy,
        release,
        changelog="crossed coordinate",
    )
    crossed = replace(
        crossed,
        version="v002",
        parent=parent.bundle.policy.version,
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
    assert selected.bundle.canonical_bytes() == parent.bundle.canonical_bytes()
    assert trace[0].changed_modules == ("retrieval", "decision")
    assert trace[0].accepted is False
    assert trace[0].accepted_bytes_sha256 == trace[0].parent_bytes_sha256


def test_controller_blocks_decision_before_non_v000_retrieval(tmp_path) -> None:
    task, parent = _bundle(tmp_path, policy=HarnessPolicy())
    decision = _Phase(
        CoordinatePhaseOutcome(
            "decision",
            replace(
                parent.bundle.policy,
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
        parent.bundle.policy,
        version="v002",
        parent=parent.bundle.policy.version,
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
