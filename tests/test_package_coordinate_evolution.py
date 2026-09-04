from __future__ import annotations

from dataclasses import replace

import pytest

from evolving_loop.co_evolution import HarnessPolicy, embed_retrieval_release
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
    retrieval_release = _accepted_release(
        tmp_path / "retrieval-child", "v002", "v001", strategy="entity_first"
    )
    retrieval_policy = replace(
        embed_retrieval_release(
            parent.bundle.policy,
            retrieval_release,
            changelog="changed Retrieval release",
        ),
        version="v002",
        parent=parent.bundle.policy.version,
    )
    child = parent.with_policy(retrieval_policy, target="retrieval")

    assert child.bundle.parent_sha256 == parent.bundle.fingerprint()
    assert child.bundle.coordinate == "retrieval"
    assert child.bundle.acceptance_evidence_sha256 is None
    assert child.registry is parent.registry
    assert child.bundle.numerical_release_payload == parent.bundle.numerical_release_payload
    assert child.bundle.numerical_manifest_sha256 == parent.bundle.numerical_manifest_sha256
    parent_fingerprints = package_principal_fingerprints(parent.bundle)
    child_fingerprints = package_principal_fingerprints(child.bundle)
    assert child_fingerprints["retrieval"] != parent_fingerprints["retrieval"]
    assert child_fingerprints["numerical"] == parent_fingerprints["numerical"]
    assert child_fingerprints["decision"] == parent_fingerprints["decision"]
    with pytest.raises(ValueError, match="crossed module ownership"):
        parent.with_policy(
            replace(parent.bundle.policy, decision_prompt="cross-coordinate change"),
            target="retrieval",
        )
    sealed = child.bundle.seal_acceptance("a" * 64)
    assert sealed.acceptance_evidence_sha256 == "a" * 64
    with pytest.raises(ValueError, match="once"):
        sealed.seal_acceptance("b" * 64)


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
    for changed, target in (
        (changed_numerical, "numerical"),
        (changed_retrieval, "retrieval"),
        (changed_decision, "decision"),
    ):
        changed_fingerprints = package_principal_fingerprints(changed)
        for coordinate in ("numerical", "retrieval", "decision"):
            if coordinate == target:
                assert changed_fingerprints[coordinate] != fingerprints[coordinate]
            else:
                assert changed_fingerprints[coordinate] == fingerprints[coordinate]


def test_numerical_mutation_replaces_release_and_registry_atomically(tmp_path) -> None:
    task, parent = _bundle(tmp_path)
    release_payload = parent.bundle.to_payload()["numerical_release_payload"]
    assert isinstance(release_payload, dict)
    release_payload["source_fingerprints"] = {"dictionary": "9" * 64}
    release = parse_numerical_supply_release(release_payload)
    package = _package()
    registry = _frozen_registry(
        (
            (
                task,
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

    child = parent.with_numerical(release, registry)

    assert child.bundle.parent_sha256 == parent.bundle.fingerprint()
    assert child.bundle.coordinate == "numerical"
    assert child.bundle.numerical_release_sha256 == release.fingerprint
    assert child.bundle.numerical_manifest_sha256 == registry.fingerprint
    assert child.registry is registry
    assert child.bundle.policy == parent.bundle.policy
    assert child.bundle.runtime_fingerprints == parent.bundle.runtime_fingerprints
    parent_fingerprints = package_principal_fingerprints(parent.bundle)
    child_fingerprints = package_principal_fingerprints(child.bundle)
    assert child_fingerprints["numerical"] != parent_fingerprints["numerical"]
    assert child_fingerprints["retrieval"] == parent_fingerprints["retrieval"]
    assert child_fingerprints["decision"] == parent_fingerprints["decision"]


# --------------------------------------------------------------------------
# Task 7: two ordered Numerical -> Retrieval -> Decision cycles
# --------------------------------------------------------------------------

import hashlib

from evolving_loop.package_candidate_proposal import PackageCandidate
from evolving_loop.package_stage_runner import PackageCoordinatePhaseOutcome


def _numerical_child(current: PackageCoordinateState) -> PackageCoordinateState:
    parent_release = parse_numerical_supply_release(
        current.bundle.to_payload()["numerical_release_payload"]
    )
    generation = int(parent_release.version[1:]) + 1
    payload = parent_release.to_payload()
    payload["version"] = f"n{generation:03d}"
    payload["parent_sha256"] = parent_release.fingerprint
    payload["source_fingerprints"] = {
        "dictionary": hashlib.sha256(f"gen{generation}".encode()).hexdigest()
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


def _retrieval_child_factory(tmp_path):
    counter = {"n": 1}

    def build(current: PackageCoordinateState) -> PackageCoordinateState:
        counter["n"] += 1
        version = f"v{counter['n']:03d}"
        release = _accepted_release(
            tmp_path / f"retrieval-{version}",
            version,
            current.bundle.policy.version,
            strategy="entity_first",
        )
        policy = replace(
            embed_retrieval_release(
                current.bundle.policy, release, changelog="accepted Retrieval"
            ),
            version=version,
            parent=current.bundle.policy.version,
        )
        return current.with_policy(policy, target="retrieval")

    return build


def _decision_child(current: PackageCoordinateState) -> PackageCoordinateState:
    policy = replace(
        current.bundle.policy,
        decision_prompt=current.bundle.policy.decision_prompt + " :: accepted",
    )
    return current.with_policy(policy, target="decision")


class _ScriptPhase:
    def __init__(self, target, script) -> None:
        self.target = target
        self._script = list(script)
        self.calls = 0

    def run(self, parent, initial, *, generation):
        action = self._script[self.calls] if self.calls < len(self._script) else None
        self.calls += 1
        if action is None:
            return PackageCoordinatePhaseOutcome(
                target=self.target,
                parent=parent,
                finalist=None,
                selected=parent,
                accepted=False,
                improved=False,
                reason=f"{self.target} phase rejected every Child",
                evidence=(),
            )
        child = action(parent)
        sealed = child.seal_acceptance(
            hashlib.sha256(f"{self.target}:{generation}".encode()).hexdigest()
        )
        return PackageCoordinatePhaseOutcome(
            target=self.target,
            parent=parent,
            finalist=PackageCandidate(
                slot=0,
                target=self.target,
                state=child,
                proposal_sha256=child.bundle.fingerprint(),
            ),
            selected=sealed,
            accepted=True,
            improved=True,
            reason=f"{self.target} phase accepted a finalist",
            evidence=(),
        )


def _controller(tmp_path, *, reject=()):
    reject = set(reject if isinstance(reject, (list, tuple, set)) else (reject,))
    if "all" in reject:
        reject = {"numerical", "retrieval", "decision"}
    retrieval_build = _retrieval_child_factory(tmp_path)

    def script(target, build):
        # cycle 1 accepts unless rejected; cycle 2 always rejects.
        return (None if target in reject else build, None)

    return PackageCoordinateController(
        _ScriptPhase("numerical", script("numerical", _numerical_child)),
        _ScriptPhase("retrieval", script("retrieval", retrieval_build)),
        _ScriptPhase("decision", script("decision", _decision_child)),
    )


def test_controller_runs_two_ordered_coordinate_cycles(tmp_path) -> None:
    _task, seed_state = _bundle(tmp_path)
    controller = _controller(tmp_path)

    selected, trace = controller.run(seed_state, seed_state)

    assert tuple(step.target for step in trace) == (
        "numerical",
        "retrieval",
        "decision",
        "numerical",
        "retrieval",
        "decision",
    )
    assert tuple(step.generation for step in trace) == tuple(range(6))
    assert selected.bundle.generation == sum(step.accepted for step in trace)
    assert [step.accepted for step in trace[:3]] == [True, True, True]
    assert [step.accepted for step in trace[3:]] == [False, False, False]
    assert all(not step.public_test_accessed for step in trace)
    for step in trace:
        if step.accepted:
            assert step.changed_modules == (step.target,)


def test_rejected_coordinate_preserves_exact_parent_state(tmp_path) -> None:
    _task, seed_state = _bundle(tmp_path)

    selected, trace = _controller(tmp_path, reject="retrieval").run(seed_state, seed_state)

    step = trace[1]
    assert step.target == "retrieval"
    assert step.accepted is False
    assert step.accepted_bytes_sha256 == step.parent_bytes_sha256
    assert step.accepted_registry_sha256 == step.parent_registry_sha256


def test_decision_skips_without_nonseed_retrieval_release(tmp_path) -> None:
    _task, seed_state = _bundle(tmp_path, policy=HarnessPolicy())

    selected, trace = _controller(tmp_path, reject="retrieval").run(seed_state, seed_state)

    assert trace[2].target == "decision"
    assert trace[2].accepted is False
    assert (
        trace[2].reason
        == "Decision phase requires a non-v000 accepted Retrieval release"
    )


def test_controller_stops_after_complete_cycle_with_no_acceptance(tmp_path) -> None:
    _task, seed_state = _bundle(tmp_path)

    selected, trace = _controller(tmp_path, reject="all").run(seed_state, seed_state)

    assert tuple(step.target for step in trace) == ("numerical", "retrieval", "decision")
    assert selected.bundle.fingerprint() == seed_state.bundle.fingerprint()
    assert all(not step.accepted for step in trace)
