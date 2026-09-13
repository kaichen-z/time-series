"""Protocol changes must not relabel frozen task-local numerical predictions."""
from dataclasses import replace

import pytest

from evolving_loop.package_numerical_supply import task_local_result_sha256
from evolving_loop.package_registry import numerical_package_fingerprint, task_registry_fingerprint
from evolving_loop.v2.cooperative.adapters import CooperativeArtifactCatalog
from evolving_loop.v2.numerical_qd.adapters import FrozenNumericalArtifactsV2
from evolving_loop.v2.protocol import ProtocolHostInputs, ProtocolRuntimeRegistry
from evolving_loop.v2.protocol.bridge import _seed_protocol
from tests.test_evolution_v2_frozen_shortlist import frozen_fixture


@pytest.fixture(scope="module")
def frozen_backbone_case():
    imported, adapter, _evidence_bytes = frozen_fixture()
    tasks = adapter.tasks
    registry = imported.envelope.restore(tasks)
    pair = FrozenNumericalArtifactsV2(imported.release, registry, imported.envelope, ())
    catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
    catalog.add_numerical(pair)

    def no_agent(*args, **kwargs):
        raise AssertionError("Numerical registry projection must not invoke an agent")

    host = ProtocolHostInputs(
        raw_fixture_records={task.numeric.task_id: task for task in tasks},
        canonical_records={task.numeric.task_id: task for task in tasks},
        catalog=catalog,
        frozen_numerical=pair,
        retrieval_factory=no_agent,
        decision_factory=no_agent,
        committed_task_metadata={
            task.numeric.task_id: task_registry_fingerprint(task) for task in tasks
        },
        l0_commitment_sha256="a" * 64,
        primary_metric_cap=5.0,
        baseline_verifier=lambda evidence: False,
        known_supports={},
    )
    return pair, tasks, host


def test_real_protocol_seed_preserves_schema2_packages_and_result_authority(frozen_backbone_case):
    """The seed must replay P2, not silently substitute last-value or mean."""
    pair, tasks, host = frozen_backbone_case
    before = pair.envelope.canonical_bytes()
    assert pair.release.schema_version == pair.envelope.schema_version == 2
    runtime = ProtocolRuntimeRegistry().resolve(_seed_protocol("a" * 64), host)
    selected = tasks[40:42]
    derived = runtime._derived_registry(pair, selected)

    assert derived.task_ids == tuple(task.numeric.task_id for task in selected)
    assert derived.release_sha256 == pair.release.fingerprint
    for task in selected:
        original = pair.registry.package_for(task)
        actual = derived.package_for(task)
        assert actual.protected_baseline.forecast == original.protected_baseline.forecast
        assert actual.final_forecast == original.final_forecast
        assert numerical_package_fingerprint(actual) == numerical_package_fingerprint(original)
        assert actual.component_fingerprints["task_local_result"] == task_local_result_sha256(
            actual.selection_decision,
            actual.component_fingerprints["hindcast_diagnostics"],
            actual.fallback_reason,
        )
    assert pair.envelope.canonical_bytes() == before


@pytest.mark.parametrize("backbone", ["last_value", "history_mean"])
def test_schema2_backbone_replacement_requires_fresh_numerical_authority(
    frozen_backbone_case, backbone,
):
    """Reject a changed forecast before it can reuse the sealed diagnostics."""
    pair, tasks, host = frozen_backbone_case
    seed = _seed_protocol("a" * 64)
    proposed = replace(seed, components=(
        replace(seed.components[0], implementation_id=backbone), *seed.components[1:],
    ))
    runtime = ProtocolRuntimeRegistry().resolve(proposed, host)
    with pytest.raises(ValueError, match="schema-2.*frozen_numerical"):
        runtime._derived_registry(pair, tasks[:1])


def test_frozen_backbone_resolves_and_preserves_the_full_registry(frozen_backbone_case):
    pair, tasks, host = frozen_backbone_case
    seed = _seed_protocol("a" * 64)
    frozen = replace(seed, components=(
        replace(seed.components[0], implementation_id="frozen_numerical"),
        *seed.components[1:],
    ))
    runtime = ProtocolRuntimeRegistry().resolve(frozen, host)
    derived = runtime._derived_registry(pair, tasks)
    assert derived.fingerprint == pair.registry.fingerprint
    assert len(derived.task_ids) == 100
