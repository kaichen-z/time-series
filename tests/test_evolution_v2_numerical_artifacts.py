"""Independent billable taxonomy oracle, not a copy of classifier conditions."""
import pytest


MATERIAL_KINDS = {
    "source", "config", "genome", "inventory", "screening_policy", "combined_policy", "recipe_policy",
    "structural_policy", "mutation_policy", "prompt", "proposer_request", "proposal_attempt",
    "executable_child", "evaluation", "rung_manifest", "task_result", "qd_entry", "cell_subset",
    "frozen_pair", "bootstrap_forecast",
}
CONTROL_KINDS = {
    "bundle", "qd_archive", "hyperband_state", "rung_record", "budget_checkpoint", "kernel_checkpoint",
    "runner_checkpoint", "manifest", "generation_status", "partial_rung", "bootstrap_preflight",
    "bootstrap_admission", "bootstrap_closure", "bootstrap_replay", "bootstrap_receipt",
}


def test_explicit_artifact_registry_matches_independent_taxonomy():
    from evolving_loop.v2.numerical_qd.artifacts import ARTIFACT_KINDS
    assert {kind.value for kind, spec in ARTIFACT_KINDS.items() if spec.billable} == MATERIAL_KINDS
    assert {kind.value for kind, spec in ARTIFACT_KINDS.items() if not spec.billable} == CONTROL_KINDS


@pytest.mark.parametrize("kind", [None, "unregistered", "budget_checkpoint", "kernel_checkpoint", "partial_rung"])
def test_material_cannot_gain_free_control_status_through_marker_keys(kind):
    from evolving_loop.v2.numerical_qd.artifacts import validate_artifact
    from evolving_loop.v2.contracts import canonical_v2_bytes
    from tests.test_evolution_v2_numerical_contracts import payloads, NumericalGenomeV2
    payload = payloads()[NumericalGenomeV2]
    payload["checkpoint_sha256"] = "f" * 64
    with pytest.raises((TypeError, ValueError)):
        validate_artifact(kind, canonical_v2_bytes(payload))
