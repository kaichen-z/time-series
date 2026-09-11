"""Independent billable taxonomy oracle, not a copy of classifier conditions."""
import pytest
import json
from dataclasses import replace
from types import SimpleNamespace


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


def bootstrap_fixture(tmp_path, *, preflight_extra=False):
    from evolving_loop.v2.kernel import SeedBootstrapAuthority
    from evolving_loop.v2.store import V2RunStore
    from tests.test_evolution_v2_numerical_runner import fixture
    config, seed, _, adapter = fixture(raw_seed=True)
    preflight = {"schema_version": 1, "stage": "seed_bootstrap", "seed_supply_sha256": seed.fingerprint,
        "input_sha256s": {"test": "1" * 64}, "protocol_sha256": config.kernel_protocol.fingerprint(),
        "budget_plan_sha256": config.budget.fingerprint(),
        "estimate": replace(config.budget.ceilings, wall_seconds=1.0).to_payload()}
    if preflight_extra:
        preflight["unpaid_payload"] = "BOOTSTRAP_SENTINEL"
    authority = SeedBootstrapAuthority(V2RunStore.create(tmp_path / "run"), config.kernel_protocol,
        config.budget, preflight, monotonic=adapter.monotonic)
    return authority, config, adapter


@pytest.mark.parametrize("kind", ["preflight", "admission", "admission_resource", "closed_forecast", "replayed_forecast"])
def test_bootstrap_real_writer_rejects_extra_fields_before_writing(tmp_path, kind):
    if kind == "preflight":
        with pytest.raises(ValueError):
            bootstrap_fixture(tmp_path, preflight_extra=True)
    else:
        authority, _, _ = bootstrap_fixture(tmp_path)
        value = {"kind": kind, "unpaid_payload": "BOOTSTRAP_SENTINEL"}
        if kind == "admission_resource":
            value = {"kind": "admission", "arguments": [0],
                "charged_use": authority.actual.to_payload() | {"unpaid_payload": "BOOTSTRAP_SENTINEL"}}
        elif kind == "admission":
            value.update(arguments=[0], charged_use=authority.actual.to_payload())
        elif kind == "closed_forecast":
            value.update(cache_sha256="a" * 64)
        else:
            value.update(cache_sha256="a" * 64, replay_index=0)
        with pytest.raises(ValueError):
            authority.event(value)
        assert authority.events == []
    assert all(b"BOOTSTRAP_SENTINEL" not in path.read_bytes() for path in (tmp_path / "run").rglob("*") if path.is_file())


@pytest.mark.parametrize("kind", ["preflight", "admission", "admission_resource", "closed_forecast", "replayed_forecast", "receipt"])
def test_bootstrap_real_resume_rejects_extra_fields_before_opening_work(tmp_path, monkeypatch, kind):
    from evolving_loop.v2.kernel import SeedBootstrapAuthority
    from evolving_loop.v2.numerical_qd import artifacts
    from evolving_loop.v2.contracts import fingerprint_payload
    from evolving_loop.v2.store import write_once_json
    host = SimpleNamespace(forecast_trusted=lambda *args, **kwargs: [0.0])
    # Build a hash-consistent malformed historical fixture below the validation
    # boundary, then restore the real boundary before resume. No decision mocked.
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "validate_artifact", lambda selected, raw: artifacts.ARTIFACT_KINDS[artifacts.ArtifactKindV2(selected)])
        authority, config, adapter = bootstrap_fixture(tmp_path, preflight_extra=kind == "preflight")
        original = SeedBootstrapAuthority.event
        def event(self, value):
            if kind == "admission_resource" and value["kind"] == "admission":
                value = value | {"charged_use": value["charged_use"] | {"unpaid_payload": "BOOTSTRAP_SENTINEL"}}
            return original(self, value | ({"unpaid_payload": "BOOTSTRAP_SENTINEL"} if value["kind"] == kind else {}))
        patch.setattr(SeedBootstrapAuthority, "event", event)
        authority.forecast(host, 0)
        authority.close("interrupted", "process_interrupted")
        if kind == "replayed_forecast":
            authority = SeedBootstrapAuthority.resume(authority.store, config.kernel_protocol, config.budget,
                authority.preflight["input_sha256s"], monotonic=adapter.monotonic)
            authority.forecast(host, 0)
            authority.close("interrupted", "process_interrupted")
    if kind == "receipt":
        previous = authority.store.root / f"seed_bootstrap_segments/{fingerprint_payload(authority.receipt)}.json"
        forged = authority.receipt | {"unpaid_payload": "BOOTSTRAP_SENTINEL"}
        previous.unlink()
        write_once_json(previous.parent / f"{fingerprint_payload(forged)}.json", forged)
    root = authority.store.root
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        SeedBootstrapAuthority.resume(authority.store, config.kernel_protocol, config.budget,
            authority.preflight["input_sha256s"], monotonic=adapter.monotonic)
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_bootstrap_real_write_and_resume_paths_use_every_registered_kind(tmp_path, monkeypatch):
    from evolving_loop.v2.numerical_qd import artifacts
    from evolving_loop.v2.kernel import SeedBootstrapAuthority
    from evolving_loop.v2.budget import ResourceUse
    original, seen = artifacts.validate_artifact, set()
    def validate(kind, raw):
        seen.add(artifacts.ArtifactKindV2(kind).value)
        return original(kind, raw)
    monkeypatch.setattr(artifacts, "validate_artifact", validate)
    authority, config, adapter = bootstrap_fixture(tmp_path)
    host = SimpleNamespace(forecast_trusted=lambda *args, **kwargs: [0.0])
    authority.forecast(host, 0)
    authority.close("interrupted", "process_interrupted")
    resumed = SeedBootstrapAuthority.resume(authority.store, config.kernel_protocol, config.budget,
        authority.preflight["input_sha256s"], monotonic=adapter.monotonic)
    assert resumed.forecast(host, 0) == [0.0]
    receipt = resumed.close("passed")
    assert receipt["resource_use"] == ResourceUse().to_payload() and receipt["status"] == "passed"
    assert {"bootstrap_preflight", "bootstrap_admission", "bootstrap_closure", "bootstrap_replay",
            "bootstrap_receipt", "bootstrap_forecast"} <= seen


@pytest.mark.parametrize("reason", [{"payload": "RECEIPT_REASON_SENTINEL"}, ["RECEIPT_REASON_SENTINEL"], "", 1])
def test_bootstrap_close_rejects_nonprimitive_reason_before_publication(tmp_path, monkeypatch, reason):
    from evolving_loop.v2.kernel import SeedBootstrapAuthority
    from evolving_loop.v2.budget import BudgetLedger
    authority, config, adapter = bootstrap_fixture(tmp_path)
    authority.forecast(SimpleNamespace(forecast_trusted=lambda *args, **kwargs: [0.0]), 0)
    root = authority.store.root
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        authority.close("interrupted", reason)
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert all(b"RECEIPT_REASON_SENTINEL" not in raw for raw in before.values())
    monkeypatch.setattr(BudgetLedger, "reserve_stage", lambda *args, **kwargs: pytest.fail("invalid receipt opened resume permit"))
    with pytest.raises(ValueError):
        SeedBootstrapAuthority.resume(authority.store, config.kernel_protocol, config.budget,
            authority.preflight["input_sha256s"], monotonic=adapter.monotonic)


RECEIPT_INVALID_FIELDS = [
    ("schema_version", 2.0), ("status", "unknown"), ("status", {"payload": "RECEIPT_SCHEMA_SENTINEL"}),
    ("preflight_sha256", "bad"), ("reservation_sha256", "bad"), ("chain_sha256", "a" * 64),
    ("allowed", 1), ("closure_reason", ""), ("closure_reason", {"payload": "RECEIPT_SCHEMA_SENTINEL"}),
    ("reason", {"payload": "RECEIPT_SCHEMA_SENTINEL"}), ("events", {}), ("events", ["bad"]),
    ("events", ["a" * 64, "a" * 64]), ("segment_index", True), ("segment_index", -1),
    ("previous_receipt_sha256", "bad"), ("previous_receipt_sha256", "a" * 64),
    ("replay_total", True), ("replay_total", -1), ("replay_consumed", 1),
    ("budget_before.schema_version", 1.0), ("budget_before.prior_elapsed_wall_seconds", 0),
    ("budget_before.finalization_started", 0), ("budget_before.exhausted_reason", {}),
    ("budget_before.closed_stage_ids", [1]), ("budget_before.closed_reservation_sha256s", ["bad"]),
    ("budget_after.open_reservations", {}), ("budget_after.charged_use.task_executions", 2),
]


@pytest.mark.parametrize("field,bad", RECEIPT_INVALID_FIELDS)
def test_bootstrap_receipt_real_writer_rejects_invalid_semantics_before_write(tmp_path, field, bad):
    from evolving_loop.v2.budget import BudgetLedger
    authority, _, _ = bootstrap_fixture(tmp_path)
    authority.forecast(SimpleNamespace(forecast_trusted=lambda *args, **kwargs: [0.0]), 0)
    value = json.loads(json.dumps(authority.close("interrupted", "process_interrupted")))
    target = value
    names = field.split(".")
    for name in names[:-1]:
        target = target[name]
    target[names[-1]] = bad
    if names[0].startswith("budget_"):
        value[names[0]]["checkpoint_sha256"] = BudgetLedger.checkpoint_sha256(value[names[0]])
    root = authority.store.root
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        authority._write_artifact(root / "seed_bootstrap_receipt.json", value)
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("reason", [{"payload": "RECEIPT_REASON_SENTINEL"}, [], "", False])
def test_bootstrap_resume_rejects_hash_consistent_invalid_receipt_reason_before_permit(tmp_path, monkeypatch, reason):
    from evolving_loop.v2.kernel import SeedBootstrapAuthority
    from evolving_loop.v2.budget import BudgetLedger
    from evolving_loop.v2.contracts import fingerprint_payload
    from evolving_loop.v2.store import write_once_json
    authority, config, adapter = bootstrap_fixture(tmp_path)
    authority.forecast(SimpleNamespace(forecast_trusted=lambda *args, **kwargs: [0.0]), 0)
    receipt = authority.close("interrupted", "process_interrupted")
    directory = authority.store.root / "seed_bootstrap_segments"
    (directory / f"{fingerprint_payload(receipt)}.json").unlink()
    forged = receipt | {"reason": reason}
    write_once_json(directory / f"{fingerprint_payload(forged)}.json", forged)
    root = authority.store.root
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    monkeypatch.setattr(BudgetLedger, "reserve_stage", lambda *args, **kwargs: pytest.fail("invalid receipt opened resume permit"))
    with pytest.raises(ValueError):
        SeedBootstrapAuthority.resume(authority.store, config.kernel_protocol, config.budget,
            authority.preflight["input_sha256s"], monotonic=adapter.monotonic)
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_bootstrap_close_verifies_event_graph_before_publishing_receipt(tmp_path):
    authority, _, _ = bootstrap_fixture(tmp_path)
    authority.forecast(SimpleNamespace(forecast_trusted=lambda *args, **kwargs: [0.0]), 0)
    authority.event({"kind": "closed_forecast", "cache_sha256": "a" * 64})
    root = authority.store.root
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        authority.close("interrupted", "process_interrupted")
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("status,reason,overrun", [
    ("passed", "unexpected_reason", False), ("failed", None, False), ("stopped", None, False),
    ("interrupted", None, False), ("passed", None, True), ("interrupted", "process_interrupted", True),
])
def test_bootstrap_close_rejects_illegal_outcome_combination_before_receipt(tmp_path, status, reason, overrun):
    authority, _, adapter = bootstrap_fixture(tmp_path)
    authority.forecast(SimpleNamespace(forecast_trusted=lambda *args, **kwargs: [0.0]), 0)
    if overrun:
        adapter.monotonic.advance(2.0)  # The committed bootstrap estimate is one second.
    root = authority.store.root
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        authority.close(status, reason)
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert not (root / "seed_bootstrap_receipt.json").exists()
    assert not (root / "run_manifest.json").exists()


def test_kernel_cannot_adopt_disallowed_bootstrap_passed_receipt(tmp_path):
    from evolving_loop.v2.kernel import EvolutionKernel
    from evolving_loop.v2.bundle import EvolutionBundleV2
    from evolving_loop.v2.contracts import fingerprint_payload
    from evolving_loop.v2.store import write_once_json
    authority, config, adapter = bootstrap_fixture(tmp_path)
    authority.forecast(SimpleNamespace(forecast_trusted=lambda *args, **kwargs: [0.0]), 0)
    adapter.monotonic.advance(2.0)
    receipt = authority.close("failed", "budget_overrun")
    assert receipt["allowed"] is False
    forged = receipt | {"status": "passed", "reason": None}
    # Simulate an invalid imported issuer and immutable receipt, not a mocked decision.
    authority.receipt = forged
    root = authority.store.root
    (root / "seed_bootstrap_receipt.json").unlink()
    (root / f"seed_bootstrap_segments/{fingerprint_payload(receipt)}.json").unlink()
    write_once_json(root / "seed_bootstrap_receipt.json", forged)
    write_once_json(root / f"seed_bootstrap_segments/{fingerprint_payload(forged)}.json", forged)
    seed = EvolutionBundleV2(2, 0, None, authority.preflight["seed_supply_sha256"], "f" * 64,
        **dict(config.fixed_bundle_components), protocol_fingerprint=config.kernel_protocol.fingerprint(),
        runtime_fingerprints=config.runtime_fingerprints, acceptance_evidence_sha256=None)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        EvolutionKernel(authority.store, config.kernel_protocol, authority.budget, seed=seed, bootstrap_authority=authority)
    assert authority.adopted is False
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
