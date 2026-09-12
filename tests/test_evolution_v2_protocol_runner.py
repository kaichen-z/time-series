from __future__ import annotations

import json

import pytest
import evolving_loop.v2.protocol.runner as protocol_runner
from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload

from evolving_loop.v2.protocol import (
    ProtocolComponentV2,
    ProtocolRuntimeRegistry,
    run_protocol_evolution,
    freeze_protocol_handoff,
)
from tests.test_evolution_v2_protocol_compatibility import _protocol, compatibility_case


def _manifest(case):
    return {
        "schema_version": 1,
        "l0_commitment": "a" * 64,
        "runtime_fingerprint": "d" * 64,
        "corpus": case.corpus.to_payload(),
        "seed_protocol": _protocol().to_payload(),
        "host_input_files": [],
        "frozen_bundle_sha256": case.bundles[0].fingerprint(),
        "replacement_templates": [
            ProtocolComponentV2("diagnostic_metric", "absolute_movement", 1, 1).to_payload(),
            ProtocolComponentV2("loader", "changed_history_json", 1, 1).to_payload(),
        ],
    }


def _config():
    return {"schema_version": 1, "profile": "smoke", "seed": 17, "max_proposals": 2, "hard_limit_seconds": 120}


def test_rejection_keeps_active_pointer_bytes(compatibility_case, tmp_path):
    """A closed rejected step must not rewrite the accepted protocol pointer."""
    case = compatibility_case
    run_protocol_evolution(tmp_path, _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=case.inputs, stop_after=1)
    before = (tmp_path / "active_protocol.json").read_bytes()
    result = run_protocol_evolution(tmp_path, _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=case.inputs)
    assert result["rejected"] == 1
    assert (tmp_path / "active_protocol.json").read_bytes() == before


def test_closed_proposal_resume_matches_completion(compatibility_case, tmp_path):
    """Resuming a sealed prefix produces the same semantic completion bytes."""
    case = compatibility_case
    full = run_protocol_evolution(tmp_path / "full", _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=case.inputs)
    run_protocol_evolution(tmp_path / "resumed", _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=case.inputs, stop_after=1)
    resumed = run_protocol_evolution(tmp_path / "resumed", _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=case.inputs)
    assert resumed == full
    assert (tmp_path / "resumed" / "completion.json").read_bytes() == (tmp_path / "full" / "completion.json").read_bytes()


def test_frozen_handoff_has_no_public_execution(compatibility_case, tmp_path):
    """Completion exports only resolved accepted references and no Public score."""
    case = compatibility_case
    result = run_protocol_evolution(tmp_path, _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=case.inputs)
    handoff = json.loads((tmp_path / "frozen_protocol_handoff.json").read_text())
    assert result["public_test_accessed"] is False
    assert handoff["public_test_accessed"] is False
    release = json.loads((tmp_path / "releases" / f"{result['active_release_sha256']}.json").read_text())
    assert handoff["bundle_sha256"] == release["frozen_bundle_sha256"]
    before = (tmp_path / "completion.json").read_bytes()
    assert run_protocol_evolution(tmp_path, _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=case.inputs) == result
    assert (tmp_path / "completion.json").read_bytes() == before


def test_seed_protocol_replays_after_newer_activation(compatibility_case, tmp_path):
    """Publication leaves the exact old protocol and its components runnable."""
    case = compatibility_case
    seed = _protocol()
    runtime = ProtocolRuntimeRegistry().resolve(seed, case.inputs.runtime_inputs)
    before = runtime.evaluate(case.bundles[0], runtime.load_tasks()[:4], "train")
    run_protocol_evolution(tmp_path, _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=case.inputs)
    after = ProtocolRuntimeRegistry().resolve(seed, case.inputs.runtime_inputs).evaluate(case.bundles[0], runtime.load_tasks()[:4], "train")
    assert after.to_payload() == before.to_payload()


def test_resume_rejects_changed_manifest_and_unsealed_evidence(compatibility_case, tmp_path):
    """A resume may consume only the exact sealed input and object prefix."""
    case = compatibility_case
    manifest = _manifest(case)
    run_protocol_evolution(tmp_path, _config(), manifest, ProtocolRuntimeRegistry(), host_inputs=case.inputs, stop_after=1)
    changed = {**manifest, "runtime_fingerprint": "e" * 64}
    with pytest.raises(ValueError, match="commitment"):
        run_protocol_evolution(tmp_path, _config(), changed, ProtocolRuntimeRegistry(), host_inputs=case.inputs)
    (tmp_path / "evidence" / ("f" * 64 + ".json")).write_text("{}")
    with pytest.raises(ValueError, match="incomplete"):
        run_protocol_evolution(tmp_path, _config(), manifest, ProtocolRuntimeRegistry(), host_inputs=case.inputs)


def test_handoff_rejects_bundle_other_than_release_binding(compatibility_case):
    """Matching L0 alone cannot substitute a different frozen Bundle."""
    case = compatibility_case
    release = __import__("evolving_loop.v2.protocol.contracts", fromlist=["ProtocolReleaseV2"]).ProtocolReleaseV2(
        1, "a" * 64, "a" * 64, "b" * 64, case.bundles[0].fingerprint(), "d" * 64
    )
    with pytest.raises(ValueError, match="frozen Bundle"):
        freeze_protocol_handoff(release, case.bundles[1], resolve=lambda _sha: {})


def test_completed_resume_rejects_tampered_active_pointer(compatibility_case, tmp_path):
    """Read-only completion still validates publication linkage before returning."""
    case = compatibility_case
    result = run_protocol_evolution(tmp_path, _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=case.inputs)
    (tmp_path / "active_protocol.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="pointer"):
        run_protocol_evolution(tmp_path, _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=case.inputs)
    assert result["status"] == "protocol_evolution_complete"


def test_checkpoint_records_measured_elapsed_budget_use(compatibility_case, tmp_path, monkeypatch):
    """Budget checkpoints record real monotonic elapsed proposal work, not a constant."""
    ticks = iter((0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5))
    monkeypatch.setattr(protocol_runner.time, "monotonic", lambda: next(ticks))
    run_protocol_evolution(tmp_path, _config(), _manifest(compatibility_case), ProtocolRuntimeRegistry(), host_inputs=compatibility_case.inputs, stop_after=1)
    checkpoint = json.loads((tmp_path / "protocol_checkpoint.json").read_text())
    assert checkpoint["budget_ledger"]["charged_use"]["wall_seconds"] == 2.5


def test_elapsed_deadline_survives_resume_with_injected_clock(compatibility_case, tmp_path):
    """A real elapsed checkpoint prevents a later resume from opening work."""
    config = {**_config(), "hard_limit_seconds": 10}
    first_ticks = iter((0.0, 0.0, 0.0, 5.0, 5.0))
    partial = run_protocol_evolution(
        tmp_path, config, _manifest(compatibility_case), ProtocolRuntimeRegistry(),
        host_inputs=compatibility_case.inputs, stop_after=1, monotonic=lambda: next(first_ticks),
    )
    assert partial["status"] == "protocol_evolution_incomplete"
    checkpoint = json.loads((tmp_path / "protocol_checkpoint.json").read_text())
    assert checkpoint["budget_ledger"]["prior_elapsed_wall_seconds"] == 5.0
    resumed_ticks = iter((100.0, 106.0))
    with pytest.raises(ValueError, match="budget denied"):
        run_protocol_evolution(
            tmp_path, config, _manifest(compatibility_case), ProtocolRuntimeRegistry(),
            host_inputs=compatibility_case.inputs, monotonic=lambda: next(resumed_ticks),
        )


@pytest.mark.parametrize("field, value", (("active_release_sha256", "0" * 64), ("frozen_handoff_sha256", "1" * 64), ("accepted", 99), ("completed_proposal_sha256s", [])))
def test_completed_resume_rejects_tampered_completion_projection(compatibility_case, tmp_path, field, value):
    """Completion is a sealed semantic projection, not an unchecked receipt."""
    run_protocol_evolution(tmp_path, _config(), _manifest(compatibility_case), ProtocolRuntimeRegistry(), host_inputs=compatibility_case.inputs)
    path = tmp_path / "completion.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_bytes(canonical_v2_bytes(payload))
    with pytest.raises(ValueError, match="completed handoff or semantic completion"):
        run_protocol_evolution(tmp_path, _config(), _manifest(compatibility_case), ProtocolRuntimeRegistry(), host_inputs=compatibility_case.inputs)


def test_handoff_requires_exact_host_bundle_evidence(compatibility_case, tmp_path):
    """Finalization cannot invent or substitute a Bundle acceptance-evidence object."""
    case = compatibility_case
    evidence_sha = case.bundles[0].acceptance_evidence_sha256
    assert evidence_sha is not None
    missing = __import__("dataclasses").replace(case.inputs, bundle_acceptance_evidence={})
    with pytest.raises(ValueError, match="missing frozen Bundle acceptance evidence|does not close"):
        run_protocol_evolution(tmp_path / "missing", _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=missing)
    wrong = __import__("dataclasses").replace(case.inputs, bundle_acceptance_evidence={evidence_sha: {"schema_version": 1, "accepted": False}})
    with pytest.raises(ValueError, match="does not match"):
        run_protocol_evolution(tmp_path / "wrong", _config(), _manifest(case), ProtocolRuntimeRegistry(), host_inputs=wrong)


def test_resume_rejects_missing_migration_object(compatibility_case, tmp_path):
    """A sealed migration mapping remains resolvable on resume."""
    manifest = _manifest(compatibility_case)
    run_protocol_evolution(tmp_path, _config(), manifest, ProtocolRuntimeRegistry(), host_inputs=compatibility_case.inputs, stop_after=5)
    evidence = json.loads(next((tmp_path / "evidence").glob("*.json")).read_text())
    # Locate the accepted schema-migration evidence rather than relying on file order.
    for path in (tmp_path / "evidence").glob("*.json"):
        candidate = json.loads(path.read_text())
        if candidate["migration_mapping"] and any(item["old_envelope_sha256"] != item["new_envelope_sha256"] for item in candidate["migration_mapping"]):
            evidence = candidate
            break
    target = evidence["migration_mapping"][0]["new_envelope_sha256"]
    (tmp_path / "migrations" / f"{target}.json").unlink()
    with pytest.raises(ValueError, match="incomplete writes"):
        run_protocol_evolution(tmp_path, _config(), manifest, ProtocolRuntimeRegistry(), host_inputs=compatibility_case.inputs)


def test_budget_overrun_does_not_publish_candidate(compatibility_case, tmp_path):
    """Slow compatibility work cannot advance an active/release/progress publication."""
    config = {**_config(), "hard_limit_seconds": 10}
    ticks = iter((0.0, 0.0, 0.0, 6.0))
    with pytest.raises(ValueError, match="budget close failed"):
        run_protocol_evolution(
            tmp_path, config, _manifest(compatibility_case), ProtocolRuntimeRegistry(),
            host_inputs=compatibility_case.inputs, monotonic=lambda: next(ticks),
        )
    assert json.loads((tmp_path / "active_protocol.json").read_text()) == _protocol().to_payload()
    assert not (tmp_path / "releases").exists()
    assert not (tmp_path / "progress.jsonl").exists()
