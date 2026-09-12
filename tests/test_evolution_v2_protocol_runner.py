from __future__ import annotations

import json

import pytest

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
