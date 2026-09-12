from __future__ import annotations

import json

from evolving_loop.v2.protocol import (
    ProtocolComponentV2,
    ProtocolRuntimeRegistry,
    run_protocol_evolution,
)
from tests.test_evolution_v2_protocol_compatibility import _protocol, compatibility_case


def _manifest(case):
    return {
        "schema_version": 1,
        "l0_commitment": "a" * 64,
        "runtime_fingerprint": "d" * 64,
        "corpus": case.corpus.to_payload(),
        "seed_protocol": _protocol().to_payload(),
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
