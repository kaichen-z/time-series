from __future__ import annotations

import json
import hashlib

import pytest

from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
from evolving_loop.v2.cli import build_parser
from evolving_loop.v2.protocol.cli import dispatch_protocol, make_smoke_inputs


def test_protocol_command_uses_unified_entrypoint():
    args = build_parser().parse_args(["protocol-evolve", "--config", "c.json", "--input-manifest", "i.json", "--output-dir", "run"])
    assert args.command == "protocol-evolve"


def test_offline_smoke_runs_all_five_kinds_and_one_rejection(tmp_path):
    """The unified offline path executes real P3 scoring without Public access."""
    inputs = make_smoke_inputs(tmp_path / "inputs")
    config = tmp_path / "smoke.json"
    config.write_bytes(b'{"hard_limit_seconds":120,"max_proposals":6,"profile":"smoke","schema_version":1,"seed":17}\n')
    result = dispatch_protocol(build_parser().parse_args(["protocol-evolve", "--config", str(config), "--input-manifest", str(inputs), "--output-dir", str(tmp_path / "run")]))
    assert result["status"] == "protocol_evolution_complete"
    assert result["accepted"] == 5
    assert result["rejected"] == 1
    assert result["public_test_accessed"] is False
    handoff = json.loads((tmp_path / "run" / "frozen_protocol_handoff.json").read_text())
    release = json.loads((tmp_path / "run" / "releases" / f"{handoff['release_sha256']}.json").read_text())
    assert handoff["bundle_sha256"] == release["frozen_bundle_sha256"]
    assert handoff["acceptance_evidence_sha256"] == release["evidence_sha256"]
    manifest = json.loads(inputs.read_text())
    assert fingerprint_payload(handoff) == result["frozen_handoff_sha256"]
    assert {record[0] for record in manifest["host_input_files"]} == {"tasks", "catalog", "runtime", "closure"}
    closure = json.loads((inputs.parent / "host_closure.json").read_text())
    bundle = next(item for item in closure["archive_bundles"] if fingerprint_payload(item) == handoff["bundle_sha256"])
    assert closure["bundle_acceptance_evidence"][bundle["acceptance_evidence_sha256"]]["accepted"] is True
    assert handoff["protocol_sha256"] == release["protocol_sha256"]
    assert handoff["bundle_sha256"] == release["frozen_bundle_sha256"]
    assert {item["kind"] for item in manifest["replacement_templates"]} == {"backbone", "loader", "verifier_strategy", "diagnostic_metric", "schema_migration"}


def test_cli_rejects_stale_or_internally_mutated_host_closure(tmp_path):
    """The CLI verifies bytes first, then rejects a changed authority closure by identity."""
    inputs = make_smoke_inputs(tmp_path / "inputs")
    config = tmp_path / "smoke.json"
    config.write_bytes(b'{"hard_limit_seconds":120,"max_proposals":6,"profile":"smoke","schema_version":1,"seed":17}\n')
    closure_path = inputs.parent / "host_closure.json"
    closure = json.loads(closure_path.read_text())
    key = next(iter(closure["bundle_acceptance_evidence"]))
    closure["bundle_acceptance_evidence"][key] = {"schema_version": 1, "accepted": False}
    closure_path.write_bytes(canonical_v2_bytes(closure))
    args = build_parser().parse_args(["protocol-evolve", "--config", str(config), "--input-manifest", str(inputs), "--output-dir", str(tmp_path / "run")])
    with pytest.raises(ValueError, match="bytes do not match"):
        dispatch_protocol(args)
    manifest = json.loads(inputs.read_text())
    for record in manifest["host_input_files"]:
        if record[1] == "host_closure.json":
            record[2] = hashlib.sha256(closure_path.read_bytes()).hexdigest()
    inputs.write_bytes(canonical_v2_bytes(manifest))
    with pytest.raises(ValueError, match="host closure"):
        dispatch_protocol(args)
