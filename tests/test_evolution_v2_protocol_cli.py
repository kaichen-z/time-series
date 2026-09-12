from __future__ import annotations

import json

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
    manifest = json.loads(inputs.read_text())
    assert {item["kind"] for item in manifest["replacement_templates"]} == {"backbone", "loader", "verifier_strategy", "diagnostic_metric", "schema_migration"}
