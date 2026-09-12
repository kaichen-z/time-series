from __future__ import annotations

import dataclasses
import json

import pytest

from evolving_loop.v2.contracts import canonical_v2_bytes

from evolving_loop.v2.source.authority import SourceAuthorityV2
from tests.build_evolution_v2_source_fixture import build_source_case


def test_canary_failure_restores_exact_pointer_bytes(tmp_path):
    case = build_source_case(tmp_path / "inputs")
    host = SourceAuthorityV2(tmp_path / "authority", case.seed_source)
    before = (tmp_path / "authority" / "active_source.json").read_bytes()
    evidence = case.evaluator.validate(case.seed_source, case.improving_source)
    host.begin_canary(case.improving_source, evidence)
    assert (tmp_path / "authority" / "active_source.json").read_bytes() == before
    failed = dataclasses.replace(evidence, passed=False, reason="canary_regression")
    assert host.finish_canary(failed) == case.seed_source.fingerprint()
    assert (tmp_path / "authority" / "active_source.json").read_bytes() == before


def test_successful_canary_updates_active_pointer(tmp_path):
    case = build_source_case(tmp_path / "inputs")
    host = SourceAuthorityV2(tmp_path / "authority", case.seed_source)
    evidence = case.evaluator.validate(case.seed_source, case.improving_source)
    host.begin_canary(case.improving_source, evidence)
    assert host.finish_canary(evidence) == case.improving_source.fingerprint()
    assert host.active_source().fingerprint() == case.improving_source.fingerprint()


def test_tampered_sealed_validation_cannot_activate(tmp_path):
    case = build_source_case(tmp_path / "inputs")
    host = SourceAuthorityV2(tmp_path / "authority", case.seed_source)
    evidence = case.evaluator.validate(case.seed_source, case.improving_source)
    host.begin_canary(case.improving_source, evidence)
    sealed = next((tmp_path / "authority" / "sealed").iterdir())
    payload = json.loads(sealed.read_text())
    payload["reason"] = "altered"
    sealed.write_bytes(canonical_v2_bytes(payload))
    with pytest.raises(ValueError, match="evidence"):
        host.finish_canary(evidence)
