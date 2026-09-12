from __future__ import annotations

import dataclasses

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
