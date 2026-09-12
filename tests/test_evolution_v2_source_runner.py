from __future__ import annotations

from evolving_loop.v2.source import SourceConfigV2, run_source_evolution
from tests.build_evolution_v2_source_fixture import build_source_case


def test_closed_validation_resume_matches_full_result(tmp_path):
    case = build_source_case(tmp_path / "inputs")
    config = SourceConfigV2.smoke(seed=17)
    full = run_source_evolution(tmp_path / "full", config, case)
    run_source_evolution(tmp_path / "resumed", config, case, stop_after="validation")
    resumed = run_source_evolution(tmp_path / "resumed", config, case, resume=True)
    assert resumed.canonical_bytes() == full.canonical_bytes()
