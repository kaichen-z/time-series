from __future__ import annotations

from evolving_loop.adjustment.controller import (
    CASCADE_CONTROLLER, IDENTITY_CONTROLLER, SEMANTIC_CONTROLLER, POOLED_CONTROLLER,
    run_controller,
)
from evolving_loop.adjustment.skill_memory import (
    SkillMemory, controller_from_payload, controller_to_payload,
)

_TS = ("2024-06-17T00:00:00", "2024-06-18T00:00:00")
_HV = tuple(10.0 if __import__("datetime").date(2024, 6, d).weekday() < 5 else 6.0 for d in range(3, 17))
_HTS = tuple(f"2024-06-{d:02d}T00:00:00" for d in range(3, 17))
_CANDS = {"toto_2_0": (100.0, 100.0)}


def test_controller_payload_roundtrips_exactly():
    for c in (IDENTITY_CONTROLLER, CASCADE_CONTROLLER, SEMANTIC_CONTROLLER, POOLED_CONTROLLER):
        back = controller_from_payload(controller_to_payload(c))
        assert back.steps == c.steps                # exact structural round-trip
        # and behaves identically when run
        a, _ = run_controller(c, _CANDS, (), _HV, _HTS, _TS, semantic_ref="weekend")
        b, _ = run_controller(back, _CANDS, (), _HV, _HTS, _TS, semantic_ref="weekend")
        assert a == b


def test_memory_persists_and_reloads(tmp_path):
    p = tmp_path / "skills.json"
    m = SkillMemory(p)
    m.add(SEMANTIC_CONTROLLER, 0.05)
    m.add(CASCADE_CONTROLLER, 0.02)
    m.save()
    m2 = SkillMemory(p)
    assert len(m2.entries) == 2
    assert m2.entries[0]["score"] == 0.05           # ranked best-first
    assert m2.seeds()[0].steps == SEMANTIC_CONTROLLER.steps


def test_memory_dedupes_and_keeps_best_score(tmp_path):
    m = SkillMemory(tmp_path / "s.json")
    m.add(SEMANTIC_CONTROLLER, 0.03)
    m.add(SEMANTIC_CONTROLLER, 0.07)                 # same structure, higher score
    assert len(m.entries) == 1 and m.entries[0]["score"] == 0.07


def test_memory_capacity_bounded(tmp_path):
    m = SkillMemory(tmp_path / "s.json", capacity=2)
    m.add(IDENTITY_CONTROLLER, 0.01)
    m.add(SEMANTIC_CONTROLLER, 0.05)
    m.add(CASCADE_CONTROLLER, 0.03)
    assert len(m.entries) == 2                       # kept the top 2
    assert {round(e["score"], 2) for e in m.entries} == {0.05, 0.03}
