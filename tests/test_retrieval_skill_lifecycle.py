from __future__ import annotations

import json
import copy
import pickle
from dataclasses import asdict, replace

import pytest

from evolving_loop.retrieval_agent.skill_library import (
    RetrievalApplicability,
    RetrievalSkill,
    RetrievalSkillError,
    RetrievalSkillLibrary,
    RetrievalSkillOperation,
)
from evolving_loop.retrieval_agent.agent import RetrievalAgent, RetrievalResult
from common.llm import FakeLLMClient
from test_evolving_harness import _task


def test_typed_skill_lifecycle_is_exported_from_retrieval_package() -> None:
    from evolving_loop.retrieval_agent import RetrievalSkillOperation as ExportedOperation

    assert ExportedOperation is RetrievalSkillOperation


def seed_skill(*, skill_id: str = "explicit_window", **overrides: object) -> RetrievalSkill:
    values: dict[str, object] = {
        "skill_id": skill_id,
        "version": 1,
        "parent_version": None,
        "stage": "both",
        "status": "accepted",
        "name": "explicit_window_search",
        "description": "Find exact event boundaries.",
        "applicability": RetrievalApplicability(
            assumption_kinds=("future_event", "regime_change"),
            gap_types=("missing_window", "missing_magnitude"),
            temporal_relations=("overlaps_future", "unknown"),
        ),
        "query_steps": ("Find the named event.",),
        "required_chain_fields": ("entity", "target", "start_timestamp", "end_timestamp"),
        "counterevidence_rule": "Search for cancellation.",
        "failure_conditions": ("The event ends before the forecast window.",),
        "validated_task_ids": ("train_1", "train_2", "train_3"),
        "validated_entities": ("north", "south"),
        "validation_smae_gain": 0.1,
        "validation_srmse_gain": 0.2,
    }
    values.update(overrides)
    return RetrievalSkill(**values)


def repaired_skill() -> RetrievalSkill:
    return seed_skill(
        status="candidate",
        description="Find inclusive event boundaries and recovery dates.",
        validation_smae_gain=None,
        validation_srmse_gain=None,
    )


def test_load_migrates_legacy_rows_and_saves_typed_schema(tmp_path) -> None:
    path = tmp_path / "skills.json"
    path.write_text(
        json.dumps(
            [
                {
                    "skill_id": "legacy_window",
                    "name": "legacy_window",
                    "description": "Find a window.",
                    "applicability": "scheduled event",
                    "query_strategy": "Find the event window.",
                    "verification_rule": "Require a quote.",
                    "created_from_task": "train_1",
                    "validation_smae": 0.1,
                    "validation_srmse": 0.2,
                }
            ]
        )
    )

    library = RetrievalSkillLibrary.load(path)

    skill = library.get_by_id("legacy_window")
    assert skill is not None
    assert (skill.version, skill.parent_version, skill.stage, skill.status) == (
        1,
        None,
        "both",
        "accepted",
    )
    assert skill.query_steps == ("Find the event window.",)
    assert skill.counterevidence_rule == "Require a quote."

    library.save()
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == 1
    assert set(saved["skills"][0]) >= {"version", "stage", "status", "query_steps"}
    assert "query_strategy" not in saved["skills"][0]


def test_typed_schema_accepts_omitted_default_audit_fields() -> None:
    raw = seed_skill().to_payload()
    for field in (
        "validated_task_ids",
        "validated_entities",
        "validation_smae_gain",
        "validation_srmse_gain",
        "merged_from_skill_ids",
        "quarantine_reason",
    ):
        raw.pop(field)
    raw["status"] = "candidate"

    loaded = RetrievalSkill.from_payload(raw)

    assert loaded.validated_task_ids == ()
    assert loaded.validation_smae_gain is None


def test_repair_preserves_identity_and_lineage(tmp_path) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", [seed_skill()])

    library.apply_operations((RetrievalSkillOperation.repair("explicit_window", repaired_skill()),))

    repaired = library.get_by_id("explicit_window")
    assert repaired is not None
    assert repaired.skill_id == "explicit_window"
    assert (repaired.version, repaired.parent_version) == (2, 1)
    assert len(library.history("explicit_window")) == 2


def test_specialization_must_narrow_applicability(tmp_path) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", [seed_skill()])
    specialized = seed_skill(
        status="candidate",
        validation_smae_gain=None,
        validation_srmse_gain=None,
    )

    with pytest.raises(RetrievalSkillError, match="narrow"):
        library.apply_operations(
            (RetrievalSkillOperation.specialize("explicit_window", specialized),)
        )

    narrowed = seed_skill(
        status="candidate",
        validation_smae_gain=None,
        validation_srmse_gain=None,
        applicability=RetrievalApplicability(
            assumption_kinds=("future_event",),
            gap_types=("missing_window",),
            temporal_relations=("overlaps_future",),
        ),
        required_chain_fields=(
            "entity",
            "target",
            "start_timestamp",
            "end_timestamp",
            "mechanism",
        ),
    )
    library.apply_operations(
        (RetrievalSkillOperation.specialize("explicit_window", narrowed),)
    )
    assert library.get_by_id("explicit_window").status == "candidate"


def _candidate_seed(*, skill_id: str = "explicit_window") -> RetrievalSkill:
    return seed_skill(
        skill_id=skill_id,
        status="candidate",
        validated_task_ids=(),
        validated_entities=(),
        validation_smae_gain=None,
        validation_srmse_gain=None,
    )


@pytest.mark.parametrize("operation_kind", ("add", "repair", "specialize", "merge"))
def test_public_skill_mutations_cannot_create_active_status(tmp_path, operation_kind) -> None:
    first = _candidate_seed()
    second = _candidate_seed(skill_id="explicit_recovery")
    library = RetrievalSkillLibrary(tmp_path / "skills.json", (first, second))
    forged = seed_skill()
    if operation_kind == "add":
        action = lambda: library.add(forged)
    elif operation_kind == "repair":
        action = lambda: library.apply_operations(
            (RetrievalSkillOperation.repair(first.skill_id, forged),)
        )
    elif operation_kind == "specialize":
        forged = replace(
            forged,
            status="specialized",
            applicability=RetrievalApplicability(
                assumption_kinds=("future_event",),
                gap_types=("missing_window",),
                temporal_relations=("overlaps_future",),
            ),
        )
        action = lambda: library.apply_operations(
            (RetrievalSkillOperation.specialize(first.skill_id, forged),)
        )
    else:
        forged = seed_skill(skill_id="forged_merge")
        action = lambda: library.apply_operations(
            (
                RetrievalSkillOperation.merge(
                    (first.skill_id, second.skill_id), forged
                ),
            )
        )

    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
        action()
    assert all(not skill.is_active for skill in library.all())


def test_copied_or_serialized_active_records_cannot_bypass_public_repair(tmp_path) -> None:
    candidate = _candidate_seed()
    active = seed_skill()
    forged_records = (
        replace(candidate, status="accepted", validation_smae_gain=0.9, validation_srmse_gain=0.9),
        copy.copy(active),
        pickle.loads(pickle.dumps(active)),
        RetrievalSkill.from_payload(asdict(active)),
    )
    forged_operation = replace(
        RetrievalSkillOperation.repair(candidate.skill_id, candidate),
        skill=active,
    )
    operations = (
        *(RetrievalSkillOperation.repair(candidate.skill_id, row) for row in forged_records),
        pickle.loads(pickle.dumps(forged_operation)),
    )
    for index, operation in enumerate(operations):
        library = RetrievalSkillLibrary(
            tmp_path / f"skills_{index}.json", (candidate,)
        )

        with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
            library.apply_operations((operation,))
        assert library.get_by_id(candidate.skill_id).status == "candidate"


def test_rejected_active_forgery_cannot_survive_save_and_reload(tmp_path) -> None:
    path = tmp_path / "skills.json"
    candidate = _candidate_seed()
    library = RetrievalSkillLibrary(path, (candidate,))

    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
        library.add(seed_skill())
    library.save()
    reloaded = RetrievalSkillLibrary.load(path)

    assert reloaded.get_by_id(candidate.skill_id).status == "candidate"
    assert len(reloaded.history(candidate.skill_id)) == 1


def test_loaded_legacy_active_skill_is_usable_but_not_publicly_revalidated(tmp_path) -> None:
    path = tmp_path / "skills.json"
    path.write_text(json.dumps([seed_skill().to_payload()]))
    library = RetrievalSkillLibrary.load(path)

    assert library.for_stage(
        "round1",
        assumption_kinds=("future_event",),
        gap_types=("missing_window",),
        temporal_relations=("overlaps_future",),
    )
    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
        library.apply_operations(
            (RetrievalSkillOperation.repair("explicit_window", seed_skill()),)
        )

    library.apply_operations(
        (RetrievalSkillOperation.quarantine("explicit_window", "retire legacy record"),)
    )
    assert library.get_by_id("explicit_window").status == "quarantined"


def test_merge_retains_predecessors_and_records_ancestry(tmp_path) -> None:
    library = RetrievalSkillLibrary(
        tmp_path / "skills.json", [seed_skill(), seed_skill(skill_id="explicit_recovery")]
    )
    successor = seed_skill(skill_id="window_and_recovery", status="candidate")

    library.apply_operations(
        (
            RetrievalSkillOperation.merge(
                ("explicit_window", "explicit_recovery"), successor
            ),
        )
    )

    merged = library.get_by_id("window_and_recovery")
    assert merged is not None
    assert merged.merged_from_skill_ids == ("explicit_recovery", "explicit_window")
    assert library.get_by_id("explicit_window").status == "quarantined"
    assert library.get_by_id("explicit_recovery").status == "quarantined"
    assert len(library.history("explicit_window")) == 2


def test_quarantine_then_reactivation_versions_the_same_skill(tmp_path) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", [seed_skill()])
    library.apply_operations(
        (RetrievalSkillOperation.quarantine("explicit_window", "unsafe quote rule"),)
    )
    assert library.get_by_id("explicit_window").status == "quarantined"

    library.apply_operations(
        (RetrievalSkillOperation.repair("explicit_window", seed_skill(status="candidate")),)
    )

    reactivated = library.get_by_id("explicit_window")
    assert reactivated is not None
    assert (reactivated.version, reactivated.parent_version, reactivated.status) == (3, 2, "candidate")


def test_skill_operations_are_atomic_and_non_destructive(tmp_path) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", [seed_skill()])
    operations = (
        RetrievalSkillOperation.repair("explicit_window", repaired_skill()),
        RetrievalSkillOperation.quarantine("unknown_skill", "not registered"),
    )

    with pytest.raises(RetrievalSkillError, match="unknown_skill"):
        library.apply_operations(operations)

    assert library.get_by_id("explicit_window").version == 1
    assert not library.path.exists()


def test_prompt_projection_is_stage_and_applicability_filtered_and_clone_is_read_only(tmp_path) -> None:
    path = tmp_path / "skills.json"
    library = RetrievalSkillLibrary(
        path,
        [
            seed_skill(stage="round1"),
            seed_skill(
                skill_id="round2_only",
                stage="round2",
                applicability=RetrievalApplicability(gap_types=("missing_window",)),
            ),
            seed_skill(skill_id="candidate", status="candidate"),
        ],
    )

    assert [
        skill.skill_id
        for skill in library.for_stage(
            "round1",
            assumption_kinds=("future_event",),
            gap_types=("missing_window",),
            temporal_relations=("overlaps_future",),
        )
    ] == ["explicit_window"]
    assert [skill.skill_id for skill in library.for_stage("round2", gap_types=("missing_window",))] == ["round2_only"]
    assert "candidate" not in library.list_for_prompt("round1")

    clone = library.clone(persist=False)
    clone.apply_operations((RetrievalSkillOperation.quarantine("explicit_window", "test"),))
    assert not path.exists()
    assert library.get_by_id("explicit_window").status == "accepted"


def test_constrained_applicability_fails_closed_without_runtime_context(tmp_path) -> None:
    applicability = RetrievalApplicability(
        assumption_kinds=("future_event",),
        gap_types=("missing_window",),
        temporal_relations=("overlaps_future",),
    )
    library = RetrievalSkillLibrary(
        tmp_path / "skills.json", [seed_skill(applicability=applicability)]
    )

    assert applicability.matches() is False
    assert applicability.matches(
        assumption_kinds=("future_event",),
        gap_types=("missing_window",),
        temporal_relations=("overlaps_future",),
    )
    assert not applicability.matches(
        assumption_kinds=("future_event",),
        gap_types=("missing_magnitude",),
        temporal_relations=("overlaps_future",),
    )
    assert library.for_stage("round1") == ()


def test_legacy_agent_projects_round_two_skills_with_prior_gap_context(tmp_path) -> None:
    library = RetrievalSkillLibrary(
        tmp_path / "skills.json",
        [
            seed_skill(
                stage="round2",
                applicability=RetrievalApplicability(gap_types=("missing_window",)),
            )
        ],
    )
    agent = RetrievalAgent(
        FakeLLMClient(
            [
                json.dumps({"evidence": [], "impacts": []}),
                json.dumps({"evidence": [], "impacts": []}),
            ]
        ),
        library,
    )
    prior = RetrievalResult("", (), (), (), False, ("missing_window",))

    agent.run(_task(), ())
    agent.run(_task(), (), prior=prior, round_index=1)

    first_payload = agent.llm.calls[0]["messages"][0]["content"]
    second_payload = agent.llm.calls[1]["messages"][0]["content"]
    assert "explicit_window_search" not in first_payload
    assert "explicit_window_search" in second_payload
