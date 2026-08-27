from __future__ import annotations

import copy
import hashlib
import json
import pickle
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import evolving_loop.retrieval_agent.skill_library as skill_library_module
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalApplicability,
    RetrievalSkill,
    RetrievalSkillError,
    RetrievalSkillLibrary,
    RetrievalSkillOperation,
    _migrate_legacy_for_operator,
)
from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    RetrievalPolicyError,
    _write_accepted_retrieval_release,
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
        "status": "candidate",
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
        "validated_task_ids": (),
        "validated_entities": (),
        "validation_smae_gain": None,
        "validation_srmse_gain": None,
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

    library = _migrate_legacy_for_operator(path)

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
    reloaded = RetrievalSkillLibrary.load_verified_checkpoint(path)
    assert reloaded.get_by_id("legacy_window").status == "accepted"


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


def _active_payload(
    *,
    skill_id: str = "explicit_window",
    status: str = "accepted",
    version: int = 1,
    parent_version: int | None = None,
    stage: str = "round1",
    applicability: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    return {
        "skill_id": skill_id,
        "version": version,
        "parent_version": parent_version,
        "stage": stage,
        "status": status,
        "name": "explicit_window_search" if skill_id == "explicit_window" else skill_id,
        "description": "Find exact event boundaries.",
        "applicability": applicability
        or {
            "assumption_kinds": [],
            "gap_types": [],
            "temporal_relations": [],
        },
        "query_steps": ["Find the named event."],
        "required_chain_fields": ["entity", "target"],
        "counterevidence_rule": "Search for cancellation.",
        "failure_conditions": ["The event is outside the forecast window."],
        "validated_task_ids": ["train_1", "train_2", "train_3"],
        "validated_entities": ["north", "south"],
        "validation_smae_gain": 0.1,
        "validation_srmse_gain": 0.2,
        "merged_from_skill_ids": [],
        "quarantine_reason": None,
    }


def _accepted_audit() -> dict[str, object]:
    return {
        "state": "accepted",
        "train_dev_split_sha256": "1" * 64,
        "verifier_sha256": "2" * 64,
        "evaluator_sha256": "3" * 64,
        "metric_sha256": "4" * 64,
        "metric_cap": 5.0,
        "train_summary": {"task_count": 80},
        "dev_summary": {"task_count": 20},
        "acceptance_reason": "all immutable Train and Dev gates passed",
    }


def _verified_active_library(
    root: Path,
    *,
    status: str = "accepted",
    skill_id: str = "explicit_window",
    stage: str = "round1",
    applicability: dict[str, list[str]] | None = None,
) -> RetrievalSkillLibrary:
    genome = replace(
        RetrievalGenome.seed(),
        version="v001",
        parent="v000",
        active_skill_ids=(skill_id,),
    )
    release = _write_accepted_retrieval_release(
        root / "releases",
        genome,
        skills=(
            _active_payload(
                status=status,
                skill_id=skill_id,
                stage=stage,
                applicability=applicability,
            ),
        ),
        audit=_accepted_audit(),
    )
    return RetrievalSkillLibrary.from_release(release.path)


def test_public_payload_and_record_constructors_reject_active_status() -> None:
    payload = _active_payload()

    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
        RetrievalSkill.from_payload(payload)
    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
        RetrievalSkill(**payload)
    quarantined = {
        **payload,
        "status": "quarantined",
        "quarantine_reason": "caller-created inactive record",
    }
    with pytest.raises(RetrievalSkillError, match="candidate|quarantine|transition"):
        RetrievalSkill.from_payload(quarantined)


def test_public_library_constructor_cannot_persist_a_loaded_active_record(tmp_path) -> None:
    active = _verified_active_library(tmp_path).get_by_id("explicit_window")
    assert active is not None and active.status == "accepted"
    destination = tmp_path / "forged.json"

    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
        RetrievalSkillLibrary(destination, (active,))

    assert not destination.exists()


@pytest.mark.parametrize("serialization", ("copy", "pickle", "asdict"))
def test_copied_or_serialized_loaded_active_records_have_no_constructor_authority(
    tmp_path, serialization
) -> None:
    active = _verified_active_library(tmp_path).get_by_id("explicit_window")
    assert active is not None
    if serialization == "copy":
        forged = copy.copy(active)
    elif serialization == "pickle":
        forged = pickle.loads(pickle.dumps(active))
    else:
        with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
            RetrievalSkill.from_payload(asdict(active))
        return

    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
        RetrievalSkillLibrary(tmp_path / f"{serialization}.json", (forged,))


def test_current_schema_file_cannot_claim_active_status(tmp_path) -> None:
    path = tmp_path / "skills.json"
    candidate = _candidate_seed().to_payload()
    active = _active_payload(version=2, parent_version=1)
    skills = [candidate, active]

    def digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    path.write_text(json.dumps({
        "schema_version": 1,
        "skills": skills,
        "provenance": {
            "skills_sha256": digest(skills),
            "active_records": [
                {"sha256": digest(active), "origin": "evaluator_promotion"}
            ],
        },
    }), encoding="utf-8")

    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted|provenance"):
        RetrievalSkillLibrary.load(path)
    with pytest.raises(RetrievalSkillError, match="checkpoint|provenance|witness|schema"):
        RetrievalSkillLibrary.load_verified_checkpoint(path)


def test_public_checkpoint_load_rejects_a_fully_recomputed_caller_witness(
    tmp_path,
) -> None:
    path = tmp_path / "skills.json"
    candidate = _candidate_seed().to_payload()
    active = _active_payload(version=2, parent_version=1)
    skills = [candidate, active]

    def digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    encoded = (json.dumps(
        {"schema_version": 1, "skills": skills},
        indent=2,
        ensure_ascii=False,
    ) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    checkpoint_sha256 = hashlib.sha256(encoded).hexdigest()
    witness = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path_sha256": digest(str(path.resolve())),
        "skills_sha256": digest(skills),
        "active_records": [
            {"sha256": digest(active), "origin": "evaluator_promotion"}
        ],
    }
    witness_path = (
        path.parent
        / f".{path.name}.provenance"
        / f"{checkpoint_sha256}.json"
    )
    witness_path.parent.mkdir()
    witness_path.write_text(
        json.dumps(witness, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RetrievalSkillError, match="authority|operator|evaluator"):
        RetrievalSkillLibrary.load_verified_checkpoint(path)


def test_replacing_candidate_library_path_with_active_payload_fails_closed(tmp_path) -> None:
    path = tmp_path / "skills.json"
    candidate = _candidate_seed()
    library = RetrievalSkillLibrary(path, (candidate,))
    library.save()
    path.write_text(
        json.dumps({"schema_version": 1, "skills": [_active_payload()]}),
        encoding="utf-8",
    )

    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted|provenance"):
        RetrievalSkillLibrary.load(path)

    assert library.get_by_id(candidate.skill_id) == candidate


def test_verified_release_hydrates_active_and_specialized_histories(tmp_path) -> None:
    accepted = _active_payload()
    specialized = _active_payload(
        status="specialized", version=2, parent_version=1
    )
    genome = replace(
        RetrievalGenome.seed(),
        version="v001",
        parent="v000",
        active_skill_ids=("explicit_window",),
    )
    release = _write_accepted_retrieval_release(
        tmp_path / "releases",
        genome,
        skills=(accepted, specialized),
        audit=_accepted_audit(),
    )

    library = RetrievalSkillLibrary.from_release(release.path)

    assert [skill.status for skill in library.history("explicit_window")] == [
        "accepted",
        "specialized",
    ]
    assert [skill.skill_id for skill in library.for_stage("round1")] == [
        "explicit_window"
    ]


def test_verified_release_factory_rechecks_file_hashes_at_hydration(tmp_path) -> None:
    library = _verified_active_library(tmp_path)
    release_path = next((tmp_path / "releases").iterdir())
    skills_path = release_path / "skills.json"
    skills_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(RetrievalPolicyError, match="skills hash"):
        RetrievalSkillLibrary.from_release(release_path)

    assert library.get_by_id("explicit_window").status == "accepted"


def test_legacy_migration_is_explicit_and_requires_both_historical_metrics(
    tmp_path,
) -> None:
    public_legacy_record = RetrievalSkill(
        "public_legacy",
        name="public_legacy",
        description="Caller-created legacy-shaped row.",
        applicability="scheduled event",
        query_strategy="Find the event window.",
        verification_rule="Require a quote.",
        created_from_task="train_0",
        validation_smae=0.1,
        validation_srmse=0.2,
    )
    assert public_legacy_record.status == "candidate"
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            [
                {
                    "skill_id": "validated_legacy",
                    "name": "validated_legacy",
                    "description": "Historical validated strategy.",
                    "applicability": "scheduled event",
                    "query_strategy": "Find the event window.",
                    "verification_rule": "Require a quote.",
                    "created_from_task": "train_1",
                    "validation_smae": 0.1,
                    "validation_srmse": 0.2,
                },
                {
                    "skill_id": "partial_legacy",
                    "name": "partial_legacy",
                    "description": "Only one historical metric.",
                    "applicability": "scheduled event",
                    "query_strategy": "Find the event window.",
                    "verification_rule": "Require a quote.",
                    "created_from_task": "train_2",
                    "validation_smae": 0.1,
                },
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RetrievalSkillError, match="legacy|migration"):
        RetrievalSkillLibrary.load(path)

    library = _migrate_legacy_for_operator(path)

    assert library.get_by_id("validated_legacy").status == "accepted"
    assert library.get_by_id("partial_legacy").status == "candidate"
    with pytest.raises(RetrievalSkillError, match="legacy|typed|schema"):
        typed_path = tmp_path / "typed-list.json"
        typed_path.write_text(json.dumps([_active_payload()]), encoding="utf-8")
        _migrate_legacy_for_operator(typed_path)


def test_public_legacy_migration_does_not_activate_caller_invented_metrics(
    tmp_path,
) -> None:
    path = tmp_path / "invented-legacy.json"
    path.write_text(
        json.dumps(
            [
                {
                    "skill_id": "invented_legacy",
                    "name": "invented_legacy",
                    "description": "Caller claims this strategy was validated.",
                    "applicability": "scheduled event",
                    "query_strategy": "Trust the caller's event window.",
                    "verification_rule": "Trust the caller's quote.",
                    "created_from_task": "invented_train_task",
                    "validation_smae": 0.9,
                    "validation_srmse": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )

    library = RetrievalSkillLibrary.migrate_legacy(path)

    assert library.get_by_id("invented_legacy").status == "candidate"
    assert library.for_stage("round1") == ()
    library.save()
    reloaded = RetrievalSkillLibrary.load(path)
    assert reloaded.get_by_id("invented_legacy").status == "candidate"
    assert reloaded.for_stage("round1") == ()


def test_trusted_operator_migration_preserves_historical_accepted_records(
    tmp_path,
) -> None:
    path = tmp_path / "operator-legacy.json"
    path.write_text(
        json.dumps(
            [
                {
                    "skill_id": "historical_skill",
                    "name": "historical_skill",
                    "description": "A genuinely operator-approved historical row.",
                    "applicability": "scheduled event",
                    "query_strategy": "Find the event window.",
                    "verification_rule": "Require an exact quote.",
                    "created_from_task": "train_1",
                    "validation_smae": 0.1,
                    "validation_srmse": 0.2,
                }
            ]
        ),
        encoding="utf-8",
    )

    library = _migrate_legacy_for_operator(path)
    reloaded = RetrievalSkillLibrary.load_verified_checkpoint(path)

    assert library.get_by_id("historical_skill").status == "accepted"
    assert reloaded.get_by_id("historical_skill").status == "accepted"
    assert "historical_skill" in reloaded.list_for_prompt("round1")


def test_quarantine_revokes_restored_accepted_checkpoint_authority(tmp_path) -> None:
    path = tmp_path / "operator-legacy.json"
    path.write_text(
        json.dumps(
            [
                {
                    "skill_id": "historical_skill",
                    "name": "historical_skill",
                    "description": "A genuinely operator-approved historical row.",
                    "applicability": "scheduled event",
                    "query_strategy": "Find the event window.",
                    "verification_rule": "Require an exact quote.",
                    "created_from_task": "train_1",
                    "validation_smae": 0.1,
                    "validation_srmse": 0.2,
                }
            ]
        ),
        encoding="utf-8",
    )
    library = _migrate_legacy_for_operator(path)
    accepted_checkpoint = path.read_bytes()

    library.apply_operations(
        (RetrievalSkillOperation.quarantine("historical_skill", "unsafe"),)
    )
    assert library.get_by_id("historical_skill").status == "quarantined"
    path.write_bytes(accepted_checkpoint)

    with pytest.raises(RetrievalSkillError, match="current|authority|epoch"):
        RetrievalSkillLibrary.load_verified_checkpoint(path)


def test_verified_checkpoint_clone_is_bound_to_current_path_epoch(tmp_path) -> None:
    path = tmp_path / "operator-legacy.json"
    path.write_text(
        json.dumps(
            [
                {
                    "skill_id": "historical_skill",
                    "name": "historical_skill",
                    "description": "A genuinely operator-approved historical row.",
                    "applicability": "scheduled event",
                    "query_strategy": "Find the event window.",
                    "verification_rule": "Require an exact quote.",
                    "created_from_task": "train_1",
                    "validation_smae": 0.1,
                    "validation_srmse": 0.2,
                }
            ]
        ),
        encoding="utf-8",
    )
    library = _migrate_legacy_for_operator(path)
    clone = library.clone(persist=False)

    library.apply_operations(
        (RetrievalSkillOperation.quarantine("historical_skill", "unsafe"),)
    )

    with pytest.raises(RetrievalSkillError, match="current|authority|epoch"):
        clone.for_stage("round1")


def test_clone_preserves_verified_active_history_without_generalizing_constructor(
    tmp_path,
) -> None:
    library = _verified_active_library(tmp_path)

    clone = library.clone(persist=False)

    assert clone.all() == library.all()
    assert clone.for_stage("round1") == library.for_stage("round1")
    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
        RetrievalSkillLibrary(tmp_path / "copy.json", clone.all())


def test_python_copy_loses_authority_while_clone_and_replay_preserve_it(
    tmp_path,
) -> None:
    library = _verified_active_library(tmp_path)
    copied = copy.copy(library)

    with pytest.raises(RetrievalSkillError, match="authority|copied|operator"):
        copied.list_for_prompt("round1")

    clone = library.clone(persist=False)
    replay = library.replay_snapshot(library.all(), persist=False)
    assert "explicit_window_search" in clone.list_for_prompt("round1")
    assert "explicit_window_search" in replay.list_for_prompt("round1")


def test_verified_source_replays_only_the_exact_skill_history(tmp_path) -> None:
    library = _verified_active_library(tmp_path)

    replay = library.replay_snapshot(library.all())

    assert replay.all() == library.all()
    assert replay.for_stage("round1") == library.for_stage("round1")
    changed = copy.copy(library.all()[0])
    object.__setattr__(changed, "description", "caller changed the active row")
    with pytest.raises(RetrievalSkillError, match="verified|absent|history"):
        library.replay_snapshot((changed,))


@pytest.mark.parametrize("operation_kind", ("add", "repair", "specialize", "merge"))
def test_public_skill_mutations_cannot_create_active_status(tmp_path, operation_kind) -> None:
    first = _candidate_seed()
    second = _candidate_seed(skill_id="explicit_recovery")
    library = RetrievalSkillLibrary(tmp_path / "skills.json", (first, second))
    forged_id = "forged_merge" if operation_kind == "merge" else "explicit_window"
    forged_status = "specialized" if operation_kind == "specialize" else "accepted"
    forged = _verified_active_library(
        tmp_path / "trusted",
        status=forged_status,
        skill_id=forged_id,
    ).get_by_id(forged_id)
    assert forged is not None
    if operation_kind == "add":
        action = lambda: library.add(forged)
    elif operation_kind == "repair":
        action = lambda: library.apply_operations(
            (RetrievalSkillOperation.repair(first.skill_id, forged),)
        )
    elif operation_kind == "specialize":
        action = lambda: library.apply_operations(
            (RetrievalSkillOperation.specialize(first.skill_id, forged),)
        )
    else:
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
    active = _verified_active_library(tmp_path / "trusted").get_by_id(
        "explicit_window"
    )
    assert active is not None
    forged_records = (
        copy.copy(active),
        pickle.loads(pickle.dumps(active)),
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

    active = _verified_active_library(tmp_path / "trusted").get_by_id(
        "explicit_window"
    )
    assert active is not None
    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
        library.add(active)
    library.save()
    reloaded = RetrievalSkillLibrary.load(path)

    assert reloaded.get_by_id(candidate.skill_id).status == "candidate"
    assert len(reloaded.history(candidate.skill_id)) == 1


def test_loaded_legacy_active_skill_is_usable_but_not_publicly_revalidated(tmp_path) -> None:
    path = tmp_path / "skills.json"
    path.write_text(
        json.dumps(
            [
                {
                    "skill_id": "explicit_window",
                    "name": "explicit_window_search",
                    "description": "Find exact event boundaries.",
                    "applicability": "scheduled event",
                    "query_strategy": "Find the named event.",
                    "verification_rule": "Search for cancellation.",
                    "created_from_task": "train_1",
                    "validation_smae": 0.1,
                    "validation_srmse": 0.2,
                }
            ]
        )
    )
    library = _migrate_legacy_for_operator(path)

    assert library.for_stage(
        "round1",
        assumption_kinds=("future_event",),
        gap_types=("missing_window",),
        temporal_relations=("overlaps_future",),
    )
    active = library.get_by_id("explicit_window")
    assert active is not None
    with pytest.raises(RetrievalSkillError, match="candidate|active|trusted"):
        library.apply_operations(
            (RetrievalSkillOperation.repair("explicit_window", active),)
        )

    library.apply_operations(
        (RetrievalSkillOperation.quarantine("explicit_window", "retire legacy record"),)
    )
    assert library.get_by_id("explicit_window").status == "quarantined"
    reloaded = RetrievalSkillLibrary.load_verified_checkpoint(path)
    assert reloaded.get_by_id("explicit_window").status == "quarantined"
    assert reloaded.for_stage("round1") == ()


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
    reloaded = RetrievalSkillLibrary.load(library.path)
    assert reloaded.get_by_id("explicit_window").status == "quarantined"

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


def test_skill_writer_ignores_attacker_fixed_temp_symlink(tmp_path) -> None:
    path = tmp_path / "skills.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("do-not-touch", encoding="utf-8")
    fixed_temporary = path.with_suffix(path.suffix + ".tmp")
    fixed_temporary.symlink_to(victim)
    library = RetrievalSkillLibrary(path, (seed_skill(),))

    library.save()

    assert victim.read_text(encoding="utf-8") == "do-not-touch"
    assert fixed_temporary.is_symlink()


def test_skill_checkpoint_read_rejects_parent_replacement_at_file_open(
    tmp_path, monkeypatch
) -> None:
    library_directory = tmp_path / "library"
    library_directory.mkdir()
    path = library_directory / "skills.json"
    RetrievalSkillLibrary(path, (seed_skill(),)).save()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / path.name).write_bytes(path.read_bytes())
    displaced = tmp_path / "displaced"
    real_open = skill_library_module.os.open
    swapped = False

    def swap_before_file_open(file, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        is_checkpoint_open = (
            dir_fd is None and Path(file) == path
        ) or (dir_fd is not None and file == path.name)
        if not swapped and is_checkpoint_open:
            swapped = True
            library_directory.rename(displaced)
            replacement.rename(library_directory)
        return real_open(file, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(skill_library_module.os, "open", swap_before_file_open)

    with pytest.raises(
        RetrievalSkillError,
        match="path|parent|directory|changed",
    ):
        RetrievalSkillLibrary.load(path)
    assert swapped


@pytest.mark.parametrize("operation", ("link", "replace"))
def test_skill_checkpoint_commit_rejects_parent_replacement_at_entry_operation(
    tmp_path, monkeypatch, operation
) -> None:
    library_directory = tmp_path / "library"
    library_directory.mkdir()
    path = library_directory / "skills.json"
    library = RetrievalSkillLibrary(path, (seed_skill(),))
    if operation == "replace":
        library.save()

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    if operation == "replace":
        (replacement / path.name).write_text("do-not-touch", encoding="utf-8")
    displaced = tmp_path / "displaced"
    real_operation = getattr(skill_library_module.os, operation)
    swapped = False

    def swap_before_entry_operation(source, destination, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            library_directory.rename(displaced)
            replacement.rename(library_directory)
            if kwargs.get("src_dir_fd") is None:
                shutil.copyfile(
                    displaced / Path(source).name,
                    library_directory / Path(source).name,
                )
        return real_operation(source, destination, *args, **kwargs)

    monkeypatch.setattr(
        skill_library_module.os,
        operation,
        swap_before_entry_operation,
    )

    with pytest.raises(
        RetrievalSkillError,
        match="path|parent|directory|changed",
    ):
        if operation == "link":
            library.save()
        else:
            library.apply_operations(
                (
                    RetrievalSkillOperation.repair(
                        "explicit_window", repaired_skill()
                    ),
                )
            )
    assert swapped
    if operation == "replace":
        assert path.read_text(encoding="utf-8") == "do-not-touch"
    else:
        assert not path.exists()


@pytest.mark.parametrize("symlink_kind", ("path", "parent"))
def test_skill_writer_rejects_symlink_paths_and_parents(tmp_path, symlink_kind) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    if symlink_kind == "path":
        victim = real_directory / "victim.json"
        victim.write_text("do-not-touch", encoding="utf-8")
        path = tmp_path / "skills.json"
        path.symlink_to(victim)
    else:
        linked_parent = tmp_path / "linked"
        linked_parent.symlink_to(real_directory, target_is_directory=True)
        path = linked_parent / "skills.json"
    library = RetrievalSkillLibrary(path, (seed_skill(),))

    with pytest.raises(RetrievalSkillError, match="symlink|unsafe|path"):
        library.save()
    if symlink_kind == "path":
        assert victim.read_text(encoding="utf-8") == "do-not-touch"


def test_prompt_projection_is_stage_and_applicability_filtered_and_clone_is_read_only(tmp_path) -> None:
    path = tmp_path / "skills.json"
    genome = replace(
        RetrievalGenome.seed(),
        version="v001",
        parent="v000",
        active_skill_ids=("explicit_window", "round2_only"),
    )
    release = _write_accepted_retrieval_release(
        tmp_path / "projection-release",
        genome,
        skills=(
            _active_payload(skill_id="explicit_window", stage="round1"),
            _active_payload(
                skill_id="round2_only",
                stage="round2",
                applicability={
                    "assumption_kinds": [],
                    "gap_types": ["missing_window"],
                    "temporal_relations": [],
                },
            ),
            seed_skill(skill_id="candidate").to_payload(),
        ),
        audit=_accepted_audit(),
    )
    library = RetrievalSkillLibrary.from_release(release.path)

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
    library = _verified_active_library(
        tmp_path,
        applicability={
            "assumption_kinds": ["future_event"],
            "gap_types": ["missing_window"],
            "temporal_relations": ["overlaps_future"],
        },
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
    library = _verified_active_library(
        tmp_path,
        stage="round2",
        applicability={
            "assumption_kinds": [],
            "gap_types": ["missing_window"],
            "temporal_relations": [],
        },
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
