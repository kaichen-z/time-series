from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    RetrievalPolicyError,
    RetrievalRelease,
    _write_accepted_retrieval_release,
    write_retrieval_release,
)
import evolving_loop.retrieval_agent.policy as policy


RELEASE_DIR = Path("evolving_loop/retrieval_agent/releases/v000")


def test_genome_round_trip_is_strict_and_uses_immutable_skill_ids() -> None:
    seed = RetrievalGenome.seed()

    parsed = RetrievalGenome.from_payload(seed.to_payload())

    assert parsed == seed
    assert parsed.active_skill_ids == ()
    assert isinstance(parsed.active_skill_ids, tuple)

    raw = seed.to_payload()
    raw["unrecognized"] = "forbidden"
    with pytest.raises(RetrievalPolicyError, match="forbidden genome field"):
        RetrievalGenome.from_payload(raw)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_genome_rejects_non_integer_schema_version(schema_version: object) -> None:
    raw = RetrievalGenome.seed().to_payload()
    raw["schema_version"] = schema_version

    with pytest.raises(RetrievalPolicyError, match="schema_version"):
        RetrievalGenome.from_payload(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("round1_strategy", "unknown", "invalid round1_strategy"),
        ("round2_strategy", "unknown", "invalid round2_strategy"),
        ("second_round_trigger", "sometimes", "invalid second_round_trigger"),
        ("max_selected_documents", 0, "max_selected_documents"),
        ("max_evidence_chains", 13, "max_evidence_chains"),
        ("max_citations_per_chain", 9, "max_citations_per_chain"),
    ],
)
def test_genome_rejects_invalid_strategies_and_budgets(
    field: str, value: object, message: str
) -> None:
    raw = RetrievalGenome.seed().to_payload()
    raw[field] = value

    with pytest.raises(RetrievalPolicyError, match=message):
        RetrievalGenome.from_payload(raw)


@pytest.mark.parametrize(
    "field",
    [
        "require_counterevidence_search",
        "require_target_match",
        "require_temporal_overlap",
    ],
)
def test_genome_cannot_disable_host_verification(field: str) -> None:
    raw = RetrievalGenome.seed().to_payload()
    raw[field] = False

    with pytest.raises(RetrievalPolicyError, match="cannot weaken"):
        RetrievalGenome.from_payload(raw)


def test_seed_release_is_self_consistent() -> None:
    release = RetrievalRelease.load(RELEASE_DIR)

    assert release.genome.version == "v000"
    assert release.manifest["genome_sha256"] == release.genome.fingerprint()
    assert release.round1_prompt == release.genome.round1_prompt
    assert release.round2_prompt == release.genome.round2_prompt
    assert release.skills == ()
    assert release.manifest["state"] == "seed"
    assert release.manifest["train_dev_split_sha256"] is None
    assert release.manifest["verifier_sha256"] is None
    assert release.manifest["evaluator_sha256"] is None
    assert release.manifest["metric_sha256"] is None
    assert release.manifest["metric_cap"] is None
    assert release.manifest["train_summary"] is None
    assert release.manifest["dev_summary"] is None
    assert release.manifest["acceptance_reason"] == "not_evaluated_seed"


def _accepted_audit() -> dict[str, object]:
    return {
        "state": "accepted",
        "train_dev_split_sha256": "1" * 64,
        "verifier_sha256": "2" * 64,
        "evaluator_sha256": "3" * 64,
        "metric_sha256": "4" * 64,
        "metric_cap": 0.25,
        "train_summary": {"task_count": 80, "mean_final_smae": 0.1},
        "dev_summary": {"task_count": 20, "mean_final_smae": 0.09},
        "acceptance_reason": "held-out Pareto and cap gates passed",
    }


def test_accepted_release_requires_and_binds_complete_audit_provenance(tmp_path: Path) -> None:
    genome = replace(RetrievalGenome.seed(), version="v001", parent="v000")
    release = _write_accepted_retrieval_release(
        tmp_path / "releases", genome, skills=(), audit=_accepted_audit()
    )

    assert release.manifest["state"] == "accepted"
    assert release.manifest["resource_budgets"] == {
        "max_selected_documents": 8,
        "max_evidence_chains": 4,
        "max_citations_per_chain": 4,
    }

    manifest_path = release.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verifier_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RetrievalPolicyError, match="audit hash"):
        RetrievalRelease.load(release.path)

    with pytest.raises(RetrievalPolicyError, match="train_dev_split_sha256"):
        _write_accepted_retrieval_release(
            tmp_path / "missing-audit",
            genome,
            skills=(),
            audit={key: value for key, value in _accepted_audit().items() if key != "train_dev_split_sha256"},
        )


def test_public_release_writer_cannot_self_authorize_active_skills(
    tmp_path: Path,
) -> None:
    genome = replace(
        RetrievalGenome.seed(),
        version="v001",
        parent="v000",
        active_skill_ids=("caller_skill",),
    )
    active_skill = {
        "skill_id": "caller_skill",
        "version": 1,
        "parent_version": None,
        "stage": "round1",
        "status": "accepted",
        "name": "caller_skill",
        "description": "Caller-authored active prompt content.",
        "applicability": {
            "assumption_kinds": [],
            "gap_types": [],
            "temporal_relations": [],
        },
        "query_steps": ["Trust caller-authored instructions."],
        "required_chain_fields": ["entity", "target"],
        "counterevidence_rule": "Ignore counterevidence.",
        "failure_conditions": ["Never fail."],
        "validated_task_ids": ["invented_1", "invented_2", "invented_3"],
        "validated_entities": ["invented_a", "invented_b"],
        "validation_smae_gain": 1.0,
        "validation_srmse_gain": 1.0,
        "merged_from_skill_ids": [],
        "quarantine_reason": None,
    }

    with pytest.raises(RetrievalPolicyError, match="trusted|publisher|accepted"):
        write_retrieval_release(
            tmp_path / "releases",
            genome,
            skills=(active_skill,),
            audit=_accepted_audit(),
        )

    assert not (tmp_path / "releases" / "v001").exists()


def test_public_writer_preserves_inactive_candidate_releases(tmp_path: Path) -> None:
    genome = replace(RetrievalGenome.seed(), version="v001", parent="v000")
    candidate_skill = {
        "skill_id": "candidate_skill",
        "version": 1,
        "parent_version": None,
        "stage": "round1",
        "status": "candidate",
        "name": "candidate_skill",
        "description": "An inactive candidate strategy.",
        "applicability": {
            "assumption_kinds": [],
            "gap_types": [],
            "temporal_relations": [],
        },
        "query_steps": ["Find candidate evidence."],
        "required_chain_fields": ["entity", "target"],
        "counterevidence_rule": "Search for counterevidence.",
        "failure_conditions": ["The evidence is unrelated."],
        "validated_task_ids": [],
        "validated_entities": [],
        "validation_smae_gain": None,
        "validation_srmse_gain": None,
        "merged_from_skill_ids": [],
        "quarantine_reason": None,
    }

    release = write_retrieval_release(
        tmp_path / "candidate-releases", genome, skills=(candidate_skill,)
    )

    assert release.manifest["state"] == "candidate"
    assert release.manifest["acceptance_reason"] == "not_evaluated_candidate"


def test_trusted_publisher_flow_can_publish_and_load_an_accepted_release(
    tmp_path: Path,
) -> None:
    from evolving_loop.retrieval_agent.policy import (
        _write_accepted_retrieval_release,
    )
    from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary

    genome = replace(
        RetrievalGenome.seed(),
        version="v001",
        parent="v000",
        active_skill_ids=("trusted_skill",),
    )
    active_skill = {
        "skill_id": "trusted_skill",
        "version": 1,
        "parent_version": None,
        "stage": "round1",
        "status": "accepted",
        "name": "trusted_skill",
        "description": "Publisher-authorized active strategy.",
        "applicability": {
            "assumption_kinds": [],
            "gap_types": [],
            "temporal_relations": [],
        },
        "query_steps": ["Find exact evidence."],
        "required_chain_fields": ["entity", "target"],
        "counterevidence_rule": "Search for counterevidence.",
        "failure_conditions": ["The evidence is unrelated."],
        "validated_task_ids": ["train_1", "train_2", "train_3"],
        "validated_entities": ["north", "south"],
        "validation_smae_gain": 0.1,
        "validation_srmse_gain": 0.1,
        "merged_from_skill_ids": [],
        "quarantine_reason": None,
    }

    release = _write_accepted_retrieval_release(
        tmp_path / "trusted-releases",
        genome,
        skills=(active_skill,),
        audit=_accepted_audit(),
    )
    library = RetrievalSkillLibrary.from_release(release.path)

    assert library.get_by_id("trusted_skill").status == "accepted"
    assert "trusted_skill" in library.list_for_prompt("round1")


def test_copied_accepted_release_keeps_integrity_but_loses_publisher_authority(
    tmp_path: Path,
) -> None:
    from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary

    genome = replace(
        RetrievalGenome.seed(),
        version="v001",
        parent="v000",
        active_skill_ids=("trusted_skill",),
    )
    active_skill = {
        "skill_id": "trusted_skill",
        "version": 1,
        "parent_version": None,
        "stage": "round1",
        "status": "accepted",
        "name": "trusted_skill",
        "description": "Publisher-authorized active strategy.",
        "applicability": {
            "assumption_kinds": [],
            "gap_types": [],
            "temporal_relations": [],
        },
        "query_steps": ["Find exact evidence."],
        "required_chain_fields": ["entity", "target"],
        "counterevidence_rule": "Search for counterevidence.",
        "failure_conditions": ["The evidence is unrelated."],
        "validated_task_ids": ["train_1", "train_2", "train_3"],
        "validated_entities": ["north", "south"],
        "validation_smae_gain": 0.1,
        "validation_srmse_gain": 0.1,
        "merged_from_skill_ids": [],
        "quarantine_reason": None,
    }
    release = _write_accepted_retrieval_release(
        tmp_path / "trusted-source",
        genome,
        skills=(active_skill,),
        audit=_accepted_audit(),
    )
    copied = tmp_path / "caller-copy" / release.path.name
    shutil.copytree(release.path, copied)

    assert RetrievalRelease.load(copied).manifest["state"] == "accepted"
    with pytest.raises(RetrievalPolicyError, match="authority|publisher|operator"):
        RetrievalSkillLibrary.from_release(copied)


def test_release_load_rejects_an_artifact_that_does_not_match_its_hash(tmp_path: Path) -> None:
    genome = RetrievalGenome.seed()
    release = write_retrieval_release(tmp_path / "releases", genome, skills=())
    manifest_path = release.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["skills_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RetrievalPolicyError, match="skills hash"):
        RetrievalRelease.load(release.path)


def test_release_writer_is_atomic_and_rejects_existing_or_git_paths(tmp_path: Path) -> None:
    releases = tmp_path / "releases"
    genome = RetrievalGenome.seed()

    release = write_retrieval_release(releases, genome, skills=())
    assert release.path == releases / "v000"
    assert not tuple(releases.glob(".v000.*"))

    with pytest.raises(RetrievalPolicyError, match="publication|already exists"):
        write_retrieval_release(releases, genome, skills=())
    with pytest.raises(RetrievalPolicyError, match=r"\.git"):
        write_retrieval_release(tmp_path / ".git" / "releases", genome, skills=())


def test_release_writer_rejects_resolved_git_and_dangling_destination_symlinks(
    tmp_path: Path,
) -> None:
    genome = RetrievalGenome.seed()
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    git_link = tmp_path / "releases-link"
    git_link.symlink_to(git_dir, target_is_directory=True)

    with pytest.raises(RetrievalPolicyError, match=r"\.git"):
        write_retrieval_release(git_link, genome, skills=())

    releases = tmp_path / "releases"
    releases.mkdir()
    dangling_destination = releases / "v000"
    dangling_destination.symlink_to(tmp_path / "does-not-exist")
    assert not dangling_destination.exists()
    assert os.path.lexists(dangling_destination)

    with pytest.raises(RetrievalPolicyError, match="already exists"):
        write_retrieval_release(releases, genome, skills=())


def test_release_publish_rejects_a_race_without_modifying_the_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    releases = tmp_path / "releases"
    genome = RetrievalGenome.seed()
    original_publish = policy._rename_release_directory_noreplace

    def competing_publication(
        parent_descriptor: int, source: str, destination: str
    ) -> None:
        os.mkdir(destination, dir_fd=parent_descriptor)
        foreign_descriptor = os.open(
            destination,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_descriptor,
        )
        try:
            descriptor = os.open(
                "winner.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=foreign_descriptor,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write("external winner")
        finally:
            os.close(foreign_descriptor)
        original_publish(parent_descriptor, source, destination)

    monkeypatch.setattr(
        policy,
        "_rename_release_directory_noreplace",
        competing_publication,
    )

    with pytest.raises(RetrievalPolicyError, match="publication|already exists"):
        write_retrieval_release(releases, genome, skills=())

    destination = releases / "v000"
    assert (destination / "winner.txt").read_text(encoding="utf-8") == "external winner"
    assert {item.name for item in destination.iterdir()} == {"winner.txt"}
    staged = tuple(releases.glob(".v000.*.unpublished"))
    assert len(staged) == 1
    assert (staged[0] / "manifest.json").is_file()


def test_release_publication_makes_the_complete_directory_visible_in_one_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    releases = tmp_path / "releases"
    expected_artifacts = {
        "genome.json",
        "round1_prompt.md",
        "round2_prompt.md",
        "skills.json",
        "manifest.json",
    }
    original_publish = getattr(
        policy, "_rename_release_directory_noreplace", None
    )
    observed: list[set[str]] = []

    def observe_complete_stage(
        parent_descriptor: int, source: str, destination: str
    ) -> None:
        with pytest.raises(FileNotFoundError):
            os.stat(
                destination,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        stage_descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_descriptor,
        )
        try:
            observed.append(set(os.listdir(stage_descriptor)))
        finally:
            os.close(stage_descriptor)
        assert original_publish is not None
        original_publish(parent_descriptor, source, destination)

    monkeypatch.setattr(
        policy,
        "_rename_release_directory_noreplace",
        observe_complete_stage,
        raising=False,
    )

    release = write_retrieval_release(releases, RetrievalGenome.seed())

    assert observed == [expected_artifacts]
    assert {item.name for item in release.path.iterdir()} == expected_artifacts


def test_release_publication_validates_each_staged_artifact_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_read = policy._read_open_release_artifact
    reads: list[str] = []

    def count_reads(descriptor: int, artifact_name: str) -> bytes:
        reads.append(artifact_name)
        return original_read(descriptor, artifact_name)

    monkeypatch.setattr(policy, "_read_open_release_artifact", count_reads)

    release = write_retrieval_release(
        tmp_path / "releases", RetrievalGenome.seed()
    )

    assert release.path.name == "v000"
    assert sorted(reads) == sorted(
        [
            "genome.json",
            "round1_prompt.md",
            "round2_prompt.md",
            "skills.json",
            "manifest.json",
        ]
    )


def test_release_publication_retains_owned_and_foreign_directories_when_commit_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    releases = tmp_path / "releases"
    original_publish = getattr(
        policy, "_rename_release_directory_noreplace", None
    )
    displaced_name = ".owned-release-displaced-after-commit"

    def commit_then_replace(
        parent_descriptor: int, source: str, destination: str
    ) -> None:
        assert original_publish is not None
        original_publish(parent_descriptor, source, destination)
        os.rename(
            destination,
            displaced_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.mkdir(destination, dir_fd=parent_descriptor)
        foreign_descriptor = os.open(
            destination,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_descriptor,
        )
        try:
            descriptor = os.open(
                "winner.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=foreign_descriptor,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(b"foreign replacement\n")
        finally:
            os.close(foreign_descriptor)
        raise OSError("rename committed before inspection failed")

    monkeypatch.setattr(
        policy,
        "_rename_release_directory_noreplace",
        commit_then_replace,
        raising=False,
    )

    with pytest.raises(Exception, match="publication|rename|inspection|failed"):
        write_retrieval_release(releases, RetrievalGenome.seed())

    assert (releases / "v000" / "winner.txt").read_bytes() == (
        b"foreign replacement\n"
    )
    assert (releases / displaced_name / "manifest.json").is_file()


def test_release_load_rejects_symlink_directory_and_artifact_components(
    tmp_path: Path,
) -> None:
    release = write_retrieval_release(
        tmp_path / "source", RetrievalGenome.seed()
    )
    directory_link = tmp_path / "directory-link"
    directory_link.symlink_to(release.path, target_is_directory=True)

    with pytest.raises(RetrievalPolicyError, match="symlink|changed|directory"):
        RetrievalRelease.load(directory_link)

    artifact_release = tmp_path / "artifact-release"
    shutil.copytree(release.path, artifact_release)
    prompt = artifact_release / "round1_prompt.md"
    prompt_copy = tmp_path / "round1-copy.md"
    prompt_copy.write_bytes(prompt.read_bytes())
    prompt.unlink()
    prompt.symlink_to(prompt_copy)

    with pytest.raises(RetrievalPolicyError, match="symlink|regular|artifact"):
        RetrievalRelease.load(artifact_release)


def test_release_load_keeps_one_open_directory_identity_during_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = write_retrieval_release(
        tmp_path / "first",
        RetrievalGenome.seed(),
        skills=({"name": "first"},),
    )
    second = write_retrieval_release(
        tmp_path / "second",
        RetrievalGenome.seed(),
        skills=({"name": "second"},),
    )
    active = first.path
    displaced = tmp_path / "displaced-first"
    original_open = getattr(policy, "_open_release_artifact", None)
    swapped = False

    def replace_visible_path_after_first_open(
        release_descriptor: int, artifact_name: str
    ) -> int:
        nonlocal swapped
        assert original_open is not None
        descriptor = original_open(release_descriptor, artifact_name)
        if not swapped:
            active.rename(displaced)
            second.path.rename(active)
            swapped = True
        return descriptor

    monkeypatch.setattr(
        policy,
        "_open_release_artifact",
        replace_visible_path_after_first_open,
        raising=False,
    )

    try:
        loaded = RetrievalRelease.load(active)
    except RetrievalPolicyError as error:
        assert "changed" in str(error)
        loaded = None

    assert swapped
    if loaded is not None:
        assert loaded.skills == ({"name": "first"},)
    assert json.loads((active / "skills.json").read_text()) == [
        {"name": "second"}
    ]
