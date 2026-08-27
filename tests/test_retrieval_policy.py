from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    RetrievalPolicyError,
    RetrievalRelease,
    write_retrieval_release,
)


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

    with pytest.raises(RetrievalPolicyError, match="already exists"):
        write_retrieval_release(releases, genome, skills=())
    with pytest.raises(RetrievalPolicyError, match=r"\.git"):
        write_retrieval_release(tmp_path / ".git" / "releases", genome, skills=())
