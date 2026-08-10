"""Bundle load/save round-trips, version numbering, and the committed seeds."""

from __future__ import annotations

import dataclasses

import pytest

from evolving_agents.bundles import AGENTS, load_bundle, load_seed, next_version, save_bundle


@pytest.mark.parametrize("agent", AGENTS)
def test_seed_bundles_load(agent: str) -> None:
    bundle = load_seed(agent)
    assert bundle.agent == agent
    assert bundle.version == "v000"
    assert bundle.parent is None
    assert bundle.system_prompt.strip()
    assert bundle.bundle_id == f"{agent}/v000"


@pytest.mark.parametrize("agent", AGENTS)
def test_seed_bundles_declare_enable_thinking(agent: str) -> None:
    assert isinstance(load_seed(agent).hyperparameters["enable_thinking"], bool)


def test_coding_seed_carries_code_templates() -> None:
    templates = load_seed("coding").code_templates
    assert templates
    assert all("def forecast(" in source for source in templates.values())


def test_save_and_load_round_trip(tmp_path) -> None:
    original = load_seed("coding")
    child = dataclasses.replace(original, version="v001", parent="v000", notes_from_evolver="test child")
    path = save_bundle(child, tmp_path)
    assert path == tmp_path / "coding" / "v001.json"

    reloaded = load_bundle(path)
    assert reloaded.version == "v001"
    assert reloaded.parent == "v000"
    assert reloaded.notes_from_evolver == "test child"
    assert reloaded.system_prompt == original.system_prompt
    assert reloaded.code_templates == original.code_templates
    assert reloaded.fewshot_examples == original.fewshot_examples
    assert reloaded.hyperparameters == original.hyperparameters


def test_next_version_counts_up(tmp_path) -> None:
    assert next_version(tmp_path, "coding") == "v000"
    save_bundle(dataclasses.replace(load_seed("coding"), version="v000"), tmp_path)
    assert next_version(tmp_path, "coding") == "v001"
    save_bundle(dataclasses.replace(load_seed("coding"), version="v001"), tmp_path)
    assert next_version(tmp_path, "coding") == "v002"


def test_next_version_ignores_unrelated_files(tmp_path) -> None:
    directory = tmp_path / "coding"
    directory.mkdir(parents=True)
    (directory / "notes.txt").write_text("ignore me", encoding="utf-8")
    (directory / "v007.json.tmp").write_text("{}", encoding="utf-8")
    assert next_version(tmp_path, "coding") == "v000"


def test_load_seed_rejects_unknown_agent() -> None:
    with pytest.raises(ValueError):
        load_seed("nonexistent")
