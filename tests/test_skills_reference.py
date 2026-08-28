from __future__ import annotations

import ast
from pathlib import Path

import pytest

from numerical_agent.evolution import analysis_skills
from numerical_agent.evolution.skills_reference import SKILLS_PATH, build_skills_reference


def public_skills() -> list[str]:
    tree = ast.parse(SKILLS_PATH.read_text(encoding="utf-8"))
    return [
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def test_every_public_skill_appears_with_a_call_signature() -> None:
    index = build_skills_reference()

    for name in public_skills():
        assert f"{name}(" in index, name


def test_no_private_helper_leaks_into_the_index() -> None:
    index = build_skills_reference()
    tree = ast.parse(SKILLS_PATH.read_text(encoding="utf-8"))
    private = [
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_")
    ]

    assert private, "expected the library to have private helpers"
    for name in private:
        assert f"{name}(" not in index, name


def test_every_skill_carries_a_one_line_summary() -> None:
    lines = build_skills_reference().splitlines()
    signatures = [i for i, line in enumerate(lines) if line.startswith("    ") and "(" in line]

    assert signatures
    for position in signatures:
        following = lines[position + 1] if position + 1 < len(lines) else ""
        if following.startswith("        "):
            assert following.strip()


def test_every_section_is_present_and_in_order() -> None:
    """The prompt lists sections by name; no letters, no gaps to keep in sync."""
    from numerical_agent.evolution.skills_reference import SECTION_TITLES

    index = build_skills_reference()
    positions = [index.index(f"{title}:") for title in SECTION_TITLES]

    assert positions == sorted(positions)


def test_the_option_constants_are_listed_so_literals_can_be_resolved() -> None:
    index = build_skills_reference()

    # Signatures name aliases like Cost and Search; the model needs their permitted values.
    for constant in ("COSTS", "SEARCHES", "PENALTIES", "AR_METHODS", "DISTANCE_METRICS"):
        assert constant in index, constant
    for value in ("'pelt'", "'kernel_rbf'", "'soft_dtw'", "'yule_walker'"):
        assert value in index, value


def test_the_index_explains_the_namespace_and_the_model_protocol() -> None:
    index = build_skills_reference()

    assert "P." in index
    assert ".extrapolate(horizon)" in index
    assert "NotApplicable" in index


def test_the_index_is_small_enough_to_sit_beside_the_module_in_a_prompt() -> None:
    index = build_skills_reference()
    source = SKILLS_PATH.read_text(encoding="utf-8")

    assert len(index) < len(source) / 8
    assert len(index.splitlines()) < 200


def test_build_skills_reference_reads_an_explicit_path(tmp_path: Path) -> None:
    stub = tmp_path / "stub.py"
    stub.write_text(
        '# Structure inference\ndef only_skill(x, k: int = 2):\n    """Just the one."""\n',
        encoding="utf-8",
    )

    index = build_skills_reference(stub)

    assert "only_skill(x, k: int=2)" in index
    assert "Just the one." in index


def test_the_index_matches_the_library_actually_imported() -> None:
    # Guards against the index drifting to a stale copy of the library on disk.
    assert SKILLS_PATH == Path(analysis_skills.__file__)


@pytest.mark.parametrize("name", ["detect_changepoints", "fit_ar", "decompose", "denoise"])
def test_key_skills_show_their_keyword_options_inline(name: str) -> None:
    line = next(
        line for line in build_skills_reference().splitlines() if line.strip().startswith(f"{name}(")
    )

    assert "=" in line, f"{name} should show its defaults"
