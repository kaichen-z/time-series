from __future__ import annotations

from pathlib import Path

import pytest

from numerical_agent.evolution import MODULE_NAME, exemplar_methods, git
from numerical_agent.evolution.execution import Task, load_methods, run_module
from numerical_agent.evolution.module import SKILLS_MODULE, read_module
from numerical_agent.seed_evolution import seed


def tasks() -> tuple[Task, ...]:
    """Two seasonal tasks long enough that none of the exemplars declines on length alone."""
    period, length, horizon = 12, 240, 12
    rising = [10.0 + 0.05 * i + 3.0 * ((i % period) - period / 2) for i in range(length + horizon)]
    flat = [50.0 + 2.0 * ((i % period) - period / 2) for i in range(length + horizon)]
    return (
        Task("rising", tuple(rising[:length]), horizon, "1 hour", tuple(rising[length:])),
        Task("flat", tuple(flat[:length]), horizon, "1 hour", tuple(flat[length:])),
    )


def test_the_exemplar_module_satisfies_the_method_contract() -> None:
    module = read_module(exemplar_methods.__file__)

    assert len(module.names()) >= 3
    for method in module.methods:
        assert method.docstring.strip()
        assert "def " in method.source


def test_every_exemplar_composes_the_skill_library_rather_than_inlining_statistics() -> None:
    module = read_module(exemplar_methods.__file__)

    for method in module.methods:
        assert "P." in method.source, f"{method.name} never calls a skill"


def test_seeding_creates_a_repository_with_one_commit(tmp_path: Path) -> None:
    repo = tmp_path / "evo"

    commit = seed(repo)

    assert commit
    assert (repo / MODULE_NAME).exists()
    assert git(repo, "log", "--oneline").count("\n") == 0
    assert "seed 5 composed forecasting methods" in git(repo, "log", "-1", "--format=%s")


def test_the_seeded_module_carries_the_skill_import_in_its_header(tmp_path: Path) -> None:
    repo = tmp_path / "evo"
    seed(repo)

    text = (repo / MODULE_NAME).read_text(encoding="utf-8")

    assert f"import {SKILLS_MODULE} as P" in text
    assert f"from {SKILLS_MODULE} import NotApplicable" in text


def test_seeding_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    repo = tmp_path / "evo"
    seed(repo)

    with pytest.raises(FileExistsError):
        seed(repo)

    assert seed(repo, force=True)


def test_the_seeded_module_loads_only_its_methods(tmp_path: Path) -> None:
    repo = tmp_path / "evo"
    seed(repo)

    _module, functions = load_methods(repo / MODULE_NAME)

    assert set(functions) == set(read_module(exemplar_methods.__file__).names())
    # Neither the skill namespace nor the imported exception is a measurable method.
    assert "P" not in functions
    assert "NotApplicable" not in functions


def test_no_exemplar_crashes_or_returns_an_invalid_forecast(tmp_path: Path) -> None:
    repo = tmp_path / "evo"
    seed(repo)

    _outcomes, reports = run_module(repo / MODULE_NAME, tasks())

    for report in reports:
        assert report.crashed == 0, f"{report.method}: {report.sample_failures}"
        assert report.invalid == 0, f"{report.method}: {report.sample_failures}"


def test_at_least_one_exemplar_forecasts_every_task(tmp_path: Path) -> None:
    repo = tmp_path / "evo"
    seed(repo)

    outcomes, _reports = run_module(repo / MODULE_NAME, tasks())
    covered = {o.task_id for o in outcomes if o.status == "success"}

    assert covered == {task.task_id for task in tasks()}


def test_a_skill_declining_a_series_is_not_applicable_rather_than_a_crash(tmp_path: Path) -> None:
    """The header imports NotApplicable from the library, so the classes are identical."""
    repo = tmp_path / "evo"
    seed(repo)
    short = Task("short", (1.0, 2.0, 3.0, 4.0, 5.0), 3, "1 hour", (6.0, 7.0, 8.0))

    _outcomes, reports = run_module(repo / MODULE_NAME, (short,))

    for report in reports:
        assert report.crashed == 0, f"{report.method}: {report.sample_failures}"
        assert report.not_applicable == 1
