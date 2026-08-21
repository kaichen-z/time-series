from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.llm import FakeLLMClient
from numerical_agent.evolution import (
    bootstrap,
    commit_module,
    evolve_once,
    git,
    init_repo,
    run_evolution,
)
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.module import MODULE_HEADER, read_module, write_module, parse_module


def method(name: str, body: str = "    return [float(history[-1])] * horizon") -> str:
    return f'def {name}(history, horizon, frequency):\n    """Use when nothing better applies."""\n{body}\n'


def seed_repo(tmp_path: Path, *names: str) -> Path:
    repo = init_repo(tmp_path / "evo")
    module = parse_module(MODULE_HEADER + "\n\n" + "\n\n".join(method(n) for n in names))
    write_module(repo / "methods.py", module)
    commit_module(repo, "seed", [])
    return repo


def tasks() -> tuple[Task, ...]:
    return (
        Task("t1", tuple(float(i) for i in range(20)), 2, "1 day", (19.0, 19.0)),
        Task("t2", tuple(float(i) for i in range(24)), 2, "1 day", (23.0, 23.0)),
    )


def scripted(operations: list[dict]) -> FakeLLMClient:
    return FakeLLMClient([json.dumps({"operations": operations})])


def test_bootstrap_writes_and_commits_a_module(tmp_path: Path) -> None:
    llm = FakeLLMClient([
        json.dumps({"code": method("naive_last")}),
        json.dumps({"code": method("naive_mean")}),
    ])
    definitions = [
        {"name": "naive_last", "description": "Repeat the last value."},
        {"name": "naive_mean", "description": "Use the historical mean."},
    ]

    module = bootstrap(tmp_path / "evo", definitions, llm)

    assert module.names() == ("naive_last", "naive_mean")
    assert read_module(tmp_path / "evo" / "methods.py").names() == module.names()
    assert git(tmp_path / "evo", "log", "--oneline").count("\n") == 0  # exactly one commit


def test_bootstrap_skips_a_definition_the_model_fails(tmp_path: Path) -> None:
    llm = FakeLLMClient([
        json.dumps({"code": method("good_one")}),
        "not json at all",
        json.dumps({"code": method("third_one")}),
    ])
    definitions = [{"name": n, "description": "d"} for n in ("good_one", "bad_one", "third_one")]

    module = bootstrap(tmp_path / "evo", definitions, llm)

    # One failure must not abort the whole bootstrap.
    assert module.names() == ("good_one", "third_one")


def test_a_generation_applies_operations_and_commits(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta", "gamma")
    llm = scripted([
        {"op": "delete", "name": "gamma", "reason": "identical scores to alpha on every task"},
    ])

    outcome = evolve_once(repo, tasks(), llm, 1)

    assert outcome.method_count == 2
    assert read_module(repo / "methods.py").names() == ("alpha", "beta")
    assert outcome.applied == ("delete gamma: identical scores to alpha on every task",)
    assert outcome.rejected == ""


def test_the_reason_becomes_the_commit_body(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    llm = scripted([{"op": "delete", "name": "beta", "reason": "crashed on 80 of 80 tasks"}])

    evolve_once(repo, tasks(), llm, 1)

    assert "crashed on 80 of 80 tasks" in git(repo, "log", "-1", "--format=%B")


def test_a_rejected_generation_leaves_the_module_untouched(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    before = (repo / "methods.py").read_text(encoding="utf-8")
    llm = scripted([{"op": "delete", "name": "does_not_exist", "reason": "bad reference"}])

    outcome = evolve_once(repo, tasks(), llm, 1)

    assert outcome.applied == ()
    assert "unknown method" in outcome.rejected
    assert (repo / "methods.py").read_text(encoding="utf-8") == before
    assert git(repo, "log", "--oneline").count("\n") == 0  # still just the seed commit


def test_invalid_generated_code_is_rejected_without_writing(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    before = (repo / "methods.py").read_text(encoding="utf-8")
    llm = scripted([
        {"op": "rewrite", "name": "alpha", "code": "def alpha(history):\n    return []\n", "reason": "x"}
    ])

    outcome = evolve_once(repo, tasks(), llm, 1)

    assert "must take exactly" in outcome.rejected
    assert (repo / "methods.py").read_text(encoding="utf-8") == before


def test_metrics_and_transcript_are_written_each_generation(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    llm = scripted([{"op": "delete", "name": "beta", "reason": "dominated"}])

    evolve_once(repo, tasks(), llm, 4)

    metrics = json.loads((repo / "generation_004_metrics.json").read_text(encoding="utf-8"))
    assert {entry["method"] for entry in metrics["reports"]} == {"alpha", "beta"}
    assert (repo / "transcripts" / "generation_004.md").exists()


def test_the_whole_module_reaches_the_prompt(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    llm = scripted([])

    evolve_once(repo, tasks(), llm, 1)

    sent = llm.calls[0]["messages"][0]["content"]
    assert "def alpha(" in sent and "def beta(" in sent
    assert "mean_mae" in sent


def test_run_evolution_stops_when_a_generation_changes_nothing(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta", "gamma")
    llm = FakeLLMClient([
        json.dumps({"operations": [{"op": "delete", "name": "gamma", "reason": "redundant"}]}),
        json.dumps({"operations": []}),
        json.dumps({"operations": [{"op": "delete", "name": "beta", "reason": "never reached"}]}),
    ])

    outcomes = run_evolution(repo, tasks(), llm, generations=3)

    assert len(outcomes) == 2
    assert read_module(repo / "methods.py").names() == ("alpha", "beta")


def test_successive_generations_build_a_readable_history(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta", "gamma", "delta")
    llm = FakeLLMClient([
        json.dumps({"operations": [{"op": "delete", "name": "delta", "reason": "crashed everywhere"}]}),
        json.dumps({"operations": [
            {"op": "merge", "names": ["beta", "gamma"], "into": "beta",
             "code": method("beta"), "reason": "identical forecasts on all tasks"}
        ]}),
    ])

    run_evolution(repo, tasks(), llm, generations=2)

    log = git(repo, "log", "--format=%s")
    assert log.splitlines() == [
        "generation 2: 1 operations",
        "generation 1: 1 operations",
        "seed",
    ]
    assert read_module(repo / "methods.py").names() == ("alpha", "beta")
