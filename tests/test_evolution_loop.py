from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.llm import FakeLLMClient
from numerical_agent import evolution
from numerical_agent.evolution import (
    bootstrap,
    commit_module,
    evolve_once,
    init_repo,
    run_evolution,
    run_git,
)
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.module import METHODS_FILE_HEADER, read_module, write_module, parse_module


def method(name: str, body: str = "    return [float(history[-1])] * horizon") -> str:
    return f'def {name}(history, horizon, frequency):\n    """Use when nothing better applies."""\n{body}\n'


def seed_repo(tmp_path: Path, *names: str) -> Path:
    repo = init_repo(tmp_path / "evo")
    module = parse_module(METHODS_FILE_HEADER + "\n\n" + "\n\n".join(method(n) for n in names))
    write_module(repo / "methods.py", module)
    commit_module(repo, "seed", [])
    return repo


def tasks() -> tuple[Task, ...]:
    return (
        Task("t1", tuple(float(i) for i in range(20)), 2, "1 day", (19.0, 19.0)),
        Task("t2", tuple(float(i) for i in range(24)), 2, "1 day", (23.0, 23.0)),
    )


def scripted(operations: list[dict]) -> FakeLLMClient:
    """The same batch twice: a rejected batch is retried once, so both attempts are scripted."""
    payload = json.dumps({"operations": operations})
    return FakeLLMClient([payload, payload])


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
    assert run_git(tmp_path / "evo", "log", "--oneline").count("\n") == 0  # exactly one commit


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

    assert "crashed on 80 of 80 tasks" in run_git(repo, "log", "-1", "--format=%B")


def test_a_rejected_generation_leaves_the_module_untouched(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    before = (repo / "methods.py").read_text(encoding="utf-8")
    llm = scripted([{"op": "delete", "name": "does_not_exist", "reason": "bad reference"}])

    outcome = evolve_once(repo, tasks(), llm, 1)

    assert outcome.applied == ()
    assert "unknown method" in outcome.rejected
    assert (repo / "methods.py").read_text(encoding="utf-8") == before
    assert run_git(repo, "log", "--oneline").count("\n") == 0  # still just the seed commit


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
    assert "mean_smae" in sent


def test_a_local_model_is_loaded_before_the_hour_of_measurement(tmp_path: Path, monkeypatch) -> None:
    """Loading after measurement lets another job take the card while we measure."""
    repo = seed_repo(tmp_path, "alpha", "beta")
    order: list[str] = []
    llm = scripted([])
    llm.preload = lambda: order.append("preload")  # type: ignore[attr-defined]
    original = evolution.run_module
    monkeypatch.setattr(
        evolution, "run_module",
        lambda path, given: (order.append("measure"), original(path, given))[1],
    )

    evolve_once(repo, tasks(), llm, 1)

    assert order == ["preload", "measure"]


def test_a_client_without_weights_to_load_is_left_alone(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")

    outcome = evolve_once(repo, tasks(), scripted([]), 1)  # FakeLLMClient has no preload

    assert outcome.converged


def test_an_unchanged_module_is_not_measured_twice(tmp_path: Path, monkeypatch) -> None:
    """Measuring 98 methods costs an hour; a rejected generation must not pay for it again."""
    repo = seed_repo(tmp_path, "alpha", "beta")
    calls: list[str] = []
    original = evolution.run_module
    monkeypatch.setattr(
        evolution, "run_module",
        lambda path, given: (calls.append(str(path)), original(path, given))[1],
    )
    llm = scripted([{"op": "delete", "name": "does_not_exist", "reason": "bad reference"}])

    evolve_once(repo, tasks(), llm, 1)
    assert len(calls) == 1  # train only; no val tasks were passed
    evolve_once(repo, tasks(), scripted([]), 2)

    # The rejected batch left methods.py untouched, so generation 2 reads the cached entry.
    assert len(calls) == 1


def test_a_changed_module_is_measured_again(tmp_path: Path, monkeypatch) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta", "gamma")
    calls: list[str] = []
    original = evolution.run_module
    monkeypatch.setattr(
        evolution, "run_module",
        lambda path, given: (calls.append(str(path)), original(path, given))[1],
    )

    evolve_once(repo, tasks(), scripted([{"op": "delete", "name": "gamma", "reason": "dominated"}]), 1)
    evolve_once(repo, tasks(), scripted([]), 2)

    assert len(calls) == 2


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

    log = run_git(repo, "log", "--format=%s")
    assert log.splitlines() == [
        "generation 2: 1 operations",
        "generation 1: 1 operations",
        "seed",
    ]
    assert read_module(repo / "methods.py").names() == ("alpha", "beta")


# --------------------------------------------------------------------------------------
# Recovery from a malformed batch
# --------------------------------------------------------------------------------------


def placeholder_batch() -> list[dict]:
    """The exact shape that killed the v002 run: an ellipsis where the code should be."""
    return [
        {"op": "delete", "name": "beta", "reason": "worse everywhere"},
        {"op": "rewrite", "name": "alpha", "code": "...", "reason": "tighten the docstring"},
    ]


def good_batch() -> list[dict]:
    # The body must actually differ, or commit_module correctly makes no commit.
    changed = method("alpha", "    return [float(history[-1]) + 1.0] * horizon")
    return [{"op": "rewrite", "name": "alpha", "code": changed, "reason": "clearer"}]


def test_a_malformed_batch_is_retried_and_the_retry_commits(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    llm = FakeLLMClient([
        json.dumps({"operations": placeholder_batch()}),
        json.dumps({"operations": good_batch()}),
    ])

    outcome = evolve_once(repo, tasks(), llm, 1)

    assert outcome.retried is True
    assert outcome.rejected == ""
    assert outcome.applied
    assert (repo / "transcripts" / "generation_001.md").exists()
    assert (repo / "transcripts" / "generation_001_retry1.md").exists()


def test_the_retry_prompt_quotes_the_rejection(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    llm = FakeLLMClient([
        json.dumps({"operations": placeholder_batch()}),
        json.dumps({"operations": good_batch()}),
    ])

    evolve_once(repo, tasks(), llm, 1)
    retry = (repo / "transcripts" / "generation_001_retry1.md").read_text(encoding="utf-8")

    assert "complete function source" in retry
    assert "rejected" in retry


def test_a_batch_malformed_twice_leaves_the_module_untouched(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    before = (repo / "methods.py").read_text(encoding="utf-8")
    commits = run_git(repo, "log", "--oneline")
    llm = FakeLLMClient([
        json.dumps({"operations": placeholder_batch()}),
        json.dumps({"operations": placeholder_batch()}),
    ])

    outcome = evolve_once(repo, tasks(), llm, 1)

    assert outcome.rejected
    assert outcome.applied == ()
    assert (repo / "methods.py").read_text(encoding="utf-8") == before
    assert run_git(repo, "log", "--oneline") == commits


def test_a_rejected_generation_no_longer_ends_the_run(tmp_path: Path) -> None:
    """The regression that cost generations 8-10 of the v002 run."""
    repo = seed_repo(tmp_path, "alpha", "beta")
    llm = FakeLLMClient([
        json.dumps({"operations": placeholder_batch()}),   # gen 1 first attempt
        json.dumps({"operations": placeholder_batch()}),   # gen 1 retry, also bad
        json.dumps({"operations": good_batch()}),          # gen 2 succeeds
    ])

    outcomes = run_evolution(repo, tasks(), llm, generations=2)

    assert len(outcomes) == 2
    assert outcomes[0].rejected
    assert outcomes[1].applied
    assert "generation 2" in run_git(repo, "log", "-1", "--format=%s")


def test_the_run_still_stops_when_the_model_proposes_nothing(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    llm = FakeLLMClient([json.dumps({"operations": []}), json.dumps({"operations": good_batch()})])

    outcomes = run_evolution(repo, tasks(), llm, generations=3)

    assert len(outcomes) == 1
    assert outcomes[0].rejected == "no operations proposed"


def test_neither_prompt_schema_shows_an_abbreviated_code_placeholder() -> None:
    from numerical_agent.evolution.prompts import WRITE_METHOD_PROMPT, IMPROVE_METHODS_PROMPT

    for prompt in (WRITE_METHOD_PROMPT, IMPROVE_METHODS_PROMPT):
        assert '"def ..."' not in prompt
        assert "def method_name(history, horizon, frequency): ..." not in prompt
    # Phrasing may tighten; the rule that a code field is never abbreviated must survive it.
    assert "never write `...`" in IMPROVE_METHODS_PROMPT
    assert "entire function" in IMPROVE_METHODS_PROMPT


def test_an_empty_response_is_retried_not_a_crash(tmp_path: Path) -> None:
    """A truncated or empty LLM response fails JSON parsing entirely -- this must recover the
    same way a malformed operation does, not propagate out of evolve_once."""
    repo = seed_repo(tmp_path, "alpha", "beta")
    llm = FakeLLMClient(["", json.dumps({"operations": good_batch()})])

    outcome = evolve_once(repo, tasks(), llm, 1)

    assert outcome.applied
    assert outcome.retried is True
    assert (repo / "transcripts" / "generation_001.md").exists()
    assert (repo / "transcripts" / "generation_001_retry1.md").exists()


def test_a_response_thats_not_json_at_all_is_also_retried(tmp_path: Path) -> None:
    """Prose instead of JSON is the same class of failure as an empty response."""
    repo = seed_repo(tmp_path, "alpha", "beta")
    llm = FakeLLMClient([
        "I'll analyze the results and get back to you.",
        json.dumps({"operations": good_batch()}),
    ])

    outcome = evolve_once(repo, tasks(), llm, 1)

    assert outcome.applied
    assert outcome.retried is True


def test_validation_is_measured_and_written_but_never_shown_to_the_model(tmp_path: Path) -> None:
    """Val is the one held-out signal; a score the model can read is one it will optimize."""
    import json

    repo = seed_repo(tmp_path, "alpha", "beta")
    train = (Task("t1", tuple(float(i % 7) for i in range(60)), 3, "1 day", (1.0, 2.0, 3.0)),)
    val = (Task("v1", tuple(float(i % 5) for i in range(60)), 3, "1 day", (2.0, 2.0, 2.0)),)
    llm = FakeLLMClient([json.dumps({"operations": [
        {"op": "delete", "name": "beta", "reason": "dominated"}
    ]})])

    outcome = evolve_once(repo, train, llm, 1, val_tasks=val)

    metrics = json.loads((repo / "generation_001_metrics.json").read_text())
    assert metrics["val_reports"], "validation results must be recorded"
    assert outcome.val_best_smae is not None

    prompt = (repo / "transcripts" / "generation_001.md").read_text()
    assert "val_reports" not in prompt
    assert "v1" not in prompt


def test_without_validation_tasks_the_loop_is_unchanged(tmp_path: Path) -> None:
    import json

    repo = seed_repo(tmp_path, "alpha", "beta")
    train = (Task("t1", tuple(float(i % 7) for i in range(60)), 3, "1 day", (1.0, 2.0, 3.0)),)
    llm = FakeLLMClient([json.dumps({"operations": [
        {"op": "delete", "name": "beta", "reason": "dominated"}
    ]})])

    outcome = evolve_once(repo, train, llm, 1)

    metrics = json.loads((repo / "generation_001_metrics.json").read_text())
    assert metrics["val_reports"] == []
    assert outcome.val_best_smae is None
