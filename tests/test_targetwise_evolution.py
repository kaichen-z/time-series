from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from common.llm import FakeLLMClient, LLMResponse
from numerical_agent.evolution import commit_module, init_repo
from numerical_agent.evolution.cache import OutcomeCache
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.module import MODULE_HEADER, ModuleError, parse_module
from numerical_agent.evolution.module import read_module, write_module
from numerical_agent.evolution.targetwise import evolve_targets_once, parse_target_proposals


SARIMA = '''def sarima_auto(history, horizon, frequency):
    """Use when seasonal SARIMA order selection is appropriate."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    best = None
    best_aic = float("inf")
    for p in range(2):
        model = SARIMAX(history, order=(p, 0, 0), seasonal_order=(1, 0, 0, 7))
        result = model.fit(disp=False)
        if result.aic < best_aic:
            best_aic = result.aic
            best = result
    return list(best.forecast(steps=horizon))
'''


ALPHA = '''def alpha(history, horizon, frequency):
    """Use when a generic test method applies."""
    return [float(history[-1])] * horizon
'''


def module():
    return parse_module(MODULE_HEADER + "\n\n" + SARIMA + "\n\n" + ALPHA)


def test_target_proposals_allow_repair_or_fork_for_verified_identity() -> None:
    proposals = parse_target_proposals(
        json.dumps({"targets": [
            {"name": "sarima_auto", "action": "repair", "reason": "timeouts"},
            {"name": "alpha", "action": "fork", "reason": "needs challenger"},
        ]}),
        module(),
        max_targets=3,
    )

    assert proposals[0].name == "sarima_auto"
    assert proposals[0].allowed_actions == ("repair", "fork")
    assert proposals[1].allowed_actions == ("fork",)


def test_target_proposals_reject_duplicate_names_before_mutation() -> None:
    response = json.dumps({"targets": [
        {"name": "alpha", "action": "fork", "reason": "one"},
        {"name": "alpha", "action": "delete", "reason": "two"},
    ]})

    with pytest.raises(ModuleError, match="duplicate selector target"):
        parse_target_proposals(response, module(), max_targets=3)


def test_target_proposals_enforce_the_configured_cap() -> None:
    response = json.dumps({"targets": [
        {"name": "sarima_auto", "action": "repair", "reason": "one"},
        {"name": "alpha", "action": "fork", "reason": "two"},
    ]})

    proposals = parse_target_proposals(response, module(), max_targets=1)

    assert [proposal.name for proposal in proposals] == ["sarima_auto"]


def test_target_proposals_reject_a_cap_above_the_hard_ceiling() -> None:
    with pytest.raises(ValueError, match="at most ten"):
        parse_target_proposals('{"targets": []}', module(), max_targets=11)


def bad_method(name: str) -> str:
    return f'''def {name}(history, horizon, frequency):
    """Use when testing independent target-wise children."""
    return [0.0] * horizon
'''


def evolution_tasks(prefix: str) -> tuple[Task, ...]:
    return (
        Task(f"{prefix}1", (1.0, 2.0, 3.0), 2, "1 day", (3.0, 3.0)),
        Task(f"{prefix}2", (4.0, 5.0, 6.0), 2, "1 day", (6.0, 6.0)),
    )


def test_invalid_target_does_not_block_a_later_improving_child(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    parent = parse_module(
        MODULE_HEADER + "\n\n" + bad_method("alpha") + "\n\n" + bad_method("beta")
    )
    write_module(repo / "methods.py", parent)
    commit_module(repo, "seed", ())
    selector = FakeLLMClient([json.dumps({"targets": [
        {"name": "alpha", "action": "fork", "reason": "first challenger"},
        {"name": "beta", "action": "fork", "reason": "second challenger"},
    ]})])
    mutator = FakeLLMClient([
        json.dumps({"operations": [
            {"op": "delete", "name": "alpha", "reason": "illegal escalation"}
        ]}),
        json.dumps({"operations": [{
            "op": "fork",
            "from": "beta",
            "new_identity": "last value challenger",
            "code": '''def beta_last(history, horizon, frequency):
    """Use when the most recent level should persist."""
    return [float(history[-1])] * horizon
''',
            "reason": "replace the biased zero forecast with a distinct challenger",
        }]}),
    ])

    outcome = evolve_targets_once(
        repo,
        evolution_tasks("train"),
        mutator,
        selector,
        generation=1,
        outcome_cache=OutcomeCache(tmp_path / "cache"),
        validation_tasks=evolution_tasks("dev"),
        screen_tasks=1,
        max_targets=3,
        isolate_methods=False,
    )

    assert len(outcome.candidates) == 2
    assert not outcome.candidates[0].accepted
    assert "allowed actions" in outcome.candidates[0].reason
    assert outcome.candidates[1].accepted and outcome.candidates[1].promoted
    assert outcome.applied == (
        "fork beta -> beta_last: replace the biased zero forecast with a distinct challenger",
    )
    assert read_module(repo / "methods.py").names() == ("alpha", "beta", "beta_last")
    assert outcome.cache_misses > 0


def test_changed_method_crash_is_not_hidden_by_portfolio_oracle(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    covering = '''def alpha(history, horizon, frequency):
    """Use as a conditional Parent cover in the crash-gate test."""
    value = float(history[-1]) if history[-1] >= 5 else 0.0
    return [value] * horizon
'''
    parent = parse_module(MODULE_HEADER + "\n\n" + covering + "\n\n" + bad_method("beta"))
    write_module(repo / "methods.py", parent)
    commit_module(repo, "seed", ())
    selector = FakeLLMClient([json.dumps({"targets": [
        {"name": "beta", "action": "fork", "reason": "conditional challenger"}
    ]})])
    mutator = FakeLLMClient([json.dumps({"operations": [{
        "op": "fork", "from": "beta", "new_identity": "low-level specialist",
        "code": '''def beta_low(history, horizon, frequency):
    """Use only for low-level histories."""
    if float(history[-1]) >= 5:
        raise RuntimeError("unsupported high level")
    return [float(history[-1])] * horizon
''',
        "reason": "improve the low-level case",
    }]})])

    result = evolve_targets_once(
        repo, evolution_tasks("train"), mutator, selector, generation=1,
        outcome_cache=OutcomeCache(tmp_path / "cache"),
        validation_tasks=evolution_tasks("dev"), screen_tasks=2,
        isolate_methods=False,
    )

    assert not result.candidates[0].accepted
    assert "crashed or invalid" in result.candidates[0].reason
    assert read_module(repo / "methods.py").names() == ("alpha", "beta")


def test_multiple_independent_children_are_rebased_and_promoted(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    parent = parse_module(
        MODULE_HEADER + "\n\n" + bad_method("alpha") + "\n\n" + bad_method("beta")
    )
    write_module(repo / "methods.py", parent)
    commit_module(repo, "seed", ())
    selector = FakeLLMClient([json.dumps({"targets": [
        {"name": "alpha", "action": "fork", "reason": "low-level specialist"},
        {"name": "beta", "action": "fork", "reason": "high-level specialist"},
    ]})])
    mutator = FakeLLMClient([
        json.dumps({"operations": [{
            "op": "fork", "from": "alpha", "new_identity": "low specialist",
            "code": '''def alpha_low(history, horizon, frequency):
    """Use only when the current level is at most three."""
    if float(history[-1]) > 3:
        raise NotApplicable("high level")
    return [float(history[-1])] * horizon
''', "reason": "cover low histories",
        }]}),
        json.dumps({"operations": [{
            "op": "fork", "from": "beta", "new_identity": "high specialist",
            "code": '''def beta_high(history, horizon, frequency):
    """Use only when the current level exceeds three."""
    if float(history[-1]) <= 3:
        raise NotApplicable("low level")
    return [float(history[-1])] * horizon
''', "reason": "cover high histories",
        }]}),
    ])

    result = evolve_targets_once(
        repo, evolution_tasks("train"), mutator, selector, generation=1,
        outcome_cache=OutcomeCache(tmp_path / "cache"),
        validation_tasks=evolution_tasks("dev"), screen_tasks=2,
        isolate_methods=False,
    )

    assert all(candidate.accepted and candidate.promoted for candidate in result.candidates)
    assert all(
        "child_method_mean_mase" in candidate.validation_metrics
        for candidate in result.candidates
    )
    assert read_module(repo / "methods.py").names() == (
        "alpha", "beta", "beta_high", "alpha_low"
    )
    log = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=repo,
        capture_output=True, text=True, check=True,
    )
    assert int(log.stdout.strip()) == 3


def test_targetwise_requires_a_clean_index_before_calling_the_selector(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    write_module(repo / "methods.py", module())
    (repo / "notes.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "notes.txt"], cwd=repo, check=True)
    commit_module(repo, "seed", ())
    (repo / "notes.txt").write_text("staged user change\n", encoding="utf-8")
    subprocess.run(["git", "add", "notes.txt"], cwd=repo, check=True)
    selector = FakeLLMClient([])

    with pytest.raises(ModuleError, match="clean Git index"):
        evolve_targets_once(
            repo, evolution_tasks("train"), FakeLLMClient([]), selector,
            generation=1, outcome_cache=OutcomeCache(tmp_path / "cache"),
            validation_tasks=evolution_tasks("dev"), isolate_methods=False,
        )

    assert selector.calls == []
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=repo,
        capture_output=True, text=True, check=True,
    )
    assert staged.stdout.strip() == "notes.txt"


def test_neutral_deletion_is_rejected_without_portfolio_improvement(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    parent = parse_module(
        MODULE_HEADER + "\n\n" + bad_method("alpha") + "\n\n" + bad_method("beta")
    )
    write_module(repo / "methods.py", parent)
    commit_module(repo, "seed", ())
    selector = FakeLLMClient([json.dumps({"targets": [
        {"name": "alpha", "action": "delete", "reason": "appears redundant"}
    ]})])
    mutator = FakeLLMClient([json.dumps({"operations": [{
        "op": "delete", "name": "alpha", "reason": "remove duplicate"
    }]})])

    result = evolve_targets_once(
        repo, evolution_tasks("train"), mutator, selector, generation=1,
        outcome_cache=OutcomeCache(tmp_path / "cache"),
        validation_tasks=evolution_tasks("dev"), screen_tasks=2,
        isolate_methods=False,
    )

    assert not result.candidates[0].accepted
    assert "did not improve" in result.candidates[0].reason
    assert read_module(repo / "methods.py").names() == ("alpha", "beta")


def test_judge_diagnoses_every_target_before_successive_halving_keeps_only_best(
    tmp_path: Path,
) -> None:
    repo = init_repo(tmp_path / "repo")
    parent = parse_module(
        MODULE_HEADER
        + "\n\n"
        + bad_method("alpha")
        + "\n\n"
        + bad_method("beta")
        + "\n\n"
        + bad_method("gamma")
    )
    write_module(repo / "methods.py", parent)
    commit_module(repo, "seed", ())
    selector = FakeLLMClient([json.dumps({"targets": [
        {"name": "alpha", "action": "fork", "reason": "best challenger"},
        {"name": "beta", "action": "fork", "reason": "weaker challenger"},
        {"name": "gamma", "action": "fork", "reason": "third challenger"},
    ]})])
    diagnosis = json.dumps({
        "failure_types": ["level_bias"],
        "summary": "The zero forecast systematically underpredicts the observed level.",
        "evidence": ["positive normalized bias across the screen tasks"],
        "mutation_guidance": ["preserve a recent level under stable histories"],
        "confidence": 0.9,
    })
    judge = FakeLLMClient([diagnosis, diagnosis, diagnosis])
    mutator = FakeLLMClient([
        json.dumps({"operations": [{
            "op": "fork", "from": "alpha", "new_identity": "last value",
            "code": '''def alpha_last(history, horizon, frequency):
    """Use when the latest level should persist."""
    return [float(history[-1])] * horizon
''', "reason": "correct the measured level bias",
        }]}),
        json.dumps({"operations": [{
            "op": "fork", "from": "beta", "new_identity": "damped level",
            "code": '''def beta_damped(history, horizon, frequency):
    """Use when a conservative recent level should persist."""
    return [float(history[-1]) - 1.0] * horizon
''', "reason": "partly correct the measured level bias",
        }]}),
        json.dumps({"operations": []}),
    ])

    result = evolve_targets_once(
        repo, evolution_tasks("train"), mutator, selector, judge=judge,
        generation=1, outcome_cache=OutcomeCache(tmp_path / "cache"),
        validation_tasks=evolution_tasks("dev"), screen_tasks=2,
        max_targets=8, full_evaluation_candidates=1, isolate_methods=False,
    )

    assert len(judge.calls) == 3
    assert all("failure_diagnosis" in call["messages"][0]["content"] for call in mutator.calls)
    assert result.candidates[0].promoted
    assert not result.candidates[1].accepted
    assert "successive halving" in result.candidates[1].reason
    assert result.candidates[1].validation_metrics == {}
    assert not result.candidates[2].accepted


class _FailThenMutate:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def complete(self, *, system: str, messages: list[dict], temperature: float = 0.0):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary model outage")
        return LLMResponse(self.response)


def test_one_mutator_failure_does_not_abort_later_independent_targets(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    parent = parse_module(
        MODULE_HEADER + "\n\n" + bad_method("alpha") + "\n\n" + bad_method("beta")
    )
    write_module(repo / "methods.py", parent)
    commit_module(repo, "seed", ())
    selector = FakeLLMClient([json.dumps({"targets": [
        {"name": "alpha", "action": "fork", "reason": "first"},
        {"name": "beta", "action": "fork", "reason": "second"},
    ]})])
    mutator = _FailThenMutate(json.dumps({"operations": [{
        "op": "fork", "from": "beta", "new_identity": "recent level",
        "code": '''def beta_last(history, horizon, frequency):
    """Use when the current level should persist."""
    return [float(history[-1])] * horizon
''',
        "reason": "correct level bias",
    }]}))

    result = evolve_targets_once(
        repo, evolution_tasks("train"), mutator, selector, generation=1,
        outcome_cache=OutcomeCache(tmp_path / "cache"),
        validation_tasks=evolution_tasks("dev"), screen_tasks=2,
        full_evaluation_candidates=1, isolate_methods=False,
    )

    assert "mutator unavailable" in result.candidates[0].reason
    assert result.candidates[0].screen_metrics == {}
    assert result.candidates[1].promoted


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"screen_tasks": 5}, "at most four"),
        ({"full_evaluation_candidates": 4}, "at most three"),
    ],
)
def test_targetwise_rejects_screening_budgets_outside_the_trusted_bounds(
    tmp_path: Path, kwargs: dict[str, int], message: str,
) -> None:
    repo = init_repo(tmp_path / "repo")
    write_module(repo / "methods.py", module())
    commit_module(repo, "seed", ())

    with pytest.raises(ValueError, match=message):
        evolve_targets_once(
            repo, evolution_tasks("train"), FakeLLMClient([]), FakeLLMClient([]),
            generation=1, outcome_cache=OutcomeCache(tmp_path / "cache"),
            validation_tasks=evolution_tasks("dev"), isolate_methods=False, **kwargs,
        )
