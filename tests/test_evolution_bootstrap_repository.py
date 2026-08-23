from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.llm import LLMResponse, TransientLLMError
from numerical_agent.evolution import git
from numerical_agent.evolution.repository_bootstrap import (
    BootstrapError,
    bootstrap_repository,
)
from numerical_agent.evolution.module import read_module


def method(name: str) -> str:
    return (
        f"def {name}(history, horizon, frequency):\n"
        f'    """Use when a last-value forecast is appropriate for {name}."""\n'
        "    return [float(history[-1])] * horizon\n"
    )


def definition(name: str) -> dict[str, object]:
    return {
        "name": name,
        "category": "baseline",
        "description": f"Implement {name}.",
        "assumptions": ["history is non-empty"],
        "failure_conditions": [],
    }


class SequenceLLM:
    """External-call double; repository state and validation remain real."""

    def __init__(self, outcomes: list[str | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def complete(self, *, system: str, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages, "temperature": temperature})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return LLMResponse(outcome)


def response(name: str) -> str:
    return json.dumps({"code": method(name)})


def test_full_bootstrap_creates_one_seed_commit_and_auditable_outputs(tmp_path: Path) -> None:
    repo = tmp_path / "v001"
    excluded = [{"name": "wrapped", "category": "calibration", "reason": "needs a base model"}]

    result = bootstrap_repository(
        repo,
        [definition("naive_last"), definition("naive_mean")],
        excluded,
        SequenceLLM([response("naive_last"), response("naive_mean")]),
        attempts_per_method=2,
    )

    assert result.total == 2
    assert result.succeeded == 2
    assert result.failed == 0
    assert read_module(repo / "methods.py").names() == ("naive_last", "naive_mean")
    assert (repo / ".git").is_dir()
    assert len(git(repo, "log", "--format=%H").splitlines()) == 1
    assert json.loads((repo / "bootstrap_summary.json").read_text())["succeeded"] == 2
    assert json.loads((repo / "excluded_methods.json").read_text()) == {"methods": excluded}
    assert (repo / "skills.py").is_file()
    assert "detect_periodicity" in (repo / "skills.py").read_text(encoding="utf-8")
    assert "skills.py" in git(repo, "ls-files").splitlines()
    assert (repo / "policies.py").is_file()
    assert "combined_timesfm_seasonal" in (repo / "policies.py").read_text(encoding="utf-8")
    assert "policies.py" in git(repo, "ls-files").splitlines()


def test_interrupted_bootstrap_resumes_without_regenerating_completed_methods(tmp_path: Path) -> None:
    repo = tmp_path / "v001"
    definitions = [definition("naive_last"), definition("naive_mean")]

    with pytest.raises(TransientLLMError, match="offline"):
        bootstrap_repository(
            repo,
            definitions,
            [],
            SequenceLLM([response("naive_last"), TransientLLMError("offline")]),
            attempts_per_method=2,
        )

    assert not (repo / "methods.py").exists()
    assert (repo / ".bootstrap" / "methods" / "naive_last.py").exists()

    resumed_llm = SequenceLLM([response("naive_mean")])
    result = bootstrap_repository(repo, definitions, [], resumed_llm, attempts_per_method=2)

    assert result.resumed == 1
    assert len(resumed_llm.calls) == 1
    assert read_module(repo / "methods.py").names() == ("naive_last", "naive_mean")


def test_invalid_method_receives_validation_feedback_and_is_retried(tmp_path: Path) -> None:
    repo = tmp_path / "v001"
    llm = SequenceLLM([response("wrong_name"), response("naive_last")])

    result = bootstrap_repository(
        repo,
        [definition("naive_last")],
        [],
        llm,
        attempts_per_method=2,
    )

    assert result.succeeded == 1
    second_prompt = str(llm.calls[1]["messages"][0]["content"])
    assert "wrong_name" in second_prompt
    assert "expected 'naive_last'" in second_prompt
    transcripts = sorted((repo / ".bootstrap" / "transcripts").glob("naive_last_attempt_*.json"))
    assert len(transcripts) == 2


def test_exhausted_invalid_method_is_recorded_without_aborting_other_methods(tmp_path: Path) -> None:
    repo = tmp_path / "v001"
    llm = SequenceLLM([
        response("wrong_one"),
        response("still_wrong"),
        response("naive_mean"),
    ])

    result = bootstrap_repository(
        repo,
        [definition("naive_last"), definition("naive_mean")],
        [],
        llm,
        attempts_per_method=2,
    )

    assert result.succeeded == 1
    assert result.failed == 1
    assert read_module(repo / "methods.py").names() == ("naive_mean",)
    summary = json.loads((repo / "bootstrap_summary.json").read_text())
    assert summary["failures"][0]["name"] == "naive_last"
    assert "expected 'naive_last'" in summary["failures"][0]["error"]


def test_resume_rejects_a_different_catalog_definition_set(tmp_path: Path) -> None:
    repo = tmp_path / "v001"
    with pytest.raises(TransientLLMError):
        bootstrap_repository(
            repo,
            [definition("naive_last"), definition("naive_mean")],
            [],
            SequenceLLM([response("naive_last"), TransientLLMError("offline")]),
            attempts_per_method=2,
        )

    with pytest.raises(BootstrapError, match="definition set does not match"):
        bootstrap_repository(
            repo,
            [definition("naive_last"), definition("linear_trend")],
            [],
            SequenceLLM([]),
            attempts_per_method=2,
        )


def test_completed_repository_is_never_overwritten_by_bootstrap(tmp_path: Path) -> None:
    repo = tmp_path / "v001"
    bootstrap_repository(
        repo,
        [definition("naive_last")],
        [],
        SequenceLLM([response("naive_last")]),
        attempts_per_method=1,
    )

    with pytest.raises(BootstrapError, match="already seeded"):
        bootstrap_repository(
            repo,
            [definition("naive_last")],
            [],
            SequenceLLM([]),
            attempts_per_method=1,
        )
