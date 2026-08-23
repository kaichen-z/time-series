from __future__ import annotations

from pathlib import Path

import pytest

from common.llm import FakeLLMClient
from numerical_agent.evolution.execution import (
    CRASHED,
    INVALID,
    NOT_APPLICABLE,
    SUCCESS,
    Outcome,
    Task,
)
from numerical_agent.evolution.filtering import (
    FilterDictionary,
    FilterEntry,
    FilterError,
    apply_filter_response,
    build_filter_dictionary,
    evaluate_filter,
    evolve_filter_once,
    parse_filter_source,
    render_filter_source,
    require_cached_portfolio_outcomes,
    _required_review_targets,
)
from numerical_agent.evolution.cache import OutcomeCache
from numerical_agent.evolution.module import MODULE_HEADER, parse_module
from numerical_agent.evolution.portfolio import (
    FLAGSHIP_METHOD_IDS,
    PolicyOutcomeCache,
    PolicyPortfolio,
    evaluate_portfolio,
)
from numerical_agent.dictionary import MethodCandidate
from numerical_agent.providers import RuntimeRegistry


def _tasks(prefix: str, count: int) -> tuple[Task, ...]:
    return tuple(
        Task(
            f"{prefix}{index}",
            (
                tuple(0.0 if value % 2 else float(value) for value in range(1, 29))
                if index % 2
                else tuple(float(value + index) for value in range(1, 29))
            ),
            2,
            "1 day",
            (30.0 + index, 31.0 + index),
        )
        for index in range(count)
    )


def _outcomes(tasks: tuple[Task, ...]) -> tuple[Outcome, ...]:
    rows = []
    for task in tasks:
        index = int(task.task_id[-1])
        rows.extend(
            (
                Outcome("stable", task.task_id, SUCCESS, mase=1.0, mae=1.0, smape=1.0),
                Outcome("weak", task.task_id, SUCCESS, mase=5.0, mae=5.0, smape=5.0),
                Outcome(
                    "intermittent",
                    task.task_id,
                    SUCCESS if index % 2 else NOT_APPLICABLE,
                    mase=0.5 if index % 2 else None,
                    mae=0.5 if index % 2 else None,
                    smape=0.5 if index % 2 else None,
                ),
            )
        )
    return tuple(rows)


def _dictionary() -> FilterDictionary:
    return FilterDictionary(
        (
            FilterEntry("stable", "statistical", "keep", (), "broad baseline"),
            FilterEntry("weak", "statistical", "keep", (), "weak baseline"),
            FilterEntry("intermittent", "statistical", "keep", (), "specialist"),
        )
    )


class _Runtime:
    def supports(self, candidate: MethodCandidate) -> bool:
        return candidate.method_id in FLAGSHIP_METHOD_IDS

    def forecast(self, candidate, history, horizon, frequency):
        return tuple(float(history[-1]) for _ in range(horizon))


def _portfolio_module():
    names = (
        "seasonal_naive", "holt_damped_trend", "croston_sba",
        "robust_loess_trend", "median_seasonal_profile_forecast",
    )
    source = "\n\n".join(
        f'''def {name}(history, horizon, frequency):
    """Use as a filtering integration fixture."""
    return [float(history[-1])] * horizon
'''
        for name in names
    )
    return parse_module(MODULE_HEADER + "\n\n" + source)


def test_filter_source_is_executable_python_literal_and_round_trips() -> None:
    dictionary = _dictionary()

    assert parse_filter_source(render_filter_source(dictionary)) == dictionary


def test_unified_python_dictionary_indexes_statistical_tsfm_and_combined() -> None:
    dictionary = build_filter_dictionary(_portfolio_module(), PolicyPortfolio.flagship5())

    assert len(dictionary.entries) == 15
    assert {entry.family for entry in dictionary.entries} == {
        "statistical", "tsfm", "combined"
    }


def test_cached_portfolio_loader_reuses_all_three_candidate_families(tmp_path: Path) -> None:
    module = _portfolio_module()
    portfolio = PolicyPortfolio.flagship5()
    task = _tasks("train", 1)[0]
    method_cache = OutcomeCache(tmp_path / "methods")
    policy_cache = PolicyOutcomeCache(tmp_path / "policies")
    runtime = _Runtime()
    registry = RuntimeRegistry(
        {"timesfm": runtime, "chronos": runtime, "tsfm_worker": runtime}
    )
    expected = evaluate_portfolio(
        module, portfolio, (task,), outcome_cache=method_cache,
        policy_cache=policy_cache, runtimes=registry, isolated_methods=False,
    )

    actual = require_cached_portfolio_outcomes(
        module, portfolio, (task,), outcome_cache=method_cache,
        policy_cache=policy_cache, isolated_methods=False,
    )

    assert actual == expected
    assert len(actual) == 15


def test_specialized_requires_a_history_only_applicability_tag() -> None:
    with pytest.raises(FilterError, match="specialized.*applicability"):
        FilterEntry("croston", "statistical", "specialized", (), "specialist")


def test_keep_method_may_declare_a_restricted_history_only_applicability() -> None:
    entry = FilterEntry(
        "inar", "statistical", "keep", ("intermittent",), "reliable count model"
    )

    assert entry.status == "keep"
    assert entry.applicability == ("intermittent",)


def test_applicability_rejects_mutually_exclusive_history_tags() -> None:
    with pytest.raises(FilterError, match="mutually exclusive.*no_zeros.*some_zeros"):
        FilterEntry(
            "croston", "statistical", "specialized",
            ("no_zeros", "some_zeros"), "contradictory gate",
        )


def test_agent_cannot_discard_without_strict_trusted_evidence() -> None:
    response = '''{
      "summary": "remove weak methods",
      "actions": [{
        "name": "weak", "status": "discard", "applicability": [], "reason": "weak"
      }]
    }'''

    with pytest.raises(FilterError, match="discard requires trusted dominance evidence"):
        apply_filter_response(_dictionary(), response, discardable=frozenset())


def test_agent_must_address_a_repeated_downstream_selection_failure() -> None:
    response = '''{
      "summary": "change an unrelated method",
      "actions": [{
        "name": "weak", "status": "repair", "applicability": [], "reason": "weak"
      }]
    }'''

    with pytest.raises(FilterError, match="must address.*intermittent"):
        apply_filter_response(
            _dictionary(), response, discardable=frozenset(),
            required_names=frozenset({"intermittent"}),
        )


def test_agent_must_review_a_method_with_partial_not_applicable_behavior(
    tmp_path: Path,
) -> None:
    train = _tasks("train", 4)
    dev = _tasks("dev", 2)
    outcomes = tuple(
        Outcome(row.method, row.task_id, NOT_APPLICABLE)
        if row.method == "weak" and int(row.task_id[-1]) % 2 == 0
        else row
        for row in _outcomes(train + dev)
    )
    agent = FakeLLMClient(
        ['''{
          "summary": "only address the selected method",
          "actions": [{
            "name": "intermittent", "status": "specialized",
            "applicability": ["intermittent"], "reason": "specialist"
          }]
        }''']
    )

    with pytest.raises(FilterError, match="must address.*weak"):
        evolve_filter_once(
            _dictionary(), train, dev, outcomes, agent,
            generation=1, transcript_dir=tmp_path,
        )


def test_history_feature_selector_reward_changes_when_bad_candidate_is_filtered() -> None:
    train = _tasks("train", 4)
    outcomes = _outcomes(train)
    parent = evaluate_filter(_dictionary(), outcomes, train, reference_outcomes=outcomes)
    child_dictionary = FilterDictionary(
        tuple(
            FilterEntry(entry.name, entry.family, "repair", (), "filtered")
            if entry.name == "intermittent"
            else entry
            for entry in _dictionary().entries
        )
    )
    child = evaluate_filter(child_dictionary, outcomes, train, reference_outcomes=outcomes)

    assert child.coverage == 1.0
    assert child.mean_mase < parent.mean_mase


def test_applicability_is_checked_before_a_keep_method_is_ranked() -> None:
    train = _tasks("train", 4)
    outcomes = _outcomes(train)
    parent = evaluate_filter(_dictionary(), outcomes, train, reference_outcomes=outcomes)
    child_dictionary = FilterDictionary(
        tuple(
            FilterEntry(entry.name, entry.family, "keep", ("intermittent",), "count only")
            if entry.name == "intermittent"
            else entry
            for entry in _dictionary().entries
        )
    )

    child = evaluate_filter(child_dictionary, outcomes, train, reference_outcomes=outcomes)

    assert child.coverage == 1.0
    assert child.mean_mase < parent.mean_mase
    assert child.eligible_counts == {
        "train0": 2,
        "train1": 3,
        "train2": 2,
        "train3": 3,
    }


def test_single_agent_child_is_accepted_only_after_train_and_dev_improve(
    tmp_path: Path,
) -> None:
    train = _tasks("train", 4)
    dev = _tasks("dev", 2)
    all_outcomes = _outcomes(train + dev)
    agent = FakeLLMClient(
        ['''{
          "summary": "exclude an unreliable general candidate",
          "actions": [{
            "name": "intermittent",
            "status": "specialized",
            "applicability": ["intermittent"],
            "reason": "Only expose it on histories matching its observed strength."
          }]
        }''']
    )

    result = evolve_filter_once(
        _dictionary(),
        train,
        dev,
        all_outcomes,
        agent,
        generation=1,
        transcript_dir=tmp_path,
    )

    assert result.agent_calls == 1
    assert result.child.entries != result.parent.entries
    assert result.train_child.mean_mase <= result.train_parent.mean_mase
    assert result.dev_child.mean_mase <= result.dev_parent.mean_mase
    assert result.accepted
    assert (tmp_path / "generation_001_filter_request.txt").is_file()
    assert (tmp_path / "generation_001_filter_response.json").is_file()


def test_reliability_improvement_accepts_filter_when_forecast_is_unchanged(
    tmp_path: Path,
) -> None:
    train = _tasks("train", 4)
    dev = _tasks("dev", 2)
    dictionary = FilterDictionary(
        (
            FilterEntry("stable", "statistical", "keep", (), "broad baseline"),
            FilterEntry("broken", "statistical", "keep", (), "unreviewed"),
            FilterEntry("specialist", "statistical", "keep", (), "not applicable here"),
        )
    )
    outcomes = tuple(
        row
        for task in train + dev
        for row in (
            Outcome("stable", task.task_id, SUCCESS, mase=1.0, mae=1.0, smape=1.0),
            Outcome("broken", task.task_id, CRASHED, detail="implementation failure"),
            Outcome("specialist", task.task_id, NOT_APPLICABLE),
        )
    )
    agent = FakeLLMClient(
        ['''{
          "summary": "quarantine an implementation that always crashes",
          "actions": [{
            "name": "broken", "status": "quarantine", "applicability": [],
            "reason": "Every executable attempt crashes."
          }]
        }''']
    )

    result = evolve_filter_once(
        dictionary, train, dev, outcomes, agent,
        generation=1, transcript_dir=tmp_path,
        required_targets=("broken",),
    )

    assert result.train_child.mean_mase == result.train_parent.mean_mase
    assert result.dev_child.mean_mase == result.dev_parent.mean_mase
    assert result.train_child.eligible_success_rate > result.train_parent.eligible_success_rate
    assert result.train_child.eligible_failure_rate < result.train_parent.eligible_failure_rate
    assert (
        result.train_child.eligible_not_applicable_rate
        > result.train_parent.eligible_not_applicable_rate
    )
    assert result.accepted
    assert "reliability" in result.reason.lower()


def test_filter_rejects_removing_healthy_candidate_without_any_quality_gain(
    tmp_path: Path,
) -> None:
    train = _tasks("train", 4)
    dev = _tasks("dev", 2)
    dictionary = FilterDictionary(
        (
            FilterEntry("stable", "statistical", "keep", (), "best baseline"),
            FilterEntry("healthy", "statistical", "keep", (), "valid challenger"),
        )
    )
    outcomes = tuple(
        row
        for task in train + dev
        for row in (
            Outcome("stable", task.task_id, SUCCESS, mase=1.0, mae=1.0, smape=1.0),
            Outcome("healthy", task.task_id, SUCCESS, mase=2.0, mae=2.0, smape=2.0),
        )
    )
    agent = FakeLLMClient(
        ['''{
          "summary": "remove a healthy but weaker challenger",
          "actions": [{
            "name": "healthy", "status": "repair", "applicability": [],
            "reason": "It is weaker than the selected method."
          }]
        }''']
    )

    result = evolve_filter_once(
        dictionary, train, dev, outcomes, agent,
        generation=1, transcript_dir=tmp_path,
        required_targets=("healthy",),
    )

    assert result.train_child.mean_mase == result.train_parent.mean_mase
    assert result.train_child.eligible_success_rate == result.train_parent.eligible_success_rate
    assert not result.accepted


def test_dev_failure_rejects_child_without_showing_dev_metrics_to_agent(tmp_path: Path) -> None:
    train = _tasks("train", 4)
    dev = _tasks("dev", 2)
    outcomes = list(_outcomes(train + dev))
    outcomes = [
        Outcome(row.method, row.task_id, INVALID, detail="dev failure")
        if row.method == "stable" and row.task_id.startswith("dev")
        else row
        for row in outcomes
    ]
    agent = FakeLLMClient(
        ['''{
          "summary": "remove the specialist",
          "actions": [{
            "name": "intermittent", "status": "repair", "applicability": [],
            "reason": "It is not reliable as a general method."
          }]
        }''']
    )

    result = evolve_filter_once(
        _dictionary(), train, dev, tuple(outcomes), agent,
        generation=1, transcript_dir=tmp_path,
    )

    request = agent.calls[0]["messages"][0]["content"]
    import json
    payload = json.loads(request)
    assert '"selection_failures"' in request
    assert '"selected": "intermittent"' in request
    assert payload["required_targets"] == ["intermittent"]
    conditional = payload["required_target_evidence"]["intermittent"]
    assert conditional["intermittent"] == {
        "tasks": 2,
        "success": 2,
        "not_applicable": 0,
        "crashed": 0,
        "invalid": 0,
        "mean_mase": 0.5,
    }
    assert conditional["dense"]["not_applicable"] == 2
    assert "dev0" not in request
    assert "dev1" not in request
    assert not result.accepted


def test_required_review_targets_are_prioritized_into_four_disjoint_batches() -> None:
    selected_names = [f"selected_{index:02d}" for index in range(5)]
    defective_names = [f"defective_{index:02d}" for index in range(30)]
    specialist_names = [f"specialist_{index:02d}" for index in range(42)]
    reports = [
        {
            "name": name,
            "success": 80,
            "not_applicable": 0,
            "crashed": 0,
            "invalid": 0,
            "mean_mase": 10.0,
        }
        for name in selected_names
    ]
    reports.extend(
        {
            "name": name,
            "success": 70,
            "not_applicable": 0,
            "crashed": 10,
            "invalid": 0,
            "mean_mase": 5.0,
        }
        for name in defective_names
    )
    reports.extend(
        {
            "name": name,
            "success": 40,
            "not_applicable": 40,
            "crashed": 0,
            "invalid": 0,
            "mean_mase": 1.0,
        }
        for name in specialist_names
    )
    failures = tuple(
        {"selected": name}
        for name in selected_names
        for _ in range(2)
    )

    reviewed: set[str] = set()
    batches = []
    for _ in range(4):
        batch = _required_review_targets(
            reports,
            failures,
            reviewed_names=frozenset(reviewed),
            limit=24,
        )
        batches.append(batch)
        assert reviewed.isdisjoint(batch)
        reviewed.update(batch)

    assert [len(batch) for batch in batches] == [24, 24, 24, 5]
    assert len(reviewed) == 77
    assert set(selected_names).issubset(batches[0])
    assert not set(specialist_names).intersection(batches[0])


def test_filter_generation_records_only_the_current_unreviewed_batch(
    tmp_path: Path,
) -> None:
    train = _tasks("train", 4)
    dev = _tasks("dev", 2)
    outcomes = _outcomes(train + dev)
    agent = FakeLLMClient(
        ['''{
          "summary": "review the remaining weak specialist",
          "actions": [{
            "name": "intermittent", "status": "specialized",
            "applicability": ["intermittent"], "reason": "specialist"
          }]
        }''']
    )

    result = evolve_filter_once(
        _dictionary(),
        train,
        dev,
        outcomes,
        agent,
        generation=2,
        transcript_dir=tmp_path,
        reviewed_names=frozenset({"weak"}),
        required_target_limit=24,
    )

    assert result.required_targets == ("intermittent",)


def test_filter_generation_uses_a_frozen_priority_batch_across_parent_changes(
    tmp_path: Path,
) -> None:
    train = _tasks("train", 4)
    dev = _tasks("dev", 2)
    outcomes = _outcomes(train + dev)
    agent = FakeLLMClient(
        ['''{
          "summary": "review exactly the scheduled candidate",
          "actions": [{
            "name": "weak", "status": "quarantine",
            "applicability": [], "reason": "scheduled batch"
          }]
        }''']
    )

    result = evolve_filter_once(
        _dictionary(),
        train,
        dev,
        outcomes,
        agent,
        generation=3,
        transcript_dir=tmp_path,
        required_targets=("weak",),
    )

    assert result.required_targets == ("weak",)
    request = agent.calls[0]["messages"][0]["content"]
    assert '"required_targets": ["weak"]' in request
