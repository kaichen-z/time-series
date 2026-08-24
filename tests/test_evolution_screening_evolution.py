from __future__ import annotations

import json

import pytest

from common.llm import FakeLLMClient
from numerical_agent.evolution.execution import CRASHED, SUCCESS, Outcome, Task
from numerical_agent.evolution.filtering import FilterDictionary, FilterEntry
from numerical_agent.evolution.screening import (
    ApplicabilityClause,
    ApplicabilityPolicy,
    FeatureTest,
    ScreeningEntry,
    ScreeningPolicy,
    ScreeningConstraints,
    profile_task,
)
from numerical_agent.evolution.screening_evolution import (
    ScreeningEvolutionError,
    apply_screening_response,
    build_train_evidence,
    complete_target_batches,
    evolve_screening_once,
    protect_train_oracles,
    select_refinement_targets,
    migrate_filter_dictionary,
    parse_screening_source,
    render_screening_source,
)


def _legacy() -> FilterDictionary:
    return FilterDictionary(
        (
            FilterEntry("stable", "statistical", "keep", (), "stable"),
            FilterEntry("special", "statistical", "specialized", ("intermittent",), "special"),
            FilterEntry("broken", "combined", "keep", (), "broken"),
            FilterEntry("timesfm", "tsfm", "keep", (), "tsfm"),
        )
    )


def test_migration_preserves_identity_status_and_legacy_and_rules() -> None:
    policy = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )

    assert [entry.name for entry in policy.entries] == ["stable", "special", "broken", "timesfm"]
    assert policy.get("special").status == "specialized"  # type: ignore[union-attr]
    assert policy.get("special").applicability == ApplicabilityPolicy(  # type: ignore[union-attr]
        (ApplicabilityClause(("intermittent",)),)
    )
    assert parse_screening_source(render_screening_source(policy)) == policy


def test_response_can_only_change_known_status_and_typed_applicability() -> None:
    parent = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )
    response = json.dumps(
        {
            "summary": "specialize the stable method to a periodic or intermittent regime",
            "actions": [
                {
                    "name": "stable",
                    "status": "specialized",
                    "any_of": [
                        {
                            "all_tags": ["intermittent"],
                            "feature_tests": [],
                        },
                        {
                            "all_tags": [],
                            "feature_tests": [
                                {
                                    "field": "periodicity_strength",
                                    "operator": ">=",
                                    "value": 0.7,
                                }
                            ],
                        },
                    ],
                    "reason": "two supported regimes",
                }
            ],
        }
    )

    child = apply_screening_response(parent, response, required_names=frozenset({"stable"}))

    assert child.get("stable") == ScreeningEntry(
        "stable",
        "statistical",
        "specialized",
        ApplicabilityPolicy(
            (
                ApplicabilityClause(("intermittent",)),
                ApplicabilityClause(
                    feature_tests=(FeatureTest("periodicity_strength", ">=", 0.7),)
                ),
            )
        ),
        "two supported regimes",
    )


def test_response_normalizes_only_omitted_empty_clause_lists() -> None:
    parent = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )
    response = json.dumps(
        {
            "summary": "specialize with compact but unambiguous clauses",
            "actions": [
                {
                    "name": "stable",
                    "status": "specialized",
                    "any_of": [
                        {"all_tags": ["periodicity:strong"]},
                        {
                            "feature_tests": [
                                {
                                    "field": "trend_strength",
                                    "operator": ">=",
                                    "value": 0.7,
                                }
                            ]
                        },
                    ],
                    "reason": "two supported regimes",
                }
            ],
        }
    )

    child = apply_screening_response(
        parent, response, required_names=frozenset({"stable"})
    )

    assert child.get("stable").applicability == ApplicabilityPolicy(  # type: ignore[union-attr]
        (
            ApplicabilityClause(("periodicity:strong",)),
            ApplicabilityClause(
                feature_tests=(FeatureTest("trend_strength", ">=", 0.7),)
            ),
        )
    )

    invalid = json.dumps(
        {
            "summary": "unknown clause key",
            "actions": [
                {
                    "name": "stable",
                    "status": "specialized",
                    "any_of": [{"all_tags": ["dense"], "query": "future"}],
                    "reason": "invalid",
                }
            ],
        }
    )
    with pytest.raises(ScreeningEvolutionError, match="invalid applicability clause"):
        apply_screening_response(parent, invalid, required_names=frozenset({"stable"}))


def test_response_rejects_unknown_feature_and_missing_required_target() -> None:
    parent = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )
    response = json.dumps(
        {
            "summary": "invalid",
            "actions": [
                {
                    "name": "stable",
                    "status": "specialized",
                    "any_of": [{"all_tags": [], "feature_tests": [
                        {"field": "future", "operator": ">", "value": 0}
                    ]}],
                    "reason": "future leak",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="unsupported profile field"):
        apply_screening_response(parent, response, required_names=frozenset({"stable"}))
    with pytest.raises(ScreeningEvolutionError, match="must address required targets"):
        apply_screening_response(
            parent,
            json.dumps({"summary": "empty", "actions": []}),
            required_names=frozenset({"broken"}),
        )
    with pytest.raises(ScreeningEvolutionError, match="unexpected targets"):
        apply_screening_response(
            parent,
            json.dumps(
                {
                    "summary": "extra mutation",
                    "actions": [
                        {
                            "name": "stable",
                            "status": "keep",
                            "any_of": [],
                            "reason": "required",
                        },
                        {
                            "name": "broken",
                            "status": "repair",
                            "any_of": [],
                            "reason": "not requested",
                        },
                    ],
                }
            ),
            required_names=frozenset({"stable"}),
        )


def test_train_evidence_learns_grouped_performance_relative_to_baseline() -> None:
    tasks = (
        Task("secret-periodic-task", tuple([1.0, 3.0, 2.0] * 12), 3, "1 day", (1.0,) * 3),
        Task("trend", tuple(float(i) for i in range(36)), 3, "1 day", (1.0,) * 3),
    )
    outcomes = (
        Outcome("candidate", "secret-periodic-task", SUCCESS, mase=0.5),
        Outcome("candidate", "trend", SUCCESS, mase=2.0),
        Outcome("baseline", "secret-periodic-task", SUCCESS, mase=1.0),
        Outcome("baseline", "trend", SUCCESS, mase=1.0),
    )
    policy = ScreeningPolicy(
        (
            ScreeningEntry(
                "candidate", "combined", "keep", ApplicabilityPolicy(()), "candidate"
            ),
            ScreeningEntry(
                "baseline", "tsfm", "keep", ApplicabilityPolicy(()), "baseline"
            ),
        ),
        ("candidate", "baseline"),
    )

    evidence = build_train_evidence(
        frozenset({"candidate"}),
        tasks,
        outcomes,
        policy=policy,
        baseline_method="baseline",
        min_group_support=1,
    )[0]

    assert evidence["family"] == "combined"
    assert evidence["relative_to_baseline"] == {
        "comparable": 2,
        "wins": 1,
        "ties": 0,
        "losses": 1,
        "win_rate": 0.5,
        "mean_relative_mase": 0.125,
    }
    groups = {row["group"]: row for row in evidence["groups"]}
    assert groups["periodicity:strong"]["wins"] == 1
    assert groups["trend:strong_up"]["losses"] == 1
    assert all("task_id" not in row for row in evidence["groups"])
    assert len(evidence["oracle_profiles"]) == 1
    assert "periodicity:strong" in evidence["oracle_profiles"][0]
    assert "secret-periodic-task" not in repr(evidence["oracle_profiles"])


def test_train_oracle_shield_adds_history_only_clause_without_exposing_ids() -> None:
    tasks = (
        Task("dense-id", (1.0,) * 12, 2, "1 day", (1.0, 1.0)),
        Task("intermittent-id", (0.0, 0.0, 4.0) * 4, 2, "1 day", (4.0, 0.0)),
    )
    outcomes = tuple(
        row
        for task in tasks
        for row in (
            Outcome(
                "candidate",
                task.task_id,
                SUCCESS,
                mase=0.5 if task.task_id == "intermittent-id" else 2.0,
                ),
                Outcome("baseline", task.task_id, SUCCESS, mase=1.0),
                Outcome("stable_one", task.task_id, SUCCESS, mase=3.0),
                Outcome("stable_two", task.task_id, SUCCESS, mase=3.0),
        )
    )
    parent = ScreeningPolicy(
        (
            ScreeningEntry(
                "candidate", "statistical", "keep", ApplicabilityPolicy(()), "candidate"
            ),
            ScreeningEntry(
                "baseline", "tsfm", "keep", ApplicabilityPolicy(()), "baseline"
            ),
            ScreeningEntry(
                "stable_one", "statistical", "keep", ApplicabilityPolicy(()), "stable"
            ),
            ScreeningEntry(
                "stable_two", "statistical", "keep", ApplicabilityPolicy(()), "stable"
            ),
        ),
        ("stable_one", "baseline", "stable_two"),
    )
    child = ScreeningPolicy(
        (
            ScreeningEntry(
                "candidate",
                "statistical",
                "specialized",
                ApplicabilityPolicy((ApplicabilityClause(("dense",)),)),
                "over-specialized",
            ),
            parent.get("baseline"),
            parent.get("stable_one"),
            parent.get("stable_two"),
        ),
        parent.fallback_names,
    )

    protected, shields = protect_train_oracles(child, tasks, outcomes)

    protected_candidate = protected.get("candidate")
    assert protected_candidate is not None
    assert protected_candidate.applicability.match(profile_task(tasks[1])) is not None
    assert shields[0].method == "candidate"
    assert "intermittency:intermittent" in shields[0].profile_tags
    assert "intermittent-id" not in repr(shields)


def test_target_batches_review_every_candidate_once() -> None:
    policy = ScreeningPolicy(
        tuple(
            ScreeningEntry(name, family, "keep", ApplicabilityPolicy(()), name)
            for name, family in (
                ("a", "statistical"),
                ("b", "tsfm"),
                ("c", "combined"),
                ("d", "statistical"),
            )
        ),
        ("a", "b", "c"),
    )

    batches = complete_target_batches(policy, (("c", "a", "b"),), batch_size=2)

    assert batches == (("c", "a"), ("b", "d"))


def test_refinement_targets_prioritize_missing_family_and_unsafe_broad_methods() -> None:
    tasks = (
        Task("one", (1.0,) * 12, 2, "1 day", (1.0, 1.0)),
        Task("two", tuple(float(i) for i in range(12)), 2, "1 day", (1.0, 1.0)),
    )
    policy = ScreeningPolicy(
        (
            ScreeningEntry("baseline", "tsfm", "keep", ApplicabilityPolicy(()), "base"),
            ScreeningEntry("worker", "tsfm", "keep", ApplicabilityPolicy(()), "worker"),
            ScreeningEntry("bad", "statistical", "keep", ApplicabilityPolicy(()), "bad"),
            ScreeningEntry("good", "statistical", "keep", ApplicabilityPolicy(()), "good"),
            ScreeningEntry("blend", "combined", "keep", ApplicabilityPolicy(()), "blend"),
        ),
        ("good", "baseline", "worker"),
    )
    outcomes = tuple(
        row
        for task in tasks
        for row in (
            Outcome("baseline", task.task_id, SUCCESS, mase=1.0),
            Outcome("worker", task.task_id, SUCCESS, mase=0.8),
            Outcome("bad", task.task_id, CRASHED),
            Outcome("good", task.task_id, SUCCESS, mase=0.5),
            Outcome("blend", task.task_id, SUCCESS, mase=0.7),
        )
    )
    constraints = ScreeningConstraints(
        baseline_method="baseline",
        min_active_candidates=1,
        max_active_candidates=3,
        min_unique_active_dictionaries=2,
        max_mean_pairwise_jaccard=0.99,
        min_group_support=1,
    )

    targets = select_refinement_targets(
        policy,
        tasks,
        outcomes,
        constraints=constraints,
        excluded_names=frozenset({"blend"}),
        required_families=("tsfm",),
        limit=3,
    )

    assert "worker" in targets  # TSFM lacks a conditioned entry.
    assert "bad" in targets  # Crashing broad method is a high-priority refinement.
    assert "baseline" not in targets
    assert "blend" not in targets


def test_refinement_can_revisit_a_specialized_rule_that_did_not_generalize() -> None:
    tasks = (Task("one", (1.0,) * 12, 2, "1 day", (1.0, 1.0)),)
    impossible = ApplicabilityPolicy((ApplicabilityClause(("signed",)),))
    policy = ScreeningPolicy(
        (
            ScreeningEntry("baseline", "tsfm", "keep", ApplicabilityPolicy(()), "base"),
            ScreeningEntry("stable", "statistical", "keep", ApplicabilityPolicy(()), "safe"),
            ScreeningEntry("blend", "combined", "specialized", impossible, "too narrow"),
        ),
        ("stable", "baseline", "blend"),
    )
    outcomes = (
        Outcome("baseline", "one", SUCCESS, mase=1.0),
        Outcome("stable", "one", SUCCESS, mase=0.8),
        Outcome("blend", "one", SUCCESS, mase=0.7),
    )
    constraints = ScreeningConstraints(
        baseline_method="baseline",
        min_active_candidates=1,
        max_active_candidates=3,
        min_unique_active_dictionaries=1,
        max_mean_pairwise_jaccard=1.0,
        min_group_support=1,
    )

    targets = select_refinement_targets(
        policy,
        tasks,
        outcomes,
        constraints=constraints,
        required_families=("combined",),
        limit=2,
    )

    assert targets[0] == "blend"


def _tasks(prefix: str) -> tuple[Task, ...]:
    return tuple(
        Task(f"{prefix}-{index}", (1.0,) * 12, 2, "1 day", (1.0, 1.0))
        for index in range(2)
    )


def _outcomes(tasks: tuple[Task, ...]) -> tuple[Outcome, ...]:
    return tuple(
        row
        for task in tasks
        for row in (
            Outcome("stable", task.task_id, SUCCESS, mase=1.0, mae=1.0, smape=1.0),
            Outcome("special", task.task_id, SUCCESS, mase=0.8, mae=0.8, smape=0.8),
            Outcome("broken", task.task_id, CRASHED),
            Outcome("timesfm", task.task_id, SUCCESS, mase=1.2, mae=1.2, smape=1.2),
        )
    )


def test_evolution_prompt_is_train_only_and_accepts_reliability_improvement(tmp_path) -> None:
    parent = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )
    train, dev = _tasks("train"), _tasks("dev")
    agent = FakeLLMClient(
        [json.dumps({
            "summary": "remove a crashing candidate from selection",
            "actions": [{
                "name": "broken", "status": "repair", "any_of": [],
                "reason": "crashes on every Train task",
            }],
        })]
    )

    result = evolve_screening_once(
        parent,
        train,
        dev,
        _outcomes(train + dev),
        agent,
        generation=1,
        required_targets=("broken",),
        transcript_dir=tmp_path,
        constraints=ScreeningConstraints(
            baseline_method="timesfm",
            min_active_candidates=1,
            max_active_candidates=4,
            min_unique_active_dictionaries=1,
            max_mean_pairwise_jaccard=1.0,
            min_group_support=1,
            required_conditioned_families=(),
        ),
    )

    request = agent.calls[0]["messages"][0]["content"]
    assert "train-0" not in request
    assert "dev-0" not in request
    assert "future" not in request.lower()
    payload = json.loads(request)
    assert payload["baseline_method"] == "timesfm"
    assert payload["constraints"]["max_active_candidates"] == 4
    assert "periodicity:none" in payload["allowed_profile_tags"]
    assert payload["train_evidence"][0]["family"] == "combined"
    assert "relative_to_baseline" in payload["train_evidence"][0]
    assert result.accepted
    assert result.child.get("broken").status == "repair"  # type: ignore[union-attr]
    assert result.gate.improved_dimensions == ("active_success_rate", "failure_exposure")


def test_evolution_repairs_one_malformed_response_without_guessing_actions(tmp_path) -> None:
    parent = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )
    train, dev = _tasks("train"), _tasks("dev")
    incomplete = json.dumps(
        {
            "summary": "forgot one required target",
            "actions": [
                {
                    "name": "stable",
                    "status": "keep",
                    "any_of": [],
                    "reason": "stable",
                }
            ],
        }
    )
    repaired = json.dumps(
        {
            "summary": "complete replacement",
            "actions": [
                {
                    "name": "stable",
                    "status": "keep",
                    "any_of": [],
                    "reason": "stable",
                },
                {
                    "name": "broken",
                    "status": "repair",
                    "any_of": [],
                    "reason": "crashes",
                },
            ],
        }
    )
    agent = FakeLLMClient([incomplete, repaired])

    result = evolve_screening_once(
        parent,
        train,
        dev,
        _outcomes(train + dev),
        agent,
        generation=1,
        required_targets=("stable", "broken"),
        transcript_dir=tmp_path,
    )

    assert result.accepted
    assert result.agent_calls == 2
    assert result.child.get("broken").status == "repair"  # type: ignore[union-attr]
    assert "must address required targets" in agent.calls[1]["messages"][-1]["content"]


def test_refinement_retries_when_required_family_is_left_broad(tmp_path) -> None:
    parent = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )
    train, dev = _tasks("train"), _tasks("dev")
    broad = json.dumps(
        {
            "summary": "left broad",
            "actions": [{
                "name": "timesfm", "status": "keep", "any_of": [],
                "reason": "broad",
            }],
        }
    )
    conditioned = json.dumps(
        {
            "summary": "conditioned",
            "actions": [{
                "name": "timesfm", "status": "specialized",
                "any_of": [{"all_tags": ["dense"], "feature_tests": []}],
                "reason": "dense history",
            }],
        }
    )
    agent = FakeLLMClient([broad, conditioned])

    result = evolve_screening_once(
        parent,
        train,
        dev,
        _outcomes(train + dev),
        agent,
        generation=1,
        required_targets=("timesfm",),
        transcript_dir=tmp_path,
        constraints=ScreeningConstraints(
            baseline_method="timesfm",
            min_active_candidates=1,
            max_active_candidates=4,
            min_unique_active_dictionaries=1,
            max_mean_pairwise_jaccard=1.0,
            min_group_support=1,
            required_conditioned_families=(),
        ),
        required_conditioning_families=("tsfm",),
    )

    assert result.agent_calls == 2
    assert result.child.get("timesfm").status == "specialized"  # type: ignore[union-attr]
    repair = (tmp_path / "generation_001_screening_repair_request.txt").read_text()
    assert "tsfm" in repair


def test_evolution_rejects_generation_after_two_malformed_responses(tmp_path) -> None:
    parent = migrate_filter_dictionary(
        _legacy(), fallback_names=("stable", "timesfm", "special")
    )
    train, dev = _tasks("train"), _tasks("dev")
    malformed = json.dumps({"summary": "missing", "actions": []})
    agent = FakeLLMClient([malformed, malformed])

    result = evolve_screening_once(
        parent,
        train,
        dev,
        _outcomes(train + dev),
        agent,
        generation=1,
        required_targets=("broken",),
        transcript_dir=tmp_path,
    )

    assert not result.accepted
    assert result.child == parent
    assert result.agent_calls == 2
    assert "invalid Agent response" in result.gate.reason


def test_rejected_batch_salvages_safe_train_generated_actions_individually(tmp_path) -> None:
    parent = ScreeningPolicy(
        (
            ScreeningEntry("stable", "statistical", "keep", ApplicabilityPolicy(()), "stable"),
            ScreeningEntry("aux", "statistical", "keep", ApplicabilityPolicy(()), "aux"),
            ScreeningEntry("oracle", "statistical", "keep", ApplicabilityPolicy(()), "oracle"),
            ScreeningEntry("broken", "combined", "keep", ApplicabilityPolicy(()), "broken"),
            ScreeningEntry("timesfm", "tsfm", "keep", ApplicabilityPolicy(()), "timesfm"),
        ),
        ("stable", "timesfm", "aux"),
    )
    train = (Task("train", (0.0, 0.0, 4.0) * 4, 2, "1 day", (1.0, 1.0)),)
    dev = (Task("dev", (1.0,) * 12, 2, "1 day", (1.0, 1.0)),)
    outcomes = tuple(
        row
        for task in train + dev
        for row in (
            Outcome("stable", task.task_id, SUCCESS, mase=1.0),
            Outcome("aux", task.task_id, SUCCESS, mase=1.1),
            Outcome("oracle", task.task_id, SUCCESS, mase=0.5),
            Outcome("broken", task.task_id, CRASHED),
            Outcome("timesfm", task.task_id, SUCCESS, mase=1.2),
        )
    )
    response = json.dumps(
        {
            "summary": "one safe and one over-specialized action",
            "actions": [
                {
                    "name": "broken",
                    "status": "repair",
                    "any_of": [],
                    "reason": "crashes",
                },
                {
                    "name": "oracle",
                    "status": "specialized",
                    "any_of": [{"all_tags": ["intermittency:intermittent"]}],
                    "reason": "Train specialist",
                },
            ],
        }
    )

    result = evolve_screening_once(
        parent,
        train,
        dev,
        outcomes,
        FakeLLMClient([response]),
        generation=1,
        required_targets=("broken", "oracle"),
        transcript_dir=tmp_path,
    )

    assert result.accepted
    assert result.child.get("broken").status == "repair"  # type: ignore[union-attr]
    assert result.child.get("oracle") == parent.get("oracle")
    assert [(row.name, row.accepted) for row in result.action_decisions] == [
        ("broken", True),
        ("oracle", False),
    ]
