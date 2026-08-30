from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT
from common.llm import FakeLLMClient
from numerical_agent.evolution import (
    _scaled_validation_accepts,
    _validate_candidate,
    bootstrap,
    commit_module,
    evolve_once,
    git,
    init_repo,
    run_evolution,
)
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.module import MODULE_HEADER, read_module, write_module, parse_module
from numerical_agent.evolution.prompts import (
    EVOLVE_SYSTEM,
    MUTATE_SYSTEM,
    SELECT_SYSTEM,
    render_evolve_user,
    render_select_user,
)
from numerical_agent.run_evolution import (
    _evolution_tasks,
    _llm_clients,
    _validate_judge_halving_configuration,
    build_parser,
)



SARIMA_EXAMPLE = '''def sarima_auto(history, horizon, frequency):
    """Use for seasonal SARIMA selection."""
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


def test_active_batch_prompts_ignore_legacy_metric_variation() -> None:
    active = {
        "method": "m",
        "mean_smae": 0.8,
        "mean_srmse": 0.9,
        "mean_mase": 1.0,
        "mean_smape": 10.0,
        "mean_mae": 2.0,
        "success": 1,
        "total": 1,
        "coverage": 1.0,
        "not_applicable": 0,
        "crashed": 0,
        "invalid": 0,
    }
    legacy_varied = {
        **active,
        "mean_mase": 999.0,
        "mean_smape": 199.0,
        "mean_mae": 999.0,
    }

    first = render_select_user(
        reports=[active], method_inventory=[], generation=1, task_count=1
    )
    second = render_select_user(
        reports=[legacy_varied], method_inventory=[], generation=1, task_count=1
    )
    evolve = render_evolve_user(
        module_source="", reports=[active], generation=1, task_count=1
    )

    assert first == second
    assert "mean_mase" not in first + evolve + SELECT_SYSTEM + EVOLVE_SYSTEM
    assert "mean_smape" not in first + evolve + SELECT_SYSTEM + EVOLVE_SYSTEM
    assert "mean_smae" in first
    assert "mean_srmse" in first


def test_batch_validation_rejects_srmse_regression_despite_smae_improvement(
    tmp_path: Path,
) -> None:
    parent = parse_module(
        MODULE_HEADER
        + "\n\ndef parent(history, horizon, frequency):\n"
        '    """Use for pair-gate testing."""\n'
        "    return [8.0, 8.0]\n"
    )
    child = parse_module(
        MODULE_HEADER
        + "\n\ndef parent(history, horizon, frequency):\n"
        '    """Use for pair-gate testing."""\n'
        "    return [10.0, 7.0]\n"
    )
    parent_path = write_module(tmp_path / "parent.py", parent)
    task = Task("t", (10.0, 10.0, 10.0), 2, "1 day", (10.0, 10.0))

    accepted, metrics = _validate_candidate(
        tmp_path, 1, parent_path, child, (task,), isolate_methods=False
    )

    assert metrics["child_mean_smae"] < metrics["parent_mean_smae"]
    assert metrics["child_mean_srmse"] > metrics["parent_mean_srmse"]
    assert not accepted


def test_batch_validation_rejects_median_only_improvement() -> None:
    metrics = {
        "parent_mean_smae": 1.0,
        "parent_mean_srmse": 1.0,
        "parent_median_smae": 1.0,
        "parent_median_srmse": 1.0,
        "child_mean_smae": 1.0,
        "child_mean_srmse": 1.0,
        "child_median_smae": 0.8,
        "child_median_srmse": 0.9,
    }

    assert not _scaled_validation_accepts(metrics)


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
    assert metrics["schema_version"] == 2
    assert metrics["metric_policy_fingerprint"] == METRIC_POLICY_FINGERPRINT
    assert {entry["method"] for entry in metrics["reports"]} == {"alpha", "beta"}
    assert (repo / "transcripts" / "generation_004.md").exists()


def test_the_whole_module_reaches_the_prompt(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    llm = scripted([])

    evolve_once(repo, tasks(), llm, 1)

    sent = llm.calls[0]["messages"][0]["content"]
    assert "def alpha(" in sent and "def beta(" in sent
    assert "mean_smae" in sent and "mean_srmse" in sent
    assert "mean_smape" not in sent and "mean_mase" not in sent


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


def test_two_stage_generation_selects_before_sending_only_target_code(
    tmp_path: Path,
) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta", "gamma")
    selector = FakeLLMClient([
        json.dumps({
            "targets": [
                {"name": "gamma", "action": "delete", "reason": "dominated on MASE"}
            ]
        })
    ])
    mutator = scripted([
        {"op": "delete", "name": "gamma", "reason": "dominated on MASE"}
    ])

    outcome = evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    selector_prompt = selector.calls[0]["messages"][0]["content"]
    selector_system = selector.calls[0]["system"]
    mutator_prompt = mutator.calls[0]["messages"][0]["content"]
    assert "mean_smae" in selector_prompt and "mean_srmse" in selector_prompt
    assert "mean_mase" not in selector_prompt
    assert "def alpha(" not in selector_prompt
    assert "def beta(" not in selector_prompt
    assert "def gamma(" not in selector_prompt
    assert "not_applicable is correct" in selector_system
    assert "Never delete" in selector_system
    assert "def gamma(" in mutator_prompt
    assert "def alpha(" not in mutator_prompt
    assert "def beta(" not in mutator_prompt
    assert outcome.applied == ("delete gamma: dominated on MASE",)


def test_two_stage_generation_rejects_an_operation_outside_selected_targets(
    tmp_path: Path,
) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta", "gamma")
    selector = FakeLLMClient([
        json.dumps({
            "targets": [
                    {"name": "gamma", "action": "repair", "reason": "crashed"}
            ]
        })
    ])
    mutator = scripted([
        {"op": "delete", "name": "alpha", "reason": "not selected"}
    ])

    outcome = evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    assert outcome.applied == ()
    assert "outside selected targets" in outcome.rejected
    assert read_module(repo / "methods.py").names() == ("alpha", "beta", "gamma")


def test_two_stage_generation_rejects_an_operation_that_changes_selected_action(
    tmp_path: Path,
) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta", "gamma")
    selector = FakeLLMClient([
        json.dumps({
            "targets": [
                {"name": "gamma", "action": "fork", "reason": "needs a challenger"}
            ]
        })
    ])
    mutator = scripted([
        {"op": "delete", "name": "gamma", "reason": "changed the selected action"}
    ])

    outcome = evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    assert outcome.applied == ()
    assert "does not match selector action" in outcome.rejected
    assert read_module(repo / "methods.py").names() == ("alpha", "beta", "gamma")


def test_cli_builds_distinct_codex_selector_and_mutator_models() -> None:
    args = build_parser().parse_args([
        "--repo", "unused",
        "--llm-backend", "codex",
        "--codex-model", "gpt-5.6-terra",
        "--codex-reasoning-effort", "medium",
        "--selector-codex-model", "gpt-5.6-luna",
        "--selector-codex-reasoning-effort", "medium",
        "--train-limit", "16",
    ])

    mutator, selector = _llm_clients(args)

    assert mutator.config.model == "gpt-5.6-terra"
    assert selector.config.model == "gpt-5.6-luna"
    assert args.train_limit == 16
    assert args.isolated_methods is True


def test_cli_exposes_targetwise_cache_and_screening_options() -> None:
    args = build_parser().parse_args([
        "--repo", "unused",
        "--evolution-strategy", "targetwise",
        "--outcome-cache-dir", "runs/cache",
        "--max-targets", "2",
        "--screen-tasks", "3",
        "--full-evaluation-candidates", "2",
        "--failure-judge",
    ])

    assert args.evolution_strategy == "targetwise"
    assert args.outcome_cache_dir == "runs/cache"
    assert args.max_targets == 2
    assert args.screen_tasks == 3
    assert args.full_evaluation_candidates == 2
    assert args.failure_judge is True


def test_judge_halving_cli_requires_the_frozen_16_plus_4_design() -> None:
    args = build_parser().parse_args([
        "--repo", "unused",
        "--evolution-strategy", "targetwise",
        "--failure-judge",
        "--max-targets", "8",
        "--screen-tasks", "4",
        "--full-evaluation-candidates", "3",
    ])
    train = tuple(tasks()[0] for _ in range(16))
    dev = tuple(tasks()[0] for _ in range(4))

    _validate_judge_halving_configuration(args, train, dev)

    with pytest.raises(ValueError, match="16 Train and 4 mini-dev"):
        _validate_judge_halving_configuration(args, train[:-1], dev)


def test_two_stage_generation_rejects_a_malformed_selector_response(
    tmp_path: Path,
) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    selector = FakeLLMClient([json.dumps({"targets": "alpha"})])
    mutator = scripted([])

    outcome = evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    assert outcome.applied == ()
    assert "targets" in outcome.rejected
    assert mutator.calls == []


def test_two_stage_generation_rejects_a_merge_without_two_distinct_methods(
    tmp_path: Path,
) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    selector = FakeLLMClient([
        json.dumps({
            "targets": [
                {"name": "alpha", "action": "merge", "reason": "possibly redundant"},
                {"name": "beta", "action": "merge", "reason": "possibly redundant"},
            ]
        })
    ])
    mutator = scripted([
        {
            "op": "merge",
            "names": ["alpha", "alpha"],
            "into": "alpha",
            "code": method("alpha"),
            "reason": "invalid duplicate merge",
        }
    ])

    outcome = evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    assert outcome.applied == ()
    assert "distinct" in outcome.rejected
    assert read_module(repo / "methods.py").names() == ("alpha", "beta")


def test_two_stage_generation_rejects_a_malformed_mutator_response(
    tmp_path: Path,
) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    selector = FakeLLMClient([
        json.dumps({
            "targets": [
                    {"name": "alpha", "action": "repair", "reason": "crashed"}
            ]
        })
    ])
    mutator = FakeLLMClient([json.dumps({"operations": "not-a-list"})])

    outcome = evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    assert outcome.applied == ()
    assert "operations" in outcome.rejected
    assert read_module(repo / "methods.py").names() == ("alpha", "beta")


def test_two_stage_prompt_exposes_identity_contracts_and_repair_fork_actions(
    tmp_path: Path,
) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    selector = FakeLLMClient([
        json.dumps({
            "targets": [
                {"name": "alpha", "action": "repair", "reason": "crashed"}
            ]
        })
    ])
    mutator = scripted([])

    evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    prompt = mutator.calls[0]["messages"][0]["content"]
    system = mutator.calls[0]["system"]
    assert "identity_contracts" in prompt
    assert '"repair_allowed": false' in prompt
    assert '"mode": "fork_only"' in prompt
    assert '"op": "repair"' in system
    assert '"op": "fork"' in system


def test_two_stage_mutator_explains_that_same_name_repair_is_constant_only() -> None:
    assert "literal constants" in MUTATE_SYSTEM
    assert "control flow, calls, variable names, operators, and returns" in MUTATE_SYSTEM
    assert "use fork" in MUTATE_SYSTEM
    assert "exact action selected" in MUTATE_SYSTEM
    assert "coverage is at least 0.5" in MUTATE_SYSTEM


def test_selector_inventory_marks_unreviewed_methods_fork_only(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    selector = FakeLLMClient([json.dumps({"targets": []})])

    evolve_once(repo, tasks(), scripted([]), 1, selector_llm=selector)

    prompt = selector.calls[0]["messages"][0]["content"]
    system = selector.calls[0]["system"]
    assert '"repair_allowed": false' in prompt
    assert '"mode": "fork_only"' in prompt
    assert "fork_only" in system


def test_two_stage_generation_rejects_deleting_an_untested_specialist(
    tmp_path: Path,
) -> None:
    repo = init_repo(tmp_path / "evo")
    specialist = method(
        "specialist",
        '    raise NotApplicable("needs a longer history")',
    )
    module = parse_module(MODULE_HEADER + "\n\n" + specialist + "\n\n" + method("general"))
    write_module(repo / "methods.py", module)
    commit_module(repo, "seed", [])
    selector = FakeLLMClient([
        json.dumps({
            "targets": [
                {"name": "specialist", "action": "delete", "reason": "zero coverage"}
            ]
        })
    ])
    mutator = scripted([
        {"op": "delete", "name": "specialist", "reason": "zero coverage"}
    ])

    outcome = evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    assert outcome.applied == ()
    assert "not sufficiently evaluated" in outcome.rejected
    assert "specialist" in read_module(repo / "methods.py").names()


def test_validation_tasks_reject_a_child_with_worse_scaled_pair(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "evo")
    alpha = method("alpha", "    return [float(history[-1])] * horizon")
    beta = method("beta", "    return [0.0] * horizon")
    module = parse_module(MODULE_HEADER + "\n\n" + alpha + "\n\n" + beta)
    write_module(repo / "methods.py", module)
    commit_module(repo, "seed", [])
    selector = FakeLLMClient([
        json.dumps({
            "targets": [
                {"name": "beta", "action": "delete", "reason": "poor on train"}
            ]
        })
    ])
    mutator = scripted([
        {"op": "delete", "name": "beta", "reason": "poor on train"}
    ])
    validation = (
        Task("dev", (5.0, 5.0, 5.0), 2, "1 day", (0.0, 0.0)),
    )

    outcome = evolve_once(
        repo,
        tasks(),
        mutator,
        1,
        selector_llm=selector,
        validation_tasks=validation,
    )

    assert outcome.applied == ()
    assert "validation scaled pair failed Pareto acceptance" in outcome.rejected
    assert read_module(repo / "methods.py").names() == ("alpha", "beta")


def test_two_stage_rejects_rewrite_even_when_the_method_was_selected(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    selector = FakeLLMClient([json.dumps({
        "targets": [{"name": "alpha", "action": "repair", "reason": "crashed"}]
    })])
    mutator = scripted([{
        "op": "rewrite",
        "name": "alpha",
        "code": method("alpha"),
        "reason": "legacy operation",
    }])

    outcome = evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    assert outcome.applied == ()
    assert "does not allow operation 'rewrite'" in outcome.rejected


def test_merge_cannot_replace_a_named_method_with_another_algorithm(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "evo")
    module = parse_module(
        MODULE_HEADER + "\n\n" + SARIMA_EXAMPLE + "\n\n" + method("beta")
    )
    write_module(repo / "methods.py", module)
    commit_module(repo, "seed", [])
    selector = FakeLLMClient([json.dumps({"targets": [
        {"name": "sarima_auto", "action": "merge", "reason": "similar"},
        {"name": "beta", "action": "merge", "reason": "similar"},
    ]})])
    mutator = scripted([{
        "op": "merge",
        "names": ["sarima_auto", "beta"],
        "into": "sarima_auto",
        "code": method("sarima_auto"),
        "reason": "replace both with last value",
    }])

    outcome = evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    assert outcome.applied == ()
    assert "violates method identity" in outcome.rejected


def test_merge_cannot_remove_an_all_not_applicable_specialist(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "evo")
    specialist = method("specialist", '    raise NotApplicable("needs another domain")')
    module = parse_module(
        MODULE_HEADER + "\n\n" + SARIMA_EXAMPLE + "\n\n" + specialist
    )
    write_module(repo / "methods.py", module)
    commit_module(repo, "seed", [])
    selector = FakeLLMClient([json.dumps({"targets": [
        {"name": "sarima_auto", "action": "merge", "reason": "combine"},
        {"name": "specialist", "action": "merge", "reason": "combine"},
    ]})])
    mutator = scripted([{
        "op": "merge",
        "names": ["sarima_auto", "specialist"],
        "into": "sarima_auto",
        "code": SARIMA_EXAMPLE,
        "reason": "combine",
    }])

    outcome = evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    assert outcome.applied == ()
    assert "specialist is not sufficiently evaluated for deletion" in outcome.rejected


def test_fork_parent_must_survive_the_complete_operation_batch(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path, "alpha", "beta")
    selector = FakeLLMClient([json.dumps({"targets": [
        {"name": "alpha", "action": "fork", "reason": "try challenger"},
        {"name": "alpha", "action": "delete", "reason": "replace parent"},
    ]})])
    mutator = scripted([
        {
            "op": "fork",
            "from": "alpha",
            "new_identity": "recent mean challenger",
            "code": method("alpha_recent_mean", "    return [sum(history[-4:]) / 4] * horizon"),
            "reason": "try challenger",
        },
        {"op": "delete", "name": "alpha", "reason": "replace parent"},
    ])

    outcome = evolve_once(repo, tasks(), mutator, 1, selector_llm=selector)

    assert outcome.applied == ()
    assert "duplicate selector target 'alpha'" in outcome.rejected
    assert mutator.calls == []
    assert read_module(repo / "methods.py").names() == ("alpha", "beta")


def test_train_limit_counts_evolution_tasks_before_validation_tail(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "partitions": {"train": {"task_ids": [f"task_{i}" for i in range(25)]}}
    }))
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("\n".join(json.dumps({
        "benchmark_id": f"task_{i}",
        "task_metadata": {"prediction_length": 1, "frequency": "1 day"},
        "series": {"history_values": [1.0, 2.0], "future_values": [3.0]},
    }) for i in range(25)))

    train, validation = _evolution_tasks(split, tasks_file, train_limit=16, validation_tail=4)

    assert [task.task_id for task in train] == [f"task_{i}" for i in range(16)]
    assert [task.task_id for task in validation] == [f"task_{i}" for i in range(16, 20)]


def test_evolution_tasks_can_load_the_public_task_directory(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "partitions": {"train": {"task_ids": ["task_2", "task_1"]}}
    }))
    directory = tmp_path / "tasks"
    directory.mkdir()
    for task_id, value in (("task_1", 1.0), ("task_2", 2.0)):
        (directory / f"{task_id}.json").write_text(json.dumps({
            "benchmark_id": task_id,
            "labels_public": True,
            "task_metadata": {"prediction_length": 1, "frequency": "1 day"},
            "series": {"history_values": [value, value], "future_values": [value]},
        }))

    train, validation = _evolution_tasks(
        split, directory, train_limit=1, validation_tail=1
    )

    assert [task.task_id for task in train] == ["task_2"]
    assert [task.task_id for task in validation] == ["task_1"]
