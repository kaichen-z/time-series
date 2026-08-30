from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import numerical_agent.run_task_conditioned_screening as screening_script
from numerical_agent.run_task_conditioned_screening import (
    SCALED_METRIC_POLICY,
    _manifest_fingerprint,
    _merge_cache_summaries,
    _paired_screening_counts,
    _report,
    _training_outcomes,
    _train_constraints_met,
    _write_policy_artifacts,
    _write_active,
    build_parser,
    load_frozen_partitions,
    main,
)
from common.evolution_core.contracts import (
    METRIC_POLICY,
    metric_policy_metadata,
    require_active_metric_policy,
)
from common.payload import strict_json_loads, write_json
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.filtering import build_filter_dictionary, render_filter_source
from numerical_agent.evolution.module import MODULE_HEADER, parse_module, write_module
from numerical_agent.evolution.portfolio import (
    CombinedPolicy,
    PolicyError,
    PolicyPortfolio,
    write_policy_file,
)
from numerical_agent.evolution.screening import (
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
    ScreeningConstraints,
)


ROOT = Path(__file__).resolve().parents[1]


class _StopAfterValidation(Exception):
    pass


def _screening_module():
    names = (
        "seasonal_naive",
        "holt_damped_trend",
        "croston_sba",
        "robust_loess_trend",
        "median_seasonal_profile_forecast",
        *(f"statistical_{index}" for index in range(88)),
    )
    source = "\n\n".join(
        f'''def {name}(history, horizon, frequency):
    """Screening namespace fixture."""
    return [0.0] * horizon
'''
        for name in names
    )
    return parse_module(MODULE_HEADER + "\n\n" + source)


def _screening_portfolio(combined_count: int) -> PolicyPortfolio:
    portfolio = PolicyPortfolio.flagship5()
    while len(portfolio.combined) < combined_count:
        index = len(portfolio.combined)
        portfolio = portfolio.add_combined(CombinedPolicy(
            f"combined_extra_{index}",
            ("toto_2_0", "seasonal_naive"),
            "median",
            fallback_parent="toto_2_0",
        ))
    return portfolio


def _screening_argv(repo: Path, output: Path, *extra: str) -> list[str]:
    return [
        "--repo", str(repo),
        "--tasks-file", "unused-tasks",
        "--outcome-cache-dir", str(repo / "method-cache"),
        "--policy-outcome-cache-dir", str(repo / "policy-cache"),
        "--target-batches-file", "unused-batches.json",
        "--output-dir", str(output),
        *extra,
    ]


def _run_until_screening_validation(
    tmp_path, monkeypatch, *, combined_count: int, max_candidates: int | None = None,
    dictionary=None,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    module = _screening_module()
    portfolio = _screening_portfolio(combined_count)
    write_module(repo / "methods.py", module)
    write_policy_file(repo / "policies.py", portfolio)
    (repo / "frozen_dictionary.py").write_text(
        render_filter_source(build_filter_dictionary(module, portfolio)),
        encoding="utf-8",
    )
    write_json(repo / "seed_manifest.json", {
        "schema_version": 2,
        **metric_policy_metadata(),
        "seed_kind": "complete_master_dictionary",
        "source_hashes": {
            "methods.py": screening_script._sha256(repo / "methods.py"),
            "policies.py": screening_script._sha256(repo / "policies.py"),
        },
    })
    task = Task("screening", (1.0, 2.0), 1, "D", (3.0,))
    monkeypatch.setattr(screening_script, "load_frozen_partitions", lambda *args, **kwargs: ((task,), (task,)))
    if dictionary is not None:
        monkeypatch.setattr(screening_script, "build_filter_dictionary", lambda *args: dictionary)
    captured = []
    original_constraints = screening_script.ScreeningConstraints

    def capture_constraints(**kwargs):
        captured.append(kwargs["max_active_candidates"])
        return original_constraints(**kwargs)

    monkeypatch.setattr(screening_script, "ScreeningConstraints", capture_constraints)
    monkeypatch.setattr(
        screening_script,
        "_training_outcomes",
        lambda *args: (_ for _ in ()).throw(_StopAfterValidation()),
    )
    extra = () if max_candidates is None else ("--screen-max-candidates", str(max_candidates))
    return _screening_argv(repo, tmp_path / "output", *extra), captured, module, portfolio


def test_partition_loader_does_not_open_dev_records_in_train_only_mode(
    tmp_path, monkeypatch
):
    split = tmp_path / "split.json"
    split.write_text(
        '{"partitions":{"train":{"task_ids":["train-1"]},'
        '"dev":{"task_ids":["task_dev_1"]}}}'.replace("train-1", "task_train_1"),
        encoding="utf-8",
    )
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    record = (
        '{{"benchmark_id":"{task_id}","series":{{"history_values":[1,2],'
        '"future_values":[3]}},"task_metadata":{{"prediction_length":1,'
        '"frequency":"D"}},"entity_name":"{entity}"}}'
    )
    (task_dir / "task_train_1.json").write_text(
        record.format(task_id="task_train_1", entity="train-entity"), encoding="utf-8"
    )
    (task_dir / "task_dev_1.json").write_text(
        record.format(task_id="task_dev_1", entity="dev-entity"), encoding="utf-8"
    )
    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path.name == "task_dev_1.json":
            raise AssertionError("Train-only loading opened a Dev record")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    train, dev = load_frozen_partitions(
        split, task_dir, train_limit=1, dev_limit=0
    )

    assert [task.task_id for task in train] == ["task_train_1"]
    assert dev == ()


def test_train_constraints_do_not_require_dev_and_cache_summaries_are_additive():
    score = SimpleNamespace(
        global_oracle_retention=1.0,
        min_active_candidates=12,
        max_active_candidates=80,
        unique_active_dictionaries=3,
        task_count=80,
        mean_pairwise_jaccard=0.9,
        conditioned_entries_by_family={
            "statistical": 1,
            "tsfm": 1,
            "combined": 1,
        },
    )
    constraints = ScreeningConstraints(max_active_candidates=103)

    assert _train_constraints_met(score, constraints)
    assert _merge_cache_summaries(
        {"statistical_hits": 10, "tsfm_misses": 2},
        {"statistical_hits": 3, "tsfm_misses": 1},
    ) == {"statistical_hits": 13, "tsfm_misses": 3}


def test_training_outcomes_rejects_duplicate_task_ids_before_cache_access():
    task = Task("duplicate", (1.0, 2.0), 1, "D", (3.0,))

    with pytest.raises(ValueError, match="duplicate task IDs"):
        _training_outcomes(SimpleNamespace(), None, None, None, (task, task))


def test_training_outcomes_preflights_namespace_before_cache_or_runtime(
    tmp_path, monkeypatch
):
    module = parse_module(
        MODULE_HEADER
        + '''

def seasonal_naive(history, horizon, frequency):
    """Satisfy the reviewed Combined parent contract."""
    return [0.0] * horizon

def holt_damped_trend(history, horizon, frequency):
    """Satisfy the reviewed Combined parent contract."""
    return [0.0] * horizon

def croston_sba(history, horizon, frequency):
    """Satisfy the reviewed Combined parent contract."""
    return [0.0] * horizon

def robust_loess_trend(history, horizon, frequency):
    """Satisfy the reviewed Combined parent contract."""
    return [0.0] * horizon

def median_seasonal_profile_forecast(history, horizon, frequency):
    """Satisfy the reviewed Combined parent contract."""
    return [0.0] * horizon

def timesfm_2_5(history, horizon, frequency):
    """An invalid statistical name collision fixture."""
    return [0.0] * horizon
'''
    )
    constructed = []

    def forbidden_cache(*args, **kwargs):
        del args, kwargs
        constructed.append("cache")
        raise AssertionError("cache construction must not occur")

    monkeypatch.setattr(
        "numerical_agent.run_task_conditioned_screening.OutcomeCache", forbidden_cache
    )
    args = SimpleNamespace(
        outcome_cache_dir=tmp_path / "cache",
        policy_outcome_cache_dir=tmp_path / "policy-cache",
    )
    task = Task("task", (1.0, 2.0), 1, "D", (3.0,))

    with pytest.raises(PolicyError, match="namespace.*timesfm_2_5"):
        _training_outcomes(args, tmp_path, module, PolicyPortfolio.flagship5(), (task,))

    assert constructed == []


def test_screening_cli_has_train_dev_but_no_public_test_option():
    parser = build_parser()
    args = parser.parse_args([
        "--repo", "repo",
        "--tasks-file", "tasks",
        "--outcome-cache-dir", "method-cache",
        "--policy-outcome-cache-dir", "policy-cache",
        "--output-dir", "output",
        "--target-batches-file", "batches.json",
        "--train-limit", "80",
        "--dev-limit", "20",
    ])
    assert args.train_limit == 80
    assert args.dev_limit == 20
    assert args.seed_manifest is None
    assert args.baseline_method == "toto_2_0"
    assert args.screen_min_candidates == 12
    assert args.screen_max_candidates is None
    assert args.screen_min_unique_dictionaries == 3
    assert args.screen_min_dev_oracle_retention == 0.9
    assert args.screen_batch_size == 8
    assert args.screen_refinement_generations == 3
    assert args.screen_refinement_batch_size == 24
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--repo", "repo", "--tasks-file", "tasks",
            "--outcome-cache-dir", "cache", "--policy-outcome-cache-dir", "cache2",
            "--output-dir", "out", "--target-batches-file", "batch",
            "--public-test-limit", "99",
        ])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--repo", "repo", "--tasks-file", "tasks",
            "--outcome-cache-dir", "cache", "--policy-outcome-cache-dir", "cache2",
            "--output-dir", "out", "--target-batches-file", "batch",
            "--seed-policy", "legacy",
        ])


def test_screening_public_contract_uses_the_runtime_namespace_formula():
    formula = "len(module.names()) + len(portfolio.names)"
    help_text = build_parser().format_help()

    assert formula in help_text
    for path in (ROOT / "README.md", ROOT / "numerical_agent" / "README.md"):
        source = path.read_text(encoding="utf-8")
        assert formula in source
        assert "93 + len(portfolio.names)" not in source
    assert "103-candidate method repository" not in help_text


@pytest.mark.parametrize("combined_count, expected_count", ((5, 103), (6, 104)))
def test_screening_cli_accepts_exact_runtime_candidate_namespace(
    tmp_path, monkeypatch, combined_count, expected_count,
):
    argv, captured, module, portfolio = _run_until_screening_validation(
        tmp_path, monkeypatch, combined_count=combined_count,
    )

    with pytest.raises(_StopAfterValidation):
        main(argv)

    assert len(module.methods) + len(portfolio.names) == expected_count
    assert captured == [expected_count]


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "extra"))
def test_screening_cli_rejects_parent_namespace_mismatch_before_cache_or_runtime(
    tmp_path, monkeypatch, mutation,
):
    module = _screening_module()
    portfolio = _screening_portfolio(5)
    entries = list(build_filter_dictionary(module, portfolio).entries)
    if mutation == "missing":
        entries.pop()
    elif mutation == "duplicate":
        entries[-1] = entries[0]
    else:
        entries.append(type(entries[0])(
            "unexpected_candidate", "statistical", "keep", (), "unexpected",
        ))
    parent = SimpleNamespace(entries=tuple(entries))
    argv, captured, _, _ = _run_until_screening_validation(
        tmp_path,
        monkeypatch,
        combined_count=5,
        dictionary=SimpleNamespace(entries=tuple(entries)),
    )
    monkeypatch.setattr(screening_script, "migrate_filter_dictionary", lambda *args, **kwargs: parent)

    with pytest.raises(ValueError, match="screening parent namespace mismatch"):
        main(argv)

    assert captured == []


@pytest.mark.parametrize(
    ("provided_ceiling", "expected_ceiling"),
    ((None, 104), (1, 1), (104, 104)),
)
def test_screening_cli_derives_or_bounds_candidate_ceiling(
    tmp_path, monkeypatch, provided_ceiling, expected_ceiling,
):
    argv, captured, _, _ = _run_until_screening_validation(
        tmp_path, monkeypatch, combined_count=6, max_candidates=provided_ceiling,
    )
    if provided_ceiling == 1:
        argv.extend(("--screen-min-candidates", "1"))

    with pytest.raises(_StopAfterValidation):
        main(argv)

    assert captured == [expected_ceiling]


@pytest.mark.parametrize("provided_ceiling", (0, 105))
def test_screening_cli_rejects_candidate_ceiling_outside_runtime_namespace(
    tmp_path, monkeypatch, provided_ceiling,
):
    argv, captured, _, _ = _run_until_screening_validation(
        tmp_path, monkeypatch, combined_count=6, max_candidates=provided_ceiling,
    )

    with pytest.raises(ValueError, match="screen max candidates must be between 1 and 104"):
        main(argv)

    assert captured == []


def test_screening_shell_forwards_formal_configuration():
    script = ROOT / "scripts" / "run_task_conditioned_screening.sh"
    completed = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    source = script.read_text(encoding="utf-8")
    for option in (
        "--repo", "--split-file", "--tasks-file", "--outcome-cache-dir",
        "--policy-outcome-cache-dir", "--train-limit", "--dev-limit",
        "--target-batches-file", "--output-dir", "--codex-model",
            "--screen-refinement-generations", "--screen-refinement-batch-size",
            "--screen-min-dev-oracle-retention",
    ):
        assert option in source
    assert 'SCREEN_MAX_CANDIDATES:-103' not in source
    assert 'MAX_CANDIDATES="${SCREEN_MAX_CANDIDATES:-}"' in source



def test_report_exposes_task_conditioning_and_family_coverage():
    score = {
        "coverage": 1.0,
        "active_success_rate": 0.8,
        "failure_exposure": 0.1,
        "not_applicable_exposure": 0.1,
        "mean_active_smae": 0.8,
        "mean_active_srmse": 0.9,
        "mean_smae": 0.8,
        "median_smae": 0.7,
        "se_smae": 0.01,
        "mean_srmse": 0.9,
        "median_srmse": 0.8,
        "se_srmse": 0.02,
        "p90_smae_raw": 6.0,
        "p95_smae_raw": 7.0,
        "p90_srmse_raw": 8.0,
        "p95_srmse_raw": 9.0,
        "smae_clipped_count": 1,
        "smae_clipped_rate": 0.1,
        "srmse_clipped_count": 2,
        "srmse_clipped_rate": 0.2,
        "global_oracle_retention": 1.0,
        "mean_active_oracle_regret": 0.0,
        "mean_active_oracle_smae_regret": 0.0,
        "mean_active_oracle_srmse_regret": 0.0,
        "active_crashed": 1,
        "active_invalid": 2,
        "active_missing": 3,
        "active_malformed_success": 4,
        "compression": 0.4,
        "mean_active_candidates": 32.5,
        "min_active_candidates": 24,
        "max_active_candidates": 40,
        "unique_active_dictionaries": 7,
        "mean_pairwise_jaccard": 0.81,
        "conditioned_entries_by_family": {
            "statistical": 9,
            "tsfm": 2,
            "combined": 3,
        },
    }
    report = _report({
        "candidate_count": 103,
        "train_tasks": 80,
        "dev_tasks": 20,
        "accepted_generations": [1, 2],
        "candidate_screening_policy_sha256": "abc",
        "frozen_screening_policy_sha256": "abc",
        "public_test_accessed": False,
        "final_constraints_met": True,
        "dev_evaluations": 1,
        "final_dev_gate": {"accepted": True, "reason": "safe on held-out Dev"},
        "train": score,
        "dev": score,
        "paired_joint_wtl": {
            "train": {"wins": 1, "ties": 2, "losses": 3, "missing": 4, "unscored": 5},
            "dev": {"wins": 5, "ties": 4, "losses": 3, "missing": 2, "unscored": 1},
        },
    })

    assert "Unique dictionaries" in report
    assert "Pairwise Jaccard" in report
    assert "Statistical / TSFM / Combined" in report
    assert "9 / 2 / 3" in report
    assert "Crash / invalid / missing / malformed" in report
    assert "1 / 2 / 3 / 4" in report
    assert "Dev evaluations: 1" in report
    assert "safe on held-out Dev" in report
    assert "Mean active sMAE" in report
    assert "Mean active sRMSE" in report
    assert "Median sMAE" in report
    assert "sMAE SE" in report
    assert "Raw P90/P95 sMAE" in report
    assert "Clipped sMAE/sRMSE" in report
    assert "Wins / Ties / Losses / Missing / Unscored" in report


def test_screening_paired_counts_conserve_both_missing_tasks() -> None:
    parent = SimpleNamespace(task_count=4, task_scaled_pairs={"same": (1.0, 1.0), "left": (1.0, 1.0)})
    child = SimpleNamespace(task_count=4, task_scaled_pairs={"same": (1.0, 1.0), "right": (1.0, 1.0)})

    counts = _paired_screening_counts(parent, child)

    assert counts == {"wins": 0, "ties": 1, "losses": 0, "missing": 2, "unscored": 1}
    assert sum(counts.values()) == 4


def test_materialized_active_dictionary_rows_are_schema_v2_policy_bound(tmp_path) -> None:
    policy = ScreeningPolicy(
        entries=(ScreeningEntry(
            "fallback", "statistical", "keep", ApplicabilityPolicy(), "safe fallback"
        ),),
        fallback_names=("fallback",),
    )
    target = tmp_path / "active.jsonl"

    _write_active(
        target,
        policy,
        (Task("task", (1.0, 2.0), 1, "D", (3.0,)),),
    )

    payload = strict_json_loads(target.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["schema_version"] == 2
    require_active_metric_policy(payload)


def test_screening_manifest_hash_binds_scaled_metric_objective():
    manifest = {"schema_version": 2, "metric_policy": SCALED_METRIC_POLICY}

    assert SCALED_METRIC_POLICY == METRIC_POLICY

    assert _manifest_fingerprint(manifest) != _manifest_fingerprint(
        {
            **manifest,
            "metric_policy": {
                **SCALED_METRIC_POLICY,
                "objective": "legacy_mase",
            },
        }
    )


def test_rejected_candidate_is_not_published_as_frozen(tmp_path):
    policy = ScreeningPolicy(
        (
            ScreeningEntry("naive", "statistical", "keep", ApplicabilityPolicy(()), "safe"),
            ScreeningEntry("timesfm", "tsfm", "keep", ApplicabilityPolicy(()), "safe"),
            ScreeningEntry("blend", "combined", "keep", ApplicabilityPolicy(()), "safe"),
        ),
        ("naive", "timesfm", "blend"),
    )

    rejected = _write_policy_artifacts(tmp_path, policy, accepted=False)

    assert (tmp_path / "candidate_screening_policy.py").is_file()
    assert not (tmp_path / "frozen_screening_policy.py").exists()
    assert rejected["frozen_screening_policy_sha256"] is None

    accepted = _write_policy_artifacts(tmp_path, policy, accepted=True)

    assert (tmp_path / "frozen_screening_policy.py").is_file()
    assert accepted["frozen_screening_policy_sha256"] is not None
