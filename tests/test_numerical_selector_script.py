from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.payload import write_json

from numerical_agent.evolution.execution import CRASHED, INVALID, SUCCESS, Outcome, Task
from numerical_agent.evolution.module import MODULE_HEADER, read_module
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    HindcastConfig,
)
from numerical_agent.evolution.portfolio import (
    CombinedPolicy,
    PolicyPortfolio,
    combine_materialized_outcome,
)
from numerical_agent.evolution.screening import (
    ApplicabilityClause,
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
)
from numerical_agent.evolution.selector_evolution import DecisionCase
from numerical_agent.providers import RuntimeRegistry
from numerical_agent.run_selector_evolution import (
    ForecastStore,
    _build_case,
    _global_ranking,
    _report,
    _score_pair_wtl,
    _write_cases,
    build_parser,
)
import common.evolution_core.contracts as evolution_contracts


ROOT = Path(__file__).resolve().parents[1]


def test_forecast_store_contains_a_hanging_statistical_method_and_recovers(tmp_path):
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def hangs_forever(history, horizon, frequency):
    """Use only to verify that native or Python hangs are contained."""
    while True:
        pass


def fast_method(history, horizon, frequency):
    """Use when a last-value forecast is sufficient."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )
    module = read_module(methods)
    store = ForecastStore(
        tmp_path / "cache",
        methods,
        None,
        module,
        PolicyPortfolio.flagship5(),
        RuntimeRegistry(),
        "screen-hash",
        statistical_time_budget_s=0.1,
    )
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="hard timeout"):
            store.forecast("hangs_forever", (1.0, 2.0), 2, "D")
        assert store.forecast("fast_method", (1.0, 2.0), 2, "D") == (2.0, 2.0)
    finally:
        store.close()
    assert time.monotonic() - started < 3.0


def test_forecast_store_reuses_identical_forecast_across_screening_policies(tmp_path):
    """Screening changes eligibility, not a candidate's numerical forecast identity."""
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def stable_method(history, horizon, frequency):
    """Use when a last-value forecast is sufficient."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )
    module = read_module(methods)
    cache = tmp_path / "cache"

    first = ForecastStore(
        cache,
        methods,
        None,
        module,
        PolicyPortfolio.flagship5(),
        RuntimeRegistry(),
        "part-1",
    )
    try:
        assert first.forecast("stable_method", (1.0, 2.0), 2, "D") == (2.0, 2.0)
        assert first.misses == 1
    finally:
        first.close()
    cached = json.loads(next(cache.glob("*.json")).read_text(encoding="utf-8"))
    assert cached["metric_policy"] == {
        **evolution_contracts.METRIC_POLICY,
        "primary": list(evolution_contracts.METRIC_POLICY["primary"]),
    }
    assert (
        cached["metric_policy_fingerprint"]
        == evolution_contracts.METRIC_POLICY_FINGERPRINT
    )

    second = ForecastStore(
        cache,
        methods,
        None,
        module,
        PolicyPortfolio.flagship5(),
        RuntimeRegistry(),
        "part-2",
    )
    try:
        assert second.forecast("stable_method", (1.0, 2.0), 2, "D") == (2.0, 2.0)
        assert second.hits == 1
        assert second.misses == 0
    finally:
        second.close()


def test_forecast_store_rejects_a_legacy_cache_row_instead_of_seeding_runtime(
    tmp_path,
):
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def stable_method(history, horizon, frequency):
    """Use when a last-value forecast is sufficient."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )
    module = read_module(methods)
    store = ForecastStore(
        tmp_path / "cache",
        methods,
        None,
        module,
        PolicyPortfolio.flagship5(),
        RuntimeRegistry(),
        "screen",
    )
    try:
        key = store._key("stable_method", (1.0, 2.0), 2, "D")
        (store.root / f"{key}.json").write_text(
            json.dumps({"key": key, "status": "success", "forecast": [2.0, 2.0]}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing metric policy"):
            store.forecast("stable_method", (1.0, 2.0), 2, "D")
    finally:
        store.close()


def test_forecast_store_identity_binds_skills_runtime_and_checkpoint(tmp_path):
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def stable_method(history, horizon, frequency):
    """Use when a last-value forecast is sufficient."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )
    skills = tmp_path / "skills.py"
    skills.write_text("SKILL_VERSION = 1\n", encoding="utf-8")
    module = read_module(methods)
    args = (module, PolicyPortfolio.flagship5(), RuntimeRegistry(), "screen")
    first = ForecastStore(
        tmp_path / "a", methods, skills, *args,
        runtime_identity={"provider": "p", "checkpoint": "v1"},
    )
    second = ForecastStore(
        tmp_path / "b", methods, skills, *args,
        runtime_identity={"provider": "p", "checkpoint": "v2"},
    )
    key1 = first._key("stable_method", (1.0, 2.0), 1, "D")
    key2 = second._key("stable_method", (1.0, 2.0), 1, "D")
    first.close()
    second.close()
    skills.write_text("SKILL_VERSION = 2\n", encoding="utf-8")
    third = ForecastStore(
        tmp_path / "c", methods, skills, *args,
        runtime_identity={"provider": "p", "checkpoint": "v1"},
    )
    key3 = third._key("stable_method", (1.0, 2.0), 1, "D")
    third.close()

    assert len({key1, key2, key3}) == 3


@pytest.mark.parametrize(
    "row",
    [
        "not-json",
        json.dumps({**evolution_contracts.metric_policy_metadata(), "key": "wrong", "status": SUCCESS, "forecast": [2.0]}),
        json.dumps({**evolution_contracts.metric_policy_metadata(), "key": "PLACEHOLDER", "status": "unknown"}),
    ],
)
def test_forecast_store_existing_malformed_or_noncanonical_row_fails_closed(tmp_path, row):
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def stable_method(history, horizon, frequency):
    """Use when a last-value forecast is sufficient."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )
    store = ForecastStore(
        tmp_path / "cache", methods, None, read_module(methods),
        PolicyPortfolio.flagship5(), RuntimeRegistry(), "screen",
    )
    try:
        key = store._key("stable_method", (1.0, 2.0), 1, "D")
        (store.root / f"{key}.json").write_text(row.replace("PLACEHOLDER", key), encoding="utf-8")
        with pytest.raises(ValueError, match="active hindcast cache row") as raised:
            store.forecast("stable_method", (1.0, 2.0), 1, "D")
        assert type(raised.value).__name__ == "CacheIntegrityError"
    finally:
        store.close()


@pytest.mark.parametrize("schema", (True, 3.0))
def test_forecast_store_rejects_noninteger_schema_aliases(tmp_path, schema):
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def stable_method(history, horizon, frequency):
    """Use when a last-value forecast is sufficient."""
    return [float(history[-1])] * horizon
''',
        encoding="utf-8",
    )
    store = ForecastStore(
        tmp_path / "cache", methods, None, read_module(methods),
        PolicyPortfolio.flagship5(), RuntimeRegistry(), "screen",
    )
    try:
        store.forecast("stable_method", (1.0, 2.0), 1, "D")
        entry = next(store.root.glob("*.json"))
        payload = json.loads(entry.read_text(encoding="utf-8"))
        payload["cache_schema"] = schema
        entry.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="schema mismatch"):
            store.forecast("stable_method", (1.0, 2.0), 1, "D")
    finally:
        store.close()


def test_corrupted_combined_leaf_cache_aborts_instead_of_falling_back(tmp_path):
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def primary_leaf(history, horizon, frequency):
    """Use as a primary leaf in cache-integrity tests."""
    return [1.0] * horizon

def fallback_leaf(history, horizon, frequency):
    """Use as a fallback leaf in cache-integrity tests."""
    return [2.0] * horizon
''',
        encoding="utf-8",
    )
    combined = CombinedPolicy(
        "combined_cache_integrity",
        ("primary_leaf", "toto_2_0"),
        "weighted_mean",
        (0.5, 0.5),
        fallback_parent="toto_2_0",
    )
    store = ForecastStore(
        tmp_path / "cache",
        methods,
        None,
        read_module(methods),
        PolicyPortfolio(PolicyPortfolio.flagship5().tsfm, (combined,)),
        RuntimeRegistry(),
        "screen",
    )
    try:
        key = store._key("primary_leaf", (1.0, 2.0), 1, "D")
        (store.root / f"{key}.json").write_text("not-json", encoding="utf-8")

        with pytest.raises(ValueError, match="malformed") as raised:
            store.forecast("combined_cache_integrity", (1.0, 2.0), 1, "D")
        assert type(raised.value).__name__ == "CacheIntegrityError"
    finally:
        store.close()


def test_forecast_store_materializes_canonical_combined_leaf_forecasts_once(tmp_path):
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def seasonal_naive(history, horizon, frequency):
    """Return a fixed statistical leaf forecast for combination tests."""
    return [100.0] * horizon
''',
        encoding="utf-8",
    )
    module = read_module(methods)
    base = PolicyPortfolio.flagship5()
    portfolio = PolicyPortfolio(
        base.tsfm,
        (
            CombinedPolicy(
                "combined_two_tsfm_median",
                ("toto_2_0", "timesfm_2_5"),
                "median",
                fallback_parent="toto_2_0",
            ),
            CombinedPolicy(
                "combined_tsfm_statistical_weighted",
                ("toto_2_0", "seasonal_naive"),
                "weighted_mean",
                (0.25, 0.75),
                fallback_parent="toto_2_0",
            ),
            CombinedPolicy(
                "combined_three_parent_weighted",
                ("toto_2_0", "timesfm_2_5", "seasonal_naive"),
                "weighted_mean",
                (0.2, 0.3, 0.5),
                fallback_parent="toto_2_0",
            ),
        ),
    )

    class CountingRuntime:
        def __init__(self):
            self.calls: list[str] = []

        def supports(self, candidate):
            return candidate.method_id in {"method_tsfm_0014", "method_tsfm_0031"}

        def forecast(self, candidate, history, horizon, frequency):
            del history, frequency
            self.calls.append(candidate.method_id)
            value = {"method_tsfm_0014": 10.0, "method_tsfm_0031": 20.0}[candidate.method_id]
            return (value,) * horizon

    runtime = CountingRuntime()
    store = ForecastStore(
        tmp_path / "cache",
        methods,
        None,
        module,
        portfolio,
        RuntimeRegistry({"timesfm": runtime, "tsfm_worker": runtime}),
        "screen-hash",
    )
    try:
        assert store.forecast("combined_two_tsfm_median", (1.0, 2.0), 2, "D") == (15.0, 15.0)
        assert store.forecast(
            "combined_tsfm_statistical_weighted", (1.0, 2.0), 2, "D"
        ) == (77.5, 77.5)
        assert store.forecast(
            "combined_three_parent_weighted", (1.0, 2.0), 2, "D"
        ) == (58.0, 58.0)
    finally:
        store.close()

    assert runtime.calls.count("method_tsfm_0014") == 1
    assert runtime.calls.count("method_tsfm_0031") == 1


@pytest.mark.parametrize("failure_status", (INVALID, CRASHED))
def test_forecast_store_materializes_shared_failed_tsfm_leaf_once_in_memory(
    tmp_path, failure_status
):
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def unused_statistical_leaf(history, horizon, frequency):
    """Satisfy the parsed module contract without entering this test graph."""
    return [0.0] * horizon
''',
        encoding="utf-8",
    )
    module = read_module(methods)
    base = PolicyPortfolio.flagship5()
    combined = tuple(
        CombinedPolicy(
            f"combined_failed_leaf_{index}",
            ("toto_2_0", "timesfm_2_5"),
            "median",
            fallback_parent="toto_2_0",
        )
        for index in range(3)
    )
    portfolio = PolicyPortfolio(base.tsfm, combined)

    class FailedLeafRuntime:
        def __init__(self):
            self.calls: list[str] = []

        def supports(self, candidate):
            return candidate.method_id in {"method_tsfm_0014", "method_tsfm_0031"}

        def forecast(self, candidate, history, horizon, frequency):
            del history, frequency
            self.calls.append(candidate.method_id)
            if candidate.method_id == "method_tsfm_0014":
                return (10.0,) * horizon
            if failure_status == INVALID:
                return ()
            raise RuntimeError("transport failed")

    runtime = FailedLeafRuntime()
    registry = RuntimeRegistry({"timesfm": runtime, "tsfm_worker": runtime})
    cache = tmp_path / "cache"
    first = ForecastStore(
        cache, methods, None, module, portfolio, registry, "screen-hash"
    )
    try:
        assert first.forecast(
            "combined_failed_leaf_0", (1.0, 2.0), 2, "D"
        ) == (10.0, 10.0)
        assert first.forecast(
            "combined_failed_leaf_1", (1.0, 2.0), 2, "D"
        ) == (10.0, 10.0)
    finally:
        first.close()

    assert runtime.calls.count("method_tsfm_0031") == 1

    second = ForecastStore(
        cache, methods, None, module, portfolio, registry, "screen-hash"
    )
    try:
        assert second.forecast(
            "combined_failed_leaf_2", (1.0, 2.0), 2, "D"
        ) == (10.0, 10.0)
    finally:
        second.close()

    assert runtime.calls.count("method_tsfm_0031") == 2


def test_forecast_store_preserves_tsfm_invalid_and_crashed_parent_statuses(tmp_path):
    methods = tmp_path / "methods.py"
    methods.write_text(
        MODULE_HEADER
        + '''

def unused_statistical_leaf(history, horizon, frequency):
    """Satisfy the parsed module contract without entering this test graph."""
    return [0.0] * horizon
''',
        encoding="utf-8",
    )
    module = read_module(methods)
    base = PolicyPortfolio.flagship5()
    policy = CombinedPolicy(
        "combined_tsfm_failure_parity",
        ("toto_2_0", "timesfm_2_5"),
        "weighted_mean",
        (0.5, 0.5),
        fallback_parent="toto_2_0",
    )
    portfolio = PolicyPortfolio(base.tsfm, (policy,))

    class FailureRuntime:
        def __init__(
            self,
            toto_failure: Exception | None = None,
            timesfm_failure: Exception | None = None,
        ):
            self.toto_failure = toto_failure
            self.timesfm_failure = timesfm_failure

        def supports(self, candidate):
            return candidate.method_id in {"method_tsfm_0014", "method_tsfm_0031"}

        def forecast(self, candidate, history, horizon, frequency):
            del history, frequency
            if candidate.method_id == "method_tsfm_0014":
                if self.toto_failure is not None:
                    raise self.toto_failure
                return (10.0,) * horizon
            if candidate.method_id == "method_tsfm_0031":
                if self.timesfm_failure is not None:
                    raise self.timesfm_failure
                return ()
            raise ValueError("unexpected candidate")

    invalid_store = ForecastStore(
        tmp_path / "invalid-cache",
        methods,
        None,
        module,
        portfolio,
        RuntimeRegistry({"timesfm": FailureRuntime(), "tsfm_worker": FailureRuntime()}),
        "screen-hash",
    )
    try:
        invalid_parent = invalid_store._materialized_leaf_outcome(
            "timesfm_2_5", (1.0, 2.0), 2, "D"
        )
        fallback = invalid_store._materialized_leaf_outcome(
            "toto_2_0", (1.0, 2.0), 2, "D"
        )
        fallback_outcome = combine_materialized_outcome(
            policy,
            {"toto_2_0": fallback, "timesfm_2_5": invalid_parent},
            task_id="history-only",
            history=(1.0, 2.0),
            horizon=2,
            frequency="D",
        )
    finally:
        invalid_store.close()

    assert invalid_parent.status == INVALID
    assert fallback_outcome.status == SUCCESS
    assert fallback_outcome.forecast == (10.0, 10.0)

    provider_store = ForecastStore(
        tmp_path / "provider-cache",
        methods,
        None,
        module,
        portfolio,
        RuntimeRegistry(
            {
                "timesfm": FailureRuntime(timesfm_failure=ValueError("provider rejected request")),
                "tsfm_worker": FailureRuntime(timesfm_failure=ValueError("provider rejected request")),
            }
        ),
        "screen-hash",
    )
    try:
        crashed_parent = provider_store._materialized_leaf_outcome(
            "timesfm_2_5", (1.0, 2.0), 2, "D"
        )
        fallback = provider_store._materialized_leaf_outcome(
            "toto_2_0", (1.0, 2.0), 2, "D"
        )
        provider_fallback = combine_materialized_outcome(
            policy,
            {"toto_2_0": fallback, "timesfm_2_5": crashed_parent},
            task_id="history-only",
            history=(1.0, 2.0),
            horizon=2,
            frequency="D",
        )
    finally:
        provider_store.close()

    assert crashed_parent.status == CRASHED
    assert provider_fallback.status == SUCCESS
    assert provider_fallback.forecast == (10.0, 10.0)

    crashed_store = ForecastStore(
        tmp_path / "crashed-cache",
        methods,
        None,
        module,
        portfolio,
        RuntimeRegistry(
            {
                "timesfm": FailureRuntime(ValueError("provider rejected request")),
                "tsfm_worker": FailureRuntime(ValueError("provider rejected request")),
            }
        ),
        "screen-hash",
    )
    try:
        crashed_fallback = crashed_store._materialized_leaf_outcome(
            "toto_2_0", (1.0, 2.0), 2, "D"
        )
        invalid_parent = crashed_store._materialized_leaf_outcome(
            "timesfm_2_5", (1.0, 2.0), 2, "D"
        )
        failed = combine_materialized_outcome(
            policy,
            {"toto_2_0": crashed_fallback, "timesfm_2_5": invalid_parent},
            task_id="history-only",
            history=(1.0, 2.0),
            horizon=2,
            frequency="D",
        )
    finally:
        crashed_store.close()

    assert crashed_fallback.status == CRASHED
    assert invalid_parent.status == INVALID
    assert failed.status == CRASHED
    assert failed.forecast == ()


def test_selector_cli_requires_frozen_screen_and_has_no_test_option():
    parser = build_parser()
    args = parser.parse_args([
        "--repo", "repo", "--screening-dir", "screen", "--tasks-file", "tasks",
        "--outcome-cache-dir", "cache", "--policy-outcome-cache-dir", "pcache",
        "--hindcast-cache-dir", "hcache", "--output-dir", "out",
        "--train-limit", "80", "--dev-limit", "20",
    ])
    assert args.train_limit == 80 and args.dev_limit == 20
    assert args.train_validation_folds == 4
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--repo", "repo", "--screening-dir", "screen", "--tasks-file", "tasks",
            "--outcome-cache-dir", "cache", "--policy-outcome-cache-dir", "pcache",
            "--hindcast-cache-dir", "hcache", "--output-dir", "out",
            "--public-test-limit", "99",
        ])


def test_task_conditioned_audit_experiment_is_train_dev_only():
    module_name = "numerical_agent.run_task_conditioned_audit_experiment"
    assert importlib.util.find_spec(module_name) is not None
    module = importlib.import_module(module_name)
    parser = module.build_parser()
    args = parser.parse_args([
        "--repo", "repo",
        "--screening-dir", "screen",
        "--parent-selector-dir", "selector",
        "--tasks-file", "tasks",
        "--outcome-cache-dir", "cache",
        "--policy-outcome-cache-dir", "pcache",
        "--hindcast-cache-dir", "hcache",
        "--output-dir", "out",
    ])

    assert args.train_limit == 80
    assert args.dev_limit == 20
    assert args.train_validation_folds == 4
    assert args.candidate_family == "change-aware"
    assert "public_test_limit" not in {action.dest for action in parser._actions}

    train_only = parser.parse_args([
        "--repo", "repo",
        "--screening-dir", "screen",
        "--parent-selector-dir", "selector",
        "--tasks-file", "tasks",
        "--outcome-cache-dir", "cache",
        "--policy-outcome-cache-dir", "pcache",
        "--hindcast-cache-dir", "hcache",
        "--output-dir", "out",
        "--train-only",
    ])
    assert train_only.train_only


def test_protected_topk_experiment_can_use_one_read_only_dev_gate():
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )

    args = experiment.build_parser().parse_args([
        "--repo", "repo",
        "--screening-dir", "screen",
        "--parent-selector-dir", "selector",
        "--tasks-file", "tasks",
        "--outcome-cache-dir", "cache",
        "--policy-outcome-cache-dir", "pcache",
        "--hindcast-cache-dir", "hcache",
        "--output-dir", "out",
        "--candidate-family", "protected-topk",
    ])

    assert args.candidate_family == "protected-topk"
    assert args.train_only is False


def test_conservative_combined_cli_rejects_dev_evaluation_before_loading_files():
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )

    with pytest.raises(ValueError, match="requires --train-only"):
        experiment.main([
            "--repo", "missing-repo",
            "--screening-dir", "missing-screen",
            "--parent-selector-dir", "missing-selector",
            "--tasks-file", "missing-tasks",
            "--outcome-cache-dir", "missing-cache",
            "--policy-outcome-cache-dir", "missing-policy-cache",
            "--hindcast-cache-dir", "missing-hindcast-cache",
            "--output-dir", "missing-output",
            "--candidate-family", "conservative-combined",
        ])


def test_joint_portfolio_cli_rejects_dev_evaluation_before_loading_files():
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )

    with pytest.raises(ValueError, match="requires --train-only"):
        experiment.main([
            "--repo", "missing-repo",
            "--screening-dir", "missing-screen",
            "--parent-selector-dir", "missing-selector",
            "--tasks-file", "missing-tasks",
            "--outcome-cache-dir", "missing-cache",
            "--policy-outcome-cache-dir", "missing-policy-cache",
            "--hindcast-cache-dir", "missing-hindcast-cache",
            "--output-dir", "missing-output",
            "--candidate-family", "joint-portfolio",
        ])


def test_protected_portfolio_cli_rejects_dev_evaluation_before_loading_files():
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )

    with pytest.raises(ValueError, match="requires --train-only"):
        experiment.main([
            "--repo", "missing-repo",
            "--screening-dir", "missing-screen",
            "--parent-selector-dir", "missing-selector",
            "--tasks-file", "missing-tasks",
            "--outcome-cache-dir", "missing-cache",
            "--policy-outcome-cache-dir", "missing-policy-cache",
            "--hindcast-cache-dir", "missing-hindcast-cache",
            "--output-dir", "missing-output",
            "--candidate-family", "protected-portfolio",
        ])


def test_task_conditioned_audit_train_only_gate_never_evaluates_dev():
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    parent = DecisionPolicy()
    child = DecisionPolicy(
        baseline_strategy="conservative_tsfm",
        tsfm_router_blend_weight=0.25,
    )

    result = experiment._gate_winner_on_dev(
        parent,
        child,
        object(),
        object(),
        (object(),),
        train_only=True,
    )

    dev_parent, dev_child, gate, accepted, frozen = result
    assert dev_parent is None
    assert dev_child is None
    assert not gate.accepted
    assert "Train-only" in gate.reason
    assert not accepted
    assert frozen == parent


def test_task_conditioned_audit_entity_loading_is_limited_to_authorized_ids(
    tmp_path, monkeypatch
):
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
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
            raise AssertionError("Entity grouping opened a Dev record")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    assert experiment._load_entity_groups(task_dir, ("task_train_1",)) == {
        "task_train_1": "train-entity"
    }


def test_task_conditioned_audit_can_select_the_two_child_conservative_router():
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    parent = DecisionPolicy()

    candidates = experiment._candidate_policies(parent, "conservative-tsfm")

    assert len(candidates) == 3
    assert candidates[0] == parent
    assert {candidate.tsfm_router_min_improvement for candidate in candidates[1:]} == {0.02}
    assert {candidate.tsfm_router_blend_weight for candidate in candidates[1:]} == {
        0.1,
        0.25,
    }
    assert {
        experiment._route_payload(candidate)["blend_weight"]
        for candidate in candidates[1:]
    } == {0.1, 0.25}


def test_task_conditioned_audit_can_select_conservative_combined_children():
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    parent = DecisionPolicy()

    candidates = experiment._candidate_policies(parent, "conservative-combined")

    assert len(candidates) == 3
    assert candidates[0] == parent
    assert all(
        candidate.baseline_strategy == "conservative_combined"
        for candidate in candidates[1:]
    )
    assert {candidate.tsfm_router_blend_weight for candidate in candidates[1:]} == {
        0.1,
        0.25,
    }


def test_task_conditioned_audit_can_select_joint_portfolio_ablations():
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    parent = DecisionPolicy()

    candidates = experiment._candidate_policies(parent, "joint-portfolio")

    assert len(candidates) == 5
    assert [candidate.baseline_strategy for candidate in candidates[1:]] == [
        "conservative_single_tsfm",
        "conservative_tsfm_portfolio",
        "conservative_tsfm_statistical",
        "conservative_joint_portfolio",
    ]


def test_task_conditioned_audit_can_select_protected_r1_r2_r3():
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    parent = DecisionPolicy()

    candidates = experiment._candidate_policies(parent, "protected-portfolio")

    assert len(candidates) == 4
    assert [candidate.baseline_strategy for candidate in candidates[1:]] == [
        "protected_single_tsfm",
        "protected_tsfm_portfolio",
        "protected_joint_residual",
    ]


def test_joint_portfolio_route_reports_the_dedicated_two_percent_margin():
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    route = experiment._route_payload(DecisionPolicy(
        baseline_strategy="conservative_joint_portfolio",
        tsfm_router_min_improvement=0.02,
    ))

    assert route["minimum_improvement"] == pytest.approx(0.02)


def test_task_conditioned_audit_writes_a_manifest_accepted_by_frozen_evaluator(tmp_path):
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    from numerical_agent.evaluate_frozen_two_stage import verify_frozen_policies

    screen_dir = tmp_path / "screen"
    selector_dir = tmp_path / "selector"
    output_dir = tmp_path / "test-output"
    screen_dir.mkdir()
    selector_dir.mkdir()
    screen_source = "SCREENING_POLICY = {}\n"
    (screen_dir / "frozen_screening_policy.py").write_text(
        screen_source, encoding="utf-8"
    )
    screen_hash = hashlib.sha256(screen_source.encode()).hexdigest()
    binding = evolution_contracts.metric_policy_metadata()
    write_json(
        screen_dir / "screening_manifest.json",
        {
            "schema_version": 2,
            **binding,
            "frozen_screening_policy_sha256": screen_hash,
            "public_test_accessed": False,
        },
    )
    decision_source = "DECISION_POLICY = {}\n"
    (selector_dir / "frozen_decision_policy.py").write_text(
        decision_source, encoding="utf-8"
    )
    parent_manifest = {
        "schema_version": 2,
        **binding,
        "frozen_global_ranking": [f"method_{index}" for index in range(103)],
        "public_test_accessed": False,
    }

    experiment._write_selector_manifest(
        selector_dir,
        parent_manifest,
        screening_hash=screen_hash,
        accepted=True,
        train_tasks=80,
        dev_tasks=20,
        final_gate={"accepted": True, "reason": "passed"},
    )

    assert verify_frozen_policies(screen_dir, selector_dir, output_dir) == (
        screen_hash,
        hashlib.sha256(decision_source.encode()).hexdigest(),
    )
    manifest = json.loads((selector_dir / "selector_manifest.json").read_text())
    assert manifest["metric_policy_fingerprint"] == evolution_contracts.METRIC_POLICY_FINGERPRINT


def test_train_only_manifest_drops_inherited_dev_results(tmp_path):
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    output = tmp_path / "selector"
    output.mkdir()
    (output / "frozen_decision_policy.py").write_text(
        "DECISION_POLICY = {}\n", encoding="utf-8"
    )

    experiment._write_selector_manifest(
        output,
        {
            "schema_version": 2,
            **evolution_contracts.metric_policy_metadata(),
            "phase": "task_conditioned_numerical_selector",
            "frozen_global_ranking": ["toto_2_0"],
            "dev": {"mean_smae": 9.0},
            "dev_parent": {"mean_smae": 8.0},
            "dev_train_winner": {"mean_smae": 7.0},
            "generations": [{"accepted": True}],
            "elapsed_seconds": 123.0,
        },
        screening_hash="screen",
        accepted=False,
        train_tasks=80,
        dev_tasks=0,
        final_gate={"accepted": False, "reason": "Train-only"},
        train_only=True,
    )

    manifest = __import__("json").loads(
        (output / "selector_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dev_tasks"] == 0
    assert manifest["dev_accepted"] is False
    assert manifest["frozen_global_ranking"] == ["toto_2_0"]
    assert manifest["metric_policy_fingerprint"] == evolution_contracts.METRIC_POLICY_FINGERPRINT
    for stale in ("dev", "dev_parent", "dev_train_winner", "generations"):
        assert stale not in manifest


def test_adaptive_overlay_assignments_report_each_task_actual_weight(monkeypatch):
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    cases = (
        SimpleNamespace(task=SimpleNamespace(task_id="a")),
        SimpleNamespace(task=SimpleNamespace(task_id="b")),
    )
    decisions = iter((
        SimpleNamespace(
            combination_type="tsfm_shrinkage_overlay", weights=(0.9, 0.1)
        ),
        SimpleNamespace(combination_type=None, weights=(1.0,)),
    ))
    monkeypatch.setattr(experiment, "_select_case", lambda policy, case: next(decisions))

    assert experiment._adaptive_overlay_assignments(DecisionPolicy(), cases) == {
        "a": 0.1
    }


def test_statistical_overlay_assignments_report_anchor_specialist_and_weight(monkeypatch):
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    cases = (SimpleNamespace(task=SimpleNamespace(task_id="a")),)
    monkeypatch.setattr(
        experiment,
        "_select_case",
        lambda policy, case: SimpleNamespace(
            combination_type="statistical_shrinkage_overlay",
            selected=("toto_2_0", "seasonal_specialist"),
            weights=(0.75, 0.25),
        ),
    )

    assert experiment._statistical_overlay_assignments(DecisionPolicy(), cases) == {
        "a": {
            "anchor": "toto_2_0",
            "specialist": "seasonal_specialist",
            "specialist_weight": 0.25,
        }
    }


def test_joint_portfolio_assignments_report_all_tsfms_and_specialist(monkeypatch):
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    cases = (SimpleNamespace(task=SimpleNamespace(task_id="a")),)
    monkeypatch.setattr(
        experiment,
        "_select_case",
        lambda policy, case: SimpleNamespace(
            combination_type="joint_tsfm_statistical_portfolio",
            selected=("toto_2_0", "timesfm_2_5", "seasonal_specialist"),
            weights=(0.375, 0.375, 0.25),
        ),
    )

    assert experiment._joint_portfolio_assignments(DecisionPolicy(), cases) == {
        "a": {
            "tsfm_members": ["toto_2_0", "timesfm_2_5"],
            "tsfm_weights": [0.375, 0.375],
            "statistical_specialist": "seasonal_specialist",
            "statistical_weight": 0.25,
            "combination_type": "joint_tsfm_statistical_portfolio",
        }
    }


def test_protected_route_assignments_report_members_weights_and_kind(monkeypatch):
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    cases = (SimpleNamespace(task=SimpleNamespace(task_id="a")),)
    monkeypatch.setattr(
        experiment,
        "_select_case",
        lambda policy, case: SimpleNamespace(
            combination_type="protected_joint_tsfm_statistical_residual",
            selected=("chronos_bolt", "moirai_2_0", "seasonal_specialist"),
            weights=(0.6, 0.2, 0.2),
        ),
    )

    assert experiment._protected_route_assignments(DecisionPolicy(), cases) == {
        "a": {
            "selected": ["chronos_bolt", "moirai_2_0", "seasonal_specialist"],
            "weights": [0.6, 0.2, 0.2],
            "combination_type": "protected_joint_tsfm_statistical_residual",
        }
    }


def test_conservative_train_gate_requires_aggregate_smae_and_srmse_improvement():
    experiment = importlib.import_module(
        "numerical_agent.run_task_conditioned_audit_experiment"
    )
    parent = SimpleNamespace(mean_smae=0.30, mean_srmse=0.50)
    unsafe = SimpleNamespace(mean_smae=0.29, mean_srmse=0.51)
    safe = SimpleNamespace(mean_smae=0.29, mean_srmse=0.49)
    passed = experiment.DecisionGateResult(True, "aggregate gates passed")

    rejected = experiment._conservative_dual_metric_train_gate(
        parent, unsafe, passed
    )
    accepted = experiment._conservative_dual_metric_train_gate(
        parent, safe, passed
    )

    assert not rejected.accepted
    assert "sRMSE" in rejected.reason
    assert accepted.accepted


def test_selector_shell_forwards_runtime_and_freeze_inputs():
    script = ROOT / "scripts" / "run_numerical_selector_evolution.sh"
    completed = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    source = script.read_text(encoding="utf-8")
    for option in (
        "--screening-dir", "--hindcast-cache-dir", "--train-limit", "--dev-limit",
        "--generations", "--train-validation-folds", "--codex-model", "--tsfm-workers-config",
    ):
        assert option in source


def test_case_artifact_serializes_ineligible_infinite_diagnostics_as_sentinel(tmp_path):
    diagnostic = CandidateDiagnostics.synthetic(
        name="failed", family="statistical", median_mase=float("inf"), eligible=False
    )
    case = DecisionCase(
        Task("t", (1.0, 2.0, 3.0), 1, "D", (4.0,)),
        ("failed",),
        {"failed": diagnostic},
        {},
        {"failed": "statistical"},
    )
    target = tmp_path / "cases.jsonl"
    _write_cases(target, (case,))
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    evolution_contracts.require_active_metric_policy(payload)
    assert payload["diagnostics"]["failed"]["median_mase"] == {
        "status": "positive_infinity",
        "value": None,
    }


def test_case_artifact_preserves_entity_group_for_train_cross_validation(tmp_path):
    case = DecisionCase(
        Task("t", (1.0, 2.0, 3.0), 1, "D", (4.0,)),
        ("a",),
        {"a": CandidateDiagnostics.synthetic(
            name="a", family="statistical", median_mase=1.0
        )},
        {"a": (4.0,)},
        {"a": "statistical"},
        group_id="entity-safe",
    )
    target = tmp_path / "cases.jsonl"

    _write_cases(target, (case,))

    assert '"group_id": "entity-safe"' in target.read_text(encoding="utf-8")


def test_build_case_marks_only_matched_screening_specialists_as_conditioned():
    task = Task("t", tuple([0.0, 0.0, 4.0] * 12), 2, "D", (0.0, 4.0))
    screening = ScreeningPolicy(
        (
            ScreeningEntry(
                "specialist",
                "statistical",
                "specialized",
                ApplicabilityPolicy((ApplicabilityClause(("intermittent",)),)),
                "intermittent specialist",
            ),
            ScreeningEntry(
                "broad",
                "statistical",
                "keep",
                ApplicabilityPolicy(),
                "broad method",
            ),
        ),
        ("broad",),
    )

    class Store:
        @staticmethod
        def forecast(name, history, horizon, frequency):
            del name, frequency
            return (float(history[-1]),) * horizon

    outcomes = {
        (name, "t"): Outcome(name, "t", "success", forecast=(0.0, 0.0))
        for name in ("specialist", "broad")
    }

    case = _build_case(
        task,
        screening,
        screening.fingerprint(),
        outcomes,
        Store(),
        HindcastConfig(folds=3),
        group_id="entity-safe",
    )

    assert case.conditioned_names == ("specialist",)
    assert case.group_id == "entity-safe"


def test_build_case_always_preserves_reviewed_tsfm_anchors():
    task = Task("t", tuple(float(index) for index in range(1, 41)), 2, "D", (1.0, 2.0))
    screening = ScreeningPolicy((
        ScreeningEntry(
            "naive_last",
            "statistical",
            "keep",
            ApplicabilityPolicy(),
            "broad fallback",
        ),
    ), ("naive_last",))

    class Store:
        tsfm = {"toto_2_0": object(), "timesfm_2_5": object()}

        @staticmethod
        def forecast(name, history, horizon, frequency):
            del name, frequency
            return (history[-1],) * horizon

    by_key = {
        (name, "t"): Outcome(
            method=name,
            task_id="t",
            status="success",
            forecast=(40.0, 40.0),
        )
        for name in ("naive_last", "toto_2_0", "timesfm_2_5")
    }

    case = _build_case(
        task,
        screening,
        "screen-hash",
        by_key,
        Store(),
        HindcastConfig(folds=3),
    )

    assert case.active_names == ("naive_last", "toto_2_0", "timesfm_2_5")
    assert case.families["toto_2_0"] == "tsfm"
    assert case.families["timesfm_2_5"] == "tsfm"
    assert case.conditioned_names == ()


def test_global_ranking_uses_the_joint_scaled_metric_pair_and_penalizes_failures():
    rows = (
        Outcome("a", "t1", "success", smae=1.0, srmse=1.0),
        Outcome("a", "t2", "success", smae=1.0, srmse=1.0),
        Outcome("b", "t1", "success", smae=0.1, srmse=0.1),
        Outcome("b", "t2", "crashed"),
        Outcome("c", "t1", "success", smae=0.5, srmse=1.5),
        Outcome("c", "t2", "success", smae=0.5, srmse=1.5),
    )
    assert _global_ranking(rows, ("t1", "t2")) == ("a", "c", "b")


def test_selector_report_leads_with_drcik_point_metrics():
    case = DecisionCase(
        Task("t", (1.0, 2.0, 3.0), 1, "D", (2.0,)),
        ("a",),
        {"a": CandidateDiagnostics.synthetic(name="a", family="statistical", median_mase=1.0)},
        {"a": (3.0,)},
        {"a": "statistical"},
    )
    from dataclasses import asdict
    from numerical_agent.evolution.numerical_selector import DecisionPolicy
    from numerical_agent.evolution.selector_evolution import evaluate_decision

    score = asdict(evaluate_decision(DecisionPolicy(ensemble_enabled=False), (case,)))
    payload = {
        "train_tasks": 1,
        "dev_tasks": 1,
        "accepted_generations": [],
        "screening_policy_sha256": "screen",
        "frozen_decision_policy_sha256": "decision",
        "public_test_accessed": False,
        "dev_accepted": False,
        "final_dev_gate": {"accepted": False, "reason": "Dev sRMSE increased"},
        "paired_joint_wtl": {
            "train": {"wins": 1, "ties": 0, "losses": 0, "missing": 0, "unscored": 0},
            "dev": {"wins": 0, "ties": 0, "losses": 1, "missing": 0, "unscored": 0},
        },
        "train": score,
        "dev": score,
        **evolution_contracts.metric_report_metadata(),
    }
    report = _report(payload)

    assert payload["primary_metrics"] == ["smae", "srmse"]
    assert set(payload["diagnostic_only"]) >= {"mase", "mae", "smape", "rmsse"}
    assert "Mean sMAE" in report
    assert "Mean sRMSE" in report
    assert "Median sMAE" in report
    assert "Median sRMSE" in report
    assert "P90/P95 sMAE" in report
    assert "Clipped sMAE/sRMSE" in report
    assert "Wins / Ties / Losses / Missing / Unscored" in report
    assert "Assumptions" in report
    assert "Verifier pool" in report
    assert "Dev accepted: `False`" in report
    assert "Dev sRMSE increased" in report


def test_selector_paired_counts_conserve_both_missing_tasks():
    parent = SimpleNamespace(task_count=4, task_scaled_pairs={"same": (1.0, 1.0), "left": (1.0, 1.0)})
    child = SimpleNamespace(task_count=4, task_scaled_pairs={"same": (1.0, 1.0), "right": (1.0, 1.0)})

    counts = _score_pair_wtl(parent, child)

    assert counts == {"wins": 0, "ties": 1, "losses": 0, "missing": 2, "unscored": 1}
    assert sum(counts.values()) == 4
