"""Executable contract for the one-task morphology smoke command."""
from __future__ import annotations

import ast
from collections import Counter
import json
import hashlib
import inspect
import os
import subprocess
import sys
import textwrap
import threading
import types
from argparse import Namespace
from pathlib import Path

import pytest

import numerical_agent.run_morphology_smoke as smoke
from common.data import Task as DrCiKTask
from common.evolution_core import contracts
from numerical_agent import evaluate_frozen_two_stage, rescore_point_forecasts
from numerical_agent import run_selector_evolution
from numerical_agent.evolution import (
    assumptions,
    cache,
    combined_evolution,
    diagnostics,
    execution,
    filtering,
    morphology_consistency,
    morphology_credit,
    numerical_loop,
    numerical_package,
    numerical_selector,
    portfolio,
    screening,
    screening_evolution,
    selector_evolution,
)
from numerical_agent.evolution.portfolio import PolicyPortfolio, render_policy_source
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.run_morphology_smoke import main


def _record(task_id: str, *, future: list[float] | None = None) -> dict[str, object]:
    return {
        "benchmark_id": task_id,
        "series": {
            "history_values": [10.0, 11.0, 10.0, 11.0] * 12,
            "future_values": future if future is not None else [11.0, 10.0, 11.0],
        },
        "task_metadata": {"prediction_length": 3, "frequency": "D"},
        "labels_public": True,
    }


def _write_tasks(path, *records: dict[str, object]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


_LEGACY_PERFORMANCE_FIELDS = {
    "mase",
    "mae",
    "smape",
    "rmsse",
    "median_mase",
    "recent_mase",
    "worst_mase",
    "mase_mad",
    "median_mae",
    "median_smape",
    "median_rmsse",
    "mean_mase",
    "mean_mae",
    "mean_smape",
    "mean_rmsse",
    "oracle_mase",
    "catastrophic_mase",
    "mase_scale",
}


def _legacy_metric_operations(module) -> list[str]:
    """Return semantic legacy-metric operations for exact-node allowlisting."""
    tree = ast.parse(Path(inspect.getsourcefile(module)).read_text(encoding="utf-8"))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    operations: list[str] = []
    functions: list[str] = []

    def record(kind: str, field: str, node: ast.AST) -> None:
        statement = node
        while not isinstance(statement, ast.stmt):
            statement = parents[statement]
        normalized = ast.dump(statement, annotate_fields=True, include_attributes=False)
        statement_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        operations.append(
            f"{functions[-1] if functions else '<module>'}:{kind}:{field}@{statement_hash}"
        )

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            functions.append(node.name)
            self.generic_visit(node)
            functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr in _LEGACY_PERFORMANCE_FIELDS:
                record("attribute", node.attr, node)
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.slice, ast.Constant) and node.slice.value in _LEGACY_PERFORMANCE_FIELDS:
                record("subscript", str(node.slice.value), node)
            self.generic_visit(node)

        def visit_keyword(self, node: ast.keyword) -> None:
            if node.arg in _LEGACY_PERFORMANCE_FIELDS:
                record("keyword", str(node.arg), node)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in _LEGACY_PERFORMANCE_FIELDS
            ):
                record("name_call", node.func.id, node)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in _LEGACY_PERFORMANCE_FIELDS
            ):
                record("mapping_get", str(node.args[0].value), node)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in _LEGACY_PERFORMANCE_FIELDS
            ):
                record("getattr", str(node.args[1].value), node)
            self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> None:
            if node.value in _LEGACY_PERFORMANCE_FIELDS:
                record("string_config", str(node.value), node)

    Visitor().visit(tree)
    return operations


def _uses_default_metric_fingerprint(function) -> bool:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and len(node.args) > 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "metric_policy_fingerprint"
        for node in ast.walk(tree)
    )


def test_semantic_metric_audit_detects_dynamic_legacy_access_patterns(tmp_path) -> None:
    source = tmp_path / "dynamic_legacy.py"
    source.write_text(
        """
def active(mapping, diagnostic, truth, forecast):
    direct = mase(truth, forecast)
    mapped = mapping.get("smape")
    reflected = getattr(diagnostic, "rmsse")
    ranking_order = ("median_mae",)
    return direct, mapped, reflected, ranking_order
""",
        encoding="utf-8",
    )
    module = types.ModuleType("dynamic_legacy")
    module.__file__ = str(source)

    operations = {
        operation.rsplit("@", 1)[0]
        for operation in _legacy_metric_operations(module)
    }

    assert {
        "active:name_call:mase",
        "active:mapping_get:smape",
        "active:getattr:rmsse",
        "active:string_config:median_mae",
    } <= operations


def test_semantic_metric_audit_binds_allowance_to_the_exact_statement(tmp_path) -> None:
    diagnostic_source = tmp_path / "diagnostic_use.py"
    diagnostic_source.write_text(
        """
def active(score):
    observed = score.mean_mase
    return {"diagnostic": observed}
""",
        encoding="utf-8",
    )
    authority_source = tmp_path / "authority_use.py"
    authority_source.write_text(
        """
def active(score):
    if score.mean_mase < 1.0:
        return "accept"
    return "reject"
""",
        encoding="utf-8",
    )
    diagnostic_module = types.ModuleType("diagnostic_use")
    diagnostic_module.__file__ = str(diagnostic_source)
    authority_module = types.ModuleType("authority_use")
    authority_module.__file__ = str(authority_source)

    assert _legacy_metric_operations(diagnostic_module) != _legacy_metric_operations(
        authority_module
    )


def _allowed_statement(statement_hash: str, *operations: str) -> Counter[str]:
    return Counter(f"{operation}@{statement_hash}" for operation in operations)


def test_scaled_metric_contract_has_no_legacy_authority_in_active_morphology_paths() -> None:
    # Exact semantic-node allowlist. Every operation is bound to the normalized AST
    # hash of its enclosing statement, so repurposing an allowed diagnostic fails.
    allowed_diagnostic_or_legacy_reader_nodes = {
        cache: sum((
            _allowed_statement(
                "520fb57f169dbcc9",
                "_outcome_from_payload:attribute:mae",
                "_outcome_from_payload:attribute:mase",
                "_outcome_from_payload:attribute:smape",
            ),
            _allowed_statement(
                "05d8aea1701e7552",
                *(
                    f"_outcome_from_payload:{kind}:{field}"
                    for kind in ("keyword", "mapping_get", "string_config")
                    for field in ("mae", "mase", "smape")
                ),
            ),
        ), Counter()),
        diagnostics: sum((
            _allowed_statement(
                "0dbcb434449879c6",
                "_aggregate:string_config:mae",
                "_aggregate:string_config:mase",
                "_aggregate:string_config:smape",
            ),
            _allowed_statement(
                "d4444a06598cce63",
                *(
                    f"diagnose_forecasts:{kind}:{field}"
                    for kind in ("attribute", "string_config")
                    for field in ("mae", "mase", "smape")
                ),
            ),
        ), Counter()),
        execution: sum((
            _allowed_statement("bf3998705f24429e", "_report:attribute:mae"),
            _allowed_statement(
                "c1a439729c061faa",
                *(
                    f"_report:attribute:{field}"
                    for field in ("mae", "mase", "smape")
                ),
                *(
                    f"_report:keyword:{field}"
                    for field in ("mean_mae", "mean_mase", "mean_smape")
                ),
            ),
            _allowed_statement("4646b75ce3c7f3f1", "_report:attribute:mase"),
            _allowed_statement("656d7b1ec514f595", "_report:attribute:smape"),
            _allowed_statement(
                "006890beae4ad0e2",
                *(
                    f"_run_one:{kind}:{field}"
                    for kind in ("keyword", "name_call")
                    for field in ("mae", "mase", "smape")
                ),
            ),
            _allowed_statement(
                "56079bd6efbd80b8",
                *(
                    f"report_payload:{kind}:{field}"
                    for kind in ("attribute", "string_config")
                    for field in ("mean_mae", "mean_mase", "mean_smape")
                ),
            ),
        ), Counter()),
        numerical_selector: sum((
            _allowed_statement(
                "7d777ed564775a67",
                *(
                    f"_normalize_legacy_decision_ranking:string_config:{field}"
                    for field in ("median_mase", "recent_mase", "worst_mase")
                ),
            ),
            _allowed_statement(
                "e112c82a57aeda5e",
                *(
                    f"_score_fold:keyword:{field}"
                    for field in ("mae", "mase", "mase_scale", "rmsse", "smape")
                ),
                *(
                    f"_score_fold:name_call:{field}"
                    for field in ("mae", "mase", "smape")
                ),
            ),
            _allowed_statement(
                "bfd3922b9088ec02",
                *(
                    f"_selection_diagnostic:keyword:{field}"
                    for field in ("mase_mad", "median_mase", "recent_mase", "worst_mase")
                ),
            ),
            _allowed_statement(
                "a8c3ef77cce5cd2b",
                "_summarize:attribute:mase",
                *(
                    f"_summarize:keyword:{field}"
                    for field in (
                        "mase_mad", "median_mae", "median_mase", "median_rmsse",
                        "median_smape", "recent_mase", "worst_mase",
                    )
                ),
                *(
                    f"_summarize:string_config:{field}"
                    for field in ("mae", "rmsse", "smape")
                ),
            ),
            _allowed_statement("27a857442157da00", "_summarize:string_config:mase"),
            _allowed_statement("07d2bf27d46bbfb4", "from_payload:string_config:catastrophic_mase"),
            _allowed_statement("6c7a47156995f494", "from_payload:string_config:catastrophic_mase"),
            _allowed_statement(
                "e669f93211913e53",
                *(
                    f"from_payload:string_config:{field}"
                    for field in (
                        "catastrophic_mase", "mase_mad", "median_mase", "median_rmsse",
                        "median_smape", "recent_mase", "worst_mase",
                    )
                ),
            ),
            _allowed_statement(
                "fb9855c476512bc7",
                *(
                    f"from_payload:string_config:{field}"
                    for field in (
                        "mase_mad", "median_mase", "median_rmsse", "median_smape",
                        "recent_mase", "worst_mase",
                    )
                ),
            ),
            _allowed_statement("90b8dc8c04d5ff3f", "from_payload:string_config:median_smape"),
            _allowed_statement(
                "02016f55256c8018",
                *(
                    f"synthetic:keyword:{field}"
                    for field in (
                        "mase_mad", "median_mae", "median_mase", "median_rmsse",
                        "median_smape", "recent_mase", "worst_mase",
                    )
                ),
            ),
        ), Counter()),
        portfolio: _allowed_statement(
            "b473b2e31ed49e42",
            *(
                f"_scored:{kind}:{field}"
                for kind in ("keyword", "name_call")
                for field in ("mae", "mase", "smape")
            ),
        ),
        selector_evolution: sum((
            _allowed_statement(
                "e669f93211913e53",
                *(
                    f"_parse_policy:string_config:{field}"
                    for field in (
                        "catastrophic_mase", "mase_mad", "median_mase", "median_rmsse",
                        "median_smape", "recent_mase", "worst_mase",
                    )
                ),
            ),
            _allowed_statement("485cdc11a41e4a57", "_parse_policy:string_config:median_smape"),
            _allowed_statement(
                "7ce93841d63789ff",
                *(
                    f"_scaled_train_summary:string_config:{field}"
                    for field in (
                        "mean_mae", "mean_mase", "mean_smape", "median_mae", "median_mase",
                    )
                ),
            ),
            _allowed_statement(
                "f8410ad12e58d206",
                *(
                    f"evaluate_decision:keyword:{field}"
                    for field in (
                        "mean_mae", "mean_mase", "mean_smape", "median_mae", "median_mase",
                    )
                ),
            ),
            _allowed_statement("68f739f795a66307", "evaluate_decision:name_call:mae"),
            _allowed_statement("b904f68bebae9a01", "evaluate_decision:name_call:mase"),
            _allowed_statement("047c8a7f7c85d51d", "evaluate_decision:name_call:smape"),
        ), Counter()),
    }
    audited = (
        assumptions,
        cache,
        combined_evolution,
        diagnostics,
        execution,
        filtering,
        morphology_consistency,
        morphology_credit,
        numerical_loop,
        numerical_package,
        numerical_selector,
        portfolio,
        screening,
        screening_evolution,
        selector_evolution,
    )
    mismatches = {}
    for module in audited:
        observed = Counter(_legacy_metric_operations(module))
        allowed = allowed_diagnostic_or_legacy_reader_nodes.get(module, Counter())
        if observed != allowed:
            mismatches[module.__name__] = {
                "unexpected": dict(observed - allowed),
                "missing_allowlisted": dict(allowed - observed),
            }

    assert mismatches == {}


def test_scaled_metric_contract_renderers_require_a_bound_fingerprint() -> None:
    renderers = (
        evaluate_frozen_two_stage._report,
        rescore_point_forecasts.render_point_report,
        run_selector_evolution._report,
    )

    assert not any(_uses_default_metric_fingerprint(function) for function in renderers)


def test_ranked_alternatives_use_the_canonical_scaled_pair_tie_break_order() -> None:
    diagnostics = {
        "z_lower_smae": CandidateDiagnostics.synthetic(
            name="z_lower_smae",
            family="tsfm",
            median_mase=1.0,
            median_smae=0.9,
            median_srmse=1.1,
        ),
        "a_name_only": CandidateDiagnostics.synthetic(
            name="a_name_only",
            family="tsfm",
            median_mase=1.0,
            median_smae=1.0,
            median_srmse=1.0,
        ),
    }

    ranked = numerical_package.ranked_forecasts(
        active_names=("z_lower_smae", "a_name_only"),
        families={name: "tsfm" for name in diagnostics},
        diagnostics=diagnostics,
        forecasts={name: (1.0, 1.0) for name in diagnostics},
    )

    assert tuple(item.name for item in ranked) == ("z_lower_smae", "a_name_only")


def test_fake_smoke_selects_one_task_freezes_then_writes_complete_result(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "nested" / "smoke.json"
    _write_tasks(tasks, _record("one"))

    completed = subprocess.run(
        [
            sys.executable, "-m", "numerical_agent.run_morphology_smoke",
            "--task-file", str(tasks), "--results-path", str(result), "--llm-backend", "fake",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["task_id"] == "one"
    assert payload["schema_version"] == 2
    assert payload["metric_policy"] == {
        **contracts.METRIC_POLICY,
        "primary": list(contracts.METRIC_POLICY["primary"]),
    }
    assert payload["metric_policy_fingerprint"] == contracts.METRIC_POLICY_FINGERPRINT
    assert payload["primary_metrics"] == ["smae", "srmse"]
    assert set(payload["diagnostic_only"]) >= {"mase", "mae", "smape", "rmsse"}
    assert payload["selection"]["decision_metrics"] == ["smae", "srmse"]
    assert payload["evolution"] == {
        "dev_read_only": True,
        "public_hidden_mutation_enabled": False,
    }
    assert set(payload["execution_by_family"]) == {"statistical", "tsfm", "combined"}
    for summary in payload["execution_by_family"].values():
        assert summary["attempted"] == summary["successful"] + summary["unavailable"]
        assert set(summary) == {
            "attempted",
            "successful",
            "unavailable",
            "successful_candidates",
            "unavailable_candidates",
        }
    assert payload["execution_by_family"]["statistical"]["successful"] > 0
    assert payload["execution_by_family"]["tsfm"]["unavailable"] > 0
    assert payload["execution_by_family"]["combined"]["unavailable"] > 0
    assert len(payload["final_forecast"]) == 3
    assert set(payload) >= {
        "task_id", "selected", "final_forecast", "protected_baseline",
        "accepted_assumptions", "rejected_assumption_reason_counts",
        "selected_history_only_diagnostics", "baseline_history_only_diagnostics",
        "candidates", "morphology", "component_fingerprints", "freeze",
    }
    assert payload["freeze"]["forecast_frozen_before_labels"] is True
    assert set(payload["freeze"]["post_freeze_trusted_diagnostics"]) == {
        "mae", "mase", "smae", "srmse"
    }
    assert payload["morphology"]["call_status"] == "completed"
    assert all(
        set(item) == {"assumption_id", "kind", "claim", "failure_condition"}
        for item in payload["accepted_assumptions"]
    )
    assert payload["candidates"]["unavailable"]
    assert "toto_2_0" in {item["name"] for item in payload["candidates"]["unavailable"]}


def test_smoke_derives_decision_metrics_from_the_validated_policy(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "smoke.json"
    _write_tasks(tasks, _record("one"))
    decision = numerical_selector.DecisionPolicy(
        ranking_order=(
            "recent_joint_scaled_error",
            "median_joint_scaled_error",
            "worst_joint_scaled_error",
            "median_smae",
            "median_srmse",
            "normalized_bias",
        )
    )
    monkeypatch.setattr(smoke, "_decision_policy", lambda artifacts: decision)

    assert main([
        "--task-file", str(tasks),
        "--results-path", str(result),
        "--llm-backend", "fake",
    ]) == 0

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["selection"]["decision_ranking_order"] == list(
        decision.ranking_order
    )
    assert payload["selection"]["decision_metrics"] == ["smae", "srmse"]


def test_ambiguous_input_fails_before_result_or_model_work(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    _write_tasks(tasks, _record("one"), _record("two"))

    with pytest.raises(ValueError, match="--task-id"):
        main([
            "--task-file", str(tasks), "--results-path", str(result), "--llm-backend", "fake",
        ])
    assert not result.exists()


def test_task_id_scans_jsonl_ids_without_decoding_an_unselected_task_body(tmp_path, monkeypatch) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    _write_tasks(tasks, _record("unselected"), _record("selected"))
    decode = smoke.json.loads

    def selected_only(value, *args, **kwargs):
        if '"benchmark_id": "unselected"' in value:
            raise AssertionError("unselected task body was decoded")
        return decode(value, *args, **kwargs)

    monkeypatch.setattr(smoke.json, "loads", selected_only)
    assert main([
        "--task-file", str(tasks), "--task-id", "selected", "--results-path", str(result),
        "--llm-backend", "fake",
    ]) == 0


def test_real_mode_requires_an_explicit_reviewed_artifact_bundle(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks, _record("one"))

    with pytest.raises(smoke.SmokeError, match="--methods-path"):
        main(["--task-file", str(tasks), "--results-path", str(tmp_path / "result.json")])


def test_reviewed_artifacts_are_content_hashed_and_decision_binds_screening(tmp_path) -> None:
    screening = tmp_path / "screening.py"
    decision = tmp_path / "decision.py"
    methods = tmp_path / "methods.py"
    skills = tmp_path / "skills.py"
    policies = tmp_path / "policies.py"
    for path in (methods, skills, policies):
        path.write_text("reviewed\n", encoding="utf-8")
    screening.write_text("CANDIDATES = ()\nFALLBACK_NAMES = ()\n", encoding="utf-8")
    decision.write_text(
        f"SCREENING_POLICY_HASH = {'0' * 64!r}\nDECISION_POLICY = {{}}\n", encoding="utf-8"
    )
    args = Namespace(
        llm_backend="fake", methods_path=str(methods), skills_path=str(skills),
        policies_path=str(policies), screening_path=str(screening), decision_path=str(decision),
    )

    first_snapshots = smoke._ArtifactSnapshots.capture(args)
    try:
        first = dict(first_snapshots.fingerprints)
    finally:
        first_snapshots.close()
    methods.write_text("reviewed changed\n", encoding="utf-8")
    snapshots = smoke._ArtifactSnapshots.capture(args)
    try:
        assert snapshots.fingerprints["reviewed_methods"] != first["reviewed_methods"]
        with pytest.raises(smoke.SmokeError, match="SCREENING_POLICY_HASH"):
            smoke._decision_policy(snapshots)
    finally:
        snapshots.close()


def test_non_overwrite_result_creation_has_one_winner_under_a_race(tmp_path) -> None:
    path = tmp_path / "nested" / "result.json"
    start = threading.Barrier(2)
    results: list[object] = []

    def write() -> None:
        start.wait()
        try:
            smoke._write_result(path, {"value": 1}, overwrite=False)
        except Exception as error:  # the losing race is the assertion target
            results.append(error)
        else:
            results.append("written")

    threads = [threading.Thread(target=write), threading.Thread(target=write)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count("written") == 1
    assert sum(isinstance(value, FileExistsError) for value in results) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_nonfinite_post_freeze_metrics_are_rejected_before_json_encoding() -> None:
    with pytest.raises(smoke.SmokeError, match="non-finite"):
        smoke._post_freeze_metrics(
            (1e308, -1e308, 1e308),
            (1e308, -1e308),
            (-1e308, 1e308),
        )


def test_labels_are_withheld_from_the_numerical_path_until_post_freeze_scoring(tmp_path, monkeypatch) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    _write_tasks(tasks, _record("one", future=[99.0, 98.0, 97.0]))
    original = smoke.run_numerical_loop
    original_future = smoke._future_values_after_freeze
    events: list[str] = []

    def history_only(task, **kwargs):
        assert task.future == ()
        package = original(task, **kwargs)
        events.append("frozen")
        return package

    def labels_after_freeze(record, horizon):
        assert events == ["frozen"]
        return original_future(record, horizon)

    monkeypatch.setattr(smoke, "run_numerical_loop", history_only)
    monkeypatch.setattr(smoke, "_future_values_after_freeze", labels_after_freeze)
    assert main([
        "--task-file", str(tasks), "--results-path", str(result), "--llm-backend", "fake",
    ]) == 0
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["freeze"]["post_freeze_trusted_diagnostics"]["mae"] > 0.0


def test_selected_future_json_is_not_decoded_until_after_package_freeze(tmp_path, monkeypatch) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    _write_tasks(tasks, _record("one", future=[99.0, 98.0, 97.0]))
    decode = smoke.json.loads
    original = smoke.run_numerical_loop
    frozen = False

    def guarded_decode(value, *args, **kwargs):
        if '"future_values": [99.0, 98.0, 97.0]' in value and not frozen:
            raise AssertionError("future labels were decoded before package freeze")
        return decode(value, *args, **kwargs)

    def freeze(task, **kwargs):
        nonlocal frozen
        package = original(task, **kwargs)
        frozen = True
        return package

    monkeypatch.setattr(smoke.json, "loads", guarded_decode)
    monkeypatch.setattr(smoke, "run_numerical_loop", freeze)
    assert main([
        "--task-file", str(tasks), "--results-path", str(result), "--llm-backend", "fake",
    ]) == 0


@pytest.mark.parametrize("reordered", [False, True])
def test_escaped_future_key_is_structurally_masked_before_freeze(
    tmp_path, monkeypatch, reordered
) -> None:
    result = tmp_path / "result.json"
    values = [99.0, 98.0, 97.0]
    record = _record("one", future=values)
    if reordered:
        record = {
            "task_metadata": record["task_metadata"],
            "series": {
                "future_values": values,
                "history_values": record["series"]["history_values"],
            },
            "labels_public": True,
            "benchmark_id": "one",
        }
    raw = json.dumps(record).replace(
        '"future_values":', '"future\\u005fvalues"   :'
    )
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(raw + "\n", encoding="utf-8")
    decode = smoke.json.loads
    original = smoke.run_numerical_loop
    frozen = False

    def guarded_decode(value, *args, **kwargs):
        if (
            'future\\u005fvalues' in value
            and '[99.0, 98.0, 97.0]' in value
            and not frozen
        ):
            raise AssertionError("escaped future labels were decoded before package freeze")
        return decode(value, *args, **kwargs)

    def freeze(task, **kwargs):
        nonlocal frozen
        package = original(task, **kwargs)
        frozen = True
        return package

    monkeypatch.setattr(smoke.json, "loads", guarded_decode)
    monkeypatch.setattr(smoke, "run_numerical_loop", freeze)
    assert main(
        [
            "--task-file", str(tasks), "--results-path", str(result), "--llm-backend", "fake",
        ]
    ) == 0


def test_selected_task_uses_the_common_history_only_task_model(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks, _record("one"))

    selected = smoke._select_one_task(tasks, None)

    assert isinstance(selected.task, DrCiKTask)
    assert selected.task.future_values == ()


def test_task_id_rejects_traversal_and_decoded_id_mismatches(tmp_path) -> None:
    directory = tmp_path / "tasks"
    directory.mkdir()
    (tmp_path / "outside.json").write_text(json.dumps(_record("outside")), encoding="utf-8")
    with pytest.raises(smoke.SmokeError, match="task-id"):
        smoke._select_one_task(directory, "../outside")

    mismatch = {"nested": {"benchmark_id": "selected"}, **_record("other")}
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks, mismatch)
    with pytest.raises(smoke.SmokeError, match="benchmark_id"):
        smoke._select_one_task(tasks, "selected")


def test_artifact_snapshot_binds_the_hashed_bytes_to_execution_input(tmp_path) -> None:
    paths = {}
    for name in ("methods", "skills", "policies", "screening", "decision"):
        path = tmp_path / f"{name}.py"
        path.write_text(f"{name}-original\n", encoding="utf-8")
        paths[f"{name}_path"] = str(path)
    Path(paths["policies_path"]).write_text(
        render_policy_source(PolicyPortfolio.flagship5()), encoding="utf-8"
    )
    snapshots = smoke._ArtifactSnapshots.capture(Namespace(llm_backend="fake", **paths))
    try:
        Path(paths["methods_path"]).write_text("methods-mutated\n", encoding="utf-8")
        Path(paths["policies_path"]).write_text("not valid policy source\n", encoding="utf-8")
        assert snapshots.text("reviewed_methods") == "methods-original\n"
        assert snapshots.fingerprints["reviewed_methods"] == hashlib.sha256(
            b"methods-original\n"
        ).hexdigest()
        assert smoke._portfolio(snapshots).names == PolicyPortfolio.flagship5().names
    finally:
        snapshots.close()


def test_task_id_selects_exactly_one_record_and_overwrite_is_explicit(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    _write_tasks(tasks, _record("one"), _record("two"))

    assert main([
        "--task-file", str(tasks), "--task-id", "two", "--results-path", str(result),
        "--llm-backend", "fake",
    ]) == 0
    assert json.loads(result.read_text(encoding="utf-8"))["task_id"] == "two"
    with pytest.raises(FileExistsError, match="--overwrite"):
        main([
            "--task-file", str(tasks), "--task-id", "two", "--results-path", str(result),
            "--llm-backend", "fake",
        ])
    assert main([
        "--task-file", str(tasks), "--task-id", "two", "--results-path", str(result),
        "--llm-backend", "fake", "--overwrite",
    ]) == 0


@pytest.mark.parametrize(
    "input_kind",
    ("task", "methods", "skills", "policies", "screening", "decision", "worker_config"),
)
@pytest.mark.parametrize("alias_kind", ("direct", "hardlink", "symlink"))
def test_overwrite_rejects_task_and_configuration_identity_aliases_before_model_work(
    tmp_path, monkeypatch, input_kind, alias_kind
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks, _record("one"))
    target = tasks
    argv = ["--task-file", str(tasks), "--llm-backend", "fake", "--overwrite"]
    if input_kind != "task":
        target = tmp_path / f"{input_kind}.input"
        target.write_text("input must not be replaced\n", encoding="utf-8")
        option = {
            "methods": "--methods-path",
            "skills": "--skills-path",
            "policies": "--policies-path",
            "screening": "--screening-path",
            "decision": "--decision-path",
            "worker_config": "--tsfm-workers-config",
        }[input_kind]
        argv.extend((option, str(target)))
    if alias_kind == "direct":
        result = target
    else:
        result = tmp_path / f"{input_kind}-{alias_kind}.json"
        if alias_kind == "hardlink":
            os.link(target, result)
        else:
            result.symlink_to(target)
    before = target.read_bytes()
    model_ran = False
    runtime_initialized = False

    def should_not_run(*_args, **_kwargs):
        nonlocal model_ran
        model_ran = True
        raise AssertionError("model/runtime work must not start for output collisions")

    def should_not_initialize_runtime(*_args, **_kwargs):
        nonlocal runtime_initialized
        runtime_initialized = True
        raise AssertionError("runtime setup must not start for output collisions")

    monkeypatch.setattr(smoke, "run_numerical_loop", should_not_run)
    monkeypatch.setattr(smoke, "_smoke_runtime_registry", should_not_initialize_runtime)
    with pytest.raises(smoke.SmokeError, match="results path aliases"):
        main([*argv, "--results-path", str(result)])
    assert target.read_bytes() == before
    assert not model_ran
    assert not runtime_initialized


def test_absent_worker_interpreter_leaves_worker_tsfms_unavailable(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    config = tmp_path / "workers.json"
    _write_tasks(tasks, _record("one"))
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "environments": {
                    "uni2ts": {"interpreter": str(tmp_path / "absent-python")}
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--task-file", str(tasks), "--results-path", str(result),
            "--llm-backend", "fake", "--tsfm-workers-config", str(config),
        ]
    ) == 0
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert {item["name"] for item in payload["candidates"]["unavailable"]} >= {
        "moirai_2_0",
    }


def test_malformed_worker_configuration_remains_fatal_in_smoke_mode(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    config = tmp_path / "workers.json"
    _write_tasks(tasks, _record("one"))
    config.write_text("not JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="TSFM deployment must be valid JSON"):
        main(
            [
                "--task-file", str(tasks), "--results-path", str(tmp_path / "result.json"),
                "--llm-backend", "fake", "--tsfm-workers-config", str(config),
            ]
        )


@pytest.mark.parametrize("task_file, results_path", [("missing.jsonl", "out.json"), ("bad.json", "")])
def test_malformed_paths_fail_cleanly(tmp_path, task_file, results_path) -> None:
    if task_file == "bad.json":
        (tmp_path / task_file).write_text("not-json\n", encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError), match="task|result|JSON|path"):
        main([
            "--task-file", str(tmp_path / task_file),
            "--results-path", str(tmp_path / results_path),
            "--llm-backend", "fake",
        ])
