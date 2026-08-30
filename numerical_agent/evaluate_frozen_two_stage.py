"""Evaluate frozen screening/selector policies exactly once on 99 Public Test tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from common.data import load_tasks
from common.evolution_core.contracts import (
    METRIC_POLICY_FINGERPRINT,
    metric_policy_metadata,
    metric_report_metadata,
    require_active_metric_policy,
)
from common.metrics import (
    drcik_point_metrics,
    linear_quantile,
    mae,
    mase,
    smape,
    standard_error,
)
from common.payload import read_json_object, write_json

from .evolution.execution import SUCCESS, Task
from .evolution.filtering import build_filter_dictionary
from .evolution.module import read_module
from .evolution.numerical_selector import (
    DecisionPolicy,
    HindcastConfig,
    select_assumption_guided_forecast,
)
from .evolution.portfolio import read_policy_file
from .evolution.screening_evolution import migrate_filter_dictionary, parse_screening_source
from .evolution.screening import profile_task
from .evolution.selector_evolution import parse_decision_source
from .main import _add_tsfm_runtime_options, _runtime_registry
from .run_selector_evolution import ForecastStore, _build_case
from .run_task_conditioned_screening import _training_outcomes


@dataclass(frozen=True)
class ForecastResult:
    task_id: str
    forecast: tuple[float, ...]
    selected: tuple[str, ...]
    families: tuple[str, ...]
    mode: str
    oracle_mase: float | None = None
    assumption_ids: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--screening-dir", required=True)
    parser.add_argument("--selector-dir", required=True)
    parser.add_argument("--split-file", default="splits/drcik_public_80_20_99_v1.json")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--outcome-cache-dir", required=True)
    parser.add_argument("--policy-outcome-cache-dir", required=True)
    parser.add_argument("--hindcast-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    _add_tsfm_runtime_options(parser)
    return parser


def verify_frozen_policies(
    screening_dir: str | Path,
    selector_dir: str | Path,
    output_dir: str | Path,
) -> tuple[str, str]:
    screen_dir = Path(screening_dir)
    decision_dir = Path(selector_dir)
    output = Path(output_dir)
    if (output / "evaluation_complete.json").exists():
        raise ValueError("Public Test evaluation has already completed")
    screen_manifest = read_json_object(screen_dir / "screening_manifest.json")
    selector_manifest = read_json_object(decision_dir / "selector_manifest.json")
    require_active_metric_policy(screen_manifest, context="active screening release")
    require_active_metric_policy(selector_manifest, context="active selector release")
    if screen_manifest.get("schema_version") != 2:
        raise ValueError("active screening release schema_version must be 2")
    if selector_manifest.get("schema_version") != 2:
        raise ValueError("active selector release schema_version must be 2")
    screen_hash = _sha256(screen_dir / "frozen_screening_policy.py")
    decision_hash = _sha256(decision_dir / "frozen_decision_policy.py")
    if screen_manifest.get("frozen_screening_policy_sha256") != screen_hash:
        raise ValueError("screening policy hash mismatch")
    if selector_manifest.get("screening_policy_sha256") != screen_hash:
        raise ValueError("decision policy is bound to a different screening policy")
    if selector_manifest.get("frozen_decision_policy_sha256") != decision_hash:
        raise ValueError("decision policy hash mismatch")
    if screen_manifest.get("public_test_accessed") is not False:
        raise ValueError("screening manifest does not certify an unopened Public Test")
    if selector_manifest.get("public_test_accessed") is not False:
        raise ValueError("selector manifest does not certify an unopened Public Test")
    return screen_hash, decision_hash


def score_forecast_results(
    tasks: Sequence[Task], results: Sequence[ForecastResult]
) -> dict[str, object]:
    by_id = {result.task_id: result for result in results}
    task_scores = []
    selected = set()
    families = set()
    ensembles = 0
    for task in tasks:
        result = by_id.get(task.task_id)
        if result is None or len(result.forecast) != task.horizon:
            continue
        truth = list(task.future)
        prediction = list(result.forecast)
        task_mase = mase(truth, prediction, list(task.history))
        point = drcik_point_metrics(truth, prediction)
        task_scores.append({
            "task_id": task.task_id,
            "mase": task_mase,
            **point,
            "smape": smape(truth, prediction),
            "rmsse": _rmsse(task.history, task.future, result.forecast),
            "oracle_regret": (
                (task_mase - result.oracle_mase) / (1.0 + result.oracle_mase)
                if result.oracle_mase is not None else 0.0
            ),
        })
        selected.update(result.selected)
        families.update(result.families)
        ensembles += result.mode == "ensemble"
    count = len(task_scores)

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in task_scores]

    mases = values("mase")
    smaes = values("smae")
    srmses = values("srmse")
    smaes_raw = values("smae_raw")
    srmses_raw = values("srmse_raw")
    smae_clipped_count = sum(bool(row["smae_clipped"]) for row in task_scores)
    srmse_clipped_count = sum(bool(row["srmse_clipped"]) for row in task_scores)
    return {
        "task_count": len(tasks),
        "coverage": count / len(tasks) if tasks else 0.0,
        "mean_mase": statistics.fmean(mases) if mases else math.inf,
        "median_mase": statistics.median(mases) if mases else math.inf,
        "mean_smae": statistics.fmean(smaes) if smaes else math.inf,
        "median_smae": statistics.median(smaes) if smaes else math.inf,
        "se_smae": standard_error(smaes) if smaes else math.inf,
        "mean_srmse": statistics.fmean(srmses) if srmses else math.inf,
        "median_srmse": statistics.median(srmses) if srmses else math.inf,
        "se_srmse": standard_error(srmses) if srmses else math.inf,
        "p90_smae": linear_quantile(smaes, 0.90) if smaes else math.inf,
        "p95_smae": linear_quantile(smaes, 0.95) if smaes else math.inf,
        "p90_srmse": linear_quantile(srmses, 0.90) if srmses else math.inf,
        "p95_srmse": linear_quantile(srmses, 0.95) if srmses else math.inf,
        "p90_smae_raw": linear_quantile(smaes_raw, 0.90) if smaes_raw else math.inf,
        "p95_smae_raw": linear_quantile(smaes_raw, 0.95) if smaes_raw else math.inf,
        "p90_srmse_raw": linear_quantile(srmses_raw, 0.90) if srmses_raw else math.inf,
        "p95_srmse_raw": linear_quantile(srmses_raw, 0.95) if srmses_raw else math.inf,
        "smae_clipped_count": smae_clipped_count,
        "smae_clipped_rate": smae_clipped_count / count if count else 1.0,
        "srmse_clipped_count": srmse_clipped_count,
        "srmse_clipped_rate": srmse_clipped_count / count if count else 1.0,
        "mean_rmsse": statistics.fmean(values("rmsse")) if mases else math.inf,
        "median_rmsse": statistics.median(values("rmsse")) if mases else math.inf,
        "mean_mae": statistics.fmean(values("mae")) if mases else math.inf,
        "median_mae": statistics.median(values("mae")) if mases else math.inf,
        "mean_smape": statistics.fmean(values("smape")) if mases else math.inf,
        "median_smape": statistics.median(values("smape")) if mases else math.inf,
        "catastrophic_rate": sum(value > 10.0 for value in mases) / count if count else 1.0,
        "mean_oracle_regret": statistics.fmean(values("oracle_regret")) if mases else math.inf,
        "method_diversity": len(selected),
        "family_diversity": len(families),
        "ensemble_rate": ensembles / count if count else 0.0,
        "per_task": task_scores,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    screening_hash, decision_hash = verify_frozen_policies(
        args.screening_dir, args.selector_dir, output
    )
    tasks = _public_test_tasks(args.split_file, args.tasks_file)
    if len(tasks) != 99:
        raise ValueError(f"frozen evaluation requires exactly 99 Public Test tasks, got {len(tasks)}")

    repo = Path(args.repo).resolve()
    screen_dir = Path(args.screening_dir).resolve()
    selector_dir = Path(args.selector_dir).resolve()
    module = read_module(repo / "methods.py")
    portfolio = read_policy_file(repo / "policies.py")
    portfolio.validate_namespace(module.names())
    candidate_names = tuple(module.names()) + portfolio.names
    screening = parse_screening_source(
        (screen_dir / "frozen_screening_policy.py").read_text(encoding="utf-8")
    )
    all_screening = migrate_filter_dictionary(
        build_filter_dictionary(module, portfolio),
        fallback_names=("naive_last", "timesfm_2_5", "toto_2_0"),
    )
    decision_policy = parse_decision_source(
        (selector_dir / "frozen_decision_policy.py").read_text(encoding="utf-8")
    )
    selector_manifest = read_json_object(selector_dir / "selector_manifest.json")
    ranking = tuple(str(name) for name in selector_manifest.get("frozen_global_ranking", ()))
    _validate_frozen_ranking(ranking, candidate_names)

    outcomes, final_cache = _training_outcomes(args, repo, module, portfolio, tasks)
    by_key = {(row.method, row.task_id): row for row in outcomes}
    runtimes = _runtime_registry(args)
    try:
        store = ForecastStore(
            args.hindcast_cache_dir,
            repo / "methods.py",
            repo / "skills.py" if (repo / "skills.py").is_file() else None,
            module,
            portfolio,
            runtimes,
            screening_hash,
        )
        try:
            config = _hindcast_config_for_policy(decision_policy)
            all_cases = tuple(
                _build_case(task, all_screening, screening_hash, by_key, store, config)
                for task in tasks
            )
            screened_cases = tuple(
                _build_case(task, screening, screening_hash, by_key, store, config)
                for task in tasks
            )
        finally:
            store.close()
    finally:
        runtimes.close()

    rows = {
        "A_current_global_ranker": tuple(
            _ranked_forecast(case, ranking, active_only=False) for case in all_cases
        ),
        "B_screening_only": tuple(
            _ranked_forecast(case, ranking, active_only=True) for case in screened_cases
        ),
        "C_decision_only": tuple(
            _selector_forecast(case, decision_policy) for case in all_cases
        ),
        "D_full_two_stage": tuple(
            _selector_forecast(case, decision_policy) for case in screened_cases
        ),
        "E_toto_reference": tuple(
            _fixed_forecast(case, "toto_2_0") for case in all_cases
        ),
    }
    scores = {name: score_forecast_results(tasks, results) for name, results in rows.items()}
    baseline_by_task = {
        row["task_id"]: (row["smae"], row["srmse"])
        for row in scores["A_current_global_ranker"]["per_task"]
    }
    paired = {
        name: _paired_counts(score["per_task"], baseline_by_task)
        for name, score in scores.items()
    }
    with (output / "per_task_results.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            payload = {
                **metric_policy_metadata(),
                "task_id": task.task_id,
                "rows": {
                    name: asdict(next(result for result in results if result.task_id == task.task_id))
                    for name, results in rows.items()
                },
            }
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    report_payload = {
        "schema_version": 2,
        **metric_report_metadata(),
        "task_count": 99,
        "screening_policy_sha256": screening_hash,
        "decision_policy_sha256": decision_hash,
        "rows": scores,
        "paired_vs_A": paired,
        "cache": {
            **final_cache,
            "hindcast_hits": store.hits,
            "hindcast_misses": store.misses,
        },
        "elapsed_seconds": time.monotonic() - started,
        "mutation_calls": 0,
        "llm_calls": 0,
    }
    write_json(output / "frozen_two_stage_results.json", report_payload)
    (output / "FINAL_TWO_STAGE_REPORT.md").write_text(
        _report(report_payload), encoding="utf-8"
    )
    # This marker is written last. Its presence makes the one-time command refuse a retry.
    write_json(output / "evaluation_complete.json", {
        "schema_version": 2,
        **metric_policy_metadata(),
        "task_count": 99,
        "screening_policy_sha256": screening_hash,
        "decision_policy_sha256": decision_hash,
        "results_sha256": _sha256(output / "frozen_two_stage_results.json"),
        "per_task_sha256": _sha256(output / "per_task_results.jsonl"),
    })
    print(json.dumps({
        "task_count": 99,
        "rows": {name: {key: value for key, value in score.items() if key != "per_task"}
                 for name, score in scores.items()},
        "paired_vs_A": paired,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _validate_frozen_ranking(
    ranking: Sequence[str], candidate_names: Sequence[str]
) -> None:
    """Require one unique ranking over the complete runtime namespace."""
    names = tuple(ranking)
    expected = set(candidate_names)
    actual = set(names)
    duplicates = tuple(sorted(name for name in actual if names.count(name) > 1))
    missing = tuple(sorted(expected - actual))
    extra = tuple(sorted(actual - expected))
    if duplicates or missing or extra:
        details = []
        if duplicates:
            details.append("duplicate=" + ", ".join(duplicates))
        if missing:
            details.append("missing=" + ", ".join(missing))
        if extra:
            details.append("extra=" + ", ".join(extra))
        raise ValueError("selector ranking namespace mismatch: " + "; ".join(details))


def _public_test_tasks(split_file: str | Path, tasks_file: str | Path) -> tuple[Task, ...]:
    split = read_json_object(split_file)
    ids = split["partitions"]["public_test"]["task_ids"]  # type: ignore[index]
    catalog = {task.task_id: task for task in load_tasks(tasks_file)}
    rows = []
    for task_id in ids:
        source = catalog.get(str(task_id))
        if source is None:
            raise ValueError(f"missing Public Test task {task_id}")
        rows.append(Task(
            source.task_id, tuple(source.history_values), source.prediction_length,
            source.frequency, tuple(source.future_values),
        ))
    return tuple(rows)


def _hindcast_config_for_policy(policy: DecisionPolicy) -> HindcastConfig:
    return HindcastConfig(
        folds=3,
        long_horizon_audit=(
            policy.long_horizon_audit_enabled or policy.long_horizon_guard_enabled
            or policy.baseline_strategy in {
                "conservative_tsfm",
                "conservative_combined",
                "conservative_single_tsfm",
                "conservative_tsfm_portfolio",
                "conservative_tsfm_statistical",
                "conservative_joint_portfolio",
            }
        ),
    )


def _ranked_forecast(case, ranking: Sequence[str], *, active_only: bool) -> ForecastResult:
    allowed = set(case.active_names) if active_only else set(case.forecasts)
    name = next((name for name in ranking if name in allowed and name in case.forecasts), None)
    if name is None:
        return ForecastResult(case.task.task_id, (), (), (), "missing")
    return _result(case, (name,), (1.0,), tuple(case.forecasts[name]), "single")


def _selector_forecast(case, policy: DecisionPolicy) -> ForecastResult:
    try:
        decision = select_assumption_guided_forecast(
            policy,
            profile=profile_task(case.task),
            active_names=case.active_names,
            diagnostics=case.diagnostics,
            forecasts=case.forecasts,
            families=case.families,
            history=case.task.history,
            conditioned_names=case.conditioned_names,
        )
    except (ValueError, KeyError):
        return ForecastResult(case.task.task_id, (), (), (), "missing")
    return _result(
        case,
        decision.selected,
        decision.weights,
        decision.forecast,
        decision.mode,
        assumption_ids=decision.assumption_ids,
    )


def _fixed_forecast(case, name: str) -> ForecastResult:
    if name not in case.forecasts:
        return ForecastResult(case.task.task_id, (), (), (), "missing")
    return _result(case, (name,), (1.0,), tuple(case.forecasts[name]), "single")


def _result(case, selected, weights, forecast, mode, *, assumption_ids=()) -> ForecastResult:
    del weights
    truth = list(case.task.future)
    oracle = min(
        (
            mase(truth, list(values), list(case.task.history))
            for name, values in case.forecasts.items()
            if name in case.active_names and len(values) == case.task.horizon
        ),
        default=None,
    )
    return ForecastResult(
        case.task.task_id,
        tuple(float(value) for value in forecast),
        tuple(selected),
        tuple(case.families.get(name, "unknown") for name in selected),
        mode,
        oracle,
        tuple(assumption_ids),
    )


def _paired_counts(
    per_task, baseline: Mapping[str, tuple[float, float]]
) -> dict[str, int]:
    result = {"wins": 0, "ties": 0, "losses": 0, "missing": 0}
    seen = set()
    for row in per_task:
        task_id = str(row["task_id"])
        seen.add(task_id)
        if task_id not in baseline:
            result["missing"] += 1
            continue
        baseline_pair = baseline[task_id]
        delta = (
            (float(row["smae"]) + float(row["srmse"]))
            - (float(baseline_pair[0]) + float(baseline_pair[1]))
        ) / 2.0
        if abs(delta) <= 1e-12:
            result["ties"] += 1
        elif delta < 0:
            result["wins"] += 1
        else:
            result["losses"] += 1
    result["missing"] += len(set(baseline) - seen)
    return result


def _rmsse(history: Sequence[float], truth: Sequence[float], forecast: Sequence[float]) -> float:
    diffs = [(history[index] - history[index - 1]) ** 2 for index in range(1, len(history))]
    scale = statistics.fmean(diffs) if diffs else 0.0
    if scale <= 1e-8:
        scale = 1.0
    error = statistics.fmean((actual - predicted) ** 2 for actual, predicted in zip(truth, forecast))
    return math.sqrt(error / scale)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _report(payload: Mapping[str, object]) -> str:
    scores = payload["rows"]
    paired = payload["paired_vs_A"]
    assert isinstance(scores, Mapping) and isinstance(paired, Mapping)
    lines = [
        "# Final Frozen Two-Stage Public Test Report",
        "",
        f"- Tasks: {payload['task_count']}",
        f"- Screening SHA-256: `{payload['screening_policy_sha256']}`",
        f"- Decision SHA-256: `{payload['decision_policy_sha256']}`",
        f"- Metric policy SHA-256: `{payload.get('metric_policy_fingerprint', METRIC_POLICY_FINGERPRINT)}`",
        "- Primary metrics: sMAE, sRMSE",
        "- Diagnostic only: MASE, MAE, sMAPE, RMSSE",
        f"- LLM / mutation calls: {payload['llm_calls']} / {payload['mutation_calls']}",
        "",
        "| Row | Mean sMAE | Median sMAE | sMAE SE | Mean sRMSE | Median sRMSE | sRMSE SE | Raw P90/P95 sMAE/sRMSE | Clipped sMAE/sRMSE | Coverage | W/T/L vs A (joint) | Mean MASE [diagnostic] | Mean RMSSE [diagnostic] | Mean MAE [diagnostic] | Methods | Families | Ensemble |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, raw in scores.items():
        score = raw
        comparison = paired[name]
        lines.append(
            f"| {name} | {score['mean_smae']:.6f} | {score['median_smae']:.6f} | "
            f"{score['se_smae']:.6f} | {score['mean_srmse']:.6f} | "
            f"{score['median_srmse']:.6f} | {score['se_srmse']:.6f} | "
            f"{score['p90_smae_raw']:.6f}/{score['p95_smae_raw']:.6f} / "
            f"{score['p90_srmse_raw']:.6f}/{score['p95_srmse_raw']:.6f} | "
            f"{score['smae_clipped_count']}/{score['srmse_clipped_count']} | "
            f"{score['coverage']:.4f} | "
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']} | "
            f"{score['mean_mase']:.6f} | {score['mean_rmsse']:.6f} | "
            f"{score['mean_mae']:.6f} | "
            f"{score['method_diversity']} | "
            f"{score['family_diversity']} | {score['ensemble_rate']:.4f} | "
            ""
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
