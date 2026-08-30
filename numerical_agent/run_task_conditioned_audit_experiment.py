"""Search one typed task-conditioned audit route on Train, then gate once on Dev."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from common.data import load_tasks_by_id
from common.evolution_core.contracts import (
    metric_report_metadata,
    require_active_metric_policy,
)
from common.payload import read_json_object, standards_json_value, write_json

from .evolution.module import read_module
from .evolution.numerical_selector import HindcastConfig
from .evolution.portfolio import read_policy_file
from .evolution.screening import profile_task
from .evolution.screening_evolution import parse_screening_source
from .evolution.selector_evolution import (
    DecisionGateResult,
    _compare_train_decisions,
    _train_rank,
    bounded_baseline_guard_candidates,
    bounded_conservative_combined_candidates,
    bounded_conservative_tsfm_candidates,
    bounded_joint_portfolio_candidates,
    bounded_protected_portfolio_candidates,
    bounded_protected_topk_candidates,
    changed_decision_task_ids,
    compare_change_aware_crossfolds,
    compare_decisions,
    evaluate_decision,
    parse_decision_source,
    render_decision_source,
    _select_case,
)
from .main import _add_tsfm_runtime_options, _runtime_registry
from .run_selector_evolution import (
    ForecastStore,
    _build_case,
    _forecast_runtime_identity,
    _score_pair_wtl,
)
from .run_task_conditioned_screening import _training_outcomes, load_frozen_partitions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--screening-dir", required=True)
    parser.add_argument("--parent-selector-dir", required=True)
    parser.add_argument("--split-file", default="splits/drcik_public_80_20_99_v1.json")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--outcome-cache-dir", required=True)
    parser.add_argument("--policy-outcome-cache-dir", required=True)
    parser.add_argument("--hindcast-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=80)
    parser.add_argument("--dev-limit", type=int, default=20)
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Search on Train only; do not load, score, or accept against Dev.",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--train-validation-folds", type=int, default=4)
    parser.add_argument(
        "--candidate-family",
        choices=(
            "change-aware",
            "conservative-tsfm",
            "conservative-combined",
            "joint-portfolio",
            "protected-portfolio",
            "protected-topk",
        ),
        default="change-aware",
    )
    _add_tsfm_runtime_options(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.candidate_family in {
        "conservative-combined", "joint-portfolio", "protected-portfolio"
    } and not args.train_only:
        raise ValueError(f"{args.candidate_family} requires --train-only")
    started = time.monotonic()
    repo = Path(args.repo).resolve()
    screen_dir = Path(args.screening_dir).resolve()
    selector_dir = Path(args.parent_selector_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    screen_path = screen_dir / "frozen_screening_policy.py"
    screen_hash = _sha256(screen_path)
    screen_manifest = read_json_object(screen_dir / "screening_manifest.json")
    require_active_metric_policy(screen_manifest, context="active screening release")
    if screen_manifest.get("schema_version") != 2:
        raise ValueError("active screening release schema_version must be 2")
    if screen_manifest.get("frozen_screening_policy_sha256") != screen_hash:
        raise ValueError("frozen screening policy hash does not match its manifest")
    parent_manifest = read_json_object(selector_dir / "selector_manifest.json")
    require_active_metric_policy(parent_manifest, context="active selector release")
    if parent_manifest.get("schema_version") != 2:
        raise ValueError("active selector release schema_version must be 2")
    if parent_manifest.get("screening_policy_sha256") != screen_hash:
        raise ValueError("parent Decision policy is bound to a different screening policy")

    screening = parse_screening_source(screen_path.read_text(encoding="utf-8"))
    parent = parse_decision_source(
        (selector_dir / "frozen_decision_policy.py").read_text(encoding="utf-8")
    )
    if parent.long_horizon_audit_enabled:
        raise ValueError("parent Decision policy already contains an audit route")

    train, dev = load_frozen_partitions(
        args.split_file,
        args.tasks_file,
        train_limit=args.train_limit,
        dev_limit=0 if args.train_only else args.dev_limit,
    )
    entities = _load_entity_groups(
        args.tasks_file, tuple(task.task_id for task in (*train, *dev))
    )
    module = read_module(repo / "methods.py")
    portfolio = read_policy_file(repo / "policies.py")
    outcomes, outcome_cache = _training_outcomes(args, repo, module, portfolio, train + dev)
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
            screen_hash,
            runtime_identity=_forecast_runtime_identity(args),
        )
        try:
            config = HindcastConfig(folds=args.folds, long_horizon_audit=True)
            cases = tuple(
                _build_case(
                    task,
                    screening,
                    screen_hash,
                    by_key,
                    store,
                    config,
                    group_id=entities.get(task.task_id, task.task_id),
                )
                for task in train + dev
            )
        finally:
            store.close()
    finally:
        runtimes.close()

    train_cases = cases[: len(train)]
    dev_cases = cases[len(train):]
    parent_train = evaluate_decision(parent, train_cases)
    winner = parent
    winner_train = parent_train
    rows = []
    for candidate in _candidate_policies(parent, args.candidate_family):
        score = evaluate_decision(candidate, train_cases)
        if candidate == parent:
            train_gate = DecisionGateResult(True, "zero-penalty parent reference")
            crossfold = DecisionGateResult(True, "zero-penalty parent reference")
        else:
            train_gate = _compare_train_decisions(parent_train, score)
            if args.candidate_family in {
                "conservative-tsfm", "conservative-combined", "joint-portfolio",
                "protected-portfolio", "protected-topk",
            }:
                train_gate = _conservative_dual_metric_train_gate(
                    parent_train, score, train_gate
                )
            crossfold = (
                compare_change_aware_crossfolds(
                    parent,
                    candidate,
                    train_cases,
                    folds=args.train_validation_folds,
                )
                if train_gate.accepted
                else DecisionGateResult(False, "aggregate Train gate failed")
            )
        eligible = train_gate.accepted and crossfold.accepted
        if eligible and _train_rank(score) < _train_rank(winner_train):
            winner = candidate
            winner_train = score
        rows.append({
            "route": _route_payload(candidate),
            "changed_train_tasks": list(changed_decision_task_ids(parent, candidate, train_cases)),
            "adaptive_overlay_assignments": _adaptive_overlay_assignments(
                candidate, train_cases
            ),
            "statistical_overlay_assignments": _statistical_overlay_assignments(
                candidate, train_cases
            ),
            "joint_portfolio_assignments": _joint_portfolio_assignments(
                candidate, train_cases
            ),
            "protected_route_assignments": _protected_route_assignments(
                candidate, train_cases
            ),
            "eligible": eligible,
            "train": asdict(score),
            "train_gate": asdict(train_gate),
            "entity_crossfold_gate": asdict(crossfold),
        })

    dev_parent, dev_winner, dev_gate, accepted, frozen = _gate_winner_on_dev(
        parent,
        winner,
        parent_train,
        winner_train,
        dev_cases,
        train_only=args.train_only,
    )
    (output / "train_winner_decision_policy.py").write_text(
        render_decision_source(winner, screening_policy_hash=screen_hash),
        encoding="utf-8",
    )
    (output / "frozen_decision_policy.py").write_text(
        render_decision_source(frozen, screening_policy_hash=screen_hash),
        encoding="utf-8",
    )
    _write_selector_manifest(
        output,
        parent_manifest,
        screening_hash=screen_hash,
        accepted=accepted,
        train_tasks=len(train),
        dev_tasks=len(dev),
        final_gate=asdict(dev_gate),
        experiment=args.candidate_family,
        train_only=args.train_only,
    )
    payload = {
        "schema_version": 2,
        **metric_report_metadata(),
        "experiment": args.candidate_family,
        "selection": (
            "Train-only search with entity-disjoint cross-validation; Dev not accessed"
            if args.train_only
            else "80 Train and change-aware entity-disjoint four-fold CV; "
            "one read-only 20 Dev gate"
        ),
        "train_tasks": len(train),
        "dev_tasks": len(dev),
        "screening_policy_sha256": screen_hash,
        "candidate_count": len(rows),
        "candidates": rows,
        "selected_route": _route_payload(winner),
        "selected_train_changes": list(changed_decision_task_ids(parent, winner, train_cases)),
        "selected_adaptive_overlay_assignments": _adaptive_overlay_assignments(
            winner, train_cases
        ),
        "selected_statistical_overlay_assignments": _statistical_overlay_assignments(
            winner, train_cases
        ),
        "selected_joint_portfolio_assignments": _joint_portfolio_assignments(
            winner, train_cases
        ),
        "selected_protected_route_assignments": _protected_route_assignments(
            winner, train_cases
        ),
        "selected_dev_changes": (
            []
            if args.train_only
            else list(changed_decision_task_ids(parent, winner, dev_cases))
        ),
        "train_parent": asdict(parent_train),
        "train_winner": asdict(winner_train),
        "dev_parent": asdict(dev_parent) if dev_parent is not None else None,
        "dev_winner": asdict(dev_winner) if dev_winner is not None else None,
        "dev_gate": asdict(dev_gate),
        "paired_joint_wtl": {
            "train": _score_pair_wtl(parent_train, winner_train),
            "dev": (
                _score_pair_wtl(dev_parent, dev_winner)
                if dev_parent is not None and dev_winner is not None
                else {"wins": 0, "ties": 0, "losses": 0, "missing": 0}
            ),
        },
        "dev_evaluated": not args.train_only,
        "accepted": accepted,
        "audit_statuses": {
            "train": _audit_statuses(train_cases),
            "dev": _audit_statuses(dev_cases),
        },
        "cache": {
            **outcome_cache,
            "hindcast_hits": store.hits,
            "hindcast_misses": store.misses,
        },
        "elapsed_seconds": time.monotonic() - started,
        "public_test_accessed": False,
    }
    write_json(output / "evaluation.json", payload)
    print(json.dumps(
        standards_json_value(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


def _gate_winner_on_dev(
    parent,
    winner,
    parent_train,
    winner_train,
    dev_cases,
    *,
    train_only: bool,
):
    if train_only:
        return (
            None,
            None,
            DecisionGateResult(False, "Train-only run; Dev was not evaluated"),
            False,
            parent,
        )
    dev_parent = evaluate_decision(parent, dev_cases)
    dev_winner = evaluate_decision(winner, dev_cases)
    gate = (
        compare_decisions(parent_train, winner_train, dev_parent, dev_winner)
        if winner != parent
        else DecisionGateResult(False, "No task-conditioned Train candidate passed selection")
    )
    accepted = winner != parent and gate.accepted
    return dev_parent, dev_winner, gate, accepted, winner if accepted else parent


def _load_entity_groups(tasks_file, task_ids) -> dict[str, str]:
    return {
        task.task_id: task.entity_name
        for task in load_tasks_by_id(tasks_file, list(task_ids))
    }


def _adaptive_overlay_assignments(policy, cases) -> dict[str, float]:
    assignments = {}
    for case in cases:
        decision = _select_case(policy, case)
        if (
            decision is not None
            and decision.combination_type == "tsfm_shrinkage_overlay"
            and len(decision.weights) == 2
        ):
            assignments[case.task.task_id] = float(decision.weights[1])
    return assignments


def _statistical_overlay_assignments(policy, cases) -> dict[str, dict[str, object]]:
    assignments = {}
    for case in cases:
        decision = _select_case(policy, case)
        if (
            decision is not None
            and decision.combination_type == "statistical_shrinkage_overlay"
            and len(decision.selected) == 2
            and len(decision.weights) == 2
        ):
            assignments[case.task.task_id] = {
                "anchor": decision.selected[0],
                "specialist": decision.selected[1],
                "specialist_weight": float(decision.weights[1]),
            }
    return assignments


def _joint_portfolio_assignments(policy, cases) -> dict[str, dict[str, object]]:
    assignments = {}
    portfolio_types = {
        "tsfm_weighted_portfolio",
        "tsfm_median_portfolio",
        "joint_tsfm_statistical_portfolio",
    }
    for case in cases:
        decision = _select_case(policy, case)
        if decision is None or decision.combination_type not in portfolio_types:
            continue
        joint = decision.combination_type == "joint_tsfm_statistical_portfolio"
        tsfm_count = len(decision.selected) - int(joint)
        if tsfm_count < 2 or len(decision.weights) != len(decision.selected):
            continue
        row = {
            "tsfm_members": list(decision.selected[:tsfm_count]),
            "tsfm_weights": [float(value) for value in decision.weights[:tsfm_count]],
            "statistical_specialist": (
                decision.selected[-1] if joint else None
            ),
            "statistical_weight": (
                float(decision.weights[-1]) if joint else 0.0
            ),
            "combination_type": decision.combination_type,
        }
        assignments[case.task.task_id] = row
    return assignments


def _protected_route_assignments(policy, cases) -> dict[str, dict[str, object]]:
    assignments = {}
    protected_types = {
        "protected_tsfm_weighted_portfolio",
        "protected_tsfm_median_portfolio",
        "protected_statistical_residual",
        "protected_joint_tsfm_statistical_residual",
    }
    for case in cases:
        decision = _select_case(policy, case)
        if decision is None or decision.combination_type not in protected_types:
            continue
        assignments[case.task.task_id] = {
            "selected": list(decision.selected),
            "weights": [float(value) for value in decision.weights],
            "combination_type": decision.combination_type,
        }
    return assignments


def _conservative_dual_metric_train_gate(
    parent, child, current_gate: DecisionGateResult
) -> DecisionGateResult:
    if not current_gate.accepted:
        return current_gate
    if not child.mean_smae + 1e-12 < parent.mean_smae:
        return DecisionGateResult(False, "Train aggregate sMAE did not improve")
    if not child.mean_srmse + 1e-12 < parent.mean_srmse:
        return DecisionGateResult(False, "Train aggregate sRMSE did not improve")
    return DecisionGateResult(
        True, "Candidate passed strict aggregate Train sMAE and sRMSE gates"
    )


def _route_payload(policy) -> dict[str, object]:
    return {
        "guard_enabled": policy.long_horizon_guard_enabled,
        "baseline_strategy": policy.baseline_strategy,
        "min_coverage": policy.long_horizon_min_coverage,
        "max_regret": policy.long_horizon_max_regret,
        "minimum_improvement": (
            policy.tsfm_router_min_improvement
            if policy.baseline_strategy in {
                "conservative_tsfm",
                "conservative_combined",
                "conservative_single_tsfm",
                "conservative_tsfm_portfolio",
                "conservative_tsfm_statistical",
                "conservative_joint_portfolio",
                "protected_single_tsfm",
                "protected_tsfm_portfolio",
                "protected_joint_residual",
            }
            else policy.ensemble_min_improvement
        ),
        "blend_weight": policy.tsfm_router_blend_weight,
    }


def _candidate_policies(parent, family: str):
    if family == "change-aware":
        return bounded_baseline_guard_candidates(parent)
    if family == "conservative-tsfm":
        return bounded_conservative_tsfm_candidates(parent)
    if family == "conservative-combined":
        return bounded_conservative_combined_candidates(parent)
    if family == "joint-portfolio":
        return bounded_joint_portfolio_candidates(parent)
    if family == "protected-portfolio":
        return bounded_protected_portfolio_candidates(parent)
    if family == "protected-topk":
        return bounded_protected_topk_candidates(parent)
    raise ValueError(f"unsupported candidate family: {family}")


def _audit_statuses(cases) -> dict[str, int]:
    counts = Counter()
    for case in cases:
        for diagnostic in case.diagnostics.values():
            status = diagnostic.long_horizon_fold.status if diagnostic.long_horizon_fold else "missing"
            counts[f"{diagnostic.family}:{status}"] += 1
    return dict(sorted(counts.items()))


def _write_selector_manifest(
    output_dir: str | Path,
    parent_manifest: dict[str, object],
    *,
    screening_hash: str,
    accepted: bool,
    train_tasks: int,
    dev_tasks: int,
    final_gate: dict[str, object],
    experiment: str = "change-aware",
    train_only: bool = False,
) -> None:
    output = Path(output_dir)
    policy_path = output / "frozen_decision_policy.py"
    inherited = parent_manifest
    if train_only:
        inherited = {
            key: parent_manifest[key]
            for key in ("schema_version", "phase", "frozen_global_ranking")
            if key in parent_manifest
        }
    manifest = {
        **inherited,
        "schema_version": 2,
        **metric_report_metadata(),
        "experiment": experiment,
        "screening_policy_sha256": screening_hash,
        "frozen_decision_policy_sha256": _sha256(policy_path),
        "train_tasks": train_tasks,
        "dev_tasks": dev_tasks,
        "dev_accepted": accepted,
        "final_dev_gate": final_gate,
        "public_test_accessed": False,
    }
    write_json(output / "selector_manifest.json", manifest)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
