"""Independent, target-wise children for forecasting-method evolution."""
from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from common.llm import LLMClient, parse_json_object
from common.evolution_core.contracts import metric_report_metadata
from common.metrics import joint_scaled_error, pareto_scaled_improvement
from common.payload import write_json

from .cache import OutcomeCache
from .diagnostics import (
    FAILURE_JUDGE_SYSTEM,
    diagnose_forecasts,
    parse_failure_diagnosis,
    render_failure_judge_user,
)
from .execution import (
    Outcome,
    Task,
    oracle_scaled_summary,
    report_payload,
    reports_from_outcomes,
)
from .identity import identity_contract
from .module import MethodModule, ModuleError, apply_operations, read_module, write_module
from .prompts import (
    TARGETWISE_MUTATE_SYSTEM,
    TARGETWISE_SELECT_SYSTEM,
    render_mutate_user,
    render_select_user,
)


@dataclass(frozen=True)
class TargetProposal:
    """One selected Parent method and the operations its child may perform."""

    name: str
    allowed_actions: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class CandidateResult:
    target: TargetProposal
    operation: Mapping[str, object] | None
    accepted: bool
    promoted: bool
    reason: str
    train_metrics: Mapping[str, float]
    validation_metrics: Mapping[str, float]
    screen_metrics: Mapping[str, float] = field(default_factory=dict)
    diagnosis: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetWiseGeneration:
    number: int
    candidates: tuple[CandidateResult, ...]
    applied: tuple[str, ...]
    method_count: int
    commit: str
    cache_hits: int
    cache_misses: int
    elapsed_seconds: float


def parse_target_proposals(
    text: str,
    module: MethodModule,
    *,
    max_targets: int,
) -> tuple[TargetProposal, ...]:
    """Parse unique selector targets and derive safe per-target action sets."""
    if max_targets < 1:
        raise ValueError("max_targets must be positive")
    if max_targets > 10:
        raise ValueError("max_targets must be at most ten")
    raw = parse_json_object(text).get("targets", [])
    if not isinstance(raw, list):
        raise ModuleError("'targets' must be a list")
    proposals: list[TargetProposal] = []
    seen: set[str] = set()
    for item in raw[:max_targets]:
        if not isinstance(item, dict):
            raise ModuleError("each selector target must be an object")
        name = str(item.get("name", "")).strip()
        action = str(item.get("action", "")).strip()
        reason = str(item.get("reason", "")).strip()
        method = module.get(name)
        if method is None:
            raise ModuleError(f"unknown selector target {name!r}")
        if name in seen:
            raise ModuleError(f"duplicate selector target {name!r}")
        if not reason:
            raise ModuleError(f"selector target {name!r} must state a reason")
        if action == "repair":
            contract = identity_contract(name, method.source)
            allowed = ("repair", "fork") if contract.payload()["repair_allowed"] else ("fork",)
        elif action == "fork":
            allowed = ("fork",)
        elif action == "delete":
            allowed = ("delete",)
        else:
            raise ModuleError(f"target-wise selector action {action!r} is not supported")
        seen.add(name)
        proposals.append(TargetProposal(name, allowed, reason))
    return tuple(proposals)


def evolve_targets_once(
    repo: str | Path,
    tasks: Sequence[Task],
    mutator: LLMClient,
    selector: LLMClient,
    *,
    generation: int,
    outcome_cache: OutcomeCache,
    validation_tasks: Sequence[Task],
    judge: LLMClient | None = None,
    screen_tasks: int = 4,
    max_targets: int = 3,
    full_evaluation_candidates: int = 3,
    isolate_methods: bool = True,
) -> TargetWiseGeneration:
    """Evaluate independent target children, then rebase and promote every survivor."""
    if not tasks:
        raise ValueError("target-wise evolution requires training tasks")
    if not validation_tasks:
        raise ValueError("target-wise evolution requires validation tasks")
    if screen_tasks < 1:
        raise ValueError("screen_tasks must be positive")
    if screen_tasks > 4:
        raise ValueError("screen_tasks must be at most four")
    if full_evaluation_candidates < 1:
        raise ValueError("full_evaluation_candidates must be positive")
    if full_evaluation_candidates > 3:
        raise ValueError("full_evaluation_candidates must be at most three")
    if max_targets > 10:
        raise ValueError("max_targets must be at most ten")
    started = time.monotonic()
    root = Path(repo)
    module_path = root / "methods.py"
    _require_clean_evolution_repo(root)
    parent = read_module(module_path)
    initial_hits, initial_misses = outcome_cache.stats.hits, outcome_cache.stats.misses

    parent_train = _evaluate_module(outcome_cache, parent, tasks, isolated=isolate_methods)
    parent_reports = reports_from_outcomes(parent.names(), parent_train, tasks)
    reports_payload = report_payload(parent_reports)
    inventory = [
        {
            "name": method.name,
            "docstring": method.docstring,
            **identity_contract(method.name, method.source).payload(),
        }
        for method in parent.methods
    ]
    select_user = render_select_user(
        reports=reports_payload,
        method_inventory=inventory,
        generation=generation,
        task_count=len(tasks),
        max_targets=max_targets,
    )
    selection = selector.complete(
        system=TARGETWISE_SELECT_SYSTEM,
        messages=[{"role": "user", "content": select_user}],
        temperature=0.0,
    )
    _write_transcript(root, generation, "selector", TARGETWISE_SELECT_SYSTEM, select_user, selection.text)
    proposals = parse_target_proposals(selection.text, parent, max_targets=max_targets)

    candidates: list[CandidateResult] = []
    screen_survivors: list[int] = []
    eligible: list[int] = []
    parent_validation: tuple[Outcome, ...] | None = None
    report_by_name = {str(report["method"]): report for report in reports_payload}
    screen = tuple(tasks[: min(screen_tasks, len(tasks))])
    screen_parent = _subset(parent_train, {task.task_id for task in screen})

    # Stage 1: every selected target gets a Train-only diagnosis and a four-task
    # child screen.  No candidate may see mini-dev during this stage.
    for index, target in enumerate(proposals, start=1):
        method = parent.get(target.name)
        assert method is not None
        diagnosis = _diagnose_target(
            root,
            generation,
            index,
            method,
            screen,
            outcome_cache,
            judge,
            isolated=isolate_methods,
        )
        selected = [{
            "name": target.name,
            "allowed_actions": list(target.allowed_actions),
            "reason": target.reason,
        }]
        user = render_mutate_user(
            reports=[report_by_name[target.name]],
            selected=selected,
            selected_source=method.source,
            all_method_names=parent.names(),
            identity_contracts=[identity_contract(method.name, method.source).payload()],
            generation=generation,
            task_count=len(tasks),
            failure_diagnosis=diagnosis,
        )
        try:
            response = mutator.complete(
                system=TARGETWISE_MUTATE_SYSTEM,
                messages=[{"role": "user", "content": user}],
                temperature=0.0,
            )
        except Exception as exc:
            candidates.append(
                _rejected(
                    target,
                    None,
                    f"mutator unavailable: {type(exc).__name__}",
                    diagnosis=diagnosis,
                )
            )
            _write_transcript(
                root,
                generation,
                f"target_{index:02d}_{target.name}",
                TARGETWISE_MUTATE_SYSTEM,
                user,
                json.dumps({"error": "mutator unavailable"}),
            )
            continue
        _write_transcript(
            root, generation, f"target_{index:02d}_{target.name}",
            TARGETWISE_MUTATE_SYSTEM, user, response.text,
        )
        try:
            operation = _parse_target_operation(response.text, target)
            if operation is None:
                candidates.append(
                    _rejected(target, None, "no operation proposed", diagnosis=diagnosis)
                )
                continue
            child, _ = apply_operations(parent, [operation])
            from . import _validate_evolved_module

            _validate_evolved_module(parent, child, [operation], reports_payload)
            screen_child = _candidate_outcomes(
                outcome_cache, parent, child, screen_parent, operation, screen,
                isolated=isolate_methods,
            )
            screen_metrics = _comparison_with_changed_method(
                parent, child, operation, screen_parent, screen_child, screen
            )
            screen_ok, screen_reason = _accept_stage(
                parent, child, operation, screen_parent, screen_child, screen, screen_metrics
            )
            if not screen_ok:
                candidates.append(
                    _rejected(
                        target,
                        operation,
                        f"screen {screen_reason}",
                        screen_metrics=screen_metrics,
                        diagnosis=diagnosis,
                    )
                )
                continue
            candidates.append(
                CandidateResult(
                    target,
                    operation,
                    False,
                    False,
                    "passed 4-task screen",
                    {},
                    {},
                    screen_metrics,
                    diagnosis,
                )
            )
            screen_survivors.append(len(candidates) - 1)
        except (ValueError, ModuleError) as exc:
            candidates.append(_rejected(target, None, str(exc), diagnosis=diagnosis))

    # Stage 2: rank survivors using trusted screen metrics.  At most the configured
    # number advance to all Train tasks and, only after that, to mini-dev.
    promoted_to_full = sorted(
        screen_survivors,
        key=lambda item: _screen_rank(candidates[item].screen_metrics),
    )[:full_evaluation_candidates]
    promoted_set = set(promoted_to_full)
    for candidate_index in screen_survivors:
        if candidate_index not in promoted_set:
            candidates[candidate_index] = replace(
                candidates[candidate_index],
                reason="pruned by successive halving after the 4-task screen",
            )

    for candidate_index in promoted_to_full:
        candidate = candidates[candidate_index]
        assert candidate.operation is not None
        try:
            child, _ = apply_operations(parent, [candidate.operation])
            child_train = _candidate_outcomes(
                outcome_cache, parent, child, parent_train, candidate.operation, tasks,
                isolated=isolate_methods,
            )
            train_metrics = _comparison_with_changed_method(
                parent, child, candidate.operation, parent_train, child_train, tasks
            )
            train_ok, train_reason = _accept_stage(
                parent, child, candidate.operation, parent_train, child_train, tasks,
                train_metrics,
            )
            if not train_ok:
                candidates[candidate_index] = replace(
                    candidate,
                    reason=f"full Train {train_reason}",
                    train_metrics=train_metrics,
                )
                continue

            if parent_validation is None:
                parent_validation = _evaluate_module(
                    outcome_cache, parent, validation_tasks, isolated=isolate_methods
                )
            child_validation = _candidate_outcomes(
                outcome_cache, parent, child, parent_validation, candidate.operation,
                validation_tasks, isolated=isolate_methods,
            )
            validation_metrics = _comparison_with_changed_method(
                parent, child, candidate.operation, parent_validation, child_validation,
                validation_tasks,
            )
            accepted, validation_reason = _accept_stage(
                parent, child, candidate.operation, parent_validation, child_validation,
                validation_tasks, validation_metrics,
            )
            candidates[candidate_index] = replace(
                candidate,
                accepted=accepted,
                reason="eligible" if accepted else f"mini-dev {validation_reason}",
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )
            if accepted:
                eligible.append(candidate_index)
        except (ValueError, ModuleError, RuntimeError) as exc:
            candidates[candidate_index] = replace(
                candidate,
                reason=f"full evaluation rejected: {exc}",
            )

    applied_list: list[str] = []
    if eligible:
        eligible.sort(
            key=lambda item: (
                joint_scaled_error(
                    candidates[item].validation_metrics["child_mean_smae"],
                    candidates[item].validation_metrics["child_mean_srmse"],
                ),
                joint_scaled_error(
                    candidates[item].validation_metrics["child_median_smae"],
                    candidates[item].validation_metrics["child_median_srmse"],
                ),
            ),
        )
        current = parent
        current_train = parent_train
        if parent_validation is None:
            parent_validation = _evaluate_module(
                outcome_cache, current, validation_tasks, isolated=isolate_methods
            )
        current_validation = parent_validation
        commit = _head(root)
        for candidate_index in eligible:
            candidate = candidates[candidate_index]
            assert candidate.operation is not None
            try:
                rebased, summaries = apply_operations(current, [candidate.operation])
                from . import _validate_evolved_module

                current_reports = report_payload(
                    reports_from_outcomes(current.names(), current_train, tasks)
                )
                _validate_evolved_module(current, rebased, [candidate.operation], current_reports)
                rebased_train = _candidate_outcomes(
                    outcome_cache, current, rebased, current_train,
                    candidate.operation, tasks, isolated=isolate_methods,
                )
                train_metrics = _comparison_with_changed_method(
                    current,
                    rebased,
                    candidate.operation,
                    current_train,
                    rebased_train,
                    tasks,
                )
                train_ok, train_reason = _accept_stage(
                    current, rebased, candidate.operation, current_train,
                    rebased_train, tasks, train_metrics,
                )
                if not train_ok:
                    raise ModuleError(f"rebase Train {train_reason}")
                rebased_validation = _candidate_outcomes(
                    outcome_cache, current, rebased, current_validation,
                    candidate.operation, validation_tasks, isolated=isolate_methods,
                )
                validation_metrics = _comparison_with_changed_method(
                    current,
                    rebased,
                    candidate.operation,
                    current_validation,
                    rebased_validation,
                    validation_tasks,
                )
                validation_ok, validation_reason = _accept_stage(
                    current, rebased, candidate.operation, current_validation,
                    rebased_validation, validation_tasks, validation_metrics,
                )
                if not validation_ok:
                    raise ModuleError(f"rebase mini-dev {validation_reason}")
                commit = _promote_module(
                    root, current, rebased,
                    f"generation {generation}: target-wise {candidate.target.name}",
                    summaries,
                )
                current = rebased
                current_train = rebased_train
                current_validation = rebased_validation
                applied_list.extend(summaries)
                candidates[candidate_index] = replace(
                    candidate, accepted=True, promoted=True, reason="promoted after rebase",
                    train_metrics=train_metrics, validation_metrics=validation_metrics,
                )
            except (ValueError, ModuleError, RuntimeError) as exc:
                candidates[candidate_index] = replace(
                    candidate, accepted=False, promoted=False,
                    reason=f"rejected after rebase: {exc}",
                )
    else:
        commit = _head(root)

    applied = tuple(applied_list)

    hits = outcome_cache.stats.hits - initial_hits
    misses = outcome_cache.stats.misses - initial_misses
    summary = {
        "schema_version": 2,
        **metric_report_metadata(),
        "generation": generation,
        "cache_hits": hits,
        "cache_misses": misses,
        "elapsed_seconds": time.monotonic() - started,
        "screen_tasks": len(screen),
        "full_evaluation_candidates": len(promoted_to_full),
        "candidates": [_candidate_payload(candidate) for candidate in candidates],
        "applied": list(applied),
        "commit": commit,
    }
    write_json(root / f"generation_{generation:03d}_targetwise.json", summary)
    return TargetWiseGeneration(
        generation, tuple(candidates), applied,
        len(read_module(module_path).names()), commit, hits, misses,
        float(summary["elapsed_seconds"]),
    )


def _evaluate_module(
    cache: OutcomeCache,
    module: MethodModule,
    tasks: Sequence[Task],
    *,
    isolated: bool,
) -> tuple[Outcome, ...]:
    return tuple(
        outcome
        for method in module.methods
        for outcome in cache.evaluate_method(method, tasks, isolated=isolated)
    )


def _diagnose_target(
    root: Path,
    generation: int,
    index: int,
    method,
    screen: Sequence[Task],
    cache: OutcomeCache,
    judge: LLMClient | None,
    *,
    isolated: bool,
) -> Mapping[str, object]:
    """Create deterministic Train diagnostics and optionally interpret them with a Judge."""
    if judge is None:
        return {}
    outcomes = cache.evaluate_method(
        method,
        screen,
        isolated=isolated,
        require_forecasts=True,
    )
    diagnostics = diagnose_forecasts(method.name, outcomes, screen)
    diagnostic_dir = root / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        diagnostic_dir / f"generation_{generation:03d}_target_{index:02d}_{method.name}.json",
        {
            "schema_version": 2,
            **metric_report_metadata(),
            "generation": generation,
            "target": method.name,
            "diagnostics": diagnostics,
        },
    )
    user = render_failure_judge_user(diagnostics)
    try:
        response = judge.complete(
            system=FAILURE_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
        )
        _write_transcript(
            root,
            generation,
            f"judge_{index:02d}_{method.name}",
            FAILURE_JUDGE_SYSTEM,
            user,
            response.text,
        )
        return parse_failure_diagnosis(response.text)
    except Exception as exc:
        # The deterministic measurements remain useful even if the optional language
        # interpretation fails.  A Judge outage must not become a forecasting failure.
        fallback = {
            "failure_types": ["judge_unavailable"],
            "summary": "The language diagnosis was unavailable; use deterministic measurements.",
            "evidence": [f"Judge response could not be validated: {type(exc).__name__}."],
            "mutation_guidance": ["Base any change only on the supplied Train measurements."],
            "confidence": 0.0,
        }
        _write_transcript(
            root,
            generation,
            f"judge_{index:02d}_{method.name}",
            FAILURE_JUDGE_SYSTEM,
            user,
            json.dumps(fallback, sort_keys=True),
        )
        return fallback


def _parse_target_operation(
    text: str, target: TargetProposal
) -> Mapping[str, object] | None:
    raw = parse_json_object(text).get("operations", [])
    if not isinstance(raw, list):
        raise ModuleError("'operations' must be a list")
    if len(raw) > 1:
        raise ModuleError("target-wise mutation permits at most one operation")
    if not raw:
        return None
    operation = raw[0]
    if not isinstance(operation, dict):
        raise ModuleError("target-wise operation must be an object")
    op = str(operation.get("op", ""))
    if op not in target.allowed_actions:
        raise ModuleError(
            f"operation {op!r} is outside allowed actions {list(target.allowed_actions)}"
        )
    touched = str(operation.get("from" if op == "fork" else "name", ""))
    if touched != target.name:
        raise ModuleError(f"operation touches {touched!r}, expected target {target.name!r}")
    if op == "fork" and not str(operation.get("new_identity", "")).strip():
        raise ModuleError("fork must state a non-empty new_identity")
    return operation


def _candidate_outcomes(
    cache: OutcomeCache,
    parent: MethodModule,
    child: MethodModule,
    parent_outcomes: Sequence[Outcome],
    operation: Mapping[str, object],
    tasks: Sequence[Task],
    *,
    isolated: bool,
) -> tuple[Outcome, ...]:
    op = str(operation["op"])
    source_name = str(operation.get("from" if op == "fork" else "name", ""))
    kept = tuple(
        outcome for outcome in parent_outcomes
        if not (op in {"repair", "delete"} and outcome.method == source_name)
    )
    if op == "delete":
        return kept
    if op == "repair":
        changed = child.get(source_name)
    else:
        new_names = set(child.names()) - set(parent.names())
        if len(new_names) != 1:
            raise ModuleError("fork must add exactly one new method")
        changed = child.get(next(iter(new_names)))
    if changed is None:
        raise ModuleError("candidate changed method could not be resolved")
    return kept + cache.evaluate_method(changed, tasks, isolated=isolated)


def _subset(outcomes: Sequence[Outcome], task_ids: set[str]) -> tuple[Outcome, ...]:
    return tuple(outcome for outcome in outcomes if outcome.task_id in task_ids)


def _comparison(
    parent: Sequence[Outcome], child: Sequence[Outcome], tasks: Sequence[Task]
) -> dict[str, float]:
    parent_metrics = oracle_scaled_summary(parent, tasks)
    child_metrics = oracle_scaled_summary(child, tasks)
    return {
        **{f"parent_{name}": value for name, value in parent_metrics.items()},
        **{f"child_{name}": value for name, value in child_metrics.items()},
    }


def _comparison_with_changed_method(
    parent: MethodModule,
    child: MethodModule,
    operation: Mapping[str, object],
    parent_outcomes: Sequence[Outcome],
    child_outcomes: Sequence[Outcome],
    tasks: Sequence[Task],
) -> dict[str, float]:
    """Add target-specific evidence for ranking when the portfolio oracle ties."""
    metrics = _comparison(parent_outcomes, child_outcomes, tasks)
    op = str(operation.get("op", ""))
    source_name = str(operation.get("from" if op == "fork" else "name", ""))
    changed_name = _changed_method_name(parent, child, operation)
    if changed_name is None:
        return metrics
    parent_report = reports_from_outcomes((source_name,), parent_outcomes, tasks)[0]
    child_report = reports_from_outcomes((changed_name,), child_outcomes, tasks)[0]
    for prefix, report in (("parent_method", parent_report), ("child_method", child_report)):
        metrics[f"{prefix}_coverage"] = float(report.coverage)
        if report.mean_smae is not None and report.mean_srmse is not None:
            metrics[f"{prefix}_mean_smae"] = float(report.mean_smae)
            metrics[f"{prefix}_mean_srmse"] = float(report.mean_srmse)
    return metrics


def _screen_rank(metrics: Mapping[str, float]) -> tuple[float, float, float]:
    """Rank screen survivors by portfolio gain, then by target-method gain."""
    mean_delta = joint_scaled_error(
        metrics["child_mean_smae"], metrics["child_mean_srmse"]
    ) - joint_scaled_error(
        metrics["parent_mean_smae"], metrics["parent_mean_srmse"]
    )
    median_delta = joint_scaled_error(
        metrics["child_median_smae"], metrics["child_median_srmse"]
    ) - joint_scaled_error(
        metrics["parent_median_smae"], metrics["parent_median_srmse"]
    )
    parent_smae = metrics.get("parent_method_mean_smae")
    parent_srmse = metrics.get("parent_method_mean_srmse")
    child_smae = metrics.get("child_method_mean_smae")
    child_srmse = metrics.get("child_method_mean_srmse")
    method_delta = (
        math.inf
        if None in (parent_smae, parent_srmse, child_smae, child_srmse)
        else joint_scaled_error(float(child_smae), float(child_srmse))
        - joint_scaled_error(float(parent_smae), float(parent_srmse))
    )
    return mean_delta, median_delta, method_delta


def _strict_non_regression(metrics: Mapping[str, float]) -> bool:
    return pareto_scaled_improvement(
        metrics["parent_mean_smae"],
        metrics["parent_mean_srmse"],
        metrics["child_mean_smae"],
        metrics["child_mean_srmse"],
    )


def _accept_stage(
    parent: MethodModule,
    child: MethodModule,
    operation: Mapping[str, object],
    parent_outcomes: Sequence[Outcome],
    child_outcomes: Sequence[Outcome],
    tasks: Sequence[Task],
    metrics: Mapping[str, float],
) -> tuple[bool, str]:
    """Gate both the changed method itself and the reconstructed portfolio."""
    op = str(operation.get("op", ""))
    tolerance = 1e-12
    changed_name = _changed_method_name(parent, child, operation)
    if changed_name is not None:
        changed_report = reports_from_outcomes((changed_name,), child_outcomes, tasks)[0]
        if changed_report.crashed or changed_report.invalid:
            return False, "changed method crashed or invalid"
        if changed_report.success == 0:
            return False, "changed method has no successful applicable task"
        if op == "repair":
            original = str(operation.get("name", ""))
            parent_report = reports_from_outcomes((original,), parent_outcomes, tasks)[0]
            if changed_report.coverage + tolerance < parent_report.coverage:
                return False, "repair reduced applicable-task coverage"
    portfolio_improved = pareto_scaled_improvement(
        metrics["parent_mean_smae"],
        metrics["parent_mean_srmse"],
        metrics["child_mean_smae"],
        metrics["child_mean_srmse"],
    )
    if portfolio_improved:
        return True, "improved"
    return False, "portfolio mean scaled metric pair did not improve under Pareto gate"


def _changed_method_name(
    parent: MethodModule,
    child: MethodModule,
    operation: Mapping[str, object],
) -> str | None:
    op = str(operation.get("op", ""))
    if op == "delete":
        return None
    if op == "repair":
        return str(operation.get("name", ""))
    if op == "fork":
        added = set(child.names()) - set(parent.names())
        if len(added) != 1:
            raise ModuleError("fork must add exactly one changed method")
        return next(iter(added))
    raise ModuleError(f"unsupported target-wise operation {op!r}")


def _require_clean_evolution_repo(root: Path) -> None:
    """Refuse to mix generated commits with pre-existing tracked or staged work."""
    index = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=root, check=False
    )
    module = subprocess.run(
        ["git", "diff", "--quiet", "--", "methods.py"], cwd=root, check=False
    )
    if index.returncode != 0 or module.returncode != 0:
        raise ModuleError(
            "target-wise evolution requires a clean Git index and unchanged methods.py"
        )


def _head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _promote_module(
    root: Path,
    before: MethodModule,
    after: MethodModule,
    subject: str,
    summaries: Sequence[str],
) -> str:
    """Write and commit only methods.py, restoring it and the index on failure."""
    path = root / "methods.py"
    old_text = before.render()
    message = subject
    if summaries:
        message += "\n\n" + "\n".join(f"- {item}" for item in summaries)
    try:
        write_module(path, after)
        add = subprocess.run(
            ["git", "add", "--", "methods.py"], cwd=root,
            capture_output=True, text=True, check=False,
        )
        if add.returncode != 0:
            raise RuntimeError(f"git add failed: {add.stderr.strip()}")
        commit = subprocess.run(
            ["git", "commit", "--quiet", "--only", "-m", message, "--", "methods.py"],
            cwd=root, capture_output=True, text=True, check=False,
        )
        if commit.returncode != 0:
            raise RuntimeError(f"git commit failed: {commit.stderr.strip()}")
        return _head(root)
    except BaseException:
        path.write_text(old_text, encoding="utf-8")
        subprocess.run(
            ["git", "reset", "--quiet", "HEAD", "--", "methods.py"],
            cwd=root, capture_output=True, check=False,
        )
        raise


def _rejected(
    target: TargetProposal,
    operation: Mapping[str, object] | None,
    reason: str,
    train_metrics: Mapping[str, float] | None = None,
    *,
    screen_metrics: Mapping[str, float] | None = None,
    diagnosis: Mapping[str, object] | None = None,
) -> CandidateResult:
    return CandidateResult(
        target,
        operation,
        False,
        False,
        reason,
        train_metrics or {},
        {},
        screen_metrics or {},
        diagnosis or {},
    )


def _candidate_payload(candidate: CandidateResult) -> dict[str, object]:
    return {
        "target": candidate.target.name,
        "allowed_actions": list(candidate.target.allowed_actions),
        "operation": dict(candidate.operation) if candidate.operation is not None else None,
        "accepted": candidate.accepted,
        "promoted": candidate.promoted,
        "reason": candidate.reason,
        "screen_metrics": dict(candidate.screen_metrics),
        "train_metrics": dict(candidate.train_metrics),
        "validation_metrics": dict(candidate.validation_metrics),
        "diagnosis": dict(candidate.diagnosis),
    }


def _write_transcript(
    root: Path,
    generation: int,
    stage: str,
    system: str,
    user: str,
    response: str,
) -> None:
    directory = root / "transcripts"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"generation_{generation:03d}_{stage}.md").write_text(
        f"# system\n\n{system}\n\n# user\n\n{user}\n\n# response\n\n{response}\n",
        encoding="utf-8",
    )
