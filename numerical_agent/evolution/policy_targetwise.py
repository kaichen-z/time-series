"""Successive-halving evolution for typed TSFM and Combined policies."""
from __future__ import annotations

import json
import math
import statistics
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from common.llm import LLMClient, parse_json_object
from common.payload import write_json

from numerical_agent.providers import RuntimeRegistry

from .cache import OutcomeCache
from .diagnostics import (
    FAILURE_JUDGE_SYSTEM,
    diagnose_forecasts,
    parse_failure_diagnosis,
    render_failure_judge_user,
)
from .execution import Outcome, SUCCESS, Task, report_payload, reports_from_outcomes
from .module import MethodModule, ModuleError, read_module
from .portfolio import (
    CombinedPolicy,
    PolicyError,
    PolicyOutcomeCache,
    PolicyPortfolio,
    TSFMPolicy,
    evaluate_portfolio,
    read_policy_file,
    write_policy_file,
)
from .prompts import (
    POLICY_MUTATE_SYSTEM,
    POLICY_SELECT_SYSTEM,
    render_policy_mutate_user,
    render_policy_select_user,
)


@dataclass(frozen=True)
class PolicyTarget:
    name: str
    reason: str


@dataclass(frozen=True)
class PolicyCandidateResult:
    target: PolicyTarget
    replacement: Mapping[str, object] | None
    accepted: bool
    promoted: bool
    reason: str
    train_metrics: Mapping[str, float] = field(default_factory=dict)
    validation_metrics: Mapping[str, float] = field(default_factory=dict)
    screen_metrics: Mapping[str, float] = field(default_factory=dict)
    diagnosis: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyGeneration:
    number: int
    candidates: tuple[PolicyCandidateResult, ...]
    applied: tuple[str, ...]
    candidate_count: int
    commit: str
    cache_hits: int
    cache_misses: int
    elapsed_seconds: float


def evolve_policies_once(
    repo: str | Path,
    tasks: Sequence[Task],
    mutator: LLMClient,
    selector: LLMClient,
    *,
    generation: int,
    outcome_cache: OutcomeCache,
    policy_cache: PolicyOutcomeCache,
    validation_tasks: Sequence[Task],
    runtimes: RuntimeRegistry,
    judge: LLMClient | None = None,
    screen_tasks: int = 4,
    max_targets: int = 3,
    full_evaluation_candidates: int = 3,
    isolate_methods: bool = True,
) -> PolicyGeneration:
    """Repair selected policies, preserving every model and parent identity."""
    if not tasks or not validation_tasks:
        raise ValueError("policy evolution requires non-empty Train and mini-dev tasks")
    if not 1 <= screen_tasks <= 4:
        raise ValueError("screen_tasks must be between one and four")
    if not 1 <= max_targets <= 10:
        raise ValueError("max_targets must be between one and ten")
    if not 1 <= full_evaluation_candidates <= 3:
        raise ValueError("full_evaluation_candidates must be between one and three")
    started = time.monotonic()
    root = Path(repo)
    _require_clean(root)
    module = read_module(root / "methods.py")
    parent = read_policy_file(root / "policies.py")
    parent.validate_statistical_parents(module.names())
    initial_hits = outcome_cache.stats.hits + policy_cache.stats.hits
    initial_misses = outcome_cache.stats.misses + policy_cache.stats.misses

    parent_train = _evaluate(
        module, parent, tasks, outcome_cache, policy_cache, runtimes, isolate_methods
    )
    all_names = module.names() + parent.names
    reports = report_payload(reports_from_outcomes(all_names, parent_train, tasks))
    select_user = render_policy_select_user(
        reports=reports,
        policies=[
            {"family": "tsfm" if isinstance(policy, TSFMPolicy) else "combined", **policy.to_payload()}
            for policy in parent.all_policies
        ],
        generation=generation,
        task_count=len(tasks),
        max_targets=max_targets,
    )
    selection = selector.complete(
        system=POLICY_SELECT_SYSTEM,
        messages=[{"role": "user", "content": select_user}],
        temperature=0.0,
    )
    _write_transcript(root, generation, "policy_selector", POLICY_SELECT_SYSTEM, select_user, selection.text)
    targets = _parse_targets(selection.text, parent, max_targets)
    report_by_name = {str(report["method"]): report for report in reports}
    screen = tuple(tasks[: min(screen_tasks, len(tasks))])
    screen_ids = {task.task_id for task in screen}
    screen_parent = _subset(parent_train, screen_ids)

    candidates: list[PolicyCandidateResult] = []
    survivors: list[int] = []
    for index, target in enumerate(targets, start=1):
        current = parent.get(target.name)
        assert current is not None
        diagnosis = _diagnose(
            root,
            generation,
            index,
            target.name,
            _subset(parent_train, screen_ids),
            screen,
            judge,
        )
        user = render_policy_mutate_user(
            report=report_by_name[target.name],
            policy=current.to_payload(),
            diagnosis=diagnosis,
            generation=generation,
            task_count=len(tasks),
        )
        try:
            response = mutator.complete(
                system=POLICY_MUTATE_SYSTEM,
                messages=[{"role": "user", "content": user}],
                temperature=0.0,
            )
            _write_transcript(
                root, generation, f"policy_{index:02d}_{target.name}",
                POLICY_MUTATE_SYSTEM, user, response.text,
            )
            replacement, reason = _parse_replacement(response.text, current)
            child = parent.replace(target.name, replacement)
            screen_child = _evaluate(
                module, child, screen, outcome_cache, policy_cache, runtimes, isolate_methods
            )
            metrics = _metrics(screen_parent, screen_child, screen, target.name)
            accepted, gate_reason = _accept(screen_parent, screen_child, screen, target.name, metrics)
            if not accepted:
                candidates.append(
                    PolicyCandidateResult(
                        target, replacement.to_payload(), False, False,
                        f"screen {gate_reason}", screen_metrics=metrics, diagnosis=diagnosis,
                    )
                )
                continue
            candidates.append(
                PolicyCandidateResult(
                    target,
                    replacement.to_payload(),
                    False,
                    False,
                    reason,
                    screen_metrics=metrics,
                    diagnosis=diagnosis,
                )
            )
            survivors.append(len(candidates) - 1)
        except Exception as error:
            candidates.append(
                PolicyCandidateResult(
                    target, None, False, False, str(error), diagnosis=diagnosis
                )
            )

    promoted = sorted(
        survivors, key=lambda item: _rank(candidates[item].screen_metrics)
    )[:full_evaluation_candidates]
    promoted_set = set(promoted)
    for candidate_index in survivors:
        if candidate_index not in promoted_set:
            candidates[candidate_index] = replace(
                candidates[candidate_index],
                reason="pruned by successive halving after the screen",
            )

    eligible: list[int] = []
    parent_validation: tuple[Outcome, ...] | None = None
    for candidate_index in promoted:
        candidate = candidates[candidate_index]
        current = parent.get(candidate.target.name)
        assert current is not None and candidate.replacement is not None
        try:
            replacement_policy = _policy_from_payload(candidate.replacement, current)
            child = parent.replace(candidate.target.name, replacement_policy)
            child_train = _evaluate(
                module, child, tasks, outcome_cache, policy_cache, runtimes, isolate_methods
            )
            train_metrics = _metrics(parent_train, child_train, tasks, candidate.target.name)
            train_ok, train_reason = _accept(
                parent_train, child_train, tasks, candidate.target.name, train_metrics
            )
            if not train_ok:
                candidates[candidate_index] = replace(
                    candidate, reason=f"full Train {train_reason}", train_metrics=train_metrics
                )
                continue
            if parent_validation is None:
                parent_validation = _evaluate(
                    module, parent, validation_tasks, outcome_cache, policy_cache,
                    runtimes, isolate_methods,
                )
            child_validation = _evaluate(
                module, child, validation_tasks, outcome_cache, policy_cache,
                runtimes, isolate_methods,
            )
            validation_metrics = _metrics(
                parent_validation, child_validation, validation_tasks, candidate.target.name
            )
            validation_ok, validation_reason = _accept(
                parent_validation,
                child_validation,
                validation_tasks,
                candidate.target.name,
                validation_metrics,
            )
            candidates[candidate_index] = replace(
                candidate,
                accepted=validation_ok,
                reason="eligible" if validation_ok else f"mini-dev {validation_reason}",
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )
            if validation_ok:
                eligible.append(candidate_index)
        except Exception as error:
            candidates[candidate_index] = replace(
                candidate, reason=f"full evaluation rejected: {error}"
            )

    applied: list[str] = []
    current = parent
    current_train = parent_train
    current_validation = parent_validation
    commit = _head(root)
    eligible.sort(
        key=lambda item: (
            candidates[item].validation_metrics["child_method_mean_mase"],
            candidates[item].validation_metrics["child_mean_mase"],
        )
    )
    for candidate_index in eligible:
        candidate = candidates[candidate_index]
        assert candidate.replacement is not None
        old = current.get(candidate.target.name)
        assert old is not None
        try:
            replacement_policy = _policy_from_payload(candidate.replacement, old)
            rebased = current.replace(candidate.target.name, replacement_policy)
            rebased_train = _evaluate(
                module, rebased, tasks, outcome_cache, policy_cache, runtimes, isolate_methods
            )
            train_metrics = _metrics(
                current_train, rebased_train, tasks, candidate.target.name
            )
            ok, reason = _accept(
                current_train, rebased_train, tasks, candidate.target.name, train_metrics
            )
            if not ok:
                raise PolicyError(f"rebase Train {reason}")
            if current_validation is None:
                current_validation = _evaluate(
                    module, current, validation_tasks, outcome_cache, policy_cache,
                    runtimes, isolate_methods,
                )
            rebased_validation = _evaluate(
                module, rebased, validation_tasks, outcome_cache, policy_cache,
                runtimes, isolate_methods,
            )
            validation_metrics = _metrics(
                current_validation,
                rebased_validation,
                validation_tasks,
                candidate.target.name,
            )
            ok, reason = _accept(
                current_validation,
                rebased_validation,
                validation_tasks,
                candidate.target.name,
                validation_metrics,
            )
            if not ok:
                raise PolicyError(f"rebase mini-dev {reason}")
            summary = f"repair policy {candidate.target.name}: {candidate.reason}"
            commit = _promote(root, current, rebased, generation, candidate.target.name, summary)
            current = rebased
            current_train = rebased_train
            current_validation = rebased_validation
            applied.append(summary)
            candidates[candidate_index] = replace(
                candidate,
                accepted=True,
                promoted=True,
                reason="promoted after rebase",
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )
        except Exception as error:
            candidates[candidate_index] = replace(
                candidate,
                accepted=False,
                promoted=False,
                reason=f"rejected after rebase: {error}",
            )

    hits = outcome_cache.stats.hits + policy_cache.stats.hits - initial_hits
    misses = outcome_cache.stats.misses + policy_cache.stats.misses - initial_misses
    payload = {
        "generation": generation,
        "candidate_count": len(module.names()) + len(current.names),
        "cache_hits": hits,
        "cache_misses": misses,
        "elapsed_seconds": time.monotonic() - started,
        "candidates": [_payload(candidate) for candidate in candidates],
        "applied": applied,
        "commit": commit,
    }
    write_json(root / f"generation_{generation:03d}_policies.json", payload)
    return PolicyGeneration(
        generation,
        tuple(candidates),
        tuple(applied),
        int(payload["candidate_count"]),
        commit,
        hits,
        misses,
        float(payload["elapsed_seconds"]),
    )


def _evaluate(
    module: MethodModule,
    policies: PolicyPortfolio,
    tasks: Sequence[Task],
    outcome_cache: OutcomeCache,
    policy_cache: PolicyOutcomeCache,
    runtimes: RuntimeRegistry,
    isolated_methods: bool,
) -> tuple[Outcome, ...]:
    return evaluate_portfolio(
        module,
        policies,
        tasks,
        outcome_cache=outcome_cache,
        policy_cache=policy_cache,
        runtimes=runtimes,
        isolated_methods=isolated_methods,
    )


def _parse_targets(text: str, policies: PolicyPortfolio, maximum: int) -> tuple[PolicyTarget, ...]:
    raw = parse_json_object(text).get("targets", [])
    if not isinstance(raw, list):
        raise PolicyError("'targets' must be a list")
    targets = []
    seen = set()
    for item in raw[:maximum]:
        if not isinstance(item, Mapping):
            raise PolicyError("each policy target must be an object")
        name = str(item.get("name", "")).strip()
        action = str(item.get("action", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if policies.get(name) is None:
            raise PolicyError(f"unknown policy target {name!r}")
        if name in seen:
            raise PolicyError(f"duplicate policy target {name!r}")
        if action != "repair":
            raise PolicyError("policy targets only support repair")
        if not reason:
            raise PolicyError("policy target requires a reason")
        seen.add(name)
        targets.append(PolicyTarget(name, reason))
    return tuple(targets)


def _parse_replacement(
    text: str, current: TSFMPolicy | CombinedPolicy
) -> tuple[TSFMPolicy | CombinedPolicy, str]:
    raw = parse_json_object(text)
    replacement = raw.get("replacement")
    reason = str(raw.get("reason", "")).strip()
    if not isinstance(replacement, Mapping):
        raise PolicyError("policy mutator must return a replacement object")
    if not reason:
        raise PolicyError("policy mutation requires a reason")
    return _policy_from_payload(replacement, current), reason


def _policy_from_payload(
    payload: Mapping[str, object], current: TSFMPolicy | CombinedPolicy
) -> TSFMPolicy | CombinedPolicy:
    expected = set(current.to_payload())
    if set(payload) != expected:
        raise PolicyError(
            f"replacement fields must exactly match {sorted(expected)!r}"
        )
    try:
        replacement = type(current)(**dict(payload))
    except (TypeError, ValueError) as error:
        raise PolicyError(str(error)) from error
    return replacement


def _metrics(
    parent: Sequence[Outcome],
    child: Sequence[Outcome],
    tasks: Sequence[Task],
    target: str,
) -> dict[str, float]:
    parent_mean, parent_median = _oracle(parent, tasks)
    child_mean, child_median = _oracle(child, tasks)
    parent_report = reports_from_outcomes((target,), parent, tasks)[0]
    child_report = reports_from_outcomes((target,), child, tasks)[0]
    metrics = {
        "parent_mean_mase": parent_mean,
        "parent_median_mase": parent_median,
        "child_mean_mase": child_mean,
        "child_median_mase": child_median,
        "parent_method_coverage": parent_report.coverage,
        "child_method_coverage": child_report.coverage,
    }
    if parent_report.mean_mase is not None:
        metrics["parent_method_mean_mase"] = parent_report.mean_mase
    if child_report.mean_mase is not None:
        metrics["child_method_mean_mase"] = child_report.mean_mase
    return metrics


def _accept(
    parent: Sequence[Outcome],
    child: Sequence[Outcome],
    tasks: Sequence[Task],
    target: str,
    metrics: Mapping[str, float],
) -> tuple[bool, str]:
    tolerance = 1e-12
    parent_report = reports_from_outcomes((target,), parent, tasks)[0]
    child_report = reports_from_outcomes((target,), child, tasks)[0]
    if child_report.crashed or child_report.invalid:
        return False, "changed policy crashed or returned invalid forecasts"
    if child_report.success == 0:
        return False, "changed policy has no successful applicable task"
    if child_report.coverage + tolerance < parent_report.coverage:
        return False, "changed policy reduced applicable-task coverage"
    if parent_report.mean_mase is None or child_report.mean_mase is None:
        return False, "changed policy lacks comparable MASE"
    if child_report.mean_mase >= parent_report.mean_mase - tolerance:
        return False, "changed policy MASE did not improve"
    if (
        metrics["child_mean_mase"] > metrics["parent_mean_mase"] + tolerance
        or metrics["child_median_mase"] > metrics["parent_median_mase"] + tolerance
    ):
        return False, "portfolio MASE regressed"
    return True, "improved"


def _oracle(outcomes: Sequence[Outcome], tasks: Sequence[Task]) -> tuple[float, float]:
    scores = []
    for task in tasks:
        usable = [
            float(outcome.mase)
            for outcome in outcomes
            if outcome.task_id == task.task_id
            and outcome.status == SUCCESS
            and outcome.mase is not None
            and math.isfinite(outcome.mase)
        ]
        scores.append(min(usable) if usable else math.inf)
    return statistics.fmean(scores), statistics.median(scores)


def _rank(metrics: Mapping[str, float]) -> tuple[float, float]:
    return (
        metrics.get("child_method_mean_mase", math.inf)
        - metrics.get("parent_method_mean_mase", math.inf),
        metrics["child_mean_mase"] - metrics["parent_mean_mase"],
    )


def _subset(outcomes: Sequence[Outcome], task_ids: set[str]) -> tuple[Outcome, ...]:
    return tuple(outcome for outcome in outcomes if outcome.task_id in task_ids)


def _diagnose(
    root: Path,
    generation: int,
    index: int,
    target: str,
    outcomes: Sequence[Outcome],
    tasks: Sequence[Task],
    judge: LLMClient | None,
) -> Mapping[str, object]:
    if judge is None:
        return {}
    selected = tuple(outcome for outcome in outcomes if outcome.method == target)
    diagnostics = diagnose_forecasts(target, selected, tasks)
    directory = root / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f"generation_{generation:03d}_policy_{index:02d}_{target}.json", diagnostics)
    user = render_failure_judge_user(diagnostics)
    try:
        response = judge.complete(
            system=FAILURE_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
        )
        _write_transcript(
            root, generation, f"policy_judge_{index:02d}_{target}",
            FAILURE_JUDGE_SYSTEM, user, response.text,
        )
        return parse_failure_diagnosis(response.text)
    except Exception as error:
        return {
            "failure_types": ["judge_unavailable"],
            "summary": "Use deterministic Train measurements only.",
            "evidence": [f"Judge unavailable: {type(error).__name__}."],
            "mutation_guidance": ["Make only a conservative typed policy repair."],
            "confidence": 0.0,
        }


def _require_clean(root: Path) -> None:
    index = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root, check=False)
    policies = subprocess.run(
        ["git", "diff", "--quiet", "--", "policies.py"], cwd=root, check=False
    )
    if index.returncode != 0 or policies.returncode != 0:
        raise ModuleError("policy evolution requires a clean index and unchanged policies.py")


def _promote(
    root: Path,
    before: PolicyPortfolio,
    after: PolicyPortfolio,
    generation: int,
    target: str,
    summary: str,
) -> str:
    path = root / "policies.py"
    try:
        write_policy_file(path, after)
        subprocess.run(["git", "add", "--", "policies.py"], cwd=root, check=True)
        subprocess.run(
            [
                "git", "commit", "--quiet", "--only", "-m",
                f"generation {generation}: policy {target}\n\n- {summary}",
                "--", "policies.py",
            ],
            cwd=root,
            check=True,
        )
        return _head(root)
    except BaseException:
        write_policy_file(path, before)
        subprocess.run(["git", "reset", "--quiet", "HEAD", "--", "policies.py"], cwd=root)
        raise


def _head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


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
        f"# System\n\n{system}\n\n# User\n\n{user}\n\n# Response\n\n{response}\n",
        encoding="utf-8",
    )


def _payload(candidate: PolicyCandidateResult) -> dict[str, object]:
    return {
        "target": candidate.target.name,
        "replacement": dict(candidate.replacement) if candidate.replacement else None,
        "accepted": candidate.accepted,
        "promoted": candidate.promoted,
        "reason": candidate.reason,
        "screen_metrics": dict(candidate.screen_metrics),
        "train_metrics": dict(candidate.train_metrics),
        "validation_metrics": dict(candidate.validation_metrics),
        "diagnosis": dict(candidate.diagnosis),
    }
