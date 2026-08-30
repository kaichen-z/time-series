"""Single-agent evolution of a Python forecasting-candidate dictionary."""
from __future__ import annotations

import ast
import json
import math
import pprint
import statistics
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from common.llm import LLMClient, parse_json_object
from common.metrics import (
    joint_scaled_error,
    linear_quantile,
    pareto_scaled_improvement,
    standard_error,
)

from .execution import (
    CRASHED,
    INVALID,
    NOT_APPLICABLE,
    SUCCESS,
    Outcome,
    Task,
    require_unique_outcome_keys,
    require_unique_task_ids,
)
from .cache import OutcomeCache
from .module import MethodModule
from .portfolio import (
    PolicyOutcomeCache,
    PolicyPortfolio,
    TSFMPolicy,
    _run_combined,
)


STATUSES = frozenset({"keep", "specialized", "repair", "quarantine", "discard"})
FAMILIES = frozenset({"statistical", "tsfm", "combined"})
EXCLUSIVE_TAG_GROUPS = (
    frozenset({"dense", "intermittent"}),
    frozenset({"no_zeros", "some_zeros", "many_zeros"}),
    frozenset({"nonnegative", "signed"}),
    frozenset({"integer_valued", "continuous_valued"}),
    frozenset({"flat", "trending"}),
)
FILTER_SYSTEM = """You are one self-evolving numerical dictionary Filter Agent. You receive
only Train evidence from executable forecasting candidates and the current Python dictionary.
Propose a conservative Child dictionary by changing candidate status or history-only
applicability. NotApplicable is correct specialist behavior, not a failure. Crash and invalid
are implementation failures. Keep broad stable useful methods; mark a narrow but useful method
specialized. Status and applicability are orthogonal: a useful keep method may also require
history-only applicability tags so it is not selected outside its valid regime. Use repair for
a valuable but faulty implementation; quarantine methods that are currently unsafe to select.
Use discard only when the request explicitly says trusted dominance evidence exists. Every name
in required_targets must appear in actions. These include repeated downstream selection failures,
partially applicable methods, and faulty implementations. For a method that succeeds on a coherent
subset and returns NotApplicable elsewhere, prefer a supplied applicability rule over global
quarantine. Use required_target_evidence to compare success and NotApplicable behavior by
history-only tag. Never invent methods, metrics, code, Dev results, or future values.

Return exactly one JSON object:
{"summary":"...","actions":[{"name":"...","status":"keep|specialized|repair|quarantine|discard","applicability":["history-only tag"],"reason":"..."}]}
Return only changed entries, at most 24. A specialized entry requires at least one supplied
history-only tag. Keep entries may use supplied history-only tags. Repair, quarantine, and
discard entries use an empty applicability list. Applicability is one conjunctive AND rule:
every listed tag must hold. Never list two mutually exclusive tags from the same family
(for example no_zeros with some_zeros). When a method succeeds across multiple alternatives
in one family, omit that family rather than pretending the alternatives are an AND condition."""


class FilterError(ValueError):
    """A filtering artifact or Agent action violates the filtering contract."""


@dataclass(frozen=True)
class FilterEntry:
    name: str
    family: str
    status: str
    applicability: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise FilterError(f"invalid candidate name {self.name!r}")
        if self.family not in FAMILIES:
            raise FilterError(f"invalid family for {self.name}: {self.family!r}")
        if self.status not in STATUSES:
            raise FilterError(f"invalid status for {self.name}: {self.status!r}")
        if not self.reason.strip():
            raise FilterError(f"{self.name} needs a non-empty reason")
        if len(set(self.applicability)) != len(self.applicability):
            raise FilterError(f"{self.name} has duplicate applicability tags")
        for group in EXCLUSIVE_TAG_GROUPS:
            overlap = sorted(group.intersection(self.applicability))
            if len(overlap) > 1:
                raise FilterError(
                    f"{self.name} has mutually exclusive applicability tags: "
                    + ", ".join(overlap)
                )
        for prefix in ("frequency:", "history:", "horizon:"):
            overlap = sorted(tag for tag in self.applicability if tag.startswith(prefix))
            if len(overlap) > 1:
                raise FilterError(
                    f"{self.name} has mutually exclusive applicability tags: "
                    + ", ".join(overlap)
                )
        if self.status == "specialized" and not self.applicability:
            raise FilterError(f"specialized {self.name} requires applicability")
        if self.status not in {"keep", "specialized"} and self.applicability:
            raise FilterError(f"only selectable methods may define applicability")


@dataclass(frozen=True)
class FilterDictionary:
    entries: tuple[FilterEntry, ...]

    def __post_init__(self) -> None:
        names = [entry.name for entry in self.entries]
        if not names:
            raise FilterError("dictionary must contain at least one candidate")
        if len(set(names)) != len(names):
            raise FilterError("dictionary contains duplicate candidate names")

    def get(self, name: str) -> FilterEntry | None:
        return next((entry for entry in self.entries if entry.name == name), None)


@dataclass(frozen=True)
class FilterScore:
    mean_smae: float
    mean_srmse: float
    median_smae: float
    median_srmse: float
    coverage: float
    selected: Mapping[str, str]
    eligible_counts: Mapping[str, int]
    eligible_attempts: int
    eligible_successes: int
    eligible_not_applicable: int
    eligible_failures: int
    eligible_crashed: int
    eligible_invalid: int
    eligible_missing: int
    eligible_malformed_success: int
    eligible_success_rate: float
    eligible_not_applicable_rate: float
    eligible_failure_rate: float
    eligible_crash_rate: float
    eligible_invalid_rate: float
    eligible_missing_rate: float
    eligible_malformed_success_rate: float
    se_smae: float = math.inf
    se_srmse: float = math.inf
    p90_smae: float = math.inf
    p95_smae: float = math.inf
    p90_srmse: float = math.inf
    p95_srmse: float = math.inf
    p90_smae_raw: float = math.inf
    p95_smae_raw: float = math.inf
    p90_srmse_raw: float = math.inf
    p95_srmse_raw: float = math.inf
    smae_clipped_count: int = 0
    smae_clipped_rate: float = 1.0
    srmse_clipped_count: int = 0
    srmse_clipped_rate: float = 1.0
    task_scaled_pairs: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    task_count: int = 0


@dataclass(frozen=True)
class FilterGeneration:
    generation: int
    parent: FilterDictionary
    child: FilterDictionary
    train_parent: FilterScore
    train_child: FilterScore
    dev_parent: FilterScore | None
    dev_child: FilterScore | None
    accepted: bool
    reason: str
    agent_calls: int
    required_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class FilterGateResult:
    accepted: bool
    reason: str


def build_filter_dictionary(
    module: MethodModule, portfolio: PolicyPortfolio
) -> FilterDictionary:
    """Index all executable Python, TSFM, and Combined candidates in one Python artifact."""
    entries = [
        FilterEntry(
            method.name,
            "statistical",
            "keep",
            (),
            method.docstring or "Unfiltered executable statistical candidate.",
        )
        for method in module.methods
    ]
    entries.extend(
        FilterEntry(
            policy.name,
            "tsfm" if isinstance(policy, TSFMPolicy) else "combined",
            "keep",
            (),
            "Unfiltered executable foundation candidate."
            if isinstance(policy, TSFMPolicy)
            else "Unfiltered executable TSFM/statistical combination.",
        )
        for policy in portfolio.all_policies
    )
    return FilterDictionary(tuple(entries))


def require_cached_portfolio_outcomes(
    module: MethodModule,
    portfolio: PolicyPortfolio,
    tasks: Sequence[Task],
    *,
    outcome_cache: OutcomeCache,
    policy_cache: PolicyOutcomeCache,
    isolated_methods: bool,
) -> tuple[Outcome, ...]:
    """Reconstruct all candidate outcomes from exact caches, never calling a model."""
    require_unique_task_ids(tasks)
    portfolio.validate_namespace(module.names())
    python_rows = tuple(
        row
        for method in module.methods
        for row in outcome_cache.require_cached_method(
            method,
            tasks,
            isolated=isolated_methods,
            require_forecasts=True,
        )
    )
    tsfm_rows = tuple(
        policy_cache.require_cached(policy, task)
        for policy in portfolio.tsfm
        for task in tasks
    )
    leaf_rows = python_rows + tsfm_rows
    require_unique_outcome_keys(leaf_rows)
    by_key = {
        (row.method, row.task_id): row for row in leaf_rows
    }
    combined_rows = tuple(
        _run_combined(policy, task, by_key)
        for policy in portfolio.combined
        for task in tasks
    )
    return python_rows + tsfm_rows + combined_rows


def render_filter_source(dictionary: FilterDictionary) -> str:
    """Render the candidate state as an AST-safe executable Python literal."""
    payload = tuple(asdict(entry) for entry in dictionary.entries)
    return (
        '"""Evolving status and applicability registry for executable candidates."""\n\n'
        f"CANDIDATES = {pprint.pformat(payload, width=100, sort_dicts=False)}\n"
    )


def parse_filter_source(source: str) -> FilterDictionary:
    """Parse only the literal CANDIDATES assignment; importing is deliberately forbidden."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise FilterError(f"dictionary source does not parse: {error}") from error
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "CANDIDATES"
    ]
    if len(assignments) != 1:
        raise FilterError("dictionary source must define CANDIDATES exactly once")
    try:
        raw = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError) as error:
        raise FilterError("CANDIDATES must be a Python literal") from error
    if not isinstance(raw, (tuple, list)):
        raise FilterError("CANDIDATES must be a sequence")
    entries = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {
            "name", "family", "status", "applicability", "reason"
        }:
            raise FilterError("each candidate must contain the exact registry fields")
        applicability = item["applicability"]
        if not isinstance(applicability, (tuple, list)):
            raise FilterError("candidate applicability must be a sequence")
        entries.append(
            FilterEntry(
                str(item["name"]), str(item["family"]), str(item["status"]),
                tuple(str(tag) for tag in applicability), str(item["reason"]),
            )
        )
    return FilterDictionary(tuple(entries))


def apply_filter_response(
    parent: FilterDictionary,
    response: str,
    *,
    discardable: frozenset[str],
    allowed_tags: frozenset[str] | None = None,
    required_names: frozenset[str] = frozenset(),
) -> FilterDictionary:
    """Apply one bounded Agent response while preserving all candidate identities."""
    payload = parse_json_object(response)
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list):
        raise FilterError("Agent response 'actions' must be a list")
    if len(raw_actions) > 24:
        raise FilterError("Agent may change at most 24 candidates per generation")
    replacements: dict[str, FilterEntry] = {}
    for raw in raw_actions:
        if not isinstance(raw, dict):
            raise FilterError("each filter action must be an object")
        name = str(raw.get("name", "")).strip()
        status = str(raw.get("status", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        applicability = raw.get("applicability", [])
        current = parent.get(name)
        if current is None:
            raise FilterError(f"unknown candidate {name!r}")
        if name in replacements:
            raise FilterError(f"duplicate action for {name}")
        if not isinstance(applicability, list):
            raise FilterError(f"applicability for {name} must be a list")
        tags = tuple(str(tag) for tag in applicability)
        if allowed_tags is not None and any(tag not in allowed_tags for tag in tags):
            raise FilterError(f"{name} uses an unavailable history-only tag")
        if status == "discard" and name not in discardable:
            raise FilterError(f"discard requires trusted dominance evidence for {name}")
        replacements[name] = FilterEntry(name, current.family, status, tags, reason)
    missing = sorted(required_names - replacements.keys())
    if missing:
        raise FilterError(f"Agent must address required targets: {', '.join(missing)}")
    return FilterDictionary(
        tuple(replacements.get(entry.name, entry) for entry in parent.entries)
    )


def evaluate_filter(
    dictionary: FilterDictionary,
    outcomes: Sequence[Outcome],
    tasks: Sequence[Task],
    *,
    reference_outcomes: Sequence[Outcome],
) -> FilterScore:
    """Select by cross-task history traits, then score selected forecasts with trusted labels."""
    require_unique_task_ids(tasks)
    require_unique_outcome_keys(outcomes)
    by_key = {(row.method, row.task_id): row for row in outcomes}
    references = tuple(reference_outcomes)
    selected: dict[str, str] = {}
    eligible_counts: dict[str, int] = {}
    smae_scores: list[float] = []
    srmse_scores: list[float] = []
    smae_raw_scores: list[float] = []
    srmse_raw_scores: list[float] = []
    smae_clipped_count = 0
    srmse_clipped_count = 0
    task_scaled_pairs: dict[str, tuple[float, float]] = {}
    eligible_attempts = 0
    eligible_successes = 0
    eligible_not_applicable = 0
    eligible_failures = 0
    eligible_crashed = 0
    eligible_invalid = 0
    eligible_missing = 0
    eligible_malformed_success = 0
    penalty = 5.0
    for task in tasks:
        task_tags = frozenset(task.characteristics())
        ranked: list[tuple[float, float, float, str]] = []
        for entry in dictionary.entries:
            if entry.status not in {"keep", "specialized"}:
                continue
            if entry.applicability and not set(entry.applicability).issubset(task_tags):
                continue
            eligible_attempts += 1
            current = by_key.get((entry.name, task.task_id))
            if (
                current is not None
                and current.status == SUCCESS
                and _finite_scaled(current)
            ):
                eligible_successes += 1
            elif current is not None and current.status == NOT_APPLICABLE:
                eligible_not_applicable += 1
            else:
                eligible_failures += 1
                if current is None:
                    eligible_missing += 1
                elif current.status == CRASHED:
                    eligible_crashed += 1
                elif current.status == INVALID:
                    eligible_invalid += 1
                elif current.status == SUCCESS:
                    eligible_malformed_success += 1
                else:
                    eligible_invalid += 1
            history_rows = [
                row
                for row in references
                if row.method == entry.name
                and row.task_id != task.task_id
                and row.status == SUCCESS
                and _finite_scaled(row)
            ]
            if history_rows:
                mean_smae = statistics.fmean(float(row.smae) for row in history_rows)
                mean_srmse = statistics.fmean(float(row.srmse) for row in history_rows)
                ranked.append(
                    (
                        joint_scaled_error(mean_smae, mean_srmse),
                        mean_smae,
                        mean_srmse,
                        entry.name,
                    )
                )
        eligible_counts[task.task_id] = len(ranked)
        if not ranked:
            smae_scores.append(penalty)
            srmse_scores.append(penalty)
            smae_raw_scores.append(math.inf)
            srmse_raw_scores.append(math.inf)
            smae_clipped_count += 1
            srmse_clipped_count += 1
            task_scaled_pairs[task.task_id] = (penalty, penalty)
            continue
        name = min(ranked)[3]
        selected[task.task_id] = name
        row = by_key.get((name, task.task_id))
        if row is None or row.status != SUCCESS or not _finite_scaled(row):
            smae_scores.append(penalty)
            srmse_scores.append(penalty)
            smae_raw_scores.append(math.inf)
            srmse_raw_scores.append(math.inf)
            smae_clipped_count += 1
            srmse_clipped_count += 1
            task_scaled_pairs[task.task_id] = (penalty, penalty)
        else:
            pair = (float(row.smae), float(row.srmse))
            smae_scores.append(pair[0])
            srmse_scores.append(pair[1])
            smae_raw_scores.append(float(row.smae_raw) if row.smae_raw is not None else math.inf)
            srmse_raw_scores.append(float(row.srmse_raw) if row.srmse_raw is not None else math.inf)
            smae_clipped_count += int(bool(row.smae_clipped))
            srmse_clipped_count += int(bool(row.srmse_clipped))
            task_scaled_pairs[task.task_id] = pair
    successes = sum(
        by_key.get((name, task_id)) is not None
        and by_key[(name, task_id)].status == SUCCESS
        for task_id, name in selected.items()
    )
    denominator = max(1, eligible_attempts)
    return FilterScore(
        mean_smae=statistics.fmean(smae_scores) if smae_scores else math.inf,
        mean_srmse=statistics.fmean(srmse_scores) if srmse_scores else math.inf,
        median_smae=statistics.median(smae_scores) if smae_scores else math.inf,
        median_srmse=statistics.median(srmse_scores) if srmse_scores else math.inf,
        coverage=successes / len(tasks) if tasks else 0.0,
        selected=selected,
        eligible_counts=eligible_counts,
        eligible_attempts=eligible_attempts,
        eligible_successes=eligible_successes,
        eligible_not_applicable=eligible_not_applicable,
        eligible_failures=eligible_failures,
        eligible_crashed=eligible_crashed,
        eligible_invalid=eligible_invalid,
        eligible_missing=eligible_missing,
        eligible_malformed_success=eligible_malformed_success,
        eligible_success_rate=eligible_successes / denominator,
        eligible_not_applicable_rate=eligible_not_applicable / denominator,
        eligible_failure_rate=eligible_failures / denominator,
        eligible_crash_rate=eligible_crashed / denominator,
        eligible_invalid_rate=eligible_invalid / denominator,
        eligible_missing_rate=eligible_missing / denominator,
        eligible_malformed_success_rate=eligible_malformed_success / denominator,
        se_smae=standard_error(smae_scores) if smae_scores else math.inf,
        se_srmse=standard_error(srmse_scores) if srmse_scores else math.inf,
        p90_smae=linear_quantile(smae_scores, 0.90) if smae_scores else math.inf,
        p95_smae=linear_quantile(smae_scores, 0.95) if smae_scores else math.inf,
        p90_srmse=linear_quantile(srmse_scores, 0.90) if srmse_scores else math.inf,
        p95_srmse=linear_quantile(srmse_scores, 0.95) if srmse_scores else math.inf,
        p90_smae_raw=linear_quantile(smae_raw_scores, 0.90) if smae_raw_scores else math.inf,
        p95_smae_raw=linear_quantile(smae_raw_scores, 0.95) if smae_raw_scores else math.inf,
        p90_srmse_raw=linear_quantile(srmse_raw_scores, 0.90) if srmse_raw_scores else math.inf,
        p95_srmse_raw=linear_quantile(srmse_raw_scores, 0.95) if srmse_raw_scores else math.inf,
        smae_clipped_count=smae_clipped_count,
        smae_clipped_rate=smae_clipped_count / len(tasks) if tasks else 1.0,
        srmse_clipped_count=srmse_clipped_count,
        srmse_clipped_rate=srmse_clipped_count / len(tasks) if tasks else 1.0,
        task_scaled_pairs=task_scaled_pairs,
        task_count=len(tasks),
    )


def evolve_filter_once(
    parent: FilterDictionary,
    train_tasks: Sequence[Task],
    dev_tasks: Sequence[Task],
    outcomes: Sequence[Outcome],
    agent: LLMClient,
    *,
    generation: int,
    transcript_dir: str | Path,
    reviewed_names: frozenset[str] = frozenset(),
    required_target_limit: int = 24,
    required_targets: Sequence[str] | None = None,
) -> FilterGeneration:
    """Produce one dictionary Child with one Agent call and a trusted Train/Dev gate."""
    train_ids = {task.task_id for task in train_tasks}
    dev_ids = {task.task_id for task in dev_tasks}
    train_outcomes = tuple(row for row in outcomes if row.task_id in train_ids)
    dev_outcomes = tuple(row for row in outcomes if row.task_id in dev_ids)
    train_parent = evaluate_filter(
        parent, train_outcomes, train_tasks, reference_outcomes=train_outcomes
    )
    reports = _evidence(parent, train_outcomes, train_tasks)
    discardable = _strictly_discardable(parent, train_outcomes, train_tasks)
    allowed_tags = frozenset(tag for task in train_tasks for tag in task.characteristics())
    selection_failures = _selection_failures(
        train_parent, train_outcomes, train_tasks
    )
    if required_targets is None:
        target_batch = _required_review_targets(
            reports,
            selection_failures,
            reviewed_names=reviewed_names,
            limit=required_target_limit,
        )
    else:
        target_batch = tuple(str(name) for name in required_targets)
        if len(target_batch) > 24:
            raise FilterError("required target batch exceeds the 24-action safety limit")
        if len(target_batch) != len(set(target_batch)):
            raise FilterError("required target batch contains duplicate candidates")
        unknown = sorted(set(target_batch) - {entry.name for entry in parent.entries})
        if unknown:
            raise FilterError(
                "required target batch contains unknown candidates: " + ", ".join(unknown)
            )
    required_target_set = frozenset(target_batch)
    request = json.dumps(
        {
            "generation": generation,
            "task_count": len(train_tasks),
            "candidate_count": len(parent.entries),
            "allowed_history_tags": sorted(allowed_tags),
            "trusted_discardable": sorted(discardable),
            "current_dictionary": [asdict(entry) for entry in parent.entries],
            "train_evidence": reports,
            "selection_failures": selection_failures,
            "required_targets": list(target_batch),
            "required_target_evidence": _conditional_evidence(
                required_target_set, train_outcomes, train_tasks
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    response = agent.complete(
        system=FILTER_SYSTEM,
        messages=[{"role": "user", "content": request}],
        temperature=0.0,
    )
    directory = Path(transcript_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"generation_{generation:03d}_filter"
    (directory / f"{prefix}_request.txt").write_text(request, encoding="utf-8")
    (directory / f"{prefix}_response.json").write_text(response.text, encoding="utf-8")
    child = apply_filter_response(
        parent,
        response.text,
        discardable=discardable,
        allowed_tags=allowed_tags,
        required_names=required_target_set,
    )
    train_child = evaluate_filter(
        child, train_outcomes, train_tasks, reference_outcomes=train_outcomes
    )
    train_gate = _compare_filter_split(
        train_parent, train_child, split="Train"
    )
    if not train_gate.accepted:
        return FilterGeneration(
            generation, parent, child, train_parent, train_child,
            None, None, False, train_gate.reason, 1, target_batch,
        )
    dev_parent = evaluate_filter(
        parent, dev_outcomes, dev_tasks, reference_outcomes=train_outcomes
    )
    dev_child = evaluate_filter(
        child, dev_outcomes, dev_tasks, reference_outcomes=train_outcomes
    )
    gate = _compare_filter_split(dev_parent, dev_child, split="Dev")
    return FilterGeneration(
        generation, parent, child, train_parent, train_child,
        dev_parent, dev_child, gate.accepted, gate.reason, 1, target_batch,
    )


def compare_filter_scores(
    train_parent: FilterScore,
    train_child: FilterScore,
    dev_parent: FilterScore,
    dev_child: FilterScore,
) -> FilterGateResult:
    """Require independent Train and Dev Pareto gains plus scaled safety."""
    train_gate = _compare_filter_split(train_parent, train_child, split="Train")
    if not train_gate.accepted:
        return train_gate
    return _compare_filter_split(dev_parent, dev_child, split="Dev")


def _compare_filter_split(
    parent: FilterScore, child: FilterScore, *, split: str
) -> FilterGateResult:
    if not _forecast_improved(parent, child):
        return FilterGateResult(
            False,
            f"rejected: {split} sMAE/sRMSE did not improve under the Pareto gate",
        )
    safety_regression = _scaled_safety_regression(parent, child)
    if safety_regression is not None:
        return FilterGateResult(
            False, f"rejected: {split} {safety_regression} regressed"
        )
    if not _forecast_non_regression(parent, child):
        return FilterGateResult(
            False, f"rejected: {split} coverage or median scaled error regressed"
        )
    if not _reliability_non_regression(parent, child):
        return FilterGateResult(False, f"rejected: {split} reliability regressed")
    return FilterGateResult(
        True,
        f"accepted: {split} scaled forecast pair improved with safety",
    )


def _forecast_non_regression(parent: FilterScore, child: FilterScore) -> bool:
    tolerance = 1e-12
    return (
        child.coverage + tolerance >= parent.coverage
        and child.mean_smae <= parent.mean_smae + tolerance
        and child.mean_srmse <= parent.mean_srmse + tolerance
        and child.median_smae <= parent.median_smae + tolerance
        and child.median_srmse <= parent.median_srmse + tolerance
    )


def _forecast_improved(parent: FilterScore, child: FilterScore) -> bool:
    tolerance = 1e-12
    return pareto_scaled_improvement(
        parent.mean_smae,
        parent.mean_srmse,
        child.mean_smae,
        child.mean_srmse,
        tolerance=tolerance,
    )


def _scaled_safety_regression(
    parent: FilterScore, child: FilterScore
) -> str | None:
    tolerance = 1e-12
    tail_fields = (
        "p90_smae",
        "p95_smae",
        "p90_srmse",
        "p95_srmse",
        "p90_smae_raw",
        "p95_smae_raw",
        "p90_srmse_raw",
        "p95_srmse_raw",
    )
    for field_name in tail_fields:
        parent_value = float(getattr(parent, field_name))
        child_value = float(getattr(child, field_name))
        if not child_value <= parent_value + tolerance:
            return field_name
    for field_name in ("smae_clipped_count", "srmse_clipped_count"):
        if int(getattr(child, field_name)) > int(getattr(parent, field_name)):
            return field_name
    return None


def _reliability_non_regression(parent: FilterScore, child: FilterScore) -> bool:
    tolerance = 1e-12
    return child.eligible_success_rate + tolerance >= parent.eligible_success_rate


def _selection_failures(
    score: FilterScore,
    outcomes: Sequence[Outcome],
    tasks: Sequence[Task],
) -> list[dict[str, object]]:
    """Expose Train-only false selections and the executable best alternative."""
    by_task: dict[str, list[Outcome]] = {}
    for row in outcomes:
        by_task.setdefault(row.task_id, []).append(row)
    failures = []
    for task in tasks:
        selected_name = score.selected.get(task.task_id)
        if selected_name is None:
            failures.append(
                {"task_id": task.task_id, "selected": None, "selected_status": "none"}
            )
            continue
        rows = by_task.get(task.task_id, [])
        selected = next((row for row in rows if row.method == selected_name), None)
        successful = [
            row for row in rows
            if row.status == SUCCESS and _finite_scaled(row)
        ]
        best = min(successful, key=_scaled_order) if successful else None
        selected_pair = (
            (float(selected.smae), float(selected.srmse))
            if selected is not None and selected.status == SUCCESS and _finite_scaled(selected)
            else None
        )
        materially_bad = (
            selected_pair is None
            or best is None
            or selected_pair[0] > float(best.smae) * 1.25 + 1e-12
            or selected_pair[1] > float(best.srmse) * 1.25 + 1e-12
        )
        if materially_bad:
            failures.append(
                {
                    "task_id": task.task_id,
                    "history_tags": list(task.characteristics()),
                    "selected": selected_name,
                    "selected_status": selected.status if selected is not None else "missing",
                    "selected_smae": selected_pair[0] if selected_pair is not None else None,
                    "selected_srmse": selected_pair[1] if selected_pair is not None else None,
                    "best_available": best.method if best is not None else None,
                    "best_smae": float(best.smae) if best is not None else None,
                    "best_srmse": float(best.srmse) if best is not None else None,
                }
            )
    return failures


def _required_selection_targets(
    failures: Sequence[Mapping[str, object]],
) -> frozenset[str]:
    """Require correction of candidates repeatedly selected into bad outcomes."""
    counts: dict[str, int] = {}
    for failure in failures:
        name = failure.get("selected")
        if isinstance(name, str) and name:
            counts[name] = counts.get(name, 0) + 1
    return frozenset(name for name, count in counts.items() if count >= 2)


def _required_review_targets(
    reports: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
    *,
    reviewed_names: frozenset[str] = frozenset(),
    limit: int = 24,
) -> tuple[str, ...]:
    """Return one deterministic priority batch of unresolved Train-evidenced targets."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 24:
        raise FilterError("required target limit must be an integer from 1 to 24")
    selected_counts: dict[str, int] = {}
    for failure in failures:
        name = failure.get("selected")
        if isinstance(name, str) and name:
            selected_counts[name] = selected_counts.get(name, 0) + 1
    selected = {
        name for name, count in selected_counts.items() if count >= 2
    }
    by_name = {str(report["name"]): report for report in reports}
    partial = {
        name
        for name, report in by_name.items()
        if int(report["success"]) > 0 and int(report["not_applicable"]) > 0
    }
    defective = {
        name
        for name, report in by_name.items()
        if int(report["crashed"]) > 0 or int(report["invalid"]) > 0
    }
    unresolved = (selected | partial | defective) - set(reviewed_names)

    def priority(name: str) -> tuple[object, ...]:
        report = by_name.get(name, {})
        defects = int(report.get("crashed", 0)) + int(report.get("invalid", 0))
        not_applicable = int(report.get("not_applicable", 0))
        mean_joint = report.get("mean_joint_scaled_error")
        finite_joint = (
            float(mean_joint)
            if isinstance(mean_joint, (int, float)) and math.isfinite(float(mean_joint))
            else -1.0
        )
        tier = 0 if name in selected else (1 if defects else 2)
        return (
            tier,
            -selected_counts.get(name, 0),
            -defects,
            -not_applicable,
            -finite_joint,
            name,
        )

    return tuple(sorted(unresolved, key=priority)[:limit])


def _conditional_evidence(
    names: frozenset[str],
    outcomes: Sequence[Outcome],
    tasks: Sequence[Task],
) -> dict[str, dict[str, dict[str, object]]]:
    """Summarize method outcomes by label-free history tag for applicability learning."""
    tags_by_task = {
        task.task_id: frozenset(task.characteristics()) for task in tasks
    }
    result: dict[str, dict[str, dict[str, object]]] = {}
    for name in sorted(names):
        rows = [row for row in outcomes if row.method == name]
        tag_rows: dict[str, dict[str, object]] = {}
        for tag in sorted({tag for tags in tags_by_task.values() for tag in tags}):
            subset = [row for row in rows if tag in tags_by_task.get(row.task_id, ())]
            if not subset:
                continue
            successful = [
                row for row in subset
                if row.status == SUCCESS and _finite_scaled(row)
            ]
            tag_rows[tag] = {
                "tasks": len(subset),
                "success": len(successful),
                "not_applicable": sum(row.status == NOT_APPLICABLE for row in subset),
                "crashed": sum(row.status == CRASHED for row in subset),
                "invalid": sum(row.status == INVALID for row in subset),
                "mean_smae": statistics.fmean(float(row.smae) for row in successful)
                if successful else None,
                "mean_srmse": statistics.fmean(float(row.srmse) for row in successful)
                if successful else None,
            }
        result[name] = tag_rows
    return result


def _evidence(
    dictionary: FilterDictionary,
    outcomes: Sequence[Outcome],
    tasks: Sequence[Task],
) -> list[dict[str, object]]:
    by_task: dict[str, list[Outcome]] = {}
    for row in outcomes:
        by_task.setdefault(row.task_id, []).append(row)
    payload = []
    for entry in dictionary.entries:
        rows = [row for row in outcomes if row.method == entry.name]
        successes = [row for row in rows if row.status == SUCCESS and _finite_scaled(row)]
        ranks = []
        top_quartile = 0
        for row in successes:
            peers = sorted(
                joint_scaled_error(float(peer.smae), float(peer.srmse))
                for peer in by_task.get(row.task_id, ())
                if peer.status == SUCCESS and _finite_scaled(peer)
            )
            if not peers:
                continue
            row_score = joint_scaled_error(float(row.smae), float(row.srmse))
            rank = sum(value < row_score for value in peers) / max(1, len(peers) - 1)
            ranks.append(rank)
            top_quartile += rank <= 0.25
        payload.append(
            {
                "name": entry.name,
                "family": entry.family,
                "current_status": entry.status,
                "current_applicability": list(entry.applicability),
                "success": len(successes),
                "not_applicable": sum(row.status == NOT_APPLICABLE for row in rows),
                "crashed": sum(row.status == CRASHED for row in rows),
                "invalid": sum(row.status == INVALID for row in rows),
                "mean_smae": statistics.fmean(float(row.smae) for row in successes)
                if successes else None,
                "mean_srmse": statistics.fmean(float(row.srmse) for row in successes)
                if successes else None,
                "mean_joint_scaled_error": statistics.fmean(
                    joint_scaled_error(float(row.smae), float(row.srmse))
                    for row in successes
                ) if successes else None,
                "median_rank_percentile": statistics.median(ranks) if ranks else None,
                "top_quartile_rate": top_quartile / len(ranks) if ranks else 0.0,
            }
        )
    return payload


def _strictly_discardable(
    dictionary: FilterDictionary,
    outcomes: Sequence[Outcome],
    tasks: Sequence[Task],
) -> frozenset[str]:
    by_method = {
        entry.name: {row.task_id: row for row in outcomes if row.method == entry.name}
        for entry in dictionary.entries
    }
    task_ids = {task.task_id for task in tasks}
    discardable = set()
    for entry in dictionary.entries:
        rows = by_method[entry.name]
        if set(rows) != task_ids or any(row.status != SUCCESS for row in rows.values()):
            continue
        for other, other_rows in by_method.items():
            if other == entry.name or set(other_rows) != task_ids:
                continue
            comparisons = [
                other_rows[task_id].status == SUCCESS
                and _finite_scaled(other_rows[task_id])
                and _finite_scaled(rows[task_id])
                and float(other_rows[task_id].smae) <= float(rows[task_id].smae) + 1e-12
                and float(other_rows[task_id].srmse) <= float(rows[task_id].srmse) + 1e-12
                for task_id in task_ids
            ]
            strict = any(
                _finite_scaled(other_rows[task_id])
                and _finite_scaled(rows[task_id])
                and (
                    float(other_rows[task_id].smae) < float(rows[task_id].smae) - 1e-12
                    or float(other_rows[task_id].srmse) < float(rows[task_id].srmse) - 1e-12
                )
                for task_id in task_ids
            )
            if all(comparisons) and strict:
                discardable.add(entry.name)
                break
    return frozenset(discardable)


def _finite_scaled(row: Outcome) -> bool:
    return (
        row.smae is not None
        and row.srmse is not None
        and math.isfinite(float(row.smae))
        and math.isfinite(float(row.srmse))
    )


def _scaled_order(row: Outcome) -> tuple[float, float, float, str]:
    assert row.smae is not None and row.srmse is not None
    smae = float(row.smae)
    srmse = float(row.srmse)
    return joint_scaled_error(smae, srmse), smae, srmse, row.method
