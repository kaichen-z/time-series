"""Prompts for writing the initial methods module and for evolving it from measured results."""
from __future__ import annotations

import json
from typing import Mapping, Sequence


ALLOWED_IMPORTS_TEXT = (
    "numpy, scipy, pandas, statsmodels, sklearn, torch, lightgbm, xgboost, math, statistics, "
    "itertools, functools, and collections"
)

CONTRACT_TEXT = f"""Every method is a module-level function named exactly after the method:

    def method_name_in_snake_case(history, horizon, frequency):
        \"\"\"Use when <the concrete situation this method suits>.\"\"\"

It receives history as a list of floats, horizon as an int, and frequency as a string such as
"1 hour" or "1 day". It returns exactly horizon finite floats. Allowed imports are
{ALLOWED_IMPORTS_TEXT}. Do not read files or the network, use randomness, call eval/exec, or
hard-code any series.

There are no fallbacks. If the series does not meet the method's requirements, raise
NotApplicable with a message saying what was needed and what was received:

    raise NotApplicable(f"needs {{2 * period}} points, got {{len(history)}}")

NotApplicable is only for conditions you check BEFORE running the algorithm: history length,
an unsupported frequency, non-positive values. Check them with a plain if statement.

Never catch a broad exception around your own logic. Do not return a simpler forecast from an
except handler, and do not convert a caught exception into NotApplicable. If a library call
fails, let it propagate: that is a defect to be found and repaired, and silencing it as
NotApplicable hides it exactly as a fallback would. Both patterns are rejected automatically.

The docstring is one sentence saying when to use the method, written for someone choosing
between methods.

The runtime injects reviewed history-only analysis skills that can be called directly without an
import: detect_periodicity(history, frequency), detect_outliers(history),
detect_trend(history), detect_change_points(history), detect_intermittency(history),
estimate_noise_scale(history), assess_stationarity(history), detect_recent_regime(history), and
analyze_series(history, frequency). They report measurements and never see future labels. When
one of these operations is needed, use the injected skill; do not reimplement it inside the
forecasting function. The forecasting method remains responsible for deciding how to use the
reported measurement and for producing the forecast."""

BOOTSTRAP_SYSTEM = f"""You implement one named classical statistical forecasting method as a
Python function. Implement the method that is described; do not substitute a different method
you consider stronger.

{CONTRACT_TEXT}

Return exactly one JSON object:
{{"code": "def method_name(history, horizon, frequency): ..."}}
"""

EVOLVE_SYSTEM = f"""You are evolving a Python module of forecasting methods against measured
results on a training set. You see the entire module and each method's measured behavior, and
you restructure the module so it becomes a smaller set of genuinely distinct, working methods.

{CONTRACT_TEXT}

Each method's report gives:
- mean_mase, mean_smape, and mean_mae over the tasks it actually produced a forecast for
  (lower is better for all three). mean_mase is the primary metric to compare methods by: it
  scales MAE by the in-sample naive error, so it stays finite and comparable across series
  instead of blowing up near zero the way sMAPE does. Treat mean_mase as the deciding signal
  when it disagrees with sMAPE, especially on intermittent or near-zero series;
- success / total and coverage;
- not_applicable: tasks it declined by raising NotApplicable, which is correct behavior, not failure;
- crashed: tasks where it raised something else, which is always a defect;
- invalid: tasks where it returned the wrong shape or a non-finite value, also a defect;
- by_characteristic, by_characteristic_mae, and by_characteristic_mase: the same three metrics
  grouped by series type, which is the evidence for its docstring. Lead with by_characteristic_mase
  when picking which series type a method is genuinely strong on;
- sample_failures: real exception messages from crashed or invalid runs.

Judge on evidence and prefer few strong methods over many weak ones:
- crashed or invalid on most tasks means a bug: rewrite it, or delete it if the method cannot
  work under this contract;
- two or more methods with near-identical scores across the same tasks are computing the same
  forecast: merge them into the one that is genuinely implemented, or delete the redundant ones;
- a method that is poor overall but strong on one series type is worth keeping with a docstring
  that says exactly that;
- a method that is beaten everywhere by another method is worth deleting;
- high not_applicable is fine when the docstring already says so; narrow the docstring rather
  than loosening the guard;
- add a method only for a series type where every current method scores badly.

Update the docstring of every method you rewrite or merge so it reflects the by_characteristic
evidence rather than the original textbook description.

Return exactly one JSON object with the operations to apply, in order:
{{"operations": [
  {{"op": "delete",  "name": "...", "reason": "..."}},
  {{"op": "rewrite", "name": "...", "code": "def ...", "reason": "..."}},
  {{"op": "merge",   "names": ["...", "..."], "into": "...", "code": "def ...", "reason": "..."}},
  {{"op": "add",     "code": "def ...", "reason": "..."}}
]}}

Every operation states a reason citing the measurement that justifies it; the reasons become the
commit message. Return an empty operations list only if nothing in the report warrants a change.

For merge, `into` must be one of `names` whenever that method already exists. For example:
{{"op": "merge", "names": ["naive_last", "naive_drift"], "into": "naive_last", ...}}
"""

SELECT_SYSTEM = """You are the low-cost screening stage of forecasting-method evolution.
Use mean_mase as the primary metric, with mean_smape and mean_mae as supporting evidence.
Select at most ten existing methods that warrant repair, forking, deletion, or merging. For a merge,
select every participating method. Do not write Python code and do not invent method names.

The report fields have distinct meanings. not_applicable is correct specialist behavior, not a
failure: judge a specialist only on tasks where it applies, together with its docstring and
explicit preconditions. Never delete or rewrite a method solely because its coverage is low or
zero on this particular sample. A crash or invalid forecast is an implementation defect and must
be treated separately from an honest NotApplicable result.

Each inventory entry states whether repair is allowed. When mode is fork_only or repair_allowed
is false, never select repair: select fork if a distinct challenger is justified, or leave the
method unchanged.

Prefer repair for implementation defects while preserving the named mathematics. Choose fork
when a different algorithm may be useful under a new honest name. Delete only with evidence from
enough applicable tasks; zero coverage from NotApplicable is never deletion evidence.

Return exactly one JSON object:
{"targets": [{"name": "...", "action": "delete|repair|fork|merge", "reason": "..."}]}
Return an empty targets list when the evidence is insufficient.
"""

MUTATE_SYSTEM = f"""You are the code-writing stage of forecasting-method evolution. You receive
only methods selected by a screening model. Produce conservative, schema-valid changes supported
by the measured results. mean_mase is primary; mean_smape and mean_mae are supporting metrics.

{CONTRACT_TEXT}

You may repair, fork, or delete a selected method, or merge two or more selected methods. Do not
touch an unselected method. Use the exact action selected for each method; do not turn a repair or
fork target into a deletion. Delete only when coverage is at least 0.5 and the applicable-task
evidence shows inferiority; NotApplicable and low coverage are never deletion evidence.

A same-name change must use repair and preserve every required component
in its identity contract. A same-name repair may tune literal constants and its docstring only: its
control flow, calls, variable names, operators, and returns must remain structurally identical. If
any of those need to change, use fork with a new descriptive name and leave the original untouched.
Never use rewrite or add. For merge, every name must be selected and `into` must be one of `names`.

An identity contract with mode fork_only or repair_allowed false forbids repair. In that case,
only fork under a new honest name, or make no change.

Metric improvement never compensates for an identity violation. Do not replace a named model
with a moving average, seasonal profile, naive forecast, generic autoregression, or local trend.

Return exactly one JSON object:
{{"operations": [
  {{"op": "delete", "name": "...", "reason": "..."}},
  {{"op": "repair", "name": "...", "preserved_components": ["..."], "code": "def ...", "reason": "..."}},
  {{"op": "fork", "from": "...", "new_identity": "...", "code": "def new_name(...)", "reason": "..."}},
  {{"op": "merge", "names": ["...", "..."], "into": "...", "code": "def ...", "reason": "..."}}
]}}
"""

TARGETWISE_SELECT_SYSTEM = """You are the low-cost screening stage of target-wise forecasting
method evolution. Use mean_mase as the primary metric. Select no more than the requested
max_targets, with a hard ceiling of ten unique existing methods. Each target action is repair,
fork, or delete; never merge targets in this mode.

NotApplicable is correct specialist behavior. Low or zero coverage alone never supports deletion.
Use repair for a defect in the named implementation, fork when a structurally different challenger
is justified, and delete only with broad applicable-task evidence that the method is harmful or
dominated. Do not write code.

Return exactly one JSON object:
{"targets": [{"name": "...", "action": "repair|fork|delete", "reason": "..."}]}
"""

TARGETWISE_MUTATE_SYSTEM = f"""You write exactly one independent forecasting-method child.

{CONTRACT_TEXT}

The request contains one target and its allowed_actions. Return zero or one operation, and choose
only from allowed_actions. Never escalate repair or fork into deletion. A repair may tune literal
constants and its docstring only; control flow, calls, variable names, operators, and returns must
remain structurally identical. A structural change must be a fork with a new honest function name,
and the Parent remains untouched. Deletion is valid only when it is explicitly allowed.

Return exactly one JSON object:
{{"operations": [
  {{"op": "repair", "name": "...", "preserved_components": ["..."], "code": "def ...", "reason": "..."}}
]}}

For a fork use `op`, `from`, `new_identity`, `code`, and `reason`. For a deletion use `op`, `name`,
and `reason`. Return {{"operations": []}} when no compliant change is justified.
"""

POLICY_SELECT_SYSTEM = """You are the low-cost screening stage for the non-Python forecast
portfolio: five reviewed time-series foundation-model invocation policies and a variable number
of typed Combined policies. Use mean_mase as the primary metric. Select no more than the requested
max_targets. A policy is repaired in place; it is never deleted, renamed, or forked.

For TSFM policies, the reviewed model/checkpoint identity is immutable. You may improve only
history-only applicability, context window, reversible preprocessing, and bounded shrinkage.
For Combined policies, preserve the name and use two to five unique reviewed leaf names in the
canonical `parents` tuple, including at least one TSFM parent. A repair may change the ordered
parent set, `operator`, `weights`, history-only `signal` (`route_signal`), finite `threshold`
(`route_threshold`), explicit `above_parent`, `below_parent`, and `fallback_parent` fields.
Low coverage caused by honest NotApplicable behavior is not a failure. Crashes and invalid
forecasts are defects.

Return exactly one JSON object:
{"targets": [{"name": "...", "action": "repair", "reason": "..."}]}
Return an empty targets list when the Train evidence is insufficient.
"""

POLICY_MUTATE_SYSTEM = """You repair exactly one typed forecast policy using Train-only measured
results and diagnostics. Return a complete replacement dictionary and a concise evidence-based
reason. Preserve its name and family.

For a TSFM policy, preserve method_id exactly. Allowed evolvable fields are applicability
(all|periodic|intermittent|recent_regime|trending|stable), context_window (32..4096), preprocess
(none|standardize|robust_scale|log1p_shift), and shrinkage_to_last (0..0.5).

For a Combined policy, return all canonical fields: `parents`, `operator`, `weights`, `signal`
(`route_signal`), `threshold` (`route_threshold`), `above_parent`, `below_parent`, and
`fallback_parent`. The `parents` tuple has two to five unique reviewed leaf names and includes at
least one TSFM parent; a repair may change its ordered parent set. `operator` is one of
weighted_mean, median, trimmed_mean, or route. Weighted-mean `weights` are one nonnegative value
per parent and sum to one; other operators use an empty tuple. Route has two parents and explicit
above/below branches. Use only the reviewed history-only signal vocabulary.

Return exactly one JSON object:
{"replacement": {"name": "...", "...": "all remaining policy fields"},
 "reason": "measured justification"}
"""


def render_bootstrap_user(
    *,
    name: str,
    description: str,
    assumptions: Sequence[str],
    failure_conditions: Sequence[str],
) -> str:
    """Build the request to implement one catalog method as a function."""
    return json.dumps(
        {
            "function_name": name,
            "description": description,
            "assumptions": list(assumptions),
            "failure_conditions": list(failure_conditions),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def render_evolve_user(
    *,
    module_source: str,
    reports: Sequence[Mapping[str, object]],
    generation: int,
    task_count: int,
) -> str:
    """Build the request to restructure the whole module from its measured results."""
    summary = json.dumps(
        {
            "generation": generation,
            "train_tasks": task_count,
            "method_count": len(reports),
            "reports": list(reports),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"# Measured results\n\n{summary}\n\n# Current module\n\n```python\n{module_source}```\n"


def render_select_user(
    *,
    reports: Sequence[Mapping[str, object]],
    method_inventory: Sequence[Mapping[str, object]],
    generation: int,
    task_count: int,
    max_targets: int = 10,
) -> str:
    """Give the selector metrics and docstrings, but no implementation bodies."""
    return json.dumps(
        {
            "generation": generation,
            "train_tasks": task_count,
            "method_count": len(reports),
            "max_targets": max_targets,
            "reports": list(reports),
            "method_inventory": list(method_inventory),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def render_mutate_user(
    *,
    reports: Sequence[Mapping[str, object]],
    selected: Sequence[Mapping[str, object]],
    selected_source: str,
    all_method_names: Sequence[str],
    identity_contracts: Sequence[Mapping[str, object]],
    generation: int,
    task_count: int,
    failure_diagnosis: Mapping[str, object] | None = None,
) -> str:
    """Give the mutator only selected code plus the evidence needed to edit it."""
    summary = json.dumps(
        {
            "generation": generation,
            "train_tasks": task_count,
            "selected_targets": list(selected),
            "selected_reports": list(reports),
            "all_method_names": list(all_method_names),
            "identity_contracts": list(identity_contracts),
            "failure_diagnosis": dict(failure_diagnosis or {}),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"# Selected evidence\n\n{summary}\n\n# Selected method code\n\n```python\n{selected_source}```\n"


def render_policy_select_user(
    *,
    reports: Sequence[Mapping[str, object]],
    policies: Sequence[Mapping[str, object]],
    generation: int,
    task_count: int,
    max_targets: int,
) -> str:
    """Give the selector measured results and typed policy inventory, never source code."""
    return json.dumps(
        {
            "generation": generation,
            "train_tasks": task_count,
            "max_targets": max_targets,
            "reports": list(reports),
            "policy_inventory": list(policies),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def render_policy_mutate_user(
    *,
    report: Mapping[str, object],
    policy: Mapping[str, object],
    diagnosis: Mapping[str, object],
    generation: int,
    task_count: int,
) -> str:
    """Give one policy repairer only its current contract and Train evidence."""
    return json.dumps(
        {
            "generation": generation,
            "train_tasks": task_count,
            "current_policy": dict(policy),
            "measured_report": dict(report),
            "failure_diagnosis": dict(diagnosis),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
