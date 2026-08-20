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
between methods."""

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
