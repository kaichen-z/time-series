"""Prompts for writing the initial methods module and for evolving it from measured results."""
from __future__ import annotations

import json
from typing import Mapping, Sequence

from .skills_index import render_index


ALLOWED_IMPORTS_TEXT = (
    "numpy, scipy, pandas, statsmodels, sklearn, torch, lightgbm, xgboost, math, statistics, "
    "itertools, functools, and collections"
)

SKILLS_INDEX = render_index()

CONTRACT_TEXT = f"""Every method is a module-level function named exactly after the method:

    def method_name_in_snake_case(history, horizon, frequency):
        \"\"\"When to use it, why it wins there, and any caveat on how it behaves.\"\"\"

It receives history as a list of floats, horizon as an int, and frequency as a string such as
"1 hour" or "1 day". It returns exactly horizon finite floats. Do not read files or the network,
use randomness, call eval/exec, or hard-code any series.

A frozen library of analysis skills is already imported as P at the top of the module, and
NotApplicable comes from it too. Build methods by composing those skills. Reimplementing a
skill inline is the single most common way to introduce a defect that the measurements cannot
see, so do not do it: the library is tested and you are not. Reach for {ALLOWED_IMPORTS_TEXT}
only for glue the library genuinely does not cover, and import it inside the function body --
a module-level import you write is discarded when the module is rewritten, and every method
that relied on it then fails.

There are no fallbacks. If the series does not meet the method's requirements, raise
NotApplicable with a message saying what was needed and what was received:

    raise NotApplicable(f"needs {{2 * period}} points, got {{len(history)}}")

NotApplicable is only for conditions you check BEFORE running the algorithm: history length,
an unsupported frequency, non-positive values. Check them with a plain if statement.

Never catch a broad exception around your own logic. Do not return a simpler forecast from an
except handler, and do not convert a caught exception into NotApplicable. If a library call
fails, let it propagate: that is a defect to be found and repaired, and silencing it as
NotApplicable hides it exactly as a fallback would. Both patterns are rejected automatically.

The docstring is written for someone choosing between methods. A single sentence is rarely
enough to justify that choice: say when to use the method, why it wins there over the
alternatives (cite the evidence), and any caveat on how it behaves or fails.

# The skill library, available as P

{SKILLS_INDEX}"""

BOOTSTRAP_SYSTEM = f"""You implement one named classical statistical forecasting method as a
Python function. Implement the method that is described; do not substitute a different method
you consider stronger.

{CONTRACT_TEXT}

Return exactly one JSON object:
{{"code": "def method_name(history, horizon, frequency): ..."}}
"""

EVOLVE_SYSTEM = f"""You are evolving a Python module of forecasting methods against measured
results on a training set. You see the entire module and each method's measured behavior, and
you grow it into a set of methods that between them cover the series types
in the data.

Methods are built by composing the frozen skill library. The interesting work is finding
combinations that no current method tries: a different cost or search in the segmentation, a
different decomposition feeding a different model, an analogue match under a different metric.
Adding a well-motivated new combination is worth more than pruning a mediocre one, because the
library spans far more combinations than any module will hold.

{CONTRACT_TEXT}

Each method's report gives:
- mean_rank: the method's average rank by MAE among the methods that forecast the same task,
  1.0 being best. This is the primary signal for comparing methods, because ranking within a
  task before averaging stops one large-magnitude series from deciding the whole comparison,
  which a mean over raw errors cannot avoid;
- mean_variance_ratio, mean_shape_correlation and mean_change_mae: whether the forecast tracks
  the series or merely sits near its level. A flat forecast has a variance ratio of 0.0 and a
  shape correlation of 0.0 however good its MAE, and its mean_change_mae equals the series' own
  volatility. A method with respectable error but a variance ratio near zero has found the mean,
  not the dynamics; say so in its docstring, and prefer a method that tracks the shape when the
  errors are close. Beware the opposite too: a variance ratio far above 1.0 is a forecast
  swinging more wildly than the truth;
- mean_mae and mean_rmse over the tasks it actually produced a forecast for (lower is better
  for both). mean_rmse penalizes large deviations more heavily, so treat a method with much
  worse mean_rmse than its mean_mae
  suggests as one that occasionally produces large errors;
- success / total and coverage;
- not_applicable: tasks it declined by raising NotApplicable, which is correct behavior, not failure;
- crashed: tasks where it raised something else, which is always a defect;
- invalid: tasks where it returned the wrong shape or a non-finite value, also a defect;
- by_characteristic_mae and by_characteristic_rmse: the same two metrics grouped by series type,
  which is the evidence for its docstring. Lead with by_characteristic_mae when picking which
  series type a method is genuinely strong on;
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
  than loosening the guard. A method that declines most tasks but is excellent on the ones it
  accepts is valuable, and its mean is not comparable with a method that accepts everything;
- prefer measured evidence. Do not keep the worse of two methods because it looks
  simpler or more familiar; if you keep it anyway, say in the reason that you are overriding
  the measurement and why.

Add methods deliberately, not only as a last resort:
- a series type where every current method scores badly needs a method built for it;
- a skill or option the module never uses -- an unused cost, search, dictionary, distance or
  model -- is a hypothesis nobody has tested yet;
- two strong methods that fail on different tasks suggest a third combining what each does
  well.

Update the docstring of every method you rewrite, merge or add so it reflects the
evidence rather than a textbook description.

Return exactly one JSON object with the operations to apply, in order:
{{"operations": [
  {{"op": "delete",  "name": "...", "reason": "..."}},
  {{"op": "rewrite", "name": "...", "code": "def ...", "reason": "..."}},
  {{"op": "merge",   "names": ["...", "..."], "into": "...", "code": "def ...", "reason": "..."}},
  {{"op": "add",     "code": "def ...", "reason": "..."}}
]}}

Operations apply in sequence to the module as your earlier operations have already changed it.
A merge removes every method it names, so do not also delete those methods afterwards, and do
not name a method an earlier operation already deleted or merged away. Each method should be
touched by exactly one operation.

Every operation states a reason citing the measurement that justifies it; the reasons become the
commit message. Return an empty operations list only if nothing in the report warrants a change.

Prefer `add` and `rewrite`. `delete` and `merge` only shrink what the seed already contained,
and a module that never adds anything can never become better than its seed.
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
