"""Prompts for writing the initial methods module and for evolving it from measured results."""
from __future__ import annotations

import json
from typing import Mapping, Sequence

from .history import History


ALLOWED_IMPORTS_SENTENCE = (
    "numpy, scipy, pandas, statsmodels, sklearn, torch, lightgbm, xgboost, math, statistics, "
    "itertools, functools, and collections"
)

METHOD_RULES = f"""Every method is a module-level function named exactly after the method:

    def method_name_in_snake_case(history, horizon, frequency):
        \"\"\"When to use it, why it wins there, and any caveat on how it behaves.\"\"\"

It receives history as a list of floats, horizon as an int, and frequency as a string such as
"1 hour" or "1 day". It returns exactly horizon finite floats. Do not read files or the network,
use randomness, call eval/exec, or hard-code any series.

Every method is self-contained: it implements its own algorithm from {ALLOWED_IMPORTS_SENTENCE},
imported inside the function body. A module-level import you write is discarded on the next
rewrite, breaking every method that relied on it. NotApplicable is already defined at module
level; do not redefine or import it.

There are no fallbacks. Check preconditions -- history length, unsupported frequency,
non-positive values -- with a plain if BEFORE running the algorithm, and decline with the reason:

    raise NotApplicable(f"needs {{2 * period}} points, got {{len(history)}}")

Never catch a broad exception around your own logic, never return a forecast from an except
handler, and never turn a caught exception into NotApplicable. A failing library call is a defect
to repair, and silencing it hides it exactly as a fallback would. Both are rejected automatically.

Write the docstring for someone choosing between methods: when to use it, why it wins there over
the alternatives, and how it fails."""

WRITE_METHOD_PROMPT = f"""You implement one named classical statistical forecasting method as a
Python function. Implement the method that is described; do not substitute a different method
you consider stronger.

{METHOD_RULES}

Return exactly one JSON object:
{{"code": "<the complete function source, starting with def>"}}
"""

IMPROVE_METHODS_PROMPT = f"""You are evolving a Python module of forecasting methods against measured
results on a training set. You see the whole module and every method's measured behavior, and
work it into a set that between them covers the series types in the data.

You cannot write a new method from nothing. Your operations are repair (rewrite), consolidation
(merge), and removal (delete), applied to the methods already in the module.

{METHOD_RULES}

Judge every method on three metrics of equal weight: mean_smae, mean_srmse and
mean_shape_correlation. None outranks the others; winning on error while losing badly on shape
is not winning.

Each method's report gives:
- mean_smae and mean_srmse. Each task's error is divided by the mean
  absolute truth over its horizon, so a slow expensive series and a fast cheap one count equally.
  Read as a fraction of the series' own magnitude: 0.1 is an error a tenth its size, 1.0 is an
  error as large as the series. Lower is better, floor 0.0, no upper bound;
- mean_shape_correlation: correlation between the forecast's shape and the truth's. 1.0 tracks
  the series, 0.0 carries no information about which way it moves. A flat forecast scores 0.0
  however good its error;
- mean_variance_ratio and mean_change_smae: supporting evidence on the same question as shape.
  0.0 is flat; far above 1.0 swings wider than the truth. Both are wrong;
- success / total, coverage;
- not_applicable: declined by raising NotApplicable -- correct behavior, not failure;
- crashed: raised something else. Always a defect;
- invalid: wrong shape or a non-finite value. Also a defect;
- smae_by_series_type: error grouped by series type -- the evidence for the docstring, and how
  to see which series a method is genuinely strong on;
- sample_failures: real exception messages from crashed or invalid runs.

Judge on evidence, and keep a set that covers the data rather than a small one:
- crashed or invalid on most tasks is a bug: rewrite it, or delete it if the method cannot work
  under this contract;
- scores matching another method on all three metrics across the same tasks means they compute
  the same forecast: merge into the one genuinely implemented, or delete the redundant one.
  Similar mean_smae alone is not redundancy -- two methods failing on different tasks both
  contribute;
- a method beaten everywhere by another, on all three metrics across the same tasks, is worth
  deleting. Beaten on error alone is not beaten;
- a method poor overall but strong on one series type stays, with a docstring saying exactly that;
- high not_applicable is fine when the docstring says so; narrow the docstring rather than
  loosening the guard. A method declining most tasks but excellent on the rest is valuable, and
  its mean is not comparable with one that accepts everything;
- do not keep the worse of two methods because it looks simpler or more familiar. If you do
  anyway, say in the reason that you are overriding the measurement, and why.

Rewrite is the operation that improves the set, not delete:
- a method crashing or scoring badly may be a repairable implementation rather than a bad idea;
- a series type every current method scores badly on is a reason to rewrite the closest method to
  handle it, stating in its docstring that this is what it now covers.

Update the docstring of every method you rewrite or merge so it reflects the evidence rather than
a textbook description.

Return exactly one JSON object with the operations to apply, in order:
{{"operations": [
  {{"op": "delete",  "name": "<method name>", "reason": "<why, meaning the reason and reasoning>"}},
  {{"op": "rewrite", "name": "<method name>", "code": "<the complete function source>", "reason": "<why>"}},
  {{"op": "merge",   "names": ["<method name>", "<method name>"], "into": "<method name>", "code": "<the complete function source>", "reason": "<why>"}}
]}}

Every `code` field carries the entire function, `def` line to last line. Never abbreviate it,
never write `...`, never send a diff, never leave a comment standing in for lines you did not
change. A rewrite restates the whole method even when only its docstring differs. An abbreviated
`code` field rejects the whole batch, so if you are short of room return fewer operations rather
than shortening any of them.

Operations apply in sequence to the module as your earlier operations have already changed it.
A merge removes every method it names, so do not also delete those afterwards, and do not name a
method an earlier operation already consumed. Touch each method with exactly one operation.

What the history above establishes is settled unless this generation's measurements contradict
it: do not re-derive it, and do not spend an operation rediscovering a bug already fixed. A method
removed in an earlier generation is gone for good; nothing here can bring it back.

Every operation states a reason citing the measurement that justifies it; the reasons become the
commit message. Return an empty operations list only if nothing in the report warrants a change.

"""

def build_write_request(
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


def describe_past_generations(history: History, live: Sequence[str]) -> str:
    """What earlier generations already established, as evidence rather than as a changelog.

    Two views of the same operations: how each surviving method came to be what it is, and
    what has already been tried and taken back out. Empty when nothing has happened yet.
    """
    if not history:
        return ""

    sections = []
    provenance = []
    for name in live:
        operations = history.for_method(name)
        if not operations:
            # A seed method nothing has touched yet has no history to report.
            continue
        lines = "\n".join(
            f"  gen {op.generation} {op.op} -- {op.reason}" for op in operations
        )
        provenance.append(f"{name}\n{lines}")
    if provenance:
        sections.append("## How the current methods got here\n\n" + "\n".join(provenance))

    buried = history.removed(live)
    if buried:
        lines = "\n".join(
            f"{op.name} (removed in generation {op.generation}) -- {op.reason}" for op in buried
        )
        sections.append(
            "## Tried and removed\n\n" + lines + "\n\nA name listed twice was added back "
            "after being removed once, and removed again."
        )
    if not sections:
        return ""
    return "# What earlier generations already established\n\n" + "\n\n".join(sections)


EMPTY_MODULE_INSTRUCTION = """# The module is empty

There are no methods yet and therefore no measurements, and no operation can create one: the
module has to be seeded before evolution can run. Return an empty operations list.

"""


def build_improve_request(
    *,
    module_source: str,
    reports: Sequence[Mapping[str, object]],
    generation: int,
    task_count: int,
    history: History | None = None,
    live: Sequence[str] = (),
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
    established = describe_past_generations(history, live) if history is not None else ""
    # The history reads as settled context, so it comes before this generation's numbers.
    prefix = f"{established}\n\n" if established else ""
    # An empty module has no measurements, which reads as "nothing warrants a change" unless
    # the request says outright that the whole set has to be written from scratch.
    opening = EMPTY_MODULE_INSTRUCTION if not reports else ""
    return (
        f"{prefix}{opening}# Measured results\n\n{summary}\n\n"
        f"# Current module\n\n```python\n{module_source}```\n"
    )



def build_retry_request(error: str) -> str:
    """Ask again after a rejected batch, quoting the exact failure."""
    return (
        "Your previous operations were rejected and nothing was applied. Operations are "
        "applied atomically, so one malformed operation discards the whole batch, including "
        "the ones that were fine.\n\n"
        f"The rejection was:\n\n    {error}\n\n"
        "Return the complete JSON object again with that fixed. Two rules account for most "
        "rejections: every `code` field must be the entire function source starting with "
        "`def`, never abbreviated or replaced by `...`; and operations apply in sequence, so "
        "an operation must not name a method an earlier operation already deleted or merged "
        "away. Return fewer operations if you need the room to write each one out in full."
    )
