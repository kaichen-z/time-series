# Task: write a better "document correction" function for a time-series forecaster (Meta-Harness)

Dr-CiK tasks: a numeric history, some text documents, and H future steps to forecast. The pipeline already
produces a strong **base forecast**. Documents describe future events; an extractor turned them into
**corrections** (steps [start, end) and a multiplier m, e.g. 1.3 = +30%). Your job is the function that decides
how to use those corrections:

```python
def adjust(view: dict) -> list[float]:   # must return exactly view["H"] finite floats
```

`view` fields (no labels): `tid, H, freq, history, base_forecast, toto_forecast, documents` (texts, truncated),
`doc_confidence, docbase` (share of methodology/boilerplate words in the documents), `cell` (frequency ×
seasonality class), `cell_toto_backtest_error, task_toto_backtest_error` (how badly Toto forecast the history
back-tests of this cell / this task; high = Toto unreliable), `sigma_main_calib` (normal deviation scale of the
series), `corrections` (list of {start, end, multiplier}).

Error per task = sMAE + sRMSE (capped at 5). Gain = Toto error − your error (positive is better).

## Start point (round 2)
`$RUN_DIR/shared/harness/0000_seed.py` is the CURRENT main method: it routes each task by `cell` to one of five
correction functions evolved in round 1 (read them at the FUNC_DIR path in the seed). Round-1 lessons, visible in
those files: shorten windows that include the "back to normal" step, fix window lengths from phrases such as
"one-hour", lift the ±50% bound for outages / zero events and short hourly surges, floor non-negative series at 0.
On test, the routed version changed 18 tasks and all 18 improved, so its rules generalise; look for NEW
mechanisms (e.g. corrections it still ignores or mis-sizes, tasks where the routed function and the others
disagree). Note: in the traces, `main_method_forecast` / `main_method_gain` are the OLDER nrd4-team method, not
the routed seed; use `shared/attempts/0000_seed.json` for the seed's per-task visible gains.
You may rewrite anything: new features from the documents or the history, per-correction magnitude
calibration, shrinking multipliers, handling overlapping corrections, different bounds, rules that depend
on the series, etc. Pure Python + math/statistics (numpy is fine); it must not read files or use the network,
and must run on 80 tasks within 10 minutes.

## Evaluation (you cannot run it yourself)
`python3 $RUN_DIR/../../submit_h.py my_harness.py --note "what/why"` returns fitness on two VISIBLE Train
folds (54 tasks), per-task gains, runtime errors, and `hidden_check` (pass/fail) from a HIDDEN fold. It becomes
the shared best (`shared/best_harness.py`) only if visible fitness improves AND the hidden check passes.
Fitness = mean over folds of (mean gain + 0.5 × mean negative gain) − 0.25 × std over folds.
Budget: a fixed number of submissions per agent (`budget_left`).

## Data you may analyse
- `shared/views_train.json`: the inputs above for all 80 Train tasks (no labels).
- `shared/traces/visible_tasks.jsonl`: for the 54 visible tasks, additionally `truth`, the main method's
  forecast and gain, `base_only_gain` (no corrections), and `corrections_solo` (gain if each correction were
  applied alone at full strength). Do not look for any other labels (dev / test / hidden fold); doing so
  invalidates the study.
- `shared/harness/` and `shared/attempts/`: every submitted function with its visible result.

## Known facts
- The fold split is new in this round (the hidden fold was never used before).
- 80 tasks is small: plain parameter search overfits (held-out folds get worse). Prefer changes with a clear
  mechanism, and check how many visible tasks a change actually affects.

## Shared memory (please use it)
`shared/notes/` (observations, hypotheses, what worked, what NEVER worked) and `shared/skills/` (reusable
analysis scripts with a README). Read other agents' notes before proposing.
