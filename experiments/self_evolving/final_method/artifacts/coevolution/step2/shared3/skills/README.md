# Visible-trace analysis tools

- `numerical_replay.py` screens changes to `numerical.program` on the 41 visible tasks without repair variants. It anchors the trace's `main_method_forecast` to the **seed** program, retains the inferred document-correction factor, and excludes repaired tasks because repaired Toto predictions are unavailable. Run `python3 numerical_replay.py` from any directory. The output is diagnostic and does not replace submission evaluation.
- `visible_decision_audit.py` audits repair and future-correction outcomes from the same visible trace. Run `python3 visible_decision_audit.py --sort effect`.

Both tools read only `shared/traces/visible_tasks.jsonl` and the shared best config. Do not use other labels.

## Other visible-trace diagnostics

`python3 shared/skills/numerical_replay.py` screens numerical changes while anchoring document effects to the seed. `python3 shared/skills/visible_decision_audit.py --sort trust` reconstructs the seed numerical forecast and reports the effect of its traced future corrections. Both exclude repair-variant tasks and are diagnostics, not full evaluator results.
