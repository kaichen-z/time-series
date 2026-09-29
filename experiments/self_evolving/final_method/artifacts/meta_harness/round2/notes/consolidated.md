# Round-two correction findings (through episode 3 audit)

Read this before starting a new candidate. Original episode notes remain as the
detailed record; this file merges their durable conclusions.

## Accepted mechanisms

- Start from `shared/best_harness.py`, not the routed seed. The seed changed 18
  visible tasks; later accepted rules cover explicit one- and two-hour surge
  durations, fourfold electrical-load measurements, modest early holiday dips,
  recurring maintenance zeros, a nonnegative physical floor, and a document
  polarity contradiction for task_79.
- The latest accepted r2 episode-3 rule scales already accepted short hourly
  surges by Toto's forecast level when task backtests indicate Toto is reliable.
  It passed the hidden check and raised visible fitness to 0.49537.
- Duration words can reveal that the extractor included the restoration bin.
  Keep the rule tied to an explicit duration and the matching correction width.
- Repeated direct measurements in documents can override a weaker extracted
  multiplier. Require agreement across reports and the same physical quantity.
- A zero floor is supported when the entire observed series is nonnegative.

## What never worked, or lacks reliable support

- A broad history-only rule that snaps a two-level series to its modal states
  improved visible task_79 but failed the hidden check. Its accepted replacement
  requires an explicit conflict between near-shutdown corrections and nonstop
  peak-operation documents.
- Generic short-surge and holiday rules built from the routed seed improved
  visible fitness but failed the hidden check. The narrow accepted versions are
  in the current best.
- Blanket activation of overlapping corrections, strong holiday reductions,
  and late holiday rebounds lacks support: visible solo corrections frequently
  harm the score.
- Parameter search on 80 Train views and single-task visible gains are weak
  evidence. Avoid tuning thresholds just to recover a visible task.

## Episode-3 audit

The current best changes 20 of 54 visible tasks from the base forecast, and
each changed task has a higher visible gain than its base-only gain. Across all
80 Train inputs it changes 31 tasks and returns valid finite outputs. Only one
unused correction has a visible solo gain over 0.01: task_97's late [43,56)
0.85 multiplier, worth only +0.026 in isolation. Its document bundle mixes
current intermittent pauses, explicit post-maintenance return to service, and
later unrelated claims. This does not justify a general routing rule.

Use `shared/skills/audit_corrections.py` to repeat the audit as the shared best
changes. It reads only the permitted round-two views and visible traces.
