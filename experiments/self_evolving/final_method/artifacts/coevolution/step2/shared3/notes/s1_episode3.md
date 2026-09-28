# s1 episode 3: no credible numerical change

I read all five episode 1–2 notes and the 21 evaluated attempts. The shared
best remains attempt 0019: approximately 95% Toto/ARIMA mix plus 5% Moirai,
20.15% shrink, phase-median repair at margin 0.30, and the original future
correction validator. Visible fitness is 0.19151 with a passing hidden check.

## Reassessment

I rebuilt a numerical replay in `shared/skills/numerical_replay.py`. The
trace's `main_method_forecast` is from the **seed**, so the replay anchors its
document factor to the seed numerical program, then applies candidate programs.
It excludes 13 tasks with repair variants, for which repaired Toto predictions
are not present. An initial replay used the current best as the anchor; I
corrected that before drawing conclusions. The corrected replay exactly
reproduces the seed on a task without repairs or future corrections.

Small 2% blends of STL-ETS, kernel ridge, or Chronos Bolt showed approximate
fitness gains of 0.00067, 0.00087, and 0.00132, respectively, on 41 unrepaired
visible tasks. These gains are dominated by task 61 (gain changes +0.1225,
+0.0953, and +0.1388). All three harm task 65, task 145, task 226, and task
63 to varying degrees. This is the same kind of narrow visible advantage that
failed hidden checks for Chronos and lower shrink in episode 1. A 2% extra
Moirai blend also reduced approximate fitness by 0.00011 in the corrected
replay; continued weight tuning is poorly motivated.

The calibrator's ratio coefficient is nearly zero in the best config. Changing
its window/statistic would therefore have little direct effect unless it
crossed a validator threshold, which is hard to support from these traces.
Prior ratio-guard changes passed hidden but reduced visible fitness by
discarding a useful task 158 correction. No extraction change was submitted:
the existing repair wins are large, and the prior alternative fill had a
better back-test yet a worse forecast.

## What NEVER worked, across episodes

- Lowering shrink to 0.10 or 0.15 and adding 5% Chronos raised visible
  fitness but failed hidden checks.
- Tight clipping harmed visible forecasting, especially task 61.
- Lower repair margins 0.10 and 0.25 admitted repairs whose back-test gains
  did not transfer; a higher 0.40 margin gave no measurable gain.
- Linear fill had a spectacular task 193 back-test but worsened its forecast.
- Correction strengths 0.8 and 1.2 failed hidden checks. Broad confidence,
  boilerplate, window, and ratio coefficient changes did not beat the best.
- Removing the accepted Moirai component while changing ARIMA performed worse
  and was confounded; it is not evidence for an isolated ARIMA adjustment.

No submission this episode. The current candidate remains the only one here
with a passing hidden check and stronger visible fitness. The reusable replay
and its limitations are documented in `shared/skills/README.md`.
