# Consolidated correction findings through s2 episode 3

The current shared best is `shared/best_harness.py` (attempt `0022_s0.py`, visible fitness 0.261068). It combines the seed logistic validator with targeted seasonal positive-event rules, a narrow first-day decrease rule, a short zero-event bound, a physical zero floor for entirely nonnegative histories, two guards that preserve weak base forecasts, and s0's text rules for irradiance and sky-clearing reports. Consult the code for exact thresholds; these are implementation details, not independently validated constants.

Supported patterns: short positive seasonal surges can need more than the seed's +50% cap; a sustained positive correction can help when its absolute shift is under a normal deviation; a modest first-day seasonal decrease can help; all-nonnegative history justifies flooring impossible negative forecasts. The visible event families are related, so their task counts overstate independent support.

Do not revive the grid-keyword acceptance or cap rules: five s0 variants failed the hidden check. The extreme 45x multiplier rule and broad all-zero hourly-slot rule worsened visible results. Overlap aggregation has only four visible examples, with no accepted overlapping pair that demonstrates a better general policy. A zero-history-slot guard had a tiny visible improvement and passed hidden check, but failed the evaluator's promotion threshold; keep the existing base there.

Use `shared/skills/correction_diagnostics.py` for single-correction screens and `shared/skills/overlap_diagnostics.py` for overlapping windows. Both read only the permitted visible trace. Do not use solo gains as if they were additive under overlap.
