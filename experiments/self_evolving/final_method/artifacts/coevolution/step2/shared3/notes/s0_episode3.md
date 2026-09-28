# s0 episode 3: reassessment and kernel-ridge test

Read the prior episode notes and attempts. The accepted best remains attempt 0019: Toto plus ARIMA plus 5% Moirai, with phase-median repair, 0.30 margin, and the original correction validator. Prior retrieval probes show no clean improvement mechanism: confidence, boilerplate, window, and ratio weighting either admitted harmful broad corrections or removed useful staged corrections. I did not change extraction instructions.

**One submission:** attempt 0022 added a 2% kernel-ridge lag-regression contribution to the accepted mix. Visible fitness rose from 0.19151 to 0.19213, but the hidden check failed, so it was rejected. It improved task 61 by only 0.0139 in the evaluated result, while hurting task 43 by 0.0236, task 193 by 0.0079, and task 226 by 0.0102. The small visible gain did not generalize.

My initial local replay mistakenly anchored the trace's `main_method_forecast` to the current best, though the trace was generated with the seed program. Its reported subset gains were invalid, and I removed the script. S1 independently supplied the correctly anchored `shared/skills/numerical_replay.py`; its corrected diagnostic finds only a small kernel-ridge benefit, dominated by task 61. I updated the shared skills README to describe that script and the decision audit.

## What NEVER worked or lacks evidence

- Local validator coefficient tuning: confidence, document boilerplate, correction window, and normalized magnitude guards failed to improve the shared best; several failed hidden checks. They cannot separate a useful broad staged correction (task 158) from a confident harmful broad correction (task 77) with the available features.
- Looser repair thresholds, linear fill, and correction strengths 0.8/1.2 failed or reduced visible fitness; the repair back-test alone does not rank future forecast quality.
- Visible-only numerical improvements repeatedly failed hidden: smaller shrink, a 5% Chronos blend, and now a 2% kernel-ridge blend. A weak visible increase is insufficient evidence to replace the best.
- Tight clipping degraded the replay. Larger Moirai doses remain risky as fold-0 harm grew, despite the accepted 3% and 5% doses.

No more submissions this episode. The accepted best remains attempt 0019. Further work needs new event-applicability evidence or a numerical change with a larger, broader effect.
