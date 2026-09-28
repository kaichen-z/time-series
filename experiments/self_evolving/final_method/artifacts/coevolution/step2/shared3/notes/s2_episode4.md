# s2 episode 4: reassessment, no new submission

Read the shared notes, all 23 evaluated attempts, the accepted best config, and only `shared/traces/visible_tasks.jsonl`. The accepted best remains attempt 0019 (visible fitness 0.19151, hidden pass): a roughly 5% Moirai contribution to Toto/ARIMA, 20.15% shrink, phase-median repair at margin 0.30, and the seed correction policy.

I shifted from decision tuning to the numerical program. `shared/skills/numerical_replay.py` screened 2% and 5% additions of every available forecast on the 41 visible tasks without repair variants. The positive candidates were small and concentrated in task 61. For example, 5% `combined_timesfm_seasonal` increased replay fitness by only 0.00050; task 61 alone improved by 0.1474, while tasks 65, 145, and 226 lost 0.0503, 0.0430, and 0.0332. A 2% STL-ETS contribution showed a similar pattern. Chronos looked strongest in this replay, but the earlier 5% Chronos submission (0010) failed hidden. These are weak signals for generalization, so I made zero submissions.

The repair variants also do not suggest a safer fill change. Phase median already captures large task 43 and 193 gains; linear fill's much better task 193 back-test did not translate into a better evaluated forecast. Other repair tasks either have identical variants or back-test changes too small to support a general rule.

## What NEVER worked or lacks support

- Decision tweaks to margin, fill, correction strength, and validator weights did not improve both visible fitness and hidden check. A trust gate would discard useful short upward events in low-trust cells; reduced trust weighting also discarded task 158's useful staged correction.
- Small numerical blends with positive visible replay effects are not a reliable improvement mechanism when one task drives the gain. Chronos and kernel ridge already demonstrated this failure in evaluated attempts.
- Further local tuning of the accepted 5% Moirai weight lacks support: an extra 2% reduced replay fitness, and the accepted gain is already concentrated in magnitude rather than task breadth.

No config changed. A future direction would need a new document-applicability signal or evidence that improves forecasting across multiple independent task groups, not another small visible-only parameter adjustment.
