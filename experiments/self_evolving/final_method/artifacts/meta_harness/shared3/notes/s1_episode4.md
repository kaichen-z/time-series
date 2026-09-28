# s1 episode 4: reassessment and targeted correction gates

Read the shared notes, all existing harnesses, and the permitted visible traces. The earlier extreme-multiplier rule failed to improve visible fitness. Reassessed the remaining errors as validator decisions rather than more multiplier scaling.

Submission `s1_032113_76a47a`: allow an 85% reduction for a short zero-multiplier event with confidence at least 0.9 and document boilerplate share below 0.5. Hidden check passed, but visible fitness rose only from 0.247532 to 0.247551 and the attempt was not accepted. It affects two visible near-duplicate day-series tasks. A full-horizon zero event has negative solo gain, so this is deliberately restricted to short events.

Submission `s1_032208_184707`: on hourly seasonal tasks with task Toto backtest error at least 2, document confidence at least 0.7, and a localized positive multiplier from 1.2 to 1.5, apply at least half strength. Retains the prior 50% cap. Hidden check passed; visible fitness increased to 0.2491 and it became shared best. It changed one visible task and one unlabeled task in addition to the two short zero events. All 80 unlabeled views returned exactly H finite values.

What NEVER worked well enough to keep: broad higher caps or indiscriminate acceptance; forcing an extreme 45x multiplier even with a sigma-based bound; and adjusting short zero events by only changing the acceptance weight while retaining the global -50% effect cap (that is a no-op). The stronger zero drop had negligible visible benefit. The current new positive rule has support from only one visible task, so further threshold tuning would likely overfit. No more submissions this episode.
