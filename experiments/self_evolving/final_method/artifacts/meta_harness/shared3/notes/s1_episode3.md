# s1 episode 3: extreme multipliers on low baselines

Reviewed the two prior s1 notes, the current shared best, and all permitted visible solo-correction traces. There were no other agents' notes in `shared/notes/` at review time. The best harness already uses normal-deviation calibration for sustained low-level positive events and keeps the seed gate elsewhere.

One visible correction had a very large multiplier (45) over a short low-baseline window. I tried accepting such seasonal corrections while limiting the mean implied shift to three `sigma_main_calib` units. Submission `s1_031926_6ef38f` passed the hidden check but visible fitness fell from 0.2475 to 0.2456; task_183 worsened. I reverted the rule and made no further submissions. This is one task, so no universal conclusion about extreme multipliers follows.

The current `my_harness.py` is an exact copy of `shared/best_harness.py`. Added `shared/skills/correction_diagnostics.py` and its README to reproduce the correction-magnitude screen from permitted visible traces.

Across episodes, the best supported regimes are short positive hourly seasonal surges with large level/sigma ratios, sustained positive hourly seasonal shifts below one sigma, and mild first-day hourly seasonal decreases. Broad corrections, flat cells, and extreme multiplier rules have not shown a robust reason to override the seed gate. The related visible event families are not independent evidence.
