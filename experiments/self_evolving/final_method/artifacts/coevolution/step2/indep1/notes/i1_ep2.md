# i1, episode 2: reassessment

Episode 1's broad validator changes, lower repair margin, and linear history fill NEVER worked as improvements: every submission fell below the shared seed, and all but linear fill failed the hidden check. The validator already admits most short, large documented events; those corrections usually help. I am therefore leaving retrieval unchanged.

I reconstructed the seed's numerical forecast from visible method traces. Only task_43 and task_193 accept history repair at the 0.30 margin. Task_43 also receives a half-strength future correction, which explains why naive offline comparisons of numerical candidates were misleading. After accounting for both repairs, the seed's ARIMA weight near 0.059 and shrink near 0.20 are locally strong. Larger shifts hurt one or both folds, especially the ship trace in task_61.

A 0.02 weight on `stl_ets` alongside the seed terms gives a small offline estimated gain (~0.0011 fitness) in both visible folds. Its benefit is spread over a range of series, including task_43, task_64, task_98, and task_112; it also harms task_193 and a few others. This is a cautious test of a seasonal model blend, not a claim that the visible gain will generalize.

Submission 1 confirmed a visible gain: fitness 0.1907 versus seed 0.1889, with both folds improving. The hidden check failed, so the config was rejected. A tiny visible numerical improvement is not reliable evidence of generalization here. This episode used one submission and made no accepted change. I stopped instead of tuning nearby weights against the visible tasks.
