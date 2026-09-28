# Agent i3, episode 4 — reassessment

The earlier decision sweeps are exhausted: zero strength, weaker repair margin, linear fill, stronger ratio penalty, confidence reweighting, and global correction damping never yielded an accepted improvement. Larger TimesFM share and lower shrink raised visible fitness but failed the hidden check, so a small visible gain from tuning the blend is not enough evidence of generalization.

Visible correction records show useful short events with multipliers 2–5 and some useful broad reductions. A blanket correction gate or duration penalty risks removing these, especially task_43's large benefit. The current ratio coefficient is tiny, making calibrator changes unlikely to matter unless paired with a stronger ratio weight. Clipping the numerical forecast to the history range reduces visible fitness strongly; relaxed clipping changes almost no tasks. I will explore a different numerical mechanism with a conservative submission budget.

Submission 1: a 0.05 difference between the seasonal TimesFM router and plain TimesFM raised visible fitness from 0.19477 to 0.1959, but failed the hidden check. This repeats the episode 2 pattern: plausible small numerical diversification can improve visible folds without generalizing. Do not increase this seasonal correction based on the visible slope alone.
Submission 2: a 0.02 STL/ETS minus Toto adjustment raised visible fitness to 0.1964 on both folds, but failed the hidden check. Most visible tasks worsened slightly; gains were concentrated in a few tasks, so the aggregate improvement was fragile. A third small model blend would repeat this failure mode without new evidence.

## Closeout

Two submissions used; neither was accepted. The shared best config is unchanged. Both seasonal adjustment candidates improved the visible metric and failed the hidden check. These small additive numerical corrections are now an observed dead end without independent evidence of a broader regime in which they help. A grid around the accepted ARIMA/TimesFM blend showed only small visible improvements toward more TimesFM, already known to fail hidden checks at larger weights. I stopped before spending the remaining budget on another local blend search or a speculative extraction-instruction rewrite.

What never worked across this agent's episodes: blanket removal or damping of document corrections; lowering repair margin; switching repair fill to linear; broad confidence reweighting; a stronger ratio penalty; adding Chronos or larger TimesFM; lowering shrink; and the two episode-4 seasonal numerical differences. Visible improvement alone repeatedly failed the hidden check, so no candidate from these directions should replace the shared best.
