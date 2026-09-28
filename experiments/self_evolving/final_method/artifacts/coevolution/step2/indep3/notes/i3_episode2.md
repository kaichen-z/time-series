# Agent i3, episode 2 — numerical diversification

The episode 1 decision changes did not generalise: removing future corrections failed the hidden check and reduced visible fitness; admitting weaker history repairs also failed; linear fill reduced visible fitness. A stricter repair margin changed almost nothing. Broadly changing the decision role is unlikely to help without a more specific document or series signal.

I reconstructed the seed forecast from the visible traces and verified that the simulated joint error reproduces its visible fitness exactly. Small blends of TimesFM 2.5 and Chronos Bolt help across both folds. The improvement is modest and comes from a diversified forecast, not from choosing a single alternative model. TimesFM alone is known to be worse than Toto on most train tasks, so its role here should remain small.

## Submissions and results

| Change | Visible fitness | Hidden check | Accepted |
| --- | ---: | --- | --- |
| 10% TimesFM in seed base ensemble | 0.1948 | pass | yes |
| 10% TimesFM plus 5% Chronos Bolt | 0.1990 | fail | no |
| 10% TimesFM, shrink reduced from 20.15% to 15% | 0.2008 | fail | no |
| 15% TimesFM, original shrink | 0.1969 | fail | no |

Stopped after four of the five allowed episode submissions. The accepted 10% TimesFM blend is in `shared/best_config.json`.

## What never worked

The episode 1 decision changes did not work: zeroing document correction strength and lowering the repair threshold failed the hidden check; changing fill to linear reduced visible fitness. In this episode, adding Chronos, reducing shrink, and increasing TimesFM beyond 10% all improved visible fitness but failed hidden checks. The simulator was accurate for visible fitness, yet visible optimization alone did not generalise. Do not repeat these changes without a different mechanism or independent evidence. Small ensemble diversification helped once, but its benefit had a narrow range.
