# s0 episode 4: small Chronos diversification failed hidden

I reread the shared notes and all evaluated attempts. The accepted best is still attempt 0019: Toto/ARIMA with about 5% Moirai, 20.15% shrink, phase-median repair at a 0.30 back-test margin, and the original future-correction validator.

I reassessed retrieval first. Across earlier episodes, confidence, boilerplate, window length, ratio, and trust coefficient changes either admitted harmful broad events or removed the useful staged correction on task 158. The current visible trace does not support another global validator change. Linear repair fill improved task 193's history back-test but worsened its future forecast; lowering or raising the repair threshold did not help. I did not spend an extraction-instruction submission because the large repair wins on tasks 43 and 193 are fragile and there was no trace-supported wording change.

I then tested a different role: a 2% Chronos Bolt term added to the accepted numerical mix, preserving the original relative weights of the three terms. This is a smaller version of the 5% Chronos probe that improved visible fitness but failed hidden earlier. The visible-only replay indicated a small gain across 24 of 41 tasks without repair variants, but it cannot reconstruct repaired Toto forecasts.

**One submission:** attempt 0023 gave visible fitness 0.19350 versus 0.19151 for the accepted best, but failed the hidden check and was rejected. The main visible changes versus best were task 61 +0.0575, task 112 +0.0365, task 43 -0.0288, and task 193 -0.0206. The gain again depends heavily on a few tasks and does not generalize. No extraction instructions were changed. The shared best remains attempt 0019.

## What NEVER worked or lacks support

- Chronos Bolt diversification at both 5% and 2% improves visible fitness but fails hidden checks. Do not retry a smaller dose based only on these 54 tasks.
- Small kernel-ridge diversification likewise improved visible fitness but failed hidden. These are repeated examples of visible-only numerical improvements that should not displace the accepted best.
- Global validator coefficient tuning cannot resolve the key correction conflicts with the available features; stronger ratio and lower trust specifically remove task 158's useful staged event.
- Linear fill, looser repair thresholds, stronger or weaker correction strength, and tight clipping failed in prior submissions or replay. The history back-test alone is not a reliable fill selector for future error.

I stopped after one submission because the remaining nearby parameter changes had no mechanism supported by the visible audit and the two Chronos doses both failed hidden checks.
