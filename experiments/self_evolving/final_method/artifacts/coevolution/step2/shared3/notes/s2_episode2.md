# s2 episode 2: decision reassessment and small Moirai blend

Started from seed visible fitness 0.18894. Read all episode-1 notes and visible traces. Used three submissions; no extraction-instruction changes.

## What NEVER worked in this direction

Episode 1: reducing the repair margin to 0.1 admitted a harmful repair on task_112 and failed hidden; margin 0.25 reduced visible fitness. Linear fill lowered task_193's repair back-test error greatly but worsened its forecast and reduced visible fitness. Correction strengths 0.8 and 1.2 both failed hidden. S0's confidence and boilerplate validator changes did not beat seed. These observations argue against more local decision tuning without a new signal.

This episode's ratio-guard test changed validator `ratio` from -0.00696 to -0.05. It passed hidden but reduced visible fitness to 0.1876. It removed a helpful downward correction on task_158 (gain fell by 0.0944); the apparent protection against extreme multipliers did not help here. Do not strengthen this guard based on the visible examples.

## Accepted change

A small Moirai 2.0 term diversifies the Toto/ARIMA forecast while keeping shrink, repair, extraction, and future corrections unchanged. At 3% effective weight, visible fitness rose to 0.19069 and hidden passed (attempt 0016). At 5% effective weight, visible fitness rose to 0.1915 and hidden passed (latest accepted attempt, weight 0.06103 against Toto 1.1005169 and ARIMA 0.059). Shared best now contains the 5% blend.

The 5% result has 29 better and 25 worse visible tasks versus the seed's 31 better and 23 worse. Thus, the gain is in magnitude, not breadth. Fold 0 fell slightly from the 3% blend while fold 1 improved. I stopped at 5% rather than tune the weight further; the local visible replay showed harm growing with blend weight, and further tuning risks overfit.

Mechanism: Moirai was one of the stronger alternatives to Toto on the visible tasks as a standalone method, and low-weight mixing offsets some Toto-specific errors. It does not require changing document decisions. Important caveat: visible performance is driven partly by a few large gains, and the hidden check only reports pass/fail.
