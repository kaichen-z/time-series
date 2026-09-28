# s1 episode 4: stop numerical tuning without a transferable signal

I reassessed the accepted attempt 0019 (visible fitness 0.19151, hidden pass) against every shared attempt and the visible trace. I made no submissions this episode; the allowance is a maximum, and the candidates below lacked a credible path to hidden improvement.

## New audit

Using `shared/skills/numerical_replay.py`, I screened small blends into the current Toto/ARIMA/Moirai program on the 41 visible tasks without repair variants. The apparent gains from Chronos Bolt, STL-ETS, and the TimesFM seasonal router were concentrated in fold 1 while **each harmed fold 0**. For example, a 2% Chronos blend changed the sum of joint-error gains by -0.049 on fold 0 and +0.215 on fold 1; task 61 alone contributed +0.139. A 2% STL-ETS blend gave -0.063 and +0.218, with task 61 contributing +0.122. A 2% TimesFM seasonal blend gave -0.056 and +0.172, with task 61 contributing +0.108. These values are replay diagnostics, not evaluator fitness, because the 13 repair-variant tasks are omitted.

The document correction trace also shows why a simple validator change is unlikely to help: high-impact useful adjustments include short upward corrections at low back-test error (tasks 128/138, 131/141) and a staged downward correction at moderate error (task 158), while a broad correction at task 77 is correctly rejected. Confidence, window length, trust, and correction magnitude do not separate these cases consistently. Existing validator attempts already tested several of those axes and failed to beat the accepted configuration.

## What NEVER worked

- Visible-only numerical gains from less shrink, Chronos, or kernel ridge failed hidden checks in prior episodes. Further tiny blend changes repeat that search, with the additional warning that fold 0 and fold 1 disagree.
- Tight clipping, lower repair margins, linear repair fill, global correction strength changes, and broad validator coefficient changes did not improve the shared best. The accepted 5% Moirai diversification remains the only change with both visible improvement and a passing hidden check.

Current recommendation: keep `shared/best_config.json` at attempt 0019 until a new data-derived feature or a genuinely different mechanism becomes available. Do not interpret a small visible replay improvement from a new numerical term as evidence of transfer.
