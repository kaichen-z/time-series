# s0 episode 1: retrieval and repair

Starting point: seed visible fitness 0.18894. Read current instructions, all visible trace metadata, and available prior notes. Short hourly future corrections with multipliers 2–5 often help; broad full-horizon corrections are mostly rejected. An accepted multiplier 45 on task_183 and multiplier 5 on task_146 appear harmful. The current validator has a large positive coefficient on absolute multiplier size and nearly zero confidence weight.

## Submission 1: linear repair fill

Visible fitness 0.18264, hidden pass, not accepted. Task_193 gain fell from 1.9472 to 1.4933 despite a much better history back-test for linear fill. Back-test improvement does not guarantee horizon improvement; keep phase-median.

## Submission 2: confidence and locality validator rewrite

Visible fitness 0.1402, hidden fail. It admitted harmful corrections on tasks 151, 158, 199, and degraded task 43; large simultaneous coefficient changes are unsafe.

## Submission 3: confidence coefficient 0.5

Visible fitness 0.1876, hidden fail. Newly admitted full-horizon multiplier 0.65 on task 77 caused a -0.288 gain. Confidence alone does not establish correction reliability.

## Submission 4: docbase coefficient -3.5

Visible fitness 0.18634, hidden pass, not accepted. It reduced task_158 gain by 0.094 and task_87 by 0.048. Stronger blanket boilerplate penalty discards useful corrections.

## Submission 5: window-fraction coefficient 0

Visible fitness 0.18775, hidden fail, not accepted. It reduced task_158 gain by 0.094. The current positive window weight helps at least one staged, genuine full-horizon adjustment.

## Takeaway

No submission improved the shared best, which remains the seed configuration. The repair back-test is useful but fill-method ranking did not transfer to the forecast on task_193. Correction confidence, document boilerplate, and window length cannot be safely modified in isolation here: visible counterexamples include task_77 (confident broad but false), task_158 (staged broad and useful), and task_183 (extreme multiplier but harmful). Future work should seek a mechanism or feature that distinguishes event relevance and effect size before changing the validator globally.
