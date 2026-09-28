# i1, episode 2 observations

Read episode 1 notes and examined visible traces plus the allowed Train views. Submitted five variants (attempts 0006–0010). Accepted attempt 0006 is the new shared best: visible fitness 0.24702 versus 0.24275, hidden check passed. It uses the episode 1 short spike rule, plus a holiday document rule for hourly 72-step tasks whose first document is an official/policy/administrative holiday or public-observance notice. It rejects upward holiday corrections and applies the first downward holiday correction at no more than 20% below base.

Visible changes in attempt 0006: task_145 +0.0698, task_153 +0.0968, task_158 +0.0131, task_146 +0.0262; task_151 -0.0069, task_159 -0.0012. Thus the gain is concentrated in a few holiday tasks, including related task_153/158 series. The rule is more reliable for a full-day correction starting at step 0 than for partial-day traffic corrections, but the sample is too small to specialize further confidently.

Attempts 0007 and 0008 were no-ops: a proposed larger downward clip for short zero-multiplier corrections did not overcome the validator's half-strength multiplier. Attempt 0009 forced full zero and lowered fitness to 0.2467: task_104/87 truth had near-zero values for the first four corrected steps but rebounded on the fifth, so full-window zero overshot. Attempt 0010 allowed weak decaying post-holiday downward effects on days 2 and 3. Fitness moved only +0.000023 and was not accepted by the harness's improvement threshold. Do not treat this tiny difference as evidence for the extra rule.

Episode submissions exhausted. Accepted function is `shared/harness/0006_i1.py` / `shared/best_harness.py`; private copy at `ws_i1/my_harness.py`.
